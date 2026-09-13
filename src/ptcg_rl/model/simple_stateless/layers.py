"""Pre-LayerNorm Transformer blocks over packed variable-length tokens."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache
from itertools import pairwise
from typing import Any, NoReturn, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from ptcg_rl.activation_precision import preserve_cuda_bfloat16_activation
from ptcg_rl.model.simple_stateless.packed import PackedTokenBatch

native_varlen_attn: Any
try:
    from torch.nn.attention.varlen import varlen_attn as native_varlen_attn
except ImportError:  # pragma: no cover - depends on the installed Torch build.
    native_varlen_attn = None


def _aten_varlen_attn_compat(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seq_q: Tensor,
    cu_seq_k: Tensor,
    max_q: int,
    max_k: int,
) -> Tensor:
    """Call the inference-only ATen kernel exported before Torch's wrapper."""
    output, _logsumexp, _rng_state, _unused, _debug_mask = (
        torch.ops.aten._flash_attention_forward(
            query,
            key,
            value,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            0.0,
            False,
            False,
            scale=None,
            window_size_left=-1,
            window_size_right=-1,
        )
    )
    return cast(Tensor, output)


varlen_attn: Any = native_varlen_attn or (
    _aten_varlen_attn_compat
    if hasattr(torch.ops.aten, "_flash_attention_forward")
    else None
)


class PackedRolloutInductorError(RuntimeError):
    """Packed rollout compilation failed and must be explicitly disabled."""


class PackedLearnerInductorError(RuntimeError):
    """Packed learner compilation failed and must be explicitly disabled."""


_PackedRolloutRunner = Callable[..., Tensor]
_PackedRolloutSegment = tuple[int, tuple[Tensor, ...]]
_ROLLOUT_INDUCTOR_MAX_SEGMENT_LAYERS = 9


class PackedSelfAttention(nn.Module):
    """Shared self-attention with a varlen FlashAttention CUDA path."""

    def __init__(self, *, d_model: int, num_heads: int) -> None:
        """Initialize packed QKV and output projections."""
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("attention dimensions must be positive and divisible")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, batch: PackedTokenBatch) -> Tensor:
        """Attend independently within each packed sequence."""
        qkv = self.qkv(batch.tokens)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(-1, self.num_heads, self.head_dim)
        key = key.view(-1, self.num_heads, self.head_dim)
        value = value.view(-1, self.num_heads, self.head_dim)
        if (
            query.is_cuda
            and query.dtype in {torch.bfloat16, torch.float16}
            and varlen_attn is not None
        ):
            attended = cast(
                Tensor,
                varlen_attn(
                    query,
                    key,
                    value,
                    batch.cu_seqlens,
                    batch.cu_seqlens,
                    batch.max_seqlen,
                    batch.max_seqlen,
                ),
            )
        else:
            attended = _reference_varlen_attention(
                query,
                key,
                value,
                offsets=batch.offsets,
            )
        return cast(Tensor, self.output(attended.reshape(-1, self.d_model)))


class PackedTransformerBlock(nn.Module):
    """One pre-LN attention/MLP block with scaled residual branches."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        residual_scale: float,
    ) -> None:
        """Initialize one dropout-free GELU block."""
        super().__init__()
        if feedforward_dim <= 0 or residual_scale <= 0.0:
            raise ValueError("feedforward width and residual scale must be positive")
        self.residual_scale = residual_scale
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = PackedSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
        )
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, d_model),
        )

    def forward(self, batch: PackedTokenBatch) -> PackedTokenBatch:
        """Transform packed tokens without padding or cross-sequence mixing."""
        attention_input = batch.with_tokens(self.attention_norm(batch.tokens))
        tokens = batch.tokens + self.residual_scale * self.attention(attention_input)
        tokens = tokens + self.residual_scale * self.feedforward(
            self.feedforward_norm(tokens)
        )
        return batch.with_tokens(tokens)


class PackedTransformerTrunk(nn.Module):
    """Deep shared stateless Transformer with a final LayerNorm."""

    def __init__(
        self,
        *,
        d_model: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        residual_scale_layers: int | None = None,
    ) -> None:
        """Initialize all shared layers and their depth-scaled residuals."""
        super().__init__()
        if num_layers <= 0:
            raise ValueError("Transformer trunk must contain at least one layer")
        scale_layers = (
            num_layers if residual_scale_layers is None else residual_scale_layers
        )
        if scale_layers <= 0:
            raise ValueError("Transformer residual-scale depth must be positive")
        residual_scale = 1.0 / math.sqrt(2.0 * scale_layers)
        self._d_model = d_model
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self._residual_scale = residual_scale
        self.layers = nn.ModuleList(
            PackedTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                residual_scale=residual_scale,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self._rollout_inductor_runner: _PackedRolloutRunner | None = None
        self._rollout_inductor_segments: tuple[_PackedRolloutSegment, ...] = ()
        self._rollout_inductor_callback_layers: tuple[int, ...] = ()
        self._rollout_inductor_failure: str | None = None
        self._learner_inductor_runner: _PackedRolloutRunner | None = None
        self._learner_inductor_segments: tuple[_PackedRolloutSegment, ...] = ()
        self._learner_inductor_callback_layers: tuple[int, ...] = ()
        self._learner_inductor_failure: str | None = None

    @property
    def supports_bfloat16_rollout_inductor(self) -> bool:
        """Return whether this Torch build provides the required CUDA kernel."""
        return varlen_attn is not None

    @property
    def uses_bfloat16_rollout_inductor(self) -> bool:
        """Return whether immutable BF16 rollout segments are enabled."""
        return (
            self._rollout_inductor_runner is not None
            and self._rollout_inductor_failure is None
        )

    def enable_bfloat16_rollout_inductor(
        self,
        *,
        layer_boundary_callback_layers: tuple[int, ...] = (),
    ) -> None:
        """Compile tensor-only BF16 CUDA segments for an immutable rollout clone."""
        self._validate_bfloat16_rollout_inductor_model()
        callback_layers = _validate_rollout_callback_layers(
            layer_boundary_callback_layers,
            num_layers=len(self.layers),
        )
        segment_ends = _rollout_segment_ends(
            num_layers=len(self.layers),
            callback_layers=callback_layers,
        )
        segments = _bind_rollout_segments(
            self.layers,
            segment_ends=segment_ends,
        )
        try:
            runner = _compile_packed_rollout_segment()
        except Exception as error:
            self._set_rollout_inductor_failure(error)
        self._rollout_inductor_runner = runner
        self._rollout_inductor_segments = segments
        self._rollout_inductor_callback_layers = callback_layers
        self._rollout_inductor_failure = None

    def disable_bfloat16_rollout_inductor(self) -> None:
        """Return to the ordinary eager trunk after an explicit opt-out."""
        self._rollout_inductor_runner = None
        self._rollout_inductor_segments = ()
        self._rollout_inductor_callback_layers = ()
        self._rollout_inductor_failure = None

    @property
    def uses_bfloat16_learner_inductor(self) -> bool:
        """Return whether trainable BF16 shared-backbone segments are enabled."""
        return (
            self._learner_inductor_runner is not None
            and self._learner_inductor_failure is None
        )

    def enable_bfloat16_learner_inductor(
        self,
        *,
        layer_boundary_callback_layers: tuple[int, ...] = (),
    ) -> None:
        """Compile differentiable BF16 segments around eager route callbacks."""
        self._validate_bfloat16_learner_inductor_model()
        callback_layers = _validate_rollout_callback_layers(
            layer_boundary_callback_layers,
            num_layers=len(self.layers),
        )
        segments = _bind_rollout_segments(
            self.layers,
            segment_ends=_rollout_segment_ends(
                num_layers=len(self.layers),
                callback_layers=callback_layers,
            ),
        )
        try:
            runner = _compile_packed_rollout_segment()
        except Exception as error:
            self._set_learner_inductor_failure(error)
        self._learner_inductor_runner = runner
        self._learner_inductor_segments = segments
        self._learner_inductor_callback_layers = callback_layers
        self._learner_inductor_failure = None

    def disable_bfloat16_learner_inductor(self) -> None:
        """Return the learner trunk to ordinary eager execution."""
        self._learner_inductor_runner = None
        self._learner_inductor_segments = ()
        self._learner_inductor_callback_layers = ()
        self._learner_inductor_failure = None

    def forward(
        self,
        batch: PackedTokenBatch,
        *,
        layer_boundary_callback: (
            Callable[[int, PackedTokenBatch], PackedTokenBatch] | None
        ) = None,
        normalize_output: bool = True,
    ) -> PackedTokenBatch:
        """Run shared blocks and optional additive transforms at their boundaries."""
        if self._learner_inductor_failure is not None:
            raise PackedLearnerInductorError(
                "packed BF16 learner Inductor is failed closed; call "
                "disable_bfloat16_learner_inductor() before using eager execution. "
                f"Original error: {self._learner_inductor_failure}"
            )
        if self._rollout_inductor_failure is not None:
            raise PackedRolloutInductorError(
                "packed BF16 rollout Inductor is failed closed; call "
                "disable_bfloat16_rollout_inductor() before using eager execution. "
                f"Original error: {self._rollout_inductor_failure}"
            )
        if self.training and self._learner_inductor_runner is not None:
            return self._forward_bfloat16_learner_inductor(
                batch,
                layer_boundary_callback=layer_boundary_callback,
                normalize_output=normalize_output,
            )
        if not self.training and self._rollout_inductor_runner is not None:
            return self._forward_bfloat16_rollout_inductor(
                batch,
                layer_boundary_callback=layer_boundary_callback,
                normalize_output=normalize_output,
            )
        for layer_number, layer in enumerate(self.layers, start=1):
            batch = layer(batch)
            if layer_boundary_callback is not None:
                batch = layer_boundary_callback(layer_number, batch)
        return self.normalize(batch) if normalize_output else batch

    def normalize(self, batch: PackedTokenBatch) -> PackedTokenBatch:
        """Apply the architecture-owned final normalization once."""
        return batch.with_tokens(
            preserve_cuda_bfloat16_activation(self.output_norm(batch.tokens))
        )

    def _forward_bfloat16_learner_inductor(
        self,
        batch: PackedTokenBatch,
        *,
        layer_boundary_callback: (
            Callable[[int, PackedTokenBatch], PackedTokenBatch] | None
        ),
        normalize_output: bool,
    ) -> PackedTokenBatch:
        """Run differentiable compiled segments with eager route boundaries."""
        runner = self._learner_inductor_runner
        if runner is None:
            raise RuntimeError("packed BF16 learner Inductor is not enabled")
        if (
            not batch.tokens.is_cuda
            or batch.tokens.dtype != torch.bfloat16
            or not batch.cu_seqlens.is_cuda
        ):
            raise TypeError(
                "packed BF16 learner Inductor requires CUDA BF16 packed inputs"
            )
        callback_layers = self._learner_inductor_callback_layers
        if callback_layers and layer_boundary_callback is None:
            raise RuntimeError(
                "packed BF16 learner callback is missing at compiled boundaries"
            )
        for end_layer, weights in self._learner_inductor_segments:
            try:
                tokens = runner(
                    batch.tokens,
                    batch.cu_seqlens,
                    batch.max_seqlen,
                    weights,
                    self._num_heads,
                    self._head_dim,
                    self._d_model,
                    self._residual_scale,
                    self.output_norm.eps,
                )
            except Exception as error:
                self._set_learner_inductor_failure(error)
            batch = batch.with_tokens(tokens)
            if end_layer in callback_layers:
                if layer_boundary_callback is None:
                    raise RuntimeError("compiled learner stage callback is missing")
                batch = layer_boundary_callback(end_layer, batch)
        return self.normalize(batch) if normalize_output else batch

    def _forward_bfloat16_rollout_inductor(
        self,
        batch: PackedTokenBatch,
        *,
        layer_boundary_callback: (
            Callable[[int, PackedTokenBatch], PackedTokenBatch] | None
        ),
        normalize_output: bool,
    ) -> PackedTokenBatch:
        """Run four-ish compiled segments while keeping route stages eager."""
        runner = self._rollout_inductor_runner
        if runner is None:
            raise RuntimeError("packed BF16 rollout Inductor is not enabled")
        if self.training:
            raise RuntimeError("packed BF16 rollout Inductor requires eval mode")
        if (
            not batch.tokens.is_cuda
            or batch.tokens.dtype != torch.bfloat16
            or not batch.cu_seqlens.is_cuda
        ):
            raise TypeError(
                "packed BF16 rollout Inductor requires CUDA BF16 packed inputs"
            )
        callback_layers = self._rollout_inductor_callback_layers
        if callback_layers and layer_boundary_callback is None:
            raise RuntimeError(
                "packed BF16 rollout callback is missing at compiled boundaries"
            )
        for end_layer, weights in self._rollout_inductor_segments:
            try:
                tokens = runner(
                    batch.tokens,
                    batch.cu_seqlens,
                    batch.max_seqlen,
                    weights,
                    self._num_heads,
                    self._head_dim,
                    self._d_model,
                    self._residual_scale,
                    self.output_norm.eps,
                )
            except Exception as error:
                self._set_rollout_inductor_failure(error)
            batch = batch.with_tokens(tokens)
            if end_layer in callback_layers:
                if layer_boundary_callback is None:
                    raise RuntimeError("compiled rollout stage callback is missing")
                batch = layer_boundary_callback(end_layer, batch)
        return self.normalize(batch) if normalize_output else batch

    def _validate_bfloat16_rollout_inductor_model(self) -> None:
        """Require the immutable pure-BF16 CUDA shadow contract."""
        if self.training:
            raise ValueError("packed BF16 rollout Inductor requires eval mode")
        parameters = tuple(self.parameters())
        floating = tuple(
            parameter for parameter in parameters if parameter.is_floating_point()
        )
        if not floating:
            raise ValueError("packed BF16 rollout trunk has no floating parameters")
        if any(parameter.requires_grad for parameter in floating):
            raise ValueError(
                "packed BF16 rollout Inductor requires frozen parameters"
            )
        if any(
            not parameter.is_cuda or parameter.dtype != torch.bfloat16
            for parameter in floating
        ):
            raise ValueError(
                "packed BF16 rollout Inductor requires CUDA BF16 parameters"
            )
        if varlen_attn is None:
            raise RuntimeError("packed BF16 rollout Inductor requires varlen_attn")

    def _validate_bfloat16_learner_inductor_model(self) -> None:
        """Require an FP32 CUDA master whose activations can run in BF16."""
        parameters = tuple(self.parameters())
        floating = tuple(
            parameter for parameter in parameters if parameter.is_floating_point()
        )
        if not floating:
            raise ValueError("packed BF16 learner trunk has no floating parameters")
        if any(
            not parameter.is_cuda or parameter.dtype != torch.float32
            for parameter in floating
        ):
            raise ValueError(
                "packed BF16 learner Inductor requires CUDA FP32 master parameters"
            )
        if varlen_attn is None:
            raise RuntimeError("packed BF16 learner Inductor requires varlen_attn")

    def _set_rollout_inductor_failure(self, error: Exception) -> NoReturn:
        """Latch one compilation/runtime failure without silently falling back."""
        failure = f"{type(error).__name__}: {error}"
        self._rollout_inductor_failure = failure
        raise PackedRolloutInductorError(
            "packed BF16 rollout Inductor failed and is now failed closed; "
            "explicitly call disable_bfloat16_rollout_inductor() to use eager "
            f"execution. Original error: {failure}"
        ) from error

    def _set_learner_inductor_failure(self, error: Exception) -> NoReturn:
        """Latch one learner compilation/runtime failure without fallback."""
        failure = f"{type(error).__name__}: {error}"
        self._learner_inductor_failure = failure
        raise PackedLearnerInductorError(
            "packed BF16 learner Inductor failed and is now failed closed; "
            "explicitly call disable_bfloat16_learner_inductor() to use eager "
            f"execution. Original error: {failure}"
        ) from error


def _validate_rollout_callback_layers(
    callback_layers: tuple[int, ...],
    *,
    num_layers: int,
) -> tuple[int, ...]:
    """Validate eager stage boundaries around compiled shared segments."""
    if (
        tuple(sorted(set(callback_layers))) != callback_layers
        or any(layer <= 0 or layer > num_layers for layer in callback_layers)
    ):
        raise ValueError(
            "packed rollout callback layers must be unique, ordered trunk layers"
        )
    return callback_layers


def _rollout_segment_ends(
    *,
    num_layers: int,
    callback_layers: tuple[int, ...],
) -> tuple[int, ...]:
    """Split at semantic callbacks and cap compile graph depth between them."""
    semantic_ends = callback_layers
    if not semantic_ends or semantic_ends[-1] != num_layers:
        semantic_ends += (num_layers,)
    segment_ends: list[int] = []
    start = 0
    for semantic_end in semantic_ends:
        while semantic_end - start > _ROLLOUT_INDUCTOR_MAX_SEGMENT_LAYERS:
            start += _ROLLOUT_INDUCTOR_MAX_SEGMENT_LAYERS
            segment_ends.append(start)
        if semantic_end != start:
            segment_ends.append(semantic_end)
            start = semantic_end
    return tuple(segment_ends)


def _bind_rollout_segments(
    layers: nn.ModuleList,
    *,
    segment_ends: tuple[int, ...],
) -> tuple[_PackedRolloutSegment, ...]:
    """Bind tensor parameters once for an immutable rollout shadow."""
    segments: list[_PackedRolloutSegment] = []
    start = 0
    for end in segment_ends:
        weights = tuple(
            tensor
            for layer in layers[start:end]
            for tensor in _rollout_block_tensors(
                cast(PackedTransformerBlock, layer)
            )
        )
        segments.append((end, weights))
        start = end
    return tuple(segments)


def _rollout_block_tensors(block: PackedTransformerBlock) -> tuple[Tensor, ...]:
    """Flatten one fixed block's affine tensors in compiled-kernel order."""
    feedforward_input = cast(nn.Linear, block.feedforward[0])
    feedforward_output = cast(nn.Linear, block.feedforward[2])
    return cast(
        tuple[Tensor, ...],
        (
            block.attention_norm.weight,
            block.attention_norm.bias,
            block.attention.qkv.weight,
            block.attention.qkv.bias,
            block.attention.output.weight,
            block.attention.output.bias,
            block.feedforward_norm.weight,
            block.feedforward_norm.bias,
            feedforward_input.weight,
            feedforward_input.bias,
            feedforward_output.weight,
            feedforward_output.bias,
        ),
    )


@cache
def _compile_packed_rollout_segment() -> _PackedRolloutRunner:
    """Return the process-wide parameterized packed Transformer runner."""
    return cast(
        _PackedRolloutRunner,
        torch.compile(
            _packed_rollout_segment_cuda,
            dynamic=True,
            fullgraph=True,
            mode="default",
        ),
    )


def _packed_rollout_segment_cuda(
    tokens: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    weights: tuple[Tensor, ...],
    num_heads: int,
    head_dim: int,
    d_model: int,
    residual_scale: float,
    layer_norm_eps: float,
) -> Tensor:
    """Run one immutable packed segment as tensor operations in one strict graph."""
    for offset in range(0, len(weights), 12):
        attention_input = functional.layer_norm(
            tokens,
            (d_model,),
            weights[offset],
            weights[offset + 1],
            layer_norm_eps,
        )
        qkv = functional.linear(
            attention_input,
            weights[offset + 2],
            weights[offset + 3],
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(-1, num_heads, head_dim)
        key = key.view(-1, num_heads, head_dim)
        value = value.view(-1, num_heads, head_dim)
        attended = cast(
            Tensor,
            varlen_attn(
                query,
                key,
                value,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
            ),
        )
        tokens = tokens + residual_scale * functional.linear(
            attended.reshape(-1, d_model),
            weights[offset + 4],
            weights[offset + 5],
        )
        feedforward_input = functional.layer_norm(
            tokens,
            (d_model,),
            weights[offset + 6],
            weights[offset + 7],
            layer_norm_eps,
        )
        feedforward_hidden = functional.gelu(
            functional.linear(
                feedforward_input,
                weights[offset + 8],
                weights[offset + 9],
            )
        )
        tokens = tokens + residual_scale * functional.linear(
            feedforward_hidden,
            weights[offset + 10],
            weights[offset + 11],
        )
    return tokens


def initialize_simple_stateless_module(module: nn.Module) -> None:
    """Apply the architecture's fresh orthogonal initialization in place."""
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.orthogonal_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.LayerNorm):
            if child.weight is not None:
                nn.init.ones_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def _reference_varlen_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    offsets: tuple[int, ...],
) -> Tensor:
    """Autograd-safe CPU/float32 reference for development and deployment."""
    outputs: list[Tensor] = []
    for start, end in pairwise(offsets):
        q_row = query[start:end].transpose(0, 1).unsqueeze(0)
        k_row = key[start:end].transpose(0, 1).unsqueeze(0)
        v_row = value[start:end].transpose(0, 1).unsqueeze(0)
        attended = functional.scaled_dot_product_attention(
            q_row,
            k_row,
            v_row,
            dropout_p=0.0,
            is_causal=False,
        )
        outputs.append(attended.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)
