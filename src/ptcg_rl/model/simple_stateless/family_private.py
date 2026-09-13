"""Routed family-private upper Transformer blocks for architecture v3."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from typing import cast

import torch
from torch import Tensor, nn

from ptcg_rl.decks.registry import DeckFamilyRoute
from ptcg_rl.model.simple_stateless.layers import (
    PackedLearnerInductorError,
    PackedRolloutInductorError,
    PackedTransformerBlock,
    _compile_packed_rollout_segment,
    _rollout_block_tensors,
    initialize_simple_stateless_module,
    varlen_attn,
)
from ptcg_rl.model.simple_stateless.packed import PackedTokenBatch
from ptcg_rl.model.simple_stateless.routing import SimpleExactRoutePlan

FAMILY_PRIVATE_SHARED_LAYERS = 15
FAMILY_PRIVATE_CLONED_LAYERS = 5
FAMILY_PRIVATE_APPENDED_LAYERS = 3
FAMILY_PRIVATE_INHERITED_LAYERS = (
    FAMILY_PRIVATE_SHARED_LAYERS + FAMILY_PRIVATE_CLONED_LAYERS
)
FAMILY_PRIVATE_ACTIVE_LAYERS = (
    FAMILY_PRIVATE_SHARED_LAYERS
    + FAMILY_PRIVATE_CLONED_LAYERS
    + FAMILY_PRIVATE_APPENDED_LAYERS
)

_PackedFamilyRunner = Callable[..., Tensor]


@dataclass(frozen=True)
class _CompiledFamilyTail:
    """Tensor-only bindings for one physical tail's two homogeneous stages."""

    cloned_weights: tuple[Tensor, ...]
    appended_weights: tuple[Tensor, ...]
    cloned_residual_scale: float
    appended_residual_scale: float
    layer_norm_eps: float


def _block(
    *,
    d_model: int,
    num_heads: int,
    feedforward_dim: int,
    residual_scale: float,
) -> PackedTransformerBlock:
    return PackedTransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        residual_scale=residual_scale,
    )


class FamilyPrivateTransformerTail(nn.Module):
    """Five inherited upper blocks followed by three identity-start blocks."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
    ) -> None:
        """Create one independently trainable strategy-family tail."""
        super().__init__()
        inherited_scale = 1.0 / math.sqrt(
            2.0 * FAMILY_PRIVATE_INHERITED_LAYERS
        )
        appended_scale = 1.0 / math.sqrt(
            2.0 * FAMILY_PRIVATE_ACTIVE_LAYERS
        )
        self.cloned_layers = nn.ModuleList(
            _block(
                d_model=d_model,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                residual_scale=inherited_scale,
            )
            for _ in range(FAMILY_PRIVATE_CLONED_LAYERS)
        )
        self.appended_layers = nn.ModuleList(
            _block(
                d_model=d_model,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                residual_scale=appended_scale,
            )
            for _ in range(FAMILY_PRIVATE_APPENDED_LAYERS)
        )

    def forward_cloned(self, batch: PackedTokenBatch) -> PackedTokenBatch:
        """Run the five blocks cloned from source layers 16 through 20."""
        for raw_layer in self.cloned_layers:
            layer = cast(PackedTransformerBlock, raw_layer)
            batch = layer(batch)
        return batch

    def forward_appended(self, batch: PackedTokenBatch) -> PackedTokenBatch:
        """Run the three newly appended family-private blocks."""
        for raw_layer in self.appended_layers:
            layer = cast(PackedTransformerBlock, raw_layer)
            batch = layer(batch)
        return batch

    def zero_appended_outputs(self) -> None:
        """Make every new residual block an exact identity at initialization."""
        with torch.no_grad():
            for raw_layer in self.appended_layers:
                layer = cast(PackedTransformerBlock, raw_layer)
                layer.attention.output.weight.zero_()
                if layer.attention.output.bias is not None:
                    layer.attention.output.bias.zero_()
                final = layer.feedforward[-1]
                if not isinstance(final, nn.Linear):
                    raise TypeError("family-private feedforward output must be linear")
                final.weight.zero_()
                if final.bias is not None:
                    final.bias.zero_()

    def initialize_appended_layers(self) -> None:
        """Initialize only target-new blocks, then restore identity startup."""
        initialize_simple_stateless_module(self.appended_layers)
        self.zero_appended_outputs()

    def inert_output_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        """Return the zero projections that prove strict identity startup."""
        names: list[tuple[str, nn.Parameter]] = []
        for index, raw_layer in enumerate(self.appended_layers):
            layer = cast(PackedTransformerBlock, raw_layer)
            attention_bias = layer.attention.output.bias
            if attention_bias is None:
                raise RuntimeError("family-private attention output requires a bias")
            names.extend(
                (
                    (
                        f"appended_layers.{index}.attention.output.weight",
                        cast(nn.Parameter, layer.attention.output.weight),
                    ),
                    (
                        f"appended_layers.{index}.attention.output.bias",
                        attention_bias,
                    ),
                )
            )
            final = layer.feedforward[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("family-private feedforward output must be linear")
            if final.bias is None:
                raise RuntimeError("family-private feedforward output requires a bias")
            names.extend(
                (
                    (
                        f"appended_layers.{index}.feedforward.2.weight",
                        cast(nn.Parameter, final.weight),
                    ),
                    (f"appended_layers.{index}.feedforward.2.bias", final.bias),
                )
            )
        return tuple(names)


class FamilyPrivateStrategyBank(nn.Module):
    """Share one upper tail across exact variants in each declared family."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        exact_module_keys_by_digest: dict[str, str],
        family_routes: tuple[DeckFamilyRoute, ...],
        include_generic_upper: bool,
    ) -> None:
        """Create unique physical family tails and an offline generic upper path."""
        super().__init__()
        self._exact_to_family: dict[str, str] = {}
        family_keys = tuple(sorted({route.module_key for route in family_routes}))
        for route in family_routes:
            exact_key = exact_module_keys_by_digest.get(route.deck_digest)
            if exact_key is None:
                raise ValueError("family route has no matching exact strategy")
            self._exact_to_family[exact_key] = route.module_key
        if len(self._exact_to_family) != len(family_routes):
            raise ValueError("family routes do not map every exact strategy")
        inherited_scale = 1.0 / math.sqrt(
            2.0 * FAMILY_PRIVATE_INHERITED_LAYERS
        )
        self.generic_upper = (
            nn.ModuleList(
                _block(
                    d_model=d_model,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    residual_scale=inherited_scale,
                )
                for _ in range(FAMILY_PRIVATE_CLONED_LAYERS)
            )
            if include_generic_upper
            else None
        )
        self.tails = nn.ModuleDict(
            {
                module_key: FamilyPrivateTransformerTail(
                    d_model=d_model,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                )
                for module_key in family_keys
            }
        )
        self._d_model = d_model
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self._compiled_tails: dict[str, _CompiledFamilyTail] = {}
        self._rollout_inductor_runner: _PackedFamilyRunner | None = None
        self._rollout_inductor_failure: str | None = None
        self._learner_inductor_runner: _PackedFamilyRunner | None = None
        self._learner_inductor_failure: str | None = None

    @property
    def family_module_keys(self) -> tuple[str, ...]:
        """Return the unique physical family lineages in state-dict order."""
        return tuple(self.tails.keys())

    @property
    def uses_bfloat16_rollout_inductor(self) -> bool:
        """Return whether all family tails use the shared rollout runner."""
        return (
            self._rollout_inductor_runner is not None
            and self._rollout_inductor_failure is None
        )

    def enable_bfloat16_rollout_inductor(self) -> None:
        """Bind every frozen BF16 family tail to one parameterized runner."""
        if self.training:
            raise ValueError("family-tail rollout Inductor requires eval mode")
        floating = tuple(
            parameter
            for parameter in self.parameters()
            if parameter.is_floating_point()
        )
        if not floating:
            raise ValueError("family-tail rollout Inductor has no parameters")
        if any(parameter.requires_grad for parameter in floating):
            raise ValueError("family-tail rollout Inductor requires frozen parameters")
        if any(
            not parameter.is_cuda or parameter.dtype != torch.bfloat16
            for parameter in floating
        ):
            raise ValueError(
                "family-tail rollout Inductor requires CUDA BF16 parameters"
            )
        if varlen_attn is None:
            raise RuntimeError("family-tail rollout Inductor requires varlen_attn")
        try:
            runner = _compile_packed_rollout_segment()
        except Exception as error:
            self._set_rollout_inductor_failure(error)
        self._compiled_tails = self._bind_compiled_tails()
        self._rollout_inductor_runner = runner
        self._rollout_inductor_failure = None

    def disable_bfloat16_rollout_inductor(self) -> None:
        """Return frozen family tails to explicit eager block execution."""
        self._rollout_inductor_runner = None
        self._rollout_inductor_failure = None
        if self._learner_inductor_runner is None:
            self._compiled_tails = {}

    @property
    def uses_bfloat16_learner_inductor(self) -> bool:
        """Return whether all family tails use the shared learner runner."""
        return (
            self._learner_inductor_runner is not None
            and self._learner_inductor_failure is None
        )

    def enable_bfloat16_learner_inductor(self) -> None:
        """Bind every FP32-master family tail to one differentiable runner."""
        floating = tuple(
            parameter
            for parameter in self.parameters()
            if parameter.is_floating_point()
        )
        if not floating:
            raise ValueError("family-tail learner Inductor has no parameters")
        if any(
            not parameter.is_cuda or parameter.dtype != torch.float32
            for parameter in floating
        ):
            raise ValueError(
                "family-tail learner Inductor requires CUDA FP32 parameters"
            )
        if varlen_attn is None:
            raise RuntimeError("family-tail learner Inductor requires varlen_attn")
        try:
            runner = _compile_packed_rollout_segment()
        except Exception as error:
            self._set_learner_inductor_failure(error)
        self._compiled_tails = self._bind_compiled_tails()
        self._learner_inductor_runner = runner
        self._learner_inductor_failure = None

    def disable_bfloat16_learner_inductor(self) -> None:
        """Return trainable family tails to explicit eager block execution."""
        self._learner_inductor_runner = None
        self._learner_inductor_failure = None
        if self._rollout_inductor_runner is None:
            self._compiled_tails = {}

    def zero_appended_outputs(self) -> None:
        """Make every newly appended family block exactly inert."""
        for tail in self.tails.values():
            if not isinstance(tail, FamilyPrivateTransformerTail):
                raise TypeError("family tail bank contains an unexpected module")
            tail.zero_appended_outputs()

    def initialize_appended_layers(self) -> None:
        """Initialize only state that is not inherited from the source policy."""
        for tail in self.tails.values():
            if not isinstance(tail, FamilyPrivateTransformerTail):
                raise TypeError("family tail bank contains an unexpected module")
            tail.initialize_appended_layers()

    def initialize_tail_from_generic(self, module_key: str) -> None:
        """Clone the neutral generic upper path into one new family lineage."""
        generic = self.generic_upper
        if generic is None:
            raise ValueError("new family initialization requires the generic upper path")
        if module_key not in self.tails:
            raise ValueError(f"unknown family tail module key: {module_key}")
        raw_tail = self.tails[module_key]
        if not isinstance(raw_tail, FamilyPrivateTransformerTail):
            raise ValueError(f"unknown family tail module key: {module_key}")
        raw_tail.cloned_layers.load_state_dict(generic.state_dict(), strict=True)
        raw_tail.zero_appended_outputs()
        generic_state = generic.state_dict()
        cloned_state = raw_tail.cloned_layers.state_dict()
        if generic_state.keys() != cloned_state.keys() or any(
            not torch.equal(cloned_state[name], value)
            for name, value in generic_state.items()
        ):
            raise RuntimeError("new family tail differs from the generic upper path")
        for name, parameter in raw_tail.inert_output_named_parameters():
            if parameter.is_meta or torch.count_nonzero(parameter).item() != 0:
                raise RuntimeError(
                    f"new family appended output is not inert: {module_key}.{name}"
                )

    def initialize_tail_from_family(
        self,
        module_key: str,
        *,
        source_module_key: str,
    ) -> None:
        """Clone one learned family tail into a distinct new lineage."""
        if module_key == source_module_key:
            raise ValueError("family tail initialization cannot clone itself")
        try:
            raw_target = self.tails[module_key]
        except KeyError as error:
            raise ValueError(
                f"unknown target family tail module key: {module_key}"
            ) from error
        try:
            raw_source = self.tails[source_module_key]
        except KeyError as error:
            raise ValueError(
                f"unknown source family tail module key: {source_module_key}"
            ) from error
        if not isinstance(raw_target, FamilyPrivateTransformerTail):
            raise ValueError(f"unknown target family tail module key: {module_key}")
        if not isinstance(raw_source, FamilyPrivateTransformerTail):
            raise ValueError(
                f"unknown source family tail module key: {source_module_key}"
            )
        raw_target.load_state_dict(raw_source.state_dict(), strict=True)
        source_state = raw_source.state_dict()
        target_state = raw_target.state_dict()
        if source_state.keys() != target_state.keys() or any(
            value.is_meta or not torch.equal(target_state[name], value)
            for name, value in source_state.items()
        ):
            raise RuntimeError("new family tail differs from its declared source")

    def inert_output_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        """Enumerate all new zero projections with stable state-dict names."""
        return tuple(
            (f"tails.{module_key}.{name}", parameter)
            for module_key, tail in self.tails.items()
            if isinstance(tail, FamilyPrivateTransformerTail)
            for name, parameter in tail.inert_output_named_parameters()
        )

    def apply_cloned(
        self,
        batch: PackedTokenBatch,
        *,
        route_plan: SimpleExactRoutePlan,
        allow_unrouted_rows: bool,
    ) -> PackedTokenBatch:
        """Run old layers 16-20 through family or offline generic ownership."""
        family_rows = self._family_rows(route_plan)
        self._raise_if_inductor_failed()
        contiguous = self._apply_contiguous_family_partition(
            batch,
            family_rows,
            appended=False,
        )
        if contiguous is not None:
            return contiguous
        covered: set[int] = set()
        for _family_key, rows in family_rows:
            covered.update(rows)
        output = batch.tokens
        for family_key, rows in family_rows:
            tail = self.tails[family_key]
            if not isinstance(tail, FamilyPrivateTransformerTail):
                raise TypeError("family tail bank contains an unexpected module")
            output = _apply_rows(
                batch,
                output,
                rows,
                partial(
                    self._apply_tail_stage,
                    family_key=family_key,
                    tail=tail,
                    appended=False,
                ),
            )
        unrouted = tuple(row for row in range(batch.batch_size) if row not in covered)
        if unrouted:
            if not allow_unrouted_rows:
                raise ValueError("family-private route does not cover the batch")
            if self.generic_upper is None:
                raise ValueError("fixed family-private model cannot route foreign rows")

            def apply_generic(selected: PackedTokenBatch) -> PackedTokenBatch:
                for raw_layer in self.generic_upper or ():
                    layer = cast(PackedTransformerBlock, raw_layer)
                    selected = layer(selected)
                return selected

            output = _apply_rows(batch, output, unrouted, apply_generic)
        return batch.with_tokens(output)

    def apply_appended(
        self,
        batch: PackedTokenBatch,
        *,
        route_plan: SimpleExactRoutePlan,
    ) -> PackedTokenBatch:
        """Run only exact family rows through the three newly appended blocks."""
        self._raise_if_inductor_failed()
        family_rows = self._family_rows(route_plan)
        contiguous = self._apply_contiguous_family_partition(
            batch,
            family_rows,
            appended=True,
        )
        if contiguous is not None:
            return contiguous
        output = batch.tokens
        for family_key, rows in family_rows:
            tail = self.tails[family_key]
            if not isinstance(tail, FamilyPrivateTransformerTail):
                raise TypeError("family tail bank contains an unexpected module")
            output = _apply_rows(
                batch,
                output,
                rows,
                partial(
                    self._apply_tail_stage,
                    family_key=family_key,
                    tail=tail,
                    appended=True,
                ),
            )
        return batch.with_tokens(output)

    def _apply_contiguous_family_partition(
        self,
        batch: PackedTokenBatch,
        family_rows: tuple[tuple[str, tuple[int, ...]], ...],
        *,
        appended: bool,
    ) -> PackedTokenBatch | None:
        """Run a complete route-contiguous partition without gather/scatter."""
        ordered = _contiguous_family_partition(
            family_rows,
            batch_size=batch.batch_size,
        )
        if ordered is None:
            return None
        outputs: list[Tensor] = []
        for family_key, rows in ordered:
            tail = self.tails[family_key]
            if not isinstance(tail, FamilyPrivateTransformerTail):
                raise TypeError("family tail bank contains an unexpected module")
            selected = _slice_packed_rows(
                batch,
                start_row=rows[0],
                end_row=rows[-1] + 1,
            )
            outputs.append(
                self._apply_tail_stage(
                    selected,
                    family_key=family_key,
                    tail=tail,
                    appended=appended,
                ).tokens
            )
        tokens = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        return batch.with_tokens(tokens)

    def _apply_tail_stage(
        self,
        batch: PackedTokenBatch,
        *,
        family_key: str,
        tail: FamilyPrivateTransformerTail,
        appended: bool,
    ) -> PackedTokenBatch:
        """Run one eager or parameterized-compiled homogeneous tail stage."""
        runner = (
            self._learner_inductor_runner
            if self.training
            else self._rollout_inductor_runner
        )
        if runner is None:
            return (
                tail.forward_appended(batch)
                if appended
                else tail.forward_cloned(batch)
            )
        if (
            not batch.tokens.is_cuda
            or batch.tokens.dtype != torch.bfloat16
            or not batch.cu_seqlens.is_cuda
        ):
            raise TypeError(
                "family-tail Inductor requires CUDA BF16 packed inputs"
            )
        compiled = self._compiled_tails.get(family_key)
        if compiled is None:
            raise RuntimeError("family-tail Inductor has no parameter binding")
        weights = (
            compiled.appended_weights if appended else compiled.cloned_weights
        )
        residual_scale = (
            compiled.appended_residual_scale
            if appended
            else compiled.cloned_residual_scale
        )
        try:
            tokens = runner(
                batch.tokens,
                batch.cu_seqlens,
                batch.max_seqlen,
                weights,
                self._num_heads,
                self._head_dim,
                self._d_model,
                residual_scale,
                compiled.layer_norm_eps,
            )
        except Exception as error:
            if self.training:
                self._set_learner_inductor_failure(error)
            self._set_rollout_inductor_failure(error)
        return batch.with_tokens(tokens)

    def _bind_compiled_tails(self) -> dict[str, _CompiledFamilyTail]:
        """Flatten each tail's parameters without registering duplicate state."""
        return {
            family_key: _bind_compiled_tail(tail)
            for family_key, tail in self.tails.items()
            if isinstance(tail, FamilyPrivateTransformerTail)
        }

    def _raise_if_inductor_failed(self) -> None:
        """Keep a compiler/runtime failure latched until explicit disable."""
        if self._learner_inductor_failure is not None:
            raise PackedLearnerInductorError(
                "family-tail learner Inductor is failed closed; call "
                "disable_bfloat16_learner_inductor() before eager execution. "
                f"Original error: {self._learner_inductor_failure}"
            )
        if self._rollout_inductor_failure is not None:
            raise PackedRolloutInductorError(
                "family-tail rollout Inductor is failed closed; call "
                "disable_bfloat16_rollout_inductor() before eager execution. "
                f"Original error: {self._rollout_inductor_failure}"
            )

    def _set_rollout_inductor_failure(self, error: Exception) -> None:
        """Latch one rollout compiler failure without eager fallback."""
        failure = f"{type(error).__name__}: {error}"
        self._rollout_inductor_failure = failure
        raise PackedRolloutInductorError(
            "family-tail rollout Inductor failed and is now failed closed; "
            f"original error: {failure}"
        ) from error

    def _set_learner_inductor_failure(self, error: Exception) -> None:
        """Latch one learner compiler failure without eager fallback."""
        failure = f"{type(error).__name__}: {error}"
        self._learner_inductor_failure = failure
        raise PackedLearnerInductorError(
            "family-tail learner Inductor failed and is now failed closed; "
            f"original error: {failure}"
        ) from error

    def _family_rows(
        self,
        route_plan: SimpleExactRoutePlan,
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        grouped: defaultdict[str, list[int]] = defaultdict(list)
        for exact_key, rows in route_plan.host_groups:
            family_key = self._exact_to_family.get(exact_key)
            if family_key is None or family_key not in self.tails:
                raise ValueError("exact route has no family-private tail")
            grouped[family_key].extend(rows)
        return tuple(
            (family_key, tuple(sorted(rows)))
            for family_key, rows in sorted(grouped.items())
        )


def _bind_compiled_tail(
    raw_tail: nn.Module,
) -> _CompiledFamilyTail:
    """Create one parameter binding after checking homogeneous block metadata."""
    if not isinstance(raw_tail, FamilyPrivateTransformerTail):
        raise TypeError("family tail bank contains an unexpected module")
    cloned = tuple(
        cast(PackedTransformerBlock, layer) for layer in raw_tail.cloned_layers
    )
    appended = tuple(
        cast(PackedTransformerBlock, layer) for layer in raw_tail.appended_layers
    )
    layers = cloned + appended
    epsilons = {
        float(norm.eps)
        for layer in layers
        for norm in (layer.attention_norm, layer.feedforward_norm)
    }
    if len(epsilons) != 1:
        raise ValueError("family-tail layer norms must use one epsilon")
    cloned_scales = {float(layer.residual_scale) for layer in cloned}
    appended_scales = {float(layer.residual_scale) for layer in appended}
    if len(cloned_scales) != 1 or len(appended_scales) != 1:
        raise ValueError("family-tail stages must use homogeneous residual scales")
    return _CompiledFamilyTail(
        cloned_weights=tuple(
            tensor for layer in cloned for tensor in _rollout_block_tensors(layer)
        ),
        appended_weights=tuple(
            tensor for layer in appended for tensor in _rollout_block_tensors(layer)
        ),
        cloned_residual_scale=next(iter(cloned_scales)),
        appended_residual_scale=next(iter(appended_scales)),
        layer_norm_eps=next(iter(epsilons)),
    )


def _apply_rows(
    source: PackedTokenBatch,
    output: Tensor,
    rows: tuple[int, ...],
    transform: Callable[[PackedTokenBatch], PackedTokenBatch],
) -> Tensor:
    """Transform selected packed sequences and scatter them into source order."""
    if not rows:
        return output
    sequences = tuple(
        source.tokens[source.offsets[row] : source.offsets[row + 1]]
        for row in rows
    )
    selected = PackedTokenBatch.from_sequences(sequences)
    transformed = transform(selected)
    if not isinstance(transformed, PackedTokenBatch):
        raise TypeError("packed row transform returned an invalid batch")
    indices = torch.cat(
        tuple(
            torch.arange(
                source.offsets[row],
                source.offsets[row + 1],
                device=source.tokens.device,
            )
            for row in rows
        )
    )
    return output.index_copy(0, indices, transformed.tokens)


def _contiguous_family_partition(
    family_rows: tuple[tuple[str, tuple[int, ...]], ...],
    *,
    batch_size: int,
) -> tuple[tuple[str, tuple[int, ...]], ...] | None:
    """Order family groups when they exactly partition rows into intervals."""
    if not family_rows:
        return None
    ordered = tuple(
        sorted(
            family_rows,
            key=lambda item: item[1][0] if item[1] else batch_size,
        )
    )
    next_row = 0
    for _family_key, rows in ordered:
        if (
            not rows
            or rows[0] != next_row
            or rows[-1] != next_row + len(rows) - 1
            or any(right != left + 1 for left, right in pairwise(rows))
        ):
            return None
        next_row += len(rows)
    return ordered if next_row == batch_size else None


def _slice_packed_rows(
    batch: PackedTokenBatch,
    *,
    start_row: int,
    end_row: int,
) -> PackedTokenBatch:
    """Return one contiguous packed row interval with a zero-copy token view."""
    if not 0 <= start_row < end_row <= batch.batch_size:
        raise ValueError("packed row interval is invalid")
    token_start = batch.offsets[start_row]
    token_end = batch.offsets[end_row]
    offsets = tuple(
        offset - token_start
        for offset in batch.offsets[start_row : end_row + 1]
    )
    return PackedTokenBatch(
        tokens=batch.tokens[token_start:token_end],
        cu_seqlens=(
            batch.cu_seqlens[start_row : end_row + 1] - token_start
        ),
        max_seqlen=max(right - left for left, right in pairwise(offsets)),
        offsets=offsets,
    )


__all__ = [
    "FAMILY_PRIVATE_ACTIVE_LAYERS",
    "FAMILY_PRIVATE_APPENDED_LAYERS",
    "FAMILY_PRIVATE_CLONED_LAYERS",
    "FAMILY_PRIVATE_INHERITED_LAYERS",
    "FAMILY_PRIVATE_SHARED_LAYERS",
    "FamilyPrivateStrategyBank",
    "FamilyPrivateTransformerTail",
]
