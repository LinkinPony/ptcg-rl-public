"""Dense exact-deck private strategy modules.

This module contains only route-independent neural-network building blocks.  A
caller selects the exact-deck module before invoking them; the modules therefore
have no dependency on deck registries, route plans, or legacy LoRA machinery.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Self, cast

import torch
from torch import Tensor, nn

DEFAULT_OPTION_SET_BOTTLENECK_DIM = 384
DEFAULT_OPTION_SET_ATTENTION_HEADS = 6
DEFAULT_OPTION_SET_FEEDFORWARD_DIM = 1536
DEFAULT_VALUE_HIDDEN_DIM = 512
DEFAULT_VALUE_BOTTLENECK_DIM = 256


class OptionSetInteractionBlock(nn.Module):
    """Return a set-equivariant option residual conditioned on state tokens.

    Options have no positional encoding.  ``option_valid_mask`` uses ``True``
    for real legal options, while ``state_padding_mask`` follows PyTorch's
    padding convention and uses ``True`` for ignored state tokens.  The final
    projection is zero initialized, so this block is output-inert at migration.
    """

    def __init__(
        self,
        d_model: int,
        *,
        bottleneck_dim: int = DEFAULT_OPTION_SET_BOTTLENECK_DIM,
        attention_heads: int = DEFAULT_OPTION_SET_ATTENTION_HEADS,
        feedforward_dim: int = DEFAULT_OPTION_SET_FEEDFORWARD_DIM,
        dropout: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the private option-set interaction block."""
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if bottleneck_dim % attention_heads != 0:
            raise ValueError("bottleneck_dim must be divisible by attention_heads")
        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.option_projection = nn.Linear(
            d_model,
            bottleneck_dim,
            device=device,
            dtype=dtype,
        )
        self.self_norm = nn.LayerNorm(
            bottleneck_dim,
            device=device,
            dtype=dtype,
        )
        self.self_attention = nn.MultiheadAttention(
            bottleneck_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        self.cross_norm = nn.LayerNorm(
            bottleneck_dim,
            device=device,
            dtype=dtype,
        )
        self.state_projection = nn.Linear(
            d_model,
            bottleneck_dim,
            device=device,
            dtype=dtype,
        )
        self.cross_attention = nn.MultiheadAttention(
            bottleneck_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        self.ffn_norm = nn.LayerNorm(
            bottleneck_dim,
            device=device,
            dtype=dtype,
        )
        self.ffn = nn.Sequential(
            nn.Linear(
                bottleneck_dim,
                feedforward_dim,
                device=device,
                dtype=dtype,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                feedforward_dim,
                bottleneck_dim,
                device=device,
                dtype=dtype,
            ),
        )
        self.residual_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(
            bottleneck_dim,
            d_model,
            device=device,
            dtype=dtype,
        )
        _zero_linear(self.output_projection)

    def forward(
        self,
        option_embeddings: Tensor,
        state_embeddings: Tensor,
        *,
        option_valid_mask: Tensor,
        state_padding_mask: Tensor,
        residual_modulators: Mapping[str, Callable[[Tensor], Tensor]] | None = None,
    ) -> Tensor:
        """Return a ``[B, M, D]`` residual for the legal option set."""
        _validate_option_state_inputs(
            option_embeddings,
            state_embeddings,
            option_valid_mask=option_valid_mask,
            state_padding_mask=state_padding_mask,
            d_model=self.d_model,
        )
        if int(option_embeddings.shape[1]) == 0:
            return torch.zeros_like(option_embeddings)

        option_hidden = cast(Tensor, self.option_projection(option_embeddings))
        option_hidden = _mask_sequence(option_hidden, option_valid_mask)
        safe_option_padding_mask = _safe_padding_mask(~option_valid_mask)
        self_inputs = self.self_norm(option_hidden)
        self_delta = cast(
            Tensor,
            self.self_attention(
                self_inputs,
                self_inputs,
                self_inputs,
                key_padding_mask=safe_option_padding_mask,
                need_weights=False,
            )[0],
        )
        if residual_modulators is not None:
            self_delta = _modulate_residual(
                self_delta,
                residual_modulators,
                site="attention",
            )
        option_hidden = option_hidden + self.residual_dropout(self_delta)
        option_hidden = _mask_sequence(option_hidden, option_valid_mask)

        state_hidden = cast(Tensor, self.state_projection(state_embeddings))
        safe_state_padding_mask = _safe_padding_mask(state_padding_mask)
        state_hidden = state_hidden.masked_fill(
            state_padding_mask.unsqueeze(-1),
            0.0,
        )
        cross_query = self.cross_norm(option_hidden)
        cross_delta = cast(
            Tensor,
            self.cross_attention(
                cross_query,
                state_hidden,
                state_hidden,
                key_padding_mask=safe_state_padding_mask,
                need_weights=False,
            )[0],
        )
        if residual_modulators is not None:
            cross_delta = _modulate_residual(
                cross_delta,
                residual_modulators,
                site="attention",
            )
        option_hidden = option_hidden + self.residual_dropout(cross_delta)
        option_hidden = _mask_sequence(option_hidden, option_valid_mask)

        ffn_delta = cast(Tensor, self.ffn(self.ffn_norm(option_hidden)))
        if residual_modulators is not None:
            ffn_delta = _modulate_residual(
                ffn_delta,
                residual_modulators,
                site="feedforward",
            )
        option_hidden = option_hidden + self.residual_dropout(ffn_delta)
        option_hidden = _mask_sequence(option_hidden, option_valid_mask)
        output = cast(Tensor, self.output_projection(option_hidden))
        return _mask_sequence(output, option_valid_mask)


class DensePrivatePolicyStrategy(nn.Module):
    """One deck's independent dense policy/count decision parameters.

    ``linears`` may contain any stable decision-target names.  In architecture
    v3 those normally include every formerly routed policy projection as well
    as the count-head projections selected by the caller.  Every supplied
    module and the legacy global adapter are deep-copied.
    """

    def __init__(
        self,
        linears: Mapping[str, nn.Linear],
        stop_embedding: Tensor,
        global_adapter: nn.Module,
        *,
        count_hidden_dim: int | None = None,
        option_set_bottleneck_dim: int = DEFAULT_OPTION_SET_BOTTLENECK_DIM,
        option_set_attention_heads: int = DEFAULT_OPTION_SET_ATTENTION_HEADS,
        option_set_feedforward_dim: int = DEFAULT_OPTION_SET_FEEDFORWARD_DIM,
        dropout: float = 0.0,
    ) -> None:
        """Clone the legacy decision path and add inert set-aware capacity."""
        super().__init__()
        if stop_embedding.ndim != 1 or int(stop_embedding.numel()) <= 0:
            raise ValueError("stop_embedding must be a non-empty rank-one tensor")
        if not linears:
            raise ValueError("linears must contain at least one decision projection")
        invalid_names = tuple(name for name in linears if not name or "." in name)
        if invalid_names:
            raise ValueError("linear target names must be non-empty module keys")
        if any(not isinstance(module, nn.Linear) for module in linears.values()):
            raise TypeError("every decision projection must be nn.Linear")

        d_model = int(stop_embedding.numel())
        resolved_count_hidden_dim = _resolve_count_hidden_dim(
            linears,
            requested=count_hidden_dim,
            d_model=d_model,
        )
        self.linears = nn.ModuleDict(
            {name: copy.deepcopy(module) for name, module in linears.items()}
        )
        self.stop_embedding = nn.Parameter(stop_embedding.detach().clone())
        self.global_adapter = copy.deepcopy(global_adapter)
        self.option_set = OptionSetInteractionBlock(
            d_model,
            bottleneck_dim=option_set_bottleneck_dim,
            attention_heads=option_set_attention_heads,
            feedforward_dim=option_set_feedforward_dim,
            dropout=dropout,
            device=stop_embedding.device,
            dtype=stop_embedding.dtype,
        )
        self.count_set_projection = nn.Linear(
            d_model,
            resolved_count_hidden_dim,
            device=stop_embedding.device,
            dtype=stop_embedding.dtype,
        )
        _zero_linear(self.count_set_projection)

    def decision_linear(self, target: str, inputs: Tensor) -> Tensor:
        """Apply one cloned decision projection by its stable target name."""
        if target not in self.linears:
            raise KeyError(f"unknown private policy target: {target}")
        return cast(Tensor, self.linears[target](inputs))

    def global_residual(self, global_embedding: Tensor) -> Tensor:
        """Return the cloned legacy dense global-policy residual."""
        return cast(Tensor, self.global_adapter(global_embedding))

    def option_delta(
        self,
        option_embeddings: Tensor,
        state_embeddings: Tensor,
        *,
        option_valid_mask: Tensor,
        state_padding_mask: Tensor,
    ) -> Tensor:
        """Return the zero-initialized set-aware option residual."""
        return cast(
            Tensor,
            self.option_set(
                option_embeddings,
                state_embeddings,
                option_valid_mask=option_valid_mask,
                state_padding_mask=state_padding_mask,
            ),
        )

    def count_set_hidden(
        self,
        option_embeddings: Tensor,
        *,
        option_valid_mask: Tensor,
    ) -> Tensor:
        """Return an invariant zero-initialized count-head hidden residual."""
        pooled = _masked_mean(option_embeddings, option_valid_mask)
        return cast(Tensor, self.count_set_projection(pooled))


class DensePrivateTransformerBlock(nn.Module):
    """One cloned upper Transformer layer and optional dense residual."""

    def __init__(
        self,
        layer: nn.TransformerEncoderLayer,
        residual: nn.Module | None = None,
    ) -> None:
        """Clone an upper layer and its legacy ordinary residual adapter."""
        super().__init__()
        self.layer = copy.deepcopy(layer)
        self.residual = copy.deepcopy(residual)

    def forward(self, inputs: Tensor, *, padding_mask: Tensor) -> Tensor:
        """Apply the private dense layer and its optional residual."""
        _validate_padding_mask(inputs, padding_mask)
        output = cast(
            Tensor,
            self.layer(
                inputs,
                src_key_padding_mask=padding_mask,
                is_causal=False,
            ),
        )
        if self.residual is not None:
            output = output + cast(Tensor, self.residual(output))
        return output


class DensePrivateTransformerStack(nn.Module):
    """A deck-private clone of the upper state Transformer and output norm."""

    def __init__(
        self,
        upper_layers: Sequence[nn.TransformerEncoderLayer],
        *,
        output_norm: nn.Module,
        residual_adapters: Sequence[nn.Module | None] | None = None,
    ) -> None:
        """Clone upper layers, aligned residual adapters, and the output norm."""
        super().__init__()
        if not upper_layers:
            raise ValueError("upper_layers must be non-empty")
        if residual_adapters is None:
            resolved_residuals: Sequence[nn.Module | None] = (None,) * len(upper_layers)
        else:
            if len(residual_adapters) != len(upper_layers):
                raise ValueError(
                    "residual_adapters must align one-to-one with upper_layers"
                )
            resolved_residuals = residual_adapters
        self.layers = nn.ModuleList(
            DensePrivateTransformerBlock(layer, residual)
            for layer, residual in zip(
                upper_layers,
                resolved_residuals,
                strict=True,
            )
        )
        self.output_norm = copy.deepcopy(output_norm)

    @classmethod
    def from_shared(
        cls,
        upper_layers: Sequence[nn.TransformerEncoderLayer],
        *,
        output_norm: nn.Module,
        residual_adapters: Sequence[nn.Module | None] | None = None,
    ) -> Self:
        """Construct a physically independent stack from shared modules."""
        return cls(
            upper_layers,
            output_norm=output_norm,
            residual_adapters=residual_adapters,
        )

    def forward(self, inputs: Tensor, *, padding_mask: Tensor) -> Tensor:
        """Apply every private upper layer, private norm, and padding mask."""
        _validate_padding_mask(inputs, padding_mask)
        output = inputs
        for block in self.layers:
            output = block(output, padding_mask=padding_mask)
        output = cast(Tensor, self.output_norm(output))
        return output.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class DensePrivateRootValueHead(nn.Module):
    """One deck's cloned root-value path plus zero-output dense residual."""

    def __init__(
        self,
        base_hidden: nn.Linear,
        base_output: nn.Linear,
        legacy_residual: nn.Module,
        *,
        base_activation: nn.Module | None = None,
        dense_hidden_dim: int = DEFAULT_VALUE_HIDDEN_DIM,
        dense_bottleneck_dim: int = DEFAULT_VALUE_BOTTLENECK_DIM,
    ) -> None:
        """Clone the scalar path without applying the root tanh."""
        super().__init__()
        _initialize_value_head(
            self,
            base_hidden,
            base_output,
            legacy_residual,
            base_activation=base_activation,
            dense_hidden_dim=dense_hidden_dim,
            dense_bottleneck_dim=dense_bottleneck_dim,
        )

    @classmethod
    def from_sequential(
        cls,
        base_path: nn.Sequential,
        legacy_residual: nn.Module,
        *,
        dense_hidden_dim: int = DEFAULT_VALUE_HIDDEN_DIM,
        dense_bottleneck_dim: int = DEFAULT_VALUE_BOTTLENECK_DIM,
    ) -> Self:
        """Clone Linear/activation/Linear modules from a legacy root path."""
        hidden, activation, output = _sequential_value_path(base_path)
        return cls(
            hidden,
            output,
            legacy_residual,
            base_activation=activation,
            dense_hidden_dim=dense_hidden_dim,
            dense_bottleneck_dim=dense_bottleneck_dim,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return a squeezed private root pre-tanh scalar."""
        return _forward_value_head(self, inputs)


class DensePrivatePrefixValueHead(nn.Module):
    """One deck's cloned prefix-delta path plus dense residual capacity."""

    def __init__(
        self,
        base_hidden: nn.Linear,
        base_output: nn.Linear,
        legacy_residual: nn.Module,
        *,
        base_activation: nn.Module | None = None,
        dense_hidden_dim: int = DEFAULT_VALUE_HIDDEN_DIM,
        dense_bottleneck_dim: int = DEFAULT_VALUE_BOTTLENECK_DIM,
    ) -> None:
        """Clone the scalar delta path and initialize an inert dense branch."""
        super().__init__()
        _initialize_value_head(
            self,
            base_hidden,
            base_output,
            legacy_residual,
            base_activation=base_activation,
            dense_hidden_dim=dense_hidden_dim,
            dense_bottleneck_dim=dense_bottleneck_dim,
        )

    @classmethod
    def from_sequential(
        cls,
        base_path: nn.Sequential,
        legacy_residual: nn.Module,
        *,
        dense_hidden_dim: int = DEFAULT_VALUE_HIDDEN_DIM,
        dense_bottleneck_dim: int = DEFAULT_VALUE_BOTTLENECK_DIM,
    ) -> Self:
        """Clone Linear/activation/Linear modules from a legacy prefix path."""
        hidden, activation, output = _sequential_value_path(base_path)
        return cls(
            hidden,
            output,
            legacy_residual,
            base_activation=activation,
            dense_hidden_dim=dense_hidden_dim,
            dense_bottleneck_dim=dense_bottleneck_dim,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return a squeezed private prefix-value delta."""
        return _forward_value_head(self, inputs)


def _initialize_value_head(
    module: nn.Module,
    base_hidden: nn.Linear,
    base_output: nn.Linear,
    legacy_residual: nn.Module,
    *,
    base_activation: nn.Module | None,
    dense_hidden_dim: int,
    dense_bottleneck_dim: int,
) -> None:
    """Install common value modules while preserving stable state key names."""
    if base_hidden.in_features <= 0 or base_hidden.out_features <= 0:
        raise ValueError("base_hidden dimensions must be positive")
    if base_output.in_features != base_hidden.out_features:
        raise ValueError("base value projections must have aligned hidden dimensions")
    if base_output.out_features != 1:
        raise ValueError("base_output must produce one scalar")
    if dense_hidden_dim <= 0 or dense_bottleneck_dim <= 0:
        raise ValueError("dense residual dimensions must be positive")

    module.add_module("base_hidden", copy.deepcopy(base_hidden))
    module.add_module(
        "base_activation",
        copy.deepcopy(base_activation) if base_activation is not None else nn.GELU(),
    )
    module.add_module("base_output", copy.deepcopy(base_output))
    module.add_module("legacy_residual", copy.deepcopy(legacy_residual))
    device = base_hidden.weight.device
    dtype = base_hidden.weight.dtype
    dense_residual = nn.Sequential(
        nn.LayerNorm(base_hidden.in_features, device=device, dtype=dtype),
        nn.Linear(
            base_hidden.in_features,
            dense_hidden_dim,
            device=device,
            dtype=dtype,
        ),
        nn.GELU(),
        nn.Linear(
            dense_hidden_dim,
            dense_bottleneck_dim,
            device=device,
            dtype=dtype,
        ),
        nn.GELU(),
        nn.Linear(dense_bottleneck_dim, 1, device=device, dtype=dtype),
    )
    _zero_linear(cast(nn.Linear, dense_residual[-1]))
    module.add_module("dense_residual", dense_residual)


def _forward_value_head(module: nn.Module, inputs: Tensor) -> Tensor:
    """Evaluate the common cloned base, legacy, and dense value branches."""
    base_hidden = module.get_submodule("base_hidden")
    base_activation = module.get_submodule("base_activation")
    base_output = module.get_submodule("base_output")
    legacy_residual = module.get_submodule("legacy_residual")
    dense_residual = module.get_submodule("dense_residual")
    hidden = base_activation(base_hidden(inputs))
    output = base_output(hidden)
    output = output + legacy_residual(inputs) + dense_residual(inputs)
    if int(output.shape[-1]) != 1:
        raise RuntimeError("private value head branches must produce one scalar")
    return cast(Tensor, output.squeeze(-1))


def _sequential_value_path(
    base_path: nn.Sequential,
) -> tuple[nn.Linear, nn.Module, nn.Linear]:
    """Extract the legacy two-Linear value path without its final transform."""
    if len(base_path) < 3:
        raise ValueError("base_path must contain Linear, activation, and Linear")
    hidden = base_path[0]
    activation = base_path[1]
    output = base_path[2]
    if not isinstance(hidden, nn.Linear) or not isinstance(output, nn.Linear):
        raise TypeError("base_path positions 0 and 2 must be nn.Linear")
    return hidden, activation, output


def _resolve_count_hidden_dim(
    linears: Mapping[str, nn.Linear],
    *,
    requested: int | None,
    d_model: int,
) -> int:
    """Infer the count residual width when the caller does not provide it."""
    if requested is not None:
        if requested <= 0:
            raise ValueError("count_hidden_dim must be positive")
        return requested
    count_state = linears.get("count_state_projection")
    if count_state is not None:
        return count_state.out_features
    return d_model


def _masked_mean(inputs: Tensor, valid_mask: Tensor) -> Tensor:
    """Return a permutation-invariant mean, with zero for an empty set."""
    if inputs.ndim != 3:
        raise ValueError("option embeddings must have shape [B, M, D]")
    if valid_mask.dtype != torch.bool:
        raise TypeError("option_valid_mask must have dtype bool")
    if tuple(valid_mask.shape) != tuple(inputs.shape[:2]):
        raise ValueError("option_valid_mask must have shape [B, M]")
    weights = valid_mask.to(dtype=inputs.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (inputs * weights).sum(dim=1) / denominator


def _mask_sequence(inputs: Tensor, valid_mask: Tensor) -> Tensor:
    """Zero invalid sequence positions using a True-is-valid mask."""
    return inputs.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


def _modulate_residual(
    inputs: Tensor,
    modulators: Mapping[str, Callable[[Tensor], Tensor]],
    *,
    site: str,
) -> Tensor:
    """Apply one required external residual-boundary modulation."""
    if site not in modulators:
        raise KeyError(f"missing option-set residual modulator: {site}")
    output = modulators[site](inputs)
    if output.shape != inputs.shape:
        raise ValueError("option-set residual modulator changed tensor shape")
    return output


def _safe_padding_mask(padding_mask: Tensor) -> Tensor:
    """Avoid all-masked attention rows while retaining zero-valued sentinels."""
    if int(padding_mask.shape[1]) == 0:
        return padding_mask
    all_padding = padding_mask.all(dim=1)
    safe = padding_mask.clone()
    safe[:, 0] = safe[:, 0] & ~all_padding
    return safe


def _validate_option_state_inputs(
    option_embeddings: Tensor,
    state_embeddings: Tensor,
    *,
    option_valid_mask: Tensor,
    state_padding_mask: Tensor,
    d_model: int,
) -> None:
    """Validate explicit option/state tensor and mask contracts."""
    if option_embeddings.ndim != 3:
        raise ValueError("option_embeddings must have shape [B, M, D]")
    if state_embeddings.ndim != 3:
        raise ValueError("state_embeddings must have shape [B, S, D]")
    if int(option_embeddings.shape[0]) != int(state_embeddings.shape[0]):
        raise ValueError("option and state batches must align")
    if int(option_embeddings.shape[2]) != d_model:
        raise ValueError("option embedding width does not match d_model")
    if int(state_embeddings.shape[2]) != d_model:
        raise ValueError("state embedding width does not match d_model")
    if option_valid_mask.dtype != torch.bool:
        raise TypeError("option_valid_mask must have dtype bool")
    if state_padding_mask.dtype != torch.bool:
        raise TypeError("state_padding_mask must have dtype bool")
    if tuple(option_valid_mask.shape) != tuple(option_embeddings.shape[:2]):
        raise ValueError("option_valid_mask must have shape [B, M]")
    if tuple(state_padding_mask.shape) != tuple(state_embeddings.shape[:2]):
        raise ValueError("state_padding_mask must have shape [B, S]")


def _validate_padding_mask(inputs: Tensor, padding_mask: Tensor) -> None:
    """Validate a Transformer batch-first padding mask."""
    if inputs.ndim != 3:
        raise ValueError("Transformer inputs must have shape [B, S, D]")
    if padding_mask.dtype != torch.bool:
        raise TypeError("padding_mask must have dtype bool")
    if tuple(padding_mask.shape) != tuple(inputs.shape[:2]):
        raise ValueError("padding_mask must have shape [B, S]")


def _zero_linear(layer: nn.Linear) -> None:
    """Initialize a Linear module to return exact zeros."""
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


__all__ = [
    "DEFAULT_OPTION_SET_ATTENTION_HEADS",
    "DEFAULT_OPTION_SET_BOTTLENECK_DIM",
    "DEFAULT_OPTION_SET_FEEDFORWARD_DIM",
    "DEFAULT_VALUE_BOTTLENECK_DIM",
    "DEFAULT_VALUE_HIDDEN_DIM",
    "DensePrivatePolicyStrategy",
    "DensePrivatePrefixValueHead",
    "DensePrivateRootValueHead",
    "DensePrivateTransformerBlock",
    "DensePrivateTransformerStack",
    "OptionSetInteractionBlock",
]
