"""Exact-deck capsules for compositional strategy architecture v4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn

from ptcg_rl.model.deck_lora import LowRankAdapter

ProjectionDomain = Literal["transformer", "policy"]


@dataclass(frozen=True)
class ProjectionShape:
    """Static shape and bias contract for one compositional projection."""

    in_features: int
    out_features: int
    has_bias: bool

    def __post_init__(self) -> None:
        """Reject invalid projection metadata at the construction boundary."""
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("projection feature dimensions must be positive")


class ExactOutputResidual(nn.Module):
    """Zero-output exact-deck bottleneck with an arbitrary output width."""

    def __init__(
        self,
        in_features: int,
        bottleneck_dim: int,
        out_features: int,
    ) -> None:
        """Build a normalized residual whose initial output is exactly zero."""
        super().__init__()
        if min(in_features, bottleneck_dim, out_features) <= 0:
            raise ValueError("exact residual dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)
        self.down = nn.Linear(in_features, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, out_features)
        _zero_linear(self.up)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return the exact residual over any leading input dimensions."""
        if int(inputs.shape[-1]) != self.in_features:
            raise ValueError(
                f"exact residual inputs must end in width {self.in_features}"
            )
        return cast(Tensor, self.up(self.activation(self.down(self.norm(inputs)))))


class ExactDeckCapsule(nn.Module):
    """All parameters owned exclusively by one immutable exact-deck lineage."""

    def __init__(
        self,
        *,
        d_model: int,
        basis_count: int,
        transformer_shapes: Mapping[str, ProjectionShape],
        policy_shapes: Mapping[str, ProjectionShape],
        transformer_exact_rank: int,
        policy_exact_rank: int,
        transformer_layer_keys: Sequence[str],
        option_set_width: int,
        option_adapter_bottleneck_dim: int,
        policy_global_bottleneck_dim: int,
        count_hidden_dim: int,
        value_bottleneck_dim: int,
    ) -> None:
        """Build one small, physically independent exact strategy subtree."""
        super().__init__()
        dimensions = (
            d_model,
            basis_count,
            transformer_exact_rank,
            policy_exact_rank,
            option_set_width,
            option_adapter_bottleneck_dim,
            policy_global_bottleneck_dim,
            count_hidden_dim,
            value_bottleneck_dim,
        )
        if min(dimensions) <= 0:
            raise ValueError("exact capsule dimensions must be positive")
        _validate_target_keys(transformer_shapes, label="transformer")
        _validate_target_keys(policy_shapes, label="policy")
        layer_keys = tuple(transformer_layer_keys)
        if not layer_keys or len(layer_keys) != len(set(layer_keys)):
            raise ValueError("transformer layer keys must be non-empty and unique")
        if any(not key or "." in key for key in layer_keys):
            raise ValueError("transformer layer keys must be stable module keys")

        self.transformer_residuals = _low_rank_bank(
            transformer_shapes,
            rank=transformer_exact_rank,
        )
        self.policy_residuals = _low_rank_bank(
            policy_shapes,
            rank=policy_exact_rank,
        )
        self.transformer_route_biases = _route_biases(
            transformer_shapes,
            basis_count=basis_count,
        )
        self.policy_route_biases = _route_biases(
            policy_shapes,
            basis_count=basis_count,
        )
        self.transformer_bias_offsets = _bias_offsets(transformer_shapes)
        self.policy_bias_offsets = _bias_offsets(policy_shapes)

        norm_offsets: dict[str, nn.Parameter] = {}
        film_offsets: dict[str, nn.Parameter] = {}
        for layer_key in layer_keys:
            for norm_name in ("norm1", "norm2"):
                norm_offsets[f"{layer_key}_{norm_name}_weight"] = nn.Parameter(
                    torch.zeros(d_model)
                )
                norm_offsets[f"{layer_key}_{norm_name}_bias"] = nn.Parameter(
                    torch.zeros(d_model)
                )
            for site in ("attention", "feedforward"):
                film_offsets[f"{layer_key}_{site}"] = nn.Parameter(
                    torch.zeros(2, d_model)
                )
        norm_offsets["output_weight"] = nn.Parameter(torch.zeros(d_model))
        norm_offsets["output_bias"] = nn.Parameter(torch.zeros(d_model))
        self.transformer_norm_offsets = nn.ParameterDict(norm_offsets)
        self.transformer_film_offsets = nn.ParameterDict(film_offsets)

        self.policy_film_offsets = nn.ParameterDict(
            {
                "option_attention": nn.Parameter(torch.zeros(2, option_set_width)),
                "option_feedforward": nn.Parameter(torch.zeros(2, option_set_width)),
            }
        )
        self.option_adapter = ExactOutputResidual(
            d_model,
            option_adapter_bottleneck_dim,
            d_model,
        )
        self.policy_global_residual = ExactOutputResidual(
            d_model,
            policy_global_bottleneck_dim,
            d_model,
        )
        self.count_set_projection = nn.Linear(d_model, count_hidden_dim)
        _zero_linear(self.count_set_projection)
        self.stop_delta = nn.Parameter(torch.zeros(d_model))
        self.value_calibrators = nn.ModuleDict(
            {
                "root": ExactOutputResidual(
                    d_model,
                    value_bottleneck_dim,
                    1,
                ),
                "prefix": ExactOutputResidual(
                    d_model,
                    value_bottleneck_dim,
                    1,
                ),
                "action_state": ExactOutputResidual(
                    d_model,
                    value_bottleneck_dim,
                    3,
                ),
                "action": ExactOutputResidual(
                    2 * d_model,
                    value_bottleneck_dim,
                    3,
                ),
            }
        )

    def residual(
        self,
        domain: ProjectionDomain,
        target: str,
    ) -> LowRankAdapter:
        """Return this capsule's exact factor for one projection target."""
        bank = self._residual_bank(domain)
        if target not in bank:
            raise KeyError(f"unknown {domain} capsule target: {target}")
        return cast(LowRankAdapter, bank[target])

    def route_bias(self, domain: ProjectionDomain, target: str) -> Tensor:
        """Return the signed shared-basis coefficient bias for one target."""
        bank = self._route_bias_bank(domain)
        if target not in bank:
            raise KeyError(f"unknown {domain} capsule route target: {target}")
        return cast(Tensor, bank[target])

    def bias_offset(self, domain: ProjectionDomain, target: str) -> Tensor | None:
        """Return a projection bias offset, or ``None`` for a biasless base."""
        bank = self._bias_offset_bank(domain)
        return cast(Tensor, bank[target]) if target in bank else None

    def _residual_bank(self, domain: ProjectionDomain) -> nn.ModuleDict:
        return (
            self.transformer_residuals
            if domain == "transformer"
            else self.policy_residuals
        )

    def _route_bias_bank(self, domain: ProjectionDomain) -> nn.ParameterDict:
        return (
            self.transformer_route_biases
            if domain == "transformer"
            else self.policy_route_biases
        )

    def _bias_offset_bank(self, domain: ProjectionDomain) -> nn.ParameterDict:
        return (
            self.transformer_bias_offsets
            if domain == "transformer"
            else self.policy_bias_offsets
        )


class FixedExactDeckCapsule(nn.Module):
    """Non-linear exact-deck state that cannot be folded into dense weights."""

    def __init__(
        self,
        *,
        d_model: int,
        option_adapter_bottleneck_dim: int,
        policy_global_bottleneck_dim: int,
        value_bottleneck_dim: int,
    ) -> None:
        """Build the minimal single-deck capsule retained by fixed export."""
        super().__init__()
        dimensions = (
            d_model,
            option_adapter_bottleneck_dim,
            policy_global_bottleneck_dim,
            value_bottleneck_dim,
        )
        if min(dimensions) <= 0:
            raise ValueError("fixed exact capsule dimensions must be positive")
        self.transformer_norm_offsets = nn.ParameterDict()
        self.transformer_film_offsets = nn.ParameterDict()
        self.policy_film_offsets = nn.ParameterDict()
        self.option_adapter = ExactOutputResidual(
            d_model,
            option_adapter_bottleneck_dim,
            d_model,
        )
        self.policy_global_residual = ExactOutputResidual(
            d_model,
            policy_global_bottleneck_dim,
            d_model,
        )
        self.value_calibrators = nn.ModuleDict(
            {
                "root": ExactOutputResidual(
                    d_model,
                    value_bottleneck_dim,
                    1,
                ),
                "prefix": ExactOutputResidual(
                    d_model,
                    value_bottleneck_dim,
                    1,
                ),
                "action_state": ExactOutputResidual(
                    d_model,
                    value_bottleneck_dim,
                    3,
                ),
                "action": ExactOutputResidual(
                    2 * d_model,
                    value_bottleneck_dim,
                    3,
                ),
            }
        )

    def residual(
        self,
        domain: ProjectionDomain,
        target: str,
    ) -> LowRankAdapter:
        """Reject access to factors already merged into the fixed dense base."""
        raise RuntimeError(
            f"fixed capsule has no {domain} projection residual {target!r}"
        )

    def route_bias(self, domain: ProjectionDomain, target: str) -> Tensor:
        """Reject access to router state removed by fixed export."""
        raise RuntimeError(f"fixed capsule has no {domain} route bias {target!r}")

    def bias_offset(self, domain: ProjectionDomain, target: str) -> Tensor | None:
        """Reject access to projection offsets already folded into biases."""
        raise RuntimeError(f"fixed capsule has no {domain} projection bias {target!r}")


def exact_capsule(
    capsules: nn.ModuleDict,
    module_key: str,
) -> ExactDeckCapsule | FixedExactDeckCapsule:
    """Return one validated exact capsule from the root-owned bank."""
    if module_key not in capsules:
        raise KeyError(f"unknown exact capsule route: {module_key}")
    capsule = capsules[module_key]
    if not isinstance(capsule, (ExactDeckCapsule, FixedExactDeckCapsule)):
        raise TypeError(f"route {module_key!r} is not an exact deck capsule")
    return capsule


def _low_rank_bank(
    shapes: Mapping[str, ProjectionShape],
    *,
    rank: int,
) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            target: LowRankAdapter(
                shape.in_features,
                shape.out_features,
                rank,
                alpha=float(rank),
            )
            for target, shape in sorted(shapes.items())
        }
    )


def _route_biases(
    shapes: Mapping[str, ProjectionShape],
    *,
    basis_count: int,
) -> nn.ParameterDict:
    return nn.ParameterDict(
        {target: nn.Parameter(torch.zeros(basis_count)) for target in sorted(shapes)}
    )


def _bias_offsets(
    shapes: Mapping[str, ProjectionShape],
) -> nn.ParameterDict:
    return nn.ParameterDict(
        {
            target: nn.Parameter(torch.zeros(shape.out_features))
            for target, shape in sorted(shapes.items())
            if shape.has_bias
        }
    )


def _validate_target_keys(
    shapes: Mapping[str, ProjectionShape],
    *,
    label: str,
) -> None:
    if not shapes:
        raise ValueError(f"{label} projection shapes must be non-empty")
    if any(not key or "." in key for key in shapes):
        raise ValueError(f"{label} target names must be stable module keys")


def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


__all__ = [
    "ExactDeckCapsule",
    "ExactOutputResidual",
    "FixedExactDeckCapsule",
    "ProjectionDomain",
    "ProjectionShape",
    "exact_capsule",
]
