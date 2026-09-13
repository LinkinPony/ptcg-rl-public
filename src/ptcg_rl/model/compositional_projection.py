"""Shared-basis projections and conditioning for DCCR architecture v4."""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from ptcg_rl.model.compositional_capsule import (
    ExactDeckCapsule,
    ProjectionDomain,
    exact_capsule,
)
from ptcg_rl.model.deck_conditioning import DeckRoutePlan
from ptcg_rl.model.deck_lora import routed_low_rank_contributions

FilmDomain = Literal["transformer", "policy"]


class SharedCompositionalLinear(nn.Module):
    """Combine shared low-rank bases and one exact-deck residual."""

    def __init__(
        self,
        *,
        target: str,
        domain: ProjectionDomain,
        in_features: int,
        out_features: int,
        deck_dim: int,
        basis_count: int,
        shared_rank: int,
        router_hidden_dim: int,
    ) -> None:
        """Build one mergeable projection delta and its content router."""
        super().__init__()
        dimensions = (
            in_features,
            out_features,
            deck_dim,
            basis_count,
            shared_rank,
            router_hidden_dim,
        )
        if min(dimensions) <= 0:
            raise ValueError("compositional projection dimensions must be positive")
        if not target or "." in target:
            raise ValueError("compositional target must be a stable module key")
        self.target = target
        self.domain = domain
        self.in_features = in_features
        self.out_features = out_features
        self.deck_dim = deck_dim
        self.basis_count = basis_count
        self.shared_rank = shared_rank
        self.shared_a = nn.Parameter(torch.empty(basis_count, shared_rank, in_features))
        self.shared_b = nn.Parameter(
            torch.zeros(basis_count, out_features, shared_rank)
        )
        self.router_norm = nn.LayerNorm(deck_dim)
        self.router = nn.Sequential(
            nn.Linear(deck_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, basis_count),
        )
        nn.init.kaiming_uniform_(self.shared_a, a=5**0.5)
        _zero_linear(cast(nn.Linear, self.router[-1]))

    def forward(
        self,
        base: nn.Linear,
        inputs: Tensor,
        *,
        route_plan: DeckRoutePlan,
        capsules: nn.ModuleDict,
    ) -> Tensor:
        """Apply the external dense base and every compositional contribution."""
        if (
            base.in_features != self.in_features
            or base.out_features != self.out_features
        ):
            raise ValueError("base Linear shape does not match compositional target")
        return self.add_delta(
            cast(Tensor, base(inputs)),
            inputs,
            route_plan=route_plan,
            capsules=capsules,
        )

    def add_delta(
        self,
        output: Tensor,
        inputs: Tensor,
        *,
        route_plan: DeckRoutePlan,
        capsules: nn.ModuleDict,
    ) -> Tensor:
        """Add shared bases, the selected exact factor, and exact bias offset."""
        self._validate_inputs(output, inputs, route_plan)
        deck_embeddings = _required_deck_embeddings(route_plan, width=self.deck_dim)
        coefficients = self.coefficients(
            deck_embeddings,
            route_plan=route_plan,
            capsules=capsules,
        )
        result = output + self.shared_delta(inputs, coefficients).to(dtype=output.dtype)
        result = result + self._exact_delta(
            inputs,
            route_plan=route_plan,
            capsules=capsules,
            output_dtype=output.dtype,
        )
        bias = self._exact_bias(
            inputs,
            route_plan=route_plan,
            capsules=capsules,
            output_dtype=output.dtype,
        )
        if bias is not None:
            result = result + bias
        return result

    def coefficients(
        self,
        deck_embeddings: Tensor,
        *,
        route_plan: DeckRoutePlan,
        capsules: nn.ModuleDict,
    ) -> Tensor:
        """Return bounded signed coefficients for every batch row."""
        if tuple(deck_embeddings.shape) != (
            route_plan.batch_size,
            self.deck_dim,
        ):
            raise ValueError("deck embeddings must have shape [B, deck_dim]")
        raw = cast(Tensor, self.router(self.router_norm(deck_embeddings)))
        exact_bias = torch.zeros_like(raw)
        row_indices: list[Tensor] = []
        contributions: list[Tensor] = []
        for group in route_plan.groups:
            capsule = exact_capsule(capsules, group.module_key)
            row_indices.append(group.row_indices)
            contributions.append(
                capsule.route_bias(self.domain, self.target)
                .unsqueeze(0)
                .expand(int(group.row_indices.numel()), -1)
                .to(device=raw.device, dtype=raw.dtype)
            )
        if contributions:
            exact_bias = exact_bias.index_copy(
                0,
                torch.cat(row_indices),
                torch.cat(contributions),
            )
        return 2.0 * torch.tanh((raw + exact_bias) / 2.0)

    def shared_delta(self, inputs: Tensor, coefficients: Tensor) -> Tensor:
        """Evaluate all shared bases as two dense GEMMs and row-wise scaling."""
        if int(inputs.shape[-1]) != self.in_features:
            raise ValueError(
                f"compositional inputs must end in width {self.in_features}"
            )
        if tuple(coefficients.shape) != (
            int(inputs.shape[0]),
            self.basis_count,
        ):
            raise ValueError("coefficients must have shape [B, basis_count]")
        a_cat = self.shared_a.reshape(
            self.basis_count * self.shared_rank,
            self.in_features,
        )
        b_cat = self.shared_b.permute(1, 0, 2).reshape(
            self.out_features,
            self.basis_count * self.shared_rank,
        )
        hidden = functional.linear(inputs, a_cat)
        row_scales = coefficients.repeat_interleave(self.shared_rank, dim=-1).to(
            dtype=hidden.dtype
        )
        scale_shape = (
            int(inputs.shape[0]),
            *((1,) * (inputs.ndim - 2)),
            self.basis_count * self.shared_rank,
        )
        hidden = hidden * row_scales.reshape(scale_shape)
        return functional.linear(hidden, b_cat)

    def materialized_delta_weight(
        self,
        deck_embedding: Tensor,
        capsule: ExactDeckCapsule,
    ) -> Tensor:
        """Return one route's FP32 merge-ready effective weight delta."""
        if tuple(deck_embedding.shape) not in {
            (self.deck_dim,),
            (1, self.deck_dim),
        }:
            raise ValueError("materialization requires one deck embedding")
        embedding = deck_embedding.reshape(1, self.deck_dim).float()
        module_dtype = self.shared_a.dtype
        raw = cast(
            Tensor,
            self.router(self.router_norm(embedding.to(dtype=module_dtype))),
        ).float()
        raw = raw + capsule.route_bias(self.domain, self.target).float().unsqueeze(0)
        coefficients = 2.0 * torch.tanh(raw / 2.0)
        shared = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.shared_a.device,
            dtype=torch.float32,
        )
        for basis_index in range(self.basis_count):
            shared = shared + coefficients[0, basis_index] * torch.matmul(
                self.shared_b[basis_index].float(),
                self.shared_a[basis_index].float(),
            )
        return (
            shared
            + capsule.residual(
                self.domain,
                self.target,
            )
            .delta_weight()
            .float()
        )

    def _exact_delta(
        self,
        inputs: Tensor,
        *,
        route_plan: DeckRoutePlan,
        capsules: nn.ModuleDict,
        output_dtype: torch.dtype,
    ) -> Tensor:
        residual = torch.zeros(
            (*inputs.shape[:-1], self.out_features),
            device=inputs.device,
            dtype=output_dtype,
        )
        dispatch = route_plan.lora_dispatch
        if dispatch is not None:
            adapters = tuple(
                exact_capsule(capsules, module_key).residual(
                    self.domain,
                    self.target,
                )
                for module_key in dispatch.module_keys
            )
            dispatch_rows, contributions = routed_low_rank_contributions(
                inputs,
                dispatch,
                adapters,
                cache_owner=self,
            )
            if dispatch.contiguous_prefix:
                residual.narrow(0, 0, int(dispatch_rows.numel())).copy_(
                    contributions.to(dtype=output_dtype)
                )
                return residual
            return residual.index_copy(
                0,
                dispatch_rows,
                contributions.to(dtype=output_dtype),
            )
        row_indices: list[Tensor] = []
        contribution_rows: list[Tensor] = []
        for group in route_plan.groups:
            capsule = exact_capsule(capsules, group.module_key)
            row_indices.append(group.row_indices)
            contribution_rows.append(
                capsule.residual(self.domain, self.target)(
                    inputs.index_select(0, group.row_indices)
                ).to(dtype=output_dtype)
            )
        if not contribution_rows:
            return residual
        return residual.index_copy(
            0,
            torch.cat(row_indices),
            torch.cat(contribution_rows),
        )

    def _exact_bias(
        self,
        inputs: Tensor,
        *,
        route_plan: DeckRoutePlan,
        capsules: nn.ModuleDict,
        output_dtype: torch.dtype,
    ) -> Tensor | None:
        row_indices: list[Tensor] = []
        contributions: list[Tensor] = []
        for group in route_plan.groups:
            capsule = exact_capsule(capsules, group.module_key)
            offset = capsule.bias_offset(self.domain, self.target)
            if offset is None:
                continue
            row_indices.append(group.row_indices)
            contributions.append(
                offset.unsqueeze(0)
                .expand(int(group.row_indices.numel()), -1)
                .to(device=inputs.device, dtype=output_dtype)
            )
        if not contributions:
            return None
        batch_bias = torch.zeros(
            (route_plan.batch_size, self.out_features),
            device=inputs.device,
            dtype=output_dtype,
        ).index_copy(
            0,
            torch.cat(row_indices),
            torch.cat(contributions),
        )
        return batch_bias.reshape(
            route_plan.batch_size,
            *((1,) * (inputs.ndim - 2)),
            self.out_features,
        )

    def _validate_inputs(
        self,
        output: Tensor,
        inputs: Tensor,
        route_plan: DeckRoutePlan,
    ) -> None:
        if inputs.ndim < 2 or int(inputs.shape[-1]) != self.in_features:
            raise ValueError(
                f"compositional inputs must have shape [B, ..., {self.in_features}]"
            )
        if int(inputs.shape[0]) != route_plan.batch_size:
            raise ValueError("compositional inputs must align with route plan")
        expected = (*inputs.shape[:-1], self.out_features)
        if tuple(output.shape) != expected:
            raise ValueError(
                f"compositional output must have shape {expected}, got {output.shape}"
            )


class FixedCompositionalLinear(nn.Module):
    """Parameter-free marker for a projection merged into its dense base."""

    def __init__(self, *, in_features: int, out_features: int) -> None:
        """Record the strict external base shape used by fixed execution."""
        super().__init__()
        if min(in_features, out_features) <= 0:
            raise ValueError("fixed projection dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features

    def forward(
        self,
        base: nn.Linear,
        inputs: Tensor,
        *,
        route_plan: DeckRoutePlan,
        capsules: nn.ModuleDict,
    ) -> Tensor:
        """Evaluate the already-merged external dense projection."""
        output = cast(Tensor, base(inputs))
        return self.add_delta(
            output,
            inputs,
            route_plan=route_plan,
            capsules=capsules,
        )

    def add_delta(
        self,
        output: Tensor,
        inputs: Tensor,
        *,
        route_plan: DeckRoutePlan,
        capsules: nn.ModuleDict,
    ) -> Tensor:
        """Validate alignment and return the already-merged output unchanged."""
        del capsules
        if inputs.ndim < 2 or int(inputs.shape[-1]) != self.in_features:
            raise ValueError("fixed compositional projection input shape is invalid")
        if int(inputs.shape[0]) != route_plan.batch_size:
            raise ValueError("fixed compositional inputs must align with route plan")
        expected = (*inputs.shape[:-1], self.out_features)
        if tuple(output.shape) != expected:
            raise ValueError("fixed compositional projection output shape is invalid")
        return output


class DeckConditionedFiLM(nn.Module):
    """Apply content-generated and exact-offset affine conditioning."""

    def __init__(
        self,
        d_model: int,
        deck_dim: int,
        hidden_dim: int,
        *,
        domain: FilmDomain,
        site: str,
    ) -> None:
        """Build an initially inert deck-conditioned affine generator."""
        super().__init__()
        if min(d_model, deck_dim, hidden_dim) <= 0:
            raise ValueError("FiLM dimensions must be positive")
        if not site or "." in site:
            raise ValueError("FiLM site must be a stable module key")
        self.d_model = d_model
        self.deck_dim = deck_dim
        self.domain = domain
        self.site = site
        self.norm = nn.LayerNorm(deck_dim)
        self.generator = nn.Sequential(
            nn.Linear(deck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * d_model),
        )
        _zero_linear(cast(nn.Linear, self.generator[-1]))

    def forward(
        self,
        inputs: Tensor,
        *,
        route_plan: DeckRoutePlan,
        capsules: nn.ModuleDict,
    ) -> Tensor:
        """Apply ``(1 + gamma) * inputs + beta`` for each routed row."""
        if inputs.ndim < 2 or int(inputs.shape[-1]) != self.d_model:
            raise ValueError(f"FiLM inputs must end in width {self.d_model}")
        if int(inputs.shape[0]) != route_plan.batch_size:
            raise ValueError("FiLM inputs must align with route plan")
        deck_embeddings = _required_deck_embeddings(route_plan, width=self.deck_dim)
        generated = cast(
            Tensor,
            self.generator(self.norm(deck_embeddings)),
        ).reshape(route_plan.batch_size, 2, self.d_model)
        offsets = torch.zeros_like(generated)
        row_indices: list[Tensor] = []
        contributions: list[Tensor] = []
        for group in route_plan.groups:
            capsule = exact_capsule(capsules, group.module_key)
            bank = (
                capsule.transformer_film_offsets
                if self.domain == "transformer"
                else capsule.policy_film_offsets
            )
            if self.site not in bank:
                raise KeyError(f"missing exact FiLM site {self.site!r}")
            row_indices.append(group.row_indices)
            contributions.append(
                bank[self.site]
                .unsqueeze(0)
                .expand(int(group.row_indices.numel()), -1, -1)
                .to(device=generated.device, dtype=generated.dtype)
            )
        if contributions:
            offsets = offsets.index_copy(
                0,
                torch.cat(row_indices),
                torch.cat(contributions),
            )
        gamma, beta = (generated + offsets).unbind(dim=1)
        affine_shape = (
            route_plan.batch_size,
            *((1,) * (inputs.ndim - 2)),
            self.d_model,
        )
        return (1.0 + gamma.reshape(affine_shape)) * inputs + beta.reshape(affine_shape)

    def materialized_affine(
        self,
        deck_embedding: Tensor,
        capsule: ExactDeckCapsule,
    ) -> tuple[Tensor, Tensor]:
        """Return one route's fixed FP32 ``gamma`` and ``beta`` vectors."""
        if tuple(deck_embedding.shape) not in {
            (self.deck_dim,),
            (1, self.deck_dim),
        }:
            raise ValueError("FiLM materialization requires one deck embedding")
        module_dtype = next(self.parameters()).dtype
        embedding = deck_embedding.reshape(1, self.deck_dim).to(dtype=module_dtype)
        generated = cast(
            Tensor,
            self.generator(self.norm(embedding)),
        ).reshape(2, self.d_model)
        bank = (
            capsule.transformer_film_offsets
            if self.domain == "transformer"
            else capsule.policy_film_offsets
        )
        if self.site not in bank:
            raise KeyError(f"missing exact FiLM site {self.site!r}")
        gamma, beta = (generated.float() + bank[self.site].float()).unbind(dim=0)
        return gamma, beta


class FixedDeckFiLM(nn.Module):
    """Single-deck FiLM constants with no router or dynamic route state."""

    def __init__(self, d_model: int) -> None:
        """Build fixed affine vectors populated by FP32 export materialization."""
        super().__init__()
        if d_model <= 0:
            raise ValueError("fixed FiLM width must be positive")
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.zeros(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        inputs: Tensor,
        *,
        route_plan: DeckRoutePlan,
        capsules: nn.ModuleDict,
    ) -> Tensor:
        """Apply one immutable deck's affine modulation to every batch row."""
        del capsules
        if inputs.ndim < 2 or int(inputs.shape[-1]) != self.d_model:
            raise ValueError("fixed FiLM input shape is invalid")
        if int(inputs.shape[0]) != route_plan.batch_size:
            raise ValueError("fixed FiLM inputs must align with route plan")
        shape = (1, *((1,) * (inputs.ndim - 2)), self.d_model)
        gamma = self.gamma.to(device=inputs.device, dtype=inputs.dtype).reshape(shape)
        beta = self.beta.to(device=inputs.device, dtype=inputs.dtype).reshape(shape)
        return (1.0 + gamma) * inputs + beta


def routed_norm_parameters(
    norm: nn.LayerNorm,
    *,
    offset_key: str,
    route_plan: DeckRoutePlan,
    capsules: nn.ModuleDict,
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return per-row LayerNorm affine parameters with exact offsets."""
    if norm.weight is None or norm.bias is None or len(norm.normalized_shape) != 1:
        raise ValueError("compositional LayerNorm requires one affine feature axis")
    width = int(norm.normalized_shape[0])
    weight = norm.weight.to(device=reference.device, dtype=reference.dtype).expand(
        route_plan.batch_size, width
    )
    bias = norm.bias.to(device=reference.device, dtype=reference.dtype).expand(
        route_plan.batch_size, width
    )
    row_indices: list[Tensor] = []
    weight_offsets: list[Tensor] = []
    bias_offsets: list[Tensor] = []
    for group in route_plan.groups:
        capsule = exact_capsule(capsules, group.module_key)
        weight_key = f"{offset_key}_weight"
        bias_key = f"{offset_key}_bias"
        if (
            weight_key not in capsule.transformer_norm_offsets
            or bias_key not in capsule.transformer_norm_offsets
        ):
            raise KeyError(f"missing exact norm offsets for {offset_key!r}")
        row_count = int(group.row_indices.numel())
        row_indices.append(group.row_indices)
        weight_offsets.append(
            capsule.transformer_norm_offsets[weight_key]
            .unsqueeze(0)
            .expand(row_count, -1)
            .to(device=reference.device, dtype=reference.dtype)
        )
        bias_offsets.append(
            capsule.transformer_norm_offsets[bias_key]
            .unsqueeze(0)
            .expand(row_count, -1)
            .to(device=reference.device, dtype=reference.dtype)
        )
    if row_indices:
        indices = torch.cat(row_indices)
        weight = weight.clone().index_add(0, indices, torch.cat(weight_offsets))
        bias = bias.clone().index_add(0, indices, torch.cat(bias_offsets))
    return weight, bias


def apply_routed_layer_norm(
    inputs: Tensor,
    norm: nn.LayerNorm,
    *,
    offset_key: str,
    route_plan: DeckRoutePlan,
    capsules: nn.ModuleDict,
) -> Tensor:
    """Evaluate LayerNorm with a distinct small affine offset per exact route."""
    weight, bias = routed_norm_parameters(
        norm,
        offset_key=offset_key,
        route_plan=route_plan,
        capsules=capsules,
        reference=inputs,
    )
    normalized = functional.layer_norm(
        inputs,
        norm.normalized_shape,
        weight=None,
        bias=None,
        eps=norm.eps,
    )
    affine_shape = (
        route_plan.batch_size,
        *((1,) * (inputs.ndim - 2)),
        int(inputs.shape[-1]),
    )
    return normalized * weight.reshape(affine_shape) + bias.reshape(affine_shape)


def _required_deck_embeddings(
    route_plan: DeckRoutePlan,
    *,
    width: int,
) -> Tensor:
    embeddings = route_plan.deck_embeddings
    if embeddings is None:
        raise ValueError("routed compositional execution requires deck embeddings")
    if tuple(embeddings.shape) != (route_plan.batch_size, width):
        raise ValueError("route-plan deck embeddings have an invalid shape")
    return embeddings


def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


__all__ = [
    "DeckConditionedFiLM",
    "FixedCompositionalLinear",
    "FixedDeckFiLM",
    "SharedCompositionalLinear",
    "apply_routed_layer_norm",
    "routed_norm_parameters",
]
