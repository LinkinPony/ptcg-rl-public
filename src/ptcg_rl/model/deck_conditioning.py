"""Exact-deck routing and private residual modules for policy specialization."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Self, cast

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from torch import Tensor, nn

from ptcg_rl.cards.card_encoder import CardEncoder
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.decks.identity import DECK_SIZE
from ptcg_rl.decks.registry import (
    DeckExpertRoute,
    PrivateDeckProfile,
    ResolvedDeckExpertRegistry,
    ResolvedPrivateDeckRegistry,
)
from ptcg_rl.model.deck_lora import RoutedLoRADispatch

DECK_CONDITIONING_ARCHITECTURE_VERSION = 1
DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION = 2
DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION = 3
DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION = 4

TransformerLoRATarget = Literal[
    "attention_qkv",
    "attention_output",
    "ffn_input",
    "ffn_output",
]
PolicyLoRATarget = Literal[
    "scalar_input",
    "scalar_output",
    "dynamic_input",
    "dynamic_output",
    "option_projection",
    "selected_projection",
    "ordered_history_projection",
    "decoder_cardinality_projection",
    "query_input",
    "query_output",
]
DeckPrivateRoute = PrivateDeckProfile | DeckExpertRoute


class DeckLoRAConfig(BaseModel):
    """Portable schema for routed exact-deck low-rank weight deltas."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = 64
    alpha: float = 64.0
    dropout: float = 0.0
    transformer_layer_indices: tuple[int, ...] = ()
    transformer_targets: tuple[TransformerLoRATarget, ...] = (
        "attention_qkv",
        "attention_output",
        "ffn_input",
        "ffn_output",
    )
    policy_targets: tuple[PolicyLoRATarget, ...] = (
        "scalar_input",
        "scalar_output",
        "dynamic_input",
        "dynamic_output",
        "option_projection",
        "selected_projection",
        "ordered_history_projection",
        "decoder_cardinality_projection",
        "query_input",
        "query_output",
    )
    export_mode: Literal["routed", "merged"] = "routed"

    @field_validator("rank")
    @classmethod
    def valid_rank(cls, value: int) -> int:
        """Require a positive low-rank dimension."""
        if value <= 0:
            raise ValueError("LoRA rank must be positive")
        return value

    @field_validator("alpha")
    @classmethod
    def valid_alpha(cls, value: float) -> float:
        """Require a finite positive LoRA scale numerator."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("LoRA alpha must be finite and positive")
        return value

    @field_validator("dropout")
    @classmethod
    def valid_dropout(cls, value: float) -> float:
        """Keep routed weight merging exact until LoRA dropout is implemented."""
        if value != 0.0:
            raise ValueError("routed weight LoRA currently requires dropout=0")
        return value

    @field_validator("transformer_layer_indices")
    @classmethod
    def valid_transformer_layer_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Require deterministic unique non-negative layer indices."""
        if any(index < 0 for index in value):
            raise ValueError("LoRA layer indices must be non-negative")
        if len(value) != len(set(value)):
            raise ValueError("LoRA layer indices must be unique")
        return tuple(sorted(value))

    @field_validator("transformer_targets", "policy_targets")
    @classmethod
    def valid_unique_targets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicate or empty target declarations."""
        if not value:
            raise ValueError("LoRA target declarations must be non-empty")
        if len(value) != len(set(value)):
            raise ValueError("LoRA target declarations must be unique")
        return value

    def resolved_transformer_layers(self, *, num_layers: int) -> tuple[int, ...]:
        """Return explicit target layers, treating an empty tuple as all layers."""
        return self.transformer_layer_indices or tuple(range(num_layers))


class DensePrivateStrategyConfig(BaseModel):
    """Portable schema for routed dense private strategy modules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shared_transformer_layers: int = 8
    option_set_width: int = 384
    option_set_attention_heads: int = 6
    option_set_feedforward_dim: int = 1024
    value_hidden_dim: int = 512
    value_bottleneck_dim: int = 256
    export_mode: Literal["routed", "fixed"] = "routed"

    @field_validator(
        "shared_transformer_layers",
        "option_set_width",
        "option_set_attention_heads",
        "option_set_feedforward_dim",
        "value_hidden_dim",
        "value_bottleneck_dim",
    )
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive dense-private dimensions."""
        if value <= 0:
            raise ValueError("dense-private dimensions must be positive")
        return value

    @model_validator(mode="after")
    def compatible_attention_width(self) -> Self:
        """Require a valid multi-head option-set bottleneck."""
        if self.option_set_width % self.option_set_attention_heads != 0:
            raise ValueError(
                "option_set_width must be divisible by option_set_attention_heads"
            )
        return self


class CompositionalStrategyConfig(BaseModel):
    """Portable schema for deck-conditioned compositional exact capsules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transformer_layer_indices: tuple[int, ...] = (8, 9, 10, 11)
    shared_basis_count: int = 4
    transformer_shared_rank: int = 16
    transformer_exact_rank: int = 8
    policy_shared_rank: int = 8
    policy_exact_rank: int = 8
    router_hidden_dim: int = 128
    option_set_width: int = 384
    option_set_attention_heads: int = 6
    option_set_feedforward_dim: int = 1024
    option_adapter_bottleneck_dim: int = 32
    value_bottleneck_dim: int = 32
    export_mode: Literal["routed", "fixed"] = "routed"

    @field_validator(
        "shared_basis_count",
        "transformer_shared_rank",
        "transformer_exact_rank",
        "policy_shared_rank",
        "policy_exact_rank",
        "router_hidden_dim",
        "option_set_width",
        "option_set_attention_heads",
        "option_set_feedforward_dim",
        "option_adapter_bottleneck_dim",
        "value_bottleneck_dim",
    )
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject unusable compositional dimensions."""
        if value <= 0:
            raise ValueError("compositional strategy dimensions must be positive")
        return value

    @field_validator("transformer_layer_indices")
    @classmethod
    def valid_transformer_layers(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Normalize a non-empty deterministic set of target layers."""
        if not value:
            raise ValueError("compositional strategy requires target layers")
        if any(index < 0 for index in value):
            raise ValueError("compositional layer indices must be non-negative")
        if len(value) != len(set(value)):
            raise ValueError("compositional layer indices must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def compatible_attention_width(self) -> Self:
        """Require a valid shared option-set attention bottleneck."""
        if self.option_set_width % self.option_set_attention_heads != 0:
            raise ValueError(
                "option_set_width must be divisible by option_set_attention_heads"
            )
        return self


class DeckConditioningConfig(BaseModel):
    """Portable architecture and exact private-deck registry configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    architecture_version: int = DECK_CONDITIONING_ARCHITECTURE_VERSION
    resolved_registry_sha256: str | None = None
    deck_size: int = DECK_SIZE
    encoder_hidden_dim: int = 256
    adapter_layer_indices: tuple[int, ...] = (9, 10, 11)
    adapter_bottleneck_dim: int = 64
    policy_bottleneck_dim: int = 64
    value_bottleneck_dim: int = 64
    adapter_dropout: float = 0.0
    deck_context_mode: Literal["encoded", "folded"] = "encoded"
    private_profiles: tuple[PrivateDeckProfile, ...] = ()
    expert_routes: tuple[DeckExpertRoute, ...] = ()
    lora: DeckLoRAConfig | None = None
    dense_private: DensePrivateStrategyConfig | None = None
    compositional: CompositionalStrategyConfig | None = None

    @field_validator(
        "architecture_version",
        "deck_size",
        "encoder_hidden_dim",
        "adapter_bottleneck_dim",
        "policy_bottleneck_dim",
        "value_bottleneck_dim",
    )
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive architecture dimensions and versions."""
        if value <= 0:
            raise ValueError("deck conditioning values must be positive")
        return value

    @field_validator("adapter_layer_indices")
    @classmethod
    def valid_adapter_layer_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Require deterministic unique non-negative Transformer layer indices."""
        if any(index < 0 for index in value):
            raise ValueError("adapter layer indices must be non-negative")
        if len(value) != len(set(value)):
            raise ValueError("adapter layer indices must be unique")
        return tuple(sorted(value))

    @field_validator("adapter_dropout")
    @classmethod
    def valid_adapter_dropout(cls, value: float) -> float:
        """Reject invalid adapter dropout probabilities."""
        if value < 0.0 or value >= 1.0:
            raise ValueError("adapter_dropout must be in [0, 1)")
        return value

    @model_validator(mode="after")
    def consistent_portable_registry(self) -> Self:
        """Validate portable registry identity when it has been resolved."""
        if self.architecture_version not in {
            DECK_CONDITIONING_ARCHITECTURE_VERSION,
            DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION,
            DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
            DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
        }:
            raise ValueError("unsupported deck conditioning architecture version")
        if self.deck_size != DECK_SIZE:
            raise ValueError(f"deck_size must equal {DECK_SIZE}")
        if self.architecture_version == DECK_CONDITIONING_ARCHITECTURE_VERSION:
            if (
                self.expert_routes
                or self.lora is not None
                or self.dense_private is not None
                or self.compositional is not None
                or self.deck_context_mode != "encoded"
            ):
                raise ValueError("architecture v1 cannot declare expert routes or LoRA")
            if self.resolved_registry_sha256 is not None:
                ResolvedPrivateDeckRegistry(
                    profiles=self.private_profiles,
                    resolved_registry_sha256=self.resolved_registry_sha256,
                )
            elif self.private_profiles:
                raise ValueError(
                    "private profiles require a resolved registry fingerprint"
                )
            return self

        if self.private_profiles:
            raise ValueError("expert-routed architectures cannot declare v1 profiles")
        if self.architecture_version == DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION:
            if (
                self.lora is None
                or self.dense_private is not None
                or self.compositional is not None
            ):
                raise ValueError("architecture v2 requires only a LoRA configuration")
            if self.lora.export_mode == "merged" and len(self.expert_routes) != 1:
                raise ValueError(
                    "merged architecture v2 requires exactly one expert route"
                )
            if self.deck_context_mode == "folded" and self.lora.export_mode != "merged":
                raise ValueError("folded deck context requires merged LoRA export mode")
        elif (
            self.architecture_version
            == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
        ):
            if (
                self.lora is not None
                or self.dense_private is None
                or self.compositional is not None
            ):
                raise ValueError(
                    "architecture v3 requires dense_private and forbids LoRA"
                )
            if (
                self.dense_private.export_mode == "fixed"
                and len(self.expert_routes) != 1
            ):
                raise ValueError(
                    "fixed architecture v3 requires exactly one expert route"
                )
            if (
                self.deck_context_mode == "folded"
                and self.dense_private.export_mode != "fixed"
            ):
                raise ValueError(
                    "folded deck context requires fixed dense-private export mode"
                )
            expert_ids = tuple(route.expert_id for route in self.expert_routes)
            if len(expert_ids) != len(set(expert_ids)):
                raise ValueError(
                    "architecture v3 requires one distinct expert per exact deck"
                )
        else:
            if (
                self.lora is not None
                or self.dense_private is not None
                or self.compositional is None
            ):
                raise ValueError(
                    "architecture v4 requires compositional and forbids legacy "
                    "private strategies"
                )
            if (
                self.compositional.export_mode == "fixed"
                and len(self.expert_routes) != 1
            ):
                raise ValueError(
                    "fixed architecture v4 requires exactly one expert route"
                )
            if (
                self.deck_context_mode == "folded"
                and self.compositional.export_mode != "fixed"
            ):
                raise ValueError(
                    "folded deck context requires fixed compositional export mode"
                )
            expert_ids = tuple(route.expert_id for route in self.expert_routes)
            if len(expert_ids) != len(set(expert_ids)):
                raise ValueError(
                    "architecture v4 requires one distinct capsule per exact deck"
                )
        if self.resolved_registry_sha256 is not None:
            ResolvedDeckExpertRegistry(
                routes=self.expert_routes,
                resolved_registry_sha256=self.resolved_registry_sha256,
            )
        elif self.expert_routes:
            raise ValueError("expert routes require a resolved registry fingerprint")
        return self

    def validate_for_model(self, *, num_layers: int, max_card_id: int) -> None:
        """Validate build-time dimensions against the concrete shared model."""
        if not self.enabled:
            return
        if self.resolved_registry_sha256 is None:
            raise ValueError(
                "enabled deck conditioning requires resolved_registry_sha256"
            )
        if any(index >= num_layers for index in self.adapter_layer_indices):
            raise ValueError("adapter layer index exceeds state encoder depth")
        routes = self.active_routes
        if any(
            card_id > max_card_id
            for profile in routes
            for card_id in profile.canonical_card_ids
        ):
            raise ValueError("private profile card ID exceeds CardEncoder range")
        if self.lora is not None and any(
            index >= num_layers
            for index in self.lora.resolved_transformer_layers(num_layers=num_layers)
        ):
            raise ValueError("LoRA layer index exceeds state encoder depth")
        if (
            self.dense_private is not None
            and self.dense_private.shared_transformer_layers >= num_layers
        ):
            raise ValueError(
                "dense-private shared layer count must leave a private upper stack"
            )
        if self.compositional is not None and any(
            index >= num_layers
            for index in self.compositional.transformer_layer_indices
        ):
            raise ValueError("compositional layer index exceeds state encoder depth")

    @property
    def active_routes(self) -> tuple[DeckPrivateRoute, ...]:
        """Return the architecture-specific exact-deck route declarations."""
        if self.architecture_version == DECK_CONDITIONING_ARCHITECTURE_VERSION:
            return self.private_profiles
        return self.expert_routes

    @property
    def profile_by_signature(self) -> dict[str, DeckPrivateRoute]:
        """Return the exact private registry keyed by canonical signature."""
        return {profile.signature: profile for profile in self.active_routes}


@dataclass(frozen=True)
class DeckCompositionGroup:
    """Rows sharing one exact deck composition."""

    full_signature: str
    row_indices: Tensor


@dataclass(frozen=True)
class DeckRouteGroup:
    """Rows activating one private ModuleDict subtree."""

    module_key: str
    row_indices: Tensor


@dataclass(frozen=True)
class DeckRoutePlan:
    """Model-relative private routes resolved once for a DeckBatch."""

    batch_size: int
    resolved_registry_sha256: str
    composition_groups: tuple[DeckCompositionGroup, ...]
    groups: tuple[DeckRouteGroup, ...]
    generic_row_indices: Tensor
    composition_representative_indices: Tensor
    composition_inverse_indices: Tensor
    lora_dispatch: RoutedLoRADispatch | None = None
    deck_embeddings: Tensor | None = None
    exact_capsules: nn.ModuleDict | None = None


class DeckEncoder(nn.Module):
    """Permutation-invariant shared encoder for a complete counted deck."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        """Build the documented phi/mean/rho set encoder."""
        super().__init__()
        self.phi = _deck_mlp(d_model, hidden_dim)
        self.rho = _deck_mlp(d_model, hidden_dim)

    def forward(self, card_ids: Tensor, *, card_encoder: CardEncoder) -> Tensor:
        """Encode complete decks without registering another CardEncoder alias."""
        if card_ids.ndim != 2 or card_ids.shape[1] != DECK_SIZE:
            raise ValueError(f"deck cards must have shape [B, {DECK_SIZE}]")
        card_embeddings = card_encoder(card_ids)
        composition = self.phi(card_embeddings).mean(dim=1)
        return cast(Tensor, self.rho(composition))


class PrivateResidualAdapter(nn.Module):
    """Zero-output bottleneck residual over an arbitrary leading shape."""

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        """Initialize a trainable down projection and inert up projection."""
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, d_model)
        _zero_linear(self.up)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return this exact deck's residual."""
        return cast(
            Tensor,
            self.up(self.dropout(self.activation(self.down(self.layer_norm(inputs))))),
        )


class PrivateScalarResidual(nn.Module):
    """Zero-output private scalar critic residual."""

    def __init__(self, d_model: int, bottleneck_dim: int) -> None:
        """Initialize a hidden projection and inert scalar output."""
        super().__init__()
        self.hidden = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(bottleneck_dim, 1)
        _zero_linear(self.output)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return a private pre-tanh or prefix scalar residual."""
        return cast(
            Tensor,
            self.output(self.activation(self.hidden(inputs))),
        )


def resolve_deck_route_plan(
    decks: DeckBatch,
    config: DeckConditioningConfig,
) -> DeckRoutePlan:
    """Resolve persistent deck identities relative to one model registry."""
    if not config.enabled or config.resolved_registry_sha256 is None:
        raise ValueError("cannot resolve routes for disabled deck conditioning")
    profile_by_signature = config.profile_by_signature
    route_keys = tuple(
        (
            ""
            if profile_by_signature.get(signature) is None
            else profile_by_signature[signature].module_key
        )
        for signature in decks.signatures
    )
    return _cached_deck_route_plan(
        decks.signatures,
        route_keys,
        config.resolved_registry_sha256,
        config.architecture_version,
        str(decks.card_ids.device),
    )


@lru_cache(maxsize=128)
def _cached_deck_route_plan(
    signatures: tuple[str, ...],
    route_keys: tuple[str, ...],
    resolved_registry_sha256: str,
    architecture_version: int,
    device_name: str,
) -> DeckRoutePlan:
    """Build and retain immutable routing indices for one static row layout.

    CUDA Graph capture may not construct arbitrary CUDA indices from a Python
    list because that introduces an unpinned host-to-device copy. Decode graph
    warmup resolves this cache before capture, so the captured model call only
    reads already-resident index tensors. The bounded cache is also useful for
    repeated eager and learner layouts.
    """
    if len(signatures) != len(route_keys):
        raise ValueError("deck signatures and route keys must align")
    device = torch.device(device_name)
    composition_rows: dict[str, list[int]] = {}
    for row, signature in enumerate(signatures):
        composition_rows.setdefault(signature, []).append(row)

    private_rows: dict[str, list[int]] = defaultdict(list)
    generic_rows: list[int] = []
    for row, module_key in enumerate(route_keys):
        if not module_key:
            generic_rows.append(row)
        else:
            private_rows[module_key].append(row)

    composition_groups = tuple(
        DeckCompositionGroup(
            full_signature=signature,
            row_indices=_row_indices(rows, device=device),
        )
        for signature, rows in composition_rows.items()
    )
    composition_positions = {
        group.full_signature: index for index, group in enumerate(composition_groups)
    }
    representatives = tuple(rows[0] for rows in composition_rows.values())
    inverse = tuple(composition_positions[signature] for signature in signatures)
    route_groups = tuple(
        DeckRouteGroup(
            module_key=module_key,
            row_indices=_row_indices(rows, device=device),
        )
        for module_key, rows in sorted(private_rows.items())
    )
    return DeckRoutePlan(
        batch_size=len(signatures),
        resolved_registry_sha256=resolved_registry_sha256,
        composition_groups=composition_groups,
        groups=route_groups,
        generic_row_indices=_row_indices(generic_rows, device=device),
        composition_representative_indices=_row_indices(
            representatives,
            device=device,
        ),
        composition_inverse_indices=_row_indices(inverse, device=device),
        lora_dispatch=(
            None
            if architecture_version
            in {
                DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
            }
            else RoutedLoRADispatch.from_routes(
                route_groups,
                disjoint_rows=True,
            )
        ),
    )


def encode_deck_compositions(
    decks: DeckBatch,
    plan: DeckRoutePlan,
    encoder: DeckEncoder,
    *,
    card_encoder: CardEncoder,
) -> Tensor:
    """Encode each unique composition once and expand it to original rows."""
    unique_cards = decks.card_ids.index_select(
        0,
        plan.composition_representative_indices,
    )
    unique_embeddings = encoder(unique_cards, card_encoder=card_encoder)
    return cast(
        Tensor,
        unique_embeddings.index_select(0, plan.composition_inverse_indices),
    )


def apply_private_residual(
    inputs: Tensor,
    plan: DeckRoutePlan,
    modules: nn.ModuleDict,
    *,
    output_dim: int | None = None,
) -> Tensor:
    """Apply only active exact-deck modules and return their dense residual."""
    if int(inputs.shape[0]) != plan.batch_size:
        raise ValueError("private residual inputs must align with route plan")
    residual_width = int(inputs.shape[-1]) if output_dim is None else output_dim
    if residual_width <= 0:
        raise ValueError("private residual output_dim must be positive")
    residual = inputs.new_zeros((*inputs.shape[:-1], residual_width))
    row_indices: list[Tensor] = []
    contributions: list[Tensor] = []
    for group in plan.groups:
        module = modules[group.module_key]
        selected = inputs.index_select(0, group.row_indices)
        row_indices.append(group.row_indices)
        contributions.append(module(selected).to(dtype=residual.dtype))
    if not contributions:
        return residual
    return residual.index_copy_(
        0,
        torch.cat(row_indices),
        torch.cat(contributions),
    )


def private_residual_modules(
    profiles: tuple[DeckPrivateRoute, ...],
    *,
    d_model: int,
    bottleneck_dim: int,
    dropout: float = 0.0,
) -> nn.ModuleDict:
    """Build one physically independent residual adapter per exact deck."""
    module_keys = tuple(sorted({profile.module_key for profile in profiles}))
    return nn.ModuleDict(
        {
            module_key: PrivateResidualAdapter(
                d_model,
                bottleneck_dim,
                dropout=dropout,
            )
            for module_key in module_keys
        }
    )


def private_scalar_modules(
    profiles: tuple[DeckPrivateRoute, ...],
    *,
    d_model: int,
    bottleneck_dim: int,
) -> nn.ModuleDict:
    """Build one physically independent scalar residual per exact deck."""
    module_keys = tuple(sorted({profile.module_key for profile in profiles}))
    return nn.ModuleDict(
        {
            module_key: PrivateScalarResidual(d_model, bottleneck_dim)
            for module_key in module_keys
        }
    )


def adapter_layer_key(layer_index: int) -> str:
    """Return a deterministic state_dict segment for a Transformer layer."""
    return f"layer_{layer_index:02d}"


def _deck_mlp(d_model: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model),
    )


def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _row_indices(
    rows: tuple[int, ...] | list[int],
    *,
    device: torch.device,
) -> Tensor:
    if not rows:
        return torch.empty(0, dtype=torch.long, device=device)
    first = int(rows[0])
    if all(int(row) == first for row in rows):
        return torch.full(
            (len(rows),),
            first,
            dtype=torch.long,
            device=device,
        )
    if all(int(row) == first + offset for offset, row in enumerate(rows)):
        return torch.arange(
            first,
            first + len(rows),
            dtype=torch.long,
            device=device,
        )
    return torch.tensor(rows, dtype=torch.long, device=device)
