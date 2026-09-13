"""Observation token tensorization and Transformer state encoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor, nn
from torch.nn import functional

from ptcg_rl.actions.encoding import (
    VIRTUAL_AREA,
    StateToken,
    StateTokenLayout,
)
from ptcg_rl.cards.card_encoder import CardEncoder
from ptcg_rl.context import DECK_FLOW_FEATURE_SIZE, HISTORY_COUNTER_SIZE
from ptcg_rl.engine.constants import AreaType
from ptcg_rl.engine.protocols import ObservationInput
from ptcg_rl.model.compositional_capsule import ProjectionShape
from ptcg_rl.model.compositional_projection import (
    DeckConditionedFiLM,
    FixedCompositionalLinear,
    FixedDeckFiLM,
    SharedCompositionalLinear,
    apply_routed_layer_norm,
)
from ptcg_rl.model.deck_conditioning import (
    DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
    DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
    DeckConditioningConfig,
    DeckRouteGroup,
    DeckRoutePlan,
    PrivateResidualAdapter,
    adapter_layer_key,
    apply_private_residual,
    private_residual_modules,
)
from ptcg_rl.model.deck_lora import RoutedLinearLoRA
from ptcg_rl.model.private_strategy import DensePrivateTransformerStack

LEGACY_TOKEN_SCALAR_SIZE = 55
PUBLIC_STATE_SCALAR_SIZE = 4
TOKEN_SCALAR_SIZE = LEGACY_TOKEN_SCALAR_SIZE + PUBLIC_STATE_SCALAR_SIZE
PUBLIC_STATE_SCALAR_START = LEGACY_TOKEN_SCALAR_SIZE
GLOBAL_DECK_FLOW_SCALAR_START = 0
GLOBAL_LONG_TURN_SCALAR = 14
GLOBAL_UNCLIPPED_HISTORY_SCALAR_START = 15
GLOBAL_CONTEXT_SCALAR_SIZE = (
    GLOBAL_UNCLIPPED_HISTORY_SCALAR_START + HISTORY_COUNTER_SIZE
)
ENERGY_TYPE_COUNT = 12
AREA_OOV_INDEX = 13
AREA_EMBEDDING_COUNT = AREA_OOV_INDEX + 1

OWNER_UNKNOWN = 0
OWNER_SELF = 1
OWNER_OPPONENT = 2
OWNER_SHARED = 3
OWNER_ROLE_COUNT = 4

TOKEN_KIND_TO_INDEX: dict[str, int] = {
    "global": 0,
    "special_condition": 1,
    "active": 2,
    "bench": 3,
    "hand": 4,
    "discard": 5,
    "prize": 6,
    "stadium": 7,
    "looking": 8,
    "deck": 9,
    "contextCard": 10,
    "effect": 11,
    "own_unseen": 12,
    "opponent_revealed": 13,
    "opponent_belief": 14,
}
TOKEN_KIND_OOV_INDEX = len(TOKEN_KIND_TO_INDEX)
TOKEN_KIND_COUNT = TOKEN_KIND_OOV_INDEX + 1

ENTITY_SLOT_OOV_INDEX = 17
ENTITY_SLOT_COUNT = ENTITY_SLOT_OOV_INDEX + 1

ATTACHMENT_KIND_ENERGY = 1
ATTACHMENT_KIND_TOOL = 2
ATTACHMENT_KIND_PRE_EVOLUTION = 3
ATTACHMENT_KIND_COUNT = 3

LEGACY_STATE_ENCODER_MISSING_KEYS = frozenset(
    {
        "state_encoder.attachment_kind_gates",
        "state_encoder.entity_slot_embedding.weight",
        "state_encoder.global_context_projection.weight",
        "state_encoder.public_state_projection.weight",
    }
)


@dataclass(frozen=True)
class StateTokenFeatures:
    """Tensor-ready features for one encoded observation token sequence."""

    card_ids: tuple[int, ...]
    areas: tuple[int, ...]
    owner_roles: tuple[int, ...]
    token_kinds: tuple[int, ...]
    scalars: tuple[tuple[float, ...], ...]
    last_attack_ids: tuple[int, ...]
    layout: StateTokenLayout | None = None
    attachment_card_ids: tuple[int, ...] = ()
    attachment_parent_indices: tuple[int, ...] = ()
    attachment_kinds: tuple[int, ...] = ()
    entity_slots: tuple[int, ...] = ()

    def without_layout(self) -> StateTokenFeatures:
        """Return an equivalent numeric feature row without layout metadata."""
        if self.layout is None:
            return self
        return StateTokenFeatures(
            card_ids=self.card_ids,
            areas=self.areas,
            owner_roles=self.owner_roles,
            token_kinds=self.token_kinds,
            scalars=self.scalars,
            last_attack_ids=self.last_attack_ids,
            layout=None,
            attachment_card_ids=self.attachment_card_ids,
            attachment_parent_indices=self.attachment_parent_indices,
            attachment_kinds=self.attachment_kinds,
            entity_slots=self.entity_slots,
        )


@dataclass(frozen=True)
class StateTokenArrayFeatures:
    """Numpy-backed features for one encoded observation token sequence."""

    card_ids: np.ndarray
    areas: np.ndarray
    owner_roles: np.ndarray
    token_kinds: np.ndarray
    scalars: np.ndarray
    last_attack_ids: np.ndarray
    attachment_card_ids: np.ndarray
    attachment_parent_indices: np.ndarray
    attachment_kinds: np.ndarray
    entity_slots: np.ndarray
    layout: StateTokenLayout | None = None

    def without_layout(self) -> StateTokenArrayFeatures:
        """Return an equivalent numeric feature row without layout metadata."""
        if self.layout is None:
            return self
        return StateTokenArrayFeatures(
            card_ids=self.card_ids,
            areas=self.areas,
            owner_roles=self.owner_roles,
            token_kinds=self.token_kinds,
            scalars=self.scalars,
            last_attack_ids=self.last_attack_ids,
            attachment_card_ids=self.attachment_card_ids,
            attachment_parent_indices=self.attachment_parent_indices,
            attachment_kinds=self.attachment_kinds,
            entity_slots=self.entity_slots,
            layout=None,
        )

    def to_features(self) -> StateTokenFeatures:
        """Return the tuple-backed compatibility representation."""
        return StateTokenFeatures(
            card_ids=tuple(int(value) for value in self.card_ids),
            areas=tuple(int(value) for value in self.areas),
            owner_roles=tuple(int(value) for value in self.owner_roles),
            token_kinds=tuple(int(value) for value in self.token_kinds),
            scalars=tuple(tuple(float(value) for value in row) for row in self.scalars),
            last_attack_ids=tuple(int(value) for value in self.last_attack_ids),
            layout=self.layout,
            attachment_card_ids=tuple(int(value) for value in self.attachment_card_ids),
            attachment_parent_indices=tuple(
                int(value) for value in self.attachment_parent_indices
            ),
            attachment_kinds=tuple(int(value) for value in self.attachment_kinds),
            entity_slots=tuple(int(value) for value in self.entity_slots),
        )


StateTokenInput = StateTokenFeatures | StateTokenArrayFeatures


@dataclass(frozen=True)
class StateBatch:
    """Padded batch of tokenized observations."""

    card_ids: Tensor
    areas: Tensor
    owner_roles: Tensor
    token_kinds: Tensor
    scalars: Tensor
    last_attack_ids: Tensor
    padding_mask: Tensor
    attachment_card_ids: Tensor | None = None
    attachment_parent_indices: Tensor | None = None
    attachment_kinds: Tensor | None = None
    entity_slots: Tensor | None = None
    root_input_fingerprints: tuple[str, ...] = ()
    sequence_lengths: tuple[int, ...] = ()


@dataclass(frozen=True)
class StateEncoderOutput:
    """Encoded per-token and pooled global state embeddings."""

    token_embeddings: Tensor
    global_embedding: Tensor
    padding_mask: Tensor


class StateEncoderConfig(BaseModel):
    """Config for the entity-set Transformer state encoder."""

    model_config = ConfigDict(extra="forbid")

    d_model: int = 128
    num_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int | None = None
    dropout: float = 0.0
    # Pre-LN keeps deep encoders away from the representation-collapse attractor
    # observed with post-LN at lr 3e-4; default False so checkpoints saved
    # before this field existed rebuild with their original post-LN semantics.
    norm_first: bool = False

    @field_validator("d_model", "num_layers", "attention_heads")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive dimensions and counts."""
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("feedforward_dim")
    @classmethod
    def valid_feedforward_dim(cls, value: int | None) -> int | None:
        """Reject non-positive feed-forward dimensions."""
        if value is not None and value <= 0:
            raise ValueError("feedforward_dim must be positive when set")
        return value

    @field_validator("dropout")
    @classmethod
    def valid_dropout(cls, value: float) -> float:
        """Reject invalid dropout rates."""
        if value < 0.0 or value >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        return value


class _ZeroInitializedLinear(nn.Linear):
    """Linear adapter whose construction does not advance the model RNG."""

    def reset_parameters(self) -> None:
        """Initialize the compatibility adapter as an exact zero residual."""
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


def encode_observation_tokens(
    observation: ObservationInput,
    *,
    layout: StateTokenLayout | None = None,
) -> StateTokenFeatures:
    """Build tensor-ready token features aligned with ``StateTokenLayout``."""
    return encode_observation_token_arrays(
        observation,
        layout=layout,
    ).to_features()


def encode_observation_token_arrays(
    observation: ObservationInput,
    *,
    layout: StateTokenLayout | None = None,
) -> StateTokenArrayFeatures:
    """Build numpy token features aligned with ``StateTokenLayout``."""
    active_layout = layout or StateTokenLayout.from_observation(observation)
    current = _field(observation, "current")
    select = _field(observation, "select")
    players = tuple(_sequence(_field(current, "players", ())))

    token_count = len(active_layout.tokens)
    card_ids = np.zeros(token_count, dtype=np.int64)
    areas = np.zeros(token_count, dtype=np.int64)
    owner_roles = np.full(token_count, OWNER_UNKNOWN, dtype=np.int64)
    token_kinds = np.full(token_count, TOKEN_KIND_OOV_INDEX, dtype=np.int64)
    scalars = np.zeros((token_count, TOKEN_SCALAR_SIZE), dtype=np.float32)
    last_attack_ids = np.zeros(token_count, dtype=np.int64)
    attachment_card_ids: list[int] = []
    attachment_parent_indices: list[int] = []
    attachment_kinds: list[int] = []
    entity_slots = np.zeros(token_count, dtype=np.uint8)
    for token_index, token in enumerate(active_layout.tokens):
        card_ids[token_index] = max(0, int(token.card_id))
        areas[token_index] = _safe_area_index(token.key.area)
        owner_roles[token_index] = _owner_role(
            token.key.player_index,
            active_layout.your_index,
        )
        token_kinds[token_index] = TOKEN_KIND_TO_INDEX.get(
            token.kind,
            TOKEN_KIND_OOV_INDEX,
        )
        scalars[token_index, :] = tuple(
            _token_scalars(token, current, select, players, active_layout)
        )
        last_attack_ids[token_index] = max(0, int(token.last_attack_id))
        pokemon = _pokemon_for_token(token, players)
        if pokemon is not None:
            for card_id, attachment_kind in _pokemon_attachments(pokemon):
                attachment_card_ids.append(_uint16_value(card_id, "card ID"))
                attachment_parent_indices.append(
                    _uint16_value(token_index, "attachment parent token index")
                )
                attachment_kinds.append(attachment_kind)
            entity_slots[token_index] = min(
                ENTITY_SLOT_OOV_INDEX,
                max(0, int(token.key.index) + 1),
            )

    return StateTokenArrayFeatures(
        card_ids=card_ids,
        areas=areas,
        owner_roles=owner_roles,
        token_kinds=token_kinds,
        scalars=scalars,
        last_attack_ids=last_attack_ids,
        attachment_card_ids=np.asarray(attachment_card_ids, dtype=np.uint16),
        attachment_parent_indices=np.asarray(
            attachment_parent_indices,
            dtype=np.uint16,
        ),
        attachment_kinds=np.asarray(attachment_kinds, dtype=np.uint8),
        entity_slots=entity_slots,
        layout=active_layout,
    )


def collate_state_tokens(
    features: Sequence[StateTokenInput],
    *,
    device: torch.device | str | None = None,
) -> StateBatch:
    """Pad token feature sequences into a ``StateBatch``."""
    if not features:
        raise ValueError("features must be non-empty")
    max_tokens = max(len(item.card_ids) for item in features)
    if max_tokens <= 0:
        raise ValueError("each feature sequence must contain at least one token")

    batch_size = len(features)
    card_rows = np.zeros((batch_size, max_tokens), dtype=np.int64)
    area_rows = np.zeros((batch_size, max_tokens), dtype=np.int64)
    owner_rows = np.full((batch_size, max_tokens), OWNER_UNKNOWN, dtype=np.int64)
    kind_rows = np.full((batch_size, max_tokens), TOKEN_KIND_OOV_INDEX, dtype=np.int64)
    scalar_rows = np.zeros(
        (batch_size, max_tokens, TOKEN_SCALAR_SIZE),
        dtype=np.float32,
    )
    last_attack_rows = np.zeros((batch_size, max_tokens), dtype=np.int64)
    max_attachments = max(1, *(len(item.attachment_card_ids) for item in features))
    attachment_card_rows = np.zeros(
        (batch_size, max_attachments),
        dtype=np.uint16,
    )
    attachment_parent_rows = np.zeros(
        (batch_size, max_attachments),
        dtype=np.uint16,
    )
    attachment_kind_rows = np.zeros(
        (batch_size, max_attachments),
        dtype=np.uint8,
    )
    entity_slot_rows = np.zeros((batch_size, max_tokens), dtype=np.uint8)
    padding_rows = np.ones((batch_size, max_tokens), dtype=np.bool_)
    for row_index, item in enumerate(features):
        length = len(item.card_ids)
        card_rows[row_index, :length] = item.card_ids
        area_rows[row_index, :length] = item.areas
        owner_rows[row_index, :length] = item.owner_roles
        kind_rows[row_index, :length] = item.token_kinds
        scalar_rows[row_index, :length, :] = item.scalars
        last_attack_rows[row_index, :length] = item.last_attack_ids
        _copy_attachments(
            attachment_card_rows[row_index],
            attachment_parent_rows[row_index],
            attachment_kind_rows[row_index],
            item,
            token_count=length,
        )
        if len(item.entity_slots) not in (0, length):
            raise ValueError("entity_slots must align with state tokens")
        if len(item.entity_slots) == length:
            entity_slot_rows[row_index, :length] = item.entity_slots
        padding_rows[row_index, :length] = False

    return StateBatch(
        card_ids=torch.as_tensor(card_rows, device=device),
        areas=torch.as_tensor(area_rows, device=device),
        owner_roles=torch.as_tensor(owner_rows, device=device),
        token_kinds=torch.as_tensor(kind_rows, device=device),
        scalars=torch.as_tensor(scalar_rows, device=device),
        last_attack_ids=torch.as_tensor(
            last_attack_rows,
            device=device,
        ),
        padding_mask=torch.as_tensor(padding_rows, device=device),
        attachment_card_ids=torch.as_tensor(attachment_card_rows, device=device),
        attachment_parent_indices=torch.as_tensor(
            attachment_parent_rows,
            device=device,
        ),
        attachment_kinds=torch.as_tensor(attachment_kind_rows, device=device),
        entity_slots=torch.as_tensor(entity_slot_rows, device=device),
        sequence_lengths=tuple(len(item.card_ids) for item in features),
    )


class StateEncoder(nn.Module):
    """Encode visible state entity tokens with a small Transformer encoder."""

    def __init__(
        self,
        *,
        config: StateEncoderConfig | None = None,
        card_encoder: CardEncoder,
        deck_conditioning: DeckConditioningConfig | None = None,
    ) -> None:
        """Initialize the state encoder."""
        super().__init__()
        self.config = config or StateEncoderConfig()
        if self.config.d_model % self.config.attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")

        d_model = self.config.d_model
        self.card_encoder = card_encoder
        self.area_embedding = nn.Embedding(AREA_EMBEDDING_COUNT, d_model)
        self.owner_embedding = nn.Embedding(OWNER_ROLE_COUNT, d_model)
        self.kind_embedding = nn.Embedding(TOKEN_KIND_COUNT, d_model)
        self.entity_slot_embedding = nn.Embedding(ENTITY_SLOT_COUNT, d_model)
        self.attachment_kind_gates = nn.Parameter(
            torch.zeros(ATTACHMENT_KIND_COUNT, d_model)
        )
        nn.init.zeros_(self.entity_slot_embedding.weight)
        self.scalar_projection = nn.Sequential(
            nn.Linear(LEGACY_TOKEN_SCALAR_SIZE, d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d_model, d_model),
        )
        self.global_context_projection = _ZeroInitializedLinear(
            GLOBAL_CONTEXT_SCALAR_SIZE,
            d_model,
            bias=False,
        )
        self.public_state_projection = _ZeroInitializedLinear(
            PUBLIC_STATE_SCALAR_SIZE,
            d_model,
            bias=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.config.attention_heads,
            dim_feedforward=self.config.feedforward_dim or d_model * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=self.config.norm_first,
        )
        self.transformer = nn.TransformerEncoder(layer, self.config.num_layers)
        self.layer_norm: nn.Module = nn.LayerNorm(d_model)
        conditioning = (
            deck_conditioning
            if deck_conditioning is not None and deck_conditioning.enabled
            else None
        )
        self.private_adapters = nn.ModuleDict()
        self.private_lora = nn.ModuleDict()
        self.private_strategy_stacks = nn.ModuleDict()
        self.compositional_projections = nn.ModuleDict()
        self.compositional_film = nn.ModuleDict()
        self.compositional_shared_adapters = nn.ModuleDict()
        self._dense_private_shared_layers: int | None = None
        self._compositional_layer_indices: frozenset[int] = frozenset()
        self._compositional_fixed = False
        self._route_major_routing = False
        if conditioning is not None:
            self._route_major_routing = (
                self.config.dropout == 0.0 and conditioning.adapter_dropout == 0.0
            )
            if (
                conditioning.architecture_version
                == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
            ):
                dense_config = conditioning.dense_private
                if dense_config is None:
                    raise ValueError(
                        "dense-private state encoder requires dense_private config"
                    )
                shared_layers = dense_config.shared_transformer_layers
                if any(
                    layer_index < shared_layers
                    for layer_index in conditioning.adapter_layer_indices
                ):
                    raise ValueError(
                        "dense-private residual adapters must be in the private "
                        "Transformer stack"
                    )
                self._dense_private_shared_layers = shared_layers
                with torch.random.fork_rng(devices=[]):
                    self.private_strategy_stacks = _private_strategy_stacks(
                        conditioning,
                        upper_layers=tuple(self.transformer.layers[shared_layers:]),
                        output_norm=self.layer_norm,
                        d_model=d_model,
                    )
                if dense_config.export_mode == "fixed":
                    # The selected private stack already owns the only upper
                    # layers and output norm that a fixed-deck runtime can use.
                    # Removing the unused generic copies is what keeps the
                    # deployment artifact within Kaggle's archive limit.
                    self.transformer.layers = nn.ModuleList(
                        tuple(self.transformer.layers[:shared_layers])
                    )
                    self.transformer.num_layers = shared_layers
                    self.layer_norm = nn.Identity()
            elif (
                conditioning.architecture_version
                == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
            ):
                compositional = conditioning.compositional
                if compositional is None:
                    raise ValueError(
                        "compositional state encoder requires architecture config"
                    )
                self._compositional_layer_indices = frozenset(
                    compositional.transformer_layer_indices
                )
                feedforward_dim = self.config.feedforward_dim or d_model * 4
                fixed_export = compositional.export_mode == "fixed"
                self.compositional_projections = nn.ModuleDict(
                    {
                        adapter_layer_key(layer_index): _compositional_layer_modules(
                            layer_index=layer_index,
                            d_model=d_model,
                            feedforward_dim=feedforward_dim,
                            deck_dim=d_model,
                            basis_count=compositional.shared_basis_count,
                            shared_rank=compositional.transformer_shared_rank,
                            router_hidden_dim=compositional.router_hidden_dim,
                            fixed=fixed_export,
                        )
                        for layer_index in compositional.transformer_layer_indices
                    }
                )
                self.compositional_film = nn.ModuleDict(
                    {
                        adapter_layer_key(layer_index): nn.ModuleDict(
                            {
                                site: (
                                    FixedDeckFiLM(d_model)
                                    if fixed_export
                                    else DeckConditionedFiLM(
                                        d_model,
                                        d_model,
                                        compositional.router_hidden_dim,
                                        domain="transformer",
                                        site=(
                                            f"{adapter_layer_key(layer_index)}_{site}"
                                        ),
                                    )
                                )
                                for site in ("attention", "feedforward")
                            }
                        )
                        for layer_index in compositional.transformer_layer_indices
                    }
                )
                self._compositional_fixed = fixed_export
                self.compositional_shared_adapters = nn.ModuleDict(
                    {
                        adapter_layer_key(layer_index): PrivateResidualAdapter(
                            d_model,
                            conditioning.adapter_bottleneck_dim,
                            dropout=conditioning.adapter_dropout,
                        )
                        for layer_index in conditioning.adapter_layer_indices
                        if layer_index in self._compositional_layer_indices
                    }
                )
            else:
                self.private_adapters = nn.ModuleDict(
                    {
                        adapter_layer_key(layer_index): private_residual_modules(
                            conditioning.active_routes,
                            d_model=d_model,
                            bottleneck_dim=conditioning.adapter_bottleneck_dim,
                            dropout=conditioning.adapter_dropout,
                        )
                        for layer_index in conditioning.adapter_layer_indices
                    }
                )
                if (
                    conditioning.lora is not None
                    and conditioning.lora.export_mode == "routed"
                ):
                    with torch.random.fork_rng(devices=[]):
                        self.private_lora = _transformer_lora_modules(
                            conditioning,
                            num_layers=self.config.num_layers,
                            d_model=d_model,
                            feedforward_dim=(
                                self.config.feedforward_dim or d_model * 4
                            ),
                        )

    def forward(
        self,
        batch: StateBatch,
        *,
        attack_embedding: nn.Embedding | None = None,
    ) -> StateEncoderOutput:
        """Return per-token and global embeddings for a padded state batch."""
        token_inputs = self._token_inputs(batch, attack_embedding=attack_embedding)
        encoded = cast(
            Tensor,
            self.transformer(token_inputs, src_key_padding_mask=batch.padding_mask),
        )
        return self._finish_encoding(encoded, batch.padding_mask)

    def forward_conditioned(
        self,
        batch: StateBatch,
        *,
        deck_global_residual: Tensor,
        route_plan: DeckRoutePlan,
        exact_capsules: nn.ModuleDict | None = None,
        attack_embedding: nn.Embedding | None = None,
    ) -> StateEncoderOutput:
        """Encode state tokens with global deck context and private adapters."""
        token_inputs = self._token_inputs(batch, attack_embedding=attack_embedding)
        if tuple(deck_global_residual.shape) != (
            int(token_inputs.shape[0]),
            int(token_inputs.shape[2]),
        ):
            raise ValueError("deck global residual must have shape [B, D]")
        encoded = torch.cat(
            (
                token_inputs[:, :1, :] + deck_global_residual.unsqueeze(1),
                token_inputs[:, 1:, :],
            ),
            dim=1,
        )
        if self._dense_private_shared_layers is not None:
            encoded = self._forward_dense_private(
                encoded,
                padding_mask=batch.padding_mask,
                route_plan=route_plan,
            )
            return self._finish_normalized_encoding(encoded, batch.padding_mask)
        if self._compositional_layer_indices:
            if exact_capsules is None:
                raise ValueError(
                    "compositional state encoding requires exact capsule bank"
                )
            encoded = self._forward_compositional(
                encoded,
                padding_mask=batch.padding_mask,
                route_plan=route_plan,
                exact_capsules=exact_capsules,
            )
            return self._finish_normalized_encoding(encoded, batch.padding_mask)

        layer_plan = route_plan
        layer_padding_mask = batch.padding_mask
        restore_row_indices: Tensor | None = None
        if self._route_major_routing and route_plan.lora_dispatch is not None:
            (
                encoded,
                layer_padding_mask,
                layer_plan,
                restore_row_indices,
            ) = _pack_route_major_rows(encoded, batch.padding_mask, route_plan)
        attention_mask = (
            _routed_attention_mask(encoded, layer_padding_mask)
            if layer_plan.groups and self.private_lora
            else None
        )
        for layer_index, layer in enumerate(self.transformer.layers):
            layer_key = adapter_layer_key(layer_index)
            if layer_plan.groups and layer_key in self.private_lora:
                encoded = _forward_routed_lora_layer(
                    cast(nn.TransformerEncoderLayer, layer),
                    cast(nn.ModuleDict, self.private_lora[layer_key]),
                    encoded,
                    padding_mask=layer_padding_mask,
                    route_plan=layer_plan,
                    attention_mask=attention_mask,
                )
            else:
                encoded = layer(
                    encoded,
                    src_key_padding_mask=layer_padding_mask,
                    is_causal=False,
                )
            if layer_plan.groups and layer_key in self.private_adapters:
                encoded = encoded + apply_private_residual(
                    encoded,
                    layer_plan,
                    cast(nn.ModuleDict, self.private_adapters[layer_key]),
                )
        if self.transformer.norm is not None:
            encoded = self.transformer.norm(encoded)
        if restore_row_indices is not None:
            encoded = encoded.index_select(0, restore_row_indices)
        return self._finish_encoding(encoded, batch.padding_mask)

    def validate_frozen_prefix_layers(self, prefix_layers: int) -> None:
        """Validate a route-independent compositional prefix boundary."""
        if not self._compositional_layer_indices:
            raise ValueError(
                "conditioned prefix freezing requires compositional routing"
            )
        first_routed_layer = min(self._compositional_layer_indices)
        if prefix_layers <= 0 or prefix_layers > first_routed_layer:
            raise ValueError(
                "prefix_layers must end before the first compositional layer"
            )

    def _forward_compositional(
        self,
        inputs: Tensor,
        *,
        padding_mask: Tensor,
        route_plan: DeckRoutePlan,
        exact_capsules: nn.ModuleDict,
    ) -> Tensor:
        """Run one shared Transformer with upper compositional residuals."""
        if int(inputs.shape[0]) != route_plan.batch_size:
            raise ValueError("compositional inputs must align with route plan")
        encoded = inputs
        attention_mask: Tensor | None = None
        for layer_index, raw_layer in enumerate(self.transformer.layers):
            layer = cast(nn.TransformerEncoderLayer, raw_layer)
            if layer_index not in self._compositional_layer_indices:
                encoded = layer(
                    encoded,
                    src_key_padding_mask=padding_mask,
                    is_causal=False,
                )
                continue
            if attention_mask is None:
                attention_mask = _routed_attention_mask(encoded, padding_mask)
            layer_key = adapter_layer_key(layer_index)
            encoded = _forward_compositional_layer(
                layer,
                cast(nn.ModuleDict, self.compositional_projections[layer_key]),
                cast(nn.ModuleDict, self.compositional_film[layer_key]),
                encoded,
                padding_mask=padding_mask,
                route_plan=route_plan,
                capsules=exact_capsules,
                layer_key=layer_key,
                attention_mask=attention_mask,
                fixed=self._compositional_fixed,
            )
            if layer_key in self.compositional_shared_adapters:
                encoded = encoded + self.compositional_shared_adapters[layer_key](
                    encoded
                )
        if self.transformer.norm is not None:
            encoded = self.transformer.norm(encoded)
        if not isinstance(self.layer_norm, nn.LayerNorm):
            raise TypeError("compositional state encoder requires LayerNorm output")
        if self._compositional_fixed:
            return cast(Tensor, self.layer_norm(encoded))
        return apply_routed_layer_norm(
            encoded,
            self.layer_norm,
            offset_key="output",
            route_plan=route_plan,
            capsules=exact_capsules,
        )

    def _forward_dense_private(
        self,
        inputs: Tensor,
        *,
        padding_mask: Tensor,
        route_plan: DeckRoutePlan,
    ) -> Tensor:
        """Run the shared lower trunk and route only active private upper stacks."""
        if int(inputs.shape[0]) != route_plan.batch_size:
            raise ValueError("dense-private inputs must align with route plan")
        shared_layers = self._dense_private_shared_layers
        if shared_layers is None:
            raise RuntimeError("dense-private Transformer split is not configured")

        shared = inputs
        for layer_index in range(shared_layers):
            shared = self.transformer.layers[layer_index](
                shared,
                src_key_padding_mask=padding_mask,
                is_causal=False,
            )

        row_indices: list[Tensor] = []
        routed_outputs: list[Tensor] = []
        for group in route_plan.groups:
            if group.module_key not in self.private_strategy_stacks:
                raise ValueError(
                    f"missing dense-private stack for route {group.module_key!r}"
                )
            selected_padding_mask = padding_mask.index_select(0, group.row_indices)
            selected = shared.index_select(0, group.row_indices)
            routed_outputs.append(
                self.private_strategy_stacks[group.module_key](
                    selected,
                    padding_mask=selected_padding_mask,
                )
            )
            row_indices.append(group.row_indices)

        generic_rows = route_plan.generic_row_indices
        if int(generic_rows.numel()) > 0:
            if len(self.transformer.layers) < self.config.num_layers:
                raise ValueError(
                    "fixed dense-private state encoder cannot use a generic route"
                )
            generic = shared.index_select(0, generic_rows)
            generic_padding_mask = padding_mask.index_select(0, generic_rows)
            for layer_index in range(shared_layers, self.config.num_layers):
                generic = self.transformer.layers[layer_index](
                    generic,
                    src_key_padding_mask=generic_padding_mask,
                    is_causal=False,
                )
            if self.transformer.norm is not None:
                generic = self.transformer.norm(generic)
            routed_outputs.append(self.layer_norm(generic))
            row_indices.append(generic_rows)

        routed_row_count = sum(int(rows.numel()) for rows in row_indices)
        if routed_row_count != route_plan.batch_size:
            raise ValueError("dense-private route plan must partition every batch row")
        return inputs.new_zeros(inputs.shape).index_copy(
            0,
            torch.cat(row_indices),
            torch.cat(routed_outputs),
        )

    def _token_inputs(
        self,
        batch: StateBatch,
        *,
        attack_embedding: nn.Embedding | None,
    ) -> Tensor:
        """Build the unchanged shared token inputs before the Transformer."""
        shared_scalars, global_context_scalars, public_state_scalars = (
            _split_global_context_scalars(batch)
        )
        token_inputs = (
            self.card_encoder(batch.card_ids)
            + self.area_embedding(_safe_embedding_rows(batch.areas, AREA_OOV_INDEX))
            + self.owner_embedding(
                _safe_embedding_rows(batch.owner_roles, OWNER_UNKNOWN)
            )
            + self.kind_embedding(
                _safe_embedding_rows(batch.token_kinds, TOKEN_KIND_OOV_INDEX)
            )
            + self.scalar_projection(shared_scalars)
            + self.global_context_projection(global_context_scalars)
            + self.public_state_projection(public_state_scalars)
        )
        if batch.attachment_card_ids is not None:
            attachment_features = self._attachment_features(batch, token_inputs)
            token_inputs = token_inputs + attachment_features
        if batch.entity_slots is not None:
            token_inputs = token_inputs + self.entity_slot_embedding(
                _safe_embedding_rows(
                    batch.entity_slots,
                    ENTITY_SLOT_OOV_INDEX,
                ).to(dtype=torch.long)
            )
        if attack_embedding is not None:
            safe_attack_ids = torch.where(
                (batch.last_attack_ids >= 0)
                & (batch.last_attack_ids < attack_embedding.num_embeddings),
                batch.last_attack_ids,
                torch.zeros_like(batch.last_attack_ids),
            )
            token_inputs = token_inputs + attack_embedding(safe_attack_ids)
        return cast(Tensor, token_inputs)

    def _finish_encoding(
        self,
        encoded: Tensor,
        padding_mask: Tensor,
    ) -> StateEncoderOutput:
        """Apply the existing final normalization and padding semantics."""
        return self._finish_normalized_encoding(
            self.layer_norm(encoded),
            padding_mask,
        )

    @staticmethod
    def _finish_normalized_encoding(
        encoded: Tensor,
        padding_mask: Tensor,
    ) -> StateEncoderOutput:
        """Mask padding and pool an already normalized routed representation."""
        encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return StateEncoderOutput(
            token_embeddings=encoded,
            global_embedding=encoded[:, 0, :],
            padding_mask=padding_mask,
        )

    def _attachment_features(
        self,
        batch: StateBatch,
        token_inputs: Tensor,
    ) -> Tensor:
        """Pool sparse attached-card embeddings into parent tokens by kind."""
        card_ids = batch.attachment_card_ids
        parent_indices = batch.attachment_parent_indices
        kinds = batch.attachment_kinds
        if card_ids is None or parent_indices is None or kinds is None:
            raise ValueError("attachment state tensors must be provided together")

        card_ids_long = card_ids.to(dtype=torch.long)
        parent_indices_long = parent_indices.to(dtype=torch.long)
        kinds_long = kinds.to(dtype=torch.long)
        batch_size, token_count, d_model = token_inputs.shape
        valid = (
            (card_ids_long > 0)
            & (parent_indices_long >= 0)
            & (parent_indices_long < token_count)
            & (kinds_long >= 1)
            & (kinds_long <= ATTACHMENT_KIND_COUNT)
        )
        safe_parents = torch.where(
            valid,
            parent_indices_long,
            torch.zeros_like(parent_indices_long),
        )
        safe_kinds = torch.where(
            valid,
            kinds_long - 1,
            torch.zeros_like(kinds_long),
        )
        attachment_gates = self.attachment_kind_gates[safe_kinds]
        attachment_embeddings = self.card_encoder(card_ids_long)
        attachment_embeddings = (
            attachment_embeddings * attachment_gates * valid.unsqueeze(-1)
        )
        pooled = attachment_embeddings.new_zeros((batch_size, token_count, d_model))
        pooled.scatter_add_(
            1,
            safe_parents.unsqueeze(-1).expand_as(attachment_embeddings),
            attachment_embeddings,
        )
        return cast(Tensor, pooled)


def _pack_route_major_rows(
    inputs: Tensor,
    padding_mask: Tensor,
    route_plan: DeckRoutePlan,
) -> tuple[Tensor, Tensor, DeckRoutePlan, Tensor]:
    """Pack expert rows once so every routed Transformer projection uses views."""
    dispatch = route_plan.lora_dispatch
    if dispatch is None:
        raise ValueError("route-major packing requires an active LoRA dispatch")
    packed_row_indices = torch.cat(
        (dispatch.row_indices, route_plan.generic_row_indices)
    )
    if int(packed_row_indices.numel()) != route_plan.batch_size:
        raise ValueError("route plan must partition every batch row exactly once")
    packed_positions = torch.arange(
        route_plan.batch_size,
        dtype=torch.long,
        device=packed_row_indices.device,
    )
    restore_row_indices = torch.empty_like(packed_row_indices)
    restore_row_indices.scatter_(0, packed_row_indices, packed_positions)

    packed_groups: list[DeckRouteGroup] = []
    start = 0
    for module_key, row_count in zip(
        dispatch.module_keys,
        dispatch.row_counts,
        strict=True,
    ):
        stop = start + row_count
        packed_groups.append(
            DeckRouteGroup(
                module_key=module_key,
                row_indices=packed_positions[start:stop],
            )
        )
        start = stop
    packed_plan = replace(
        route_plan,
        groups=tuple(packed_groups),
        generic_row_indices=packed_positions[start:],
        lora_dispatch=dispatch.as_contiguous_prefix(),
    )
    return (
        inputs.index_select(0, packed_row_indices),
        padding_mask.index_select(0, packed_row_indices),
        packed_plan,
        restore_row_indices,
    )


def _transformer_lora_modules(
    conditioning: DeckConditioningConfig,
    *,
    num_layers: int,
    d_model: int,
    feedforward_dim: int,
) -> nn.ModuleDict:
    """Build routed LoRA banks without replacing legacy base parameter names."""
    config = conditioning.lora
    if config is None:
        return nn.ModuleDict()
    target_shapes = {
        "attention_qkv": (d_model, d_model * 3),
        "attention_output": (d_model, d_model),
        "ffn_input": (d_model, feedforward_dim),
        "ffn_output": (feedforward_dim, d_model),
    }
    return nn.ModuleDict(
        {
            adapter_layer_key(layer_index): nn.ModuleDict(
                {
                    target: RoutedLinearLoRA(
                        conditioning.active_routes,
                        in_features=target_shapes[target][0],
                        out_features=target_shapes[target][1],
                        rank=config.rank,
                        alpha=config.alpha,
                    )
                    for target in config.transformer_targets
                }
            )
            for layer_index in config.resolved_transformer_layers(num_layers=num_layers)
        }
    )


def compositional_transformer_shapes(
    config: StateEncoderConfig,
    *,
    layer_indices: Sequence[int],
) -> dict[str, ProjectionShape]:
    """Return stable exact-capsule shapes for compositional Transformer targets."""
    d_model = config.d_model
    feedforward_dim = config.feedforward_dim or d_model * 4
    shapes: dict[str, ProjectionShape] = {}
    for layer_index in layer_indices:
        layer_key = adapter_layer_key(layer_index)
        shapes.update(
            {
                f"{layer_key}_attention_qkv": ProjectionShape(
                    d_model,
                    3 * d_model,
                    True,
                ),
                f"{layer_key}_attention_output": ProjectionShape(
                    d_model,
                    d_model,
                    True,
                ),
                f"{layer_key}_ffn_input": ProjectionShape(
                    d_model,
                    feedforward_dim,
                    True,
                ),
                f"{layer_key}_ffn_output": ProjectionShape(
                    feedforward_dim,
                    d_model,
                    True,
                ),
            }
        )
    return shapes


def _compositional_layer_modules(
    *,
    layer_index: int,
    d_model: int,
    feedforward_dim: int,
    deck_dim: int,
    basis_count: int,
    shared_rank: int,
    router_hidden_dim: int,
    fixed: bool,
) -> nn.ModuleDict:
    """Build all four DCCR projection banks for one Transformer layer."""
    layer_key = adapter_layer_key(layer_index)
    shapes = {
        "attention_qkv": (d_model, 3 * d_model),
        "attention_output": (d_model, d_model),
        "ffn_input": (d_model, feedforward_dim),
        "ffn_output": (feedforward_dim, d_model),
    }
    return nn.ModuleDict(
        {
            target: (
                FixedCompositionalLinear(
                    in_features=in_features,
                    out_features=out_features,
                )
                if fixed
                else SharedCompositionalLinear(
                    target=f"{layer_key}_{target}",
                    domain="transformer",
                    in_features=in_features,
                    out_features=out_features,
                    deck_dim=deck_dim,
                    basis_count=basis_count,
                    shared_rank=shared_rank,
                    router_hidden_dim=router_hidden_dim,
                )
            )
            for target, (in_features, out_features) in shapes.items()
        }
    )


def _forward_compositional_layer(
    layer: nn.TransformerEncoderLayer,
    projections: nn.ModuleDict,
    film: nn.ModuleDict,
    inputs: Tensor,
    *,
    padding_mask: Tensor,
    route_plan: DeckRoutePlan,
    capsules: nn.ModuleDict,
    layer_key: str,
    attention_mask: Tensor,
    fixed: bool,
) -> Tensor:
    """Evaluate one Transformer layer with DCCR projection and FiLM deltas."""
    x = inputs
    if layer.norm_first:
        normalized = _compositional_layer_norm(
            x,
            layer.norm1,
            offset_key=f"{layer_key}_norm1",
            route_plan=route_plan,
            capsules=capsules,
            fixed=fixed,
        )
        attention = _compositional_attention_block(
            layer,
            projections,
            normalized,
            padding_mask=padding_mask,
            route_plan=route_plan,
            capsules=capsules,
            attention_mask=attention_mask,
        )
        x = x + cast(Any, film["attention"])(
            attention,
            route_plan=route_plan,
            capsules=capsules,
        )
        normalized = _compositional_layer_norm(
            x,
            layer.norm2,
            offset_key=f"{layer_key}_norm2",
            route_plan=route_plan,
            capsules=capsules,
            fixed=fixed,
        )
        feedforward = _compositional_feedforward_block(
            layer,
            projections,
            normalized,
            route_plan=route_plan,
            capsules=capsules,
        )
        return cast(
            Tensor,
            x
            + cast(Any, film["feedforward"])(
                feedforward,
                route_plan=route_plan,
                capsules=capsules,
            ),
        )

    attention = _compositional_attention_block(
        layer,
        projections,
        x,
        padding_mask=padding_mask,
        route_plan=route_plan,
        capsules=capsules,
        attention_mask=attention_mask,
    )
    attention = cast(Any, film["attention"])(
        attention,
        route_plan=route_plan,
        capsules=capsules,
    )
    x = _compositional_layer_norm(
        x + attention,
        layer.norm1,
        offset_key=f"{layer_key}_norm1",
        route_plan=route_plan,
        capsules=capsules,
        fixed=fixed,
    )
    feedforward = _compositional_feedforward_block(
        layer,
        projections,
        x,
        route_plan=route_plan,
        capsules=capsules,
    )
    feedforward = cast(Any, film["feedforward"])(
        feedforward,
        route_plan=route_plan,
        capsules=capsules,
    )
    return _compositional_layer_norm(
        x + feedforward,
        layer.norm2,
        offset_key=f"{layer_key}_norm2",
        route_plan=route_plan,
        capsules=capsules,
        fixed=fixed,
    )


def _compositional_layer_norm(
    inputs: Tensor,
    norm: nn.LayerNorm,
    *,
    offset_key: str,
    route_plan: DeckRoutePlan,
    capsules: nn.ModuleDict,
    fixed: bool,
) -> Tensor:
    """Apply routed offsets or a fixed export's already-folded LayerNorm."""
    if fixed:
        return cast(Tensor, norm(inputs))
    return apply_routed_layer_norm(
        inputs,
        norm,
        offset_key=offset_key,
        route_plan=route_plan,
        capsules=capsules,
    )


def _compositional_attention_block(
    layer: nn.TransformerEncoderLayer,
    projections: nn.ModuleDict,
    inputs: Tensor,
    *,
    padding_mask: Tensor,
    route_plan: DeckRoutePlan,
    capsules: nn.ModuleDict,
    attention_mask: Tensor,
) -> Tensor:
    """Apply self-attention with compositional QKV and output projections."""
    del padding_mask
    attention = layer.self_attn
    in_proj_weight = attention.in_proj_weight
    if in_proj_weight is None or not attention._qkv_same_embed_dim:
        raise RuntimeError("compositional Transformer requires packed QKV weights")
    qkv = functional.linear(inputs, in_proj_weight, attention.in_proj_bias)
    qkv = cast(Any, projections["attention_qkv"]).add_delta(
        qkv,
        inputs,
        route_plan=route_plan,
        capsules=capsules,
    )
    query, key, value = qkv.chunk(3, dim=-1)
    batch_size, token_count, d_model = inputs.shape
    num_heads = attention.num_heads
    head_dim = d_model // num_heads
    query = query.view(batch_size, token_count, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, token_count, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, token_count, num_heads, head_dim).transpose(1, 2)
    attended = functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=attention.dropout if layer.training else 0.0,
        is_causal=False,
    )
    attended = (
        attended.transpose(1, 2).contiguous().view(batch_size, token_count, d_model)
    )
    output = cast(Tensor, attention.out_proj(attended))
    output = cast(Any, projections["attention_output"]).add_delta(
        output,
        attended,
        route_plan=route_plan,
        capsules=capsules,
    )
    return cast(Tensor, layer.dropout1(output))


def _compositional_feedforward_block(
    layer: nn.TransformerEncoderLayer,
    projections: nn.ModuleDict,
    inputs: Tensor,
    *,
    route_plan: DeckRoutePlan,
    capsules: nn.ModuleDict,
) -> Tensor:
    """Apply the feed-forward block with both shared and exact factors."""
    hidden = cast(Tensor, layer.linear1(inputs))
    hidden = cast(Any, projections["ffn_input"]).add_delta(
        hidden,
        inputs,
        route_plan=route_plan,
        capsules=capsules,
    )
    activated = layer.dropout(layer.activation(hidden))
    output = cast(Tensor, layer.linear2(activated))
    output = cast(Any, projections["ffn_output"]).add_delta(
        output,
        activated,
        route_plan=route_plan,
        capsules=capsules,
    )
    return cast(Tensor, layer.dropout2(output))


def _private_strategy_stacks(
    conditioning: DeckConditioningConfig,
    *,
    upper_layers: Sequence[nn.TransformerEncoderLayer],
    output_norm: nn.Module,
    d_model: int,
) -> nn.ModuleDict:
    """Clone one dense upper Transformer stack for every private strategy."""
    dense_config = conditioning.dense_private
    if dense_config is None:
        raise ValueError("dense-private stack construction requires its config")
    private_layer_start = dense_config.shared_transformer_layers
    residual_layer_indices = set(conditioning.adapter_layer_indices)
    module_keys = tuple(
        sorted({profile.module_key for profile in conditioning.active_routes})
    )
    return nn.ModuleDict(
        {
            module_key: DensePrivateTransformerStack(
                upper_layers,
                output_norm=output_norm,
                residual_adapters=tuple(
                    PrivateResidualAdapter(
                        d_model,
                        conditioning.adapter_bottleneck_dim,
                        dropout=conditioning.adapter_dropout,
                    )
                    if private_layer_start + relative_index in residual_layer_indices
                    else None
                    for relative_index in range(len(upper_layers))
                ),
            )
            for module_key in module_keys
        }
    )


def _forward_routed_lora_layer(
    layer: nn.TransformerEncoderLayer,
    lora_modules: nn.ModuleDict,
    inputs: Tensor,
    *,
    padding_mask: Tensor,
    route_plan: DeckRoutePlan,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Evaluate one Transformer layer with factorized routed projections."""
    if int(inputs.shape[0]) != route_plan.batch_size:
        raise ValueError("LoRA Transformer inputs must align with route plan")
    x = inputs
    if layer.norm_first:
        x = x + _routed_attention_block(
            layer,
            lora_modules,
            layer.norm1(x),
            padding_mask=padding_mask,
            route_plan=route_plan,
            attention_mask=attention_mask,
        )
        return x + _routed_feedforward_block(
            layer,
            lora_modules,
            layer.norm2(x),
            route_plan=route_plan,
        )
    x = layer.norm1(
        x
        + _routed_attention_block(
            layer,
            lora_modules,
            x,
            padding_mask=padding_mask,
            route_plan=route_plan,
            attention_mask=attention_mask,
        )
    )
    return cast(
        Tensor,
        layer.norm2(
            x
            + _routed_feedforward_block(
                layer,
                lora_modules,
                x,
                route_plan=route_plan,
            )
        ),
    )


def _routed_attention_block(
    layer: nn.TransformerEncoderLayer,
    modules: nn.ModuleDict,
    inputs: Tensor,
    *,
    padding_mask: Tensor,
    route_plan: DeckRoutePlan,
    attention_mask: Tensor | None,
) -> Tensor:
    """Apply self-attention with factorized QKV and output deltas."""
    attention = layer.self_attn
    in_proj_weight = attention.in_proj_weight
    if in_proj_weight is None or not attention._qkv_same_embed_dim:
        raise RuntimeError("routed Transformer LoRA requires packed QKV weights")
    qkv = functional.linear(inputs, in_proj_weight, attention.in_proj_bias)
    if "attention_qkv" in modules:
        qkv = _add_routed_lora_delta_(
            modules,
            "attention_qkv",
            qkv,
            inputs,
            route_plan=route_plan,
        )
    query, key, value = qkv.chunk(3, dim=-1)
    batch_size, token_count, d_model = inputs.shape
    num_heads = attention.num_heads
    head_dim = d_model // num_heads
    query = query.view(batch_size, token_count, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, token_count, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, token_count, num_heads, head_dim).transpose(1, 2)
    if attention_mask is None:
        attention_mask = _routed_attention_mask(inputs, padding_mask)
    attended = functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=attention.dropout if layer.training else 0.0,
        is_causal=False,
    )
    attended = (
        attended.transpose(1, 2).contiguous().view(batch_size, token_count, d_model)
    )
    output = cast(Tensor, attention.out_proj(attended))
    if "attention_output" in modules:
        output = _add_routed_lora_delta_(
            modules,
            "attention_output",
            output,
            attended,
            route_plan=route_plan,
        )
    return cast(Tensor, layer.dropout1(output))


def _routed_attention_mask(inputs: Tensor, padding_mask: Tensor) -> Tensor:
    """Build the shared additive padding mask once for all routed layers."""
    batch_size, token_count = padding_mask.shape
    attention_mask = inputs.new_zeros((batch_size, 1, 1, token_count))
    return attention_mask.masked_fill(
        padding_mask[:, None, None, :],
        float("-inf"),
    )


def _routed_feedforward_block(
    layer: nn.TransformerEncoderLayer,
    modules: nn.ModuleDict,
    inputs: Tensor,
    *,
    route_plan: DeckRoutePlan,
) -> Tensor:
    """Apply the feed-forward block with factorized routed deltas."""
    hidden = cast(Tensor, layer.linear1(inputs))
    if "ffn_input" in modules:
        hidden = _add_routed_lora_delta_(
            modules,
            "ffn_input",
            hidden,
            inputs,
            route_plan=route_plan,
        )
    activated = layer.dropout(layer.activation(hidden))
    output = cast(Tensor, layer.linear2(activated))
    if "ffn_output" in modules:
        output = _add_routed_lora_delta_(
            modules,
            "ffn_output",
            output,
            activated,
            route_plan=route_plan,
        )
    return cast(Tensor, layer.dropout2(output))


def _add_routed_lora_delta_(
    modules: nn.ModuleDict,
    target: str,
    output: Tensor,
    inputs: Tensor,
    *,
    route_plan: DeckRoutePlan,
) -> Tensor:
    """Add one Transformer projection delta to active expert rows."""
    bank = cast(RoutedLinearLoRA, modules[target])
    return bank.add_routed_delta_(
        output,
        inputs,
        route_plan.groups,
        dispatch=route_plan.lora_dispatch,
    )


def _token_scalars(
    token: StateToken,
    current: Any,
    select: Any,
    players: Sequence[Any],
    layout: StateTokenLayout,
) -> list[float]:
    scalars = [0.0] * TOKEN_SCALAR_SIZE
    _set_owner_scalars(scalars, token.key.player_index, layout.your_index)
    if token.kind == "global":
        _fill_global_scalars(scalars, current, select, players, layout.your_index)
        _fill_history_scalars(scalars, layout)
        _fill_deck_flow_scalars(scalars, layout)
        _fill_belief_global_scalars(scalars, layout)
        return scalars
    if token.count > 0:
        scalars[52] = _clamp(float(token.count) / 60.0)

    pokemon = _pokemon_for_token(token, players)
    if pokemon is not None:
        _fill_pokemon_scalars(scalars, pokemon)
        player = _player_by_index(players, token.key.player_index)
        if player is not None and token.key.area == int(AreaType.ACTIVE):
            _fill_special_condition_scalars(scalars, player)
    return scalars


def _split_global_context_scalars(batch: StateBatch) -> tuple[Tensor, Tensor, Tensor]:
    """Separate new global features from the checkpoint-era shared projection."""
    if int(batch.scalars.shape[-1]) != TOKEN_SCALAR_SIZE:
        raise ValueError(
            "state scalar width does not match the current public input schema"
        )
    context_values = batch.scalars[..., :GLOBAL_CONTEXT_SCALAR_SIZE]
    global_tokens = (batch.token_kinds == TOKEN_KIND_TO_INDEX["global"]).unsqueeze(-1)
    zero_context = torch.zeros_like(context_values)
    shared_context = torch.where(global_tokens, zero_context, context_values)
    shared_scalars = torch.cat(
        (
            shared_context,
            batch.scalars[..., GLOBAL_CONTEXT_SCALAR_SIZE:LEGACY_TOKEN_SCALAR_SIZE],
        ),
        dim=-1,
    )
    global_context = torch.where(global_tokens, context_values, zero_context)
    public_values = batch.scalars[..., PUBLIC_STATE_SCALAR_START:]
    public_state = torch.where(
        global_tokens,
        public_values,
        torch.zeros_like(public_values),
    )
    return shared_scalars, global_context, public_state


def _fill_pokemon_scalars(scalars: list[float], pokemon: Any) -> None:
    hp = max(0.0, _float_field(pokemon, "hp", 0.0))
    max_hp = max(1.0, _float_field(pokemon, "maxHp", 0.0))
    scalars[0] = _clamp(hp / max_hp)
    scalars[1] = _clamp((max_hp - hp) / max_hp)
    scalars[2] = _clamp(max_hp / 400.0)
    for energy in _sequence(_field(pokemon, "energies", ())):
        energy_index = _int_value(energy)
        if 0 <= energy_index < ENERGY_TYPE_COUNT:
            scalars[4 + energy_index] += 1.0 / 8.0
    for index in range(4, 4 + ENERGY_TYPE_COUNT):
        scalars[index] = _clamp(scalars[index])
    scalars[16] = _clamp(len(_sequence(_field(pokemon, "tools", ()))) / 4.0)
    scalars[22] = 1.0 if bool(_field(pokemon, "appearThisTurn", False)) else 0.0
    scalars[23] = _clamp(len(_sequence(_field(pokemon, "preEvolution", ()))) / 3.0)


def _fill_special_condition_scalars(scalars: list[float], player: Any) -> None:
    scalars[17] = 1.0 if bool(_field(player, "poisoned", False)) else 0.0
    scalars[18] = 1.0 if bool(_field(player, "burned", False)) else 0.0
    scalars[19] = 1.0 if bool(_field(player, "asleep", False)) else 0.0
    scalars[20] = 1.0 if bool(_field(player, "paralyzed", False)) else 0.0
    scalars[21] = 1.0 if bool(_field(player, "confused", False)) else 0.0


def _fill_global_scalars(
    scalars: list[float],
    current: Any,
    select: Any,
    players: Sequence[Any],
    your_index: int,
) -> None:
    opponent_index = 1 - your_index if your_index in (0, 1) else -1
    own_player = _player_by_index(players, your_index)
    opponent_player = _player_by_index(players, opponent_index)
    scalars[24] = _clamp(_float_field(current, "turn", 0.0) / 100.0)
    scalars[25] = _clamp(_float_field(current, "turnActionCount", 0.0) / 20.0)
    scalars[26] = 1.0 if _int_field(current, "firstPlayer", -1) == your_index else 0.0
    scalars[27] = 1.0 if bool(_field(current, "supporterPlayed", False)) else 0.0
    scalars[28] = 1.0 if bool(_field(current, "stadiumPlayed", False)) else 0.0
    scalars[29] = 1.0 if bool(_field(current, "energyAttached", False)) else 0.0
    scalars[30] = 1.0 if bool(_field(current, "retreated", False)) else 0.0
    scalars[31] = _float_field(current, "result", 0.0) / 2.0
    scalars[32] = _clamp(_zone_count(own_player, "prize") / 6.0)
    scalars[33] = _clamp(_zone_count(opponent_player, "prize") / 6.0)
    scalars[34] = _clamp(_float_field(own_player, "deckCount", 0.0) / 60.0)
    scalars[35] = _clamp(_float_field(opponent_player, "deckCount", 0.0) / 60.0)
    scalars[36] = _clamp(_float_field(own_player, "handCount", 0.0) / 20.0)
    scalars[37] = _clamp(_float_field(opponent_player, "handCount", 0.0) / 20.0)
    scalars[38] = _clamp(_float_field(select, "minCount", 0.0) / 8.0)
    scalars[39] = _clamp(_float_field(select, "maxCount", 0.0) / 8.0)
    scalars[40] = _clamp(_float_field(select, "remainDamageCounter", 0.0) / 50.0)
    scalars[41] = _clamp(_float_field(select, "remainEnergyCost", 0.0) / 12.0)
    scalars[42] = _clamp(_float_field(select, "type", 0.0) / 64.0)
    scalars[43] = _clamp(_float_field(select, "context", 0.0) / 64.0)
    _fill_bench_capacity_scalars(
        scalars,
        own_player=own_player,
        opponent_player=opponent_player,
    )
    scalars[GLOBAL_LONG_TURN_SCALAR] = _saturating_ratio(
        _float_field(current, "turn", 0.0),
        scale=100.0,
    )


def _fill_bench_capacity_scalars(
    scalars: list[float],
    *,
    own_player: Any | None,
    opponent_player: Any | None,
) -> None:
    """Encode both public bench limits without conflating missing with zero."""
    for offset, player in enumerate((own_player, opponent_player)):
        value_index = PUBLIC_STATE_SCALAR_START + offset * 2
        presence_index = value_index + 1
        if player is None or not _has_field(player, "benchMax"):
            continue
        scalars[value_index] = _float_field(player, "benchMax", 0.0) / 8.0
        scalars[presence_index] = 1.0


def _fill_history_scalars(scalars: list[float], layout: StateTokenLayout) -> None:
    if len(layout.context_features.history_counts) != HISTORY_COUNTER_SIZE:
        raise ValueError("history context has invalid width")
    for index, value in enumerate(layout.context_features.history_counts):
        scalars[44 + index] = _clamp(float(value) / 40.0)
        scalars[GLOBAL_UNCLIPPED_HISTORY_SCALAR_START + index] = _saturating_ratio(
            float(value), scale=20.0
        )


def _fill_deck_flow_scalars(
    scalars: list[float],
    layout: StateTokenLayout,
) -> None:
    """Encode public deck circulation without changing checkpoint dimensions."""
    values = layout.context_features.deck_flow_counts
    if len(values) != DECK_FLOW_FEATURE_SIZE:
        raise ValueError("deck-flow context has invalid width")
    for player_offset in (0, 7):
        output_offset = GLOBAL_DECK_FLOW_SCALAR_START + player_offset
        scalars[output_offset] = _saturating_ratio(
            float(values[player_offset]),
            scale=40.0,
        )
        scalars[output_offset + 1] = _saturating_ratio(
            float(values[player_offset + 1]),
            scale=20.0,
        )
        scalars[output_offset + 2] = _saturating_ratio(
            float(values[player_offset + 2]),
            scale=4.0,
        )
        scalars[output_offset + 3] = _saturating_ratio(
            float(values[player_offset + 3]),
            scale=4.0,
        )
        scalars[output_offset + 4] = _signed_saturating_ratio(
            float(values[player_offset + 4]),
            scale=4.0,
        )
        scalars[output_offset + 5] = _saturating_ratio(
            float(values[player_offset + 5]),
            scale=10.0,
        )
        scalars[output_offset + 6] = _saturating_ratio(
            float(values[player_offset + 6]),
            scale=10.0,
        )


def _fill_belief_global_scalars(
    scalars: list[float],
    layout: StateTokenLayout,
) -> None:
    scalars[53] = _clamp(float(layout.context_features.opponent_belief_entropy))
    scalars[54] = 1.0 if layout.context_features.opponent_belief_empty else 0.0


def _set_owner_scalars(
    scalars: list[float],
    player_index: int,
    your_index: int,
) -> None:
    if player_index == your_index:
        scalars[3] = 1.0
    elif player_index in (0, 1):
        scalars[3] = -1.0


def _pokemon_for_token(token: StateToken, players: Sequence[Any]) -> Any | None:
    if token.key.area not in {int(AreaType.ACTIVE), int(AreaType.BENCH)}:
        return None
    player = _player_by_index(players, token.key.player_index)
    if player is None:
        return None
    field_name = "active" if token.key.area == int(AreaType.ACTIVE) else "bench"
    return _item(_sequence(_field(player, field_name, ())), token.key.index)


def _pokemon_attachments(pokemon: Any) -> tuple[tuple[int, int], ...]:
    """Return visible attached card IDs with their engine attachment kind."""
    attachments: list[tuple[int, int]] = []
    fields = (
        ("energyCards", ATTACHMENT_KIND_ENERGY),
        ("tools", ATTACHMENT_KIND_TOOL),
        ("preEvolution", ATTACHMENT_KIND_PRE_EVOLUTION),
    )
    for field_name, attachment_kind in fields:
        for card in _sequence(_field(pokemon, field_name, ())):
            card_id = _int_field(card, "id", 0)
            if card_id > 0:
                attachments.append((card_id, attachment_kind))
    return tuple(attachments)


def _copy_attachments(
    card_destination: np.ndarray,
    parent_destination: np.ndarray,
    kind_destination: np.ndarray,
    item: StateTokenInput,
    *,
    token_count: int,
) -> None:
    card_ids = item.attachment_card_ids
    parents = item.attachment_parent_indices
    kinds = item.attachment_kinds
    if len(card_ids) != len(parents) or len(card_ids) != len(kinds):
        raise ValueError("attachment feature arrays must align")
    for attachment_index, (card_id, parent, kind) in enumerate(
        zip(card_ids, parents, kinds, strict=True)
    ):
        if parent < 0 or parent >= token_count:
            raise ValueError("attachment parent token index is out of range")
        if kind < 1 or kind > ATTACHMENT_KIND_COUNT:
            raise ValueError("attachment kind is out of range")
        card_destination[attachment_index] = _uint16_value(card_id, "card ID")
        parent_destination[attachment_index] = _uint16_value(
            parent,
            "attachment parent token index",
        )
        kind_destination[attachment_index] = kind


def _uint16_value(value: int, name: str) -> int:
    normalized = int(value)
    if normalized < 0 or normalized > np.iinfo(np.uint16).max:
        raise ValueError(f"{name} does not fit uint16: {normalized}")
    return normalized


def _player_by_index(players: Sequence[Any], player_index: int) -> Any | None:
    if player_index < 0 or player_index >= len(players):
        return None
    return players[player_index]


def _zone_count(player: Any | None, field_name: str) -> float:
    if player is None:
        return 0.0
    return float(len(_sequence(_field(player, field_name, ()))))


def _safe_area_index(area: int) -> int:
    if VIRTUAL_AREA <= area < AREA_OOV_INDEX:
        return int(area)
    return AREA_OOV_INDEX


def _owner_role(player_index: int, your_index: int) -> int:
    if player_index == your_index:
        return OWNER_SELF
    if player_index in (0, 1):
        return OWNER_OPPONENT
    if player_index < 0:
        return OWNER_SHARED
    return OWNER_UNKNOWN


def _safe_embedding_rows(values: Tensor, oov_index: int) -> Tensor:
    return torch.where(
        (values >= 0) & (values <= oov_index),
        values,
        torch.full_like(values, oov_index),
    )


def _item(values: Sequence[Any], index: int) -> Any | None:
    if index < 0 or index >= len(values):
        return None
    return values[index]


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _has_field(value: Any, name: str) -> bool:
    if isinstance(value, Mapping):
        return name in value
    return hasattr(value, name)


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_value(value: Any) -> int:
    return int(value) if value is not None else 0


def _int_field(value: Any, name: str, default: int) -> int:
    field_value = _field(value, name, default)
    return int(field_value) if field_value is not None else default


def _float_field(value: Any, name: str, default: float) -> float:
    field_value = _field(value, name, default)
    return float(field_value) if field_value is not None else default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def _saturating_ratio(value: float, *, scale: float) -> float:
    normalized = max(0.0, float(value))
    return normalized / (normalized + scale)


def _signed_saturating_ratio(value: float, *, scale: float) -> float:
    magnitude = abs(float(value))
    if magnitude == 0.0:
        return 0.0
    return (1.0 if value > 0.0 else -1.0) * magnitude / (magnitude + scale)
