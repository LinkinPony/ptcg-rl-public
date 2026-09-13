"""Public-event encoding and pure recurrent decision-state transitions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from torch import Tensor, nn
from torch.nn import functional

from ptcg_rl.activation_precision import preserve_cuda_bfloat16_activation
from ptcg_rl.cards.card_encoder import CardEncoder
from ptcg_rl.context import (
    PUBLIC_EVENT_ACTOR_ROLE_COUNT,
    PUBLIC_EVENT_AREA_COUNT,
    PUBLIC_EVENT_CATEGORICAL_COUNTS,
    PUBLIC_EVENT_ENTITY_COUNT,
    PUBLIC_EVENT_SCHEMA_FINGERPRINT,
    PUBLIC_EVENT_TYPE_COUNT,
    PublicEventBatch,
    validate_public_event_batch,
)

RECURRENT_POLICY_ARCHITECTURE_VERSION = 1
RECURRENT_POLICY_INITIALIZATION_SEED = 2026072201


class RecurrentPolicyConfig(BaseModel):
    """Config for the shared public-event encoder and decision LSTM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_version: int = RECURRENT_POLICY_ARCHITECTURE_VERSION
    public_event_schema_fingerprint: str = PUBLIC_EVENT_SCHEMA_FINGERPRINT
    hidden_size: int | None = None
    event_hidden_size: int | None = None
    num_layers: int = 1
    dropout: float = 0.0
    serial_hash_buckets: int = 4096
    attack_hash_buckets: int = 1024

    @field_validator("architecture_version")
    @classmethod
    def valid_architecture_version(cls, value: int) -> int:
        """Require the only recurrent architecture implemented here."""
        if value != RECURRENT_POLICY_ARCHITECTURE_VERSION:
            raise ValueError("unsupported recurrent policy architecture version")
        return value

    @field_validator("public_event_schema_fingerprint")
    @classmethod
    def valid_public_event_schema(cls, value: str) -> str:
        """Bind model weights to the exact event reconstruction contract."""
        if value != PUBLIC_EVENT_SCHEMA_FINGERPRINT:
            raise ValueError("recurrent policy public event schema mismatch")
        return value

    @field_validator(
        "hidden_size",
        "event_hidden_size",
        "num_layers",
        "serial_hash_buckets",
        "attack_hash_buckets",
    )
    @classmethod
    def valid_positive_dimension(cls, value: int | None) -> int | None:
        """Reject non-positive recurrent dimensions and bucket counts."""
        if value is not None and value <= 0:
            raise ValueError("recurrent dimensions must be positive")
        return value

    @field_validator("dropout")
    @classmethod
    def valid_dropout(cls, value: float) -> float:
        """Reject non-finite or invalid dropout probabilities."""
        if not math.isfinite(value) or value < 0.0 or value >= 1.0:
            raise ValueError("recurrent dropout must be finite and in [0, 1)")
        return value

    @model_validator(mode="after")
    def valid_layer_dropout(self) -> RecurrentPolicyConfig:
        """Avoid PyTorch's silent single-layer recurrent dropout no-op."""
        if self.num_layers == 1 and self.dropout != 0.0:
            raise ValueError("single-layer recurrent policy requires zero dropout")
        return self

    def resolved_hidden_size(self, d_model: int) -> int:
        """Return the configured recurrent width or the state width."""
        return d_model if self.hidden_size is None else self.hidden_size

    def resolved_event_hidden_size(self, d_model: int) -> int:
        """Return the configured intra-decision event width or state width."""
        return d_model if self.event_hidden_size is None else self.event_hidden_size


@dataclass(frozen=True)
class RecurrentPolicyState:
    """Explicit LSTM state proposed by one pure decision transition."""

    hidden: Tensor
    cell: Tensor

    def __post_init__(self) -> None:
        """Reject state tensors that cannot describe the same LSTM lease."""
        if self.hidden.ndim != 3 or self.cell.ndim != 3:
            raise ValueError("recurrent state tensors must have shape [L, B, H]")
        if self.hidden.shape != self.cell.shape:
            raise ValueError("recurrent hidden and cell shapes differ")
        if self.hidden.device != self.cell.device:
            raise ValueError("recurrent hidden and cell devices differ")
        if self.hidden.dtype != self.cell.dtype:
            raise ValueError("recurrent hidden and cell dtypes differ")

    @property
    def batch_size(self) -> int:
        """Return the number of independently isolated decision sequences."""
        return int(self.hidden.shape[1])

    def detach(self) -> RecurrentPolicyState:
        """Detach both tensors at an explicit actor or truncation boundary."""
        return RecurrentPolicyState(self.hidden.detach(), self.cell.detach())


def select_recurrent_policy_state_rows(
    state: RecurrentPolicyState,
    indices: Tensor,
) -> RecurrentPolicyState:
    """Select or repeat recurrent rows along the LSTM batch dimension."""
    _validate_recurrent_policy_state(state)
    if not isinstance(indices, Tensor):
        raise TypeError("recurrent state row indices must be a tensor")
    if indices.ndim != 1:
        raise ValueError("recurrent state row indices must be one-dimensional")
    if indices.dtype != torch.long:
        raise TypeError("recurrent state row indices must use torch.long")
    if indices.device != state.hidden.device:
        raise ValueError("recurrent state row indices must share the state device")
    if indices.numel() and bool(
        ((indices < 0) | (indices >= state.batch_size)).any().item()
    ):
        raise IndexError("recurrent state row index is out of range")
    return RecurrentPolicyState(
        hidden=state.hidden.index_select(1, indices),
        cell=state.cell.index_select(1, indices),
    )


def concatenate_recurrent_policy_states(
    states: Sequence[RecurrentPolicyState],
) -> RecurrentPolicyState:
    """Concatenate compatible recurrent states along batch dimension one."""
    if not states:
        raise ValueError("cannot concatenate an empty recurrent state sequence")
    for state in states:
        _validate_recurrent_policy_state(state)
    reference = states[0]
    layout = (reference.hidden.shape[0], reference.hidden.shape[2])
    for state in states[1:]:
        if (state.hidden.shape[0], state.hidden.shape[2]) != layout:
            raise ValueError("recurrent states have incompatible non-batch shapes")
        if state.hidden.device != reference.hidden.device:
            raise ValueError("recurrent states must use the same device")
        if state.hidden.dtype != reference.hidden.dtype:
            raise TypeError("recurrent states must use the same dtype")
    return RecurrentPolicyState(
        hidden=torch.cat(tuple(state.hidden for state in states), dim=1),
        cell=torch.cat(tuple(state.cell for state in states), dim=1),
    )


class PublicEventEncoder(nn.Module):
    """Encode one bounded ordered public-event delta into a decision vector."""

    def __init__(
        self,
        d_model: int,
        config: RecurrentPolicyConfig,
    ) -> None:
        """Initialize field-aware embeddings and an ordered event GRU."""
        super().__init__()
        if d_model <= 0:
            raise ValueError("event encoder d_model must be positive")
        self.d_model = d_model
        self.config = config
        event_hidden_size = config.resolved_event_hidden_size(d_model)
        self.event_type_embedding = nn.Embedding(
            PUBLIC_EVENT_TYPE_COUNT,
            d_model,
            padding_idx=0,
        )
        self.actor_embedding = nn.Embedding(
            PUBLIC_EVENT_ACTOR_ROLE_COUNT,
            d_model,
        )
        self.from_area_embedding = nn.Embedding(PUBLIC_EVENT_AREA_COUNT, d_model)
        self.to_area_embedding = nn.Embedding(PUBLIC_EVENT_AREA_COUNT, d_model)
        self.entity_field_embedding = nn.Embedding(
            PUBLIC_EVENT_ENTITY_COUNT,
            d_model,
        )
        self.entity_field_projections = nn.ModuleList(
            nn.Linear(d_model, d_model, bias=False)
            for _index in range(PUBLIC_EVENT_ENTITY_COUNT)
        )
        self.serial_embedding = nn.Embedding(
            config.serial_hash_buckets + 1,
            d_model,
            padding_idx=0,
        )
        self.attack_embedding = nn.Embedding(
            config.attack_hash_buckets + 1,
            d_model,
            padding_idx=0,
        )
        self.categorical_embeddings = nn.ModuleList(
            nn.Embedding(count, d_model) for count in PUBLIC_EVENT_CATEGORICAL_COUNTS
        )
        self.value_projection = nn.Linear(2, d_model, bias=False)
        self.event_input_norm = nn.LayerNorm(d_model)
        self.event_cell = nn.GRUCell(d_model, event_hidden_size)
        self.event_projection = nn.Linear(event_hidden_size, d_model, bias=False)
        self.overflow_projection = nn.Linear(
            PUBLIC_EVENT_TYPE_COUNT * PUBLIC_EVENT_ACTOR_ROLE_COUNT,
            d_model,
            bias=False,
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        batch: PublicEventBatch,
        *,
        card_encoder: CardEncoder,
    ) -> Tensor:
        """Return an order-sensitive vector for every decision event delta."""
        validate_public_event_batch(batch)
        if card_encoder.d_model != self.d_model:
            raise ValueError("event and card encoder dimensions differ")
        tokens = self._event_tokens(batch, card_encoder=card_encoder)
        batch_size, event_width, _ = tokens.shape
        history = tokens.new_zeros((batch_size, self.event_cell.hidden_size))
        for index in range(event_width):
            proposed = self.event_cell(tokens[:, index, :], history)
            valid = ~batch.padding_mask[:, index]
            history = torch.where(valid.unsqueeze(-1), proposed, history)

        overflow = (
            torch.log1p(batch.overflow_type_actor_counts.to(dtype=torch.float32))
            .to(dtype=self.overflow_projection.weight.dtype)
            .flatten(start_dim=1)
        )
        encoded = self.event_projection(history) + self.overflow_projection(overflow)
        encoded = preserve_cuda_bfloat16_activation(self.output_norm(encoded))
        has_evidence = (~batch.padding_mask).any(dim=1) | (
            batch.overflow_type_actor_counts != 0
        ).flatten(start_dim=1).any(dim=1)
        return encoded.masked_fill(~has_evidence.unsqueeze(-1), 0.0)

    def _event_tokens(
        self,
        batch: PublicEventBatch,
        *,
        card_encoder: CardEncoder,
    ) -> Tensor:
        token = (
            self.event_type_embedding(batch.event_types)
            + self.actor_embedding(batch.actor_roles)
            + self.from_area_embedding(batch.from_areas)
            + self.to_area_embedding(batch.to_areas)
        )
        entity_mask = batch.entity_mask.unsqueeze(-1)
        cards = card_encoder(batch.card_ids)
        serial_indices = _masked_hash_indices(
            batch.serials,
            batch.entity_mask,
            buckets=self.config.serial_hash_buckets,
        )
        serials = self.serial_embedding(serial_indices)
        field_ids = torch.arange(
            PUBLIC_EVENT_ENTITY_COUNT,
            dtype=torch.long,
            device=token.device,
        )
        fields = self.entity_field_embedding(field_ids).view(
            1,
            1,
            PUBLIC_EVENT_ENTITY_COUNT,
            self.d_model,
        )
        entity_inputs = cards + serials + fields
        entity_values = torch.stack(
            tuple(
                projection(entity_inputs[:, :, index, :])
                for index, projection in enumerate(self.entity_field_projections)
            ),
            dim=2,
        ).masked_fill(~entity_mask, 0.0)
        entity_count = (
            batch.entity_mask.sum(dim=-1, keepdim=True)
            .to(dtype=entity_values.dtype)
            .clamp(min=1)
        )
        token = token + entity_values.sum(dim=2) / entity_count.sqrt()

        attack_indices = _masked_hash_indices(
            batch.attack_ids,
            batch.attack_id_mask,
            buckets=self.config.attack_hash_buckets,
        )
        token = token + self.attack_embedding(attack_indices)
        signed_values = torch.sign(batch.values) * torch.log1p(batch.values.abs())
        value_features = torch.stack(
            (
                signed_values * batch.value_mask,
                batch.value_mask.to(dtype=batch.values.dtype),
            ),
            dim=-1,
        ).to(dtype=self.value_projection.weight.dtype)
        token = token + self.value_projection(value_features)
        for index, embedding in enumerate(self.categorical_embeddings):
            token = token + embedding(batch.categorical_values[:, :, index])
        token = preserve_cuda_bfloat16_activation(self.event_input_norm(token))
        return token.masked_fill(batch.padding_mask.unsqueeze(-1), 0.0)


class RecurrentPolicyCore(nn.Module):
    """Shared pure LSTM transition between DCCR and all decision heads."""

    def __init__(self, d_model: int, config: RecurrentPolicyConfig) -> None:
        """Initialize deterministic event and recurrent parameters."""
        super().__init__()
        if d_model <= 0:
            raise ValueError("recurrent d_model must be positive")
        self.d_model = d_model
        self.config = config
        hidden_size = config.resolved_hidden_size(d_model)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(RECURRENT_POLICY_INITIALIZATION_SEED)
            self.event_encoder = PublicEventEncoder(d_model, config)
            self.input_projection = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.cells = nn.ModuleList(
                nn.LSTMCell(d_model if index == 0 else hidden_size, hidden_size)
                for index in range(config.num_layers)
            )
            self.residual_projection = nn.Linear(hidden_size, d_model)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RecurrentPolicyState:
        """Create one zero state without storing it inside the model."""
        if batch_size <= 0:
            raise ValueError("recurrent batch_size must be positive")
        shape = (
            self.config.num_layers,
            batch_size,
            self.config.resolved_hidden_size(self.d_model),
        )
        return RecurrentPolicyState(
            hidden=torch.zeros(shape, device=device, dtype=dtype),
            cell=torch.zeros(shape, device=device, dtype=dtype),
        )

    def step(
        self,
        state_embedding: Tensor,
        events: PublicEventBatch,
        *,
        card_encoder: CardEncoder,
        previous_state: RecurrentPolicyState,
    ) -> tuple[Tensor, RecurrentPolicyState]:
        """Propose one decision embedding and state; mutate no caller state."""
        if state_embedding.ndim != 2 or state_embedding.shape[-1] != self.d_model:
            raise ValueError("recurrent step state embedding must have shape [B, D]")
        if events.batch_size != int(state_embedding.shape[0]):
            raise ValueError("recurrent step events must align with state rows")
        event_embedding = self.event_encoder(events, card_encoder=card_encoder)
        current = self._validated_state(previous_state, state_embedding)
        output, proposed = self._step_embeddings(
            state_embedding,
            event_embedding,
            current,
        )
        return (state_embedding + self.residual_projection(output), proposed)

    def unroll(
        self,
        state_embeddings: Tensor,
        events: PublicEventBatch,
        padding_mask: Tensor,
        *,
        card_encoder: CardEncoder,
        initial_state: RecurrentPolicyState,
    ) -> tuple[Tensor, RecurrentPolicyState]:
        """Replay complete left-aligned sequences from an explicit boundary.

        This first architecture intentionally has no internal reset mask: every
        row is one full game-seat-policy sequence and ``initial_state`` names
        its externally validated reset or burn-in boundary.
        """
        if state_embeddings.ndim != 3 or state_embeddings.shape[-1] != self.d_model:
            raise ValueError(
                "recurrent unroll state embeddings must have shape [B, S, D]"
            )
        batch_size, sequence_length, _ = state_embeddings.shape
        if padding_mask.dtype != torch.bool or padding_mask.shape != (
            batch_size,
            sequence_length,
        ):
            raise ValueError("recurrent padding_mask must be bool [B, S]")
        if padding_mask.device != state_embeddings.device:
            raise ValueError("recurrent padding mask and states use different devices")
        if events.batch_size != batch_size * sequence_length:
            raise ValueError("recurrent unroll events must flatten [B, S] rows")
        if bool((~padding_mask).sum(dim=1).eq(0).any().item()):
            raise ValueError("recurrent sequences must contain at least one decision")
        if bool(((~padding_mask[:, 1:]) & padding_mask[:, :-1]).any().item()):
            raise ValueError("recurrent sequence padding must be right-aligned")
        flat_events = self.event_encoder(events, card_encoder=card_encoder)
        event_embeddings = flat_events.reshape(
            batch_size,
            sequence_length,
            self.d_model,
        )
        current = self._validated_state(initial_state, state_embeddings[:, 0, :])
        decisions: list[Tensor] = []
        for index in range(sequence_length):
            state_row = state_embeddings[:, index, :]
            output, proposed = self._step_embeddings(
                state_row,
                event_embeddings[:, index, :],
                current,
            )
            valid = ~padding_mask[:, index]
            current = RecurrentPolicyState(
                hidden=torch.where(
                    valid.view(1, batch_size, 1),
                    proposed.hidden,
                    current.hidden,
                ),
                cell=torch.where(
                    valid.view(1, batch_size, 1),
                    proposed.cell,
                    current.cell,
                ),
            )
            decision = state_row + self.residual_projection(output)
            decisions.append(decision.masked_fill(~valid.unsqueeze(-1), 0.0))
        return (torch.stack(decisions, dim=1), current)

    def _validated_state(
        self,
        state: RecurrentPolicyState,
        reference: Tensor,
    ) -> RecurrentPolicyState:
        batch_size = int(reference.shape[0])
        expected_shape = (
            self.config.num_layers,
            batch_size,
            self.config.resolved_hidden_size(self.d_model),
        )
        if state.hidden.shape != expected_shape:
            raise ValueError("recurrent state shape is incompatible with this batch")
        if state.hidden.device != reference.device:
            raise ValueError("recurrent state and input devices differ")
        if state.hidden.dtype != reference.dtype:
            raise ValueError("recurrent state and input dtypes differ")
        return state

    def _step_embeddings(
        self,
        state_embedding: Tensor,
        event_embedding: Tensor,
        previous_state: RecurrentPolicyState,
    ) -> tuple[Tensor, RecurrentPolicyState]:
        layer_input = self.input_projection(
            torch.cat((state_embedding, event_embedding), dim=-1)
        )
        hidden_rows: list[Tensor] = []
        cell_rows: list[Tensor] = []
        for layer_index, cell in enumerate(self.cells):
            hidden, cell_state = cell(
                layer_input,
                (
                    previous_state.hidden[layer_index],
                    previous_state.cell[layer_index],
                ),
            )
            # Autocast may produce reduced-precision LSTM outputs even when the
            # snapshot and committed actor state are FP32. Keep the persistent
            # state lease in its caller-declared dtype so the next decision has
            # the same explicit state contract.
            hidden_rows.append(hidden.to(dtype=previous_state.hidden.dtype))
            cell_rows.append(cell_state.to(dtype=previous_state.cell.dtype))
            layer_input = hidden
            if layer_index + 1 < len(self.cells) and self.config.dropout:
                layer_input = functional.dropout(
                    layer_input,
                    p=self.config.dropout,
                    training=self.training,
                )
        return (
            hidden_rows[-1],
            RecurrentPolicyState(
                hidden=torch.stack(hidden_rows),
                cell=torch.stack(cell_rows),
            ),
        )


def _masked_hash_indices(
    values: Tensor,
    mask: Tensor,
    *,
    buckets: int,
) -> Tensor:
    hashed = torch.remainder(values.to(dtype=torch.long), buckets) + 1
    return torch.where(mask, hashed, torch.zeros_like(hashed))


def _validate_recurrent_policy_state(state: RecurrentPolicyState) -> None:
    """Revalidate a recurrent state at reusable batch-operation boundaries."""
    if not isinstance(state, RecurrentPolicyState):
        raise TypeError("recurrent state must be a RecurrentPolicyState")
    if not isinstance(state.hidden, Tensor) or not isinstance(state.cell, Tensor):
        raise TypeError("recurrent hidden and cell values must be tensors")
    if state.hidden.ndim != 3 or state.cell.ndim != 3:
        raise ValueError("recurrent state tensors must have shape [L, B, H]")
    if state.hidden.shape != state.cell.shape:
        raise ValueError("recurrent hidden and cell shapes differ")
    if state.hidden.device != state.cell.device:
        raise ValueError("recurrent hidden and cell devices differ")
    if state.hidden.dtype != state.cell.dtype:
        raise TypeError("recurrent hidden and cell dtypes differ")


__all__ = [
    "RECURRENT_POLICY_ARCHITECTURE_VERSION",
    "RECURRENT_POLICY_INITIALIZATION_SEED",
    "PublicEventEncoder",
    "RecurrentPolicyConfig",
    "RecurrentPolicyCore",
    "RecurrentPolicyState",
    "concatenate_recurrent_policy_states",
    "select_recurrent_policy_state_rows",
]
