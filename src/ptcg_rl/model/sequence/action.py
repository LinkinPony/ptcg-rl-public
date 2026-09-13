"""Stable accepted-action records and their local temporal encoder."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch
from torch import Tensor, nn

from ptcg_rl.actions.encoding import (
    SCALAR_FEATURE_SIZE,
    EncodedOptionArrayFeatures,
)
from ptcg_rl.actions.selection import ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
from ptcg_rl.activation_precision import preserve_cuda_bfloat16_activation
from ptcg_rl.cards.card_encoder import CardEncoder
from ptcg_rl.model.policy import (
    CONTEXT_EMBEDDING_COUNT,
    CONTEXT_OOV_INDEX,
    KNOWN_CONTEXT_COUNT,
    KNOWN_OPTION_TYPE_COUNT,
    MAX_ENTITY_SLOTS,
    OPTION_TYPE_EMBEDDING_COUNT,
    OPTION_TYPE_OOV_INDEX,
)
from ptcg_rl.model.state_encoder import (
    AREA_EMBEDDING_COUNT,
    OWNER_ROLE_COUNT,
    TOKEN_KIND_COUNT,
    TOKEN_SCALAR_SIZE,
    StateTokenArrayFeatures,
)

ACCEPTED_ACTION_SCHEMA_VERSION = 1
_ACTION_IDENTITY_DOMAIN = b"ptcg-rl/accepted-complete-action/v1\x00"
_ENTITY_NUMERIC_SIZE = MAX_ENTITY_SLOTS * TOKEN_SCALAR_SIZE
_OPTION_NUMERIC_SIZE = SCALAR_FEATURE_SIZE + _ENTITY_NUMERIC_SIZE
_StableOptionPayload: TypeAlias = tuple[
    bytes,
    int,
    int,
    int,
    int,
    tuple[float, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[float, ...],
]


@dataclass(frozen=True)
class AcceptedActionRecord:
    """Raw, stable representation of one actually served complete action."""

    schema_version: int
    stable_identity: str
    prompt_context: int
    option_types: tuple[int, ...]
    option_contexts: tuple[int, ...]
    card_ids: tuple[int, ...]
    attack_ids: tuple[int, ...]
    option_scalars: tuple[tuple[float, ...], ...]
    entity_card_ids: tuple[tuple[int, ...], ...]
    entity_areas: tuple[tuple[int, ...], ...]
    entity_owner_roles: tuple[tuple[int, ...], ...]
    entity_token_kinds: tuple[tuple[int, ...], ...]
    entity_scalars: tuple[tuple[float, ...], ...]
    ordered: bool
    stop_sampled: bool
    accepted: bool
    fallback: bool

    def __post_init__(self) -> None:
        """Reject partial or internally inconsistent action payloads."""
        if self.schema_version != ACCEPTED_ACTION_SCHEMA_VERSION:
            raise ValueError("unsupported accepted-action schema")
        if len(self.stable_identity) != 64:
            raise ValueError("accepted-action identity must be SHA-256")
        size = len(self.option_types)
        fields = (
            self.option_contexts,
            self.card_ids,
            self.attack_ids,
            self.option_scalars,
            self.entity_card_ids,
            self.entity_areas,
            self.entity_owner_roles,
            self.entity_token_kinds,
            self.entity_scalars,
        )
        if any(len(field) != size for field in fields):
            raise ValueError("accepted-action option fields are misaligned")
        if any(len(values) != SCALAR_FEATURE_SIZE for values in self.option_scalars):
            raise ValueError("accepted-action option scalar width differs")
        for field in (
            self.entity_card_ids,
            self.entity_areas,
            self.entity_owner_roles,
            self.entity_token_kinds,
        ):
            if any(len(values) != MAX_ENTITY_SLOTS for values in field):
                raise ValueError("accepted-action entity categorical width differs")
        if any(len(values) != _ENTITY_NUMERIC_SIZE for values in self.entity_scalars):
            raise ValueError("accepted-action entity scalar width differs")
        if not self.accepted:
            raise ValueError("only accepted served actions may enter temporal history")
        if any(
            not math.isfinite(value)
            for row in (*self.option_scalars, *self.entity_scalars)
            for value in row
        ):
            raise ValueError("accepted-action numeric fields must be finite")


@dataclass(frozen=True)
class AcceptedActionBatch:
    """Padded tensor representation of accepted complete actions."""

    option_types: Tensor
    option_contexts: Tensor
    card_ids: Tensor
    attack_ids: Tensor
    option_scalars: Tensor
    entity_card_ids: Tensor
    entity_areas: Tensor
    entity_owner_roles: Tensor
    entity_token_kinds: Tensor
    entity_scalars: Tensor
    valid_mask: Tensor
    prompt_contexts: Tensor
    lengths: Tensor
    ordered: Tensor
    stop_sampled: Tensor
    fallback: Tensor
    stable_identities: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        """Return the number of complete actions."""
        return int(self.option_types.shape[0])

    def __post_init__(self) -> None:
        """Validate the padded action tensor contract."""
        batch_size, width = self.option_types.shape
        categorical = (
            self.option_contexts,
            self.card_ids,
            self.attack_ids,
        )
        if any(value.shape != (batch_size, width) for value in categorical):
            raise ValueError("accepted-action categorical tensors are misaligned")
        if self.option_scalars.shape != (
            batch_size,
            width,
            SCALAR_FEATURE_SIZE,
        ):
            raise ValueError("accepted-action option scalar tensor differs")
        entity_shape = (batch_size, width, MAX_ENTITY_SLOTS)
        if any(
            value.shape != entity_shape
            for value in (
                self.entity_card_ids,
                self.entity_areas,
                self.entity_owner_roles,
                self.entity_token_kinds,
            )
        ):
            raise ValueError("accepted-action entity tensors are misaligned")
        if self.entity_scalars.shape != (
            batch_size,
            width,
            MAX_ENTITY_SLOTS,
            TOKEN_SCALAR_SIZE,
        ):
            raise ValueError("accepted-action entity scalar tensor differs")
        if self.valid_mask.shape != (batch_size, width):
            raise ValueError("accepted-action valid mask differs")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("accepted-action valid mask must be boolean")
        for value in (
            self.prompt_contexts,
            self.lengths,
            self.ordered,
            self.stop_sampled,
            self.fallback,
        ):
            if value.shape != (batch_size,):
                raise ValueError("accepted-action row tensors are misaligned")
        if self.stable_identities and len(self.stable_identities) != batch_size:
            raise ValueError("accepted-action identities are misaligned")


def build_accepted_action_record(
    *,
    state: StateTokenArrayFeatures,
    options: EncodedOptionArrayFeatures,
    action: tuple[int, ...],
    min_count: int,
    max_count: int,
    stop_sampled: bool,
    fallback: bool = False,
) -> AcceptedActionRecord:
    """Resolve transient option indices into stable actor-visible semantics."""
    _validate_accepted_action(
        options=options,
        action=action,
        min_count=min_count,
        max_count=max_count,
    )
    prompt_context = int(options.contexts[0]) if len(options) else CONTEXT_OOV_INDEX
    ordered = prompt_context not in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
    encoded = tuple(
        _stable_option_payload(state=state, options=options, option_index=index)
        for index in action
    )
    if not ordered:
        encoded = tuple(sorted(encoded, key=lambda item: item[0]))
    return _accepted_action_record_from_payloads(
        prompt_context=prompt_context,
        encoded=encoded,
        ordered=ordered,
        stop_sampled=stop_sampled,
        fallback=fallback,
    )


def build_accepted_action_records(
    *,
    states: Sequence[StateTokenArrayFeatures],
    options: Sequence[EncodedOptionArrayFeatures],
    actions: Sequence[tuple[int, ...]],
    min_counts: Sequence[int],
    max_counts: Sequence[int],
    stop_sampled: Sequence[bool],
    fallback: Sequence[bool] | None = None,
) -> tuple[AcceptedActionRecord, ...]:
    """Resolve an aligned host batch with shape-grouped NumPy gathers."""
    from ptcg_rl.model.sequence.host_action import (
        build_host_accepted_action_records,
    )

    return build_host_accepted_action_records(
        states=states,
        options=options,
        actions=actions,
        min_counts=min_counts,
        max_counts=max_counts,
        stop_sampled=stop_sampled,
        fallback=fallback,
    )


def _accepted_action_record_from_payloads(
    *,
    prompt_context: int,
    encoded: tuple[_StableOptionPayload, ...],
    ordered: bool,
    stop_sampled: bool,
    fallback: bool,
) -> AcceptedActionRecord:
    """Finalize one already resolved stable payload sequence."""
    identity = _complete_action_identity(
        prompt_context=prompt_context,
        encoded=encoded,
        stop_sampled=stop_sampled,
        fallback=fallback,
    )
    return AcceptedActionRecord(
        schema_version=ACCEPTED_ACTION_SCHEMA_VERSION,
        stable_identity=identity,
        prompt_context=prompt_context,
        option_types=tuple(item[1] for item in encoded),
        option_contexts=tuple(item[2] for item in encoded),
        card_ids=tuple(item[3] for item in encoded),
        attack_ids=tuple(item[4] for item in encoded),
        option_scalars=tuple(item[5] for item in encoded),
        entity_card_ids=tuple(item[6] for item in encoded),
        entity_areas=tuple(item[7] for item in encoded),
        entity_owner_roles=tuple(item[8] for item in encoded),
        entity_token_kinds=tuple(item[9] for item in encoded),
        entity_scalars=tuple(item[10] for item in encoded),
        ordered=ordered,
        stop_sampled=bool(stop_sampled),
        accepted=True,
        fallback=bool(fallback),
    )


def collate_accepted_actions(
    records: tuple[AcceptedActionRecord, ...],
    *,
    device: torch.device | str | None = None,
) -> AcceptedActionBatch:
    """Pad stable accepted-action records without losing selection order."""
    if not records:
        raise ValueError("accepted-action collation requires records")
    width = max(1, *(len(record.option_types) for record in records))
    batch_size = len(records)

    def integers(shape: tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape, dtype=np.int64)

    option_types = integers((batch_size, width))
    option_contexts = integers((batch_size, width))
    card_ids = integers((batch_size, width))
    attack_ids = integers((batch_size, width))
    option_scalars = np.zeros(
        (batch_size, width, SCALAR_FEATURE_SIZE),
        dtype=np.float32,
    )
    entity_card_ids = integers((batch_size, width, MAX_ENTITY_SLOTS))
    entity_areas = integers((batch_size, width, MAX_ENTITY_SLOTS))
    entity_owners = integers((batch_size, width, MAX_ENTITY_SLOTS))
    entity_kinds = integers((batch_size, width, MAX_ENTITY_SLOTS))
    entity_scalars = np.zeros(
        (batch_size, width, MAX_ENTITY_SLOTS, TOKEN_SCALAR_SIZE),
        dtype=np.float32,
    )
    valid = np.zeros((batch_size, width), dtype=np.bool_)
    for row, record in enumerate(records):
        length = len(record.option_types)
        if not length:
            continue
        target = (row, slice(0, length))
        option_types[target] = record.option_types
        option_contexts[target] = record.option_contexts
        card_ids[target] = record.card_ids
        attack_ids[target] = record.attack_ids
        option_scalars[target] = record.option_scalars
        entity_card_ids[target] = record.entity_card_ids
        entity_areas[target] = record.entity_areas
        entity_owners[target] = record.entity_owner_roles
        entity_kinds[target] = record.entity_token_kinds
        entity_scalars[target] = np.asarray(record.entity_scalars).reshape(
            length,
            MAX_ENTITY_SLOTS,
            TOKEN_SCALAR_SIZE,
        )
        valid[target] = True

    def tensor(value: np.ndarray) -> Tensor:
        return torch.as_tensor(value, device=device)

    return AcceptedActionBatch(
        option_types=tensor(option_types),
        option_contexts=tensor(option_contexts),
        card_ids=tensor(card_ids),
        attack_ids=tensor(attack_ids),
        option_scalars=tensor(option_scalars),
        entity_card_ids=tensor(entity_card_ids),
        entity_areas=tensor(entity_areas),
        entity_owner_roles=tensor(entity_owners),
        entity_token_kinds=tensor(entity_kinds),
        entity_scalars=tensor(entity_scalars),
        valid_mask=tensor(valid),
        prompt_contexts=tensor(
            np.asarray([record.prompt_context for record in records], dtype=np.int64)
        ),
        lengths=tensor(
            np.asarray([len(record.option_types) for record in records], dtype=np.int64)
        ),
        ordered=tensor(
            np.asarray([record.ordered for record in records], dtype=np.bool_)
        ),
        stop_sampled=tensor(
            np.asarray([record.stop_sampled for record in records], dtype=np.bool_)
        ),
        fallback=tensor(
            np.asarray([record.fallback for record in records], dtype=np.bool_)
        ),
        stable_identities=tuple(record.stable_identity for record in records),
    )


class AcceptedActionEncoder(nn.Module):
    """Encode raw accepted complete actions without prospective features."""

    def __init__(self, *, d_model: int, attack_hash_buckets: int) -> None:
        """Build field-aware option encoders and an ordered local GRU."""
        super().__init__()
        self.d_model = d_model
        self.attack_hash_buckets = attack_hash_buckets
        self.option_type_embedding = nn.Embedding(
            OPTION_TYPE_EMBEDDING_COUNT,
            d_model,
        )
        self.context_embedding = nn.Embedding(
            CONTEXT_EMBEDDING_COUNT,
            d_model,
        )
        self.attack_embedding = nn.Embedding(
            attack_hash_buckets + 1,
            d_model,
            padding_idx=0,
        )
        self.area_embedding = nn.Embedding(AREA_EMBEDDING_COUNT, d_model)
        self.owner_embedding = nn.Embedding(OWNER_ROLE_COUNT, d_model)
        self.kind_embedding = nn.Embedding(TOKEN_KIND_COUNT, d_model)
        self.option_scalar_projection = nn.Linear(
            SCALAR_FEATURE_SIZE,
            d_model,
            bias=False,
        )
        self.entity_scalar_projection = nn.Linear(
            TOKEN_SCALAR_SIZE,
            d_model,
            bias=False,
        )
        self.entity_role_embedding = nn.Embedding(MAX_ENTITY_SLOTS, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.cell = nn.GRUCell(d_model, d_model)
        self.prompt_projection = nn.Linear(5, d_model, bias=False)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        batch: AcceptedActionBatch,
        *,
        card_encoder: CardEncoder,
    ) -> Tensor:
        """Return one order-sensitive temporal ACTION token per record."""
        option_types = _safe_indices(
            batch.option_types,
            known=KNOWN_OPTION_TYPE_COUNT,
            oov=OPTION_TYPE_OOV_INDEX,
        )
        option_contexts = _safe_indices(
            batch.option_contexts,
            known=KNOWN_CONTEXT_COUNT,
            oov=CONTEXT_OOV_INDEX,
        )
        attack_indices = torch.where(
            batch.attack_ids > 0,
            torch.remainder(batch.attack_ids, self.attack_hash_buckets) + 1,
            torch.zeros_like(batch.attack_ids),
        )
        token = (
            self.option_type_embedding(option_types)
            + self.context_embedding(option_contexts)
            + self.attack_embedding(attack_indices)
            + card_encoder(batch.card_ids)
            + self.option_scalar_projection(
                batch.option_scalars.to(
                    dtype=self.option_scalar_projection.weight.dtype
                )
            )
        )
        for slot in range(MAX_ENTITY_SLOTS):
            token = (
                token
                + self.entity_role_embedding.weight[slot]
                + (
                    card_encoder(batch.entity_card_ids[:, :, slot])
                    + self.area_embedding(batch.entity_areas[:, :, slot])
                    + self.owner_embedding(batch.entity_owner_roles[:, :, slot])
                    + self.kind_embedding(batch.entity_token_kinds[:, :, slot])
                    + self.entity_scalar_projection(
                        batch.entity_scalars[:, :, slot].to(
                            dtype=self.entity_scalar_projection.weight.dtype
                        )
                    )
                )
            )
        token = preserve_cuda_bfloat16_activation(self.input_norm(token))
        history = token.new_zeros((batch.batch_size, self.d_model))
        for index in range(int(token.shape[1])):
            proposed = self.cell(token[:, index], history)
            history = torch.where(
                batch.valid_mask[:, index].unsqueeze(-1),
                proposed,
                history,
            )
        prompt = torch.stack(
            (
                batch.lengths.to(dtype=history.dtype).clamp(max=32.0) / 32.0,
                batch.ordered.to(dtype=history.dtype),
                batch.stop_sampled.to(dtype=history.dtype),
                batch.fallback.to(dtype=history.dtype),
                torch.ones_like(batch.lengths, dtype=history.dtype),
            ),
            dim=-1,
        )
        history = history + self.context_embedding(
            _safe_indices(
                batch.prompt_contexts,
                known=KNOWN_CONTEXT_COUNT,
                oov=CONTEXT_OOV_INDEX,
            )
        )
        return preserve_cuda_bfloat16_activation(
            self.output_norm(history + self.prompt_projection(prompt))
        )


def _validate_accepted_action(
    *,
    options: EncodedOptionArrayFeatures,
    action: tuple[int, ...],
    min_count: int,
    max_count: int,
) -> None:
    """Reject a sampled index sequence outside its served legal set."""
    if (
        len(action) < min_count
        or len(action) > max_count
        or len(set(action)) != len(action)
        or any(index < 0 or index >= len(options) for index in action)
    ):
        raise ValueError("accepted action violates its legal option set")


def _stable_option_payload(
    *,
    state: StateTokenArrayFeatures,
    options: EncodedOptionArrayFeatures,
    option_index: int,
) -> _StableOptionPayload:
    option_type = int(options.option_types[option_index])
    context = int(options.contexts[option_index])
    card_id = int(options.card_ids[option_index])
    attack_id = int(options.attack_ids[option_index])
    option_scalars = tuple(float(value) for value in options.scalars[option_index])
    entity_card_ids: list[int] = []
    entity_areas: list[int] = []
    entity_owners: list[int] = []
    entity_kinds: list[int] = []
    entity_scalars: list[float] = []
    for slot in range(MAX_ENTITY_SLOTS):
        present = bool(options.entity_slot_mask[option_index, slot])
        token_index = int(options.entity_slots[option_index, slot]) if present else 0
        if present and not 0 <= token_index < len(state.card_ids):
            raise ValueError("accepted-action entity reference is out of range")
        entity_card_ids.append(int(state.card_ids[token_index]) if present else 0)
        entity_areas.append(int(state.areas[token_index]) if present else 0)
        entity_owners.append(int(state.owner_roles[token_index]) if present else 0)
        entity_kinds.append(int(state.token_kinds[token_index]) if present else 0)
        scalars = (
            tuple(float(value) for value in state.scalars[token_index])
            if present
            else (0.0,) * TOKEN_SCALAR_SIZE
        )
        entity_scalars.extend(scalars)
    semantic = _semantic_bytes(
        option_type,
        context,
        card_id,
        attack_id,
        option_scalars,
        tuple(entity_card_ids),
        tuple(entity_areas),
        tuple(entity_owners),
        tuple(entity_kinds),
        tuple(entity_scalars),
    )
    return (
        semantic,
        option_type,
        context,
        card_id,
        attack_id,
        option_scalars,
        tuple(entity_card_ids),
        tuple(entity_areas),
        tuple(entity_owners),
        tuple(entity_kinds),
        tuple(entity_scalars),
    )


def _semantic_bytes(
    option_type: int,
    context: int,
    card_id: int,
    attack_id: int,
    option_scalars: tuple[float, ...],
    entity_card_ids: tuple[int, ...],
    entity_areas: tuple[int, ...],
    entity_owners: tuple[int, ...],
    entity_kinds: tuple[int, ...],
    entity_scalars: tuple[float, ...],
) -> bytes:
    integers = (
        option_type,
        context,
        card_id,
        attack_id,
        *entity_card_ids,
        *entity_areas,
        *entity_owners,
        *entity_kinds,
    )
    numeric = (*option_scalars, *entity_scalars)
    return struct.pack(f"<{len(integers)}q{len(numeric)}f", *integers, *numeric)


def _complete_action_identity(
    *,
    prompt_context: int,
    encoded: tuple[_StableOptionPayload, ...],
    stop_sampled: bool,
    fallback: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(_ACTION_IDENTITY_DOMAIN)
    digest.update(
        struct.pack(
            "<qI??",
            prompt_context,
            len(encoded),
            stop_sampled,
            fallback,
        )
    )
    for item in encoded:
        payload = item[0]
        if not isinstance(payload, bytes):
            raise TypeError("stable action payload must be bytes")
        digest.update(struct.pack("<I", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _safe_indices(values: Tensor, *, known: int, oov: int) -> Tensor:
    return torch.where(
        (values >= 0) & (values < known),
        values,
        torch.full_like(values, oov),
    )


__all__ = [
    "ACCEPTED_ACTION_SCHEMA_VERSION",
    "AcceptedActionBatch",
    "AcceptedActionEncoder",
    "AcceptedActionRecord",
    "build_accepted_action_record",
    "build_accepted_action_records",
    "collate_accepted_actions",
]
