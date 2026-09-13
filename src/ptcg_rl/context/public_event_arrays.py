"""Compact array storage and tensor collation for public event deltas."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ptcg_rl.context.public_events import (
    PUBLIC_EVENT_ACTOR_ROLE_COUNT,
    PUBLIC_EVENT_AREA_COUNT,
    PUBLIC_EVENT_CATEGORICAL_COUNTS,
    PUBLIC_EVENT_CATEGORICAL_SIZE,
    PUBLIC_EVENT_ENTITY_COUNT,
    PUBLIC_EVENT_TYPE_COUNT,
    PUBLIC_EVENT_WINDOW,
    PublicEvent,
    PublicEventActorRole,
    PublicEventDelta,
    PublicEventOverflowCount,
)

_INT32_MAX = int(np.iinfo(np.int32).max)


@dataclass(frozen=True)
class PublicEventBatch:
    """Padded tensor batch of per-decision public event deltas."""

    event_types: Tensor
    actor_roles: Tensor
    from_areas: Tensor
    to_areas: Tensor
    card_ids: Tensor
    serials: Tensor
    entity_mask: Tensor
    attack_ids: Tensor
    attack_id_mask: Tensor
    values: Tensor
    value_mask: Tensor
    categorical_values: Tensor
    padding_mask: Tensor
    overflow_type_actor_counts: Tensor

    @property
    def batch_size(self) -> int:
        """Return the number of decision rows in the batch."""
        return int(self.event_types.shape[0])


@dataclass(frozen=True)
class PublicEventArrayBlock:
    """CSR-style compact event deltas aligned with trajectory decisions."""

    event_offsets: np.ndarray
    event_types: np.ndarray
    actor_roles: np.ndarray
    from_areas: np.ndarray
    to_areas: np.ndarray
    card_ids: np.ndarray
    serials: np.ndarray
    entity_mask: np.ndarray
    attack_ids: np.ndarray
    attack_id_mask: np.ndarray
    values: np.ndarray
    value_mask: np.ndarray
    categorical_values: np.ndarray
    overflow_offsets: np.ndarray
    overflow_event_types: np.ndarray
    overflow_actor_roles: np.ndarray
    overflow_counts: np.ndarray

    @property
    def decision_count(self) -> int:
        """Return the number of aligned trajectory decisions."""
        return max(0, int(self.event_offsets.shape[0]) - 1)

    @property
    def event_count(self) -> int:
        """Return the number of flattened retained event rows."""
        return int(self.event_types.shape[0])

    @property
    def overflow_entry_count(self) -> int:
        """Return the number of flattened sparse overflow entries."""
        return int(self.overflow_event_types.shape[0])

    def delta_at(self, index: int) -> PublicEventDelta:
        """Materialize one decision delta for diagnostics and object paths."""
        if not 0 <= index < self.decision_count:
            raise IndexError("public event decision index is out of range")
        start = int(self.event_offsets[index])
        stop = int(self.event_offsets[index + 1])
        overflow_start = int(self.overflow_offsets[index])
        overflow_stop = int(self.overflow_offsets[index + 1])
        return PublicEventDelta(
            events=tuple(self._event_at(row) for row in range(start, stop)),
            overflow=tuple(
                PublicEventOverflowCount(
                    event_type=int(self.overflow_event_types[row]),
                    actor_role=PublicEventActorRole(
                        int(self.overflow_actor_roles[row])
                    ),
                    count=int(self.overflow_counts[row]),
                )
                for row in range(overflow_start, overflow_stop)
            ),
        )

    def _event_at(self, index: int) -> PublicEvent:
        return PublicEvent(
            event_type=int(self.event_types[index]),
            actor_role=PublicEventActorRole(int(self.actor_roles[index])),
            from_area=int(self.from_areas[index]),
            to_area=int(self.to_areas[index]),
            card_ids=tuple(int(value) for value in self.card_ids[index]),
            serials=tuple(int(value) for value in self.serials[index]),
            entity_mask=tuple(bool(value) for value in self.entity_mask[index]),
            attack_id=int(self.attack_ids[index]),
            attack_id_present=bool(self.attack_id_mask[index]),
            value=float(self.values[index]),
            value_present=bool(self.value_mask[index]),
            categorical_values=tuple(
                int(value) for value in self.categorical_values[index]
            ),
        )


def build_public_event_array_block(
    deltas: Sequence[PublicEventDelta],
) -> PublicEventArrayBlock:
    """Flatten bounded decision deltas into compact trajectory arrays."""
    event_offsets = [0]
    overflow_offsets = [0]
    events: list[PublicEvent] = []
    overflow: list[PublicEventOverflowCount] = []
    for delta in deltas:
        events.extend(delta.events)
        overflow.extend(delta.overflow)
        event_offsets.append(len(events))
        overflow_offsets.append(len(overflow))
    if len(events) > _INT32_MAX or len(overflow) > _INT32_MAX:
        raise ValueError("public event array exceeds int32 offset capacity")
    block = PublicEventArrayBlock(
        event_offsets=np.asarray(event_offsets, dtype=np.int32),
        event_types=np.asarray([event.event_type for event in events], dtype=np.uint8),
        actor_roles=np.asarray(
            [int(event.actor_role) for event in events], dtype=np.uint8
        ),
        from_areas=np.asarray([event.from_area for event in events], dtype=np.uint8),
        to_areas=np.asarray([event.to_area for event in events], dtype=np.uint8),
        card_ids=_event_matrix(
            (event.card_ids for event in events),
            dtype=np.uint16,
            width=PUBLIC_EVENT_ENTITY_COUNT,
        ),
        serials=_event_matrix(
            (event.serials for event in events),
            dtype=np.int32,
            width=PUBLIC_EVENT_ENTITY_COUNT,
        ),
        entity_mask=_event_matrix(
            (event.entity_mask for event in events),
            dtype=np.bool_,
            width=PUBLIC_EVENT_ENTITY_COUNT,
        ),
        attack_ids=np.asarray([event.attack_id for event in events], dtype=np.int32),
        attack_id_mask=np.asarray(
            [event.attack_id_present for event in events], dtype=np.bool_
        ),
        values=np.asarray([event.value for event in events], dtype=np.float32),
        value_mask=np.asarray(
            [event.value_present for event in events], dtype=np.bool_
        ),
        categorical_values=_event_matrix(
            (event.categorical_values for event in events),
            dtype=np.uint8,
            width=PUBLIC_EVENT_CATEGORICAL_SIZE,
        ),
        overflow_offsets=np.asarray(overflow_offsets, dtype=np.int32),
        overflow_event_types=np.asarray(
            [item.event_type for item in overflow], dtype=np.uint8
        ),
        overflow_actor_roles=np.asarray(
            [int(item.actor_role) for item in overflow], dtype=np.uint8
        ),
        overflow_counts=np.asarray([item.count for item in overflow], dtype=np.int32),
    )
    validate_public_event_array_block(block)
    return block


def collate_public_event_deltas(
    deltas: Sequence[PublicEventDelta],
    *,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
) -> PublicEventBatch:
    """Pad object deltas into a tensor batch without losing event order."""
    if all(
        isinstance(delta, PublicEventDelta) and not delta.events and not delta.overflow
        for delta in deltas
    ):
        return _empty_public_event_batch(
            len(deltas),
            device=device,
            pin_memory=pin_memory,
        )
    result = public_event_batch_from_array_block(
        build_public_event_array_block(deltas),
        tuple(range(len(deltas))),
        device=device,
    )
    if not pin_memory:
        return result
    return pin_public_event_batch(result)


def _empty_public_event_batch(
    batch_size: int,
    *,
    device: torch.device | str | None,
    pin_memory: bool,
) -> PublicEventBatch:
    """Allocate the common no-event payload without a per-row Python pass."""
    target = torch.device("cpu" if device is None else device)
    if pin_memory and target.type != "cpu":
        raise ValueError("only CPU public-event batches can use pinned memory")

    def zeros(shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        return torch.zeros(
            shape,
            dtype=dtype,
            device=target,
            pin_memory=pin_memory,
        )

    vector_shape = (batch_size, 1)
    entity_shape = (*vector_shape, PUBLIC_EVENT_ENTITY_COUNT)
    return PublicEventBatch(
        event_types=zeros(vector_shape, torch.long),
        actor_roles=zeros(vector_shape, torch.long),
        from_areas=zeros(vector_shape, torch.long),
        to_areas=zeros(vector_shape, torch.long),
        card_ids=zeros(entity_shape, torch.long),
        serials=zeros(entity_shape, torch.long),
        entity_mask=zeros(entity_shape, torch.bool),
        attack_ids=zeros(vector_shape, torch.long),
        attack_id_mask=zeros(vector_shape, torch.bool),
        values=zeros(vector_shape, torch.float32),
        value_mask=zeros(vector_shape, torch.bool),
        categorical_values=zeros(
            (*vector_shape, PUBLIC_EVENT_CATEGORICAL_SIZE),
            torch.long,
        ),
        padding_mask=torch.ones(
            vector_shape,
            dtype=torch.bool,
            device=target,
            pin_memory=pin_memory,
        ),
        overflow_type_actor_counts=zeros(
            (
                batch_size,
                PUBLIC_EVENT_TYPE_COUNT,
                PUBLIC_EVENT_ACTOR_ROLE_COUNT,
            ),
            torch.long,
        ),
    )


def move_public_event_batch(
    batch: PublicEventBatch,
    *,
    device: torch.device | str,
    non_blocking: bool = False,
) -> PublicEventBatch:
    """Move one pre-collated event payload without rebuilding object deltas."""
    target = torch.device(device)
    return _map_public_event_batch_tensors(
        batch,
        lambda value: value.to(
            device=target,
            non_blocking=non_blocking,
        ),
    )


def pin_public_event_batch(batch: PublicEventBatch) -> PublicEventBatch:
    """Copy one CPU public-event batch into page-locked host storage."""
    if batch.event_types.device.type != "cpu":
        raise ValueError("only CPU public-event batches can use pinned memory")
    return _map_public_event_batch_tensors(
        batch,
        lambda value: value.pin_memory(),
    )


def _map_public_event_batch_tensors(
    batch: PublicEventBatch,
    transform: Callable[[Tensor], Tensor],
) -> PublicEventBatch:
    """Apply one storage transform to every aligned public-event tensor."""
    validate_public_event_batch(batch)
    result = PublicEventBatch(
        event_types=transform(batch.event_types),
        actor_roles=transform(batch.actor_roles),
        from_areas=transform(batch.from_areas),
        to_areas=transform(batch.to_areas),
        card_ids=transform(batch.card_ids),
        serials=transform(batch.serials),
        entity_mask=transform(batch.entity_mask),
        attack_ids=transform(batch.attack_ids),
        attack_id_mask=transform(batch.attack_id_mask),
        values=transform(batch.values),
        value_mask=transform(batch.value_mask),
        categorical_values=transform(batch.categorical_values),
        padding_mask=transform(batch.padding_mask),
        overflow_type_actor_counts=transform(batch.overflow_type_actor_counts),
    )
    validate_public_event_batch(result)
    return result


def select_public_event_batch_rows(
    batch: PublicEventBatch,
    indices: Tensor,
) -> PublicEventBatch:
    """Select or repeat decision rows without changing their event width."""
    validate_public_event_batch(batch)
    _validate_batch_row_indices(
        indices,
        batch_size=batch.batch_size,
        device=batch.event_types.device,
    )

    def select(values: Tensor) -> Tensor:
        return values.index_select(0, indices)

    return PublicEventBatch(
        event_types=select(batch.event_types),
        actor_roles=select(batch.actor_roles),
        from_areas=select(batch.from_areas),
        to_areas=select(batch.to_areas),
        card_ids=select(batch.card_ids),
        serials=select(batch.serials),
        entity_mask=select(batch.entity_mask),
        attack_ids=select(batch.attack_ids),
        attack_id_mask=select(batch.attack_id_mask),
        values=select(batch.values),
        value_mask=select(batch.value_mask),
        categorical_values=select(batch.categorical_values),
        padding_mask=select(batch.padding_mask),
        overflow_type_actor_counts=select(batch.overflow_type_actor_counts),
    )


def concatenate_public_event_batches(
    batches: Sequence[PublicEventBatch],
) -> PublicEventBatch:
    """Right-pad event widths and concatenate batches along their row axis."""
    if not batches:
        raise ValueError("cannot concatenate an empty public event batch sequence")
    for batch in batches:
        validate_public_event_batch(batch)
    device = batches[0].event_types.device
    if any(batch.event_types.device != device for batch in batches[1:]):
        raise ValueError("public event batches must use the same device")
    event_width = max(int(batch.event_types.shape[1]) for batch in batches)

    def concatenate(name: str, *, padding_value: int | float | bool) -> Tensor:
        values = tuple(getattr(batch, name) for batch in batches)
        return torch.cat(
            tuple(
                _right_pad_public_event_tensor(
                    value,
                    event_width=event_width,
                    padding_value=padding_value,
                )
                for value in values
            ),
            dim=0,
        )

    result = PublicEventBatch(
        event_types=concatenate("event_types", padding_value=0),
        actor_roles=concatenate("actor_roles", padding_value=0),
        from_areas=concatenate("from_areas", padding_value=0),
        to_areas=concatenate("to_areas", padding_value=0),
        card_ids=concatenate("card_ids", padding_value=0),
        serials=concatenate("serials", padding_value=0),
        entity_mask=concatenate("entity_mask", padding_value=False),
        attack_ids=concatenate("attack_ids", padding_value=0),
        attack_id_mask=concatenate("attack_id_mask", padding_value=False),
        values=concatenate("values", padding_value=0.0),
        value_mask=concatenate("value_mask", padding_value=False),
        categorical_values=concatenate("categorical_values", padding_value=0),
        padding_mask=concatenate("padding_mask", padding_value=True),
        overflow_type_actor_counts=torch.cat(
            tuple(batch.overflow_type_actor_counts for batch in batches),
            dim=0,
        ),
    )
    validate_public_event_batch(result)
    return result


def public_event_batch_from_array_block(
    block: PublicEventArrayBlock,
    indices: Sequence[int],
    *,
    device: torch.device | str | None = None,
) -> PublicEventBatch:
    """Gather CSR rows into a padded event tensor batch."""
    validate_public_event_array_block(block)
    normalized = tuple(int(index) for index in indices)
    if any(index < 0 or index >= block.decision_count for index in normalized):
        raise IndexError("public event batch index is out of range")
    source_rows = np.asarray(normalized, dtype=np.int64)
    event_starts = block.event_offsets[source_rows].astype(
        np.int64,
        copy=False,
    )
    event_widths = (
        block.event_offsets[source_rows + 1].astype(np.int64, copy=False) - event_starts
    )
    event_width = max(
        1,
        int(event_widths.max()) if event_widths.size else 0,
    )
    batch_size = len(normalized)
    arrays = _empty_padded_arrays(batch_size, event_width)
    overflow_counts = np.zeros(
        (batch_size, PUBLIC_EVENT_TYPE_COUNT, PUBLIC_EVENT_ACTOR_ROLE_COUNT),
        dtype=np.int32,
    )
    target_rows, target_columns, event_indices = _ragged_gather_indices(
        event_starts,
        event_widths,
    )
    for name in (
        "event_types",
        "actor_roles",
        "from_areas",
        "to_areas",
        "card_ids",
        "serials",
        "entity_mask",
        "attack_ids",
        "attack_id_mask",
        "values",
        "value_mask",
        "categorical_values",
    ):
        arrays[name][target_rows, target_columns] = getattr(block, name)[event_indices]
    arrays["padding_mask"][target_rows, target_columns] = False

    overflow_starts = block.overflow_offsets[source_rows].astype(
        np.int64,
        copy=False,
    )
    overflow_widths = (
        block.overflow_offsets[source_rows + 1].astype(
            np.int64,
            copy=False,
        )
        - overflow_starts
    )
    overflow_rows, _, overflow_indices = _ragged_gather_indices(
        overflow_starts,
        overflow_widths,
    )
    overflow_counts[
        overflow_rows,
        block.overflow_event_types[overflow_indices],
        block.overflow_actor_roles[overflow_indices],
    ] = block.overflow_counts[overflow_indices]

    def tensor(name: str, *, dtype: torch.dtype) -> Tensor:
        return torch.as_tensor(arrays[name], dtype=dtype, device=device)

    return PublicEventBatch(
        event_types=tensor("event_types", dtype=torch.long),
        actor_roles=tensor("actor_roles", dtype=torch.long),
        from_areas=tensor("from_areas", dtype=torch.long),
        to_areas=tensor("to_areas", dtype=torch.long),
        card_ids=tensor("card_ids", dtype=torch.long),
        serials=tensor("serials", dtype=torch.long),
        entity_mask=tensor("entity_mask", dtype=torch.bool),
        attack_ids=tensor("attack_ids", dtype=torch.long),
        attack_id_mask=tensor("attack_id_mask", dtype=torch.bool),
        values=tensor("values", dtype=torch.float32),
        value_mask=tensor("value_mask", dtype=torch.bool),
        categorical_values=tensor("categorical_values", dtype=torch.long),
        padding_mask=tensor("padding_mask", dtype=torch.bool),
        overflow_type_actor_counts=torch.as_tensor(
            overflow_counts,
            dtype=torch.long,
            device=device,
        ),
    )


def _ragged_gather_indices(
    starts: np.ndarray,
    widths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand CSR row starts into vectorized padded-target/source indices."""
    target_rows = np.repeat(
        np.arange(widths.shape[0], dtype=np.int64),
        widths,
    )
    target_columns = np.arange(target_rows.size, dtype=np.int64)
    if target_columns.size:
        target_columns -= np.repeat(
            np.cumsum(widths, dtype=np.int64) - widths,
            widths,
        )
    source_indices = np.repeat(starts, widths) + target_columns
    return target_rows, target_columns, source_indices


def validate_public_event_batch(batch: PublicEventBatch) -> None:
    """Reject public-event tensors with incompatible shapes, devices, or dtypes."""
    if not isinstance(batch, PublicEventBatch):
        raise TypeError("public event batch must be a PublicEventBatch")
    if not isinstance(batch.event_types, Tensor):
        raise TypeError("PublicEventBatch.event_types must be a tensor")
    if batch.event_types.ndim != 2:
        raise ValueError("public event tensors must have shape [B, W]")
    batch_size, event_width = batch.event_types.shape
    if event_width <= 0 or event_width > PUBLIC_EVENT_WINDOW:
        raise ValueError(
            "public event width must be positive and within the bounded window"
        )
    vector_shape = (batch_size, event_width)
    entity_shape = (*vector_shape, PUBLIC_EVENT_ENTITY_COUNT)
    categorical_shape = (*vector_shape, PUBLIC_EVENT_CATEGORICAL_SIZE)
    overflow_shape = (
        batch_size,
        PUBLIC_EVENT_TYPE_COUNT,
        PUBLIC_EVENT_ACTOR_ROLE_COUNT,
    )
    field_specs: tuple[tuple[str, tuple[int, ...], torch.dtype], ...] = (
        ("event_types", vector_shape, torch.long),
        ("actor_roles", vector_shape, torch.long),
        ("from_areas", vector_shape, torch.long),
        ("to_areas", vector_shape, torch.long),
        ("card_ids", entity_shape, torch.long),
        ("serials", entity_shape, torch.long),
        ("entity_mask", entity_shape, torch.bool),
        ("attack_ids", vector_shape, torch.long),
        ("attack_id_mask", vector_shape, torch.bool),
        ("values", vector_shape, torch.float32),
        ("value_mask", vector_shape, torch.bool),
        ("categorical_values", categorical_shape, torch.long),
        ("padding_mask", vector_shape, torch.bool),
        ("overflow_type_actor_counts", overflow_shape, torch.long),
    )
    device = batch.event_types.device
    for name, expected_shape, expected_dtype in field_specs:
        values = getattr(batch, name)
        if not isinstance(values, Tensor):
            raise TypeError(f"PublicEventBatch.{name} must be a tensor")
        if values.shape != expected_shape:
            raise ValueError(
                f"PublicEventBatch.{name} must have shape {expected_shape}, got "
                f"{tuple(values.shape)}"
            )
        if values.dtype != expected_dtype:
            raise TypeError(
                f"PublicEventBatch.{name} must use {expected_dtype}, got {values.dtype}"
            )
        if values.device != device:
            raise ValueError("public event tensors must use the same device")


def validate_public_event_array_block(block: PublicEventArrayBlock) -> None:
    """Reject corrupt event offsets, shapes, dtypes, and categorical values."""
    _validate_offsets(block.event_offsets, block.event_count, "public event")
    _validate_offsets(
        block.overflow_offsets,
        block.overflow_entry_count,
        "public event overflow",
    )
    if block.overflow_offsets.shape != block.event_offsets.shape:
        raise ValueError("public event and overflow decision counts differ")
    event_count = block.event_count
    for name, dtype in (
        ("event_types", np.uint8),
        ("actor_roles", np.uint8),
        ("from_areas", np.uint8),
        ("to_areas", np.uint8),
        ("attack_ids", np.int32),
        ("attack_id_mask", np.bool_),
        ("values", np.float32),
        ("value_mask", np.bool_),
    ):
        values = getattr(block, name)
        if values.dtype != dtype or values.shape != (event_count,):
            raise TypeError(f"{name} has the wrong public event shape or dtype")
    matrix_fields: tuple[tuple[str, Any, int], ...] = (
        ("card_ids", np.uint16, PUBLIC_EVENT_ENTITY_COUNT),
        ("serials", np.int32, PUBLIC_EVENT_ENTITY_COUNT),
        ("entity_mask", np.bool_, PUBLIC_EVENT_ENTITY_COUNT),
        ("categorical_values", np.uint8, PUBLIC_EVENT_CATEGORICAL_SIZE),
    )
    for matrix_name, matrix_dtype, width in matrix_fields:
        values = getattr(block, matrix_name)
        if values.dtype != matrix_dtype or values.shape != (event_count, width):
            raise TypeError(f"{matrix_name} has the wrong public event shape or dtype")
    overflow_count = block.overflow_entry_count
    for name, dtype in (
        ("overflow_event_types", np.uint8),
        ("overflow_actor_roles", np.uint8),
        ("overflow_counts", np.int32),
    ):
        values = getattr(block, name)
        if values.dtype != dtype or values.shape != (overflow_count,):
            raise TypeError(f"{name} has the wrong public event shape or dtype")
    if bool(
        np.any(block.event_offsets[1:] - block.event_offsets[:-1] > PUBLIC_EVENT_WINDOW)
    ):
        raise ValueError("public event decision row exceeds the bounded window")
    event_widths = block.event_offsets[1:] - block.event_offsets[:-1]
    overflow_widths = block.overflow_offsets[1:] - block.overflow_offsets[:-1]
    if bool(np.any((overflow_widths > 0) & (event_widths != PUBLIC_EVENT_WINDOW))):
        raise ValueError("public event overflow requires a full retained window")
    _validate_domains(block.event_types, block.actor_roles)
    if event_count:
        if not bool((block.from_areas < PUBLIC_EVENT_AREA_COUNT).all()) or not bool(
            (block.to_areas < PUBLIC_EVENT_AREA_COUNT).all()
        ):
            raise ValueError("public event area is outside the embedding domain")
        if not bool(np.isfinite(block.values).all()):
            raise ValueError("public event values must be finite")
        hidden_entities = ~block.entity_mask
        if bool(np.any(block.card_ids[hidden_entities] != 0)) or bool(
            np.any(block.serials[hidden_entities] != 0)
        ):
            raise ValueError("masked public event identity must be canonical zero")
        if bool(np.any((~block.attack_id_mask) & (block.attack_ids != 0))):
            raise ValueError("masked public event attack ID must be canonical zero")
        if bool(
            np.any(
                (~block.value_mask) & ((block.values != 0.0) | np.signbit(block.values))
            )
        ):
            raise ValueError("masked public event value must be canonical zero")
        for column, count in enumerate(PUBLIC_EVENT_CATEGORICAL_COUNTS):
            if not bool((block.categorical_values[:, column] < count).all()):
                raise ValueError("public event categorical value is outside its domain")
    _validate_domains(block.overflow_event_types, block.overflow_actor_roles)
    if overflow_count and not bool((block.overflow_counts > 0).all()):
        raise ValueError("public event overflow counts must be positive")
    for decision_index in range(block.decision_count):
        rows = range(
            int(block.overflow_offsets[decision_index]),
            int(block.overflow_offsets[decision_index + 1]),
        )
        keys = tuple(
            (int(block.overflow_event_types[row]), int(block.overflow_actor_roles[row]))
            for row in rows
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("public event overflow entries are not canonical")


def public_event_array_field_names() -> tuple[str, ...]:
    """Return transport field order for one compact public event block."""
    return tuple(PublicEventArrayBlock.__dataclass_fields__)


def _event_matrix(
    rows: Any,
    *,
    dtype: Any,
    width: int,
) -> np.ndarray:
    return np.asarray(list(rows), dtype=dtype).reshape((-1, width))


def _empty_padded_arrays(batch_size: int, event_width: int) -> dict[str, np.ndarray]:
    shape = (batch_size, event_width)
    return {
        "event_types": np.zeros(shape, dtype=np.uint8),
        "actor_roles": np.zeros(shape, dtype=np.uint8),
        "from_areas": np.zeros(shape, dtype=np.uint8),
        "to_areas": np.zeros(shape, dtype=np.uint8),
        "card_ids": np.zeros((*shape, PUBLIC_EVENT_ENTITY_COUNT), dtype=np.uint16),
        "serials": np.zeros((*shape, PUBLIC_EVENT_ENTITY_COUNT), dtype=np.int32),
        "entity_mask": np.zeros((*shape, PUBLIC_EVENT_ENTITY_COUNT), dtype=np.bool_),
        "attack_ids": np.zeros(shape, dtype=np.int32),
        "attack_id_mask": np.zeros(shape, dtype=np.bool_),
        "values": np.zeros(shape, dtype=np.float32),
        "value_mask": np.zeros(shape, dtype=np.bool_),
        "categorical_values": np.zeros(
            (*shape, PUBLIC_EVENT_CATEGORICAL_SIZE), dtype=np.uint8
        ),
        "padding_mask": np.ones(shape, dtype=np.bool_),
    }


def _right_pad_public_event_tensor(
    values: Tensor,
    *,
    event_width: int,
    padding_value: int | float | bool,
) -> Tensor:
    current_width = int(values.shape[1])
    if current_width == event_width:
        return values
    padding_shape = (values.shape[0], event_width - current_width, *values.shape[2:])
    padding = torch.full(
        padding_shape,
        padding_value,
        dtype=values.dtype,
        device=values.device,
    )
    return torch.cat((values, padding), dim=1)


def _validate_batch_row_indices(
    indices: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    if not isinstance(indices, Tensor):
        raise TypeError("public event row indices must be a tensor")
    if indices.ndim != 1:
        raise ValueError("public event row indices must be one-dimensional")
    if indices.dtype != torch.long:
        raise TypeError("public event row indices must use torch.long")
    if indices.device != device:
        raise ValueError("public event row indices must share the batch device")
    if indices.numel() and bool(((indices < 0) | (indices >= batch_size)).any().item()):
        raise IndexError("public event row index is out of range")


def _validate_offsets(offsets: np.ndarray, row_count: int, name: str) -> None:
    if offsets.dtype != np.int32 or offsets.ndim != 1 or offsets.shape[0] == 0:
        raise TypeError(f"{name} offsets must be non-empty int32 [N + 1]")
    if int(offsets[0]) != 0 or bool(np.any(offsets[1:] < offsets[:-1])):
        raise ValueError(f"{name} offsets must start at zero and be monotonic")
    if int(offsets[-1]) != row_count:
        raise ValueError(f"{name} offsets do not match flattened rows")


def _validate_domains(event_types: np.ndarray, actor_roles: np.ndarray) -> None:
    if event_types.size and not bool(
        ((event_types >= 1) & (event_types < PUBLIC_EVENT_TYPE_COUNT)).all()
    ):
        raise ValueError("public event type is outside the embedding domain")
    if actor_roles.size and not bool(
        (actor_roles < PUBLIC_EVENT_ACTOR_ROLE_COUNT).all()
    ):
        raise ValueError("public event actor role is invalid")


__all__ = [
    "PublicEventArrayBlock",
    "PublicEventBatch",
    "build_public_event_array_block",
    "collate_public_event_deltas",
    "concatenate_public_event_batches",
    "move_public_event_batch",
    "pin_public_event_batch",
    "public_event_array_field_names",
    "public_event_batch_from_array_block",
    "select_public_event_batch_rows",
    "validate_public_event_array_block",
    "validate_public_event_batch",
]
