"""Direct tensor encoding for native fixed-column public event logs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ptcg_rl.context.public_event_arrays import (
    PublicEventArrayBlock,
    PublicEventBatch,
    pin_public_event_batch,
    public_event_batch_from_array_block,
)
from ptcg_rl.context.public_events import (
    PUBLIC_EVENT_ACTOR_ROLE_COUNT,
    PUBLIC_EVENT_CATEGORICAL_SIZE,
    PUBLIC_EVENT_ENTITY_COUNT,
    PUBLIC_EVENT_ENTITY_FIELDS,
    PUBLIC_EVENT_WINDOW,
)
from ptcg_rl.engine.native_public_events import (
    _FIELD_PARAMETERS,
    _HIDDEN_IDENTITY_LOG_TYPES,
    _KNOWN_AREAS,
    _PUBLIC_EVENT_AREA_OOV,
    _selected_rows,
    _validate_native_public_logs,
)
from ptcg_rl.engine.native_public_history import LOG_PARAM_WIDTH

if TYPE_CHECKING:
    from ptcg_rl.engine.native_training import NativeTrainingBatchView


def native_public_event_batch(
    batch: NativeTrainingBatchView,
    *,
    rows: npt.ArrayLike | None = None,
    pin_memory: bool = False,
) -> PublicEventBatch:
    """Encode native fixed-column logs directly into model input tensors."""
    _validate_native_public_logs(batch)
    selected = (
        np.arange(batch.batch_size, dtype=np.int64)
        if rows is None
        else _selected_rows(rows, size=batch.batch_size)
    )
    block = _native_public_event_array_block(batch, selected)
    result = public_event_batch_from_array_block(
        block,
        tuple(range(block.decision_count)),
        device="cpu",
    )
    return pin_public_event_batch(result) if pin_memory else result


def _native_public_event_array_block(
    batch: NativeTrainingBatchView,
    selected: npt.NDArray[np.int64],
) -> PublicEventArrayBlock:
    starts = batch.log_offsets[selected].astype(np.int64, copy=False)
    stops = batch.log_offsets[selected + 1].astype(np.int64, copy=False)
    widths = np.minimum(stops - starts, PUBLIC_EVENT_WINDOW)
    retained_starts = stops - widths
    retained_indices = _ragged_source_indices(retained_starts, widths)
    player_indices = np.repeat(
        batch.select_player[selected].astype(np.int32, copy=False),
        widths,
    )
    projected = _project_native_public_events(
        batch,
        retained_indices,
        player_indices=player_indices,
    )
    event_offsets = np.empty(selected.size + 1, dtype=np.int32)
    event_offsets[0] = 0
    np.cumsum(widths, dtype=np.int64, out=event_offsets[1:])

    overflow_offsets = [0]
    overflow_event_types: list[int] = []
    overflow_actor_roles: list[int] = []
    overflow_counts: list[int] = []
    for source_row, start, retained_start in zip(
        selected,
        starts,
        retained_starts,
        strict=True,
    ):
        dropped_indices = np.arange(start, retained_start, dtype=np.int64)
        if dropped_indices.size:
            raw_types = batch.log_type[dropped_indices].astype(
                np.int64,
                copy=False,
            )
            actor_roles = _native_actor_roles(
                batch,
                raw_types,
                dropped_indices,
                player_indices=np.full(
                    dropped_indices.size,
                    int(batch.select_player[source_row]),
                    dtype=np.int32,
                ),
            )
            packed = (raw_types + 1) * PUBLIC_EVENT_ACTOR_ROLE_COUNT + actor_roles
            keys, counts = np.unique(packed, return_counts=True)
            overflow_event_types.extend(
                int(key // PUBLIC_EVENT_ACTOR_ROLE_COUNT) for key in keys
            )
            overflow_actor_roles.extend(
                int(key % PUBLIC_EVENT_ACTOR_ROLE_COUNT) for key in keys
            )
            overflow_counts.extend(int(count) for count in counts)
        overflow_offsets.append(len(overflow_counts))

    return PublicEventArrayBlock(
        event_offsets=event_offsets,
        event_types=projected["event_types"],
        actor_roles=projected["actor_roles"],
        from_areas=projected["from_areas"],
        to_areas=projected["to_areas"],
        card_ids=projected["card_ids"],
        serials=projected["serials"],
        entity_mask=projected["entity_mask"],
        attack_ids=projected["attack_ids"],
        attack_id_mask=projected["attack_id_mask"],
        values=projected["values"],
        value_mask=projected["value_mask"],
        categorical_values=projected["categorical_values"],
        overflow_offsets=np.asarray(overflow_offsets, dtype=np.int32),
        overflow_event_types=np.asarray(
            overflow_event_types,
            dtype=np.uint8,
        ),
        overflow_actor_roles=np.asarray(
            overflow_actor_roles,
            dtype=np.uint8,
        ),
        overflow_counts=np.asarray(overflow_counts, dtype=np.int32),
    )


def _project_native_public_events(
    batch: NativeTrainingBatchView,
    indices: npt.NDArray[np.int64],
    *,
    player_indices: npt.NDArray[np.int32],
) -> dict[str, np.ndarray]:
    raw_types = batch.log_type[indices].astype(np.int64, copy=False)
    event_count = int(indices.size)
    hidden_identity = np.isin(
        raw_types,
        tuple(_HIDDEN_IDENTITY_LOG_TYPES),
    )
    card_ids = np.zeros(
        (event_count, PUBLIC_EVENT_ENTITY_COUNT),
        dtype=np.uint16,
    )
    serials = np.zeros(
        (event_count, PUBLIC_EVENT_ENTITY_COUNT),
        dtype=np.int32,
    )
    entity_mask = np.zeros(
        (event_count, PUBLIC_EVENT_ENTITY_COUNT),
        dtype=np.bool_,
    )
    for column, (card_field, serial_field) in enumerate(PUBLIC_EVENT_ENTITY_FIELDS):
        raw_card_ids, card_present = _native_field_values(
            batch,
            raw_types,
            indices,
            card_field,
        )
        raw_serials, serial_present = _native_field_values(
            batch,
            raw_types,
            indices,
            serial_field,
        )
        visible = ~hidden_identity
        card_present &= visible
        serial_present &= visible
        if np.any(
            card_present
            & ((raw_card_ids < 0) | (raw_card_ids > np.iinfo(np.uint16).max))
        ):
            raise ValueError("public event card ID exceeds uint16 storage")
        card_ids[card_present, column] = raw_card_ids[card_present].astype(
            np.uint16,
            copy=False,
        )
        serials[serial_present, column] = raw_serials[serial_present]
        entity_mask[:, column] = card_present | serial_present

    attack_ids, attack_present = _native_field_values(
        batch,
        raw_types,
        indices,
        "attackId",
    )
    attack_present &= ~hidden_identity
    attack_ids[~attack_present] = 0
    values, value_present = _native_field_values(
        batch,
        raw_types,
        indices,
        "value",
    )
    categorical_values = np.zeros(
        (event_count, PUBLIC_EVENT_CATEGORICAL_SIZE),
        dtype=np.uint8,
    )
    for column, field in enumerate(
        ("putDamageCounter", "isRecover", "head", "hasBasicPokemon")
    ):
        raw_values, present = _native_field_values(
            batch,
            raw_types,
            indices,
            field,
        )
        categorical_values[present, column] = np.where(
            raw_values[present] != 0,
            2,
            1,
        )
    for column, field, minimum, maximum in (
        (4, "result", 0, 2),
        (5, "reason", 1, 4),
    ):
        raw_values, present = _native_field_values(
            batch,
            raw_types,
            indices,
            field,
        )
        in_range = present & (raw_values >= minimum) & (raw_values <= maximum)
        categorical_values[in_range, column] = raw_values[in_range] - minimum + 1
        categorical_values[present & ~in_range, column] = maximum - minimum + 2

    return {
        "event_types": (raw_types + 1).astype(np.uint8, copy=False),
        "actor_roles": _native_actor_roles(
            batch,
            raw_types,
            indices,
            player_indices=player_indices,
        ).astype(np.uint8, copy=False),
        "from_areas": _native_areas(
            *_native_field_values(
                batch,
                raw_types,
                indices,
                "fromArea",
            )
        ),
        "to_areas": _native_areas(
            *_native_field_values(
                batch,
                raw_types,
                indices,
                "toArea",
            )
        ),
        "card_ids": card_ids,
        "serials": serials,
        "entity_mask": entity_mask,
        "attack_ids": attack_ids,
        "attack_id_mask": attack_present,
        "values": values.astype(np.float32, copy=False),
        "value_mask": value_present,
        "categorical_values": categorical_values,
    }


def _native_field_values(
    batch: NativeTrainingBatchView,
    raw_types: npt.NDArray[np.int64],
    indices: npt.NDArray[np.int64],
    field: str,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.bool_]]:
    parameters = np.asarray(_FIELD_PARAMETERS[field], dtype=np.int8)[raw_types]
    present = parameters >= 0
    values = np.zeros(indices.size, dtype=np.int32)
    for parameter in range(LOG_PARAM_WIDTH):
        selected = parameters == parameter
        values[selected] = batch.log_params[parameter][indices[selected]]
    return values, present


def _native_actor_roles(
    batch: NativeTrainingBatchView,
    raw_types: npt.NDArray[np.int64],
    indices: npt.NDArray[np.int64],
    *,
    player_indices: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    actors, present = _native_field_values(
        batch,
        raw_types,
        indices,
        "playerIndex",
    )
    valid = present & np.isin(actors, (0, 1)) & np.isin(player_indices, (0, 1))
    roles = np.zeros(indices.size, dtype=np.int32)
    roles[valid] = np.where(actors[valid] == player_indices[valid], 1, 2)
    return roles


def _native_areas(
    values: npt.NDArray[np.int32],
    present: npt.NDArray[np.bool_],
) -> npt.NDArray[np.uint8]:
    result = np.zeros(values.size, dtype=np.uint8)
    known = present & np.isin(values, tuple(_KNOWN_AREAS))
    result[known] = values[known].astype(np.uint8, copy=False)
    result[present & ~known] = _PUBLIC_EVENT_AREA_OOV
    return result


def _ragged_source_indices(
    starts: npt.NDArray[np.int64],
    widths: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    row_offsets = np.repeat(starts, widths)
    columns = np.arange(row_offsets.size, dtype=np.int64)
    if columns.size:
        columns -= np.repeat(
            np.cumsum(widths, dtype=np.int64) - widths,
            widths,
        )
    return row_offsets + columns


__all__ = ["native_public_event_batch"]
