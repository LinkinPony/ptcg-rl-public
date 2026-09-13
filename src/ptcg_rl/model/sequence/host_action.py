"""Shape-grouped host materialization for accepted complete actions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
from torch import Tensor

from ptcg_rl.actions.encoding import (
    SCALAR_FEATURE_SIZE,
    EncodedOptionArrayFeatures,
)
from ptcg_rl.actions.selection import ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
from ptcg_rl.model.policy import CONTEXT_OOV_INDEX, MAX_ENTITY_SLOTS, OptionBatch
from ptcg_rl.model.sequence.action import (
    AcceptedActionRecord,
    _accepted_action_record_from_payloads,
    _StableOptionPayload,
    _validate_accepted_action,
    build_accepted_action_record,
)
from ptcg_rl.model.state_encoder import (
    TOKEN_SCALAR_SIZE,
    StateBatch,
    StateTokenArrayFeatures,
)

_ENTITY_NUMERIC_SIZE = MAX_ENTITY_SLOTS * TOKEN_SCALAR_SIZE
_MIN_BATCHED_OPTION_PAYLOADS = 16


def build_host_accepted_action_records(
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
    batch_size = len(states)
    row_inputs = (
        options,
        actions,
        min_counts,
        max_counts,
        stop_sampled,
    )
    if any(len(values) != batch_size for values in row_inputs):
        raise ValueError("accepted-action batch inputs are misaligned")
    if fallback is not None and len(fallback) != batch_size:
        raise ValueError("accepted-action fallback rows are misaligned")
    if not batch_size:
        return ()

    fallback_rows: Sequence[bool] = (
        (False,) * batch_size if fallback is None else fallback
    )
    groups: dict[tuple[int, int, int], list[int]] = {}
    prompt_contexts: list[int] = []
    ordered_rows: list[bool] = []
    for row, (state, row_options, action, minimum, maximum) in enumerate(
        zip(
            states,
            options,
            actions,
            min_counts,
            max_counts,
            strict=True,
        )
    ):
        _validate_accepted_action(
            options=row_options,
            action=action,
            min_count=minimum,
            max_count=maximum,
        )
        _validate_action_feature_shapes(state=state, options=row_options)
        prompt_context = (
            int(row_options.contexts[0]) if len(row_options) else CONTEXT_OOV_INDEX
        )
        prompt_contexts.append(prompt_context)
        ordered_rows.append(prompt_context not in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS)
        key = (len(state.card_ids), len(row_options), len(action))
        groups.setdefault(key, []).append(row)

    records: list[AcceptedActionRecord | None] = [None] * batch_size
    for (_state_size, _option_size, action_size), group_rows in groups.items():
        if not action_size:
            for row in group_rows:
                records[row] = _accepted_action_record_from_payloads(
                    prompt_context=prompt_contexts[row],
                    encoded=(),
                    ordered=ordered_rows[row],
                    stop_sampled=stop_sampled[row],
                    fallback=fallback_rows[row],
                )
            continue
        if len(group_rows) * action_size < _MIN_BATCHED_OPTION_PAYLOADS:
            for row in group_rows:
                records[row] = build_accepted_action_record(
                    state=states[row],
                    options=options[row],
                    action=actions[row],
                    min_count=min_counts[row],
                    max_count=max_counts[row],
                    stop_sampled=stop_sampled[row],
                    fallback=fallback_rows[row],
                )
            continue
        payload_rows = _stable_option_payload_batch(
            states=states,
            options=options,
            actions=actions,
            rows=group_rows,
        )
        for row, encoded in zip(group_rows, payload_rows, strict=True):
            if not ordered_rows[row]:
                encoded = tuple(sorted(encoded, key=lambda item: item[0]))
            records[row] = _accepted_action_record_from_payloads(
                prompt_context=prompt_contexts[row],
                encoded=encoded,
                ordered=ordered_rows[row],
                stop_sampled=stop_sampled[row],
                fallback=fallback_rows[row],
            )

    if any(record is None for record in records):
        raise RuntimeError("accepted-action batch did not materialize every row")
    return tuple(cast(AcceptedActionRecord, record) for record in records)


def build_tensor_host_accepted_action_records(
    *,
    states: StateBatch,
    options: OptionBatch,
    action_offsets: npt.NDArray[np.int64],
    action_choices: npt.NDArray[np.int32],
    min_counts: Sequence[int],
    max_counts: Sequence[int],
    stop_sampled: npt.NDArray[np.bool_],
) -> tuple[AcceptedActionRecord, ...]:
    """Materialize stable actions directly from a CPU tensor-native batch.

    Only sampled options and the state entities referenced by those options are
    converted to Python values.  This is the native rollout path: it avoids
    reconstructing one full state/option object graph for every served row.
    """
    batch_size = int(states.card_ids.shape[0])
    if batch_size <= 0:
        return ()
    _validate_tensor_host_inputs(
        states=states,
        options=options,
        action_offsets=action_offsets,
        action_choices=action_choices,
        min_counts=min_counts,
        max_counts=max_counts,
        stop_sampled=stop_sampled,
    )
    lengths = np.diff(action_offsets)
    width = max(1, int(lengths.max(initial=0)))
    valid = np.arange(width, dtype=np.int64)[None, :] < lengths[:, None]
    action_indices = np.zeros((batch_size, width), dtype=np.int64)
    if action_choices.size:
        rows = np.repeat(np.arange(batch_size, dtype=np.int64), lengths)
        positions = np.arange(action_choices.size, dtype=np.int64) - np.repeat(
            action_offsets[:-1],
            lengths,
        )
        action_indices[rows, positions] = action_choices

    option_lengths = _option_lengths(options)
    selected_lengths = option_lengths[:, None]
    if np.any(valid & (action_indices >= selected_lengths)):
        raise ValueError("accepted action references an unserved option")
    for row in range(batch_size):
        start = int(action_offsets[row])
        stop = int(action_offsets[row + 1])
        choices = action_choices[start:stop]
        if np.unique(choices).size != choices.size:
            raise ValueError("accepted action repeats an option")

    row_axis = np.arange(batch_size, dtype=np.int64)[:, None]
    option_types = _numpy(options.option_types)[row_axis, action_indices]
    option_contexts = _numpy(options.contexts)[row_axis, action_indices]
    card_ids = _numpy(options.card_ids)[row_axis, action_indices]
    attack_ids = _numpy(options.attack_ids)[row_axis, action_indices]
    option_scalars = _numpy(options.scalars)[row_axis, action_indices]
    entity_indices = _numpy(options.entity_slots)[row_axis, action_indices].astype(
        np.int64,
        copy=False,
    )
    entity_present = _numpy(options.entity_slot_mask)[
        row_axis,
        action_indices,
    ].astype(np.bool_, copy=False)

    state_lengths = _state_lengths(states)
    invalid_entities = (
        valid[..., None]
        & entity_present
        & ((entity_indices < 0) | (entity_indices >= state_lengths[:, None, None]))
    )
    if np.any(invalid_entities):
        raise ValueError("accepted-action entity reference is out of range")
    safe_entities = np.where(entity_present, entity_indices, 0)
    state_axis = np.arange(batch_size, dtype=np.int64)[:, None, None]
    entity_card_ids = _numpy(states.card_ids)[state_axis, safe_entities]
    entity_areas = _numpy(states.areas)[state_axis, safe_entities]
    entity_owners = _numpy(states.owner_roles)[state_axis, safe_entities]
    entity_kinds = _numpy(states.token_kinds)[state_axis, safe_entities]
    entity_scalars = _numpy(states.scalars)[state_axis, safe_entities]

    present = valid[..., None] & entity_present
    option_types = np.where(valid, option_types, 0)
    option_contexts = np.where(valid, option_contexts, 0)
    card_ids = np.where(valid, card_ids, 0)
    attack_ids = np.where(valid, attack_ids, 0)
    option_scalars = np.where(valid[..., None], option_scalars, np.float32(0.0))
    entity_card_ids = np.where(present, entity_card_ids, 0)
    entity_areas = np.where(present, entity_areas, 0)
    entity_owners = np.where(present, entity_owners, 0)
    entity_kinds = np.where(present, entity_kinds, 0)
    entity_scalars = np.where(
        present[..., None],
        entity_scalars,
        np.float32(0.0),
    )

    integer_fields = np.asarray(
        np.concatenate(
            (
                option_types[..., None],
                option_contexts[..., None],
                card_ids[..., None],
                attack_ids[..., None],
                entity_card_ids,
                entity_areas,
                entity_owners,
                entity_kinds,
            ),
            axis=-1,
        ),
        dtype="<i8",
        order="C",
    )
    flat_entity_scalars = entity_scalars.reshape(
        batch_size,
        width,
        _ENTITY_NUMERIC_SIZE,
    )
    numeric_fields = np.asarray(
        np.concatenate((option_scalars, flat_entity_scalars), axis=-1),
        dtype="<f4",
        order="C",
    )
    semantic_bytes = np.concatenate(
        (
            integer_fields.view(np.uint8).reshape(batch_size, width, -1),
            numeric_fields.view(np.uint8).reshape(batch_size, width, -1),
        ),
        axis=-1,
    )
    prompt_contexts = _numpy(options.contexts)[:, 0]
    records: list[AcceptedActionRecord] = []
    for row in range(batch_size):
        length = int(lengths[row])
        payloads: list[_StableOptionPayload] = []
        for option in range(length):
            payloads.append(
                (
                    semantic_bytes[row, option].tobytes(),
                    int(option_types[row, option]),
                    int(option_contexts[row, option]),
                    int(card_ids[row, option]),
                    int(attack_ids[row, option]),
                    tuple(float(value) for value in option_scalars[row, option]),
                    tuple(int(value) for value in entity_card_ids[row, option]),
                    tuple(int(value) for value in entity_areas[row, option]),
                    tuple(int(value) for value in entity_owners[row, option]),
                    tuple(int(value) for value in entity_kinds[row, option]),
                    tuple(float(value) for value in flat_entity_scalars[row, option]),
                )
            )
        prompt_context = int(prompt_contexts[row])
        ordered = prompt_context not in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
        encoded = tuple(
            payloads if ordered else sorted(payloads, key=lambda item: item[0])
        )
        records.append(
            _accepted_action_record_from_payloads(
                prompt_context=prompt_context,
                encoded=encoded,
                ordered=ordered,
                stop_sampled=bool(stop_sampled[row]),
                fallback=False,
            )
        )
    return tuple(records)


def _validate_tensor_host_inputs(
    *,
    states: StateBatch,
    options: OptionBatch,
    action_offsets: npt.NDArray[np.int64],
    action_choices: npt.NDArray[np.int32],
    min_counts: Sequence[int],
    max_counts: Sequence[int],
    stop_sampled: npt.NDArray[np.bool_],
) -> None:
    """Reject a malformed tensor/trace join before semantic materialization."""
    batch_size = int(states.card_ids.shape[0])
    if (
        int(options.option_types.shape[0]) != batch_size
        or len(min_counts) != batch_size
        or len(max_counts) != batch_size
        or stop_sampled.shape != (batch_size,)
        or action_offsets.shape != (batch_size + 1,)
        or action_offsets.dtype != np.int64
        or action_choices.dtype != np.int32
        or int(action_offsets[0]) != 0
        or int(action_offsets[-1]) != int(action_choices.size)
        or np.any(np.diff(action_offsets) < 0)
    ):
        raise ValueError("tensor-native accepted-action inputs are misaligned")
    if (
        states.card_ids.device.type != "cpu"
        or options.option_types.device.type != "cpu"
    ):
        raise ValueError("tensor-native accepted actions require CPU source tensors")
    lengths = np.diff(action_offsets)
    minimum = np.asarray(min_counts, dtype=np.int64)
    maximum = np.asarray(max_counts, dtype=np.int64)
    if np.any(lengths < minimum) or np.any(lengths > maximum):
        raise ValueError("accepted action violates its legal option count")
    if np.any(action_choices < 0):
        raise ValueError("accepted action choices must be non-negative")


def _option_lengths(options: OptionBatch) -> npt.NDArray[np.int64]:
    if options.option_lengths:
        return np.asarray(options.option_lengths, dtype=np.int64)
    return cast(
        npt.NDArray[np.int64],
        np.count_nonzero(_numpy(options.valid_options), axis=1).astype(
            np.int64,
            copy=False,
        ),
    )


def _state_lengths(states: StateBatch) -> npt.NDArray[np.int64]:
    if states.sequence_lengths:
        return np.asarray(states.sequence_lengths, dtype=np.int64)
    return cast(
        npt.NDArray[np.int64],
        np.count_nonzero(~_numpy(states.padding_mask), axis=1).astype(
            np.int64,
            copy=False,
        ),
    )


def _numpy(value: Tensor) -> np.ndarray:
    if value.device.type != "cpu":
        raise ValueError("tensor-native accepted actions require CPU tensors")
    return value.detach().numpy()


def _validate_action_feature_shapes(
    *,
    state: StateTokenArrayFeatures,
    options: EncodedOptionArrayFeatures,
) -> None:
    """Validate the array fields read by accepted-action materialization."""
    state_size = len(state.card_ids)
    if any(
        values.shape != (state_size,)
        for values in (
            state.areas,
            state.owner_roles,
            state.token_kinds,
        )
    ) or state.scalars.shape != (state_size, TOKEN_SCALAR_SIZE):
        raise ValueError("accepted-action state fields are misaligned")

    option_size = len(options)
    if any(
        values.shape != (option_size,)
        for values in (
            options.contexts,
            options.card_ids,
            options.attack_ids,
        )
    ):
        raise ValueError("accepted-action option fields are misaligned")
    if options.scalars.shape != (option_size, SCALAR_FEATURE_SIZE):
        raise ValueError("accepted-action option scalar fields are misaligned")
    entity_shape = (option_size, MAX_ENTITY_SLOTS)
    if (
        options.entity_slots.shape != entity_shape
        or options.entity_slot_mask.shape != entity_shape
    ):
        raise ValueError("accepted-action option entity fields are misaligned")


def _stable_option_payload_batch(
    *,
    states: Sequence[StateTokenArrayFeatures],
    options: Sequence[EncodedOptionArrayFeatures],
    actions: Sequence[tuple[int, ...]],
    rows: Sequence[int],
) -> tuple[tuple[_StableOptionPayload, ...], ...]:
    """Gather one same-shaped row group and bulk-convert its Python payloads."""
    group_size = len(rows)
    action_size = len(actions[rows[0]])
    state_size = len(states[rows[0]].card_ids)
    action_indices = np.asarray(
        [actions[row] for row in rows],
        dtype=np.intp,
    )
    batch_indices = np.arange(group_size, dtype=np.intp)[:, None]

    option_types = np.stack(
        [options[row].option_types for row in rows],
        axis=0,
    )[batch_indices, action_indices]
    option_contexts = np.stack(
        [options[row].contexts for row in rows],
        axis=0,
    )[batch_indices, action_indices]
    card_ids = np.stack(
        [options[row].card_ids for row in rows],
        axis=0,
    )[batch_indices, action_indices]
    attack_ids = np.stack(
        [options[row].attack_ids for row in rows],
        axis=0,
    )[batch_indices, action_indices]
    option_scalars = np.stack(
        [options[row].scalars for row in rows],
        axis=0,
    )[batch_indices, action_indices]
    entity_slots = np.stack(
        [options[row].entity_slots for row in rows],
        axis=0,
    )[batch_indices, action_indices].astype(np.intp, copy=False)
    entity_present = np.stack(
        [options[row].entity_slot_mask for row in rows],
        axis=0,
    )[batch_indices, action_indices].astype(np.bool_, copy=False)

    invalid_entities = entity_present & (
        (entity_slots < 0) | (entity_slots >= state_size)
    )
    if bool(np.any(invalid_entities)):
        raise ValueError("accepted-action entity reference is out of range")
    if bool(np.any(entity_present)):
        safe_entities = np.where(entity_present, entity_slots, 0)
        state_indices = np.arange(group_size, dtype=np.intp)[:, None, None]
        entity_card_ids = np.stack(
            [states[row].card_ids for row in rows],
            axis=0,
        )[state_indices, safe_entities]
        entity_areas = np.stack(
            [states[row].areas for row in rows],
            axis=0,
        )[state_indices, safe_entities]
        entity_owners = np.stack(
            [states[row].owner_roles for row in rows],
            axis=0,
        )[state_indices, safe_entities]
        entity_kinds = np.stack(
            [states[row].token_kinds for row in rows],
            axis=0,
        )[state_indices, safe_entities]
        entity_scalars = np.stack(
            [states[row].scalars for row in rows],
            axis=0,
        )[state_indices, safe_entities]
        entity_card_ids = np.where(entity_present, entity_card_ids, 0)
        entity_areas = np.where(entity_present, entity_areas, 0)
        entity_owners = np.where(entity_present, entity_owners, 0)
        entity_kinds = np.where(entity_present, entity_kinds, 0)
        entity_scalars = np.where(
            entity_present[..., None],
            entity_scalars,
            np.float32(0.0),
        )
    else:
        entity_shape = (group_size, action_size, MAX_ENTITY_SLOTS)
        entity_card_ids = np.zeros(entity_shape, dtype=np.int64)
        entity_areas = np.zeros(entity_shape, dtype=np.int64)
        entity_owners = np.zeros(entity_shape, dtype=np.int64)
        entity_kinds = np.zeros(entity_shape, dtype=np.int64)
        entity_scalars = np.zeros(
            (*entity_shape, TOKEN_SCALAR_SIZE),
            dtype=np.float32,
        )

    integer_fields = np.asarray(
        np.concatenate(
            (
                option_types[..., None],
                option_contexts[..., None],
                card_ids[..., None],
                attack_ids[..., None],
                entity_card_ids,
                entity_areas,
                entity_owners,
                entity_kinds,
            ),
            axis=-1,
        ),
        dtype="<i8",
        order="C",
    )
    flat_entity_scalars = entity_scalars.reshape(
        group_size,
        action_size,
        _ENTITY_NUMERIC_SIZE,
    )
    numeric_fields = np.asarray(
        np.concatenate((option_scalars, flat_entity_scalars), axis=-1),
        dtype="<f4",
        order="C",
    )
    semantic_bytes = np.concatenate(
        (
            integer_fields.view(np.uint8).reshape(
                group_size,
                action_size,
                -1,
            ),
            numeric_fields.view(np.uint8).reshape(
                group_size,
                action_size,
                -1,
            ),
        ),
        axis=-1,
    )

    option_type_rows = cast(list[list[int]], option_types.tolist())
    option_context_rows = cast(list[list[int]], option_contexts.tolist())
    card_id_rows = cast(list[list[int]], card_ids.tolist())
    attack_id_rows = cast(list[list[int]], attack_ids.tolist())
    option_scalar_rows = cast(
        list[list[list[float]]],
        option_scalars.tolist(),
    )
    entity_card_rows = cast(
        list[list[list[int]]],
        entity_card_ids.tolist(),
    )
    entity_area_rows = cast(list[list[list[int]]], entity_areas.tolist())
    entity_owner_rows = cast(list[list[list[int]]], entity_owners.tolist())
    entity_kind_rows = cast(list[list[list[int]]], entity_kinds.tolist())
    entity_scalar_rows = cast(
        list[list[list[float]]],
        flat_entity_scalars.tolist(),
    )
    payload_rows: list[tuple[_StableOptionPayload, ...]] = []
    for row in range(group_size):
        payloads: list[_StableOptionPayload] = []
        for option in range(action_size):
            payloads.append(
                (
                    semantic_bytes[row, option].tobytes(),
                    option_type_rows[row][option],
                    option_context_rows[row][option],
                    card_id_rows[row][option],
                    attack_id_rows[row][option],
                    tuple(option_scalar_rows[row][option]),
                    tuple(entity_card_rows[row][option]),
                    tuple(entity_area_rows[row][option]),
                    tuple(entity_owner_rows[row][option]),
                    tuple(entity_kind_rows[row][option]),
                    tuple(entity_scalar_rows[row][option]),
                )
            )
        payload_rows.append(tuple(payloads))
    return tuple(payload_rows)


__all__ = [
    "build_host_accepted_action_records",
    "build_tensor_host_accepted_action_records",
]
