"""Column conversion and gathering for object-free native trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from torch import Tensor

from ptcg_rl.actions.encoding import SCALAR_FEATURE_SIZE
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.native_public_context import NativeKnownOpponentBatch
from ptcg_rl.model.policy import MAX_ENTITY_SLOTS
from ptcg_rl.model.sequence.action import AcceptedActionRecord
from ptcg_rl.model.state_encoder import TOKEN_SCALAR_SIZE
from ptcg_rl.rl.native_policy_batch import NativeSimpleStatelessPolicyBatch
from ptcg_rl.rl.native_policy_trace import NativePolicyNumpyTrace
from ptcg_rl.rl.native_sequence_trajectory import (
    build_native_sequence_decision_arrays,
)
from ptcg_rl.rl.policy_inputs import SimpleStatelessActorRow

Array = npt.NDArray[np.generic]
Int64Array = npt.NDArray[np.int64]

_ROW_BITS = 32
_ROW_MASK = (1 << _ROW_BITS) - 1

_DENSE_FIELDS = (
    "action_logprobs",
    "root_values",
    "stop_sampled",
    "min_counts",
    "max_counts",
    "belief_scalars",
)
_SEQUENCE_DENSE_FIELDS = (
    "engine_fact_producer_fingerprints",
    "sequence_request_ids",
    "accepted_action_stable_ids",
    "accepted_action_prompt_contexts",
    "accepted_action_ordered",
    "accepted_action_fallback",
)
_RAGGED_FIELDS = (
    (
        "state_offsets",
        (
            "state_card_ids",
            "state_areas",
            "state_owner_roles",
            "state_token_kinds",
            "state_scalars",
            "state_last_attack_ids",
            "state_entity_slots",
        ),
    ),
    (
        "attachment_offsets",
        (
            "attachment_card_ids",
            "attachment_parent_indices",
            "attachment_kinds",
        ),
    ),
    (
        "option_offsets",
        (
            "option_types",
            "option_contexts",
            "option_entity_slots",
            "option_entity_slot_mask",
            "option_attack_ids",
            "option_card_ids",
            "option_scalars",
            "option_dynamic_effect_features",
            "option_dynamic_effect_masks",
        ),
    ),
    (
        "belief_offsets",
        (
            "belief_card_ids",
            "belief_expected_counts",
        ),
    ),
    (
        "known_offsets",
        (
            "known_card_ids",
            "known_counts",
        ),
    ),
    (
        "action_offsets",
        ("action_choices",),
    ),
    (
        "token_offsets",
        (
            "token_logprobs",
            "prefix_values",
        ),
    ),
)
_SEQUENCE_RAGGED_FIELDS = (
    (
        "event_offsets",
        (
            "event_types",
            "event_actor_roles",
            "event_from_areas",
            "event_to_areas",
            "event_card_ids",
            "event_serials",
            "event_entity_mask",
            "event_attack_ids",
            "event_attack_id_mask",
            "event_values",
            "event_value_mask",
            "event_categorical_values",
        ),
    ),
    (
        "event_overflow_offsets",
        (
            "event_overflow_types",
            "event_overflow_actor_roles",
            "event_overflow_counts",
        ),
    ),
)
_ACCEPTED_ACTION_FIELDS = (
    "accepted_action_option_types",
    "accepted_action_option_contexts",
    "accepted_action_card_ids",
    "accepted_action_attack_ids",
    "accepted_action_option_scalars",
    "accepted_action_entity_card_ids",
    "accepted_action_entity_areas",
    "accepted_action_entity_owner_roles",
    "accepted_action_entity_token_kinds",
    "accepted_action_entity_scalars",
)


@dataclass(frozen=True, slots=True)
class NativeDecisionChunk:
    """One immutable CPU column block aligned to an inference batch."""

    row_count: int
    arrays: Mapping[str, Array]


@dataclass(frozen=True, slots=True)
class _ChunkSelection:
    """One source chunk's rows and aligned output destinations."""

    chunk_id: int
    output_rows: Int64Array
    local_rows: Int64Array


def build_native_decision_chunk(
    batch: NativeSimpleStatelessPolicyBatch,
    known: NativeKnownOpponentBatch,
    trace: NativePolicyNumpyTrace,
    *,
    batch_rows: npt.ArrayLike | None = None,
    sequence_rows: Sequence[SimpleStatelessActorRow] | None = None,
    accepted_actions: Sequence[AcceptedActionRecord] | None = None,
) -> NativeDecisionChunk:
    """Gather policy rows once into compact CPU trajectory columns."""
    selected = _batch_rows(batch_rows, size=batch.batch_size)
    rows = int(selected.size)
    if known.batch_size != rows or trace.batch_size != rows:
        raise ValueError("native trajectory inputs must be aligned non-empty rows")
    states = batch.states
    options = batch.options
    state_mask: npt.NDArray[np.bool_] = ~np.asarray(
        _tensor_numpy(states.padding_mask)[selected],
        dtype=np.bool_,
    )
    option_mask: npt.NDArray[np.bool_] = np.asarray(
        _tensor_numpy(options.valid_options)[selected],
        dtype=np.bool_,
    )
    _validate_prefix_mask(state_mask, name="state")
    _validate_prefix_mask(option_mask, name="option")
    if np.any(state_mask.sum(axis=1) <= 0):
        raise ValueError("native trajectory state rows must contain tokens")
    if np.any(option_mask.sum(axis=1) <= 0):
        raise ValueError("native trajectory option rows must contain choices")

    state_arrays = _state_columns(batch, state_mask, selected)
    attachment_arrays = _attachment_columns(batch, selected)
    option_arrays = _option_columns(batch, option_mask, selected)
    belief_arrays = _belief_columns(batch, selected)
    known_arrays = _known_columns(known, rows=rows)
    trace_arrays = _trace_columns(trace)
    arrays: dict[str, Array] = {
        **state_arrays,
        **attachment_arrays,
        **option_arrays,
        **belief_arrays,
        **known_arrays,
        **trace_arrays,
    }
    if (sequence_rows is None) != (accepted_actions is None):
        raise ValueError("native sequence rows and actions must be supplied together")
    if sequence_rows is not None and accepted_actions is not None:
        if len(sequence_rows) != rows or len(accepted_actions) != rows:
            raise ValueError("native sequence trajectory rows differ from trace")
        if any(
            len(action.option_types)
            != int(trace.action_offsets[index + 1] - trace.action_offsets[index])
            for index, action in enumerate(accepted_actions)
        ):
            raise ValueError("native accepted actions differ from sampled lengths")
        arrays.update(
            build_native_sequence_decision_arrays(
                sequence_rows,
                accepted_actions,
            )
        )
    return NativeDecisionChunk(row_count=rows, arrays=arrays)


def native_row_references(chunk_id: int, row_count: int) -> Int64Array:
    """Pack one chunk's row coordinates into scalar int64 references."""
    if chunk_id < 0 or chunk_id > np.iinfo(np.int32).max:
        raise ValueError("native trajectory chunk ID is out of range")
    if row_count <= 0 or row_count > _ROW_MASK:
        raise ValueError("native trajectory chunk row count is out of range")
    return (np.int64(chunk_id) << np.int64(_ROW_BITS)) | np.arange(
        row_count, dtype=np.int64
    )


def gather_native_decision_columns(
    chunks: Mapping[int, NativeDecisionChunk],
    references: Int64Array,
) -> dict[str, Array]:
    """Gather arbitrary fragment-ordered rows from immutable column blocks."""
    if references.ndim != 1 or references.size <= 0:
        raise ValueError("native trajectory part requires decision references")
    chunk_ids, local_rows = _unpack_references(references)
    groups: list[_ChunkSelection] = []
    for raw_chunk_id in np.unique(chunk_ids):
        chunk_id = int(raw_chunk_id)
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise RuntimeError("native trajectory references a released chunk")
        output_rows = np.flatnonzero(chunk_ids == chunk_id)
        selected = local_rows[output_rows]
        if np.any(selected < 0) or np.any(selected >= chunk.row_count):
            raise RuntimeError("native trajectory row reference is out of range")
        groups.append(
            _ChunkSelection(
                chunk_id=chunk_id,
                output_rows=output_rows,
                local_rows=selected,
            )
        )

    first_arrays = chunks[groups[0].chunk_id].arrays
    sequence = "sequence_request_ids" in first_arrays
    if any(
        ("sequence_request_ids" in chunks[group.chunk_id].arrays) != sequence
        for group in groups[1:]
    ):
        raise ValueError("native trajectory chunks mix sequence schemas")
    output: dict[str, Array] = {}
    dense_fields = (
        (*_DENSE_FIELDS, *_SEQUENCE_DENSE_FIELDS)
        if sequence
        else _DENSE_FIELDS
    )
    for field in dense_fields:
        output[field] = _gather_dense(
            chunks,
            groups,
            row_count=int(references.size),
            field=field,
        )
    ragged_fields = list(_RAGGED_FIELDS)
    if sequence:
        ragged_fields.extend(_SEQUENCE_RAGGED_FIELDS)
        ragged_fields = [
            (
                offsets,
                (*values, *_ACCEPTED_ACTION_FIELDS)
                if offsets == "action_offsets"
                else values,
            )
            for offsets, values in ragged_fields
        ]
    for offsets_field, value_fields in ragged_fields:
        offsets, values = _gather_ragged(
            chunks,
            groups,
            row_count=int(references.size),
            offsets_field=offsets_field,
            value_fields=value_fields,
        )
        output[offsets_field] = offsets
        output.update(values)
    return output


def referenced_chunk_ids(references: Sequence[Int64Array]) -> set[int]:
    """Return chunk IDs retained by live or unpublished fragments."""
    result: set[int] = set()
    for rows in references:
        if rows.size:
            chunk_ids, _local_rows = _unpack_references(rows)
            result.update(int(value) for value in np.unique(chunk_ids))
    return result


def _state_columns(
    batch: NativeSimpleStatelessPolicyBatch,
    mask: npt.NDArray[np.bool_],
    rows: Int64Array,
) -> dict[str, Array]:
    states = batch.states
    source_shape = (batch.batch_size, mask.shape[1])
    _require_shape(states.card_ids, source_shape, "state card IDs")
    _require_shape(states.areas, source_shape, "state areas")
    _require_shape(states.owner_roles, source_shape, "state owner roles")
    _require_shape(states.token_kinds, source_shape, "state token kinds")
    _require_shape(
        states.scalars,
        (*source_shape, TOKEN_SCALAR_SIZE),
        "state scalars",
    )
    _require_shape(
        states.last_attack_ids,
        source_shape,
        "state last attack IDs",
    )
    if states.entity_slots is None:
        raise ValueError("native state batch requires entity slots")
    _require_shape(states.entity_slots, source_shape, "state entity slots")
    return {
        "state_offsets": _mask_offsets(mask),
        "state_card_ids": _masked_tensor(states.card_ids, rows, mask, np.int32),
        "state_areas": _masked_tensor(states.areas, rows, mask, np.int16),
        "state_owner_roles": _masked_tensor(
            states.owner_roles,
            rows,
            mask,
            np.int8,
        ),
        "state_token_kinds": _masked_tensor(
            states.token_kinds,
            rows,
            mask,
            np.int8,
        ),
        "state_scalars": _masked_tensor(states.scalars, rows, mask, np.float32),
        "state_last_attack_ids": _masked_tensor(
            states.last_attack_ids,
            rows,
            mask,
            np.int32,
        ),
        "state_entity_slots": _masked_tensor(
            states.entity_slots,
            rows,
            mask,
            np.uint8,
        ),
    }


def _attachment_columns(
    batch: NativeSimpleStatelessPolicyBatch,
    rows: Int64Array,
) -> dict[str, Array]:
    states = batch.states
    if (
        states.attachment_card_ids is None
        or states.attachment_parent_indices is None
        or states.attachment_kinds is None
    ):
        raise ValueError("native state batch requires attachment columns")
    source_kinds = _tensor_numpy(states.attachment_kinds, np.int8)
    if source_kinds.ndim != 2 or source_kinds.shape[0] != batch.batch_size:
        raise ValueError("native attachment rows do not align with decisions")
    kinds = source_kinds[rows]
    mask = kinds != 0
    _validate_prefix_mask(mask, name="attachment")
    _require_shape(
        states.attachment_card_ids,
        (batch.batch_size, mask.shape[1]),
        "attachment card IDs",
    )
    _require_shape(
        states.attachment_parent_indices,
        (batch.batch_size, mask.shape[1]),
        "attachment parent indices",
    )
    return {
        "attachment_offsets": _mask_offsets(mask),
        "attachment_card_ids": _masked_tensor(
            states.attachment_card_ids,
            rows,
            mask,
            np.int32,
        ),
        "attachment_parent_indices": _masked_tensor(
            states.attachment_parent_indices,
            rows,
            mask,
            np.int32,
        ),
        "attachment_kinds": np.ascontiguousarray(kinds[mask], dtype=np.int8),
    }


def _option_columns(
    batch: NativeSimpleStatelessPolicyBatch,
    mask: npt.NDArray[np.bool_],
    rows: Int64Array,
) -> dict[str, Array]:
    options = batch.options
    _selected_rows, width = mask.shape
    source_shape = (batch.batch_size, width)
    _require_shape(options.option_types, source_shape, "option types")
    _require_shape(options.contexts, source_shape, "option contexts")
    _require_shape(
        options.entity_slots,
        (*source_shape, MAX_ENTITY_SLOTS),
        "option entity slots",
    )
    _require_shape(
        options.entity_slot_mask,
        (*source_shape, MAX_ENTITY_SLOTS),
        "option entity slot mask",
    )
    _require_shape(options.attack_ids, source_shape, "option attack IDs")
    _require_shape(options.card_ids, source_shape, "option card IDs")
    _require_shape(
        options.scalars,
        (*source_shape, SCALAR_FEATURE_SIZE),
        "option scalars",
    )
    _require_shape(
        options.dynamic_effect_features,
        (*source_shape, DYNAMIC_EFFECT_FEATURE_SIZE),
        "option dynamic effect features",
    )
    _require_shape(
        options.dynamic_effect_masks,
        source_shape,
        "option dynamic effect masks",
    )
    minimums = _tensor_numpy(options.min_counts, np.int64)[rows]
    maximums = _tensor_numpy(options.max_counts, np.int64)[rows]
    declared_minimums = np.asarray(batch.min_counts, dtype=np.int64)[rows]
    declared_maximums = np.asarray(batch.max_counts, dtype=np.int64)[rows]
    if (
        minimums.shape != (_selected_rows,)
        or maximums.shape != (_selected_rows,)
        or not np.array_equal(minimums, declared_minimums)
        or not np.array_equal(maximums, declared_maximums)
    ):
        raise ValueError("native option cardinality metadata is inconsistent")
    return {
        "option_offsets": _mask_offsets(mask),
        "option_types": _masked_tensor(
            options.option_types,
            rows,
            mask,
            np.int16,
        ),
        "option_contexts": _masked_tensor(
            options.contexts,
            rows,
            mask,
            np.int16,
        ),
        "option_entity_slots": _masked_tensor(
            options.entity_slots,
            rows,
            mask,
            np.int32,
        ),
        "option_entity_slot_mask": _masked_tensor(
            options.entity_slot_mask,
            rows,
            mask,
            np.bool_,
        ),
        "option_attack_ids": _masked_tensor(
            options.attack_ids,
            rows,
            mask,
            np.int32,
        ),
        "option_card_ids": _masked_tensor(
            options.card_ids,
            rows,
            mask,
            np.int32,
        ),
        "option_scalars": _masked_tensor(
            options.scalars,
            rows,
            mask,
            np.float32,
        ),
        "option_dynamic_effect_features": _masked_tensor(
            options.dynamic_effect_features,
            rows,
            mask,
            np.float32,
        ),
        "option_dynamic_effect_masks": _masked_tensor(
            options.dynamic_effect_masks,
            rows,
            mask,
            np.bool_,
        ),
        "min_counts": minimums.astype(np.int16, copy=True),
        "max_counts": maximums.astype(np.int16, copy=True),
    }


def _belief_columns(
    batch: NativeSimpleStatelessPolicyBatch,
    selected: Int64Array,
) -> dict[str, Array]:
    summary = batch.belief_summary
    card_ids = _tensor_numpy(summary.card_ids, np.int64)
    expected = _tensor_numpy(summary.expected_counts, np.float32)
    valid: npt.NDArray[np.bool_] = np.asarray(
        _tensor_numpy(summary.valid_mask),
        dtype=np.bool_,
    )
    scalars = _tensor_numpy(summary.scalars, np.float32)
    if (
        card_ids.ndim != 2
        or expected.shape != card_ids.shape
        or valid.shape != card_ids.shape
        or scalars.shape != (card_ids.shape[0], 4)
    ):
        raise ValueError("native belief summary tensors are misaligned")
    inverse: Int64Array
    if summary.row_indices is None:
        inverse = np.arange(batch.batch_size, dtype=np.int64)
    else:
        inverse = np.asarray(
            _tensor_numpy(summary.row_indices),
            dtype=np.int64,
        )
    if (
        inverse.shape != (batch.batch_size,)
        or np.any(inverse < 0)
        or np.any(inverse >= card_ids.shape[0])
    ):
        raise ValueError("native belief inverse rows are invalid")
    rows = inverse[selected]
    selected_valid = valid[rows]
    _validate_prefix_mask(selected_valid, name="belief")
    return {
        "belief_offsets": _mask_offsets(selected_valid),
        "belief_card_ids": np.ascontiguousarray(
            card_ids[rows][selected_valid],
            dtype=np.int32,
        ),
        "belief_expected_counts": np.ascontiguousarray(
            expected[rows][selected_valid],
            dtype=np.float32,
        ),
        "belief_scalars": np.ascontiguousarray(scalars[rows], dtype=np.float32),
    }


def _known_columns(
    known: NativeKnownOpponentBatch,
    *,
    rows: int,
) -> dict[str, Array]:
    offsets = np.asarray(known.offsets, dtype=np.int64)
    card_ids = np.asarray(known.card_ids, dtype=np.int32)
    counts = np.asarray(known.counts, dtype=np.int64)
    _validate_offsets(
        offsets,
        rows=rows,
        values=card_ids.shape[0],
        name="known opponent",
    )
    if counts.shape != card_ids.shape:
        raise ValueError("native known-opponent values are misaligned")
    if np.any(counts > np.iinfo(np.int16).max):
        raise ValueError("native known-opponent count exceeds int16")
    return {
        "known_offsets": offsets.copy(),
        "known_card_ids": card_ids.copy(),
        "known_counts": counts.astype(np.int16, copy=True),
    }


def _trace_columns(trace: NativePolicyNumpyTrace) -> dict[str, Array]:
    return {
        "action_offsets": trace.action_offsets.copy(),
        "action_choices": trace.action_choices.copy(),
        "action_logprobs": trace.action_logprobs.copy(),
        "token_offsets": trace.token_offsets.copy(),
        "token_logprobs": trace.token_logprobs.copy(),
        "prefix_values": trace.prefix_values.copy(),
        "root_values": trace.root_values.copy(),
        "stop_sampled": trace.stop_sampled.copy(),
    }


def _gather_dense(
    chunks: Mapping[int, NativeDecisionChunk],
    groups: Sequence[_ChunkSelection],
    *,
    row_count: int,
    field: str,
) -> Array:
    first = chunks[groups[0].chunk_id].arrays[field]
    result = np.empty((row_count, *first.shape[1:]), dtype=first.dtype)
    for group in groups:
        source = chunks[group.chunk_id].arrays[field]
        result[group.output_rows] = source[group.local_rows]
    return result


def _gather_ragged(
    chunks: Mapping[int, NativeDecisionChunk],
    groups: Sequence[_ChunkSelection],
    *,
    row_count: int,
    offsets_field: str,
    value_fields: tuple[str, ...],
) -> tuple[Int64Array, dict[str, Array]]:
    lengths = np.empty(row_count, dtype=np.int64)
    source_offsets: dict[int, Int64Array] = {}
    for group in groups:
        offsets = chunks[group.chunk_id].arrays[offsets_field].astype(
            np.int64,
            copy=False,
        )
        source_offsets[group.chunk_id] = offsets
        lengths[group.output_rows] = (
            offsets[group.local_rows + 1] - offsets[group.local_rows]
        )
    output_offsets = np.zeros(row_count + 1, dtype=np.int64)
    output_offsets[1:] = np.cumsum(lengths, dtype=np.int64)

    gather_indices: list[tuple[_ChunkSelection, Int64Array, Int64Array]] = []
    for group in groups:
        row_lengths = lengths[group.output_rows]
        count = int(row_lengths.sum())
        if count == 0:
            continue
        local_positions = np.arange(count, dtype=np.int64) - np.repeat(
            np.cumsum(row_lengths, dtype=np.int64) - row_lengths,
            row_lengths,
        )
        source_indices = np.repeat(
            source_offsets[group.chunk_id][group.local_rows],
            row_lengths,
        )
        source_indices += local_positions
        destination_indices = np.repeat(
            output_offsets[group.output_rows],
            row_lengths,
        )
        destination_indices += local_positions
        gather_indices.append(
            (group, source_indices, destination_indices)
        )

    gathered: dict[str, Array] = {}
    for field in value_fields:
        first = chunks[groups[0].chunk_id].arrays[field]
        result = np.empty(
            (int(output_offsets[-1]), *first.shape[1:]),
            dtype=first.dtype,
        )
        for group, source_indices, destination_indices in gather_indices:
            result[destination_indices] = chunks[group.chunk_id].arrays[field][
                source_indices
            ]
        gathered[field] = result
    return output_offsets, gathered


def _unpack_references(references: Int64Array) -> tuple[Int64Array, Int64Array]:
    if np.any(references < 0):
        raise RuntimeError("native trajectory row reference is negative")
    return references >> np.int64(_ROW_BITS), references & np.int64(_ROW_MASK)


def _masked_tensor(
    tensor: Tensor,
    rows: Int64Array,
    mask: npt.NDArray[np.bool_],
    dtype: npt.DTypeLike,
) -> Array:
    values = _tensor_numpy(tensor)
    return np.ascontiguousarray(values[rows][mask], dtype=dtype)


def _batch_rows(
    values: npt.ArrayLike | None,
    *,
    size: int,
) -> Int64Array:
    if values is None:
        return np.arange(size, dtype=np.int64)
    rows = np.asarray(values)
    if rows.ndim != 1 or rows.size <= 0:
        raise ValueError("native trajectory batch rows must be non-empty")
    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError("native trajectory batch rows must use integers")
    normalized = rows.astype(np.int64, copy=False)
    if (
        np.any(normalized < 0)
        or np.any(normalized >= size)
        or np.unique(normalized).size != normalized.size
    ):
        raise ValueError("native trajectory batch rows are invalid")
    return normalized


def _tensor_numpy(
    tensor: Tensor,
    dtype: npt.DTypeLike | None = None,
) -> Array:
    if tensor.device.type != "cpu":
        raise ValueError("native trajectory page accepts CPU tensors only")
    values = tensor.detach().numpy()
    if dtype is None:
        return values
    return np.asarray(values, dtype=dtype)


def _mask_offsets(mask: npt.NDArray[np.bool_]) -> Int64Array:
    result = np.zeros(mask.shape[0] + 1, dtype=np.int64)
    result[1:] = np.cumsum(mask.sum(axis=1, dtype=np.int64), dtype=np.int64)
    return result


def _validate_prefix_mask(
    mask: npt.NDArray[np.bool_],
    *,
    name: str,
) -> None:
    if mask.ndim != 2:
        raise ValueError(f"native {name} mask must have shape [batch, width]")
    if mask.shape[1] > 1 and np.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ValueError(f"native {name} mask is not prefix packed")


def _validate_offsets(
    offsets: Int64Array,
    *,
    rows: int,
    values: int,
    name: str,
) -> None:
    if (
        offsets.shape != (rows + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != values
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(f"native {name} offsets are invalid")


def _require_shape(tensor: Tensor, shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"native {name} must have shape {shape}")


__all__ = [
    "NativeDecisionChunk",
    "build_native_decision_chunk",
    "gather_native_decision_columns",
    "native_row_references",
    "referenced_chunk_ids",
]
