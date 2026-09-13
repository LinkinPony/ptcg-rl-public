"""Row selection for native training SoA/CSR batch views."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt

from ptcg_rl.engine.native_training import NativeTrainingBatchView

Array = npt.NDArray[np.generic]
Int32Array = npt.NDArray[np.int32]
Uint32Array = npt.NDArray[np.uint32]
_MISSING_ROW = np.iinfo(np.uint32).max


def concatenate_native_training_views(
    views: Sequence[NativeTrainingBatchView],
) -> NativeTrainingBatchView:
    """Concatenate disjoint native rows and rebase every nested CSR pointer."""
    selected = tuple(views)
    if not selected:
        raise ValueError("native view concatenation requires at least one batch")
    if any(view.batch_size <= 0 for view in selected):
        raise ValueError("native view concatenation requires non-empty batches")
    if len(selected) == 1:
        return selected[0]
    slots = np.concatenate(tuple(view.slots for view in selected))
    if np.unique(slots).size != slots.size:
        raise ValueError("native view concatenation requires disjoint slots")

    option_offsets, option_columns = _concatenate_csr(
        tuple(view.option_offsets for view in selected),
        tuple((view.option_type, *view.option_params) for view in selected),
    )
    visible_offsets, visible_columns = _concatenate_csr(
        tuple(view.visible_card_offsets for view in selected),
        tuple(
            (
                view.visible_card_owner,
                view.visible_card_area,
                view.visible_card_area_index,
                view.visible_card_id,
                view.visible_card_serial,
                view.visible_card_hp,
                view.visible_card_max_hp,
                view.visible_card_appear_this_turn,
            )
            for view in selected
        ),
    )
    visible_bases = _cumulative_value_bases(
        tuple(view.visible_card_count for view in selected)
    )
    attachment_offsets, attachment_columns = _concatenate_csr(
        tuple(view.attachment_offsets for view in selected),
        tuple(
            (
                view.attachment_parent,
                view.attachment_kind,
                view.attachment_card_id,
                view.attachment_card_serial,
                view.attachment_energy_type,
                view.attachment_energy_units,
            )
            for view in selected
        ),
    )
    attachment_parent = np.concatenate(
        tuple(
            view.attachment_parent + visible_base
            for view, visible_base in zip(
                selected,
                visible_bases,
                strict=True,
            )
        )
    ).astype(np.uint32, copy=False)
    log_offsets, log_columns = _concatenate_csr(
        tuple(view.log_offsets for view in selected),
        tuple((view.log_type, view.log_param_count, *view.log_params) for view in selected),
    )
    return NativeTrainingBatchView(
        slots=slots,
        status=np.concatenate(tuple(view.status for view in selected)),
        error=np.concatenate(tuple(view.error for view in selected)),
        select_player=np.concatenate(tuple(view.select_player for view in selected)),
        select_type=np.concatenate(tuple(view.select_type for view in selected)),
        select_context=np.concatenate(tuple(view.select_context for view in selected)),
        select_min=np.concatenate(tuple(view.select_min for view in selected)),
        select_max=np.concatenate(tuple(view.select_max for view in selected)),
        result=np.concatenate(tuple(view.result for view in selected)),
        turn=np.concatenate(tuple(view.turn for view in selected)),
        option_offsets=option_offsets,
        option_type=cast(Int32Array, option_columns[0]),
        option_params=(
            cast(Int32Array, option_columns[1]),
            cast(Int32Array, option_columns[2]),
            cast(Int32Array, option_columns[3]),
            cast(Int32Array, option_columns[4]),
            cast(Int32Array, option_columns[5]),
        ),
        turn_action_count=np.concatenate(
            tuple(view.turn_action_count for view in selected)
        ),
        first_player=np.concatenate(tuple(view.first_player for view in selected)),
        turn_flags=np.concatenate(tuple(view.turn_flags for view in selected)),
        remain_damage_counter=np.concatenate(
            tuple(view.remain_damage_counter for view in selected)
        ),
        remain_energy_cost=np.concatenate(
            tuple(view.remain_energy_cost for view in selected)
        ),
        player_deck_counts=_concatenate_pairs(
            tuple(view.player_deck_counts for view in selected)
        ),
        player_hand_counts=_concatenate_pairs(
            tuple(view.player_hand_counts for view in selected)
        ),
        player_prize_counts=_concatenate_pairs(
            tuple(view.player_prize_counts for view in selected)
        ),
        player_bench_max=_concatenate_pairs(
            tuple(view.player_bench_max for view in selected)
        ),
        player_status_flags=(
            np.concatenate(
                tuple(view.player_status_flags[0] for view in selected)
            ),
            np.concatenate(
                tuple(view.player_status_flags[1] for view in selected)
            ),
        ),
        looking_mode=np.concatenate(tuple(view.looking_mode for view in selected)),
        select_deck_visible=np.concatenate(
            tuple(view.select_deck_visible for view in selected)
        ),
        context_card_row=_concatenate_optional_visible_rows(
            tuple(view.context_card_row for view in selected),
            visible_bases=visible_bases,
        ),
        effect_card_row=_concatenate_optional_visible_rows(
            tuple(view.effect_card_row for view in selected),
            visible_bases=visible_bases,
        ),
        visible_card_offsets=visible_offsets,
        visible_card_owner=cast(Int32Array, visible_columns[0]),
        visible_card_area=cast(Int32Array, visible_columns[1]),
        visible_card_area_index=cast(Int32Array, visible_columns[2]),
        visible_card_id=cast(Int32Array, visible_columns[3]),
        visible_card_serial=cast(Int32Array, visible_columns[4]),
        visible_card_hp=cast(Int32Array, visible_columns[5]),
        visible_card_max_hp=cast(Int32Array, visible_columns[6]),
        visible_card_appear_this_turn=cast(Int32Array, visible_columns[7]),
        attachment_offsets=attachment_offsets,
        attachment_parent=attachment_parent,
        attachment_kind=cast(Int32Array, attachment_columns[1]),
        attachment_card_id=cast(Int32Array, attachment_columns[2]),
        attachment_card_serial=cast(Int32Array, attachment_columns[3]),
        attachment_energy_type=cast(Int32Array, attachment_columns[4]),
        attachment_energy_units=cast(Int32Array, attachment_columns[5]),
        log_offsets=log_offsets,
        log_type=cast(Int32Array, log_columns[0]),
        log_param_count=cast(Uint32Array, log_columns[1]),
        log_params=(
            cast(Int32Array, log_columns[2]),
            cast(Int32Array, log_columns[3]),
            cast(Int32Array, log_columns[4]),
            cast(Int32Array, log_columns[5]),
            cast(Int32Array, log_columns[6]),
            cast(Int32Array, log_columns[7]),
            cast(Int32Array, log_columns[8]),
        ),
        selection_advance_count=np.concatenate(
            tuple(view.selection_advance_count for view in selected)
        ),
        _owner=selected[0]._owner,
    )


def select_native_training_rows(
    view: NativeTrainingBatchView,
    rows: npt.ArrayLike,
) -> NativeTrainingBatchView:
    """Copy selected rows while preserving every nested CSR relationship."""
    selected = _validated_rows(rows, size=view.batch_size)
    (
        option_offsets,
        option_columns,
        _option_source_starts,
        _option_destination_starts,
    ) = _select_csr(
        view.option_offsets,
        selected,
        (
            view.option_type,
            *view.option_params,
        ),
    )
    (
        visible_offsets,
        visible_columns,
        visible_source_starts,
        visible_destination_starts,
    ) = _select_csr(
        view.visible_card_offsets,
        selected,
        (
            view.visible_card_owner,
            view.visible_card_area,
            view.visible_card_area_index,
            view.visible_card_id,
            view.visible_card_serial,
            view.visible_card_hp,
            view.visible_card_max_hp,
            view.visible_card_appear_this_turn,
        ),
    )
    (
        attachment_offsets,
        attachment_columns,
        _attachment_source_starts,
        _attachment_destination_starts,
    ) = _select_csr(
        view.attachment_offsets,
        selected,
        (
            view.attachment_parent,
            view.attachment_kind,
            view.attachment_card_id,
            view.attachment_card_serial,
            view.attachment_energy_type,
            view.attachment_energy_units,
        ),
    )
    attachment_parent = _remap_attachment_parents(
        attachment_columns[0],
        attachment_offsets=attachment_offsets,
        visible_source_starts=visible_source_starts,
        visible_source_stops=view.visible_card_offsets[selected + 1],
        visible_destination_starts=visible_destination_starts,
    )
    (
        log_offsets,
        log_columns,
        _log_source_starts,
        _log_destination_starts,
    ) = _select_csr(
        view.log_offsets,
        selected,
        (
            view.log_type,
            view.log_param_count,
            *view.log_params,
        ),
    )
    context_card_row = _remap_optional_visible_rows(
        view.context_card_row[selected],
        source_starts=visible_source_starts,
        source_stops=view.visible_card_offsets[selected + 1],
        destination_starts=visible_destination_starts,
        name="context card",
    )
    effect_card_row = _remap_optional_visible_rows(
        view.effect_card_row[selected],
        source_starts=visible_source_starts,
        source_stops=view.visible_card_offsets[selected + 1],
        destination_starts=visible_destination_starts,
        name="effect card",
    )
    return NativeTrainingBatchView(
        slots=view.slots[selected].copy(),
        status=view.status[selected].copy(),
        error=view.error[selected].copy(),
        select_player=view.select_player[selected].copy(),
        select_type=view.select_type[selected].copy(),
        select_context=view.select_context[selected].copy(),
        select_min=view.select_min[selected].copy(),
        select_max=view.select_max[selected].copy(),
        result=view.result[selected].copy(),
        turn=view.turn[selected].copy(),
        option_offsets=option_offsets,
        option_type=cast(Int32Array, option_columns[0]),
        option_params=(
            cast(Int32Array, option_columns[1]),
            cast(Int32Array, option_columns[2]),
            cast(Int32Array, option_columns[3]),
            cast(Int32Array, option_columns[4]),
            cast(Int32Array, option_columns[5]),
        ),
        turn_action_count=view.turn_action_count[selected].copy(),
        first_player=view.first_player[selected].copy(),
        turn_flags=view.turn_flags[selected].copy(),
        remain_damage_counter=view.remain_damage_counter[selected].copy(),
        remain_energy_cost=view.remain_energy_cost[selected].copy(),
        player_deck_counts=(
            view.player_deck_counts[0][selected].copy(),
            view.player_deck_counts[1][selected].copy(),
        ),
        player_hand_counts=(
            view.player_hand_counts[0][selected].copy(),
            view.player_hand_counts[1][selected].copy(),
        ),
        player_prize_counts=(
            view.player_prize_counts[0][selected].copy(),
            view.player_prize_counts[1][selected].copy(),
        ),
        player_bench_max=(
            view.player_bench_max[0][selected].copy(),
            view.player_bench_max[1][selected].copy(),
        ),
        player_status_flags=(
            view.player_status_flags[0][selected].copy(),
            view.player_status_flags[1][selected].copy(),
        ),
        looking_mode=view.looking_mode[selected].copy(),
        select_deck_visible=view.select_deck_visible[selected].copy(),
        context_card_row=context_card_row,
        effect_card_row=effect_card_row,
        visible_card_offsets=visible_offsets,
        visible_card_owner=cast(Int32Array, visible_columns[0]),
        visible_card_area=cast(Int32Array, visible_columns[1]),
        visible_card_area_index=cast(Int32Array, visible_columns[2]),
        visible_card_id=cast(Int32Array, visible_columns[3]),
        visible_card_serial=cast(Int32Array, visible_columns[4]),
        visible_card_hp=cast(Int32Array, visible_columns[5]),
        visible_card_max_hp=cast(Int32Array, visible_columns[6]),
        visible_card_appear_this_turn=cast(
            Int32Array,
            visible_columns[7],
        ),
        attachment_offsets=attachment_offsets,
        attachment_parent=attachment_parent,
        attachment_kind=cast(Int32Array, attachment_columns[1]),
        attachment_card_id=cast(Int32Array, attachment_columns[2]),
        attachment_card_serial=cast(Int32Array, attachment_columns[3]),
        attachment_energy_type=cast(Int32Array, attachment_columns[4]),
        attachment_energy_units=cast(Int32Array, attachment_columns[5]),
        log_offsets=log_offsets,
        log_type=cast(Int32Array, log_columns[0]),
        log_param_count=cast(Uint32Array, log_columns[1]),
        log_params=(
            cast(Int32Array, log_columns[2]),
            cast(Int32Array, log_columns[3]),
            cast(Int32Array, log_columns[4]),
            cast(Int32Array, log_columns[5]),
            cast(Int32Array, log_columns[6]),
            cast(Int32Array, log_columns[7]),
            cast(Int32Array, log_columns[8]),
        ),
        selection_advance_count=view.selection_advance_count[selected].copy(),
        _owner=view._owner,
    )


def _validated_rows(values: npt.ArrayLike, *, size: int) -> npt.NDArray[np.int64]:
    rows = np.asarray(values)
    if rows.ndim != 1 or rows.size <= 0:
        raise ValueError("native row selection must be a non-empty vector")
    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError("native row selection must use an integer dtype")
    normalized = rows.astype(np.int64, copy=False)
    if (
        np.any(normalized < 0)
        or np.any(normalized >= size)
        or np.unique(normalized).size != normalized.size
    ):
        raise ValueError("native row selection must be unique and in range")
    return normalized


def _select_csr(
    offsets: npt.NDArray[np.uint32],
    rows: npt.NDArray[np.int64],
    columns: Sequence[Array],
) -> tuple[
    npt.NDArray[np.uint32],
    tuple[Array, ...],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
]:
    source_starts = offsets[rows]
    source_stops = offsets[rows + 1]
    lengths = source_stops.astype(np.int64) - source_starts.astype(np.int64)
    destination_offsets = np.zeros(rows.size + 1, dtype=np.uint32)
    np.cumsum(lengths, dtype=np.uint32, out=destination_offsets[1:])
    selected_columns = tuple(
        _copy_csr_values(
            column,
            source_starts=source_starts,
            source_stops=source_stops,
            destination_offsets=destination_offsets,
        )
        for column in columns
    )
    return (
        destination_offsets,
        selected_columns,
        source_starts,
        destination_offsets[:-1],
    )


def _copy_csr_values(
    values: Array,
    *,
    source_starts: npt.NDArray[np.uint32],
    source_stops: npt.NDArray[np.uint32],
    destination_offsets: npt.NDArray[np.uint32],
) -> Array:
    shape = (int(destination_offsets[-1]), *values.shape[1:])
    result = np.empty(shape, dtype=values.dtype)
    for row, (source_start, source_stop) in enumerate(
        zip(source_starts, source_stops, strict=True)
    ):
        destination_start = int(destination_offsets[row])
        destination_stop = int(destination_offsets[row + 1])
        result[destination_start:destination_stop] = values[
            int(source_start) : int(source_stop)
        ]
    return result


def _remap_optional_visible_rows(
    values: npt.NDArray[np.uint32],
    *,
    source_starts: npt.NDArray[np.uint32],
    source_stops: npt.NDArray[np.uint32],
    destination_starts: npt.NDArray[np.uint32],
    name: str,
) -> npt.NDArray[np.uint32]:
    result = np.full(values.shape, _MISSING_ROW, dtype=np.uint32)
    present = values != _MISSING_ROW
    if np.any(present & ((values < source_starts) | (values >= source_stops))):
        raise ValueError(f"native {name} pointer is outside its visible row")
    result[present] = (
        destination_starts[present] + values[present] - source_starts[present]
    )
    return result


def _remap_attachment_parents(
    values: Array,
    *,
    attachment_offsets: npt.NDArray[np.uint32],
    visible_source_starts: npt.NDArray[np.uint32],
    visible_source_stops: npt.NDArray[np.uint32],
    visible_destination_starts: npt.NDArray[np.uint32],
) -> npt.NDArray[np.uint32]:
    result = np.asarray(values, dtype=np.uint32).copy()
    for row in range(attachment_offsets.size - 1):
        start = int(attachment_offsets[row])
        stop = int(attachment_offsets[row + 1])
        parents = result[start:stop]
        source_start = visible_source_starts[row]
        source_stop = visible_source_stops[row]
        if np.any((parents < source_start) | (parents >= source_stop)):
            raise ValueError("native attachment parent is outside its visible row")
        result[start:stop] = visible_destination_starts[row] + parents - source_start
    return result


def _concatenate_csr(
    offsets: Sequence[npt.NDArray[np.uint32]],
    columns: Sequence[Sequence[Array]],
) -> tuple[npt.NDArray[np.uint32], tuple[Array, ...]]:
    if len(offsets) != len(columns) or not offsets:
        raise ValueError("native CSR concatenation inputs must align")
    column_count = len(columns[0])
    if any(len(batch_columns) != column_count for batch_columns in columns):
        raise ValueError("native CSR concatenation columns must align")
    lengths = np.concatenate(
        tuple(
            np.diff(batch_offsets.astype(np.int64, copy=False))
            for batch_offsets in offsets
        )
    )
    total = int(np.sum(lengths, dtype=np.int64))
    if total > int(np.iinfo(np.uint32).max):
        raise OverflowError("native CSR concatenation exceeds uint32 capacity")
    destination_offsets = np.zeros(lengths.size + 1, dtype=np.uint32)
    np.cumsum(lengths, dtype=np.uint32, out=destination_offsets[1:])
    destination_columns = tuple(
        np.concatenate(
            tuple(batch_columns[column] for batch_columns in columns)
        )
        for column in range(column_count)
    )
    return destination_offsets, destination_columns


def _cumulative_value_bases(counts: Sequence[int]) -> tuple[np.uint32, ...]:
    total = 0
    bases: list[np.uint32] = []
    for count in counts:
        if count < 0 or total + count > int(np.iinfo(np.uint32).max):
            raise OverflowError("native pointer concatenation exceeds uint32 capacity")
        bases.append(np.uint32(total))
        total += count
    return tuple(bases)


def _concatenate_optional_visible_rows(
    values: Sequence[npt.NDArray[np.uint32]],
    *,
    visible_bases: Sequence[np.uint32],
) -> npt.NDArray[np.uint32]:
    adjusted: list[npt.NDArray[np.uint32]] = []
    for batch_values, base in zip(values, visible_bases, strict=True):
        copied = batch_values.copy()
        present = copied != _MISSING_ROW
        copied[present] += base
        adjusted.append(copied)
    return np.concatenate(tuple(adjusted))


def _concatenate_pairs(
    values: Sequence[tuple[Int32Array, Int32Array]],
) -> tuple[Int32Array, Int32Array]:
    return (
        cast(Int32Array, np.concatenate(tuple(value[0] for value in values))),
        cast(Int32Array, np.concatenate(tuple(value[1] for value in values))),
    )


__all__ = [
    "concatenate_native_training_views",
    "select_native_training_rows",
]
