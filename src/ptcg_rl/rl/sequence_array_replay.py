"""Context-closure plans over compact sequence fragment columns."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ptcg_rl.rl.stateless_array_replay import StatelessArrayOptimizerWindow
from ptcg_rl.rl.stateless_ppo import StatelessLogicalBatch

ArrayCoordinate = tuple[int, int, int]
SequenceKey = tuple[str, int]
TargetKey = tuple[str, int, int]


@dataclass(frozen=True)
class ArraySequenceReplayIndex:
    """Small coordinate index over immutable compact game-seat tapes."""

    by_sequence: dict[SequenceKey, dict[int, ArrayCoordinate]]
    target_keys: tuple[TargetKey, ...]


@dataclass(frozen=True)
class ArraySequenceMicrobatchPlan:
    """Packed source coordinates and target positions for temporal replay."""

    decision_part_indices: npt.NDArray[np.int32]
    decision_rows: npt.NDArray[np.int64]
    fragment_part_indices: npt.NDArray[np.int32]
    fragment_rows: npt.NDArray[np.int64]
    sequence_offsets: tuple[int, ...]
    block_indices: tuple[int, ...]
    target_row_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject any packed-coordinate alignment error."""
        row_count = int(self.decision_rows.shape[0])
        if (
            row_count <= 0
            or self.decision_part_indices.shape != (row_count,)
            or self.fragment_part_indices.shape != (row_count,)
            or self.fragment_rows.shape != (row_count,)
            or len(self.block_indices) != row_count
            or len(self.sequence_offsets) < 2
            or self.sequence_offsets[0] != 0
            or self.sequence_offsets[-1] != row_count
            or any(
                index < 0 or index >= row_count
                for index in self.target_row_indices
            )
        ):
            raise ValueError("array sequence microbatch plan is invalid")


def build_array_sequence_replay_index(
    window: StatelessArrayOptimizerWindow,
) -> ArraySequenceReplayIndex:
    """Index compact raw context without reconstructing trajectory objects."""
    by_sequence: dict[SequenceKey, dict[int, ArrayCoordinate]] = {}
    for part_index, arrays in enumerate(window.source_arrays):
        if "fragment_schema_versions" not in arrays:
            raise ValueError("array sequence replay requires schema-v2 fragments")
        fragment_offsets = np.asarray(
            arrays["fragment_decision_offsets"],
            dtype=np.int64,
        )
        for fragment_row in range(fragment_offsets.shape[0] - 1):
            start = int(fragment_offsets[fragment_row])
            stop = int(fragment_offsets[fragment_row + 1])
            first_clock = int(arrays["start_decision_indices"][fragment_row])
            clocks = np.asarray(
                arrays["decision_indices"][start:stop],
                dtype=np.int64,
            )
            expected = np.arange(
                first_clock,
                first_clock + stop - start,
                dtype=np.int64,
            )
            if not np.array_equal(clocks, expected):
                raise ValueError("array sequence fragment clock is discontinuous")
            sequence = by_sequence.setdefault(
                (
                    str(arrays["game_ids"][fragment_row]),
                    int(arrays["seats"][fragment_row]),
                ),
                {},
            )
            for decision_row, decision_index in zip(
                range(start, stop),
                clocks,
                strict=True,
            ):
                clock = int(decision_index)
                if clock in sequence:
                    raise ValueError("array sequence raw tape repeats a decision")
                sequence[clock] = (
                    part_index,
                    decision_row,
                    fragment_row,
                )

    target_keys: list[TargetKey] = []
    for part_index, decision_row in zip(
        window.decision_part_indices,
        window.decision_rows,
        strict=True,
    ):
        arrays = window.source_arrays[int(part_index)]
        fragment_row = int(arrays["decision_fragment_indices"][int(decision_row)])
        key = (
            str(arrays["game_ids"][fragment_row]),
            int(arrays["seats"][fragment_row]),
            int(arrays["decision_indices"][int(decision_row)]),
        )
        coordinate = by_sequence.get(key[:2], {}).get(key[2])
        expected_coordinate = (
            int(part_index),
            int(decision_row),
            fragment_row,
        )
        if coordinate != expected_coordinate:
            raise ValueError("array sequence target is absent from its raw tape")
        target_keys.append(key)
    return ArraySequenceReplayIndex(
        by_sequence=by_sequence,
        target_keys=tuple(target_keys),
    )


def schedule_array_sequence_logical_batches(
    window: StatelessArrayOptimizerWindow,
    replay_index: ArraySequenceReplayIndex,
    *,
    target_decisions: int | None,
    locality_chunk_decisions: int,
) -> tuple[StatelessLogicalBatch, ...]:
    """Balance optimizer steps while preserving contiguous temporal chunks."""
    if locality_chunk_decisions <= 0:
        raise ValueError("array sequence locality chunk must be positive")
    route_deck_digests = window.route_deck_digests
    if target_decisions is None:
        return (
            StatelessLogicalBatch(
                indices=tuple(range(window.decision_count)),
                deck_digests=tuple(sorted(set(route_deck_digests))),
            ),
        )
    if target_decisions <= 0:
        raise ValueError("array sequence logical batch target must be positive")
    batch_count = math.ceil(window.decision_count / target_decisions)
    targets_by_deck_and_sequence: defaultdict[
        str,
        defaultdict[SequenceKey, list[int]],
    ] = defaultdict(lambda: defaultdict(list))
    for index, (deck_digest, key) in enumerate(
        zip(
            route_deck_digests,
            replay_index.target_keys,
            strict=True,
        )
    ):
        targets_by_deck_and_sequence[deck_digest][key[:2]].append(index)
    batches: list[list[int]] = [[] for _ in range(batch_count)]
    chunk_size = min(locality_chunk_decisions, target_decisions)
    for deck_digest in sorted(targets_by_deck_and_sequence):
        for sequence_key in sorted(targets_by_deck_and_sequence[deck_digest]):
            sequence = sorted(
                targets_by_deck_and_sequence[deck_digest][sequence_key],
                key=lambda index: replay_index.target_keys[index][2],
            )
            for start in range(0, len(sequence), chunk_size):
                chunk = sequence[start : start + chunk_size]
                batch_index = min(
                    range(batch_count),
                    key=lambda index: (len(batches[index]), index),
                )
                batches[batch_index].extend(chunk)
    logical_batches = tuple(
        StatelessLogicalBatch(
            indices=tuple(sorted(indices)),
            deck_digests=tuple(
                sorted({route_deck_digests[index] for index in indices})
            ),
        )
        for indices in batches
    )
    covered = tuple(index for batch in logical_batches for index in batch.indices)
    if sorted(covered) != list(range(window.decision_count)):
        raise RuntimeError("array sequence batches did not cover every target")
    return logical_batches


def plan_array_sequence_microbatch(
    window: StatelessArrayOptimizerWindow,
    target_indices: tuple[int, ...],
    *,
    max_context_blocks: int,
    replay_index: ArraySequenceReplayIndex,
) -> ArraySequenceMicrobatchPlan:
    """Close targets over bounded contiguous compact predecessors."""
    if max_context_blocks < 2:
        raise ValueError("array sequence context must contain at least two blocks")
    if not target_indices:
        raise ValueError("array sequence microbatch requires targets")
    targets_by_sequence: dict[SequenceKey, list[int]] = {}
    target_keys: list[TargetKey] = []
    for target_index in target_indices:
        if not 0 <= target_index < window.decision_count:
            raise IndexError("array sequence target index is out of range")
        key = replay_index.target_keys[target_index]
        targets_by_sequence.setdefault(key[:2], []).append(key[2])
        target_keys.append(key)

    coordinates: list[ArrayCoordinate] = []
    offsets = [0]
    packed_blocks: list[int] = []
    packed_lookup: dict[TargetKey, int] = {}
    # Preserve first-target order so an upstream route-major microbatch remains
    # route-major after predecessor closure. Each game-seat tape is still
    # internally chronological, and target_row_indices restores caller order.
    for sequence_key, target_clocks in targets_by_sequence.items():
        first_target = min(target_clocks)
        last_target = max(target_clocks)
        first_context = max(0, first_target - max_context_blocks + 1)
        source = replay_index.by_sequence.get(sequence_key)
        if source is None:
            raise ValueError("array sequence target has no raw tape")
        for decision_index in range(first_context, last_target + 1):
            coordinate = source.get(decision_index)
            if coordinate is None:
                raise ValueError(
                    "array sequence target predecessor closure is discontinuous"
                )
            packed_lookup[(*sequence_key, decision_index)] = len(coordinates)
            coordinates.append(coordinate)
            packed_blocks.append(decision_index)
        offsets.append(len(coordinates))
    return ArraySequenceMicrobatchPlan(
        decision_part_indices=np.asarray(
            [coordinate[0] for coordinate in coordinates],
            dtype=np.int32,
        ),
        decision_rows=np.asarray(
            [coordinate[1] for coordinate in coordinates],
            dtype=np.int64,
        ),
        fragment_part_indices=np.asarray(
            [coordinate[0] for coordinate in coordinates],
            dtype=np.int32,
        ),
        fragment_rows=np.asarray(
            [coordinate[2] for coordinate in coordinates],
            dtype=np.int64,
        ),
        sequence_offsets=tuple(offsets),
        block_indices=tuple(packed_blocks),
        target_row_indices=tuple(packed_lookup[key] for key in target_keys),
    )


__all__ = [
    "ArraySequenceMicrobatchPlan",
    "ArraySequenceReplayIndex",
    "build_array_sequence_replay_index",
    "plan_array_sequence_microbatch",
    "schedule_array_sequence_logical_batches",
]
