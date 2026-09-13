"""Context-closure plans for current-weight temporal PPO reconstruction."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from ptcg_rl.rl.stateless_ppo import StatelessLogicalBatch
from ptcg_rl.rl.stateless_replay import (
    SequenceContextDecision,
    StatelessOptimizerWindow,
)


@dataclass(frozen=True)
class SequenceReplayIndex:
    """One update-local lookup over immutable raw game-seat tapes."""

    by_sequence: dict[
        tuple[str, int],
        dict[int, SequenceContextDecision],
    ]


@dataclass(frozen=True)
class SequenceMicrobatchPlan:
    """Complete raw context rows plus loss-row positions in caller order."""

    rows: tuple[SequenceContextDecision, ...]
    sequence_offsets: tuple[int, ...]
    block_indices: tuple[int, ...]
    target_row_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate packed block and target coordinates."""
        if (
            not self.rows
            or len(self.block_indices) != len(self.rows)
            or len(self.sequence_offsets) < 2
            or self.sequence_offsets[0] != 0
            or self.sequence_offsets[-1] != len(self.rows)
        ):
            raise ValueError("sequence microbatch plan is structurally invalid")
        if any(
            index < 0 or index >= len(self.rows)
            for index in self.target_row_indices
        ):
            raise ValueError("sequence target row index is out of range")


def build_sequence_replay_index(
    window: StatelessOptimizerWindow,
) -> SequenceReplayIndex:
    """Index raw context once before repeated temporal microbatch planning."""
    if not window.sequence_context:
        raise ValueError("optimizer window has no raw sequence context")
    by_sequence: dict[
        tuple[str, int],
        dict[int, SequenceContextDecision],
    ] = {}
    for row in window.sequence_context:
        sequence = by_sequence.setdefault((row.game_id, row.seat), {})
        decision_index = row.decision.decision_index
        if decision_index in sequence:
            raise ValueError("sequence raw tape repeats a decision")
        sequence[decision_index] = row
    return SequenceReplayIndex(by_sequence=by_sequence)


def schedule_sequence_logical_batches(
    window: StatelessOptimizerWindow,
    *,
    target_decisions: int | None,
    locality_chunk_decisions: int,
) -> tuple[StatelessLogicalBatch, ...]:
    """Balance optimizer steps without round-robin scattering temporal runs."""
    if locality_chunk_decisions <= 0:
        raise ValueError("sequence locality chunk must be positive")
    if target_decisions is None:
        return (
            StatelessLogicalBatch(
                indices=tuple(range(len(window.targets))),
                deck_digests=tuple(sorted(set(window.deck_digests))),
            ),
        )
    if target_decisions <= 0:
        raise ValueError("sequence logical batch target must be positive")
    batch_count = math.ceil(len(window.targets) / target_decisions)
    targets_by_deck_and_sequence: defaultdict[
        str,
        defaultdict[tuple[str, int], list[int]],
    ] = defaultdict(lambda: defaultdict(list))
    for index, target in enumerate(window.targets):
        targets_by_deck_and_sequence[target.deck_digest][
            (target.game_id, target.seat)
        ].append(index)
    batches: list[list[int]] = [[] for _ in range(batch_count)]
    chunk_size = min(locality_chunk_decisions, target_decisions)
    for deck_digest in sorted(targets_by_deck_and_sequence):
        targets_by_sequence = targets_by_deck_and_sequence[deck_digest]
        for sequence_key in sorted(targets_by_sequence):
            sequence = sorted(
                targets_by_sequence[sequence_key],
                key=lambda index: window.targets[index].decision.decision_index,
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
                sorted({window.deck_digests[index] for index in indices})
            ),
        )
        for indices in batches
    )
    covered = tuple(
        index for batch in logical_batches for index in batch.indices
    )
    if sorted(covered) != list(range(len(window.targets))):
        raise RuntimeError("sequence logical batches did not cover every target")
    return logical_batches


def plan_sequence_microbatch(
    window: StatelessOptimizerWindow,
    target_indices: tuple[int, ...],
    *,
    max_context_blocks: int,
    replay_index: SequenceReplayIndex | None = None,
) -> SequenceMicrobatchPlan:
    """Close each target over its bounded contiguous game-seat predecessors."""
    if max_context_blocks < 2:
        raise ValueError("sequence context window must contain at least two blocks")
    if not target_indices:
        raise ValueError("sequence microbatch requires loss targets")
    indexed = (
        build_sequence_replay_index(window)
        if replay_index is None
        else replay_index
    )
    targets_by_sequence: dict[tuple[str, int], list[int]] = {}
    target_keys: list[tuple[str, int, int]] = []
    for target_index in target_indices:
        if not 0 <= target_index < len(window.targets):
            raise IndexError("sequence PPO target index is out of range")
        target = window.targets[target_index]
        key = (target.game_id, target.seat)
        decision_index = target.decision.decision_index
        targets_by_sequence.setdefault(key, []).append(decision_index)
        target_keys.append((*key, decision_index))

    rows: list[SequenceContextDecision] = []
    offsets = [0]
    packed_blocks: list[int] = []
    packed_lookup: dict[tuple[str, int, int], int] = {}
    # Preserve first-target order so an upstream route-major microbatch remains
    # route-major after predecessor closure. Each game-seat tape is still
    # internally chronological, and target_row_indices restores caller order.
    for sequence_key, target_clocks in targets_by_sequence.items():
        first_target = min(target_clocks)
        last_target = max(target_clocks)
        first_context = max(0, first_target - max_context_blocks + 1)
        source = indexed.by_sequence.get(sequence_key)
        if source is None:
            raise ValueError("sequence target has no game-seat raw tape")
        sequence_rows: list[SequenceContextDecision] = []
        for decision_index in range(first_context, last_target + 1):
            context_row = source.get(decision_index)
            if context_row is None:
                raise ValueError(
                    "sequence target predecessor closure is discontinuous"
                )
            packed_lookup[(*sequence_key, decision_index)] = (
                len(rows) + len(sequence_rows)
            )
            sequence_rows.append(context_row)
            packed_blocks.append(decision_index)
        rows.extend(sequence_rows)
        offsets.append(len(rows))
    return SequenceMicrobatchPlan(
        rows=tuple(rows),
        sequence_offsets=tuple(offsets),
        block_indices=tuple(packed_blocks),
        target_row_indices=tuple(packed_lookup[key] for key in target_keys),
    )


__all__ = [
    "SequenceMicrobatchPlan",
    "SequenceReplayIndex",
    "build_sequence_replay_index",
    "plan_sequence_microbatch",
    "schedule_sequence_logical_batches",
]
