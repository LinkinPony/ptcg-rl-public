"""Bounded target/context planning for causal replay behavior cloning."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ptcg_rl.training.simple_stateless_pretrain_data import (
    REPLAY_SPLITS,
    LoadedPretrainingPart,
    ReplayPretrainingDatasetManifest,
    ReplaySplit,
    load_pretraining_metadata_columns,
)

_MONITOR_DOMAIN = b"ptcg-rl/temporal-pretraining-monitor/v2\x00"
_MONITOR_MINIMUM_PARTS_PER_STRATUM = 4


@dataclass(frozen=True)
class TemporalPretrainingBatchPlan:
    """One target set and its complete bounded predecessor closure."""

    target_indices: tuple[int, ...]
    context_indices: tuple[int, ...]
    sequence_offsets: tuple[int, ...]
    block_indices: tuple[int, ...]
    target_row_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Require aligned packed sequences and one position per target."""
        context_rows = len(self.context_indices)
        if (
            not self.target_indices
            or context_rows <= 0
            or len(self.block_indices) != context_rows
            or len(self.sequence_offsets) < 2
            or self.sequence_offsets[0] != 0
            or self.sequence_offsets[-1] != context_rows
            or len(self.target_row_indices) != len(self.target_indices)
            or any(
                index < 0 or index >= context_rows
                for index in self.target_row_indices
            )
        ):
            raise ValueError("temporal pretraining batch plan is invalid")


@dataclass(frozen=True)
class TemporalMonitorPlan:
    """Dataset-fingerprint-selected train rows grouped by compact part."""

    fingerprint: str
    rows_by_part: tuple[tuple[int, ...], ...]
    split: ReplaySplit = "train"

    @property
    def target_count(self) -> int:
        """Return the number of selected monitor decisions."""
        return sum(len(rows) for rows in self.rows_by_part)


@dataclass(frozen=True)
class _TemporalTargetChunk:
    rows: tuple[int, ...]
    context_blocks: int
    context_state_tokens: int
    target_options: int


def temporal_epoch_batches(
    part: LoadedPretrainingPart,
    *,
    target_decisions: int,
    target_chunk_decisions: int,
    max_context_blocks: int,
    seed: int,
    selected_targets: tuple[int, ...] | None = None,
    maximum_batch_context_blocks: int | None = None,
    maximum_batch_context_state_tokens: int | None = None,
    maximum_batch_target_options: int | None = None,
) -> tuple[TemporalPretrainingBatchPlan, ...]:
    """Cover selected decisions exactly once with locality-preserving chunks."""
    if target_decisions <= 0 or target_chunk_decisions <= 0:
        raise ValueError("temporal target budgets must be positive")
    if target_chunk_decisions > max_context_blocks:
        raise ValueError("temporal target chunk exceeds context geometry")
    if any(
        value is not None and value <= 0
        for value in (
            maximum_batch_context_blocks,
            maximum_batch_context_state_tokens,
            maximum_batch_target_options,
        )
    ):
        raise ValueError("temporal batch geometry budgets must be positive")
    target_set = (
        frozenset(range(part.example_count))
        if selected_targets is None
        else frozenset(int(index) for index in selected_targets)
    )
    if not target_set or any(
        index < 0 or index >= part.example_count for index in target_set
    ):
        raise ValueError("temporal target selection is invalid")
    chunks = tuple(
        _target_chunk_geometry(
            part,
            rows=rows,
            max_context_blocks=max_context_blocks,
            require_state_tokens=maximum_batch_context_state_tokens is not None,
            require_target_options=maximum_batch_target_options is not None,
        )
        for rows in _target_chunks(
            part,
            target_set=target_set,
            chunk_size=target_chunk_decisions,
        )
    )
    grouped: list[list[tuple[int, ...]]] = []
    pending: list[tuple[int, ...]] = []
    pending_targets = 0
    pending_context_blocks = 0
    pending_context_state_tokens = 0
    pending_target_options = 0
    for chunk in chunks:
        if (
            chunk.context_blocks
            > (
                maximum_batch_context_blocks
                if maximum_batch_context_blocks is not None
                else chunk.context_blocks
            )
            or chunk.context_state_tokens
            > (
                maximum_batch_context_state_tokens
                if maximum_batch_context_state_tokens is not None
                else chunk.context_state_tokens
            )
            or chunk.target_options
            > (
                maximum_batch_target_options
                if maximum_batch_target_options is not None
                else chunk.target_options
            )
        ):
            raise ValueError("one temporal target chunk exceeds batch geometry")
        if pending and (
            pending_targets + len(chunk.rows) > target_decisions
            or (
                maximum_batch_context_blocks is not None
                and pending_context_blocks + chunk.context_blocks
                > maximum_batch_context_blocks
            )
            or (
                maximum_batch_context_state_tokens is not None
                and pending_context_state_tokens + chunk.context_state_tokens
                > maximum_batch_context_state_tokens
            )
            or (
                maximum_batch_target_options is not None
                and pending_target_options + chunk.target_options
                > maximum_batch_target_options
            )
        ):
            grouped.append(pending)
            pending = []
            pending_targets = 0
            pending_context_blocks = 0
            pending_context_state_tokens = 0
            pending_target_options = 0
        pending.append(chunk.rows)
        pending_targets += len(chunk.rows)
        pending_context_blocks += chunk.context_blocks
        pending_context_state_tokens += chunk.context_state_tokens
        pending_target_options += chunk.target_options
    if pending:
        grouped.append(pending)
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(grouped))
    plans = tuple(
        _batch_plan(
            part,
            chunks=tuple(grouped[int(index)]),
            max_context_blocks=max_context_blocks,
        )
        for index in order
    )
    covered = tuple(index for plan in plans for index in plan.target_indices)
    if len(covered) != len(set(covered)) or frozenset(covered) != target_set:
        raise RuntimeError("temporal batches did not cover every target exactly once")
    return plans


def _target_chunk_geometry(
    part: LoadedPretrainingPart,
    *,
    rows: tuple[int, ...],
    max_context_blocks: int,
    require_state_tokens: bool,
    require_target_options: bool,
) -> _TemporalTargetChunk:
    sequence_offsets = np.asarray(part.arrays["sequence_offsets"], dtype=np.int64)
    sequence = int(
        np.searchsorted(sequence_offsets[1:], rows[0], side="right")
    )
    sequence_start = int(sequence_offsets[sequence])
    if rows[-1] >= int(sequence_offsets[sequence + 1]):
        raise ValueError("temporal target chunk crosses a sequence")
    context_start = max(sequence_start, rows[0] - max_context_blocks + 1)
    context_stop = rows[-1] + 1
    context_state_tokens = 0
    if require_state_tokens:
        if "state_offsets" not in part.arrays:
            raise ValueError("temporal batch geometry requires state offsets")
        state_offsets = np.asarray(part.arrays["state_offsets"], dtype=np.int64)
        context_state_tokens = int(
            state_offsets[context_stop] - state_offsets[context_start]
        )
    target_options = 0
    if require_target_options:
        if "option_offsets" not in part.arrays:
            raise ValueError("temporal batch geometry requires option offsets")
        option_offsets = np.asarray(part.arrays["option_offsets"], dtype=np.int64)
        target_options = int(
            option_offsets[rows[-1] + 1] - option_offsets[rows[0]]
        )
    return _TemporalTargetChunk(
        rows=rows,
        context_blocks=context_stop - context_start,
        context_state_tokens=context_state_tokens,
        target_options=target_options,
    )


def build_temporal_monitor_plan(
    dataset: ReplayPretrainingDatasetManifest,
    *,
    dataset_dir: Path,
    maximum_targets: int,
    target_chunk_decisions: int = 16,
    split: ReplaySplit = "train",
) -> TemporalMonitorPlan:
    """Select a deterministic date/route-stratified causal split monitor."""
    if not dataset.complete:
        raise ValueError("temporal monitor requires a complete dataset")
    if maximum_targets <= 0 or target_chunk_decisions <= 0:
        raise ValueError("temporal monitor target budget must be positive")
    if split not in REPLAY_SPLITS:
        raise ValueError("temporal monitor split is invalid")
    split_code = REPLAY_SPLITS.index(split)
    dataset_fingerprint = dataset.fingerprint
    strata: Counter[tuple[int, str]] = Counter()
    part_strata: list[Counter[tuple[int, str]]] = []
    for record in dataset.parts:
        arrays = load_pretraining_metadata_columns(
            dataset_dir / "parts" / record.filename,
            columns=("date_indices", "route_expert_ids", "split_codes"),
        )
        if len(arrays["episode_ids"]) != record.examples:
            raise ValueError("monitor metadata row count changed")
        counts = Counter(
            (
                int(raw_date),
                str(raw_route) or "generic",
            )
            for raw_date, raw_route, raw_split in zip(
                arrays["date_indices"],
                arrays["route_expert_ids"],
                arrays["split_codes"],
                strict=True,
            )
            if int(raw_split) == split_code
        )
        strata.update(counts)
        part_strata.append(counts)
    total = sum(strata.values())
    if total <= 0:
        raise ValueError(f"temporal monitor dataset has no {split} decisions")
    budget = min(maximum_targets, total)
    quotas = _stratified_quotas(strata, budget=budget)
    candidate_parts = _monitor_candidate_parts(
        part_strata,
        quotas=quotas,
        dataset_fingerprint=dataset_fingerprint,
    )
    chunks: dict[
        tuple[int, str],
        list[tuple[int, int, int, int]],
    ] = defaultdict(list)
    for part_index in candidate_parts:
        record = dataset.parts[part_index]
        arrays = load_pretraining_metadata_columns(
            dataset_dir / "parts" / record.filename,
            columns=(
                "player_indices",
                "date_indices",
                "route_expert_ids",
                "split_codes",
            ),
        )
        if len(arrays["episode_ids"]) != record.examples:
            raise ValueError("monitor metadata row count changed")
        row = 0
        while row < record.examples:
            if int(arrays["split_codes"][row]) != split_code:
                row += 1
                continue
            stratum = (
                int(arrays["date_indices"][row]),
                str(arrays["route_expert_ids"][row]) or "generic",
            )
            episode_id = int(arrays["episode_ids"][row])
            player_index = int(arrays["player_indices"][row])
            stop = row + 1
            while (
                stop < record.examples
                and stop - row < target_chunk_decisions
                and int(arrays["episode_ids"][stop]) == episode_id
                and int(arrays["player_indices"][stop]) == player_index
                and int(arrays["date_indices"][stop]) == stratum[0]
                and int(arrays["split_codes"][stop]) == split_code
                and (
                    str(arrays["route_expert_ids"][stop]) or "generic"
                )
                == stratum[1]
            ):
                stop += 1
            rank = _monitor_chunk_rank(
                dataset_fingerprint,
                part_index=part_index,
                row=row,
            )
            chunks[stratum].append((rank, part_index, row, stop))
            row = stop
    rows: list[list[int]] = [[] for _record in dataset.parts]
    for stratum, quota in quotas.items():
        remaining = quota
        for _rank, part_index, start, stop in sorted(chunks[stratum]):
            selected_stop = min(stop, start + remaining)
            rows[part_index].extend(range(start, selected_stop))
            remaining -= selected_stop - start
            if remaining == 0:
                break
        if remaining != 0:
            raise RuntimeError("temporal monitor chunks cannot fill quota")
    canonical = tuple(tuple(sorted(part_rows)) for part_rows in rows)
    selected = sum(len(part_rows) for part_rows in canonical)
    if selected != budget:
        raise RuntimeError("temporal monitor selection did not fill its budget")
    fingerprint = hashlib.sha256(
        _MONITOR_DOMAIN
        + dataset_fingerprint.encode("ascii")
        + b"\x00"
        + split.encode("ascii")
        + b"\x00"
        + target_chunk_decisions.to_bytes(8, "little")
        + b"\x00"
        + repr(canonical).encode("ascii")
    ).hexdigest()
    return TemporalMonitorPlan(
        fingerprint=fingerprint,
        rows_by_part=canonical,
        split=split,
    )


def _target_chunks(
    part: LoadedPretrainingPart,
    *,
    target_set: frozenset[int],
    chunk_size: int,
) -> tuple[tuple[int, ...], ...]:
    offsets = np.asarray(part.arrays["sequence_offsets"], dtype=np.int64)
    chunks: list[tuple[int, ...]] = []
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        run: list[int] = []
        for row in range(int(start), int(stop)):
            if row not in target_set:
                if run:
                    chunks.extend(_split_run(run, chunk_size=chunk_size))
                    run = []
                continue
            run.append(row)
        if run:
            chunks.extend(_split_run(run, chunk_size=chunk_size))
    return tuple(chunks)


def _split_run(rows: list[int], *, chunk_size: int) -> list[tuple[int, ...]]:
    return [
        tuple(rows[start : start + chunk_size])
        for start in range(0, len(rows), chunk_size)
    ]


def _batch_plan(
    part: LoadedPretrainingPart,
    *,
    chunks: tuple[tuple[int, ...], ...],
    max_context_blocks: int,
) -> TemporalPretrainingBatchPlan:
    decision_indices = np.asarray(part.arrays["decision_indices"], dtype=np.int64)
    sequence_offsets = np.asarray(part.arrays["sequence_offsets"], dtype=np.int64)
    row_to_sequence = np.searchsorted(
        sequence_offsets[1:],
        np.arange(part.example_count),
        side="right",
    )
    context_indices: list[int] = []
    packed_offsets = [0]
    block_indices: list[int] = []
    target_indices: list[int] = []
    target_positions: list[int] = []
    for chunk in chunks:
        sequence = int(row_to_sequence[chunk[0]])
        if any(int(row_to_sequence[row]) != sequence for row in chunk):
            raise ValueError("temporal target chunk crosses a sequence")
        sequence_start = int(sequence_offsets[sequence])
        first = chunk[0]
        last = chunk[-1]
        context_start = max(sequence_start, first - max_context_blocks + 1)
        positions: dict[int, int] = {}
        for row in range(context_start, last + 1):
            positions[row] = len(context_indices)
            context_indices.append(row)
            block_indices.append(int(decision_indices[row]))
        target_indices.extend(chunk)
        target_positions.extend(positions[row] for row in chunk)
        packed_offsets.append(len(context_indices))
    return TemporalPretrainingBatchPlan(
        target_indices=tuple(target_indices),
        context_indices=tuple(context_indices),
        sequence_offsets=tuple(packed_offsets),
        block_indices=tuple(block_indices),
        target_row_indices=tuple(target_positions),
    )


def _stratified_quotas(
    counts: Counter[tuple[int, str]],
    *,
    budget: int,
) -> dict[tuple[int, str], int]:
    total = sum(counts.values())
    if budget < len(counts):
        selected = set(
            sorted(
                counts,
                key=lambda item: (-counts[item], item),
            )[:budget]
        )
        return {key: int(key in selected) for key in counts}
    quotas = {
        key: min(count, max(1, budget * count // total))
        for key, count in counts.items()
    }
    while sum(quotas.values()) > budget:
        key = max(
            (key for key in quotas if quotas[key] > 1),
            key=lambda item: (quotas[item], item),
        )
        quotas[key] -= 1
    while sum(quotas.values()) < budget:
        candidates = [
            key for key, count in counts.items() if quotas[key] < count
        ]
        key = max(
            candidates,
            key=lambda item: (
                counts[item] / (quotas[item] + 1),
                item,
            ),
        )
        quotas[key] += 1
    return quotas


def _monitor_candidate_parts(
    part_strata: list[Counter[tuple[int, str]]],
    *,
    quotas: dict[tuple[int, str], int],
    dataset_fingerprint: str,
) -> tuple[int, ...]:
    """Choose a deterministic compact shard set that can satisfy every quota."""
    available_parts = Counter(
        stratum
        for counts in part_strata
        for stratum, count in counts.items()
        if count > 0 and quotas[stratum] > 0
    )
    required_parts = {
        stratum: min(
            _MONITOR_MINIMUM_PARTS_PER_STRATUM,
            available_parts[stratum],
        )
        for stratum, quota in quotas.items()
        if quota > 0
    }
    remaining = dict(quotas)
    contributors: Counter[tuple[int, str]] = Counter()
    selected: list[int] = []
    order = sorted(
        range(len(part_strata)),
        key=lambda part_index: (
            _monitor_part_rank(
                dataset_fingerprint,
                part_index=part_index,
            ),
            part_index,
        ),
    )
    for part_index in order:
        counts = part_strata[part_index]
        contributes = any(
            count > 0
            and (
                remaining.get(stratum, 0) > 0
                or contributors[stratum] < required_parts.get(stratum, 0)
            )
            for stratum, count in counts.items()
        )
        if not contributes:
            continue
        selected.append(part_index)
        for stratum, count in counts.items():
            if count <= 0 or quotas.get(stratum, 0) <= 0:
                continue
            remaining[stratum] = max(0, remaining[stratum] - count)
            contributors[stratum] += 1
        if all(count == 0 for count in remaining.values()) and all(
            contributors[stratum] >= required
            for stratum, required in required_parts.items()
        ):
            break
    if any(count > 0 for count in remaining.values()):
        raise RuntimeError("temporal monitor candidate parts cannot fill quotas")
    return tuple(sorted(selected))


def _monitor_part_rank(
    dataset_fingerprint: str,
    *,
    part_index: int,
) -> int:
    digest = hashlib.sha256(
        _MONITOR_DOMAIN
        + dataset_fingerprint.encode("ascii")
        + b"\x00part\x00"
        + part_index.to_bytes(8, "little")
    ).digest()
    return int.from_bytes(digest, "big")


def _monitor_chunk_rank(
    dataset_fingerprint: str,
    *,
    part_index: int,
    row: int,
) -> int:
    digest = hashlib.sha256(
        _MONITOR_DOMAIN
        + dataset_fingerprint.encode("ascii")
        + b"\x00chunk\x00"
        + part_index.to_bytes(8, "little")
        + row.to_bytes(8, "little")
    ).digest()
    return int.from_bytes(digest, "big")


__all__ = [
    "TemporalMonitorPlan",
    "TemporalPretrainingBatchPlan",
    "build_temporal_monitor_plan",
    "temporal_epoch_batches",
]
