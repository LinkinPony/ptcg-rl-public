"""Native row control, status validation, and scripted-slot lifecycle."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.engine.native_training_view import (
    concatenate_native_training_views,
    select_native_training_rows,
)
from ptcg_rl.rl.native_collection_games import NativeLiveGame
from ptcg_rl.rl.native_policy_context import (
    NativePublicContextBatch,
    concatenate_native_public_context_batches,
    select_native_public_context_rows,
)
from ptcg_rl.rl.native_scripted_policy import NativeScriptedPolicy

_READY = 1
_FINISHED = 2


@dataclass(frozen=True, slots=True)
class NativeSelectionAdvanceResult:
    """Exact engine work plus rows that must be retired at the step limit."""

    count: int
    step_limit_rows: npt.NDArray[np.int64]


def select_native_rollout_rows(
    view: NativeTrainingBatchView,
    context: NativePublicContextBatch,
    rows: npt.ArrayLike,
) -> tuple[NativeTrainingBatchView, NativePublicContextBatch]:
    """Select and rebase aligned engine and public-context rows."""
    return (
        select_native_training_rows(view, rows),
        select_native_public_context_rows(context, rows),
    )


def concatenate_native_rollout_batches(
    batches: Sequence[tuple[NativeTrainingBatchView, NativePublicContextBatch]],
) -> tuple[NativeTrainingBatchView, NativePublicContextBatch]:
    """Concatenate aligned rollout batches after slot reuse."""
    selected = tuple(batches)
    if not selected:
        raise ValueError("native rollout concatenation requires at least one batch")
    for view, context in selected:
        if view.batch_size != context.batch_size:
            raise ValueError("native rollout batch view and context do not align")
    return (
        concatenate_native_training_views(tuple(view for view, _ in selected)),
        concatenate_native_public_context_batches(
            tuple(context for _, context in selected)
        ),
    )


def require_native_ready(view: NativeTrainingBatchView) -> None:
    """Require clean non-terminal rows before action dispatch."""
    _require_clean(view)
    bad = np.flatnonzero(view.status != _READY)
    if bad.size:
        raise RuntimeError(f"native arena has non-ready live rows: {bad[:8].tolist()}")


def require_native_step_output(view: NativeTrainingBatchView) -> None:
    """Require exact clean READY/FINISHED status and result pairing."""
    _require_clean(view)
    terminal = view.result >= 0
    finished = view.status == _FINISHED
    if np.any(terminal != finished) or np.any((~finished) & (view.status != _READY)):
        raise RuntimeError("native arena result and clean slot status are inconsistent")


def native_finished_rows(
    view: NativeTrainingBatchView,
) -> npt.NDArray[np.int64]:
    """Return clean terminal rows after step-output validation."""
    require_native_step_output(view)
    return np.flatnonzero(view.status == _FINISHED)


def accumulate_native_selection_advances(
    view: NativeTrainingBatchView,
    live: Mapping[int, NativeLiveGame],
    *,
    maximum_engine_steps: int,
    submitted_actions: bool,
) -> NativeSelectionAdvanceResult:
    """Apply exact source-engine callback counts and identify isolated limits.

    A terminal reached exactly on the limit is accepted because the Python
    collector checks terminal before its step-limit branch. A non-terminal at
    the limit, or a terminal reached only after crossing it, is retired without
    invalidating unrelated rows from the same engine batch. Reset callbacks
    still count as engine work, but retirement waits for a submitted action.
    """
    if maximum_engine_steps <= 0:
        raise ValueError("native maximum engine steps must be positive")
    updates: list[tuple[NativeLiveGame, int]] = []
    step_limit_rows: list[int] = []
    total = 0
    for row, raw_slot in enumerate(view.slots):
        try:
            game = live[int(raw_slot)]
        except KeyError as error:
            raise KeyError("native advance row references an unknown slot") from error
        advances = int(view.selection_advance_count[row])
        if submitted_actions and advances <= 0:
            raise RuntimeError(
                "successful native step did not count its submitted selection"
            )
        updated = game.engine_steps + advances
        terminal = int(view.status[row]) == _FINISHED
        exceeded = (
            updated > maximum_engine_steps
            if terminal
            else updated >= maximum_engine_steps
        )
        if submitted_actions and exceeded:
            step_limit_rows.append(row)
        updates.append((game, updated))
        total += advances
    for game, updated in updates:
        game.engine_steps = updated
    return NativeSelectionAdvanceResult(
        count=total,
        step_limit_rows=np.asarray(step_limit_rows, dtype=np.int64),
    )


def reset_native_scripted(
    policies: Mapping[str, NativeScriptedPolicy],
    live: Mapping[int, NativeLiveGame],
    *,
    seeds: Mapping[int, int],
) -> None:
    """Bind one exact legacy-compatible Python MT stream per scripted slot."""
    slots_by_policy: defaultdict[str, list[int]] = defaultdict(list)
    for slot, game in live.items():
        assignment = game.assignment.curriculum
        if assignment.lane == "scripted":
            slots_by_policy[assignment.opponent_id].append(slot)
    for opponent_id, slots in slots_by_policy.items():
        try:
            policy = policies[opponent_id]
        except KeyError as error:
            raise KeyError(
                f"native scripted policy is absent: {opponent_id}"
            ) from error
        policy.reset(
            np.asarray(slots, dtype=np.uint32),
            np.asarray([seeds[slot] for slot in slots], dtype=np.uint32),
        )


def release_native_scripted(
    policies: Mapping[str, NativeScriptedPolicy],
    slots: npt.ArrayLike,
    live: Mapping[int, NativeLiveGame],
) -> None:
    """Release terminal scripted streams without touching other slots."""
    slots_by_policy: defaultdict[str, list[int]] = defaultdict(list)
    for raw_slot in np.asarray(slots):
        slot = int(raw_slot)
        game = live[slot]
        assignment = game.assignment.curriculum
        if assignment.lane == "scripted":
            slots_by_policy[assignment.opponent_id].append(slot)
    for opponent_id, selected in slots_by_policy.items():
        policies[opponent_id].release(np.asarray(selected, dtype=np.uint32))


def release_all_native_scripted(
    policies: Mapping[str, NativeScriptedPolicy],
    live: Mapping[int, NativeLiveGame],
) -> None:
    """Best-effort release used only while unwinding a failed collection."""
    if not live:
        return
    with suppress(BaseException):
        release_native_scripted(
            policies,
            np.asarray(tuple(live), dtype=np.uint32),
            live,
        )


def native_engine_seeds(start: int, count: int) -> np.ndarray:
    """Return explicit wrapping uint32 source-engine seeds."""
    maximum = int(np.iinfo(np.uint32).max) + 1
    return ((np.arange(count, dtype=np.uint64) + int(start)) % maximum).astype(
        np.uint32
    )


def _require_clean(view: NativeTrainingBatchView) -> None:
    bad = np.flatnonzero(view.error)
    if bad.size:
        rows = bad[:8]
        raise RuntimeError(
            "native arena reported row errors: "
            f"rows={rows.tolist()} errors={view.error[rows].tolist()}"
        )


__all__ = [
    "NativeSelectionAdvanceResult",
    "accumulate_native_selection_advances",
    "concatenate_native_rollout_batches",
    "native_engine_seeds",
    "native_finished_rows",
    "release_all_native_scripted",
    "release_native_scripted",
    "require_native_ready",
    "require_native_step_output",
    "reset_native_scripted",
    "select_native_rollout_rows",
]
