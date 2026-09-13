"""Lifecycle for one full-capacity mixed native source-engine arena."""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ptcg_rl.belief.public_catalog import PublicDeckCatalog
from ptcg_rl.engine.native_rollout import NativeRolloutEncoder
from ptcg_rl.engine.native_training import (
    NativeTrainingBatchView,
    NativeTrainingLane,
    NativeTrainingOutputBuffer,
)
from ptcg_rl.engine.native_training_view import (
    concatenate_native_training_views,
    select_native_training_rows,
)
from ptcg_rl.rl.native_collection_actions import NativeActionAccumulator
from ptcg_rl.rl.native_collection_control import (
    accumulate_native_selection_advances,
    native_finished_rows,
    release_all_native_scripted,
    release_native_scripted,
    require_native_ready,
    reset_native_scripted,
)
from ptcg_rl.rl.native_collection_delivery import NativePartDelivery
from ptcg_rl.rl.native_collection_games import (
    NativeLiveGame,
    native_candidate_score,
    native_deck_rows,
    native_deck_rows_for_games,
)
from ptcg_rl.rl.native_route_scheduler import (
    NativeFrozenBatchScheduler,
    NativeRouteKey,
    NativeRouteKind,
    native_assignment_engine_seeds,
    native_assignment_scripted_seeds,
)
from ptcg_rl.rl.native_scripted_policy import NativeScriptedPolicy
from ptcg_rl.rl.native_sequence_runtime import NativePendingSequenceDecision
from ptcg_rl.rl.native_trajectory_page import NativeTrajectoryPage
from ptcg_rl.rl.stateless_collection import StatelessGameOutcome

if TYPE_CHECKING:
    from ptcg_rl.rl.native_scripted_mixed75 import NativeScriptedActionBatch
    from ptcg_rl.rl.native_stateless_collection import NativeStatelessCollector

Int64Array = npt.NDArray[np.int64]
UInt32Array = npt.NDArray[np.uint32]
Int32Array = npt.NDArray[np.int32]


@dataclass(slots=True)
class NativeRouteArena:
    """One live engine lane containing the lease-ordered opponent mixture."""

    assignment_indices: frozenset[int]
    capacity: int
    pending: deque[tuple[int, NativeLiveGame]]
    live: dict[int, NativeLiveGame]
    live_assignment_indices: dict[int, int]
    completed_assignment_indices: set[int]
    released_assignment_indices: set[int]
    lane: NativeTrainingLane
    encoder: NativeRolloutEncoder
    route_encoders: Mapping[str, NativeRolloutEncoder]
    page: NativeTrajectoryPage
    buffers: tuple[NativeTrainingOutputBuffer, NativeTrainingOutputBuffer]
    view: NativeTrainingBatchView
    scripted_policies: Mapping[str, NativeScriptedPolicy]
    batch_index: int = 0
    window_draining: bool = False

    def encoder_for_artifact(
        self,
        artifact_sha256: str | None,
    ) -> NativeRolloutEncoder:
        """Resolve the input-contract encoder owned by one policy route."""
        if artifact_sha256 is None:
            return self.encoder
        return self.route_encoders.get(artifact_sha256, self.encoder)

    @property
    def encoders(self) -> tuple[NativeRolloutEncoder, ...]:
        """Return every distinct encoder synchronized with this arena."""
        values = (self.encoder, *self.route_encoders.values())
        return tuple(dict.fromkeys(values))


def _owned_encoders(arena: NativeRouteArena) -> tuple[NativeRolloutEncoder, ...]:
    """Return route encoders while preserving synthetic arena compatibility."""
    values = getattr(arena, "encoders", None)
    return tuple(values) if values is not None else (arena.encoder,)


def _retains_trajectories(collector: NativeStatelessCollector) -> bool:
    """Treat collectors predating evaluation mode as trajectory retaining."""
    return bool(getattr(collector, "retain_trajectories", True))


@dataclass(frozen=True, slots=True)
class _NativeArenaResources:
    """One reusable native lane and its fixed-capacity host buffers."""

    lane: NativeTrainingLane
    encoder: NativeRolloutEncoder
    route_encoders: Mapping[str, NativeRolloutEncoder]
    buffers: tuple[NativeTrainingOutputBuffer, NativeTrainingOutputBuffer]


class NativeArenaResourcePool:
    """Keep native engine lanes and encoders alive across collection shards."""

    def __init__(
        self,
        *,
        resource_count: int,
        lane_capacity: int,
        options_per_lane: int,
        library_path: Path | None,
        catalog: PublicDeckCatalog,
        input_contract_fingerprint: str,
        route_input_contracts: Mapping[
            str,
            tuple[PublicDeckCatalog, str],
        ]
        | None = None,
    ) -> None:
        if resource_count <= 0 or lane_capacity <= 0 or options_per_lane <= 0:
            raise ValueError("native arena resource capacities must be positive")
        self.resource_count = int(resource_count)
        self.lane_capacity = int(lane_capacity)
        self._stack = ExitStack()
        self._resources: list[_NativeArenaResources] = []
        for _index in range(resource_count):
            lane = self._stack.enter_context(
                NativeTrainingLane(
                    lane_capacity,
                    library_path=library_path,
                )
            )
            encoder = self._stack.enter_context(
                NativeRolloutEncoder(
                    slot_capacity=lane_capacity,
                    library=lane.library,
                    catalog=catalog,
                    input_contract_fingerprint=input_contract_fingerprint,
                )
            )
            route_encoders = {
                artifact_sha256: self._stack.enter_context(
                    NativeRolloutEncoder(
                        slot_capacity=lane_capacity,
                        library=lane.library,
                        catalog=route_catalog,
                        input_contract_fingerprint=route_contract,
                    )
                )
                for artifact_sha256, (
                    route_catalog,
                    route_contract,
                ) in (route_input_contracts or {}).items()
            }
            buffers = tuple(
                NativeTrainingOutputBuffer(
                    slot_capacity=lane_capacity,
                    option_capacity=lane_capacity * options_per_lane,
                )
                for _buffer_index in range(2)
            )
            self._resources.append(
                _NativeArenaResources(
                    lane=lane,
                    encoder=encoder,
                    route_encoders=route_encoders,
                    buffers=(buffers[0], buffers[1]),
                )
            )
        self._available = list(range(resource_count - 1, -1, -1))
        self._lock = threading.Lock()
        self._closed = False

    def acquire(
        self,
        capacity: int,
        *,
        owner_stack: ExitStack,
    ) -> _NativeArenaResources:
        """Lease one clean fixed-capacity lane until the owner stack exits."""
        if not 1 <= capacity <= self.lane_capacity:
            raise ValueError("native arena lease exceeds its persistent capacity")
        with self._lock:
            if self._closed:
                raise RuntimeError("native arena resource pool is closed")
            if not self._available:
                raise RuntimeError("native arena resource pool is exhausted")
            index = self._available.pop()
        owner_stack.callback(self._release, index)
        return self._resources[index]

    def close(self) -> None:
        """Release every native lane after all shard leases have returned."""
        with self._lock:
            if self._closed:
                return
            if len(self._available) != len(self._resources):
                raise RuntimeError("native arena resources are still leased")
            self._closed = True
        self._stack.close()

    def _release(self, index: int) -> None:
        with self._lock:
            if index in self._available:
                raise RuntimeError("native arena resource was released twice")
            self._available.append(index)


@dataclass(slots=True)
class NativeArenaDispatch:
    """Ready rows and their single-use action accumulator."""

    arena: NativeRouteArena
    actions: NativeActionAccumulator
    current_rows: Int64Array
    past_rows: dict[str, Int64Array]
    historical_rows: dict[str, Int64Array]
    scripted_rows: dict[str, Int64Array]
    parked_past_rows: dict[str, Int64Array] = field(default_factory=dict)
    parked_historical_rows: dict[str, Int64Array] = field(default_factory=dict)
    pending_sequence: list[NativePendingSequenceDecision] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class NativeRouteArenaEngineStep:
    """Immutable private-lane input safe to execute on an engine worker."""

    lane: NativeTrainingLane
    slots: UInt32Array
    action_offsets: UInt32Array
    action_choices: Int32Array
    output: NativeTrainingOutputBuffer


@dataclass(frozen=True, slots=True)
class NativePreparedArenaAdvance:
    """Coordinator-owned context surrounding one private-lane engine step."""

    dispatch: NativeArenaDispatch
    engine_step: NativeRouteArenaEngineStep
    submitted_players: Int32Array
    acting_rows: Int64Array | None
    parked_view: NativeTrainingBatchView | None


@dataclass(frozen=True, slots=True)
class NativePreparedScriptedAction:
    """One CPU-scripted result awaiting owner-side action publication."""

    dispatch: NativeArenaDispatch
    rows: Int64Array
    actions: NativeScriptedActionBatch
    work_seconds: float


@dataclass(frozen=True, slots=True)
class NativePreparedScriptedActions:
    """Immutable scripted-policy work produced away from the GPU owner."""

    batches: tuple[NativePreparedScriptedAction, ...]
    started_at: float
    finished_at: float

    @property
    def rows(self) -> int:
        """Return the number of scripted decisions in this preparation."""
        return sum(int(batch.rows.size) for batch in self.batches)


def prepare_arena_dispatches(
    collector: NativeStatelessCollector,
    arenas: Sequence[NativeRouteArena],
    *,
    counters: Counter[str],
    frozen_scheduler: NativeFrozenBatchScheduler | None = None,
) -> tuple[NativeArenaDispatch, ...]:
    """Create complete action accumulators for all active route arenas."""
    staged: list[
        tuple[
            NativeRouteArena,
            NativeActionAccumulator,
            Int64Array,
            Int64Array,
            dict[str, Int64Array],
            dict[str, Int64Array],
            dict[str, Int64Array],
        ]
    ] = []
    for arena in arenas:
        actions = NativeActionAccumulator(arena.view)
        forced_rows = actions.fill_forced()
        counters["forced_rows"] += int(forced_rows.size)
        current, past, historical, scripted = collector._dispatch_rows(
            arena.view,
            arena.live,
            forced_rows,
        )
        staged.append(
            (arena, actions, forced_rows, current, past, historical, scripted)
        )
    released = _plan_frozen_releases(
        staged,
        counters=counters,
        frozen_scheduler=frozen_scheduler,
    )
    dispatches: list[NativeArenaDispatch] = []
    for arena, actions, forced_rows, current, past, historical, scripted in staged:
        served_past, parked_past = _split_frozen_routes(
            past,
            kind="past_self",
            released=released,
        )
        served_historical, parked_historical = _split_frozen_routes(
            historical,
            kind="historical",
            released=released,
        )
        dispatch = NativeArenaDispatch(
            arena=arena,
            actions=actions,
            current_rows=current,
            past_rows=served_past,
            historical_rows=served_historical,
            scripted_rows=scripted,
            parked_past_rows=parked_past,
            parked_historical_rows=parked_historical,
        )
        _require_complete_dispatch(
            collector,
            dispatch,
            forced_rows=forced_rows,
        )
        dispatches.append(dispatch)
    return tuple(dispatches)


def _plan_frozen_releases(
    staged: Sequence[
        tuple[
            NativeRouteArena,
            NativeActionAccumulator,
            Int64Array,
            Int64Array,
            dict[str, Int64Array],
            dict[str, Int64Array],
            dict[str, Int64Array],
        ]
    ],
    *,
    counters: Counter[str],
    frozen_scheduler: NativeFrozenBatchScheduler | None,
) -> frozenset[NativeRouteKey] | None:
    """Return the frozen routes serving this wave, or None to serve all."""
    if frozen_scheduler is None:
        return None
    ready: Counter[NativeRouteKey] = Counter()
    other_work_rows = 0
    draining = False
    for arena, _actions, forced, current, past, historical, scripted in staged:
        draining = draining or arena.window_draining
        other_work_rows += int(forced.size) + int(current.size)
        other_work_rows += sum(int(rows.size) for rows in scripted.values())
        for artifact_sha256, rows in past.items():
            ready[
                NativeRouteKey(kind="past_self", artifact_sha256=artifact_sha256)
            ] += int(rows.size)
        for artifact_sha256, rows in historical.items():
            ready[
                NativeRouteKey(kind="historical", artifact_sha256=artifact_sha256)
            ] += int(rows.size)
    plan = frozen_scheduler.plan_wave(
        ready,
        draining=draining,
        other_work_rows=other_work_rows,
    )
    counters["frozen_pending_row_waves"] += plan.parked_rows
    counters["frozen_threshold_releases"] += plan.threshold_releases
    counters["frozen_deadline_releases"] += plan.deadline_releases
    counters["frozen_forced_releases"] += plan.forced_releases
    return plan.released


def _split_frozen_routes(
    grouped: dict[str, Int64Array],
    *,
    kind: NativeRouteKind,
    released: frozenset[NativeRouteKey] | None,
) -> tuple[dict[str, Int64Array], dict[str, Int64Array]]:
    """Partition one frozen route family into served and parked rows."""
    if released is None:
        return grouped, {}
    served: dict[str, Int64Array] = {}
    parked: dict[str, Int64Array] = {}
    for artifact_sha256, rows in grouped.items():
        route = NativeRouteKey(kind=kind, artifact_sha256=artifact_sha256)
        if route in released:
            served[artifact_sha256] = rows
        else:
            parked[artifact_sha256] = rows
    return served, parked


def prepare_scripted_actions(
    _collector: NativeStatelessCollector,
    dispatches: Sequence[NativeArenaDispatch],
) -> NativePreparedScriptedActions:
    """Run scripted policies without mutating owner-owned accumulators."""
    started_at = time.perf_counter()
    prepared: list[NativePreparedScriptedAction] = []
    for dispatch in dispatches:
        for opponent_id, rows in dispatch.scripted_rows.items():
            policy_started = time.perf_counter()
            try:
                policy = dispatch.arena.scripted_policies[opponent_id]
            except KeyError as error:
                raise KeyError(
                    f"native scripted policy is absent: {opponent_id}"
                ) from error
            actions = policy.act_batch(
                dispatch.arena.view,
                rows,
                lane=dispatch.arena.lane,
            )
            prepared.append(
                NativePreparedScriptedAction(
                    dispatch=dispatch,
                    rows=rows,
                    actions=actions,
                    work_seconds=time.perf_counter() - policy_started,
                )
            )
    return NativePreparedScriptedActions(
        batches=tuple(prepared),
        started_at=started_at,
        finished_at=time.perf_counter(),
    )


def apply_prepared_scripted_actions(
    prepared: NativePreparedScriptedActions,
    *,
    timings: defaultdict[str, float],
) -> None:
    """Publish prepared scripted actions on the sole coordinator thread."""
    for batch in prepared.batches:
        batch.dispatch.actions.fill_scripted(batch.rows, batch.actions)
        timings["scripted"] += batch.work_seconds


def serve_scripted(
    collector: NativeStatelessCollector,
    dispatches: Sequence[NativeArenaDispatch],
    *,
    timings: defaultdict[str, float],
) -> None:
    """Synchronously fill all CPU-scripted rows for non-prefetched paths."""
    apply_prepared_scripted_actions(
        prepare_scripted_actions(collector, dispatches),
        timings=timings,
    )


def prepare_route_arena_advance(
    dispatch: NativeArenaDispatch,
    *,
    timings: defaultdict[str, float],
) -> NativePreparedArenaAdvance | None:
    """Prepare immutable engine input without advancing shared arena state."""
    arena = dispatch.arena
    input_started = time.perf_counter()
    parked_rows = _parked_dispatch_rows(dispatch)
    parked_view: NativeTrainingBatchView | None = None
    acting_rows: Int64Array | None = None
    if parked_rows.size:
        # Copy the waiting rows out of the ping-pong output buffers before the
        # subset step and refill reuse them. Parked engine slots stay frozen at
        # their pending decision, so the copied view stays valid across waves.
        parked_view = select_native_training_rows(arena.view, parked_rows)
        acting_mask = np.ones(arena.view.batch_size, dtype=np.bool_)
        acting_mask[parked_rows] = False
        acting_rows = np.flatnonzero(acting_mask)
        if acting_rows.size == 0:
            # Every live row waits on a frozen release; hold the whole arena.
            arena.view = parked_view
            timings["input"] += time.perf_counter() - input_started
            return None
    action_offsets, action_choices = (
        dispatch.actions.finish()
        if acting_rows is None
        else dispatch.actions.finish(rows=acting_rows)
    )
    step_slots = (
        arena.view.slots.copy()
        if acting_rows is None
        else arena.view.slots[acting_rows].copy()
    )
    submitted_players = (
        arena.view.select_player.copy()
        if acting_rows is None
        else arena.view.select_player[acting_rows].copy()
    )
    if np.any((submitted_players != 0) & (submitted_players != 1)):
        raise RuntimeError("native submitted action has an invalid acting player")
    timings["input"] += time.perf_counter() - input_started
    return NativePreparedArenaAdvance(
        dispatch=dispatch,
        engine_step=NativeRouteArenaEngineStep(
            lane=arena.lane,
            slots=step_slots,
            action_offsets=action_offsets,
            action_choices=action_choices,
            output=arena.buffers[(arena.batch_index + 1) % 2],
        ),
        submitted_players=submitted_players,
        acting_rows=acting_rows,
        parked_view=parked_view,
    )


def run_route_arena_engine_step(
    engine_step: NativeRouteArenaEngineStep,
) -> NativeTrainingBatchView:
    """Run only one arena-private native step; safe for an engine worker."""
    return engine_step.lane.step(
        engine_step.slots,
        engine_step.action_offsets,
        engine_step.action_choices,
        output=engine_step.output,
    )


def finish_route_arena_advance(
    collector: NativeStatelessCollector,
    prepared: NativePreparedArenaAdvance,
    next_view: NativeTrainingBatchView,
    *,
    delivery: NativePartDelivery,
    outcomes: list[StatelessGameOutcome],
    lane_scores: defaultdict[str, list[float]],
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> None:
    """Finalize one native step on the coordinator and update shared state."""
    dispatch = prepared.dispatch
    arena = dispatch.arena
    _resolve_pending_sequence(
        dispatch,
        next_view,
        acting_rows=prepared.acting_rows,
    )
    finished_rows = native_finished_rows(next_view)
    advances = accumulate_native_selection_advances(
        next_view,
        arena.live,
        maximum_engine_steps=collector.maximum_engine_steps,
        submitted_actions=True,
    )
    counters["engine_steps"] += advances.count
    step_limit_rows = advances.step_limit_rows
    if prepared.submitted_players.shape != (next_view.batch_size,):
        raise RuntimeError("native submitted players do not align with step output")
    terminal_rows = np.setdiff1d(
        finished_rows,
        step_limit_rows,
        assume_unique=True,
    )
    input_started = time.perf_counter()
    for encoder in _owned_encoders(arena):
        encoder.consume_step(next_view)
    timings["input"] += time.perf_counter() - input_started

    if terminal_rows.size:
        _finish_terminals(
            collector,
            arena,
            next_view,
            terminal_rows,
            delivery=delivery,
            outcomes=outcomes,
            lane_scores=lane_scores,
            timings=timings,
        )
    if step_limit_rows.size:
        _finish_step_limits(
            collector,
            arena,
            next_view,
            step_limit_rows,
            prepared.submitted_players[step_limit_rows],
            delivery=delivery,
            outcomes=outcomes,
            lane_scores=lane_scores,
            timings=timings,
            counters=counters,
        )
    already_retired = np.union1d(terminal_rows, step_limit_rows)
    immediate_cutoff = bool(
        getattr(collector, "immediate_whole_game_cutoff_on_drain", False)
        and getattr(arena, "window_draining", False)
    )
    if immediate_cutoff:
        window_cutoff_rows = np.setdiff1d(
            np.arange(next_view.batch_size, dtype=np.int64),
            already_retired,
            assume_unique=True,
        )
    else:
        window_cutoff_rows = _window_cutoff_rows(
            collector,
            arena,
            next_view,
            excluded_rows=already_retired,
        )
    if window_cutoff_rows.size:
        finish_cutoffs = (
            _finish_immediate_window_cutoffs
            if immediate_cutoff
            else _finish_window_cutoffs
        )
        finish_cutoffs(
            collector,
            arena,
            next_view,
            window_cutoff_rows,
            delivery=delivery,
            outcomes=outcomes,
            timings=timings,
            counters=counters,
        )
    retired_rows = np.union1d(already_retired, window_cutoff_rows)
    retired_slots = next_view.slots[retired_rows].copy()

    if not retired_rows.size and prepared.parked_view is None:
        arena.view = next_view
        require_native_ready(arena.view)
        arena.batch_index += 1
        return

    rollout_batches: list[NativeTrainingBatchView] = []
    remaining = np.ones(next_view.batch_size, dtype=np.bool_)
    remaining[retired_rows] = False
    remaining_rows = np.flatnonzero(remaining)
    if remaining_rows.size:
        rollout_batches.append(select_native_training_rows(next_view, remaining_rows))
    replacement = None
    if not getattr(arena, "window_draining", False):
        replacement = _refill_route(
            collector,
            arena,
            retired_slots,
            timings=timings,
            counters=counters,
        )
    if replacement is not None:
        rollout_batches.append(replacement)
    if prepared.parked_view is not None:
        rollout_batches.append(prepared.parked_view)
    if not arena.live:
        return
    arena.view = concatenate_native_training_views(rollout_batches)
    require_native_ready(arena.view)
    arena.batch_index += 1


def abort_route_arena_advance(prepared: NativePreparedArenaAdvance) -> None:
    """Rollback provisional sequence rows after an engine-worker failure."""
    _abort_pending_sequence(prepared.dispatch)


def abort_route_arena_dispatch(dispatch: NativeArenaDispatch) -> None:
    """Rollback provisional sequence rows before engine ownership transfers."""
    _abort_pending_sequence(dispatch)


def advance_route_arena(
    collector: NativeStatelessCollector,
    dispatch: NativeArenaDispatch,
    *,
    delivery: NativePartDelivery,
    outcomes: list[StatelessGameOutcome],
    lane_scores: defaultdict[str, list[float]],
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> None:
    """Synchronously prepare, run, and finish one mixed native arena step."""
    prepared = prepare_route_arena_advance(
        dispatch,
        timings=timings,
    )
    if prepared is None:
        return
    engine_started = time.perf_counter()
    try:
        next_view = run_route_arena_engine_step(prepared.engine_step)
    except BaseException:
        abort_route_arena_advance(prepared)
        raise
    timings["engine"] += time.perf_counter() - engine_started
    finish_route_arena_advance(
        collector,
        prepared,
        next_view,
        delivery=delivery,
        outcomes=outcomes,
        lane_scores=lane_scores,
        timings=timings,
        counters=counters,
    )


def start_native_arena(
    collector: NativeStatelessCollector,
    prepared: Mapping[int, NativeLiveGame],
    *,
    capacity: int,
    lane_stack: ExitStack,
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> NativeRouteArena:
    """Start one engine lane over every native assignment in lease order."""
    assignment_order = tuple(prepared)
    if assignment_order != tuple(range(len(prepared))):
        raise ValueError("native mixed arena assignments are not lease ordered")
    if not 1 <= capacity <= len(assignment_order):
        raise ValueError("native mixed arena capacity is invalid")
    pending = deque((index, prepared[index]) for index in assignment_order)
    live: dict[int, NativeLiveGame] = {}
    live_assignment_indices: dict[int, int] = {}
    for slot in range(capacity):
        original_index, game = pending.popleft()
        game.slot = slot
        live[slot] = game
        live_assignment_indices[slot] = original_index
    decks = native_deck_rows(live)
    page = NativeTrajectoryPage(
        collector.identity,
        fragments_per_part=collector.fragments_per_part,
    )
    if _retains_trajectories(collector):
        page.register_games(
            tuple(
                trajectory
                for game in live.values()
                for trajectory in game.trajectory_games(
                    mirror_bilateral=collector.mirror_bilateral_trajectories
                )
            )
        )
    if collector.arena_resource_pool is None:
        buffers = (
            NativeTrainingOutputBuffer(
                slot_capacity=capacity,
                option_capacity=capacity * collector.options_per_lane,
            ),
            NativeTrainingOutputBuffer(
                slot_capacity=capacity,
                option_capacity=capacity * collector.options_per_lane,
            ),
        )
        lane = lane_stack.enter_context(
            NativeTrainingLane(
                capacity,
                library_path=collector.library_path,
            )
        )
        encoder = lane_stack.enter_context(
            NativeRolloutEncoder(
                slot_capacity=capacity,
                library=lane.library,
                catalog=collector.catalog,
                input_contract_fingerprint=(
                    collector.identity.input_contract_fingerprint
                ),
            )
        )
    else:
        resources = collector.arena_resource_pool.acquire(
            capacity,
            owner_stack=lane_stack,
        )
        lane = resources.lane
        encoder = resources.encoder
        buffers = resources.buffers
    scripted_ids = {
        prepared[index].assignment.curriculum.opponent_id
        for index in assignment_order
        if prepared[index].assignment.curriculum.lane == "scripted"
    }
    scripted_policies = {
        opponent_id: collector.scripted_policies[opponent_id].clone_empty()
        for opponent_id in scripted_ids
    }
    try:
        _reset_arena_scripted(
            scripted_policies,
            live,
            seed=collector.seed,
        )
        engine_started = time.perf_counter()
        view = lane.reset(
            decks,
            native_assignment_engine_seeds(
                collector.seed,
                tuple(game.assignment for game in live.values()),
            ),
            slots=np.arange(capacity, dtype=np.uint32),
            output=buffers[0],
        )
        timings["engine"] += time.perf_counter() - engine_started
        require_native_ready(view)
        advances = accumulate_native_selection_advances(
            view,
            live,
            maximum_engine_steps=collector.maximum_engine_steps,
            submitted_actions=False,
        )
        counters["engine_steps"] += advances.count
        input_started = time.perf_counter()
        if collector.arena_resource_pool is None:
            route_encoders: Mapping[str, NativeRolloutEncoder] = {
                artifact_sha256: lane_stack.enter_context(
                    NativeRolloutEncoder(
                        slot_capacity=capacity,
                        library=lane.library,
                        catalog=route_catalog,
                        input_contract_fingerprint=route_contract,
                    )
                )
                for artifact_sha256, (
                    route_catalog,
                    route_contract,
                ) in collector.route_input_contracts.items()
            }
        else:
            route_encoders = resources.route_encoders
        for owned_encoder in (encoder, *route_encoders.values()):
            owned_encoder.consume_reset(view, decks)
        timings["input"] += time.perf_counter() - input_started
    except BaseException:
        release_all_native_scripted(scripted_policies, live)
        raise
    arena = NativeRouteArena(
        assignment_indices=frozenset(assignment_order),
        capacity=capacity,
        pending=pending,
        live=live,
        live_assignment_indices=live_assignment_indices,
        completed_assignment_indices=set(),
        released_assignment_indices=set(),
        lane=lane,
        encoder=encoder,
        route_encoders=route_encoders,
        page=page,
        buffers=buffers,
        view=view,
        scripted_policies=scripted_policies,
    )
    counters["games_started"] += len(live)
    _require_assignment_conservation(arena)
    return arena


def _finish_terminals(
    collector: NativeStatelessCollector,
    arena: NativeRouteArena,
    view: NativeTrainingBatchView,
    terminal_rows: Int64Array,
    *,
    delivery: NativePartDelivery,
    outcomes: list[StatelessGameOutcome],
    lane_scores: defaultdict[str, list[float]],
    timings: defaultdict[str, float],
) -> None:
    terminal_slots = view.slots[terminal_rows]
    trajectory_slots: list[int] = []
    trajectory_seats: list[int] = []
    trajectory_rewards: list[float] = []
    for row in terminal_rows:
        game = arena.live[int(view.slots[row])]
        result = int(view.result[row])
        score = native_candidate_score(result, game.candidate_seat)
        lane_scores[game.assignment.curriculum.lane].append(score)
        outcomes.append(game.outcome(score=score))
        trajectories = (
            game.trajectory_games(
                mirror_bilateral=collector.mirror_bilateral_trajectories
            )
            if _retains_trajectories(collector)
            else ()
        )
        for trajectory in trajectories:
            trajectory_slots.append(trajectory.slot)
            trajectory_seats.append(trajectory.candidate_seat)
            perspective_score = native_candidate_score(
                result,
                trajectory.candidate_seat,
            )
            trajectory_rewards.append(2.0 * perspective_score - 1.0)
    if trajectory_slots:
        input_started = time.perf_counter()
        delivery.accept(
            arena.page.finish_terminals(
                np.asarray(trajectory_slots, dtype=np.int64),
                np.asarray(trajectory_rewards, dtype=np.float32),
                seat_ids=np.asarray(trajectory_seats, dtype=np.int8),
            )
        )
        timings["input"] += time.perf_counter() - input_started
    _release_arena_scripted(collector, arena, terminal_slots)
    _release_sequence_games(collector, arena, terminal_slots)
    _retire_arena_slots(arena, terminal_slots, reason="terminal")
    for encoder in _owned_encoders(arena):
        encoder.clear_slots(terminal_slots)
    _require_assignment_conservation(arena)


def _finish_step_limits(
    collector: NativeStatelessCollector,
    arena: NativeRouteArena,
    view: NativeTrainingBatchView,
    step_limit_rows: Int64Array,
    losing_seats: Int32Array,
    *,
    delivery: NativePartDelivery,
    outcomes: list[StatelessGameOutcome],
    lane_scores: defaultdict[str, list[float]],
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> None:
    """Adjudicate over-limit games against the last submitted-action seat."""
    step_limit_slots = view.slots[step_limit_rows].copy()
    if losing_seats.shape != step_limit_slots.shape:
        raise ValueError("step-limit rows and losing seats must be aligned")
    if np.any((losing_seats != 0) & (losing_seats != 1)):
        raise ValueError("step-limit losing seat must be 0 or 1")
    games = tuple(arena.live[int(slot)] for slot in step_limit_slots)
    trajectories = (
        tuple(
            trajectory
            for game in games
            for trajectory in game.trajectory_games(
                mirror_bilateral=collector.mirror_bilateral_trajectories
            )
        )
        if _retains_trajectories(collector)
        else ()
    )
    if trajectories:
        input_started = time.perf_counter()
        arena.page.discard_games(trajectories)
        delivery.discard_games(tuple(game.game_id for game in games))
        timings["input"] += time.perf_counter() - input_started
    for game, raw_losing_seat in zip(games, losing_seats, strict=True):
        outcome = game.step_limit_outcome(
            losing_seat=int(raw_losing_seat)
        ).model_copy(update={"candidate_decisions": 0})
        if outcome.candidate_score is None:
            raise AssertionError("step-limit adjudication omitted its score")
        outcomes.append(outcome)
        lane_scores[game.assignment.curriculum.lane].append(outcome.candidate_score)
        candidate_decisions = int(game.candidate_decisions)
        mirror_decisions = int(game.mirror_opponent_decisions)
        counters["candidate_trajectory_rows_discarded"] += candidate_decisions
        counters["mirror_opponent_trajectory_rows_discarded"] += mirror_decisions
        counters["trainable_trajectory_rows_discarded"] += (
            candidate_decisions + mirror_decisions
        )
        counters["step_limit_games"] += 1
    _release_arena_scripted(collector, arena, step_limit_slots)
    _release_sequence_games(collector, arena, step_limit_slots)
    _retire_arena_slots(arena, step_limit_slots, reason="step limit")
    for encoder in _owned_encoders(arena):
        encoder.clear_slots(step_limit_slots)
    _require_assignment_conservation(arena)


def begin_arena_window_drain(
    collector: NativeStatelessCollector,
    arena: NativeRouteArena,
    *,
    outcomes: list[StatelessGameOutcome],
    counters: Counter[str],
    immediate_whole_game_cutoff: bool = False,
) -> None:
    """Stop admission and latch every live game for fragment-boundary drain."""
    if arena.window_draining:
        return
    arena.window_draining = True
    if immediate_whole_game_cutoff:
        if any(
            game.window_draining or game.sealed_trajectory_seats
            for game in arena.live.values()
        ):
            raise RuntimeError("native immediate cutoff found a partially sealed game")
        counters["immediate_window_drain_arenas"] += 1
    else:
        for game in arena.live.values():
            if not game.begin_window_drain():
                raise RuntimeError("native live game entered window drain twice")
    _release_pending_reservations(arena, counters=counters)
    counters["window_drain_arenas"] += 1
    _require_assignment_conservation(arena)


def _release_pending_reservations(
    arena: NativeRouteArena,
    *,
    counters: Counter[str],
) -> None:
    """Release assignments that never entered the engine."""
    while arena.pending:
        assignment_index, _game = arena.pending.popleft()
        if (
            assignment_index in arena.completed_assignment_indices
            or assignment_index in arena.released_assignment_indices
        ):
            raise RuntimeError("native pending reservation was already settled")
        arena.released_assignment_indices.add(assignment_index)
        counters["unstarted_reservations_released"] += 1


def _window_cutoff_rows(
    collector: NativeStatelessCollector,
    arena: NativeRouteArena,
    view: NativeTrainingBatchView,
    *,
    excluded_rows: Int64Array,
) -> Int64Array:
    """Return nonterminal rows whose trainable perspectives are all sealed."""
    if not getattr(arena, "window_draining", False):
        return np.asarray((), dtype=np.int64)
    excluded = {int(row) for row in excluded_rows}
    return np.asarray(
        [
            row
            for row in range(view.batch_size)
            if (
                row not in excluded
                and arena.live[int(view.slots[row])].is_window_drain_complete(
                    mirror_bilateral=collector.mirror_bilateral_trajectories
                )
            )
        ],
        dtype=np.int64,
    )


def _finish_window_cutoffs(
    collector: NativeStatelessCollector,
    arena: NativeRouteArena,
    view: NativeTrainingBatchView,
    cutoff_rows: Int64Array,
    *,
    delivery: NativePartDelivery,
    outcomes: list[StatelessGameOutcome],
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> None:
    """Roll back and retire cutoff games without publishing them to PPO."""
    cutoff_slots = view.slots[cutoff_rows].copy()
    games = tuple(arena.live[int(slot)] for slot in cutoff_slots)
    if any(
        not game.is_window_drain_complete(
            mirror_bilateral=collector.mirror_bilateral_trajectories
        )
        for game in games
    ):
        raise RuntimeError("native window cutoff contains an unsealed game")
    input_started = time.perf_counter()
    trajectory_games = tuple(
        trajectory
        for game in games
        for trajectory in game.trajectory_games(
            mirror_bilateral=collector.mirror_bilateral_trajectories,
            include_sealed=True,
        )
    )
    if trajectory_games:
        arena.page.discard_sealed_games(trajectory_games)
        delivery.discard_games(tuple(game.game_id for game in games))
    for game in games:
        candidate_decisions = int(game.candidate_decisions)
        mirror_decisions = int(game.mirror_opponent_decisions)
        counters["candidate_trajectory_rows_discarded"] += candidate_decisions
        counters["mirror_opponent_trajectory_rows_discarded"] += mirror_decisions
        counters["trainable_trajectory_rows_discarded"] += (
            candidate_decisions + mirror_decisions
        )
        outcomes.append(
            game.cancelled_outcome(
                status="window_cutoff",
                retained_candidate_decisions=0,
            )
        )
        counters["window_cutoff_games"] += 1
    _release_arena_scripted(collector, arena, cutoff_slots)
    _release_sequence_games(collector, arena, cutoff_slots)
    _retire_arena_slots(arena, cutoff_slots, reason="window cutoff")
    for encoder in _owned_encoders(arena):
        encoder.clear_slots(cutoff_slots)
    timings["input"] += time.perf_counter() - input_started
    _require_assignment_conservation(arena)


def _finish_immediate_window_cutoffs(
    collector: NativeStatelessCollector,
    arena: NativeRouteArena,
    view: NativeTrainingBatchView,
    cutoff_rows: Int64Array,
    *,
    delivery: NativePartDelivery,
    outcomes: list[StatelessGameOutcome],
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> None:
    """Discard nonterminal games atomically after the current engine wave."""
    cutoff_slots = view.slots[cutoff_rows].copy()
    games = tuple(arena.live[int(slot)] for slot in cutoff_slots)
    if any(
        game.window_draining or game.sealed_trajectory_seats for game in games
    ):
        raise RuntimeError("native immediate cutoff found a sealed trajectory")
    input_started = time.perf_counter()
    trajectory_games = tuple(
        trajectory
        for game in games
        for trajectory in game.trajectory_games(
            mirror_bilateral=collector.mirror_bilateral_trajectories
        )
    )
    if trajectory_games:
        arena.page.discard_games(trajectory_games)
        delivery.discard_games(tuple(game.game_id for game in games))
    for game in games:
        candidate_decisions = int(game.candidate_decisions)
        mirror_decisions = int(game.mirror_opponent_decisions)
        counters["candidate_trajectory_rows_discarded"] += candidate_decisions
        counters["mirror_opponent_trajectory_rows_discarded"] += mirror_decisions
        counters["trainable_trajectory_rows_discarded"] += (
            candidate_decisions + mirror_decisions
        )
        outcomes.append(
            game.cancelled_outcome(
                status="window_cutoff",
                retained_candidate_decisions=0,
            )
        )
        counters["window_cutoff_games"] += 1
        counters["immediate_window_cutoff_games"] += 1
    _release_arena_scripted(collector, arena, cutoff_slots)
    _release_sequence_games(collector, arena, cutoff_slots)
    _retire_arena_slots(arena, cutoff_slots, reason="immediate window cutoff")
    for encoder in _owned_encoders(arena):
        encoder.clear_slots(cutoff_slots)
    timings["input"] += time.perf_counter() - input_started
    _require_assignment_conservation(arena)


def _retire_arena_slots(
    arena: NativeRouteArena,
    slots: npt.ArrayLike,
    *,
    reason: str,
) -> None:
    slot_rows = np.ascontiguousarray(slots, dtype=np.uint32)
    for raw_slot in slot_rows:
        slot = int(raw_slot)
        if slot not in arena.live_assignment_indices:
            raise RuntimeError(f"native {reason} slot has no live assignment")
        assignment_index = arena.live_assignment_indices[slot]
        if assignment_index in arena.completed_assignment_indices:
            raise RuntimeError(f"native assignment reached {reason} more than once")

    # BattleData owns the engine state and its complete log history. Retired
    # games must release that ownership immediately instead of leaving every
    # terminal state resident in the worker-lifetime arena until the next
    # reset or process exit.
    arena.lane.clear_slots(slot_rows)
    for raw_slot in slot_rows:
        slot = int(raw_slot)
        assignment_index = arena.live_assignment_indices.pop(slot)
        arena.completed_assignment_indices.add(assignment_index)
        del arena.live[slot]


def _resolve_pending_sequence(
    dispatch: NativeArenaDispatch,
    view: NativeTrainingBatchView,
    *,
    acting_rows: Int64Array | None,
) -> None:
    """Commit only actions accepted by the source engine."""
    if not dispatch.pending_sequence:
        return
    source_rows = (
        np.arange(dispatch.arena.view.batch_size, dtype=np.int64)
        if acting_rows is None
        else acting_rows
    )
    output_by_source = {
        int(source_row): output_row for output_row, source_row in enumerate(source_rows)
    }
    try:
        for pending in dispatch.pending_sequence:
            try:
                output_row = output_by_source[pending.arena_row]
            except KeyError as error:
                raise RuntimeError(
                    "pending sequence action was not submitted to the engine"
                ) from error
            if int(view.error[output_row]) != 0 or int(view.status[output_row]) not in (
                1,
                2,
            ):
                pending.actor.abort_decision(pending.row, pending.trace)
                continue
            pending.actor.commit_decision(pending.row, pending.trace)
            identity = pending.row.sequence_identity
            if identity is None:
                raise RuntimeError("native committed sequence row lost its identity")
            game = dispatch.arena.live[int(view.slots[output_row])]
            if game.sequence_decisions_by_seat[identity.seat] != (
                identity.decision_index
            ):
                raise RuntimeError("native sequence decision clock diverged")
            game.sequence_decisions_by_seat[identity.seat] += 1
    except BaseException:
        _abort_pending_sequence(dispatch)
        raise
    dispatch.pending_sequence.clear()


def _abort_pending_sequence(dispatch: NativeArenaDispatch) -> None:
    """Best-effort rollback of every still-provisional sequence row."""
    pending_rows = tuple(dispatch.pending_sequence)
    dispatch.pending_sequence.clear()
    for pending in pending_rows:
        try:
            pending.actor.abort_decision(pending.row, pending.trace)
        except RuntimeError:
            # A preceding row may already have committed before a later
            # invariant failed. The enclosing collection fails closed.
            continue


def _release_sequence_games(
    collector: NativeStatelessCollector,
    arena: NativeRouteArena,
    slots: npt.ArrayLike,
) -> None:
    release = getattr(collector, "_release_sequence_game", None)
    if release is None:
        return
    for raw_slot in np.asarray(slots):
        release(arena.live[int(raw_slot)])


def _refill_route(
    collector: NativeStatelessCollector,
    arena: NativeRouteArena,
    terminal_slots: npt.NDArray[np.uint32],
    *,
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> NativeTrainingBatchView | None:
    replacement_games: list[NativeLiveGame] = []
    replacement_slots: list[int] = []
    if terminal_slots.size and arena.pending:
        for raw_slot in terminal_slots[: len(arena.pending)]:
            original_index, game = arena.pending.popleft()
            slot = int(raw_slot)
            if slot in arena.live or slot in arena.live_assignment_indices:
                raise RuntimeError("native refill attempted to replace a live slot")
            game.slot = slot
            arena.live[slot] = game
            arena.live_assignment_indices[slot] = original_index
            replacement_games.append(game)
            replacement_slots.append(slot)
    if not replacement_games:
        return None
    _require_assignment_conservation(arena)
    replacement_live = {game.slot: game for game in replacement_games}
    if _retains_trajectories(collector):
        arena.page.register_games(
            tuple(
                trajectory
                for game in replacement_games
                for trajectory in game.trajectory_games(
                    mirror_bilateral=collector.mirror_bilateral_trajectories
                )
            )
        )
    _reset_arena_scripted(
        arena.scripted_policies,
        replacement_live,
        seed=collector.seed,
    )
    decks = native_deck_rows_for_games(replacement_games)
    slots = np.asarray(replacement_slots, dtype=np.uint32)
    engine_started = time.perf_counter()
    view = arena.lane.reset(
        decks,
        native_assignment_engine_seeds(
            collector.seed,
            tuple(game.assignment for game in replacement_games),
        ),
        slots=slots,
        output=arena.buffers[arena.batch_index % 2],
    )
    timings["engine"] += time.perf_counter() - engine_started
    require_native_ready(view)
    advances = accumulate_native_selection_advances(
        view,
        arena.live,
        maximum_engine_steps=collector.maximum_engine_steps,
        submitted_actions=False,
    )
    counters["engine_steps"] += advances.count
    input_started = time.perf_counter()
    for encoder in _owned_encoders(arena):
        encoder.consume_reset(view, decks)
    timings["input"] += time.perf_counter() - input_started
    counters["games_started"] += len(replacement_games)
    return view


def release_arena_scripted(
    _collector: NativeStatelessCollector,
    arena: NativeRouteArena,
) -> None:
    """Best-effort release of the mixed arena's isolated policy runtimes."""
    release_all_native_scripted(
        arena.scripted_policies,
        arena.live,
    )


def finish_native_arena(
    arena: NativeRouteArena,
    *,
    delivery: NativePartDelivery,
) -> None:
    """Publish the final complete partial part only after arena termination."""
    _require_assignment_conservation(arena)
    if arena.live or arena.pending:
        raise RuntimeError("cannot publish a live native mixed arena")
    settled = arena.completed_assignment_indices | arena.released_assignment_indices
    if settled != set(arena.assignment_indices):
        raise RuntimeError("native mixed arena did not settle every reservation")
    if arena.page.live_game_count:
        raise RuntimeError("native mixed arena retained live trajectories")
    delivery.accept(arena.page.publish_ready(include_partial=True))
    if arena.page.decision_chunk_count:
        raise RuntimeError("native mixed arena retained unpublished trajectory rows")


def _reset_arena_scripted(
    policies: Mapping[str, NativeScriptedPolicy],
    live: Mapping[int, NativeLiveGame],
    *,
    seed: int,
) -> None:
    games = tuple(live.values())
    generated = native_assignment_scripted_seeds(
        seed,
        tuple(game.assignment for game in games),
    )
    reset_native_scripted(
        policies,
        live,
        seeds={
            game.slot: int(value) for game, value in zip(games, generated, strict=True)
        },
    )


def _parked_dispatch_rows(dispatch: NativeArenaDispatch) -> Int64Array:
    """Return the sorted union of rows waiting for a batched frozen release."""
    groups = (
        *dispatch.parked_past_rows.values(),
        *dispatch.parked_historical_rows.values(),
    )
    if not groups:
        return np.asarray((), dtype=np.int64)
    return np.sort(np.concatenate(groups))


def _require_complete_dispatch(
    collector: NativeStatelessCollector,
    dispatch: NativeArenaDispatch,
    *,
    forced_rows: Int64Array,
) -> None:
    """Require every ready row exactly once under its immutable policy route."""
    view = dispatch.arena.view
    seen: npt.NDArray[np.bool_] = np.zeros(view.batch_size, dtype=np.bool_)

    def record(rows: Int64Array, route: tuple[str, str]) -> None:
        selected = np.asarray(rows)
        if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
            raise TypeError(
                "native dispatch rows must be a one-dimensional integer array"
            )
        for raw_row in selected:
            row = int(raw_row)
            if row < 0 or row >= view.batch_size:
                raise RuntimeError("native dispatch row exceeds its ready batch")
            if seen[row]:
                raise RuntimeError("native ready row was dispatched more than once")
            expected = _expected_dispatch_route(collector, dispatch, row)
            if route != expected:
                raise RuntimeError("native ready row crossed its policy route")
            seen[row] = True

    forced = np.asarray(forced_rows)
    if forced.ndim != 1 or not np.issubdtype(forced.dtype, np.integer):
        raise TypeError("native forced rows must be a one-dimensional integer array")
    for raw_row in forced:
        row = int(raw_row)
        if row < 0 or row >= view.batch_size or seen[row]:
            raise RuntimeError("native forced row assignment is invalid")
        seen[row] = True

    current_route = (
        "current",
        collector.identity.behavior_policy_fingerprint,
    )
    record(dispatch.current_rows, current_route)
    for artifact_sha256, rows in dispatch.past_rows.items():
        record(rows, ("past_self", artifact_sha256))
    for artifact_sha256, rows in dispatch.historical_rows.items():
        record(rows, ("historical", artifact_sha256))
    for runtime_id, rows in dispatch.scripted_rows.items():
        record(rows, ("scripted", runtime_id))
    for artifact_sha256, rows in dispatch.parked_past_rows.items():
        record(rows, ("past_self", artifact_sha256))
    for artifact_sha256, rows in dispatch.parked_historical_rows.items():
        record(rows, ("historical", artifact_sha256))
    if not np.all(seen):
        raise RuntimeError("native ready rows were omitted from action dispatch")


def _expected_dispatch_route(
    collector: NativeStatelessCollector,
    dispatch: NativeArenaDispatch,
    row: int,
) -> tuple[str, str]:
    view = dispatch.arena.view
    try:
        game = dispatch.arena.live[int(view.slots[row])]
    except KeyError as error:
        raise RuntimeError(
            "native ready row references an unknown live slot"
        ) from error
    if (
        int(view.select_player[row]) == game.candidate_seat
        or game.assignment.curriculum.lane == "mirror"
    ):
        return (
            "current",
            collector.identity.behavior_policy_fingerprint,
        )
    route = collector._cohort_group_key(game)
    if route.kind == "scripted":
        return (route.kind, route.runtime_id)
    return (route.kind, route.artifact_sha256)


def _require_assignment_conservation(arena: NativeRouteArena) -> None:
    """Require every reservation to occupy exactly one lifecycle partition."""
    pending = [index for index, _game in arena.pending]
    live = list(arena.live_assignment_indices.values())
    completed = list(arena.completed_assignment_indices)
    released = list(arena.released_assignment_indices)
    combined = pending + live + completed + released
    if (
        len(combined) != len(set(combined))
        or frozenset(combined) != arena.assignment_indices
    ):
        raise RuntimeError("native mixed arena lost or duplicated an assignment")
    if set(arena.live) != set(arena.live_assignment_indices):
        raise RuntimeError("native mixed arena live slot bindings differ")
    if len(arena.live) > arena.capacity:
        raise RuntimeError("native mixed arena exceeded its slot capacity")
    if any(game.slot != slot for slot, game in arena.live.items()):
        raise RuntimeError("native mixed arena game slot binding differs")


def _release_arena_scripted(
    _collector: NativeStatelessCollector,
    arena: NativeRouteArena,
    slots: npt.ArrayLike,
) -> None:
    release_native_scripted(
        arena.scripted_policies,
        slots,
        arena.live,
    )


__all__ = [
    "NativeArenaDispatch",
    "NativePreparedArenaAdvance",
    "NativePreparedScriptedAction",
    "NativePreparedScriptedActions",
    "NativeRouteArena",
    "NativeRouteArenaEngineStep",
    "apply_prepared_scripted_actions",
    "abort_route_arena_advance",
    "abort_route_arena_dispatch",
    "advance_route_arena",
    "begin_arena_window_drain",
    "finish_route_arena_advance",
    "finish_native_arena",
    "prepare_arena_dispatches",
    "prepare_scripted_actions",
    "prepare_route_arena_advance",
    "release_arena_scripted",
    "run_route_arena_engine_step",
    "serve_scripted",
    "start_native_arena",
]
