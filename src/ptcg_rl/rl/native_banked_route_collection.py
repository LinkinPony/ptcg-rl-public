"""Production adapter for fixed-bank native route collection."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.rl.native_banked_collection import (
    BankedNativeCollectionCallbacks,
    BankedNativeCollectionReport,
    NativeBankLayout,
    run_banked_native_collection_core,
)
from ptcg_rl.rl.native_collection_delivery import NativePartDelivery
from ptcg_rl.rl.native_collection_games import build_native_live_games
from ptcg_rl.rl.native_route_arena import (
    NativeArenaDispatch,
    NativePreparedArenaAdvance,
    NativeRouteArena,
    abort_route_arena_advance,
    abort_route_arena_dispatch,
    begin_arena_window_drain,
    finish_native_arena,
    finish_route_arena_advance,
    prepare_arena_dispatches,
    prepare_route_arena_advance,
    release_arena_scripted,
    run_route_arena_engine_step,
    start_native_arena,
)
from ptcg_rl.rl.native_route_collection import (
    _abort_prepared_policy_routes,
    _collection_result,
    _external_drain_requested,
    _finish_prepared_scripted_routes,
    _net_trainable_decisions,
    _prepare_policy_route_wave,
    _PreparedPolicyRouteWave,
    _route_inference_executor,
    _serve_prepared_policy_routes,
)
from ptcg_rl.rl.native_route_scheduler import (
    NativeArenaKey,
    NativeRouteKey,
    plan_native_artifact_banks,
)
from ptcg_rl.rl.stateless_collection import (
    StatelessAssignedGame,
    StatelessCollectionResult,
    StatelessGameOutcome,
)

if TYPE_CHECKING:
    from ptcg_rl.rl.native_policy_bank import NativePolicyInferenceWaveTicket
    from ptcg_rl.rl.native_stateless_collection import NativeStatelessCollector

_MIN_ARENAS_PER_BANK = 2


@dataclass(slots=True)
class _NativeBankArenaState:
    """All mutable result state owned by one physical engine arena."""

    arena: NativeRouteArena
    delivery: NativePartDelivery
    outcomes: list[StatelessGameOutcome]
    lane_scores: defaultdict[str, list[float]]
    timings: defaultdict[str, float]
    counters: Counter[str]
    advance_started_at: float = 0.0
    finished: bool = False


@dataclass(frozen=True, slots=True)
class _PreparedNativeBankPolicy:
    """One bank frozen at a policy boundary with host work already queued."""

    arenas: tuple[NativeRouteArena, ...]
    wave: _PreparedPolicyRouteWave


@dataclass(frozen=True, slots=True)
class _SubmittedNativeBankPolicy:
    """One sealed CUDA wave whose host actions are not visible yet."""

    prepared: _PreparedNativeBankPolicy
    ticket: NativePolicyInferenceWaveTicket


class _NativeRouteBankCallbacks(
    BankedNativeCollectionCallbacks[
        NativeRouteArena,
        _PreparedNativeBankPolicy,
        _SubmittedNativeBankPolicy,
        NativeArenaDispatch,
        NativePreparedArenaAdvance,
        NativeTrainingBatchView,
    ]
):
    """Bind the generic A/B state machine to native route operations."""

    def __init__(
        self,
        collector: NativeStatelessCollector,
        states: Sequence[_NativeBankArenaState],
        *,
        route_executor: ThreadPoolExecutor | None,
        scripted_executor: ThreadPoolExecutor | None,
        cancellation_event: threading.Event | None,
        coordinator_timings: defaultdict[str, float],
        coordinator_counters: Counter[str],
        artifact_batches: Counter[NativeRouteKey],
        artifact_rows: Counter[NativeRouteKey],
        generators: dict[NativeRouteKey, torch.Generator],
    ) -> None:
        self._collector = collector
        self._states = {id(state.arena): state for state in states}
        self._route_executor = route_executor
        self._scripted_executor = scripted_executor
        self._cancellation_event = cancellation_event
        self._timings = coordinator_timings
        self._counters = coordinator_counters
        self._artifact_batches = artifact_batches
        self._artifact_rows = artifact_rows
        self._generators = generators

    @staticmethod
    def is_live(arena: NativeRouteArena) -> bool:
        return bool(arena.live)

    def prepare_policy(
        self,
        arenas: tuple[NativeRouteArena, ...],
    ) -> _PreparedNativeBankPolicy:
        """Freeze one bank and immediately queue host encoding and fact probes."""
        self._require_not_cancelled()
        self._begin_bank_drain_if_required(arenas)
        dispatches = prepare_arena_dispatches(
            self._collector,
            arenas,
            counters=self._counters,
            frozen_scheduler=None,
        )
        if any(
            dispatch.parked_past_rows or dispatch.parked_historical_rows
            for dispatch in dispatches
        ):
            raise RuntimeError("banked native collection parked a frozen route")
        cohort_rows = sum(
            int(dispatch.arena.view.batch_size) for dispatch in dispatches
        )
        self._counters["policy_cohort_batches"] += 1
        self._counters["policy_cohort_rows"] += cohort_rows
        self._counters["policy_cohort_max_rows"] = max(
            self._counters["policy_cohort_max_rows"],
            cohort_rows,
        )
        return _PreparedNativeBankPolicy(
            arenas=arenas,
            wave=_prepare_policy_route_wave(
                self._collector,
                dispatches,
                route_executor=self._route_executor,
                scripted_executor=self._scripted_executor,
            ),
        )

    def submit_prepared_policy(
        self,
        prepared: _PreparedNativeBankPolicy,
        *,
        overlap: Callable[[], None],
    ) -> _SubmittedNativeBankPolicy:
        """Seal one CUDA wave while peer banks advance on the owner thread."""
        self._require_not_cancelled()
        # A peer wave may have crossed the global decision budget after this
        # ticket was frozen. Latch the bank before its engine step; the frozen
        # view remains valid because it has not advanced while prefetched.
        self._begin_bank_drain_if_required(prepared.arenas)
        ticket = _serve_prepared_policy_routes(
            self._collector,
            prepared.wave,
            generators=self._generators,
            delivery={
                id(arena): self._state(arena).delivery for arena in prepared.arenas
            },
            timings=self._timings,
            counters=self._counters,
            artifact_batches=self._artifact_batches,
            artifact_rows=self._artifact_rows,
            overlap=overlap,
            defer_completion=True,
        )
        if ticket is None:
            raise RuntimeError("banked CUDA policy submission returned no ticket")
        return _SubmittedNativeBankPolicy(
            prepared=prepared,
            ticket=ticket,
        )

    def finish_policy(
        self,
        submitted: _SubmittedNativeBankPolicy,
    ) -> tuple[NativeArenaDispatch, ...]:
        """Resolve one exact CUDA boundary and publish its host actions."""
        self._require_not_cancelled()
        _finish_prepared_scripted_routes(
            submitted.prepared.wave,
            timings=self._timings,
            counters=self._counters,
        )
        wait_started_at = time.perf_counter()
        submitted.ticket.finish()
        self._timings["policy_completion_wait"] += time.perf_counter() - wait_started_at
        return submitted.prepared.wave.dispatches

    @staticmethod
    def abort_policy(prepared: _PreparedNativeBankPolicy) -> None:
        _abort_prepared_policy_routes(prepared.wave)

    @staticmethod
    def abort_submitted_policy(
        submitted: _SubmittedNativeBankPolicy,
    ) -> None:
        submitted.ticket.abort()
        _abort_prepared_policy_routes(submitted.prepared.wave)
        for dispatch in submitted.prepared.wave.dispatches:
            abort_route_arena_dispatch(dispatch)

    def prepare_engine(
        self,
        arena: NativeRouteArena,
        dispatch: NativeArenaDispatch,
    ) -> NativePreparedArenaAdvance | None:
        if dispatch.arena is not arena:
            raise RuntimeError("banked policy dispatch changed arena ownership")
        # Policy completion is the atomic publish boundary. Latch any newly
        # crossed decision budget here, before this arena can advance, without
        # introducing a fallible operation after sequence proposals publish.
        self._begin_bank_drain_if_required((arena,))
        state = self._state(arena)
        prepared = prepare_route_arena_advance(
            dispatch,
            timings=state.timings,
        )
        if prepared is not None:
            state.advance_started_at = time.perf_counter()
        return prepared

    @staticmethod
    def run_engine(
        prepared: NativePreparedArenaAdvance,
    ) -> NativeTrainingBatchView:
        return run_route_arena_engine_step(prepared.engine_step)

    def finish_engine(
        self,
        arena: NativeRouteArena,
        prepared: NativePreparedArenaAdvance,
        result: NativeTrainingBatchView,
    ) -> None:
        state = self._state(arena)
        state.timings["engine"] += max(
            time.perf_counter() - state.advance_started_at,
            0.0,
        )
        finish_route_arena_advance(
            self._collector,
            prepared,
            result,
            delivery=state.delivery,
            outcomes=state.outcomes,
            lane_scores=state.lane_scores,
            timings=state.timings,
            counters=state.counters,
        )
        state.advance_started_at = 0.0
        if arena.live:
            return
        finish_native_arena(arena, delivery=state.delivery)
        state.finished = True

    @staticmethod
    def abort_engine(prepared: NativePreparedArenaAdvance) -> None:
        abort_route_arena_advance(prepared)

    def _begin_bank_drain_if_required(
        self,
        arenas: Sequence[NativeRouteArena],
    ) -> bool:
        if not _external_drain_requested(self._collector):
            budget = getattr(self._collector, "trainable_decision_budget", None)
            if budget is None:
                return False
            trainable_decisions = _net_trainable_decisions(
                self._counters,
                tuple(state.counters for state in self._states.values()),
            )
            if trainable_decisions < int(budget):
                return False
        changed = False
        for arena in arenas:
            if arena.window_draining:
                continue
            state = self._state(arena)
            begin_arena_window_drain(
                self._collector,
                arena,
                outcomes=state.outcomes,
                counters=state.counters,
                immediate_whole_game_cutoff=bool(
                    getattr(
                        self._collector,
                        "immediate_whole_game_cutoff_on_drain",
                        False,
                    )
                ),
            )
            changed = True
        return changed

    def _state(self, arena: NativeRouteArena) -> _NativeBankArenaState:
        try:
            return self._states[id(arena)]
        except KeyError as error:
            raise RuntimeError("banked native arena is not registered") from error

    def _require_not_cancelled(self) -> None:
        if self._cancellation_event is not None and self._cancellation_event.is_set():
            raise RuntimeError("native arena collection was cancelled")


def collect_native_banked_assigned(
    collector: NativeStatelessCollector,
    assignments: Sequence[StatelessAssignedGame],
    *,
    cancellation_event: threading.Event | None = None,
) -> StatelessCollectionResult:
    """Collect one lease through an artifact-coherent fixed bank ring."""
    physical_arenas = int(collector.engine_shards)
    if physical_arenas < 4:
        raise ValueError("banked native collection requires at least four arenas")
    if len(assignments) < physical_arenas:
        raise ValueError("banked native collection cannot fill every physical arena")
    started_at = time.perf_counter()
    prepared = build_native_live_games(
        assignments,
        identity=collector.identity,
        active_decks=collector.active_decks,
        opponent_decks=collector.opponent_decks,
        members=collector.members,
        scripted_bindings=collector.scripted_bindings,
        scripted_policy_ids=frozenset(collector.scripted_policies),
    )
    group_keys = tuple(
        collector._cohort_group_key(prepared[index])
        for index in range(len(assignments))
    )
    execution_bank_count = _execution_bank_count(
        group_keys,
        physical_arenas=physical_arenas,
    )
    physical_arena_counts = _physical_arena_counts(
        physical_arenas,
        bank_count=execution_bank_count,
    )
    execution_banks = plan_native_artifact_banks(
        assignments,
        bank_count=execution_bank_count,
        group_keys=group_keys,
    )
    physical_groups = _split_execution_banks(
        execution_banks,
        physical_arena_counts=physical_arena_counts,
    )
    configured_capacity = collector.arena_capacity or len(prepared)
    total_capacity = min(configured_capacity, len(prepared))
    capacities = _allocate_physical_group_capacities(
        total_capacity,
        tuple(len(group) for group in physical_groups),
    )
    configure_past_slots = getattr(
        collector,
        "_configure_past_temporal_cache_slots",
        None,
    )
    if configure_past_slots is not None:
        configure_past_slots(
            assignments,
            arena_capacity=total_capacity,
        )

    coordinator_timings: defaultdict[str, float] = defaultdict(float)
    coordinator_counters: Counter[str] = Counter()
    artifact_batches: Counter[NativeRouteKey] = Counter()
    artifact_rows: Counter[NativeRouteKey] = Counter()
    generators: dict[NativeRouteKey, torch.Generator] = {}
    states: list[_NativeBankArenaState] = []
    core_report: BankedNativeCollectionReport | None = None

    with ExitStack() as lane_stack:
        try:
            for group, capacity in zip(
                physical_groups,
                capacities,
                strict=True,
            ):
                timings: defaultdict[str, float] = defaultdict(float)
                counters: Counter[str] = Counter()
                arena_prepared = {
                    local_index: prepared[assignment_index]
                    for local_index, assignment_index in enumerate(group)
                }
                states.append(
                    _NativeBankArenaState(
                        arena=start_native_arena(
                            collector,
                            arena_prepared,
                            capacity=capacity,
                            lane_stack=lane_stack,
                            timings=timings,
                            counters=counters,
                        ),
                        delivery=NativePartDelivery(
                            part_sink=getattr(
                                collector,
                                "compact_part_sink",
                                None,
                            )
                        ),
                        outcomes=[],
                        lane_scores=defaultdict(list),
                        timings=timings,
                        counters=counters,
                    )
                )
            route_executor = _route_inference_executor(collector)
            if route_executor is not None:
                lane_stack.enter_context(route_executor)
            scripted_executor = (
                lane_stack.enter_context(
                    ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix="native-scripted-prefetch",
                    )
                )
                if collector.scripted_policies
                else None
            )
            coordinator_timings["startup"] += time.perf_counter() - started_at
            callbacks = _NativeRouteBankCallbacks(
                collector,
                states,
                route_executor=route_executor,
                scripted_executor=scripted_executor,
                cancellation_event=cancellation_event,
                coordinator_timings=coordinator_timings,
                coordinator_counters=coordinator_counters,
                artifact_batches=artifact_batches,
                artifact_rows=artifact_rows,
                generators=generators,
            )
            core_report = run_banked_native_collection_core(
                tuple(state.arena for state in states),
                callbacks,
                layout=NativeBankLayout(
                    banks=_physical_bank_layout(physical_arena_counts)
                ),
                coalesce_two_bank_layout=(
                    execution_bank_count == 2
                    and collector.policy_cohort_slots == total_capacity
                ),
                policy_group_bank_limit=collector.policy_group_bank_limit,
                policy_cohort_wait_seconds=(
                    float(getattr(collector, "policy_cohort_wait_ms", 0.0)) / 1000.0
                ),
            )
            coordinator_timings["bank_engine_wait"] += sum(
                core_report.bank_wait_seconds
            )
            coordinator_timings["bank_policy_overlap"] += (
                core_report.policy_overlap_seconds
            )
            coordinator_counters["bank_policy_prefetches"] += (
                core_report.policy_prefetches
            )
            coordinator_counters["bank_engine_barriers"] += core_report.engine_barriers
            coordinator_counters["bank_policy_groups"] += core_report.policy_waves
            coordinator_counters["bank_policy_group_members"] += (
                core_report.policy_group_member_banks
            )
            coordinator_counters["bank_policy_group_max_size"] = max(
                coordinator_counters["bank_policy_group_max_size"],
                core_report.policy_group_max_size,
            )
            coordinator_counters["bank_policy_coalescing_misses"] += (
                core_report.policy_coalescing_misses
            )
            coordinator_timings["policy_cohort_wait"] += (
                core_report.policy_cohort_wait_seconds
            )
            coordinator_counters["policy_cohort_wait_events"] += (
                core_report.policy_cohort_wait_events
            )
            coordinator_counters["policy_cohort_wait_harvests"] += (
                core_report.policy_cohort_wait_harvests
            )
            coordinator_timings["bank_gpu_feed_gap"] += core_report.gpu_feed_gap_seconds
            coordinator_counters["bank_gpu_feed_gap_events"] += (
                core_report.gpu_feed_gap_events
            )
        except BaseException:
            for state in states:
                release_arena_scripted(collector, state.arena)
            raise

    if core_report is None or not all(state.finished for state in states):
        raise RuntimeError("banked native collection left an unfinished arena")
    return _merge_banked_result(
        collector,
        assignments,
        states=states,
        coordinator_timings=coordinator_timings,
        coordinator_counters=coordinator_counters,
        artifact_batches=artifact_batches,
        artifact_rows=artifact_rows,
        started_at=started_at,
        native_engine_arenas=physical_arenas,
    )


def _split_execution_banks(
    execution_banks: Sequence[Sequence[int]],
    *,
    physical_arena_counts: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Stripe each artifact bank across its permanent physical arenas."""
    if len(execution_banks) < 2:
        raise ValueError("native artifact planner produced fewer than two banks")
    if len(execution_banks) != len(physical_arena_counts):
        raise ValueError("native execution banks and arena counts differ")
    physical: list[tuple[int, ...]] = []
    for bank, arena_count in zip(
        execution_banks,
        physical_arena_counts,
        strict=True,
    ):
        if len(bank) < arena_count:
            raise ValueError("native artifact bank cannot fill its physical arenas")
        physical.extend(
            tuple(bank[offset::arena_count]) for offset in range(arena_count)
        )
    if len(physical) != sum(physical_arena_counts) or any(
        not group for group in physical
    ):
        raise RuntimeError("native physical arena plan is incomplete")
    return tuple(physical)


def _allocate_physical_group_capacities(
    total_capacity: int,
    assignment_counts: Sequence[int],
) -> tuple[int, ...]:
    """Allocate live slots proportionally while retaining a queue per arena."""
    counts = tuple(int(value) for value in assignment_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("native physical arena assignments must be positive")
    if total_capacity < len(counts) or total_capacity > sum(counts):
        raise ValueError("native live capacity cannot cover physical arenas")
    if total_capacity == sum(counts):
        return counts
    exact = tuple(total_capacity * value / sum(counts) for value in counts)
    capacities = [
        max(1, min(counts[index], int(value))) for index, value in enumerate(exact)
    ]
    add_order = sorted(
        range(len(counts)),
        key=lambda index: (-(exact[index] - int(exact[index])), index),
    )
    while sum(capacities) < total_capacity:
        changed = False
        for index in add_order:
            if capacities[index] >= counts[index]:
                continue
            capacities[index] += 1
            changed = True
            if sum(capacities) == total_capacity:
                break
        if not changed:
            raise RuntimeError("native physical arena capacity could not grow")
    remove_order = tuple(reversed(add_order))
    while sum(capacities) > total_capacity:
        changed = False
        for index in remove_order:
            if capacities[index] <= 1:
                continue
            capacities[index] -= 1
            changed = True
            if sum(capacities) == total_capacity:
                break
        if not changed:
            raise RuntimeError("native physical arena capacity could not shrink")
    return tuple(capacities)


def _execution_bank_count(
    group_keys: Sequence[NativeArenaKey],
    *,
    physical_arenas: int,
) -> int:
    """Use the smallest ring that preserves frozen-artifact affinity."""
    frozen_artifacts = {
        key.artifact_sha256
        for key in group_keys
        if key.kind in {"past_self", "historical"}
    }
    bank_count = max(2, len(frozen_artifacts))
    if physical_arenas < bank_count * _MIN_ARENAS_PER_BANK:
        raise ValueError("native frozen artifacts exceed execution-bank capacity")
    return bank_count


def _physical_arena_counts(
    physical_arenas: int,
    *,
    bank_count: int,
) -> tuple[int, ...]:
    """Distribute engine arenas across the fixed policy-bank ring."""
    base, extra = divmod(physical_arenas, bank_count)
    counts = tuple(base + int(bank_index < extra) for bank_index in range(bank_count))
    if any(count < _MIN_ARENAS_PER_BANK for count in counts):
        raise ValueError("native policy banks require at least two arenas")
    return counts


def _physical_bank_layout(
    physical_arena_counts: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Map bank-local arena counts onto contiguous adapter state indices."""
    cursor = 0
    banks: list[tuple[int, ...]] = []
    for count in physical_arena_counts:
        banks.append(tuple(range(cursor, cursor + count)))
        cursor += count
    return tuple(banks)


def _merge_banked_result(
    collector: NativeStatelessCollector,
    assignments: Sequence[StatelessAssignedGame],
    *,
    states: Sequence[_NativeBankArenaState],
    coordinator_timings: defaultdict[str, float],
    coordinator_counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
    started_at: float,
    native_engine_arenas: int,
) -> StatelessCollectionResult:
    """Merge physical arena ownership before existing invariant checks."""
    delivery = NativePartDelivery()
    outcomes: list[StatelessGameOutcome] = []
    lane_scores: defaultdict[str, list[float]] = defaultdict(list)
    timings: defaultdict[str, float] = defaultdict(float)
    counters: Counter[str] = Counter()
    counters.update(coordinator_counters)
    for key, seconds in coordinator_timings.items():
        timings[key] += seconds
    for state in states:
        delivery.accept(state.delivery.retained)
        outcomes.extend(state.outcomes)
        counters.update(state.counters)
        for key, seconds in state.timings.items():
            timings[key] += seconds
        for lane, scores in state.lane_scores.items():
            lane_scores[lane].extend(scores)

    elapsed = time.perf_counter() - started_at
    measured_non_engine = (
        timings["input"]
        + timings["current"]
        + timings["past"]
        + timings["historical"]
        + timings["scripted"]
        + timings["policy_host_prepare_wait"]
        + timings["policy_completion_wait"]
        - timings["policy_route_overlap"]
    )
    timings["engine_control"] = max(elapsed - measured_non_engine, 0.0)
    _maybe_dump_shard_diagnostics(timings, counters, elapsed_seconds=elapsed)
    return _collection_result(
        assignments,
        delivery=delivery,
        outcomes=outcomes,
        lane_scores=lane_scores,
        timings=timings,
        counters=counters,
        started_at=started_at,
        native_engine_arenas=native_engine_arenas,
        trainable_decision_budget=getattr(
            collector,
            "trainable_decision_budget",
            None,
        ),
        artifact_batches=artifact_batches,
        artifact_rows=artifact_rows,
    )


def _maybe_dump_shard_diagnostics(
    timings: defaultdict[str, float],
    counters: Counter[str],
    *,
    elapsed_seconds: float,
) -> None:
    """Write raw per-shard timings when shard diagnostics are requested."""
    directory = os.environ.get("PTCG_RL_SHARD_DIAG_DIR")
    if not directory:
        return
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_id = os.environ.get("PTCG_RL_WORKER_ID", "unknown")
    payload = {
        "format": "ptcg-shard-diagnostics-v1",
        "worker_id": worker_id,
        "pid": os.getpid(),
        "written_at_unix": time.time(),
        "elapsed_seconds": elapsed_seconds,
        "timings": dict(timings),
        "counters": dict(counters),
    }
    output_path = output_dir / f"{worker_id}_{os.getpid()}_{time.monotonic_ns()}.json"
    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload))
    tmp_path.replace(output_path)


__all__ = ["collect_native_banked_assigned"]
