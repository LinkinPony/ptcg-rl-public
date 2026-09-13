"""Mixed native engine arenas driven by one inference-wave coordinator."""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import torch

from ptcg_rl.rl.native_collection_delivery import (
    NativePartDelivery,
    discard_compact_assignments,
)
from ptcg_rl.rl.native_collection_games import (
    NativeLiveGame,
    build_native_live_games,
)
from ptcg_rl.rl.native_route_arena import (
    NativeArenaDispatch,
    NativePreparedArenaAdvance,
    NativePreparedScriptedActions,
    NativeRouteArena,
    abort_route_arena_advance,
    advance_route_arena,
    apply_prepared_scripted_actions,
    begin_arena_window_drain,
    finish_native_arena,
    finish_route_arena_advance,
    prepare_arena_dispatches,
    prepare_route_arena_advance,
    prepare_scripted_actions,
    release_arena_scripted,
    run_route_arena_engine_step,
    serve_scripted,
    start_native_arena,
)
from ptcg_rl.rl.native_route_inference import (
    PendingNativeRouteSubmission,
    PreparedNativeCurrent,
    PreparedNativePast,
    begin_current,
    begin_past,
    prepare_current,
    prepare_past,
    resolve_past_executors,
    serve_historical,
    submit_current,
    submit_past,
)
from ptcg_rl.rl.native_route_scheduler import (
    NativeFrozenBatchScheduler,
    NativeRouteKey,
)
from ptcg_rl.rl.stateless_collection import (
    NativeArtifactInferenceReport,
    NativeInferenceRouteKind,
    StatelessAssignedGame,
    StatelessCollectionReport,
    StatelessCollectionResult,
    StatelessGameOutcome,
)

if TYPE_CHECKING:
    from ptcg_rl.engine.native_training import NativeTrainingBatchView
    from ptcg_rl.rl.native_policy_bank import NativePolicyInferenceWaveTicket
    from ptcg_rl.rl.native_policy_inference import NativePolicyInferenceExecutor
    from ptcg_rl.rl.native_stateless_collection import NativeStatelessCollector
    from ptcg_rl.rl.sequence_actor import GeneralistSequenceActorPolicy


_MAX_POLICY_PREPARE_WORKERS = 8


@dataclass(slots=True)
class _RouteShardState:
    """Mutable collection state owned by one independent native arena."""

    shard_index: int
    arena: NativeRouteArena
    delivery: NativePartDelivery
    outcomes: list[StatelessGameOutcome]
    lane_scores: defaultdict[str, list[float]]
    timings: defaultdict[str, float]
    counters: Counter[str]
    advance_future: Future[NativeTrainingBatchView] | None = None
    prepared_advance: NativePreparedArenaAdvance | None = None
    advance_started_at: float = 0.0
    finished: bool = False


@dataclass(frozen=True, slots=True)
class _PreparedPolicyRoutes:
    """Host-only inputs for one immutable inference cohort."""

    current: PreparedNativeCurrent | None
    past: PreparedNativePast
    prepare_wall_seconds: float
    prepare_work_seconds: float
    prepare_task_count: int


@dataclass(frozen=True, slots=True)
class _TimedCurrentPreparation:
    prepared: PreparedNativeCurrent | None
    started_at: float
    finished_at: float


@dataclass(frozen=True, slots=True)
class _TimedPastPreparation:
    prepared: PreparedNativePast
    started_at: float
    finished_at: float


@dataclass(frozen=True, slots=True)
class _PolicyPreparationTicket:
    """Ordered route CPU work queued behind one native encoder worker."""

    current_future: Future[_TimedCurrentPreparation] | None
    past_futures: tuple[Future[_TimedPastPreparation], ...]


@dataclass(frozen=True, slots=True)
class _PreparedPolicyRouteWave:
    """One immutable cohort whose host preparation may run ahead of CUDA."""

    dispatches: tuple[NativeArenaDispatch, ...]
    serial: _PreparedPolicyRoutes | None
    ticket: _PolicyPreparationTicket | None
    scripted_future: Future[NativePreparedScriptedActions] | None
    scripted_submitted_at: float | None


@dataclass(frozen=True, slots=True)
class _PolicyCohort:
    """One FIFO cohort selected out of the ready-arena set."""

    states: tuple[_RouteShardState, ...]
    dispatches: tuple[NativeArenaDispatch, ...]
    preparation: _PolicyPreparationTicket


@contextmanager
def _policy_inference_wave(
    collector: NativeStatelessCollector,
    *,
    defer_completion: bool = False,
) -> Iterator[NativePolicyInferenceWaveTicket | None]:
    """Keep compatibility fakes synchronous while production batches D2H."""
    policy_bank = getattr(collector, "policy_bank", None)
    if policy_bank is None:
        yield None
        return
    if defer_completion:
        with policy_bank.deferred_wave() as ticket:
            yield ticket
        return
    with policy_bank.wave():
        yield None


def _route_inference_executor(
    collector: NativeStatelessCollector,
) -> ThreadPoolExecutor | None:
    """Create CPU preparation lanes that keep recurrent CUDA work supplied."""
    actor = getattr(collector, "actor", None)
    if actor is None:
        return None
    device = getattr(actor, "device", None)
    if (
        not bool(getattr(actor, "uses_generalist_sequence", False))
        or device is None
        or torch.device(device).type != "cuda"
    ):
        return None
    worker_count = min(
        max(int(getattr(collector, "engine_shards", 2)), 2),
        _MAX_POLICY_PREPARE_WORKERS,
    )
    return ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="native-policy-prepare",
    )


def _prepare_policy_routes_serial(
    collector: NativeStatelessCollector,
    dispatches: tuple[NativeArenaDispatch, ...],
    past_executors: Mapping[
        str,
        NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
    ],
) -> _PreparedPolicyRoutes:
    """Prepare one CPU cohort serially when no CUDA pipeline is available."""
    started_at = time.perf_counter()
    current = prepare_current(collector, dispatches)
    past = prepare_past(
        collector,
        dispatches,
        executors=past_executors,
    )
    work_seconds = past.prepare_seconds + (
        0.0 if current is None else current.prepare_seconds
    )
    return _PreparedPolicyRoutes(
        current=current,
        past=past,
        prepare_wall_seconds=time.perf_counter() - started_at,
        prepare_work_seconds=work_seconds,
        prepare_task_count=int(current is not None) + len(past.routes),
    )


def _submit_policy_preparation(
    collector: NativeStatelessCollector,
    dispatches: tuple[NativeArenaDispatch, ...],
    past_executors: Mapping[
        str,
        NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
    ],
    *,
    route_executor: ThreadPoolExecutor,
) -> _PolicyPreparationTicket:
    """Queue current then exact past tasks in deterministic route order."""
    current_future = (
        route_executor.submit(
            _timed_prepare_current,
            collector,
            dispatches,
        )
        if any(dispatch.current_rows.size for dispatch in dispatches)
        else None
    )
    past_futures: list[Future[_TimedPastPreparation]] = []
    try:
        for artifact_sha256, executor in past_executors.items():
            route_dispatches = tuple(
                replace(
                    dispatch,
                    past_rows={artifact_sha256: dispatch.past_rows[artifact_sha256]},
                )
                for dispatch in dispatches
                if artifact_sha256 in dispatch.past_rows
            )
            if not route_dispatches:
                raise RuntimeError("resolved past route has no policy rows")
            past_futures.append(
                route_executor.submit(
                    _timed_prepare_past,
                    collector,
                    route_dispatches,
                    artifact_sha256,
                    executor,
                )
            )
    except BaseException:
        _abort_policy_preparation_ticket(
            _PolicyPreparationTicket(
                current_future=current_future,
                past_futures=tuple(past_futures),
            )
        )
        raise
    return _PolicyPreparationTicket(
        current_future=current_future,
        past_futures=tuple(past_futures),
    )


def _timed_prepare_current(
    collector: NativeStatelessCollector,
    dispatches: tuple[NativeArenaDispatch, ...],
) -> _TimedCurrentPreparation:
    """Prepare current host inputs and retain actual task boundaries."""
    started_at = time.perf_counter()
    prepared = prepare_current(collector, dispatches)
    return _TimedCurrentPreparation(
        prepared=prepared,
        started_at=started_at,
        finished_at=time.perf_counter(),
    )


def _timed_prepare_past(
    collector: NativeStatelessCollector,
    dispatches: tuple[NativeArenaDispatch, ...],
    artifact_sha256: str,
    executor: NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
) -> _TimedPastPreparation:
    """Prepare one exact artifact without changing its route position."""
    started_at = time.perf_counter()
    prepared = prepare_past(
        collector,
        dispatches,
        executors={artifact_sha256: executor},
    )
    if (
        len(prepared.routes) != 1
        or prepared.routes[0].route.artifact_sha256 != artifact_sha256
    ):
        raise RuntimeError("past preparation changed exact route identity")
    return _TimedPastPreparation(
        prepared=prepared,
        started_at=started_at,
        finished_at=time.perf_counter(),
    )


def _submit_policy_routes(
    collector: NativeStatelessCollector,
    dispatches: tuple[NativeArenaDispatch, ...],
    prepared: _PreparedPolicyRoutes,
    *,
    generators: dict[NativeRouteKey, torch.Generator],
    delivery: NativePartDelivery | Mapping[int, NativePartDelivery],
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
    overlap: Callable[[], None] | None = None,
    defer_completion: bool = False,
    serve_scripted_inline: bool = True,
) -> NativePolicyInferenceWaveTicket | None:
    """Submit one prepared wave from the sole actor/GPU owner thread."""
    policy_bank = getattr(collector, "policy_bank", None)
    wave_started_at = time.perf_counter()
    accounted_before = sum(
        timings[key] for key in ("input", "current", "past", "historical", "scripted")
    )
    overlap_before = timings["policy_route_overlap"]
    explicit_overlap_seconds = 0.0
    with _policy_inference_wave(
        collector,
        defer_completion=defer_completion,
    ) as policy_ticket:
        submit_current(
            collector,
            prepared.current,
            generators=generators,
            delivery=delivery,
            timings=timings,
            counters=counters,
            artifact_batches=artifact_batches,
            artifact_rows=artifact_rows,
        )
        submit_past(
            collector,
            prepared.past,
            generators=generators,
            timings=timings,
            counters=counters,
            artifact_batches=artifact_batches,
            artifact_rows=artifact_rows,
        )
        serve_historical(
            collector,
            dispatches,
            timings=timings,
            counters=counters,
            artifact_batches=artifact_batches,
            artifact_rows=artifact_rows,
        )
        if serve_scripted_inline:
            serve_scripted(
                collector,
                dispatches,
                timings=timings,
            )
        if overlap is not None:
            overlap_started_at = time.perf_counter()
            overlap()
            explicit_overlap_seconds += time.perf_counter() - overlap_started_at
    try:
        wave_elapsed = time.perf_counter() - wave_started_at
        submission_elapsed = max(wave_elapsed - explicit_overlap_seconds, 0.0)
        accounted_after = sum(
            timings[key]
            for key in ("input", "current", "past", "historical", "scripted")
        )
        exclusive_accounted = accounted_after - accounted_before
        route_overlap = max(exclusive_accounted - submission_elapsed, 0.0)
        if route_overlap:
            timings["policy_route_overlap"] += route_overlap
            counters["policy_route_overlap_waves"] += 1
        if policy_bank is not None:
            added_overlap = timings["policy_route_overlap"] - overlap_before
            timings["current"] += max(
                submission_elapsed - (exclusive_accounted - added_overlap),
                0.0,
            )
    except BaseException:
        _abort_untransferred_policy_ticket(policy_ticket)
        raise
    return policy_ticket


def _submit_policy_preparation_ticket(
    collector: NativeStatelessCollector,
    dispatches: tuple[NativeArenaDispatch, ...],
    ticket: _PolicyPreparationTicket,
    *,
    generators: dict[NativeRouteKey, torch.Generator],
    delivery: NativePartDelivery | Mapping[int, NativePartDelivery],
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
    overlap: Callable[[], None] | None = None,
    defer_completion: bool = False,
    serve_scripted_inline: bool = True,
) -> NativePolicyInferenceWaveTicket | None:
    """Fill host/fact waits with whichever deterministic route prefix is ready."""
    policy_bank = getattr(collector, "policy_bank", None)
    wave_started_at = time.perf_counter()
    accounted_before = sum(
        timings[key] for key in ("input", "current", "past", "historical", "scripted")
    )
    overlap_before = timings["policy_route_overlap"]
    prepare_wait_before = timings["policy_host_prepare_wait"]
    explicit_overlap_seconds = 0.0
    pending_routes: deque[PendingNativeRouteSubmission] = deque()
    with _policy_inference_wave(
        collector,
        defer_completion=defer_completion,
    ) as policy_ticket:
        try:
            cpu_routes_served = False

            def serve_cpu_routes_once() -> None:
                nonlocal cpu_routes_served
                if cpu_routes_served:
                    return
                serve_historical(
                    collector,
                    dispatches,
                    timings=timings,
                    counters=counters,
                    artifact_batches=artifact_batches,
                    artifact_rows=artifact_rows,
                )
                if serve_scripted_inline:
                    serve_scripted(
                        collector,
                        dispatches,
                        timings=timings,
                    )
                cpu_routes_served = True

            remaining_past = deque(ticket.past_futures)

            def submit_prepared_past(
                future: Future[_TimedPastPreparation],
                *,
                prior_wait_seconds: float,
            ) -> None:
                prepared_past = _wait_past_preparation(
                    future,
                    prior_wait_seconds=prior_wait_seconds,
                    timings=timings,
                    counters=counters,
                )
                pending_past = begin_past(
                    collector,
                    prepared_past,
                    generators=generators,
                    timings=timings,
                    counters=counters,
                    artifact_batches=artifact_batches,
                    artifact_rows=artifact_rows,
                )
                pending_routes.extend(pending_past.routes)
                _resume_ready_routes(pending_routes)

            current_future = ticket.current_future
            current_prior_wait_seconds = 0.0
            # Current batches are normally much larger than an exact past
            # route. If the next immutable past artifact finishes first, queue
            # that prefix instead of leaving CUDA idle behind current host
            # encoding. Past artifacts retain their declared order, and every
            # route owns an independent generator and temporal cache.
            while (
                current_future is not None
                and not current_future.done()
                and remaining_past
            ):
                next_past = remaining_past[0]
                prior_wait_seconds = _resume_routes_until_preparation_ready(
                    (
                        cast(Future[object], current_future),
                        cast(Future[object], next_past),
                    ),
                    pending_routes,
                )
                if current_future.done():
                    current_prior_wait_seconds += prior_wait_seconds
                    break
                if not next_past.done():
                    raise RuntimeError("policy preparation race returned no route")
                remaining_past.popleft()
                submit_prepared_past(
                    next_past,
                    prior_wait_seconds=prior_wait_seconds,
                )
                counters["policy_host_prepare_past_bypasses"] += 1
                # The first CUDA prefix is queued. Fill the remaining current
                # host wait with independent CPU-only routes exactly once.
                serve_cpu_routes_once()

            current_wait_seconds = current_prior_wait_seconds
            if current_future is not None and not current_future.done():
                current_wait_seconds += _resume_routes_until_preparation_ready(
                    (cast(Future[object], current_future),),
                    pending_routes,
                )
            current = _wait_current_preparation(
                current_future,
                prior_wait_seconds=current_wait_seconds,
                timings=timings,
                counters=counters,
            )
            pending_current = begin_current(
                collector,
                current,
                generators=generators,
                delivery=delivery,
                timings=timings,
                counters=counters,
                artifact_batches=artifact_batches,
                artifact_rows=artifact_rows,
            )
            if pending_current is not None:
                pending_routes.append(pending_current)
            _resume_ready_routes(pending_routes)
            # Preserve current-first service whenever current was already
            # ready, including collectors without an exact past route.
            serve_cpu_routes_once()

            for future in remaining_past:
                prior_wait_seconds = _resume_routes_until_preparation_ready(
                    (cast(Future[object], future),),
                    pending_routes,
                )
                submit_prepared_past(
                    future,
                    prior_wait_seconds=prior_wait_seconds,
                )
            # Run the overlap callback before draining fact-gated routes.
            # Peer-bank preparation launches its own host encoding and engine
            # fact probes, so the next wave's fact latency resolves while this
            # wave waits on its own facts instead of strictly after them.
            if overlap is not None:
                overlap_started_at = time.perf_counter()
                overlap()
                explicit_overlap_seconds += time.perf_counter() - overlap_started_at
                overlap = None
            _drain_pending_routes(pending_routes)
        except BaseException:
            for pending in reversed(pending_routes):
                pending.cancel()
            raise
    try:
        counters["policy_host_prepare_batches"] += 1
        wave_elapsed = time.perf_counter() - wave_started_at
        submission_elapsed = max(wave_elapsed - explicit_overlap_seconds, 0.0)
        accounted_after = sum(
            timings[key]
            for key in ("input", "current", "past", "historical", "scripted")
        )
        exclusive_accounted = accounted_after - accounted_before
        route_overlap = max(exclusive_accounted - submission_elapsed, 0.0)
        if route_overlap:
            timings["policy_route_overlap"] += route_overlap
            counters["policy_route_overlap_waves"] += 1
        if policy_bank is not None:
            added_overlap = timings["policy_route_overlap"] - overlap_before
            added_prepare_wait = (
                timings["policy_host_prepare_wait"] - prepare_wait_before
            )
            timings["current"] += max(
                submission_elapsed
                - (exclusive_accounted - added_overlap)
                - added_prepare_wait,
                0.0,
            )
    except BaseException:
        _abort_untransferred_policy_ticket(policy_ticket)
        raise
    return policy_ticket


def _abort_untransferred_policy_ticket(
    ticket: NativePolicyInferenceWaveTicket | None,
) -> None:
    """Best-effort abort a sealed wave whose caller never received ownership."""
    if ticket is None:
        return
    with suppress(BaseException):
        ticket.abort()


def _wait_current_preparation(
    future: Future[_TimedCurrentPreparation] | None,
    *,
    prior_wait_seconds: float = 0.0,
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> PreparedNativeCurrent | None:
    """Join current host input after accounting owner waits raced elsewhere."""
    if future is None:
        return None
    wait_started_at = time.perf_counter()
    result = future.result()
    _record_route_preparation(
        prepare_seconds=(
            0.0 if result.prepared is None else result.prepared.prepare_seconds
        ),
        wall_seconds=result.finished_at - result.started_at,
        wait_seconds=(prior_wait_seconds + time.perf_counter() - wait_started_at),
        timings=timings,
        counters=counters,
    )
    return result.prepared


def _wait_past_preparation(
    future: Future[_TimedPastPreparation],
    *,
    prior_wait_seconds: float = 0.0,
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> PreparedNativePast:
    """Join one exact artifact in stable route order."""
    wait_started_at = time.perf_counter()
    result = future.result()
    _record_route_preparation(
        prepare_seconds=result.prepared.prepare_seconds,
        wall_seconds=result.finished_at - result.started_at,
        wait_seconds=(prior_wait_seconds + time.perf_counter() - wait_started_at),
        timings=timings,
        counters=counters,
    )
    return result.prepared


def _resume_routes_until_preparation_ready(
    futures: Sequence[Future[object]],
    pending_routes: deque[PendingNativeRouteSubmission],
) -> float:
    """Race host preparations against every fact-gated GPU route."""
    if not futures:
        raise ValueError("policy preparation race requires at least one future")
    wait_seconds = 0.0
    while not any(future.done() for future in futures):
        _resume_ready_routes(pending_routes)
        if any(future.done() for future in futures):
            break
        wait_targets = list(futures)
        for pending in pending_routes:
            ready_future = pending.ready_future
            if ready_future is None:
                raise RuntimeError("unready native route has no readiness future")
            wait_targets.append(cast(Future[object], ready_future))
        wait_started_at = time.perf_counter()
        wait(
            tuple(wait_targets),
            return_when=FIRST_COMPLETED,
        )
        wait_seconds += time.perf_counter() - wait_started_at
    return wait_seconds


def _resume_ready_routes(
    pending_routes: deque[PendingNativeRouteSubmission],
) -> None:
    """Resume every fact-ready route without waiting behind a slow sibling."""
    blocked: deque[PendingNativeRouteSubmission] = deque()
    while pending_routes:
        pending = pending_routes.popleft()
        if pending.ready():
            pending.resume()
        else:
            blocked.append(pending)
    pending_routes.extend(blocked)


def _drain_pending_routes(
    pending_routes: deque[PendingNativeRouteSubmission],
) -> None:
    """Wait by readiness, preserving order only among routes still blocked."""
    while pending_routes:
        _resume_ready_routes(pending_routes)
        if not pending_routes:
            return
        ready_futures: list[Future[object]] = []
        for pending in pending_routes:
            ready_future = pending.ready_future
            if ready_future is None:
                raise RuntimeError("unready native route has no readiness future")
            ready_futures.append(cast(Future[object], ready_future))
        wait(tuple(ready_futures), return_when=FIRST_COMPLETED)


def _record_route_preparation(
    *,
    prepare_seconds: float,
    wall_seconds: float,
    wait_seconds: float,
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> None:
    """Record work and the portion hidden behind GPU/other CPU routes."""
    timings["policy_host_prepare"] += prepare_seconds
    timings["policy_host_prepare_wall"] += wall_seconds
    timings["policy_host_prepare_wait"] += wait_seconds
    timings["policy_host_prepare_overlap"] += max(
        wall_seconds - wait_seconds,
        0.0,
    )
    counters["policy_host_prepare_tasks"] += 1


def _serve_policy_routes(
    collector: NativeStatelessCollector,
    dispatches: Sequence[NativeArenaDispatch],
    *,
    route_executor: ThreadPoolExecutor | None,
    generators: dict[NativeRouteKey, torch.Generator],
    delivery: NativePartDelivery | Mapping[int, NativePartDelivery],
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
    overlap: Callable[[], None] | None = None,
) -> None:
    """Synchronously prepare and submit one cohort with coordinator ownership."""
    prepared = _prepare_policy_route_wave(
        collector,
        dispatches,
        route_executor=route_executor,
    )
    _serve_prepared_policy_routes(
        collector,
        prepared,
        generators=generators,
        delivery=delivery,
        timings=timings,
        counters=counters,
        artifact_batches=artifact_batches,
        artifact_rows=artifact_rows,
        overlap=overlap,
    )


def _prepare_policy_route_wave(
    collector: NativeStatelessCollector,
    dispatches: Sequence[NativeArenaDispatch],
    *,
    route_executor: ThreadPoolExecutor | None,
    scripted_executor: ThreadPoolExecutor | None = None,
) -> _PreparedPolicyRouteWave:
    """Start host encoding and fact probes without touching actor or CUDA state."""
    selected = tuple(dispatches)
    scripted_submitted_at: float | None = None
    scripted_future: Future[NativePreparedScriptedActions] | None = None
    if scripted_executor is not None and any(
        dispatch.scripted_rows for dispatch in selected
    ):
        scripted_submitted_at = time.perf_counter()
        scripted_future = scripted_executor.submit(
            prepare_scripted_actions,
            collector,
            selected,
        )
    try:
        past_executors = resolve_past_executors(collector, selected)
        if route_executor is not None:
            return _PreparedPolicyRouteWave(
                dispatches=selected,
                serial=None,
                ticket=_submit_policy_preparation(
                    collector,
                    selected,
                    past_executors,
                    route_executor=route_executor,
                ),
                scripted_future=scripted_future,
                scripted_submitted_at=scripted_submitted_at,
            )
        return _PreparedPolicyRouteWave(
            dispatches=selected,
            serial=_prepare_policy_routes_serial(
                collector,
                selected,
                past_executors,
            ),
            ticket=None,
            scripted_future=scripted_future,
            scripted_submitted_at=scripted_submitted_at,
        )
    except BaseException:
        _abort_scripted_preparation_future(scripted_future)
        raise


def _serve_prepared_policy_routes(
    collector: NativeStatelessCollector,
    prepared_wave: _PreparedPolicyRouteWave,
    *,
    generators: dict[NativeRouteKey, torch.Generator],
    delivery: NativePartDelivery | Mapping[int, NativePartDelivery],
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
    overlap: Callable[[], None] | None = None,
    defer_completion: bool = False,
) -> NativePolicyInferenceWaveTicket | None:
    """Consume one prepared cohort on the sole actor and CUDA owner thread."""
    selected = prepared_wave.dispatches
    ticket = prepared_wave.ticket
    serial = prepared_wave.serial
    serve_scripted_inline = prepared_wave.scripted_future is None
    if ticket is not None:
        if serial is not None:
            raise RuntimeError("policy route wave has two preparation owners")
        policy_ticket = _submit_policy_preparation_ticket(
            collector,
            selected,
            ticket,
            generators=generators,
            delivery=delivery,
            timings=timings,
            counters=counters,
            artifact_batches=artifact_batches,
            artifact_rows=artifact_rows,
            overlap=overlap,
            defer_completion=defer_completion,
            serve_scripted_inline=serve_scripted_inline,
        )
    else:
        if serial is None:
            raise RuntimeError("policy route wave has no prepared inputs")
        timings["policy_host_prepare"] += serial.prepare_work_seconds
        timings["policy_host_prepare_wall"] += serial.prepare_wall_seconds
        timings["policy_host_prepare_wait"] += serial.prepare_wall_seconds
        counters["policy_host_prepare_batches"] += 1
        counters["policy_host_prepare_tasks"] += serial.prepare_task_count
        policy_ticket = _submit_policy_routes(
            collector,
            selected,
            serial,
            generators=generators,
            delivery=delivery,
            timings=timings,
            counters=counters,
            artifact_batches=artifact_batches,
            artifact_rows=artifact_rows,
            overlap=overlap,
            defer_completion=defer_completion,
            serve_scripted_inline=serve_scripted_inline,
        )
    if prepared_wave.scripted_future is not None and not defer_completion:
        _finish_prepared_scripted_routes(
            prepared_wave,
            timings=timings,
            counters=counters,
        )
    return policy_ticket


def _finish_prepared_scripted_routes(
    prepared_wave: _PreparedPolicyRouteWave,
    *,
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> None:
    """Join prefetched scripted work and publish actions on the owner thread."""
    future = prepared_wave.scripted_future
    submitted_at = prepared_wave.scripted_submitted_at
    if future is None:
        if submitted_at is not None:
            raise RuntimeError("scripted submission time has no future")
        return
    if submitted_at is None:
        raise RuntimeError("scripted future has no submission time")
    wait_started_at = time.perf_counter()
    scripted = future.result()
    wait_seconds = time.perf_counter() - wait_started_at
    apply_prepared_scripted_actions(scripted, timings=timings)
    timings["scripted_prepare_wait"] += wait_seconds
    timings["scripted_prepare_queue"] += max(
        scripted.started_at - submitted_at,
        0.0,
    )
    timings["scripted_prepare_overlap"] += max(
        scripted.finished_at - submitted_at - wait_seconds,
        0.0,
    )
    counters["scripted_prepare_batches"] += len(scripted.batches)
    counters["scripted_prepare_rows"] += scripted.rows


def _abort_prepared_policy_routes(
    prepared_wave: _PreparedPolicyRouteWave,
) -> None:
    """Drain host and fact readers before their native arena may be closed."""
    _abort_scripted_preparation_future(prepared_wave.scripted_future)
    if prepared_wave.serial is not None:
        _drain_prepared_policy_facts(
            prepared_wave.serial.current,
            prepared_wave.serial.past,
        )
    ticket = prepared_wave.ticket
    if ticket is None:
        return
    _abort_policy_preparation_ticket(ticket)


def _abort_scripted_preparation_future(
    future: Future[NativePreparedScriptedActions] | None,
) -> None:
    """Cancel queued scripted work or join it before releasing its arena."""
    if future is None or future.cancel():
        return
    with suppress(BaseException):
        future.result()


def _abort_policy_preparation_ticket(
    ticket: _PolicyPreparationTicket,
) -> None:
    """Drain one host-preparation queue and every nested native fact reader."""
    if ticket.current_future is not None:
        current_future = ticket.current_future
        if not current_future.cancel():
            with suppress(BaseException):
                current_result = current_future.result()
                _drain_prepared_policy_facts(current_result.prepared, None)
    for past_future in ticket.past_futures:
        if past_future.cancel():
            continue
        with suppress(BaseException):
            past_result = past_future.result()
            _drain_prepared_policy_facts(None, past_result.prepared)


def _drain_prepared_policy_facts(
    current: PreparedNativeCurrent | None,
    past: PreparedNativePast | None,
) -> None:
    """Cancel a queued fact probe or join it if native reads already started."""
    futures = []
    if current is not None and current.fact_future is not None:
        futures.append(current.fact_future)
    if past is not None:
        futures.extend(
            route.fact_future for route in past.routes if route.fact_future is not None
        )
    for future in futures:
        if future.cancel():
            continue
        with suppress(BaseException):
            future.result()


def collect_native_route_assigned(
    collector: NativeStatelessCollector,
    assignments: Sequence[StatelessAssignedGame],
    *,
    cancellation_event: threading.Event | None = None,
) -> StatelessCollectionResult:
    """Collect one immutable PPO lease through one global inference owner."""
    if not assignments:
        raise ValueError("native collection requires assigned games")
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
    total_capacity = min(
        collector.arena_capacity or len(prepared),
        len(prepared),
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
    # Resolve and validate every immutable inference route before mutating the
    # source engine. The engine lane itself is policy-agnostic; ready rows are
    # routed by their live assignment below.
    for index in range(len(assignments)):
        collector._cohort_group_key(prepared[index])
    shard_count = min(
        collector.engine_shards,
        total_capacity,
        len(prepared),
    )
    if shard_count > 1:
        return _collect_sharded_route_assigned(
            collector,
            assignments,
            prepared=prepared,
            total_capacity=total_capacity,
            shard_count=shard_count,
            cancellation_event=cancellation_event,
            started_at=started_at,
        )
    delivery = NativePartDelivery(
        part_sink=getattr(collector, "compact_part_sink", None)
    )
    outcomes: list[StatelessGameOutcome] = []
    lane_scores: defaultdict[str, list[float]] = defaultdict(list)
    timings: defaultdict[str, float] = defaultdict(float)
    counters: Counter[str] = Counter()
    artifact_batches: Counter[NativeRouteKey] = Counter()
    artifact_rows: Counter[NativeRouteKey] = Counter()
    generators: dict[NativeRouteKey, torch.Generator] = {}
    frozen_scheduler = _frozen_batch_scheduler(collector)
    arena: NativeRouteArena | None = None

    with ExitStack() as lane_stack:
        try:
            route_executor = _route_inference_executor(collector)
            if route_executor is not None:
                lane_stack.enter_context(route_executor)
            arena = start_native_arena(
                collector,
                prepared,
                capacity=total_capacity,
                lane_stack=lane_stack,
                timings=timings,
                counters=counters,
            )
            while arena.live:
                _require_not_cancelled(cancellation_event)
                _begin_decision_budget_drain(
                    collector,
                    ((arena, outcomes, counters),),
                    trainable_decisions=_net_trainable_decisions(
                        counters,
                        (counters,),
                    ),
                )
                (dispatch,) = prepare_arena_dispatches(
                    collector,
                    (arena,),
                    counters=counters,
                    frozen_scheduler=frozen_scheduler,
                )
                dispatches = (dispatch,)
                _serve_policy_routes(
                    collector,
                    dispatches,
                    route_executor=route_executor,
                    generators=generators,
                    delivery=delivery,
                    timings=timings,
                    counters=counters,
                    artifact_batches=artifact_batches,
                    artifact_rows=artifact_rows,
                )
                _begin_decision_budget_drain(
                    collector,
                    ((arena, outcomes, counters),),
                    trainable_decisions=_net_trainable_decisions(
                        counters,
                        (counters,),
                    ),
                )
                _require_not_cancelled(cancellation_event)
                advance_route_arena(
                    collector,
                    dispatch,
                    delivery=delivery,
                    outcomes=outcomes,
                    lane_scores=lane_scores,
                    timings=timings,
                    counters=counters,
                )
            finish_native_arena(arena, delivery=delivery)
        except BaseException:
            if arena is not None:
                release_arena_scripted(collector, arena)
            raise

    elapsed = time.perf_counter() - started_at
    measured_non_engine = (
        timings["input"]
        + timings["current"]
        + timings["past"]
        + timings["historical"]
        + timings["scripted"]
        + timings["policy_host_prepare_wait"]
        - timings["policy_route_overlap"]
    )
    timings["engine_control"] = max(elapsed - measured_non_engine, 0.0)
    return _collection_result(
        assignments,
        delivery=delivery,
        outcomes=outcomes,
        lane_scores=lane_scores,
        timings=timings,
        counters=counters,
        started_at=started_at,
        native_engine_arenas=1,
        trainable_decision_budget=getattr(
            collector,
            "trainable_decision_budget",
            None,
        ),
        artifact_batches=artifact_batches,
        artifact_rows=artifact_rows,
    )


def _collect_sharded_route_assigned(
    collector: NativeStatelessCollector,
    assignments: Sequence[StatelessAssignedGame],
    *,
    prepared: Mapping[int, NativeLiveGame],
    total_capacity: int,
    shard_count: int,
    cancellation_event: threading.Event | None,
    started_at: float,
) -> StatelessCollectionResult:
    """Continuously feed one GPU owner from independently advancing arenas."""
    prepared_shards = _partition_prepared_games(
        prepared,
        shard_count=shard_count,
    )
    capacities = _partition_capacity(
        total_capacity,
        shard_count=shard_count,
    )
    states: list[_RouteShardState] = []
    artifact_batches: Counter[NativeRouteKey] = Counter()
    artifact_rows: Counter[NativeRouteKey] = Counter()
    generators: dict[NativeRouteKey, torch.Generator] = {}
    frozen_scheduler = _frozen_batch_scheduler(collector)
    coordinator_timings: defaultdict[str, float] = defaultdict(float)
    coordinator_counters: Counter[str] = Counter()

    with ExitStack() as lane_stack:
        try:
            route_executor = _route_inference_executor(collector)
            if route_executor is not None:
                lane_stack.enter_context(route_executor)
            for shard_index, (shard_prepared, capacity) in enumerate(
                zip(prepared_shards, capacities, strict=True)
            ):
                shard_timings: defaultdict[str, float] = defaultdict(float)
                shard_counters: Counter[str] = Counter()
                states.append(
                    _RouteShardState(
                        shard_index=shard_index,
                        arena=start_native_arena(
                            collector,
                            shard_prepared,
                            capacity=capacity,
                            lane_stack=lane_stack,
                            timings=shard_timings,
                            counters=shard_counters,
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
                        timings=shard_timings,
                        counters=shard_counters,
                    )
                )

            with ThreadPoolExecutor(
                max_workers=shard_count,
                thread_name_prefix="native-arena",
            ) as executor:
                ready = {state.shard_index: state for state in states}
                inflight: dict[
                    Future[NativeTrainingBatchView],
                    _RouteShardState,
                ] = {}
                configured_cohort_slots = getattr(
                    collector,
                    "policy_cohort_slots",
                    None,
                )
                cohort_slot_limit = (
                    total_capacity
                    if configured_cohort_slots is None
                    else int(configured_cohort_slots)
                )
                prepared_cohorts: deque[_PolicyCohort] = deque()
                coordinator_timings["startup"] += time.perf_counter() - started_at
                while ready or inflight or prepared_cohorts:
                    _require_not_cancelled(cancellation_event)

                    completed_advances = tuple(
                        future for future in inflight if future.done()
                    )
                    if completed_advances:
                        _harvest_completed_advances(
                            collector,
                            completed_advances,
                            inflight=inflight,
                            ready=ready,
                        )
                    if not prepared_cohorts:
                        _fence_decision_budget_if_required(
                            collector,
                            states,
                            inflight=inflight,
                            ready=ready,
                            coordinator_timings=coordinator_timings,
                            coordinator_counters=coordinator_counters,
                        )

                    if not ready and not prepared_cohorts:
                        if not inflight:
                            break
                        feed_wait_started_at = time.perf_counter()
                        completed, _pending = wait(
                            tuple(inflight),
                            return_when=FIRST_COMPLETED,
                        )
                        coordinator_timings["gpu_feed_wait"] += (
                            time.perf_counter() - feed_wait_started_at
                        )
                        coordinator_counters["gpu_feed_wait_events"] += 1
                        completed = {
                            future for future in inflight if future.done()
                        } | completed
                        _harvest_completed_advances(
                            collector,
                            tuple(completed),
                            inflight=inflight,
                            ready=ready,
                        )
                        continue

                    if route_executor is None:
                        _coalesce_ready_cohort(
                            collector,
                            ready,
                            inflight=inflight,
                            slot_limit=cohort_slot_limit,
                            timings=coordinator_timings,
                            counters=coordinator_counters,
                        )
                        _fence_decision_budget_if_required(
                            collector,
                            states,
                            inflight=inflight,
                            ready=ready,
                            coordinator_timings=coordinator_timings,
                            coordinator_counters=coordinator_counters,
                        )
                        cohort = _take_ready_cohort(
                            ready,
                            slot_limit=cohort_slot_limit,
                        )
                        dispatches = _prepare_policy_cohort_dispatches(
                            collector,
                            cohort,
                            counters=coordinator_counters,
                            frozen_scheduler=frozen_scheduler,
                        )
                        _serve_policy_routes(
                            collector,
                            dispatches,
                            route_executor=None,
                            generators=generators,
                            delivery={
                                id(state.arena): state.delivery for state in cohort
                            },
                            timings=coordinator_timings,
                            counters=coordinator_counters,
                            artifact_batches=artifact_batches,
                            artifact_rows=artifact_rows,
                        )
                    else:
                        if not prepared_cohorts:
                            _coalesce_ready_cohort(
                                collector,
                                ready,
                                inflight=inflight,
                                slot_limit=cohort_slot_limit,
                                timings=coordinator_timings,
                                counters=coordinator_counters,
                            )
                            _fence_decision_budget_if_required(
                                collector,
                                states,
                                inflight=inflight,
                                ready=ready,
                                coordinator_timings=coordinator_timings,
                                coordinator_counters=coordinator_counters,
                            )
                            prepared_cohorts.append(
                                _enqueue_policy_cohort(
                                    collector,
                                    ready,
                                    route_executor=route_executor,
                                    slot_limit=cohort_slot_limit,
                                    counters=coordinator_counters,
                                    frozen_scheduler=frozen_scheduler,
                                )
                            )
                        trainable_decisions = _net_trainable_decisions(
                            coordinator_counters,
                            tuple(state.counters for state in states),
                        )
                        if (
                            len(prepared_cohorts) < 2
                            and ready
                            and _policy_prefetch_allowed(
                                collector,
                                prepared_cohorts,
                                trainable_decisions=trainable_decisions,
                                cohort_slot_limit=cohort_slot_limit,
                            )
                        ):
                            prepared_cohorts.append(
                                _enqueue_policy_cohort(
                                    collector,
                                    ready,
                                    route_executor=route_executor,
                                    slot_limit=cohort_slot_limit,
                                    counters=coordinator_counters,
                                    frozen_scheduler=frozen_scheduler,
                                )
                            )
                            coordinator_counters["policy_host_prefetch_batches"] += 1
                        selected_cohort = prepared_cohorts.popleft()
                        cohort = selected_cohort.states
                        dispatches = selected_cohort.dispatches
                        _submit_policy_preparation_ticket(
                            collector,
                            dispatches,
                            selected_cohort.preparation,
                            generators=generators,
                            delivery={
                                id(state.arena): state.delivery for state in cohort
                            },
                            timings=coordinator_timings,
                            counters=coordinator_counters,
                            artifact_batches=artifact_batches,
                            artifact_rows=artifact_rows,
                        )
                    _require_not_cancelled(cancellation_event)
                    for state, dispatch in zip(
                        cohort,
                        dispatches,
                        strict=True,
                    ):
                        prepared_advance = prepare_route_arena_advance(
                            dispatch,
                            timings=state.timings,
                        )
                        if prepared_advance is None:
                            ready[state.shard_index] = state
                            continue
                        state.prepared_advance = prepared_advance
                        state.advance_started_at = time.perf_counter()
                        future = executor.submit(
                            run_route_arena_engine_step,
                            prepared_advance.engine_step,
                        )
                        state.advance_future = future
                        inflight[future] = state
        except BaseException:
            for state in states:
                if state.prepared_advance is not None:
                    abort_route_arena_advance(state.prepared_advance)
                    state.prepared_advance = None
                    state.advance_future = None
                release_arena_scripted(collector, state.arena)
            raise

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
        - timings["policy_route_overlap"]
    )
    timings["engine_control"] = max(elapsed - measured_non_engine, 0.0)
    return _collection_result(
        assignments,
        delivery=delivery,
        outcomes=outcomes,
        lane_scores=lane_scores,
        timings=timings,
        counters=counters,
        started_at=started_at,
        native_engine_arenas=shard_count,
        trainable_decision_budget=getattr(
            collector,
            "trainable_decision_budget",
            None,
        ),
        artifact_batches=artifact_batches,
        artifact_rows=artifact_rows,
    )


def _take_ready_cohort(
    ready: dict[int, _RouteShardState],
    *,
    slot_limit: int,
) -> tuple[_RouteShardState, ...]:
    """Take a fair ready prefix without creating another all-shard barrier."""
    if slot_limit <= 0:
        raise ValueError("native policy cohort slot limit must be positive")
    selected: list[_RouteShardState] = []
    selected_rows = 0
    ordered = sorted(
        ready.values(),
        key=lambda state: (state.arena.batch_index, state.shard_index),
    )
    for state in ordered:
        rows = int(state.arena.view.batch_size)
        if rows <= 0:
            raise RuntimeError("ready native arena has no policy rows")
        if selected and selected_rows + rows > slot_limit:
            break
        selected.append(state)
        selected_rows += rows
        if selected_rows >= slot_limit:
            break
    if not selected:
        raise RuntimeError("native feeder failed to select a ready arena")
    for state in selected:
        removed = ready.pop(state.shard_index)
        if removed is not state:
            raise RuntimeError("native ready arena identity changed")
        if state.advance_future is not None or state.prepared_advance is not None:
            raise RuntimeError("native ready arena still has an in-flight advance")
    return tuple(selected)


def _coalesce_ready_cohort(
    collector: NativeStatelessCollector,
    ready: dict[int, _RouteShardState],
    *,
    inflight: dict[Future[NativeTrainingBatchView], _RouteShardState],
    slot_limit: int,
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> None:
    """Briefly gather near-simultaneous shards without restoring a barrier."""
    wait_ms = float(getattr(collector, "policy_cohort_wait_ms", 0.0))
    if wait_ms <= 0.0 or not ready or not inflight:
        return
    ready_rows = sum(int(state.arena.view.batch_size) for state in ready.values())
    if ready_rows >= slot_limit:
        return
    started_at = time.perf_counter()
    deadline = started_at + wait_ms / 1000.0
    harvested = False
    while inflight and ready_rows < slot_limit:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            break
        completed, _pending = wait(
            tuple(inflight),
            timeout=remaining,
            return_when=FIRST_COMPLETED,
        )
        if not completed:
            break
        completed = {future for future in inflight if future.done()} | completed
        _harvest_completed_advances(
            collector,
            tuple(completed),
            inflight=inflight,
            ready=ready,
        )
        harvested = True
        ready_rows = sum(int(state.arena.view.batch_size) for state in ready.values())
    timings["policy_cohort_wait"] += time.perf_counter() - started_at
    counters["policy_cohort_wait_events"] += 1
    if harvested:
        counters["policy_cohort_wait_harvests"] += 1


def _prepare_policy_cohort_dispatches(
    collector: NativeStatelessCollector,
    cohort: tuple[_RouteShardState, ...],
    *,
    counters: Counter[str],
    frozen_scheduler: NativeFrozenBatchScheduler | None,
) -> tuple[NativeArenaDispatch, ...]:
    """Create immutable dispatches and account one selected policy cohort."""
    dispatches = prepare_arena_dispatches(
        collector,
        tuple(state.arena for state in cohort),
        counters=counters,
        frozen_scheduler=frozen_scheduler,
    )
    cohort_rows = sum(int(dispatch.arena.view.batch_size) for dispatch in dispatches)
    counters["policy_cohort_batches"] += 1
    counters["policy_cohort_rows"] += cohort_rows
    counters["policy_cohort_max_rows"] = max(
        counters["policy_cohort_max_rows"],
        cohort_rows,
    )
    return dispatches


def _enqueue_policy_cohort(
    collector: NativeStatelessCollector,
    ready: dict[int, _RouteShardState],
    *,
    route_executor: ThreadPoolExecutor,
    slot_limit: int,
    counters: Counter[str],
    frozen_scheduler: NativeFrozenBatchScheduler | None,
) -> _PolicyCohort:
    """Freeze one ready cohort and enqueue only its CPU host preparation."""
    cohort = _take_ready_cohort(
        ready,
        slot_limit=slot_limit,
    )
    dispatches = _prepare_policy_cohort_dispatches(
        collector,
        cohort,
        counters=counters,
        frozen_scheduler=frozen_scheduler,
    )
    # This call may construct a lazy BF16 shadow and therefore deliberately
    # remains on the coordinator. The worker only reads the resolved actors.
    past_executors = resolve_past_executors(collector, dispatches)
    return _PolicyCohort(
        states=cohort,
        dispatches=dispatches,
        preparation=_submit_policy_preparation(
            collector,
            dispatches,
            past_executors,
            route_executor=route_executor,
        ),
    )


def _policy_prefetch_allowed(
    collector: NativeStatelessCollector,
    prepared_cohorts: Sequence[_PolicyCohort],
    *,
    trainable_decisions: int,
    cohort_slot_limit: int,
) -> bool:
    """Avoid preparing another full cohort near the decision-budget tail."""
    budget = getattr(collector, "trainable_decision_budget", None)
    if budget is None:
        return True
    queued_current_rows = sum(
        int(dispatch.current_rows.size)
        for cohort in prepared_cohorts
        for dispatch in cohort.dispatches
    )
    remaining = int(budget) - trainable_decisions
    return remaining > queued_current_rows + cohort_slot_limit


def _harvest_completed_advances(
    collector: NativeStatelessCollector,
    futures: Sequence[Future[NativeTrainingBatchView]],
    *,
    inflight: dict[Future[NativeTrainingBatchView], _RouteShardState],
    ready: dict[int, _RouteShardState],
) -> None:
    """Finalize completed private engine steps on the GPU-owner thread."""
    selected = sorted(
        set(futures),
        key=lambda future: inflight[future].shard_index,
    )
    for future in selected:
        state = inflight.pop(future)
        prepared = state.prepared_advance
        if prepared is None or state.advance_future is not future:
            raise RuntimeError("native engine future lost its prepared advance")
        try:
            next_view = future.result()
        except BaseException:
            abort_route_arena_advance(prepared)
            state.prepared_advance = None
            state.advance_future = None
            raise
        state.timings["engine"] += max(
            time.perf_counter() - state.advance_started_at,
            0.0,
        )
        finish_route_arena_advance(
            collector,
            prepared,
            next_view,
            delivery=state.delivery,
            outcomes=state.outcomes,
            lane_scores=state.lane_scores,
            timings=state.timings,
            counters=state.counters,
        )
        state.prepared_advance = None
        state.advance_future = None
        state.advance_started_at = 0.0
        if state.arena.live:
            if state.shard_index in ready:
                raise RuntimeError("native arena became ready more than once")
            ready[state.shard_index] = state
            continue
        finish_native_arena(
            state.arena,
            delivery=state.delivery,
        )
        state.finished = True


def _external_drain_requested(collector: NativeStatelessCollector) -> bool:
    """Return whether the coordinator advised finishing this window early."""
    signal = getattr(collector, "external_drain_signal", None)
    return bool(signal is not None and signal())


def _decision_budget_fence_required(
    collector: NativeStatelessCollector,
    states: Sequence[_RouteShardState],
    *,
    trainable_decisions: int,
) -> bool:
    """Fence once at the bounded-window tail before globally latching drain."""
    budget = getattr(collector, "trainable_decision_budget", None)
    budget_reached = budget is not None and trainable_decisions >= budget
    return bool(
        (budget_reached or _external_drain_requested(collector))
        and states
        and any(not state.arena.window_draining for state in states)
    )


def _fence_decision_budget_if_required(
    collector: NativeStatelessCollector,
    states: Sequence[_RouteShardState],
    *,
    inflight: dict[Future[NativeTrainingBatchView], _RouteShardState],
    ready: dict[int, _RouteShardState],
    coordinator_timings: defaultdict[str, float],
    coordinator_counters: Counter[str],
) -> bool:
    """Fence engine futures and globally latch drain at the budget boundary."""
    unfinished = tuple(state for state in states if not state.finished)
    trainable_decisions = _net_trainable_decisions(
        coordinator_counters,
        tuple(state.counters for state in states),
    )
    if not _decision_budget_fence_required(
        collector,
        unfinished,
        trainable_decisions=trainable_decisions,
    ):
        return False
    if inflight:
        fence_started_at = time.perf_counter()
        _harvest_completed_advances(
            collector,
            tuple(inflight),
            inflight=inflight,
            ready=ready,
        )
        coordinator_timings["budget_fence_wait"] += (
            time.perf_counter() - fence_started_at
        )
        unfinished = tuple(state for state in states if not state.finished)
        trainable_decisions = _net_trainable_decisions(
            coordinator_counters,
            tuple(state.counters for state in states),
        )
    return _begin_decision_budget_drain(
        collector,
        tuple((state.arena, state.outcomes, state.counters) for state in unfinished),
        trainable_decisions=trainable_decisions,
    )


def _partition_prepared_games(
    prepared: Mapping[int, NativeLiveGame],
    *,
    shard_count: int,
) -> tuple[dict[int, NativeLiveGame], ...]:
    """Stably stripe lease-ordered games into balanced mixed arenas."""
    if shard_count <= 0 or shard_count > len(prepared):
        raise ValueError("native engine shard count is invalid")
    if tuple(prepared) != tuple(range(len(prepared))):
        raise ValueError("native prepared games are not lease ordered")
    groups: list[list[NativeLiveGame]] = [[] for _index in range(shard_count)]
    for assignment_index, game in prepared.items():
        groups[assignment_index % shard_count].append(game)
    return tuple(dict(enumerate(group)) for group in groups)


def _partition_capacity(
    total_capacity: int,
    *,
    shard_count: int,
) -> tuple[int, ...]:
    """Split one global live-slot budget across non-empty engine shards."""
    if shard_count <= 0 or total_capacity < shard_count:
        raise ValueError("native engine shard capacity is invalid")
    base, remainder = divmod(total_capacity, shard_count)
    return tuple(
        base + int(shard_index < remainder) for shard_index in range(shard_count)
    )


def _require_not_cancelled(event: threading.Event | None) -> None:
    if event is not None and event.is_set():
        raise RuntimeError("native arena collection was cancelled")


def _frozen_batch_scheduler(
    collector: NativeStatelessCollector,
) -> NativeFrozenBatchScheduler | None:
    """Create one per-window release scheduler when batching is enabled."""
    min_rows = int(getattr(collector, "frozen_batch_min_rows", 1))
    max_wait_waves = int(getattr(collector, "frozen_batch_max_wait_waves", 1))
    if min_rows <= 1:
        return None
    return NativeFrozenBatchScheduler(
        min_rows=min_rows,
        max_wait_waves=max_wait_waves,
    )


def _begin_decision_budget_drain(
    collector: NativeStatelessCollector,
    arenas: Sequence[
        tuple[
            NativeRouteArena,
            list[StatelessGameOutcome],
            Counter[str],
        ]
    ],
    *,
    trainable_decisions: int,
) -> bool:
    """Latch active games so cutoff data can be rolled back by game."""
    budget = getattr(collector, "trainable_decision_budget", None)
    budget_reached = budget is not None and trainable_decisions >= budget
    if not budget_reached and not _external_drain_requested(collector):
        return False
    changed = False
    for arena, outcomes, counters in arenas:
        if arena.window_draining:
            continue
        begin_arena_window_drain(
            collector,
            arena,
            outcomes=outcomes,
            counters=counters,
            immediate_whole_game_cutoff=bool(
                getattr(
                    collector,
                    "immediate_whole_game_cutoff_on_drain",
                    False,
                )
            ),
        )
        changed = True
    return changed


def _net_trainable_decisions(
    appended_counters: Counter[str],
    discard_counters: Sequence[Counter[str]],
) -> int:
    """Return appended rows after subtracting watchdog-discarded open tails."""
    appended = int(appended_counters["trainable_trajectory_rows_appended"])
    discarded = sum(
        int(counters["trainable_trajectory_rows_discarded"])
        for counters in discard_counters
    )
    retained = appended - discarded
    if retained < 0:
        raise RuntimeError("native discarded trajectory rows exceed appended rows")
    return retained


def _collection_result(
    assignments: Sequence[StatelessAssignedGame],
    *,
    delivery: NativePartDelivery,
    outcomes: list[StatelessGameOutcome],
    lane_scores: Mapping[str, list[float]],
    timings: Mapping[str, float],
    counters: Counter[str],
    started_at: float,
    native_engine_arenas: int,
    trainable_decision_budget: int | None = None,
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
) -> StatelessCollectionResult:
    elapsed = max(time.perf_counter() - started_at, 1.0e-9)
    terminal_outcomes = tuple(
        outcome for outcome in outcomes if outcome.status == "engine_terminal"
    )
    step_limit_outcomes = tuple(
        outcome for outcome in outcomes if outcome.status == "step_limit"
    )
    window_cutoff_outcomes = tuple(
        outcome for outcome in outcomes if outcome.status == "window_cutoff"
    )
    nonterminal_outcomes = step_limit_outcomes + window_cutoff_outcomes
    if any(outcome.candidate_decisions for outcome in nonterminal_outcomes):
        raise RuntimeError("native nonterminal outcome retained candidate decisions")
    if any(outcome.status == "infrastructure_error" for outcome in outcomes):
        raise RuntimeError(
            "successful native collection cannot publish infrastructure errors"
        )
    nonterminal_assignment_ids = frozenset(
        outcome.curriculum_assignment_id for outcome in nonterminal_outcomes
    )
    if nonterminal_assignment_ids:
        # Parts may have streamed before a later cutoff revokes their games.
        # Apply the same final assignment transaction as the coordinator so
        # the worker's result counts describe only engine-terminal games.
        terminal_delivery = NativePartDelivery()
        for part in delivery.retained:
            retained = discard_compact_assignments(
                part,
                tuple(nonterminal_assignment_ids),
            )
            if retained is not None:
                terminal_delivery.accept((retained,))
        delivery = terminal_delivery
    trajectory_decisions = delivery.decision_count
    retained_candidate_decisions = sum(
        outcome.candidate_decisions for outcome in outcomes
    )
    candidate_decisions = int(counters["candidate_trajectory_rows"]) - int(
        counters["candidate_trajectory_rows_discarded"]
    )
    mirror_decisions = int(counters["mirror_opponent_trajectory_rows"]) - int(
        counters["mirror_opponent_trajectory_rows_discarded"]
    )
    if candidate_decisions < 0 or mirror_decisions < 0:
        raise RuntimeError("native cutoff rollback exceeds collected rows")
    expected_outcome_ids = {
        assignment.curriculum.assignment_id for assignment in assignments
    }
    if len(expected_outcome_ids) != len(assignments):
        raise RuntimeError("native route collection received duplicate reservations")
    actual_outcome_ids = Counter(
        outcome.curriculum_assignment_id for outcome in outcomes
    )
    if (
        any(count != 1 for count in actual_outcome_ids.values())
        or not set(actual_outcome_ids) <= expected_outcome_ids
    ):
        raise RuntimeError("native route collection duplicated or replaced outcomes")
    started_assignments = tuple(
        assignment
        for assignment in assignments
        if assignment.curriculum.assignment_id in actual_outcome_ids
    )
    games_started = int(counters["games_started"]) or len(started_assignments)
    if games_started != len(started_assignments):
        raise RuntimeError("native started-game count differs from terminal outcomes")
    released_reservations = len(assignments) - games_started
    observed_released = int(counters["unstarted_reservations_released"])
    if observed_released and observed_released != released_reservations:
        raise RuntimeError("native unstarted reservation count differs")
    if retained_candidate_decisions != candidate_decisions:
        raise RuntimeError(
            "native delivered candidate decisions differ from retained outcomes"
        )
    if candidate_decisions + mirror_decisions != trajectory_decisions:
        raise RuntimeError(
            "native trainable decisions differ from trajectory delivery: "
            f"candidate={candidate_decisions}, mirror={mirror_decisions}, "
            f"delivery={trajectory_decisions}"
        )
    if set(artifact_batches) != set(artifact_rows):
        raise RuntimeError("native artifact batch and row telemetry differ")
    outcome_order = {
        assignment.curriculum.assignment_id: index
        for index, assignment in enumerate(started_assignments)
    }
    outcomes.sort(key=lambda outcome: outcome_order[outcome.curriculum_assignment_id])
    lane_games = Counter(
        assignment.curriculum.lane for assignment in started_assignments
    )
    seat_games = Counter(
        str(assignment.balance.seat) for assignment in started_assignments
    )
    member_games = Counter(
        assignment.curriculum.member_id
        for assignment in started_assignments
        if assignment.curriculum.member_id
    )
    artifact_inference = tuple(
        NativeArtifactInferenceReport(
            route_kind=_inference_route_kind(route),
            artifact_sha256=route.artifact_sha256,
            batches=artifact_batches[route],
            rows=artifact_rows[route],
        )
        for route in sorted(artifact_batches)
    )
    decision_budget_reached = (
        trainable_decision_budget is not None
        and trajectory_decisions >= trainable_decision_budget
    )
    return StatelessCollectionResult(
        fragments=(),
        compact_parts=delivery.retained,
        compact_part_paths=(),
        assignments=started_assignments,
        outcomes=tuple(outcomes),
        report=StatelessCollectionReport(
            games_started=games_started,
            assignment_reservations=len(assignments),
            unstarted_reservations_released=released_reservations,
            games_finished=len(terminal_outcomes),
            games_cancelled=len(step_limit_outcomes) + len(window_cutoff_outcomes),
            games_window_cutoff=len(window_cutoff_outcomes),
            games_immediate_window_cutoff=int(
                counters["immediate_window_cutoff_games"]
            ),
            native_engine_arenas=native_engine_arenas,
            engine_steps=int(counters["engine_steps"]),
            candidate_decisions=trajectory_decisions,
            mirror_opponent_decisions=mirror_decisions,
            native_trainable_decisions=trajectory_decisions,
            native_trainable_decision_budget=trainable_decision_budget,
            native_trainable_decision_budget_reached=decision_budget_reached,
            native_trainable_decision_budget_overshoot=(
                max(trajectory_decisions - trainable_decision_budget, 0)
                if trainable_decision_budget is not None
                else 0
            ),
            fragments=delivery.fragment_count,
            elapsed_seconds=elapsed,
            decisions_per_second=trajectory_decisions / elapsed,
            native_phase_seconds=elapsed,
            input_seconds=timings["input"],
            current_policy_seconds=timings["current"],
            past_self_policy_seconds=timings["past"],
            historical_policy_seconds=timings["historical"],
            scripted_policy_seconds=timings["scripted"],
            native_policy_route_overlap_seconds=timings["policy_route_overlap"],
            native_policy_route_overlap_waves=int(
                counters["policy_route_overlap_waves"]
            ),
            native_policy_cohort_batches=int(counters["policy_cohort_batches"]),
            native_policy_cohort_rows=int(counters["policy_cohort_rows"]),
            native_policy_cohort_max_rows=int(counters["policy_cohort_max_rows"]),
            native_policy_cohort_wait_seconds=timings["policy_cohort_wait"],
            native_policy_cohort_wait_events=int(counters["policy_cohort_wait_events"]),
            native_policy_cohort_wait_harvests=int(
                counters["policy_cohort_wait_harvests"]
            ),
            native_gpu_feed_wait_seconds=timings["gpu_feed_wait"],
            native_gpu_feed_wait_events=int(counters["gpu_feed_wait_events"]),
            native_policy_host_prepare_seconds=timings["policy_host_prepare"],
            native_policy_host_prepare_wall_seconds=timings["policy_host_prepare_wall"],
            native_policy_host_prepare_wait_seconds=timings["policy_host_prepare_wait"],
            native_policy_host_prepare_overlap_seconds=timings[
                "policy_host_prepare_overlap"
            ],
            native_policy_host_prepare_past_bypasses=int(
                counters["policy_host_prepare_past_bypasses"]
            ),
            native_policy_completion_wait_seconds=timings["policy_completion_wait"],
            native_scripted_prefetch_wait_seconds=timings["scripted_prepare_wait"],
            native_scripted_prefetch_queue_seconds=timings["scripted_prepare_queue"],
            native_scripted_prefetch_overlap_seconds=timings[
                "scripted_prepare_overlap"
            ],
            native_scripted_prefetch_batches=int(counters["scripted_prepare_batches"]),
            native_scripted_prefetch_rows=int(counters["scripted_prepare_rows"]),
            native_bank_engine_wait_seconds=timings["bank_engine_wait"],
            native_bank_policy_overlap_seconds=timings["bank_policy_overlap"],
            native_bank_policy_prefetches=int(counters["bank_policy_prefetches"]),
            native_bank_engine_barriers=int(counters["bank_engine_barriers"]),
            native_bank_policy_groups=int(counters["bank_policy_groups"]),
            native_bank_policy_group_members=int(counters["bank_policy_group_members"]),
            native_bank_policy_group_max_size=int(
                counters["bank_policy_group_max_size"]
            ),
            native_bank_policy_coalescing_misses=int(
                counters["bank_policy_coalescing_misses"]
            ),
            native_bank_gpu_feed_gap_seconds=timings["bank_gpu_feed_gap"],
            native_bank_gpu_feed_gap_events=int(counters["bank_gpu_feed_gap_events"]),
            native_startup_seconds=timings["startup"],
            native_budget_fence_wait_seconds=timings["budget_fence_wait"],
            engine_fact_seconds=timings["engine_facts"],
            engine_fact_wait_seconds=timings["engine_fact_wait"],
            engine_fact_overlap_seconds=timings["engine_fact_overlap"],
            engine_fact_roots=int(counters["engine_fact_roots"]),
            engine_fact_eligible_options=int(counters["engine_fact_eligible_options"]),
            engine_fact_native_batch_calls=int(
                counters["engine_fact_native_batch_calls"]
            ),
            engine_fact_native_transitions=int(
                counters["engine_fact_native_transitions"]
            ),
            engine_fact_unresolved_worlds=int(
                counters["engine_fact_unresolved_worlds"]
            ),
            engine_fact_resolved_options=int(counters["engine_fact_resolved_options"]),
            engine_control_seconds=timings["engine_control"],
            current_policy_batches=int(counters["current_batches"]),
            current_policy_rows=int(counters["current_rows"]),
            past_self_policy_batches=int(counters["past_batches"]),
            past_self_policy_rows=int(counters["past_rows"]),
            historical_policy_batches=int(counters["historical_batches"]),
            historical_policy_rows=int(counters["historical_rows"]),
            frozen_pending_row_waves=int(counters["frozen_pending_row_waves"]),
            frozen_threshold_releases=int(counters["frozen_threshold_releases"]),
            frozen_deadline_releases=int(counters["frozen_deadline_releases"]),
            frozen_forced_releases=int(counters["frozen_forced_releases"]),
            native_artifact_inference=artifact_inference,
            lane_games=dict(sorted(lane_games.items())),
            seat_games=dict(sorted(seat_games.items())),
            pfsp_member_games=dict(sorted(member_games.items())),
            candidate_score_by_lane={
                lane: sum(scores) / len(scores)
                for lane, scores in sorted(lane_scores.items())
                if scores
            },
        ),
    )


def _inference_route_kind(
    route: NativeRouteKey,
) -> NativeInferenceRouteKind:
    if route.kind == "scripted":
        raise RuntimeError("scripted CPU work cannot enter GPU artifact telemetry")
    return route.kind


__all__ = ["collect_native_route_assigned"]
