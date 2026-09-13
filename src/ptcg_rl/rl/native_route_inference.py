"""Artifact-coherent inference waves for native rollout arenas."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import torch

from ptcg_rl.context.public_event_arrays import (
    PublicEventBatch,
    concatenate_public_event_batches,
    pin_public_event_batch,
)
from ptcg_rl.engine.native_prospective_facts import (
    NativeEngineFactBatch,
    NativeEngineFactSource,
)
from ptcg_rl.engine.native_public_context import NativeKnownOpponentBatch
from ptcg_rl.engine.native_public_events import native_public_event_batch
from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.engine.native_training_view import (
    concatenate_native_training_views,
    select_native_training_rows,
)
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.sequence.action import AcceptedActionRecord
from ptcg_rl.rl.native_collection_delivery import NativePartDelivery
from ptcg_rl.rl.native_policy_batch import (
    NativeSimpleStatelessPolicyBatch,
    move_native_simple_stateless_batch,
)
from ptcg_rl.rl.native_policy_inference import NativePolicyInferenceExecutor
from ptcg_rl.rl.native_policy_selection import select_native_policy_trace_rows
from ptcg_rl.rl.native_policy_trace import (
    NativePolicyActionHostTransfer,
    NativePolicyNumpyActionBatch,
    NativePolicyNumpyTrace,
    NativePolicyTraceHostTransfer,
)
from ptcg_rl.rl.native_rollout_batch import (
    NativeRolloutSource,
    encode_native_rollout_sources,
)
from ptcg_rl.rl.native_route_arena import NativeArenaDispatch
from ptcg_rl.rl.native_route_scheduler import (
    NativeRouteKey,
    native_route_generator_seed,
)
from ptcg_rl.rl.native_sequence_input import build_native_sequence_actor_rows
from ptcg_rl.rl.native_sequence_runtime import (
    NativePendingSequenceDecision,
    native_actions_from_actor_batch,
    native_trace_from_actor_batch,
)
from ptcg_rl.rl.policy_inputs import SimpleStatelessActorRow
from ptcg_rl.rl.sequence_actor import GeneralistSequenceActorPolicy
from ptcg_rl.rl.sequence_actor_transfer import SequenceActorHostTransfer
from ptcg_rl.rl.stateless_actor import StatelessActorBatchTrace

if TYPE_CHECKING:
    from ptcg_rl.rl.native_stateless_collection import NativeStatelessCollector

Int64Array = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _PastRequest:
    dispatch: NativeArenaDispatch
    rows: Int64Array


@dataclass(frozen=True, slots=True)
class _HistoricalRequest:
    dispatch: NativeArenaDispatch
    rows: Int64Array
    view: NativeTrainingBatchView
    known: NativeKnownOpponentBatch
    member_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PendingCurrent:
    requests: tuple[NativeArenaDispatch, ...]
    host_batch: NativeSimpleStatelessPolicyBatch
    transfer: NativePolicyTraceHostTransfer


@dataclass(frozen=True, slots=True)
class _PendingPast:
    requests: tuple[_PastRequest, ...]
    host_batch: NativeSimpleStatelessPolicyBatch
    transfer: NativePolicyActionHostTransfer


@dataclass(frozen=True, slots=True)
class _PendingSequenceCurrent:
    requests: tuple[NativeArenaDispatch, ...]
    row_segments: tuple[tuple[SimpleStatelessActorRow, ...], ...]
    host_batch: NativeSimpleStatelessPolicyBatch
    host_public_events: PublicEventBatch | None
    transfer: SequenceActorHostTransfer


@dataclass(frozen=True, slots=True)
class _PendingSequencePast:
    requests: tuple[_PastRequest, ...]
    row_segments: tuple[tuple[SimpleStatelessActorRow, ...], ...]
    host_batch: NativeSimpleStatelessPolicyBatch
    host_public_events: PublicEventBatch | None
    transfer: SequenceActorHostTransfer


@dataclass(frozen=True, slots=True)
class PreparedNativeCurrent:
    """CPU-prepared current-policy inputs with no actor state mutation."""

    requests: tuple[NativeArenaDispatch, ...]
    host_batch: NativeSimpleStatelessPolicyBatch
    fact_future: Future[NativeEngineFactBatch] | None
    row_segments: tuple[tuple[SimpleStatelessActorRow, ...], ...] | None
    host_public_events: PublicEventBatch | None
    route: NativeRouteKey
    prepare_seconds: float


@dataclass(frozen=True, slots=True)
class _PreparedNativePastRoute:
    requests: tuple[_PastRequest, ...]
    executor: NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy
    host_batch: NativeSimpleStatelessPolicyBatch
    fact_future: Future[NativeEngineFactBatch] | None
    row_segments: tuple[tuple[SimpleStatelessActorRow, ...], ...] | None
    host_public_events: PublicEventBatch | None
    route: NativeRouteKey


@dataclass(frozen=True, slots=True)
class PreparedNativePast:
    """CPU-prepared exact-checkpoint inputs for one policy cohort."""

    routes: tuple[_PreparedNativePastRoute, ...]
    prepare_seconds: float


@dataclass(slots=True)
class PendingNativeRouteSubmission:
    """Opaque owner-thread continuation for one fact-gated CUDA route."""

    _resume_callback: Callable[[], None] | None
    _cancel_callback: Callable[[], None] | None
    _ready_future: Future[NativeEngineFactBatch] | None = None
    _resumed: bool = False
    _cancelled: bool = False

    @property
    def ready_future(self) -> Future[NativeEngineFactBatch] | None:
        """Expose the fact future for event-driven owner-thread scheduling."""
        return self._ready_future

    def ready(self) -> bool:
        """Return whether resume can pass its fact barrier without waiting."""
        return self._ready_future is None or self._ready_future.done()

    def resume(self) -> None:
        """Resume the route exactly once after other prefixes are enqueued."""
        if self._resumed:
            return
        if self._cancelled:
            raise RuntimeError("native route submission was cancelled")
        callback = self._resume_callback
        if callback is None:
            raise RuntimeError("native route submission has no resume callback")
        try:
            callback()
            self._resumed = True
        finally:
            self._resume_callback = None
            self._cancel_callback = None

    def cancel(self) -> None:
        """Drop an unresumed route prefix without consuming its RNG."""
        if self._resumed or self._cancelled:
            return
        callback = self._cancel_callback
        self._cancelled = True
        try:
            if callback is not None:
                callback()
        finally:
            self._resume_callback = None
            self._cancel_callback = None


@dataclass(frozen=True, slots=True)
class PendingNativePastSubmissions:
    """Stable exact-artifact sequence of fact-gated route continuations."""

    routes: tuple[PendingNativeRouteSubmission, ...]

    def resume(self) -> None:
        """Resume past artifacts in their immutable preparation order."""
        for route in self.routes:
            route.resume()

    def cancel(self) -> None:
        """Cancel every still-unresumed artifact in reverse launch order."""
        for route in reversed(self.routes):
            route.cancel()


def resolve_past_executors(
    collector: NativeStatelessCollector,
    dispatches: Sequence[NativeArenaDispatch],
) -> dict[
    str,
    NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
]:
    """Resolve lazy past actors on the coordinator before host preparation."""
    artifacts = dict.fromkeys(
        artifact_sha256
        for dispatch in dispatches
        for artifact_sha256 in dispatch.past_rows
    )
    return {
        artifact_sha256: collector._past_executor(artifact_sha256)
        for artifact_sha256 in artifacts
    }


def prepare_current(
    collector: NativeStatelessCollector,
    dispatches: Sequence[NativeArenaDispatch],
) -> PreparedNativeCurrent | None:
    """Encode a current-policy cohort without touching RNG, CUDA, or actor state."""
    requests = tuple(dispatch for dispatch in dispatches if dispatch.current_rows.size)
    if not requests:
        return None
    started_at = time.perf_counter()
    sources = tuple(
        _rollout_source(dispatch, dispatch.current_rows) for dispatch in requests
    )
    fact_future = (
        _submit_native_engine_facts(
            collector,
            tuple((dispatch, dispatch.current_rows) for dispatch in requests),
        )
        if isinstance(collector.actor, GeneralistSequenceActorPolicy)
        else None
    )
    try:
        host_batch = encode_native_rollout_sources(
            sources,
            pin_memory=collector.actor.device.type == "cuda",
        )
        route = NativeRouteKey(
            "current",
            collector.identity.behavior_policy_fingerprint,
        )
        row_segments = (
            _build_sequence_row_segments(
                collector,
                requests,
                host_batch=host_batch,
                row_sets=tuple(dispatch.current_rows for dispatch in requests),
                materialize_public_events=bool(
                    getattr(collector.actor, "retain_raw_blocks", True)
                ),
            )
            if isinstance(collector.actor, GeneralistSequenceActorPolicy)
            else None
        )
        host_public_events = (
            _collate_sequence_public_events(
                requests,
                row_sets=tuple(dispatch.current_rows for dispatch in requests),
                pin_memory=(
                    collector.actor.device.type == "cuda" and torch.cuda.is_available()
                ),
            )
            if row_segments is not None
            else None
        )
        return PreparedNativeCurrent(
            requests=requests,
            host_batch=host_batch,
            fact_future=fact_future,
            row_segments=row_segments,
            host_public_events=host_public_events,
            route=route,
            prepare_seconds=time.perf_counter() - started_at,
        )
    except BaseException:
        _cancel_or_drain_native_engine_fact_futures((fact_future,))
        raise


def begin_current(
    collector: NativeStatelessCollector,
    prepared: PreparedNativeCurrent | None,
    *,
    generators: dict[NativeRouteKey, torch.Generator],
    delivery: NativePartDelivery | Mapping[int, NativePartDelivery],
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
) -> PendingNativeRouteSubmission | None:
    """Submit current work through its fact barrier without blocking when pending."""
    if prepared is None:
        return None
    requests = prepared.requests
    host_batch = prepared.host_batch
    fact_future = prepared.fact_future
    route = prepared.route
    started_at = time.perf_counter()
    if isinstance(collector.actor, GeneralistSequenceActorPolicy):
        if prepared.row_segments is None:
            raise RuntimeError("prepared sequence current rows are absent")
        policy_bank = getattr(collector, "policy_bank", None)
        if policy_bank is not None and policy_bank.wave_active:
            pending_submission: PendingNativeRouteSubmission | None = None
            with policy_bank.route_stream(
                route,
                device=collector.actor.device,
            ):
                if (
                    collector.actor.device.type == "cuda"
                    and fact_future is not None
                    and not fact_future.done()
                ):
                    pending_submission = _begin_sequence_current(
                        collector,
                        requests,
                        host_batch=host_batch,
                        fact_future=fact_future,
                        row_segments=prepared.row_segments,
                        host_public_events=prepared.host_public_events,
                        route=route,
                        generators=generators,
                        delivery=delivery,
                        timings=timings,
                        counters=counters,
                        artifact_batches=artifact_batches,
                        artifact_rows=artifact_rows,
                        started_at=started_at,
                    )
                else:
                    _serve_sequence_current(
                        collector,
                        requests,
                        host_batch=host_batch,
                        fact_future=fact_future,
                        row_segments=prepared.row_segments,
                        host_public_events=prepared.host_public_events,
                        route=route,
                        generators=generators,
                        delivery=delivery,
                        timings=timings,
                        counters=counters,
                        artifact_batches=artifact_batches,
                        artifact_rows=artifact_rows,
                        started_at=started_at,
                    )
            if pending_submission is not None:
                if fact_future is not None and fact_future.done():
                    pending_submission.resume()
                    return None
                return pending_submission
        else:
            _serve_sequence_current(
                collector,
                requests,
                host_batch=host_batch,
                fact_future=fact_future,
                row_segments=prepared.row_segments,
                host_public_events=prepared.host_public_events,
                route=route,
                generators=generators,
                delivery=delivery,
                timings=timings,
                counters=counters,
                artifact_batches=artifact_batches,
                artifact_rows=artifact_rows,
                started_at=started_at,
            )
        return None
    policy_bank = getattr(collector, "policy_bank", None)
    if policy_bank is not None and policy_bank.wave_active:
        with policy_bank.route_stream(route, device=collector.actor.device):
            device_batch = move_native_simple_stateless_batch(
                host_batch,
                device=collector.actor.device,
                non_blocking=collector.actor.device.type == "cuda",
            )
            transfer = collector.actor.sample_device(
                device_batch,
                temperature=_current_policy_temperature(collector),
                generator=_route_generator(
                    collector,
                    route,
                    device=collector.actor.device,
                    generators=generators,
                ),
                evaluation_action_only=_evaluation_action_only(collector),
            ).defer_to_host(
                copy_stream=policy_bank.copy_stream(collector.actor.device),
            )
        timings["current"] += time.perf_counter() - started_at
        counters["current_batches"] += 1
        counters["current_rows"] += host_batch.batch_size
        artifact_batches[route] += 1
        artifact_rows[route] += host_batch.batch_size
        pending = _PendingCurrent(
            requests=requests,
            host_batch=host_batch,
            transfer=transfer,
        )
        policy_bank.defer(
            lambda: _complete_current(
                collector,
                pending,
                delivery=delivery,
                timings=timings,
                counters=counters,
            )
        )
        return None
    device_batch = move_native_simple_stateless_batch(
        host_batch,
        device=collector.actor.device,
        non_blocking=collector.actor.device.type == "cuda",
    )
    trace = collector.actor.sample(
        device_batch,
        temperature=_current_policy_temperature(collector),
        generator=_route_generator(
            collector,
            route,
            device=collector.actor.device,
            generators=generators,
        ),
        evaluation_action_only=_evaluation_action_only(collector),
    )
    timings["current"] += time.perf_counter() - started_at
    counters["current_batches"] += 1
    counters["current_rows"] += host_batch.batch_size
    artifact_batches[route] += 1
    artifact_rows[route] += host_batch.batch_size

    start = 0
    for dispatch in requests:
        rows = dispatch.current_rows
        stop = start + int(rows.size)
        segment_trace = select_native_policy_trace_rows(
            trace,
            np.arange(start, stop, dtype=np.int64),
        )
        apply_current_trace(
            collector,
            dispatch,
            batch=host_batch,
            trace=segment_trace,
            batch_start=start,
            delivery=_delivery_for(dispatch, delivery),
            timings=timings,
            counters=counters,
        )
        start = stop
    return None


def resume_current(pending: PendingNativeRouteSubmission | None) -> None:
    """Resume a fact-gated current route when one was returned by begin."""
    if pending is not None:
        pending.resume()


def submit_current(
    collector: NativeStatelessCollector,
    prepared: PreparedNativeCurrent | None,
    *,
    generators: dict[NativeRouteKey, torch.Generator],
    delivery: NativePartDelivery | Mapping[int, NativePartDelivery],
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
) -> None:
    """Submit and synchronously resume the compatible current-policy API."""
    resume_current(
        begin_current(
            collector,
            prepared,
            generators=generators,
            delivery=delivery,
            timings=timings,
            counters=counters,
            artifact_batches=artifact_batches,
            artifact_rows=artifact_rows,
        )
    )


def serve_current(
    collector: NativeStatelessCollector,
    dispatches: Sequence[NativeArenaDispatch],
    *,
    generators: dict[NativeRouteKey, torch.Generator],
    delivery: NativePartDelivery | Mapping[int, NativePartDelivery],
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
) -> None:
    """Prepare then synchronously submit one merged current-policy cohort."""
    prepared = prepare_current(collector, dispatches)
    if prepared is not None:
        timings["policy_host_prepare"] += prepared.prepare_seconds
    submit_current(
        collector,
        prepared,
        generators=generators,
        delivery=delivery,
        timings=timings,
        counters=counters,
        artifact_batches=artifact_batches,
        artifact_rows=artifact_rows,
    )


def prepare_past(
    collector: NativeStatelessCollector,
    dispatches: Sequence[NativeArenaDispatch],
    *,
    executors: Mapping[
        str,
        NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
    ]
    | None = None,
) -> PreparedNativePast:
    """Encode exact past-policy inputs without invoking an actor or CUDA."""
    started_at = time.perf_counter()
    grouped: defaultdict[str, list[_PastRequest]] = defaultdict(list)
    for dispatch in dispatches:
        for artifact_sha256, rows in dispatch.past_rows.items():
            grouped[artifact_sha256].append(_PastRequest(dispatch=dispatch, rows=rows))
    resolved = (
        resolve_past_executors(collector, dispatches)
        if executors is None
        else executors
    )
    prepared_routes: list[_PreparedNativePastRoute] = []
    fact_futures: list[Future[NativeEngineFactBatch]] = []
    try:
        for artifact_sha256, requests in grouped.items():
            sources = tuple(
                _artifact_rollout_source(
                    request.dispatch,
                    request.rows,
                    artifact_sha256=artifact_sha256,
                )
                for request in requests
            )
            try:
                executor = resolved[artifact_sha256]
            except KeyError as error:
                raise RuntimeError(
                    "past executor was not resolved before host preparation"
                ) from error
            fact_future = (
                _submit_native_engine_facts(
                    collector,
                    tuple((request.dispatch, request.rows) for request in requests),
                    artifact_sha256=artifact_sha256,
                )
                if isinstance(executor, GeneralistSequenceActorPolicy)
                else None
            )
            if fact_future is not None:
                fact_futures.append(fact_future)
            host_batch = encode_native_rollout_sources(
                sources,
                pin_memory=executor.device.type == "cuda",
            )
            route = NativeRouteKey("past_self", artifact_sha256)
            row_segments = (
                _build_sequence_row_segments(
                    collector,
                    tuple(request.dispatch for request in requests),
                    host_batch=host_batch,
                    row_sets=tuple(request.rows for request in requests),
                    materialize_public_events=bool(
                        getattr(executor, "retain_raw_blocks", True)
                    ),
                )
                if isinstance(executor, GeneralistSequenceActorPolicy)
                else None
            )
            host_public_events = (
                _collate_sequence_public_events(
                    tuple(request.dispatch for request in requests),
                    row_sets=tuple(request.rows for request in requests),
                    pin_memory=(
                        executor.device.type == "cuda" and torch.cuda.is_available()
                    ),
                )
                if row_segments is not None
                else None
            )
            prepared_routes.append(
                _PreparedNativePastRoute(
                    requests=tuple(requests),
                    executor=executor,
                    host_batch=host_batch,
                    fact_future=fact_future,
                    row_segments=row_segments,
                    host_public_events=host_public_events,
                    route=route,
                )
            )
        return PreparedNativePast(
            routes=tuple(prepared_routes),
            prepare_seconds=time.perf_counter() - started_at,
        )
    except BaseException:
        _cancel_or_drain_native_engine_fact_futures(fact_futures)
        raise


def begin_past(
    collector: NativeStatelessCollector,
    prepared: PreparedNativePast,
    *,
    generators: dict[NativeRouteKey, torch.Generator],
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
) -> PendingNativePastSubmissions:
    """Enqueue exact past prefixes without blocking on serialized fact futures."""
    pending_submissions: list[PendingNativeRouteSubmission] = []
    for prepared_route in prepared.routes:
        started_at = time.perf_counter()
        requests = prepared_route.requests
        executor = prepared_route.executor
        host_batch = prepared_route.host_batch
        fact_future = prepared_route.fact_future
        route = prepared_route.route
        if isinstance(executor, GeneralistSequenceActorPolicy):
            if prepared_route.row_segments is None:
                raise RuntimeError("prepared sequence past rows are absent")
            policy_bank = getattr(collector, "policy_bank", None)
            if policy_bank is not None and policy_bank.wave_active:
                with policy_bank.route_stream(route, device=executor.device):
                    if executor.device.type == "cuda" and fact_future is not None:
                        pending_submissions.append(
                            _begin_sequence_past(
                                collector,
                                requests,
                                executor=executor,
                                host_batch=host_batch,
                                fact_future=fact_future,
                                row_segments=prepared_route.row_segments,
                                host_public_events=(prepared_route.host_public_events),
                                route=route,
                                generators=generators,
                                timings=timings,
                                counters=counters,
                                artifact_batches=artifact_batches,
                                artifact_rows=artifact_rows,
                                started_at=started_at,
                            )
                        )
                    else:
                        _serve_sequence_past(
                            collector,
                            requests,
                            executor=executor,
                            host_batch=host_batch,
                            fact_future=fact_future,
                            row_segments=prepared_route.row_segments,
                            host_public_events=(prepared_route.host_public_events),
                            route=route,
                            generators=generators,
                            timings=timings,
                            counters=counters,
                            artifact_batches=artifact_batches,
                            artifact_rows=artifact_rows,
                            started_at=started_at,
                        )
            else:
                _serve_sequence_past(
                    collector,
                    requests,
                    executor=executor,
                    host_batch=host_batch,
                    fact_future=fact_future,
                    row_segments=prepared_route.row_segments,
                    host_public_events=prepared_route.host_public_events,
                    route=route,
                    generators=generators,
                    timings=timings,
                    counters=counters,
                    artifact_batches=artifact_batches,
                    artifact_rows=artifact_rows,
                    started_at=started_at,
                )
            continue
        policy_bank = getattr(collector, "policy_bank", None)
        if policy_bank is not None and policy_bank.wave_active:
            with policy_bank.route_stream(route, device=executor.device):
                device_batch = move_native_simple_stateless_batch(
                    host_batch,
                    device=executor.device,
                    non_blocking=executor.device.type == "cuda",
                )
                transfer = executor.sample_actions_device(
                    device_batch,
                    temperature=_frozen_policy_temperature(
                        collector,
                        route.artifact_sha256
                    ),
                    generator=_route_generator(
                        collector,
                        route,
                        device=executor.device,
                        generators=generators,
                    ),
                ).defer_to_host(
                    copy_stream=policy_bank.copy_stream(executor.device),
                )
            timings["past"] += time.perf_counter() - started_at
            counters["past_batches"] += 1
            counters["past_rows"] += host_batch.batch_size
            artifact_batches[route] += 1
            artifact_rows[route] += host_batch.batch_size
            pending = _PendingPast(
                requests=tuple(requests),
                host_batch=host_batch,
                transfer=transfer,
            )
            policy_bank.defer(
                lambda pending=pending: _complete_past(
                    pending,
                    timings=timings,
                )
            )
            continue
        device_batch = move_native_simple_stateless_batch(
            host_batch,
            device=executor.device,
            non_blocking=executor.device.type == "cuda",
        )
        actions = executor.sample_actions(
            device_batch,
            temperature=_frozen_policy_temperature(
                collector,
                route.artifact_sha256,
            ),
            generator=_route_generator(
                collector,
                route,
                device=executor.device,
                generators=generators,
            ),
        )
        timings["past"] += time.perf_counter() - started_at
        counters["past_batches"] += 1
        counters["past_rows"] += host_batch.batch_size
        artifact_batches[route] += 1
        artifact_rows[route] += host_batch.batch_size

        start = 0
        for request in requests:
            stop = start + int(request.rows.size)
            request.dispatch.actions.fill_action_batch(
                request.rows,
                _select_action_rows(actions, start=start, stop=stop),
            )
            start = stop
    return PendingNativePastSubmissions(routes=tuple(pending_submissions))


def resume_past(pending: PendingNativePastSubmissions) -> None:
    """Resume fact-gated past routes in stable exact-artifact order."""
    pending.resume()


def submit_past(
    collector: NativeStatelessCollector,
    prepared: PreparedNativePast,
    *,
    generators: dict[NativeRouteKey, torch.Generator],
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
) -> None:
    """Submit and synchronously resume the compatible past-policy API."""
    resume_past(
        begin_past(
            collector,
            prepared,
            generators=generators,
            timings=timings,
            counters=counters,
            artifact_batches=artifact_batches,
            artifact_rows=artifact_rows,
        )
    )


def serve_past(
    collector: NativeStatelessCollector,
    dispatches: Sequence[NativeArenaDispatch],
    *,
    generators: dict[NativeRouteKey, torch.Generator],
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
) -> None:
    """Prepare then synchronously submit all exact past-policy routes."""
    prepared = prepare_past(collector, dispatches)
    timings["policy_host_prepare"] += prepared.prepare_seconds
    submit_past(
        collector,
        prepared,
        generators=generators,
        timings=timings,
        counters=counters,
        artifact_batches=artifact_batches,
        artifact_rows=artifact_rows,
    )


def _build_sequence_row_segments(
    collector: NativeStatelessCollector,
    dispatches: Sequence[NativeArenaDispatch],
    *,
    host_batch: NativeSimpleStatelessPolicyBatch,
    row_sets: Sequence[Int64Array],
    materialize_public_events: bool,
) -> tuple[tuple[SimpleStatelessActorRow, ...], ...]:
    """Build recurrent compatibility rows entirely from host snapshots."""
    if len(dispatches) != len(row_sets):
        raise ValueError("sequence dispatch and row segments differ")
    segments: list[tuple[SimpleStatelessActorRow, ...]] = []
    start = 0
    for dispatch, native_rows in zip(dispatches, row_sets, strict=True):
        stop = start + int(native_rows.size)
        segments.append(
            build_native_sequence_actor_rows(
                host_batch,
                batch_rows=np.arange(start, stop, dtype=np.int64),
                view=dispatch.arena.view,
                native_rows=native_rows,
                live=dispatch.arena.live,
                engine_fact_producer_fingerprint=getattr(
                    collector,
                    "engine_fact_producer_fingerprint",
                    None,
                ),
                materialize_public_events=materialize_public_events,
                materialize_raw_payloads=materialize_public_events,
            )
        )
        start = stop
    if start != host_batch.batch_size:
        raise RuntimeError("sequence row segments differ from encoded host batch")
    return tuple(segments)


def _collate_sequence_public_events(
    dispatches: Sequence[NativeArenaDispatch],
    *,
    row_sets: Sequence[Int64Array],
    pin_memory: bool,
) -> PublicEventBatch:
    """Encode native fixed-column logs without per-event Python objects."""
    if len(dispatches) != len(row_sets) or not dispatches:
        raise ValueError("sequence dispatch and event row segments differ")
    result = concatenate_public_event_batches(
        tuple(
            native_public_event_batch(
                dispatch.arena.view,
                rows=rows,
            )
            for dispatch, rows in zip(dispatches, row_sets, strict=True)
        )
    )
    return pin_public_event_batch(result) if pin_memory else result


def _begin_sequence_current(
    collector: NativeStatelessCollector,
    requests: tuple[NativeArenaDispatch, ...],
    *,
    host_batch: NativeSimpleStatelessPolicyBatch,
    fact_future: Future[NativeEngineFactBatch],
    row_segments: tuple[tuple[SimpleStatelessActorRow, ...], ...],
    host_public_events: PublicEventBatch | None,
    route: NativeRouteKey,
    generators: dict[NativeRouteKey, torch.Generator],
    delivery: NativePartDelivery | Mapping[int, NativePartDelivery],
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
    started_at: float,
) -> PendingNativeRouteSubmission:
    """Enqueue current snapshot/temporal work and stop before the fact join."""
    actor = collector.actor
    if not isinstance(actor, GeneralistSequenceActorPolicy):
        raise TypeError("sequence current prefix requires a sequence actor")
    policy_bank = getattr(collector, "policy_bank", None)
    if policy_bank is None or not policy_bank.wave_active:
        raise RuntimeError("sequence current prefix requires an active policy wave")
    actor_rows = tuple(row for segment in row_segments for row in segment)
    device_batch = move_native_simple_stateless_batch(
        host_batch,
        device=actor.device,
        non_blocking=True,
        move_exact_deck_tensors=False,
    )
    actor_continuation = actor.begin_preencoded_deferred(
        actor_rows,
        device_batch,
        temperature=_current_policy_temperature(collector),
        generator=_route_generator(
            collector,
            route,
            device=actor.device,
            generators=generators,
        ),
        copy_stream=policy_bank.copy_stream(actor.device),
        public_event_batch=host_public_events,
        host_semantic_batch=host_batch,
        evaluation_action_only=_evaluation_action_only(collector),
    )
    prefix_seconds = time.perf_counter() - started_at

    def resume() -> None:
        resume_started_at = time.perf_counter()
        with policy_bank.route_stream(route, device=actor.device):
            transfer = actor_continuation.resume(
                await_option_features=partial(
                    _finish_native_engine_facts,
                    fact_future,
                    (host_batch.options, device_batch.options),
                    timings=timings,
                    counters=counters,
                ),
            )
        timings["current"] += prefix_seconds + time.perf_counter() - resume_started_at
        counters["current_batches"] += 1
        counters["current_rows"] += host_batch.batch_size
        artifact_batches[route] += 1
        artifact_rows[route] += host_batch.batch_size
        pending = _PendingSequenceCurrent(
            requests=requests,
            row_segments=row_segments,
            host_batch=host_batch,
            host_public_events=host_public_events,
            transfer=transfer,
        )
        policy_bank.defer(
            lambda: _complete_sequence_current(
                collector,
                pending,
                delivery=delivery,
                timings=timings,
                counters=counters,
            ),
            on_abort=transfer.cancel,
        )

    return PendingNativeRouteSubmission(
        _resume_callback=resume,
        _cancel_callback=actor_continuation.cancel,
        _ready_future=fact_future,
    )


def _serve_sequence_current(
    collector: NativeStatelessCollector,
    requests: tuple[NativeArenaDispatch, ...],
    *,
    host_batch: NativeSimpleStatelessPolicyBatch,
    fact_future: Future[NativeEngineFactBatch] | None,
    row_segments: tuple[tuple[SimpleStatelessActorRow, ...], ...],
    route: NativeRouteKey,
    generators: dict[NativeRouteKey, torch.Generator],
    delivery: NativePartDelivery | Mapping[int, NativePartDelivery],
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
    started_at: float,
    host_public_events: PublicEventBatch | None = None,
) -> None:
    """Serve one global recurrent current-policy wave."""
    actor = collector.actor
    if not isinstance(actor, GeneralistSequenceActorPolicy):
        raise TypeError("sequence current service requires a sequence actor")
    actor_rows = tuple(row for segment in row_segments for row in segment)
    device_batch = move_native_simple_stateless_batch(
        host_batch,
        device=actor.device,
        non_blocking=actor.device.type == "cuda",
        move_exact_deck_tensors=False,
    )
    policy_bank = getattr(collector, "policy_bank", None)
    if (
        policy_bank is not None
        and policy_bank.wave_active
        and actor.device.type == "cuda"
    ):
        transfer = actor.sample_preencoded_deferred(
            actor_rows,
            device_batch,
            temperature=_current_policy_temperature(collector),
            generator=_route_generator(
                collector,
                route,
                device=actor.device,
                generators=generators,
            ),
            copy_stream=policy_bank.copy_stream(actor.device),
            public_event_batch=host_public_events,
            host_semantic_batch=host_batch,
            await_option_features=(
                None
                if fact_future is None
                else partial(
                    _finish_native_engine_facts,
                    fact_future,
                    (host_batch.options, device_batch.options),
                    timings=timings,
                    counters=counters,
                )
            ),
            evaluation_action_only=_evaluation_action_only(collector),
        )
        timings["current"] += time.perf_counter() - started_at
        counters["current_batches"] += 1
        counters["current_rows"] += host_batch.batch_size
        artifact_batches[route] += 1
        artifact_rows[route] += host_batch.batch_size
        pending = _PendingSequenceCurrent(
            requests=requests,
            row_segments=row_segments,
            host_batch=host_batch,
            host_public_events=host_public_events,
            transfer=transfer,
        )
        policy_bank.defer(
            lambda: _complete_sequence_current(
                collector,
                pending,
                delivery=delivery,
                timings=timings,
                counters=counters,
            ),
            on_abort=transfer.cancel,
        )
        return
    actor_trace = actor.sample_preencoded(
        actor_rows,
        device_batch,
        temperature=_current_policy_temperature(collector),
        generator=_route_generator(
            collector,
            route,
            device=actor.device,
            generators=generators,
        ),
        public_event_batch=host_public_events,
        host_semantic_batch=host_batch,
        await_option_features=(
            None
            if fact_future is None
            else partial(
                _finish_native_engine_facts,
                fact_future,
                (host_batch.options, device_batch.options),
                timings=timings,
                counters=counters,
            )
        ),
        evaluation_action_only=_evaluation_action_only(collector),
    )
    timings["current"] += time.perf_counter() - started_at
    counters["current_batches"] += 1
    counters["current_rows"] += host_batch.batch_size
    artifact_batches[route] += 1
    artifact_rows[route] += host_batch.batch_size
    _scatter_sequence_current(
        collector,
        actor,
        requests=requests,
        row_segments=row_segments,
        host_batch=host_batch,
        actor_trace=actor_trace,
        delivery=delivery,
        timings=timings,
        counters=counters,
    )


def _scatter_sequence_current(
    collector: NativeStatelessCollector,
    actor: GeneralistSequenceActorPolicy,
    *,
    requests: tuple[NativeArenaDispatch, ...],
    row_segments: tuple[tuple[SimpleStatelessActorRow, ...], ...],
    host_batch: NativeSimpleStatelessPolicyBatch,
    actor_trace: StatelessActorBatchTrace,
    native_trace: NativePolicyNumpyTrace | None = None,
    delivery: NativePartDelivery | Mapping[int, NativePartDelivery],
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> None:
    """Scatter one finalized recurrent current-policy batch."""
    trace = (
        native_trace_from_actor_batch(actor.identity, actor_trace)
        if native_trace is None
        else native_trace
    )
    if getattr(trace, "identity", actor.identity) != actor.identity:
        raise ValueError("native sequence trace changed behavior identity")
    start = 0
    for dispatch, rows, actor_row_segment in zip(
        requests,
        (request.current_rows for request in requests),
        row_segments,
        strict=True,
    ):
        stop = start + int(rows.size)
        local_indices = np.arange(start, stop, dtype=np.int64)
        segment_trace = select_native_policy_trace_rows(trace, local_indices)
        decision_segment = actor_trace.decisions[start:stop]
        accepted_actions: list[AcceptedActionRecord] = []
        for arena_row, actor_row, decision in zip(
            rows,
            actor_row_segment,
            decision_segment,
            strict=True,
        ):
            accepted = decision.accepted_action
            if accepted is None:
                raise RuntimeError("sequence current action was not staged")
            accepted_actions.append(accepted)
            dispatch.pending_sequence.append(
                NativePendingSequenceDecision(
                    actor=actor,
                    arena_row=int(arena_row),
                    row=actor_row,
                    trace=decision,
                )
            )
        apply_current_trace(
            collector,
            dispatch,
            batch=host_batch,
            trace=segment_trace,
            batch_start=start,
            delivery=_delivery_for(dispatch, delivery),
            timings=timings,
            counters=counters,
            sequence_rows=actor_row_segment,
            accepted_actions=tuple(accepted_actions),
        )
        start = stop


def _begin_sequence_past(
    collector: NativeStatelessCollector,
    requests: Sequence[_PastRequest],
    *,
    executor: GeneralistSequenceActorPolicy,
    host_batch: NativeSimpleStatelessPolicyBatch,
    fact_future: Future[NativeEngineFactBatch],
    row_segments: tuple[tuple[SimpleStatelessActorRow, ...], ...],
    host_public_events: PublicEventBatch | None,
    route: NativeRouteKey,
    generators: dict[NativeRouteKey, torch.Generator],
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
    started_at: float,
) -> PendingNativeRouteSubmission:
    """Enqueue one frozen snapshot/temporal prefix before its fact join."""
    policy_bank = getattr(collector, "policy_bank", None)
    if policy_bank is None or not policy_bank.wave_active:
        raise RuntimeError("sequence past prefix requires an active policy wave")
    actor_rows = tuple(row for segment in row_segments for row in segment)
    device_batch = move_native_simple_stateless_batch(
        host_batch,
        device=executor.device,
        non_blocking=True,
        move_exact_deck_tensors=False,
    )
    actor_continuation = executor.begin_preencoded_deferred(
        actor_rows,
        device_batch,
        temperature=_frozen_policy_temperature(collector, route.artifact_sha256),
        generator=_route_generator(
            collector,
            route,
            device=executor.device,
            generators=generators,
        ),
        copy_stream=policy_bank.copy_stream(executor.device),
        public_event_batch=host_public_events,
        host_semantic_batch=host_batch,
        evaluation_action_only=_evaluation_action_only(collector),
    )
    prefix_seconds = time.perf_counter() - started_at

    def resume() -> None:
        resume_started_at = time.perf_counter()
        with policy_bank.route_stream(route, device=executor.device):
            transfer = actor_continuation.resume(
                await_option_features=partial(
                    _finish_native_engine_facts,
                    fact_future,
                    (host_batch.options, device_batch.options),
                    timings=timings,
                    counters=counters,
                ),
            )
        timings["past"] += prefix_seconds + time.perf_counter() - resume_started_at
        counters["past_batches"] += 1
        counters["past_rows"] += host_batch.batch_size
        artifact_batches[route] += 1
        artifact_rows[route] += host_batch.batch_size
        pending = _PendingSequencePast(
            requests=tuple(requests),
            row_segments=row_segments,
            host_batch=host_batch,
            host_public_events=host_public_events,
            transfer=transfer,
        )
        policy_bank.defer(
            lambda: _complete_sequence_past(
                pending,
                executor=executor,
                timings=timings,
            ),
            on_abort=transfer.cancel,
        )

    return PendingNativeRouteSubmission(
        _resume_callback=resume,
        _cancel_callback=actor_continuation.cancel,
        _ready_future=fact_future,
    )


def _serve_sequence_past(
    collector: NativeStatelessCollector,
    requests: Sequence[_PastRequest],
    *,
    executor: GeneralistSequenceActorPolicy,
    host_batch: NativeSimpleStatelessPolicyBatch,
    fact_future: Future[NativeEngineFactBatch] | None,
    row_segments: tuple[tuple[SimpleStatelessActorRow, ...], ...],
    route: NativeRouteKey,
    generators: dict[NativeRouteKey, torch.Generator],
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
    started_at: float,
    host_public_events: PublicEventBatch | None = None,
) -> None:
    """Serve one accumulated recurrent frozen-policy route."""
    actor_rows = tuple(row for segment in row_segments for row in segment)
    device_batch = move_native_simple_stateless_batch(
        host_batch,
        device=executor.device,
        non_blocking=executor.device.type == "cuda",
        move_exact_deck_tensors=False,
    )
    policy_bank = getattr(collector, "policy_bank", None)
    if (
        policy_bank is not None
        and policy_bank.wave_active
        and executor.device.type == "cuda"
    ):
        transfer = executor.sample_preencoded_deferred(
            actor_rows,
            device_batch,
            temperature=_frozen_policy_temperature(
                collector,
                route.artifact_sha256,
            ),
            generator=_route_generator(
                collector,
                route,
                device=executor.device,
                generators=generators,
            ),
            copy_stream=policy_bank.copy_stream(executor.device),
            public_event_batch=host_public_events,
            host_semantic_batch=host_batch,
            await_option_features=(
                None
                if fact_future is None
                else partial(
                    _finish_native_engine_facts,
                    fact_future,
                    (host_batch.options, device_batch.options),
                    timings=timings,
                    counters=counters,
                )
            ),
            evaluation_action_only=_evaluation_action_only(collector),
        )
        timings["past"] += time.perf_counter() - started_at
        counters["past_batches"] += 1
        counters["past_rows"] += host_batch.batch_size
        artifact_batches[route] += 1
        artifact_rows[route] += host_batch.batch_size
        pending = _PendingSequencePast(
            requests=tuple(requests),
            row_segments=row_segments,
            host_batch=host_batch,
            host_public_events=host_public_events,
            transfer=transfer,
        )
        policy_bank.defer(
            lambda pending=pending, executor=executor: _complete_sequence_past(
                pending,
                executor=executor,
                timings=timings,
            ),
            on_abort=transfer.cancel,
        )
        return
    actor_trace = executor.sample_preencoded(
        actor_rows,
        device_batch,
        temperature=_frozen_policy_temperature(collector, route.artifact_sha256),
        generator=_route_generator(
            collector,
            route,
            device=executor.device,
            generators=generators,
        ),
        public_event_batch=host_public_events,
        host_semantic_batch=host_batch,
        await_option_features=(
            None
            if fact_future is None
            else partial(
                _finish_native_engine_facts,
                fact_future,
                (host_batch.options, device_batch.options),
                timings=timings,
                counters=counters,
            )
        ),
        evaluation_action_only=_evaluation_action_only(collector),
    )
    timings["past"] += time.perf_counter() - started_at
    counters["past_batches"] += 1
    counters["past_rows"] += host_batch.batch_size
    artifact_batches[route] += 1
    artifact_rows[route] += host_batch.batch_size
    _scatter_sequence_past(
        tuple(requests),
        row_segments,
        executor=executor,
        actor_trace=actor_trace,
    )


def _scatter_sequence_past(
    requests: tuple[_PastRequest, ...],
    row_segments: tuple[tuple[SimpleStatelessActorRow, ...], ...],
    *,
    executor: GeneralistSequenceActorPolicy,
    actor_trace: StatelessActorBatchTrace,
    native_trace: NativePolicyNumpyTrace | None = None,
) -> None:
    """Scatter one finalized recurrent frozen-policy batch."""
    actions = (
        native_actions_from_actor_batch(executor.identity, actor_trace)
        if native_trace is None
        else NativePolicyNumpyActionBatch(
            identity=native_trace.identity,
            action_offsets=native_trace.action_offsets,
            action_choices=native_trace.action_choices,
        )
    )
    if actions.identity != executor.identity:
        raise ValueError("native sequence actions changed behavior identity")
    start = 0
    for request, actor_row_segment in zip(
        requests,
        row_segments,
        strict=True,
    ):
        stop = start + int(request.rows.size)
        request.dispatch.actions.fill_action_batch(
            request.rows,
            _select_action_rows(actions, start=start, stop=stop),
        )
        for arena_row, actor_row, decision in zip(
            request.rows,
            actor_row_segment,
            actor_trace.decisions[start:stop],
            strict=True,
        ):
            request.dispatch.pending_sequence.append(
                NativePendingSequenceDecision(
                    actor=executor,
                    arena_row=int(arena_row),
                    row=actor_row,
                    trace=decision,
                )
            )
        start = stop


def _complete_sequence_current(
    collector: NativeStatelessCollector,
    pending: _PendingSequenceCurrent,
    *,
    delivery: NativePartDelivery | Mapping[int, NativePartDelivery],
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> None:
    """Finalize and scatter one recurrent current batch after shared D2H."""
    actor = collector.actor
    if not isinstance(actor, GeneralistSequenceActorPolicy):
        raise TypeError("sequence current completion requires a sequence actor")
    started_at = time.perf_counter()
    actor_trace, native_trace = _finish_sequence_transfer_native(pending.transfer)
    timings["current"] += time.perf_counter() - started_at
    _scatter_sequence_current(
        collector,
        actor,
        requests=pending.requests,
        row_segments=pending.row_segments,
        host_batch=pending.host_batch,
        actor_trace=actor_trace,
        native_trace=native_trace,
        delivery=delivery,
        timings=timings,
        counters=counters,
    )


def _complete_sequence_past(
    pending: _PendingSequencePast,
    *,
    executor: GeneralistSequenceActorPolicy,
    timings: defaultdict[str, float],
) -> None:
    """Finalize and scatter one recurrent frozen batch after shared D2H."""
    started_at = time.perf_counter()
    actor_trace, native_trace = _finish_sequence_transfer_native(pending.transfer)
    timings["past"] += time.perf_counter() - started_at
    _scatter_sequence_past(
        pending.requests,
        pending.row_segments,
        executor=executor,
        actor_trace=actor_trace,
        native_trace=native_trace,
    )


def _finish_sequence_transfer_native(
    transfer: Any,
) -> tuple[StatelessActorBatchTrace, NativePolicyNumpyTrace | None]:
    """Finish a sequence transfer while retaining compatibility test doubles."""
    finish_native = getattr(transfer, "finish_native_ready", None)
    if callable(finish_native):
        return cast(
            tuple[StatelessActorBatchTrace, NativePolicyNumpyTrace | None],
            finish_native(),
        )
    return cast(StatelessActorBatchTrace, transfer.finish_ready()), None


def _complete_current(
    collector: NativeStatelessCollector,
    pending: _PendingCurrent,
    *,
    delivery: NativePartDelivery | Mapping[int, NativePartDelivery],
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> None:
    """Resolve one wave's full trace after the shared D2H boundary."""
    started_at = time.perf_counter()
    trace = pending.transfer.finish_ready()
    timings["current"] += time.perf_counter() - started_at
    start = 0
    for dispatch in pending.requests:
        rows = dispatch.current_rows
        stop = start + int(rows.size)
        segment_trace = select_native_policy_trace_rows(
            trace,
            np.arange(start, stop, dtype=np.int64),
        )
        apply_current_trace(
            collector,
            dispatch,
            batch=pending.host_batch,
            trace=segment_trace,
            batch_start=start,
            delivery=_delivery_for(dispatch, delivery),
            timings=timings,
            counters=counters,
        )
        start = stop


def _complete_past(
    pending: _PendingPast,
    *,
    timings: defaultdict[str, float],
) -> None:
    """Resolve and scatter exact checkpoint actions after batched D2H."""
    started_at = time.perf_counter()
    actions = pending.transfer.finish_ready()
    timings["past"] += time.perf_counter() - started_at
    start = 0
    for request in pending.requests:
        stop = start + int(request.rows.size)
        request.dispatch.actions.fill_action_batch(
            request.rows,
            _select_action_rows(actions, start=start, stop=stop),
        )
        start = stop


def serve_historical(
    collector: NativeStatelessCollector,
    dispatches: Sequence[NativeArenaDispatch],
    *,
    timings: defaultdict[str, float],
    counters: Counter[str],
    artifact_batches: Counter[NativeRouteKey],
    artifact_rows: Counter[NativeRouteKey],
) -> None:
    """Run one native legacy batch per exact historical checkpoint."""
    grouped: defaultdict[str, list[_HistoricalRequest]] = defaultdict(list)
    for dispatch in dispatches:
        for artifact_sha256, rows in dispatch.historical_rows.items():
            view = select_native_training_rows(dispatch.arena.view, rows)
            members = tuple(
                dispatch.arena.live[int(slot)].required_member() for slot in view.slots
            )
            if any(
                member.source != "historical_anchor"
                or member.policy_sha256 != artifact_sha256
                for member in members
            ):
                raise RuntimeError(
                    "historical dispatch rows differ from their checkpoint artifact"
                )
            known = _encoder_for_artifact(
                dispatch, artifact_sha256
            ).known_opponent_batch(
                view.slots,
                view.select_player,
            )
            grouped[artifact_sha256].append(
                _HistoricalRequest(
                    dispatch=dispatch,
                    rows=rows,
                    view=view,
                    known=known,
                    member_ids=tuple(member.member_id for member in members),
                )
            )

    for artifact_sha256, requests in grouped.items():
        started_at = time.perf_counter()
        view = _concatenate_model_views(tuple(request.view for request in requests))
        known = _concatenate_known_opponents(
            tuple(request.known for request in requests)
        )
        member_ids = tuple(
            member_id for request in requests for member_id in request.member_ids
        )
        sources = tuple(
            _artifact_rollout_source(
                request.dispatch,
                request.rows,
                artifact_sha256=artifact_sha256,
            )
            for request in requests
        )
        host_batch = encode_native_rollout_sources(
            sources,
            pin_memory=False,
        )
        actions = collector.historical_pool.act_many_preencoded(
            view,
            host_batch.states,
            host_batch.options,
            known,
            member_ids=member_ids,
            deck_signatures=host_batch.deck_signatures,
            model_encoding_fingerprint=(sources[0].encoder.model_encoding_fingerprint),
        )
        timings["historical"] += time.perf_counter() - started_at
        route = NativeRouteKey("historical", artifact_sha256)
        counters["historical_batches"] += 1
        counters["historical_rows"] += view.batch_size
        artifact_batches[route] += 1
        artifact_rows[route] += view.batch_size

        start = 0
        for request in requests:
            stop = start + int(request.rows.size)
            request.dispatch.actions.fill_actions(
                request.rows,
                actions[start:stop],
            )
            start = stop


def apply_current_trace(
    collector: NativeStatelessCollector,
    dispatch: NativeArenaDispatch,
    *,
    batch: NativeSimpleStatelessPolicyBatch,
    trace: NativePolicyNumpyTrace,
    batch_start: int,
    delivery: NativePartDelivery,
    timings: defaultdict[str, float],
    counters: Counter[str],
    sequence_rows: Sequence[SimpleStatelessActorRow] | None = None,
    accepted_actions: Sequence[AcceptedActionRecord] | None = None,
) -> None:
    """Scatter one global current trace segment into its arena."""
    rows = dispatch.current_rows
    if trace.batch_size != int(rows.size):
        raise ValueError("native current trace rows differ from arena request")
    if batch_start < 0 or batch_start + trace.batch_size > batch.batch_size:
        raise ValueError("native current trace slice exceeds its shared batch")
    dispatch.actions.fill_trace(rows, trace)
    trajectory_local = collector._trajectory_local_rows(
        dispatch.arena.view,
        rows,
        dispatch.arena.live,
    )
    if trajectory_local.size:
        _append_trajectory_rows(
            batch,
            dispatch,
            trace,
            trajectory_local,
            batch_start=batch_start,
            delivery=delivery,
            timings=timings,
            counters=counters,
            sequence_rows=sequence_rows,
            accepted_actions=accepted_actions,
        )


def _append_trajectory_rows(
    batch: NativeSimpleStatelessPolicyBatch,
    dispatch: NativeArenaDispatch,
    segment_trace: NativePolicyNumpyTrace,
    trajectory_local: Int64Array,
    *,
    batch_start: int,
    delivery: NativePartDelivery,
    timings: defaultdict[str, float],
    counters: Counter[str],
    sequence_rows: Sequence[SimpleStatelessActorRow] | None = None,
    accepted_actions: Sequence[AcceptedActionRecord] | None = None,
) -> None:
    input_started = time.perf_counter()
    rows = dispatch.current_rows
    trajectory_rows = rows[trajectory_local]
    trajectory_trace = select_native_policy_trace_rows(
        segment_trace,
        trajectory_local,
    )
    known = dispatch.arena.encoder.known_opponent_batch(
        dispatch.arena.view.slots[trajectory_rows],
        dispatch.arena.view.select_player[trajectory_rows],
    )
    trajectory_games = tuple(
        dispatch.arena.live[int(dispatch.arena.view.slots[row])]
        for row in trajectory_rows
    )
    selected_sequence_rows = (
        None
        if sequence_rows is None
        else tuple(sequence_rows[int(index)] for index in trajectory_local)
    )
    selected_actions = (
        None
        if accepted_actions is None
        else tuple(accepted_actions[int(index)] for index in trajectory_local)
    )
    seal_if_full = np.asarray(
        [getattr(game, "window_draining", False) for game in trajectory_games],
        dtype=np.bool_,
    )
    if np.any(seal_if_full):
        parts, sealed = dispatch.arena.page.append_or_seal_batch(
            batch,
            dispatch.arena.view.slots[trajectory_rows],
            known,
            trajectory_trace,
            seat_ids=dispatch.arena.view.select_player[trajectory_rows],
            seal_if_full=seal_if_full,
            batch_rows=batch_start + trajectory_local,
            sequence_rows=selected_sequence_rows,
            accepted_actions=selected_actions,
        )
    else:
        parts = dispatch.arena.page.append_batch(
            batch,
            dispatch.arena.view.slots[trajectory_rows],
            known,
            trajectory_trace,
            seat_ids=dispatch.arena.view.select_player[trajectory_rows],
            batch_rows=batch_start + trajectory_local,
            sequence_rows=selected_sequence_rows,
            accepted_actions=selected_actions,
        )
        sealed = np.zeros(trajectory_rows.size, dtype=np.bool_)
    delivery.accept(parts)
    for raw_row, game, row_sealed in zip(
        trajectory_rows,
        trajectory_games,
        sealed,
        strict=True,
    ):
        seat = int(dispatch.arena.view.select_player[raw_row])
        if bool(row_sealed):
            if not game.seal_trajectory_seat(seat):
                raise RuntimeError("native trajectory perspective sealed twice")
            counters["window_sealed_trajectory_perspectives"] += 1
            continue
        counters["trainable_trajectory_rows_appended"] += 1
        if seat == game.candidate_seat:
            game.candidate_decisions += 1
            counters["candidate_trajectory_rows"] += 1
        else:
            game.mirror_opponent_decisions += 1
            counters["mirror_opponent_trajectory_rows"] += 1
    timings["input"] += time.perf_counter() - input_started


def _submit_native_engine_facts(
    collector: NativeStatelessCollector,
    requests: Sequence[tuple[NativeArenaDispatch, Int64Array]],
    *,
    artifact_sha256: str | None = None,
) -> Future[NativeEngineFactBatch] | None:
    """Start exact probes before native rollout encoding enters the CPU path."""
    producer = collector.engine_fact_producer
    if producer is None:
        return None
    sources: list[NativeEngineFactSource] = []
    output_rows = 0
    output_options = 0
    for dispatch, rows in requests:
        view = dispatch.arena.view
        slots = view.slots[rows]
        perspectives = view.select_player[rows]
        option_counts = view.option_offsets[rows + 1] - view.option_offsets[rows]
        output_rows += int(rows.size)
        output_options = max(
            output_options,
            int(option_counts.max(initial=0)),
        )
        sources.append(
            NativeEngineFactSource(
                lane=dispatch.arena.lane,
                view=view,
                rows=rows,
                known=_encoder_for_artifact(
                    dispatch, artifact_sha256
                ).known_opponent_batch(
                    slots,
                    perspectives,
                ),
                live=dispatch.arena.live,
            )
        )
    if output_rows <= 0 or output_options <= 0:
        raise RuntimeError("native sequence fact wave has no policy options")
    return producer.submit(
        sources,
        output_shape=(output_rows, output_options),
    )


def _cancel_or_drain_native_engine_fact_futures(
    futures: Sequence[Future[NativeEngineFactBatch] | None],
) -> None:
    """Release every submitted fact reader before its arena can be reused."""
    for future in futures:
        if future is None:
            continue
        cancelled = False
        with suppress(BaseException):
            cancelled = future.cancel()
        if cancelled:
            continue
        with suppress(BaseException):
            future.result()


def _finish_native_engine_facts(
    future: Future[NativeEngineFactBatch] | None,
    options: Sequence[OptionBatch],
    *,
    timings: defaultdict[str, float],
    counters: Counter[str],
) -> None:
    """Join an overlapped fact wave and inject its exact option features."""
    if future is None:
        return
    wait_started_at = time.perf_counter()
    result = future.result()
    wait_seconds = time.perf_counter() - wait_started_at
    for option_batch in options:
        result.inject(option_batch)
    timings["engine_facts"] += result.elapsed_seconds
    timings["engine_fact_wait"] += wait_seconds
    timings["engine_fact_overlap"] += max(
        result.elapsed_seconds - wait_seconds,
        0.0,
    )
    timings["engine_fact_sample"] += result.sample_seconds
    timings["engine_fact_native"] += result.native_seconds
    timings["engine_fact_queue"] += result.queue_seconds
    timings["engine_fact_chunk_wall"] += result.chunk_wall_seconds
    counters["engine_fact_roots"] += result.roots
    counters["engine_fact_eligible_options"] += result.eligible_options
    counters["engine_fact_native_batch_calls"] += result.native_batch_calls
    counters["engine_fact_native_transitions"] += result.native_transitions
    counters["engine_fact_unresolved_worlds"] += result.unresolved_worlds
    counters["engine_fact_resolved_options"] += int(result.masks.sum())


def _current_policy_temperature(collector: NativeStatelessCollector) -> float:
    """Read the collector decode mode while preserving the training default."""
    return float(getattr(collector, "current_policy_temperature", 1.0))


def _frozen_policy_temperature(
    collector: NativeStatelessCollector,
    artifact_sha256: str,
) -> float:
    """Read an exact frozen decode mode while preserving the training default."""
    accessor = getattr(collector, "frozen_policy_temperature", None)
    return 1.0 if accessor is None else float(accessor(artifact_sha256))


def _evaluation_action_only(collector: NativeStatelessCollector) -> bool:
    """Return whether structural traces are legal for this collection."""
    return bool(getattr(collector, "evaluation_action_only", False))


def _route_generator(
    collector: NativeStatelessCollector,
    route: NativeRouteKey,
    *,
    device: torch.device,
    generators: dict[NativeRouteKey, torch.Generator],
) -> torch.Generator:
    existing = generators.get(route)
    if existing is not None:
        return existing
    generator = torch.Generator(device=device)
    generator.manual_seed(native_route_generator_seed(collector.seed, route))
    generators[route] = generator
    return generator


def _delivery_for(
    dispatch: NativeArenaDispatch,
    delivery: NativePartDelivery | Mapping[int, NativePartDelivery],
) -> NativePartDelivery:
    if isinstance(delivery, NativePartDelivery):
        return delivery
    try:
        return delivery[id(dispatch.arena)]
    except KeyError as error:
        raise RuntimeError(
            "native inference delivery is absent for an arena"
        ) from error


def _rollout_source(
    dispatch: NativeArenaDispatch,
    rows: Int64Array,
) -> NativeRolloutSource:
    """Describe selected slots without copying any nested engine columns."""
    return NativeRolloutSource.from_arrays(
        dispatch.arena.encoder,
        dispatch.arena.view.slots[rows],
        dispatch.arena.view.select_player[rows],
    )


def _artifact_rollout_source(
    dispatch: NativeArenaDispatch,
    rows: Int64Array,
    *,
    artifact_sha256: str,
) -> NativeRolloutSource:
    """Select a route encoder, falling back for synthetic legacy arenas."""
    resolver = getattr(dispatch.arena, "encoder_for_artifact", None)
    if not callable(resolver):
        return _rollout_source(dispatch, rows)
    return NativeRolloutSource.from_arrays(
        resolver(artifact_sha256),
        dispatch.arena.view.slots[rows],
        dispatch.arena.view.select_player[rows],
    )


def _encoder_for_artifact(
    dispatch: NativeArenaDispatch,
    artifact_sha256: str | None,
) -> Any:
    """Resolve an artifact encoder without widening synthetic test fixtures."""
    if artifact_sha256 is None:
        return dispatch.arena.encoder
    resolver = getattr(dispatch.arena, "encoder_for_artifact", None)
    return resolver(artifact_sha256) if callable(resolver) else dispatch.arena.encoder


def _concatenate_model_views(
    batches: Sequence[NativeTrainingBatchView],
) -> NativeTrainingBatchView:
    """Concatenate arena-local rows under temporary model-only identities."""
    if not batches:
        raise ValueError("native inference concatenation requires rows")
    if len(batches) == 1:
        return batches[0]
    rebased: list[NativeTrainingBatchView] = []
    next_row = 0
    for view in batches:
        stop = next_row + view.batch_size
        rebased.append(
            replace(
                view,
                slots=np.arange(next_row, stop, dtype=np.uint32),
            )
        )
        next_row = stop
    return concatenate_native_training_views(rebased)


def _concatenate_known_opponents(
    batches: Sequence[NativeKnownOpponentBatch],
) -> NativeKnownOpponentBatch:
    if not batches:
        raise ValueError("known-opponent concatenation requires rows")
    lengths = np.concatenate(
        tuple(np.diff(batch.offsets.astype(np.uint64, copy=False)) for batch in batches)
    )
    total_values = int(lengths.sum(dtype=np.uint64))
    if total_values > int(np.iinfo(np.uint32).max):
        raise OverflowError("known-opponent concatenation exceeds uint32 capacity")
    offsets = np.zeros(lengths.size + 1, dtype=np.uint32)
    np.cumsum(lengths, dtype=np.uint32, out=offsets[1:])
    return NativeKnownOpponentBatch(
        offsets=offsets,
        card_ids=np.concatenate(tuple(batch.card_ids for batch in batches)),
        counts=np.concatenate(tuple(batch.counts for batch in batches)),
    )


def _select_action_rows(
    actions: NativePolicyNumpyActionBatch,
    *,
    start: int,
    stop: int,
) -> NativePolicyNumpyActionBatch:
    if start < 0 or stop <= start or stop > actions.batch_size:
        raise ValueError("native action batch slice is invalid")
    choice_start = int(actions.action_offsets[start])
    choice_stop = int(actions.action_offsets[stop])
    return NativePolicyNumpyActionBatch(
        identity=actions.identity,
        action_offsets=(actions.action_offsets[start : stop + 1].copy() - choice_start),
        action_choices=actions.action_choices[choice_start:choice_stop].copy(),
    )


__all__ = [
    "PendingNativePastSubmissions",
    "PendingNativeRouteSubmission",
    "PreparedNativeCurrent",
    "PreparedNativePast",
    "apply_current_trace",
    "begin_current",
    "begin_past",
    "prepare_current",
    "prepare_past",
    "resolve_past_executors",
    "resume_current",
    "resume_past",
    "serve_current",
    "serve_historical",
    "serve_past",
    "submit_current",
    "submit_past",
]
