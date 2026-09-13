"""Bounded post-behavior lane for native counterfactual macro teaching."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

from ptcg_rl.rl.engine_teacher import (
    EngineTeacherBatchCompletion,
    EngineTeacherRequest,
    EngineTeacherTarget,
)
from ptcg_rl.rl.macro_teacher import MacroTeacherRequest
from ptcg_rl.rl.planner_behavior_service import (
    PlannerBehaviorService,
    release_rollout_planner_batch_contexts,
)
from ptcg_rl.runtime.planner_telemetry import PlannerTelemetryAccumulator


@dataclass(frozen=True, slots=True)
class _QueuedMacroBatch:
    batch_id: int
    requests: tuple[MacroTeacherRequest, ...]


_STOP = object()


class BackgroundMacroTeacherProducer:
    """Resolve retained native planner batches without changing PPO behavior."""

    def __init__(
        self,
        service: PlannerBehaviorService,
        *,
        queue_batches: int,
    ) -> None:
        if queue_batches <= 0:
            raise ValueError("background macro teacher queue must be positive")
        self._service = service
        self._tasks: queue.Queue[_QueuedMacroBatch | object] = queue.Queue(
            maxsize=queue_batches
        )
        self._results: queue.Queue[EngineTeacherBatchCompletion] = queue.Queue()
        self._pending: set[int] = set()
        self._pending_lock = threading.Lock()
        self._summary_lock = threading.Lock()
        self._latencies_ms: deque[float] = deque(maxlen=4_096)
        self._next_batch_id = 0
        self._submitted_batches = 0
        self._submitted_requests = 0
        self._completed_batches = 0
        self._completed_requests = 0
        self._resolved_targets = 0
        self._saturated_batches = 0
        self._saturated_requests = 0
        self._cleanup_failures = 0
        self._last_cleanup_error = ""
        self._worker_failures = 0
        self._last_worker_error = ""
        self._eligible_requests = 0
        self._attempted_requests = 0
        self._absent_reasons: Counter[str] = Counter()
        self._planner_telemetry = PlannerTelemetryAccumulator(latency_window=4_096)
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="native-macro-teacher-background-lane",
            daemon=True,
        )
        self._thread.start()

    def produce(self, request: EngineTeacherRequest) -> EngineTeacherTarget | None:
        """Reject accidental synchronous use."""
        del request
        raise RuntimeError("native macro teacher requires submit_async")

    def submit_async(
        self,
        requests: Sequence[EngineTeacherRequest | MacroTeacherRequest],
    ) -> int | None:
        """Queue retained root batches after behavior has reached the engine."""
        if self._closed:
            raise RuntimeError("background macro teacher is closed")
        raw_requests = tuple(requests)
        frozen = cast(tuple[MacroTeacherRequest, ...], raw_requests)
        if not frozen:
            raise ValueError("macro teacher batch cannot be empty")
        if any(not isinstance(request, MacroTeacherRequest) for request in frozen):
            raise TypeError("native macro teacher received a legacy request")
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        task = _QueuedMacroBatch(batch_id=batch_id, requests=frozen)
        try:
            self._tasks.put_nowait(task)
        except queue.Full:
            self._release_unique_batches(frozen)
            with self._summary_lock:
                self._saturated_batches += 1
                self._saturated_requests += len(frozen)
            return None
        with self._pending_lock:
            self._pending.add(batch_id)
        with self._summary_lock:
            self._submitted_batches += 1
            self._submitted_requests += len(frozen)
        return batch_id

    def poll_completed(self) -> tuple[EngineTeacherBatchCompletion, ...]:
        """Return every result currently available without blocking rollout."""
        completed: list[EngineTeacherBatchCompletion] = []
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            self._consume_pending(result.batch_id)
            completed.append(result)
        return tuple(completed)

    def drain(self) -> tuple[EngineTeacherBatchCompletion, ...]:
        """Wait for accepted native work at the orderly actor boundary."""
        completed = list(self.poll_completed())
        while self._pending_count() > 0:
            result = self._results.get()
            self._consume_pending(result.batch_id)
            completed.append(result)
        return tuple(completed)

    def summary(self) -> Mapping[str, Any]:
        """Return coverage, latency, and bounded-queue diagnostics."""
        with self._summary_lock:
            latencies = tuple(self._latencies_ms)
            submitted = self._submitted_requests
            resolved = self._resolved_targets
            summary: dict[str, Any] = {
                "macro_teacher_async": True,
                "macro_teacher_submitted_batches": self._submitted_batches,
                "macro_teacher_submitted_requests": submitted,
                "macro_teacher_completed_batches": self._completed_batches,
                "macro_teacher_completed_requests": self._completed_requests,
                "macro_teacher_resolved_targets": resolved,
                "macro_teacher_coverage": resolved / submitted if submitted else 0.0,
                "macro_teacher_saturated_batches": self._saturated_batches,
                "macro_teacher_saturated_requests": self._saturated_requests,
                "macro_teacher_cleanup_failures": self._cleanup_failures,
                "macro_teacher_last_cleanup_error": self._last_cleanup_error,
                "macro_teacher_worker_failures": self._worker_failures,
                "macro_teacher_last_worker_error": self._last_worker_error,
                "macro_teacher_eligible_requests": self._eligible_requests,
                "macro_teacher_attempted_requests": self._attempted_requests,
                "macro_teacher_absent_reasons": dict(
                    sorted(self._absent_reasons.items())
                ),
                "macro_teacher_latency_p50_ms": _percentile(latencies, 0.50),
                "macro_teacher_latency_p95_ms": _percentile(latencies, 0.95),
            }
        summary["macro_teacher_pending_batches"] = self._pending_count()
        summary["macro_teacher_queue_depth"] = self._tasks.qsize()
        summary["macro_teacher_planner"] = asdict(self._planner_telemetry.summary())
        service_stats = self._service.stats()
        summary["macro_teacher_native_peak_active_rows"] = (
            service_stats.peak_active_rows
        )
        return summary

    def close(self) -> None:
        """Drain accepted work before its planner runtime is released."""
        if self._closed:
            return
        self.drain()
        self._closed = True
        self._tasks.put(_STOP)
        self._thread.join()
        if self._thread.is_alive():  # pragma: no cover - unbounded host failure
            raise RuntimeError("background macro teacher did not stop")

    def _run(self) -> None:
        while True:
            task = self._tasks.get()
            if task is _STOP:
                return
            queued = cast(_QueuedMacroBatch, task)
            started = time.perf_counter()
            targets: tuple[EngineTeacherTarget | None, ...]
            try:
                targets = self._resolve(queued.requests)
            except Exception as exc:  # Auxiliary evidence is fail-open.
                targets = (None,) * len(queued.requests)
                with self._summary_lock:
                    self._worker_failures += 1
                    self._last_worker_error = _error_detail(exc)
            elapsed_ms = (time.perf_counter() - started) * 1_000.0
            with self._summary_lock:
                self._latencies_ms.append(elapsed_ms)
                self._completed_batches += 1
                self._completed_requests += len(queued.requests)
                self._resolved_targets += sum(target is not None for target in targets)
            self._results.put(
                EngineTeacherBatchCompletion(
                    batch_id=queued.batch_id,
                    targets=targets,
                )
            )

    def _resolve(
        self,
        requests: tuple[MacroTeacherRequest, ...],
    ) -> tuple[EngineTeacherTarget | None, ...]:
        results: list[EngineTeacherTarget | None] = [None] * len(requests)
        groups: dict[int, tuple[Any, list[tuple[int, int]]]] = {}
        for result_index, request in enumerate(requests):
            key = id(request.planner_batch)
            if key not in groups:
                groups[key] = (request.planner_batch, [])
            groups[key][1].append((result_index, request.row_index))
        for batch, requested_rows in groups.values():
            planned = tuple(self._service.plan_batch(batch))
            if len(planned) != len(tuple(batch.rows)):
                raise RuntimeError("native macro planner returned the wrong batch size")
            for result_index, row_index in requested_rows:
                decision = planned[row_index]
                self._record_planner_decision(decision)
                results[result_index] = (
                    None if decision is None else decision.macro_teacher_target
                )
        return tuple(results)

    def _record_planner_decision(self, decision: Any | None) -> None:
        """Accumulate diagnostic eligibility, absence, and stage telemetry."""
        if decision is None:
            with self._summary_lock:
                self._absent_reasons["not_candidate"] += 1
            return
        fallback = decision.planner_behavior.fallback_reason
        target = decision.macro_teacher_target
        reason = fallback.name.lower() if decision.used_base_trace else None
        with self._summary_lock:
            if reason != "ineligible":
                self._eligible_requests += 1
            if reason not in {"ineligible", "model_lease_capacity", "queue_full"}:
                self._attempted_requests += 1
            if target is None:
                self._absent_reasons[reason or "evidence_absent"] += 1
        runtime = decision.runtime_stats
        self._planner_telemetry.record_decision(
            decision.telemetry_events,
            planner_used=not decision.used_base_trace,
            fallback_reason=reason,
            cache_hits={
                "root_context": runtime.root_context_hits,
                "engine_reuse": runtime.engine_reuse_hits,
                "leaf_reuse": runtime.leaf_reuse_hits,
                "prefix_reuse": runtime.prefix_reuse_count,
            },
            cache_misses={
                "root_context": runtime.root_context_misses,
                "engine_reuse": runtime.engine_reuse_misses,
                "leaf_reuse": runtime.leaf_reuse_misses,
            },
            native_occupancy=self._service.stats().peak_active_rows,
        )

    def _release_unique_batches(
        self,
        requests: Sequence[MacroTeacherRequest],
    ) -> None:
        released: set[int] = set()
        for request in requests:
            key = id(request.planner_batch)
            if key in released:
                continue
            try:
                release_rollout_planner_batch_contexts(request.planner_batch)
            except Exception as exc:
                # Rejected auxiliary work is fail-open. Remote contexts retain
                # a bounded server-side TTL, so a congested cleanup RPC must
                # not terminate the rollout process.
                with self._summary_lock:
                    self._cleanup_failures += 1
                    self._last_cleanup_error = _error_detail(exc)
            released.add(key)

    def _pending_count(self) -> int:
        with self._pending_lock:
            return len(self._pending)

    def _consume_pending(self, batch_id: int) -> None:
        with self._pending_lock:
            if batch_id not in self._pending:
                raise RuntimeError("unknown background macro teacher completion")
            self._pending.remove(batch_id)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _error_detail(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:512]}"


__all__ = ["BackgroundMacroTeacherProducer"]
