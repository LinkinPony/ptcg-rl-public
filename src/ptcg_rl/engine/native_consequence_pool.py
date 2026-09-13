"""Persistent bounded pool of isolated native consequence lanes."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from types import TracebackType
from typing import Self

from ptcg_rl.engine.native_consequence import NativeConsequenceLane
from ptcg_rl.engine.native_consequence_pool_contract import (
    NativeConsequenceCall,
    NativeConsequencePoolConfig,
    NativeConsequencePoolResult,
    NativeConsequencePoolStats,
    maximum_call_host_bytes,
)
from ptcg_rl.runtime.work_ledger import (
    PlannerRequestLedger,
    PlannerWorkReservation,
    PlannerWorkStopReason,
)


class NativeConsequencePoolError(RuntimeError):
    """Base error for bounded native-lane service failures."""


class NativeConsequencePoolSaturatedError(NativeConsequencePoolError):
    """Raised when bounded admission has no available job slot."""


class NativeConsequencePoolDeadlineError(NativeConsequencePoolError):
    """Raised when no native chunk can finish before the return guard."""


class NativeConsequenceLanePool:
    """Own fixed native arenas and a bounded persistent executor.

    ``ctypes.CDLL`` calls release the GIL while native engine work executes.
    Each worker checks the request-global return guard before acquiring a lane;
    an unexpectedly slow call may finish in the background but can never cause
    a replacement call to exceed the same deterministic ledger.
    """

    def __init__(
        self,
        config: NativeConsequencePoolConfig,
        *,
        library_path: Path | str | None = None,
        lane_factory: Callable[[], NativeConsequenceLane] | None = None,
    ) -> None:
        self.config = config
        factory = lane_factory or (
            lambda: NativeConsequenceLane(library_path=library_path)
        )
        created: list[NativeConsequenceLane] = []
        try:
            for _ in range(config.lane_count):
                created.append(factory())
        except Exception:
            for lane in created:
                lane.close()
            raise
        self._lanes: queue.LifoQueue[NativeConsequenceLane] = queue.LifoQueue(
            maxsize=config.lane_count
        )
        for lane in created:
            self._lanes.put_nowait(lane)
        self._executor = ThreadPoolExecutor(
            max_workers=config.lane_count,
            thread_name_prefix="native-consequence",
        )
        self._admission = threading.BoundedSemaphore(config.max_inflight_jobs)
        self._lock = threading.Lock()
        self._closed = False
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._saturated = 0
        self._deadline_rejected = 0
        self._active = 0
        self._peak_active = 0
        self._queue_wait_seconds = 0.0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def submit(
        self,
        call: NativeConsequenceCall,
        *,
        ledger: PlannerRequestLedger,
    ) -> Future[NativeConsequencePoolResult]:
        """Admit one call without ever growing an unbounded executor queue."""
        if call.transitions <= 0:
            raise ValueError("native consequence call must contain at least one cell")
        if call.transitions > self.config.max_transitions_per_call:
            raise ValueError("native consequence call exceeds pool chunk capacity")
        reservation = ledger.reserve(
            transitions=call.transitions,
            native_calls=1,
            host_bytes=maximum_call_host_bytes(call),
            expected_seconds=max(
                self.config.call_guard_seconds,
                ledger.limits.native_call_guard_seconds,
            ),
        )
        if reservation is None:
            stop_reason = ledger.snapshot().stop_reason
            with self._lock:
                self._deadline_rejected += int(
                    stop_reason is PlannerWorkStopReason.DEADLINE_GUARD
                )
            if stop_reason is PlannerWorkStopReason.DEADLINE_GUARD:
                raise NativeConsequencePoolDeadlineError(
                    "native call rejected by the request deadline guard"
                )
            raise NativeConsequencePoolError(
                f"native call rejected by request work limit: {stop_reason}"
            )
        if not self._admission.acquire(blocking=False):
            ledger.complete(reservation, elapsed_seconds=0.0, success=False)
            ledger.stop(PlannerWorkStopReason.NATIVE_POOL_SATURATED)
            with self._lock:
                self._saturated += 1
            raise NativeConsequencePoolSaturatedError("native job admission is full")
        with self._lock:
            if self._closed:
                self._admission.release()
                ledger.complete(reservation, elapsed_seconds=0.0, success=False)
                raise RuntimeError("cannot submit to a closed native lane pool")
            self._submitted += 1
        try:
            future = self._executor.submit(
                self._run_reserved, call, ledger, reservation
            )
        except Exception:
            self._admission.release()
            ledger.complete(reservation, elapsed_seconds=0.0, success=False)
            raise
        future.add_done_callback(lambda _future: self._admission.release())
        return future

    def run(
        self,
        call: NativeConsequenceCall,
        *,
        ledger: PlannerRequestLedger,
    ) -> NativeConsequencePoolResult:
        """Wait only until the request return guard, never for a slow native call.

        Native engine calls cannot be interrupted safely. On an unexpected
        overrun the worker retains its lane and completes cleanup in the
        background, while this caller can immediately take the base fallback.
        """
        future = self.submit(call, ledger=ledger)
        remaining = max(0.0, ledger.deadline_monotonic - time.monotonic())
        try:
            return future.result(timeout=remaining)
        except FutureTimeout as exc:
            ledger.stop(PlannerWorkStopReason.DEADLINE_GUARD)
            with self._lock:
                self._deadline_rejected += 1
            raise NativeConsequencePoolDeadlineError(
                "native call exceeded the request return deadline"
            ) from exc

    def close(self) -> None:
        """Drain bounded in-flight work and destroy every native lane once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        while True:
            try:
                lane = self._lanes.get_nowait()
            except queue.Empty:
                break
            lane.close()

    def stats(self) -> NativeConsequencePoolStats:
        """Return an immutable service telemetry snapshot."""
        with self._lock:
            return NativeConsequencePoolStats(
                lanes=self.config.lane_count,
                submitted=self._submitted,
                completed=self._completed,
                failed=self._failed,
                saturated=self._saturated,
                deadline_rejected=self._deadline_rejected,
                active=self._active,
                peak_active=self._peak_active,
                queue_wait_seconds=self._queue_wait_seconds,
            )

    def _run_reserved(
        self,
        call: NativeConsequenceCall,
        ledger: PlannerRequestLedger,
        reservation: PlannerWorkReservation,
    ) -> NativeConsequencePoolResult:
        started = time.perf_counter()
        wait_started = time.perf_counter()
        start_deadline = ledger.deadline_monotonic - max(
            self.config.call_guard_seconds,
            ledger.limits.native_call_guard_seconds,
        )
        timeout = max(0.0, start_deadline - time.monotonic())
        if timeout <= 0.0:
            self._record_failure(deadline=True)
            ledger.complete(
                reservation,
                elapsed_seconds=time.perf_counter() - started,
                success=False,
            )
            raise NativeConsequencePoolDeadlineError("native lane start guard expired")
        try:
            lane = self._lanes.get(timeout=timeout)
        except queue.Empty as exc:
            self._record_failure(deadline=True)
            ledger.complete(
                reservation,
                elapsed_seconds=time.perf_counter() - started,
                success=False,
            )
            raise NativeConsequencePoolDeadlineError(
                "no native lane became available before the start guard"
            ) from exc
        queue_wait = time.perf_counter() - wait_started
        with self._lock:
            self._active += 1
            self._peak_active = max(self._peak_active, self._active)
            self._queue_wait_seconds += queue_wait
        success = False
        try:
            batch = lane.run(
                call.state_token,
                hidden_worlds=call.hidden_worlds,
                candidate_actions=call.candidate_actions,
                producer_contract_fingerprint=call.producer_contract_fingerprint,
                root_player=call.root_player,
                manual_coin=call.manual_coin,
                stochastic_seed=call.stochastic_seed,
                max_cells=call.max_cells,
                max_engine_steps=call.max_engine_steps,
                max_forced_steps=call.max_forced_steps,
                max_observation_bytes=call.max_observation_bytes,
            )
            success = True
            return NativeConsequencePoolResult(
                batch=batch,
                queue_wait_seconds=queue_wait,
            )
        finally:
            self._lanes.put_nowait(lane)
            elapsed = time.perf_counter() - started
            ledger.complete(reservation, elapsed_seconds=elapsed, success=success)
            with self._lock:
                self._active -= 1
                self._completed += int(success)
                self._failed += int(not success)

    def _record_failure(self, *, deadline: bool) -> None:
        with self._lock:
            self._failed += 1
            self._deadline_rejected += int(deadline)


__all__ = [
    "NativeConsequenceCall",
    "NativeConsequenceLanePool",
    "NativeConsequencePoolConfig",
    "NativeConsequencePoolDeadlineError",
    "NativeConsequencePoolError",
    "NativeConsequencePoolResult",
    "NativeConsequencePoolSaturatedError",
    "NativeConsequencePoolStats",
]
