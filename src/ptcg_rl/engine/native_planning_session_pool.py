"""Persistent bounded pool of pinned native planning-session lanes."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from types import TracebackType
from typing import Self

from ptcg_rl.engine.native_planning_session import (
    NativePlanningSession,
    NativePlanningSessionBatchResult,
    NativePlanningSessionHandle,
    NativePlanningSessionLane,
)
from ptcg_rl.engine.native_planning_session_payload import (
    native_planning_session_schema_fingerprint,
)
from ptcg_rl.engine.native_planning_session_pool_contract import (
    NativePlanningSessionContinueCall,
    NativePlanningSessionOpenCall,
    NativePlanningSessionPoolConfig,
    NativePlanningSessionPoolResult,
    NativePlanningSessionPoolStats,
    maximum_continue_host_bytes,
    maximum_open_host_bytes,
)
from ptcg_rl.runtime.work_ledger import (
    PlannerRequestLedger,
    PlannerWorkReservation,
    PlannerWorkStopReason,
)


class NativePlanningSessionPoolError(RuntimeError):
    """Base error for bounded pinned-session service failures."""


class NativePlanningSessionPoolSaturatedError(NativePlanningSessionPoolError):
    """Raised when no bounded service or pinned-lane slot is available."""


class NativePlanningSessionPoolDeadlineError(NativePlanningSessionPoolError):
    """Raised when a native call cannot finish before the return guard."""


class NativePlanningSessionLease:
    """One request-local session that pins its native lane until close."""

    def __init__(
        self,
        *,
        pool: NativePlanningSessionLanePool,
        lane: NativePlanningSessionLane,
        session: NativePlanningSession,
        open_queue_wait_seconds: float,
    ) -> None:
        self._pool = pool
        self._lane = lane
        self._session = session
        self._operation_lock = threading.RLock()
        self._released = False
        self._abandoned = False
        self.open_queue_wait_seconds = open_queue_wait_seconds

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("cannot enter a closed planning-session lease")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    @property
    def initial_result(self) -> NativePlanningSessionBatchResult:
        """Return the validated candidate-major OPEN response."""
        return self._session.initial_result

    @property
    def generation(self) -> int:
        """Return the lane-local generation identity."""
        return self._session.generation

    @property
    def closed(self) -> bool:
        """Return whether the lease can execute another continuation."""
        with self._operation_lock:
            return self._released or self._abandoned or self._session.closed

    def continue_batch(
        self,
        call: NativePlanningSessionContinueCall,
        *,
        ledger: PlannerRequestLedger,
    ) -> NativePlanningSessionPoolResult:
        """Advance aligned handles under the same global request ledger."""
        return self._pool._continue(self, call, ledger=ledger)

    def release(self, handles: Sequence[NativePlanningSessionHandle]) -> None:
        """Release unchosen lane-local handles without serializing them."""
        with self._operation_lock:
            self._require_active()
            self._session.release(handles)

    def close(self) -> None:
        """Close the generation and return or replace its lane exactly once."""
        self._pool._close_lease(self)

    def _require_active(self) -> None:
        if self._released or self._abandoned or self._session.closed:
            raise RuntimeError("planning-session lease is no longer active")


class NativePlanningSessionLanePool:
    """Own fixed native arenas and lend each lane to one whole search tree."""

    def __init__(
        self,
        config: NativePlanningSessionPoolConfig,
        *,
        library_path: Path | str | None = None,
        lane_factory: Callable[[], NativePlanningSessionLane] | None = None,
    ) -> None:
        self.config = config
        self._factory = lane_factory or (
            lambda: NativePlanningSessionLane(library_path=library_path)
        )
        created: list[NativePlanningSessionLane] = []
        try:
            for _ in range(config.lane_count):
                created.append(self._factory())
        except Exception:
            for lane in created:
                lane.close()
            raise
        library_fingerprints = {
            str(getattr(lane, "engine_library_fingerprint", ""))
            for lane in created
        }
        abi_fingerprints = {
            str(getattr(lane, "native_abi_fingerprint", ""))
            for lane in created
        }
        if len(library_fingerprints) != 1 or len(abi_fingerprints) != 1:
            for lane in created:
                lane.close()
            raise RuntimeError("native planning lanes loaded different engine identities")
        self._engine_library_fingerprint = library_fingerprints.pop()
        self._native_abi_fingerprint = abi_fingerprints.pop()
        self._native_schema_fingerprint = native_planning_session_schema_fingerprint()
        self._available: queue.LifoQueue[NativePlanningSessionLane] = queue.LifoQueue(
            maxsize=config.lane_count
        )
        for lane in created:
            self._available.put_nowait(lane)
        self._executor = ThreadPoolExecutor(
            max_workers=config.lane_count,
            thread_name_prefix="native-planning-session",
        )
        self._admission = threading.BoundedSemaphore(config.max_inflight_jobs)
        self._lock = threading.Lock()
        self._leases: set[NativePlanningSessionLease] = set()
        self._closed = False
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._saturated = 0
        self._deadline_rejected = 0
        self._active_calls = 0
        self._active_sessions = 0
        self._peak_active_calls = 0
        self._lanes_replaced = 0
        self._queue_wait_seconds = 0.0

    @property
    def engine_library_fingerprint(self) -> str:
        """Return the exact library content shared by every resident lane."""
        return self._engine_library_fingerprint

    @property
    def native_abi_fingerprint(self) -> str:
        """Return the exact validated v5 ABI shared by every lane."""
        return self._native_abi_fingerprint

    @property
    def native_schema_fingerprint(self) -> str:
        """Return the Python v5 payload and endpoint schema identity."""
        return self._native_schema_fingerprint

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

    def open_session(
        self,
        call: NativePlanningSessionOpenCall,
        *,
        ledger: PlannerRequestLedger,
    ) -> NativePlanningSessionLease:
        """Open a root grid and yield a session that keeps its lane pinned."""
        self._validate_transitions(call.transitions)
        reservation = self._reserve(
            ledger,
            transitions=call.transitions,
            host_bytes=maximum_open_host_bytes(call),
        )
        self._admit(ledger, reservation)
        wait_started = time.perf_counter()
        lane = self._take_lane(ledger, reservation)
        queue_wait = time.perf_counter() - wait_started
        submitted = time.perf_counter()
        with self._lock:
            if self._closed:
                self._available.put_nowait(lane)
                self._admission.release()
                ledger.complete(reservation, elapsed_seconds=0.0, success=False)
                raise RuntimeError("cannot open a session on a closed pool")
            self._submitted += 1
        try:
            future = self._executor.submit(
                self._run_open,
                lane,
                call,
                ledger,
                reservation,
                queue_wait,
                submitted,
            )
        except Exception:
            self._return_or_replace_lane(lane)
            self._admission.release()
            ledger.complete(reservation, elapsed_seconds=0.0, success=False)
            raise
        future.add_done_callback(lambda _future: self._admission.release())
        remaining = max(0.0, ledger.deadline_monotonic - time.monotonic())
        try:
            session = future.result(timeout=remaining)
        except FutureTimeout as exc:
            ledger.stop(PlannerWorkStopReason.DEADLINE_GUARD)
            with self._lock:
                self._deadline_rejected += 1
            future.add_done_callback(self._cleanup_abandoned_open)
            raise NativePlanningSessionPoolDeadlineError(
                "planning-session OPEN exceeded the request return deadline"
            ) from exc
        lease = NativePlanningSessionLease(
            pool=self,
            lane=lane,
            session=session,
            open_queue_wait_seconds=queue_wait,
        )
        with self._lock:
            if self._closed:
                close_immediately = True
            else:
                close_immediately = False
                self._leases.add(lease)
                self._active_sessions += 1
        if close_immediately:
            self._discard_session(lane, session)
            raise RuntimeError("planning-session pool closed during OPEN")
        return lease

    def close(self) -> None:
        """Drain calls, close live generations, and destroy all lanes."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._lock:
            leases = tuple(self._leases)
        for lease in leases:
            lease.close()
        while True:
            try:
                lane = self._available.get_nowait()
            except queue.Empty:
                break
            lane.close()

    def stats(self) -> NativePlanningSessionPoolStats:
        """Return an immutable service telemetry snapshot."""
        with self._lock:
            return NativePlanningSessionPoolStats(
                lanes=self.config.lane_count,
                submitted=self._submitted,
                completed=self._completed,
                failed=self._failed,
                saturated=self._saturated,
                deadline_rejected=self._deadline_rejected,
                active_calls=self._active_calls,
                active_sessions=self._active_sessions,
                peak_active_calls=self._peak_active_calls,
                lanes_replaced=self._lanes_replaced,
                queue_wait_seconds=self._queue_wait_seconds,
            )

    def _continue(
        self,
        lease: NativePlanningSessionLease,
        call: NativePlanningSessionContinueCall,
        *,
        ledger: PlannerRequestLedger,
    ) -> NativePlanningSessionPoolResult:
        self._validate_transitions(call.transitions)
        if len(call.handles) != len(call.actions):
            raise ValueError("continuation handles and actions must align")
        with lease._operation_lock:  # pylint: disable=protected-access
            lease._require_active()  # pylint: disable=protected-access
            if any(handle.generation != lease.generation for handle in call.handles):
                raise ValueError("continuation handles differ from the lease")
            reservation = self._reserve(
                ledger,
                transitions=call.transitions,
                host_bytes=maximum_continue_host_bytes(call),
            )
            self._admit(ledger, reservation)
            submitted = time.perf_counter()
            with self._lock:
                if self._closed:
                    self._admission.release()
                    ledger.complete(
                        reservation,
                        elapsed_seconds=0.0,
                        success=False,
                    )
                    raise RuntimeError("cannot continue a session on a closed pool")
                self._submitted += 1
            try:
                future = self._executor.submit(
                    self._run_continue,
                    lease,
                    call,
                    ledger,
                    reservation,
                    submitted,
                )
            except Exception:
                self._admission.release()
                ledger.complete(reservation, elapsed_seconds=0.0, success=False)
                raise
            future.add_done_callback(lambda _future: self._admission.release())
            remaining = max(0.0, ledger.deadline_monotonic - time.monotonic())
            try:
                return future.result(timeout=remaining)
            except FutureTimeout as exc:
                lease._abandoned = True  # pylint: disable=protected-access
                ledger.stop(PlannerWorkStopReason.DEADLINE_GUARD)
                with self._lock:
                    self._deadline_rejected += 1
                future.add_done_callback(
                    lambda _future: self._cleanup_abandoned_lease(lease)
                )
                raise NativePlanningSessionPoolDeadlineError(
                    "planning-session CONTINUE exceeded the request return deadline"
                ) from exc
            except Exception:
                lease._released = True  # pylint: disable=protected-access
                self._retire_lease(lease)
                raise

    def _run_open(
        self,
        lane: NativePlanningSessionLane,
        call: NativePlanningSessionOpenCall,
        ledger: PlannerRequestLedger,
        reservation: PlannerWorkReservation,
        queue_wait: float,
        submitted: float,
    ) -> NativePlanningSession:
        started = time.perf_counter()
        self._record_call_start(queue_wait + max(0.0, started - submitted))
        success = False
        try:
            session = lane.open_session(
                call.state_token,
                hidden_worlds=call.hidden_worlds,
                candidate_actions=call.candidate_actions,
                producer_contract_fingerprint=call.producer_contract_fingerprint,
                root_player=call.root_player,
                manual_coin=call.manual_coin,
                max_state_slots=call.max_state_slots,
                caps=call.caps,
            )
            success = True
            return session
        finally:
            elapsed = time.perf_counter() - started
            ledger.complete(reservation, elapsed_seconds=elapsed, success=success)
            self._record_call_finish(success)
            if not success:
                self._return_or_replace_lane(lane)

    def _run_continue(
        self,
        lease: NativePlanningSessionLease,
        call: NativePlanningSessionContinueCall,
        ledger: PlannerRequestLedger,
        reservation: PlannerWorkReservation,
        submitted: float,
    ) -> NativePlanningSessionPoolResult:
        started = time.perf_counter()
        queue_wait = max(0.0, started - submitted)
        self._record_call_start(queue_wait)
        success = False
        try:
            batch = lease._session.continue_batch(  # pylint: disable=protected-access
                call.handles,
                call.actions,
                caps=call.caps,
            )
            success = True
            return NativePlanningSessionPoolResult(
                batch=batch,
                queue_wait_seconds=queue_wait,
            )
        finally:
            elapsed = time.perf_counter() - started
            ledger.complete(reservation, elapsed_seconds=elapsed, success=success)
            self._record_call_finish(success)

    def _close_lease(self, lease: NativePlanningSessionLease) -> None:
        with lease._operation_lock:  # pylint: disable=protected-access
            if lease._released:  # pylint: disable=protected-access
                return
            if lease._abandoned:  # pylint: disable=protected-access
                return
            lease._released = True  # pylint: disable=protected-access
            self._retire_lease(lease)

    def _retire_lease(self, lease: NativePlanningSessionLease) -> None:
        try:
            lease._session.close()  # pylint: disable=protected-access
        finally:
            with self._lock:
                if lease in self._leases:
                    self._leases.remove(lease)
                    self._active_sessions -= 1
            self._return_or_replace_lane(lease._lane)  # pylint: disable=protected-access

    def _cleanup_abandoned_open(
        self,
        future: Future[NativePlanningSession],
    ) -> None:
        try:
            session = future.result()
        except Exception:
            return
        lane = session._lane  # pylint: disable=protected-access
        self._discard_session(lane, session)

    def _cleanup_abandoned_lease(self, lease: NativePlanningSessionLease) -> None:
        with lease._operation_lock:  # pylint: disable=protected-access
            if lease._released:  # pylint: disable=protected-access
                return
            lease._released = True  # pylint: disable=protected-access
            self._retire_lease(lease)

    def _discard_session(
        self,
        lane: NativePlanningSessionLane,
        session: NativePlanningSession,
    ) -> None:
        try:
            session.close()
        finally:
            self._return_or_replace_lane(lane)

    def _reserve(
        self,
        ledger: PlannerRequestLedger,
        *,
        transitions: int,
        host_bytes: int,
    ) -> PlannerWorkReservation:
        reservation = ledger.reserve(
            transitions=transitions,
            native_calls=1,
            host_bytes=host_bytes,
            expected_seconds=max(
                self.config.call_guard_seconds,
                ledger.limits.native_call_guard_seconds,
            ),
        )
        if reservation is not None:
            return reservation
        reason = ledger.snapshot().stop_reason
        with self._lock:
            self._deadline_rejected += int(
                reason is PlannerWorkStopReason.DEADLINE_GUARD
            )
        if reason is PlannerWorkStopReason.DEADLINE_GUARD:
            raise NativePlanningSessionPoolDeadlineError(
                "planning-session call rejected by the request deadline guard"
            )
        raise NativePlanningSessionPoolError(
            f"planning-session call rejected by request work limit: {reason}"
        )

    def _admit(
        self,
        ledger: PlannerRequestLedger,
        reservation: PlannerWorkReservation,
    ) -> None:
        if self._admission.acquire(blocking=False):
            return
        ledger.complete(reservation, elapsed_seconds=0.0, success=False)
        ledger.stop(PlannerWorkStopReason.NATIVE_POOL_SATURATED)
        with self._lock:
            self._saturated += 1
        raise NativePlanningSessionPoolSaturatedError(
            "planning-session job admission is full"
        )

    def _take_lane(
        self,
        ledger: PlannerRequestLedger,
        reservation: PlannerWorkReservation,
    ) -> NativePlanningSessionLane:
        start_deadline = ledger.deadline_monotonic - max(
            self.config.call_guard_seconds,
            ledger.limits.native_call_guard_seconds,
        )
        timeout = max(0.0, start_deadline - time.monotonic())
        if timeout <= 0.0:
            self._admission.release()
            ledger.complete(reservation, elapsed_seconds=0.0, success=False)
            ledger.stop(PlannerWorkStopReason.DEADLINE_GUARD)
            with self._lock:
                self._failed += 1
                self._deadline_rejected += 1
            raise NativePlanningSessionPoolDeadlineError(
                "planning-session lane start guard expired"
            )
        try:
            return self._available.get(timeout=timeout)
        except queue.Empty as exc:
            self._admission.release()
            ledger.complete(reservation, elapsed_seconds=0.0, success=False)
            ledger.stop(PlannerWorkStopReason.NATIVE_POOL_SATURATED)
            with self._lock:
                self._failed += 1
                self._saturated += 1
            raise NativePlanningSessionPoolSaturatedError(
                "no pinned planning-session lane became available"
            ) from exc

    def _return_or_replace_lane(self, lane: NativePlanningSessionLane) -> None:
        replacement = lane
        if lane.closed:
            with self._lock:
                closed = self._closed
            if closed:
                return
            replacement = self._factory()
            with self._lock:
                self._lanes_replaced += 1
        with self._lock:
            closed = self._closed
        if closed:
            replacement.close()
        else:
            self._available.put_nowait(replacement)

    def _validate_transitions(self, transitions: int) -> None:
        if transitions <= 0:
            raise ValueError("planning-session call must contain a transition")
        if transitions > self.config.max_transitions_per_call:
            raise ValueError("planning-session call exceeds pool chunk capacity")

    def _record_call_start(self, queue_wait_seconds: float) -> None:
        with self._lock:
            self._active_calls += 1
            self._peak_active_calls = max(
                self._peak_active_calls,
                self._active_calls,
            )
            self._queue_wait_seconds += queue_wait_seconds

    def _record_call_finish(self, success: bool) -> None:
        with self._lock:
            self._active_calls -= 1
            self._completed += int(success)
            self._failed += int(not success)


__all__ = [
    "NativePlanningSessionContinueCall",
    "NativePlanningSessionLanePool",
    "NativePlanningSessionLease",
    "NativePlanningSessionOpenCall",
    "NativePlanningSessionPoolConfig",
    "NativePlanningSessionPoolDeadlineError",
    "NativePlanningSessionPoolError",
    "NativePlanningSessionPoolResult",
    "NativePlanningSessionPoolSaturatedError",
    "NativePlanningSessionPoolStats",
]
