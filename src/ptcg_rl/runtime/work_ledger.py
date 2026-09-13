"""Request-global deterministic work and deadline accounting."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class PlannerWorkStopReason(StrEnum):
    """Stable reasons why no additional request work may start."""

    ACTIVE = "active"
    CANDIDATE_BUDGET = "candidate_budget"
    NODE_BUDGET = "node_budget"
    TRANSITION_BUDGET = "transition_budget"
    NATIVE_CALL_BUDGET = "native_call_budget"
    GPU_ROW_BUDGET = "gpu_row_budget"
    HOST_BYTE_BUDGET = "host_byte_budget"
    DEADLINE_GUARD = "deadline_guard"
    NATIVE_POOL_SATURATED = "native_pool_saturated"
    EXPLICIT_FALLBACK = "explicit_fallback"


class PlannerWorkLimits(BaseModel):
    """Resolved deterministic caps for one planning request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_candidates: int
    max_nodes: int
    max_transitions: int
    max_native_calls: int
    max_gpu_rows: int
    max_host_bytes: int
    max_native_transitions_per_call: int
    native_call_guard_seconds: float

    @field_validator(
        "max_candidates",
        "max_nodes",
        "max_transitions",
        "max_native_calls",
        "max_gpu_rows",
        "max_host_bytes",
        "max_native_transitions_per_call",
    )
    @classmethod
    def positive_limit(cls, value: int) -> int:
        """Require every resolved capacity to be positive."""
        if value <= 0:
            raise ValueError("planner work limits must be positive")
        return value

    @field_validator("native_call_guard_seconds")
    @classmethod
    def nonnegative_finite_guard(cls, value: float) -> float:
        """Require a finite non-negative profile-derived call guard."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "native_call_guard_seconds must be finite and non-negative"
            )
        return value

    @model_validator(mode="after")
    def call_fits_request(self) -> Self:
        """A single native chunk cannot exceed the request transition cap."""
        if self.max_native_transitions_per_call > self.max_transitions:
            raise ValueError(
                "max_native_transitions_per_call cannot exceed max_transitions"
            )
        return self


@dataclass(frozen=True)
class PlannerWorkReservation:
    """One immutable work allocation consumed when a stage is launched."""

    reservation_id: int
    candidates: int
    nodes: int
    transitions: int
    native_calls: int
    gpu_rows: int
    host_bytes: int


@dataclass(frozen=True)
class PlannerWorkSnapshot:
    """Thread-safe request work/stop telemetry snapshot."""

    candidates: int
    nodes: int
    transitions: int
    native_calls: int
    gpu_rows: int
    host_bytes: int
    completed_reservations: int
    failed_reservations: int
    stage_seconds: float
    stop_reason: PlannerWorkStopReason
    remaining_seconds: float


class PlannerRequestLedger:
    """Atomically share one work/deadline budget across a planner request.

    Reservations are never returned to the budget. A failed or timed-out call
    still consumed service capacity and cannot be retried under a fresh node
    budget. Wall-clock time is only a return guard; deterministic counters are
    the normal stopping mechanism.
    """

    def __init__(
        self,
        limits: PlannerWorkLimits,
        *,
        deadline_monotonic: float,
        started_monotonic: float | None = None,
    ) -> None:
        if not math.isfinite(deadline_monotonic) or deadline_monotonic <= 0.0:
            raise ValueError("planner deadline must be finite and positive")
        self.limits = limits
        self.deadline_monotonic = float(deadline_monotonic)
        started = (
            time.monotonic() if started_monotonic is None else float(started_monotonic)
        )
        if not math.isfinite(started) or started <= 0.0:
            raise ValueError("planner start time must be finite and positive")
        if started > self.deadline_monotonic:
            raise ValueError("planner start time cannot follow its deadline")
        self.started_monotonic = started
        self._lock = threading.Lock()
        self._next_reservation_id = 0
        self._used = [0, 0, 0, 0, 0, 0]
        self._completed: set[int] = set()
        self._failed_reservations = 0
        self._stage_seconds = 0.0
        self._stop_reason = PlannerWorkStopReason.ACTIVE

    def reserve(
        self,
        *,
        candidates: int = 0,
        nodes: int = 0,
        transitions: int = 0,
        native_calls: int = 0,
        gpu_rows: int = 0,
        host_bytes: int = 0,
        expected_seconds: float = 0.0,
        now_monotonic: float | None = None,
    ) -> PlannerWorkReservation | None:
        """Reserve one bounded chunk, or return ``None`` before work starts."""
        requested = (
            candidates,
            nodes,
            transitions,
            native_calls,
            gpu_rows,
            host_bytes,
        )
        if any(value < 0 for value in requested):
            raise ValueError("planner work reservations must be non-negative")
        if not any(requested):
            raise ValueError("planner work reservation must consume some capacity")
        if transitions > self.limits.max_native_transitions_per_call and native_calls:
            raise ValueError("native chunk exceeds max_native_transitions_per_call")
        if not math.isfinite(expected_seconds) or expected_seconds < 0.0:
            raise ValueError("expected_seconds must be finite and non-negative")
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if not math.isfinite(now):
            raise ValueError("now_monotonic must be finite")
        with self._lock:
            if self._stop_reason is not PlannerWorkStopReason.ACTIVE:
                return None
            if now + expected_seconds >= self.deadline_monotonic:
                self._stop_reason = PlannerWorkStopReason.DEADLINE_GUARD
                return None
            limits = (
                self.limits.max_candidates,
                self.limits.max_nodes,
                self.limits.max_transitions,
                self.limits.max_native_calls,
                self.limits.max_gpu_rows,
                self.limits.max_host_bytes,
            )
            reasons = (
                PlannerWorkStopReason.CANDIDATE_BUDGET,
                PlannerWorkStopReason.NODE_BUDGET,
                PlannerWorkStopReason.TRANSITION_BUDGET,
                PlannerWorkStopReason.NATIVE_CALL_BUDGET,
                PlannerWorkStopReason.GPU_ROW_BUDGET,
                PlannerWorkStopReason.HOST_BYTE_BUDGET,
            )
            for used, increment, limit, reason in zip(
                self._used, requested, limits, reasons, strict=True
            ):
                if used + increment > limit:
                    self._stop_reason = reason
                    return None
            reservation = PlannerWorkReservation(
                reservation_id=self._next_reservation_id,
                candidates=candidates,
                nodes=nodes,
                transitions=transitions,
                native_calls=native_calls,
                gpu_rows=gpu_rows,
                host_bytes=host_bytes,
            )
            self._next_reservation_id += 1
            for index, increment in enumerate(requested):
                self._used[index] += increment
            return reservation

    def complete(
        self,
        reservation: PlannerWorkReservation,
        *,
        elapsed_seconds: float,
        success: bool,
    ) -> None:
        """Record completion exactly once without returning its allocation."""
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        with self._lock:
            if reservation.reservation_id >= self._next_reservation_id:
                raise ValueError("reservation does not belong to this ledger")
            if reservation.reservation_id in self._completed:
                raise ValueError("reservation was already completed")
            self._completed.add(reservation.reservation_id)
            self._failed_reservations += int(not success)
            self._stage_seconds += elapsed_seconds

    def stop(self, reason: PlannerWorkStopReason) -> None:
        """Record an explicit fallback without overwriting an earlier reason."""
        if reason is PlannerWorkStopReason.ACTIVE:
            raise ValueError("cannot explicitly stop with the active reason")
        with self._lock:
            if self._stop_reason is PlannerWorkStopReason.ACTIVE:
                self._stop_reason = reason

    def snapshot(self, *, now_monotonic: float | None = None) -> PlannerWorkSnapshot:
        """Return immutable counters and current deadline reserve."""
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        with self._lock:
            return PlannerWorkSnapshot(
                candidates=self._used[0],
                nodes=self._used[1],
                transitions=self._used[2],
                native_calls=self._used[3],
                gpu_rows=self._used[4],
                host_bytes=self._used[5],
                completed_reservations=len(self._completed),
                failed_reservations=self._failed_reservations,
                stage_seconds=self._stage_seconds,
                stop_reason=self._stop_reason,
                remaining_seconds=max(0.0, self.deadline_monotonic - now),
            )


__all__ = [
    "PlannerRequestLedger",
    "PlannerWorkLimits",
    "PlannerWorkReservation",
    "PlannerWorkSnapshot",
    "PlannerWorkStopReason",
]
