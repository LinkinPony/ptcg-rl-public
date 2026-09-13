"""Fixed byte-admission ledgers for concurrent planner requests."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ptcg_rl.rl.planner_runtime_identity import PlannerBufferLimits


@dataclass(frozen=True, slots=True)
class PlannerBufferLedgerStats:
    """Current and peak occupancy of each declared staging domain."""

    active_requests: int
    peak_active_requests: int
    rejected_requests: int
    active_host_bytes: int
    peak_host_bytes: int
    host_capacity_bytes: int
    active_ipc_bytes: int
    peak_ipc_bytes: int
    ipc_capacity_bytes: int
    active_gpu_bytes: int
    peak_gpu_bytes: int
    gpu_capacity_bytes: int


@dataclass(slots=True)
class PlannerBufferReservation:
    """One worst-case request reservation released exactly once."""

    ledger: PlannerBufferLedger
    host_bytes: int
    ipc_bytes: int
    gpu_bytes: int
    released: bool = False

    def release(self) -> None:
        """Return this reservation to its fixed-capacity ledger."""
        self.ledger.release(self)


class PlannerBufferLedger:
    """Admit work only when all declared byte domains have capacity."""

    def __init__(
        self,
        limits: PlannerBufferLimits,
        *,
        cells_per_request: int,
        unique_leaves_per_request: int,
    ) -> None:
        if cells_per_request <= 0:
            raise ValueError("planner cells_per_request must be positive")
        if unique_leaves_per_request <= 0:
            raise ValueError("planner unique_leaves_per_request must be positive")
        self._limits = limits
        self._host_per_request = cells_per_request * limits.host_bytes_per_cell
        self._ipc_per_request = cells_per_request * limits.ipc_bytes_per_cell
        self._gpu_per_request = (
            unique_leaves_per_request * limits.gpu_bytes_per_unique_leaf
        )
        self._lock = threading.Lock()
        self._active_requests = 0
        self._peak_active_requests = 0
        self._rejected_requests = 0
        self._active_host_bytes = 0
        self._peak_host_bytes = 0
        self._active_ipc_bytes = 0
        self._peak_ipc_bytes = 0
        self._active_gpu_bytes = 0
        self._peak_gpu_bytes = 0

    def try_acquire(self) -> PlannerBufferReservation | None:
        """Reserve one fixed worst-case request footprint without blocking."""
        with self._lock:
            fits = (
                self._active_requests < self._limits.max_inflight_requests
                and self._active_host_bytes + self._host_per_request
                <= self._limits.host_pool_bytes
                and self._active_ipc_bytes + self._ipc_per_request
                <= self._limits.ipc_pool_bytes
                and self._active_gpu_bytes + self._gpu_per_request
                <= self._limits.gpu_staging_bytes
            )
            if not fits:
                self._rejected_requests += 1
                return None
            self._active_requests += 1
            self._active_host_bytes += self._host_per_request
            self._active_ipc_bytes += self._ipc_per_request
            self._active_gpu_bytes += self._gpu_per_request
            self._peak_active_requests = max(
                self._peak_active_requests,
                self._active_requests,
            )
            self._peak_host_bytes = max(
                self._peak_host_bytes,
                self._active_host_bytes,
            )
            self._peak_ipc_bytes = max(
                self._peak_ipc_bytes,
                self._active_ipc_bytes,
            )
            self._peak_gpu_bytes = max(
                self._peak_gpu_bytes,
                self._active_gpu_bytes,
            )
        return PlannerBufferReservation(
            ledger=self,
            host_bytes=self._host_per_request,
            ipc_bytes=self._ipc_per_request,
            gpu_bytes=self._gpu_per_request,
        )

    def release(self, reservation: PlannerBufferReservation) -> None:
        """Release one reservation, rejecting foreign or duplicate tokens."""
        if reservation.ledger is not self:
            raise ValueError("planner buffer reservation belongs to another ledger")
        with self._lock:
            if reservation.released:
                raise RuntimeError("planner buffer reservation was already released")
            reservation.released = True
            self._active_requests -= 1
            self._active_host_bytes -= reservation.host_bytes
            self._active_ipc_bytes -= reservation.ipc_bytes
            self._active_gpu_bytes -= reservation.gpu_bytes
            if min(
                self._active_requests,
                self._active_host_bytes,
                self._active_ipc_bytes,
                self._active_gpu_bytes,
            ) < 0:
                raise AssertionError("planner buffer ledger occupancy underflowed")

    def stats(self) -> PlannerBufferLedgerStats:
        """Return an immutable thread-safe occupancy snapshot."""
        with self._lock:
            return PlannerBufferLedgerStats(
                active_requests=self._active_requests,
                peak_active_requests=self._peak_active_requests,
                rejected_requests=self._rejected_requests,
                active_host_bytes=self._active_host_bytes,
                peak_host_bytes=self._peak_host_bytes,
                host_capacity_bytes=self._limits.host_pool_bytes,
                active_ipc_bytes=self._active_ipc_bytes,
                peak_ipc_bytes=self._peak_ipc_bytes,
                ipc_capacity_bytes=self._limits.ipc_pool_bytes,
                active_gpu_bytes=self._active_gpu_bytes,
                peak_gpu_bytes=self._peak_gpu_bytes,
                gpu_capacity_bytes=self._limits.gpu_staging_bytes,
            )


__all__ = [
    "PlannerBufferLedger",
    "PlannerBufferLedgerStats",
    "PlannerBufferReservation",
]
