"""Bounded immutable model-snapshot leases for action-critical requests."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, Self, TypeVar

_SnapshotT = TypeVar("_SnapshotT")


class ModelLeaseCapacityError(RuntimeError):
    """Raised when a request or publication would exceed resident capacity."""


@dataclass
class _SnapshotEntry(Generic[_SnapshotT]):
    snapshot: _SnapshotT
    leases: int = 0
    retired: bool = False


@dataclass(frozen=True)
class ModelSnapshotPoolStats:
    """Current bounded residency and in-flight lease counters."""

    current_version: int | None
    resident_versions: tuple[int, ...]
    leases_by_version: Mapping[int, int]
    in_flight_leases: int
    max_resident_snapshots: int
    max_in_flight_leases: int
    rejected_acquires: int
    rejected_publishes: int


class ModelLease(Generic[_SnapshotT]):
    """One idempotently releasable immutable snapshot lease."""

    def __init__(
        self,
        *,
        pool: ModelSnapshotPool[_SnapshotT],
        version: int,
        snapshot: _SnapshotT,
    ) -> None:
        self.version = int(version)
        self.snapshot = snapshot
        self._pool = pool
        self._released = False
        self._lock = threading.Lock()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.release()

    def release(self) -> None:
        """Release this lease exactly once."""
        with self._lock:
            if self._released:
                return
            self._released = True
        self._pool._release(self.version)  # pylint: disable=protected-access


class ModelSnapshotPool(Generic[_SnapshotT]):
    """Atomically publish snapshots without unbounded checkpoint residency."""

    def __init__(
        self,
        *,
        max_resident_snapshots: int,
        max_in_flight_leases: int,
    ) -> None:
        if max_resident_snapshots <= 0 or max_in_flight_leases <= 0:
            raise ValueError("model lease capacities must be positive")
        self.max_resident_snapshots = int(max_resident_snapshots)
        self.max_in_flight_leases = int(max_in_flight_leases)
        self._lock = threading.Lock()
        self._entries: dict[int, _SnapshotEntry[_SnapshotT]] = {}
        self._current_version: int | None = None
        self._in_flight = 0
        self._rejected_acquires = 0
        self._rejected_publishes = 0

    def publish(self, version: int, snapshot: _SnapshotT) -> None:
        """Atomically make a strictly newer snapshot current.

        If active leases occupy every resident slot, publication is rejected
        before retaining another large model. The current snapshot remains
        unchanged and callers can retry publication after bounded work drains.
        """
        if version < 0:
            raise ValueError("model version must be non-negative")
        with self._lock:
            if self._current_version is not None and version <= self._current_version:
                raise ValueError("published model versions must strictly increase")
            leased_residents = sum(
                int(entry.leases > 0) for entry in self._entries.values()
            )
            if leased_residents + 1 > self.max_resident_snapshots:
                self._rejected_publishes += 1
                raise ModelLeaseCapacityError(
                    "resident model snapshot capacity is occupied by leases"
                )
            for entry in self._entries.values():
                entry.retired = True
            self._remove_unleased_retired_locked()
            self._entries[version] = _SnapshotEntry(snapshot=snapshot)
            self._current_version = version
            self._remove_unleased_retired_locked()

    def can_publish(self, version: int) -> bool:
        """Return whether a newer snapshot can be admitted without retention."""
        if version < 0:
            raise ValueError("model version must be non-negative")
        with self._lock:
            if self._current_version is not None and version <= self._current_version:
                return False
            leased_residents = sum(
                int(entry.leases > 0) for entry in self._entries.values()
            )
            return leased_residents + 1 <= self.max_resident_snapshots

    def resident_snapshot(self, version: int | None = None) -> _SnapshotT:
        """Return a resident immutable snapshot without creating another lease.

        This is intended for routing calls already protected by a root request
        lease. Callers must not use it to start new action-critical work.
        """
        with self._lock:
            selected = self._current_version if version is None else int(version)
            entry = self._entries.get(selected) if selected is not None else None
            if entry is None:
                raise ModelLeaseCapacityError("requested model snapshot is unavailable")
            return entry.snapshot

    def has_resident_version(self, version: int) -> bool:
        """Return whether an immutable snapshot version remains resident."""
        with self._lock:
            return int(version) in self._entries

    def acquire(self, version: int | None = None) -> ModelLease[_SnapshotT]:
        """Acquire the current or explicitly named resident version."""
        with self._lock:
            selected = self._current_version if version is None else version
            if selected is None or selected not in self._entries:
                self._rejected_acquires += 1
                raise ModelLeaseCapacityError("requested model snapshot is unavailable")
            if self._in_flight >= self.max_in_flight_leases:
                self._rejected_acquires += 1
                raise ModelLeaseCapacityError("model lease capacity is full")
            entry = self._entries[selected]
            entry.leases += 1
            self._in_flight += 1
            snapshot = entry.snapshot
        return ModelLease(pool=self, version=selected, snapshot=snapshot)

    def stats(self) -> ModelSnapshotPoolStats:
        """Return an immutable pool snapshot."""
        with self._lock:
            return ModelSnapshotPoolStats(
                current_version=self._current_version,
                resident_versions=tuple(sorted(self._entries)),
                leases_by_version={
                    version: entry.leases
                    for version, entry in sorted(self._entries.items())
                },
                in_flight_leases=self._in_flight,
                max_resident_snapshots=self.max_resident_snapshots,
                max_in_flight_leases=self.max_in_flight_leases,
                rejected_acquires=self._rejected_acquires,
                rejected_publishes=self._rejected_publishes,
            )

    def _release(self, version: int) -> None:
        with self._lock:
            entry = self._entries.get(version)
            if entry is None or entry.leases <= 0:
                raise RuntimeError("model lease release has no matching acquire")
            entry.leases -= 1
            self._in_flight -= 1
            self._remove_unleased_retired_locked()

    def _remove_unleased_retired_locked(self) -> None:
        removable = tuple(
            version
            for version, entry in self._entries.items()
            if entry.retired and entry.leases == 0
        )
        for version in removable:
            del self._entries[version]


__all__ = [
    "ModelLease",
    "ModelLeaseCapacityError",
    "ModelSnapshotPool",
    "ModelSnapshotPoolStats",
]
