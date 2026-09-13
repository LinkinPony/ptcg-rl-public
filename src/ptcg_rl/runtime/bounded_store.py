"""Thread-safe bounded admission storage with no live-entry eviction."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


class BoundedStoreCapacityError(RuntimeError):
    """Raised when an atomic insert cannot fit without evicting live data."""


@dataclass(frozen=True, slots=True)
class BoundedStoreStats:
    """Lifetime access/admission counters and current occupancy."""

    accesses: int
    misses: int
    inserts: int
    rejected_inserts: int
    size: int
    capacity: int


class BoundedAdmissionStore(Generic[_KeyT, _ValueT]):
    """Store explicit-lifecycle entries and reject capacity pressure atomically."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("bounded store capacity must be positive")
        self.capacity = int(capacity)
        self._lock = threading.Lock()
        self._values: dict[_KeyT, _ValueT] = {}
        self._accesses = 0
        self._misses = 0
        self._inserts = 0
        self._rejected_inserts = 0

    def get(self, key: _KeyT) -> _ValueT | None:
        """Return one entry without changing its explicit lifecycle."""
        with self._lock:
            value = self._values.get(key)
            if value is None:
                self._misses += 1
                return None
            self._accesses += 1
            return value

    def put(self, key: _KeyT, value: _ValueT) -> None:
        """Insert/update one entry without evicting another active key."""
        self.put_many({key: value})

    def put_many(self, values: Mapping[_KeyT, _ValueT]) -> None:
        """Atomically insert a batch or reject it before changing the store."""
        if not values:
            return
        with self._lock:
            new_keys = sum(key not in self._values for key in values)
            if len(self._values) + new_keys > self.capacity:
                self._rejected_inserts += new_keys
                raise BoundedStoreCapacityError(
                    "bounded store has no capacity for the complete batch"
                )
            self._values.update(values)
            self._inserts += new_keys

    def remove(self, key: _KeyT) -> bool:
        """Explicitly release one entry."""
        with self._lock:
            if key not in self._values:
                return False
            del self._values[key]
            return True

    def clear(self) -> None:
        """Drop all entries during owner shutdown or identity invalidation."""
        with self._lock:
            self._values.clear()

    def stats(self) -> BoundedStoreStats:
        """Return an immutable occupancy snapshot."""
        with self._lock:
            return BoundedStoreStats(
                accesses=self._accesses,
                misses=self._misses,
                inserts=self._inserts,
                rejected_inserts=self._rejected_inserts,
                size=len(self._values),
                capacity=self.capacity,
            )


__all__ = [
    "BoundedAdmissionStore",
    "BoundedStoreCapacityError",
    "BoundedStoreStats",
]
