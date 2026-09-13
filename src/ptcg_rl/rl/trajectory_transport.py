"""Bounded shared-memory transport for compact rollout trajectories."""

from __future__ import annotations

import ctypes
import pickle
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from ptcg_rl.rl.experience import GameTrajectory

_SLOT_FREE = 0
_SLOT_WRITING = 1
_SLOT_READY = 2
_SLOT_READING = 3
_POLL_SECONDS = 0.0005
_ATOMIC_ACQUIRE = 2
_ATOMIC_RELEASE = 3

_ATOMIC_INIT_LOCK = threading.Lock()
_LIBATOMIC: Any | None = None
_ATOMIC_LOAD: Any | None = None
_ATOMIC_STORE: Any | None = None
_ATOMIC_IS_LOCK_FREE: Any | None = None


def _initialize_atomic_runtime() -> None:
    """Load libatomic only when the crash-recoverable ring is selected."""
    global _ATOMIC_IS_LOCK_FREE, _ATOMIC_LOAD, _ATOMIC_STORE, _LIBATOMIC
    if _ATOMIC_LOAD is not None:
        return
    with _ATOMIC_INIT_LOCK:
        if _ATOMIC_LOAD is not None:
            return
        try:
            library = ctypes.CDLL("libatomic.so.1")
        except OSError as exc:
            raise RuntimeError(
                "shared trajectory rings require the libatomic runtime"
            ) from exc
        atomic_load = library.__atomic_load_4
        atomic_load.argtypes = (ctypes.c_void_p, ctypes.c_int)
        atomic_load.restype = ctypes.c_uint32
        atomic_store = library.__atomic_store_4
        atomic_store.argtypes = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int)
        atomic_store.restype = None
        atomic_is_lock_free = library.__atomic_is_lock_free
        atomic_is_lock_free.argtypes = (ctypes.c_size_t, ctypes.c_void_p)
        atomic_is_lock_free.restype = ctypes.c_bool
        _LIBATOMIC = library
        _ATOMIC_LOAD = atomic_load
        _ATOMIC_STORE = atomic_store
        _ATOMIC_IS_LOCK_FREE = atomic_is_lock_free


@dataclass(frozen=True, slots=True)
class SharedTrajectoryRingStats:
    """Local-process counters for one ring endpoint."""

    puts: int
    gets: int
    shared_payloads: int
    inline_payloads: int
    payload_bytes: int
    max_payload_bytes: int
    recovered_writing_slots: int


class SharedTrajectoryRing:
    """Move compact trajectories through actor-owned anonymous shared slots.

    Each producer owns a disjoint lane of slots and the learner is the only
    consumer.  A producer writes ``FREE -> WRITING -> READY``; the learner uses
    ``READY -> READING -> FREE``.  No queue feeder, pipe frame, or process-shared
    lock participates in the handoff.  A restarted actor can therefore reclaim
    only the ``WRITING`` slots in its lane while preserving every fully
    published ``READY`` trajectory.

    Slot payloads are bounded deliberately.  Sending an oversized pickle inline
    would reintroduce a pipe and make hard actor failure capable of corrupting
    the transport, so callers must size slots for the largest accepted compact
    trajectory.
    """

    def __init__(
        self,
        context: Any,
        *,
        slots: int,
        slot_bytes: int,
        producer_count: int = 1,
        allow_inline_oversize: bool = False,
    ) -> None:
        if slots <= 0:
            raise ValueError("shared trajectory ring slots must be positive")
        if slot_bytes <= 0:
            raise ValueError("shared trajectory ring slot_bytes must be positive")
        if producer_count <= 0:
            raise ValueError("shared trajectory ring producer_count must be positive")
        if slots < producer_count:
            raise ValueError(
                "shared trajectory ring needs at least one slot per producer"
            )
        if allow_inline_oversize:
            raise ValueError(
                "crash-recoverable shared trajectory rings cannot inline "
                "oversized payloads; increase slot_bytes"
            )
        self.slots = int(slots)
        self.slot_bytes = int(slot_bytes)
        self.producer_count = int(producer_count)
        self.allow_inline_oversize = False
        self._storage = context.RawArray("B", self.slots * self.slot_bytes)
        self._slot_states = context.RawArray("I", self.slots)
        self._slot_sizes = context.RawArray("Q", self.slots)
        _initialize_atomic_runtime()
        assert _ATOMIC_IS_LOCK_FREE is not None
        if not _ATOMIC_IS_LOCK_FREE(
            ctypes.sizeof(ctypes.c_uint32),
            ctypes.addressof(self._slot_states),
        ):
            raise RuntimeError(
                "shared trajectory slot states require lock-free 32-bit atomics"
            )
        self._producer_index: int | None = None
        self._producer_cursor = 0
        self._consumer_cursor = 0
        self._local_lock = threading.Lock()
        self._producer_lock = threading.Lock()
        self._puts = 0
        self._gets = 0
        self._shared_payloads = 0
        self._payload_bytes = 0
        self._max_payload_bytes = 0
        self._recovered_writing_slots = 0

    def bind_producer(self, producer_index: int) -> int:
        """Bind this process endpoint and reclaim its interrupted writes."""
        if not 0 <= producer_index < self.producer_count:
            raise ValueError("shared trajectory producer index is out of range")
        if self._producer_index == producer_index:
            return 0
        if self._producer_index is not None:
            raise RuntimeError("shared trajectory endpoint is already bound")
        self._producer_index = producer_index
        recovered = 0
        for slot in self._owned_slots(producer_index):
            if self._load_slot_state(slot) != _SLOT_WRITING:
                continue
            self._slot_sizes[slot] = 0
            self._store_slot_state(slot, _SLOT_FREE)
            recovered += 1
        with self._local_lock:
            self._recovered_writing_slots += recovered
        return recovered

    def put(
        self,
        item: GameTrajectory,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Serialize and publish one trajectory in the bound producer lane."""
        if not isinstance(item, GameTrajectory):
            raise TypeError("shared trajectory ring accepts GameTrajectory values")
        payload = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
        size_bytes = len(payload)
        if size_bytes > self.slot_bytes:
            raise ValueError(
                "trajectory payload exceeds crash-safe shared ring slot: "
                f"{size_bytes} > {self.slot_bytes}; increase slot_bytes"
            )
        with self._producer_lock:
            slot = self._acquire_free_slot(block=block, timeout=timeout)
            try:
                start = slot * self.slot_bytes
                view = memoryview(self._storage).cast("B")
                try:
                    view[start : start + size_bytes] = payload
                finally:
                    view.release()
                self._slot_sizes[slot] = size_bytes
                self._store_slot_state(slot, _SLOT_READY)
            except BaseException:
                self._slot_sizes[slot] = 0
                self._store_slot_state(slot, _SLOT_FREE)
                raise
        self._record_put(size_bytes)

    def put_nowait(self, item: GameTrajectory) -> None:
        """Publish one trajectory without waiting for a free owned slot."""
        self.put(item, block=False)

    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> GameTrajectory:
        """Read, deserialize, and release one ready trajectory slot."""
        slot = self._acquire_ready_slot(block=block, timeout=timeout)
        try:
            size_bytes = int(self._slot_sizes[slot])
            if not 0 < size_bytes <= self.slot_bytes:
                raise ValueError("shared trajectory slot has an invalid payload size")
            start = slot * self.slot_bytes
            view = memoryview(self._storage).cast("B")
            try:
                payload = bytes(view[start : start + size_bytes])
            finally:
                view.release()
            trajectory = pickle.loads(payload)
            if not isinstance(trajectory, GameTrajectory):
                raise TypeError("shared trajectory payload is not a GameTrajectory")
            self._record_get()
            return trajectory
        finally:
            self._slot_sizes[slot] = 0
            self._store_slot_state(slot, _SLOT_FREE)

    def get_nowait(self) -> GameTrajectory:
        """Return one ready trajectory without blocking."""
        return self.get(block=False)

    def qsize(self) -> int:
        """Return the current number of fully published trajectories."""
        return sum(
            self._load_slot_state(slot) == _SLOT_READY for slot in range(self.slots)
        )

    def empty(self) -> bool:
        """Return whether no trajectory is fully published."""
        return self.qsize() == 0

    def full(self) -> bool:
        """Return whether no producer lane currently has a free slot."""
        return all(
            self._load_slot_state(slot) != _SLOT_FREE for slot in range(self.slots)
        )

    def cancel_join_thread(self) -> None:
        """Retain queue-compatible cleanup; the ring has no feeder thread."""

    def close(self) -> None:
        """Retain queue-compatible cleanup; anonymous arrays need no unlink."""

    def stats(self) -> SharedTrajectoryRingStats:
        """Return counters collected by this process-local endpoint instance."""
        with self._local_lock:
            return SharedTrajectoryRingStats(
                puts=self._puts,
                gets=self._gets,
                shared_payloads=self._shared_payloads,
                inline_payloads=0,
                payload_bytes=self._payload_bytes,
                max_payload_bytes=self._max_payload_bytes,
                recovered_writing_slots=self._recovered_writing_slots,
            )

    def __getstate__(self) -> dict[str, Any]:
        """Drop process-local endpoint state while reducing for spawn."""
        state = dict(self.__dict__)
        state.pop("_local_lock", None)
        state.pop("_producer_lock", None)
        state["_producer_index"] = None
        state["_producer_cursor"] = 0
        state["_consumer_cursor"] = 0
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore process-local endpoint state in a spawned process."""
        self.__dict__.update(state)
        self._local_lock = threading.Lock()
        self._producer_lock = threading.Lock()

    def _owned_slots(self, producer_index: int) -> range:
        return range(producer_index, self.slots, self.producer_count)

    def _bound_producer_index(self) -> int:
        if self._producer_index is None:
            if self.producer_count != 1:
                raise RuntimeError(
                    "multi-producer shared trajectory endpoint must be bound"
                )
            self.bind_producer(0)
        assert self._producer_index is not None
        return self._producer_index

    def _acquire_free_slot(
        self,
        *,
        block: bool,
        timeout: float | None,
    ) -> int:
        self._validate_wait(block=block, timeout=timeout)
        producer_index = self._bound_producer_index()
        owned = tuple(self._owned_slots(producer_index))
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            for offset in range(len(owned)):
                cursor = (self._producer_cursor + offset) % len(owned)
                slot = owned[cursor]
                if self._load_slot_state(slot) != _SLOT_FREE:
                    continue
                self._store_slot_state(slot, _SLOT_WRITING)
                self._producer_cursor = (cursor + 1) % len(owned)
                return slot
            if not block or self._deadline_expired(deadline):
                raise queue.Full
            self._wait(deadline)

    def _acquire_ready_slot(
        self,
        *,
        block: bool,
        timeout: float | None,
    ) -> int:
        self._validate_wait(block=block, timeout=timeout)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            for offset in range(self.slots):
                slot = (self._consumer_cursor + offset) % self.slots
                if self._load_slot_state(slot) != _SLOT_READY:
                    continue
                self._store_slot_state(slot, _SLOT_READING)
                self._consumer_cursor = (slot + 1) % self.slots
                return slot
            if not block or self._deadline_expired(deadline):
                raise queue.Empty
            self._wait(deadline)

    @staticmethod
    def _validate_wait(*, block: bool, timeout: float | None) -> None:
        if not block and timeout is not None:
            raise ValueError("can't specify a timeout for a non-blocking operation")
        if timeout is not None and timeout < 0.0:
            raise ValueError("timeout must be non-negative")

    @staticmethod
    def _deadline_expired(deadline: float | None) -> bool:
        return deadline is not None and time.monotonic() >= deadline

    @staticmethod
    def _wait(deadline: float | None) -> None:
        if deadline is None:
            time.sleep(_POLL_SECONDS)
            return
        time.sleep(max(0.0, min(_POLL_SECONDS, deadline - time.monotonic())))

    def _record_put(self, size_bytes: int) -> None:
        with self._local_lock:
            self._puts += 1
            self._shared_payloads += 1
            self._payload_bytes += size_bytes
            self._max_payload_bytes = max(self._max_payload_bytes, size_bytes)

    def _record_get(self) -> None:
        with self._local_lock:
            self._gets += 1

    def _load_slot_state(self, slot: int) -> int:
        _initialize_atomic_runtime()
        assert _ATOMIC_LOAD is not None
        address = ctypes.addressof(self._slot_states) + (
            slot * ctypes.sizeof(ctypes.c_uint32)
        )
        return int(_ATOMIC_LOAD(address, _ATOMIC_ACQUIRE))

    def _store_slot_state(self, slot: int, state: int) -> None:
        _initialize_atomic_runtime()
        assert _ATOMIC_STORE is not None
        address = ctypes.addressof(self._slot_states) + (
            slot * ctypes.sizeof(ctypes.c_uint32)
        )
        _ATOMIC_STORE(address, state, _ATOMIC_RELEASE)


__all__ = ["SharedTrajectoryRing", "SharedTrajectoryRingStats"]
