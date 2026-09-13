"""Single-owner CUDA execution for exact multi-policy native rollout inference."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from typing import Any, Literal, cast

import torch

from ptcg_rl.rl.native_route_scheduler import NativeRouteKey

_StreamKind = Literal["current", "frozen"]
_StreamLane = tuple[_StreamKind, int]
_DeferredCompletion = tuple[Callable[[], None], Callable[[], None] | None]
_CURRENT_STREAM_PRIORITY = -1
# Frozen and past-self inference is collection-critical too: leaving it at the
# default priority lets an overlapping learner starve most rollout routes.
_FROZEN_STREAM_PRIORITY = _CURRENT_STREAM_PRIORITY
_FROZEN_STREAM_LANES = 8


class _NativePolicyCudaStreamLease:
    """Exclusive window lease over one persistent CUDA stream lane pool."""

    def __init__(
        self,
        pool: NativePolicyCudaStreamPool,
        token: object,
    ) -> None:
        """Bind one opaque pool lease token."""
        self._pool = pool
        self._token = token
        self._closed = False

    def compute_stream(
        self,
        *,
        device: torch.device,
        lane: _StreamLane,
    ) -> Any:
        """Return the persistent compute stream for one device lane."""
        if self._closed:
            raise RuntimeError("native policy CUDA stream lease is closed")
        return self._pool._compute_stream(  # noqa: SLF001 - paired lease owner.
            self._token,
            device=device,
            lane=lane,
        )

    def copy_stream(self, *, device: torch.device) -> Any:
        """Return the persistent copy stream for one device."""
        if self._closed:
            raise RuntimeError("native policy CUDA stream lease is closed")
        return self._pool._copy_stream(  # noqa: SLF001 - paired lease owner.
            self._token,
            device=device,
        )

    def close(self) -> None:
        """Release this window without destroying the shared CUDA streams."""
        if self._closed:
            return
        self._pool._release(self._token)  # noqa: SLF001 - paired lease owner.
        self._closed = True


class NativePolicyCudaStreamPool:
    """Persist fixed native-policy CUDA lanes across serial collection windows."""

    def __init__(self) -> None:
        """Create an unleased pool without initializing CUDA eagerly."""
        self._compute_streams: dict[tuple[torch.device, _StreamLane], Any] = {}
        self._copy_streams: dict[torch.device, Any] = {}
        self._active_lease: object | None = None
        self._lock = threading.Lock()
        self._closed = False

    def lease(self) -> _NativePolicyCudaStreamLease:
        """Exclusively lease all lanes to one serial collection window."""
        with self._lock:
            if self._closed:
                raise RuntimeError("native policy CUDA stream pool is closed")
            if self._active_lease is not None:
                raise RuntimeError(
                    "native policy CUDA stream pool already has an active window"
                )
            token = object()
            self._active_lease = token
        return _NativePolicyCudaStreamLease(self, token)

    def close(self) -> None:
        """Destroy persistent streams after the final window lease drains."""
        with self._lock:
            if self._closed:
                return
            if self._active_lease is not None:
                raise RuntimeError(
                    "cannot close native policy CUDA stream pool with an active window"
                )
            self._closed = True
            self._compute_streams.clear()
            self._copy_streams.clear()

    def _compute_stream(
        self,
        token: object,
        *,
        device: torch.device,
        lane: _StreamLane,
    ) -> Any:
        """Resolve one fixed compute lane for the active lease."""
        with self._lock:
            self._require_active_lease(token)
            key = (device, lane)
            stream = self._compute_streams.get(key)
            if stream is None:
                stream_factory = cast(Any, torch.cuda.Stream)
                stream = stream_factory(
                    device=device,
                    priority=_stream_priority(lane[0]),
                )
                self._compute_streams[key] = stream
            return stream

    def _copy_stream(
        self,
        token: object,
        *,
        device: torch.device,
    ) -> Any:
        """Resolve one fixed D2H stream for the active lease."""
        with self._lock:
            self._require_active_lease(token)
            stream = self._copy_streams.get(device)
            if stream is None:
                stream_factory = cast(Any, torch.cuda.Stream)
                stream = stream_factory(device=device)
                self._copy_streams[device] = stream
            return stream

    def _release(self, token: object) -> None:
        """Release exactly the active serial lease."""
        with self._lock:
            self._require_active_lease(token)
            self._active_lease = None

    def _require_active_lease(self, token: object) -> None:
        """Reject stale or foreign window access while holding ``_lock``."""
        if self._closed:
            raise RuntimeError("native policy CUDA stream pool is closed")
        if self._active_lease is not token:
            raise RuntimeError("native policy CUDA stream lease is not active")


class NativePolicyInferenceWaveTicket:
    """Own one sealed CUDA wave until its host completions are resolved."""

    def __init__(self, bank: NativePolicyInferenceBank) -> None:
        """Create an open ticket owned by ``bank.deferred_wave()``."""
        self._bank = bank
        self._events: tuple[Any, ...] = ()
        self._deferred: tuple[_DeferredCompletion, ...] = ()
        self._state: Literal[
            "open",
            "sealed",
            "resolving",
            "finished",
            "aborted",
        ] = "open"
        self._lock = threading.Lock()

    def finish(self) -> None:
        """Wait for this wave and publish each deferred host result once."""
        claimed = self._claim()
        if claimed is None:
            return
        events, deferred = claimed
        state: Literal["finished", "aborted"] = "finished"
        try:
            _synchronize_events(events)
            for complete, _abort in deferred:
                complete()
        except BaseException:
            state = "aborted"
            _abort_deferred(deferred)
            raise
        finally:
            self._resolve(state)

    def abort(self) -> None:
        """Safely drain this wave and release its deferred reservations."""
        claimed = self._claim()
        if claimed is None:
            return
        events, deferred = claimed
        try:
            # Preserve best-effort cleanup after a poisoned CUDA context.
            with suppress(BaseException):
                _synchronize_events(events)
            _abort_deferred(deferred)
        finally:
            self._resolve("aborted")

    def _seal(
        self,
        *,
        events: tuple[Any, ...],
        deferred: tuple[_DeferredCompletion, ...],
    ) -> None:
        """Bind immutable completion boundaries after context submission."""
        with self._lock:
            if self._state != "open":
                raise RuntimeError("native policy wave ticket is not open")
            self._events = events
            self._deferred = deferred
            self._state = "sealed"

    def _abort_open(self) -> None:
        """Mark a context-aborted ticket resolved without publishing it."""
        with self._lock:
            if self._state != "open":
                raise RuntimeError("native policy wave ticket is not open")
            self._state = "aborted"

    def _claim(
        self,
    ) -> tuple[tuple[Any, ...], tuple[_DeferredCompletion, ...]] | None:
        """Claim the sole finish/abort operation for this ticket."""
        with self._lock:
            if self._state in ("finished", "aborted"):
                return None
            if self._state == "open":
                raise RuntimeError("native policy wave ticket is not sealed")
            if self._state == "resolving":
                raise RuntimeError("native policy wave ticket is resolving")
            self._state = "resolving"
            return self._events, self._deferred

    def _resolve(self, state: Literal["finished", "aborted"]) -> None:
        """Release retained transfers and unregister from the owning bank."""
        with self._lock:
            if self._state != "resolving":
                raise RuntimeError("native policy wave ticket is not resolving")
            self._events = ()
            self._deferred = ()
            self._state = state
        self._bank._release_ticket(self)


class NativePolicyInferenceBank:
    """Own route-aware compute lanes and one host-transfer boundary per device."""

    def __init__(
        self,
        *,
        stream_pool: NativePolicyCudaStreamPool | None = None,
    ) -> None:
        """Create an empty bank without initializing CUDA eagerly."""
        self._owns_stream_pool = stream_pool is None
        self._stream_pool = stream_pool or NativePolicyCudaStreamPool()
        self._stream_lease = self._stream_pool.lease()
        self._route_streams: dict[NativeRouteKey, tuple[torch.device, Any]] = {}
        self._compute_streams: dict[tuple[torch.device, _StreamLane], Any] = {}
        self._copy_streams: dict[torch.device, Any] = {}
        self._frozen_route_lanes: dict[NativeRouteKey, int] = {}
        self._wave_routes: set[NativeRouteKey] = set()
        self._wave_copy_devices: set[torch.device] = set()
        self._deferred: list[_DeferredCompletion] = []
        self._wave_active = False
        self._stream_lock = threading.Lock()
        self._outstanding: set[NativePolicyInferenceWaveTicket] = set()
        self._closed = False

    @property
    def wave_active(self) -> bool:
        """Return whether route submissions may defer their host boundary."""
        return self._wave_active

    @contextmanager
    def wave(self) -> Iterator[None]:
        """Submit all exact policies before one copy-stream synchronization."""
        with self.deferred_wave() as ticket:
            yield
        ticket.finish()

    @contextmanager
    def deferred_wave(self) -> Iterator[NativePolicyInferenceWaveTicket]:
        """Seal a wave without waiting for its CUDA or host completions."""
        if self._closed:
            raise RuntimeError("native policy inference bank is closed")
        if self._wave_active:
            raise RuntimeError("native policy inference waves cannot nest")
        ticket = NativePolicyInferenceWaveTicket(self)
        self._wave_active = True
        self._wave_routes.clear()
        self._wave_copy_devices.clear()
        self._deferred.clear()
        try:
            yield ticket
        except BaseException:
            routes, copy_devices, deferred = self._take_active_wave()
            try:
                self._synchronize_route_streams(routes)
                self._synchronize_copy_streams(copy_devices)
            except BaseException:
                # Preserve the exception that aborted the arena wave. CUDA
                # faults will still poison the owning process and stop rollout.
                pass
            _abort_deferred(deferred)
            ticket._abort_open()
            raise
        else:
            routes, copy_devices, deferred = self._take_active_wave()
            try:
                events = self._record_completion_events(routes, copy_devices)
            except BaseException:
                try:
                    self._synchronize_route_streams(routes)
                    self._synchronize_copy_streams(copy_devices)
                except BaseException:
                    pass
                _abort_deferred(deferred)
                ticket._abort_open()
                raise
            ticket._seal(events=events, deferred=deferred)
            with self._stream_lock:
                self._outstanding.add(ticket)
        finally:
            self._deferred.clear()
            self._wave_routes.clear()
            self._wave_copy_devices.clear()
            self._wave_active = False

    @contextmanager
    def route_stream(
        self,
        route: NativeRouteKey,
        *,
        device: torch.device | str,
    ) -> Iterator[None]:
        """Run one immutable policy on its persistent device compute lane."""
        if self._closed:
            raise RuntimeError("native policy inference bank is closed")
        resolved = torch.device(device)
        if resolved.type != "cuda":
            yield
            return
        stream = self._route_stream(route, device=resolved)
        if self._wave_active:
            self._wave_routes.add(route)
        with torch.cuda.stream(stream):
            yield

    def copy_stream(self, device: torch.device | str) -> Any | None:
        """Return the shared asynchronous D2H stream for one CUDA device."""
        if self._closed:
            raise RuntimeError("native policy inference bank is closed")
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return None
        with self._stream_lock:
            existing = self._copy_streams.get(resolved)
            if existing is None:
                existing = self._stream_lease.copy_stream(device=resolved)
                self._copy_streams[resolved] = existing
            self._wave_copy_devices.add(resolved)
        return existing

    def defer(
        self,
        completion: Callable[[], None],
        *,
        on_abort: Callable[[], None] | None = None,
    ) -> None:
        """Run a host scatter after every wave D2H copy is visible."""
        if self._closed:
            raise RuntimeError("native policy inference bank is closed")
        if not self._wave_active:
            raise RuntimeError("native policy completion requires an active wave")
        with self._stream_lock:
            self._deferred.append((completion, on_abort))

    def close(self) -> None:
        """Drop window-local streams and abort any retained completion."""
        if self._closed:
            return
        if self._wave_active:
            raise RuntimeError("cannot close an active native policy wave")
        with self._stream_lock:
            if self._outstanding:
                raise RuntimeError(
                    "cannot close native policy inference bank with "
                    "outstanding wave tickets"
                )
        self._closed = True
        deferred = tuple(self._deferred)
        self._deferred.clear()
        _abort_deferred(deferred)
        self._wave_routes.clear()
        self._wave_copy_devices.clear()
        self._route_streams.clear()
        self._compute_streams.clear()
        self._copy_streams.clear()
        self._frozen_route_lanes.clear()
        self._stream_lease.close()
        if self._owns_stream_pool:
            self._stream_pool.close()

    def _route_stream(
        self,
        route: NativeRouteKey,
        *,
        device: torch.device,
    ) -> Any:
        with self._stream_lock:
            existing = self._route_streams.get(route)
            if existing is not None:
                bound_device, stream = existing
                if bound_device != device:
                    raise ValueError("one native policy route changed CUDA device")
                return stream
            lane = self._stream_lane(route)
            stream_key = (device, lane)
            stream = self._compute_streams.get(stream_key)
            if stream is None:
                stream = self._stream_lease.compute_stream(
                    device=device,
                    lane=lane,
                )
                self._compute_streams[stream_key] = stream
            # A cold route may have initialized model-local CUDA tensors on the
            # caller's stream before it is bound here. A lane may serve multiple
            # exact artifacts, so establish this dependency for every cold route.
            stream.wait_stream(torch.cuda.current_stream(device))
            self._route_streams[route] = (device, stream)
            return stream

    def _stream_lane(self, route: NativeRouteKey) -> _StreamLane:
        """Give the first active frozen artifacts distinct compute lanes."""
        if route.kind == "current":
            return ("current", 0)
        lane = self._frozen_route_lanes.get(route)
        if lane is None:
            lane = len(self._frozen_route_lanes) % _FROZEN_STREAM_LANES
            self._frozen_route_lanes[route] = lane
        return ("frozen", lane)

    def _take_active_wave(
        self,
    ) -> tuple[
        tuple[NativeRouteKey, ...],
        tuple[torch.device, ...],
        tuple[_DeferredCompletion, ...],
    ]:
        """Snapshot and detach mutable submission state from the active wave."""
        with self._stream_lock:
            routes = tuple(sorted(self._wave_routes))
            copy_devices = tuple(sorted(self._wave_copy_devices, key=str))
            deferred = tuple(self._deferred)
            self._wave_routes.clear()
            self._wave_copy_devices.clear()
            self._deferred.clear()
        return routes, copy_devices, deferred

    def _record_completion_events(
        self,
        routes: tuple[NativeRouteKey, ...],
        copy_devices: tuple[torch.device, ...],
    ) -> tuple[Any, ...]:
        """Record exact per-wave boundaries without synchronizing a stream."""
        events: list[Any] = []
        recorded_streams: set[int] = set()
        for route in routes:
            stream = self._route_streams[route][1]
            stream_identity = id(stream)
            if stream_identity in recorded_streams:
                continue
            events.append(_record_stream_event(stream))
            recorded_streams.add(stream_identity)
        for device in copy_devices:
            stream = self._copy_streams[device]
            stream_identity = id(stream)
            if stream_identity in recorded_streams:
                continue
            events.append(_record_stream_event(stream))
            recorded_streams.add(stream_identity)
        return tuple(events)

    def _synchronize_copy_streams(
        self,
        copy_devices: tuple[torch.device, ...],
    ) -> None:
        for device in copy_devices:
            self._copy_streams[device].synchronize()

    def _synchronize_route_streams(
        self,
        routes: tuple[NativeRouteKey, ...],
    ) -> None:
        synchronized: set[int] = set()
        for route in routes:
            stream = self._route_streams[route][1]
            stream_identity = id(stream)
            if stream_identity in synchronized:
                continue
            stream.synchronize()
            synchronized.add(stream_identity)

    def _release_ticket(self, ticket: NativePolicyInferenceWaveTicket) -> None:
        """Forget one resolved ticket while retaining persistent CUDA lanes."""
        with self._stream_lock:
            self._outstanding.discard(ticket)


def _stream_priority(kind: _StreamKind) -> int:
    """Keep every rollout inference route above the overlapping learner."""
    if kind == "current":
        return _CURRENT_STREAM_PRIORITY
    return _FROZEN_STREAM_PRIORITY


def _record_stream_event(stream: Any) -> Any:
    """Record a lightweight completion boundary on one CUDA stream."""
    event_factory = cast(Any, torch.cuda.Event)
    event = event_factory()
    event.record(stream)
    return event


def _synchronize_events(events: tuple[Any, ...]) -> None:
    """Wait only through this ticket's recorded stream boundaries."""
    for event in events:
        event.synchronize()


def _abort_deferred(deferred: tuple[_DeferredCompletion, ...]) -> None:
    """Best-effort cleanup while preserving the wave's primary failure."""
    for _complete, abort in reversed(deferred):
        if abort is None:
            continue
        with suppress(BaseException):
            abort()


__all__ = [
    "NativePolicyCudaStreamPool",
    "NativePolicyInferenceBank",
    "NativePolicyInferenceWaveTicket",
]
