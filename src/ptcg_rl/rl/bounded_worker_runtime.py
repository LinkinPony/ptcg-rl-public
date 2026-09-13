"""Socket protocol and child runtime for deadline-bounded native work."""

from __future__ import annotations

import ctypes
import os
import pickle
import select
import signal
import socket
import struct
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol


class WorkerInitializer(Protocol):
    """Construct process-local state after the subprocess starts."""

    def __call__(self, payload: Any) -> Any:
        """Return state retained for subsequent work items."""


class WorkerHandler(Protocol):
    """Execute one work item inside the isolated subprocess."""

    def __call__(
        self,
        state: Any,
        payload: Any,
        broker: WorkerBroker,
        deadline: float,
    ) -> Any:
        """Return a serializable result before ``deadline``."""


@dataclass(frozen=True)
class Ready:
    """Child startup completed."""


@dataclass(frozen=True)
class StartupFailure:
    """Child startup failed before accepting work."""

    error_type: str
    message: str


@dataclass(frozen=True)
class Work:
    """One isolated work request."""

    work_id: int
    payload: Any
    deadline: float


@dataclass(frozen=True)
class BrokerCall:
    """One child call into a parent-owned service."""

    work_id: int
    call_id: int
    payload: Any


@dataclass(frozen=True)
class BrokerReply:
    """Parent service result or bounded error."""

    work_id: int
    call_id: int
    result: Any = None
    error_type: str | None = None
    message: str = ""


@dataclass(frozen=True)
class WorkResult:
    """Child work result or bounded error."""

    work_id: int
    result: Any = None
    error_type: str | None = None
    message: str = ""


@dataclass(frozen=True)
class Shutdown:
    """Orderly idle-child shutdown request."""


class WorkerBroker:
    """Child-side synchronous calls into resources retained by the parent."""

    def __init__(
        self, channel: socket.socket, *, work_id: int, deadline: float
    ) -> None:
        self._channel = channel
        self._work_id = int(work_id)
        self._deadline = float(deadline)
        self._next_call_id = 0

    def call(self, payload: Any) -> Any:
        """Invoke the parent broker without extending the work deadline."""
        if time.perf_counter() >= self._deadline:
            raise TimeoutError("worker broker deadline expired")
        call_id = self._next_call_id
        self._next_call_id += 1
        send_message(
            self._channel,
            BrokerCall(
                work_id=self._work_id,
                call_id=call_id,
                payload=payload,
            ),
            deadline=self._deadline,
        )
        response = receive_message(self._channel, deadline=self._deadline)
        if not isinstance(response, BrokerReply):
            raise RuntimeError("worker broker received an invalid reply")
        if response.work_id != self._work_id or response.call_id != call_id:
            raise RuntimeError("worker broker reply identity mismatch")
        if response.error_type is not None:
            if response.error_type == "TimeoutError":
                raise TimeoutError(response.message)
            raise RuntimeError(f"{response.error_type}: {response.message}")
        return response.result


def worker_main(
    channel: socket.socket,
    initializer: WorkerInitializer,
    handler: WorkerHandler,
    initializer_payload: Any,
    expected_parent_pid: int,
) -> None:
    """Own native state and execute one serialized work item at a time."""
    _arm_parent_death_signal(expected_parent_pid)
    try:
        state = initializer(initializer_payload)
    except BaseException as exc:
        with suppress(OSError, TimeoutError):
            send_message(
                channel,
                StartupFailure(type(exc).__name__, bounded_message(exc)),
                deadline=time.perf_counter() + 1.0,
            )
        channel.close()
        return
    try:
        send_message(channel, Ready(), deadline=time.perf_counter() + 1.0)
        while True:
            message = receive_message(channel, deadline=None)
            if isinstance(message, Shutdown):
                return
            if not isinstance(message, Work):
                raise RuntimeError("bounded worker received an invalid command")
            broker = WorkerBroker(
                channel,
                work_id=message.work_id,
                deadline=message.deadline,
            )
            try:
                result = handler(
                    state,
                    message.payload,
                    broker,
                    message.deadline,
                )
                response = WorkResult(work_id=message.work_id, result=result)
            except BaseException as exc:
                response = WorkResult(
                    work_id=message.work_id,
                    error_type=type(exc).__name__,
                    message=bounded_message(exc),
                )
            send_message(channel, response, deadline=message.deadline)
    except (EOFError, BrokenPipeError, ConnectionError, OSError, TimeoutError):
        return
    finally:
        channel.close()


def send_message(
    channel: socket.socket,
    message: Any,
    *,
    deadline: float | None,
) -> None:
    """Send one framed pickle without waiting past an absolute deadline."""
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    frame = struct.pack("!Q", len(payload)) + payload
    _set_socket_timeout(channel, deadline)
    channel.sendall(frame)


def receive_message(
    channel: socket.socket,
    *,
    deadline: float | None,
) -> Any:
    """Receive one size-limited framed pickle by an absolute deadline."""
    _wait_readable(channel, deadline)
    header = _receive_exact(channel, 8, deadline=deadline)
    size = struct.unpack("!Q", header)[0]
    if size > 256 * 1024 * 1024:
        raise ValueError("bounded worker message exceeds the size limit")
    payload = _receive_exact(channel, size, deadline=deadline)
    return pickle.loads(payload)


def bounded_message(exc: BaseException) -> str:
    """Remove newlines and cap diagnostic payload size."""
    return str(exc).replace("\n", " ")[:512]


def _receive_exact(
    channel: socket.socket,
    size: int,
    *,
    deadline: float | None,
) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        _set_socket_timeout(channel, deadline)
        chunk = channel.recv(remaining)
        if not chunk:
            raise EOFError("bounded worker channel closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _wait_readable(channel: socket.socket, deadline: float | None) -> None:
    if deadline is None:
        return
    timeout = deadline - time.perf_counter()
    if timeout <= 0.0:
        raise TimeoutError("bounded worker deadline expired")
    readable, _, _ = select.select((channel,), (), (), timeout)
    if not readable:
        raise TimeoutError("bounded worker deadline expired")


def _set_socket_timeout(channel: socket.socket, deadline: float | None) -> None:
    if deadline is None:
        channel.settimeout(None)
        return
    timeout = deadline - time.perf_counter()
    if timeout <= 0.0:
        raise TimeoutError("bounded worker deadline expired")
    channel.settimeout(timeout)


def _arm_parent_death_signal(expected_parent_pid: int) -> None:
    """Kill a native worker if its actor parent exits during a blocked call."""
    if os.name != "posix" or not hasattr(signal, "SIGKILL"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(1, int(signal.SIGKILL), 0, 0, 0)
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != expected_parent_pid:
        os._exit(1)


__all__ = [
    "BrokerCall",
    "BrokerReply",
    "Ready",
    "Shutdown",
    "StartupFailure",
    "Work",
    "WorkerBroker",
    "WorkerHandler",
    "WorkerInitializer",
    "WorkResult",
    "bounded_message",
    "receive_message",
    "send_message",
    "worker_main",
]
