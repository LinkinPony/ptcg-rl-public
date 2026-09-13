"""Three-channel ROUTER/DEALER socket ownership for native collection."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any, Self

import zmq


@dataclass(frozen=True, slots=True)
class NativeDistributedEndpoints:
    """Resolved control, artifact, and data endpoints."""

    control: str
    artifact: str
    data: str

    @classmethod
    def tcp(cls, host: str, *, control: int, artifact: int, data: int) -> Self:
        """Construct three TCP endpoints from one host and distinct ports."""
        if len({control, artifact, data}) != 3:
            raise ValueError("native distributed endpoint ports must be distinct")
        try:
            address = ip_address(host)
        except ValueError as exc:
            raise ValueError(
                "native distributed endpoints require a private IP literal"
            ) from exc
        if address.is_unspecified or not (address.is_private or address.is_loopback):
            raise ValueError(
                "native distributed endpoints require a private IP literal"
            )
        return cls(
            control=f"tcp://{host}:{control}",
            artifact=f"tcp://{host}:{artifact}",
            data=f"tcp://{host}:{data}",
        )


class NativeCoordinatorSockets:
    """Own the three coordinator ROUTER sockets on one thread."""

    def __init__(
        self,
        endpoints: NativeDistributedEndpoints,
        *,
        io_threads: int,
        high_watermark: int,
    ) -> None:
        """Bind all channels or release every partial bind on failure."""
        self.context = zmq.Context(io_threads=io_threads)
        self.control = self.context.socket(zmq.ROUTER)
        self.artifact = self.context.socket(zmq.ROUTER)
        self.data = self.context.socket(zmq.ROUTER)
        self._closed = False
        try:
            for socket in (self.control, self.artifact, self.data):
                _configure_socket(socket, high_watermark=high_watermark)
                socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
            self.control.bind(endpoints.control)
            self.artifact.bind(endpoints.artifact)
            self.data.bind(endpoints.data)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close all ROUTER channels without a blocking linger."""
        if self._closed:
            return
        self._closed = True
        for socket in (self.control, self.artifact, self.data):
            socket.close(linger=0)
        self.context.term()

    def __enter__(self) -> NativeCoordinatorSockets:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class NativeWorkerSockets:
    """Own the three worker DEALER sockets under one session identity."""

    def __init__(
        self,
        endpoints: NativeDistributedEndpoints,
        *,
        worker_id: str,
        session_id: str,
        io_threads: int,
        high_watermark: int,
    ) -> None:
        """Connect independent channels with channel-specific routing IDs."""
        self.context = zmq.Context(io_threads=io_threads)
        self.control = self.context.socket(zmq.DEALER)
        self.artifact = self.context.socket(zmq.DEALER)
        self.data = self.context.socket(zmq.DEALER)
        self._closed = False
        try:
            for channel, socket in (
                ("control", self.control),
                ("artifact", self.artifact),
                ("data", self.data),
            ):
                _configure_socket(socket, high_watermark=high_watermark)
                routing_id = f"{worker_id}/{session_id}/{channel}".encode()
                socket.setsockopt(zmq.ROUTING_ID, routing_id)
            self.control.connect(endpoints.control)
            self.artifact.connect(endpoints.artifact)
            self.data.connect(endpoints.data)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close all DEALER channels without a blocking linger."""
        if self._closed:
            return
        self._closed = True
        for socket in (self.control, self.artifact, self.data):
            socket.close(linger=0)
        self.context.term()

    def __enter__(self) -> NativeWorkerSockets:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def recv_router_multipart(
    socket: Any,
    *,
    flags: int = 0,
    copy: bool = False,
) -> tuple[bytes, list[Any]]:
    """Split the mandatory ROUTER identity from application frames."""
    message = socket.recv_multipart(flags=flags, copy=copy)
    if len(message) < 2:
        raise ValueError("native distributed ROUTER message is truncated")
    routing_frame = memoryview(message[0]).cast("B")
    if not routing_frame:
        raise ValueError("native distributed ROUTER identity is empty")
    return bytes(routing_frame), list(message[1:])


def send_router_multipart(
    socket: Any,
    routing_id: bytes,
    frames: tuple[Any, ...] | list[Any],
    *,
    flags: int = 0,
    copy: bool = False,
) -> None:
    """Send application frames to one exact ROUTER peer."""
    if not routing_id or not frames:
        raise ValueError("native distributed ROUTER response is empty")
    socket.send_multipart(
        (routing_id, *frames),
        flags=flags,
        copy=copy,
    )


def try_send_router_multipart(
    socket: Any,
    routing_id: bytes,
    frames: tuple[Any, ...] | list[Any],
    *,
    copy: bool = False,
) -> bool:
    """Try one ROUTER response without blocking the shared coordinator loop.

    A DEALER may disappear after its request has been received but before the
    ROUTER sends the response. That peer-local race must not block or terminate
    the single coordinator I/O thread. The worker will rebuild its session
    after the missing response; unexpected socket failures remain fatal.
    """
    try:
        send_router_multipart(
            socket,
            routing_id,
            frames,
            flags=zmq.DONTWAIT,
            copy=copy,
        )
    except zmq.ZMQError as exc:
        if exc.errno in {zmq.EAGAIN, zmq.EHOSTUNREACH}:
            return False
        raise
    return True


def _configure_socket(socket: Any, *, high_watermark: int) -> None:
    if high_watermark <= 0:
        raise ValueError("native distributed high watermark must be positive")
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, high_watermark)
    socket.setsockopt(zmq.RCVHWM, high_watermark)
    socket.setsockopt(zmq.IMMEDIATE, 1)


__all__ = [
    "NativeCoordinatorSockets",
    "NativeDistributedEndpoints",
    "NativeWorkerSockets",
    "recv_router_multipart",
    "send_router_multipart",
    "try_send_router_multipart",
]
