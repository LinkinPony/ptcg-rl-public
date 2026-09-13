"""Length-prefixed msgpack transport for persistent native match executors."""

from __future__ import annotations

import io
import os
import select
import struct
import time
from collections.abc import Mapping
from typing import Any, BinaryIO

import msgpack

_HEADER = struct.Struct("!Q")
_MAXIMUM_FRAME_BYTES = 64 * 1024 * 1024


class NativeMatchProtocolError(RuntimeError):
    """Raised when a native match process violates the framing contract."""


def read_frame(stream: BinaryIO) -> dict[str, Any] | None:
    """Read one mapping frame, returning ``None`` only at a clean EOF."""
    header = _read_exact(stream, _HEADER.size, allow_clean_eof=True)
    if header is None:
        return None
    (size,) = _HEADER.unpack(header)
    if size <= 0 or size > _MAXIMUM_FRAME_BYTES:
        raise NativeMatchProtocolError(f"invalid native match frame size: {size}")
    payload = _read_exact(stream, size, allow_clean_eof=False)
    assert payload is not None
    try:
        decoded = msgpack.unpackb(payload, raw=False)
    except (msgpack.ExtraData, msgpack.FormatError, ValueError) as error:
        raise NativeMatchProtocolError("invalid native match msgpack") from error
    if not isinstance(decoded, Mapping):
        raise NativeMatchProtocolError("native match frame must contain a mapping")
    return {str(key): value for key, value in decoded.items()}


def write_frame(stream: BinaryIO, payload: Mapping[str, Any]) -> None:
    """Atomically publish one mapping frame to a buffered binary stream."""
    encoded = msgpack.packb(dict(payload), use_bin_type=True)
    if not encoded or len(encoded) > _MAXIMUM_FRAME_BYTES:
        raise NativeMatchProtocolError("native match response frame is too large")
    stream.write(_HEADER.pack(len(encoded)))
    stream.write(encoded)
    stream.flush()


def encode_frame(payload: Mapping[str, Any]) -> bytes:
    """Encode one complete frame for subprocess clients and tests."""
    stream = io.BytesIO()
    write_frame(stream, payload)
    return stream.getvalue()


def read_frame_fd(fd: int, *, timeout_seconds: float) -> dict[str, Any]:
    """Read one complete frame from a pipe with a whole-response deadline."""
    if timeout_seconds <= 0.0:
        raise ValueError("native match response timeout must be positive")
    deadline = time.monotonic() + timeout_seconds
    header = _read_fd_exact(fd, _HEADER.size, deadline=deadline)
    (size,) = _HEADER.unpack(header)
    if size <= 0 or size > _MAXIMUM_FRAME_BYTES:
        raise NativeMatchProtocolError(f"invalid native match frame size: {size}")
    encoded = _read_fd_exact(fd, size, deadline=deadline)
    try:
        decoded = msgpack.unpackb(encoded, raw=False)
    except (msgpack.ExtraData, msgpack.FormatError, ValueError) as error:
        raise NativeMatchProtocolError("invalid native match msgpack") from error
    if not isinstance(decoded, Mapping):
        raise NativeMatchProtocolError("native match frame must contain a mapping")
    return {str(key): value for key, value in decoded.items()}


def _read_exact(
    stream: BinaryIO,
    size: int,
    *,
    allow_clean_eof: bool,
) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if allow_clean_eof and remaining == size:
                return None
            raise NativeMatchProtocolError("truncated native match frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_fd_exact(fd: int, size: int, *, deadline: float) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        timeout = deadline - time.monotonic()
        if timeout <= 0.0:
            raise TimeoutError("native match executor response timed out")
        readable, _, _ = select.select((fd,), (), (), timeout)
        if not readable:
            raise TimeoutError("native match executor response timed out")
        chunk = os.read(fd, remaining)
        if not chunk:
            raise NativeMatchProtocolError("native match executor closed its pipe")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__ = [
    "NativeMatchProtocolError",
    "encode_frame",
    "read_frame",
    "read_frame_fd",
    "write_frame",
]
