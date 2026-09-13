"""Strict HELLO/READY compatibility handshake for native rollout workers."""

from __future__ import annotations

from typing import Any, cast

import msgpack

from ptcg_rl.rl.native_distributed.contracts import (
    NativeRolloutWindowIdentity,
    NativeRolloutWorkerIdentity,
)
from ptcg_rl.rl.native_distributed.control import NativeRolloutProtocolError

_HANDSHAKE_SCHEMA = "ptcg-rl/native-rollout-handshake/v1"
_HELLO_FIELDS = ("schema", "message_type", "worker")
_READY_FIELDS = ("schema", "message_type", "worker", "window")
_MAX_HANDSHAKE_BYTES = 1 << 16


def encode_worker_hello(worker: NativeRolloutWorkerIdentity) -> bytes:
    """Encode one worker's complete immutable compatibility identity."""
    return _pack(
        {
            "schema": _HANDSHAKE_SCHEMA,
            "message_type": "HELLO",
            "worker": worker.model_dump(mode="python"),
        }
    )


def decode_worker_hello(
    frame: Any,
    *,
    expected_worker: NativeRolloutWorkerIdentity,
) -> NativeRolloutWorkerIdentity:
    """Decode HELLO and reject any worker or compatibility mismatch."""
    fields = _unpack(frame, expected_fields=_HELLO_FIELDS)
    if fields["message_type"] != "HELLO":
        raise NativeRolloutProtocolError(
            "native rollout handshake message is not HELLO"
        )
    worker = _worker(fields["worker"])
    if worker != expected_worker:
        raise NativeRolloutProtocolError(
            "native rollout HELLO worker compatibility identity differs"
        )
    return worker


def encode_worker_ready(
    worker: NativeRolloutWorkerIdentity,
    window: NativeRolloutWindowIdentity,
) -> bytes:
    """Bind READY to the accepted worker and exact behavior window."""
    return _pack(
        {
            "schema": _HANDSHAKE_SCHEMA,
            "message_type": "READY",
            "worker": worker.model_dump(mode="python"),
            "window": window.model_dump(mode="python"),
        }
    )


def decode_worker_ready(
    frame: Any,
    *,
    expected_worker: NativeRolloutWorkerIdentity,
    expected_window: NativeRolloutWindowIdentity,
) -> tuple[NativeRolloutWorkerIdentity, NativeRolloutWindowIdentity]:
    """Decode READY and require exact worker and rollout-window identities."""
    fields = _unpack(frame, expected_fields=_READY_FIELDS)
    if fields["message_type"] != "READY":
        raise NativeRolloutProtocolError(
            "native rollout handshake message is not READY"
        )
    worker = _worker(fields["worker"])
    window = _window(fields["window"])
    if worker != expected_worker or window != expected_window:
        raise NativeRolloutProtocolError(
            "native rollout READY compatibility identity differs"
        )
    return worker, window


def send_worker_hello(
    socket: Any,
    worker: NativeRolloutWorkerIdentity,
    *,
    flags: int = 0,
    copy: bool = True,
) -> None:
    """Send HELLO as exactly one ZeroMQ multipart frame."""
    socket.send_multipart(
        (encode_worker_hello(worker),),
        flags=flags,
        copy=copy,
    )


def recv_worker_hello(
    socket: Any,
    *,
    expected_worker: NativeRolloutWorkerIdentity,
    flags: int = 0,
    copy: bool = False,
) -> NativeRolloutWorkerIdentity:
    """Receive exactly one HELLO frame and validate compatibility."""
    message = socket.recv_multipart(flags=flags, copy=copy)
    if len(message) != 1:
        raise NativeRolloutProtocolError(
            "native rollout HELLO must have exactly one frame"
        )
    return decode_worker_hello(message[0], expected_worker=expected_worker)


def send_worker_ready(
    socket: Any,
    worker: NativeRolloutWorkerIdentity,
    window: NativeRolloutWindowIdentity,
    *,
    flags: int = 0,
    copy: bool = True,
) -> None:
    """Send READY as exactly one ZeroMQ multipart frame."""
    socket.send_multipart(
        (encode_worker_ready(worker, window),),
        flags=flags,
        copy=copy,
    )


def recv_worker_ready(
    socket: Any,
    *,
    expected_worker: NativeRolloutWorkerIdentity,
    expected_window: NativeRolloutWindowIdentity,
    flags: int = 0,
    copy: bool = False,
) -> tuple[NativeRolloutWorkerIdentity, NativeRolloutWindowIdentity]:
    """Receive exactly one READY frame and validate both identities."""
    message = socket.recv_multipart(flags=flags, copy=copy)
    if len(message) != 1:
        raise NativeRolloutProtocolError(
            "native rollout READY must have exactly one frame"
        )
    return decode_worker_ready(
        message[0],
        expected_worker=expected_worker,
        expected_window=expected_window,
    )


def _pack(fields: dict[str, object]) -> bytes:
    return cast(bytes, msgpack.packb(fields, use_bin_type=True))


def _unpack(
    frame: Any,
    *,
    expected_fields: tuple[str, ...],
) -> dict[str, object]:
    try:
        view = memoryview(frame)
        if not view.c_contiguous:
            raise TypeError
        data = view.cast("B")
    except (TypeError, ValueError) as exc:
        raise NativeRolloutProtocolError(
            "native rollout handshake frame is not contiguous bytes"
        ) from exc
    if data.nbytes > _MAX_HANDSHAKE_BYTES:
        raise NativeRolloutProtocolError(
            "native rollout handshake frame exceeds the size limit"
        )
    try:
        unpacked = msgpack.unpackb(
            bytes(data),
            raw=False,
            strict_map_key=True,
            object_pairs_hook=_strict_map,
        )
    except Exception as exc:
        raise NativeRolloutProtocolError(
            "native rollout handshake frame is invalid msgpack"
        ) from exc
    if not isinstance(unpacked, dict) or tuple(unpacked) != expected_fields:
        raise NativeRolloutProtocolError("native rollout handshake fields are invalid")
    fields = cast(dict[str, object], unpacked)
    if fields["schema"] != _HANDSHAKE_SCHEMA:
        raise NativeRolloutProtocolError(
            "native rollout handshake schema is unsupported"
        )
    return fields


def _strict_map(pairs: list[tuple[object, object]]) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate msgpack map key")
        result[key] = value
    return result


def _worker(value: object) -> NativeRolloutWorkerIdentity:
    try:
        return NativeRolloutWorkerIdentity.model_validate(value)
    except Exception as exc:
        raise NativeRolloutProtocolError(
            "native rollout handshake worker identity is invalid"
        ) from exc


def _window(value: object) -> NativeRolloutWindowIdentity:
    try:
        return NativeRolloutWindowIdentity.model_validate(value)
    except Exception as exc:
        raise NativeRolloutProtocolError(
            "native rollout handshake window identity is invalid"
        ) from exc


__all__ = [
    "decode_worker_hello",
    "decode_worker_ready",
    "encode_worker_hello",
    "encode_worker_ready",
    "recv_worker_hello",
    "recv_worker_ready",
    "send_worker_hello",
    "send_worker_ready",
]
