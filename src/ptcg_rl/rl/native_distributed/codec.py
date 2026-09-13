"""Msgpack header plus raw ndarray frames for native rollout parts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import msgpack
import numpy as np
import numpy.typing as npt

from ptcg_rl.rl.native_distributed.array_schema import (
    COMPACT_FRAGMENT_ARRAY_FIELDS,
    SEQUENCE_COMPACT_FRAGMENT_ARRAY_FIELDS,
    compact_fragment_array_fields,
    expected_wire_dtype,
    expected_wire_ndim,
    prepare_compact_fragment_arrays,
    validate_compact_fragment_arrays,
)
from ptcg_rl.rl.native_distributed.contracts import NativeRolloutPartIdentity
from ptcg_rl.rl.stateless_fragment_io import (
    SEQUENCE_FRAGMENT_ARRAY_SCHEMA,
    STATELESS_FRAGMENT_ARRAY_SCHEMA,
    CompactFragmentPart,
)

Array = npt.NDArray[np.generic]

_MESSAGE_SCHEMA = "ptcg-rl/native-rollout-part/v1"
_CODEC = "msgpack-header+raw-contiguous-ndarray-multipart"
_HEADER_FIELDS = (
    "schema",
    "codec",
    "compression",
    "array_schema",
    "part",
    "fields",
)
_ARRAY_HEADER_FIELDS = ("name", "dtype", "shape", "nbytes", "sha256")
_MAX_HEADER_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class DecodedNativeRolloutPart:
    """A decoded part borrowing its ZeroMQ payload buffers.

    The NumPy columns are read-only views. ``_frame_owners`` deliberately keeps
    every received ``zmq.Frame`` alive for this object's lifetime. Callers
    should retain this wrapper, rather than only its ``part`` attribute, until
    the learner has finished consuming the arrays.
    """

    identity: NativeRolloutPartIdentity
    part: CompactFragmentPart
    header_bytes: int
    payload_bytes: int
    _frame_owners: tuple[Any, ...] = field(repr=False)

    @property
    def frame_count(self) -> int:
        """Return the number of borrowed raw ndarray frames."""
        return len(self._frame_owners)


def encode_compact_fragment_part(
    identity: NativeRolloutPartIdentity,
    part: CompactFragmentPart,
) -> tuple[bytes, tuple[memoryview, ...]]:
    """Encode one full memory-only part without compression or persistence."""
    if part.path is not None:
        raise ValueError("distributed native rollout accepts only memory-only parts")
    if (
        identity.fragment_count != part.fragment_count
        or identity.decision_count != part.decision_count
    ):
        raise ValueError("native rollout part identity row counts do not match arrays")
    fields = compact_fragment_array_fields(part.arrays)
    arrays = prepare_compact_fragment_arrays(part.arrays)
    frames: list[memoryview] = []
    field_headers: list[dict[str, Any]] = []
    for name, array in zip(fields, arrays, strict=True):
        frame = memoryview(cast(Any, array.reshape(-1))).cast("B")
        frames.append(frame)
        field_headers.append(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "nbytes": frame.nbytes,
                "sha256": hashlib.sha256(frame).hexdigest(),
            }
        )
    header = {
        "schema": _MESSAGE_SCHEMA,
        "codec": _CODEC,
        "compression": "none",
        "array_schema": (
            SEQUENCE_FRAGMENT_ARRAY_SCHEMA
            if fields == SEQUENCE_COMPACT_FRAGMENT_ARRAY_FIELDS
            else STATELESS_FRAGMENT_ARRAY_SCHEMA
        ),
        "part": identity.model_dump(mode="python"),
        "fields": field_headers,
    }
    return msgpack.packb(header, use_bin_type=True), tuple(frames)


def decode_compact_fragment_part(
    header: bytes | bytearray | memoryview,
    frames: Sequence[Any],
) -> DecodedNativeRolloutPart:
    """Strictly decode raw frames as borrowed, read-only NumPy views."""
    header_view = _byte_view(header, name="header")
    if header_view.nbytes > _MAX_HEADER_BYTES:
        raise ValueError("native rollout header exceeds the size limit")
    unpacked = msgpack.unpackb(
        bytes(header_view),
        raw=False,
        strict_map_key=True,
    )
    if not isinstance(unpacked, dict) or tuple(unpacked) != _HEADER_FIELDS:
        raise ValueError("native rollout header fields are invalid")
    header_map = cast(dict[str, Any], unpacked)
    if (
        header_map["schema"] != _MESSAGE_SCHEMA
        or header_map["codec"] != _CODEC
        or header_map["compression"] != "none"
        or header_map["array_schema"]
        not in {
            STATELESS_FRAGMENT_ARRAY_SCHEMA,
            SEQUENCE_FRAGMENT_ARRAY_SCHEMA,
        }
    ):
        raise ValueError("native rollout header contract is unsupported")
    fields = (
        SEQUENCE_COMPACT_FRAGMENT_ARRAY_FIELDS
        if header_map["array_schema"] == SEQUENCE_FRAGMENT_ARRAY_SCHEMA
        else COMPACT_FRAGMENT_ARRAY_FIELDS
    )
    identity = NativeRolloutPartIdentity.model_validate(header_map["part"])
    raw_fields = header_map["fields"]
    if not isinstance(raw_fields, list):
        raise ValueError("native rollout array headers must be a list")
    if len(raw_fields) != len(fields):
        raise ValueError("native rollout array header count is invalid")
    if len(frames) != len(fields):
        raise ValueError("native rollout payload is truncated or has extra frames")

    owners = tuple(frames)
    arrays: dict[str, Array] = {}
    payload_bytes = 0
    for expected_name, raw_field, owner in zip(
        fields,
        raw_fields,
        owners,
        strict=True,
    ):
        if not isinstance(raw_field, dict) or tuple(raw_field) != _ARRAY_HEADER_FIELDS:
            raise ValueError("native rollout array header fields are invalid")
        metadata = cast(dict[str, Any], raw_field)
        if metadata["name"] != expected_name:
            raise ValueError("native rollout array field order is invalid")
        dtype = _header_dtype(expected_name, metadata["dtype"])
        shape = _header_shape(
            expected_name,
            metadata["shape"],
            ndim=expected_wire_ndim(expected_name),
        )
        expected_nbytes = math.prod(shape) * dtype.itemsize
        nbytes = metadata["nbytes"]
        if type(nbytes) is not int or nbytes != expected_nbytes:
            raise ValueError(
                f"native rollout field {expected_name} has invalid nbytes metadata"
            )
        frame = _byte_view(owner, name=expected_name)
        if frame.nbytes != expected_nbytes:
            raise ValueError(
                f"native rollout field {expected_name} frame length mismatch"
            )
        expected_hash = metadata["sha256"]
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
            or hashlib.sha256(frame).hexdigest() != expected_hash
        ):
            raise ValueError(
                f"native rollout field {expected_name} frame hash mismatch"
            )
        values = np.frombuffer(frame, dtype=dtype, count=math.prod(shape))
        array = values.reshape(shape)
        array.setflags(write=False)
        arrays[expected_name] = array
        payload_bytes += frame.nbytes

    validate_compact_fragment_arrays(arrays)
    part = CompactFragmentPart(path=None, arrays=arrays)
    if (
        identity.fragment_count != part.fragment_count
        or identity.decision_count != part.decision_count
    ):
        raise ValueError("native rollout part identity row counts do not match arrays")
    return DecodedNativeRolloutPart(
        identity=identity,
        part=part,
        header_bytes=header_view.nbytes,
        payload_bytes=payload_bytes,
        _frame_owners=owners,
    )


def send_compact_fragment_part(
    socket: Any,
    identity: NativeRolloutPartIdentity,
    part: CompactFragmentPart,
    *,
    flags: int = 0,
    copy: bool = False,
) -> None:
    """Send one encoded part as a ZeroMQ multipart message."""
    header, frames = encode_compact_fragment_part(identity, part)
    socket.send_multipart((header, *frames), flags=flags, copy=copy)


def recv_compact_fragment_part(
    socket: Any,
    *,
    flags: int = 0,
    copy: bool = False,
) -> DecodedNativeRolloutPart:
    """Receive one multipart message, borrowing ZeroMQ frames when requested."""
    message = socket.recv_multipart(flags=flags, copy=copy)
    if not message:
        raise ValueError("native rollout multipart message is empty")
    header = _byte_view(message[0], name="header")
    return decode_compact_fragment_part(header, message[1:])


def _header_dtype(name: str, value: object) -> np.dtype[np.generic]:
    if not isinstance(value, str):
        raise ValueError(f"native rollout field {name} dtype is invalid")
    expected = expected_wire_dtype(name)
    if expected is not None:
        if value != expected.str:
            raise ValueError(f"native rollout field {name} dtype mismatch")
        return expected
    if (
        len(value) < 3
        or not value.startswith("<U")
        or not value[2:].isdigit()
        or int(value[2:]) <= 0
    ):
        raise ValueError(f"native rollout field {name} Unicode dtype mismatch")
    return np.dtype(value)


def _header_shape(name: str, value: object, *, ndim: int) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != ndim
        or any(type(dimension) is not int or dimension < 0 for dimension in value)
    ):
        raise ValueError(f"native rollout field {name} shape is invalid")
    return tuple(cast(list[int], value))


def _byte_view(value: Any, *, name: str) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError as exc:
        raise ValueError(f"native rollout {name} frame is not bytes-like") from exc
    if not view.c_contiguous:
        raise ValueError(f"native rollout {name} frame is not contiguous")
    try:
        return view.cast("B")
    except TypeError as exc:
        raise ValueError(f"native rollout {name} frame cannot be byte-cast") from exc
