"""Raw, memory-only behavior-policy tensor transport."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import msgpack
import numpy as np
import numpy.typing as npt

from ptcg_rl.rl.native_distributed.contracts import NativeRolloutWindowIdentity
from ptcg_rl.rl.native_distributed.control import NativeRolloutProtocolError

Array = npt.NDArray[np.generic]
TensorSource = Mapping[str, Array] | Sequence[tuple[str, Array]]

_MESSAGE_SCHEMA = "ptcg-rl/native-behavior-policy/v1"
_CODEC = "msgpack-header+raw-contiguous-ndarray-multipart"
_HEADER_FIELDS = (
    "schema",
    "codec",
    "compression",
    "window",
    "artifact_fingerprint",
    "tensors",
)
_TENSOR_HEADER_FIELDS = ("name", "dtype", "shape", "nbytes", "sha256")
_MAX_HEADER_BYTES = 1 << 24
_MAX_TENSOR_COUNT = 100_000
_MAX_TENSOR_NAME_BYTES = 1 << 10
_SAFE_DTYPE_KINDS = frozenset("biufc")
_MODEL_STATE_DOMAIN = b"ptcg-rl/full-model-state/v1\x00"
_NUMPY_DTYPE_TO_TORCH_NAME = {
    "|b1": "bool",
    "|u1": "uint8",
    "<u2": "uint16",
    "<u4": "uint32",
    "<u8": "uint64",
    "|i1": "int8",
    "<i2": "int16",
    "<i4": "int32",
    "<i8": "int64",
    "<f2": "float16",
    "<f4": "float32",
    "<f8": "float64",
    "<c8": "complex64",
    "<c16": "complex128",
}


@dataclass(frozen=True, slots=True)
class BehaviorPolicyTensorSpec:
    """Expected tensor name, dtype, and shape in canonical wire order."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_tensor_name(self.name)
        _parse_wire_dtype(self.dtype)
        if type(self.shape) is not tuple or any(
            type(dimension) is not int or dimension < 0 for dimension in self.shape
        ):
            raise ValueError("behavior-policy tensor shape is invalid")


class DecodedBehaviorPolicyWeights:
    """Read-only tensor views borrowing received ZeroMQ frames.

    Keep this wrapper alive while loading or using its arrays. ``release()``
    drops all wrapper-owned views and frame references explicitly; accessing
    ``tensors`` afterward is an error.
    """

    __slots__ = (
        "artifact_fingerprint",
        "header_bytes",
        "identity",
        "payload_bytes",
        "_frame_owners",
        "_released",
        "_tensor_view",
        "_tensors",
    )

    def __init__(
        self,
        *,
        identity: NativeRolloutWindowIdentity,
        artifact_fingerprint: str,
        tensors: dict[str, Array],
        header_bytes: int,
        payload_bytes: int,
        frame_owners: tuple[Any, ...],
    ) -> None:
        self.identity = identity
        self.artifact_fingerprint = artifact_fingerprint
        self.header_bytes = header_bytes
        self.payload_bytes = payload_bytes
        self._tensors = tensors
        self._tensor_view = MappingProxyType(tensors)
        self._frame_owners = frame_owners
        self._released = False

    @property
    def tensors(self) -> Mapping[str, Array]:
        """Return read-only borrowed arrays until explicit release."""
        if self._released:
            raise RuntimeError("behavior-policy tensor frames have been released")
        return self._tensor_view

    @property
    def frame_count(self) -> int:
        """Return the number of raw frames still retained by this wrapper."""
        return len(self._frame_owners)

    @property
    def released(self) -> bool:
        """Report whether explicit ownership release has occurred."""
        return self._released

    def release(self) -> None:
        """Drop all wrapper-owned array views and ZeroMQ frame references."""
        if self._released:
            return
        self._tensors.clear()
        self._frame_owners = ()
        self._released = True

    def __enter__(self) -> DecodedBehaviorPolicyWeights:
        """Retain frames for a bounded context."""
        if self._released:
            raise RuntimeError("behavior-policy tensor frames have been released")
        return self

    def __exit__(self, *_args: object) -> None:
        """Release frame ownership when leaving a bounded context."""
        self.release()


def behavior_policy_tensor_specs(
    tensors: TensorSource,
) -> tuple[BehaviorPolicyTensorSpec, ...]:
    """Describe tensor names, dtypes, and shapes in canonical wire order."""
    return tuple(
        BehaviorPolicyTensorSpec(
            name=name,
            dtype=array.dtype.str,
            shape=tuple(array.shape),
        )
        for name, array in _prepare_tensors(tensors)
    )


def behavior_policy_artifact_fingerprint(tensors: TensorSource) -> str:
    """Exactly reproduce the production full-model-state fingerprint."""
    prepared = _prepare_tensors(tensors)
    return _canonical_model_state_fingerprint(prepared)


def encode_behavior_policy_weights(
    identity: NativeRolloutWindowIdentity,
    tensors: TensorSource,
) -> tuple[bytes, tuple[memoryview, ...]]:
    """Encode an exact behavior policy without pickle, compression, or disk."""
    prepared = _prepare_tensors(tensors)
    metadata = _tensor_metadata(prepared)
    artifact_fingerprint = _canonical_model_state_fingerprint(prepared)
    if artifact_fingerprint != identity.behavior_policy_fingerprint:
        raise ValueError(
            "behavior-policy artifact fingerprint differs from window identity"
        )
    frames = tuple(memoryview(cast(Any, array)).cast("B") for _, array in prepared)
    header = {
        "schema": _MESSAGE_SCHEMA,
        "codec": _CODEC,
        "compression": "none",
        "window": identity.model_dump(mode="python"),
        "artifact_fingerprint": artifact_fingerprint,
        "tensors": metadata,
    }
    return cast(bytes, msgpack.packb(header, use_bin_type=True)), frames


def decode_behavior_policy_weights(
    header: Any,
    frames: Sequence[Any],
    *,
    expected_identity: NativeRolloutWindowIdentity,
    expected_tensors: Sequence[BehaviorPolicyTensorSpec],
) -> DecodedBehaviorPolicyWeights:
    """Decode and authenticate borrowed tensor frames against local schema."""
    header_view = _byte_view(header, name="header")
    if header_view.nbytes > _MAX_HEADER_BYTES:
        raise NativeRolloutProtocolError(
            "behavior-policy header exceeds the size limit"
        )
    try:
        unpacked = msgpack.unpackb(
            bytes(header_view),
            raw=False,
            strict_map_key=True,
            object_pairs_hook=_strict_map,
        )
    except Exception as exc:
        raise NativeRolloutProtocolError(
            "behavior-policy header is invalid msgpack"
        ) from exc
    if not isinstance(unpacked, dict) or tuple(unpacked) != _HEADER_FIELDS:
        raise NativeRolloutProtocolError("behavior-policy header fields are invalid")
    fields = cast(dict[str, object], unpacked)
    if (
        fields["schema"] != _MESSAGE_SCHEMA
        or fields["codec"] != _CODEC
        or fields["compression"] != "none"
    ):
        raise NativeRolloutProtocolError("behavior-policy wire contract is unsupported")
    identity = _window_identity(fields["window"])
    if identity != expected_identity:
        raise NativeRolloutProtocolError(
            "behavior-policy rollout-window identity differs"
        )
    artifact_fingerprint = fields["artifact_fingerprint"]
    if not _valid_sha256(artifact_fingerprint):
        raise NativeRolloutProtocolError(
            "behavior-policy artifact fingerprint is invalid"
        )
    if artifact_fingerprint != expected_identity.behavior_policy_fingerprint:
        raise NativeRolloutProtocolError(
            "behavior-policy artifact fingerprint differs from window identity"
        )

    specs = _validate_expected_specs(expected_tensors)
    raw_tensors = fields["tensors"]
    if not isinstance(raw_tensors, list) or len(raw_tensors) != len(specs):
        raise NativeRolloutProtocolError(
            "behavior-policy tensor header count is invalid"
        )
    if len(frames) != len(specs):
        raise NativeRolloutProtocolError(
            "behavior-policy payload is truncated or has extra frames"
        )

    owners = tuple(frames)
    decoded: dict[str, Array] = {}
    canonical_frames: list[tuple[BehaviorPolicyTensorSpec, memoryview]] = []
    payload_bytes = 0
    for spec, raw_metadata, owner in zip(
        specs,
        raw_tensors,
        owners,
        strict=True,
    ):
        metadata = _validate_tensor_header(raw_metadata, spec)
        dtype = _parse_wire_dtype(cast(str, metadata["dtype"]))
        shape = tuple(cast(list[int], metadata["shape"]))
        expected_nbytes = math.prod(shape) * dtype.itemsize
        frame = _byte_view(owner, name=spec.name)
        if frame.nbytes != expected_nbytes:
            raise NativeRolloutProtocolError(
                f"behavior-policy tensor {spec.name} frame length mismatch"
            )
        if hashlib.sha256(frame).hexdigest() != metadata["sha256"]:
            raise NativeRolloutProtocolError(
                f"behavior-policy tensor {spec.name} frame hash mismatch"
            )
        values = np.frombuffer(frame, dtype=dtype, count=math.prod(shape))
        array = values.reshape(shape)
        array.setflags(write=False)
        decoded[spec.name] = array
        canonical_frames.append((spec, frame))
        payload_bytes += frame.nbytes

    actual_artifact_fingerprint = _canonical_model_state_fingerprint_from_frames(
        canonical_frames
    )
    if actual_artifact_fingerprint != artifact_fingerprint:
        raise NativeRolloutProtocolError(
            "behavior-policy whole artifact fingerprint mismatch"
        )
    return DecodedBehaviorPolicyWeights(
        identity=identity,
        artifact_fingerprint=artifact_fingerprint,
        tensors=decoded,
        header_bytes=header_view.nbytes,
        payload_bytes=payload_bytes,
        frame_owners=owners,
    )


def send_behavior_policy_weights(
    socket: Any,
    identity: NativeRolloutWindowIdentity,
    tensors: TensorSource,
    *,
    flags: int = 0,
    copy: bool = False,
) -> None:
    """Send one raw behavior-policy artifact as a multipart message."""
    header, frames = encode_behavior_policy_weights(identity, tensors)
    socket.send_multipart((header, *frames), flags=flags, copy=copy)


def recv_behavior_policy_weights(
    socket: Any,
    *,
    expected_identity: NativeRolloutWindowIdentity,
    expected_tensors: Sequence[BehaviorPolicyTensorSpec],
    flags: int = 0,
    copy: bool = False,
) -> DecodedBehaviorPolicyWeights:
    """Receive a raw behavior policy while borrowing ZeroMQ frame storage."""
    message = socket.recv_multipart(flags=flags, copy=copy)
    if not message:
        raise NativeRolloutProtocolError("behavior-policy multipart message is empty")
    return decode_behavior_policy_weights(
        message[0],
        message[1:],
        expected_identity=expected_identity,
        expected_tensors=expected_tensors,
    )


def _prepare_tensors(tensors: TensorSource) -> tuple[tuple[str, Array], ...]:
    source = tensors.items() if isinstance(tensors, Mapping) else tensors
    items = list(source)
    if not items or len(items) > _MAX_TENSOR_COUNT:
        raise ValueError("behavior-policy tensor count is invalid")
    prepared: list[tuple[str, Array]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("behavior-policy tensor entry is invalid")
        name, value = item
        _validate_tensor_name(name)
        if name in seen:
            raise ValueError(f"behavior-policy tensor name is duplicated: {name}")
        seen.add(name)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"behavior-policy tensor {name} is not an ndarray")
        _validate_source_dtype(name, value.dtype)
        array = (
            value
            if value.flags.c_contiguous
            else np.array(value, dtype=value.dtype, order="C", copy=True)
        )
        prepared.append((name, cast(Array, array)))
    prepared.sort(key=lambda item: item[0])
    return tuple(prepared)


def _tensor_metadata(
    tensors: Sequence[tuple[str, Array]],
) -> list[dict[str, object]]:
    metadata: list[dict[str, object]] = []
    for name, array in tensors:
        frame = memoryview(cast(Any, array)).cast("B")
        metadata.append(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "nbytes": frame.nbytes,
                "sha256": hashlib.sha256(frame).hexdigest(),
            }
        )
    return metadata


def _canonical_model_state_fingerprint(
    tensors: Sequence[tuple[str, Array]],
) -> str:
    framed = tuple(
        (
            BehaviorPolicyTensorSpec(
                name=name,
                dtype=array.dtype.str,
                shape=tuple(array.shape),
            ),
            memoryview(cast(Any, array)).cast("B"),
        )
        for name, array in tensors
    )
    return _canonical_model_state_fingerprint_from_frames(framed)


def _canonical_model_state_fingerprint_from_frames(
    tensors: Sequence[tuple[BehaviorPolicyTensorSpec, memoryview]],
) -> str:
    digest = hashlib.sha256()
    digest.update(_MODEL_STATE_DOMAIN)
    digest.update(struct.pack(">Q", len(tensors)))
    for spec, payload in tensors:
        encoded_name = spec.name.encode("utf-8")
        encoded_dtype = _torch_dtype_name(spec.dtype).encode("ascii")
        digest.update(struct.pack(">I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">I", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack(">I", len(spec.shape)))
        for dimension in spec.shape:
            digest.update(struct.pack(">Q", dimension))
        digest.update(struct.pack(">Q", payload.nbytes))
        digest.update(payload)
    return digest.hexdigest()


def _validate_expected_specs(
    expected: Sequence[BehaviorPolicyTensorSpec],
) -> tuple[BehaviorPolicyTensorSpec, ...]:
    specs = tuple(expected)
    if not specs or len(specs) > _MAX_TENSOR_COUNT:
        raise ValueError("expected behavior-policy tensor count is invalid")
    if tuple(sorted(specs, key=lambda spec: spec.name)) != specs:
        raise ValueError("expected behavior-policy tensors are not canonical")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("expected behavior-policy tensor name is duplicated")
    return specs


def _validate_tensor_header(
    value: object,
    spec: BehaviorPolicyTensorSpec,
) -> dict[str, object]:
    if not isinstance(value, dict) or tuple(value) != _TENSOR_HEADER_FIELDS:
        raise NativeRolloutProtocolError(
            f"behavior-policy tensor {spec.name} header fields are invalid"
        )
    metadata = cast(dict[str, object], value)
    if metadata["name"] != spec.name or metadata["dtype"] != spec.dtype:
        raise NativeRolloutProtocolError(
            f"behavior-policy tensor {spec.name} schema or order differs"
        )
    shape = metadata["shape"]
    if (
        not isinstance(shape, list)
        or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        or shape != list(spec.shape)
    ):
        raise NativeRolloutProtocolError(
            f"behavior-policy tensor {spec.name} shape differs"
        )
    dtype = _parse_wire_dtype(spec.dtype)
    expected_nbytes = math.prod(spec.shape) * dtype.itemsize
    if type(metadata["nbytes"]) is not int or metadata["nbytes"] != expected_nbytes:
        raise NativeRolloutProtocolError(
            f"behavior-policy tensor {spec.name} nbytes metadata is invalid"
        )
    if not _valid_sha256(metadata["sha256"]):
        raise NativeRolloutProtocolError(
            f"behavior-policy tensor {spec.name} frame hash is invalid"
        )
    return metadata


def _window_identity(value: object) -> NativeRolloutWindowIdentity:
    try:
        return NativeRolloutWindowIdentity.model_validate(value)
    except Exception as exc:
        raise NativeRolloutProtocolError(
            "behavior-policy rollout-window identity is invalid"
        ) from exc


def _strict_map(pairs: list[tuple[object, object]]) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate msgpack map key")
        result[key] = value
    return result


def _validate_tensor_name(value: object) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > _MAX_TENSOR_NAME_BYTES
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("behavior-policy tensor name is invalid")


def _validate_source_dtype(name: str, dtype: np.dtype[np.generic]) -> None:
    if (
        dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.kind not in _SAFE_DTYPE_KINDS
        or (dtype.byteorder == ">" and dtype.itemsize > 1)
    ):
        raise ValueError(f"behavior-policy tensor {name} dtype is unsupported")
    _parse_wire_dtype(dtype.str)


def _parse_wire_dtype(value: object) -> np.dtype[np.generic]:
    if not isinstance(value, str):
        raise ValueError("behavior-policy tensor dtype is invalid")
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("behavior-policy tensor dtype is invalid") from exc
    if (
        dtype.str != value
        or dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.kind not in _SAFE_DTYPE_KINDS
        or (dtype.byteorder == ">" and dtype.itemsize > 1)
    ):
        raise ValueError("behavior-policy tensor dtype is unsupported")
    _torch_dtype_name(dtype.str)
    return cast(np.dtype[np.generic], dtype)


def _torch_dtype_name(wire_dtype: str) -> str:
    try:
        return _NUMPY_DTYPE_TO_TORCH_NAME[wire_dtype]
    except KeyError as exc:
        raise ValueError(
            "behavior-policy tensor dtype has no canonical torch mapping"
        ) from exc


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _byte_view(value: Any, *, name: str) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError as exc:
        raise NativeRolloutProtocolError(
            f"behavior-policy {name} frame is not bytes-like"
        ) from exc
    if not view.c_contiguous:
        raise NativeRolloutProtocolError(
            f"behavior-policy {name} frame is not contiguous"
        )
    try:
        return view.cast("B")
    except TypeError as exc:
        raise NativeRolloutProtocolError(
            f"behavior-policy {name} frame cannot be byte-cast"
        ) from exc


__all__ = [
    "BehaviorPolicyTensorSpec",
    "DecodedBehaviorPolicyWeights",
    "behavior_policy_artifact_fingerprint",
    "behavior_policy_tensor_specs",
    "decode_behavior_policy_weights",
    "encode_behavior_policy_weights",
    "recv_behavior_policy_weights",
    "send_behavior_policy_weights",
]
