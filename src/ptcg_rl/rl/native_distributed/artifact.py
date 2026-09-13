"""Raw dual-fingerprint BF16 rollout-artifact transport."""

from __future__ import annotations

import hashlib
import math
import struct
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import msgpack
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.native_distributed.contracts import (
    NativeBfloat16ArtifactManifest,
    NativeBfloat16TensorSpec,
)
from ptcg_rl.rl.native_distributed.control import NativeRolloutProtocolError

Array = npt.NDArray[np.generic]

_SCHEMA = "ptcg-rl/native-bfloat16-rollout-artifact/v1"
_CODEC = "msgpack-header+raw-contiguous-tensor-multipart"
_HEADER_FIELDS = ("schema", "codec", "compression", "manifest")
_WIRE_DOMAIN = b"ptcg-rl/native-bfloat16-rollout-artifact/v1\x00"
NATIVE_BFLOAT16_ARTIFACT_SEMANTICS_FINGERPRINT = hashlib.sha256(
    _WIRE_DOMAIN + _SCHEMA.encode() + b"\x00" + _CODEC.encode() + b"\x00bf16"
).hexdigest()
_MAX_HEADER_BYTES = 1 << 24
_MAX_TENSORS = 100_000
_TORCH_TO_NUMPY: dict[torch.dtype, np.dtype[np.generic]] = {
    torch.bool: np.dtype("|b1"),
    torch.uint8: np.dtype("|u1"),
    torch.int8: np.dtype("|i1"),
    torch.int16: np.dtype("<i2"),
    torch.int32: np.dtype("<i4"),
    torch.int64: np.dtype("<i8"),
    torch.float16: np.dtype("<f2"),
    torch.float32: np.dtype("<f4"),
    torch.float64: np.dtype("<f8"),
}
_NUMPY_TO_TORCH: dict[str, torch.dtype] = {
    dtype.str: torch_dtype for torch_dtype, dtype in _TORCH_TO_NUMPY.items()
}


@dataclass(frozen=True, slots=True)
class EncodedBfloat16RolloutArtifact:
    """Prepared host artifact whose frames can be sent to many workers."""

    manifest: NativeBfloat16ArtifactManifest
    header: bytes
    frames: tuple[memoryview, ...]
    _owners: tuple[Array, ...]
    preparation_timing: NativeArtifactPreparationTiming

    @property
    def wire_bytes(self) -> int:
        """Return header plus payload bytes."""
        return len(self.header) + sum(frame.nbytes for frame in self.frames)


class DecodedBfloat16RolloutArtifact:
    """Borrowed tensor frames verified against both artifact identities."""

    __slots__ = ("manifest", "_frame_owners", "_released", "_tensors")

    def __init__(
        self,
        *,
        manifest: NativeBfloat16ArtifactManifest,
        tensors: dict[str, Tensor],
        frame_owners: tuple[Any, ...],
    ) -> None:
        self.manifest = manifest
        self._tensors = tensors
        self._frame_owners = frame_owners
        self._released = False

    @property
    def tensors(self) -> Mapping[str, Tensor]:
        """Return borrowed CPU tensors until explicit release."""
        if self._released:
            raise RuntimeError("BF16 rollout artifact frames have been released")
        return self._tensors

    @property
    def frame_count(self) -> int:
        """Return the retained raw-frame count."""
        return len(self._frame_owners)

    def materialize_state_dict(
        self,
        *,
        device: torch.device | str = "cpu",
    ) -> dict[str, Tensor]:
        """Copy verified frames into an independently owned model state."""
        if self._released:
            raise RuntimeError("BF16 rollout artifact frames have been released")
        target = torch.device(device)
        return {
            name: tensor.clone().to(device=target)
            for name, tensor in self._tensors.items()
        }

    def release(self) -> None:
        """Drop all borrowed tensor views and frame owners."""
        if self._released:
            return
        self._tensors.clear()
        self._frame_owners = ()
        self._released = True


@dataclass(frozen=True, slots=True)
class NativeArtifactPreparationTiming:
    """Measured model-sized passes used to construct one wire artifact."""

    source_fingerprint_seconds: float
    bf16_conversion_seconds: float
    wire_hashing_seconds: float


def prepare_bfloat16_rollout_artifact(
    tensors: Mapping[str, Tensor],
    *,
    artifact_id: str,
    kind: Literal["current", "past_self"],
    source_policy_version: int,
    expected_source_fp32_fingerprint: str,
    model_config_fingerprint: str,
    exact_registry_fingerprint: str,
    precomputed_source_fp32_fingerprint: str | None = None,
) -> EncodedBfloat16RolloutArtifact:
    """Verify an FP32 source and prepare its canonical BF16 wire projection."""
    if not tensors or len(tensors) > _MAX_TENSORS:
        raise ValueError("BF16 rollout artifact tensor count is invalid")
    fingerprint_started_at = time.perf_counter()
    actual_source = (
        canonical_model_state_fingerprint(tensors)
        if precomputed_source_fp32_fingerprint is None
        else precomputed_source_fp32_fingerprint
    )
    source_fingerprint_seconds = time.perf_counter() - fingerprint_started_at
    if actual_source != expected_source_fp32_fingerprint:
        raise ValueError("BF16 rollout artifact FP32 source fingerprint differs")
    conversion_started_at = time.perf_counter()
    prepared: list[tuple[str, str, Array]] = []
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"rollout artifact value is not a tensor: {name}")
        if tensor.is_floating_point():
            if tensor.dtype != torch.float32:
                raise TypeError(
                    f"BF16 rollout artifacts require an FP32 floating source: {name}"
                )
            array = (
                tensor.detach()
                .to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
                .view(torch.uint16)
                .numpy()
                .astype("<u2", copy=False)
            )
            logical_dtype = "bfloat16"
        else:
            try:
                dtype = _TORCH_TO_NUMPY[tensor.dtype]
            except KeyError as exc:
                raise TypeError(
                    f"rollout artifact tensor dtype is unsupported: {name}"
                ) from exc
            array = (
                tensor.detach()
                .to(device="cpu")
                .contiguous()
                .numpy()
                .astype(dtype, copy=False)
            )
            logical_dtype = str(tensor.dtype).removeprefix("torch.")
        prepared.append((name, logical_dtype, cast(Array, array)))
    bf16_conversion_seconds = time.perf_counter() - conversion_started_at
    hashing_started_at = time.perf_counter()
    specs, wire_fingerprint = _tensor_specs_and_wire_fingerprint(prepared)
    wire_hashing_seconds = time.perf_counter() - hashing_started_at
    manifest = NativeBfloat16ArtifactManifest(
        artifact_id=artifact_id,
        kind=kind,
        source_policy_version=source_policy_version,
        source_fp32_fingerprint=actual_source,
        wire_bf16_fingerprint=wire_fingerprint,
        model_config_fingerprint=model_config_fingerprint,
        exact_registry_fingerprint=exact_registry_fingerprint,
        tensors=specs,
    )
    owners = tuple(array for _name, _logical_dtype, array in prepared)
    frames = tuple(memoryview(cast(Any, array)).cast("B") for array in owners)
    header = cast(
        bytes,
        msgpack.packb(
            {
                "schema": _SCHEMA,
                "codec": _CODEC,
                "compression": "none",
                "manifest": manifest.model_dump(mode="python"),
            },
            use_bin_type=True,
        ),
    )
    return EncodedBfloat16RolloutArtifact(
        manifest=manifest,
        header=header,
        frames=frames,
        _owners=owners,
        preparation_timing=NativeArtifactPreparationTiming(
            source_fingerprint_seconds=source_fingerprint_seconds,
            bf16_conversion_seconds=bf16_conversion_seconds,
            wire_hashing_seconds=wire_hashing_seconds,
        ),
    )


def decode_bfloat16_rollout_artifact(
    header: Any,
    frames: Sequence[Any],
    *,
    expected_manifest: NativeBfloat16ArtifactManifest,
) -> DecodedBfloat16RolloutArtifact:
    """Strictly verify and borrow one BF16 rollout artifact."""
    if sys.byteorder != "little":
        raise NativeRolloutProtocolError(
            "BF16 rollout artifact decoding requires a little-endian host"
        )
    header_view = _byte_view(header, name="header")
    if header_view.nbytes > _MAX_HEADER_BYTES:
        raise NativeRolloutProtocolError("BF16 rollout artifact header is too large")
    try:
        raw = msgpack.unpackb(
            bytes(header_view),
            raw=False,
            strict_map_key=True,
            object_pairs_hook=_strict_map,
            use_list=False,
        )
    except Exception as exc:
        raise NativeRolloutProtocolError(
            "BF16 rollout artifact header is invalid msgpack"
        ) from exc
    if not isinstance(raw, dict) or tuple(raw) != _HEADER_FIELDS:
        raise NativeRolloutProtocolError(
            "BF16 rollout artifact header fields are invalid"
        )
    fields = cast(dict[str, object], raw)
    if (
        fields["schema"] != _SCHEMA
        or fields["codec"] != _CODEC
        or fields["compression"] != "none"
    ):
        raise NativeRolloutProtocolError(
            "BF16 rollout artifact wire contract is unsupported"
        )
    try:
        manifest = NativeBfloat16ArtifactManifest.model_validate(fields["manifest"])
    except Exception as exc:
        raise NativeRolloutProtocolError(
            "BF16 rollout artifact manifest is invalid"
        ) from exc
    if manifest != expected_manifest:
        raise NativeRolloutProtocolError("BF16 rollout artifact manifest differs")
    if len(frames) != len(manifest.tensors):
        raise NativeRolloutProtocolError(
            "BF16 rollout artifact payload is truncated or has extra frames"
        )

    owners = tuple(frames)
    tensors: dict[str, Tensor] = {}
    fingerprint_inputs: list[tuple[str, str, memoryview]] = []
    for spec, owner in zip(manifest.tensors, owners, strict=True):
        frame = _byte_view(owner, name=spec.name)
        if frame.nbytes != spec.nbytes:
            raise NativeRolloutProtocolError(
                f"BF16 rollout artifact tensor {spec.name} length differs"
            )
        if hashlib.sha256(frame).hexdigest() != spec.sha256:
            raise NativeRolloutProtocolError(
                f"BF16 rollout artifact tensor {spec.name} hash differs"
            )
        dtype = _logical_torch_dtype(spec)
        expected_nbytes = math.prod(spec.shape) * dtype.itemsize
        if expected_nbytes != frame.nbytes:
            raise NativeRolloutProtocolError(
                f"BF16 rollout artifact tensor {spec.name} shape differs"
            )
        tensor = torch.frombuffer(frame, dtype=dtype, count=math.prod(spec.shape))
        tensors[spec.name] = tensor.reshape(spec.shape)
        fingerprint_inputs.append((spec.name, spec.logical_dtype, frame))
    if _wire_fingerprint_from_frames(fingerprint_inputs, manifest.tensors) != (
        manifest.wire_bf16_fingerprint
    ):
        raise NativeRolloutProtocolError(
            "BF16 rollout artifact whole-wire fingerprint differs"
        )
    return DecodedBfloat16RolloutArtifact(
        manifest=manifest,
        tensors=tensors,
        frame_owners=owners,
    )


def send_bfloat16_rollout_artifact(
    socket: Any,
    artifact: EncodedBfloat16RolloutArtifact,
    *,
    flags: int = 0,
    copy: bool = False,
) -> None:
    """Send one prepared artifact without another model-sized copy."""
    socket.send_multipart(
        (artifact.header, *artifact.frames),
        flags=flags,
        copy=copy,
    )


def recv_bfloat16_rollout_artifact(
    socket: Any,
    *,
    expected_manifest: NativeBfloat16ArtifactManifest,
    flags: int = 0,
    copy: bool = False,
) -> DecodedBfloat16RolloutArtifact:
    """Receive and verify one raw BF16 rollout artifact."""
    message = socket.recv_multipart(flags=flags, copy=copy)
    if not message:
        raise NativeRolloutProtocolError("BF16 rollout artifact message is empty")
    return decode_bfloat16_rollout_artifact(
        message[0],
        message[1:],
        expected_manifest=expected_manifest,
    )


def _tensor_specs_and_wire_fingerprint(
    tensors: Sequence[tuple[str, str, Array]],
) -> tuple[tuple[NativeBfloat16TensorSpec, ...], str]:
    frames = tuple(
        (
            name,
            logical_dtype,
            memoryview(cast(Any, array)).cast("B"),
        )
        for name, logical_dtype, array in tensors
    )
    specs = tuple(
        NativeBfloat16TensorSpec(
            name=name,
            logical_dtype=logical_dtype,
            wire_dtype=array.dtype.str,
            shape=tuple(int(value) for value in array.shape),
            nbytes=frame.nbytes,
            sha256=hashlib.sha256(frame).hexdigest(),
        )
        for (name, logical_dtype, array), (_frame_name, _frame_dtype, frame) in zip(
            tensors,
            frames,
            strict=True,
        )
    )
    return specs, _wire_fingerprint_from_frames(frames, specs)


def _wire_fingerprint_from_frames(
    tensors: Sequence[tuple[str, str, memoryview]],
    specs: Sequence[NativeBfloat16TensorSpec],
) -> str:
    digest = hashlib.sha256()
    digest.update(_WIRE_DOMAIN)
    digest.update(struct.pack(">Q", len(tensors)))
    for (name, logical_dtype, frame), spec in zip(tensors, specs, strict=True):
        encoded_name = name.encode("utf-8")
        encoded_logical_dtype = logical_dtype.encode("ascii")
        encoded_wire_dtype = spec.wire_dtype.encode("ascii")
        digest.update(struct.pack(">I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">I", len(encoded_logical_dtype)))
        digest.update(encoded_logical_dtype)
        digest.update(struct.pack(">I", len(encoded_wire_dtype)))
        digest.update(encoded_wire_dtype)
        digest.update(struct.pack(">I", len(spec.shape)))
        for dimension in spec.shape:
            digest.update(struct.pack(">Q", dimension))
        digest.update(struct.pack(">Q", frame.nbytes))
        digest.update(frame)
    return digest.hexdigest()


def _logical_torch_dtype(spec: NativeBfloat16TensorSpec) -> torch.dtype:
    if spec.logical_dtype == "bfloat16":
        if spec.wire_dtype != "<u2":
            raise NativeRolloutProtocolError(
                f"BF16 rollout artifact tensor {spec.name} wire dtype differs"
            )
        return torch.bfloat16
    try:
        dtype = _NUMPY_TO_TORCH[spec.wire_dtype]
    except KeyError as exc:
        raise NativeRolloutProtocolError(
            f"BF16 rollout artifact tensor {spec.name} dtype is unsupported"
        ) from exc
    if str(dtype).removeprefix("torch.") != spec.logical_dtype:
        raise NativeRolloutProtocolError(
            f"BF16 rollout artifact tensor {spec.name} logical dtype differs"
        )
    return dtype


def _strict_map(pairs: list[tuple[object, object]]) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate msgpack map key")
        result[key] = value
    return result


def _byte_view(value: Any, *, name: str) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError as exc:
        raise NativeRolloutProtocolError(
            f"BF16 rollout artifact {name} is not bytes-like"
        ) from exc
    if not view.c_contiguous:
        raise NativeRolloutProtocolError(
            f"BF16 rollout artifact {name} is not contiguous"
        )
    try:
        return view.cast("B")
    except TypeError as exc:
        raise NativeRolloutProtocolError(
            f"BF16 rollout artifact {name} cannot be byte-cast"
        ) from exc


__all__ = [
    "DecodedBfloat16RolloutArtifact",
    "EncodedBfloat16RolloutArtifact",
    "NATIVE_BFLOAT16_ARTIFACT_SEMANTICS_FINGERPRINT",
    "NativeArtifactPreparationTiming",
    "decode_bfloat16_rollout_artifact",
    "prepare_bfloat16_rollout_artifact",
    "recv_bfloat16_rollout_artifact",
    "send_bfloat16_rollout_artifact",
]
