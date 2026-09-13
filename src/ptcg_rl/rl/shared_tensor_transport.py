"""Reusable exact-dtype tensor slots for shared-memory model transport."""

from __future__ import annotations

import hashlib
import json
import struct
import time
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import TypedDict, cast

import torch

TENSOR_TRANSPORT = "tensor_slots_v1"
_HEADER_BYTES = 64
_HEADER = struct.Struct("<Qq")


class TensorDescriptor(TypedDict):
    """One tensor's exact location and type inside a shared slot."""

    name: str
    dtype: str
    shape: list[int]
    offset: int
    numel: int
    nbytes: int


@dataclass(frozen=True, slots=True)
class SharedTensorPublication:
    """One complete but not yet externally advertised tensor slot."""

    name: str
    size_bytes: int
    sequence: int
    layout_path: Path
    layout_fingerprint: str
    copy_seconds: float


@dataclass(frozen=True, slots=True)
class SharedTensorReadSpec:
    """Manifest-bound inputs needed to read one tensor slot safely."""

    name: str
    size_bytes: int
    version: int
    sequence: int
    layout_path: Path
    layout_fingerprint: str


class SharedTensorSlotPublisher:
    """Own and alternate a fixed set of model-shaped shared-memory slots."""

    def __init__(
        self,
        directory: Path,
        *,
        filename_prefix: str,
        slot_count: int,
    ) -> None:
        if slot_count < 2:
            raise ValueError("shared tensor slot_count must be at least two")
        self.directory = Path(directory)
        self.filename_prefix = filename_prefix
        self.slot_count = int(slot_count)
        self._segments: dict[str, shared_memory.SharedMemory] = {}
        self._signature: tuple[tuple[str, tuple[int, ...], str], ...] | None = None
        self._descriptors: tuple[TensorDescriptor, ...] = ()
        self._slot_names: tuple[str, ...] = ()
        self._layout_path: Path | None = None
        self._layout_fingerprint: str | None = None
        self._size_bytes = 0
        self._next_slot = 0
        self._sequence = 0
        self._retired_names: list[str] = []
        self._retired_layouts: list[Path] = []

    @property
    def owned_names(self) -> frozenset[str]:
        """Return every live shared-memory name owned by this publisher."""
        return frozenset(self._segments)

    def write(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        version: int,
    ) -> SharedTensorPublication:
        """Copy a state into the inactive slot without advertising it."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self._ensure_slots(state_dict)
        slot_name = self._slot_names[self._next_slot]
        self._next_slot = (self._next_slot + 1) % len(self._slot_names)
        self._sequence += 2
        sequence = self._sequence
        segment = self._segments[slot_name]
        started_at = time.perf_counter()
        buffer = segment.buf
        try:
            if buffer is None:
                raise RuntimeError("shared tensor slot has no writable buffer")
            _HEADER.pack_into(buffer, 0, sequence - 1, version)
            for descriptor in self._descriptors:
                source = state_dict[descriptor["name"]]
                if descriptor["numel"] == 0:
                    continue
                destination = torch.frombuffer(
                    buffer,
                    dtype=source.dtype,
                    count=descriptor["numel"],
                    offset=descriptor["offset"],
                )
                destination.copy_(source.reshape(-1), non_blocking=False)
                del destination
            _HEADER.pack_into(buffer, 0, sequence, version)
        finally:
            del buffer
        if self._layout_path is None or self._layout_fingerprint is None:
            raise RuntimeError("shared tensor layout was not initialized")
        return SharedTensorPublication(
            name=slot_name,
            size_bytes=self._size_bytes,
            sequence=sequence,
            layout_path=self._layout_path,
            layout_fingerprint=self._layout_fingerprint,
            copy_seconds=time.perf_counter() - started_at,
        )

    def commit(self) -> None:
        """Retire topology slots only after the caller advertises the new one."""
        for name in self._retired_names:
            self._close_segment(name, unlink=True)
        self._retired_names.clear()
        for path in self._retired_layouts:
            path.unlink(missing_ok=True)
        self._retired_layouts.clear()

    def close(self) -> None:
        """Close and unlink every slot and layout owned by this publisher."""
        for name in tuple(self._segments):
            self._close_segment(name, unlink=True)
        for path in (*self._retired_layouts, self._layout_path):
            if path is not None:
                path.unlink(missing_ok=True)
        self._retired_layouts.clear()

    def _ensure_slots(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        signature = tuple(
            (name, tuple(tensor.shape), str(tensor.dtype))
            for name, tensor in sorted(state_dict.items())
        )
        if signature == self._signature and self._slot_names:
            return
        descriptors, size_bytes = tensor_layout(state_dict)
        layout_id = uuid.uuid4().hex
        layout_path = self.directory / f"shared_layout_{layout_id}.json"
        layout_record = {
            "format": TENSOR_TRANSPORT,
            "size_bytes": size_bytes,
            "tensors": descriptors,
        }
        layout_payload = json.dumps(
            layout_record,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        layout_fingerprint = hashlib.sha256(layout_payload).hexdigest()
        atomic_write_bytes(layout_path, layout_payload)

        new_names: list[str] = []
        try:
            for slot in range(self.slot_count):
                name = f"{self.filename_prefix}_slot{slot}_{layout_id}"
                segment = shared_memory.SharedMemory(
                    name=name,
                    create=True,
                    size=size_bytes,
                )
                self._segments[name] = segment
                new_names.append(name)
        except BaseException:
            for name in new_names:
                self._close_segment(name, unlink=True)
            layout_path.unlink(missing_ok=True)
            raise

        self._retired_names.extend(self._slot_names)
        if self._layout_path is not None:
            self._retired_layouts.append(self._layout_path)
        self._signature = signature
        self._descriptors = descriptors
        self._slot_names = tuple(new_names)
        self._layout_path = layout_path
        self._layout_fingerprint = layout_fingerprint
        self._size_bytes = size_bytes
        self._next_slot = 0

    def _close_segment(self, name: str, *, unlink: bool) -> None:
        segment = self._segments.pop(name, None)
        if segment is None:
            return
        segment.close()
        if unlink:
            with suppress(FileNotFoundError):
                segment.unlink()


def load_shared_tensor_slot(
    spec: SharedTensorReadSpec,
    *,
    map_location: str | torch.device,
) -> tuple[Mapping[str, torch.Tensor], float]:
    """Copy one stable tensor slot, rejecting concurrent overwrite races."""
    layout_payload = spec.layout_path.read_bytes()
    if hashlib.sha256(layout_payload).hexdigest() != spec.layout_fingerprint:
        raise ValueError("shared tensor layout differs from its fingerprint")
    raw_layout = json.loads(layout_payload)
    if not isinstance(raw_layout, dict) or raw_layout.get("format") != TENSOR_TRANSPORT:
        raise ValueError("shared tensor layout format is invalid")
    if int(raw_layout.get("size_bytes", -1)) != spec.size_bytes:
        raise ValueError("shared tensor layout size differs from publication")
    descriptors = raw_layout.get("tensors")
    if not isinstance(descriptors, list):
        raise ValueError("shared tensor layout descriptors are invalid")

    started_at = time.perf_counter()
    segment = shared_memory.SharedMemory(name=spec.name, create=False)
    buffer = segment.buf
    loaded: dict[str, torch.Tensor] = {}
    try:
        if buffer is None:
            raise RuntimeError("shared tensor slot has no readable buffer")
        sequence_before, version_before = _HEADER.unpack_from(buffer, 0)
        if sequence_before != spec.sequence or version_before != spec.version:
            raise FileNotFoundError(
                "shared tensor slot was superseded before it could be read"
            )
        device = torch.device(map_location)
        for raw_descriptor in descriptors:
            descriptor = validated_tensor_descriptor(
                raw_descriptor,
                size_bytes=spec.size_bytes,
            )
            dtype = torch_dtype(descriptor["dtype"])
            shape = tuple(descriptor["shape"])
            target = torch.empty(shape, dtype=dtype, device=device)
            if descriptor["numel"]:
                source = torch.frombuffer(
                    buffer,
                    dtype=dtype,
                    count=descriptor["numel"],
                    offset=descriptor["offset"],
                ).reshape(shape)
                target.copy_(source, non_blocking=False)
                del source
            loaded[descriptor["name"]] = target
        sequence_after, version_after = _HEADER.unpack_from(buffer, 0)
        if sequence_after != sequence_before or version_after != version_before:
            loaded.clear()
            raise FileNotFoundError(
                "shared tensor slot changed while it was being read"
            )
    finally:
        del buffer
        segment.close()
    return loaded, time.perf_counter() - started_at


def tensor_layout(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[tuple[TensorDescriptor, ...], int]:
    """Build a deterministic aligned layout for contiguous CPU tensors."""
    offset = _HEADER_BYTES
    descriptors: list[TensorDescriptor] = []
    for name, tensor in sorted(state_dict.items()):
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError(
                "shared tensor publication requires contiguous CPU tensors"
            )
        offset = align(offset, max(64, tensor.element_size()))
        numel = tensor.numel()
        nbytes = numel * tensor.element_size()
        descriptors.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "offset": offset,
                "numel": numel,
                "nbytes": nbytes,
            }
        )
        offset += nbytes
    return tuple(descriptors), max(_HEADER_BYTES, offset)


def validated_tensor_descriptor(
    raw: object,
    *,
    size_bytes: int,
) -> TensorDescriptor:
    """Validate untrusted layout metadata before creating tensor views."""
    if not isinstance(raw, dict):
        raise ValueError("shared tensor descriptor must be an object")
    name = raw.get("name")
    dtype = raw.get("dtype")
    shape = raw.get("shape")
    if not isinstance(name, str) or not name or not isinstance(dtype, str):
        raise ValueError("shared tensor descriptor name/dtype is invalid")
    if not isinstance(shape, list) or any(
        not isinstance(dimension, int) or dimension < 0 for dimension in shape
    ):
        raise ValueError("shared tensor descriptor shape is invalid")
    descriptor = cast(TensorDescriptor, raw)
    offset = int(descriptor["offset"])
    numel = int(descriptor["numel"])
    nbytes = int(descriptor["nbytes"])
    expected_numel = 1
    for dimension in shape:
        expected_numel *= dimension
    expected_nbytes = expected_numel * torch_dtype(dtype).itemsize
    if numel != expected_numel or nbytes != expected_nbytes:
        raise ValueError("shared tensor descriptor size is invalid")
    if offset < _HEADER_BYTES or offset + nbytes > size_bytes:
        raise ValueError("shared tensor descriptor exceeds its slot")
    return descriptor


def torch_dtype(name: str) -> torch.dtype:
    """Resolve a serialized exact torch dtype name."""
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported shared tensor dtype: {name}")
    return value


def align(value: int, alignment: int) -> int:
    """Round a byte offset up to an alignment boundary."""
    return ((value + alignment - 1) // alignment) * alignment


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace a small non-durable transport metadata file."""
    pending = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        pending.write_bytes(payload)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


__all__ = [
    "SharedTensorPublication",
    "SharedTensorReadSpec",
    "SharedTensorSlotPublisher",
    "TENSOR_TRANSPORT",
    "atomic_write_bytes",
    "load_shared_tensor_slot",
]
