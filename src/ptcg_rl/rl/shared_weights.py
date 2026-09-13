"""Shared-memory policy weight publication for async rollout serving.

The default transport keeps a fixed pair of raw tensor slots.  Publications
copy exact-dtype CPU tensors into the inactive slot and atomically advertise
that slot only after every tensor is complete.  This removes the model-sized
``torch.save``/``bytes``/``torch.load`` round trip from the hot path.  A small
sequence header detects a reader racing a later overwrite of the same slot.

The legacy serialized transport and manifests remain readable for rolling
upgrades and old run recovery.
"""

from __future__ import annotations

import io
import json
import time
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.model_publication import (
    PreparedModelState,
    prepare_model_state,
    refresh_prepared_model_state,
    validate_model_fingerprint,
)
from ptcg_rl.rl.shared_tensor_transport import (
    TENSOR_TRANSPORT,
    SharedTensorReadSpec,
    SharedTensorSlotPublisher,
    atomic_write_bytes,
    load_shared_tensor_slot,
)

_TENSOR_TRANSPORT = TENSOR_TRANSPORT
_LEGACY_TRANSPORT = "torch_save_v1"


class SharedMemoryWeightPublisherConfig(BaseModel):
    """Config for shared-memory policy weight publication."""

    model_config = ConfigDict(extra="forbid")

    keep_last: int = 2
    filename_prefix: str = "policy_shm"
    transport: Literal["tensor_slots", "legacy_serialized"] = "tensor_slots"
    tensor_slot_count: int = 2

    @field_validator("keep_last")
    @classmethod
    def valid_keep_last(cls, value: int) -> int:
        """Reject invalid retention counts."""
        if value <= 0:
            raise ValueError("keep_last must be positive")
        return value

    @field_validator("tensor_slot_count")
    @classmethod
    def valid_tensor_slot_count(cls, value: int) -> int:
        """Require at least an active and an inactive tensor slot."""
        if value < 2:
            raise ValueError("tensor_slot_count must be at least two")
        return value

    @field_validator("filename_prefix")
    @classmethod
    def valid_filename_prefix(cls, value: str) -> str:
        """Reject empty or path-like shared-memory prefixes."""
        cleaned = value.strip()
        if not cleaned or "/" in cleaned or "\\" in cleaned:
            raise ValueError("filename_prefix must be a single filename segment")
        return cleaned


@dataclass(frozen=True, slots=True)
class SharedWeightPublicationTiming:
    """Measured stages for the most recent hot publication."""

    prepare_copy_seconds: float
    fingerprint_seconds: float
    slot_copy_seconds: float
    manifest_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class SharedWeightLoadTiming:
    """Measured stages for the most recent shared-weight load."""

    manifest_seconds: float
    tensor_copy_seconds: float
    fingerprint_seconds: float
    model_load_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class SharedMemoryWeights:
    """Metadata for one shared-memory policy publication."""

    version: int
    name: str
    size_bytes: int
    latest_path: Path
    published_at: str
    metadata: Mapping[str, Any]
    model_fingerprint: str | None = None
    transport: str = _LEGACY_TRANSPORT
    sequence: int | None = None
    layout_path: Path | None = None
    layout_fingerprint: str | None = None


class SharedMemoryWeightPublisher:
    """Publish model snapshots through reusable exact-dtype shared tensors."""

    def __init__(
        self,
        directory: Path,
        config: SharedMemoryWeightPublisherConfig | None = None,
    ) -> None:
        """Initialize the shared-memory publisher root."""
        self.directory = Path(directory)
        self.config = config or SharedMemoryWeightPublisherConfig()
        self._segments: dict[str, shared_memory.SharedMemory] = {}
        self._published_names: list[str] = []
        self._prepared_state: PreparedModelState | None = None
        self._serialization_buffer = io.BytesIO()
        self._tensor_publisher = SharedTensorSlotPublisher(
            self.directory,
            filename_prefix=self.config.filename_prefix,
            slot_count=self.config.tensor_slot_count,
        )
        self._last_timing: SharedWeightPublicationTiming | None = None

    @property
    def latest_path(self) -> Path:
        """Return the latest shared-memory manifest path."""
        return self.directory / "shared_latest.json"

    @property
    def last_timing(self) -> SharedWeightPublicationTiming | None:
        """Return timings for the most recent completed publication."""
        return self._last_timing

    def prepare_state(self, state_dict: Mapping[str, Any]) -> PreparedModelState:
        """Refresh and retain one reusable model-sized CPU staging snapshot."""
        self._prepared_state = refresh_prepared_model_state(
            state_dict,
            self._prepared_state,
        )
        return self._prepared_state

    def publish(
        self,
        state_dict: Mapping[str, Any] | PreparedModelState,
        *,
        version: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> SharedMemoryWeights:
        """Publish one complete version and atomically advance the manifest."""
        if version < 0:
            raise ValueError("version must be non-negative")
        started_at = time.perf_counter()
        self.directory.mkdir(parents=True, exist_ok=True)
        prepared = prepare_model_state(state_dict)
        if self.config.transport == "legacy_serialized":
            weights, slot_copy_seconds, manifest_seconds = self._publish_legacy(
                prepared,
                version=version,
                metadata=metadata,
            )
        else:
            weights, slot_copy_seconds, manifest_seconds = self._publish_tensors(
                prepared,
                version=version,
                metadata=metadata,
            )
        self._last_timing = SharedWeightPublicationTiming(
            prepare_copy_seconds=prepared.copy_seconds,
            fingerprint_seconds=prepared.fingerprint_seconds,
            slot_copy_seconds=slot_copy_seconds,
            manifest_seconds=manifest_seconds,
            total_seconds=time.perf_counter() - started_at,
        )
        return weights

    def close(self) -> None:
        """Close and unlink all shared-memory slots owned by this publisher."""
        owned_names = set(self._segments) | set(self._tensor_publisher.owned_names)
        for name in tuple(self._segments):
            self._close_segment(name, unlink=True)
        latest = read_latest_shared_memory_weights(self.directory)
        if latest is not None and latest.name in owned_names:
            self.latest_path.unlink(missing_ok=True)
        self._tensor_publisher.close()
        self._prepared_state = None
        self._serialization_buffer.close()

    def _publish_tensors(
        self,
        prepared: PreparedModelState,
        *,
        version: int,
        metadata: Mapping[str, Any] | None,
    ) -> tuple[SharedMemoryWeights, float, float]:
        slot = self._tensor_publisher.write(prepared.state_dict, version=version)
        slot_copy_seconds = slot.copy_seconds

        published_at = datetime.now(UTC).isoformat()
        record = {
            "format": _TENSOR_TRANSPORT,
            "version": version,
            "name": slot.name,
            "size_bytes": slot.size_bytes,
            "published_at": published_at,
            "model_fingerprint": prepared.model_fingerprint,
            "metadata": dict(metadata or {}),
            "sequence": slot.sequence,
            "layout_file": slot.layout_path.name,
            "layout_fingerprint": slot.layout_fingerprint,
        }
        manifest_started_at = time.perf_counter()
        _atomic_write_json(self.latest_path, record)
        manifest_seconds = time.perf_counter() - manifest_started_at
        self._tensor_publisher.commit()
        return (
            SharedMemoryWeights(
                version=version,
                name=slot.name,
                size_bytes=slot.size_bytes,
                latest_path=self.latest_path,
                published_at=published_at,
                metadata=metadata or {},
                model_fingerprint=prepared.model_fingerprint,
                transport=_TENSOR_TRANSPORT,
                sequence=slot.sequence,
                layout_path=slot.layout_path,
                layout_fingerprint=slot.layout_fingerprint,
            ),
            slot_copy_seconds,
            manifest_seconds,
        )

    def _publish_legacy(
        self,
        prepared: PreparedModelState,
        *,
        version: int,
        metadata: Mapping[str, Any] | None,
    ) -> tuple[SharedMemoryWeights, float, float]:
        copy_started_at = time.perf_counter()
        payload_size, payload = _serialize_state_dict_into_buffer(
            prepared.state_dict,
            self._serialization_buffer,
        )
        self._prune_before_publish()
        name = f"{self.config.filename_prefix}_{version}_{uuid.uuid4().hex}"
        try:
            segment = shared_memory.SharedMemory(
                name=name,
                create=True,
                size=payload_size,
            )
        except BaseException:
            payload.release()
            raise
        buffer = segment.buf
        try:
            if buffer is None:
                raise RuntimeError("shared-memory segment has no writable buffer")
            buffer[:payload_size] = payload[:payload_size]
        except BaseException:
            segment.close()
            with suppress(FileNotFoundError):
                segment.unlink()
            raise
        finally:
            payload.release()
            del buffer
        slot_copy_seconds = time.perf_counter() - copy_started_at
        self._segments[name] = segment
        self._published_names.append(name)

        published_at = datetime.now(UTC).isoformat()
        record = {
            "format": _LEGACY_TRANSPORT,
            "version": version,
            "name": name,
            "size_bytes": payload_size,
            "published_at": published_at,
            "model_fingerprint": prepared.model_fingerprint,
            "metadata": dict(metadata or {}),
        }
        manifest_started_at = time.perf_counter()
        _atomic_write_json(self.latest_path, record)
        manifest_seconds = time.perf_counter() - manifest_started_at
        self._prune_old_segments()
        return (
            SharedMemoryWeights(
                version=version,
                name=name,
                size_bytes=payload_size,
                latest_path=self.latest_path,
                published_at=published_at,
                metadata=metadata or {},
                model_fingerprint=prepared.model_fingerprint,
                transport=_LEGACY_TRANSPORT,
            ),
            slot_copy_seconds,
            manifest_seconds,
        )

    def _prune_old_segments(self) -> None:
        old_names = self._published_names[: -self.config.keep_last]
        self._published_names = self._published_names[-self.config.keep_last :]
        for name in old_names:
            self._close_segment(name, unlink=True)

    def _prune_before_publish(self) -> None:
        """Reserve room without invalidating the manifest-visible segment."""
        retained_count = max(1, self.config.keep_last - 1)
        old_names = self._published_names[:-retained_count]
        self._published_names = self._published_names[-retained_count:]
        for name in old_names:
            self._close_segment(name, unlink=True)

    def _close_segment(self, name: str, *, unlink: bool) -> None:
        segment = self._segments.pop(name, None)
        if segment is None:
            return
        segment.close()
        if unlink:
            with suppress(FileNotFoundError):
                segment.unlink()


class SharedMemoryWeightLoader:
    """Poll shared-memory policy weights and load newer versions into a model."""

    def __init__(
        self,
        directory: Path,
        *,
        map_location: str = "cpu",
        strict: bool = True,
        verify_fingerprint: bool = False,
    ) -> None:
        """Initialize loader state.

        The tensor-slot seqlock and immutable layout are sufficient for hot
        serving by default. Set ``verify_fingerprint`` at durable/audit
        boundaries when a second full model hash is required.
        """
        self.directory = Path(directory)
        self.map_location = map_location
        self.strict = bool(strict)
        self.verify_fingerprint = bool(verify_fingerprint)
        self._loaded_version: int | None = None
        self._last_timing: SharedWeightLoadTiming | None = None

    @property
    def loaded_version(self) -> int | None:
        """Return the newest loaded shared-memory version."""
        return self._loaded_version

    @property
    def last_timing(self) -> SharedWeightLoadTiming | None:
        """Return timings for the most recent completed load."""
        return self._last_timing

    def latest_version(self) -> int | None:
        """Return the latest advertised shared-memory version, if present."""
        latest = read_latest_shared_memory_weights(self.directory)
        return None if latest is None else latest.version

    def poll(self, model: torch.nn.Module) -> SharedMemoryWeights | None:
        """Load the latest shared-memory weights if the version changed."""
        started_at = time.perf_counter()
        loaded = self._poll_state_dict_with_timing()
        if loaded is None:
            return None
        latest, state_dict, manifest_seconds, copy_seconds, fingerprint_seconds = loaded
        model_load_started_at = time.perf_counter()
        model.load_state_dict(
            _state_dict_from_checkpoint(state_dict), strict=self.strict
        )
        model_load_seconds = time.perf_counter() - model_load_started_at
        self.mark_loaded(latest)
        self._last_timing = SharedWeightLoadTiming(
            manifest_seconds=manifest_seconds,
            tensor_copy_seconds=copy_seconds,
            fingerprint_seconds=fingerprint_seconds,
            model_load_seconds=model_load_seconds,
            total_seconds=time.perf_counter() - started_at,
        )
        return latest

    def poll_state_dict(
        self,
    ) -> tuple[SharedMemoryWeights, Mapping[str, Any]] | None:
        """Read and validate a newer immutable state without mutating a model."""
        started_at = time.perf_counter()
        loaded = self._poll_state_dict_with_timing()
        if loaded is None:
            return None
        latest, state_dict, manifest_seconds, copy_seconds, fingerprint_seconds = loaded
        self._last_timing = SharedWeightLoadTiming(
            manifest_seconds=manifest_seconds,
            tensor_copy_seconds=copy_seconds,
            fingerprint_seconds=fingerprint_seconds,
            model_load_seconds=0.0,
            total_seconds=time.perf_counter() - started_at,
        )
        return (latest, state_dict)

    def _poll_state_dict_with_timing(
        self,
    ) -> (
        tuple[
            SharedMemoryWeights,
            Mapping[str, Any],
            float,
            float,
            float,
        ]
        | None
    ):
        manifest_started_at = time.perf_counter()
        latest = read_latest_shared_memory_weights(self.directory)
        manifest_seconds = time.perf_counter() - manifest_started_at
        if latest is None or (
            self._loaded_version is not None and latest.version <= self._loaded_version
        ):
            return None
        state_dict, copy_seconds, fingerprint_seconds = (
            _load_shared_memory_state_dict_with_timing(
                latest,
                map_location=self.map_location,
                verify_fingerprint=self.verify_fingerprint,
            )
        )
        return (
            latest,
            state_dict,
            manifest_seconds,
            copy_seconds,
            fingerprint_seconds,
        )

    def mark_loaded(self, weights: SharedMemoryWeights) -> None:
        """Advance the poll cursor after a model snapshot was atomically published."""
        if self._loaded_version is not None and weights.version <= self._loaded_version:
            raise ValueError(
                "shared weight loader requires a strictly newer publication"
            )
        self._loaded_version = int(weights.version)


def read_latest_shared_memory_weights(directory: Path) -> SharedMemoryWeights | None:
    """Return latest shared-memory weight metadata, if present."""
    latest_path = Path(directory) / "shared_latest.json"
    if not latest_path.exists():
        return None
    record = json.loads(latest_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"shared weight record must be a JSON object: {latest_path}")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("shared weight record metadata must be an object")
    transport = str(record.get("format", _LEGACY_TRANSPORT))
    if transport not in (_TENSOR_TRANSPORT, _LEGACY_TRANSPORT):
        raise ValueError(f"unsupported shared weight transport: {transport}")
    layout_path: Path | None = None
    layout_fingerprint: str | None = None
    sequence: int | None = None
    if transport == _TENSOR_TRANSPORT:
        layout_file = record.get("layout_file")
        if not isinstance(layout_file, str) or Path(layout_file).name != layout_file:
            raise ValueError("shared tensor layout_file must be a filename")
        layout_path = latest_path.parent / layout_file
        raw_layout_fingerprint = record.get("layout_fingerprint")
        if not isinstance(raw_layout_fingerprint, str):
            raise ValueError("shared tensor layout fingerprint is missing")
        layout_fingerprint = validate_model_fingerprint(
            raw_layout_fingerprint,
            name="layout_fingerprint",
        )
        sequence = int(record["sequence"])
        if sequence <= 0 or sequence % 2:
            raise ValueError("shared tensor sequence must be positive and even")
    return SharedMemoryWeights(
        version=int(record["version"]),
        name=str(record["name"]),
        size_bytes=int(record["size_bytes"]),
        latest_path=latest_path,
        published_at=str(record["published_at"]),
        metadata=metadata,
        model_fingerprint=_optional_model_fingerprint(record),
        transport=transport,
        sequence=sequence,
        layout_path=layout_path,
        layout_fingerprint=layout_fingerprint,
    )


def load_shared_memory_state_dict(
    weights: SharedMemoryWeights,
    *,
    map_location: str | torch.device = "cpu",
    verify_fingerprint: bool = True,
) -> Mapping[str, Any]:
    """Load a state dict, optionally verifying its full content identity."""
    state_dict, _copy_seconds, _fingerprint_seconds = (
        _load_shared_memory_state_dict_with_timing(
            weights,
            map_location=map_location,
            verify_fingerprint=verify_fingerprint,
        )
    )
    return state_dict


def _load_shared_memory_state_dict_with_timing(
    weights: SharedMemoryWeights,
    *,
    map_location: str | torch.device,
    verify_fingerprint: bool,
) -> tuple[Mapping[str, Any], float, float]:
    if weights.transport == _TENSOR_TRANSPORT:
        loaded, copy_seconds = _load_tensor_slot(
            weights,
            map_location=map_location,
        )
    else:
        copy_started_at = time.perf_counter()
        loaded = _load_legacy_shared_state(weights, map_location=map_location)
        copy_seconds = time.perf_counter() - copy_started_at
    fingerprint_seconds = 0.0
    if verify_fingerprint and weights.model_fingerprint is not None:
        fingerprint_started_at = time.perf_counter()
        state_dict = _state_dict_from_checkpoint(loaded)
        actual_fingerprint = canonical_model_state_fingerprint(state_dict)
        fingerprint_seconds = time.perf_counter() - fingerprint_started_at
        if actual_fingerprint != weights.model_fingerprint:
            raise ValueError("shared weight payload differs from its model fingerprint")
    return loaded, copy_seconds, fingerprint_seconds


def _load_tensor_slot(
    weights: SharedMemoryWeights,
    *,
    map_location: str | torch.device,
) -> tuple[Mapping[str, torch.Tensor], float]:
    if (
        weights.layout_path is None
        or weights.layout_fingerprint is None
        or weights.sequence is None
    ):
        raise ValueError("shared tensor publication has no complete layout metadata")
    return load_shared_tensor_slot(
        SharedTensorReadSpec(
            name=weights.name,
            size_bytes=weights.size_bytes,
            version=weights.version,
            sequence=weights.sequence,
            layout_path=weights.layout_path,
            layout_fingerprint=weights.layout_fingerprint,
        ),
        map_location=map_location,
    )


def _load_legacy_shared_state(
    weights: SharedMemoryWeights,
    *,
    map_location: str | torch.device,
) -> Mapping[str, Any]:
    segment = shared_memory.SharedMemory(name=weights.name, create=False)
    try:
        buffer = segment.buf
        if buffer is None:
            raise RuntimeError("shared-memory segment has no readable buffer")
        payload = buffer[: weights.size_bytes].tobytes()
        del buffer
    finally:
        segment.close()
    loaded = torch.load(io.BytesIO(payload), map_location=map_location)
    if not isinstance(loaded, Mapping):
        raise TypeError("shared weights must contain a state_dict mapping")
    return loaded


def _serialize_state_dict_into_buffer(
    state_dict: Mapping[str, Any],
    buffer: io.BytesIO,
) -> tuple[int, memoryview]:
    """Serialize into reusable storage for legacy rolling-upgrade readers."""
    buffer.seek(0)
    torch.save(dict(state_dict), buffer)
    size = buffer.tell()
    buffer.truncate(size)
    return (size, buffer.getbuffer())


def _optional_model_fingerprint(record: Mapping[str, Any]) -> str | None:
    raw = record.get("model_fingerprint")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("shared weight model_fingerprint must be a string")
    return validate_model_fingerprint(raw)


def _state_dict_from_checkpoint(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    atomic_write_bytes(path, payload)


__all__ = [
    "SharedMemoryWeightLoader",
    "SharedMemoryWeightPublisher",
    "SharedMemoryWeightPublisherConfig",
    "SharedMemoryWeights",
    "SharedWeightLoadTiming",
    "SharedWeightPublicationTiming",
    "load_shared_memory_state_dict",
    "read_latest_shared_memory_weights",
]
