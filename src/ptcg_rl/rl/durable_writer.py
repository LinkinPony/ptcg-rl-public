"""Bounded asynchronous publication of immutable durable artifacts."""

from __future__ import annotations

import copy
import hashlib
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self, cast

import orjson
import torch

_MANIFEST_FORMAT = "bounded_async_durable_artifacts_v1"


class AsyncWriteBusyError(RuntimeError):
    """Raised when a non-blocking submission finds an artifact in flight."""


@dataclass(frozen=True, slots=True)
class DurableWriteTiming:
    """Fine-grained wall time for one durable artifact publication."""

    write_seconds: float
    file_fsync_seconds: float
    fingerprint_seconds: float
    publish_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class DurableArtifact:
    """One complete file made visible through the durable manifest."""

    path: Path
    size_bytes: int
    published_at: str
    sha256: str | None
    metadata: Mapping[str, Any]
    timing: DurableWriteTiming


class BoundedAsyncDurableWriter:
    """Overlap one immutable artifact write with foreground computation.

    At most one artifact is running. A second blocking ``submit`` waits for the
    first; a non-blocking submission raises :class:`AsyncWriteBusyError`.
    ``barrier`` surfaces worker failures. ``close`` is a clean-shutdown barrier,
    so a successful return means the final artifact and manifest are durable.
    """

    def __init__(
        self,
        directory: Path,
        *,
        manifest_name: str = "durable_manifest.json",
        manifest_history: int = 32,
    ) -> None:
        """Initialize a single-worker durable publication lane."""
        if Path(manifest_name).name != manifest_name:
            raise ValueError("manifest_name must be one filename segment")
        if manifest_history <= 0:
            raise ValueError("manifest_history must be positive")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / manifest_name
        self.manifest_history = int(manifest_history)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="durable-writer",
        )
        self._lock = threading.Lock()
        self._pending: Future[DurableArtifact] | None = None
        self._closed = False
        self._last_result: DurableArtifact | None = None

    @property
    def busy(self) -> bool:
        """Return whether an artifact has not crossed its barrier yet."""
        with self._lock:
            return self._pending is not None and not self._pending.done()

    @property
    def last_result(self) -> DurableArtifact | None:
        """Return the last artifact observed by a barrier."""
        with self._lock:
            return self._last_result

    def submit(
        self,
        path: Path,
        write: Callable[[Path], None],
        *,
        metadata: Mapping[str, Any] | None = None,
        compute_sha256: bool = True,
        block: bool = True,
        overwrite: bool = False,
    ) -> Future[DurableArtifact]:
        """Submit a file producer after applying bounded backpressure."""
        final_path = self._validated_target(path)
        with self._lock:
            if self._closed:
                raise RuntimeError("durable writer is closed")
            previous = self._pending
            if previous is not None and not previous.done() and not block:
                raise AsyncWriteBusyError("one durable artifact is already in flight")
            if previous is not None:
                self._last_result = previous.result()
                self._pending = None
            if final_path.exists() and not overwrite:
                raise FileExistsError(f"durable artifact already exists: {final_path}")
            future = self._executor.submit(
                self._write_one,
                final_path,
                write,
                dict(metadata or {}),
                bool(compute_sha256),
                bool(overwrite),
            )
            self._pending = future
            return future

    def barrier(self) -> DurableArtifact | None:
        """Wait for the in-flight write and surface any worker failure."""
        with self._lock:
            pending = self._pending
            if pending is None:
                return self._last_result
            result = pending.result()
            self._last_result = result
            self._pending = None
            return result

    def flush(self) -> DurableArtifact | None:
        """Alias for the explicit durability barrier."""
        return self.barrier()

    def close(self) -> None:
        """Commit pending work, surface failures, and stop the worker thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.barrier()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        self.close()

    def _validated_target(self, path: Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.directory / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.directory) or resolved == self.directory:
            raise ValueError("durable artifact must be a file below writer directory")
        return resolved

    def _write_one(
        self,
        final_path: Path,
        write: Callable[[Path], None],
        metadata: dict[str, Any],
        compute_sha256: bool,
        overwrite: bool,
    ) -> DurableArtifact:
        started_at = time.perf_counter()
        final_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path = final_path.parent / (
            f".{final_path.name}.pending-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            write_started_at = time.perf_counter()
            write(pending_path)
            write_seconds = time.perf_counter() - write_started_at
            if not pending_path.is_file():
                raise RuntimeError("durable writer callback did not create its file")

            fsync_started_at = time.perf_counter()
            with pending_path.open("rb") as source:
                os.fsync(source.fileno())
            file_fsync_seconds = time.perf_counter() - fsync_started_at

            fingerprint: str | None = None
            fingerprint_seconds = 0.0
            if compute_sha256:
                fingerprint_started_at = time.perf_counter()
                fingerprint = _file_sha256(pending_path)
                fingerprint_seconds = time.perf_counter() - fingerprint_started_at

            if final_path.exists() and not overwrite:
                raise FileExistsError(f"durable artifact already exists: {final_path}")
            os.replace(pending_path, final_path)
            _fsync_directory(final_path.parent)
            size_bytes = final_path.stat().st_size
            published_at = datetime.now(UTC).isoformat()
            publish_started_at = time.perf_counter()
            self._publish_manifest(
                path=final_path,
                size_bytes=size_bytes,
                published_at=published_at,
                fingerprint=fingerprint,
                metadata=metadata,
            )
            publish_seconds = time.perf_counter() - publish_started_at
            timing = DurableWriteTiming(
                write_seconds=write_seconds,
                file_fsync_seconds=file_fsync_seconds,
                fingerprint_seconds=fingerprint_seconds,
                publish_seconds=publish_seconds,
                total_seconds=time.perf_counter() - started_at,
            )
            return DurableArtifact(
                path=final_path,
                size_bytes=size_bytes,
                published_at=published_at,
                sha256=fingerprint,
                metadata=metadata,
                timing=timing,
            )
        finally:
            pending_path.unlink(missing_ok=True)

    def _publish_manifest(
        self,
        *,
        path: Path,
        size_bytes: int,
        published_at: str,
        fingerprint: str | None,
        metadata: Mapping[str, Any],
    ) -> None:
        records: list[dict[str, Any]] = []
        if self.manifest_path.exists():
            raw = orjson.loads(self.manifest_path.read_bytes())
            if not isinstance(raw, dict) or raw.get("format") != _MANIFEST_FORMAT:
                raise RuntimeError("durable artifact manifest format is invalid")
            raw_records = raw.get("artifacts")
            if not isinstance(raw_records, list):
                raise RuntimeError("durable artifact manifest records are invalid")
            records = [cast(dict[str, Any], item) for item in raw_records]
        record = {
            "path": str(path.relative_to(self.directory)),
            "size_bytes": size_bytes,
            "published_at": published_at,
            "sha256": fingerprint,
            "metadata": dict(metadata),
        }
        records.append(record)
        records = records[-self.manifest_history :]
        manifest = {
            "format": _MANIFEST_FORMAT,
            "latest": record,
            "artifacts": records,
        }
        payload = orjson.dumps(
            manifest,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
        pending = self.directory / (
            f".{self.manifest_path.name}.pending-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            with pending.open("xb") as destination:
                destination.write(payload)
                destination.write(b"\n")
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(pending, self.manifest_path)
            _fsync_directory(self.directory)
        finally:
            pending.unlink(missing_ok=True)


def freeze_durable_value(value: Any) -> Any:
    """Detach mutable tensor trees into an immutable CPU-owned snapshot.

    This function is intentionally synchronous. Callers must cross this
    boundary before handing a payload to a background serializer; otherwise an
    optimizer step can mutate storage while ``torch.save`` is reading it.
    """
    if isinstance(value, torch.Tensor):
        frozen = value.detach().to(device="cpu", copy=True)
        return frozen.contiguous() if frozen.layout == torch.strided else frozen
    if isinstance(value, Mapping):
        return {
            copy.deepcopy(key): freeze_durable_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [freeze_durable_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(freeze_durable_value(item) for item in value)
    if isinstance(value, set):
        return {freeze_durable_value(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(freeze_durable_value(item) for item in value)
    return copy.deepcopy(value)


def adopt_frozen_durable_value(value: Any) -> Any:
    """Transfer an already CPU-frozen tensor tree without copying storage.

    The caller must give up every mutable reference after this call. Container
    shells are rebuilt so later structural edits cannot affect serialization;
    CPU tensor storage is intentionally retained to avoid a second model-sized
    copy.
    """
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("owned durable tensors must already be on CPU")
        return value.detach()
    if isinstance(value, Mapping):
        return {
            copy.deepcopy(key): adopt_frozen_durable_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [adopt_frozen_durable_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(adopt_frozen_durable_value(item) for item in value)
    if isinstance(value, set):
        return {adopt_frozen_durable_value(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(adopt_frozen_durable_value(item) for item in value)
    return copy.deepcopy(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "AsyncWriteBusyError",
    "BoundedAsyncDurableWriter",
    "DurableArtifact",
    "DurableWriteTiming",
    "adopt_frozen_durable_value",
    "freeze_durable_value",
]
