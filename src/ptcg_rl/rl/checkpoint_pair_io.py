"""Low-level durable I/O for exact checkpoint-pair publication."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO, cast

import torch


def publish_torch_file(path: Path, payload: Any) -> tuple[int, str]:
    """Serialize, hash, fsync, and atomically publish one tensor file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = _pending_path(path)
    try:
        with pending.open("xb") as raw:
            destination = _HashingWriter(raw)
            torch.save(payload, cast(BinaryIO, destination))
            destination.flush()
            os.fsync(raw.fileno())
            size_bytes = destination.bytes_written
            sha256 = destination.hexdigest
        _publish_immutable_with_fsync(pending, path)
        return size_bytes, sha256
    finally:
        pending.unlink(missing_ok=True)


def publish_bytes_file(path: Path, payload: bytes) -> None:
    """Publish one immutable metadata file."""
    atomic_write_bytes(path, payload, overwrite=False)


def atomic_write_bytes(path: Path, payload: bytes, *, overwrite: bool) -> None:
    """Fsync and atomically replace one small pointer or manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = _pending_path(path)
    try:
        with pending.open("xb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        if overwrite:
            os.replace(pending, path)
            fsync_directory(path.parent)
        else:
            _publish_immutable_with_fsync(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def json_payload(record: Mapping[str, Any]) -> bytes:
    """Encode a deterministic, human-readable-compatible pointer record."""
    return (
        json.dumps(record, sort_keys=True, default=str, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def fsync_directory(path: Path) -> None:
    """Persist directory entry changes."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def pair_manifest_version(path: Path) -> int | None:
    """Parse a numbered exact-pair manifest filename."""
    raw = path.stem.removeprefix("checkpoint_pair_v")
    return int(raw) if raw.isdigit() else None


def _pending_path(path: Path) -> Path:
    return path.parent / f".{path.name}.pending-{os.getpid()}-{uuid.uuid4().hex}"


def _publish_no_replace(pending: Path, path: Path) -> None:
    """Atomically publish ``pending`` without replacing an immutable target."""
    try:
        os.link(pending, path)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable checkpoint already exists: {path}") from exc
    pending.unlink()


def _publish_immutable_with_fsync(pending: Path, path: Path) -> None:
    """Publish and clean only this writer's inode on pre-commit failure."""
    pending_stat = pending.stat()
    pending_identity = (pending_stat.st_dev, pending_stat.st_ino)
    try:
        _publish_no_replace(pending, path)
        fsync_directory(path.parent)
    except BaseException:
        _remove_owned_publication(
            path,
            expected_identity=pending_identity,
        )
        raise


def _remove_owned_publication(
    path: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Best-effort remove a failed writer's hard-linked immutable target."""
    try:
        published_stat = path.stat()
    except FileNotFoundError:
        return
    if (published_stat.st_dev, published_stat.st_ino) != expected_identity:
        return
    path.unlink()
    # Preserve the original publication error if durability of the cleanup
    # cannot itself be confirmed.
    with suppress(OSError):
        fsync_directory(path.parent)


class _HashingWriter:
    """Fingerprint torch's sequential byte stream without rereading the file."""

    def __init__(self, destination: BinaryIO) -> None:
        self._destination = destination
        self._digest = hashlib.sha256()
        self.bytes_written = 0

    def write(self, payload: bytes) -> int:
        """Write one serialized chunk and include exactly those bytes in SHA-256."""
        written = self._destination.write(payload)
        if written != len(payload):
            raise OSError("short checkpoint write")
        self._digest.update(payload)
        self.bytes_written += written
        return written

    def flush(self) -> None:
        """Flush Python buffering before the caller fsyncs the descriptor."""
        self._destination.flush()

    @property
    def hexdigest(self) -> str:
        """Return the digest for every byte accepted by the destination."""
        return self._digest.hexdigest()


__all__ = [
    "atomic_write_bytes",
    "fsync_directory",
    "json_payload",
    "pair_manifest_version",
    "publish_bytes_file",
    "publish_torch_file",
]
