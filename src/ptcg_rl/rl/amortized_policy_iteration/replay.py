"""Atomic compact NPZ shards for policy-iteration training evidence."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np
import numpy.typing as npt
import orjson

ReplayShardKind = Literal[
    "root", "protected_root", "candidate", "consequence", "target"
]

_MANIFEST_NAME = "manifest.json"
_MANIFEST_DOMAIN = b"ptcg-rl/amortized-policy-iteration-replay/v1\x00"
_STUDENT_KINDS = frozenset({"root", "candidate", "target"})
_FORBIDDEN_STUDENT_COLUMN_FRAGMENTS = (
    "particle",
    "world_id",
    "hidden",
    "opponent_hand_identity",
    "deck_order",
    "search_begin_input",
    "state_token",
)


class _ReplayPartManifest(TypedDict):
    index: int
    file: str
    rows: int
    sha256: str
    columns: list[str]


class _ReplayManifest(TypedDict):
    format: str
    kind: ReplayShardKind
    identity: dict[str, str | int]
    identity_fingerprint: str
    compression: bool
    parts: list[_ReplayPartManifest]
    committed_rows: int
    retired_parts: int


@dataclass(frozen=True, slots=True)
class ReplayIdentity:
    """Immutable provenance required to append or resume a replay directory."""

    run_id: str
    model_schema_fingerprint: str
    belief_fingerprint: str
    proposal_fingerprint: str
    engine_fingerprint: str
    schema_version: int

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("replay run_id must be non-empty")
        if self.schema_version <= 0:
            raise ValueError("replay schema_version must be positive")
        for name, value in (
            ("model", self.model_schema_fingerprint),
            ("belief", self.belief_fingerprint),
            ("proposal", self.proposal_fingerprint),
            ("engine", self.engine_fingerprint),
        ):
            if not _is_sha256(value):
                raise ValueError(f"replay {name} fingerprint must be SHA-256")

    @property
    def fingerprint(self) -> str:
        """Return a canonical digest binding every resume-sensitive identity."""
        return hashlib.sha256(
            _MANIFEST_DOMAIN + orjson.dumps(self.as_dict(), option=orjson.OPT_SORT_KEYS)
        ).hexdigest()

    def as_dict(self) -> dict[str, str | int]:
        """Return the small JSON manifest representation."""
        return {
            "run_id": self.run_id,
            "model_schema_fingerprint": self.model_schema_fingerprint,
            "belief_fingerprint": self.belief_fingerprint,
            "proposal_fingerprint": self.proposal_fingerprint,
            "engine_fingerprint": self.engine_fingerprint,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class CommittedReplayPart:
    """One manifest-visible complete NPZ shard."""

    index: int
    path: Path
    rows: int
    sha256: str
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplayWriteTiming:
    """Measured stages for one atomically committed replay part."""

    encode_seconds: float
    file_fsync_seconds: float
    fingerprint_seconds: float
    manifest_seconds: float
    retire_seconds: float
    total_seconds: float


class AtomicReplayShardWriter:
    """Publish complete compact parts before atomically advancing a manifest."""

    def __init__(
        self,
        directory: Path,
        *,
        kind: ReplayShardKind,
        identity: ReplayIdentity,
        compress: bool,
        max_committed_shards: int,
    ) -> None:
        """Create or resume one exact-identity shard stream."""
        if kind not in ("root", "protected_root", "candidate", "consequence", "target"):
            raise ValueError(f"unsupported replay shard kind: {kind}")
        if max_committed_shards <= 0:
            raise ValueError("max_committed_shards must be positive")
        self.directory = Path(directory)
        self.kind = kind
        self.identity = identity
        self.compress = bool(compress)
        self.max_committed_shards = int(max_committed_shards)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.directory / _MANIFEST_NAME
        self._manifest = self._load_or_initialize_manifest()
        self._last_timing: ReplayWriteTiming | None = None

    @property
    def committed_rows(self) -> int:
        """Return exact rows reachable through the current manifest."""
        return sum(int(part["rows"]) for part in self._manifest["parts"])

    @property
    def last_timing(self) -> ReplayWriteTiming | None:
        """Return stage timings for the most recently committed part."""
        return self._last_timing

    def append(
        self, columns: Mapping[str, npt.NDArray[np.generic]]
    ) -> CommittedReplayPart:
        """Atomically append one aligned, non-object NumPy column batch."""
        started_at = time.perf_counter()
        normalized, row_count = _validated_columns(columns, kind=self.kind)
        parts = list(self._manifest["parts"])
        index = 0 if not parts else int(parts[-1]["index"]) + 1
        final_path = self.directory / f"part-{index:08d}.npz"
        if final_path.exists():
            raise FileExistsError(
                f"uncommitted replay part already exists: {final_path}"
            )
        pending_path = self.directory / f".{final_path.name}.pending-{os.getpid()}"
        try:
            encode_started_at = time.perf_counter()
            with pending_path.open("xb") as destination:
                save = cast(
                    Any,
                    np.savez_compressed if self.compress else np.savez,
                )
                save(destination, **normalized)
                destination.flush()
                encode_seconds = time.perf_counter() - encode_started_at
                fsync_started_at = time.perf_counter()
                os.fsync(destination.fileno())
                file_fsync_seconds = time.perf_counter() - fsync_started_at
            os.replace(pending_path, final_path)
            _fsync_directory(self.directory)
            fingerprint_started_at = time.perf_counter()
            fingerprint = _file_sha256(final_path)
            fingerprint_seconds = time.perf_counter() - fingerprint_started_at
            part: _ReplayPartManifest = {
                "index": index,
                "file": final_path.name,
                "rows": row_count,
                "sha256": fingerprint,
                "columns": sorted(normalized),
            }
            parts.append(part)
            retired = parts[: -self.max_committed_shards]
            parts = parts[-self.max_committed_shards :]
            manifest = cast(_ReplayManifest, dict(self._manifest))
            manifest["parts"] = parts
            manifest["committed_rows"] = sum(int(item["rows"]) for item in parts)
            manifest["retired_parts"] = self._manifest["retired_parts"] + len(retired)
            manifest_started_at = time.perf_counter()
            self._publish_manifest(manifest)
            manifest_seconds = time.perf_counter() - manifest_started_at
            self._manifest = manifest
            retire_started_at = time.perf_counter()
            for retired_part in retired:
                retired_path = self.directory / str(retired_part["file"])
                if retired_path.exists():
                    retired_path.unlink()
            if retired:
                _fsync_directory(self.directory)
            retire_seconds = time.perf_counter() - retire_started_at
        except BaseException:
            if pending_path.exists():
                pending_path.unlink()
            raise
        self._last_timing = ReplayWriteTiming(
            encode_seconds=encode_seconds,
            file_fsync_seconds=file_fsync_seconds,
            fingerprint_seconds=fingerprint_seconds,
            manifest_seconds=manifest_seconds,
            retire_seconds=retire_seconds,
            total_seconds=time.perf_counter() - started_at,
        )
        return _part_from_manifest(self.directory, part)

    def committed_parts(self) -> tuple[CommittedReplayPart, ...]:
        """Return only shards referenced by the durable manifest."""
        return tuple(
            _part_from_manifest(self.directory, part)
            for part in self._manifest["parts"]
        )

    def iter_parts(self) -> Iterator[Mapping[str, npt.NDArray[np.generic]]]:
        """Stream one verified committed shard at a time."""
        for part in self.committed_parts():
            if _file_sha256(part.path) != part.sha256:
                raise RuntimeError(f"committed replay part is corrupt: {part.path}")
            with np.load(part.path, allow_pickle=False) as payload:
                names = tuple(sorted(payload.files))
                if names != part.columns:
                    raise RuntimeError("committed replay columns differ from manifest")
                yield {name: np.asarray(payload[name]) for name in names}

    def _load_or_initialize_manifest(self) -> _ReplayManifest:
        expected: _ReplayManifest = {
            "format": "compact_npz_parts_v1",
            "kind": self.kind,
            "identity": self.identity.as_dict(),
            "identity_fingerprint": self.identity.fingerprint,
            "compression": self.compress,
            "parts": [],
            "committed_rows": 0,
            "retired_parts": 0,
        }
        if not self._manifest_path.exists():
            self._publish_manifest(expected)
            self._quarantine_uncommitted_parts([])
            return expected
        raw_untyped = orjson.loads(self._manifest_path.read_bytes())
        if not isinstance(raw_untyped, dict):
            raise RuntimeError("replay manifest is not a JSON object")
        raw = cast(_ReplayManifest, raw_untyped)
        for key in ("format", "kind", "identity_fingerprint", "compression"):
            if raw.get(key) != expected[key]:
                raise RuntimeError(f"replay manifest {key} differs on resume")
        if raw.get("identity") != expected["identity"]:
            raise RuntimeError("replay manifest identity differs on resume")
        parts = raw.get("parts")
        if not isinstance(parts, list):
            raise RuntimeError("replay manifest parts must be a list")
        if parts and not isinstance(parts[0], dict):
            raise RuntimeError("replay manifest part is invalid")
        first_index = int(parts[0].get("index", -1)) if parts else 0
        for offset, part in enumerate(parts):
            if not isinstance(part, dict) or int(part.get("index", -1)) != (
                first_index + offset
            ):
                raise RuntimeError("replay manifest part indices are not contiguous")
            parsed = _part_from_manifest(self.directory, part)
            if not parsed.path.is_file() or _file_sha256(parsed.path) != parsed.sha256:
                raise RuntimeError("replay manifest references a missing/corrupt part")
        expected_rows = sum(int(part["rows"]) for part in parts)
        if int(raw.get("committed_rows", -1)) != expected_rows:
            raise RuntimeError("replay manifest committed row count differs")
        self._quarantine_uncommitted_parts(parts)
        return raw

    def _quarantine_uncommitted_parts(
        self,
        parts: Sequence[Mapping[str, object]],
    ) -> None:
        """Keep crash leftovers invisible while freeing the next part index."""
        committed_names = {str(part["file"]) for part in parts}
        leftovers = tuple(
            path
            for path in self.directory.iterdir()
            if (
                path.is_file()
                and (
                    (path.name.startswith("part-") and path.suffix == ".npz")
                    or ".pending-" in path.name
                )
                and path.name not in committed_names
            )
        )
        if not leftovers:
            return
        quarantine = self.directory / "uncommitted"
        quarantine.mkdir(exist_ok=True)
        for ordinal, path in enumerate(sorted(leftovers)):
            destination = quarantine / f"{path.name}.recovered-{os.getpid()}-{ordinal}"
            os.replace(path, destination)
        _fsync_directory(quarantine)
        _fsync_directory(self.directory)

    def _publish_manifest(self, manifest: Mapping[str, object]) -> None:
        pending = self.directory / f".{_MANIFEST_NAME}.pending-{os.getpid()}"
        payload = orjson.dumps(
            manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
        )
        try:
            with pending.open("xb") as destination:
                destination.write(payload)
                destination.write(b"\n")
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(pending, self._manifest_path)
            _fsync_directory(self.directory)
        finally:
            if pending.exists():
                pending.unlink()


class BoundedAsyncReplayShardWriter:
    """Overlap at most one replay part commit with foreground computation."""

    def __init__(self, writer: AtomicReplayShardWriter) -> None:
        """Wrap one synchronous writer with a bounded single-worker lane."""
        self._writer = writer
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"replay-{writer.kind}",
        )
        self._lock = threading.Lock()
        self._pending: Future[CommittedReplayPart] | None = None
        self._pending_rows = 0
        self._closed = False

    @property
    def committed_rows(self) -> int:
        """Return rows already made manifest-visible."""
        with self._lock:
            return self._writer.committed_rows

    @property
    def pending_rows(self) -> int:
        """Return logical rows owned by the in-flight part."""
        with self._lock:
            return self._pending_rows

    @property
    def last_timing(self) -> ReplayWriteTiming | None:
        """Return timings for the latest completed underlying append."""
        with self._lock:
            return self._writer.last_timing

    def append(
        self,
        columns: Mapping[str, npt.NDArray[np.generic]],
        *,
        copy: bool = True,
    ) -> Future[CommittedReplayPart]:
        """Submit owned columns after waiting for the previous part, if any."""
        normalized, row_count = _validated_columns(columns, kind=self._writer.kind)
        owned = (
            {name: values.copy(order="C") for name, values in normalized.items()}
            if copy
            else normalized
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("async replay writer is closed")
            if self._pending is not None:
                self._pending.result()
                self._pending = None
                self._pending_rows = 0
            future = self._executor.submit(self._writer.append, owned)
            self._pending = future
            self._pending_rows = row_count
            return future

    def barrier(self) -> CommittedReplayPart | None:
        """Wait for the in-flight part and surface worker failures."""
        with self._lock:
            if self._pending is None:
                return None
            part = self._pending.result()
            self._pending = None
            self._pending_rows = 0
            return part

    def committed_parts(self) -> tuple[CommittedReplayPart, ...]:
        """Barrier and return the underlying manifest-visible parts."""
        self.barrier()
        return self._writer.committed_parts()

    def iter_parts(self) -> Iterator[Mapping[str, npt.NDArray[np.generic]]]:
        """Barrier and stream verified parts from the underlying writer."""
        self.barrier()
        yield from self._writer.iter_parts()

    def close(self) -> None:
        """Commit pending work and stop the background thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.barrier()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)


def _validated_columns(
    columns: Mapping[str, npt.NDArray[np.generic]],
    *,
    kind: ReplayShardKind,
) -> tuple[dict[str, npt.NDArray[np.generic]], int]:
    if not columns:
        raise ValueError("replay part must contain columns")
    if set(columns) == {"record_payload", "record_offsets"}:
        return _validated_compact_records(columns)
    normalized: dict[str, npt.NDArray[np.generic]] = {}
    row_count: int | None = None
    for raw_name, raw_values in sorted(columns.items()):
        name = str(raw_name).strip()
        if not name or name != raw_name:
            raise ValueError("replay column names must be non-empty canonical strings")
        lowered = name.lower()
        if kind in _STUDENT_KINDS and any(
            fragment in lowered for fragment in _FORBIDDEN_STUDENT_COLUMN_FRAGMENTS
        ):
            raise ValueError(
                f"protected engine field cannot enter {kind} shard: {name}"
            )
        values = np.asarray(raw_values)
        if values.ndim <= 0:
            raise ValueError(f"replay column must have a row dimension: {name}")
        if values.dtype.hasobject:
            raise TypeError(f"replay column cannot use object dtype: {name}")
        if row_count is None:
            row_count = int(values.shape[0])
        elif int(values.shape[0]) != row_count:
            raise ValueError("replay columns must have aligned row counts")
        normalized[name] = np.ascontiguousarray(values)
    if row_count is None or row_count <= 0:
        raise ValueError("replay part must contain at least one row")
    return normalized, row_count


def _validated_compact_records(
    columns: Mapping[str, npt.NDArray[np.generic]],
) -> tuple[dict[str, npt.NDArray[np.generic]], int]:
    """Validate one unpadded concatenated MessagePack record segment."""
    payload = np.asarray(columns["record_payload"])
    offsets = np.asarray(columns["record_offsets"])
    if payload.ndim != 1 or payload.dtype != np.dtype(np.uint8):
        raise TypeError("compact replay payload must be one-dimensional uint8")
    if offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer):
        raise TypeError("compact replay offsets must be one-dimensional integers")
    if offsets.size < 2:
        raise ValueError("compact replay must contain at least one record")
    normalized_offsets = np.ascontiguousarray(offsets, dtype=np.uint64)
    if int(normalized_offsets[0]) != 0 or int(normalized_offsets[-1]) != payload.size:
        raise ValueError("compact replay offsets must span the complete payload")
    if np.any(normalized_offsets[1:] < normalized_offsets[:-1]):
        raise ValueError("compact replay offsets must be monotonic")
    return (
        {
            "record_payload": np.ascontiguousarray(payload),
            "record_offsets": normalized_offsets,
        },
        int(normalized_offsets.size - 1),
    )


def _part_from_manifest(
    directory: Path,
    raw: _ReplayPartManifest,
) -> CommittedReplayPart:
    index = int(raw["index"])
    name = str(raw["file"])
    expected_name = f"part-{index:08d}.npz"
    if name != expected_name:
        raise RuntimeError("replay part filename differs from its index")
    sha256 = str(raw["sha256"])
    if not _is_sha256(sha256):
        raise RuntimeError("replay part fingerprint is invalid")
    columns = raw["columns"]
    if not isinstance(columns, list) or any(
        not isinstance(item, str) for item in columns
    ):
        raise RuntimeError("replay part columns are invalid")
    return CommittedReplayPart(
        index=index,
        path=directory / name,
        rows=int(raw["rows"]),
        sha256=sha256,
        columns=tuple(sorted(columns)),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "AtomicReplayShardWriter",
    "BoundedAsyncReplayShardWriter",
    "CommittedReplayPart",
    "ReplayIdentity",
    "ReplayShardKind",
    "ReplayWriteTiming",
]
