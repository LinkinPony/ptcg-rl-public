"""Non-blocking immutable learner metrics bound to checkpoint pairs."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.rl.performance_state import atomic_write_json

LEARNER_METRIC_SCHEMA_VERSION = 1
_FINGERPRINT_LENGTH = 64


class LearnerMetricRecord(BaseModel):
    """One compact learner observation attached to an immutable checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["checkpoint-learner-metric-v1"] = "checkpoint-learner-metric-v1"
    recorded_at_utc: str
    run_version: str
    update_index: int = Field(ge=0)
    optimizer_step_index: int = Field(ge=0)
    checkpoint_version: int = Field(ge=0)
    pair_manifest_sha256: str
    policy_sha256: str
    learner_state_sha256: str
    decisions: int = Field(gt=0)
    fragments_seen: int = Field(gt=0)
    fragments_stale: int = Field(ge=0)
    loss: float
    policy_loss: float
    value_loss: float
    belief_loss: float
    entropy: float
    ratio_mean: float
    approximate_kl: float
    clip_fraction: float = Field(ge=0.0, le=1.0)
    gradient_norm: float = Field(ge=0.0)
    learning_rate: float = Field(ge=0.0)
    kept_decisions_per_second: float = Field(ge=0.0)
    collection_seconds: float = Field(ge=0.0)
    learner_seconds: float = Field(ge=0.0)
    checkpoint_seconds: float = Field(ge=0.0)
    total_seconds: float = Field(ge=0.0)
    cuda_peak_allocated_bytes: int | None = Field(default=None, ge=0)
    cuda_peak_reserved_bytes: int | None = Field(default=None, ge=0)
    cuda_allocation_retries: int | None = Field(default=None, ge=0)
    cuda_ooms: int | None = Field(default=None, ge=0)

    @field_validator(
        "pair_manifest_sha256",
        "policy_sha256",
        "learner_state_sha256",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require a lowercase SHA-256 artifact identity."""
        normalized = value.strip().lower()
        if len(normalized) != _FINGERPRINT_LENGTH or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("learner metric artifact identities must be SHA-256")
        return normalized

    @model_validator(mode="after")
    def coherent_checkpoint(self) -> Self:
        """A checkpoint metric may only describe its own update boundary."""
        if self.checkpoint_version != self.update_index + 1:
            raise ValueError(
                "checkpoint metric version must immediately follow update index"
            )
        if self.fragments_stale > self.fragments_seen:
            raise ValueError("stale fragments exceed observed fragments")
        return self


class LearnerMetricWriterStatus(BaseModel):
    """Non-gating telemetry for the asynchronous metric writer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    queued_records: int = Field(ge=0)
    committed_records: int = Field(ge=0)
    dropped_records: int = Field(ge=0)
    last_error: str | None = None


_SCHEMA = pa.schema(
    [
        pa.field("format", pa.string(), nullable=False),
        pa.field("recorded_at_utc", pa.string(), nullable=False),
        pa.field("run_version", pa.string(), nullable=False),
        pa.field("update_index", pa.int64(), nullable=False),
        pa.field("optimizer_step_index", pa.int64(), nullable=False),
        pa.field("checkpoint_version", pa.int64(), nullable=False),
        pa.field("pair_manifest_sha256", pa.string(), nullable=False),
        pa.field("policy_sha256", pa.string(), nullable=False),
        pa.field("learner_state_sha256", pa.string(), nullable=False),
        pa.field("decisions", pa.int64(), nullable=False),
        pa.field("fragments_seen", pa.int64(), nullable=False),
        pa.field("fragments_stale", pa.int64(), nullable=False),
        pa.field("loss", pa.float64(), nullable=False),
        pa.field("policy_loss", pa.float64(), nullable=False),
        pa.field("value_loss", pa.float64(), nullable=False),
        pa.field("belief_loss", pa.float64(), nullable=False),
        pa.field("entropy", pa.float64(), nullable=False),
        pa.field("ratio_mean", pa.float64(), nullable=False),
        pa.field("approximate_kl", pa.float64(), nullable=False),
        pa.field("clip_fraction", pa.float64(), nullable=False),
        pa.field("gradient_norm", pa.float64(), nullable=False),
        pa.field("learning_rate", pa.float64(), nullable=False),
        pa.field("kept_decisions_per_second", pa.float64(), nullable=False),
        pa.field("collection_seconds", pa.float64(), nullable=False),
        pa.field("learner_seconds", pa.float64(), nullable=False),
        pa.field("checkpoint_seconds", pa.float64(), nullable=False),
        pa.field("total_seconds", pa.float64(), nullable=False),
        pa.field("cuda_peak_allocated_bytes", pa.int64()),
        pa.field("cuda_peak_reserved_bytes", pa.int64()),
        pa.field("cuda_allocation_retries", pa.int64()),
        pa.field("cuda_ooms", pa.int64()),
    ]
)


class LearnerMetricHistory:
    """Read validated immutable learner metric shards."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.history_dir = run_dir / "performance" / "learner_metrics"
        self.manifest_path = run_dir / "performance" / "learner_metrics_manifest.json"

    def commit(self, record: LearnerMetricRecord) -> dict[str, Any]:
        """Atomically publish one idempotent checkpoint metric."""
        self.history_dir.mkdir(parents=True, exist_ok=True)
        final = self.history_dir / (
            f"part-v{record.checkpoint_version:08d}-"
            f"{record.pair_manifest_sha256[:12]}.parquet"
        )
        temporary = final.with_name(f".{final.name}.{os.getpid()}.{time.time_ns()}.tmp")
        table = pa.Table.from_pylist(
            [record.model_dump(mode="json")],
            schema=_SCHEMA,
        )
        pq.write_table(table, temporary, compression="zstd")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if final.exists():
            if _sha256(final) != _sha256(temporary):
                temporary.unlink(missing_ok=True)
                raise FileExistsError(f"conflicting learner metric shard: {final}")
            temporary.unlink()
        else:
            temporary.replace(final)
        item = {
            "relative_path": str(final.relative_to(self.run_dir)),
            "checkpoint_version": record.checkpoint_version,
            "pair_manifest_sha256": record.pair_manifest_sha256,
            "rows": 1,
            "size_bytes": final.stat().st_size,
            "sha256": _sha256(final),
        }
        manifest = self.read_manifest()
        files = [
            entry
            for entry in _manifest_files(manifest)
            if str(entry["relative_path"]) != item["relative_path"]
        ]
        files.append(item)
        files.sort(key=lambda entry: int(entry["checkpoint_version"]))
        atomic_write_json(
            self.manifest_path,
            {
                "schema_version": LEARNER_METRIC_SCHEMA_VERSION,
                "updated_at_epoch_seconds": time.time(),
                "files": files,
            },
        )
        return item

    def read_manifest(self) -> dict[str, Any]:
        """Return a validated manifest, or an empty manifest."""
        if not self.manifest_path.is_file():
            return {
                "schema_version": LEARNER_METRIC_SCHEMA_VERSION,
                "files": [],
            }
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("learner metric manifest must be an object")
        if int(payload.get("schema_version", -1)) != LEARNER_METRIC_SCHEMA_VERSION:
            raise ValueError("unsupported learner metric manifest schema")
        _manifest_files(payload)
        return payload

    def load(self) -> tuple[LearnerMetricRecord, ...]:
        """Load every locally present metric after fingerprint validation."""
        records: list[LearnerMetricRecord] = []
        for item in _manifest_files(self.read_manifest()):
            path = self.run_dir / str(item["relative_path"])
            if not path.is_file():
                raise FileNotFoundError(f"missing learner metric shard: {path}")
            if path.stat().st_size != int(item["size_bytes"]):
                raise ValueError(f"learner metric shard size mismatch: {path}")
            if _sha256(path) != str(item["sha256"]):
                raise ValueError(f"learner metric shard fingerprint mismatch: {path}")
            rows = pq.read_table(path, schema=_SCHEMA).to_pylist()
            if len(rows) != 1:
                raise ValueError(f"learner metric shard must contain one row: {path}")
            record = LearnerMetricRecord.model_validate(rows[0])
            if record.checkpoint_version != int(
                item["checkpoint_version"]
            ) or record.pair_manifest_sha256 != str(item["pair_manifest_sha256"]):
                raise ValueError(f"learner metric manifest identity mismatch: {path}")
            records.append(record)
        records.sort(key=lambda record: record.checkpoint_version)
        return tuple(records)


class NonBlockingLearnerMetricWriter:
    """Move checkpoint metric serialization away from the learner hot path."""

    def __init__(self, run_dir: Path, *, queue_size: int = 8) -> None:
        if queue_size <= 0:
            raise ValueError("learner metric queue size must be positive")
        self._history = LearnerMetricHistory(run_dir)
        self._queue: queue.Queue[LearnerMetricRecord | None] = queue.Queue(
            maxsize=queue_size
        )
        self._lock = threading.Lock()
        self._committed = 0
        self._dropped = 0
        self._last_error: str | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="learner-metric-writer",
            daemon=True,
        )
        self._thread.start()

    def publish(self, record: LearnerMetricRecord) -> bool:
        """Enqueue without blocking; return false when telemetry was dropped."""
        if self._closed:
            raise RuntimeError("learner metric writer is closed")
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False
        return True

    @property
    def status(self) -> LearnerMetricWriterStatus:
        """Return a lock-protected snapshot without disk access."""
        with self._lock:
            return LearnerMetricWriterStatus(
                queued_records=self._queue.qsize(),
                committed_records=self._committed,
                dropped_records=self._dropped,
                last_error=self._last_error,
            )

    def close(self) -> None:
        """Flush accepted records at normal training shutdown."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            record = self._queue.get()
            try:
                if record is None:
                    return
                self._history.commit(record)
                with self._lock:
                    self._committed += 1
            except Exception as error:  # noqa: BLE001 - telemetry cannot stop training.
                with self._lock:
                    self._last_error = f"{type(error).__name__}: {error}"
            finally:
                self._queue.task_done()


def _manifest_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("files", [])
    if not isinstance(raw, list):
        raise ValueError("learner metric manifest files must be a list")
    files: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("learner metric manifest entries must be objects")
        relative = Path(str(item.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("unsafe learner metric history path")
        files.append(item)
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
