"""Append-only Parquet history for RL performance windows."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.rl.performance_state import (
    atomic_write_json,
    normalize_outcome_cell,
    normalize_performance_window,
)

HISTORY_SCHEMA_VERSION = 2

_LEGACY_CELL_TYPE = pa.struct(
    [
        pa.field("candidate_deck_label", pa.string(), nullable=False),
        pa.field("opponent_kind", pa.string(), nullable=False),
        pa.field("opponent_deck_label", pa.string(), nullable=False),
        pa.field("opponent_id", pa.string(), nullable=False),
        pa.field("candidate_seat", pa.int8(), nullable=False),
        pa.field("wins", pa.int64(), nullable=False),
        pa.field("losses", pa.int64(), nullable=False),
        pa.field("draws", pa.int64(), nullable=False),
    ]
)


def _history_schema(cell_type: pa.StructType) -> pa.Schema:
    return pa.schema(
        [
            pa.field("minute_index", pa.int64(), nullable=False),
            pa.field("started_at_utc", pa.string(), nullable=False),
            pa.field("ended_at_utc", pa.string(), nullable=False),
            pa.field("ended_at_epoch_seconds", pa.float64(), nullable=False),
            pa.field("decoded_games", pa.int64(), nullable=False),
            pa.field("stale_excluded_games", pa.int64(), nullable=False),
            pa.field("queued_games", pa.int64(), nullable=False),
            pa.field("missing_metadata_games", pa.int64(), nullable=False),
            pa.field("missing_metadata_fields_json", pa.string(), nullable=False),
            pa.field("worker_games_json", pa.string(), nullable=False),
            pa.field("policy_version_min", pa.int64()),
            pa.field("policy_version_max", pa.int64()),
            pa.field("slices_json", pa.string(), nullable=False),
            pa.field("cells", pa.list_(cell_type), nullable=False),
        ]
    )


LEGACY_HISTORY_SCHEMA = _history_schema(_LEGACY_CELL_TYPE)

_CELL_TYPE = pa.struct(
    [
        pa.field("candidate_deck_label", pa.string(), nullable=False),
        pa.field("opponent_kind", pa.string(), nullable=False),
        pa.field("opponent_stratum", pa.string(), nullable=False),
        pa.field("opponent_deck_label", pa.string(), nullable=False),
        pa.field("opponent_id", pa.string(), nullable=False),
        pa.field("candidate_seat", pa.int8(), nullable=False),
        pa.field("wins", pa.int64(), nullable=False),
        pa.field("losses", pa.int64(), nullable=False),
        pa.field("draws", pa.int64(), nullable=False),
    ]
)

HISTORY_SCHEMA = _history_schema(_CELL_TYPE)


class PerformanceHistoryStore:
    """Publish complete immutable Parquet shards and an atomic manifest."""

    def __init__(self, history_dir: Path, manifest_path: Path) -> None:
        self.history_dir = history_dir
        self.manifest_path = manifest_path

    def commit(self, windows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        """Commit unseen complete windows as one immutable shard."""
        normalized = self._unseen_windows(windows)
        if not normalized:
            return None
        first = int(normalized[0]["minute_index"])
        last = int(normalized[-1]["minute_index"])
        self.history_dir.mkdir(parents=True, exist_ok=True)
        final = self.history_dir / f"part-{first:08d}-{last:08d}.parquet"
        rows = [_history_row(window) for window in normalized]
        table = pa.Table.from_pylist(rows, schema=HISTORY_SCHEMA)
        temporary = final.with_name(
            f".{final.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        pq.write_table(table, temporary, compression="zstd")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if final.exists():
            if _sha256(final) != _sha256(temporary):
                temporary.unlink(missing_ok=True)
                raise FileExistsError(f"conflicting performance shard: {final}")
            temporary.unlink()
        else:
            temporary.replace(final)
        record = {
            "relative_path": str(final.relative_to(self.manifest_path.parent.parent)),
            "first_minute_index": first,
            "last_minute_index": last,
            "rows": len(rows),
            "size_bytes": final.stat().st_size,
            "sha256": _sha256(final),
        }
        manifest = self.read_manifest()
        parts = [
            part
            for part in _manifest_parts(manifest)
            if str(part.get("relative_path")) != record["relative_path"]
        ]
        parts.append(record)
        parts.sort(key=lambda part: int(part["first_minute_index"]))
        atomic_write_json(
            self.manifest_path,
            {
                "schema_version": HISTORY_SCHEMA_VERSION,
                "updated_at_epoch_seconds": time.time(),
                "files": parts,
            },
        )
        return record

    def read_manifest(self) -> dict[str, Any]:
        """Read a valid manifest or return an empty one."""
        if not self.manifest_path.is_file():
            return {"schema_version": HISTORY_SCHEMA_VERSION, "files": []}
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("performance history manifest must be an object")
        if int(payload.get("schema_version", -1)) not in (1, HISTORY_SCHEMA_VERSION):
            raise ValueError("unsupported performance history manifest schema")
        _manifest_parts(payload)
        return cast(dict[str, Any], payload)

    def last_committed_minute(self) -> int:
        """Return the largest immutable minute index."""
        parts = _manifest_parts(self.read_manifest())
        return max((int(part["last_minute_index"]) for part in parts), default=0)

    def load_windows(self) -> list[dict[str, Any]]:
        """Load and validate all locally present immutable windows."""
        return self.load_windows_from_parts(
            _manifest_parts(self.read_manifest())
        )

    def load_windows_from_parts(
        self,
        parts: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Load selected manifest parts after applying the normal validations."""
        windows: dict[int, dict[str, Any]] = {}
        for part in _manifest_parts({"files": list(parts)}):
            path = self.manifest_path.parent.parent / str(part["relative_path"])
            if not path.is_file():
                raise FileNotFoundError(f"missing performance shard: {path}")
            if path.stat().st_size != int(part["size_bytes"]):
                raise ValueError(f"performance shard size mismatch: {path}")
            if _sha256(path) != str(part["sha256"]):
                raise ValueError(f"performance shard fingerprint mismatch: {path}")
            table = pq.read_table(path)
            if not any(
                table.schema.equals(schema, check_metadata=False)
                for schema in (LEGACY_HISTORY_SCHEMA, HISTORY_SCHEMA)
            ):
                raise ValueError(f"unsupported performance shard schema: {path}")
            legacy_schema = table.schema.equals(
                LEGACY_HISTORY_SCHEMA,
                check_metadata=False,
            )
            for row in table.to_pylist():
                window = _window_from_history_row(
                    row,
                    rebuild_stratum_slices=legacy_schema,
                )
                windows[int(window["minute_index"])] = window
        return [windows[index] for index in sorted(windows)]

    def _unseen_windows(
        self, windows: Sequence[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        last = self.last_committed_minute()
        normalized = [
            window
            for window in windows
            if window.get("complete") is True
            and int(window.get("minute_index") or 0) > last
        ]
        normalized.sort(key=lambda window: int(window["minute_index"]))
        return normalized


def _history_row(window: Mapping[str, Any]) -> dict[str, Any]:
    cells = window.get("cells")
    raw_cells = cells if isinstance(cells, list) else []
    return {
        "minute_index": int(window["minute_index"]),
        "started_at_utc": str(window["started_at_utc"]),
        "ended_at_utc": str(window["ended_at_utc"]),
        "ended_at_epoch_seconds": float(window["ended_at_epoch_seconds"]),
        "decoded_games": int(window.get("decoded_games", 0)),
        "stale_excluded_games": int(window.get("stale_excluded_games", 0)),
        "queued_games": int(window.get("queued_games", 0)),
        "missing_metadata_games": int(window.get("missing_metadata_games", 0)),
        "missing_metadata_fields_json": json.dumps(
            window.get("missing_metadata_fields", {}), sort_keys=True
        ),
        "worker_games_json": json.dumps(window.get("worker_games", {}), sort_keys=True),
        "policy_version_min": window.get("policy_version_min"),
        "policy_version_max": window.get("policy_version_max"),
        "slices_json": json.dumps(window.get("slices", {}), sort_keys=True),
        "cells": [
            normalize_outcome_cell(cell)
            for cell in raw_cells
            if isinstance(cell, Mapping)
        ],
    }


def _window_from_history_row(
    row: Mapping[str, Any],
    *,
    rebuild_stratum_slices: bool,
) -> dict[str, Any]:
    window = {
        "minute_index": int(row["minute_index"]),
        "complete": True,
        "started_at_utc": str(row["started_at_utc"]),
        "ended_at_utc": str(row["ended_at_utc"]),
        "ended_at_epoch_seconds": float(row["ended_at_epoch_seconds"]),
        "decoded_games": int(row["decoded_games"]),
        "stale_excluded_games": int(row["stale_excluded_games"]),
        "queued_games": int(row["queued_games"]),
        "missing_metadata_games": int(row["missing_metadata_games"]),
        "missing_metadata_fields": json.loads(
            str(row["missing_metadata_fields_json"])
        ),
        "worker_games": json.loads(str(row["worker_games_json"])),
        "policy_version_min": row.get("policy_version_min"),
        "policy_version_max": row.get("policy_version_max"),
        "slices": json.loads(str(row["slices_json"])),
        "cells": [
            normalize_outcome_cell(cell)
            for cell in list(row.get("cells") or [])
            if isinstance(cell, Mapping)
        ],
    }
    return normalize_performance_window(
        window,
        rebuild_stratum_slices=rebuild_stratum_slices,
    )


def _manifest_parts(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("files", [])
    if not isinstance(raw, list):
        raise ValueError("performance history manifest files must be a list")
    parts: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("performance history manifest entries must be objects")
        relative = Path(str(item.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("unsafe performance history path")
        parts.append(cast(dict[str, Any], item))
    return parts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
