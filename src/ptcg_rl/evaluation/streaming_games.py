"""Crash-safe streaming storage for bundle-gauntlet game rows."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO, cast

import pyarrow as pa
import pyarrow.parquet as pq

_FINGERPRINT_KEY = b"ptcg_rl.bundle_evaluation_fingerprint"
_TOTAL_GAMES_KEY = b"ptcg_rl.bundle_evaluation_total_games"
_EXPECTED_GAMES_KEY = b"ptcg_rl.bundle_evaluation_expected_games"
_STORE_VERSION = 2
_PART_PATTERN = re.compile(r"part-(\d{6})\.parquet")
_OPTIONAL_FLOAT_FIELDS = frozenset(
    {
        "candidate_runtime_min_search_start_remaining_overage",
        "opponent_runtime_min_search_start_remaining_overage",
    }
)


class StreamingGameStore:
    """Persist completed games in atomic Parquet parts and compact at the end.

    A completed part is the recovery boundary. Temporary files are never
    counted, and the progress manifest is advisory: recovery always scans the
    committed Parquet parts so a crash between the two atomic renames is safe.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        evaluation_fingerprint: str,
        total_games: int,
        compression: str,
        result_shard_size: int,
        expected_game_indices: Sequence[int] | None = None,
    ) -> None:
        if total_games <= 0:
            raise ValueError("streaming game store requires positive total_games")
        if result_shard_size <= 0:
            raise ValueError("result_shard_size must be positive")
        self.output_dir = output_dir
        self.parts_dir = output_dir / "games_parts"
        self.games_path = output_dir / "games.parquet"
        self.progress_path = output_dir / "progress.json"
        self._lock_path = output_dir / ".games.lock"
        self._fingerprint = evaluation_fingerprint
        self._campaign_total_games = total_games
        self._expected_indices = _resolve_expected_indices(
            total_games,
            expected_game_indices,
        )
        self._total_games = len(self._expected_indices)
        self._expected_games_fingerprint = _indices_fingerprint(
            self._expected_indices
        )
        self._sparse_indices = expected_game_indices is not None
        self._compression = compression
        self._result_shard_size = result_shard_size
        self._completed_indices: set[int] = set()
        self._parts: list[Path] = []
        self._schema: pa.Schema | None = None
        self._terminal_counts: Counter[str] = Counter()
        self._error_actor_counts: Counter[str] = Counter()
        self._next_part_index = 0
        self._started_monotonic = time.monotonic()
        self._lock: TextIO | None = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        try:
            self._remove_abandoned_temporary_files()
            self._validate_progress_identity()
            self._load_parts()
            if not self._parts and self.games_path.exists():
                raise ValueError(
                    "output directory contains a legacy games.parquet without "
                    "recoverable games_parts; use a fresh output directory"
                )
            if len(self._completed_indices) < self._total_games:
                self.games_path.unlink(missing_ok=True)
            self._resumed_games = len(self._completed_indices)
            self._write_progress(status="running_games")
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> StreamingGameStore:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    @property
    def completed_indices(self) -> frozenset[int]:
        """Return game indices already durable in committed Parquet parts."""
        return frozenset(self._completed_indices)

    @property
    def resumed_games(self) -> int:
        """Return the number of committed games found when this session opened."""
        return self._resumed_games

    @property
    def terminal_reason_counts(self) -> Mapping[str, int]:
        """Return terminal-reason counts across all committed parts."""
        return dict(sorted(self._terminal_counts.items()))

    @property
    def error_actor_counts(self) -> Mapping[str, int]:
        """Return non-empty error-actor counts across all committed parts."""
        return dict(sorted(self._error_actor_counts.items()))

    @property
    def result_parts(self) -> int:
        """Return the number of atomically committed Parquet parts."""
        return len(self._parts)

    def append(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Atomically commit one non-empty result shard."""
        if not rows:
            return
        materialized = _canonicalize_rows(rows)
        materialized.sort(key=lambda row: int(row["game_index"]))
        indices = [int(row["game_index"]) for row in materialized]
        if len(indices) != len(set(indices)):
            raise ValueError("one result shard contains duplicate game_index values")
        invalid = [index for index in indices if index not in self._expected_indices]
        if invalid:
            raise ValueError(f"result shard contains out-of-range games: {invalid}")
        duplicates = sorted(set(indices) & self._completed_indices)
        if duplicates:
            raise ValueError(f"games are already committed: {duplicates}")

        table = _normalize_optional_float_columns(pa.Table.from_pylist(materialized))
        schema = table.schema.remove_metadata()
        if self._schema is not None and not schema.equals(self._schema):
            raise ValueError("bundle game row schema changed within one evaluation")
        metadata = dict(table.schema.metadata or {})
        metadata[_FINGERPRINT_KEY] = self._fingerprint.encode("utf-8")
        metadata[_TOTAL_GAMES_KEY] = str(self._campaign_total_games).encode("ascii")
        if self._sparse_indices:
            metadata[_EXPECTED_GAMES_KEY] = self._expected_games_fingerprint.encode(
                "ascii"
            )
        table = table.replace_schema_metadata(metadata)

        part_path = self.parts_dir / f"part-{self._next_part_index:06d}.parquet"
        temporary = self.parts_dir / f".{part_path.name}.tmp"
        pq.write_table(table, temporary, compression=self._compression)
        temporary.replace(part_path)
        _fsync_directory(self.parts_dir)

        if self._schema is None:
            self._schema = schema
        self._parts.append(part_path)
        self._next_part_index += 1
        self._completed_indices.update(indices)
        self._observe_rows(materialized)
        self._write_progress(status="running_games")

    def progress_payload(
        self,
        *,
        games_finished: int | None = None,
    ) -> dict[str, Any]:
        """Build exact durable progress plus current-session throughput fields."""
        committed = len(self._completed_indices)
        finished = committed if games_finished is None else games_finished
        if not committed <= finished <= self._total_games:
            raise ValueError(
                "games_finished must be between committed games and total games"
            )
        elapsed = max(0.0, time.monotonic() - self._started_monotonic)
        session_finished = max(0, finished - self._resumed_games)
        rate = session_finished / elapsed if elapsed > 0.0 else 0.0
        remaining_finished = self._total_games - finished
        eta = remaining_finished / rate if rate > 0.0 else None
        return {
            "games_total": self._total_games,
            "campaign_games_total": self._campaign_total_games,
            "games_committed": committed,
            "games_finished": finished,
            "games_remaining": self._total_games - committed,
            "progress_fraction": committed / self._total_games,
            "progress_percent": 100.0 * committed / self._total_games,
            "resumed_games": self._resumed_games,
            "result_parts": len(self._parts),
            "result_shard_size": self._result_shard_size,
            "max_replay_games_after_hard_crash": self._result_shard_size - 1,
            "session_elapsed_seconds": elapsed,
            "session_games_per_second": rate,
            "eta_seconds": eta,
            "progress_path": str(self.progress_path),
            "games_parts_dir": str(self.parts_dir),
        }

    def compact(self) -> Path:
        """Stream committed parts into the conventional final Parquet file."""
        completed = len(self._completed_indices)
        if self._completed_indices != self._expected_indices:
            raise ValueError(
                f"cannot compact incomplete evaluation: {completed}/{self._total_games}"
            )
        temporary = self.games_path.with_name(f".{self.games_path.name}.tmp")
        temporary.unlink(missing_ok=True)
        writer: pq.ParquetWriter | None = None
        written_rows = 0
        try:
            for part_path in self._parts:
                parquet_file = pq.ParquetFile(part_path)
                for batch in parquet_file.iter_batches(batch_size=65_536):
                    table = pa.Table.from_batches([batch])
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temporary,
                            table.schema,
                            compression=self._compression,
                        )
                    writer.write_table(table)
                    written_rows += table.num_rows
        finally:
            if writer is not None:
                writer.close()
        if writer is None or written_rows != self._total_games:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                "committed result parts did not compact to the expected row count"
            )
        temporary.replace(self.games_path)
        _fsync_directory(self.output_dir)
        self._write_progress(status="completed_games")
        return self.games_path

    def close(self) -> None:
        """Release the single-writer lock."""
        if self._lock is None:
            return
        fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
        self._lock.close()
        self._lock = None

    def _acquire_lock(self) -> None:
        self._lock = self._lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            self._lock = None
            raise RuntimeError(
                f"another evaluation writer holds {self._lock_path}"
            ) from exc

    def _validate_progress_identity(self) -> None:
        if not self.progress_path.exists():
            return
        raw = json.loads(self.progress_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"invalid progress manifest: {self.progress_path}")
        fingerprint = str(raw.get("evaluation_fingerprint", ""))
        total_games = int(raw.get("games_total", -1))
        campaign_total_games = int(
            raw.get("campaign_games_total", total_games)
        )
        expected_fingerprint = raw.get("expected_games_fingerprint")
        expected_mismatch = self._sparse_indices and (
            expected_fingerprint != self._expected_games_fingerprint
        )
        if (
            fingerprint != self._fingerprint
            or total_games != self._total_games
            or campaign_total_games != self._campaign_total_games
            or expected_mismatch
        ):
            raise ValueError(
                "existing streamed results do not match this evaluation config"
            )

    def _load_parts(self) -> None:
        for part_path in sorted(self.parts_dir.glob("part-*.parquet")):
            match = _PART_PATTERN.fullmatch(part_path.name)
            if match is None:
                raise ValueError(f"invalid result part name: {part_path}")
            parquet_file = pq.ParquetFile(part_path)
            metadata = parquet_file.metadata.metadata or {}
            fingerprint = metadata.get(_FINGERPRINT_KEY, b"").decode("utf-8")
            total_games = int(metadata.get(_TOTAL_GAMES_KEY, b"-1"))
            expected_fingerprint = metadata.get(_EXPECTED_GAMES_KEY, b"").decode(
                "ascii"
            )
            expected_mismatch = self._sparse_indices and (
                expected_fingerprint != self._expected_games_fingerprint
            )
            if (
                fingerprint != self._fingerprint
                or total_games != self._campaign_total_games
                or expected_mismatch
            ):
                raise ValueError(f"result part belongs to another run: {part_path}")
            schema = parquet_file.schema_arrow.remove_metadata()
            if self._schema is not None and not schema.equals(self._schema):
                raise ValueError(f"result part schema mismatch: {part_path}")
            self._schema = schema
            required = {"game_index", "terminal_reason", "error_actor"}
            missing = required - set(schema.names)
            if missing:
                raise ValueError(
                    f"result part is missing columns {missing}: {part_path}"
                )
            for batch in parquet_file.iter_batches(columns=sorted(required)):
                rows = cast(list[dict[str, Any]], batch.to_pylist())
                indices = [int(row["game_index"]) for row in rows]
                invalid = [
                    index for index in indices if index not in self._expected_indices
                ]
                duplicates = sorted(set(indices) & self._completed_indices)
                if invalid or duplicates or len(indices) != len(set(indices)):
                    raise ValueError(
                        f"invalid or duplicate game indices in {part_path}: "
                        f"invalid={invalid}, duplicate={duplicates}"
                    )
                self._completed_indices.update(indices)
                self._observe_rows(rows)
            self._parts.append(part_path)
            self._next_part_index = max(
                self._next_part_index,
                int(match.group(1)) + 1,
            )
        if len(self._completed_indices) > self._total_games:
            raise ValueError("streamed result count exceeds planned games")

    def _observe_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            self._terminal_counts[str(row["terminal_reason"])] += 1
            error_actor = str(row["error_actor"])
            if error_actor:
                self._error_actor_counts[error_actor] += 1

    def _write_progress(self, *, status: str) -> None:
        payload = {
            "store_version": _STORE_VERSION,
            "status": status,
            "evaluation_fingerprint": self._fingerprint,
            "campaign_games_total": self._campaign_total_games,
            "expected_games_fingerprint": self._expected_games_fingerprint,
            **self.progress_payload(),
            "updated_at_unix": time.time(),
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json_atomic(self.progress_path, payload)

    def _remove_abandoned_temporary_files(self) -> None:
        for temporary in self.parts_dir.glob(".part-*.parquet.tmp"):
            temporary.unlink()
        self.games_path.with_name(f".{self.games_path.name}.tmp").unlink(
            missing_ok=True
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _resolve_expected_indices(
    total_games: int,
    expected_game_indices: Sequence[int] | None,
) -> frozenset[int]:
    if expected_game_indices is None:
        return frozenset(range(total_games))
    materialized = tuple(int(index) for index in expected_game_indices)
    if not materialized:
        raise ValueError("expected_game_indices must not be empty")
    if len(materialized) != len(set(materialized)):
        raise ValueError("expected_game_indices must be unique")
    invalid = sorted(index for index in materialized if not 0 <= index < total_games)
    if invalid:
        raise ValueError(f"expected_game_indices are out of range: {invalid}")
    return frozenset(materialized)


def _indices_fingerprint(indices: frozenset[int]) -> str:
    encoded = json.dumps(sorted(indices), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Require identical fields and emit a stable Arrow column order."""
    column_names = tuple(sorted(rows[0]))
    expected = set(column_names)
    materialized: list[dict[str, Any]] = []
    for row in rows:
        actual = set(row)
        if actual != expected:
            raise ValueError(
                "one result shard contains inconsistent row fields: "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        materialized.append({name: row[name] for name in column_names})
    return materialized


def _normalize_optional_float_columns(table: pa.Table) -> pa.Table:
    """Prevent all-null early shards from freezing nullable floats as null type."""
    for field_name in _OPTIONAL_FLOAT_FIELDS & set(table.column_names):
        column_index = table.schema.get_field_index(field_name)
        if pa.types.is_null(table.field(column_index).type):
            table = table.set_column(
                column_index,
                field_name,
                pa.array([None] * table.num_rows, type=pa.float64()),
            )
    return table


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["StreamingGameStore"]
