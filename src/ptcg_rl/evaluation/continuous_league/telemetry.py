"""Asynchronous, crash-safe Parquet export of compact game telemetry."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.evaluation.continuous_league.ledger import LeagueLedger

_PART = re.compile(r"part-(\d{8})\.parquet")


class TelemetryExporter:
    """Stream SQLite result rows to immutable atomic Parquet shards."""

    def __init__(self, ledger: LeagueLedger) -> None:
        self.ledger = ledger
        configured = ledger.config.telemetry_dir
        self.output_dir = (
            configured if configured.is_absolute() else ledger.repo_root / configured
        ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_available(self) -> int:
        """Export at most one configured shard and return its row count."""
        metadata_cursor = int(self.ledger.metadata("telemetry_result_seq") or 0)
        published_cursor = self._published_cursor()
        cursor = max(metadata_cursor, published_cursor)
        if cursor != metadata_cursor:
            # Recover the narrow crash window after an atomic shard publish but
            # before the SQLite export cursor transaction committed.
            self.ledger.advance_telemetry_cursor(cursor)
        with self.ledger.database.read() as connection:
            rows = connection.execute(
                """SELECT r.*, m.side_a_bundle_id, m.side_b_bundle_id,
                          m.schedule_reason
                   FROM results r JOIN matches m USING(match_id)
                   WHERE r.result_seq > ? ORDER BY r.result_seq
                   LIMIT ?""",
                (cursor, self.ledger.config.telemetry_shard_rows),
            ).fetchall()
        if not rows:
            return 0
        records: list[dict[str, Any]] = []
        for row in rows:
            records.append(
                {
                    "result_seq": int(row["result_seq"]),
                    "event_seq": (
                        None if row["event_seq"] is None else int(row["event_seq"])
                    ),
                    "match_id": str(row["match_id"]),
                    "worker_id": str(row["worker_id"]),
                    "side_a_bundle_id": str(row["side_a_bundle_id"]),
                    "side_b_bundle_id": str(row["side_b_bundle_id"]),
                    "outcome": str(row["outcome"]),
                    "terminal_reason": str(row["terminal_reason"]),
                    "schedule_reason": str(row["schedule_reason"]),
                    "started_at": str(row["started_at"]),
                    "finished_at": str(row["finished_at"]),
                    "steps": int(row["steps"]),
                    "duration_seconds": float(row["duration_seconds"]),
                    "telemetry_sha256": str(row["telemetry_sha256"]),
                    "telemetry_msgpack": bytes(row["telemetry_msgpack"]),
                }
            )
        last_seq = int(records[-1]["result_seq"])
        part_index = self._next_part_index()
        path = self.output_dir / f"part-{part_index:08d}.parquet"
        pending = self.output_dir / f".{path.name}.pending-{os.getpid()}"
        table = pa.Table.from_pylist(records)
        pq.write_table(table, pending, compression="zstd")
        pending.replace(path)
        _fsync_directory(self.output_dir)
        self.ledger.advance_telemetry_cursor(last_seq)
        return len(records)

    def _next_part_index(self) -> int:
        indices = [
            int(match.group(1))
            for path in self.output_dir.glob("part-*.parquet")
            if (match := _PART.fullmatch(path.name)) is not None
        ]
        return 0 if not indices else max(indices) + 1

    def _published_cursor(self) -> int:
        parts = sorted(
            (
                (int(match.group(1)), path)
                for path in self.output_dir.glob("part-*.parquet")
                if (match := _PART.fullmatch(path.name)) is not None
            ),
            key=lambda item: item[0],
        )
        if not parts:
            return 0
        table = pq.read_table(parts[-1][1], columns=["result_seq"])
        values = table.column("result_seq").to_pylist()
        if not values:
            raise ValueError("continuous league telemetry shard is empty")
        return max(int(value) for value in values)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["TelemetryExporter"]
