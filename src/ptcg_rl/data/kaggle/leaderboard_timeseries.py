"""Columnar time-series store for periodic Kaggle leaderboard snapshots.

Layout under one store root (default
``data/external/kaggle_leaderboard_timeseries``):

- ``parts/<YYYYMMDD>/part-<HHMMSS>.parquet``: one file per poll, full
  leaderboard rows for that minute (current UTC day only).
- ``daily/leaderboard-<YYYYMMDD>.parquet``: completed days, streamed-compacted
  from that day's parts and then the parts are removed.
- ``polls.jsonl``: one small provenance record per poll (capture time, CSV
  sha256, row count, outcome). This is the poll index used for idempotent
  backfill and health analysis.
- ``latest/``: rolling copy of the newest raw archive plus its manifest, kept
  for provenance and for downstream consumers of the raw zip.

All rows share one schema; every poll stores the full leaderboard so analysis
never needs forward-filling. Parquet with zstd compression keeps 15 days of
per-minute snapshots compact while remaining directly queryable.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

SCHEMA = pa.schema(
    [
        ("polled_at", pa.timestamp("ms", tz="UTC")),
        ("rank", pa.int32()),
        ("team_id", pa.int64()),
        ("team_name", pa.string()),
        ("last_submission", pa.timestamp("ms", tz="UTC")),
        ("score", pa.float64()),
        ("submission_count", pa.int32()),
        ("members", pa.string()),
    ]
)

_CSV_HEADER = [
    "Rank",
    "TeamId",
    "TeamName",
    "LastSubmissionDate",
    "Score",
    "SubmissionCount",
    "TeamMemberUserNames",
]
_COMPACT_BATCH_ROWS = 500_000
POLL_LOG_NAME = "polls.jsonl"


def parse_archive(
    archive: Path,
    polled_at: datetime,
) -> tuple[pa.Table, dict[str, Any]]:
    """Parse one downloaded leaderboard zip into rows plus a digest record."""
    archive_sha256 = _sha256_file(archive)
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected exactly one CSV member, found {members}")
        raw = bundle.read(members[0])
    csv_sha256 = hashlib.sha256(raw).hexdigest()
    table = _parse_csv(raw, polled_at)
    digest = {
        "archive_sha256": archive_sha256,
        "archive_bytes": archive.stat().st_size,
        "member": members[0],
        "csv_sha256": csv_sha256,
        "csv_bytes": len(raw),
        "row_count": table.num_rows,
    }
    return table, digest


def write_part(root: Path, table: pa.Table, polled_at: datetime) -> Path:
    """Atomically publish one poll's rows; returns the store-relative path."""
    moment = polled_at.astimezone(UTC)
    relative = (
        Path("parts")
        / moment.strftime("%Y%m%d")
        / (moment.strftime("part-%H%M%S") + ".parquet")
    )
    final = root / relative
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name("." + final.name + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(final)
    return relative


def compact_completed_days(root: Path, *, today: str) -> list[str]:
    """Compact every parts/<day> older than ``today`` into one daily file."""
    parts_root = root / "parts"
    if not parts_root.is_dir():
        return []
    compacted = []
    for day_dir in sorted(parts_root.iterdir()):
        if not day_dir.is_dir() or day_dir.name >= today:
            continue
        _compact_day(root, day_dir)
        compacted.append(day_dir.name)
    return compacted


def _compact_day(root: Path, day_dir: Path) -> None:
    """Stream one day's daily file (if any) plus parts into a new daily file."""
    day = day_dir.name
    daily = root / "daily" / f"leaderboard-{day}.parquet"
    parts = sorted(day_dir.glob("part-*.parquet"))
    if not parts:
        shutil.rmtree(day_dir)
        return
    daily.parent.mkdir(parents=True, exist_ok=True)
    temporary = daily.with_name("." + daily.name + ".tmp")
    pending: list[pa.Table] = []
    pending_rows = 0
    rows_total = 0
    with pq.ParquetWriter(temporary, SCHEMA, compression="zstd") as writer:

        def flush() -> None:
            nonlocal pending, pending_rows
            if pending:
                writer.write_table(pa.concat_tables(pending))
                pending = []
                pending_rows = 0

        sources: list[Path] = ([daily] if daily.is_file() else []) + parts
        for source in sources:
            handle = pq.ParquetFile(source)
            for batch in handle.iter_batches(batch_size=_COMPACT_BATCH_ROWS):
                table = pa.Table.from_batches([batch]).cast(SCHEMA)
                pending.append(table)
                pending_rows += table.num_rows
                rows_total += table.num_rows
                if pending_rows >= _COMPACT_BATCH_ROWS:
                    flush()
        flush()
    temporary.replace(daily)
    shutil.rmtree(day_dir)
    logger.info(
        "compacted {} ({} parts, {} rows) into {}",
        day,
        len(parts),
        rows_total,
        daily.name,
    )


def data_files(root: Path) -> list[Path]:
    """All Parquet files of the store in chronological order."""
    daily = sorted((root / "daily").glob("leaderboard-*.parquet"))
    parts = sorted((root / "parts").glob("*/part-*.parquet"))
    return daily + parts


def append_poll(root: Path, record: dict[str, Any]) -> None:
    """Append one poll provenance record to the poll log."""
    with (root / POLL_LOG_NAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _parse_csv(raw: bytes, polled_at: datetime) -> pa.Table:
    reader = csv.reader(io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig"))
    header = next(reader, None)
    if header != _CSV_HEADER:
        raise ValueError(f"unexpected leaderboard CSV header: {header}")
    ranks: list[int] = []
    team_ids: list[int] = []
    team_names: list[str] = []
    last_submissions: list[datetime | None] = []
    scores: list[float] = []
    submission_counts: list[int] = []
    members: list[str] = []
    for row in reader:
        if len(row) != len(_CSV_HEADER):
            raise ValueError(f"malformed leaderboard CSV row: {row!r}")
        ranks.append(int(row[0]))
        team_ids.append(int(row[1]))
        team_names.append(row[2])
        last_submissions.append(_parse_timestamp(row[3]))
        scores.append(float(row[4]))
        submission_counts.append(int(row[5]))
        members.append(row[6])
    moment = polled_at.astimezone(UTC)
    return pa.table(
        {
            "polled_at": pa.array([moment] * len(ranks), SCHEMA.field(0).type),
            "rank": pa.array(ranks, SCHEMA.field(1).type),
            "team_id": pa.array(team_ids, SCHEMA.field(2).type),
            "team_name": pa.array(team_names, SCHEMA.field(3).type),
            "last_submission": pa.array(last_submissions, SCHEMA.field(4).type),
            "score": pa.array(scores, SCHEMA.field(5).type),
            "submission_count": pa.array(submission_counts, SCHEMA.field(6).type),
            "members": pa.array(members, SCHEMA.field(7).type),
        },
        schema=SCHEMA,
    )


def _parse_timestamp(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "POLL_LOG_NAME",
    "SCHEMA",
    "append_poll",
    "compact_completed_days",
    "data_files",
    "parse_archive",
    "write_part",
]
