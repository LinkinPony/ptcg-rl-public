"""Prepare full Kaggle step shards for behavior-cloning training."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent import futures
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.data.kaggle import episode_sync
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps import environment, records


class PrepareBCDataConfig(BaseModel):
    """Hydra-backed config for full BC data sync and step extraction."""

    model_config = ConfigDict(extra="forbid")

    source_index: str = episode_sync.DEFAULT_INDEX_DATASET
    index_dir: Path = Path("data/external/kaggle_top_episodes_index/latest")
    replay_root: Path = Path("data/external/kaggle_top_episodes_daily")
    dates: list[str] = Field(default_factory=list)
    refresh_index: bool = True
    sync_missing: bool = True
    force_download: bool = False
    kaggle_binary: str = "kaggle"

    output_dir: Path = Path("outputs/kaggle_steps/full")
    force_extract: bool = False
    extract_workers: int = 1
    replay_workers_per_date: int = 1
    max_episodes_per_date: int | None = None
    rows_per_shard: int = 50_000
    drop_forced_actions: bool = True
    normalize_unordered_actions: bool = True
    fast_prefix_bytes: int = 65_536
    parser_chunk_size: int = records.DEFAULT_CHUNK_SIZE
    compression: str = "zstd"
    validate_shards: bool = True
    fail_on_missing_replays: bool = True

    @field_validator("dates")
    @classmethod
    def valid_dates(cls, values: list[str]) -> list[str]:
        """Reject malformed date strings early."""
        for value in values:
            datetime.strptime(value, "%Y-%m-%d")
        return values

    @field_validator("max_episodes_per_date")
    @classmethod
    def valid_optional_positive_int(cls, value: int | None) -> int | None:
        """Reject non-positive optional episode limits."""
        if value is not None and value <= 0:
            raise ValueError("max_episodes_per_date must be positive when set")
        return value

    @field_validator("extract_workers")
    @classmethod
    def valid_extract_workers(cls, value: int) -> int:
        """Reject non-positive extraction worker counts."""
        if value <= 0:
            raise ValueError("extract_workers must be positive")
        return value

    @field_validator("replay_workers_per_date")
    @classmethod
    def valid_replay_workers_per_date(cls, value: int) -> int:
        """Reject non-positive replay worker counts."""
        if value <= 0:
            raise ValueError("replay_workers_per_date must be positive")
        return value

    @model_validator(mode="after")
    def valid_worker_topology(self) -> PrepareBCDataConfig:
        """Avoid nested process pools, which can leave orphan workers on abort."""
        if self.extract_workers > 1 and self.replay_workers_per_date > 1:
            raise ValueError(
                "extract_workers and replay_workers_per_date cannot both exceed 1"
            )
        return self

    @field_validator("rows_per_shard", "fast_prefix_bytes", "parser_chunk_size")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive size limits."""
        if value <= 0:
            raise ValueError("size limits must be positive")
        return value


def run(config: PrepareBCDataConfig) -> dict[str, Any]:
    """Sync replay data, extract date-level shards, and write a full manifest."""
    start = time.perf_counter()
    output_dir = deck_records.repo_path(config.output_dir)
    logger.info(
        "starting full BC data preparation output_dir={} dates={} sync_missing={} "
        "extract_workers={} replay_workers_per_date={} rows_per_shard={}",
        deck_records.display_path(output_dir),
        config.dates or "all",
        config.sync_missing,
        config.extract_workers,
        config.replay_workers_per_date,
        config.rows_per_shard,
    )
    sync_report = _sync_missing_replays(config) if config.sync_missing else None
    index_manifest = deck_records.repo_path(config.index_dir) / "manifest.csv"
    index_rows = _read_index_manifest(index_manifest)
    target_rows = _target_index_rows(index_rows, config.dates)
    logger.info(
        "loaded index manifest path={} target_dates={}",
        deck_records.display_path(index_manifest),
        [row["date"] for row in target_rows],
    )

    by_date_dir = output_dir / "by_date"
    by_date_dir.mkdir(parents=True, exist_ok=True)

    missing_dates: list[str] = []
    extract_rows: list[dict[str, str]] = []
    date_reports_by_date: dict[str, dict[str, Any]] = {}
    for row in target_rows:
        date = row["date"]
        replay_dir = deck_records.repo_path(config.replay_root) / date
        if not any(replay_dir.glob("*.json")):
            missing_dates.append(date)
            date_reports_by_date[date] = (
                {
                    "date": date,
                    "status": "missing_replays",
                    "replay_dir": deck_records.display_path(replay_dir),
                }
            )
            continue
        extract_rows.append(row)

    if missing_dates:
        logger.warning("missing replay JSON files for dates={}", missing_dates)
    if missing_dates and config.fail_on_missing_replays:
        raise FileNotFoundError(f"missing replay JSON files for dates: {missing_dates}")

    date_reports_by_date.update(
        _extract_date_reports(
            rows=extract_rows,
            config=config,
            by_date_dir=by_date_dir,
        )
    )
    date_reports = [
        date_reports_by_date[row["date"]]
        for row in target_rows
        if row["date"] in date_reports_by_date
    ]

    aggregate = _aggregate_manifest(
        config=config,
        output_dir=output_dir,
        index_manifest=index_manifest,
        index_rows=target_rows,
        sync_report=sync_report,
        date_reports=date_reports,
        missing_dates=missing_dates,
    )
    _write_json(output_dir / "manifest.json", aggregate)
    elapsed = time.perf_counter() - start
    summary = aggregate["summary"]
    logger.info(
        "finished full BC data preparation output_dir={} dates={} rows={} shards={} "
        "bytes={} seconds={:.2f}",
        deck_records.display_path(output_dir),
        summary["dates"],
        summary["rows"],
        summary["shards"],
        summary["bytes"],
        elapsed,
    )
    print(json.dumps(_console_summary(aggregate), indent=2, sort_keys=True))
    return aggregate


def validate_step_manifest(
    manifest_path: Path,
    *,
    require_shards: bool = True,
) -> dict[str, int]:
    """Validate a step manifest and return aggregate shard counts."""
    resolved_manifest = deck_records.repo_path(manifest_path)
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    raw_shards = manifest.get("shards", [])
    if not isinstance(raw_shards, Sequence):
        raise ValueError(f"manifest shards must be a list: {resolved_manifest}")
    if require_shards and not raw_shards:
        raise ValueError(f"manifest has no shards: {resolved_manifest}")

    expected_schema = records.step_row_schema()
    rows = 0
    bytes_count = 0
    shard_count = 0
    for raw_shard in raw_shards:
        if not isinstance(raw_shard, Mapping):
            raise ValueError(f"invalid shard entry in {resolved_manifest}")
        raw_path = raw_shard.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"shard missing path in {resolved_manifest}")
        shard_path = deck_records.repo_path(Path(raw_path))
        if not shard_path.exists():
            raise FileNotFoundError(f"shard does not exist: {shard_path}")
        parquet_file = pq.ParquetFile(shard_path)
        if not parquet_file.schema_arrow.equals(
            expected_schema,
            check_metadata=False,
        ):
            raise ValueError(f"unexpected shard schema: {shard_path}")
        shard_rows = int(parquet_file.metadata.num_rows)
        expected_rows = raw_shard.get("rows")
        if expected_rows is not None and int(expected_rows) != shard_rows:
            raise ValueError(
                f"manifest row count mismatch for {shard_path}: "
                f"{expected_rows} != {shard_rows}"
            )
        rows += shard_rows
        bytes_count += shard_path.stat().st_size
        shard_count += 1
    return {"rows": rows, "bytes": bytes_count, "shards": shard_count}


def _sync_missing_replays(config: PrepareBCDataConfig) -> dict[str, Any]:
    sync_config = episode_sync.EpisodeSyncConfig(
        source_index=config.source_index,
        index_dir=config.index_dir,
        replay_root=config.replay_root,
        dates=config.dates,
        date_selection="all_missing",
        refresh_index=config.refresh_index,
        dry_run=False,
        force=config.force_download,
        kaggle_binary=config.kaggle_binary,
    )
    return episode_sync.run(sync_config)


def _ensure_date_manifest(
    row: Mapping[str, str],
    config: PrepareBCDataConfig,
    by_date_dir: Path,
) -> dict[str, Any]:
    date = row["date"]
    date_dir = by_date_dir / date
    manifest_path = date_dir / "manifest.json"
    if not config.force_extract and manifest_path.exists():
        try:
            validation = validate_step_manifest(
                manifest_path,
                require_shards=True,
            )
            logger.info(
                "reusing existing date step manifest date={} rows={} shards={} path={}",
                date,
                validation["rows"],
                validation["shards"],
                deck_records.display_path(manifest_path),
            )
            return _date_report(
                row,
                status="skipped_existing",
                manifest_path=manifest_path,
                validation=validation,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "existing date manifest is invalid; re-extracting date={} path={} error={}",
                date,
                deck_records.display_path(manifest_path),
                exc,
            )

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{date}.",
            dir=by_date_dir,
        )
    )
    try:
        logger.info(
            "extracting date step shards date={} temp_dir={} replay_workers={}",
            date,
            deck_records.display_path(temp_dir),
            config.replay_workers_per_date,
        )
        extract_config = environment.KaggleStepExtractionConfig(
            replay_root=config.replay_root,
            dates=[date],
            output_dir=temp_dir,
            max_episodes=config.max_episodes_per_date,
            rows_per_shard=config.rows_per_shard,
            drop_forced_actions=config.drop_forced_actions,
            normalize_unordered_actions=config.normalize_unordered_actions,
            fast_prefix_bytes=config.fast_prefix_bytes,
            parser_chunk_size=config.parser_chunk_size,
            compression=config.compression,
            replay_workers=config.replay_workers_per_date,
        )
        environment.run(extract_config)
        temp_manifest = temp_dir / "manifest.json"
        if config.validate_shards:
            validate_step_manifest(temp_manifest, require_shards=True)
        if date_dir.exists():
            shutil.rmtree(date_dir)
        shutil.move(str(temp_dir), str(date_dir))
        _rewrite_manifest_shard_paths(date_dir / "manifest.json", date_dir)
        validation = (
            validate_step_manifest(manifest_path, require_shards=True)
            if config.validate_shards
            else _manifest_counts(manifest_path)
        )
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    logger.info(
        "finished date step shards date={} rows={} shards={} bytes={} manifest={}",
        date,
        validation["rows"],
        validation["shards"],
        validation["bytes"],
        deck_records.display_path(manifest_path),
    )
    return _date_report(
        row,
        status="extracted",
        manifest_path=manifest_path,
        validation=validation,
    )


def _extract_date_reports(
    *,
    rows: Sequence[dict[str, str]],
    config: PrepareBCDataConfig,
    by_date_dir: Path,
) -> dict[str, dict[str, Any]]:
    if not rows:
        return {}
    if config.extract_workers <= 1 or len(rows) <= 1:
        serial_reports: dict[str, dict[str, Any]] = {}
        start = time.perf_counter()
        total = len(rows)
        for index, row in enumerate(rows, start=1):
            serial_reports[row["date"]] = _ensure_date_manifest(
                row,
                config,
                by_date_dir,
            )
            _log_date_progress(
                index=index,
                total=total,
                reports=serial_reports,
                start=start,
            )
        return serial_reports

    reports: dict[str, dict[str, Any]] = {}
    config_data = config.model_dump(mode="json")
    worker_count = min(config.extract_workers, len(rows))
    start = time.perf_counter()
    logger.info(
        "extracting date manifests in parallel workers={} dates={}",
        worker_count,
        [row["date"] for row in rows],
    )
    with futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_date = {
            executor.submit(
                _ensure_date_manifest_worker,
                row,
                config_data,
                str(by_date_dir),
            ): row["date"]
            for row in rows
        }
        for future in futures.as_completed(future_to_date):
            date = future_to_date[future]
            reports[date] = future.result()
            _log_date_progress(
                index=len(reports),
                total=len(rows),
                reports=reports,
                start=start,
            )
            print(
                json.dumps(
                    {
                        "date": date,
                        "rows": reports[date].get("rows", 0),
                        "shards": reports[date].get("shards", 0),
                        "status": reports[date]["status"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return reports


def _log_date_progress(
    *,
    index: int,
    total: int,
    reports: Mapping[str, Mapping[str, Any]],
    start: float,
) -> None:
    """Log aggregate progress across date-level extraction reports."""
    elapsed = max(time.perf_counter() - start, 1.0e-9)
    rows = sum(int(report.get("rows", 0)) for report in reports.values())
    shards = sum(int(report.get("shards", 0)) for report in reports.values())
    bytes_count = sum(int(report.get("bytes", 0)) for report in reports.values())
    logger.info(
        "full BC data progress dates={}/{} rows={} shards={} bytes={} "
        "seconds={:.1f} rows_per_second={:.1f}",
        index,
        total,
        rows,
        shards,
        bytes_count,
        elapsed,
        rows / elapsed,
    )


def _ensure_date_manifest_worker(
    row: dict[str, str],
    config_data: Mapping[str, Any],
    by_date_dir: str,
) -> dict[str, Any]:
    config = PrepareBCDataConfig.model_validate(config_data)
    return _ensure_date_manifest(row, config, Path(by_date_dir))


def _date_report(
    row: Mapping[str, str],
    *,
    status: str,
    manifest_path: Path,
    validation: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "date": row["date"],
        "status": status,
        "manifest_path": deck_records.display_path(manifest_path),
        "rows": int(validation["rows"]),
        "bytes": int(validation["bytes"]),
        "shards": int(validation["shards"]),
        "episode_count_index": _int_or_none(row.get("episode_count")),
        "total_bytes_index": _int_or_none(row.get("total_bytes")),
    }


def _aggregate_manifest(
    *,
    config: PrepareBCDataConfig,
    output_dir: Path,
    index_manifest: Path,
    index_rows: Sequence[Mapping[str, str]],
    sync_report: dict[str, Any] | None,
    date_reports: Sequence[Mapping[str, Any]],
    missing_dates: Sequence[str],
) -> dict[str, Any]:
    shards: list[dict[str, Any]] = []
    summary = {
        "dates": 0,
        "rows": 0,
        "bytes": 0,
        "shards": 0,
        "missing_dates": len(missing_dates),
    }
    by_date: dict[str, dict[str, Any]] = {}
    for report in date_reports:
        date = str(report["date"])
        by_date[date] = dict(report)
        if report.get("status") == "missing_replays":
            continue
        manifest_path = deck_records.repo_path(Path(str(report["manifest_path"])))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for shard in manifest.get("shards", []):
            if isinstance(shard, Mapping):
                shards.append(dict(shard))
        summary["dates"] += 1
        summary["rows"] += int(report.get("rows", 0))
        summary["bytes"] += int(report.get("bytes", 0))
        summary["shards"] += int(report.get("shards", 0))

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "index_manifest": deck_records.display_path(index_manifest),
        "index_dates": [row["date"] for row in index_rows],
        "sync_report": sync_report,
        "summary": summary,
        "missing_dates": list(missing_dates),
        "by_date": by_date,
        "schema": records.step_row_schema().to_string(),
        "shards": shards,
        "output_dir": deck_records.display_path(output_dir),
    }


def _manifest_counts(manifest_path: Path) -> dict[str, int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = 0
    bytes_count = 0
    shard_count = 0
    for shard in manifest.get("shards", []):
        if not isinstance(shard, Mapping):
            continue
        rows += int(shard.get("rows", 0))
        bytes_count += int(shard.get("bytes", 0))
        shard_count += 1
    return {"rows": rows, "bytes": bytes_count, "shards": shard_count}


def _rewrite_manifest_shard_paths(manifest_path: Path, shard_dir: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rewritten_shards: list[dict[str, Any]] = []
    for shard in manifest.get("shards", []):
        if not isinstance(shard, Mapping):
            continue
        raw_path = shard.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        rewritten = dict(shard)
        rewritten["path"] = deck_records.display_path(shard_dir / Path(raw_path).name)
        rewritten_shards.append(rewritten)
    manifest["shards"] = rewritten_shards
    manifest["output_dir"] = deck_records.display_path(shard_dir)
    _write_json(manifest_path, manifest)


def _read_index_manifest(path: Path) -> list[dict[str, str]]:
    resolved_path = deck_records.repo_path(path)
    with resolved_path.open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    required = {
        "date",
        "daily_dataset_slug",
        "episode_count",
        "total_bytes",
    }
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
    return rows


def _target_index_rows(
    rows: Sequence[dict[str, str]],
    dates: Sequence[str],
) -> list[dict[str, str]]:
    if not dates:
        return list(rows)
    by_date = {row["date"]: row for row in rows}
    missing_dates = [date for date in dates if date not in by_date]
    if missing_dates:
        raise ValueError(f"dates not present in manifest: {missing_dates}")
    return [by_date[date] for date in dates]


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "output_dir": report["output_dir"],
        "index_manifest": report["index_manifest"],
        "summary": report["summary"],
        "missing_dates": report["missing_dates"],
    }


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
