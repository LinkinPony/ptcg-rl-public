"""Hydra entry point for step-level Kaggle replay extraction."""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent import futures
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps import records


class KaggleStepExtractionConfig(BaseModel):
    """Hydra-backed config for step-level behavior-cloning extraction."""

    model_config = ConfigDict(extra="forbid")

    replay_root: Path = Path("data/external/kaggle_top_episodes_daily")
    dates: list[str] = []
    output_dir: Path = Path("outputs/kaggle_steps/latest")
    max_episodes: int | None = None
    rows_per_shard: int = 50_000
    drop_forced_actions: bool = True
    normalize_unordered_actions: bool = True
    fast_prefix_bytes: int = 65_536
    parser_chunk_size: int = records.DEFAULT_CHUNK_SIZE
    compression: str = "zstd"
    replay_workers: int = 1

    @field_validator("dates")
    @classmethod
    def valid_dates(cls, values: list[str]) -> list[str]:
        """Reject malformed date strings early."""
        for value in values:
            datetime.strptime(value, "%Y-%m-%d")
        return values

    @field_validator("max_episodes")
    @classmethod
    def positive_optional_limit(cls, value: int | None) -> int | None:
        """Reject non-positive optional limits."""
        if value is not None and value <= 0:
            raise ValueError("max_episodes must be positive when set")
        return value

    @field_validator("rows_per_shard", "fast_prefix_bytes", "parser_chunk_size")
    @classmethod
    def positive_limit(cls, value: int) -> int:
        """Reject non-positive size limits."""
        if value <= 0:
            raise ValueError("size limits must be positive")
        return value

    @field_validator("replay_workers")
    @classmethod
    def positive_worker_count(cls, value: int) -> int:
        """Reject non-positive worker counts."""
        if value <= 0:
            raise ValueError("replay_workers must be positive")
        return value


@dataclass
class _ShardWriter:
    """Buffered Parquet shard writer."""

    output_dir: Path
    rows_per_shard: int
    compression: str

    def __post_init__(self) -> None:
        self._buffer: list[records.Row] = []
        self._shards: list[dict[str, Any]] = []
        self._schema = records.step_row_schema()

    @property
    def shards(self) -> list[dict[str, Any]]:
        """Return manifest rows for flushed shards."""
        return list(self._shards)

    def add_rows(self, rows: list[records.Row]) -> None:
        """Add rows and flush full shards."""
        if not rows:
            return
        self._buffer.extend(rows)
        while len(self._buffer) >= self.rows_per_shard:
            self._flush_rows(self.rows_per_shard)

    def close(self) -> None:
        """Flush the final partial shard."""
        if self._buffer:
            self._flush_rows(len(self._buffer))

    def _flush_rows(self, count: int) -> None:
        shard_rows = self._buffer[:count]
        del self._buffer[:count]
        shard_index = len(self._shards)
        shard_path = self.output_dir / f"steps-{shard_index:05d}.parquet"
        start = time.perf_counter()
        logger.info(
            "writing step shard index={} rows={} path={}",
            shard_index,
            len(shard_rows),
            deck_records.display_path(shard_path),
        )
        table = pa.Table.from_pylist(shard_rows, schema=self._schema)
        pq.write_table(table, shard_path, compression=self.compression)
        elapsed = time.perf_counter() - start
        self._shards.append(
            {
                "path": deck_records.display_path(shard_path),
                "rows": len(shard_rows),
                "bytes": shard_path.stat().st_size,
            }
        )
        logger.info(
            "wrote step shard index={} rows={} bytes={} seconds={:.2f}",
            shard_index,
            len(shard_rows),
            shard_path.stat().st_size,
            elapsed,
        )


def run(config: KaggleStepExtractionConfig) -> dict[str, Any]:
    """Extract step-level rows from local Kaggle replay JSON files."""
    start = time.perf_counter()
    replay_root = deck_records.repo_path(config.replay_root)
    output_dir = deck_records.repo_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "starting Kaggle step extraction replay_root={} dates={} output_dir={} "
        "max_episodes={} rows_per_shard={} replay_workers={}",
        deck_records.display_path(replay_root),
        config.dates or "all",
        deck_records.display_path(output_dir),
        config.max_episodes,
        config.rows_per_shard,
        config.replay_workers,
    )

    writer = _ShardWriter(
        output_dir=output_dir,
        rows_per_shard=config.rows_per_shard,
        compression=config.compression,
    )
    counters: Counter[str] = Counter()
    by_date: dict[str, Counter[str]] = {}

    replay_paths = tuple(deck_records.iter_replay_paths(replay_root, set(config.dates)))
    if config.max_episodes is not None:
        if len(replay_paths) > config.max_episodes:
            counters["max_episodes_reached"] += 1
        replay_paths = replay_paths[: config.max_episodes]
    logger.info("selected {} replay JSON files for extraction", len(replay_paths))

    if config.replay_workers > 1 and len(replay_paths) > 1:
        _extract_parallel(replay_paths, config, writer, counters, by_date)
    else:
        _extract_serial(replay_paths, config, writer, counters, by_date)

    writer.close()
    report = _summary_report(config, output_dir, counters, by_date, writer.shards)
    _write_json(output_dir / "manifest.json", report)
    elapsed = time.perf_counter() - start
    logger.info(
        "finished Kaggle step extraction output_dir={} replays={} rows={} shards={} "
        "seconds={:.2f} rows_per_second={:.1f}",
        deck_records.display_path(output_dir),
        counters["episode_json_files"],
        counters["rows"],
        len(writer.shards),
        elapsed,
        counters["rows"] / elapsed if elapsed > 0.0 else 0.0,
    )
    print(json.dumps(_console_summary(output_dir, report), indent=2, sort_keys=True))
    return report


def _extract_serial(
    replay_paths: tuple[Path, ...],
    config: KaggleStepExtractionConfig,
    writer: _ShardWriter,
    counters: Counter[str],
    by_date: dict[str, Counter[str]],
) -> None:
    start = time.perf_counter()
    total = len(replay_paths)
    for replay_path in replay_paths:
        date_counts = by_date.setdefault(replay_path.parent.name, Counter())
        counters["episode_json_files"] += 1
        date_counts["episode_json_files"] += 1
        try:
            rows, replay_counters = records.extract_replay_rows(
                replay_path,
                drop_forced_actions=config.drop_forced_actions,
                normalize_unordered_actions=config.normalize_unordered_actions,
                fast_prefix_bytes=config.fast_prefix_bytes,
                chunk_size=config.parser_chunk_size,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            counters["scan_errors"] += 1
            date_counts["scan_errors"] += 1
            print(f"failed to extract {deck_records.display_path(replay_path)}: {exc}")
            continue
        writer.add_rows(rows)
        counters.update(replay_counters)
        date_counts.update(replay_counters)
        _log_extraction_progress(counters, total=total, start=start)


def _extract_parallel(
    replay_paths: tuple[Path, ...],
    config: KaggleStepExtractionConfig,
    writer: _ShardWriter,
    counters: Counter[str],
    by_date: dict[str, Counter[str]],
) -> None:
    worker_count = min(config.replay_workers, len(replay_paths))
    config_data = config.model_dump(mode="json")
    start = time.perf_counter()
    total = len(replay_paths)
    logger.info(
        "extracting replay JSON files in parallel workers={} replays={}",
        worker_count,
        total,
    )
    with futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_path = {
            executor.submit(_extract_replay_worker, replay_path, config_data): replay_path
            for replay_path in replay_paths
        }
        for future in futures.as_completed(future_to_path):
            replay_path = future_to_path[future]
            date_counts = by_date.setdefault(replay_path.parent.name, Counter())
            counters["episode_json_files"] += 1
            date_counts["episode_json_files"] += 1
            rows, replay_counters, error = future.result()
            if error is not None:
                counters["scan_errors"] += 1
                date_counts["scan_errors"] += 1
                print(f"failed to extract {deck_records.display_path(replay_path)}: {error}")
                continue
            writer.add_rows(rows)
            counters.update(replay_counters)
            date_counts.update(replay_counters)
            _log_extraction_progress(counters, total=total, start=start)


def _extract_replay_worker(
    replay_path: Path,
    config_data: dict[str, Any],
) -> tuple[list[records.Row], Counter[str], str | None]:
    config = KaggleStepExtractionConfig.model_validate(config_data)
    try:
        rows, replay_counters = records.extract_replay_rows(
            replay_path,
            drop_forced_actions=config.drop_forced_actions,
            normalize_unordered_actions=config.normalize_unordered_actions,
            fast_prefix_bytes=config.fast_prefix_bytes,
            chunk_size=config.parser_chunk_size,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [], Counter(), str(exc)
    return rows, replay_counters, None


def _log_extraction_progress(
    counters: Counter[str],
    *,
    total: int,
    start: float,
) -> None:
    """Log bounded replay-extraction progress from aggregate counters."""
    processed = counters["episode_json_files"]
    if processed == 0:
        return
    if processed < total and processed % 500 != 0:
        return
    elapsed = max(time.perf_counter() - start, 1.0e-9)
    logger.info(
        "step extraction progress replays={}/{} rows={} forced_dropped={} "
        "scan_errors={} seconds={:.1f} replays_per_second={:.1f} rows_per_second={:.1f}",
        processed,
        total,
        counters["rows"],
        counters["forced_actions_dropped"],
        counters["scan_errors"],
        elapsed,
        processed / elapsed,
        counters["rows"] / elapsed,
    )


def _summary_report(
    config: KaggleStepExtractionConfig,
    output_dir: Path,
    counters: Counter[str],
    by_date: dict[str, Counter[str]],
    shards: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "summary": dict(sorted(counters.items())),
        "by_date": {
            date: dict(sorted(date_counts.items()))
            for date, date_counts in sorted(by_date.items())
        },
        "schema": records.step_row_schema().to_string(),
        "shards": shards,
        "output_dir": deck_records.display_path(output_dir),
    }


def _console_summary(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_dir": deck_records.display_path(output_dir),
        "summary": report["summary"],
        "shards": report["shards"],
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="data/kaggle_steps",
)
def main(hydra_config: DictConfig) -> None:
    """Hydra entry point."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = KaggleStepExtractionConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    run(config)


if __name__ == "__main__":
    main()
