"""Throughput micro-benchmark for step-level replay extraction."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps import records as step_records

BYTES_PER_MIB = 1024.0 * 1024.0


class KaggleStepBenchmarkConfig(BaseModel):
    """Config for step-extraction throughput micro-benchmarks."""

    model_config = ConfigDict(extra="forbid")

    replay_root: Path = Path("data/external/kaggle_top_episodes_daily")
    dates: list[str] = ["2026-06-16"]
    replay_paths: tuple[Path, ...] = ()
    max_replays: int | None = 150
    drop_forced_actions: bool = True
    normalize_unordered_actions: bool = True
    fast_prefix_bytes: int = 65_536
    chunk_size: int = step_records.DEFAULT_CHUNK_SIZE
    target_dataset_gib: float | None = 283.0
    max_examples: int = 20
    output_path: Path | None = Path("outputs/kaggle_steps/benchmark/summary.json")

    @field_validator("dates")
    @classmethod
    def valid_dates(cls, values: list[str]) -> list[str]:
        """Reject malformed date strings early."""
        for value in values:
            datetime.strptime(value, "%Y-%m-%d")
        return values

    @field_validator("max_replays")
    @classmethod
    def valid_optional_positive_int(cls, value: int | None) -> int | None:
        """Reject non-positive optional replay limits."""
        if value is not None and value <= 0:
            raise ValueError("max_replays must be positive when set")
        return value

    @field_validator("fast_prefix_bytes", "chunk_size")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive byte limits."""
        if value <= 0:
            raise ValueError("byte limits must be positive")
        return value

    @field_validator("target_dataset_gib")
    @classmethod
    def valid_optional_positive_float(cls, value: float | None) -> float | None:
        """Reject non-positive optional dataset-size estimates."""
        if value is not None and value <= 0.0:
            raise ValueError("target_dataset_gib must be positive when set")
        return value

    @field_validator("max_examples")
    @classmethod
    def valid_non_negative_int(cls, value: int) -> int:
        """Reject negative example limits."""
        if value < 0:
            raise ValueError("max_examples must be non-negative")
        return value


@dataclass
class _BenchmarkStats:
    """Mutable aggregate benchmark counters."""

    counters: Counter[str] = field(default_factory=Counter)
    by_date: dict[str, Counter[str]] = field(default_factory=dict)
    extraction_counters: Counter[str] = field(default_factory=Counter)
    scan_error_examples: list[dict[str, Any]] = field(default_factory=list)
    total_seconds: float = 0.0
    total_bytes: int = 0


def run_kaggle_step_benchmark(
    config: KaggleStepBenchmarkConfig,
) -> dict[str, Any]:
    """Benchmark step extraction over configured replay files."""
    report = benchmark_replay_paths(_resolve_replay_paths(config), config=config)
    return report


def benchmark_replay_paths(
    replay_paths: Iterable[Path],
    *,
    config: KaggleStepBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Benchmark explicit replay paths; useful for tests and focused runs."""
    paths = tuple(replay_paths)
    active_config = config or KaggleStepBenchmarkConfig(
        replay_paths=paths,
        output_path=None,
    )
    stats = _BenchmarkStats()
    for replay_path in paths:
        if _limit_reached(stats.counters["replays_seen"], active_config.max_replays):
            break
        _benchmark_replay(replay_path, config=active_config, stats=stats)

    report = _report(active_config, stats)
    _write_report(active_config.output_path, report)
    return report


def _resolve_replay_paths(config: KaggleStepBenchmarkConfig) -> tuple[Path, ...]:
    if config.replay_paths:
        return tuple(deck_records.repo_path(path) for path in config.replay_paths)
    replay_root = deck_records.repo_path(config.replay_root)
    return tuple(deck_records.iter_replay_paths(replay_root, set(config.dates)))


def _benchmark_replay(
    replay_path: Path,
    *,
    config: KaggleStepBenchmarkConfig,
    stats: _BenchmarkStats,
) -> None:
    stats.counters["replays_seen"] += 1
    date_counts = stats.by_date.setdefault(replay_path.parent.name, Counter())
    date_counts["replays_seen"] += 1
    try:
        replay_bytes = replay_path.stat().st_size
        start = time.perf_counter()
        rows, counters = step_records.extract_replay_rows(
            replay_path,
            drop_forced_actions=config.drop_forced_actions,
            normalize_unordered_actions=config.normalize_unordered_actions,
            fast_prefix_bytes=config.fast_prefix_bytes,
            chunk_size=config.chunk_size,
        )
        elapsed = time.perf_counter() - start
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        stats.counters["scan_errors"] += 1
        date_counts["scan_errors"] += 1
        if len(stats.scan_error_examples) < config.max_examples:
            stats.scan_error_examples.append(
                {
                    "replay_path": deck_records.display_path(replay_path),
                    "error": str(exc),
                }
            )
        return

    stats.total_seconds += elapsed
    stats.total_bytes += replay_bytes
    stats.extraction_counters.update(counters)
    stats.counters["rows"] += len(rows)
    date_counts["rows"] += len(rows)
    date_counts["bytes"] += replay_bytes


def _report(
    config: KaggleStepBenchmarkConfig,
    stats: _BenchmarkStats,
) -> dict[str, Any]:
    throughput = _throughput(stats, target_dataset_gib=config.target_dataset_gib)
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "summary": {
            **dict(sorted(stats.counters.items())),
            "bytes": stats.total_bytes,
            "seconds": stats.total_seconds,
            "mib": stats.total_bytes / BYTES_PER_MIB,
        },
        "throughput": throughput,
        "extraction_counters": dict(sorted(stats.extraction_counters.items())),
        "by_date": {
            date: dict(sorted(counter.items()))
            for date, counter in sorted(stats.by_date.items())
        },
        "scan_error_examples": stats.scan_error_examples,
    }


def _throughput(
    stats: _BenchmarkStats,
    *,
    target_dataset_gib: float | None,
) -> dict[str, float | None]:
    total_mib = stats.total_bytes / BYTES_PER_MIB
    mib_per_second = _safe_rate(total_mib, stats.total_seconds)
    target_hours = None
    if mib_per_second is not None and target_dataset_gib is not None:
        target_mib = target_dataset_gib * 1024.0
        target_hours = target_mib / mib_per_second / 3600.0
    return {
        "replays_per_second": _safe_rate(
            float(stats.counters["replays_seen"]),
            stats.total_seconds,
        ),
        "rows_per_second": _safe_rate(float(stats.counters["rows"]), stats.total_seconds),
        "mib_per_second": mib_per_second,
        "seconds_per_replay": _safe_rate(
            stats.total_seconds,
            float(stats.counters["replays_seen"]),
        ),
        "seconds_per_mib": _safe_rate(stats.total_seconds, total_mib),
        "estimated_target_dataset_hours": target_hours,
    }


def _safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _write_report(path: Path | None, report: Mapping[str, Any]) -> None:
    if path is None:
        return
    output_path = deck_records.repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _limit_reached(count: int, limit: int | None) -> bool:
    return limit is not None and count >= limit
