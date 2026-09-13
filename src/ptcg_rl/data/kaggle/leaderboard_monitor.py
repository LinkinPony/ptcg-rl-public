"""Periodic full-leaderboard monitor writing a Parquet time series.

Every poll downloads ``kaggle competitions leaderboard <competition>
--download``, parses the full leaderboard CSV, and appends it to the columnar
store described in :mod:`ptcg_rl.data.kaggle.leaderboard_timeseries`. The raw
zip is kept only as a rolling ``latest/`` copy for provenance; parsed rows are
the durable artifact. Completed UTC days are compacted into one daily Parquet
file automatically.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.data.kaggle import leaderboard_timeseries as store
from ptcg_rl.rl.performance_state import atomic_write_json

REPO_ROOT = Path(__file__).resolve().parents[4]
_STATUS = "status.json"
_LOCK = "monitor.lock"
_STAGING = ".monitor_staging"


class LeaderboardMonitorConfig(BaseModel):
    """Hydra-backed config for the leaderboard time-series monitor."""

    model_config = ConfigDict(extra="forbid")

    competition: str = "pokemon-tcg-ai-battle"
    store_root: Path = Path("data/external/kaggle_leaderboard_timeseries")
    kaggle_binary: str = "kaggle"
    interval_seconds: float = 60.0
    download_timeout_seconds: float = 300.0
    keep_latest_archive: bool = True
    max_polls: int = 0
    max_consecutive_failures: int = 120

    @field_validator("interval_seconds", "download_timeout_seconds")
    @classmethod
    def positive_seconds(cls, value: float) -> float:
        """Reject non-positive durations."""
        if value <= 0:
            raise ValueError("durations must be positive")
        return value

    @field_validator("max_polls", "max_consecutive_failures")
    @classmethod
    def non_negative_count(cls, value: int) -> int:
        """Reject negative counts; zero means unlimited."""
        if value < 0:
            raise ValueError("counts must be >= 0 (0 means unlimited)")
        return value


class _StopRequestedError(Exception):
    """Raised from a signal handler to stop the loop gracefully."""


def run(config: LeaderboardMonitorConfig) -> dict[str, Any]:
    """Poll the competition leaderboard until stopped or bounded out."""
    if shutil.which(config.kaggle_binary) is None:
        raise RuntimeError(f"Kaggle CLI binary not found: {config.kaggle_binary}")
    root = _repo_path(config.store_root)
    root.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    counters = {"polls": 0, "captured": 0, "errors": 0}
    consecutive_failures = 0
    _install_signal_handlers()
    logger.info(
        "monitoring {} leaderboard every {:.0f}s into {}",
        config.competition,
        config.interval_seconds,
        root,
    )
    with _exclusive_lock(root / _LOCK):
        _write_store_manifest(config, root)
        store.compact_completed_days(root, today=_utc_day())
        next_tick = time.monotonic()
        try:
            while True:
                record = _poll_once(config, root)
                counters["polls"] += 1
                if record["status"] == "captured":
                    counters["captured"] += 1
                    consecutive_failures = 0
                else:
                    counters["errors"] += 1
                    consecutive_failures += 1
                store.append_poll(root, record)
                compacted = store.compact_completed_days(root, today=_utc_day())
                atomic_write_json(
                    root / _STATUS,
                    {
                        "schema_version": 2,
                        "competition": config.competition,
                        "pid": os.getpid(),
                        "started_at_utc": started_at,
                        "interval_seconds": config.interval_seconds,
                        "last_poll": record,
                        "last_compacted_days": compacted or None,
                        "consecutive_failures": consecutive_failures,
                        **counters,
                    },
                )
                if (
                    config.max_consecutive_failures
                    and consecutive_failures >= config.max_consecutive_failures
                ):
                    raise RuntimeError(
                        f"{consecutive_failures} consecutive poll failures; "
                        f"last error: {record.get('error')}"
                    )
                if config.max_polls and counters["polls"] >= config.max_polls:
                    break
                next_tick += config.interval_seconds
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
        except _StopRequestedError:
            logger.info("stop signal received; exiting after completed poll")
    summary: dict[str, Any] = {"started_at_utc": started_at, **counters}
    logger.info("monitor finished: {}", summary)
    return summary


def _poll_once(config: LeaderboardMonitorConfig, root: Path) -> dict[str, Any]:
    """Download, parse, and store one full leaderboard snapshot."""
    poll_start = time.monotonic()
    polled_at = datetime.now(UTC).replace(microsecond=0)
    record: dict[str, Any] = {
        "polled_at_utc": _format_utc(polled_at),
        "status": "error",
        "source": "live",
        "part": None,
        "rows": None,
        "csv_sha256": None,
        "error": None,
    }
    staging = root / _STAGING
    try:
        archive = _download_archive(config, staging)
        table, digest = store.parse_archive(archive, polled_at)
        part = store.write_part(root, table, polled_at)
        record.update(
            status="captured",
            part=str(part),
            rows=digest["row_count"],
            csv_sha256=digest["csv_sha256"],
        )
        if config.keep_latest_archive:
            _publish_latest(config, root, archive, polled_at, digest)
        logger.info(
            "captured {} ({} rows, csv sha256 {})",
            part.name,
            digest["row_count"],
            str(digest["csv_sha256"])[:12],
        )
    except _StopRequestedError:
        raise
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        logger.warning("poll failed: {}", record["error"])
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        record["duration_seconds"] = round(time.monotonic() - poll_start, 3)
    return record


def _download_archive(config: LeaderboardMonitorConfig, staging: Path) -> Path:
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    command = [
        config.kaggle_binary,
        "competitions",
        "leaderboard",
        config.competition,
        "--download",
        "--path",
        str(staging),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=config.download_timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"kaggle CLI exited with {completed.returncode}: {detail[-500:]}"
        )
    archives = sorted(staging.glob("*.zip"))
    if len(archives) != 1:
        raise FileNotFoundError(
            f"expected exactly one downloaded zip in {staging}, found {archives}"
        )
    return archives[0]


def _publish_latest(
    config: LeaderboardMonitorConfig,
    root: Path,
    archive: Path,
    polled_at: datetime,
    digest: dict[str, Any],
) -> None:
    """Replace the rolling copy of the newest raw archive."""
    latest = root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    temporary = latest / ("." + archive.name + ".tmp")
    shutil.move(str(archive), temporary)
    temporary.replace(latest / archive.name)
    atomic_write_json(
        latest / "manifest.json",
        {
            "schema_version": 1,
            "competition": config.competition,
            "captured_at_utc": _format_utc(polled_at),
            "source_command": (
                f"{config.kaggle_binary} competitions leaderboard "
                f"{config.competition} --download"
            ),
            "archive": archive.name,
            **digest,
        },
    )


def _write_store_manifest(config: LeaderboardMonitorConfig, root: Path) -> None:
    path = root / "store_manifest.json"
    if path.is_file():
        return
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "competition": config.competition,
            "created_at_utc": _utc_now(),
            "source_command": (
                f"{config.kaggle_binary} competitions leaderboard "
                f"{config.competition} --download"
            ),
            "columns": [field.name for field in store.SCHEMA],
            "layout": {
                "parts": "parts/<YYYYMMDD>/part-<HHMMSS>.parquet (current day)",
                "daily": "daily/leaderboard-<YYYYMMDD>.parquet (completed days)",
                "poll_log": store.POLL_LOG_NAME,
                "latest_raw": "latest/",
            },
        },
    )


def _install_signal_handlers() -> None:
    def _handler(signum: int, _frame: Any) -> None:
        raise _StopRequestedError(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("a leaderboard monitor is already running") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _utc_day() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


def _format_utc(moment: datetime) -> str:
    return (
        moment.astimezone(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def _utc_now() -> str:
    return _format_utc(datetime.now(UTC))


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="data/kaggle_leaderboard_monitor",
)
def main(hydra_config: DictConfig) -> None:
    """Hydra entry point."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to an object")
    run(LeaderboardMonitorConfig.model_validate(cast(dict[str, Any], raw)))


if __name__ == "__main__":
    main()


__all__ = ["LeaderboardMonitorConfig", "main", "run"]
