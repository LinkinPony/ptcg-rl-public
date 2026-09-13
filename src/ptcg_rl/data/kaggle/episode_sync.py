"""Sync Kaggle top-episode datasets for local replay analysis."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.data.kaggle import episode_archive

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INDEX_DATASET = "kaggle/pokemon-tcg-ai-battle-episodes-index"
_TEMP_DIRS: list[tempfile.TemporaryDirectory[str]] = []


class EpisodeSyncConfig(BaseModel):
    """Hydra-backed config for syncing Kaggle top-episode replay data."""

    model_config = ConfigDict(extra="forbid")

    source_index: str = DEFAULT_INDEX_DATASET
    index_dir: Path = Path("data/external/kaggle_top_episodes_index/latest")
    replay_root: Path = Path("data/external/kaggle_top_episodes_daily")
    dates: list[str] = Field(default_factory=list)
    date_selection: Literal["latest_missing", "all_missing", "latest"] = (
        "latest_missing"
    )
    refresh_index: bool = True
    dry_run: bool = False
    force: bool = False
    kaggle_binary: str = "kaggle"
    archive_after_sync: bool = False
    archive_root: Path = Path("data/external/kaggle_top_episodes_archives")
    archive_keep_raw_dates: int = 2
    archive_compression_level: int = 1
    zstd_binary: str = "zstd"

    @field_validator("dates")
    @classmethod
    def valid_dates(cls, values: list[str]) -> list[str]:
        """Reject malformed date strings early."""
        for value in values:
            datetime.strptime(value, "%Y-%m-%d")
        return values

    @field_validator("archive_keep_raw_dates")
    @classmethod
    def valid_archive_keep_raw_dates(cls, value: int) -> int:
        """Reject a negative raw replay retention count."""
        if value < 0:
            raise ValueError("archive_keep_raw_dates must be non-negative")
        return value

    @field_validator("archive_compression_level")
    @classmethod
    def valid_archive_compression_level(cls, value: int) -> int:
        """Restrict compression to regular zstd levels."""
        if value < 1 or value > 19:
            raise ValueError("archive_compression_level must be in [1, 19]")
        return value


def run(config: EpisodeSyncConfig) -> dict[str, Any]:
    """Refresh the index and download selected daily episode datasets."""
    start = time.perf_counter()
    logger.info(
        "starting Kaggle episode sync index_dir={} replay_root={} dates={} "
        "date_selection={} refresh_index={} dry_run={} force={}",
        _display_path(_repo_path(config.index_dir)),
        _display_path(_repo_path(config.replay_root)),
        config.dates or "auto",
        config.date_selection,
        config.refresh_index,
        config.dry_run,
        config.force,
    )
    _ensure_kaggle_available(config.kaggle_binary)
    manifest_path = _load_manifest_path(config)
    manifest_rows = _read_manifest(manifest_path)
    targets = _target_rows(manifest_rows, config)
    logger.info(
        "selected Kaggle episode datasets count={} dates={}",
        len(targets),
        [row["date"] for row in targets],
    )
    actions: list[dict[str, Any]] = []

    for row in targets:
        action = _sync_daily_dataset(row, config)
        actions.append(action)

    archive_report = None
    if config.archive_after_sync:
        archive_report = episode_archive.run(
            episode_archive.EpisodeArchiveConfig(
                replay_root=config.replay_root,
                archive_root=config.archive_root,
                index_manifest=config.index_dir / "manifest.csv",
                keep_raw_dates=config.archive_keep_raw_dates,
                compression_level=config.archive_compression_level,
                zstd_binary=config.zstd_binary,
                dry_run=config.dry_run,
            )
        )

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "index_manifest": _display_path(manifest_path),
        "manifest_rows": len(manifest_rows),
        "selected_dates": [row["date"] for row in targets],
        "actions": actions,
        "archive_report": archive_report,
    }
    if not config.dry_run:
        index_dir = _repo_path(config.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        _write_json(index_dir / "sync_summary.json", report)
    elapsed = time.perf_counter() - start
    logger.info(
        "finished Kaggle episode sync selected_dates={} actions={} seconds={:.2f}",
        report["selected_dates"],
        [action["status"] for action in actions],
        elapsed,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))
    return report


def _load_manifest_path(config: EpisodeSyncConfig) -> Path:
    if config.dry_run:
        existing_manifest = _repo_path(config.index_dir) / "manifest.csv"
        if existing_manifest.exists() and not config.refresh_index:
            return existing_manifest
        temp_dir = tempfile.TemporaryDirectory(prefix="pokemon_tcg_episode_index_")
        temp_path = Path(temp_dir.name)
        _download_index_manifest(config, temp_path)
        _TEMP_DIRS.append(temp_dir)
        return temp_path / "manifest.csv"

    index_dir = _repo_path(config.index_dir)
    manifest_path = index_dir / "manifest.csv"
    if config.refresh_index or not manifest_path.exists():
        index_dir.mkdir(parents=True, exist_ok=True)
        _download_index_manifest(config, index_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.csv not found after index refresh: {index_dir}"
        )
    return manifest_path


def _download_index_manifest(config: EpisodeSyncConfig, output_dir: Path) -> None:
    logger.info(
        "downloading Kaggle episode index dataset={} output_dir={}",
        config.source_index,
        _display_path(output_dir),
    )
    _run_command(
        [
            config.kaggle_binary,
            "datasets",
            "download",
            config.source_index,
            "-f",
            "manifest.csv",
            "-p",
            str(output_dir),
            "--unzip",
        ]
    )


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    required = {
        "date",
        "daily_dataset_slug",
        "daily_dataset_url",
        "episode_count",
        "total_bytes",
    }
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
    return rows


def _target_rows(
    manifest_rows: list[dict[str, str]],
    config: EpisodeSyncConfig,
) -> list[dict[str, str]]:
    if config.dates:
        by_date = {row["date"]: row for row in manifest_rows}
        missing_dates = [date for date in config.dates if date not in by_date]
        if missing_dates:
            raise ValueError(f"dates not present in manifest: {missing_dates}")
        return [by_date[date] for date in config.dates]

    if not manifest_rows:
        return []
    if config.date_selection == "latest":
        return [manifest_rows[-1]]

    missing_rows = [
        row
        for row in manifest_rows
        if not _date_is_available(row["date"], config)
    ]
    if config.date_selection == "all_missing":
        return missing_rows
    if not missing_rows:
        return []
    return [missing_rows[-1]]


def _sync_daily_dataset(
    row: dict[str, str],
    config: EpisodeSyncConfig,
) -> dict[str, Any]:
    date = row["date"]
    target_dir = _repo_path(config.replay_root) / date
    dataset_ref = _daily_dataset_ref(row)
    already_present = _date_is_available(date, config)
    if already_present and not config.force:
        logger.info(
            "skipping existing Kaggle episode dataset date={} target_dir={}",
            date,
            _display_path(target_dir),
        )
        return {
            "date": date,
            "dataset": dataset_ref,
            "status": "skipped_existing",
            "target_dir": _display_path(target_dir),
            "storage": "raw" if _date_has_replays(target_dir) else "archive",
        }
    if config.dry_run:
        logger.info(
            "planned Kaggle episode dataset download date={} dataset={} "
            "target_dir={} episode_count={} total_bytes={}",
            date,
            dataset_ref,
            _display_path(target_dir),
            _int_or_none(row.get("episode_count")),
            _int_or_none(row.get("total_bytes")),
        )
        return {
            "date": date,
            "dataset": dataset_ref,
            "status": "planned_download",
            "target_dir": _display_path(target_dir),
            "episode_count": _int_or_none(row.get("episode_count")),
            "total_bytes": _int_or_none(row.get("total_bytes")),
        }

    target_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    logger.info(
        "downloading Kaggle episode dataset date={} dataset={} target_dir={} "
        "expected_episodes={} expected_bytes={}",
        date,
        dataset_ref,
        _display_path(target_dir),
        _int_or_none(row.get("episode_count")),
        _int_or_none(row.get("total_bytes")),
    )
    command = [
        config.kaggle_binary,
        "datasets",
        "download",
        dataset_ref,
        "-p",
        str(target_dir),
        "--unzip",
    ]
    if config.force:
        command.append("--force")
    _run_command(command)
    json_files = len(list(target_dir.glob("*.json")))
    logger.info(
        "downloaded Kaggle episode dataset date={} json_files={} seconds={:.2f}",
        date,
        json_files,
        time.perf_counter() - start,
    )
    return {
        "date": date,
        "dataset": dataset_ref,
        "status": "downloaded",
        "target_dir": _display_path(target_dir),
        "json_files": json_files,
    }


def _daily_dataset_ref(row: dict[str, str]) -> str:
    slug = row["daily_dataset_slug"]
    if "/" in slug:
        return slug
    return f"kaggle/{slug}"


def _date_has_replays(date_dir: Path) -> bool:
    return date_dir.is_dir() and any(date_dir.glob("*.json"))


def _date_is_available(date: str, config: EpisodeSyncConfig) -> bool:
    date_dir = _repo_path(config.replay_root) / date
    return _date_has_replays(date_dir) or episode_archive.archive_is_available(
        config.archive_root,
        date,
    )


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _ensure_kaggle_available(kaggle_binary: str) -> None:
    if shutil.which(kaggle_binary) is None:
        raise RuntimeError(f"Kaggle CLI binary not found: {kaggle_binary}")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "actions": report["actions"],
        "archive_summary": (
            report["archive_report"]["summary"]
            if report["archive_report"] is not None
            else None
        ),
        "index_manifest": report["index_manifest"],
        "manifest_rows": report["manifest_rows"],
        "selected_dates": report["selected_dates"],
    }


def _int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="data/kaggle_episode_sync",
)
def main(hydra_config: DictConfig) -> None:
    """Hydra entry point."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = EpisodeSyncConfig.model_validate(cast(dict[str, Any], raw_config))
    run(config)


if __name__ == "__main__":
    main()
