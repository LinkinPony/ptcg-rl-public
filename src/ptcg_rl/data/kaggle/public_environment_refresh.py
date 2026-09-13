"""End-to-end refresh coordinator for Kaggle Daily public environment data."""

from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.dashboard.repository import DashboardRepository
from ptcg_rl.data.kaggle import episode_archive
from ptcg_rl.data.kaggle.public_environment_analysis import build_snapshots
from ptcg_rl.data.kaggle.public_environment_ingest import (
    compact_required_dates,
    read_index,
    refresh_index,
    repo_path,
)
from ptcg_rl.data.kaggle.public_environment_models import PublicEnvironmentConfig
from ptcg_rl.rl.performance_state import atomic_write_json

REPO_ROOT = Path(__file__).resolve().parents[4]


def run(config: PublicEnvironmentConfig) -> dict[str, Any]:
    """Refresh compact corpus, score the selected roster, and archive legacy raw."""
    output_root = repo_path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    progress = _ProgressReporter(config.progress_path)
    progress.publish(
        phase="starting",
        message="正在初始化公开环境刷新",
        percent=0.0,
    )
    with _exclusive_lock(output_root / "refresh.lock"):
        try:
            run_id = _resolve_run_id(config.run_id)
            logger.info("refreshing public Daily environment run={}", run_id)
            progress.publish(
                phase="index",
                message="正在检查 Kaggle Daily 小型索引",
                completed=0,
                total=1,
                percent=2.0,
            )
            manifest_path, index_refreshed = refresh_index(config)
            index_rows = read_index(manifest_path)
            progress.publish(
                phase="index",
                message=("索引已刷新" if index_refreshed else "索引仍在节流期，已复用"),
                completed=1,
                total=1,
                percent=10.0,
            )
            corpus = compact_required_dates(
                config,
                index_rows,
                progress=progress.compaction,
            )
            latest_date = str(corpus["latest_date"])
            progress.publish(
                phase="snapshot",
                message="正在构建并校验 1/2/7/14 个完整日统计快照",
                percent=78.0,
                current_date=latest_date,
            )
            snapshots = build_snapshots(
                config,
                run_id=run_id,
                latest_date=latest_date,
            )
            progress.publish(
                phase="archive",
                message="正在校验并归档遗留原始回放",
                percent=92.0,
                current_date=latest_date,
            )
            archive_report = _archive_legacy_raw(config, corpus)
            progress.publish(
                phase="publishing",
                message="正在原子发布最新快照",
                percent=98.0,
                current_date=latest_date,
            )
            report = {
                "schema_version": 1,
                "completed_at_utc": _utc_now(),
                "run_id": run_id,
                "checkpoint_version": config.checkpoint_version,
                "index_refreshed": index_refreshed,
                "index_manifest": str(manifest_path.relative_to(REPO_ROOT)),
                "corpus": corpus,
                "snapshot": snapshots,
                "archive": archive_report,
            }
            atomic_write_json(output_root / "refresh_summary.json", report)
            progress.publish(
                phase="completed",
                message="公开环境快照刷新完成",
                completed=1,
                total=1,
                percent=100.0,
                current_date=latest_date,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return report
        except BaseException as error:
            progress.publish(
                phase="failed",
                message=f"{type(error).__name__}: {error}",
                percent=progress.percent,
            )
            raise


class _ProgressReporter:
    """Atomically publish bounded progress for an optional dashboard receipt."""

    def __init__(self, path: Path | None) -> None:
        self.path = repo_path(path) if path is not None else None
        self.percent = 0.0

    def publish(
        self,
        *,
        phase: str,
        message: str,
        percent: float,
        completed: int = 0,
        total: int = 0,
        current_date: str | None = None,
    ) -> None:
        """Publish one monotonic progress observation."""
        self.percent = max(self.percent, min(100.0, percent))
        if self.path is None:
            return
        atomic_write_json(
            self.path,
            {
                "schema_version": 1,
                "phase": phase,
                "message": message,
                "completed": completed,
                "total": total,
                "percent": self.percent,
                "current_date": current_date,
                "updated_at_utc": _utc_now(),
            },
        )

    def compaction(
        self,
        completed: int,
        total: int,
        current_date: str,
        status: str,
    ) -> None:
        """Map per-date compaction progress into the overall refresh range."""
        percent = 10.0 + 65.0 * completed / max(1, total)
        message = (
            f"正在处理 {current_date}（可能包含下载与流式压实）"
            if status == "processing"
            else f"{current_date} 已{('复用' if status == 'reused' else '压实')}"
        )
        self.publish(
            phase="compaction",
            message=message,
            completed=completed,
            total=total,
            percent=percent,
            current_date=current_date,
        )


def _resolve_run_id(value: str) -> str:
    repository = DashboardRepository.from_config(
        REPO_ROOT / "configs" / "rl" / "dashboard.yaml",
        repo_root=REPO_ROOT,
    )
    return repository.resolve_run_id(value)


def _archive_legacy_raw(
    config: PublicEnvironmentConfig,
    corpus: dict[str, Any],
) -> dict[str, Any] | None:
    if not config.remove_legacy_raw_after_archive:
        return None
    compacted_dates = {
        str(action["date"]) for action in cast(list[dict[str, Any]], corpus["actions"])
    }
    replay_root = repo_path(config.replay_root)
    raw_dates = (
        sorted(
            path.name
            for path in replay_root.iterdir()
            if path.is_dir()
            and path.name in compacted_dates
            and any(path.glob("*.json"))
        )
        if replay_root.is_dir()
        else []
    )
    if not raw_dates:
        return None
    return episode_archive.run(
        episode_archive.EpisodeArchiveConfig(
            replay_root=config.replay_root,
            archive_root=config.archive_root,
            index_manifest=config.index_dir / "manifest.csv",
            dates=raw_dates,
            keep_raw_dates=0,
            remove_source=True,
            compression_level=config.archive_compression_level,
        )
    )


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "a public-environment refresh is already running"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="data/kaggle_public_environment",
)
def main(hydra_config: DictConfig) -> None:
    """Hydra entry point."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to an object")
    run(PublicEnvironmentConfig.model_validate(cast(dict[str, Any], raw)))


if __name__ == "__main__":
    main()


__all__ = ["main", "run"]
