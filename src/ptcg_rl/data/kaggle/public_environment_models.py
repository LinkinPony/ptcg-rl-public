"""Validated configuration for compact Kaggle public-environment analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.evaluation.posterior import PosteriorEvaluationConfig

WindowDays = Literal[1, 2, 7, 14]


class PublicEnvironmentConfig(BaseModel):
    """Configuration shared by refresh, compaction, and snapshot scoring."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = "current"
    checkpoint_version: int | None = Field(default=None, ge=0)
    source_index: str = "kaggle/pokemon-tcg-ai-battle-episodes-index"
    index_dir: Path = Path("data/external/kaggle_top_episodes_index/latest")
    package_root: Path = Path("data/external/kaggle_top_episodes_packages")
    replay_root: Path = Path("data/external/kaggle_top_episodes_daily")
    archive_root: Path = Path("data/external/kaggle_top_episodes_archives")
    processed_root: Path = Path("data/processed/kaggle_public_environment")
    output_root: Path = Path("outputs/kaggle_public_environment")
    progress_path: Path | None = None
    run_root: Path = Path("outputs/training/rl")
    card_data_csv: Path = Path("data/EN_Card_Data.csv")
    kaggle_binary: str = "kaggle"
    network_enabled: bool = True
    network_min_interval_hours: float = Field(default=6.0, gt=0.0)
    bootstrap_days: int = Field(default=7, ge=1)
    windows: tuple[WindowDays, ...] = (1, 2, 7, 14)
    prefix_bytes: int = Field(default=262_144, ge=65_536)
    max_prefix_bytes: int = Field(default=4_194_304, ge=65_536)
    parquet_batch_rows: int = Field(default=2_048, ge=2)
    min_explicit_opponent_sides: int = Field(default=20, ge=1)
    min_rank_sides: int = Field(default=20, ge=1)
    min_rank_sides_per_seat: int = Field(default=5, ge=1)
    min_matchup_sides: int = Field(default=10, ge=1)
    min_matchup_sides_per_seat: int = Field(default=2, ge=1)
    top_meta_decks: int = Field(default=50, ge=1)
    matchup_sample_count: int = Field(default=8_192, ge=1_000)
    remove_legacy_raw_after_archive: bool = True
    archive_compression_level: int = Field(default=1, ge=1, le=19)
    posterior: PosteriorEvaluationConfig = Field(
        default_factory=PosteriorEvaluationConfig
    )

    @field_validator("windows")
    @classmethod
    def valid_windows(cls, value: tuple[WindowDays, ...]) -> tuple[WindowDays, ...]:
        """Require unique supported windows in ascending order."""
        if tuple(sorted(set(value))) != value:
            raise ValueError("windows must be unique and sorted")
        return value

    @model_validator(mode="after")
    def valid_prefixes(self) -> PublicEnvironmentConfig:
        """Keep bounded prefix retries monotonic."""
        if self.max_prefix_bytes < self.prefix_bytes:
            raise ValueError("max_prefix_bytes must be at least prefix_bytes")
        return self


__all__ = ["PublicEnvironmentConfig", "WindowDays"]
