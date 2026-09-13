"""Validated configuration for recent Kaggle BC source selection."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RecentBCSelectionConfig(BaseModel):
    """Immutable source and quality gates for one recent BC selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dates: tuple[str, ...]
    processed_root: Path
    package_root: Path
    output_dir: Path
    run_resolved_config_path: Path
    rank10_team_deck_summary_path: Path
    rank10_collection_manifest_path: Path
    card_data_csv: Path
    leaderboard_path: Path | None = None
    leaderboard_top_fraction: float = Field(default=0.1, gt=0.0, le=1.0)
    leaderboard_max_rank: int = Field(default=20, ge=1)
    leaderboard_min_score: float = 1100.0
    require_group_quality: bool = True
    require_roster_coverage: bool = False
    write_pretraining_source: bool = False
    min_group_sides: int = Field(default=24, ge=1)
    min_sides_per_seat: int = Field(default=6, ge=1)
    min_dates: int = Field(default=2, ge=1)
    min_wilson_lcb: float = Field(default=0.55, ge=0.0, le=1.0)
    wilson_z: float = Field(default=1.6448536269514722, gt=0.0)
    validation_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    split_seed: int = 20260815
    validate_selected_replays: bool = True

    @field_validator("leaderboard_min_score")
    @classmethod
    def finite_leaderboard_score(cls, value: float) -> float:
        """Reject a score threshold that cannot define a cohort."""
        if not math.isfinite(value):
            raise ValueError("leaderboard_min_score must be finite")
        return value

    @model_validator(mode="after")
    def coherent_training_source(self) -> Self:
        """Training-ready output must be exact-routed and replay-validated."""
        if self.write_pretraining_source and (
            not self.require_roster_coverage
            or not self.validate_selected_replays
            or self.leaderboard_path is None
        ):
            raise ValueError(
                "pretraining source output requires leaderboard gating, exact "
                "roster coverage, and replay validation"
            )
        return self

    @field_validator("dates")
    @classmethod
    def valid_dates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require a stable ascending list of distinct ISO dates."""
        if not value or tuple(sorted(set(value))) != value:
            raise ValueError("dates must be non-empty, unique, and sorted")
        for item in value:
            datetime.strptime(item, "%Y-%m-%d")
        return value


__all__ = ["RecentBCSelectionConfig"]
