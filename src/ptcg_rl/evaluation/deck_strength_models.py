"""Configuration and intermediate models for deck-strength scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.meta_distribution import MetaDistributionConfig
from ptcg_rl.evaluation.posterior import PosteriorEvaluationConfig


class DeckStrengthConfig(BaseModel):
    """Hydra-backed configuration for bundle strength post-processing."""

    model_config = ConfigDict(extra="forbid")

    games_paths: tuple[Path, ...]
    side_observations_path: Path
    output_dir: Path = Path("outputs/evaluation/deck_strength/latest")
    meta: MetaDistributionConfig = Field(default_factory=MetaDistributionConfig)
    posterior: PosteriorEvaluationConfig = Field(
        default_factory=PosteriorEvaluationConfig
    )
    pilot_weights: dict[str, float] = Field(default_factory=dict)
    robust_meta_concentration: float = 1_000_000_000.0
    require_balanced_seats: bool = True
    read_batch_size: int = 65_536
    compression: str = "zstd"
    min_effective_meta_coverage: float = 0.8
    max_unresolved_fraction: float = 0.01
    max_prior_only_meta_mass: float = 0.1
    require_explicit_multi_pilot_weights: bool = True

    @field_validator("games_paths")
    @classmethod
    def nonempty_game_paths(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        """Require at least one independently produced game shard."""
        if not value:
            raise ValueError("games_paths must contain at least one Parquet file")
        return value

    @field_validator("pilot_weights")
    @classmethod
    def valid_pilot_weights(cls, value: dict[str, float]) -> dict[str, float]:
        """Normalize pilot ids and reject ambiguous or invalid weights."""
        output: dict[str, float] = {}
        for raw_pilot, raw_weight in value.items():
            pilot = str(raw_pilot).strip()
            weight = float(raw_weight)
            if not pilot:
                raise ValueError("pilot_weights contains an empty pilot id")
            if pilot in output:
                raise ValueError(f"duplicate normalized pilot id: {pilot!r}")
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("pilot weights must be finite and non-negative")
            output[pilot] = weight
        return output

    @field_validator("robust_meta_concentration")
    @classmethod
    def positive_robust_concentration(cls, value: float) -> float:
        """Keep the robust support effectively uniform in posterior draws."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("robust_meta_concentration must be positive")
        return value

    @field_validator("read_batch_size")
    @classmethod
    def positive_batch_size(cls, value: int) -> int:
        """Require bounded, positive streaming batches."""
        if value <= 0:
            raise ValueError("read_batch_size must be positive")
        return value

    @field_validator(
        "min_effective_meta_coverage",
        "max_unresolved_fraction",
        "max_prior_only_meta_mass",
    )
    @classmethod
    def probability_threshold(cls, value: float) -> float:
        """Validate diagnostic reference probabilities."""
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("quality thresholds must be between zero and one")
        return value

    @field_validator("compression")
    @classmethod
    def nonempty_compression(cls, value: str) -> str:
        """Reject blank Parquet codec names."""
        value = value.strip()
        if not value:
            raise ValueError("compression must be non-empty")
        return value

    @model_validator(mode="after")
    def unique_resolved_game_paths(self) -> DeckStrengthConfig:
        """Prevent accidental double counting through path aliases."""
        resolved = [
            records.repo_path(path).expanduser().resolve() for path in self.games_paths
        ]
        if len(set(resolved)) != len(resolved):
            raise ValueError(
                "games_paths contains the same resolved file more than once"
            )
        return self


@dataclass(frozen=True)
class MetaAllocation:
    """Opponent-bundle allocation of the exact-signature target meta."""

    weights: Mapping[str, float]
    rows: tuple[Mapping[str, Any], ...]
    default_pilot_signatures: tuple[str, ...]
    implicit_multi_pilot_signatures: tuple[str, ...]


__all__ = ["DeckStrengthConfig", "MetaAllocation"]
