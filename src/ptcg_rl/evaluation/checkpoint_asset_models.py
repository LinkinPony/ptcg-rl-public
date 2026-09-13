"""Typed inputs for checkpoint-selection asset preparation."""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator


class CheckpointAssetInput(BaseModel):
    """Raw checkpoint and immutable FP16/protocol output paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_tag: str
    raw_checkpoint_path: Path
    asset_path: Path
    protocol_archive_path: Path

    @field_validator("checkpoint_tag")
    @classmethod
    def nonempty_tag(cls, value: str) -> str:
        """Reject blank checkpoint tags."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("checkpoint_tag must be non-empty")
        return normalized


class CheckpointAssetAuditConfig(BaseModel):
    """Hydra-backed export, replay parity, and submission validation config."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    checkpoints: tuple[CheckpointAssetInput, ...]
    deck_path: Path
    belief_summary_path: Path
    static_features_path: Path
    replay_glob: str
    team_name: str
    max_decisions: int = 200
    decisions_per_phase_per_replay: int = 2
    early_turn_max: int = 3
    mid_turn_max: int = 9
    near_tie_logit_margin: float = 1.0e-3
    min_top1_match_rate: float = 0.99
    output_manifest_path: Path
    output_parquet_path: Path

    @field_validator("experiment_id", "team_name", "replay_glob")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        """Reject blank audit identity fields."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("asset audit text fields must be non-empty")
        return normalized

    @field_validator("checkpoints")
    @classmethod
    def unique_checkpoints(
        cls, value: tuple[CheckpointAssetInput, ...]
    ) -> tuple[CheckpointAssetInput, ...]:
        """Require at least one unique tag and output path."""
        tags = [item.checkpoint_tag for item in value]
        assets = [item.asset_path for item in value]
        if not tags:
            raise ValueError("checkpoints must not be empty")
        if len(set(tags)) != len(tags) or len(set(assets)) != len(assets):
            raise ValueError("checkpoint tags and asset paths must be unique")
        return value

    @field_validator(
        "max_decisions",
        "decisions_per_phase_per_replay",
    )
    @classmethod
    def positive_count(cls, value: int) -> int:
        """Require positive replay sample counts."""
        if value <= 0:
            raise ValueError("asset audit sample counts must be positive")
        return value

    @field_validator("near_tie_logit_margin")
    @classmethod
    def nonnegative_margin(cls, value: float) -> float:
        """Require a finite non-negative tie margin."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("near_tie_logit_margin must be finite and non-negative")
        return value

    @field_validator("min_top1_match_rate")
    @classmethod
    def valid_match_rate(cls, value: float) -> float:
        """Restrict the parity reference to the unit interval."""
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("min_top1_match_rate must be in [0, 1]")
        return value


__all__ = ["CheckpointAssetAuditConfig", "CheckpointAssetInput"]
