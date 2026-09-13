"""Static opponent schedules for configurable RL deck rosters."""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FixedFrozenPolicyConfig(BaseModel):
    """One immutable frozen pilot that is independent of opponent decks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_id: str
    checkpoint_path: Path
    weight: float = 1.0
    checkpoint_size_bytes: int | None = None
    checkpoint_sha256: str | None = None

    @field_validator("opponent_id")
    @classmethod
    def valid_opponent_id(cls, value: str) -> str:
        """Normalize and reject empty pilot identifiers."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("fixed frozen policy opponent_id must be non-empty")
        return cleaned

    @field_validator("weight")
    @classmethod
    def valid_weight(cls, value: float) -> float:
        """Require a finite positive pilot sampling weight."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("fixed frozen policy weight must be positive")
        return value

    @field_validator("checkpoint_size_bytes")
    @classmethod
    def valid_checkpoint_size(cls, value: int | None) -> int | None:
        """Reject invalid optional checkpoint sizes."""
        if value is not None and value < 0:
            raise ValueError("checkpoint_size_bytes must be non-negative")
        return value

    @field_validator("checkpoint_sha256")
    @classmethod
    def valid_checkpoint_sha256(cls, value: str | None) -> str | None:
        """Normalize and validate an optional checkpoint digest."""
        if value is None:
            return None
        cleaned = value.strip().lower()
        if len(cleaned) != 64 or any(
            char not in "0123456789abcdef" for char in cleaned
        ):
            raise ValueError("checkpoint_sha256 must be a 64-character hex digest")
        return cleaned

    @model_validator(mode="after")
    def valid_fingerprint(self) -> FixedFrozenPolicyConfig:
        """Require the optional checkpoint fingerprint as a complete pair."""
        has_size = self.checkpoint_size_bytes is not None
        has_sha = self.checkpoint_sha256 is not None
        if has_size != has_sha:
            raise ValueError("checkpoint size and SHA256 must be provided together")
        return self


class StaticOpponentScheduleConfig(BaseModel):
    """Fixed pilot weights and candidate-specific deck multipliers.

    The opponent deck pool supplies the base weights. A row overrides selected
    opponent multipliers for one candidate, while omitted rows and cells use
    ``default_matchup_multiplier``. This makes every configured deck covered by
    default when the roster changes, without hard-coding its cardinality.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    default_matchup_multiplier: float = 1.0
    matchup_multipliers: dict[str, dict[str, float]] = Field(default_factory=dict)

    @field_validator("default_matchup_multiplier")
    @classmethod
    def valid_default_multiplier(cls, value: float) -> float:
        """Require non-zero default coverage for every configured matchup."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("default matchup multiplier must be positive")
        return value

    @field_validator("matchup_multipliers")
    @classmethod
    def valid_matchup_multipliers(
        cls,
        value: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """Normalize labels and reject zero or invalid cell multipliers."""
        cleaned: dict[str, dict[str, float]] = {}
        for raw_candidate, raw_row in value.items():
            candidate = raw_candidate.strip()
            if not candidate:
                raise ValueError("static schedule candidate labels must be non-empty")
            if candidate in cleaned:
                raise ValueError(
                    f"duplicate normalized static schedule candidate: {candidate}"
                )
            row: dict[str, float] = {}
            for raw_opponent, raw_multiplier in raw_row.items():
                opponent = raw_opponent.strip()
                if not opponent:
                    raise ValueError(
                        "static schedule opponent labels must be non-empty"
                    )
                if opponent in row:
                    raise ValueError(
                        "duplicate normalized static schedule opponent: "
                        f"{candidate}/{opponent}"
                    )
                multiplier = float(raw_multiplier)
                if not math.isfinite(multiplier) or multiplier <= 0.0:
                    raise ValueError(
                        "static schedule matchup multipliers must be positive"
                    )
                row[opponent] = multiplier
            cleaned[candidate] = row
        return cleaned

    def matchup_multiplier(self, candidate: str, opponent: str) -> float:
        """Return the fixed multiplier for one candidate/opponent cell."""
        if not self.enabled:
            return 1.0
        return self.matchup_multipliers.get(candidate, {}).get(
            opponent,
            self.default_matchup_multiplier,
        )
