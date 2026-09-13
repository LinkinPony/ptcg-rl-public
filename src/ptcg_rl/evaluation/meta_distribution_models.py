"""Validated configuration and result models for metagame weighting."""

from __future__ import annotations

import math
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RatingBandConfig(BaseModel):
    """Relative row weights for a target leaderboard-rating band."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: float | None = None
    maximum: float | None = None
    outside_weight: float = 0.0
    missing_weight: float = 1.0

    @field_validator("minimum", "maximum")
    @classmethod
    def finite_bound(cls, value: float | None) -> float | None:
        """Require finite optional rating bounds."""
        if value is not None and not math.isfinite(value):
            raise ValueError("rating bounds must be finite")
        return value

    @field_validator("outside_weight", "missing_weight")
    @classmethod
    def nonnegative_weight(cls, value: float) -> float:
        """Require finite non-negative relative weights."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("rating weights must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def ordered_bounds(self) -> RatingBandConfig:
        """Reject an inverted closed interval."""
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum rating must not exceed maximum rating")
        return self


class MetaDistributionConfig(BaseModel):
    """Configuration for metagame observation weighting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signature_column: str = "deck_signature"
    archetype_column: str = "deck_label"
    date_column: str | None = "date"
    team_column: str | None = "team_name"
    rating_column: str | None = "leaderboard_score"
    rating_band: RatingBandConfig | None = None
    recency_half_life_days: float | None = 7.0
    reference_date: date | None = None
    missing_date_weight: float = 1.0
    equalize_team_days: bool = True
    other_name: str = "__other__"
    parquet_batch_size: int = 65_536

    @field_validator("signature_column", "archetype_column", "other_name")
    @classmethod
    def nonempty_required_name(cls, value: str) -> str:
        """Reject blank required column and bucket names."""
        value = value.strip()
        if not value:
            raise ValueError("column and bucket names must be non-empty")
        return value

    @field_validator("date_column", "team_column", "rating_column")
    @classmethod
    def nonempty_optional_name(cls, value: str | None) -> str | None:
        """Reject blank optional column names."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("optional column names must be non-empty")
        return value

    @field_validator("recency_half_life_days")
    @classmethod
    def positive_half_life(cls, value: float | None) -> float | None:
        """Require a finite positive optional half-life."""
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError("recency_half_life_days must be positive")
        return value

    @field_validator("missing_date_weight")
    @classmethod
    def nonnegative_missing_date_weight(cls, value: float) -> float:
        """Require a finite non-negative missing-date weight."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("missing_date_weight must be non-negative")
        return value

    @field_validator("parquet_batch_size")
    @classmethod
    def positive_batch_size(cls, value: int) -> int:
        """Require a positive streaming batch size."""
        if value <= 0:
            raise ValueError("parquet_batch_size must be positive")
        return value


class VariantWeight(BaseModel):
    """Weight of one exact deck signature inside an archetype."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signature: str
    archetype: str
    raw_observations: int
    effective_observations: float
    conditional_weight: float
    meta_weight: float


class ArchetypeWeight(BaseModel):
    """Global metagame weight and exact-variant mixture for one archetype."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    archetype: str
    raw_observations: int
    effective_observations: float
    meta_weight: float
    variants: tuple[VariantWeight, ...]


class CoverageDiagnostics(BaseModel):
    """Coverage and data-quality diagnostics for a built distribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_rows: int
    rating_positive_weight_rows: int
    contributing_rows: int
    zero_weight_rows: int
    tracked_rows: int
    other_rows: int
    raw_coverage: float
    effective_coverage: float
    observed_signature_count: int
    tracked_signature_count: int
    other_observed_signature_count: int
    dated_rows: int
    missing_or_invalid_date_rows: int
    future_date_rows: int
    rating_in_band_rows: int
    rating_out_of_band_rows: int
    missing_or_invalid_rating_rows: int
    rows_with_team_day: int
    rows_without_team_day: int
    unique_team_days: int
    date_column_available: bool
    team_column_available: bool
    rating_column_available: bool
    recency_applied: bool
    team_day_equalization_applied: bool
    rating_band_applied: bool
    other_reason_counts: dict[str, int] = Field(default_factory=dict)


class MetaDistribution(BaseModel):
    """Normalized hierarchical metagame distribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_date: date | None
    total_effective_observations: float
    archetypes: tuple[ArchetypeWeight, ...]
    other_name: str
    other_mass: float
    diagnostics: CoverageDiagnostics

    def exact_variant_weights(self) -> dict[str, float]:
        """Return global weights keyed by tracked exact deck signature."""
        weights: dict[str, float] = {}
        for archetype in self.archetypes:
            if archetype.archetype == self.other_name:
                continue
            for variant in archetype.variants:
                weights[variant.signature] = (
                    weights.get(variant.signature, 0.0) + variant.meta_weight
                )
        return weights
