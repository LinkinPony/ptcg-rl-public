"""Validated public models for the RL performance dashboard."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WindowName = Literal["cumulative", "15m", "60m"]
ScopeName = Literal["segment", "lineage"]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class DashboardLineageConfig(BaseModel):
    """Ordered runtime segments belonging to one logical training lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    display_name: str
    segments: tuple[str, ...]

    @field_validator("display_name")
    @classmethod
    def non_empty_display_name(cls, value: str) -> str:
        """Reject empty lineage labels."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("lineage display_name must be non-empty")
        return cleaned

    @field_validator("segments")
    @classmethod
    def valid_segments(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require unique safe run directory names."""
        if not value or len(value) != len(set(value)):
            raise ValueError("lineage segments must be non-empty and unique")
        for segment in value:
            _safe_run_id(segment)
        return value


class DashboardConfig(BaseModel):
    """Local run discovery and presentation settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_root: str = "outputs/training/rl"
    current_run: str
    refresh_seconds: int = 15
    league_database_path: str = "outputs/evaluation/continuous_league/league.sqlite3"
    league_coordinator_url: str = "http://127.0.0.1:8788"
    deck_aliases: dict[str, str] = Field(default_factory=dict)
    family_aliases: dict[str, str] = Field(default_factory=dict)
    deck_catalogs: tuple[str, ...] = ()
    lineages: dict[str, DashboardLineageConfig] = Field(default_factory=dict)

    @field_validator("current_run")
    @classmethod
    def valid_current_run(cls, value: str) -> str:
        """Require one safe current run identifier."""
        return _safe_run_id(value)

    @field_validator("refresh_seconds")
    @classmethod
    def valid_refresh_seconds(cls, value: int) -> int:
        """Reject non-positive browser refresh intervals."""
        if value <= 0:
            raise ValueError("refresh_seconds must be positive")
        return value

    @field_validator("league_database_path")
    @classmethod
    def valid_league_database_path(cls, value: str) -> str:
        """Keep the default league database inside the repository workspace."""
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("league_database_path must be a safe relative path")
        return path.as_posix()

    @field_validator("league_coordinator_url")
    @classmethod
    def valid_league_coordinator_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("league_coordinator_url must be an HTTP URL")
        return cleaned

    @field_validator("deck_catalogs")
    @classmethod
    def valid_deck_catalogs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Keep deck-name catalogs inside the repository config boundary."""
        for raw_path in value:
            path = Path(raw_path)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("deck_catalogs must contain safe relative paths")
        return value

    @field_validator("family_aliases")
    @classmethod
    def valid_family_aliases(cls, value: dict[str, str]) -> dict[str, str]:
        """Require stable family identities and useful presentation labels."""
        for family_id, display_name in value.items():
            if _SHA256_PATTERN.fullmatch(family_id) is None:
                raise ValueError("family_alias keys must be full lowercase SHA256 IDs")
            if not display_name.strip():
                raise ValueError("family_alias values must be non-empty")
        return value

    @model_validator(mode="after")
    def current_run_has_at_most_one_lineage(self) -> Self:
        """Keep lineage aggregation unambiguous."""
        memberships = [
            name
            for name, lineage in self.lineages.items()
            if self.current_run in lineage.segments
        ]
        if len(memberships) > 1:
            raise ValueError("current_run belongs to more than one lineage")
        return self


class OutcomeStats(BaseModel):
    """One outcome aggregate with both raw and draw-aware rates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    games: int
    wins: int
    draws: int
    losses: int
    win_rate: float | None
    score_rate: float | None


class RunInfo(BaseModel):
    """One locally discoverable training run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    display_name: str
    current: bool
    lineage_id: str | None
    schema_version: int
    updated_at_utc: str | None
    minute_index: int
    data_state: Literal["ready", "stale", "invalid"]
    detail: str | None = None


class DeckRow(BaseModel):
    """Controller-sliced performance for one target deck."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_label: str
    deck_hash: str
    display_name: str
    slices: dict[str, OutcomeStats]


class PerformanceTable(BaseModel):
    """Deck table and pooled totals for one selected time range."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    scope: ScopeName
    window: WindowName
    started_at_utc: str | None
    ended_at_utc: str | None
    source_segments: tuple[str, ...]
    overall: dict[str, OutcomeStats]
    decks: tuple[DeckRow, ...]
    exact_dimensions_available: bool


class SeriesPoint(BaseModel):
    """One time-series point paired with its sample count."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minute_index: int
    ended_at_utc: str
    games: int
    win_rate: float | None
    score_rate: float | None


class DeckSeries(BaseModel):
    """One target deck's rolling performance series."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_label: str
    deck_hash: str
    display_name: str
    points: tuple[SeriesPoint, ...]


class MatchupRow(BaseModel):
    """Exact candidate/opponent/controller/seat aggregate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_deck_label: str
    candidate_deck_hash: str | None
    candidate_display_name: str
    opponent_kind: str
    opponent_deck_label: str
    opponent_deck_hash: str | None
    opponent_display_name: str
    opponent_id: str
    candidate_seat: int
    outcomes: OutcomeStats


class HealthPayload(BaseModel):
    """Normalized operational status plus data-freshness diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    data_age_seconds: float | None
    data_state: Literal["ready", "stale", "invalid"]
    status: dict[str, object]
    warnings: tuple[str, ...] = ()


def _safe_run_id(value: str) -> str:
    cleaned = value.strip()
    if not cleaned or cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise ValueError("run identifiers must be one non-empty path component")
    return cleaned
