"""Validated models for live Kaggle top-team replay analysis."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ReplaySource = Literal["daily_reuse", "live_cache", "live_download"]
SideSource = Literal[
    "inventory_daily",
    "inventory_live",
    "inventory_inferred",
    "daily_discovered",
    "live_discovered",
]
Result = Literal["win", "loss", "draw", "other"]


class TargetTeam(BaseModel):
    """One leaderboard cohort team and its active public submissions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    team_id: str
    team_name: str
    submission_ids: tuple[int, ...]

    @field_validator("team_id", "team_name")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        """Reject empty identities."""
        value = value.strip()
        if not value:
            raise ValueError("team identities must be non-empty")
        return value

    @field_validator("submission_ids")
    @classmethod
    def valid_submission_ids(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        """Require a non-empty, duplicate-free active-submission set."""
        if not values or any(value <= 0 for value in values):
            raise ValueError("submission_ids must contain positive identifiers")
        if len(values) != len(set(values)):
            raise ValueError("submission_ids must not contain duplicates")
        return values


class TopTeamLiveConfig(BaseModel):
    """Hydra-backed config for a fixed-window top-team replay cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    competition: str
    start_time_utc: datetime
    end_time_utc: datetime
    daily_replay_root: Path
    replay_output_dir: Path
    output_dir: Path
    card_data_csv: Path
    download_workers: int = 8
    kaggle_binary: str = "kaggle"
    teams: tuple[TargetTeam, ...]

    @field_validator("competition", "kaggle_binary")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        """Reject empty command and competition strings."""
        value = value.strip()
        if not value:
            raise ValueError("strings must be non-empty")
        return value

    @field_validator("start_time_utc", "end_time_utc")
    @classmethod
    def utc_datetime(cls, value: datetime) -> datetime:
        """Require explicit timezones and normalize boundaries to UTC."""
        if value.tzinfo is None:
            raise ValueError("window timestamps must include a timezone")
        return value.astimezone(UTC)

    @field_validator("download_workers")
    @classmethod
    def positive_workers(cls, value: int) -> int:
        """Reject an unusable worker count."""
        if value <= 0:
            raise ValueError("download_workers must be positive")
        return value

    @field_validator("teams")
    @classmethod
    def non_empty_teams(cls, values: tuple[TargetTeam, ...]) -> tuple[TargetTeam, ...]:
        """Require at least one target team."""
        if not values:
            raise ValueError("teams must not be empty")
        return values

    @model_validator(mode="after")
    def valid_cohort(self) -> Self:
        """Reject ambiguous windows, names, ids, and submission ownership."""
        if self.start_time_utc >= self.end_time_utc:
            raise ValueError("start_time_utc must precede end_time_utc")
        ids = [team.team_id for team in self.teams]
        names = [team.team_name.casefold() for team in self.teams]
        submissions = [value for team in self.teams for value in team.submission_ids]
        if len(ids) != len(set(ids)):
            raise ValueError("team_id values must be unique")
        if len(names) != len(set(names)):
            raise ValueError("team_name values must be case-insensitively unique")
        if len(submissions) != len(set(submissions)):
            raise ValueError("a submission_id may belong to only one target team")
        return self


class EpisodeAssociation(BaseModel):
    """Target teams and submissions associated with one inventory episode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    team_id: str
    team_name: str
    submission_ids: tuple[int, ...]
    player_index: int = Field(ge=0, le=1)
    opponent_team_name: str
    reward: float | None
    agent_state: str


class InventoryEpisode(BaseModel):
    """One completed public episode after cross-submission deduplication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: int = Field(gt=0)
    create_time_utc: datetime
    end_time_utc: datetime
    state: str
    episode_type: str
    associations: tuple[EpisodeAssociation, ...]


class ManifestEpisode(InventoryEpisode):
    """One verified inventory replay and its content identity."""

    source: ReplaySource
    replay_path: Path
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReplayManifest(BaseModel):
    """Atomic replay manifest for one fixed inventory window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    created_at_utc: datetime
    competition: str
    start_time_utc: datetime
    end_time_utc: datetime
    submission_ids: tuple[int, ...]
    episode_count: int = Field(ge=0)
    association_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    episodes: tuple[ManifestEpisode, ...]


class SideObservation(BaseModel):
    """One target-team player-side observation with an exact deck."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: int = Field(gt=0)
    create_time_utc: datetime | None
    end_time_utc: datetime | None
    date: str
    team_id: str
    team_name: str
    player_index: int
    opponent_team_name: str
    reward: float | None
    status: str
    result: Result
    deck_signature: str
    deck_hash: str
    deck_label: str
    deck_ids: tuple[int, ...]
    unique_card_ids: int
    total_cards: int
    pokemon_summary: str
    top_cards: str
    opponent_deck_signature: str
    opponent_deck_hash: str
    opponent_deck_label: str
    submission_ids: tuple[int, ...]
    source: SideSource
    replay_source: ReplaySource
    replay_path: Path
    size_bytes: int = Field(gt=0)
    deck_evidence_episode_id: int = Field(gt=0)
    exact_replay: bool
