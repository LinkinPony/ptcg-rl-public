"""Typed contracts for observed training deck-strength evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ptcg_rl.dashboard.models import OutcomeStats

TrainingEvidenceRange = Literal[
    "checkpoint",
    "recent_15m",
    "recent_60m",
    "cumulative",
]
TrainingController = Literal[
    "all",
    "self_play",
    "sentinel",
    "adaptive_history",
    "legacy_unknown",
    "frozen",
    "scripted",
    "stationary",
]
OpponentSet = Literal["active", "all"]


class TrainingPosterior(BaseModel):
    """Draw-aware posterior that preserves the observed sampling distribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observed: OutcomeStats
    posterior_mean: float | None = Field(default=None, ge=0.0, le=1.0)
    credible_low: float | None = Field(default=None, ge=0.0, le=1.0)
    credible_high: float | None = Field(default=None, ge=0.0, le=1.0)
    probability_above_half: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_state: Literal["ready", "unavailable"]


class TrainingMatchupSummary(BaseModel):
    """Compact opponent-deck aggregate used in the decision rail."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_deck_label: str
    opponent_deck_hash: str | None
    opponent_display_name: str
    outcomes: OutcomeStats


class TrainingDeckStanding(BaseModel):
    """One exact candidate deck under one independently selected evidence range."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int | None = Field(default=None, ge=1)
    deck_label: str
    deck_hash: str
    deck_digest: str | None = None
    display_name: str
    family_id: str | None = None
    family_display_name: str | None = None
    route_compatible: bool | None = None
    posterior: TrainingPosterior
    controller_scores: dict[str, OutcomeStats]
    seat_scores: dict[int, OutcomeStats]
    opponent_count: int = Field(ge=0)
    strongest_matchup: TrainingMatchupSummary | None = None
    weakest_matchup: TrainingMatchupSummary | None = None


class TrainingDeckFamily(BaseModel):
    """One shared strategy family containing one or more exact decks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family_id: str
    display_name: str
    deck_labels: tuple[str, ...]


class TrainingEvidenceRangeMetadata(BaseModel):
    """Selection provenance for one evidence range."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    range: TrainingEvidenceRange
    available: bool
    unavailable_reason: str | None = None
    started_at_utc: str | None = None
    ended_at_utc: str | None = None
    windows_considered: int = Field(ge=0)
    windows_selected: int = Field(ge=0)
    mixed_version_windows_excluded: int = Field(ge=0)
    unknown_version_windows_excluded: int = Field(ge=0)


class TrainingEvidenceRangeStandings(BaseModel):
    """Compact standings plus the exact source selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: TrainingEvidenceRangeMetadata
    standings: tuple[TrainingDeckStanding, ...]


class TrainingDeckStrengthPayload(BaseModel):
    """All four independently ranked observed-training evidence ranges."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    checkpoint_version: int | None = Field(default=None, ge=0)
    checkpoint_pair_manifest_sha256: str | None = None
    semantics: Literal["observed_training_distribution_wdl_posterior_v1"]
    source_warning: str
    exact_dimensions_available: bool
    active_deck_labels: tuple[str, ...]
    families: tuple[TrainingDeckFamily, ...]
    stationary_opponent_kinds: tuple[str, ...]
    ranges: dict[TrainingEvidenceRange, TrainingEvidenceRangeStandings]
    warnings: tuple[str, ...]


class TrainingSeriesPoint(BaseModel):
    """One immutable training window point with its sample count."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minute_index: int
    ended_at_utc: str
    games: int = Field(ge=0)
    score_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class TrainingDeckSeries(BaseModel):
    """Observed window series for one active candidate deck."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_label: str
    deck_hash: str
    display_name: str
    points: tuple[TrainingSeriesPoint, ...]


class TrainingDeckSeriesPayload(BaseModel):
    """Range- and controller-bound multi-deck trend evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    range: TrainingEvidenceRange
    checkpoint_version: int | None = Field(default=None, ge=0)
    controller: TrainingController
    available: bool
    unavailable_reason: str | None = None
    series: tuple[TrainingDeckSeries, ...]


class TrainingMatrixCell(BaseModel):
    """Candidate-deck by opponent-deck aggregate; pilot identities are separate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_deck_label: str
    candidate_deck_hash: str
    candidate_display_name: str
    opponent_deck_label: str
    opponent_deck_hash: str | None
    opponent_display_name: str
    pilot_count: int = Field(ge=0)
    posterior: TrainingPosterior


class TrainingDeckMatrixPayload(BaseModel):
    """Filtered matchup heatmap evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    range: TrainingEvidenceRange
    checkpoint_version: int | None = Field(default=None, ge=0)
    controller: TrainingController
    candidate_seat: int | None = Field(default=None, ge=0, le=1)
    opponent_set: OpponentSet
    available: bool
    unavailable_reason: str | None = None
    cells: tuple[TrainingMatrixCell, ...]


class TrainingMatchupDetail(BaseModel):
    """Exact deck/opponent/pilot/seat evidence retained for drill-down."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_kind: str
    opponent_stratum: str
    opponent_deck_label: str
    opponent_deck_hash: str | None
    opponent_display_name: str
    opponent_id: str
    candidate_seat: int = Field(ge=0, le=1)
    posterior: TrainingPosterior


class TrainingMatchupSeatSummary(BaseModel):
    """Filtered matchup evidence aggregated for one candidate seat."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_seat: int = Field(ge=0, le=1)
    posterior: TrainingPosterior


class TrainingMatchupStratumSummary(BaseModel):
    """Filtered matchup evidence aggregated for one opponent stratum."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_kind: str
    opponent_stratum: str
    pilot_count: int = Field(ge=0)
    posterior: TrainingPosterior


class TrainingOpponentMatchupSummary(BaseModel):
    """One exact opponent deck with pilot identities retained below it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_deck_label: str
    opponent_deck_hash: str | None
    opponent_display_name: str
    pilot_count: int = Field(ge=0)
    posterior: TrainingPosterior
    seat_breakdown: tuple[TrainingMatchupSeatSummary, ...]
    stratum_breakdown: tuple[TrainingMatchupStratumSummary, ...]


class TrainingDeckMatchupsPayload(BaseModel):
    """Focused exact-identity evidence for one candidate deck."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    candidate_deck_label: str
    range: TrainingEvidenceRange
    checkpoint_version: int | None = Field(default=None, ge=0)
    controller: TrainingController
    candidate_seat: int | None = Field(default=None, ge=0, le=1)
    opponent_set: OpponentSet
    available: bool
    unavailable_reason: str | None = None
    overall: TrainingPosterior
    opponent_deck_count: int = Field(ge=0)
    pilot_count: int = Field(ge=0)
    seat_breakdown: tuple[TrainingMatchupSeatSummary, ...]
    stratum_breakdown: tuple[TrainingMatchupStratumSummary, ...]
    opponents: tuple[TrainingOpponentMatchupSummary, ...]
    matchups: tuple[TrainingMatchupDetail, ...]
