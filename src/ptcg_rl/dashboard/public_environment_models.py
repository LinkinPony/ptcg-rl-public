"""Public API contracts for Kaggle Daily environment snapshots."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EnvironmentWindow = Literal[1, 2, 7, 14]
CoverageState = Literal["verified_responder", "evidence_blind_spot", "unresolved"]


class EnvironmentScoreSemantics(BaseModel):
    """Human-visible statistical contract for one snapshot."""

    model_config = ConfigDict(extra="allow", frozen=True)

    submission_data_used: bool = False
    primary: str | None = None
    meta_weighting: str | None = None
    draw_score: float | None = None
    non_done: str | None = None
    seat_weights: dict[str, float] = Field(default_factory=dict)


class EnvironmentQuality(BaseModel):
    """Completeness and evidence diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episodes: int = Field(default=0, ge=0)
    sides: int = Field(default=0, ge=0)
    valid_episodes: int = Field(default=0, ge=0)
    unresolved_episodes: int = Field(default=0, ge=0)
    source_missing_episodes: int = Field(default=0, ge=0)
    source_missing_bytes: int = Field(default=0, ge=0)
    explicit_meta_mass: float = Field(default=0.0, ge=0.0, le=1.0)
    unknown_tail_mass: float = Field(default=0.0, ge=0.0, le=1.0)
    eligible_roster_decks: int = Field(default=0, ge=0)
    roster_decks: int = Field(default=0, ge=0)


class EnvironmentOverview(BaseModel):
    """Global environment concentration summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exact_decks: int = Field(default=0, ge=0)
    pilot_keys: int = Field(default=0, ge=0)
    meta_hhi: float = Field(default=0.0, ge=0.0, le=1.0)
    top_10_share: float = Field(default=0.0, ge=0.0, le=1.0)


class EnvironmentMetaDeck(BaseModel):
    """One exact deck observed in the public Daily meta."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_hash: str | None
    deck_digest: str
    deck_label: str | None
    display_name: str | None = None
    active_roster: bool
    sides: int = Field(ge=0)
    share: float = Field(ge=0.0, le=1.0)
    early_share: float = Field(ge=0.0, le=1.0)
    late_share: float = Field(ge=0.0, le=1.0)
    share_delta: float
    new_in_late_half: bool
    unique_pilots: int = Field(ge=0)
    pilot_hhi: float = Field(ge=0.0, le=1.0)
    effective_pilots: float = Field(ge=0.0)


class EnvironmentRosterStanding(BaseModel):
    """One active training deck evaluated only from public Daily evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_label: str
    deck_hash: str
    deck_digest: str
    display_name: str | None = None
    family_id: str | None
    rank: int | None = Field(default=None, ge=1)
    evidence_status: Literal["eligible", "sparse"]
    environment_sides: int = Field(ge=0)
    valid_games: int = Field(ge=0)
    first_games: int = Field(ge=0)
    second_games: int = Field(ge=0)
    observed_score: float | None = Field(default=None, ge=0.0, le=1.0)
    deploy_mean: float = Field(ge=0.0, le=1.0)
    deploy_credible_low: float = Field(ge=0.0, le=1.0)
    deploy_credible_high: float = Field(ge=0.0, le=1.0)
    deploy_lcb: float = Field(ge=0.0, le=1.0)
    matchup_cvar_mean: float = Field(ge=0.0, le=1.0)
    probability_above_even: float = Field(ge=0.0, le=1.0)
    observed_meta_mass: float = Field(ge=0.0, le=1.0)
    prior_only_meta_mass: float = Field(ge=0.0, le=1.0)


class EnvironmentCoverageRow(BaseModel):
    """Evidence-gated roster coverage for one major public opponent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_deck_hash: str | None
    opponent_deck_digest: str
    opponent_deck_label: str | None
    meta_share: float = Field(ge=0.0, le=1.0)
    state: CoverageState
    best_candidate_deck_hash: str | None
    best_lcb: float | None = Field(default=None, ge=0.0, le=1.0)
    eligible_candidates: int = Field(ge=0)


class EnvironmentMatchupCell(BaseModel):
    """Seat-standardized candidate versus public exact-deck cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_deck_hash: str
    candidate_deck_digest: str
    opponent_deck_hash: str | None
    opponent_deck_digest: str
    opponent_deck_label: str | None
    games: int = Field(ge=0)
    first_games: int = Field(ge=0)
    second_games: int = Field(ge=0)
    evidence_eligible: bool
    posterior_mean: float = Field(ge=0.0, le=1.0)
    credible_low: float = Field(ge=0.0, le=1.0)
    credible_high: float = Field(ge=0.0, le=1.0)
    lcb: float = Field(ge=0.0, le=1.0)


class PublicEnvironmentPayload(BaseModel):
    """One immutable public-environment window snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    available: bool
    run_id: str
    checkpoint_version: int | None
    window_days: EnvironmentWindow
    as_of_date: str | None
    required_dates: tuple[str, ...] = ()
    missing_dates: tuple[str, ...] = ()
    generated_at_utc: str | None
    snapshot_fingerprint: str | None
    source_scope: str
    unavailable_reason: str | None = None
    score_semantics: EnvironmentScoreSemantics
    quality: EnvironmentQuality = Field(default_factory=EnvironmentQuality)
    overview: EnvironmentOverview = Field(default_factory=EnvironmentOverview)
    meta_decks: tuple[EnvironmentMetaDeck, ...] = ()
    roster_standings: tuple[EnvironmentRosterStanding, ...] = ()
    coverage: tuple[EnvironmentCoverageRow, ...] = ()
    matrix: tuple[EnvironmentMatchupCell, ...] = ()


class PublicEnvironmentMatrixPayload(BaseModel):
    """Large matchup matrix split from the summary response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    checkpoint_version: int | None
    window_days: EnvironmentWindow
    available: bool
    snapshot_fingerprint: str | None
    cells: tuple[EnvironmentMatchupCell, ...]


class PublicEnvironmentMatchupsPayload(PublicEnvironmentMatrixPayload):
    """Matrix subset for one authoritative roster deck hash."""

    deck_hash: str


__all__ = [
    "EnvironmentWindow",
    "PublicEnvironmentMatchupsPayload",
    "PublicEnvironmentMatrixPayload",
    "PublicEnvironmentPayload",
]
