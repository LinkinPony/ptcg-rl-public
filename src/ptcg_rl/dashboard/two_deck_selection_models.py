"""Typed contracts for the two-submission deck portfolio proxy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ptcg_rl.dashboard.public_environment_models import EnvironmentWindow
from ptcg_rl.dashboard.training_deck_strength_models import TrainingEvidenceRange


class TwoDeckSelectionQuality(BaseModel):
    """Input completeness and identity gates for one recommendation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    training_available: bool
    public_environment_available: bool
    active_candidates: int = Field(ge=0)
    eligible_candidates: int = Field(ge=0)
    candidate_pairs: int = Field(ge=0)
    known_meta_mass: float = Field(ge=0.0, le=1.0)
    unknown_meta_mass: float = Field(ge=0.0, le=1.0)
    unexpanded_explicit_meta_mass: float = Field(ge=0.0, le=1.0)
    rare_unknown_meta_mass: float = Field(ge=0.0, le=1.0)
    warnings: tuple[str, ...]


class TwoDeckCandidateEvidence(BaseModel):
    """Training, public, and reweighted evidence for one exact deck."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_label: str
    deck_hash: str
    deck_digest: str
    display_name: str
    route_compatible: bool
    training_games: int = Field(ge=0)
    training_first_games: int = Field(ge=0)
    training_second_games: int = Field(ge=0)
    training_mean: float = Field(ge=0.0, le=1.0)
    training_credible_low: float = Field(ge=0.0, le=1.0)
    training_credible_high: float = Field(ge=0.0, le=1.0)
    public_evidence_status: Literal["eligible", "sparse"]
    public_games: int = Field(ge=0)
    public_deploy_mean: float = Field(ge=0.0, le=1.0)
    public_deploy_lcb: float = Field(ge=0.0, le=1.0)
    proxy_rank: int = Field(ge=1)
    proxy_mean: float = Field(ge=0.0, le=1.0)
    proxy_credible_low: float = Field(ge=0.0, le=1.0)
    proxy_credible_high: float = Field(ge=0.0, le=1.0)
    proxy_lcb: float = Field(ge=0.0, le=1.0)
    observed_meta_mass: float = Field(ge=0.0, le=1.0)
    prior_only_meta_mass: float = Field(ge=0.0, le=1.0)


class TwoDeckPairRecommendation(BaseModel):
    """One pair ranked by the posterior maximum of its two global scores."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    deck_a_label: str
    deck_a_hash: str
    deck_a_digest: str
    deck_a_display_name: str
    deck_b_label: str
    deck_b_hash: str
    deck_b_digest: str
    deck_b_display_name: str
    expected_best_score: float = Field(ge=0.0, le=1.0)
    credible_low: float = Field(ge=0.0, le=1.0)
    credible_high: float = Field(ge=0.0, le=1.0)
    best_score_lcb: float = Field(ge=0.0, le=1.0)
    probability_at_least_one_above_even: float = Field(ge=0.0, le=1.0)
    joint_downside_probability: float = Field(ge=0.0, le=1.0)
    diversification_gain: float = Field(ge=0.0, le=1.0)
    score_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)
    expected_regret: float = Field(ge=0.0, le=1.0)
    common_weak_meta_mass: float = Field(ge=0.0, le=1.0)
    shared_observed_meta_mass: float = Field(ge=0.0, le=1.0)


class TwoDeckSelectionPayload(BaseModel):
    """Artifact-bound recommendation for the competition's two tracked slots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    available: bool
    semantics: Literal["training_matchups_kaggle_daily_meta_portfolio_proxy_v1"] = (
        "training_matchups_kaggle_daily_meta_portfolio_proxy_v1"
    )
    run_id: str
    checkpoint_version: int | None = Field(default=None, ge=0)
    checkpoint_pair_manifest_sha256: str | None = None
    training_range: TrainingEvidenceRange
    training_started_at_utc: str | None = None
    training_ended_at_utc: str | None = None
    public_window_days: EnvironmentWindow
    public_as_of_date: str | None = None
    public_snapshot_fingerprint: str | None = None
    objective: Literal["expected_max_meta_weighted_score_proxy"] = (
        "expected_max_meta_weighted_score_proxy"
    )
    quality: TwoDeckSelectionQuality
    recommendation: TwoDeckPairRecommendation | None = None
    pairs: tuple[TwoDeckPairRecommendation, ...] = ()
    candidates: tuple[TwoDeckCandidateEvidence, ...] = ()
    warnings: tuple[str, ...] = ()


__all__ = [
    "TwoDeckCandidateEvidence",
    "TwoDeckPairRecommendation",
    "TwoDeckSelectionPayload",
    "TwoDeckSelectionQuality",
]
