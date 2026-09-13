"""Typed contracts for the training decision workbench API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptcg_rl.dashboard.models import OutcomeStats, RunInfo, WindowName
from ptcg_rl.rl.learner_metric_history import LearnerMetricRecord
from ptcg_rl.rl.training_game_statistics import TrainingGameStatistics


class CheckpointInfo(BaseModel):
    """One locally present immutable checkpoint pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    version: int = Field(ge=0)
    pair_manifest_sha256: str
    policy_sha256: str
    learner_state_sha256: str
    model_config_fingerprint: str | None = None
    training_roster_fingerprint: str | None = None
    exact_registry_fingerprint: str | None = None
    active_exact_deck_digests: tuple[str, ...] = ()
    updated_at_utc: str
    metric_available: bool
    current: bool


class PosteriorStats(BaseModel):
    """Seat-balanced W/D/L posterior evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observed: OutcomeStats
    seat_games: dict[int, int]
    seat_balanced: bool
    posterior_mean: float | None = Field(default=None, ge=0.0, le=1.0)
    credible_low: float | None = Field(default=None, ge=0.0, le=1.0)
    credible_high: float | None = Field(default=None, ge=0.0, le=1.0)
    probability_above_half: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_state: Literal["ready", "missing_seat", "unavailable"]


class DeckEvidence(BaseModel):
    """One candidate deck's live training-pool evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_label: str
    deck_hash: str
    display_name: str
    posterior: PosteriorStats
    controller_scores: dict[str, OutcomeStats]
    opponent_count: int = Field(ge=0)


class DeckEvidencePayload(BaseModel):
    """Posterior deck standings for one exact segment and time window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    window: WindowName
    semantics: Literal["training_pool_wdl_posterior_v1"]
    overall: PosteriorStats
    decks: tuple[DeckEvidence, ...]
    exact_dimensions_available: bool


class MatchupEvidence(BaseModel):
    """One exact opponent identity retained under a candidate deck."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_kind: str
    opponent_deck_label: str
    opponent_display_name: str
    opponent_id: str
    posterior: PosteriorStats


class DeckMatchupPayload(BaseModel):
    """Focused matchup evidence that never returns the full global matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    candidate_deck_label: str
    window: WindowName
    matchups: tuple[MatchupEvidence, ...]


class WorkbenchAlert(BaseModel):
    """Actionable health conclusion derived from observed telemetry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: Literal["info", "warning", "critical"]
    title: str
    detail: str


class ProgressSummary(BaseModel):
    """Normalized learner progress without freezing profile defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    update_index: int | None = Field(default=None, ge=0)
    target_updates: int | None = Field(default=None, ge=0)
    optimizer_step_index: int | None = Field(default=None, ge=0)
    decisions_seen: int | None = Field(default=None, ge=0)
    target_decisions: int | None = Field(default=None, ge=0)
    kept_decisions_per_second: float | None = Field(default=None, ge=0.0)


class CurrentLearnerSnapshot(BaseModel):
    """Small latest-value snapshot complementing durable history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    loss: float | None = None
    policy_loss: float | None = None
    value_loss: float | None = None
    belief_loss: float | None = None
    entropy: float | None = None
    approximate_kl: float | None = None
    clip_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    gradient_norm: float | None = Field(default=None, ge=0.0)
    learning_rate: float | None = Field(default=None, ge=0.0)
    collection_seconds: float | None = Field(default=None, ge=0.0)
    learner_seconds: float | None = Field(default=None, ge=0.0)
    checkpoint_seconds: float | None = Field(default=None, ge=0.0)
    cuda_peak_allocated_bytes: int | None = Field(default=None, ge=0)
    cuda_peak_reserved_bytes: int | None = Field(default=None, ge=0)
    fragments_stale: int | None = Field(default=None, ge=0)


class WorkbenchSummary(BaseModel):
    """The compact answer to whether one run needs attention now."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: RunInfo
    data_age_seconds: float | None = Field(default=None, ge=0.0)
    data_state: Literal["ready", "stale", "invalid"]
    warnings: tuple[str, ...]
    checkpoint: CheckpointInfo | None
    training_games: TrainingGameStatistics
    progress: ProgressSummary
    latest_learner: CurrentLearnerSnapshot
    evidence: PosteriorStats
    alerts: tuple[WorkbenchAlert, ...]


class LearnerSeriesPayload(BaseModel):
    """Artifact-bound checkpoint learner history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    records: tuple[LearnerMetricRecord, ...]
    complete: bool
    warning: str | None = None


class ComparisonReference(BaseModel):
    """One run/checkpoint selected for side-by-side comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    checkpoint_version: int | None = Field(default=None, ge=0)


class ComparisonRequest(BaseModel):
    """A bounded local comparison query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    references: tuple[ComparisonReference, ...]

    @model_validator(mode="after")
    def valid_reference_count(self) -> ComparisonRequest:
        """Keep comparison output readable and avoid repeated work."""
        if not 2 <= len(self.references) <= 4:
            raise ValueError("comparison requires between two and four references")
        identities = {
            (reference.run_id, reference.checkpoint_version)
            for reference in self.references
        }
        if len(identities) != len(self.references):
            raise ValueError("comparison references must be unique")
        return self


class ComparisonRow(BaseModel):
    """Comparable evidence for one selected run/checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference: ComparisonReference
    checkpoint: CheckpointInfo | None
    learner_metric: LearnerMetricRecord | None
    training_pool: PosteriorStats
    compatible_group: str | None
    warnings: tuple[str, ...]


class ComparisonPayload(BaseModel):
    """Bounded side-by-side comparison with compatibility diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: tuple[ComparisonRow, ...]
    all_compatible: bool
