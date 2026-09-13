"""Typed API contracts for opponent-allocation observability."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ptcg_rl.rl.opponent_pool.adaptive import PortfolioName
from ptcg_rl.rl.opponent_pool.adaptive_report import (
    AdaptivePortfolioAllocationReport,
)
from ptcg_rl.rl.opponent_pool.role_budget import RoleBudgetName
from ptcg_rl.rl.opponent_pool.role_budget_report import RoleBudgetRoleReport

AllocationMode = Literal["adaptive", "role_budget"]

AllocationSort = Literal[
    "debt",
    "weakness",
    "uncertainty",
    "target",
    "change",
    "planned_games",
]


class OpponentAllocationExecutionView(BaseModel):
    """Live native quota execution state for the active collect window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    window_id: str
    window_sequence: int = Field(ge=0)
    window_state: Literal["idle", "collecting", "committed", "aborted"]
    exposure_cohort_games: int = Field(gt=0)
    initial_wave_workers_issued: int = Field(ge=0)
    initial_wave_workers_total: int = Field(gt=0)
    assignment_pool_issued: int = Field(ge=0)
    assignment_pool_total: int = Field(gt=0)
    target_decisions: int = Field(gt=0)
    accepted_decisions: int = Field(ge=0)
    provisional_decisions: int = Field(ge=0)
    inflight_decision_credit: int = Field(ge=0)
    shards_issued: int = Field(ge=0)
    shards_completed: int = Field(ge=0)


class OpponentAllocationCandidateView(BaseModel):
    """Candidate target enriched with stable human-facing deck identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_deck_digest: str
    candidate_deck_label: str | None
    candidate_deck_hash: str | None
    candidate_display_name: str
    base_share: float
    target_share: float
    previous_target_share: float | None = None
    posterior_score: float
    worst_posterior_score: float
    matchup_cells: int
    evidence_cells: int
    planned_games: int
    actual_games: int
    actual_decisions: int


class OpponentAllocationArtifactView(BaseModel):
    """One active artifact under either supported allocation policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    source_fingerprint: str
    source_policy_version: int = Field(ge=0)
    stratum: str
    role: RoleBudgetName | None = None
    route_count: int = Field(gt=0)
    target_share: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    posterior_score: float | None = Field(default=None, ge=0.0, le=1.0)
    worst_posterior_score: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_cells: int | None = Field(default=None, ge=0)
    planned_games: int = Field(ge=0)
    planned_expected_decisions: float | None = Field(default=None, ge=0.0)
    actual_games: int = Field(ge=0)
    actual_decisions: int = Field(ge=0)


class OpponentAllocationSummaryPayload(BaseModel):
    """Compact current-window allocation view; exact cells are paged separately."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    run_id: str
    allocation_mode: AllocationMode | None = None
    recorded_at_utc: str | None = None
    detail: str | None = None
    window_sequence: int | None = Field(default=None, ge=0)
    plan_id: str | None = None
    target_fingerprint: str | None = None
    predecessor_state_fingerprint: str | None = None
    committed_state_fingerprint: str | None = None
    revision_fingerprint: str | None = None
    evidence_cells: int = Field(default=0, ge=0)
    low_evidence_cells: int = Field(default=0, ge=0)
    matchup_count: int = Field(default=0, ge=0)
    execution: OpponentAllocationExecutionView | None = None
    candidates: tuple[OpponentAllocationCandidateView, ...] = ()
    portfolios: tuple[AdaptivePortfolioAllocationReport, ...] = ()
    roles: tuple[RoleBudgetRoleReport, ...] = ()
    artifacts: tuple[OpponentAllocationArtifactView, ...] = ()


class OpponentAllocationMatchupView(BaseModel):
    """Exact allocator cell with stable candidate/opponent display identities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_deck_digest: str
    candidate_deck_label: str | None
    candidate_deck_hash: str | None
    candidate_display_name: str
    artifact_id: str
    source_fingerprint: str
    source_policy_version: int = Field(ge=0)
    stratum: str
    route_id: str
    opponent_deck_digest: str
    opponent_deck_label: str | None
    opponent_deck_hash: str | None
    opponent_display_name: str
    candidate_seat: Literal[0, 1]
    portfolio: PortfolioName | None = None
    role: RoleBudgetName | None = None
    target_share: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    global_target_share: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    previous_target_share: float | None = Field(default=None, gt=0.0, le=1.0)
    target_delta: float | None = None
    posterior_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    slow_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    posterior_stddev: float = Field(ge=0.0, allow_inf_nan=False)
    effective_evidence: float = Field(ge=0.0, allow_inf_nan=False)
    utility: float | None = Field(default=None, gt=0.0)
    components: dict[PortfolioName, float]
    expected_decisions_per_game: float = Field(gt=0.0, allow_inf_nan=False)
    decision_mass: float = Field(ge=0.0, allow_inf_nan=False)
    normalized_decision_mass: float | None = Field(default=None, ge=0.0)
    decision_debt_before: float | None = None
    projected_decision_debt: float | None = None
    planned_games: int = Field(ge=0)
    planned_expected_decisions: float = Field(ge=0.0, allow_inf_nan=False)
    actual_games: int = Field(ge=0)
    actual_decisions: int = Field(ge=0)
    actual_score_sum: float | None = Field(default=None, ge=0.0)


class OpponentAllocationMatchupPage(BaseModel):
    """Server-filtered page over the detailed exact-cell artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    run_id: str
    allocation_mode: AllocationMode | None = None
    window_sequence: int | None = Field(default=None, ge=0)
    target_fingerprint: str | None = None
    total: int = Field(default=0, ge=0)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(gt=0)
    sort: AllocationSort
    candidate_deck_digest: str | None = None
    artifact_id: str | None = None
    portfolio: PortfolioName | None = None
    role: RoleBudgetName | None = None
    candidate_seat: Literal[0, 1] | None = None
    rows: tuple[OpponentAllocationMatchupView, ...] = ()


__all__ = [
    "AllocationMode",
    "AllocationSort",
    "OpponentAllocationArtifactView",
    "OpponentAllocationExecutionView",
    "OpponentAllocationCandidateView",
    "OpponentAllocationMatchupPage",
    "OpponentAllocationMatchupView",
    "OpponentAllocationSummaryPayload",
]
