"""Audit report for one settled fixed role-budget opponent window."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptcg_rl.rl.opponent_pool._identity import Sha256, canonical_fingerprint
from ptcg_rl.rl.opponent_pool.adaptive import AdaptiveCellScore, PortfolioName
from ptcg_rl.rl.opponent_pool.commit import OpponentOutcome
from ptcg_rl.rl.opponent_pool.models import CandidateSeat, QuotaStratum
from ptcg_rl.rl.opponent_pool.planner import QuotaWindowPlan
from ptcg_rl.rl.opponent_pool.role_budget import (
    ROLE_ORDER,
    RoleBudgetAllocationSnapshot,
    RoleBudgetName,
    role_for_stratum,
)
from ptcg_rl.rl.opponent_pool.state import LeagueState

_MatchupKey = tuple[str, str, int]
_EXECUTED = frozenset(("engine_terminal", "window_cutoff", "step_limit"))


class RoleBudgetCandidateReport(BaseModel):
    """Candidate target and settled allocation for one window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_deck_digest: Sha256
    base_share: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    target_share: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    posterior_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    worst_posterior_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    matchup_cells: int = Field(gt=0)
    evidence_cells: int = Field(ge=0)
    planned_games: int = Field(ge=0)
    actual_games: int = Field(ge=0)
    actual_decisions: int = Field(ge=0)


class RoleBudgetRoleReport(BaseModel):
    """Fixed target and actual mass for one unambiguous allocation reason."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: RoleBudgetName
    target_share: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    artifact_count: int = Field(gt=0)
    planned_games: int = Field(ge=0)
    planned_expected_decisions: float = Field(ge=0.0, allow_inf_nan=False)
    actual_games: int = Field(ge=0)
    actual_decisions: int = Field(ge=0)
    actual_decision_share: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class RoleBudgetArtifactReport(BaseModel):
    """Artifact-first target and observations, independent of route count."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: Sha256
    source_fingerprint: Sha256
    source_policy_version: int = Field(ge=0)
    stratum: QuotaStratum
    role: RoleBudgetName
    route_count: int = Field(gt=0)
    target_share: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    planned_games: int = Field(ge=0)
    actual_games: int = Field(ge=0)
    actual_decisions: int = Field(ge=0)


class RoleBudgetMatchupReport(BaseModel):
    """Exact route-seat evidence and target under its explicit role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_deck_digest: Sha256
    artifact_id: Sha256
    route_id: Sha256
    opponent_deck_digest: Sha256
    candidate_seat: CandidateSeat
    role: RoleBudgetName
    target_share: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    global_target_share: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    posterior_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    slow_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    posterior_stddev: float = Field(ge=0.0, allow_inf_nan=False)
    effective_evidence: float = Field(ge=0.0, allow_inf_nan=False)
    components: dict[PortfolioName, float]
    expected_decisions_per_game: float = Field(gt=0.0, allow_inf_nan=False)
    decision_mass: float = Field(ge=0.0, allow_inf_nan=False)
    planned_games: int = Field(ge=0)
    planned_expected_decisions: float = Field(ge=0.0, allow_inf_nan=False)
    actual_games: int = Field(ge=0)
    actual_decisions: int = Field(ge=0)

    @property
    def matchup_key(self) -> _MatchupKey:
        """Return the core exact-matchup key."""
        return (self.candidate_deck_digest, self.route_id, self.candidate_seat)


class RoleBudgetOpponentAllocationReport(BaseModel):
    """Detailed evidence that fixed budgets reached actual quota cells."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["role-budget-opponent-allocation-report-v1"] = (
        "role-budget-opponent-allocation-report-v1"
    )
    window_sequence: int = Field(ge=0)
    plan_id: Sha256
    target_fingerprint: Sha256
    predecessor_state_fingerprint: Sha256
    committed_state_fingerprint: Sha256
    revision_fingerprint: Sha256
    evidence_cells: int = Field(ge=0)
    low_evidence_cells: int = Field(ge=0)
    candidates: tuple[RoleBudgetCandidateReport, ...]
    roles: tuple[RoleBudgetRoleReport, ...]
    artifacts: tuple[RoleBudgetArtifactReport, ...]
    matchups: tuple[RoleBudgetMatchupReport, ...]

    @model_validator(mode="after")
    def canonical_rows(self) -> Self:
        """Keep all report tables deterministic and unambiguous."""
        candidate_keys = tuple(item.candidate_deck_digest for item in self.candidates)
        artifact_keys = tuple(item.artifact_id for item in self.artifacts)
        matchup_keys = tuple(item.matchup_key for item in self.matchups)
        if candidate_keys != tuple(sorted(candidate_keys)):
            raise ValueError("role-budget candidate rows are not canonical")
        if tuple(item.role for item in self.roles) != ROLE_ORDER:
            raise ValueError("role-budget role rows are incomplete")
        if artifact_keys != tuple(sorted(artifact_keys)):
            raise ValueError("role-budget artifact rows are not canonical")
        if matchup_keys != tuple(sorted(matchup_keys)) or len(matchup_keys) != len(
            set(matchup_keys)
        ):
            raise ValueError("role-budget matchup rows are not canonical")
        return self

    @property
    def fingerprint(self) -> str:
        """Return a stable report identity for status and audit consumers."""
        return canonical_fingerprint(
            "role-budget-opponent-allocation-report-v1",
            self.model_dump(mode="json"),
        )


def build_role_budget_allocation_report(
    *,
    predecessor_state_fingerprint: str,
    committed_state: LeagueState,
    committed_state_fingerprint: str,
    snapshot: RoleBudgetAllocationSnapshot,
    scores: Sequence[AdaptiveCellScore],
    matchup_decision_mass: Mapping[_MatchupKey, float],
    base_candidate_shares: Mapping[str, float],
    plan: QuotaWindowPlan,
    outcomes: Sequence[OpponentOutcome],
) -> RoleBudgetOpponentAllocationReport:
    """Join immutable targets, planned cells, and accepted observations."""
    score_by_key = {item.identity.matchup_key: item for item in scores}
    target_by_key = {item.matchup_key: item.share for item in snapshot.target_weights}
    active_keys = {item.key for item in committed_state.revision.active_matchups}
    if (
        set(score_by_key) != active_keys
        or set(target_by_key) != active_keys
        or set(matchup_decision_mass) != active_keys
    ):
        raise ValueError("role-budget report inputs differ from active matchups")

    artifacts = {item.artifact_id: item for item in committed_state.revision.artifacts}
    routes = {item.route_id: item for item in committed_state.revision.routes}
    entries = {item.artifact_id: item for item in committed_state.revision.entries}
    planned_games: Counter[_MatchupKey] = Counter()
    cells = {item.cell_index: item for item in plan.cells}
    for cell in plan.cells:
        planned_games[cell.matchup_key] += cell.game_count
    actual_games: Counter[_MatchupKey] = Counter()
    actual_decisions: Counter[_MatchupKey] = Counter()
    for outcome in outcomes:
        if outcome.status not in _EXECUTED:
            continue
        key = cells[outcome.assignment_index].matchup_key
        actual_games[key] += 1
        actual_decisions[key] += outcome.candidate_decisions

    matchup_rows: list[RoleBudgetMatchupReport] = []
    for key in sorted(active_keys):
        score = score_by_key[key]
        identity = score.identity
        stratum = entries[identity.artifact_id].stratum
        target = target_by_key[key]
        planned_expected = planned_games[key] * score.expected_decisions
        matchup_rows.append(
            RoleBudgetMatchupReport(
                candidate_deck_digest=identity.candidate_deck_digest,
                artifact_id=identity.artifact_id,
                route_id=identity.route_id,
                opponent_deck_digest=identity.opponent_deck_digest,
                candidate_seat=identity.candidate_seat,
                role=role_for_stratum(stratum),
                target_share=target,
                global_target_share=(
                    snapshot.candidate_target_shares[identity.candidate_deck_digest]
                    * target
                    / 2.0
                ),
                posterior_score=score.posterior_score,
                slow_score=score.slow_score,
                posterior_stddev=score.posterior_stddev,
                effective_evidence=score.effective_evidence,
                components=dict(score.components),
                expected_decisions_per_game=score.expected_decisions,
                decision_mass=matchup_decision_mass[key],
                planned_games=planned_games[key],
                planned_expected_decisions=planned_expected,
                actual_games=actual_games[key],
                actual_decisions=actual_decisions[key],
            )
        )

    grouped_candidates: defaultdict[str, list[RoleBudgetMatchupReport]] = defaultdict(
        list
    )
    grouped_artifacts: defaultdict[str, list[RoleBudgetMatchupReport]] = defaultdict(
        list
    )
    for row in matchup_rows:
        grouped_candidates[row.candidate_deck_digest].append(row)
        grouped_artifacts[row.artifact_id].append(row)
    candidates = tuple(
        RoleBudgetCandidateReport(
            candidate_deck_digest=candidate,
            base_share=base_candidate_shares[candidate],
            target_share=snapshot.candidate_target_shares[candidate],
            posterior_score=sum(row.posterior_score for row in rows) / len(rows),
            worst_posterior_score=min(row.posterior_score for row in rows),
            matchup_cells=len(rows),
            evidence_cells=sum(row.effective_evidence > 0.0 for row in rows),
            planned_games=sum(row.planned_games for row in rows),
            actual_games=sum(row.actual_games for row in rows),
            actual_decisions=sum(row.actual_decisions for row in rows),
        )
        for candidate, rows in sorted(grouped_candidates.items())
    )
    total_actual = sum(row.actual_decisions for row in matchup_rows)
    roles = tuple(
        RoleBudgetRoleReport(
            role=role,
            target_share=snapshot.role_target_mass[role],
            artifact_count=len(
                {row.artifact_id for row in matchup_rows if row.role == role}
            ),
            planned_games=sum(
                row.planned_games for row in matchup_rows if row.role == role
            ),
            planned_expected_decisions=sum(
                row.planned_expected_decisions
                for row in matchup_rows
                if row.role == role
            ),
            actual_games=sum(
                row.actual_games for row in matchup_rows if row.role == role
            ),
            actual_decisions=(
                decisions := sum(
                    row.actual_decisions for row in matchup_rows if row.role == role
                )
            ),
            actual_decision_share=(
                0.0 if total_actual == 0 else decisions / float(total_actual)
            ),
        )
        for role in ROLE_ORDER
    )
    route_counts = Counter(route.artifact_id for route in routes.values())
    artifact_rows = tuple(
        RoleBudgetArtifactReport(
            artifact_id=artifact_id,
            source_fingerprint=artifacts[artifact_id].source_fingerprint,
            source_policy_version=artifacts[artifact_id].source_policy_version,
            stratum=entries[artifact_id].stratum,
            role=role_for_stratum(entries[artifact_id].stratum),
            route_count=route_counts[artifact_id],
            target_share=sum(row.global_target_share for row in rows),
            planned_games=sum(row.planned_games for row in rows),
            actual_games=sum(row.actual_games for row in rows),
            actual_decisions=sum(row.actual_decisions for row in rows),
        )
        for artifact_id, rows in sorted(grouped_artifacts.items())
    )
    return RoleBudgetOpponentAllocationReport(
        window_sequence=plan.window_sequence,
        plan_id=plan.plan_id,
        target_fingerprint=snapshot.fingerprint,
        predecessor_state_fingerprint=predecessor_state_fingerprint,
        committed_state_fingerprint=committed_state_fingerprint,
        revision_fingerprint=committed_state.revision.fingerprint,
        evidence_cells=snapshot.evidence_cells,
        low_evidence_cells=snapshot.low_evidence_cells,
        candidates=candidates,
        roles=roles,
        artifacts=artifact_rows,
        matchups=tuple(matchup_rows),
    )


__all__ = [
    "RoleBudgetOpponentAllocationReport",
    "build_role_budget_allocation_report",
]
