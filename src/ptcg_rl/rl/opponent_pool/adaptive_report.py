"""Bounded observability artifact for one settled adaptive allocation window."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptcg_rl.rl.opponent_pool._identity import Sha256, canonical_fingerprint
from ptcg_rl.rl.opponent_pool.adaptive import (
    AdaptiveAllocationSnapshot,
    AdaptiveCellScore,
    PortfolioName,
)
from ptcg_rl.rl.opponent_pool.commit import OpponentOutcome
from ptcg_rl.rl.opponent_pool.models import CandidateSeat, QuotaStratum
from ptcg_rl.rl.opponent_pool.planner import QuotaWindowPlan
from ptcg_rl.rl.opponent_pool.state import LeagueState

_MatchupKey = tuple[str, str, int]
_EXECUTED_STATUSES = frozenset(("engine_terminal", "window_cutoff", "step_limit"))


class AdaptiveCandidateAllocationReport(BaseModel):
    """Joint target and observed window mass for one candidate deck."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_deck_digest: Sha256
    base_share: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    target_share: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    previous_target_share: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    posterior_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    worst_posterior_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    matchup_cells: int = Field(gt=0)
    evidence_cells: int = Field(ge=0)
    planned_games: int = Field(ge=0)
    actual_games: int = Field(ge=0)
    actual_decisions: int = Field(ge=0)


class AdaptivePortfolioAllocationReport(BaseModel):
    """Target and settled mass attributed to one learning objective."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    portfolio: PortfolioName
    target_share: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    planned_games: int = Field(ge=0)
    planned_expected_decisions: float = Field(ge=0.0, allow_inf_nan=False)
    actual_games: int = Field(ge=0)
    actual_decisions: int = Field(ge=0)
    actual_decision_share: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class AdaptiveArtifactAllocationReport(BaseModel):
    """Active frozen-policy inventory and its aggregate allocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: Sha256
    source_fingerprint: Sha256
    source_policy_version: int = Field(ge=0)
    stratum: QuotaStratum
    route_count: int = Field(gt=0)
    target_share: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    posterior_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    worst_posterior_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    evidence_cells: int = Field(ge=0)
    planned_games: int = Field(ge=0)
    planned_expected_decisions: float = Field(ge=0.0, allow_inf_nan=False)
    actual_games: int = Field(ge=0)
    actual_decisions: int = Field(ge=0)


class AdaptiveMatchupAllocationReport(BaseModel):
    """One exact route-seat cell with target, evidence, and debt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_deck_digest: Sha256
    artifact_id: Sha256
    source_fingerprint: Sha256
    source_policy_version: int = Field(ge=0)
    stratum: QuotaStratum
    route_id: Sha256
    opponent_deck_digest: Sha256
    candidate_seat: CandidateSeat
    portfolio: PortfolioName
    target_share: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    global_target_share: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    previous_target_share: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    target_delta: float | None = Field(default=None, allow_inf_nan=False)
    posterior_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    slow_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    posterior_stddev: float = Field(ge=0.0, allow_inf_nan=False)
    effective_evidence: float = Field(ge=0.0, allow_inf_nan=False)
    utility: float = Field(gt=0.0, allow_inf_nan=False)
    components: dict[PortfolioName, float]
    expected_decisions_per_game: float = Field(gt=0.0, allow_inf_nan=False)
    decision_mass: float = Field(ge=0.0, allow_inf_nan=False)
    normalized_decision_mass: float = Field(ge=0.0, allow_inf_nan=False)
    decision_debt_before: float = Field(allow_inf_nan=False)
    projected_decision_debt: float = Field(allow_inf_nan=False)
    planned_games: int = Field(ge=0)
    planned_expected_decisions: float = Field(ge=0.0, allow_inf_nan=False)
    actual_games: int = Field(ge=0)
    actual_decisions: int = Field(ge=0)
    actual_score_sum: float = Field(ge=0.0, allow_inf_nan=False)

    @property
    def matchup_key(self) -> _MatchupKey:
        """Return the normalized planner key."""
        return (self.candidate_deck_digest, self.route_id, self.candidate_seat)


class AdaptiveOpponentAllocationReport(BaseModel):
    """Immutable detailed view written separately from the hot status file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["adaptive-opponent-allocation-report-v1"] = (
        "adaptive-opponent-allocation-report-v1"
    )
    window_sequence: int = Field(ge=0)
    plan_id: Sha256
    target_fingerprint: Sha256
    predecessor_state_fingerprint: Sha256
    committed_state_fingerprint: Sha256
    revision_fingerprint: Sha256
    evidence_cells: int = Field(ge=0)
    low_evidence_cells: int = Field(ge=0)
    candidates: tuple[AdaptiveCandidateAllocationReport, ...]
    portfolios: tuple[AdaptivePortfolioAllocationReport, ...]
    artifacts: tuple[AdaptiveArtifactAllocationReport, ...]
    matchups: tuple[AdaptiveMatchupAllocationReport, ...]

    @model_validator(mode="after")
    def canonical_rows(self) -> Self:
        """Keep the detail artifact deterministic and safe to page."""
        candidate_keys = tuple(item.candidate_deck_digest for item in self.candidates)
        artifact_keys = tuple(item.artifact_id for item in self.artifacts)
        matchup_keys = tuple(item.matchup_key for item in self.matchups)
        if candidate_keys != tuple(sorted(candidate_keys)):
            raise ValueError("adaptive candidate report rows are not canonical")
        if artifact_keys != tuple(sorted(artifact_keys)):
            raise ValueError("adaptive artifact report rows are not canonical")
        if matchup_keys != tuple(sorted(matchup_keys)):
            raise ValueError("adaptive matchup report rows are not canonical")
        if len(matchup_keys) != len(set(matchup_keys)):
            raise ValueError("adaptive matchup report rows are duplicated")
        return self

    @property
    def fingerprint(self) -> str:
        """Return a stable identity for API caching and audit links."""
        return canonical_fingerprint(
            "adaptive-opponent-allocation-report-v1",
            self.model_dump(mode="json"),
        )


def build_adaptive_allocation_report(
    *,
    predecessor_state_fingerprint: str,
    committed_state: LeagueState,
    committed_state_fingerprint: str,
    snapshot: AdaptiveAllocationSnapshot,
    scores: Sequence[AdaptiveCellScore],
    previous_targets: Mapping[_MatchupKey, float],
    previous_candidate_targets: Mapping[str, float],
    matchup_decision_mass: Mapping[_MatchupKey, float],
    base_candidate_shares: Mapping[str, float],
    plan: QuotaWindowPlan,
    outcomes: Sequence[OpponentOutcome],
) -> AdaptiveOpponentAllocationReport:
    """Join pre-window evidence, immutable quotas, and settled observations."""
    if snapshot.window_sequence != plan.window_sequence:
        raise ValueError("adaptive report target and plan clocks differ")
    score_by_key = {item.identity.matchup_key: item for item in scores}
    target_by_key = {item.matchup_key: item.share for item in snapshot.target_weights}
    active_keys = {item.key for item in committed_state.revision.active_matchups}
    if set(score_by_key) != active_keys or set(target_by_key) != active_keys:
        raise ValueError("adaptive report inputs differ from the committed revision")
    if set(matchup_decision_mass) != active_keys:
        raise ValueError("adaptive report decision mass is incomplete")

    artifacts = {item.artifact_id: item for item in committed_state.revision.artifacts}
    routes = {item.route_id: item for item in committed_state.revision.routes}
    entries = {item.artifact_id: item for item in committed_state.revision.entries}
    route_artifacts = {
        route_id: artifacts[route.artifact_id] for route_id, route in routes.items()
    }
    planned_games: Counter[_MatchupKey] = Counter()
    planned_expected: defaultdict[_MatchupKey, float] = defaultdict(float)
    cells = {item.cell_index: item for item in plan.cells}
    for cell in plan.cells:
        key = cell.matchup_key
        planned_games[key] += cell.game_count
        planned_expected[key] += cell.game_count * score_by_key[key].expected_decisions

    actual_games: Counter[_MatchupKey] = Counter()
    actual_decisions: Counter[_MatchupKey] = Counter()
    actual_scores: defaultdict[_MatchupKey, float] = defaultdict(float)
    for outcome in outcomes:
        if outcome.status not in _EXECUTED_STATUSES:
            continue
        key = cells[outcome.assignment_index].matchup_key
        actual_games[key] += 1
        actual_decisions[key] += outcome.candidate_decisions
        if outcome.candidate_score is not None:
            actual_scores[key] += outcome.candidate_score

    mass_by_candidate_seat: defaultdict[tuple[str, int], float] = defaultdict(float)
    planned_mass_by_candidate_seat: defaultdict[tuple[str, int], float] = defaultdict(
        float
    )
    for key in active_keys:
        candidate_seat = (key[0], key[2])
        mass_by_candidate_seat[candidate_seat] += matchup_decision_mass[key]
        planned_mass_by_candidate_seat[candidate_seat] += planned_expected[key]

    matchup_rows: list[AdaptiveMatchupAllocationReport] = []
    for key in sorted(active_keys):
        score = score_by_key[key]
        identity = score.identity
        artifact = route_artifacts[identity.route_id]
        target = target_by_key[key]
        previous = previous_targets.get(key)
        candidate_seat = (identity.candidate_deck_digest, identity.candidate_seat)
        total_mass = mass_by_candidate_seat[candidate_seat]
        projected_total = total_mass + planned_mass_by_candidate_seat[candidate_seat]
        global_target = (
            snapshot.candidate_target_shares[identity.candidate_deck_digest]
            * target
            / 2.0
        )
        matchup_rows.append(
            AdaptiveMatchupAllocationReport(
                candidate_deck_digest=identity.candidate_deck_digest,
                artifact_id=identity.artifact_id,
                source_fingerprint=artifact.source_fingerprint,
                source_policy_version=artifact.source_policy_version,
                stratum=entries[identity.artifact_id].stratum,
                route_id=identity.route_id,
                opponent_deck_digest=identity.opponent_deck_digest,
                candidate_seat=identity.candidate_seat,
                portfolio=score.dominant_portfolio,
                target_share=target,
                global_target_share=global_target,
                previous_target_share=previous,
                target_delta=None if previous is None else target - previous,
                posterior_score=score.posterior_score,
                slow_score=score.slow_score,
                posterior_stddev=score.posterior_stddev,
                effective_evidence=score.effective_evidence,
                utility=score.utility,
                components=score.components,
                expected_decisions_per_game=score.expected_decisions,
                decision_mass=matchup_decision_mass[key],
                normalized_decision_mass=matchup_decision_mass[key] / target,
                decision_debt_before=target * total_mass - matchup_decision_mass[key],
                projected_decision_debt=target * projected_total
                - matchup_decision_mass[key]
                - planned_expected[key],
                planned_games=planned_games[key],
                planned_expected_decisions=planned_expected[key],
                actual_games=actual_games[key],
                actual_decisions=actual_decisions[key],
                actual_score_sum=actual_scores[key],
            )
        )

    candidate_rows = _candidate_rows(
        matchup_rows,
        snapshot=snapshot,
        base_candidate_shares=base_candidate_shares,
        previous_candidate_targets=previous_candidate_targets,
    )
    portfolio_rows = _portfolio_rows(matchup_rows, snapshot=snapshot)
    artifact_rows = _artifact_rows(matchup_rows, committed_state=committed_state)
    return AdaptiveOpponentAllocationReport(
        window_sequence=plan.window_sequence,
        plan_id=plan.plan_id,
        target_fingerprint=snapshot.fingerprint,
        predecessor_state_fingerprint=predecessor_state_fingerprint,
        committed_state_fingerprint=committed_state_fingerprint,
        revision_fingerprint=committed_state.revision.fingerprint,
        evidence_cells=snapshot.evidence_cells,
        low_evidence_cells=snapshot.low_evidence_cells,
        candidates=candidate_rows,
        portfolios=portfolio_rows,
        artifacts=artifact_rows,
        matchups=tuple(matchup_rows),
    )


def _candidate_rows(
    matchups: Sequence[AdaptiveMatchupAllocationReport],
    *,
    snapshot: AdaptiveAllocationSnapshot,
    base_candidate_shares: Mapping[str, float],
    previous_candidate_targets: Mapping[str, float],
) -> tuple[AdaptiveCandidateAllocationReport, ...]:
    grouped: defaultdict[str, list[AdaptiveMatchupAllocationReport]] = defaultdict(list)
    for row in matchups:
        grouped[row.candidate_deck_digest].append(row)
    result: list[AdaptiveCandidateAllocationReport] = []
    for candidate, rows in sorted(grouped.items()):
        weight_total = sum(row.target_share for row in rows)
        result.append(
            AdaptiveCandidateAllocationReport(
                candidate_deck_digest=candidate,
                base_share=base_candidate_shares[candidate],
                target_share=snapshot.candidate_target_shares[candidate],
                previous_target_share=previous_candidate_targets.get(candidate),
                posterior_score=sum(
                    row.posterior_score * row.target_share for row in rows
                )
                / weight_total,
                worst_posterior_score=min(row.posterior_score for row in rows),
                matchup_cells=len(rows),
                evidence_cells=sum(row.effective_evidence > 0.0 for row in rows),
                planned_games=sum(row.planned_games for row in rows),
                actual_games=sum(row.actual_games for row in rows),
                actual_decisions=sum(row.actual_decisions for row in rows),
            )
        )
    return tuple(result)


def _portfolio_rows(
    matchups: Sequence[AdaptiveMatchupAllocationReport],
    *,
    snapshot: AdaptiveAllocationSnapshot,
) -> tuple[AdaptivePortfolioAllocationReport, ...]:
    names: tuple[PortfolioName, ...] = (
        "counter",
        "frontier",
        "probe",
        "rehearsal",
        "staleness",
    )
    total_actual_decisions = sum(row.actual_decisions for row in matchups)
    return tuple(
        AdaptivePortfolioAllocationReport(
            portfolio=name,
            target_share=snapshot.portfolio_target_mass[name],
            planned_games=sum(
                row.planned_games for row in matchups if row.portfolio == name
            ),
            planned_expected_decisions=sum(
                row.planned_expected_decisions
                for row in matchups
                if row.portfolio == name
            ),
            actual_games=sum(
                row.actual_games for row in matchups if row.portfolio == name
            ),
            actual_decisions=(
                decisions := sum(
                    row.actual_decisions for row in matchups if row.portfolio == name
                )
            ),
            actual_decision_share=(
                0.0
                if total_actual_decisions == 0
                else decisions / float(total_actual_decisions)
            ),
        )
        for name in names
    )


def _artifact_rows(
    matchups: Sequence[AdaptiveMatchupAllocationReport],
    *,
    committed_state: LeagueState,
) -> tuple[AdaptiveArtifactAllocationReport, ...]:
    grouped: defaultdict[str, list[AdaptiveMatchupAllocationReport]] = defaultdict(list)
    for row in matchups:
        grouped[row.artifact_id].append(row)
    artifacts = {item.artifact_id: item for item in committed_state.revision.artifacts}
    entries = {item.artifact_id: item for item in committed_state.revision.entries}
    routes = {item.route_id: item for item in committed_state.revision.routes}
    result: list[AdaptiveArtifactAllocationReport] = []
    for artifact_id, rows in sorted(grouped.items()):
        artifact = artifacts[artifact_id]
        target_total = sum(row.global_target_share for row in rows)
        weighted_score = sum(
            row.posterior_score * row.global_target_share for row in rows
        )
        result.append(
            AdaptiveArtifactAllocationReport(
                artifact_id=artifact_id,
                source_fingerprint=artifact.source_fingerprint,
                source_policy_version=artifact.source_policy_version,
                stratum=entries[artifact_id].stratum,
                route_count=sum(
                    route.artifact_id == artifact_id for route in routes.values()
                ),
                target_share=target_total,
                posterior_score=(
                    weighted_score / target_total
                    if target_total > 0.0
                    else math.fsum(row.posterior_score for row in rows) / len(rows)
                ),
                worst_posterior_score=min(row.posterior_score for row in rows),
                evidence_cells=sum(row.effective_evidence > 0.0 for row in rows),
                planned_games=sum(row.planned_games for row in rows),
                planned_expected_decisions=sum(
                    row.planned_expected_decisions for row in rows
                ),
                actual_games=sum(row.actual_games for row in rows),
                actual_decisions=sum(row.actual_decisions for row in rows),
            )
        )
    return tuple(result)


__all__ = [
    "AdaptiveArtifactAllocationReport",
    "AdaptiveCandidateAllocationReport",
    "AdaptiveMatchupAllocationReport",
    "AdaptiveOpponentAllocationReport",
    "AdaptivePortfolioAllocationReport",
    "build_adaptive_allocation_report",
]
