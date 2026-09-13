"""Atomic commit of window outcomes into normalized matchup statistics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptcg_rl.rl.opponent_pool.models import ActiveMatchup
from ptcg_rl.rl.opponent_pool.planner import (
    OpponentAssignment,
    OpponentQuotaCell,
    QuotaWindowPlan,
    WindowPlan,
)
from ptcg_rl.rl.opponent_pool.state import LeagueState, MatchupStat

OutcomeStatus = Literal[
    "engine_terminal",
    "window_cutoff",
    "step_limit",
    "cancelled",
    "infrastructure_error",
]

_EXECUTED_STATUSES: frozenset[OutcomeStatus] = frozenset(
    {"engine_terminal", "window_cutoff", "step_limit"}
)


class OpponentOutcome(BaseModel):
    """Observed result for one assignment; missing assignments are unresolved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assignment_index: int = Field(ge=0)
    status: OutcomeStatus
    candidate_decisions: int = Field(ge=0)
    candidate_score: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def score_matches_status(self) -> Self:
        """Only real terminals and adjudicated step limits carry scores."""
        if self.status == "engine_terminal":
            if self.candidate_score is None:
                raise ValueError("terminal outcome requires candidate_score")
            if self.candidate_score not in {0.0, 0.5, 1.0}:
                raise ValueError("candidate_score must be win, draw, or loss")
        elif self.status == "step_limit" and self.candidate_score is not None:
            if self.candidate_score not in {0.0, 1.0}:
                raise ValueError("step-limit score must identify a winner")
        elif self.candidate_score is not None:
            raise ValueError("non-terminal outcome cannot have candidate_score")
        return self


def _validate_plan_against_revision(
    state: LeagueState,
    plan: WindowPlan,
) -> None:
    """Ensure every planned identity remains meaningful in the active graph."""
    artifact_ids = {
        artifact.artifact_id for artifact in state.revision.artifacts
    }
    routes = {route.route_id: route for route in state.revision.routes}
    entries = {entry.artifact_id: entry for entry in state.revision.entries}
    active_matchups = {
        matchup.key for matchup in state.revision.active_matchups
    }
    if not set(plan.selected_artifact_ids).issubset(artifact_ids):
        raise ValueError("plan selects an inactive artifact")
    for assignment in plan.assignments:
        route = routes.get(assignment.route_id)
        if route is None or route.artifact_id != assignment.artifact_id:
            raise ValueError("assignment route does not resolve to its artifact")
        if entries[assignment.artifact_id].stratum != assignment.stratum:
            raise ValueError("assignment stratum does not match active entry")
        if assignment.matchup_key not in active_matchups:
            raise ValueError("assignment matchup is not active")


def _updated_stat(
    previous: MatchupStat | None,
    assignment: OpponentAssignment | OpponentQuotaCell,
    outcome: OpponentOutcome,
    window_sequence: int,
) -> MatchupStat:
    """Apply one eligible outcome to one normalized row."""
    matchup = ActiveMatchup(
        candidate_deck_digest=assignment.candidate_deck_digest,
        route_id=assignment.route_id,
        candidate_seat=assignment.candidate_seat,
    )
    base = previous or MatchupStat(matchup=matchup)
    executed = outcome.status in _EXECUTED_STATUSES
    learning = executed and outcome.candidate_decisions > 0
    score = outcome.candidate_score
    return MatchupStat(
        matchup=matchup,
        executed_games=base.executed_games + int(executed),
        learning_exposures=base.learning_exposures + int(learning),
        trainable_decisions=(
            base.trainable_decisions
            + (outcome.candidate_decisions if learning else 0)
        ),
        wins=base.wins + int(score == 1.0),
        draws=base.draws + int(score == 0.5),
        losses=base.losses + int(score == 0.0),
        last_executed_window=(
            window_sequence if executed else base.last_executed_window
        ),
        last_learning_exposure_window=(
            window_sequence
            if learning
            else base.last_learning_exposure_window
        ),
    )


def commit_window(
    state: LeagueState,
    plan: WindowPlan,
    outcomes: Sequence[OpponentOutcome],
) -> LeagueState:
    """Atomically commit eligible outcomes after stale-state validation."""
    if plan.revision_fingerprint != state.revision.fingerprint:
        raise ValueError("window plan targets a stale revision")
    if plan.base_state_fingerprint != state.fingerprint:
        raise ValueError("window plan targets a stale committed state")
    if plan.window_sequence != state.next_window_sequence:
        raise ValueError("window sequence does not match committed state")
    _validate_plan_against_revision(state, plan)

    assignments = {
        assignment.assignment_index: assignment
        for assignment in plan.assignments
    }
    outcomes_by_index: dict[int, OpponentOutcome] = {}
    for outcome in outcomes:
        if outcome.assignment_index not in assignments:
            raise ValueError("outcome references an unknown assignment")
        if outcome.assignment_index in outcomes_by_index:
            raise ValueError("assignment outcome is duplicated")
        outcomes_by_index[outcome.assignment_index] = outcome

    stats = {stat.matchup.key: stat for stat in state.matchup_stats}
    for assignment in plan.assignments:
        observed_outcome = outcomes_by_index.get(assignment.assignment_index)
        if (
            observed_outcome is None
            or observed_outcome.status not in _EXECUTED_STATUSES
        ):
            continue
        key = assignment.matchup_key
        stats[key] = _updated_stat(
            stats.get(key),
            assignment,
            observed_outcome,
            plan.window_sequence,
        )

    return LeagueState(
        revision=state.revision,
        generation=state.generation + 1,
        next_window_sequence=state.next_window_sequence + 1,
        matchup_stats=tuple(sorted(stats.values(), key=lambda item: item.matchup.key)),
        last_committed_plan_id=plan.plan_id,
        last_transition_id=state.last_transition_id,
    )


def commit_quota_window(
    state: LeagueState,
    plan: QuotaWindowPlan,
    outcomes: Sequence[OpponentOutcome],
) -> LeagueState:
    """Commit issued quota-cell observations without expanding unissued games."""
    if plan.revision_fingerprint != state.revision.fingerprint:
        raise ValueError("quota window targets a stale revision")
    if plan.base_state_fingerprint != state.fingerprint:
        raise ValueError("quota window targets a stale committed state")
    if plan.window_sequence != state.next_window_sequence:
        raise ValueError("quota window sequence does not match committed state")
    artifact_ids = {item.artifact_id for item in state.revision.artifacts}
    routes = {item.route_id: item for item in state.revision.routes}
    entries = {item.artifact_id: item for item in state.revision.entries}
    active_matchups = {item.key for item in state.revision.active_matchups}
    if not set(plan.selected_artifact_ids) <= artifact_ids:
        raise ValueError("quota window selects an inactive artifact")
    cells = {item.cell_index: item for item in plan.cells}
    for cell in plan.cells:
        route = routes.get(cell.route_id)
        if route is None or route.artifact_id != cell.artifact_id:
            raise ValueError("quota route does not resolve to its artifact")
        if entries[cell.artifact_id].stratum != cell.stratum:
            raise ValueError("quota stratum differs from its active entry")
        if cell.matchup_key not in active_matchups:
            raise ValueError("quota matchup is not active")

    observed_counts: dict[int, int] = dict.fromkeys(cells, 0)
    stats = {item.matchup.key: item for item in state.matchup_stats}
    for outcome in outcomes:
        try:
            cell = cells[outcome.assignment_index]
        except KeyError as exc:
            raise ValueError("quota outcome references an unknown cell") from exc
        observed_counts[cell.cell_index] += 1
        if observed_counts[cell.cell_index] > cell.game_count:
            raise ValueError("quota outcomes exceed their planned cell")
        if outcome.status not in _EXECUTED_STATUSES:
            continue
        stats[cell.matchup_key] = _updated_stat(
            stats.get(cell.matchup_key),
            cell,
            outcome,
            plan.window_sequence,
        )

    return LeagueState(
        revision=state.revision,
        generation=state.generation + 1,
        next_window_sequence=state.next_window_sequence + 1,
        matchup_stats=tuple(sorted(stats.values(), key=lambda item: item.matchup.key)),
        last_committed_plan_id=plan.plan_id,
        last_transition_id=state.last_transition_id,
    )
