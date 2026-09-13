"""Exact single-writer performance evidence for simple-stateless training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ptcg_rl.rl.performance_state import PerformanceOutcome
from ptcg_rl.rl.stateless_collection import (
    StatelessAssignedGame,
    StatelessGameOutcome,
)

_OPPONENT_KIND_BY_LANE = {
    "mirror": "self_play",
    "pfsp": "frozen",
    "scripted": "scripted",
}


def stateless_performance_outcomes(
    assignments: Sequence[StatelessAssignedGame],
    outcomes: Sequence[StatelessGameOutcome],
    *,
    active_deck_labels: Mapping[str, str],
    policy_version: int,
    opponent_strata_by_assignment: Mapping[str, str],
) -> tuple[PerformanceOutcome, ...]:
    """Project one complete assignment cohort to exact scored diagnostics."""
    expected = {
        assignment.curriculum.assignment_id: assignment for assignment in assignments
    }
    actual = {outcome.curriculum_assignment_id: outcome for outcome in outcomes}
    if len(expected) != len(assignments) or len(actual) != len(outcomes):
        raise ValueError("performance cohort contains duplicate assignment IDs")
    if set(actual) != set(expected):
        raise ValueError("performance outcomes do not cover assigned games")
    if set(opponent_strata_by_assignment) != set(expected):
        raise ValueError("performance strata do not cover assigned games")

    projected: list[PerformanceOutcome] = []
    for assignment in assignments:
        balance = assignment.balance
        curriculum = assignment.curriculum
        outcome = actual[curriculum.assignment_id]
        if outcome.balance_assignment_id != balance.assignment_id:
            raise ValueError("performance outcome crossed deck-balance identity")
        if (
            curriculum.candidate_deck_digest != balance.deck_digest
            or curriculum.candidate_seat != balance.seat
        ):
            raise ValueError(
                "performance assignment candidate identity is inconsistent"
            )
        score = outcome.candidate_score
        scored = outcome.status == "engine_terminal" or (
            outcome.status == "step_limit" and score is not None
        )
        if not scored:
            if score is not None:
                raise ValueError("unresolved performance outcome carried a score")
            continue
        allowed_scores = (
            (0.0, 1.0)
            if outcome.status == "step_limit"
            else (0.0, 0.5, 1.0)
        )
        if score is None or score not in allowed_scores:
            raise ValueError("scored performance outcome has an invalid score")
        try:
            candidate_label = active_deck_labels[balance.deck_digest]
        except KeyError as error:
            raise KeyError("candidate deck has no performance label") from error
        opponent_label = active_deck_labels.get(
            curriculum.opponent_deck_digest,
            f"opponent_{curriculum.opponent_deck_digest[:12]}",
        )
        projected.append(
            PerformanceOutcome(
                candidate_deck_label=candidate_label,
                opponent_kind=_OPPONENT_KIND_BY_LANE[curriculum.lane],
                opponent_deck_label=opponent_label,
                opponent_id=curriculum.opponent_id,
                candidate_seat=balance.seat,
                candidate_reward=2.0 * score - 1.0,
                policy_version=policy_version,
                opponent_stratum=opponent_strata_by_assignment[
                    curriculum.assignment_id
                ],
            )
        )
    return tuple(projected)


def stateless_opponent_strata(
    assignments: Sequence[StatelessAssignedGame],
    *,
    member_sources: Mapping[str, str],
) -> dict[str, str]:
    """Bind every assignment to a stratum from its lane and PFSP member source."""
    strata: dict[str, str] = {}
    for assigned in assignments:
        assignment = assigned.curriculum
        if assignment.assignment_id in strata:
            raise ValueError("performance cohort contains duplicate assignment IDs")
        if assignment.lane == "mirror":
            stratum = "self_play"
        elif assignment.lane == "scripted":
            stratum = "scripted"
        else:
            if not assignment.member_id:
                raise ValueError("PFSP performance assignment omitted its member ID")
            source = member_sources.get(assignment.member_id)
            if source in {"historical_anchor", "fixed_stateless_anchor"}:
                stratum = "sentinel"
            elif source == "past_self":
                stratum = "adaptive_history"
            else:
                raise ValueError(
                    "PFSP performance assignment has no authoritative member source"
                )
        strata[assignment.assignment_id] = stratum
    return strata


__all__ = ["stateless_opponent_strata", "stateless_performance_outcomes"]
