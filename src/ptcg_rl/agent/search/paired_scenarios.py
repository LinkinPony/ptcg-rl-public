"""Nonanticipative weighted aggregation over a fixed paired-scenario grid."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ptcg_rl.agent.search.hierarchical_contract import OptionOutcome
from ptcg_rl.agent.search.planner_fallback import (
    PlannerEvidenceError,
    PlannerFallbackReason,
)
from ptcg_rl.agent.search.planner_scoring import PlannerScoringConfig
from ptcg_rl.engine.compact_consequence import SemanticEndpoint

Float32Array = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class CandidateScenarioAggregate:
    """One root option's paired-support value summary."""

    candidate_fingerprint: str
    weighted_mean: float
    weighted_std: float
    robust_score: float
    downside_minimum: float
    terminal_weight: float
    same_seat_main_weight: float
    turn_handoff_weight: float
    information_history_forked: bool


@dataclass(frozen=True, slots=True)
class PairedScenarioAggregation:
    """Comparable candidate summaries under one support/controller identity."""

    root_information_history_fingerprint: str
    scenario_support_fingerprint: str
    continuation_controller_fingerprint: str
    scorer_fingerprint: str
    candidates: tuple[CandidateScenarioAggregate, ...]

    @property
    def robust_scores(self) -> Float32Array:
        """Return immutable candidate-aligned robust scores."""
        values = np.asarray(
            tuple(item.robust_score for item in self.candidates),
            dtype=np.float32,
        )
        values.setflags(write=False)
        return values


def aggregate_paired_option_outcomes(
    outcomes: tuple[OptionOutcome, ...],
    *,
    cell_scores: npt.ArrayLike,
    scorer_fingerprint: str,
    config: PlannerScoringConfig,
) -> PairedScenarioAggregation:
    """Aggregate candidate-major scores without scenario-conditioned actions."""
    if not outcomes:
        raise PlannerEvidenceError(
            PlannerFallbackReason.EVIDENCE_ABSENT,
            "paired aggregation requires at least one option outcome",
        )
    candidate_fingerprints = tuple(
        outcome.root_candidate_fingerprint for outcome in outcomes
    )
    if len(set(candidate_fingerprints)) != len(candidate_fingerprints):
        raise PlannerEvidenceError(
            PlannerFallbackReason.FINGERPRINT_MISMATCH,
            "paired aggregation contains duplicate root candidates",
        )
    reference = outcomes[0]
    if scorer_fingerprint != config.scorer_fingerprint:
        raise PlannerEvidenceError(
            PlannerFallbackReason.FINGERPRINT_MISMATCH,
            "leaf scores were produced by a different scorer contract",
        )
    if reference.controller.scorer_fingerprint != scorer_fingerprint:
        raise PlannerEvidenceError(
            PlannerFallbackReason.MODEL_VERSION_MISMATCH,
            "continuation controller uses a different scorer contract",
        )
    scenario_count = len(reference.cells)
    _validate_common_contract(outcomes, reference)
    _validate_cross_candidate_nonanticipativity(outcomes)
    scores = np.asarray(cell_scores, dtype=np.float64)
    expected_shape = (len(outcomes) * scenario_count,)
    if scores.shape != expected_shape or not bool(np.isfinite(scores).all()):
        raise PlannerEvidenceError(
            PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
            "cell scores do not form a complete finite candidate-scenario grid",
        )
    score_grid = scores.reshape(len(outcomes), scenario_count)
    _validate_terminal_scores(outcomes, score_grid)
    weights = np.asarray(
        tuple(cell.weight for cell in reference.cells),
        dtype=np.float64,
    )

    aggregates: list[CandidateScenarioAggregate] = []
    for candidate_index, outcome in enumerate(outcomes):
        candidate_scores = score_grid[candidate_index]
        mean = float(np.dot(weights, candidate_scores))
        variance = float(np.dot(weights, np.square(candidate_scores - mean)))
        std = math.sqrt(max(0.0, variance))
        endpoints = tuple(cell.endpoint for cell in outcome.cells)
        aggregates.append(
            CandidateScenarioAggregate(
                candidate_fingerprint=outcome.root_candidate_fingerprint,
                weighted_mean=mean,
                weighted_std=std,
                robust_score=mean - config.risk_std_weight * std,
                downside_minimum=float(candidate_scores.min()),
                terminal_weight=_endpoint_weight(
                    endpoints,
                    weights,
                    SemanticEndpoint.TERMINAL,
                ),
                same_seat_main_weight=_endpoint_weight(
                    endpoints,
                    weights,
                    SemanticEndpoint.SAME_SEAT_MAIN,
                ),
                turn_handoff_weight=_endpoint_weight(
                    endpoints,
                    weights,
                    SemanticEndpoint.TURN_HANDOFF,
                ),
                information_history_forked=(
                    len(
                        {
                            cell.information_history_fingerprint
                            for cell in outcome.cells
                        }
                    )
                    > 1
                ),
            )
        )
    return PairedScenarioAggregation(
        root_information_history_fingerprint=(
            reference.root_information_history_fingerprint
        ),
        scenario_support_fingerprint=reference.scenario_support_fingerprint,
        continuation_controller_fingerprint=(
            reference.controller.controller_fingerprint
        ),
        scorer_fingerprint=scorer_fingerprint,
        candidates=tuple(aggregates),
    )


def _validate_common_contract(
    outcomes: tuple[OptionOutcome, ...],
    reference: OptionOutcome,
) -> None:
    reference_support = tuple(
        (cell.scenario_fingerprint, cell.weight) for cell in reference.cells
    )
    for outcome in outcomes[1:]:
        if outcome.root_information_history_fingerprint != (
            reference.root_information_history_fingerprint
        ):
            raise PlannerEvidenceError(
                PlannerFallbackReason.FINGERPRINT_MISMATCH,
                "candidate outcomes refer to different root information histories",
            )
        if outcome.scenario_support_fingerprint != (
            reference.scenario_support_fingerprint
        ):
            raise PlannerEvidenceError(
                PlannerFallbackReason.FINGERPRINT_MISMATCH,
                "candidate outcomes use different scenario supports",
            )
        if outcome.controller.controller_fingerprint != (
            reference.controller.controller_fingerprint
        ):
            raise PlannerEvidenceError(
                PlannerFallbackReason.MODEL_VERSION_MISMATCH,
                "candidate outcomes use different continuation controllers",
            )
        support = tuple(
            (cell.scenario_fingerprint, cell.weight) for cell in outcome.cells
        )
        if support != reference_support:
            raise PlannerEvidenceError(
                PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                "candidate outcomes do not share identical paired weights",
            )


def _validate_cross_candidate_nonanticipativity(
    outcomes: tuple[OptionOutcome, ...],
) -> None:
    action_by_history: dict[str, str] = {}
    for outcome in outcomes:
        for decision in outcome.continuation_decisions:
            previous = action_by_history.setdefault(
                decision.information_history_fingerprint,
                decision.action_fingerprint,
            )
            if previous != decision.action_fingerprint:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.NONANTICIPATIVITY_VIOLATION,
                    "shared controller changed action in one information history",
                )


def _validate_terminal_scores(
    outcomes: tuple[OptionOutcome, ...],
    score_grid: np.ndarray,
) -> None:
    for candidate_index, outcome in enumerate(outcomes):
        for cell in outcome.cells:
            if cell.terminal_result is None:
                continue
            score = np.float32(score_grid[candidate_index, cell.scenario_index])
            expected = np.float32(cell.terminal_result)
            if score != expected:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.FINGERPRINT_MISMATCH,
                    "terminal cell score contradicts its exact W/D/L result",
                )


def _endpoint_weight(
    endpoints: tuple[SemanticEndpoint, ...],
    weights: np.ndarray,
    target: SemanticEndpoint,
) -> float:
    return float(
        math.fsum(
            float(weight)
            for endpoint, weight in zip(endpoints, weights, strict=True)
            if endpoint is target
        )
    )


__all__ = [
    "CandidateScenarioAggregate",
    "PairedScenarioAggregation",
    "aggregate_paired_option_outcomes",
]
