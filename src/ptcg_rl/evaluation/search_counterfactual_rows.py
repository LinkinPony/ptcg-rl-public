"""Normalized root, candidate, and world rows for the S1 audit."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.agent.search.reranker import MacroSearchResult, MacroWorldEvaluation


def build_counterfactual_rows(
    *,
    episode_id: int,
    step_index: int,
    seat: int,
    split: str,
    phase: str,
    terminal_value: float | None,
    root_value: float,
    serving_observation: Mapping[str, Any],
    greedy_action: tuple[int, ...],
    result: MacroSearchResult,
    priors: Mapping[tuple[int, ...], float],
    identity: Mapping[str, Any],
    timing: Mapping[str, float],
    policy_entropy: float,
    policy_top_gap: float,
) -> tuple[
    dict[str, Any],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    MacroSearchResult,
]:
    """Normalize one in-memory root group for streaming Parquet output."""
    root_id = f"{episode_id}:{step_index}:{seat}"
    campaign_fp = str(identity["campaign_fp"])
    stage_fp = str(identity["stage_fp"])
    action_index = {
        action: index for index, action in enumerate(result.candidates.actions)
    }
    score_by_action = {score.action: score for score in result.decision.action_scores}
    evaluations_by_action: dict[
        tuple[int, ...], list[MacroWorldEvaluation]
    ] = {action: [] for action in result.candidates.actions}
    evaluation_rows: list[dict[str, Any]] = []
    for evaluation in result.evaluations:
        evaluations_by_action.setdefault(evaluation.action, []).append(evaluation)
        evaluation_rows.append(
            {
                "campaign_fp": campaign_fp,
                "stage_fp": stage_fp,
                "root_id": root_id,
                "candidate_index": action_index.get(evaluation.action, -1),
                "world_index": evaluation.world_index,
                "action": list(evaluation.action),
                "endpoint": evaluation.endpoint.value,
                "steps": evaluation.steps,
                "engine_score": evaluation.engine_score,
                "critic_value": evaluation.critic_value,
                "stop_detail": evaluation.stop_detail,
                "error": evaluation.error,
            }
        )
    candidate_rows: list[dict[str, Any]] = []
    for candidate_index, (action, sources) in enumerate(
        zip(result.candidates.actions, result.candidates.sources, strict=True)
    ):
        action_evaluations = evaluations_by_action[action]
        engine_scores = [
            float(row.engine_score)
            for row in action_evaluations
            if row.engine_score is not None
        ]
        critic_values = [
            float(row.critic_value)
            for row in action_evaluations
            if row.critic_value is not None
        ]
        action_score = score_by_action.get(action)
        candidate_rows.append(
            {
                "campaign_fp": campaign_fp,
                "stage_fp": stage_fp,
                "root_id": root_id,
                "candidate_index": candidate_index,
                "action": list(action),
                "sources": list(sources),
                "prior": priors.get(action),
                "paired_worlds": len(engine_scores),
                "engine_mean": _mean(engine_scores),
                "engine_std": _std(engine_scores),
                "critic_mean": _mean(critic_values),
                "critic_std": _std(critic_values),
                "mean_delta": action_score.mean_delta if action_score else None,
                "std_delta": action_score.std_delta if action_score else None,
                "robust_delta": action_score.robust_delta if action_score else None,
                "downside_cvar": action_score.downside_cvar if action_score else None,
                "minimum_delta": action_score.minimum_delta if action_score else None,
            }
        )
    select = _mapping(serving_observation.get("select"))
    root_row = {
        "campaign_fp": campaign_fp,
        "stage_fp": stage_fp,
        "root_id": root_id,
        "episode_id": episode_id,
        "split": split,
        "step_index": step_index,
        "seat": seat,
        "phase": phase,
        "turn": _int_field(serving_observation.get("current"), "turn", -1),
        "select_context": _int_field(select, "context", -1),
        "option_count": len(_sequence(select.get("option"))),
        "terminal_value": terminal_value,
        "resolved": terminal_value is not None,
        "root_value": root_value,
        "policy_entropy": policy_entropy,
        "policy_top_gap": policy_top_gap,
        "greedy_action": list(greedy_action),
        "recommended_action": list(result.decision.selected_action),
        "recommendation_changed": result.decision.action_changed,
        "selection_reason": result.decision.reason,
        "candidate_count": len(result.candidates.actions),
        "greedy_included": result.candidates.contains(greedy_action),
        "illegal_candidate_count": sum(
            not is_legal_action(select, action)
            for action in result.candidates.actions
        ),
        "worlds_requested": result.worlds_requested,
        "worlds_sampled": result.worlds_sampled,
        "worlds_completed": result.worlds_completed,
        "complete_coverage": result.complete_coverage,
        "stop_reason": result.stop_reason,
        "transitions": result.transitions,
        "engine_sessions": result.engine_sessions,
        "state_pool_peak": result.state_pool_peak,
        "state_leaks": result.state_leaks,
        **timing,
        "telemetry_complete": telemetry_complete(result, timing),
    }
    return root_row, tuple(candidate_rows), tuple(evaluation_rows), result


def prior_diagnostics(
    priors: Mapping[tuple[int, ...], float],
) -> tuple[float, float]:
    """Return candidate-policy entropy and top-one/top-two probability gap."""
    probabilities = sorted(
        (float(value) for value in priors.values() if value > 0.0),
        reverse=True,
    )
    entropy = -sum(value * math.log(value) for value in probabilities)
    gap = probabilities[0] - probabilities[1] if len(probabilities) >= 2 else 1.0
    return entropy, gap


def telemetry_complete(
    result: MacroSearchResult,
    timing: Mapping[str, float],
) -> bool:
    """Validate compact timing and root/action/world telemetry coverage."""
    if any(not math.isfinite(value) or value < 0.0 for value in timing.values()):
        return False
    if len(result.candidates.actions) != len(result.candidates.sources):
        return False
    candidate_actions = set(result.candidates.actions)
    keys = [(row.world_index, row.action) for row in result.evaluations]
    return len(keys) == len(set(keys)) and all(
        row.action in candidate_actions for row in result.evaluations
    )


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std(values: Sequence[float]) -> float | None:
    return statistics.pstdev(values) if values else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_field(value: Any, name: str, default: int) -> int:
    if isinstance(value, Mapping):
        item = value.get(name, default)
    else:
        item = getattr(value, name, default)
    return int(item) if item is not None else default
