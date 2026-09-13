"""Pure diagnostic metrics for exhaustive-fit candidate supports."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ptcg_rl.evaluation.candidate_regret_sources import Action


@dataclass(frozen=True, slots=True)
class RegretAtK:
    """One nested-support regret and near-best recall observation."""

    k: int
    retained_count: int
    exhaustive_best_score: float
    retained_best_score: float
    best_regret: float
    epsilon_recall: bool


def candidate_regret_curve(
    exhaustive_scores: Mapping[Action, float],
    retained_actions: Sequence[Action],
    *,
    k_values: Sequence[int],
    epsilon: float,
) -> tuple[RegretAtK, ...]:
    """Compare nested constructor support with a genuinely exhaustive set."""
    if not exhaustive_scores:
        raise ValueError("candidate regret requires exhaustive candidate scores")
    if not retained_actions:
        raise ValueError("candidate regret requires at least one retained action")
    if any(action not in exhaustive_scores for action in retained_actions):
        raise ValueError("retained actions are outside the exhaustive support")
    if any(not math.isfinite(float(score)) for score in exhaustive_scores.values()):
        raise ValueError("candidate scores must be finite")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    ks = tuple(int(value) for value in k_values)
    if not ks or any(value <= 0 for value in ks):
        raise ValueError("k_values must contain positive integers")
    exhaustive_best = max(float(score) for score in exhaustive_scores.values())
    near_best_floor = exhaustive_best - epsilon
    output: list[RegretAtK] = []
    for k in ks:
        prefix = tuple(retained_actions[:k])
        if not prefix:
            raise ValueError("a K prefix unexpectedly contains no candidate")
        scores = tuple(float(exhaustive_scores[action]) for action in prefix)
        retained_best = max(scores)
        output.append(
            RegretAtK(
                k=k,
                retained_count=len(prefix),
                exhaustive_best_score=exhaustive_best,
                retained_best_score=retained_best,
                best_regret=max(0.0, exhaustive_best - retained_best),
                epsilon_recall=any(score >= near_best_floor for score in scores),
            )
        )
    return tuple(output)


def post_state_alias_counts(
    successor_fingerprints: Sequence[str],
    scores: Sequence[float],
    *,
    tolerance: float = 1.0e-6,
) -> tuple[int, int]:
    """Count exact-successor aliases and impossible score disagreements."""
    if len(successor_fingerprints) != len(scores):
        raise ValueError("successor fingerprints and scores must align")
    groups: dict[str, list[float]] = {}
    for fingerprint, score in zip(successor_fingerprints, scores, strict=True):
        groups.setdefault(fingerprint, []).append(float(score))
    alias_groups = sum(len(values) > 1 for values in groups.values())
    score_disagreements = sum(
        len(values) > 1 and max(values) - min(values) > tolerance
        for values in groups.values()
    )
    return alias_groups, score_disagreements


__all__ = ["RegretAtK", "candidate_regret_curve", "post_state_alias_counts"]
