"""Streaming metrics for paired engine sibling value rankings."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast

from ptcg_rl.agent.search.macro import MacroEndpoint
from ptcg_rl.evaluation.teacher_sibling_config import TeacherSiblingReferences

_TIE_EPSILON = 1.0e-12


@dataclass(frozen=True)
class SiblingEvaluation:
    """One action resolved in one fixed hidden-information world."""

    action: tuple[int, ...]
    world_index: int
    endpoint: MacroEndpoint
    engine_score: float | None
    initial_value: float | None
    trained_value: float | None
    leaf_available: bool
    error: str | None = None


class TeacherSiblingMetrics:
    """Accumulate root-grouped paired rankings without retaining the dataset."""

    def __init__(self) -> None:
        self.roots = 0
        self.teacher_candidate_roots = 0
        self.evaluations = 0
        self.leaf_evaluations = 0
        self.engine_score_evaluations = 0
        self.engine_errors = 0
        self.complete_roots = 0
        self.illegal_candidates = 0
        self.state_leaks = 0
        self.max_state_pool_peak = 0
        self.endpoint_counts: Counter[str] = Counter()
        self.comparable_pairs = 0
        self.initial_pair_credit = 0.0
        self.trained_pair_credit = 0.0
        self.initial_top1_regret_sum = 0.0
        self.trained_top1_regret_sum = 0.0
        self.top1_regret_groups = 0
        self.teacher_rank_roots = 0
        self.initial_teacher_mrr_sum = 0.0
        self.trained_teacher_mrr_sum = 0.0
        self.initial_teacher_top1_credit = 0.0
        self.trained_teacher_top1_credit = 0.0
        self.teacher_pairs = 0
        self.initial_teacher_pair_credit = 0.0
        self.trained_teacher_pair_credit = 0.0

    def update(
        self,
        *,
        root_id: str,
        teacher_action: tuple[int, ...],
        candidate_actions: Sequence[tuple[int, ...]],
        evaluations: Sequence[SiblingEvaluation],
        worlds_requested: int,
        illegal_candidates: int,
        state_leaks: int,
        state_pool_peak: int,
        behavior_kind: str,
    ) -> tuple[dict[str, Any], ...]:
        """Consume one root and return its engine-supervised pair rows."""
        self.roots += 1
        normalized_candidates = tuple(tuple(action) for action in candidate_actions)
        self.teacher_candidate_roots += int(teacher_action in normalized_candidates)
        self.illegal_candidates += int(illegal_candidates)
        self.state_leaks += int(state_leaks)
        self.max_state_pool_peak = max(self.max_state_pool_peak, state_pool_peak)
        self.evaluations += len(evaluations)
        self.leaf_evaluations += sum(row.leaf_available for row in evaluations)
        self.engine_score_evaluations += sum(
            row.engine_score is not None and math.isfinite(row.engine_score)
            for row in evaluations
        )
        self.engine_errors += sum(
            row.error is not None or row.endpoint == MacroEndpoint.ENGINE_ERROR
            for row in evaluations
        )
        self.endpoint_counts.update(row.endpoint.value for row in evaluations)
        expected_keys = {
            (world_index, action)
            for world_index in range(worlds_requested)
            for action in normalized_candidates
        }
        actual_keys = {(row.world_index, row.action) for row in evaluations}
        self.complete_roots += int(actual_keys == expected_keys)

        rows_by_world: dict[int, list[SiblingEvaluation]] = defaultdict(list)
        for evaluation in evaluations:
            rows_by_world[evaluation.world_index].append(evaluation)

        pair_rows: list[dict[str, Any]] = []
        for world_index, world_rows in sorted(rows_by_world.items()):
            pair_rows.extend(
                self._update_world_pairs(
                    root_id=root_id,
                    world_index=world_index,
                    evaluations=world_rows,
                    behavior_kind=behavior_kind,
                )
            )
            self._update_top1_regret(world_rows)
        self._update_teacher_ranks(teacher_action, evaluations)
        return tuple(pair_rows)

    def summary(self, references: TeacherSiblingReferences) -> dict[str, Any]:
        """Return integrity checks and initial-versus-trained paired metrics."""
        initial_pairwise = _safe_rate(
            self.initial_pair_credit,
            self.comparable_pairs,
        )
        trained_pairwise = _safe_rate(
            self.trained_pair_credit,
            self.comparable_pairs,
        )
        initial_regret = _safe_rate(
            self.initial_top1_regret_sum,
            self.top1_regret_groups,
        )
        trained_regret = _safe_rate(
            self.trained_top1_regret_sum,
            self.top1_regret_groups,
        )
        initial_mrr = _safe_rate(
            self.initial_teacher_mrr_sum,
            self.teacher_rank_roots,
        )
        trained_mrr = _safe_rate(
            self.trained_teacher_mrr_sum,
            self.teacher_rank_roots,
        )
        initial_teacher_pairwise = _safe_rate(
            self.initial_teacher_pair_credit,
            self.teacher_pairs,
        )
        trained_teacher_pairwise = _safe_rate(
            self.trained_teacher_pair_credit,
            self.teacher_pairs,
        )
        pairwise_improvement = _difference(trained_pairwise, initial_pairwise)
        regret_increase = _difference(trained_regret, initial_regret)
        teacher_mrr_improvement = _difference(trained_mrr, initial_mrr)
        teacher_pairwise_improvement = _difference(
            trained_teacher_pairwise,
            initial_teacher_pairwise,
        )
        leaf_coverage = _safe_rate(self.leaf_evaluations, self.evaluations)
        engine_score_coverage = _safe_rate(
            self.engine_score_evaluations,
            self.evaluations,
        )
        teacher_recall = _safe_rate(self.teacher_candidate_roots, self.roots)
        integrity_checks = {
            "root_count": self.roots >= references.min_roots,
            "complete_world_coverage": self.complete_roots == self.roots,
            "leaf_coverage": _at_least(
                leaf_coverage,
                references.min_leaf_coverage_rate,
            ),
            "engine_score_coverage": _at_least(
                engine_score_coverage,
                references.min_engine_score_coverage_rate,
            ),
            "teacher_candidate_recall": _at_least(
                teacher_recall,
                references.min_teacher_candidate_recall,
            ),
            "comparable_pairs": (
                self.comparable_pairs >= references.min_comparable_pairs
            ),
            "candidate_legality": self.illegal_candidates == 0,
            "engine_errors": self.engine_errors == 0,
            "state_lifecycle": self.state_leaks == 0,
        }
        benefit_checks = {
            "pairwise_accuracy": _at_least(
                pairwise_improvement,
                references.min_pairwise_accuracy_improvement,
            ),
            "top1_regret": regret_increase is not None
            and regret_increase <= references.max_top1_regret_increase,
            "teacher_mrr": _at_least(
                teacher_mrr_improvement,
                references.min_teacher_mrr_improvement,
            ),
            "teacher_pairwise": _at_least(
                teacher_pairwise_improvement,
                references.min_teacher_pairwise_improvement,
            ),
        }
        warnings = [
            f"integrity:{name}"
            for name, observed in integrity_checks.items()
            if not observed
        ]
        warnings.extend(
            f"benefit:{name}"
            for name, observed in benefit_checks.items()
            if not observed
        )
        return {
            "roots": self.roots,
            "evaluations": self.evaluations,
            "leaf_evaluations": self.leaf_evaluations,
            "leaf_coverage_rate": leaf_coverage,
            "engine_score_evaluations": self.engine_score_evaluations,
            "engine_score_coverage_rate": engine_score_coverage,
            "complete_roots": self.complete_roots,
            "teacher_candidate_recall": teacher_recall,
            "comparable_pairs": self.comparable_pairs,
            "top1_regret_groups": self.top1_regret_groups,
            "teacher_rank_roots": self.teacher_rank_roots,
            "teacher_pairs": self.teacher_pairs,
            "engine_errors": self.engine_errors,
            "illegal_candidates": self.illegal_candidates,
            "state_leaks": self.state_leaks,
            "max_state_pool_peak": self.max_state_pool_peak,
            "endpoint_counts": dict(sorted(self.endpoint_counts.items())),
            "initial": {
                "engine_pairwise_accuracy": initial_pairwise,
                "engine_top1_regret": initial_regret,
                "teacher_mrr": initial_mrr,
                "teacher_top1_credit": _safe_rate(
                    self.initial_teacher_top1_credit,
                    self.teacher_rank_roots,
                ),
                "teacher_pairwise_accuracy": initial_teacher_pairwise,
            },
            "trained": {
                "engine_pairwise_accuracy": trained_pairwise,
                "engine_top1_regret": trained_regret,
                "teacher_mrr": trained_mrr,
                "teacher_top1_credit": _safe_rate(
                    self.trained_teacher_top1_credit,
                    self.teacher_rank_roots,
                ),
                "teacher_pairwise_accuracy": trained_teacher_pairwise,
            },
            "deltas": {
                "engine_pairwise_accuracy": pairwise_improvement,
                "engine_top1_regret": regret_increase,
                "teacher_mrr": teacher_mrr_improvement,
                "teacher_pairwise_accuracy": teacher_pairwise_improvement,
            },
            "decision_role": "diagnostic_only",
            "integrity_checks": integrity_checks,
            "benefit_checks": benefit_checks,
            "diagnostic_warnings": warnings,
        }

    def _update_world_pairs(
        self,
        *,
        root_id: str,
        world_index: int,
        evaluations: Sequence[SiblingEvaluation],
        behavior_kind: str,
    ) -> list[dict[str, Any]]:
        pair_rows: list[dict[str, Any]] = []
        for left, right in combinations(evaluations, 2):
            if not _comparable(left, right):
                continue
            assert left.engine_score is not None
            assert right.engine_score is not None
            assert left.initial_value is not None
            assert right.initial_value is not None
            assert left.trained_value is not None
            assert right.trained_value is not None
            engine_delta = left.engine_score - right.engine_score
            if abs(engine_delta) <= _TIE_EPSILON:
                continue
            initial_delta = left.initial_value - right.initial_value
            trained_delta = left.trained_value - right.trained_value
            initial_credit = _direction_credit(initial_delta, engine_delta)
            trained_credit = _direction_credit(trained_delta, engine_delta)
            self.comparable_pairs += 1
            self.initial_pair_credit += initial_credit
            self.trained_pair_credit += trained_credit
            pair_rows.append(
                {
                    "root_id": root_id,
                    "world_index": world_index,
                    "left_action": list(left.action),
                    "right_action": list(right.action),
                    "preferred_action": list(
                        left.action if engine_delta > 0.0 else right.action
                    ),
                    "engine_delta": engine_delta,
                    "initial_value_delta": initial_delta,
                    "trained_value_delta": trained_delta,
                    "initial_credit": initial_credit,
                    "trained_credit": trained_credit,
                    "behavior_kind": behavior_kind,
                    "ppo_ratio_eligible": False,
                }
            )
        return pair_rows

    def _update_top1_regret(
        self,
        evaluations: Sequence[SiblingEvaluation],
    ) -> None:
        valid = [row for row in evaluations if _fully_scored(row)]
        for group in _comparable_groups(valid):
            if len(group) < 2:
                continue
            assert all(row.engine_score is not None for row in group)
            assert all(row.initial_value is not None for row in group)
            assert all(row.trained_value is not None for row in group)
            best_engine = max(cast(float, row.engine_score) for row in group)
            initial_choice = max(
                group,
                key=lambda row: cast(float, row.initial_value),
            )
            trained_choice = max(
                group,
                key=lambda row: cast(float, row.trained_value),
            )
            self.initial_top1_regret_sum += max(
                0.0,
                best_engine - cast(float, initial_choice.engine_score),
            )
            self.trained_top1_regret_sum += max(
                0.0,
                best_engine - cast(float, trained_choice.engine_score),
            )
            self.top1_regret_groups += 1

    def _update_teacher_ranks(
        self,
        teacher_action: tuple[int, ...],
        evaluations: Sequence[SiblingEvaluation],
    ) -> None:
        initial_values: dict[tuple[int, ...], list[float]] = defaultdict(list)
        trained_values: dict[tuple[int, ...], list[float]] = defaultdict(list)
        for row in evaluations:
            if row.error is not None or not row.leaf_available:
                continue
            if row.initial_value is not None and math.isfinite(row.initial_value):
                initial_values[row.action].append(row.initial_value)
            if row.trained_value is not None and math.isfinite(row.trained_value):
                trained_values[row.action].append(row.trained_value)
        shared_actions = set(initial_values) & set(trained_values)
        if teacher_action not in shared_actions or len(shared_actions) < 2:
            return
        initial_means = {
            action: sum(initial_values[action]) / len(initial_values[action])
            for action in shared_actions
        }
        trained_means = {
            action: sum(trained_values[action]) / len(trained_values[action])
            for action in shared_actions
        }
        self.teacher_rank_roots += 1
        initial_mrr, initial_top1 = _rank_credit(initial_means, teacher_action)
        trained_mrr, trained_top1 = _rank_credit(trained_means, teacher_action)
        self.initial_teacher_mrr_sum += initial_mrr
        self.trained_teacher_mrr_sum += trained_mrr
        self.initial_teacher_top1_credit += initial_top1
        self.trained_teacher_top1_credit += trained_top1
        for action in shared_actions - {teacher_action}:
            self.teacher_pairs += 1
            self.initial_teacher_pair_credit += _positive_credit(
                initial_means[teacher_action] - initial_means[action]
            )
            self.trained_teacher_pair_credit += _positive_credit(
                trained_means[teacher_action] - trained_means[action]
            )


def _fully_scored(row: SiblingEvaluation) -> bool:
    return (
        row.error is None
        and row.leaf_available
        and row.engine_score is not None
        and math.isfinite(row.engine_score)
        and row.initial_value is not None
        and math.isfinite(row.initial_value)
        and row.trained_value is not None
        and math.isfinite(row.trained_value)
    )


def _comparable(left: SiblingEvaluation, right: SiblingEvaluation) -> bool:
    if not _fully_scored(left) or not _fully_scored(right):
        return False
    return (
        left.endpoint == right.endpoint
        or MacroEndpoint.TERMINAL in {left.endpoint, right.endpoint}
    )


def _comparable_groups(
    rows: Sequence[SiblingEvaluation],
) -> tuple[tuple[SiblingEvaluation, ...], ...]:
    terminal = [row for row in rows if row.endpoint == MacroEndpoint.TERMINAL]
    groups: dict[MacroEndpoint, list[SiblingEvaluation]] = defaultdict(list)
    for row in rows:
        if row.endpoint != MacroEndpoint.TERMINAL:
            groups[row.endpoint].append(row)
    output = [tuple(group) for group in groups.values() if len(group) >= 2]
    if len(terminal) >= 2:
        output.append(tuple(terminal))
    return tuple(output)


def _direction_credit(predicted_delta: float, target_delta: float) -> float:
    if abs(predicted_delta) <= _TIE_EPSILON:
        return 0.5
    return float((predicted_delta > 0.0) == (target_delta > 0.0))


def _positive_credit(delta: float) -> float:
    if abs(delta) <= _TIE_EPSILON:
        return 0.5
    return float(delta > 0.0)


def _rank_credit(
    values: Mapping[tuple[int, ...], float],
    target: tuple[int, ...],
) -> tuple[float, float]:
    target_value = values[target]
    greater = sum(value > target_value + _TIE_EPSILON for value in values.values())
    tied = sum(abs(value - target_value) <= _TIE_EPSILON for value in values.values())
    average_rank = 1.0 + greater + 0.5 * (tied - 1)
    top1_credit = 1.0 / tied if greater == 0 else 0.0
    return 1.0 / average_rank, top1_credit


def _safe_rate(numerator: float | int, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


__all__ = ["SiblingEvaluation", "TeacherSiblingMetrics"]
