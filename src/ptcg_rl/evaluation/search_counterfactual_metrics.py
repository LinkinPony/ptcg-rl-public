"""Streaming S1 counterfactual, critic-ranking, and calibration metrics."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.agent.search.config import PairedRerankConfig, ScoreMode
from ptcg_rl.agent.search.macro import MacroEndpoint
from ptcg_rl.agent.search.reranker import MacroSearchResult
from ptcg_rl.agent.search.scoring import PairedWorldScore, select_paired_action


class CounterfactualMetricReferences(BaseModel):
    """Pre-registered S1 sample and ranking reference bands."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_roots: int = 500
    min_holdout_roots: int = 200
    min_complete_coverage_rate: float = 0.95
    min_holdout_comparable_pairs: int = 200
    min_holdout_pairwise_accuracy: float = 0.55
    max_holdout_top1_regret: float = 0.10
    min_telemetry_coverage: float = 0.999

    @field_validator(
        "min_roots",
        "min_holdout_roots",
        "min_holdout_comparable_pairs",
    )
    @classmethod
    def positive_counts(cls, value: int) -> int:
        """Require positive diagnostic sample references."""
        if value <= 0:
            raise ValueError("counterfactual metric sample references must be positive")
        return value

    @field_validator(
        "min_complete_coverage_rate",
        "min_holdout_pairwise_accuracy",
        "min_telemetry_coverage",
    )
    @classmethod
    def probability_reference(cls, value: float) -> float:
        """Restrict reference rates to the unit interval."""
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("counterfactual metric rate references must be in [0, 1]")
        return value

    @field_validator("max_holdout_top1_regret")
    @classmethod
    def finite_regret(cls, value: float) -> float:
        """Require a finite non-negative regret reference."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("counterfactual regret reference must be finite and non-negative")
        return value


@dataclass
class _SplitStats:
    roots: int = 0
    complete_roots: int = 0
    greedy_included: int = 0
    telemetry_roots: int = 0
    state_leaks: int = 0
    illegal_candidates: int = 0
    deadline_overruns: int = 0
    max_deadline_overshoot: float = 0.0
    max_state_pool_peak: int = 0
    engine_errors: int = 0
    deadline_roots: int = 0
    step_cap_roots: int = 0
    sibling_pairs: int = 0
    sibling_pair_credit: float = 0.0
    top1_regret_sum: float = 0.0
    top1_regret_count: int = 0
    root_values: list[tuple[float, float]] = field(default_factory=list)
    brier_by_episode: dict[int, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    search_seconds: list[float] = field(default_factory=list)
    whole_act_seconds: list[float] = field(default_factory=list)
    endpoint_counts: Counter[str] = field(default_factory=Counter)
    decision_roots: Counter[str] = field(default_factory=Counter)
    decision_changes: Counter[str] = field(default_factory=Counter)
    engine_supported_changes: Counter[str] = field(default_factory=Counter)


class CounterfactualAuditStats:
    """Accumulate root-grouped S1 evidence without loading persisted shards."""

    def __init__(
        self,
        *,
        rerank: PairedRerankConfig,
        margin_sweep: Sequence[float],
    ) -> None:
        self._rerank = rerank
        self._margin_sweep = tuple(float(margin) for margin in margin_sweep)
        self._splits: dict[str, _SplitStats] = {
            "dev": _SplitStats(),
            "holdout": _SplitStats(),
        }

    def update(
        self,
        *,
        split: str,
        episode_id: int,
        terminal_value: float | None,
        root_value: float | None,
        greedy_action: tuple[int, ...],
        result: MacroSearchResult,
        search_seconds: float,
        whole_act_seconds: float,
        deadline_overshoot_seconds: float,
        illegal_candidates: int,
        telemetry_complete: bool,
    ) -> None:
        """Consume one root and all of its paired-world evaluations."""
        stats = self._split(split)
        stats.roots += 1
        stats.complete_roots += int(result.complete_coverage)
        stats.greedy_included += int(result.candidates.contains(greedy_action))
        stats.telemetry_roots += int(telemetry_complete)
        stats.state_leaks += int(result.state_leaks)
        stats.illegal_candidates += int(illegal_candidates)
        stats.deadline_overruns += int(deadline_overshoot_seconds > 0.0)
        stats.max_deadline_overshoot = max(
            stats.max_deadline_overshoot,
            float(deadline_overshoot_seconds),
        )
        stats.max_state_pool_peak = max(
            stats.max_state_pool_peak,
            result.state_pool_peak,
        )
        stats.search_seconds.append(float(search_seconds))
        stats.whole_act_seconds.append(float(whole_act_seconds))
        for evaluation in result.evaluations:
            stats.endpoint_counts[evaluation.endpoint.value] += 1
            stats.engine_errors += int(
                evaluation.endpoint == MacroEndpoint.ENGINE_ERROR
            )
            stats.deadline_roots += int(evaluation.endpoint == MacroEndpoint.DEADLINE)
            stats.step_cap_roots += int(evaluation.endpoint == MacroEndpoint.STEP_CAP)

        self._update_critic_ranking(stats, result)
        self._update_decisions(
            stats,
            greedy_action,
            result,
            margins=(
                self._margin_sweep
                if split == "dev"
                else (self._rerank.switch_margin,)
            ),
        )
        if (
            terminal_value is not None
            and root_value is not None
            and math.isfinite(terminal_value)
            and math.isfinite(root_value)
        ):
            clipped_value = min(max(float(root_value), -1.0), 1.0)
            target = min(max(float(terminal_value), -1.0), 1.0)
            predicted_probability = (clipped_value + 1.0) / 2.0
            target_probability = (target + 1.0) / 2.0
            brier = (predicted_probability - target_probability) ** 2
            stats.root_values.append((clipped_value, target))
            stats.brier_by_episode[int(episode_id)].append(brier)

    def summary(self, references: CounterfactualMetricReferences) -> dict[str, Any]:
        """Return split metrics plus diagnostic correctness observations."""
        split_summaries = {
            name: _split_summary(stats) for name, stats in self._splits.items()
        }
        total = _merge_stats(tuple(self._splits.values()))
        total_summary = _split_summary(total)
        holdout = split_summaries["holdout"]
        correctness_checks = {
            "root_count": total.roots >= references.min_roots,
            "holdout_root_count": (
                self._splits["holdout"].roots >= references.min_holdout_roots
            ),
            "greedy_candidate_recall": total.greedy_included == total.roots,
            "complete_coverage": (
                float(total_summary["complete_coverage_rate"])
                >= references.min_complete_coverage_rate
            ),
            "state_lifecycle": total.state_leaks == 0,
            "candidate_legality": total.illegal_candidates == 0,
            "deadline_overrun": total.deadline_overruns == 0,
            "state_pool_limit": total.max_state_pool_peak <= 128,
            "engine_errors": total.engine_errors == 0,
            "telemetry_coverage": (
                float(total_summary["telemetry_coverage"])
                >= references.min_telemetry_coverage
            ),
        }
        critic_rank_checks = {
            "comparable_pairs": (
                self._splits["holdout"].sibling_pairs
                >= references.min_holdout_comparable_pairs
            ),
            "pairwise_accuracy": (
                float(holdout["critic_pairwise_accuracy"])
                >= references.min_holdout_pairwise_accuracy
            ),
            "top1_regret": (
                float(holdout["critic_top1_engine_regret"])
                <= references.max_holdout_top1_regret
            ),
        }
        warnings = [
            f"correctness:{name}"
            for name, observed in correctness_checks.items()
            if not observed
        ]
        warnings.extend(
            f"critic_rank:{name}"
            for name, observed in critic_rank_checks.items()
            if not observed
        )
        return {
            "splits": split_summaries,
            "total": total_summary,
            "decision_role": "diagnostic_only",
            "correctness_checks": correctness_checks,
            "critic_rank_checks": critic_rank_checks,
            "diagnostic_warnings": warnings,
        }

    def _split(self, split: str) -> _SplitStats:
        if split not in self._splits:
            raise ValueError(f"unknown replay split: {split}")
        return self._splits[split]

    def _update_critic_ranking(
        self,
        stats: _SplitStats,
        result: MacroSearchResult,
    ) -> None:
        rows_by_world: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for evaluation in result.evaluations:
            if (
                evaluation.endpoint != MacroEndpoint.SAME_SEAT_MAIN
                or evaluation.engine_score is None
                or evaluation.critic_value is None
                or not math.isfinite(evaluation.engine_score)
                or not math.isfinite(evaluation.critic_value)
            ):
                continue
            rows_by_world[evaluation.world_index].append(
                (evaluation.engine_score, evaluation.critic_value)
            )
        for rows in rows_by_world.values():
            for left, right in combinations(rows, 2):
                engine_delta = left[0] - right[0]
                if abs(engine_delta) <= 1.0e-12:
                    continue
                critic_delta = left[1] - right[1]
                stats.sibling_pairs += 1
                if critic_delta == 0.0:
                    stats.sibling_pair_credit += 0.5
                elif (critic_delta > 0.0) == (engine_delta > 0.0):
                    stats.sibling_pair_credit += 1.0
            if len(rows) >= 2:
                best_engine = max(value[0] for value in rows)
                critic_choice = max(rows, key=lambda value: value[1])
                stats.top1_regret_sum += max(0.0, best_engine - critic_choice[0])
                stats.top1_regret_count += 1

    def _update_decisions(
        self,
        stats: _SplitStats,
        greedy_action: tuple[int, ...],
        result: MacroSearchResult,
        *,
        margins: Sequence[float],
    ) -> None:
        world_scores = _world_scores(result)
        engine_evidence = select_paired_action(
            world_scores,
            actions=result.candidates.actions,
            greedy_action=greedy_action,
            worlds_requested=result.worlds_requested,
            config=self._rerank.model_copy(
                update={"score_mode": "engine_only", "switch_margin": 1.0e6}
            ),
        )
        engine_by_action = {
            score.action: score for score in engine_evidence.action_scores
        }
        for score_mode in (
            "engine_only",
            "engine_value_tiebreak",
            "value_primary",
        ):
            for margin in margins:
                key = _decision_key(score_mode, margin)
                config = self._rerank.model_copy(
                    update={"score_mode": score_mode, "switch_margin": margin}
                )
                decision = select_paired_action(
                    world_scores,
                    actions=result.candidates.actions,
                    greedy_action=greedy_action,
                    worlds_requested=result.worlds_requested,
                    config=config,
                )
                stats.decision_roots[key] += 1
                if not decision.action_changed:
                    continue
                stats.decision_changes[key] += 1
                evidence = engine_by_action.get(decision.selected_action)
                if (
                    evidence is not None
                    and evidence.robust_delta > 0.0
                    and evidence.downside_cvar >= config.minimum_downside_delta
                ):
                    stats.engine_supported_changes[key] += 1


def _world_scores(result: MacroSearchResult) -> tuple[PairedWorldScore, ...]:
    return tuple(
        PairedWorldScore(
            action=evaluation.action,
            world_index=evaluation.world_index,
            endpoint=evaluation.endpoint,
            engine_score=evaluation.engine_score,
            critic_value=evaluation.critic_value,
            error=evaluation.error,
        )
        for evaluation in result.evaluations
    )


def _decision_key(score_mode: ScoreMode, margin: float) -> str:
    return f"{score_mode}@{margin:.6g}"


def _split_summary(stats: _SplitStats) -> dict[str, Any]:
    roots = stats.roots
    values = stats.root_values
    decisions = {
        key: {
            "roots": stats.decision_roots[key],
            "changes": stats.decision_changes[key],
            "change_rate": _rate(stats.decision_changes[key], stats.decision_roots[key]),
            "engine_supported_changes": stats.engine_supported_changes[key],
            "engine_support_rate": _rate(
                stats.engine_supported_changes[key],
                stats.decision_changes[key],
            ),
        }
        for key in sorted(stats.decision_roots)
    }
    return {
        "roots": roots,
        "complete_coverage_rate": _rate(stats.complete_roots, roots),
        "greedy_candidate_recall": _rate(stats.greedy_included, roots),
        "telemetry_coverage": _rate(stats.telemetry_roots, roots),
        "state_leaks": stats.state_leaks,
        "illegal_candidates": stats.illegal_candidates,
        "deadline_overruns": stats.deadline_overruns,
        "max_deadline_overshoot_seconds": stats.max_deadline_overshoot,
        "max_state_pool_peak": stats.max_state_pool_peak,
        "engine_errors": stats.engine_errors,
        "deadline_evaluations": stats.deadline_roots,
        "step_cap_evaluations": stats.step_cap_roots,
        "endpoint_counts": dict(sorted(stats.endpoint_counts.items())),
        "critic_comparable_pairs": stats.sibling_pairs,
        "critic_pairwise_accuracy": _rate_float(
            stats.sibling_pair_credit,
            stats.sibling_pairs,
        ),
        "critic_top1_engine_regret": _rate_float(
            stats.top1_regret_sum,
            stats.top1_regret_count,
        ),
        "root_value_count": len(values),
        "root_value_brier": _brier(values),
        "root_value_ece": _ece(values),
        "root_value_auc": _auc(values),
        "episode_cluster_brier": _episode_cluster_brier(stats.brier_by_episode),
        "search_seconds": _distribution(stats.search_seconds),
        "whole_act_seconds": _distribution(stats.whole_act_seconds),
        "decisions": decisions,
    }


def _merge_stats(stats_rows: Sequence[_SplitStats]) -> _SplitStats:
    merged = _SplitStats()
    for stats in stats_rows:
        for name in (
            "roots",
            "complete_roots",
            "greedy_included",
            "telemetry_roots",
            "state_leaks",
            "illegal_candidates",
            "deadline_overruns",
            "engine_errors",
            "deadline_roots",
            "step_cap_roots",
            "sibling_pairs",
            "top1_regret_count",
        ):
            setattr(merged, name, getattr(merged, name) + getattr(stats, name))
        merged.sibling_pair_credit += stats.sibling_pair_credit
        merged.top1_regret_sum += stats.top1_regret_sum
        merged.max_deadline_overshoot = max(
            merged.max_deadline_overshoot,
            stats.max_deadline_overshoot,
        )
        merged.max_state_pool_peak = max(
            merged.max_state_pool_peak,
            stats.max_state_pool_peak,
        )
        merged.root_values.extend(stats.root_values)
        merged.search_seconds.extend(stats.search_seconds)
        merged.whole_act_seconds.extend(stats.whole_act_seconds)
        merged.endpoint_counts.update(stats.endpoint_counts)
        merged.decision_roots.update(stats.decision_roots)
        merged.decision_changes.update(stats.decision_changes)
        merged.engine_supported_changes.update(stats.engine_supported_changes)
        for episode_id, values in stats.brier_by_episode.items():
            merged.brier_by_episode[episode_id].extend(values)
    return merged


def _brier(values: Sequence[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    return statistics.fmean(
        (((prediction + 1.0) / 2.0) - ((target + 1.0) / 2.0)) ** 2
        for prediction, target in values
    )


def _ece(values: Sequence[tuple[float, float]], bins: int = 10) -> float:
    if not values:
        return 0.0
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for prediction, target in values:
        probability = (prediction + 1.0) / 2.0
        index = min(bins - 1, int(probability * bins))
        buckets[index].append((probability, (target + 1.0) / 2.0))
    return sum(
        (len(bucket) / len(values))
        * abs(
            statistics.fmean(row[0] for row in bucket)
            - statistics.fmean(row[1] for row in bucket)
        )
        for bucket in buckets
        if bucket
    )


def _auc(values: Sequence[tuple[float, float]]) -> float:
    binary = [(prediction, target) for prediction, target in values if target != 0.0]
    positives = [row for row in binary if row[1] > 0.0]
    negatives = [row for row in binary if row[1] < 0.0]
    if not positives or not negatives:
        return 0.0
    credit = 0.0
    for positive in positives:
        for negative in negatives:
            if positive[0] > negative[0]:
                credit += 1.0
            elif positive[0] == negative[0]:
                credit += 0.5
    return credit / float(len(positives) * len(negatives))


def _episode_cluster_brier(values: Mapping[int, Sequence[float]]) -> float:
    if not values:
        return 0.0
    return statistics.fmean(statistics.fmean(rows) for rows in values.values() if rows)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "p50": _quantile(ordered, 0.50),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": ordered[-1],
    }


def _quantile(ordered: Sequence[float], probability: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _rate(numerator: int, denominator: int) -> float:
    return numerator / float(denominator) if denominator else 0.0


def _rate_float(numerator: float, denominator: int) -> float:
    return numerator / float(denominator) if denominator else 0.0
