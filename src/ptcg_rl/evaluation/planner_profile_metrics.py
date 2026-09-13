"""Streaming profile summaries and one direct low-regret Pareto decision."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ptcg_rl.evaluation.planner_profile_config import (
    PlannerProfilePointConfig,
    PlannerProfileResourceContract,
)
from ptcg_rl.evaluation.planner_profile_records import (
    PlannerProfileDecisionRecord,
    PlannerProfileRunRecord,
)
from ptcg_rl.evaluation.planner_profile_writer import PlannerProfileWriter
from ptcg_rl.runtime.planner_telemetry import PlannerStage, PlannerStageEvent

_PLANNER_REQUIRED_STAGES = frozenset(
    {
        PlannerStage.QUEUE_WAIT,
        PlannerStage.BASE_PROPOSAL,
        PlannerStage.CANDIDATE_CONSTRUCTION,
        PlannerStage.AGGREGATION,
        PlannerStage.CANDIDATE_MODEL,
        PlannerStage.BEHAVIOR_SAMPLE,
    }
)


@dataclass(frozen=True)
class PlannerProfilePointSummary:
    """Bounded scalar diagnostics for one complete profile point."""

    point_id: str
    budget_id: str
    environment: str
    planner_enabled: bool
    decisions: int
    shape_counts: Mapping[str, int]
    eligible_decisions: int
    equivalence_probe_decisions: int
    planner_decisions: int
    fallback_decisions: int
    failed_decisions: int
    deadline_exceeded_decisions: int
    fallback_reasons: Mapping[str, int]
    total_p50_ms: float
    total_p95_ms: float
    total_p99_ms: float
    total_max_ms: float
    stage_p50_ms: Mapping[str, float]
    stage_p95_ms: Mapping[str, float]
    stage_p99_ms: Mapping[str, float]
    mean_candidate_count: float
    mean_unique_leaf_ratio: float
    mean_gpu_batch_fill: float
    mean_prefix_reuse_count: float
    mean_native_lane_occupancy: float
    mean_ipc_bytes: float | None
    actor_policy_wait_p95_ms: float
    planner_queue_wait_p95_ms: float
    mean_base_action_regret: float | None
    mean_served_action_regret: float | None
    mean_candidate_best_regret: float | None
    oracle_decisions: int
    oracle_executions: int
    served_epsilon_optimal_rate: float | None
    candidate_epsilon_recall_rate: float | None
    mean_value_calibration_error: float | None
    scenario_support_rows: int
    scenario_support_manifest_fingerprint: str | None
    batch_position_decisions: Mapping[str, int]
    batch_position_valid_evidence_rates: Mapping[str, float]
    batch_position_fallback_rates: Mapping[str, float]
    batch_position_queue_wait_p95_ms: Mapping[str, float]
    ipc_measurement_unavailable_reasons: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerProfileSelection:
    """One fixed budget or an explicit result that quality was unidentifiable."""

    selected_budget_id: str | None
    resource_feasible_budget_ids: tuple[str, ...]
    quality_rankable_budget_ids: tuple[str, ...]
    pareto_budget_ids: tuple[str, ...]
    invalid_reasons: Mapping[str, tuple[str, ...]]
    unranked_reasons: Mapping[str, tuple[str, ...]]


class PlannerProfileSink:
    """Validate point identity, stream rows, and retain scalar diagnostics."""

    def __init__(
        self,
        *,
        campaign_id: str,
        point: PlannerProfilePointConfig,
        writer: PlannerProfileWriter,
    ) -> None:
        self._campaign_id = campaign_id
        self._point = point
        self._writer = writer
        self._decisions = 0
        self._shape_counts: Counter[str] = Counter()
        self._eligible = 0
        self._probe = 0
        self._planner = 0
        self._failed = 0
        self._deadline_exceeded = 0
        self._fallback_reasons: Counter[str] = Counter()
        self._total_ms: list[float] = []
        self._stages_ms: defaultdict[str, list[float]] = defaultdict(list)
        self._candidate_count = 0
        self._unique_leaf_ratios: list[float] = []
        self._gpu_batch_fills: list[float] = []
        self._prefix_reuse_counts: list[int] = []
        self._native_lane_occupancies: list[int] = []
        self._ipc_bytes: list[int] = []
        self._ipc_measurement_unavailable_reasons: Counter[str] = Counter()
        self._actor_policy_wait_ms: list[float] = []
        self._planner_queue_wait_ms: list[float] = []
        self._base_regrets: list[float] = []
        self._served_regrets: list[float] = []
        self._candidate_regrets: list[float] = []
        self._served_epsilon: list[bool] = []
        self._candidate_epsilon: list[bool] = []
        self._calibration_errors: list[float] = []
        self._scenario_supports: dict[tuple[int, str], str] = {}
        self._oracle_keys: set[tuple[int, str]] = set()
        self._oracle_elapsed_ms = 0.0
        self._oracle_executions = 0
        self._batch_position_decisions: Counter[int] = Counter()
        self._batch_position_valid: Counter[int] = Counter()
        self._batch_position_fallback: Counter[int] = Counter()
        self._batch_position_queue_wait: defaultdict[int, list[float]] = defaultdict(
            list
        )

    @property
    def decisions(self) -> int:
        return self._decisions

    @property
    def scenario_supports(self) -> Mapping[tuple[int, str], str]:
        return dict(self._scenario_supports)

    @property
    def oracle_keys(self) -> frozenset[tuple[int, str]]:
        return frozenset(self._oracle_keys)

    @property
    def oracle_elapsed_ms(self) -> float:
        return self._oracle_elapsed_ms

    def record(
        self,
        decision: PlannerProfileDecisionRecord,
        events: Sequence[PlannerStageEvent],
    ) -> None:
        """Record one service/agent decision with complete stage telemetry."""
        if decision.campaign_id != self._campaign_id:
            raise ValueError("profile decision has the wrong campaign identity")
        if (
            decision.point_id != self._point.point_id
            or decision.budget_id != self._point.budget_id
            or decision.environment != self._point.environment
        ):
            raise ValueError("profile decision has the wrong point identity")
        if decision.decision_index != self._decisions:
            raise ValueError("profile decision indices must be contiguous")
        stages = tuple(event.stage for event in events)
        for event in events:
            if not math.isfinite(event.seconds) or event.seconds < 0.0:
                raise ValueError("profile stage time must be finite and non-negative")
            if event.rows < 0 or event.bytes_count < 0 or event.batch_capacity < 0:
                raise ValueError("profile stage shape must be non-negative")
            if event.batch_capacity and event.rows > event.batch_capacity:
                raise ValueError("profile stage rows exceed batch capacity")
        queue_wait_ms = sum(
            event.seconds * 1_000.0
            for event in events
            if event.stage is PlannerStage.QUEUE_WAIT
        )
        if not math.isclose(
            queue_wait_ms,
            decision.planner_queue_wait_ms,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError("planner queue wait differs from genuine stage timing")
        native_rows = sum(
            event.rows for event in events if event.stage is PlannerStage.NATIVE_ENGINE
        )
        if native_rows != decision.engine_transitions:
            raise ValueError("engine transition count differs from native events")
        gpu_events = tuple(
            event for event in events if event.stage is PlannerStage.GPU_LEAF_VALUE
        )
        if (
            sum(event.rows for event in gpu_events) != decision.gpu_rows
            or sum(event.batch_capacity for event in gpu_events)
            != decision.gpu_batch_capacity
        ):
            raise ValueError("GPU aggregate differs from genuine leaf events")
        if decision.planner_used and not _PLANNER_REQUIRED_STAGES.issubset(stages):
            raise ValueError("planner decision is missing required stage timing")
        if (
            decision.planner_used
            and decision.engine_transitions > 0
            and not {
                PlannerStage.NATIVE_QUEUE_WAIT,
                PlannerStage.NATIVE_ENGINE,
            }.issubset(stages)
        ):
            raise ValueError("planner native work is missing genuine engine timing")
        if (
            decision.planner_used
            and decision.unique_leaf_count > 0
            and not {
                PlannerStage.TENSORIZATION,
                PlannerStage.GPU_LEAF_VALUE,
            }.issubset(stages)
        ):
            raise ValueError("planner leaf work is missing tensor/model timing")
        if decision.planner_used and not self._point.planner_enabled:
            raise ValueError("planner-off control emitted planner behavior")
        if decision.policy_runtime_agent_used:
            raise ValueError(
                "compact independent roots must use the stateless shared service"
            )
        if decision.environment == "packaged_cpu_acttime" and (
            decision.gpu_rows != 0 or decision.gpu_batch_capacity != 0
        ):
            raise ValueError("packaged CPU point claimed GPU execution rows")
        if decision.environment == "h200_mps" and (decision.policy_runtime_agent_used):
            raise ValueError("H200 point must exercise the rollout service path")
        if (
            decision.environment == "h200_mps"
            and decision.planner_used
            and (decision.gpu_rows <= 0 or decision.gpu_batch_capacity <= 0)
        ):
            raise ValueError("H200 planner point omitted GPU batch execution")
        support_key = (decision.corpus_repetition, decision.corpus_row_id)
        if decision.scenario_support_fingerprint is not None:
            if support_key in self._scenario_supports:
                raise ValueError("profile point repeats a scenario support root")
            self._scenario_supports[support_key] = decision.scenario_support_fingerprint
        position = decision.batch_row_position
        self._batch_position_decisions[position] += 1
        self._batch_position_valid[position] += int(decision.planner_used)
        self._batch_position_fallback[position] += int(
            decision.fallback_reason is not None
        )
        self._batch_position_queue_wait[position].append(decision.planner_queue_wait_ms)
        self._writer.append_decision(decision, events)
        self._decisions += 1
        self._shape_counts.update(decision.decision_shapes)
        self._eligible += int(decision.planner_eligible)
        self._probe += int(decision.equivalence_probe_used)
        self._planner += int(decision.planner_used)
        self._failed += int(decision.failed)
        self._deadline_exceeded += int(decision.deadline_exceeded)
        if decision.fallback_reason is not None:
            self._fallback_reasons[decision.fallback_reason] += 1
        self._total_ms.append(decision.total_latency_ms)
        for event in events:
            self._stages_ms[str(event.stage)].append(event.seconds * 1000.0)
        self._candidate_count += decision.candidate_count
        if decision.consequence_cell_count > 0:
            self._unique_leaf_ratios.append(
                decision.unique_leaf_count / decision.consequence_cell_count
            )
        if decision.gpu_batch_capacity > 0:
            self._gpu_batch_fills.append(
                decision.gpu_rows / decision.gpu_batch_capacity
            )
        self._prefix_reuse_counts.append(decision.prefix_reuse_count)
        self._native_lane_occupancies.append(decision.native_lane_occupancy)
        if decision.ipc_bytes is None:
            assert decision.ipc_measurement_unavailable_reason is not None
            self._ipc_measurement_unavailable_reasons[
                decision.ipc_measurement_unavailable_reason
            ] += 1
        else:
            self._ipc_bytes.append(decision.ipc_bytes)
        self._actor_policy_wait_ms.append(decision.actor_policy_wait_ms)
        self._planner_queue_wait_ms.append(decision.planner_queue_wait_ms)
        if decision.base_action_regret is not None:
            self._base_regrets.append(decision.base_action_regret)
        if decision.served_action_regret is not None:
            self._served_regrets.append(decision.served_action_regret)
            self._oracle_keys.add(support_key)
            assert decision.oracle_latency_ms is not None
            self._oracle_elapsed_ms += decision.oracle_latency_ms
            self._oracle_executions += int(decision.oracle_executed)
        if decision.candidate_best_regret is not None:
            self._candidate_regrets.append(decision.candidate_best_regret)
        if decision.served_epsilon_optimal is not None:
            self._served_epsilon.append(decision.served_epsilon_optimal)
        if decision.candidate_epsilon_recall is not None:
            self._candidate_epsilon.append(decision.candidate_epsilon_recall)
        if decision.value_calibration_error is not None:
            self._calibration_errors.append(decision.value_calibration_error)

    def summary(self) -> PlannerProfilePointSummary:
        return PlannerProfilePointSummary(
            point_id=self._point.point_id,
            budget_id=self._point.budget_id,
            environment=self._point.environment,
            planner_enabled=self._point.planner_enabled,
            decisions=self._decisions,
            shape_counts=dict(sorted(self._shape_counts.items())),
            eligible_decisions=self._eligible,
            equivalence_probe_decisions=self._probe,
            planner_decisions=self._planner,
            fallback_decisions=sum(self._fallback_reasons.values()),
            failed_decisions=self._failed,
            deadline_exceeded_decisions=self._deadline_exceeded,
            fallback_reasons=dict(sorted(self._fallback_reasons.items())),
            total_p50_ms=_percentile(self._total_ms, 0.50),
            total_p95_ms=_percentile(self._total_ms, 0.95),
            total_p99_ms=_percentile(self._total_ms, 0.99),
            total_max_ms=max(self._total_ms, default=0.0),
            stage_p50_ms=_stage_percentiles(self._stages_ms, 0.50),
            stage_p95_ms=_stage_percentiles(self._stages_ms, 0.95),
            stage_p99_ms=_stage_percentiles(self._stages_ms, 0.99),
            mean_candidate_count=(
                self._candidate_count / self._decisions if self._decisions else 0.0
            ),
            mean_unique_leaf_ratio=_mean(self._unique_leaf_ratios),
            mean_gpu_batch_fill=_mean(self._gpu_batch_fills),
            mean_prefix_reuse_count=_mean(self._prefix_reuse_counts),
            mean_native_lane_occupancy=_mean(self._native_lane_occupancies),
            mean_ipc_bytes=(_mean(self._ipc_bytes) if self._ipc_bytes else None),
            actor_policy_wait_p95_ms=_percentile(self._actor_policy_wait_ms, 0.95),
            planner_queue_wait_p95_ms=_percentile(self._planner_queue_wait_ms, 0.95),
            mean_base_action_regret=(
                _mean(self._base_regrets) if self._base_regrets else None
            ),
            mean_served_action_regret=(
                _mean(self._served_regrets) if self._served_regrets else None
            ),
            mean_candidate_best_regret=(
                _mean(self._candidate_regrets) if self._candidate_regrets else None
            ),
            oracle_decisions=len(self._served_regrets),
            oracle_executions=self._oracle_executions,
            served_epsilon_optimal_rate=(
                sum(self._served_epsilon) / len(self._served_epsilon)
                if self._served_epsilon
                else None
            ),
            candidate_epsilon_recall_rate=(
                sum(self._candidate_epsilon) / len(self._candidate_epsilon)
                if self._candidate_epsilon
                else None
            ),
            mean_value_calibration_error=(
                _mean(self._calibration_errors) if self._calibration_errors else None
            ),
            scenario_support_rows=len(self._scenario_supports),
            scenario_support_manifest_fingerprint=(
                _scenario_support_manifest_fingerprint(self._scenario_supports)
                if self._scenario_supports
                else None
            ),
            batch_position_decisions={
                str(position): count
                for position, count in sorted(self._batch_position_decisions.items())
            },
            batch_position_valid_evidence_rates={
                str(position): _safe_fraction(
                    self._batch_position_valid[position],
                    count,
                )
                for position, count in sorted(self._batch_position_decisions.items())
            },
            batch_position_fallback_rates={
                str(position): _safe_fraction(
                    self._batch_position_fallback[position],
                    count,
                )
                for position, count in sorted(self._batch_position_decisions.items())
            },
            batch_position_queue_wait_p95_ms={
                str(position): _percentile(values, 0.95)
                for position, values in sorted(self._batch_position_queue_wait.items())
            },
            ipc_measurement_unavailable_reasons=dict(
                sorted(self._ipc_measurement_unavailable_reasons.items())
            ),
        )


def select_profile_budget(
    contract: PlannerProfileResourceContract,
    *,
    point_summaries: Sequence[PlannerProfilePointSummary],
    run_records: Sequence[PlannerProfileRunRecord],
) -> PlannerProfileSelection:
    """Choose once from resource-valid points using direct Pareto objectives."""
    summaries = {(item.budget_id, item.environment): item for item in point_summaries}
    runs = {(item.budget_id, item.environment): item for item in run_records}
    budgets = sorted({key[0] for key in summaries} | {key[0] for key in runs})
    invalid: dict[str, tuple[str, ...]] = {}
    unranked: dict[str, tuple[str, ...]] = {}
    resource_feasible: list[str] = []
    rankable: list[str] = []
    objectives: dict[str, tuple[float, ...]] = {}
    for budget_id in budgets:
        h200_summary = summaries.get((budget_id, "h200_mps"))
        packaged_summary = summaries.get((budget_id, "packaged_cpu_acttime"))
        h200_run = runs.get((budget_id, "h200_mps"))
        packaged_run = runs.get((budget_id, "packaged_cpu_acttime"))
        invalid_reasons: list[str] = []
        if any(
            item is None
            for item in (h200_summary, packaged_summary, h200_run, packaged_run)
        ):
            invalid_reasons.append("incomplete_paired_campaign")
        else:
            assert h200_summary is not None
            assert packaged_summary is not None
            assert h200_run is not None
            assert packaged_run is not None
            if h200_run.failed_decisions or packaged_run.failed_decisions:
                invalid_reasons.append("failed_decision_evidence")
            if packaged_run.act_time_replay_exhausted_episodes:
                invalid_reasons.append("packaged_act_time_exhausted")
            if packaged_run.peak_host_bytes > contract.packaged_hard_peak_host_bytes:
                invalid_reasons.append("packaged_hard_host_memory")
            if packaged_run.peak_vram_bytes != 0:
                invalid_reasons.append("packaged_gpu_execution")
            if h200_run.peak_vram_bytes > contract.h200_hard_peak_vram_bytes:
                invalid_reasons.append("h200_hard_vram")
            if h200_run.peak_vram_bytes == 0:
                invalid_reasons.append("h200_gpu_execution_not_measured")
            if h200_run.peak_host_bytes > contract.h200_hard_peak_host_bytes:
                invalid_reasons.append("h200_hard_host_memory")
            if h200_run.learner_updates <= 0 or h200_run.learner_kernel_rows <= 0:
                invalid_reasons.append("learner_kernel_workload_not_measured")
            expected_calls = packaged_run.act_time_replay_decisions
            if packaged_run.policy_runtime_agent_calls != expected_calls:
                invalid_reasons.append("packaged_agent_path_not_measured")
            h200_quality = h200_summary.mean_served_action_regret
            packaged_quality = packaged_summary.mean_served_action_regret
            if (h200_quality is None) != (packaged_quality is None) or (
                h200_summary.oracle_decisions != packaged_summary.oracle_decisions
            ):
                invalid_reasons.append("unpaired_oracle_evidence")
        if invalid_reasons:
            invalid[budget_id] = tuple(invalid_reasons)
            continue
        resource_feasible.append(budget_id)
        assert h200_summary is not None
        assert packaged_summary is not None
        assert h200_run is not None
        assert packaged_run is not None
        h200_regret = h200_summary.mean_served_action_regret
        packaged_regret = packaged_summary.mean_served_action_regret
        if h200_regret is None or packaged_regret is None:
            unranked[budget_id] = ("shape_coverage_served_regret_not_identified",)
            continue
        rankable.append(budget_id)
        objectives[budget_id] = (
            _mean((h200_regret, packaged_regret)),
            packaged_run.act_time_replay_elapsed_seconds,
            -packaged_run.act_time_replay_min_remaining_seconds,
            -h200_run.valid_evidence_decisions_per_second,
            -h200_run.learner_kernel_rows_per_second,
            float(h200_run.peak_vram_bytes),
            h200_run.cpu_lane_utilization,
        )
    pareto = tuple(
        budget_id
        for budget_id in rankable
        if not any(
            _dominates(objectives[other], objectives[budget_id])
            for other in rankable
            if other != budget_id
        )
    )
    selected = (
        min(pareto, key=lambda item: (*objectives[item], item)) if pareto else None
    )
    return PlannerProfileSelection(
        selected_budget_id=selected,
        resource_feasible_budget_ids=tuple(resource_feasible),
        quality_rankable_budget_ids=tuple(rankable),
        pareto_budget_ids=pareto,
        invalid_reasons=dict(sorted(invalid.items())),
        unranked_reasons=dict(sorted(unranked.items())),
    )


def point_summary_dict(summary: PlannerProfilePointSummary) -> dict[str, object]:
    """Convert one summary to JSON-native values."""
    return {
        "point_id": summary.point_id,
        "budget_id": summary.budget_id,
        "environment": summary.environment,
        "planner_enabled": summary.planner_enabled,
        "decisions": summary.decisions,
        "shape_counts": dict(summary.shape_counts),
        "eligible_decisions": summary.eligible_decisions,
        "equivalence_probe_decisions": summary.equivalence_probe_decisions,
        "planner_decisions": summary.planner_decisions,
        "fallback_decisions": summary.fallback_decisions,
        "failed_decisions": summary.failed_decisions,
        "deadline_exceeded_decisions": summary.deadline_exceeded_decisions,
        "fallback_reasons": dict(summary.fallback_reasons),
        "total_p50_ms": summary.total_p50_ms,
        "total_p95_ms": summary.total_p95_ms,
        "total_p99_ms": summary.total_p99_ms,
        "total_max_ms": summary.total_max_ms,
        "stage_p50_ms": dict(summary.stage_p50_ms),
        "stage_p95_ms": dict(summary.stage_p95_ms),
        "stage_p99_ms": dict(summary.stage_p99_ms),
        "mean_candidate_count": summary.mean_candidate_count,
        "mean_unique_leaf_ratio": summary.mean_unique_leaf_ratio,
        "mean_gpu_batch_fill": summary.mean_gpu_batch_fill,
        "mean_prefix_reuse_count": summary.mean_prefix_reuse_count,
        "mean_native_lane_occupancy": summary.mean_native_lane_occupancy,
        "mean_ipc_bytes": summary.mean_ipc_bytes,
        "ipc_measurement_unavailable_reasons": dict(
            summary.ipc_measurement_unavailable_reasons
        ),
        "actor_policy_wait_p95_ms": summary.actor_policy_wait_p95_ms,
        "planner_queue_wait_p95_ms": summary.planner_queue_wait_p95_ms,
        "mean_base_action_regret": summary.mean_base_action_regret,
        "mean_served_action_regret": summary.mean_served_action_regret,
        "mean_candidate_best_regret": summary.mean_candidate_best_regret,
        "oracle_decisions": summary.oracle_decisions,
        "oracle_executions": summary.oracle_executions,
        "served_epsilon_optimal_rate": summary.served_epsilon_optimal_rate,
        "candidate_epsilon_recall_rate": summary.candidate_epsilon_recall_rate,
        "mean_value_calibration_error": summary.mean_value_calibration_error,
        "scenario_support_rows": summary.scenario_support_rows,
        "scenario_support_manifest_fingerprint": (
            summary.scenario_support_manifest_fingerprint
        ),
        "batch_position_decisions": dict(summary.batch_position_decisions),
        "batch_position_valid_evidence_rates": dict(
            summary.batch_position_valid_evidence_rates
        ),
        "batch_position_fallback_rates": dict(summary.batch_position_fallback_rates),
        "batch_position_queue_wait_p95_ms": dict(
            summary.batch_position_queue_wait_p95_ms
        ),
    }


def _stage_percentiles(
    values: Mapping[str, Sequence[float]],
    quantile: float,
) -> dict[str, float]:
    return {
        name: _percentile(items, quantile) for name, items in sorted(values.items())
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _mean(values: Sequence[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _scenario_support_manifest_fingerprint(
    supports: Mapping[tuple[int, str], str],
) -> str:
    payload = [
        {
            "corpus_repetition": repetition,
            "corpus_row_id": row_id,
            "scenario_support_fingerprint": fingerprint,
        }
        for (repetition, row_id), fingerprint in sorted(supports.items())
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True)) and any(
        a < b for a, b in zip(left, right, strict=True)
    )


__all__ = [
    "PlannerProfilePointSummary",
    "PlannerProfileSelection",
    "PlannerProfileSink",
    "point_summary_dict",
    "select_profile_budget",
]
