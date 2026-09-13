"""Campaign summary and selected-runtime artifacts for planner profiling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ptcg_rl.agent.search.root_information_tensorizer import (
    ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT,
)
from ptcg_rl.evaluation.planner_profile_config import IntegratedPlannerProfileConfig
from ptcg_rl.evaluation.planner_profile_metrics import (
    PlannerProfilePointSummary,
    PlannerProfileSelection,
    point_summary_dict,
)
from ptcg_rl.evaluation.planner_profile_records import PlannerProfileRunRecord


def build_campaign_summary(
    config: IntegratedPlannerProfileConfig,
    *,
    checkpoint_sha256: str,
    model_fingerprint: str,
    native_library_sha256: str,
    native_abi_fingerprint: str,
    native_schema_fingerprint: str,
    belief_prior_sha256: str,
    belief_runtime_fingerprint: str,
    act_time_replay_assets: Sequence[Mapping[str, Any]],
    corpus_rows: int,
    corpus_shape_counts: Mapping[str, int],
    point_summaries: Sequence[PlannerProfilePointSummary],
    run_records: Sequence[PlannerProfileRunRecord],
    selection: PlannerProfileSelection,
    part_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Build a small summary with explicit estimand and topology boundaries."""
    h200_runtime = next(
        config.runtime_for(point)
        for point in config.points
        if point.environment == "h200_mps"
    )
    actor_count = h200_runtime.planner.batching.actor_count
    root_rows_per_actor = h200_runtime.planner.batching.max_root_rows_per_request
    return {
        "campaign_id": config.campaign_id,
        "checkpoint_sha256": checkpoint_sha256,
        "model_fingerprint": model_fingerprint,
        "model_migration_seed": config.model_migration_seed,
        "native_library_sha256": native_library_sha256,
        "native_abi_fingerprint": native_abi_fingerprint,
        "native_schema_fingerprint": native_schema_fingerprint,
        "belief_prior_sha256": belief_prior_sha256,
        "belief_runtime_fingerprint": belief_runtime_fingerprint,
        "act_time_replay_assets": [dict(item) for item in act_time_replay_assets],
        "decision_corpus_sha256": config.expected_decision_corpus_sha256,
        "decision_corpus_rows": corpus_rows,
        "decision_corpus_shape_counts": dict(corpus_shape_counts),
        "tensor_schema_fingerprint": ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT,
        "quality_estimand": "uniform_shape_coverage_served_action_regret",
        "quality_applicability": (
            "The compact corpus is a multi-label structural coverage sample, not "
            "a deployment-frequency-weighted outcome distribution."
        ),
        "act_time_applicability": (
            "Ordered third-party episodes are deck-bound runtime stress evidence; "
            "they are not deployment action-quality evidence."
        ),
        "selection_objectives": [
            "min_mean_paired_served_action_regret",
            "min_ordered_act_time_total_elapsed_seconds",
            "max_ordered_act_time_worst_remaining_seconds",
            "max_h200_valid_evidence_decisions_per_second",
            "max_h200_learner_kernel_rows_per_second",
            "min_h200_peak_vram_bytes",
            "min_h200_cpu_lane_utilization",
        ],
        "compact_root_latency_role": "diagnostic_only",
        "learner_kernel_applicability": (
            "The fixed 1024-row full-objective synthetic learner kernel measures "
            "relative H200 contention only; it is not rollout-distribution, GAE, "
            "or end-to-end learner-consumption evidence."
        ),
        "learner_kernel_runtime_id": config.learner_kernel_runtime_id,
        "h200_serving_geometry": {
            "actor_count": actor_count,
            "root_rows_per_actor_request": root_rows_per_actor,
            "aggregate_root_wave_rows": actor_count * root_rows_per_actor,
            "formal_collection_num_concurrent_games_per_actor": (
                config.formal_collection_num_concurrent_games_per_actor
            ),
            "v1_live_game_rate_comparable": False,
        },
        "point_execution_order_method": "campaign_id_sha256_permutation_v1",
        "point_execution_order": [item.point_id for item in point_summaries],
        "points": [point_summary_dict(item) for item in point_summaries],
        "runs": [record.model_dump(mode="json") for record in run_records],
        "relative_controls": _relative_controls(point_summaries, run_records),
        "selection": {
            "selected_budget_id": selection.selected_budget_id,
            "resource_feasible_budget_ids": list(
                selection.resource_feasible_budget_ids
            ),
            "quality_rankable_budget_ids": list(selection.quality_rankable_budget_ids),
            "pareto_budget_ids": list(selection.pareto_budget_ids),
            "invalid_reasons": {
                key: list(value) for key, value in selection.invalid_reasons.items()
            },
            "unranked_reasons": {
                key: list(value) for key, value in selection.unranked_reasons.items()
            },
        },
        "part_counts": dict(part_counts),
        "selected_runtime_artifact": (
            str(config.output_dir / "selected_runtime.json")
            if selection.selected_budget_id is not None
            else None
        ),
    }


def selected_runtime_artifact(
    config: IntegratedPlannerProfileConfig,
    *,
    selection: PlannerProfileSelection,
) -> dict[str, Any] | None:
    """Return exact train/serve blocks for the one direct campaign decision."""
    budget_id = selection.selected_budget_id
    if budget_id is None:
        return None
    selected = {
        point.environment: point
        for point in config.points
        if point.budget_id == budget_id
    }
    if set(selected) != {"h200_mps", "packaged_cpu_acttime"}:
        raise ValueError("selected profile budget has no exact paired runtime")
    result: dict[str, Any] = {
        "campaign_id": config.campaign_id,
        "selected_budget_id": budget_id,
        "quality_estimand": "uniform_shape_coverage_served_action_regret",
        "checkpoint_path": str(config.checkpoint_path),
        "checkpoint_sha256": config.expected_checkpoint_sha256,
        "model_fingerprint": config.expected_model_fingerprint,
        "model_migration_seed": config.model_migration_seed,
        "policy_version": config.policy_version,
        "proposal_version": config.proposal_version,
        "learner_kernel_runtime_id": config.learner_kernel_runtime_id,
    }
    for environment, point in sorted(selected.items()):
        runtime = config.runtime_for(point)
        model_identity = config.model_identity_for(point)
        identity = runtime.planner.resolve_for_lease(
            model_fingerprint=model_identity.model_fingerprint,
            policy_version=model_identity.policy_version,
            proposal_version=model_identity.proposal_version,
        )
        result[environment] = {
            "point_id": point.point_id,
            "runtime_id": point.runtime_id,
            "planner_enabled": point.planner_enabled,
            "model_identity": model_identity.model_dump(mode="json"),
            "runtime_fingerprint": identity.runtime_fingerprint,
            "planner": runtime.planner.model_dump(mode="json"),
            "belief": runtime.belief.model_dump(mode="json"),
        }
        if environment == "h200_mps":
            result[environment]["formal_topology"] = {
                "collection_num_concurrent_games_per_actor": (
                    config.formal_collection_num_concurrent_games_per_actor
                ),
                "actor_count": runtime.planner.batching.actor_count,
                "aggregate_root_wave_rows": (
                    runtime.planner.batching.actor_count
                    * runtime.planner.batching.max_root_rows_per_request
                ),
                "v1_live_game_rate_comparable": False,
            }
        if runtime.packaged is not None:
            result[environment]["packaged_agent"] = {
                "device": runtime.packaged.device,
                "act_time": runtime.packaged.act_time.model_dump(mode="json"),
                "package_asset_id": runtime.packaged.package_asset_id,
                "planner_enabled_by_default": (
                    runtime.packaged.planner_enabled_by_default
                ),
            }
    return result


def _relative_controls(
    summaries: Sequence[PlannerProfilePointSummary],
    records: Sequence[PlannerProfileRunRecord],
) -> dict[str, dict[str, float]]:
    control_summaries = {
        item.environment: item for item in summaries if not item.planner_enabled
    }
    control_runs: dict[str, PlannerProfileRunRecord] = {
        item.environment: item for item in records if not item.planner_enabled
    }
    result: dict[str, dict[str, float]] = {}
    for summary, run in zip(summaries, records, strict=True):
        if not summary.planner_enabled:
            continue
        control_summary = control_summaries[summary.environment]
        control_run = control_runs[summary.environment]
        result[summary.point_id] = {
            "p95_latency_increment_ms": (
                summary.total_p95_ms - control_summary.total_p95_ms
            ),
            "inference_decisions_per_second_ratio": _safe_ratio(
                run.inference_decisions_per_second,
                control_run.inference_decisions_per_second,
            ),
            "learner_kernel_rows_per_second_ratio": _safe_ratio(
                run.learner_kernel_rows_per_second,
                control_run.learner_kernel_rows_per_second,
            ),
            "peak_vram_increment_bytes": float(
                run.peak_vram_bytes - control_run.peak_vram_bytes
            ),
            "act_time_replay_elapsed_seconds_increment": (
                run.act_time_replay_elapsed_seconds
                - control_run.act_time_replay_elapsed_seconds
            ),
            "act_time_replay_min_remaining_seconds_increment": (
                run.act_time_replay_min_remaining_seconds
                - control_run.act_time_replay_min_remaining_seconds
            ),
        }
    return result


def _safe_ratio(value: float, baseline: float) -> float:
    return 0.0 if baseline <= 0.0 else value / baseline


__all__ = ["build_campaign_summary", "selected_runtime_artifact"]
