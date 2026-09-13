"""Stable Arrow schemas for integrated planner profile records."""

from __future__ import annotations

import pyarrow as pa

from ptcg_rl.evaluation.planner_profile_records import PlannerProfileRunRecord


def decision_schema() -> pa.Schema:
    """Return the stable per-decision profile schema."""
    string_fields = (
        "campaign_id",
        "point_id",
        "budget_id",
        "environment",
        "corpus_row_id",
        "model_fingerprint",
        "runtime_fingerprint",
        "planner_fingerprint",
    )
    count_fields = (
        "corpus_repetition",
        "decision_index",
        "batch_row_position",
        "policy_version",
        "proposal_version",
        "legal_action_count",
        "candidate_count",
        "scenario_count",
        "belief_world_count",
        "chance_outcome_count",
        "engine_transitions",
        "prefix_nodes",
        "prefix_reuse_count",
        "unique_leaf_count",
        "consequence_cell_count",
        "native_chunk_size",
        "gpu_rows",
        "gpu_batch_capacity",
        "native_lane_occupancy",
        "oracle_candidate_count",
        "oracle_scenario_count",
        "oracle_engine_transitions",
    )
    bool_fields = (
        "planner_eligible",
        "equivalence_probe_used",
        "planner_used",
        "policy_runtime_agent_used",
        "support_exhaustive",
        "scenario_grid_complete",
        "rules_exact",
        "leaf_bootstrapped",
        "failed",
        "deadline_exceeded",
        "oracle_compatible",
        "oracle_executed",
        "oracle_scenario_grid_complete",
        "oracle_rules_exact",
    )
    return pa.schema(
        [
            *[pa.field(name, pa.string(), nullable=False) for name in string_fields],
            pa.field(
                "decision_shapes",
                pa.list_(pa.string()),
                nullable=False,
            ),
            *[pa.field(name, pa.int64(), nullable=False) for name in count_fields],
            *[pa.field(name, pa.bool_(), nullable=False) for name in bool_fields],
            pa.field("fallback_reason", pa.string()),
            pa.field("oracle_exclusion_reason", pa.string()),
            pa.field("oracle_provenance_fingerprint", pa.string()),
            pa.field("scenario_support_fingerprint", pa.string()),
            pa.field("ipc_bytes", pa.int64()),
            pa.field("ipc_measurement_unavailable_reason", pa.string()),
            pa.field("base_action_fingerprint", pa.string(), nullable=False),
            pa.field("selected_action_fingerprint", pa.string(), nullable=False),
            *[
                pa.field(name, pa.float64(), nullable=False)
                for name in (
                    "model_lease_lifetime_ms",
                    "actor_policy_wait_ms",
                    "planner_queue_wait_ms",
                    "total_latency_ms",
                )
            ],
            pa.field("base_planner_agree", pa.bool_()),
            pa.field("oracle_latency_ms", pa.float64()),
            pa.field("base_action_regret", pa.float64()),
            pa.field("served_action_regret", pa.float64()),
            pa.field("candidate_best_regret", pa.float64()),
            pa.field("served_epsilon_optimal", pa.bool_()),
            pa.field("candidate_epsilon_recall", pa.bool_()),
            pa.field("value_calibration_error", pa.float64()),
        ]
    )


def stage_schema() -> pa.Schema:
    """Return the one-row-per-stage timing schema."""
    return pa.schema(
        [
            pa.field("campaign_id", pa.string(), nullable=False),
            pa.field("point_id", pa.string(), nullable=False),
            pa.field("environment", pa.string(), nullable=False),
            pa.field("decision_index", pa.int64(), nullable=False),
            pa.field("stage", pa.string(), nullable=False),
            pa.field("seconds", pa.float64(), nullable=False),
            pa.field("rows", pa.int64(), nullable=False),
            pa.field("bytes_count", pa.int64(), nullable=False),
            pa.field("batch_capacity", pa.int64(), nullable=False),
        ]
    )


def run_schema() -> pa.Schema:
    """Infer the stable run schema directly from the validated model."""
    sample = _run_schema_sample().model_dump(mode="json")
    fields: list[pa.Field] = []
    for name, value in sample.items():
        if name == "gil_utilization":
            fields.append(pa.field(name, pa.float64()))
            continue
        if name == "gil_measurement_unavailable_reason":
            fields.append(pa.field(name, pa.string()))
            continue
        if name in {
            "packaged_archive_fingerprint",
            "packaged_required_files_fingerprint",
        }:
            fields.append(pa.field(name, pa.string()))
            continue
        if isinstance(value, bool):
            dtype = pa.bool_()
        elif isinstance(value, int):
            dtype = pa.int64()
        elif isinstance(value, float):
            dtype = pa.float64()
        else:
            dtype = pa.string()
        fields.append(pa.field(name, dtype, nullable=False))
    return pa.schema(fields)


def _run_schema_sample() -> PlannerProfileRunRecord:
    digest = "0" * 64
    return PlannerProfileRunRecord(
        campaign_id="schema",
        point_id="schema",
        budget_id="schema",
        environment="h200_mps",
        planner_enabled=False,
        checkpoint_sha256=digest,
        decision_corpus_sha256=digest,
        workload_fingerprint=digest,
        machine_fingerprint=digest,
        model_fingerprint=digest,
        runtime_fingerprint=digest,
        planner_fingerprint=digest,
        controller_fingerprint=digest,
        constructor_fingerprint=digest,
        scorer_fingerprint=digest,
        tensor_schema_fingerprint=digest,
        native_abi_fingerprint=digest,
        native_schema_fingerprint=digest,
        native_library_fingerprint=digest,
        packaged_archive_fingerprint=None,
        packaged_required_files_fingerprint=None,
        policy_version=0,
        proposal_version=0,
        decisions=0,
        valid_planner_evidence_decisions=0,
        learner_warmup_updates=0,
        learner_updates=0,
        learner_optimizer_steps=0,
        learner_kernel_rows=0,
        policy_runtime_agent_warmup_calls=0,
        policy_runtime_agent_calls=0,
        packaged_isolated_processes=0,
        act_time_replay_episodes=0,
        act_time_replay_decisions=0,
        act_time_replay_exhausted_episodes=0,
        act_time_replay_elapsed_seconds=0.0,
        act_time_replay_min_remaining_seconds=0.0,
        oracle_decisions=0,
        oracle_elapsed_seconds=0.0,
        elapsed_seconds=0.0,
        inference_decisions_per_second=0.0,
        learner_kernel_rows_per_second=0.0,
        valid_evidence_decisions_per_second=0.0,
        checkpoint_cadence_seconds=0.0,
        cpu_lane_utilization=0.0,
        gil_utilization=0.0,
        gil_measurement_unavailable_reason=None,
        actor_idle_fraction=0.0,
        peak_vram_bytes=0,
        peak_host_bytes=0,
        fallback_decisions=0,
        failed_decisions=0,
        deadline_exceeded_decisions=0,
    )


__all__ = ["decision_schema", "run_schema", "stage_schema"]
