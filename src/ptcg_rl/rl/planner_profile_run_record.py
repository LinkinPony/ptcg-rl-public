"""Point-level resource and throughput record construction."""

from __future__ import annotations

from dataclasses import dataclass

from ptcg_rl.evaluation.planner_profile_config import (
    IntegratedPlannerProfileConfig,
    PlannerProfileModelIdentity,
    PlannerProfilePointConfig,
    PlannerProfileRuntimeConfig,
)
from ptcg_rl.evaluation.planner_profile_records import PlannerProfileRunRecord
from ptcg_rl.rl.planner_profile_learner import ProfileLearnerResult
from ptcg_rl.rl.planner_profile_oracle import ProfileOracleCacheValue
from ptcg_rl.rl.planner_profile_packaged import PackagedReplayResult
from ptcg_rl.rl.planner_profile_resources import (
    ProfileResourcePeak,
    profile_machine_fingerprint,
)
from ptcg_rl.rl.planner_runtime_identity import ResolvedPlannerRuntimeIdentity

_GIL_UNAVAILABLE = "exact_gil_ownership_api_unavailable"


@dataclass(frozen=True, slots=True)
class ProfileRunMeasurements:
    """Scalar outputs retained after both concurrent workloads finish."""

    decisions: int
    elapsed_seconds: float
    actor_busy_seconds: float
    native_seconds: float
    valid_planner_decisions: int
    fallback_decisions: int
    failed_decisions: int
    deadline_decisions: int
    resource_peak: ProfileResourcePeak
    learner: ProfileLearnerResult | None
    replay: PackagedReplayResult | None
    oracle_executions: tuple[ProfileOracleCacheValue, ...]
    archive_sha256: str | None
    required_files_fingerprint: str | None


def build_profile_run_record(
    *,
    config: IntegratedPlannerProfileConfig,
    point: PlannerProfilePointConfig,
    runtime: PlannerProfileRuntimeConfig,
    model_identity: PlannerProfileModelIdentity,
    resolved: ResolvedPlannerRuntimeIdentity,
    measured: ProfileRunMeasurements,
) -> PlannerProfileRunRecord:
    """Build one validated run row from already-measured physical work."""
    elapsed = measured.elapsed_seconds
    learner = measured.learner
    replay = measured.replay
    learner_kernel_rows = 0 if learner is None else learner.kernel_rows
    learner_seconds = 0.0 if learner is None else learner.elapsed_seconds
    replay_decisions = 0 if replay is None else replay.decisions
    replay_episodes = 0 if replay is None else replay.episodes
    oracle_seconds = (
        sum(
            value.result.latency_ms
            for value in measured.oracle_executions
            if value.result.latency_ms is not None
        )
        / 1_000.0
    )
    actor_count = runtime.planner.batching.actor_count
    cpu_utilization = min(
        1.0,
        measured.native_seconds / max(elapsed * actor_count, 1.0e-9),
    )
    actor_idle = 1.0 - min(
        1.0,
        measured.actor_busy_seconds / max(elapsed * actor_count, 1.0e-9),
    )
    identity = resolved.runtime_identity
    workload_fingerprint = (
        measured.archive_sha256 if learner is None else learner.workload_fingerprint
    )
    if workload_fingerprint is None:
        raise ValueError("profile run has no exact workload fingerprint")
    return PlannerProfileRunRecord(
        campaign_id=config.campaign_id,
        point_id=point.point_id,
        budget_id=point.budget_id,
        environment=point.environment,
        planner_enabled=point.planner_enabled,
        checkpoint_sha256=model_identity.checkpoint_sha256,
        decision_corpus_sha256=config.expected_decision_corpus_sha256,
        workload_fingerprint=workload_fingerprint,
        machine_fingerprint=profile_machine_fingerprint(),
        model_fingerprint=identity.model_fingerprint,
        runtime_fingerprint=resolved.runtime_fingerprint,
        planner_fingerprint=identity.planner_fingerprint,
        controller_fingerprint=identity.controller_fingerprint,
        constructor_fingerprint=identity.constructor_fingerprint,
        scorer_fingerprint=identity.scorer_fingerprint,
        tensor_schema_fingerprint=resolved.tensor_schema_fingerprint,
        native_abi_fingerprint=config.expected_native_abi_fingerprint,
        native_schema_fingerprint=config.expected_native_schema_fingerprint,
        native_library_fingerprint=config.expected_native_library_sha256,
        packaged_archive_fingerprint=measured.archive_sha256,
        packaged_required_files_fingerprint=(measured.required_files_fingerprint),
        policy_version=identity.policy_version,
        proposal_version=identity.proposal_version,
        decisions=measured.decisions,
        valid_planner_evidence_decisions=measured.valid_planner_decisions,
        learner_warmup_updates=0 if learner is None else learner.warmup_updates,
        learner_updates=0 if learner is None else learner.updates,
        learner_optimizer_steps=0 if learner is None else learner.optimizer_steps,
        learner_kernel_rows=learner_kernel_rows,
        policy_runtime_agent_warmup_calls=(
            0 if runtime.packaged is None else runtime.packaged.warmup_decisions
        ),
        policy_runtime_agent_calls=replay_decisions,
        packaged_isolated_processes=0 if replay is None else replay.processes,
        act_time_replay_episodes=replay_episodes,
        act_time_replay_decisions=replay_decisions,
        act_time_replay_exhausted_episodes=(
            0 if replay is None else replay.exhausted_episodes
        ),
        act_time_replay_elapsed_seconds=(
            0.0 if replay is None else replay.elapsed_seconds
        ),
        act_time_replay_min_remaining_seconds=(
            0.0 if replay is None else replay.min_remaining_seconds
        ),
        oracle_decisions=len(measured.oracle_executions),
        oracle_elapsed_seconds=oracle_seconds,
        elapsed_seconds=elapsed,
        inference_decisions_per_second=measured.decisions / elapsed,
        learner_kernel_rows_per_second=(
            0.0 if learner_seconds <= 0.0 else learner_kernel_rows / learner_seconds
        ),
        valid_evidence_decisions_per_second=(
            measured.valid_planner_decisions / elapsed
        ),
        checkpoint_cadence_seconds=(
            0.0
            if learner is None or learner.updates <= 0
            else learner_seconds / learner.updates
        ),
        cpu_lane_utilization=cpu_utilization,
        gil_utilization=None,
        gil_measurement_unavailable_reason=_GIL_UNAVAILABLE,
        actor_idle_fraction=actor_idle,
        peak_vram_bytes=max(
            measured.resource_peak.vram_bytes,
            0 if learner is None else learner.peak_vram_bytes,
        ),
        peak_host_bytes=measured.resource_peak.host_bytes,
        fallback_decisions=measured.fallback_decisions,
        failed_decisions=measured.failed_decisions,
        deadline_exceeded_decisions=measured.deadline_decisions,
    )


__all__ = ["ProfileRunMeasurements", "build_profile_run_record"]
