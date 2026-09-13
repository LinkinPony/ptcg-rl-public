"""Correctness-preserving merge helpers for hybrid stateless collection."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from ptcg_rl.rl.stateless_collection import (
    NativeArtifactInferenceReport,
    NativeInferenceRouteKind,
    StatelessAssignedGame,
    StatelessCollectionReport,
    StatelessCollectionResult,
)


def partition_hybrid_assignments(
    assignments: Sequence[StatelessAssignedGame],
) -> tuple[
    tuple[StatelessAssignedGame, ...],
    tuple[StatelessAssignedGame, ...],
]:
    """Route vectorizable lanes to native and exact scripts to Python."""
    native: list[StatelessAssignedGame] = []
    scripted: list[StatelessAssignedGame] = []
    for assignment in assignments:
        lane = assignment.curriculum.lane
        if lane == "scripted":
            scripted.append(assignment)
        elif lane in {"mirror", "pfsp"}:
            native.append(assignment)
        else:
            raise ValueError(f"hybrid collection received an unknown lane: {lane}")
    return tuple(native), tuple(scripted)


def merge_hybrid_collection_results(
    results: Sequence[StatelessCollectionResult],
    *,
    assignments: Sequence[StatelessAssignedGame],
    elapsed_seconds: float,
    native_phase_seconds: float = 0.0,
    scripted_phase_seconds: float = 0.0,
) -> StatelessCollectionResult:
    """Merge concurrent lane shards without changing assignment identity."""
    if not results:
        raise ValueError("hybrid collection requires at least one result")
    if elapsed_seconds <= 0.0:
        raise ValueError("hybrid collection elapsed time must be positive")
    expected = {
        assignment.curriculum.assignment_id: assignment for assignment in assignments
    }
    if len(expected) != len(assignments):
        raise ValueError("hybrid collection received duplicate assignments")

    observed_assignments: dict[str, StatelessAssignedGame] = {}
    observed_outcomes = {}
    reports: list[StatelessCollectionReport] = []
    for result in results:
        if result.fragments:
            raise ValueError("hybrid collection requires compact fragment parts")
        reports.append(result.report)
        for assignment in result.assignments:
            assignment_id = assignment.curriculum.assignment_id
            if assignment_id in observed_assignments:
                raise ValueError("hybrid collection duplicated an assignment")
            observed_assignments[assignment_id] = assignment
        for outcome in result.outcomes:
            assignment_id = outcome.curriculum_assignment_id
            if assignment_id in observed_outcomes:
                raise ValueError("hybrid collection duplicated an outcome")
            observed_outcomes[assignment_id] = outcome

    if observed_assignments != expected:
        raise ValueError("hybrid collection changed assignment coverage")
    if set(observed_outcomes) != set(expected):
        raise ValueError("hybrid collection changed outcome coverage")

    merged_report = _merge_reports(
        reports,
        elapsed_seconds=elapsed_seconds,
        native_phase_seconds=native_phase_seconds,
        scripted_phase_seconds=scripted_phase_seconds,
    )
    return StatelessCollectionResult(
        fragments=(),
        compact_parts=tuple(
            part for result in results for part in result.compact_parts
        ),
        compact_part_paths=tuple(
            path for result in results for path in result.compact_part_paths
        ),
        assignments=tuple(assignments),
        outcomes=tuple(
            observed_outcomes[assignment.curriculum.assignment_id]
            for assignment in assignments
        ),
        report=merged_report,
    )


def _merge_reports(
    reports: Sequence[StatelessCollectionReport],
    *,
    elapsed_seconds: float,
    native_phase_seconds: float,
    scripted_phase_seconds: float,
) -> StatelessCollectionReport:
    """Add shard counters while retaining wall-clock collection throughput."""
    lane_games: Counter[str] = Counter()
    seat_games: Counter[str] = Counter()
    member_games: Counter[str] = Counter()
    lane_score_total: defaultdict[str, float] = defaultdict(float)
    lane_score_games: Counter[str] = Counter()
    artifact_batches: Counter[tuple[NativeInferenceRouteKind, str]] = Counter()
    artifact_rows: Counter[tuple[NativeInferenceRouteKind, str]] = Counter()
    decision_budgets = {
        report.native_trainable_decision_budget
        for report in reports
        if report.native_trainable_decision_budget is not None
    }
    if len(decision_budgets) > 1:
        raise ValueError("hybrid collection mixed native decision budgets")
    trainable_decision_budget = next(iter(decision_budgets), None)
    for report in reports:
        lane_games.update(report.lane_games)
        seat_games.update(report.seat_games)
        member_games.update(report.pfsp_member_games)
        for lane, score in report.candidate_score_by_lane.items():
            games = report.lane_games.get(lane, 0)
            lane_score_total[lane] += score * games
            lane_score_games[lane] += games
        for item in report.native_artifact_inference:
            key = (item.route_kind, item.artifact_sha256)
            artifact_batches[key] += item.batches
            artifact_rows[key] += item.rows

    decisions = sum(report.candidate_decisions for report in reports)
    return StatelessCollectionReport(
        games_started=sum(report.games_started for report in reports),
        assignment_reservations=sum(
            report.assignment_reservations for report in reports
        ),
        unstarted_reservations_released=sum(
            report.unstarted_reservations_released for report in reports
        ),
        games_finished=sum(report.games_finished for report in reports),
        games_cancelled=sum(report.games_cancelled for report in reports),
        games_window_cutoff=sum(report.games_window_cutoff for report in reports),
        games_immediate_window_cutoff=sum(
            report.games_immediate_window_cutoff for report in reports
        ),
        native_engine_arenas=sum(report.native_engine_arenas for report in reports),
        engine_steps=sum(report.engine_steps for report in reports),
        candidate_decisions=decisions,
        mirror_opponent_decisions=sum(
            report.mirror_opponent_decisions for report in reports
        ),
        native_trainable_decisions=sum(
            report.native_trainable_decisions for report in reports
        ),
        native_trainable_decision_budget=trainable_decision_budget,
        native_trainable_decision_budget_reached=any(
            report.native_trainable_decision_budget_reached for report in reports
        ),
        native_trainable_decision_budget_overshoot=sum(
            report.native_trainable_decision_budget_overshoot for report in reports
        ),
        fragments=sum(report.fragments for report in reports),
        elapsed_seconds=elapsed_seconds,
        decisions_per_second=decisions / elapsed_seconds,
        native_phase_seconds=native_phase_seconds,
        scripted_phase_seconds=scripted_phase_seconds,
        input_seconds=sum(report.input_seconds for report in reports),
        current_policy_seconds=sum(report.current_policy_seconds for report in reports),
        past_self_policy_seconds=sum(
            report.past_self_policy_seconds for report in reports
        ),
        historical_policy_seconds=sum(
            report.historical_policy_seconds for report in reports
        ),
        scripted_policy_seconds=sum(
            report.scripted_policy_seconds for report in reports
        ),
        native_policy_route_overlap_seconds=sum(
            report.native_policy_route_overlap_seconds for report in reports
        ),
        native_policy_route_overlap_waves=sum(
            report.native_policy_route_overlap_waves for report in reports
        ),
        native_policy_cohort_batches=sum(
            report.native_policy_cohort_batches for report in reports
        ),
        native_policy_cohort_rows=sum(
            report.native_policy_cohort_rows for report in reports
        ),
        native_policy_cohort_max_rows=max(
            (report.native_policy_cohort_max_rows for report in reports),
            default=0,
        ),
        native_policy_cohort_wait_seconds=sum(
            report.native_policy_cohort_wait_seconds for report in reports
        ),
        native_policy_cohort_wait_events=sum(
            report.native_policy_cohort_wait_events for report in reports
        ),
        native_policy_cohort_wait_harvests=sum(
            report.native_policy_cohort_wait_harvests for report in reports
        ),
        native_gpu_feed_wait_seconds=sum(
            report.native_gpu_feed_wait_seconds for report in reports
        ),
        native_gpu_feed_wait_events=sum(
            report.native_gpu_feed_wait_events for report in reports
        ),
        native_policy_host_prepare_seconds=sum(
            report.native_policy_host_prepare_seconds for report in reports
        ),
        native_policy_host_prepare_wall_seconds=sum(
            report.native_policy_host_prepare_wall_seconds for report in reports
        ),
        native_policy_host_prepare_wait_seconds=sum(
            report.native_policy_host_prepare_wait_seconds for report in reports
        ),
        native_policy_host_prepare_overlap_seconds=sum(
            report.native_policy_host_prepare_overlap_seconds for report in reports
        ),
        native_policy_host_prepare_past_bypasses=sum(
            report.native_policy_host_prepare_past_bypasses for report in reports
        ),
        native_policy_completion_wait_seconds=sum(
            report.native_policy_completion_wait_seconds for report in reports
        ),
        native_scripted_prefetch_wait_seconds=sum(
            report.native_scripted_prefetch_wait_seconds for report in reports
        ),
        native_scripted_prefetch_queue_seconds=sum(
            report.native_scripted_prefetch_queue_seconds for report in reports
        ),
        native_scripted_prefetch_overlap_seconds=sum(
            report.native_scripted_prefetch_overlap_seconds for report in reports
        ),
        native_scripted_prefetch_batches=sum(
            report.native_scripted_prefetch_batches for report in reports
        ),
        native_scripted_prefetch_rows=sum(
            report.native_scripted_prefetch_rows for report in reports
        ),
        native_bank_engine_wait_seconds=sum(
            report.native_bank_engine_wait_seconds for report in reports
        ),
        native_bank_policy_overlap_seconds=sum(
            report.native_bank_policy_overlap_seconds for report in reports
        ),
        native_bank_policy_prefetches=sum(
            report.native_bank_policy_prefetches for report in reports
        ),
        native_bank_engine_barriers=sum(
            report.native_bank_engine_barriers for report in reports
        ),
        native_bank_policy_groups=sum(
            report.native_bank_policy_groups for report in reports
        ),
        native_bank_policy_group_members=sum(
            report.native_bank_policy_group_members for report in reports
        ),
        native_bank_policy_group_max_size=max(
            (report.native_bank_policy_group_max_size for report in reports),
            default=0,
        ),
        native_bank_policy_coalescing_misses=sum(
            report.native_bank_policy_coalescing_misses for report in reports
        ),
        native_bank_gpu_feed_gap_seconds=sum(
            report.native_bank_gpu_feed_gap_seconds for report in reports
        ),
        native_bank_gpu_feed_gap_events=sum(
            report.native_bank_gpu_feed_gap_events for report in reports
        ),
        native_startup_seconds=sum(report.native_startup_seconds for report in reports),
        native_budget_fence_wait_seconds=sum(
            report.native_budget_fence_wait_seconds for report in reports
        ),
        engine_fact_seconds=sum(report.engine_fact_seconds for report in reports),
        engine_fact_wait_seconds=sum(
            report.engine_fact_wait_seconds for report in reports
        ),
        engine_fact_overlap_seconds=sum(
            report.engine_fact_overlap_seconds for report in reports
        ),
        engine_fact_roots=sum(report.engine_fact_roots for report in reports),
        engine_fact_eligible_options=sum(
            report.engine_fact_eligible_options for report in reports
        ),
        engine_fact_native_batch_calls=sum(
            report.engine_fact_native_batch_calls for report in reports
        ),
        engine_fact_native_transitions=sum(
            report.engine_fact_native_transitions for report in reports
        ),
        engine_fact_unresolved_worlds=sum(
            report.engine_fact_unresolved_worlds for report in reports
        ),
        engine_fact_resolved_options=sum(
            report.engine_fact_resolved_options for report in reports
        ),
        engine_control_seconds=sum(report.engine_control_seconds for report in reports),
        current_policy_batches=sum(report.current_policy_batches for report in reports),
        current_policy_rows=sum(report.current_policy_rows for report in reports),
        past_self_policy_batches=sum(
            report.past_self_policy_batches for report in reports
        ),
        past_self_policy_rows=sum(report.past_self_policy_rows for report in reports),
        historical_policy_batches=sum(
            report.historical_policy_batches for report in reports
        ),
        historical_policy_rows=sum(report.historical_policy_rows for report in reports),
        frozen_pending_row_waves=sum(
            report.frozen_pending_row_waves for report in reports
        ),
        frozen_threshold_releases=sum(
            report.frozen_threshold_releases for report in reports
        ),
        frozen_deadline_releases=sum(
            report.frozen_deadline_releases for report in reports
        ),
        frozen_forced_releases=sum(report.frozen_forced_releases for report in reports),
        native_process_workers=sum(report.native_process_workers for report in reports),
        native_inference_requests=sum(
            report.native_inference_requests for report in reports
        ),
        native_inference_threshold_batches=sum(
            report.native_inference_threshold_batches for report in reports
        ),
        native_inference_deadline_batches=sum(
            report.native_inference_deadline_batches for report in reports
        ),
        native_inference_unblock_batches=sum(
            report.native_inference_unblock_batches for report in reports
        ),
        native_shared_batch_rows=sum(
            report.native_shared_batch_rows for report in reports
        ),
        integrated_scripted_inference_requests=sum(
            report.integrated_scripted_inference_requests for report in reports
        ),
        integrated_scripted_inference_batches=sum(
            report.integrated_scripted_inference_batches for report in reports
        ),
        integrated_scripted_inference_rows=sum(
            report.integrated_scripted_inference_rows for report in reports
        ),
        integrated_scripted_mixed_batches=sum(
            report.integrated_scripted_mixed_batches for report in reports
        ),
        integrated_scripted_mixed_rows=sum(
            report.integrated_scripted_mixed_rows for report in reports
        ),
        native_artifact_inference=tuple(
            NativeArtifactInferenceReport(
                route_kind=route_kind,
                artifact_sha256=artifact_sha256,
                batches=artifact_batches[(route_kind, artifact_sha256)],
                rows=artifact_rows[(route_kind, artifact_sha256)],
            )
            for route_kind, artifact_sha256 in sorted(artifact_batches)
        ),
        lane_games=dict(sorted(lane_games.items())),
        seat_games=dict(sorted(seat_games.items())),
        pfsp_member_games=dict(sorted(member_games.items())),
        candidate_score_by_lane={
            lane: lane_score_total[lane] / games
            for lane, games in sorted(lane_score_games.items())
            if games
        },
    )


__all__ = [
    "merge_hybrid_collection_results",
    "partition_hybrid_assignments",
]
