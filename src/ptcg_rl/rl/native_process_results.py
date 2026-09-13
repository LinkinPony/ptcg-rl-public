"""Durable worker-part validation and telemetry merge for native processes."""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from ptcg_rl.rl.native_process_inference import NativeProcessInferenceBroker
from ptcg_rl.rl.stateless_collection import (
    NativeArtifactInferenceReport,
    StatelessAssignedGame,
    StatelessCollectionReport,
    StatelessGameOutcome,
)
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity
from ptcg_rl.rl.stateless_fragment_io import (
    CompactFragmentPart,
    load_compact_fragment_part,
)
from ptcg_rl.rl.stateless_parallel import StatelessParallelWorkerResult


def persist_native_worker_parts(
    parts: Sequence[CompactFragmentPart],
    *,
    shard_dir: Path,
) -> tuple[Path, ...]:
    """Atomically persist one worker's compact transport parts."""
    parts_dir = shard_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    for index, part in enumerate(parts):
        destination = parts_dir / f"part-{index:08d}.npz"
        temporary = parts_dir / f".{destination.name}.{uuid.uuid4().hex}.partial"
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **part.arrays)  # type: ignore[arg-type]
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        paths.append(destination)
    (shard_dir / "manifest.json").write_text(
        json.dumps(
            {
                "parts": len(parts),
                "fragments": sum(part.fragment_count for part in parts),
                "decisions": sum(part.decision_count for part in parts),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return tuple(paths)


def load_native_worker_parts(
    results: Sequence[StatelessParallelWorkerResult],
    *,
    expected_identity: StatelessFragmentIdentity,
) -> tuple[CompactFragmentPart, ...]:
    """Load parts only after validating full static identity and uniqueness."""
    parts: list[CompactFragmentPart] = []
    seen_paths: set[Path] = set()
    seen_fragments: set[str] = set()
    expected_fields: tuple[tuple[str, object], ...] = (
        ("horizons", expected_identity.horizon),
        ("behavior_policy_versions", expected_identity.behavior_policy_version),
        (
            "behavior_policy_fingerprints",
            expected_identity.behavior_policy_fingerprint,
        ),
        ("model_config_fingerprints", expected_identity.model_config_fingerprint),
        ("action_schema_fingerprints", expected_identity.action_schema_fingerprint),
        (
            "public_context_fingerprints",
            expected_identity.public_context_fingerprint,
        ),
        ("card_catalog_fingerprints", expected_identity.card_catalog_fingerprint),
        (
            "public_deck_catalog_fingerprints",
            expected_identity.public_deck_catalog_fingerprint,
        ),
        (
            "exact_registry_fingerprints",
            expected_identity.exact_registry_fingerprint,
        ),
        (
            "belief_target_semantics_fingerprints",
            expected_identity.belief_target_semantics_fingerprint,
        ),
        (
            "input_contract_fingerprints",
            expected_identity.input_contract_fingerprint,
        ),
        (
            "resolved_config_fingerprints",
            expected_identity.resolved_config_fingerprint,
        ),
    )
    for result in results:
        report = result.report
        if report is None:
            raise RuntimeError("native rollout worker omitted its report")
        if result.part_paths and result.compact_parts:
            raise RuntimeError("native rollout worker mixed compact transports")
        worker_fragments = 0
        worker_decisions = 0
        worker_parts: list[CompactFragmentPart] = []
        for path in result.part_paths:
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise RuntimeError(
                    f"native rollout worker part is unavailable: {path}"
                ) from error
            if resolved in seen_paths:
                raise RuntimeError("native rollout workers duplicated a part")
            seen_paths.add(resolved)
            worker_parts.append(load_compact_fragment_part(resolved))
        for part in result.compact_parts:
            if part.path is not None:
                raise RuntimeError("native memory transport retained a local path")
            worker_parts.append(part)
        for part in worker_parts:
            fragment_ids = tuple(str(value) for value in part.arrays["fragment_ids"])
            if seen_fragments.intersection(fragment_ids):
                raise RuntimeError("native rollout workers duplicated a fragment")
            seen_fragments.update(fragment_ids)
            for field, expected in expected_fields:
                if not np.all(np.asarray(part.arrays[field]) == expected):
                    raise RuntimeError(
                        f"native rollout worker {field} identity changed"
                    )
            worker_fragments += part.fragment_count
            worker_decisions += part.decision_count
            parts.append(part)
        if worker_fragments != report.fragments:
            raise RuntimeError("native worker fragment count differs")
        if worker_decisions != report.candidate_decisions:
            raise RuntimeError("native worker decision count differs")
    return tuple(parts)


def order_native_worker_outcomes(
    assignments: Sequence[StatelessAssignedGame],
    outcomes: Sequence[StatelessGameOutcome],
) -> tuple[StatelessGameOutcome, ...]:
    """Restore central assignment order after worker-local execution."""
    order = {
        assignment.curriculum.assignment_id: index
        for index, assignment in enumerate(assignments)
    }
    outcome_ids = tuple(outcome.curriculum_assignment_id for outcome in outcomes)
    if len(outcome_ids) != len(set(outcome_ids)) or not set(outcome_ids) <= set(order):
        raise RuntimeError("native rollout workers lost or duplicated outcomes")
    return tuple(
        sorted(
            outcomes,
            key=lambda outcome: order[outcome.curriculum_assignment_id],
        )
    )


def merge_native_process_reports(
    results: Sequence[StatelessParallelWorkerResult],
    *,
    elapsed_seconds: float,
    decision_budget: int | None,
    broker: NativeProcessInferenceBroker,
) -> StatelessCollectionReport:
    """Merge additive worker counters with central CUDA service telemetry."""
    reports = tuple(
        cast(StatelessCollectionReport, result.report) for result in results
    )
    lane_games: Counter[str] = Counter()
    seat_games: Counter[str] = Counter()
    member_games: Counter[str] = Counter()
    lane_score_total: defaultdict[str, float] = defaultdict(float)
    lane_score_games: Counter[str] = Counter()
    for report in reports:
        lane_games.update(report.lane_games)
        seat_games.update(report.seat_games)
        member_games.update(report.pfsp_member_games)
        for lane, score in report.candidate_score_by_lane.items():
            games = report.lane_games.get(lane, 0)
            lane_score_total[lane] += score * games
            lane_score_games[lane] += games
    decisions = sum(report.candidate_decisions for report in reports)
    route_seconds = broker.route_seconds
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
        native_engine_arenas=len(reports),
        engine_steps=sum(report.engine_steps for report in reports),
        candidate_decisions=decisions,
        mirror_opponent_decisions=sum(
            report.mirror_opponent_decisions for report in reports
        ),
        native_trainable_decisions=decisions,
        native_trainable_decision_budget=decision_budget,
        native_trainable_decision_budget_reached=(
            decision_budget is not None and decisions >= decision_budget
        ),
        native_trainable_decision_budget_overshoot=(
            max(decisions - decision_budget, 0) if decision_budget is not None else 0
        ),
        fragments=sum(report.fragments for report in reports),
        elapsed_seconds=elapsed_seconds,
        decisions_per_second=decisions / elapsed_seconds,
        native_phase_seconds=elapsed_seconds,
        input_seconds=sum(report.input_seconds for report in reports),
        current_policy_seconds=sum(
            seconds
            for route, seconds in route_seconds.items()
            if route.kind == "current"
        ),
        past_self_policy_seconds=sum(
            seconds
            for route, seconds in route_seconds.items()
            if route.kind == "past_self"
        ),
        historical_policy_seconds=sum(
            seconds
            for route, seconds in route_seconds.items()
            if route.kind == "historical"
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
        engine_control_seconds=max(
            elapsed_seconds - sum(route_seconds.values()),
            0.0,
        ),
        current_policy_batches=sum(
            count
            for route, count in broker.route_batches.items()
            if route.kind == "current"
        ),
        current_policy_rows=sum(
            count
            for route, count in broker.route_rows.items()
            if route.kind == "current"
        ),
        past_self_policy_batches=sum(
            count
            for route, count in broker.route_batches.items()
            if route.kind == "past_self"
        ),
        past_self_policy_rows=sum(
            count
            for route, count in broker.route_rows.items()
            if route.kind == "past_self"
        ),
        historical_policy_batches=sum(
            count
            for route, count in broker.route_batches.items()
            if route.kind == "historical"
        ),
        historical_policy_rows=sum(
            count
            for route, count in broker.route_rows.items()
            if route.kind == "historical"
        ),
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
        native_process_workers=len(reports),
        native_inference_requests=broker.requests,
        native_inference_threshold_batches=broker.threshold_batches,
        native_inference_deadline_batches=broker.deadline_batches,
        native_inference_unblock_batches=broker.unblock_batches,
        native_shared_batch_rows=broker.shared_rows,
        integrated_scripted_inference_requests=(broker.integrated_scripted_requests),
        integrated_scripted_inference_batches=(broker.integrated_scripted_batches),
        integrated_scripted_inference_rows=broker.integrated_scripted_rows,
        integrated_scripted_mixed_batches=(broker.integrated_scripted_mixed_batches),
        integrated_scripted_mixed_rows=broker.integrated_scripted_mixed_rows,
        native_artifact_inference=tuple(
            NativeArtifactInferenceReport(
                route_kind=cast(Any, route.kind),
                artifact_sha256=route.artifact_sha256,
                batches=broker.route_batches[route],
                rows=broker.route_rows[route],
            )
            for route in sorted(broker.route_batches)
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
    "load_native_worker_parts",
    "merge_native_process_reports",
    "order_native_worker_outcomes",
    "persist_native_worker_parts",
]
