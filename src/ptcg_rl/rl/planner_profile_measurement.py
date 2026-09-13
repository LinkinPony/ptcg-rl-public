"""Decision-level measurements for the integrated planner profile."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

from ptcg_rl.agent.search.planner_fallback import PlannerFallbackReason
from ptcg_rl.agent.search.prompt_actions import describe_prompt_action_space
from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.evaluation.planner_profile_config import (
    IntegratedPlannerProfileConfig,
    PlannerProfilePointConfig,
    PlannerProfileRuntimeConfig,
)
from ptcg_rl.evaluation.planner_profile_corpus_reader import (
    PlannerProfileCorpusRecord,
)
from ptcg_rl.evaluation.planner_profile_records import PlannerProfileDecisionRecord
from ptcg_rl.rl.planner_behavior_policy_contract import PlannerPolicyDecision
from ptcg_rl.rl.planner_evidence import PlannerBehaviorBranch
from ptcg_rl.rl.planner_profile_oracle import (
    MaterializedProfileRequest,
    ProfileOracleCacheValue,
)
from ptcg_rl.rl.planner_runtime_identity import ResolvedPlannerRuntimeIdentity
from ptcg_rl.runtime.planner_telemetry import PlannerStage

_ACTION_DOMAIN = b"ptcg-rl/planner-profile-action/v1\x00"
_IPC_UNAVAILABLE = "exact_queue_transport_bytes_not_exposed"


@dataclass(frozen=True, slots=True)
class ProfileDecisionTiming:
    """Measured deployment-path wall times for one actor-local row."""

    actor_policy_wait_ms: float
    total_latency_ms: float


class _OracleQuality(TypedDict):
    base_action_regret: float | None
    served_action_regret: float | None
    candidate_best_regret: float | None
    served_epsilon_optimal: bool | None
    candidate_epsilon_recall: bool | None
    value_calibration_error: float | None


def build_profile_decision_record(
    *,
    config: IntegratedPlannerProfileConfig,
    point: PlannerProfilePointConfig,
    runtime: PlannerProfileRuntimeConfig,
    resolved: ResolvedPlannerRuntimeIdentity,
    source: PlannerProfileCorpusRecord,
    decision_index: int,
    corpus_repetition: int,
    batch_row_position: int,
    base_action: tuple[int, ...],
    selected_action: tuple[int, ...],
    planner_decision: PlannerPolicyDecision | None,
    materialized: MaterializedProfileRequest,
    oracle: ProfileOracleCacheValue,
    timing: ProfileDecisionTiming,
    native_lane_occupancy: int,
) -> PlannerProfileDecisionRecord:
    """Bind shared-service evidence, exact oracle quality, and stage work."""
    evidence = None if planner_decision is None else planner_decision.planner_behavior
    planner_used = bool(
        evidence is not None
        and evidence.branch is PlannerBehaviorBranch.PLANNER_CONDITIONED
    )
    fallback = _fallback_reason(point=point, planner_decision=planner_decision)
    events = () if planner_decision is None else planner_decision.telemetry_events
    stats = None if planner_decision is None else planner_decision.runtime_stats
    legal_count = materialized.request.budget_request.legal_action_count
    belief_world_count = materialized.belief_world_count
    candidate_count = 0 if evidence is None else len(evidence.candidates)
    scenario_count = 0 if evidence is None else evidence.scenario_count
    chance_outcome_count = (
        0
        if scenario_count <= 0 or belief_world_count <= 0
        else max(1, math.ceil(scenario_count / belief_world_count))
    )
    gpu_events = tuple(
        event for event in events if event.stage is PlannerStage.GPU_LEAF_VALUE
    )
    engine_events = tuple(
        event for event in events if event.stage is PlannerStage.NATIVE_ENGINE
    )
    queue_wait_ms = sum(
        event.seconds * 1_000.0
        for event in events
        if event.stage is PlannerStage.QUEUE_WAIT
    )
    oracle_fields = _oracle_quality(
        oracle,
        base_action=base_action,
        selected_action=selected_action,
        candidate_actions=(
            ()
            if not planner_used or evidence is None
            else tuple(candidate.action for candidate in evidence.candidates)
        ),
        epsilon=config.oracle.epsilon_regret,
        outcome=source.final_root_outcome,
    )
    rules_exact = bool(
        planner_used
        and evidence is not None
        and all(candidate.rules_exact for candidate in evidence.candidates)
    )
    return PlannerProfileDecisionRecord(
        campaign_id=config.campaign_id,
        point_id=point.point_id,
        budget_id=point.budget_id,
        environment=point.environment,
        corpus_row_id=source.row_id,
        corpus_repetition=corpus_repetition,
        decision_shapes=source.shapes,
        decision_index=decision_index,
        batch_row_position=batch_row_position,
        model_fingerprint=resolved.runtime_identity.model_fingerprint,
        runtime_fingerprint=resolved.runtime_fingerprint,
        planner_fingerprint=resolved.runtime_identity.planner_fingerprint,
        scenario_support_fingerprint=materialized.support_fingerprint,
        policy_version=resolved.runtime_identity.policy_version,
        proposal_version=resolved.runtime_identity.proposal_version,
        planner_eligible=_planner_eligible(
            point=point,
            planner_decision=planner_decision,
            legal_action_count=legal_count,
        ),
        equivalence_probe_used=_equivalence_probe_used(
            source=source,
            runtime=runtime,
            planner_decision=planner_decision,
            planner_enabled=point.planner_enabled,
        ),
        planner_used=planner_used,
        policy_runtime_agent_used=False,
        fallback_reason=fallback,
        base_action_fingerprint=profile_action_fingerprint(base_action),
        selected_action_fingerprint=profile_action_fingerprint(selected_action),
        legal_action_count=legal_count,
        candidate_count=candidate_count,
        scenario_count=scenario_count,
        belief_world_count=belief_world_count,
        chance_outcome_count=chance_outcome_count,
        engine_transitions=sum(event.rows for event in engine_events),
        prefix_nodes=(0 if evidence is None else evidence.prefix_nodes_used),
        prefix_reuse_count=(0 if stats is None else stats.prefix_reuse_count),
        unique_leaf_count=(0 if stats is None else stats.unique_leaf_count),
        consequence_cell_count=(0 if stats is None else stats.consequence_cell_count),
        native_chunk_size=max((event.rows for event in engine_events), default=0),
        gpu_rows=sum(event.rows for event in gpu_events),
        gpu_batch_capacity=sum(event.batch_capacity for event in gpu_events),
        native_lane_occupancy=(native_lane_occupancy if planner_used else 0),
        ipc_bytes=None,
        ipc_measurement_unavailable_reason=_IPC_UNAVAILABLE,
        model_lease_lifetime_ms=timing.total_latency_ms,
        actor_policy_wait_ms=timing.actor_policy_wait_ms,
        planner_queue_wait_ms=queue_wait_ms,
        total_latency_ms=timing.total_latency_ms,
        base_planner_agree=(selected_action == base_action if planner_used else None),
        support_exhaustive=(False if evidence is None else evidence.support_exhaustive),
        scenario_grid_complete=(
            False if evidence is None else evidence.scenario_grid_complete
        ),
        rules_exact=rules_exact,
        leaf_bootstrapped=(False if evidence is None else evidence.leaf_bootstrapped),
        failed=False,
        deadline_exceeded=(
            evidence is not None
            and evidence.fallback_reason is PlannerFallbackReason.DEADLINE
        ),
        oracle_compatible=oracle.result.compatible,
        oracle_exclusion_reason=oracle.result.exclusion_reason,
        oracle_provenance_fingerprint=(
            oracle.provenance_fingerprint if oracle.result.compatible else None
        ),
        oracle_executed=bool(oracle.executed and oracle.result.compatible),
        oracle_candidate_count=oracle.result.candidate_count,
        oracle_scenario_count=oracle.result.scenario_count,
        oracle_engine_transitions=oracle.result.engine_transitions,
        oracle_scenario_grid_complete=oracle.result.scenario_grid_complete,
        oracle_rules_exact=oracle.result.rules_exact,
        oracle_latency_ms=(
            (oracle.result.latency_ms if oracle.executed else 0.0)
            if oracle.result.compatible
            else None
        ),
        **oracle_fields,
    )


def profile_action_fingerprint(action: tuple[int, ...]) -> str:
    """Return a privacy-safe, order-sensitive complete-action identity."""
    digest = hashlib.sha256()
    digest.update(_ACTION_DOMAIN)
    digest.update(struct.pack(">I", len(action)))
    for value in action:
        digest.update(struct.pack(">i", int(value)))
    return digest.hexdigest()


def _oracle_quality(
    oracle: ProfileOracleCacheValue,
    *,
    base_action: tuple[int, ...],
    selected_action: tuple[int, ...],
    candidate_actions: tuple[tuple[int, ...], ...],
    epsilon: float,
    outcome: int,
) -> _OracleQuality:
    result = oracle.result
    if not result.compatible:
        return {
            "base_action_regret": None,
            "served_action_regret": None,
            "candidate_best_regret": None,
            "served_epsilon_optimal": None,
            "candidate_epsilon_recall": None,
            "value_calibration_error": None,
        }
    if base_action not in result.scores or selected_action not in result.scores:
        raise RuntimeError("profile oracle omitted a served legal action")
    maximum = max(result.scores.values())
    base_regret = max(0.0, maximum - result.scores[base_action])
    served_regret = max(0.0, maximum - result.scores[selected_action])
    candidate_regret: float | None = None
    candidate_recall: bool | None = None
    if candidate_actions:
        missing = tuple(
            action for action in candidate_actions if action not in result.scores
        )
        if missing:
            raise RuntimeError("profile oracle omitted a retained candidate action")
        candidate_regret = max(
            0.0,
            maximum - max(result.scores[action] for action in candidate_actions),
        )
        candidate_recall = candidate_regret <= epsilon
    return {
        "base_action_regret": base_regret,
        "served_action_regret": served_regret,
        "candidate_best_regret": candidate_regret,
        "served_epsilon_optimal": served_regret <= epsilon,
        "candidate_epsilon_recall": candidate_recall,
        "value_calibration_error": abs(
            float(result.scores[selected_action]) - float(outcome)
        ),
    }


def _fallback_reason(
    *,
    point: PlannerProfilePointConfig,
    planner_decision: PlannerPolicyDecision | None,
) -> str | None:
    if not point.planner_enabled:
        return "control"
    if planner_decision is None:
        raise RuntimeError("planner-enabled profile row has no branch evidence")
    if not planner_decision.used_base_trace:
        return None
    return str(planner_decision.planner_behavior.fallback_reason.name).lower()


def _planner_eligible(
    *,
    point: PlannerProfilePointConfig,
    planner_decision: PlannerPolicyDecision | None,
    legal_action_count: int,
) -> bool:
    if not point.planner_enabled or legal_action_count <= 1:
        return False
    if planner_decision is None:
        return False
    return (
        planner_decision.planner_behavior.fallback_reason
        is not PlannerFallbackReason.INELIGIBLE
    )


def _equivalence_probe_used(
    *,
    source: PlannerProfileCorpusRecord,
    runtime: PlannerProfileRuntimeConfig,
    planner_decision: PlannerPolicyDecision | None,
    planner_enabled: bool,
) -> bool:
    if not planner_enabled or planner_decision is None:
        return False
    select = source.observation.get("select")
    space = describe_prompt_action_space(select)
    context_raw = select.get("context", -1) if isinstance(select, Mapping) else -1
    context = -1 if context_raw is None else int(context_raw)
    return bool(
        context != int(SelectContext.MAIN)
        and not space.ordered
        and space.min_count == 1
        and space.max_count == 1
        and 1
        < space.legal_action_count
        <= runtime.planner.planner_behavior.eligibility.single_select_probe_cap
    )


__all__ = [
    "ProfileDecisionTiming",
    "build_profile_decision_record",
    "profile_action_fingerprint",
]
