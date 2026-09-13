"""Pre-action planner sampling on one immutable retained support."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ptcg_rl.agent.search.planner_fallback import PlannerFallbackReason
from ptcg_rl.agent.search.planner_scoring import PlannerScoringConfig
from ptcg_rl.agent.search.planner_target import (
    build_detached_planner_target,
    planner_conditioned_distribution,
)
from ptcg_rl.engine.search_evidence import SEARCH_EVIDENCE_FEATURE_SIZE
from ptcg_rl.rl.planner_evidence import (
    PLANNER_SOURCE_COUNT,
    PlannerBehaviorBranch,
    PlannerBehaviorEvidence,
    PlannerCandidateEvidence,
    ScenarioSupportMode,
)


@dataclass(frozen=True, slots=True)
class PlannerCollectionMetadata:
    """Non-candidate fields fixed before an action is sampled."""

    root_information_fingerprint: str
    scenario_support_fingerprint: str
    model_fingerprint: str
    constructor_fingerprint: str
    scorer_fingerprint: str
    controller_fingerprint: str
    planner_fingerprint: str
    policy_version: int
    proposal_version: int
    constructor_version: int
    planner_version: int
    scenario_support_mode: ScenarioSupportMode
    legal_action_count: int
    scenario_count: int
    support_exhaustive: bool
    scenario_grid_complete: bool
    leaf_bootstrapped: bool
    configured_source_quotas: tuple[int, ...]
    used_source_quotas: tuple[int, ...]
    engine_transition_limit: int
    engine_transitions_used: int
    prefix_node_limit: int
    prefix_nodes_used: int
    wall_clock_limit_ms: int
    wall_clock_used_ms: int
    planner_temperature: float

    def __post_init__(self) -> None:
        """Catch the quota-width error before constructing evidence."""
        if (
            len(self.configured_source_quotas) != PLANNER_SOURCE_COUNT
            or len(self.used_source_quotas) != PLANNER_SOURCE_COUNT
        ):
            raise ValueError("planner source quota vectors have the wrong width")


@dataclass(frozen=True, slots=True)
class PlannerBehaviorSelection:
    """Complete action sampled from the distribution persisted for PPO."""

    action: tuple[int, ...]
    old_logprob: float
    evidence: PlannerBehaviorEvidence


def sample_planner_conditioned_action(
    *,
    actions: tuple[tuple[int, ...], ...],
    aggregate_features: tuple[tuple[float, ...], ...],
    source_bits: tuple[int, ...],
    rules_exact: tuple[bool, ...],
    base_logprobs_at_collection: Tensor,
    proposal_logprobs_at_collection: Tensor,
    reranker_residual_at_collection: Tensor,
    robust_scores: Tensor,
    scoring_config: PlannerScoringConfig,
    metadata: PlannerCollectionMetadata,
    generator: torch.Generator | None = None,
) -> PlannerBehaviorSelection:
    """Sample once and freeze every probability needed for replay.

    All model tensors must come from the same model-version lease.  The caller
    enforces that lease; this function binds its identity into the resulting
    evidence and refuses a scorer/config mismatch.
    """
    candidate_count = len(actions)
    if candidate_count <= 0:
        raise ValueError("planner behavior requires a retained candidate")
    if not (
        len(aggregate_features)
        == len(source_bits)
        == len(rules_exact)
        == candidate_count
    ):
        raise ValueError("planner candidate columns must align")
    for features in aggregate_features:
        if len(features) != SEARCH_EVIDENCE_FEATURE_SIZE:
            raise ValueError("planner candidate aggregate feature width is invalid")
    tensors = (
        base_logprobs_at_collection,
        proposal_logprobs_at_collection,
        reranker_residual_at_collection,
        robust_scores,
    )
    if any(tensor.shape != (candidate_count,) for tensor in tensors):
        raise ValueError("planner candidate tensors must be one aligned vector")
    if metadata.policy_version < 0:
        raise ValueError("planner behavior requires a non-negative policy version")
    if metadata.scorer_fingerprint != scoring_config.scorer_fingerprint:
        raise ValueError("planner metadata and scorer config fingerprints differ")
    if not math.isclose(
        metadata.planner_temperature,
        scoring_config.planner_temperature,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise ValueError("planner metadata and scorer temperatures differ")
    target = build_detached_planner_target(
        base_logprobs_at_collection=base_logprobs_at_collection,
        robust_scores=robust_scores,
        config=scoring_config,
    )
    behavior = planner_conditioned_distribution(
        current_base_logprobs=base_logprobs_at_collection,
        immutable_score_prior=target.score_prior,
        reranker_residual=reranker_residual_at_collection,
        config=scoring_config,
    ).detach()
    cpu_base = base_logprobs_at_collection.detach().float().cpu()
    cpu_proposal = proposal_logprobs_at_collection.detach().float().cpu()
    cpu_scores = robust_scores.detach().float().cpu()
    cpu_prior = target.score_prior.float().cpu()
    cpu_target = target.distribution.float().cpu()
    cpu_behavior = behavior.float().cpu()
    selected = int(torch.multinomial(cpu_behavior, 1, generator=generator).item())
    candidates = tuple(
        PlannerCandidateEvidence(
            action=tuple(int(index) for index in action),
            aggregate_features=tuple(float(value) for value in features),
            base_logprob=float(cpu_base[index]),
            proposal_logprob=float(cpu_proposal[index]),
            score_prior=float(cpu_prior[index]),
            target_probability=float(cpu_target[index]),
            behavior_probability=float(cpu_behavior[index]),
            robust_score=float(cpu_scores[index]),
            source_bits=int(source_bits[index]),
            rules_exact=bool(rules_exact[index]),
        )
        for index, (action, features) in enumerate(
            zip(actions, aggregate_features, strict=True)
        )
    )
    evidence = _evidence_from_metadata(
        metadata=metadata,
        branch=PlannerBehaviorBranch.PLANNER_CONDITIONED,
        fallback_reason=PlannerFallbackReason.NONE,
        candidates=candidates,
        selected_candidate_index=selected,
    )
    old_logprob = math.log(candidates[selected].behavior_probability)
    return PlannerBehaviorSelection(
        action=candidates[selected].action,
        old_logprob=old_logprob,
        evidence=evidence,
    )


def base_fallback_evidence(
    *,
    metadata: PlannerCollectionMetadata,
    reason: PlannerFallbackReason,
) -> PlannerBehaviorEvidence:
    """Build the explicit branch record for base autoregressive sampling."""
    if reason is PlannerFallbackReason.NONE:
        raise ValueError("base fallback requires an explicit reason")
    return _evidence_from_metadata(
        metadata=metadata,
        branch=PlannerBehaviorBranch.BASE_FALLBACK,
        fallback_reason=reason,
        candidates=(),
        selected_candidate_index=-1,
    )


def _evidence_from_metadata(
    *,
    metadata: PlannerCollectionMetadata,
    branch: PlannerBehaviorBranch,
    fallback_reason: PlannerFallbackReason,
    candidates: tuple[PlannerCandidateEvidence, ...],
    selected_candidate_index: int,
) -> PlannerBehaviorEvidence:
    return PlannerBehaviorEvidence(
        branch=branch,
        fallback_reason=fallback_reason,
        candidates=candidates,
        selected_candidate_index=selected_candidate_index,
        root_information_fingerprint=metadata.root_information_fingerprint,
        scenario_support_fingerprint=metadata.scenario_support_fingerprint,
        model_fingerprint=metadata.model_fingerprint,
        constructor_fingerprint=metadata.constructor_fingerprint,
        scorer_fingerprint=metadata.scorer_fingerprint,
        controller_fingerprint=metadata.controller_fingerprint,
        planner_fingerprint=metadata.planner_fingerprint,
        policy_version=metadata.policy_version,
        proposal_version=metadata.proposal_version,
        constructor_version=metadata.constructor_version,
        planner_version=metadata.planner_version,
        scenario_support_mode=metadata.scenario_support_mode,
        legal_action_count=metadata.legal_action_count,
        scenario_count=metadata.scenario_count,
        support_exhaustive=(
            metadata.support_exhaustive
            if branch is PlannerBehaviorBranch.PLANNER_CONDITIONED
            else False
        ),
        scenario_grid_complete=(
            metadata.scenario_grid_complete
            if branch is PlannerBehaviorBranch.PLANNER_CONDITIONED
            else False
        ),
        leaf_bootstrapped=(
            metadata.leaf_bootstrapped
            if branch is PlannerBehaviorBranch.PLANNER_CONDITIONED
            else False
        ),
        support_censored=(
            not metadata.support_exhaustive
            if branch is PlannerBehaviorBranch.PLANNER_CONDITIONED
            else True
        ),
        configured_source_quotas=metadata.configured_source_quotas,
        used_source_quotas=metadata.used_source_quotas,
        engine_transition_limit=metadata.engine_transition_limit,
        engine_transitions_used=metadata.engine_transitions_used,
        prefix_node_limit=metadata.prefix_node_limit,
        prefix_nodes_used=metadata.prefix_nodes_used,
        wall_clock_limit_ms=metadata.wall_clock_limit_ms,
        wall_clock_used_ms=metadata.wall_clock_used_ms,
        planner_temperature=metadata.planner_temperature,
    )


__all__ = [
    "PlannerBehaviorSelection",
    "PlannerCollectionMetadata",
    "base_fallback_evidence",
    "sample_planner_conditioned_action",
]
