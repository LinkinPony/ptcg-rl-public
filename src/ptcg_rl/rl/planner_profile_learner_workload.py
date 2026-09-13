"""Assembly and validation for the fixed synthetic learner-kernel workload."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from ptcg_rl.actions.encoding import EncodedOptionArrayFeatures
from ptcg_rl.agent.search.root_information import RootInformationLeaf
from ptcg_rl.agent.search.root_information_context import (
    RootInformationProducerContext,
    encode_root_information_producer_context,
)
from ptcg_rl.agent.search.root_information_producer import (
    root_information_belief_summary,
)
from ptcg_rl.context import OpponentBeliefFeatureProducer
from ptcg_rl.evaluation.planner_profile_config import IntegratedPlannerProfileConfig
from ptcg_rl.evaluation.planner_profile_corpus_reader import PlannerProfileCorpusReader
from ptcg_rl.evaluation.planner_profile_workloads import (
    PlannerLearnerWorkloadConfig,
    PlannerProfileRuntimeConfig,
)
from ptcg_rl.model.network import AgentPolicyValueNet
from ptcg_rl.model.root_input_fingerprint import (
    canonical_planner_root_input_fingerprint,
)
from ptcg_rl.model.state_encoder import StateTokenArrayFeatures
from ptcg_rl.rl.experience import GameTrajectory
from ptcg_rl.rl.learner import (
    LearnerBatchConfig,
    LearnerBatchResult,
    build_ppo_minibatches,
)
from ptcg_rl.rl.planner_evidence import (
    PlannerBehaviorBranch,
    PlannerBehaviorEvidence,
)
from ptcg_rl.rl.planner_profile_learner_data import (
    build_profile_learner_transitions,
)
from ptcg_rl.rl.planner_profile_learner_examples import (
    build_profile_kernel_seed_examples,
)
from ptcg_rl.rl.planner_runtime_identity import ResolvedPlannerRuntimeIdentity

_FIXED_KERNEL_ROWS = 1_024
_FIXED_MICROBATCH_SIZE = 512
_ENTROPY_DENOMINATOR = 2_048


@dataclass(frozen=True, slots=True)
class PreparedProfileLearnerKernel:
    """Canonical tensor workload and its stable, timing-free identity."""

    batch_result: LearnerBatchResult
    workload_fingerprint: str
    planner_rows: int
    root_value_rows: int
    unique_root_value_model_inputs: int


def prepare_profile_learner_kernel(
    *,
    config: IntegratedPlannerProfileConfig,
    runtime: PlannerProfileRuntimeConfig,
    workload: PlannerLearnerWorkloadConfig,
    resolved: ResolvedPlannerRuntimeIdentity,
    model: AgentPolicyValueNet,
    batch_config: LearnerBatchConfig,
) -> PreparedProfileLearnerKernel:
    """Build a fixed full-objective kernel, not an on-policy training stream.

    Every row is a one-decision synthetic trajectory. Consequently, the tensor
    mix exercises all deployed PPO objectives but its state distribution and
    one-step GAE targets are not production-equivalent training evidence.
    """
    if workload.kernel_rows_per_update != _FIXED_KERNEL_ROWS:
        raise ValueError("profile learner kernel requires exactly 1024 rows")
    reader = PlannerProfileCorpusReader(
        config.decision_corpus_path,
        expected_sha256=config.expected_decision_corpus_sha256,
    )
    records = tuple(reader)
    producer = OpponentBeliefFeatureProducer.from_config(runtime.belief.producer)
    transitions = build_profile_learner_transitions(
        records,
        replay_root=config.corpus_build.replay_root,
        replay_archive_root=config.corpus_build.replay_archive_root,
        belief_producer=producer,
        belief_summary_width=runtime.planner.tensorizer.belief_summary_dim,
    )
    seeds = build_profile_kernel_seed_examples(
        records,
        transitions=transitions,
        producer=producer,
        runtime=runtime,
        resolved=resolved,
        model=model,
        native_library_path=config.native_library_path,
    )
    trajectories = _repeat_unique_kernel_rows(
        seeds,
        row_count=workload.kernel_rows_per_update,
    )
    result = build_ppo_minibatches(
        trajectories,
        current_policy_version=config.policy_version,
        config=batch_config,
        device=None,
    )
    planner_rows, root_rows, unique_inputs = _validate_full_objective_kernel(
        result,
        trajectories=trajectories,
        expected_rows=workload.kernel_rows_per_update,
    )
    fingerprint = _kernel_workload_fingerprint(
        config=config,
        runtime=runtime,
        workload=workload,
        resolved=resolved,
        trajectories=trajectories,
    )
    return PreparedProfileLearnerKernel(
        batch_result=result,
        workload_fingerprint=fingerprint,
        planner_rows=planner_rows,
        root_value_rows=root_rows,
        unique_root_value_model_inputs=unique_inputs,
    )


def _repeat_unique_kernel_rows(
    examples: Sequence[GameTrajectory],
    *,
    row_count: int,
) -> tuple[GameTrajectory, ...]:
    eligible = tuple(
        example for example in examples if _is_full_objective_seed(example)
    )
    if not eligible:
        raise ValueError("profile learner has no full-objective kernel seed rows")
    repeated: list[GameTrajectory] = []
    encoded_contexts: set[bytes] = set()
    encoded_entropies: set[str] = set()
    model_input_fingerprints: set[str] = set()
    for row_index in range(row_count):
        base = eligible[row_index % len(eligible)]
        decision = base.decisions[0]
        leaf = decision.executed_endpoint_value_leaf
        if leaf is None:
            raise AssertionError("eligible kernel seed omitted its endpoint leaf")
        varied_leaf = _vary_endpoint_context(leaf, row_index=row_index)
        encoded_contexts.add(varied_leaf.producer_context)
        varied_context = RootInformationProducerContext.from_bytes(
            varied_leaf.producer_context
        )
        encoded_entropies.add(
            json.dumps(
                varied_context.opponent_belief_entropy,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        model_input_fingerprints.add(varied_leaf.model_input_fingerprint)
        repeated.append(
            replace(
                base,
                game_id=f"{base.game_id}-kernel-{row_index:05d}",
                decisions=(
                    replace(
                        decision,
                        executed_endpoint_value_leaf=varied_leaf,
                    ),
                ),
            )
        )
    if len(encoded_contexts) != row_count:
        raise RuntimeError("kernel producer-context perturbations are not unique")
    if len(encoded_entropies) != row_count:
        raise RuntimeError("kernel entropy perturbations are not JSON-distinct")
    if len(model_input_fingerprints) != row_count:
        raise RuntimeError("kernel root-value model inputs are not unique")
    return tuple(repeated)


def _is_full_objective_seed(trajectory: GameTrajectory) -> bool:
    if len(trajectory.decisions) != 1:
        return False
    decision = trajectory.decisions[0]
    evidence = decision.planner_behavior
    return bool(
        decision.factual_target is not None
        and decision.executed_endpoint_value_leaf is not None
        and evidence is not None
        and evidence.branch is PlannerBehaviorBranch.PLANNER_CONDITIONED
        and evidence.scenario_grid_complete
        and all(candidate.rules_exact for candidate in evidence.candidates)
    )


def _vary_endpoint_context(
    leaf: RootInformationLeaf,
    *,
    row_index: int,
) -> RootInformationLeaf:
    source = RootInformationProducerContext.from_bytes(leaf.producer_context)
    entropy32 = np.float32((row_index + 1) / _ENTROPY_DENOMINATOR)
    raw = source.model_dump(mode="python")
    raw["opponent_belief_entropy"] = float(entropy32)
    varied = RootInformationProducerContext.model_validate(raw)
    encoded = encode_root_information_producer_context(
        varied.to_features(),
        root_player=varied.root_player,
    )
    decoded = RootInformationProducerContext.from_bytes(encoded)
    if decoded != varied or decoded.to_bytes() != encoded:
        raise RuntimeError("kernel producer context failed strict re-encoding")
    if np.float32(decoded.opponent_belief_entropy) != entropy32:
        raise RuntimeError("kernel entropy perturbation is not float32 exact")
    return RootInformationLeaf(
        root_observable_state=leaf.root_observable_state,
        producer_context=encoded,
        belief_summary=root_information_belief_summary(
            decoded.to_features(),
            width=len(leaf.belief_summary),
        ),
        exact_effect=leaf.exact_effect,
        actor_relation=leaf.actor_relation,
        endpoint=leaf.endpoint,
    )


def _validate_full_objective_kernel(
    result: LearnerBatchResult,
    *,
    trajectories: Sequence[GameTrajectory],
    expected_rows: int,
) -> tuple[int, int, int]:
    if (
        result.stats.total_decisions != expected_rows
        or result.stats.kept_decisions != expected_rows
        or result.stats.stale_decisions != 0
    ):
        raise RuntimeError("profile learner kernel changed its fixed row count")
    if len(result.batches) != 2 or any(
        len(item.actions) != _FIXED_MICROBATCH_SIZE for item in result.batches
    ):
        raise RuntimeError("profile learner kernel changed its 512x2 minibatches")
    pool = result.pool
    if pool is None or pool.sample_count != expected_rows:
        raise RuntimeError("profile learner kernel omitted its canonical pool")
    batch = pool.batch
    if batch.engine_teacher_mask is not None and bool(batch.engine_teacher_mask.any()):
        raise RuntimeError("profile learner retained the retired teacher objective")
    if (
        batch.factual_effect_targets is None
        or batch.factual_actor_relations is None
        or batch.factual_next_contexts is None
        or int(batch.factual_effect_targets.shape[0]) != expected_rows
    ):
        raise RuntimeError("profile learner kernel omitted factual objective rows")
    planner = batch.planner_replay
    if planner is None or planner.group_count != expected_rows:
        raise RuntimeError("profile learner kernel requires 1024 planner rows")
    root_replay = batch.root_information_value_replay
    if root_replay is None or root_replay.row_count != expected_rows:
        raise RuntimeError("profile learner kernel requires 1024 root-value rows")
    model_input_count = int(root_replay.model_inputs.states.card_ids.shape[0])
    if model_input_count != expected_rows:
        raise RuntimeError("profile learner kernel requires 1024 model inputs")
    unique_gathers = int(torch.unique(root_replay.value_input_indices).numel())
    if unique_gathers != expected_rows:
        raise RuntimeError("root-value rows did not retain unique tensorized inputs")
    leaf_fingerprints = {
        decision.executed_endpoint_value_leaf.model_input_fingerprint
        for trajectory in trajectories
        for decision in trajectory.decisions
        if decision.executed_endpoint_value_leaf is not None
    }
    if len(leaf_fingerprints) != expected_rows:
        raise RuntimeError("kernel model-input fingerprint count is not 1024")
    minibatch_planner_rows = sum(
        0 if item.planner_replay is None else item.planner_replay.group_count
        for item in result.batches
    )
    minibatch_root_rows = sum(
        0
        if item.root_information_value_replay is None
        else item.root_information_value_replay.row_count
        for item in result.batches
    )
    if (minibatch_planner_rows, minibatch_root_rows) != (
        expected_rows,
        expected_rows,
    ):
        raise RuntimeError("kernel minibatches do not cover every objective row")
    return (planner.group_count, root_replay.row_count, model_input_count)


def _kernel_workload_fingerprint(
    *,
    config: IntegratedPlannerProfileConfig,
    runtime: PlannerProfileRuntimeConfig,
    workload: PlannerLearnerWorkloadConfig,
    resolved: ResolvedPlannerRuntimeIdentity,
    trajectories: Sequence[GameTrajectory],
) -> str:
    """Hash canonical learner semantics while excluding runtime measurements."""
    digest = hashlib.sha256()
    digest.update(b"ptcg-rl/planner-profile-learner-kernel/v2\x00")
    header = {
        "anchor_sha256": workload.anchor_checkpoint_sha256,
        "checkpoint_sha256": config.expected_checkpoint_sha256,
        "corpus_sha256": config.expected_decision_corpus_sha256,
        "model_fingerprint": config.expected_model_fingerprint,
        "policy_version": config.policy_version,
        "runtime_fingerprint": resolved.runtime_fingerprint,
        "workload": _workload_tensor_semantics(workload),
    }
    _update_canonical_json(digest, header)
    for row_index, trajectory in enumerate(trajectories):
        decision = trajectory.decisions[0]
        factual = decision.factual_target
        planner = decision.planner_behavior
        leaf = decision.executed_endpoint_value_leaf
        if factual is None or planner is None or leaf is None:
            raise RuntimeError("kernel fingerprint saw an incomplete objective row")
        if not isinstance(decision.state, StateTokenArrayFeatures) or not isinstance(
            decision.options,
            EncodedOptionArrayFeatures,
        ):
            raise TypeError("kernel fingerprint requires canonical array inputs")
        root_fingerprint = canonical_planner_root_input_fingerprint(
            decision.state,
            decision.options,
            min_count=decision.min_count,
            max_count=decision.max_count,
        )
        row = {
            "row_index": row_index,
            "root_input": root_fingerprint,
            "seat": decision.seat,
            "action": decision.action,
            "action_logprob": _float32(decision.action_logprob),
            "value_pred": _float32(decision.value_pred),
            "sampling_temperature": _optional_float32(decision.sampling_temperature),
            "outcome": _float32(trajectory.reward_for_seat(decision.seat)),
            "deck_signature": trajectory.metadata.deck_signature,
            "factual": {
                "effect": tuple(_float32(value) for value in factual.effect_features),
                "actor_relation": int(factual.actor_relation),
                "next_context": factual.next_context,
            },
            "planner": _planner_tensor_semantics(planner),
            "endpoint_model_input": leaf.model_input_fingerprint,
        }
        _update_canonical_json(digest, row)
    return digest.hexdigest()


def _workload_tensor_semantics(
    workload: PlannerLearnerWorkloadConfig,
) -> dict[str, Any]:
    """Exclude machine-local paths while binding every kernel control."""
    return {
        "updates": workload.updates,
        "warmup_updates": workload.warmup_updates,
        "kernel_rows_per_update": workload.kernel_rows_per_update,
        "microbatch_size": workload.microbatch_size,
        "gradient_accumulation_steps": workload.gradient_accumulation_steps,
        "ppo_epochs": workload.ppo_epochs,
        "max_policy_age": workload.max_policy_age,
        "engine_teacher_coefficient": workload.engine_teacher_coefficient,
        "factual_effect_coefficient": workload.factual_effect_coefficient,
        "factual_successor_coefficient": workload.factual_successor_coefficient,
        "root_information_value_coefficient": (
            workload.root_information_value_coefficient
        ),
        "candidate_rerank_coefficient": workload.candidate_rerank_coefficient,
        "proposal_distillation_coefficient": (
            workload.proposal_distillation_coefficient
        ),
        "anchor_coefficient": workload.anchor_coefficient,
        "planner_target_ratio_clip": workload.planner_target_ratio_clip,
    }


def _planner_tensor_semantics(
    planner: PlannerBehaviorEvidence,
) -> dict[str, Any]:
    """Return only planner fields consumed by PPO tensor collation."""
    return {
        "branch": int(planner.branch),
        "selected_candidate_index": planner.selected_candidate_index,
        "policy_version": planner.policy_version,
        "model_fingerprint": planner.model_fingerprint,
        "constructor_fingerprint": planner.constructor_fingerprint,
        "scorer_fingerprint": planner.scorer_fingerprint,
        "controller_fingerprint": planner.controller_fingerprint,
        "planner_fingerprint": planner.planner_fingerprint,
        "scenario_grid_complete": planner.scenario_grid_complete,
        "support_exhaustive": planner.support_exhaustive,
        "support_censored": planner.support_censored,
        "planner_temperature": _float32(planner.planner_temperature),
        "candidates": [
            {
                "action": candidate.action,
                "aggregate_features": tuple(
                    _float32(value) for value in candidate.aggregate_features
                ),
                "score_prior": _float32(candidate.score_prior),
                "target_probability": _float32(candidate.target_probability),
                "behavior_probability": _float32(candidate.behavior_probability),
                "rules_exact": candidate.rules_exact,
            }
            for candidate in planner.candidates
        ],
    }


def _float32(value: float) -> float:
    return float(np.float32(value))


def _optional_float32(value: float | None) -> float | None:
    return None if value is None else _float32(value)


def _update_canonical_json(digest: Any, payload: Any) -> None:
    digest.update(
        json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\x00")


__all__ = ["PreparedProfileLearnerKernel", "prepare_profile_learner_kernel"]
