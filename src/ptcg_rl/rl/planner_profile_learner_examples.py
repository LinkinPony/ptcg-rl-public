"""Synthetic learner-kernel seed rows built from exact public replay evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch

from ptcg_rl.actions.encoding import StateTokenLayout, encode_option_arrays
from ptcg_rl.context import GameContextFeatures, OpponentBeliefFeatureProducer
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.decks.identity import CanonicalDeck, canonicalize_deck
from ptcg_rl.evaluation.planner_profile_corpus_types import PlannerProfileCorpusRecord
from ptcg_rl.evaluation.planner_profile_workloads import PlannerProfileRuntimeConfig
from ptcg_rl.model import collate_encoded_options, collate_state_tokens
from ptcg_rl.model.network import ActionEvaluation, AgentPolicyValueNet
from ptcg_rl.model.root_input_fingerprint import (
    canonical_planner_root_input_fingerprint,
)
from ptcg_rl.model.state_encoder import encode_observation_token_arrays
from ptcg_rl.rl.collection import ModelRolloutPolicy
from ptcg_rl.rl.experience import (
    DecisionRecord,
    GameMetadata,
    GameTrajectory,
    TrajectoryDeckContext,
)
from ptcg_rl.rl.planner_behavior_policy_contract import PlannerPolicyDecision
from ptcg_rl.rl.planner_evidence import PlannerBehaviorBranch
from ptcg_rl.rl.planner_profile_learner_data import ProfileLearnerTransition
from ptcg_rl.rl.planner_runtime_factory import create_planner_behavior_runtime
from ptcg_rl.rl.planner_runtime_identity import ResolvedPlannerRuntimeIdentity
from ptcg_rl.rl.rollout import RolloutPlannerBatch, RolloutPlannerRowContext


def build_profile_kernel_seed_examples(
    records: Sequence[PlannerProfileCorpusRecord],
    *,
    transitions: Mapping[str, ProfileLearnerTransition],
    producer: OpponentBeliefFeatureProducer,
    runtime: PlannerProfileRuntimeConfig,
    resolved: ResolvedPlannerRuntimeIdentity,
    model: AgentPolicyValueNet,
    native_library_path: Path,
) -> tuple[GameTrajectory, ...]:
    """Build replay-derived seed rows for a fixed synthetic learner kernel.

    These rows intentionally are not an on-policy rollout stream. The corpus
    action is installed as the selected planner-support action so every retained
    seed can exercise the same learner objectives under a fixed workload.
    """
    model.eval()
    policy = ModelRolloutPolicy(
        model,
        policy_version=resolved.runtime_identity.policy_version,
        autocast="bf16",
        planner_context_capacity=runtime.planner.contexts.retained_root_rows,
        verified_model_fingerprint=resolved.runtime_identity.model_fingerprint,
        proposal_version=resolved.runtime_identity.proposal_version,
    )
    planner_runtime = create_planner_behavior_runtime(
        runtime_config=runtime.planner,
        sampler_config=runtime.belief.sampler,
        belief_config=runtime.belief.producer,
        stochastic_seed=runtime.belief.stochastic_seed,
        native_library_path=native_library_path,
    )
    examples: list[GameTrajectory] = []
    batch_size = runtime.planner.batching.max_root_rows_per_request
    try:
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            examples.extend(
                _build_seed_batch(
                    chunk,
                    transitions=transitions,
                    producer=producer,
                    runtime=runtime,
                    resolved=resolved,
                    policy=policy,
                    model=model,
                    service=planner_runtime.service,
                )
            )
    finally:
        planner_runtime.close()
    return tuple(examples)


def _build_seed_batch(
    records: Sequence[PlannerProfileCorpusRecord],
    *,
    transitions: Mapping[str, ProfileLearnerTransition],
    producer: OpponentBeliefFeatureProducer,
    runtime: PlannerProfileRuntimeConfig,
    resolved: ResolvedPlannerRuntimeIdentity,
    policy: ModelRolloutPolicy,
    model: AgentPolicyValueNet,
    service: Any,
) -> tuple[GameTrajectory, ...]:
    prepared = tuple(_prepare_root(record, producer=producer) for record in records)
    states = replace(
        collate_state_tokens(
            tuple(item[0] for item in prepared),
            device="cuda",
        ),
        root_input_fingerprints=tuple(item[4] for item in prepared),
    )
    options = collate_encoded_options(
        tuple(item[1] for item in prepared),
        min_counts=tuple(item[2] for item in prepared),
        max_counts=tuple(item[3] for item in prepared),
        device="cuda",
    )
    decks = tuple(canonicalize_deck(record.own_deck) for record in records)
    deck_batch = DeckBatch.from_decks(decks, device="cuda")
    actions = tuple(record.executed_action for record in records)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        evaluation = model.evaluate_actions(
            states,
            options,
            actions,
            decks=deck_batch,
            temperature=1.0,
        )
    trace = policy.sample_decode_with_trace_for_request(
        states,
        options,
        deck_batch,
        temperature=1.0,
        model_version_lease=None,
        retain_planner_context=True,
    )
    rows = tuple(
        RolloutPlannerRowContext(
            game_id=f"profile-learner-{record.row_id}",
            seat=_record_root_player(record),
            policy_role="candidate",
            should_record=True,
            observation=_augmented_observation(record, prepared[index][5]),
            context_features=prepared[index][5],
            context_snapshot=record.context_snapshot,
            deck_pair=(record.own_deck, record.own_deck),
        )
        for index, record in enumerate(records)
    )
    decisions = service.plan_batch(
        RolloutPlannerBatch(
            policy=policy,
            rows=rows,
            states=states,
            options=options,
            decks=deck_batch,
            base_actions=actions,
            base_logprobs=tuple(
                float(value)
                for value in evaluation.action_logprobs.detach().float().cpu()
            ),
            base_values=tuple(
                float(value) for value in evaluation.values.detach().float().cpu()
            ),
            policy_version=resolved.runtime_identity.policy_version,
            model_fingerprint=resolved.runtime_identity.model_fingerprint,
            proposal_version=resolved.runtime_identity.proposal_version,
            planner_context_handles=trace.planner_context_handles,
        )
    )
    if len(decisions) != len(records) or any(item is None for item in decisions):
        raise RuntimeError("profile learner planner evidence is misaligned")
    return tuple(
        _trajectory_from_decision(
            record,
            planner_decision=cast(PlannerPolicyDecision, decisions[index]),
            transition=transitions[record.row_id],
            deck=decks[index],
            prepared=prepared[index],
            evaluation=evaluation,
            evaluation_index=index,
            resolved=resolved,
        )
        for index, record in enumerate(records)
    )


def _trajectory_from_decision(
    record: PlannerProfileCorpusRecord,
    *,
    planner_decision: PlannerPolicyDecision,
    transition: ProfileLearnerTransition,
    deck: CanonicalDeck,
    prepared: tuple[Any, Any, int, int, str, GameContextFeatures],
    evaluation: ActionEvaluation,
    evaluation_index: int,
    resolved: ResolvedPlannerRuntimeIdentity,
) -> GameTrajectory:
    evidence = planner_decision.planner_behavior
    action_logprob = float(evaluation.action_logprobs[evaluation_index].item())
    token_trace: tuple[tuple[float, ...], tuple[float, ...], bool] | None
    if evidence.branch is PlannerBehaviorBranch.PLANNER_CONDITIONED:
        selected = next(
            (
                candidate_index
                for candidate_index, candidate in enumerate(evidence.candidates)
                if candidate.action == record.executed_action
            ),
            None,
        )
        if selected is None:
            raise RuntimeError("planner workload dropped its corpus action anchor")
        evidence = replace(evidence, selected_candidate_index=selected)
        selected_logprob = evidence.selected_old_logprob
        if selected_logprob is None:
            raise RuntimeError("planner workload has no selected probability")
        action_logprob = selected_logprob
        token_trace = None
    else:
        token_trace = _evaluation_token_trace(
            evaluation,
            index=evaluation_index,
        )
    root_player = _record_root_player(record)
    decision_record = DecisionRecord(
        seat=root_player,
        decision_index=0,
        state=prepared[0],
        options=prepared[1],
        min_count=prepared[2],
        max_count=prepared[3],
        action=record.executed_action,
        action_logprob=action_logprob,
        value_pred=float(evaluation.values[evaluation_index].item()),
        policy_version=resolved.runtime_identity.policy_version,
        sampling_temperature=evidence.planner_temperature,
        token_logprobs=None if token_trace is None else token_trace[0],
        prefix_value_preds=None if token_trace is None else token_trace[1],
        stop_sampled=None if token_trace is None else token_trace[2],
        factual_target=transition.factual_target,
        planner_behavior=evidence,
        executed_endpoint_value_leaf=transition.endpoint_leaf,
    )
    reward = float(record.final_root_outcome)
    return GameTrajectory(
        game_id=f"profile-learner-{record.row_id}",
        seats_reward=((reward, -reward) if root_player == 0 else (-reward, reward)),
        decisions=(decision_record,),
        metadata=GameMetadata(
            deck_signature=deck.signature,
            policy_version=resolved.runtime_identity.policy_version,
            episode_length=1,
        ),
        deck_context=TrajectoryDeckContext.from_deck_pair(
            (deck.card_ids, deck.card_ids)
        ),
    )


def _prepare_root(
    record: PlannerProfileCorpusRecord,
    *,
    producer: OpponentBeliefFeatureProducer,
) -> tuple[Any, Any, int, int, str, GameContextFeatures]:
    context_features = producer.augment(record.observation, record.context_features)
    observation = _augmented_observation(record, context_features)
    layout = StateTokenLayout.from_observation(
        observation,
        context_features=context_features,
    )
    state = encode_observation_token_arrays(observation, layout=layout)
    options = encode_option_arrays(observation.get("select"), layout)
    select = cast(Mapping[str, Any], observation["select"])
    min_count = min(len(options), max(0, int(select.get("minCount", 0))))
    max_count = min(
        len(options),
        max(min_count, int(select.get("maxCount", len(options)))),
    )
    fingerprint = canonical_planner_root_input_fingerprint(
        state,
        options,
        min_count=min_count,
        max_count=max_count,
    )
    return (state, options, min_count, max_count, fingerprint, context_features)


def _augmented_observation(
    record: PlannerProfileCorpusRecord,
    context_features: GameContextFeatures,
) -> Mapping[str, Any]:
    observation = dict(record.observation)
    observation["gameContext"] = context_features.as_observation_dict()
    return observation


def _evaluation_token_trace(
    evaluation: ActionEvaluation,
    *,
    index: int,
) -> tuple[tuple[float, ...], tuple[float, ...], bool]:
    if (
        evaluation.token_logprobs is None
        or evaluation.prefix_values is None
        or evaluation.token_mask is None
        or evaluation.stop_sampled is None
    ):
        raise RuntimeError("profile learner fallback lacks token behavior evidence")
    mask = evaluation.token_mask[index]
    token_logprobs = tuple(
        float(value)
        for value in evaluation.token_logprobs[index]
        .masked_select(mask)
        .detach()
        .float()
        .cpu()
    )
    prefix_values = tuple(
        float(value)
        for value in evaluation.prefix_values[index]
        .masked_select(mask)
        .detach()
        .float()
        .cpu()
    )
    return (
        token_logprobs,
        prefix_values,
        bool(evaluation.stop_sampled[index].item()),
    )


def _record_root_player(record: PlannerProfileCorpusRecord) -> int:
    root_player = record.context_snapshot.player_index
    if root_player not in (0, 1):
        raise ValueError("profile learner corpus root has no valid player seat")
    return root_player


__all__ = ["build_profile_kernel_seed_examples"]
