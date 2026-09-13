"""Streaming split evaluation for exact actor-private behavior cloning."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ptcg_rl.context.public_event_arrays import collate_public_event_deltas
from ptcg_rl.engine.constants import OptionType
from ptcg_rl.model.sequence.action import collate_accepted_actions
from ptcg_rl.model.simple_stateless import (
    SimpleStatelessPolicyValueNet,
    simple_count_first_rows,
)
from ptcg_rl.model.simple_stateless.backbone import (
    select_simple_stateless_backbone_rows,
)
from ptcg_rl.rl.policy_inputs import (
    collate_simple_stateless_actor_rows,
    collate_simple_stateless_observation_rows,
)
from ptcg_rl.rl.transition_distillation import (
    policy_forward_kl_losses,
    visited_option_prefix_mask,
)
from ptcg_rl.training.simple_stateless_pretrain_artifact import (
    POLICY_ACTION_TYPE_BUCKETS,
    SupervisedPolicyActionTypeMetrics,
    SupervisedPolicyEvaluationMetrics,
    supervised_outcome_weighting_record,
)
from ptcg_rl.training.simple_stateless_pretrain_data import (
    ReplayPretrainingDatasetManifest,
    ReplayPretrainingExample,
    ReplaySplit,
    iter_pretraining_rows,
    load_pretraining_part,
    pretraining_split_indices,
)
from ptcg_rl.training.simple_stateless_pretrain_scope import (
    TrainableParameterScope,
    resolve_pretraining_batch_routes,
)
from ptcg_rl.training.simple_stateless_pretrain_weighting import (
    ReplayPretrainingOutcomeWeightsConfig,
)
from ptcg_rl.training.simple_stateless_temporal_pretrain import (
    TemporalMonitorPlan,
    TemporalPretrainingBatchPlan,
    temporal_epoch_batches,
)


@dataclass(frozen=True)
class _EvaluatedRow:
    policy_nll: float
    exact: bool
    action_type: str
    baseline_policy_forward_kl: float | None = None


@dataclass(frozen=True)
class _PreparedTemporalMonitorPart:
    plans: tuple[TemporalPretrainingBatchPlan, ...]
    rows_by_index: dict[int, ReplayPretrainingExample]


def evaluate_temporal_pretraining_monitor(
    model: SimpleStatelessPolicyValueNet,
    *,
    dataset: ReplayPretrainingDatasetManifest,
    dataset_dir: Path,
    monitor: TemporalMonitorPlan,
    batch_size: int,
    target_chunk_decisions: int,
    max_context_blocks: int,
    device: torch.device,
    trainable_scope: TrainableParameterScope,
    maximum_batch_context_blocks: int,
    maximum_batch_context_state_tokens: int,
    maximum_batch_target_options: int,
) -> SupervisedPolicyEvaluationMetrics:
    """Evaluate one deterministic split monitor through causal replay."""
    episode_nll: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    decision_nll_sum = 0.0
    token_nll_sum = 0.0
    examples_seen = 0
    decode_tokens = 0
    exact_sequences = 0
    action_type_rows: Counter[str] = Counter()
    action_type_nll: defaultdict[str, float] = defaultdict(float)
    action_type_exact: Counter[str] = Counter()
    was_training = model.training
    model.eval()
    selected_parts = tuple(
        (part_index, record, selected)
        for part_index, (record, selected) in enumerate(
            zip(dataset.parts, monitor.rows_by_part, strict=True)
        )
        if selected
    )
    if not selected_parts:
        raise RuntimeError("temporal split monitor has no selected parts")
    started = time.perf_counter()
    try:
        with torch.no_grad(), ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="temporal-monitor-prefetch",
        ) as executor:
            future: Future[_PreparedTemporalMonitorPart] = executor.submit(
                _prepare_temporal_monitor_part,
                dataset_dir=dataset_dir,
                filename=selected_parts[0][1].filename,
                selected=selected_parts[0][2],
                part_index=selected_parts[0][0],
                catalog_fingerprint=dataset.public_catalog_fingerprint,
                input_contract_fingerprint=(
                    dataset.input_contract_fingerprint
                ),
                batch_size=batch_size,
                target_chunk_decisions=target_chunk_decisions,
                max_context_blocks=max_context_blocks,
                maximum_batch_context_blocks=maximum_batch_context_blocks,
                maximum_batch_context_state_tokens=(
                    maximum_batch_context_state_tokens
                ),
                maximum_batch_target_options=maximum_batch_target_options,
            )
            for position, (_part_index, _record, _selected) in enumerate(
                selected_parts
            ):
                prepared = future.result()
                if position + 1 < len(selected_parts):
                    next_part_index, next_record, next_selected = (
                        selected_parts[position + 1]
                    )
                    future = executor.submit(
                        _prepare_temporal_monitor_part,
                        dataset_dir=dataset_dir,
                        filename=next_record.filename,
                        selected=next_selected,
                        part_index=next_part_index,
                        catalog_fingerprint=(
                            dataset.public_catalog_fingerprint
                        ),
                        input_contract_fingerprint=(
                            dataset.input_contract_fingerprint
                        ),
                        batch_size=batch_size,
                        target_chunk_decisions=target_chunk_decisions,
                        max_context_blocks=max_context_blocks,
                        maximum_batch_context_blocks=(
                            maximum_batch_context_blocks
                        ),
                        maximum_batch_context_state_tokens=(
                            maximum_batch_context_state_tokens
                        ),
                        maximum_batch_target_options=(
                            maximum_batch_target_options
                        ),
                    )
                for plan in prepared.plans:
                    examples = tuple(
                        prepared.rows_by_index[index]
                        for index in plan.target_indices
                    )
                    context = tuple(
                        prepared.rows_by_index[index]
                        for index in plan.context_indices
                    )
                    rows, batch_token_nll, batch_tokens = (
                        _evaluate_temporal_batch(
                            model,
                            examples=examples,
                            context_examples=context,
                            plan=plan,
                            device=device,
                            trainable_scope=trainable_scope,
                        )
                    )
                    for example, row in zip(examples, rows, strict=True):
                        key = (example.episode_id, example.player_index)
                        episode_nll[key].append(row.policy_nll)
                        action_type_rows[row.action_type] += 1
                        action_type_nll[row.action_type] += row.policy_nll
                        action_type_exact[row.action_type] += row.exact
                    decision_nll_sum += sum(row.policy_nll for row in rows)
                    token_nll_sum += batch_token_nll
                    decode_tokens += batch_tokens
                    exact_sequences += sum(row.exact for row in rows)
                    examples_seen += len(examples)
                completed_parts = position + 1
                if (
                    completed_parts % 16 == 0
                    or completed_parts == len(selected_parts)
                ):
                    print(
                        "pretraining_startup phase=initial_monitor_progress "
                        f"parts={completed_parts}/{len(selected_parts)} "
                        f"targets={examples_seen}/{monitor.target_count} "
                        "elapsed_seconds="
                        f"{time.perf_counter() - started:.2f}",
                        flush=True,
                    )
    finally:
        model.train(was_training)
    if examples_seen != monitor.target_count or not episode_nll or decode_tokens <= 0:
        raise RuntimeError("temporal split monitor produced incomplete evidence")
    episode_normalized = sum(
        sum(values) / len(values) for values in episode_nll.values()
    ) / len(episode_nll)
    return SupervisedPolicyEvaluationMetrics(
        split=monitor.split,
        examples=examples_seen,
        episodes=len(episode_nll),
        decode_tokens=decode_tokens,
        episode_normalized_policy_nll=episode_normalized,
        decision_policy_nll=decision_nll_sum / examples_seen,
        token_policy_nll=token_nll_sum / decode_tokens,
        exact_action_sequence_accuracy=exact_sequences / examples_seen,
        action_types={
            action_type: SupervisedPolicyActionTypeMetrics(
                rows=action_type_rows[action_type],
                policy_nll=(
                    None
                    if action_type_rows[action_type] == 0
                    else action_type_nll[action_type]
                    / action_type_rows[action_type]
                ),
                exact_action_sequence_accuracy=(
                    None
                    if action_type_rows[action_type] == 0
                    else action_type_exact[action_type]
                    / action_type_rows[action_type]
                ),
            )
            for action_type in POLICY_ACTION_TYPE_BUCKETS
        },
    )


def _prepare_temporal_monitor_part(
    *,
    dataset_dir: Path,
    filename: str,
    selected: tuple[int, ...],
    part_index: int,
    catalog_fingerprint: str,
    input_contract_fingerprint: str,
    batch_size: int,
    target_chunk_decisions: int,
    max_context_blocks: int,
    maximum_batch_context_blocks: int,
    maximum_batch_context_state_tokens: int,
    maximum_batch_target_options: int,
) -> _PreparedTemporalMonitorPart:
    """Load and materialize one monitor shard while the GPU evaluates another."""
    part = load_pretraining_part(dataset_dir / "parts" / filename)
    plans = temporal_epoch_batches(
        part,
        target_decisions=batch_size,
        target_chunk_decisions=target_chunk_decisions,
        max_context_blocks=max_context_blocks,
        seed=part_index,
        selected_targets=selected,
        maximum_batch_context_blocks=maximum_batch_context_blocks,
        maximum_batch_context_state_tokens=maximum_batch_context_state_tokens,
        maximum_batch_target_options=maximum_batch_target_options,
    )
    needed_indices = tuple(
        sorted(
            {
                index
                for plan in plans
                for index in plan.context_indices
            }
        )
    )
    materialized = tuple(
        iter_pretraining_rows(
            part,
            needed_indices,
            catalog_fingerprint=catalog_fingerprint,
            input_contract_fingerprint=input_contract_fingerprint,
        )
    )
    return _PreparedTemporalMonitorPart(
        plans=plans,
        rows_by_index=dict(
            zip(needed_indices, materialized, strict=True)
        ),
    )


def evaluate_pretraining_policy(
    model: SimpleStatelessPolicyValueNet,
    *,
    dataset: ReplayPretrainingDatasetManifest,
    dataset_dir: Path,
    split: ReplaySplit,
    batch_size: int,
    device: torch.device,
    trainable_scope: TrainableParameterScope,
    outcome_weights: ReplayPretrainingOutcomeWeightsConfig | None,
    baseline_model: SimpleStatelessPolicyValueNet | None = None,
) -> SupervisedPolicyEvaluationMetrics:
    """Evaluate one complete held-out split without materializing it in memory."""
    expected_examples = dataset.split_examples(split)
    if expected_examples <= 0:
        raise ValueError(f"pretraining dataset has no {split} examples")
    episode_nll: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    episode_outcomes: dict[tuple[int, int], float] = {}
    episode_source_weight_sums: defaultdict[tuple[int, int], float] = defaultdict(
        float
    )
    decision_nll_sum = 0.0
    token_nll_sum = 0.0
    examples_seen = 0
    decode_tokens = 0
    exact_sequences = 0
    action_type_rows: Counter[str] = Counter()
    action_type_nll: defaultdict[str, float] = defaultdict(float)
    action_type_exact: Counter[str] = Counter()
    episode_baseline_kl: defaultdict[tuple[int, int], list[float]] = defaultdict(
        list
    )
    decision_baseline_kl_sum = 0.0
    if baseline_model is not None and baseline_model.config != model.config:
        raise ValueError("baseline policy topology differs from evaluated model")
    was_training = model.training
    baseline_was_training = None if baseline_model is None else baseline_model.training
    model.eval()
    if baseline_model is not None:
        baseline_model.eval()
    try:
        with torch.no_grad():
            for record in dataset.parts:
                if record.split_examples and record.split_examples[split] == 0:
                    continue
                part = load_pretraining_part(dataset_dir / "parts" / record.filename)
                split_indices = pretraining_split_indices(part, split)
                for start in range(0, int(split_indices.size), batch_size):
                    indices = split_indices[start : start + batch_size]
                    examples = tuple(
                        iter_pretraining_rows(
                            part,
                            indices.tolist(),
                            catalog_fingerprint=(dataset.public_catalog_fingerprint),
                            input_contract_fingerprint=(
                                dataset.input_contract_fingerprint
                            ),
                        )
                    )
                    if examples:
                        batch_result = _evaluate_batch(
                            model,
                            examples=examples,
                            device=device,
                            trainable_scope=trainable_scope,
                            baseline_model=baseline_model,
                        )
                        rows, batch_token_nll, batch_tokens = batch_result
                        for example, row in zip(
                            examples,
                            rows,
                            strict=True,
                        ):
                            episode_seat = (
                                example.episode_id,
                                example.player_index,
                            )
                            if outcome_weights is not None:
                                outcome_weights.multiplier(example.outcome)
                            previous_outcome = episode_outcomes.setdefault(
                                episode_seat,
                                example.outcome,
                            )
                            if previous_outcome != example.outcome:
                                raise ValueError(
                                    "one evaluated episode-seat has conflicting outcomes"
                                )
                            episode_source_weight_sums[episode_seat] += (
                                example.example_weight
                            )
                            episode_nll[episode_seat].append(row.policy_nll)
                            action_type_rows[row.action_type] += 1
                            action_type_nll[row.action_type] += row.policy_nll
                            action_type_exact[row.action_type] += row.exact
                            if row.baseline_policy_forward_kl is not None:
                                episode_baseline_kl[episode_seat].append(
                                    row.baseline_policy_forward_kl
                                )
                                decision_baseline_kl_sum += (
                                    row.baseline_policy_forward_kl
                                )
                        decision_nll_sum += sum(row.policy_nll for row in rows)
                        token_nll_sum += batch_token_nll
                        decode_tokens += batch_tokens
                        exact_sequences += sum(row.exact for row in rows)
                        examples_seen += len(examples)
    finally:
        model.train(was_training)
        if baseline_model is not None and baseline_was_training is not None:
            baseline_model.train(baseline_was_training)
    if examples_seen != expected_examples:
        raise RuntimeError(
            f"{split} evaluation row count changed: "
            f"{examples_seen}!={expected_examples}"
        )
    if not episode_nll or decode_tokens <= 0:
        raise RuntimeError(f"{split} evaluation produced no policy evidence")
    episode_normalized = sum(
        sum(values) / len(values) for values in episode_nll.values()
    ) / len(episode_nll)
    if set(episode_outcomes) != set(episode_nll):
        raise RuntimeError(f"{split} evaluation outcomes are incomplete")
    if set(episode_source_weight_sums) != set(episode_nll) or any(
        not math.isclose(weight_sum, 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6)
        for weight_sum in episode_source_weight_sums.values()
    ):
        raise ValueError(f"{split} source example weights are not episode normalized")
    outcome_weighting = (
        None
        if outcome_weights is None
        else supervised_outcome_weighting_record(
            split=split,
            multipliers=outcome_weights,
            episode_outcomes=episode_outcomes,
            source_example_weight_sum=sum(episode_source_weight_sums.values()),
        )
    )
    outcome_weighted_episode_nll = None
    if outcome_weights is not None:
        raw_outcome_weight_sum = sum(
            outcome_weights.multiplier(outcome) for outcome in episode_outcomes.values()
        )
        outcome_weighted_episode_nll = (
            sum(
                outcome_weights.multiplier(episode_outcomes[episode_id])
                * (sum(values) / len(values))
                for episode_id, values in episode_nll.items()
            )
            / raw_outcome_weight_sum
        )
    if baseline_model is not None and set(episode_baseline_kl) != set(episode_nll):
        raise RuntimeError(f"{split} evaluation baseline KL rows are incomplete")
    episode_normalized_baseline_kl = (
        None
        if baseline_model is None
        else sum(sum(values) / len(values) for values in episode_baseline_kl.values())
        / len(episode_baseline_kl)
    )
    return SupervisedPolicyEvaluationMetrics(
        split=split,
        examples=examples_seen,
        episodes=len(episode_nll),
        decode_tokens=decode_tokens,
        episode_normalized_policy_nll=episode_normalized,
        outcome_weighted_episode_policy_nll=outcome_weighted_episode_nll,
        decision_policy_nll=decision_nll_sum / examples_seen,
        token_policy_nll=token_nll_sum / decode_tokens,
        exact_action_sequence_accuracy=exact_sequences / examples_seen,
        episode_normalized_baseline_policy_forward_kl=(episode_normalized_baseline_kl),
        decision_baseline_policy_forward_kl=(
            None if baseline_model is None else decision_baseline_kl_sum / examples_seen
        ),
        outcome_weighting=outcome_weighting,
        action_types={
            action_type: SupervisedPolicyActionTypeMetrics(
                rows=action_type_rows[action_type],
                policy_nll=(
                    None
                    if action_type_rows[action_type] == 0
                    else (action_type_nll[action_type] / action_type_rows[action_type])
                ),
                exact_action_sequence_accuracy=(
                    None
                    if action_type_rows[action_type] == 0
                    else (
                        action_type_exact[action_type] / action_type_rows[action_type]
                    )
                ),
            )
            for action_type in POLICY_ACTION_TYPE_BUCKETS
        },
    )


def _evaluate_batch(
    model: SimpleStatelessPolicyValueNet,
    *,
    examples: tuple[ReplayPretrainingExample, ...],
    device: torch.device,
    trainable_scope: TrainableParameterScope,
    baseline_model: SimpleStatelessPolicyValueNet | None,
) -> tuple[list[_EvaluatedRow], float, int]:
    batch = collate_simple_stateless_actor_rows(
        tuple(example.actor_row for example in examples),
        device=device,
        deduplicate_belief=True,
    )
    routes = resolve_pretraining_batch_routes(
        batch.deck_signatures,
        model.config,
        device=device,
        trainable_scope=trainable_scope,
    )
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        state = model.encode_observation_state(
            state=batch.states,
            unique_deck_card_ids=batch.unique_deck_card_ids,
            deck_counts=batch.deck_counts,
            deck_valid_mask=batch.deck_valid_mask,
            belief_summary=batch.belief_summary,
            route_plan=routes,
            allow_unrouted_rows=routes.allow_unrouted_rows,
        )
        option_embeddings = model.encode_legal_options(
            state,
            batch.options,
            route_plan=routes,
            allow_unrouted_rows=routes.allow_unrouted_rows,
        )
        evaluation = model.heads.teacher_forced(
            state.policy,
            state.opponent_belief,
            option_embeddings,
            batch.options,
            tuple(example.action for example in examples),
            route_plan=routes,
            evaluate_prefix_values=False,
        )
        baseline_evaluation = None
        if baseline_model is not None:
            baseline_state = baseline_model.encode_observation_state(
                state=batch.states,
                unique_deck_card_ids=batch.unique_deck_card_ids,
                deck_counts=batch.deck_counts,
                deck_valid_mask=batch.deck_valid_mask,
                belief_summary=batch.belief_summary,
                route_plan=routes,
                allow_unrouted_rows=routes.allow_unrouted_rows,
            )
            baseline_option_embeddings = baseline_model.encode_legal_options(
                baseline_state,
                batch.options,
                route_plan=routes,
                allow_unrouted_rows=routes.allow_unrouted_rows,
            )
            baseline_evaluation = baseline_model.heads.teacher_forced(
                baseline_state.policy,
                baseline_state.opponent_belief,
                baseline_option_embeddings,
                batch.options,
                tuple(example.action for example in examples),
                route_plan=routes,
                evaluate_prefix_values=False,
            )
        greedy = model.heads.greedy_decode(
            state.policy,
            state.opponent_belief,
            option_embeddings,
            batch.options,
            route_plan=routes,
        )
    baseline_row_kl: list[float] | None = None
    if baseline_evaluation is not None:
        count_mask = simple_count_first_rows(batch.options)
        option_prefix_mask = visited_option_prefix_mask(
            tuple(example.action for example in examples),
            max_counts=batch.options.max_counts,
            count_mask=count_mask,
        )
        anchor_kl = policy_forward_kl_losses(
            student_step_logits=evaluation.step_logits,
            teacher_step_logits=baseline_evaluation.step_logits,
            student_count_logits=evaluation.count_logits,
            teacher_count_logits=baseline_evaluation.count_logits,
            option_prefix_mask=option_prefix_mask,
            count_mask=count_mask,
            reference=evaluation.action_logprobs,
            rows=len(examples),
        )
        baseline_row_kl = [
            max(float(value), 0.0) for value in anchor_kl.row_kl.detach().cpu().tolist()
        ]
    row_nll_tensor = -evaluation.action_logprobs.float()
    token_nll = -(evaluation.token_logprobs.float() * evaluation.token_mask).sum()
    decode_tokens = int(evaluation.token_mask.sum().item())
    rows: list[_EvaluatedRow] = []
    for row_index, (nll, predicted, example) in enumerate(
        zip(
            row_nll_tensor.cpu().tolist(),
            greedy,
            examples,
            strict=True,
        )
    ):
        rows.append(
            _EvaluatedRow(
                policy_nll=float(nll),
                exact=predicted == example.action,
                action_type=_teacher_first_action_type(example),
                baseline_policy_forward_kl=(
                    None if baseline_row_kl is None else baseline_row_kl[row_index]
                ),
            )
        )
    return (
        rows,
        float(token_nll.cpu()),
        decode_tokens,
    )


def _evaluate_temporal_batch(
    model: SimpleStatelessPolicyValueNet,
    *,
    examples: tuple[ReplayPretrainingExample, ...],
    context_examples: tuple[ReplayPretrainingExample, ...],
    plan: TemporalPretrainingBatchPlan,
    device: torch.device,
    trainable_scope: TrainableParameterScope,
) -> tuple[list[_EvaluatedRow], float, int]:
    """Evaluate one monitor batch through production temporal conditioning."""
    target_batch = collate_simple_stateless_actor_rows(
        tuple(example.actor_row for example in examples),
        device=device,
        deduplicate_belief=True,
    )
    context_batch = collate_simple_stateless_observation_rows(
        tuple(example.actor_row for example in context_examples),
        device=device,
        deduplicate_belief=True,
    )
    target_routes = resolve_pretraining_batch_routes(
        target_batch.deck_signatures,
        model.config,
        device=device,
        trainable_scope=trainable_scope,
    )
    context_routes = resolve_pretraining_batch_routes(
        context_batch.deck_signatures,
        model.config,
        device=device,
        trainable_scope=trainable_scope,
    )
    events = tuple(
        example.actor_row.public_event_delta for example in context_examples
    )
    actions = tuple(example.accepted_action for example in context_examples)
    if any(item is None for item in events) or any(item is None for item in actions):
        raise ValueError("temporal monitor context payload is incomplete")
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        snapshots = model.encode_observation_state(
            state=context_batch.states,
            unique_deck_card_ids=context_batch.unique_deck_card_ids,
            deck_counts=context_batch.deck_counts,
            deck_valid_mask=context_batch.deck_valid_mask,
            belief_summary=context_batch.belief_summary,
            route_plan=context_routes,
            allow_unrouted_rows=context_routes.allow_unrouted_rows,
        )
        temporal = model.replay_sequence(
            snapshots,
            collate_public_event_deltas(
                tuple(item for item in events if item is not None),
                device=device,
            ),
            collate_accepted_actions(
                tuple(item for item in actions if item is not None),
                device=device,
            ),
            sequence_offsets=plan.sequence_offsets,
            block_indices=torch.tensor(
                plan.block_indices,
                dtype=torch.long,
                device=device,
            ),
        )
        conditioned = model.condition_sequence(snapshots, temporal)
        state = select_simple_stateless_backbone_rows(
            conditioned,
            torch.tensor(
                plan.target_row_indices,
                dtype=torch.long,
                device=device,
            ),
        )
        option_embeddings = model.encode_legal_options(
            state,
            target_batch.options,
            route_plan=target_routes,
            allow_unrouted_rows=target_routes.allow_unrouted_rows,
        )
        evaluation = model.heads.teacher_forced(
            state.policy,
            state.opponent_belief,
            option_embeddings,
            target_batch.options,
            tuple(example.action for example in examples),
            route_plan=target_routes,
            evaluate_prefix_values=False,
        )
        greedy = model.heads.greedy_decode(
            state.policy,
            state.opponent_belief,
            option_embeddings,
            target_batch.options,
            route_plan=target_routes,
        )
    row_nll: Tensor = -evaluation.action_logprobs.float()
    token_nll = -(evaluation.token_logprobs.float() * evaluation.token_mask).sum()
    rows = [
        _EvaluatedRow(
            policy_nll=float(nll),
            exact=predicted == example.action,
            action_type=_teacher_first_action_type(example),
        )
        for nll, predicted, example in zip(
            row_nll.cpu().tolist(),
            greedy,
            examples,
            strict=True,
        )
    ]
    return rows, float(token_nll.cpu()), int(evaluation.token_mask.sum().item())


def _teacher_first_action_type(example: ReplayPretrainingExample) -> str:
    """Bucket the teacher's first selected engine option."""
    if not example.action:
        return "OTHER"
    option_type = int(example.actor_row.options.option_types[example.action[0]])
    mapping = {
        int(OptionType.PLAY): "PLAY",
        int(OptionType.ABILITY): "ABILITY",
        int(OptionType.SKILL): "ABILITY",
        int(OptionType.ATTACH): "ATTACH",
        int(OptionType.ATTACK): "ATTACK",
        int(OptionType.RETREAT): "RETREAT",
        int(OptionType.END): "END",
    }
    return mapping.get(option_type, "OTHER")


__all__ = [
    "evaluate_pretraining_policy",
    "evaluate_temporal_pretraining_monitor",
]
