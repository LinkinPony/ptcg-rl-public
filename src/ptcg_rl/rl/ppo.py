"""PPO loss and effective-update utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor
from torch.nn import functional

from ptcg_rl.actions.selection import is_unordered_set_selection
from ptcg_rl.context import PublicEventBatch, validate_public_event_batch
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.engine.factual_schema import (
    FACTUAL_ACTOR_RELATION_COUNT,
    FACTUAL_NEXT_CONTEXT_COUNT,
)
from ptcg_rl.engine.feature_vectors import (
    DYNAMIC_EFFECT_FEATURE_SIZE,
    DYNAMIC_EFFECT_MAGNITUDE_INDICES,
)
from ptcg_rl.engine.search_evidence import SEARCH_EVIDENCE_FEATURE_SIZE
from ptcg_rl.model import (
    LEGACY_STATE_ENCODER_MISSING_KEYS,
    ActionEvaluation,
    AgentNetworkConfig,
    AgentPolicyValueNet,
    OptionBatch,
    PolicyEvaluationContext,
    StateBatch,
    build_agent_policy_value_net,
)
from ptcg_rl.model.macro_outcome import MacroOutcomePrediction
from ptcg_rl.profiling import StageTimer, time_stage
from ptcg_rl.rl.macro_credit import (
    MACRO_CONTINUATION_SUMMARY_SIZE,
    MACRO_ENDPOINT_COUNT,
)
from ptcg_rl.rl.model_compatibility import DistributedModelCompatibility
from ptcg_rl.rl.planner_losses import (
    PlannerCurrentDistributions,
    PlannerImitationConfig,
    PlannerImitationLoss,
    PlannerImitationMetrics,
    PlannerReplayBatch,
    planner_current_distributions,
    planner_imitation_loss,
    planner_selected_logprobs_and_entropies,
    root_information_value_loss,
)
from ptcg_rl.rl.recurrent_runtime import PolicyArtifactIdentity
from ptcg_rl.rl.root_information_value_replay import (
    RootInformationValueReplayBatch,
)
from ptcg_rl.rl.transition_distillation import (
    TransitionDistillationConfig,
    TransitionDistillationLosses,
    transition_distillation_losses,
    visited_option_prefix_mask,
)


class PpoConfig(BaseModel):
    """Hydra/Pydantic config for PPO minibatch updates."""

    model_config = ConfigDict(extra="forbid")

    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    kl_anchor_coef: float = 0.05
    engine_teacher_coef: float = 0.0
    factual_effect_coef: float = 0.0
    factual_successor_coef: float = 0.0
    macro_conditional_coef: float = 0.0
    macro_expected_coef: float = 0.0
    candidate_rerank_coef: float = 0.0
    proposal_distillation_coef: float = 0.0
    root_information_value_coef: float = 0.0
    planner_imitation_max_policy_age: int = 0
    planner_target_ratio_clip: float = 4.0
    grad_clip_norm: float = 1.0
    critic_warmup_updates: int = 200
    target_kl_early_stop: float | None = 0.015
    target_kl_early_stop_multiplier: float = 1.5
    kl_stop_mode: Literal["training_batch_delta", "fixed_reference"] = "fixed_reference"
    kl_reference_decisions: int = 2048
    kl_reference_batch_size: int | None = None
    autocast: Literal["bf16", "off"] = "bf16"
    transition_distillation: TransitionDistillationConfig = (
        TransitionDistillationConfig()
    )

    @field_validator(
        "clip_epsilon",
        "value_clip_epsilon",
        "grad_clip_norm",
    )
    @classmethod
    def valid_positive_float(cls, value: float) -> float:
        """Reject non-positive PPO magnitudes."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("value must be finite and positive")
        return value

    @field_validator(
        "value_coef",
        "entropy_coef",
        "kl_anchor_coef",
        "engine_teacher_coef",
        "factual_effect_coef",
        "factual_successor_coef",
        "macro_conditional_coef",
        "macro_expected_coef",
        "candidate_rerank_coef",
        "proposal_distillation_coef",
        "root_information_value_coef",
    )
    @classmethod
    def valid_non_negative_float(cls, value: float) -> float:
        """Reject invalid loss coefficients."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("value must be finite and non-negative")
        return value

    @field_validator("critic_warmup_updates", "planner_imitation_max_policy_age")
    @classmethod
    def valid_non_negative_int(cls, value: int) -> int:
        """Reject negative warmup lengths."""
        if value < 0:
            raise ValueError("value must be non-negative")
        return value

    @field_validator("planner_target_ratio_clip")
    @classmethod
    def valid_planner_target_ratio_clip(cls, value: float) -> float:
        """Reject an invalid conservative planner-target ratio cap."""
        if not math.isfinite(value) or value < 1.0:
            raise ValueError(
                "planner_target_ratio_clip must be finite and at least one"
            )
        return value

    @field_validator("kl_reference_decisions")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject an empty fixed-KL reference."""
        if value <= 0:
            raise ValueError("kl_reference_decisions must be positive")
        return value

    @field_validator("kl_reference_batch_size")
    @classmethod
    def valid_optional_reference_batch_size(cls, value: int | None) -> int | None:
        """Reject an invalid fixed-reference inference batch size."""
        if value is not None and value <= 0:
            raise ValueError("kl_reference_batch_size must be positive")
        return value

    @field_validator("target_kl_early_stop")
    @classmethod
    def valid_optional_positive_float(cls, value: float | None) -> float | None:
        """Reject invalid optional KL thresholds."""
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError("target_kl_early_stop must be finite and positive")
        return value

    @field_validator("target_kl_early_stop_multiplier")
    @classmethod
    def valid_kl_multiplier(cls, value: float) -> float:
        """Reject invalid KL early-stop slack multipliers."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("target_kl_early_stop_multiplier must be positive")
        return value


@dataclass(frozen=True)
class PpoBatch:
    """Tensor-ready PPO minibatch."""

    states: StateBatch
    options: OptionBatch
    actions: Sequence[Sequence[int]]
    old_action_logprobs: Tensor
    sampling_temperatures: Tensor
    old_values: Tensor
    returns: Tensor
    advantages: Tensor
    decks: DeckBatch | None = None
    action_targets: Tensor | None = None
    sample_indices: Tensor | None = None
    old_token_logprobs: Tensor | None = None
    old_prefix_values: Tensor | None = None
    token_returns: Tensor | None = None
    token_advantages: Tensor | None = None
    token_mask: Tensor | None = None
    engine_teacher_actions: Sequence[Sequence[int]] | None = None
    engine_teacher_action_targets: Tensor | None = None
    engine_teacher_confidences: Tensor | None = None
    engine_teacher_weights: Tensor | None = None
    engine_teacher_mask: Tensor | None = None
    engine_teacher_candidate_actions: Sequence[Sequence[Sequence[int]]] | None = None
    engine_teacher_candidate_features: Sequence[Tensor] | None = None
    factual_effect_targets: Tensor | None = None
    factual_actor_relations: Tensor | None = None
    factual_next_contexts: Tensor | None = None
    macro_effect_targets: Tensor | None = None
    macro_endpoints: Tensor | None = None
    macro_continuation_summaries: Tensor | None = None
    macro_mask: Tensor | None = None
    integrity_validated: bool = False
    count_first_rows_present: bool | None = None
    objective_unit_count: int | None = None
    engine_teacher_indices: tuple[int, ...] | None = None
    token_counts: tuple[int, ...] | None = None
    planner_replay: PlannerReplayBatch | None = None
    root_information_value_replay: RootInformationValueReplayBatch | None = None
    public_events: PublicEventBatch | None = None
    sequence_offsets: Tensor | None = None
    sequence_artifacts: tuple[PolicyArtifactIdentity, ...] | None = None


class AnchorStepLogitsCache:
    """Frozen-anchor step logits cached across PPO epochs.

    The anchor model is frozen and minibatch composition is fixed for one
    learner iteration, so its teacher-forced step logits are deterministic
    per minibatch: epochs after the first reuse them instead of re-running
    the anchor forward. Scope one cache to one set of minibatches.
    """

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._step_logits: dict[tuple[int, tuple[str, ...]], tuple[Tensor, ...]] = {}
        self._sample_step_logits: dict[tuple[int, str], tuple[Tensor, ...]] = {}

    def get(
        self,
        key: int,
        *,
        decks: DeckBatch | None = None,
    ) -> tuple[Tensor, ...] | None:
        """Return cached anchor step logits for a minibatch key, if present."""
        return self._step_logits.get((key, _deck_signature_values(decks)))

    def put(
        self,
        key: int,
        step_logits: tuple[Tensor, ...],
        *,
        decks: DeckBatch | None = None,
    ) -> None:
        """Store anchor step logits for a minibatch key."""
        self._step_logits[(key, _deck_signature_values(decks))] = step_logits

    def get_samples(
        self,
        sample_indices: Tensor | None,
        actions: Sequence[Sequence[int]],
        *,
        decks: DeckBatch | None = None,
        token_mask: Tensor | None = None,
        logit_width: int | None = None,
    ) -> tuple[Tensor, ...] | None:
        """Return cached logits for a reshuffled minibatch, if all active rows exist."""
        if sample_indices is None:
            return None
        sample_ids = _sample_index_values(sample_indices)
        if len(sample_ids) != len(actions):
            return None
        signatures = _aligned_deck_signatures(decks, len(actions))
        if signatures is None:
            return None
        if token_mask is None:
            max_steps = max((len(action) for action in actions), default=0) + 1
            active_rows_by_step: tuple[tuple[bool, ...], ...] | None = None
        else:
            if token_mask.ndim != 2 or token_mask.shape[0] != len(actions):
                return None
            cpu_mask = token_mask.detach().to(device="cpu", dtype=torch.bool)
            active_columns = torch.nonzero(cpu_mask.any(dim=0), as_tuple=False)
            max_steps = (
                0
                if int(active_columns.numel()) == 0
                else int(active_columns[-1].item()) + 1
            )
            active_rows_by_step = tuple(
                tuple(bool(value) for value in cpu_mask[:, step_index].tolist())
                for step_index in range(max_steps)
            )
        step_logits: list[Tensor] = []
        for step_index in range(max_steps):
            rows: list[Tensor | None] = []
            prototype: Tensor | None = None
            for row_index, (sample_id, signature, action) in enumerate(
                zip(sample_ids, signatures, actions, strict=True)
            ):
                cached = self._sample_step_logits.get((sample_id, signature))
                row = (
                    None
                    if cached is None or step_index >= len(cached)
                    else cached[step_index]
                )
                if row is not None:
                    if logit_width is not None:
                        row = _resize_cached_step_logits(row, logit_width)
                    prototype = row
                    rows.append(row)
                    continue
                active = (
                    step_index <= len(action)
                    if active_rows_by_step is None
                    else active_rows_by_step[step_index][row_index]
                )
                if active:
                    return None
                rows.append(None)
            if prototype is None:
                return None
            step_logits.append(
                torch.stack(
                    [
                        row if row is not None else torch.zeros_like(prototype)
                        for row in rows
                    ],
                    dim=0,
                )
            )
        return tuple(step_logits)

    def put_samples(
        self,
        sample_indices: Tensor | None,
        step_logits: tuple[Tensor, ...],
        *,
        decks: DeckBatch | None = None,
    ) -> None:
        """Store per-sample anchor logits for future shuffled minibatches."""
        if sample_indices is None or not step_logits:
            return
        sample_ids = _sample_index_values(sample_indices)
        batch_size = int(step_logits[0].shape[0])
        if len(sample_ids) != batch_size:
            return
        signatures = _aligned_deck_signatures(decks, batch_size)
        if signatures is None:
            return
        for row_index, (sample_id, signature) in enumerate(
            zip(sample_ids, signatures, strict=True)
        ):
            self._sample_step_logits[(sample_id, signature)] = tuple(
                logits[row_index].detach() for logits in step_logits
            )


def _resize_cached_step_logits(logits: Tensor, width: int) -> Tensor:
    """Move cached STOP to a new padded-option width without changing valid logits."""
    if logits.ndim != 1 or width <= 0:
        raise ValueError("cached step logits require a positive one-dimensional width")
    option_width = width - 1
    option_logits = logits[:-1]
    if int(option_logits.shape[0]) < option_width:
        option_logits = torch.cat(
            (
                option_logits,
                option_logits.new_full(
                    (option_width - int(option_logits.shape[0]),),
                    float("-inf"),
                ),
            )
        )
    else:
        option_logits = option_logits[:option_width]
    return torch.cat((option_logits, logits[-1:]))


@dataclass(frozen=True)
class PpoLossBreakdown:
    """Loss tensors and scalar diagnostics for one PPO minibatch."""

    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy_loss: Tensor
    kl_anchor_loss: Tensor
    approx_kl: float | None
    approx_kl_k3: float | None
    anchor_kl: float | None
    entropy: float | None
    ratio_mean: float | None
    ratio_p95: float | None
    clip_fraction: float | None
    value_mean: float | None
    critic_warmup: bool
    engine_teacher_loss: Tensor | None = None
    engine_teacher_decisions: int = 0
    factual_effect_loss: Tensor | None = None
    factual_successor_loss: Tensor | None = None
    factual_decisions: int = 0
    macro_conditional_loss: Tensor | None = None
    macro_expected_loss: Tensor | None = None
    macro_roots: int = 0
    candidate_rerank_loss: Tensor | None = None
    proposal_distillation_loss: Tensor | None = None
    planner_decisions: int = 0
    planner_applicable_decisions: int = 0
    planner_metrics: PlannerImitationMetrics | None = None
    root_information_value_loss: Tensor | None = None
    root_information_value_rows: int = 0
    transition_distillation: bool = False
    transition_sequence_kl: Tensor | None = None
    transition_prefix_kl: Tensor | None = None
    transition_count_kl: Tensor | None = None
    transition_root_value_loss: Tensor | None = None
    transition_prefix_value_loss: Tensor | None = None
    transition_action_wdl_loss: Tensor | None = None
    transition_engine_return_loss: Tensor | None = None


@dataclass(frozen=True)
class PpoUpdateResult:
    """Result of one optimizer step."""

    breakdown: PpoLossBreakdown
    grad_norm: float | None
    should_stop: bool
    kl_stop_baseline_k3: float | None = None
    kl_stop_delta_k3: float | None = None
    kl_stop_threshold: float | None = None
    reference_kl: float | None = None
    reference_kl_k3: float | None = None
    reference_initial_kl_k3: float | None = None
    reference_decisions: int = 0
    microbatch_count: int = 1
    sample_count: int = 0
    objective_unit_count: int = 0
    engine_teacher_decisions: int = 0
    factual_decisions: int = 0
    macro_roots: int = 0
    planner_decisions: int = 0
    planner_applicable_decisions: int = 0
    root_information_value_rows: int = 0


@dataclass(frozen=True)
class PpoReferenceKl:
    """KL diagnostics evaluated on one fixed set of decision rows."""

    approx_kl: float
    approx_kl_k3: float
    decisions: int


@dataclass(frozen=True)
class _PlannerModelEvaluation:
    """Current planner branch probabilities and auxiliary objectives."""

    distributions: PlannerCurrentDistributions
    selected_logprobs: Tensor
    entropies: Tensor
    imitation: PlannerImitationLoss


def ppo_loss(
    model: AgentPolicyValueNet,
    batch: PpoBatch,
    config: PpoConfig | None = None,
    *,
    anchor_model: AgentPolicyValueNet | None = None,
    anchor_cache: AnchorStepLogitsCache | None = None,
    anchor_cache_key: int | None = None,
    update_index: int = 0,
    timer: StageTimer | None = None,
    collect_diagnostics: bool = True,
) -> PpoLossBreakdown:
    """Return PPO-clip loss plus value, entropy, and optional anchor-KL terms."""
    cfg = config or PpoConfig()
    if not batch.integrity_validated:
        _validate_batch(batch)
    with time_stage(timer, "learner_policy_forward"), _ppo_autocast_context(batch, cfg):
        evaluation = (
            _evaluate_action_sequences_for_batch(model, batch)
            if _uses_sequence_only_evaluation(batch, cfg)
            else _evaluate_actions_for_batch(model, batch)
        )
        planner_evaluation = _evaluate_planner_for_batch(
            model,
            batch,
            evaluation,
            cfg,
        )
    token_objective = _uses_token_objective(batch)
    if token_objective and planner_evaluation is not None:
        (
            current_logprobs,
            current_entropies,
            current_values,
            old_logprobs,
            old_values,
            returns,
            advantages,
            evaluation_token_mask,
        ) = _hybrid_planner_token_objective_tensors(
            evaluation,
            planner_evaluation,
            batch,
        )
    elif token_objective:
        (
            current_logprobs,
            current_entropies,
            current_values,
            old_logprobs,
            old_values,
            returns,
            advantages,
            evaluation_token_mask,
        ) = _active_token_objective_tensors(evaluation, batch)
    else:
        current_logprobs = evaluation.action_logprobs
        current_entropies = evaluation.entropies
        if planner_evaluation is not None:
            planner_indices = _planner_decision_indices(
                batch,
                device=current_logprobs.device,
            )
            current_logprobs = current_logprobs.index_copy(
                0,
                planner_indices,
                planner_evaluation.selected_logprobs.to(
                    device=current_logprobs.device,
                    dtype=current_logprobs.dtype,
                ),
            )
            current_entropies = current_entropies.index_copy(
                0,
                planner_indices,
                planner_evaluation.entropies.to(
                    device=current_entropies.device,
                    dtype=current_entropies.dtype,
                ),
            )
        current_values = evaluation.values
        old_logprobs = batch.old_action_logprobs.to(
            device=current_logprobs.device,
            dtype=current_logprobs.dtype,
        )
        advantages = batch.advantages.to(
            device=current_logprobs.device,
            dtype=current_logprobs.dtype,
        )
        old_values = batch.old_values.to(
            device=current_values.device,
            dtype=current_values.dtype,
        )
        returns = batch.returns.to(
            device=current_values.device,
            dtype=current_values.dtype,
        )
        evaluation_token_mask = None

    log_ratio = current_logprobs - old_logprobs
    ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
    log_ratio_for_kl = log_ratio.detach().float()
    approx_kl_k1 = (
        _tensor_float((-log_ratio_for_kl).mean()) if collect_diagnostics else None
    )
    collect_training_batch_kl = bool(
        collect_diagnostics or cfg.target_kl_early_stop is not None
    )
    approx_kl_k3 = (
        _tensor_float(
            (
                torch.exp(log_ratio_for_kl.clamp(min=-20.0, max=20.0))
                - 1.0
                - log_ratio_for_kl
            ).mean()
        )
        if collect_training_batch_kl
        else None
    )
    unclipped = ratio * advantages
    clipped_ratio = ratio.clamp(1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon)
    clipped = clipped_ratio * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    value_clipped = old_values + (current_values - old_values).clamp(
        min=-cfg.value_clip_epsilon,
        max=cfg.value_clip_epsilon,
    )
    value_loss = torch.maximum(
        (current_values - returns).pow(2),
        (value_clipped - returns).pow(2),
    ).mean()
    entropy_loss = -current_entropies.mean()

    critic_warmup = update_index < cfg.critic_warmup_updates
    if critic_warmup:
        policy_loss = policy_loss * 0.0
        entropy_loss = entropy_loss * 0.0
        kl_anchor_loss = current_values.sum() * 0.0
        anchor_kl = 0.0 if collect_diagnostics else None
    elif anchor_model is None or cfg.kl_anchor_coef == 0.0:
        kl_anchor_loss = current_values.sum() * 0.0
        anchor_kl = 0.0 if collect_diagnostics else None
    else:
        anchor_step_logits = _anchor_step_logits(
            anchor_model,
            batch,
            cfg,
            anchor_cache=anchor_cache,
            anchor_cache_key=anchor_cache_key,
            timer=timer,
        )
        if token_objective:
            if evaluation_token_mask is None:
                raise RuntimeError("token PPO evaluation mask is unavailable")
            kl_anchor_loss = token_kl_from_logits(
                evaluation.step_logits,
                anchor_step_logits,
                evaluation_token_mask,
                active_tokens=(
                    int(evaluation_token_mask.sum().item())
                    if planner_evaluation is not None
                    else _objective_unit_count(batch)
                ),
            )
        else:
            kl_anchor_loss = sequence_kl_from_logits(
                evaluation.step_logits,
                anchor_step_logits,
                batch.actions,
                action_targets=batch.action_targets,
            )
        anchor_kl = _tensor_float(kl_anchor_loss) if collect_diagnostics else None

    engine_teacher_decisions = _engine_teacher_decision_count(batch)
    if critic_warmup or cfg.engine_teacher_coef == 0.0 or engine_teacher_decisions == 0:
        engine_teacher_loss = current_values.sum() * 0.0
    else:
        with (
            time_stage(timer, "learner_engine_teacher_forward"),
            _ppo_autocast_context(batch, cfg),
        ):
            engine_teacher_loss = _engine_teacher_sequence_loss(
                model,
                batch,
                evaluation,
            )

    factual_decisions = _factual_decision_count(batch)
    factual_objective_enabled = any(
        coefficient > 0.0
        for coefficient in (
            cfg.factual_effect_coef,
            cfg.factual_successor_coef,
        )
    )
    if factual_objective_enabled and factual_decisions == 0:
        raise ValueError("factual PPO coefficients require dense factual targets")
    if factual_decisions == 0:
        factual_effect_loss = current_values.sum() * 0.0
        factual_successor_loss = current_values.sum() * 0.0
    else:
        (
            factual_effect_loss,
            factual_successor_loss,
        ) = _factual_action_losses(evaluation, batch)

    macro_roots = _macro_root_count(batch)
    macro_objective_enabled = (
        cfg.macro_conditional_coef > 0.0 or cfg.macro_expected_coef > 0.0
    )
    if macro_objective_enabled and batch.macro_mask is None:
        raise ValueError("macro PPO coefficients require schema-10 targets")
    if macro_roots == 0:
        macro_conditional_loss = current_values.sum() * 0.0
        macro_expected_loss = current_values.sum() * 0.0
    else:
        macro_conditional_loss, macro_expected_loss = _macro_action_losses(
            model,
            evaluation,
            batch,
        )

    if planner_evaluation is None:
        candidate_rerank_loss = current_values.sum() * 0.0
        proposal_distillation_loss = current_values.sum() * 0.0
        planner_decisions = 0
        planner_applicable_decisions = 0
        planner_metrics = None
    else:
        candidate_rerank_loss = planner_evaluation.imitation.candidate_rerank_loss
        proposal_distillation_loss = planner_evaluation.imitation.proposal_kl_loss
        planner_metrics = planner_evaluation.imitation.metrics
        planner_decisions = planner_metrics.planner_decisions
        planner_applicable_decisions = planner_metrics.applicable_decisions
        if critic_warmup:
            candidate_rerank_loss = candidate_rerank_loss * 0.0
            proposal_distillation_loss = proposal_distillation_loss * 0.0

    root_replay = batch.root_information_value_replay
    root_information_value_rows = 0 if root_replay is None else root_replay.row_count
    if root_replay is None or cfg.root_information_value_coef == 0.0:
        root_value_loss = current_values.sum() * 0.0
    else:
        with (
            time_stage(timer, "learner_root_information_value_forward"),
            _ppo_autocast_context(batch, cfg),
        ):
            root_value_loss = _root_information_value_replay_loss(
                model,
                root_replay,
                batch_size=len(batch.actions),
            )

    loss = (
        policy_loss
        + cfg.value_coef * value_loss
        + cfg.entropy_coef * entropy_loss
        + cfg.kl_anchor_coef * kl_anchor_loss
        + cfg.engine_teacher_coef * engine_teacher_loss
        + cfg.factual_effect_coef * factual_effect_loss
        + cfg.factual_successor_coef * factual_successor_loss
        + cfg.macro_conditional_coef * macro_conditional_loss
        + cfg.macro_expected_coef * macro_expected_loss
        + cfg.candidate_rerank_coef * candidate_rerank_loss
        + cfg.proposal_distillation_coef * proposal_distillation_loss
        + cfg.root_information_value_coef * root_value_loss
    )
    return PpoLossBreakdown(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        kl_anchor_loss=kl_anchor_loss,
        approx_kl=approx_kl_k1,
        approx_kl_k3=approx_kl_k3,
        anchor_kl=anchor_kl,
        entropy=_tensor_float(current_entropies.mean())
        if collect_diagnostics
        else None,
        ratio_mean=_tensor_float(ratio.mean()) if collect_diagnostics else None,
        ratio_p95=(
            _tensor_float(torch.quantile(ratio.detach().float(), 0.95))
            if collect_diagnostics
            else None
        ),
        clip_fraction=_tensor_float(
            (ratio.detach() - 1.0).abs().gt(cfg.clip_epsilon).float().mean()
        )
        if collect_diagnostics
        else None,
        value_mean=_tensor_float(current_values.mean())
        if collect_diagnostics
        else None,
        critic_warmup=critic_warmup,
        engine_teacher_loss=engine_teacher_loss,
        engine_teacher_decisions=engine_teacher_decisions,
        factual_effect_loss=factual_effect_loss,
        factual_successor_loss=factual_successor_loss,
        factual_decisions=factual_decisions,
        macro_conditional_loss=macro_conditional_loss,
        macro_expected_loss=macro_expected_loss,
        macro_roots=macro_roots,
        candidate_rerank_loss=candidate_rerank_loss,
        proposal_distillation_loss=proposal_distillation_loss,
        planner_decisions=planner_decisions,
        planner_applicable_decisions=planner_applicable_decisions,
        planner_metrics=planner_metrics,
        root_information_value_loss=root_value_loss,
        root_information_value_rows=root_information_value_rows,
    )


def _anchor_step_logits(
    anchor_model: AgentPolicyValueNet,
    batch: PpoBatch,
    config: PpoConfig,
    *,
    anchor_cache: AnchorStepLogitsCache | None,
    anchor_cache_key: int | None,
    timer: StageTimer | None,
) -> tuple[Tensor, ...]:
    if anchor_cache is not None and anchor_cache_key is not None:
        cached = anchor_cache.get(anchor_cache_key, decks=batch.decks)
        if cached is not None:
            return cached
    if anchor_cache is not None:
        cached_samples = anchor_cache.get_samples(
            batch.sample_indices,
            batch.actions,
            decks=batch.decks,
            token_mask=_decode_token_mask_for_cache(batch),
            logit_width=int(batch.options.valid_options.shape[1]) + 1,
        )
        if cached_samples is not None:
            if anchor_cache_key is not None:
                anchor_cache.put(
                    anchor_cache_key,
                    cached_samples,
                    decks=batch.decks,
                )
            return cached_samples
    with (
        torch.no_grad(),
        time_stage(timer, "learner_anchor_forward"),
        _ppo_autocast_context(batch, config),
    ):
        step_logits = anchor_model.evaluate_action_step_logits(
            batch.states,
            batch.options,
            batch.actions,
            decks=batch.decks,
            action_targets=batch.action_targets,
            temperature=batch.sampling_temperatures,
            validate_temperature=not batch.integrity_validated,
        )
    if anchor_cache is not None and anchor_cache_key is not None:
        anchor_cache.put(anchor_cache_key, step_logits, decks=batch.decks)
    if anchor_cache is not None:
        anchor_cache.put_samples(
            batch.sample_indices,
            step_logits,
            decks=batch.decks,
        )
    return step_logits


def ppo_update_step(
    model: AgentPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: PpoBatch,
    config: PpoConfig | None = None,
    *,
    anchor_model: AgentPolicyValueNet | None = None,
    anchor_cache: AnchorStepLogitsCache | None = None,
    anchor_cache_key: int | None = None,
    update_index: int = 0,
    distillation_update_index: int | None = None,
    timer: StageTimer | None = None,
    kl_stop_baseline_k3: float | None = None,
    collect_diagnostics: bool = True,
) -> PpoUpdateResult:
    """Run one legacy single-microbatch optimizer step.

    Target-KL stopping is intentionally not decided from this training batch.
    The learner attaches the independently evaluated fixed-reference KL after
    the optimizer step. ``kl_stop_baseline_k3`` remains accepted so older
    callers do not fail during migration, but it is logging-only.
    """
    return ppo_accumulated_update_step(
        model,
        optimizer,
        (batch,),
        config,
        anchor_model=anchor_model,
        anchor_cache=anchor_cache,
        anchor_cache_keys=(anchor_cache_key,),
        update_index=update_index,
        distillation_update_index=distillation_update_index,
        timer=timer,
        kl_stop_baseline_k3=kl_stop_baseline_k3,
        collect_diagnostics=collect_diagnostics,
    )


def ppo_accumulated_update_step(
    model: AgentPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batches: Sequence[PpoBatch],
    config: PpoConfig | None = None,
    *,
    anchor_model: AgentPolicyValueNet | None = None,
    anchor_cache: AnchorStepLogitsCache | None = None,
    anchor_cache_keys: Sequence[int | None] | None = None,
    update_index: int = 0,
    distillation_update_index: int | None = None,
    timer: StageTimer | None = None,
    kl_stop_baseline_k3: float | None = None,
    collect_diagnostics: bool = True,
) -> PpoUpdateResult:
    """Run one effective optimizer update over unit-weighted microbatches.

    Decision-level batches use rows as units. Token-level batches use active
    decode tokens so padding and uneven sequence lengths cannot change the
    effective objective when a batch is split into microbatches.
    """
    if not batches:
        raise ValueError("at least one PPO microbatch is required")
    if distillation_update_index is None:
        distillation_update_index = update_index
    if distillation_update_index < 0:
        raise ValueError("distillation_update_index must be non-negative")
    cfg = config or PpoConfig()
    if cfg.transition_distillation.active(distillation_update_index):
        return _transition_distillation_update_step(
            model,
            optimizer,
            batches,
            cfg,
            anchor_model=anchor_model,
            update_index=distillation_update_index,
            timer=timer,
            collect_diagnostics=collect_diagnostics,
        )
    row_counts = tuple(len(batch.actions) for batch in batches)
    if any(count <= 0 for count in row_counts):
        raise ValueError("PPO microbatches must contain at least one row")
    total_rows = sum(row_counts)
    token_modes = tuple(_uses_token_objective(batch) for batch in batches)
    if len(set(token_modes)) != 1:
        raise ValueError("PPO microbatches must use one credit unit")
    objective_counts = tuple(_objective_unit_count(batch) for batch in batches)
    total_objective_units = sum(objective_counts)
    anchor_counts = tuple(_anchor_kl_unit_count(batch) for batch in batches)
    total_anchor_units = sum(anchor_counts)
    teacher_counts = tuple(_engine_teacher_decision_count(batch) for batch in batches)
    total_teacher_decisions = sum(teacher_counts)
    factual_counts = tuple(_factual_decision_count(batch) for batch in batches)
    total_factual_decisions = sum(factual_counts)
    macro_counts = tuple(_macro_root_count(batch) for batch in batches)
    total_macro_roots = sum(macro_counts)
    planner_counts = tuple(_planner_decision_counts(batch, cfg) for batch in batches)
    total_planner_decisions = sum(counts[0] for counts in planner_counts)
    total_planner_applicable = sum(counts[1] for counts in planner_counts)
    root_value_counts = tuple(
        0
        if batch.root_information_value_replay is None
        else batch.root_information_value_replay.row_count
        for batch in batches
    )
    total_root_value_rows = sum(root_value_counts)
    cache_keys = _normalized_anchor_cache_keys(anchor_cache_keys, len(batches))
    breakdowns: list[PpoLossBreakdown] = []
    optimizer.zero_grad(set_to_none=True)
    try:
        for (
            batch,
            objective_count,
            anchor_count,
            teacher_count,
            factual_count,
            macro_count,
            planner_count,
            root_value_count,
            cache_key,
        ) in zip(
            batches,
            objective_counts,
            anchor_counts,
            teacher_counts,
            factual_counts,
            macro_counts,
            planner_counts,
            root_value_counts,
            cache_keys,
            strict=True,
        ):
            breakdown = ppo_loss(
                model,
                batch,
                cfg,
                anchor_model=anchor_model,
                anchor_cache=anchor_cache,
                anchor_cache_key=cache_key,
                update_index=update_index,
                timer=timer,
                collect_diagnostics=collect_diagnostics,
            )
            objective_weight = objective_count / total_objective_units
            anchor_loss = _cast_tensor(breakdown.kl_anchor_loss)
            teacher_loss = _required_engine_teacher_loss(breakdown)
            factual_loss = _weighted_factual_loss(breakdown, cfg)
            macro_loss = _weighted_macro_loss(breakdown, cfg)
            planner_loss = _weighted_planner_loss(breakdown, cfg)
            root_value_loss = _required_optional_loss(
                breakdown,
                "root_information_value_loss",
            )
            base_loss = (
                breakdown.loss
                - cfg.engine_teacher_coef * teacher_loss
                - factual_loss
                - macro_loss
                - planner_loss
                - cfg.root_information_value_coef * root_value_loss
                - cfg.kl_anchor_coef * anchor_loss
            )
            weighted_loss = base_loss * objective_weight
            if total_anchor_units > 0 and anchor_count > 0:
                weighted_loss = weighted_loss + (
                    cfg.kl_anchor_coef
                    * anchor_loss
                    * (anchor_count / total_anchor_units)
                )
            if total_teacher_decisions > 0 and teacher_count > 0:
                weighted_loss = weighted_loss + (
                    cfg.engine_teacher_coef
                    * teacher_loss
                    * (teacher_count / total_teacher_decisions)
                )
            if total_factual_decisions > 0 and factual_count > 0:
                weighted_loss = weighted_loss + (
                    factual_loss * (factual_count / total_factual_decisions)
                )
            if total_macro_roots > 0 and macro_count > 0:
                weighted_loss = weighted_loss + (
                    macro_loss * (macro_count / total_macro_roots)
                )
            planner_applicable = planner_count[1]
            if total_planner_applicable > 0 and planner_applicable > 0:
                weighted_loss = weighted_loss + (
                    planner_loss * (planner_applicable / total_planner_applicable)
                )
            if total_root_value_rows > 0 and root_value_count > 0:
                weighted_loss = weighted_loss + (
                    cfg.root_information_value_coef
                    * root_value_loss
                    * (root_value_count / total_root_value_rows)
                )
            with time_stage(timer, "learner_backward"):
                torch.autograd.backward(weighted_loss)
            breakdowns.append(
                _detached_loss_breakdown(
                    breakdown,
                    copy_to_cpu=collect_diagnostics,
                )
            )
        _raise_for_nonfinite_losses(breakdowns)
        if breakdowns[0].critic_warmup:
            _clear_non_value_head_gradients(model)
        with time_stage(timer, "learner_grad_clip"):
            grad_norm = _clip_optimizer_parameter_groups(
                optimizer,
                max_norm=cfg.grad_clip_norm,
            )
        with time_stage(timer, "learner_optimizer"):
            optimizer.step()
    except (FloatingPointError, RuntimeError):
        optimizer.zero_grad(set_to_none=True)
        raise

    breakdown = _unit_weighted_breakdowns(
        breakdowns,
        objective_counts,
        anchor_counts,
        teacher_counts,
        factual_counts,
        macro_counts,
        planner_counts,
        root_value_counts,
        engine_teacher_coef=cfg.engine_teacher_coef,
        kl_anchor_coef=cfg.kl_anchor_coef,
        factual_effect_coef=cfg.factual_effect_coef,
        factual_successor_coef=cfg.factual_successor_coef,
        macro_conditional_coef=cfg.macro_conditional_coef,
        macro_expected_coef=cfg.macro_expected_coef,
        candidate_rerank_coef=cfg.candidate_rerank_coef,
        proposal_distillation_coef=cfg.proposal_distillation_coef,
        root_information_value_coef=cfg.root_information_value_coef,
        copy_to_cpu=collect_diagnostics,
    )
    kl_stop_delta = (
        None
        if kl_stop_baseline_k3 is None or breakdown.approx_kl_k3 is None
        else breakdown.approx_kl_k3 - kl_stop_baseline_k3
    )
    return PpoUpdateResult(
        breakdown=breakdown,
        grad_norm=_tensor_float(grad_norm) if collect_diagnostics else None,
        should_stop=False,
        kl_stop_baseline_k3=kl_stop_baseline_k3,
        kl_stop_delta_k3=kl_stop_delta,
        kl_stop_threshold=_target_kl_stop_threshold(cfg),
        microbatch_count=len(batches),
        sample_count=total_rows,
        objective_unit_count=total_objective_units,
        engine_teacher_decisions=total_teacher_decisions,
        factual_decisions=total_factual_decisions,
        macro_roots=total_macro_roots,
        planner_decisions=total_planner_decisions,
        planner_applicable_decisions=total_planner_applicable,
        root_information_value_rows=total_root_value_rows,
    )


def _transition_distillation_update_step(
    model: AgentPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batches: Sequence[PpoBatch],
    config: PpoConfig,
    *,
    anchor_model: AgentPolicyValueNet | None,
    update_index: int,
    timer: StageTimer | None,
    collect_diagnostics: bool,
) -> PpoUpdateResult:
    """Run one fixed-budget teacher update without PPO or automatic stopping."""
    if anchor_model is None:
        raise ValueError("transition distillation requires a frozen anchor model")
    if not batches:
        raise ValueError("transition distillation requires at least one microbatch")
    if not config.transition_distillation.active(update_index):
        raise ValueError("optimizer index is outside the distillation budget")
    row_counts = tuple(len(batch.actions) for batch in batches)
    if any(count <= 0 for count in row_counts):
        raise ValueError("transition microbatches must contain rows")
    count_masks: list[Tensor] = []
    option_prefix_masks: list[Tensor] = []
    for batch in batches:
        count_mask = model.policy_head.count_first_rows(batch.options)
        teacher_count_mask = anchor_model.policy_head.count_first_rows(batch.options)
        if not torch.equal(
            count_mask.detach().to(device="cpu"),
            teacher_count_mask.detach().to(device="cpu"),
        ):
            raise RuntimeError("transition teacher and student count routes differ")
        count_masks.append(count_mask)
        option_prefix_masks.append(
            _transition_option_prefix_mask(batch, count_mask=count_mask)
        )
    prefix_counts = tuple(
        int(option_mask.sum().item()) + int(count_mask.sum().item())
        for option_mask, count_mask in zip(
            option_prefix_masks,
            count_masks,
            strict=True,
        )
    )
    count_counts = tuple(int(mask.sum().item()) for mask in count_masks)
    value_prefix_counts = tuple(
        _transition_value_prefix_count(batch) for batch in batches
    )
    total_rows = sum(row_counts)
    total_prefixes = sum(prefix_counts)
    total_count_rows = sum(count_counts)
    total_value_prefixes = sum(value_prefix_counts)
    transition = config.transition_distillation
    losses: list[TransitionDistillationLosses] = []
    optimizer.zero_grad(set_to_none=True)
    try:
        for (
            batch,
            row_count,
            prefix_count,
            count_count,
            value_prefix_count,
            option_prefix_mask,
            count_mask,
        ) in zip(
            batches,
            row_counts,
            prefix_counts,
            count_counts,
            value_prefix_counts,
            option_prefix_masks,
            count_masks,
            strict=True,
        ):
            with (
                time_stage(timer, "learner_transition_student_forward"),
                _ppo_autocast_context(batch, config),
            ):
                student = _evaluate_actions_for_batch(model, batch)
                student_action_wdl = _transition_action_wdl_logits(
                    model,
                    batch,
                    student,
                    enabled=transition.action_wdl_coef > 0.0,
                )
            with (
                torch.no_grad(),
                time_stage(timer, "learner_transition_teacher_forward"),
                _ppo_autocast_context(batch, config),
            ):
                teacher = _evaluate_actions_for_batch(anchor_model, batch)
                teacher_action_wdl = _transition_action_wdl_logits(
                    anchor_model,
                    batch,
                    teacher,
                    enabled=transition.action_wdl_coef > 0.0,
                )
            terms = transition_distillation_losses(
                student_step_logits=student.step_logits,
                teacher_step_logits=teacher.step_logits,
                student_count_logits=student.first_logits,
                teacher_count_logits=teacher.first_logits,
                option_prefix_mask=option_prefix_mask,
                count_mask=count_mask,
                student_values=student.values,
                teacher_values=teacher.values,
                student_prefix_values=student.prefix_values,
                teacher_prefix_values=teacher.prefix_values,
                token_mask=student.token_mask,
                actions=batch.actions,
                returns=batch.returns,
                student_action_wdl_logits=student_action_wdl,
                teacher_action_wdl_logits=teacher_action_wdl,
            )
            if (
                terms.rows != row_count
                or terms.prefixes != prefix_count
                or terms.count_rows != count_count
                or terms.value_prefixes != value_prefix_count
            ):
                raise RuntimeError(
                    "transition target inventory changed during forward: "
                    f"expected rows/prefixes/count/value={row_count}/"
                    f"{prefix_count}/{count_count}/{value_prefix_count}, got "
                    f"{terms.rows}/{terms.prefixes}/{terms.count_rows}/"
                    f"{terms.value_prefixes}"
                )
            row_weight = row_count / total_rows
            prefix_weight = prefix_count / total_prefixes if total_prefixes > 0 else 0.0
            value_prefix_weight = value_prefix_count / total_value_prefixes
            weighted_loss = (
                transition.sequence_kl_coef * terms.sequence_kl * row_weight
                + transition.prefix_kl_coef * terms.prefix_kl * prefix_weight
                + transition.root_value_coef * terms.root_value * row_weight
                + transition.prefix_value_coef
                * terms.prefix_value
                * value_prefix_weight
                + transition.action_wdl_coef * terms.action_wdl * row_weight
                + transition.engine_return_coef * terms.engine_return * row_weight
            )
            if not bool(torch.isfinite(weighted_loss.detach()).all().item()):
                raise FloatingPointError("non-finite transition distillation loss")
            with time_stage(timer, "learner_backward"):
                torch.autograd.backward(weighted_loss)
            losses.append(terms)
        with time_stage(timer, "learner_grad_clip"):
            grad_norm = _clip_optimizer_parameter_groups(
                optimizer,
                max_norm=config.grad_clip_norm,
            )
        with time_stage(timer, "learner_optimizer"):
            optimizer.step()
    except (FloatingPointError, RuntimeError, ValueError):
        optimizer.zero_grad(set_to_none=True)
        raise

    row_weights = tuple(count / total_rows for count in row_counts)
    prefix_weights = (
        tuple(count / total_prefixes for count in prefix_counts)
        if total_prefixes > 0
        else (0.0,) * len(prefix_counts)
    )
    count_weights = (
        tuple(count / total_count_rows for count in count_counts)
        if total_count_rows > 0
        else (0.0,) * len(count_counts)
    )
    value_prefix_weights = tuple(
        count / total_value_prefixes for count in value_prefix_counts
    )

    def combined(name: str, weights: Sequence[float]) -> Tensor:
        values = tuple(getattr(item, name).detach().float() for item in losses)
        result = sum(
            (value * weight for value, weight in zip(values, weights, strict=True)),
            start=values[0].new_zeros(()),
        )
        return _detach_loss_scalar(result, copy_to_cpu=collect_diagnostics)

    sequence_kl = combined("sequence_kl", row_weights)
    prefix_kl = combined("prefix_kl", prefix_weights)
    count_kl = combined("count_kl", count_weights)
    root_value = combined("root_value", row_weights)
    prefix_value = combined("prefix_value", value_prefix_weights)
    action_wdl = combined("action_wdl", row_weights)
    engine_return = combined("engine_return", row_weights)
    policy_loss = (
        transition.sequence_kl_coef * sequence_kl
        + transition.prefix_kl_coef * prefix_kl
    )
    value_loss = (
        transition.root_value_coef * root_value
        + transition.prefix_value_coef * prefix_value
        + transition.action_wdl_coef * action_wdl
        + transition.engine_return_coef * engine_return
    )
    zero = policy_loss.new_zeros(())
    breakdown = PpoLossBreakdown(
        loss=policy_loss + value_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=zero,
        kl_anchor_loss=prefix_kl,
        approx_kl=None,
        approx_kl_k3=None,
        anchor_kl=_tensor_float(prefix_kl) if collect_diagnostics else None,
        entropy=None,
        ratio_mean=None,
        ratio_p95=None,
        clip_fraction=None,
        value_mean=None,
        critic_warmup=False,
        transition_distillation=True,
        transition_sequence_kl=sequence_kl,
        transition_prefix_kl=prefix_kl,
        transition_count_kl=count_kl,
        transition_root_value_loss=root_value,
        transition_prefix_value_loss=prefix_value,
        transition_action_wdl_loss=action_wdl,
        transition_engine_return_loss=engine_return,
    )
    return PpoUpdateResult(
        breakdown=breakdown,
        grad_norm=_tensor_float(grad_norm) if collect_diagnostics else None,
        should_stop=False,
        kl_stop_threshold=None,
        microbatch_count=len(batches),
        sample_count=total_rows,
        objective_unit_count=total_prefixes,
    )


def _transition_option_prefix_mask(
    batch: PpoBatch,
    *,
    count_mask: Tensor,
) -> Tensor:
    """Return real option/STOP decisions, excluding forced max termination."""
    return visited_option_prefix_mask(
        batch.actions,
        max_counts=batch.options.max_counts,
        count_mask=count_mask,
    )


def _transition_value_prefix_count(batch: PpoBatch) -> int:
    max_counts = tuple(
        int(value) for value in batch.options.max_counts.detach().cpu().tolist()
    )
    if len(max_counts) != len(batch.actions):
        raise ValueError("transition max-count rows do not align with actions")
    count = sum(
        len(action) + int(len(action) < max_count)
        for action, max_count in zip(batch.actions, max_counts, strict=True)
    )
    if count <= 0:
        raise ValueError("transition distillation requires prefix values")
    return count


def _transition_action_wdl_logits(
    model: AgentPolicyValueNet,
    batch: PpoBatch,
    evaluation: ActionEvaluation,
    *,
    enabled: bool,
) -> Tensor | None:
    if not enabled:
        return None
    context = evaluation.policy_context
    if context is None:
        raise RuntimeError("transition action-WDL target has no policy context")
    candidate_actions = tuple((tuple(action),) for action in batch.actions)
    prediction = model.evaluate_action_values_from_context(
        context,
        batch.options,
        candidate_actions,
        decks=batch.decks,
        policy_temperature=batch.sampling_temperatures,
        validate_candidate_actions=not batch.integrity_validated,
    )
    return prediction.logits


def _clip_optimizer_parameter_groups(
    optimizer: torch.optim.Optimizer,
    *,
    max_norm: float,
) -> Tensor:
    """Clip disjoint optimizer groups independently and report total pre-clip norm."""
    gradients: list[Tensor] = []
    group_ranges: list[tuple[int, int]] = []
    for group in optimizer.param_groups:
        group_gradients = [
            parameter.grad
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if not group_gradients:
            continue
        start = len(gradients)
        gradients.extend(group_gradients)
        group_ranges.append((start, len(gradients)))
    if not gradients:
        return torch.zeros(())

    parameter_norms = torch._foreach_norm(gradients, 2.0)
    stacked_norms = torch.stack(parameter_norms)
    group_norms = [
        torch.linalg.vector_norm(stacked_norms[start:stop], ord=2)
        for start, stop in group_ranges
    ]
    finite = torch.stack([torch.isfinite(norm).all() for norm in group_norms])
    if not bool(finite.all().item()):
        bad_groups = []
        for group_index, group in enumerate(optimizer.param_groups):
            gradients_with_values = [
                parameter.grad
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            bad_count = sum(
                not bool(torch.isfinite(gradient).all().item())
                for gradient in gradients_with_values
            )
            if bad_count:
                bad_groups.append(
                    (
                        str(group.get("name", group_index)),
                        bad_count,
                        len(gradients_with_values),
                    )
                )
        raise FloatingPointError(
            f"non-finite PPO gradient norm; bad groups (name, bad, total)={bad_groups}"
        )

    clip_scales: list[Tensor] = []
    for norm, (start, stop) in zip(group_norms, group_ranges, strict=True):
        scale = torch.clamp(max_norm / (norm + 1.0e-6), max=1.0)
        clip_scales.extend(
            scale.to(device=gradient.device, dtype=gradient.dtype)
            for gradient in gradients[start:stop]
        )
    torch._foreach_mul_(gradients, clip_scales)

    device = group_norms[0].device
    norms = torch.stack(
        [norm.detach().to(device=device, dtype=torch.float32) for norm in group_norms]
    )
    return norms.square().sum().sqrt()


def reference_kl_from_logprobs(
    current_action_logprobs: Sequence[Tensor],
    reference_action_logprobs: Sequence[Tensor],
) -> PpoReferenceKl:
    """Return fixed-row K1/K3 KL estimates from aligned action log-probs."""
    if len(current_action_logprobs) != len(reference_action_logprobs):
        raise ValueError("current and reference log-prob chunks must align")
    if not current_action_logprobs:
        raise ValueError("fixed KL reference must contain at least one chunk")
    current_rows: list[Tensor] = []
    reference_rows: list[Tensor] = []
    for current, reference in zip(
        current_action_logprobs,
        reference_action_logprobs,
        strict=True,
    ):
        if current.ndim != 1 or reference.ndim != 1 or current.shape != reference.shape:
            raise ValueError("fixed KL log-prob chunks must have equal 1-D shapes")
        current_rows.append(current.detach().float().cpu())
        reference_rows.append(reference.detach().float().cpu())
    current = torch.cat(current_rows)
    reference = torch.cat(reference_rows)
    if current.numel() == 0:
        raise ValueError("fixed KL reference must contain at least one row")
    log_ratio = current - reference
    approx_kl = _tensor_float((-log_ratio).mean())
    approx_kl_k3 = _tensor_float(
        (torch.exp(log_ratio.clamp(min=-20.0, max=20.0)) - 1.0 - log_ratio).mean()
    )
    if not math.isfinite(approx_kl) or not math.isfinite(approx_kl_k3):
        raise FloatingPointError("non-finite fixed-reference KL")
    return PpoReferenceKl(
        approx_kl=approx_kl,
        approx_kl_k3=approx_kl_k3,
        decisions=int(current.numel()),
    )


def with_reference_kl(
    update: PpoUpdateResult,
    reference_kl: PpoReferenceKl,
    *,
    initial_reference_kl_k3: float,
    config: PpoConfig,
) -> PpoUpdateResult:
    """Attach fixed-reference diagnostics and make the sole KL-stop decision."""
    threshold = _target_kl_stop_threshold(config)
    return replace(
        update,
        should_stop=(threshold is not None and reference_kl.approx_kl_k3 > threshold),
        kl_stop_baseline_k3=initial_reference_kl_k3,
        kl_stop_delta_k3=reference_kl.approx_kl_k3 - initial_reference_kl_k3,
        kl_stop_threshold=threshold,
        reference_kl=reference_kl.approx_kl,
        reference_kl_k3=reference_kl.approx_kl_k3,
        reference_initial_kl_k3=initial_reference_kl_k3,
        reference_decisions=reference_kl.decisions,
    )


def with_training_batch_kl(
    update: PpoUpdateResult,
    *,
    baseline_k3: float | None,
    config: PpoConfig,
) -> PpoUpdateResult:
    """Attach the legacy cross-minibatch KL delta used only by control arms."""
    current_k3 = update.breakdown.approx_kl_k3
    if current_k3 is None:
        raise RuntimeError("training-batch KL was not collected")
    effective_baseline = current_k3 if baseline_k3 is None else baseline_k3
    delta = current_k3 - effective_baseline
    threshold = _target_kl_stop_threshold(config)
    return replace(
        update,
        should_stop=(
            baseline_k3 is not None and threshold is not None and delta > threshold
        ),
        kl_stop_baseline_k3=effective_baseline,
        kl_stop_delta_k3=delta,
        kl_stop_threshold=threshold,
    )


def evaluate_reference_action_logprobs(
    model: AgentPolicyValueNet,
    batch: PpoBatch,
    config: PpoConfig | None = None,
) -> Tensor:
    """Evaluate aligned log-prob units for a fixed-KL reference.

    Legacy decision batches return one summed action log-prob per row. Token
    batches return the flattened active decode-token log-probs.
    """
    cfg = config or PpoConfig()
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad(), _ppo_autocast_context(batch, cfg):
            evaluation = _evaluate_actions_for_batch(model, batch)
            planner_evaluation = _evaluate_planner_for_batch(
                model,
                batch,
                evaluation,
                cfg,
            )
    finally:
        model.train(was_training)
    if _uses_token_objective(batch) and planner_evaluation is not None:
        action_logprobs = (
            _hybrid_planner_token_objective_tensors(
                evaluation,
                planner_evaluation,
                batch,
            )[0]
            .detach()
            .float()
            .cpu()
        )
    elif _uses_token_objective(batch):
        token_logprobs = evaluation.token_logprobs
        token_mask = evaluation.token_mask
        if token_logprobs is None or token_mask is None:
            raise RuntimeError("model did not return token-level PPO evidence")
        _validate_evaluation_token_mask(token_mask, batch)
        action_logprobs = (
            token_logprobs.masked_select(token_mask).detach().float().cpu()
        )
    else:
        decision_logprobs = evaluation.action_logprobs
        if planner_evaluation is not None:
            indices = _planner_decision_indices(
                batch,
                device=decision_logprobs.device,
            )
            decision_logprobs = decision_logprobs.index_copy(
                0,
                indices,
                planner_evaluation.selected_logprobs.to(
                    device=decision_logprobs.device,
                    dtype=decision_logprobs.dtype,
                ),
            )
        action_logprobs = decision_logprobs.detach().float().cpu()
    if not bool(torch.isfinite(action_logprobs).all().item()):
        raise FloatingPointError("non-finite fixed-reference action log-probs")
    return action_logprobs


def ensure_finite_model(model: AgentPolicyValueNet) -> None:
    """Reject a model with non-finite parameters before checkpoint publication."""
    _raise_for_nonfinite_parameters(model)


def _normalized_anchor_cache_keys(
    keys: Sequence[int | None] | None,
    count: int,
) -> tuple[int | None, ...]:
    if keys is None:
        return (None,) * count
    if len(keys) != count:
        raise ValueError("anchor cache keys must align with PPO microbatches")
    return tuple(keys)


def _unit_weighted_breakdowns(
    breakdowns: Sequence[PpoLossBreakdown],
    unit_counts: Sequence[int],
    anchor_counts: Sequence[int],
    teacher_counts: Sequence[int],
    factual_counts: Sequence[int],
    macro_counts: Sequence[int],
    planner_counts: Sequence[tuple[int, int]],
    root_value_counts: Sequence[int],
    *,
    engine_teacher_coef: float,
    kl_anchor_coef: float,
    factual_effect_coef: float,
    factual_successor_coef: float,
    macro_conditional_coef: float,
    macro_expected_coef: float,
    candidate_rerank_coef: float,
    proposal_distillation_coef: float,
    root_information_value_coef: float,
    copy_to_cpu: bool,
) -> PpoLossBreakdown:
    if (
        len(breakdowns) != len(unit_counts)
        or len(breakdowns) != len(anchor_counts)
        or len(breakdowns) != len(teacher_counts)
        or len(breakdowns) != len(factual_counts)
        or len(breakdowns) != len(macro_counts)
        or len(breakdowns) != len(planner_counts)
        or len(breakdowns) != len(root_value_counts)
        or not breakdowns
    ):
        raise ValueError("loss breakdowns and unit counts must be non-empty and align")
    total_units = sum(unit_counts)
    weights = tuple(count / total_units for count in unit_counts)
    total_anchor_units = sum(anchor_counts)
    anchor_weights = tuple(
        0.0 if total_anchor_units <= 0 else count / total_anchor_units
        for count in anchor_counts
    )

    def weighted_tensor(name: str) -> Tensor:
        values = [_cast_tensor(getattr(item, name)) for item in breakdowns]
        result = sum(
            (value * weight for value, weight in zip(values, weights, strict=True)),
            start=values[0].new_zeros(()),
        )
        return _detach_loss_scalar(result, copy_to_cpu=copy_to_cpu)

    def weighted_optional(name: str) -> float | None:
        values = [getattr(item, name) for item in breakdowns]
        if any(value is None for value in values):
            return None
        return sum(
            float(value) * weight for value, weight in zip(values, weights, strict=True)
        )

    def anchor_weighted_tensor(name: str) -> Tensor:
        values = tuple(_cast_tensor(getattr(item, name)) for item in breakdowns)
        result = sum(
            (
                value * weight
                for value, weight in zip(values, anchor_weights, strict=True)
            ),
            start=values[0].new_zeros(()),
        )
        return _detach_loss_scalar(result, copy_to_cpu=copy_to_cpu)

    def anchor_weighted_optional(name: str) -> float | None:
        values = tuple(getattr(item, name) for item in breakdowns)
        if any(value is None for value in values):
            return None
        return sum(
            float(value) * weight
            for value, weight in zip(values, anchor_weights, strict=True)
        )

    teacher_values = tuple(
        _required_engine_teacher_loss(breakdown) for breakdown in breakdowns
    )
    total_teacher_decisions = sum(teacher_counts)
    if total_teacher_decisions > 0:
        engine_teacher_loss = sum(
            (
                value * (count / total_teacher_decisions)
                for value, count in zip(
                    teacher_values,
                    teacher_counts,
                    strict=True,
                )
                if count > 0
            ),
            start=teacher_values[0].new_zeros(()),
        )
    else:
        engine_teacher_loss = teacher_values[0].new_zeros(())
    total_factual_decisions = sum(factual_counts)

    def factual_weighted(name: str) -> Tensor:
        values = tuple(_required_optional_loss(item, name) for item in breakdowns)
        if total_factual_decisions <= 0:
            return values[0].new_zeros(())
        return sum(
            (
                value * (count / total_factual_decisions)
                for value, count in zip(values, factual_counts, strict=True)
                if count > 0
            ),
            start=values[0].new_zeros(()),
        )

    factual_effect_loss = factual_weighted("factual_effect_loss")
    factual_successor_loss = factual_weighted("factual_successor_loss")
    total_macro_roots = sum(macro_counts)

    def macro_weighted(name: str) -> Tensor:
        values = tuple(_required_optional_loss(item, name) for item in breakdowns)
        if total_macro_roots <= 0:
            return values[0].new_zeros(())
        return sum(
            (
                value * (count / total_macro_roots)
                for value, count in zip(values, macro_counts, strict=True)
                if count > 0
            ),
            start=values[0].new_zeros(()),
        )

    macro_conditional_loss = macro_weighted("macro_conditional_loss")
    macro_expected_loss = macro_weighted("macro_expected_loss")
    total_planner_applicable = sum(counts[1] for counts in planner_counts)

    def planner_weighted(name: str) -> Tensor:
        values = tuple(_required_optional_loss(item, name) for item in breakdowns)
        if total_planner_applicable <= 0:
            return values[0].new_zeros(())
        return sum(
            (
                value * (counts[1] / total_planner_applicable)
                for value, counts in zip(values, planner_counts, strict=True)
                if counts[1] > 0
            ),
            start=values[0].new_zeros(()),
        )

    candidate_rerank_loss = planner_weighted("candidate_rerank_loss")
    proposal_distillation_loss = planner_weighted("proposal_distillation_loss")
    total_root_value_rows = sum(root_value_counts)
    root_value_losses = tuple(
        _required_optional_loss(item, "root_information_value_loss")
        for item in breakdowns
    )
    if total_root_value_rows > 0:
        root_value_loss = sum(
            (
                value * (count / total_root_value_rows)
                for value, count in zip(
                    root_value_losses,
                    root_value_counts,
                    strict=True,
                )
                if count > 0
            ),
            start=root_value_losses[0].new_zeros(()),
        )
    else:
        root_value_loss = root_value_losses[0].new_zeros(())
    base_losses = tuple(
        _cast_tensor(item.loss)
        - engine_teacher_coef * teacher_loss
        - factual_effect_coef
        * _required_optional_loss(
            item,
            "factual_effect_loss",
        )
        - factual_successor_coef
        * _required_optional_loss(
            item,
            "factual_successor_loss",
        )
        - macro_conditional_coef
        * _required_optional_loss(
            item,
            "macro_conditional_loss",
        )
        - macro_expected_coef
        * _required_optional_loss(
            item,
            "macro_expected_loss",
        )
        - candidate_rerank_coef
        * _required_optional_loss(
            item,
            "candidate_rerank_loss",
        )
        - proposal_distillation_coef
        * _required_optional_loss(
            item,
            "proposal_distillation_loss",
        )
        - root_information_value_coef
        * _required_optional_loss(
            item,
            "root_information_value_loss",
        )
        - kl_anchor_coef * _cast_tensor(item.kl_anchor_loss)
        for item, teacher_loss in zip(
            breakdowns,
            teacher_values,
            strict=True,
        )
    )
    combined_loss = (
        sum(
            (
                value * weight
                for value, weight in zip(base_losses, weights, strict=True)
            ),
            start=base_losses[0].new_zeros(()),
        )
        + engine_teacher_coef * engine_teacher_loss
        + factual_effect_coef * factual_effect_loss
        + factual_successor_coef * factual_successor_loss
        + macro_conditional_coef * macro_conditional_loss
        + macro_expected_coef * macro_expected_loss
        + candidate_rerank_coef * candidate_rerank_loss
        + proposal_distillation_coef * proposal_distillation_loss
        + root_information_value_coef * root_value_loss
        + kl_anchor_coef * anchor_weighted_tensor("kl_anchor_loss")
    )

    return PpoLossBreakdown(
        loss=_detach_loss_scalar(combined_loss, copy_to_cpu=copy_to_cpu),
        policy_loss=weighted_tensor("policy_loss"),
        value_loss=weighted_tensor("value_loss"),
        entropy_loss=weighted_tensor("entropy_loss"),
        kl_anchor_loss=anchor_weighted_tensor("kl_anchor_loss"),
        approx_kl=weighted_optional("approx_kl"),
        approx_kl_k3=weighted_optional("approx_kl_k3"),
        anchor_kl=anchor_weighted_optional("anchor_kl"),
        entropy=weighted_optional("entropy"),
        ratio_mean=weighted_optional("ratio_mean"),
        ratio_p95=weighted_optional("ratio_p95"),
        clip_fraction=weighted_optional("clip_fraction"),
        value_mean=weighted_optional("value_mean"),
        critic_warmup=all(item.critic_warmup for item in breakdowns),
        engine_teacher_loss=_detach_loss_scalar(
            engine_teacher_loss,
            copy_to_cpu=copy_to_cpu,
        ),
        engine_teacher_decisions=total_teacher_decisions,
        factual_effect_loss=_detach_loss_scalar(
            factual_effect_loss,
            copy_to_cpu=copy_to_cpu,
        ),
        factual_successor_loss=_detach_loss_scalar(
            factual_successor_loss,
            copy_to_cpu=copy_to_cpu,
        ),
        factual_decisions=total_factual_decisions,
        macro_conditional_loss=_detach_loss_scalar(
            macro_conditional_loss,
            copy_to_cpu=copy_to_cpu,
        ),
        macro_expected_loss=_detach_loss_scalar(
            macro_expected_loss,
            copy_to_cpu=copy_to_cpu,
        ),
        macro_roots=total_macro_roots,
        candidate_rerank_loss=_detach_loss_scalar(
            candidate_rerank_loss,
            copy_to_cpu=copy_to_cpu,
        ),
        proposal_distillation_loss=_detach_loss_scalar(
            proposal_distillation_loss,
            copy_to_cpu=copy_to_cpu,
        ),
        planner_decisions=sum(counts[0] for counts in planner_counts),
        planner_applicable_decisions=total_planner_applicable,
        planner_metrics=_combined_planner_metrics(breakdowns),
        root_information_value_loss=_detach_loss_scalar(
            root_value_loss,
            copy_to_cpu=copy_to_cpu,
        ),
        root_information_value_rows=total_root_value_rows,
    )


def _cast_tensor(value: Any) -> Tensor:
    """Narrow a dynamically selected loss-breakdown field to a tensor."""
    if not isinstance(value, Tensor):
        raise TypeError("loss breakdown field must be a tensor")
    return value


def _required_engine_teacher_loss(breakdown: PpoLossBreakdown) -> Tensor:
    value = breakdown.engine_teacher_loss
    if value is None:
        return _cast_tensor(breakdown.loss).new_zeros(())
    return value


def _required_optional_loss(
    breakdown: PpoLossBreakdown,
    name: str,
) -> Tensor:
    value = getattr(breakdown, name)
    if value is None:
        return _cast_tensor(breakdown.loss).new_zeros(())
    return _cast_tensor(value)


def _weighted_factual_loss(
    breakdown: PpoLossBreakdown,
    config: PpoConfig,
) -> Tensor:
    return config.factual_effect_coef * _required_optional_loss(
        breakdown, "factual_effect_loss"
    ) + config.factual_successor_coef * _required_optional_loss(
        breakdown, "factual_successor_loss"
    )


def _weighted_macro_loss(
    breakdown: PpoLossBreakdown,
    config: PpoConfig,
) -> Tensor:
    """Return configured root-only macro terms for one microbatch."""
    return config.macro_conditional_coef * _required_optional_loss(
        breakdown,
        "macro_conditional_loss",
    ) + config.macro_expected_coef * _required_optional_loss(
        breakdown,
        "macro_expected_loss",
    )


def _weighted_planner_loss(
    breakdown: PpoLossBreakdown,
    config: PpoConfig,
) -> Tensor:
    """Return configured planner auxiliary terms for one microbatch."""
    return config.candidate_rerank_coef * _required_optional_loss(
        breakdown,
        "candidate_rerank_loss",
    ) + config.proposal_distillation_coef * _required_optional_loss(
        breakdown,
        "proposal_distillation_loss",
    )


def _planner_decision_counts(
    batch: PpoBatch,
    config: PpoConfig,
) -> tuple[int, int]:
    """Return total and imitation-applicable planner groups without a forward."""
    replay = batch.planner_replay
    if replay is None:
        return (0, 0)
    replay.validate(batch_size=len(batch.actions))
    offsets = replay.candidate_offsets.to(dtype=torch.long)
    lengths = offsets[1:] - offsets[:-1]
    exact_counts = torch.segment_reduce(
        replay.rules_exact.to(dtype=torch.float32),
        "sum",
        lengths=lengths.to(device=replay.rules_exact.device),
    )
    applicable = (
        exact_counts.eq(lengths.to(device=exact_counts.device, dtype=torch.float32))
        & replay.scenario_grid_complete.to(device=exact_counts.device)
        & replay.identity_valid.to(device=exact_counts.device)
        & replay.policy_ages.to(device=exact_counts.device).le(
            config.planner_imitation_max_policy_age
        )
    )
    return (replay.group_count, int(applicable.sum().item()))


def _combined_planner_metrics(
    breakdowns: Sequence[PpoLossBreakdown],
) -> PlannerImitationMetrics | None:
    """Sum non-overlapping microbatch diagnostics for logging."""
    metrics = tuple(
        breakdown.planner_metrics
        for breakdown in breakdowns
        if breakdown.planner_metrics is not None
    )
    if not metrics:
        return None
    return PlannerImitationMetrics(
        planner_decisions=sum(item.planner_decisions for item in metrics),
        applicable_decisions=sum(item.applicable_decisions for item in metrics),
        stale_decisions=sum(item.stale_decisions for item in metrics),
        inexact_decisions=sum(item.inexact_decisions for item in metrics),
        incomplete_grid_decisions=sum(
            item.incomplete_grid_decisions for item in metrics
        ),
        identity_invalid_decisions=sum(
            item.identity_invalid_decisions for item in metrics
        ),
        censored_support_decisions=sum(
            item.censored_support_decisions for item in metrics
        ),
        exhaustive_support_decisions=sum(
            item.exhaustive_support_decisions for item in metrics
        ),
        clipped_candidates=sum(item.clipped_candidates for item in metrics),
        candidate_count=sum(item.candidate_count for item in metrics),
    )


def _raise_for_nonfinite_losses(
    breakdowns: Sequence[PpoLossBreakdown],
) -> None:
    """Reject non-finite losses with one device synchronization per update."""
    if not breakdowns:
        raise ValueError("loss breakdowns must be non-empty")
    names: tuple[str, ...] = (
        "loss",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "kl_anchor_loss",
    )
    if any(item.engine_teacher_loss is not None for item in breakdowns):
        names = (*names, "engine_teacher_loss")
    for factual_name in (
        "factual_effect_loss",
        "factual_successor_loss",
        "macro_conditional_loss",
        "macro_expected_loss",
    ):
        if any(getattr(item, factual_name) is not None for item in breakdowns):
            names = (*names, factual_name)
    for planner_name in (
        "candidate_rerank_loss",
        "proposal_distillation_loss",
    ):
        if any(getattr(item, planner_name) is not None for item in breakdowns):
            names = (*names, planner_name)
    if any(item.root_information_value_loss is not None for item in breakdowns):
        names = (*names, "root_information_value_loss")
    entries = tuple(
        (batch_index, name, getattr(breakdown, name))
        for batch_index, breakdown in enumerate(breakdowns)
        for name in names
        if getattr(breakdown, name) is not None
    )
    values = tuple(_cast_tensor(value) for _, _, value in entries)
    device = values[0].device
    # Copy one mask so the hard gate introduces only one device synchronization.
    finite = (
        torch.stack([torch.isfinite(value).all().to(device=device) for value in values])
        .detach()
        .cpu()
        .tolist()
    )
    for (batch_index, name, _), is_finite in zip(entries, finite, strict=True):
        if not is_finite:
            raise FloatingPointError(
                f"non-finite PPO {name} in microbatch {batch_index}"
            )


def _raise_for_nonfinite_loss(breakdown: PpoLossBreakdown) -> None:
    """Validate one breakdown for direct callers and focused tests."""
    _raise_for_nonfinite_losses((breakdown,))


def _raise_for_nonfinite_parameters(model: AgentPolicyValueNet) -> None:
    try:
        torch.nn.utils.get_total_norm(
            model.parameters(),
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        if "non-finite" not in str(exc).lower():
            raise
        raise FloatingPointError("non-finite model parameter norm") from exc


def load_frozen_anchor_model(
    checkpoint_path: Path,
    *,
    fallback_config: AgentNetworkConfig | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> AgentPolicyValueNet:
    """Load a checkpoint as an eval-mode, gradient-frozen PPO anchor."""
    checkpoint = torch.load(deck_records.repo_path(checkpoint_path), map_location="cpu")
    config = (
        _checkpoint_model_config(checkpoint) or fallback_config or AgentNetworkConfig()
    )
    model = build_agent_policy_value_net(config)
    incompatible = model.load_state_dict(
        _checkpoint_state_dict(checkpoint), strict=False
    )
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {
        "opponent_hand_head.weight",
        "opponent_hand_head.bias",
    } | LEGACY_STATE_ENCODER_MISSING_KEYS
    if missing - allowed_missing or unexpected:
        raise RuntimeError("checkpoint state dict is incompatible with anchor model")
    if dtype is None:
        model = model.to(device=device)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def sequence_kl_from_logits(
    student_step_logits: Sequence[Tensor],
    anchor_step_logits: Sequence[Tensor],
    actions: Sequence[Sequence[int]],
    *,
    action_targets: Tensor | None = None,
) -> Tensor:
    """Return mean per-sample sum KL over teacher-forced decode steps."""
    if len(student_step_logits) != len(anchor_step_logits):
        raise ValueError("student and anchor step logits must align")
    if not actions:
        if student_step_logits:
            return student_step_logits[0].new_zeros(())
        if anchor_step_logits:
            return anchor_step_logits[0].new_zeros(())
        return torch.zeros(())

    total = _zero_like_logits(student_step_logits, anchor_step_logits)
    for step_index, (student_logits, anchor_logits) in enumerate(
        zip(student_step_logits, anchor_step_logits, strict=True)
    ):
        active_indices = _active_indices_for_step(
            actions,
            action_targets=action_targets,
            step_index=step_index,
            device=student_logits.device,
        )
        if int(active_indices.numel()) == 0:
            continue
        step_kl = _categorical_kl(
            student_logits.index_select(0, active_indices),
            anchor_logits.index_select(0, active_indices),
        )
        total = total + step_kl.sum()
    return total / len(actions)


def token_kl_from_logits(
    student_step_logits: Sequence[Tensor],
    anchor_step_logits: Sequence[Tensor],
    token_mask: Tensor,
    *,
    active_tokens: int | None = None,
) -> Tensor:
    """Return mean categorical KL over active teacher-forced decode tokens."""
    if len(student_step_logits) != len(anchor_step_logits):
        raise ValueError("student and anchor step logits must align")
    if token_mask.ndim != 2 or token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be a bool tensor with shape [batch, steps]")
    if token_mask.shape[1] != len(student_step_logits):
        raise ValueError("token_mask steps must align with teacher-forced logits")
    if active_tokens is None:
        active_tokens = int(token_mask.sum().item())
    if active_tokens <= 0:
        raise ValueError("token KL requires at least one active token")

    total = _zero_like_logits(student_step_logits, anchor_step_logits)
    for step_index, (student_logits, anchor_logits) in enumerate(
        zip(student_step_logits, anchor_step_logits, strict=True)
    ):
        step_mask = token_mask[:, step_index]
        if step_mask.device != student_logits.device:
            step_mask = step_mask.to(device=student_logits.device)
        active_indices = torch.nonzero(step_mask, as_tuple=False).flatten()
        if int(active_indices.numel()) == 0:
            continue
        step_kl = _categorical_kl(
            student_logits.index_select(0, active_indices),
            anchor_logits.index_select(0, active_indices),
        )
        total = total + step_kl.sum()
    return total / active_tokens


def _validate_batch(batch: PpoBatch) -> None:
    batch_size = len(batch.actions)
    _validate_recurrent_batch(batch, batch_size=batch_size)
    replay = batch.planner_replay
    if replay is not None:
        _validate_planner_replay_batch(batch, replay)
    root_replay = batch.root_information_value_replay
    if root_replay is not None:
        root_replay.validate(batch_size=batch_size)
    if batch.decks is not None and len(batch.decks) != batch_size:
        raise ValueError("decks must have the same batch size as actions")
    for name, value in (
        ("old_action_logprobs", batch.old_action_logprobs),
        ("sampling_temperatures", batch.sampling_temperatures),
        ("old_values", batch.old_values),
        ("returns", batch.returns),
        ("advantages", batch.advantages),
    ):
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape [batch_size]")
    if not bool(torch.isfinite(batch.sampling_temperatures).all().item()) or not bool(
        batch.sampling_temperatures.gt(0.0).all().item()
    ):
        raise ValueError("sampling_temperatures must be finite and positive")
    if batch.action_targets is not None and (
        batch.action_targets.ndim != 2 or batch.action_targets.shape[0] != batch_size
    ):
        raise ValueError("action_targets must have shape [batch_size, max_steps]")
    _validate_behavior_actions(batch)
    teacher_fields = (
        batch.engine_teacher_actions,
        batch.engine_teacher_action_targets,
        batch.engine_teacher_confidences,
        batch.engine_teacher_weights,
        batch.engine_teacher_mask,
    )
    teacher_fields_present = sum(value is not None for value in teacher_fields)
    if teacher_fields_present not in (0, len(teacher_fields)):
        raise ValueError("engine teacher batch fields must be provided together")
    candidate_fields = (
        batch.engine_teacher_candidate_actions,
        batch.engine_teacher_candidate_features,
    )
    candidate_fields_present = sum(value is not None for value in candidate_fields)
    if candidate_fields_present not in (0, len(candidate_fields)):
        raise ValueError("engine teacher search candidate fields must align")
    if candidate_fields_present and not teacher_fields_present:
        raise ValueError("search candidates require engine teacher targets")
    if teacher_fields_present:
        _validate_engine_teacher_batch(batch)
    factual_fields = (
        batch.factual_effect_targets,
        batch.factual_actor_relations,
        batch.factual_next_contexts,
    )
    factual_fields_present = sum(value is not None for value in factual_fields)
    if factual_fields_present not in (0, len(factual_fields)):
        raise ValueError("factual batch fields must be provided together")
    if factual_fields_present:
        _validate_factual_batch(batch)
    macro_fields = (
        batch.macro_effect_targets,
        batch.macro_endpoints,
        batch.macro_continuation_summaries,
        batch.macro_mask,
    )
    macro_fields_present = sum(value is not None for value in macro_fields)
    if macro_fields_present not in (0, len(macro_fields)):
        raise ValueError("macro batch fields must be provided together")
    if macro_fields_present:
        _validate_macro_batch(batch)
    token_fields = (
        batch.old_token_logprobs,
        batch.old_prefix_values,
        batch.token_returns,
        batch.token_advantages,
        batch.token_mask,
    )
    present_fields = sum(value is not None for value in token_fields)
    if present_fields not in (0, len(token_fields)):
        raise ValueError("token-level PPO tensors must be provided together")
    if present_fields == 0:
        return
    (
        old_token_logprobs,
        old_prefix_values,
        token_returns,
        token_advantages,
        token_mask,
    ) = _require_token_batch_tensors(batch)
    if token_mask.dtype != torch.bool:
        raise ValueError("token_mask must use bool dtype")
    if token_mask.ndim != 2 or token_mask.shape[0] != batch_size:
        raise ValueError("token_mask must have shape [batch_size, max_tokens]")
    if token_mask.shape[1] <= 0 and (
        replay is None or replay.group_count != batch_size
    ):
        raise ValueError("token-level PPO batches require at least one token column")
    for name, value in (
        ("old_token_logprobs", old_token_logprobs),
        ("old_prefix_values", old_prefix_values),
        ("token_returns", token_returns),
        ("token_advantages", token_advantages),
    ):
        if value.ndim != 2 or value.shape != token_mask.shape:
            raise ValueError(f"{name} must have the same shape as token_mask")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain only finite values")

    expected_mask = torch.zeros_like(token_mask, device="cpu")
    max_counts = batch.options.max_counts.detach().to(device="cpu").tolist()
    planner_rows = set()
    if replay is not None:
        planner_rows = {
            int(value)
            for value in replay.decision_indices.detach().to(device="cpu").tolist()
        }
    for row_index, (action, max_count_value) in enumerate(
        zip(batch.actions, max_counts, strict=True)
    ):
        max_count = int(max_count_value)
        if len(action) > max_count:
            raise ValueError("action length exceeds prompt max_count")
        token_count = (
            0
            if row_index in planner_rows
            else len(action) + int(len(action) < max_count)
        )
        if token_count <= 0 and row_index not in planner_rows:
            raise ValueError("token-level PPO rows require a sampled decode token")
        if token_count > token_mask.shape[1]:
            raise ValueError("token_mask is too narrow for the sampled action")
        expected_mask[row_index, :token_count] = True
    if not torch.equal(token_mask.detach().to(device="cpu"), expected_mask):
        raise ValueError("token_mask does not match selection/STOP semantics")

    cpu_mask = token_mask.detach().to(device="cpu")
    summed_logprobs = (
        old_token_logprobs.detach()
        .to(device="cpu")
        .masked_fill(
            ~cpu_mask,
            0.0,
        )
        .sum(dim=1)
    )
    fallback_rows = torch.tensor(
        [index not in planner_rows for index in range(batch_size)],
        dtype=torch.bool,
    )
    expected_action_logprobs = batch.old_action_logprobs.detach().to(
        device="cpu", dtype=summed_logprobs.dtype
    )
    if bool(fallback_rows.any()) and not torch.allclose(
        summed_logprobs[fallback_rows],
        expected_action_logprobs[fallback_rows],
        rtol=1.0e-4,
        atol=1.0e-5,
    ):
        raise ValueError("active old_token_logprobs must sum to old_action_logprobs")
    if bool(fallback_rows.any()):
        first_prefix_values = old_prefix_values[:, 0].detach().to(device="cpu")
        if not torch.allclose(
            first_prefix_values[fallback_rows],
            batch.old_values.detach().to(device="cpu", dtype=first_prefix_values.dtype)[
                fallback_rows
            ],
            rtol=1.0e-4,
            atol=1.0e-5,
        ):
            raise ValueError("the first old_prefix_value must equal old_values")


def _validate_recurrent_batch(batch: PpoBatch, *, batch_size: int) -> None:
    """Validate the atomic full-sequence replay contract."""
    fields = (
        batch.public_events,
        batch.sequence_offsets,
        batch.sequence_artifacts,
    )
    present = sum(value is not None for value in fields)
    if present == 0:
        return
    if present != len(fields):
        raise ValueError("recurrent PPO batch fields must be provided together")
    events = batch.public_events
    offsets = batch.sequence_offsets
    artifacts = batch.sequence_artifacts
    if events is None or offsets is None or artifacts is None:
        raise AssertionError("validated recurrent PPO fields disappeared")
    validate_public_event_batch(events)
    if events.batch_size != batch_size:
        raise ValueError("public event rows must align with PPO decisions")
    if events.event_types.device != batch.states.card_ids.device:
        raise ValueError("public events and PPO states must use the same device")
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("sequence_offsets must describe at least one sequence")
    if offsets.dtype == torch.bool or offsets.is_floating_point():
        raise TypeError("sequence_offsets must use an integer dtype")
    values = tuple(
        int(value)
        for value in offsets.detach().to(device="cpu", dtype=torch.long).tolist()
    )
    if values[0] != 0 or values[-1] != batch_size:
        raise ValueError("sequence_offsets must span the complete PPO batch")
    if any(left >= right for left, right in pairwise(values)):
        raise ValueError("sequence_offsets must be strictly increasing")
    if len(artifacts) != len(values) - 1:
        raise ValueError("sequence artifacts must align with sequence offsets")
    compatibility = artifacts[0].compatibility
    if any(artifact.compatibility != compatibility for artifact in artifacts[1:]):
        raise ValueError("recurrent PPO batch mixes incompatible model contracts")
    forbidden = (
        batch.engine_teacher_actions,
        batch.factual_effect_targets,
        batch.macro_mask,
        batch.planner_replay,
        batch.root_information_value_replay,
    )
    if any(value is not None for value in forbidden):
        raise ValueError("recurrent sequence schema is PPO-only")


def _validate_behavior_actions(batch: PpoBatch) -> None:
    """Establish the one-time action-integrity proof used by PPO fast paths."""
    minimums = batch.options.min_counts.detach().to(device="cpu").tolist()
    maximums = batch.options.max_counts.detach().to(device="cpu").tolist()
    valid_options = batch.options.valid_options.detach().to(device="cpu")
    contexts = batch.options.contexts.detach().to(device="cpu")
    option_width = int(valid_options.shape[1])
    for row, action_values in enumerate(batch.actions):
        action = tuple(int(index) for index in action_values)
        minimum = int(minimums[row])
        maximum = int(maximums[row])
        if len(action) < minimum or len(action) > maximum:
            raise ValueError("behavior action violates selection count bounds")
        if len(set(action)) != len(action) or any(
            index < 0 or index >= option_width or not bool(valid_options[row, index])
            for index in action
        ):
            raise ValueError("behavior action contains an invalid option index")
        context = (
            -1
            if option_width == 0 or not bool(valid_options[row].any())
            else int(contexts[row, 0])
        )
        if is_unordered_set_selection(
            context=context,
            min_count=minimum,
            max_count=maximum,
        ) and any(left >= right for left, right in pairwise(action)):
            raise ValueError("unordered behavior actions must be strictly increasing")


def _validate_planner_replay_batch(
    batch: PpoBatch,
    replay: PlannerReplayBatch,
) -> None:
    """Bind immutable candidate behavior evidence to actual PPO rows."""
    replay.validate(batch_size=len(batch.actions))
    offsets = replay.candidate_offsets.detach().to(device="cpu", dtype=torch.long)
    selected_indices = replay.selected_candidate_indices.detach().to(
        device="cpu", dtype=torch.long
    )
    behavior = replay.old_behavior_probabilities.detach().to(device="cpu")
    temperatures = replay.planner_temperatures.detach().to(device="cpu")
    decision_indices = replay.decision_indices.detach().to(
        device="cpu", dtype=torch.long
    )
    old_action_logprobs = batch.old_action_logprobs.detach().to(device="cpu")
    sampling_temperatures = batch.sampling_temperatures.detach().to(device="cpu")
    for group_index, decision_value in enumerate(decision_indices.tolist()):
        decision_index = int(decision_value)
        selected_index = int(selected_indices[group_index])
        selected_action = replay.candidate_actions[group_index][selected_index]
        if (
            tuple(int(value) for value in batch.actions[decision_index])
            != selected_action
        ):
            raise ValueError("planner replay selected action differs from PPO behavior")
        candidate_index = int(offsets[group_index]) + selected_index
        expected_logprob = math.log(float(behavior[candidate_index]))
        if not math.isclose(
            expected_logprob,
            float(old_action_logprobs[decision_index]),
            rel_tol=1.0e-4,
            abs_tol=1.0e-5,
        ):
            raise ValueError("planner replay old log-probability differs from PPO row")
        if not math.isclose(
            float(temperatures[group_index]),
            float(sampling_temperatures[decision_index]),
            rel_tol=1.0e-5,
            abs_tol=1.0e-5,
        ):
            raise ValueError("planner replay temperature differs from PPO row")


def validate_ppo_batch(batch: PpoBatch) -> None:
    """Validate an untrusted PPO batch before immutable row selection."""
    _validate_batch(batch)


def _validate_factual_batch(batch: PpoBatch) -> None:
    """Reject incomplete or non-finite dense actual-transition evidence."""
    effect_targets = batch.factual_effect_targets
    actor_relations = batch.factual_actor_relations
    next_contexts = batch.factual_next_contexts
    if effect_targets is None or actor_relations is None or next_contexts is None:
        raise ValueError("factual batch fields must be provided together")
    batch_size = len(batch.actions)
    if effect_targets.shape != (batch_size, DYNAMIC_EFFECT_FEATURE_SIZE):
        raise ValueError(
            "factual_effect_targets must have shape "
            f"[batch_size, {DYNAMIC_EFFECT_FEATURE_SIZE}]"
        )
    if not effect_targets.is_floating_point() or not bool(
        torch.isfinite(effect_targets).all().item()
    ):
        raise ValueError("factual_effect_targets must be finite floating point")
    for name, values, upper_bound in (
        (
            "factual_actor_relations",
            actor_relations,
            FACTUAL_ACTOR_RELATION_COUNT,
        ),
        ("factual_next_contexts", next_contexts, FACTUAL_NEXT_CONTEXT_COUNT),
    ):
        if values.ndim != 1 or values.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape [batch_size]")
        if values.dtype == torch.bool or values.is_floating_point():
            raise ValueError(f"{name} must use an integer dtype")
        if bool(((values < 0) | (values >= upper_bound)).any().item()):
            raise ValueError(f"{name} contains an out-of-range label")


def _validate_macro_batch(batch: PpoBatch) -> None:
    """Reject malformed dense root mappings and continuation summaries."""
    effects = batch.macro_effect_targets
    endpoints = batch.macro_endpoints
    summaries = batch.macro_continuation_summaries
    mask = batch.macro_mask
    if effects is None or endpoints is None or summaries is None or mask is None:
        raise ValueError("macro batch fields must be provided together")
    batch_size = len(batch.actions)
    if effects.shape != (batch_size, DYNAMIC_EFFECT_FEATURE_SIZE):
        raise ValueError("macro effect targets have an invalid shape")
    if summaries.shape != (batch_size, MACRO_CONTINUATION_SUMMARY_SIZE):
        raise ValueError("macro continuation summaries have an invalid shape")
    if not effects.is_floating_point() or not summaries.is_floating_point():
        raise ValueError("macro targets and summaries must be floating point")
    if not bool(torch.isfinite(effects).all().item()) or not bool(
        torch.isfinite(summaries).all().item()
    ):
        raise ValueError("macro targets and summaries must be finite")
    if mask.dtype != torch.bool or mask.shape != (batch_size,):
        raise ValueError("macro mask must be bool [batch_size]")
    if (
        endpoints.dtype == torch.bool
        or endpoints.is_floating_point()
        or (endpoints.shape != (batch_size,))
    ):
        raise ValueError("macro endpoints must be integer [batch_size]")
    active_endpoints = endpoints[mask]
    if bool(
        ((active_endpoints < 0) | (active_endpoints >= MACRO_ENDPOINT_COUNT))
        .any()
        .item()
    ):
        raise ValueError("macro endpoints contain an out-of-range label")
    if bool(effects[~mask].ne(0.0).any().item()) or bool(
        summaries[~mask].ne(0.0).any().item()
    ):
        raise ValueError("non-root macro rows must remain zero-filled")


def _validate_engine_teacher_batch(batch: PpoBatch) -> None:
    actions = batch.engine_teacher_actions
    action_targets = batch.engine_teacher_action_targets
    confidences = batch.engine_teacher_confidences
    weights = batch.engine_teacher_weights
    mask = batch.engine_teacher_mask
    if (
        actions is None
        or action_targets is None
        or confidences is None
        or weights is None
        or mask is None
    ):
        raise ValueError("engine teacher batch fields must be provided together")
    batch_size = len(batch.actions)
    if len(actions) != batch_size:
        raise ValueError("engine teacher actions must align with PPO rows")
    if action_targets.ndim != 2 or action_targets.shape[0] != batch_size:
        raise ValueError("engine teacher targets must be [batch_size, max_steps]")
    if mask.dtype != torch.bool or mask.ndim != 1 or mask.shape[0] != batch_size:
        raise ValueError("engine teacher mask must be bool [batch_size]")
    for name, values in (
        ("engine_teacher_confidences", confidences),
        ("engine_teacher_weights", weights),
    ):
        if values.ndim != 1 or values.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape [batch_size]")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError(f"{name} must contain finite values")
    cpu_mask = mask.detach().to(device="cpu")
    cpu_confidences = confidences.detach().to(device="cpu")
    cpu_weights = weights.detach().to(device="cpu")
    if bool((cpu_confidences[~cpu_mask] != 0.0).any().item()) or bool(
        (cpu_weights[~cpu_mask] != 0.0).any().item()
    ):
        raise ValueError("masked engine teacher rows must have zero evidence weights")
    if bool((cpu_confidences[cpu_mask] <= 0.0).any().item()) or bool(
        (cpu_confidences[cpu_mask] > 1.0).any().item()
    ):
        raise ValueError("valid engine teacher confidences must be in (0, 1]")
    if bool((cpu_weights[cpu_mask] <= 0.0).any().item()):
        raise ValueError("valid engine teacher weights must be positive")
    min_counts = batch.options.min_counts.detach().to(device="cpu").tolist()
    max_counts = batch.options.max_counts.detach().to(device="cpu").tolist()
    valid_options = batch.options.valid_options.detach().to(device="cpu")
    contexts = batch.options.contexts.detach().to(device="cpu")
    for row_index, action in enumerate(actions):
        normalized = tuple(int(index) for index in action)
        if len(normalized) < int(min_counts[row_index]) or len(normalized) > int(
            max_counts[row_index]
        ):
            raise ValueError("engine teacher action violates count bounds")
        if len(set(normalized)) != len(normalized) or any(
            index < 0
            or index >= int(valid_options.shape[1])
            or not bool(valid_options[row_index, index])
            for index in normalized
        ):
            raise ValueError("engine teacher action contains an invalid option index")
        context = int(contexts[row_index, 0].item())
        minimum = int(min_counts[row_index])
        maximum = int(max_counts[row_index])
        order_sensitive = context == int(SelectContext.SKILL_ORDER) or (
            maximum > 1
            and not is_unordered_set_selection(
                context=context,
                min_count=minimum,
                max_count=maximum,
            )
        )
        if not order_sensitive and any(
            left >= right for left, right in pairwise(normalized)
        ):
            raise ValueError(
                "unordered engine teacher actions must be strictly increasing"
            )
    _validate_engine_teacher_search_candidates(batch)


def _validate_engine_teacher_search_candidates(batch: PpoBatch) -> None:
    """Reject corrupt or target-misaligned complete-action search evidence."""
    candidate_actions = batch.engine_teacher_candidate_actions
    candidate_features = batch.engine_teacher_candidate_features
    if candidate_actions is None and candidate_features is None:
        return
    if candidate_actions is None or candidate_features is None:
        raise ValueError("engine teacher search candidate fields must align")
    teacher_actions = batch.engine_teacher_actions
    mask = batch.engine_teacher_mask
    if teacher_actions is None or mask is None:
        raise ValueError("search candidates require engine teacher targets")
    batch_size = len(batch.actions)
    if len(candidate_actions) != batch_size or len(candidate_features) != batch_size:
        raise ValueError("engine teacher search candidates must align with PPO rows")

    cpu_mask = mask.detach().to(device="cpu")
    min_counts = batch.options.min_counts.detach().to(device="cpu").tolist()
    max_counts = batch.options.max_counts.detach().to(device="cpu").tolist()
    valid_options = batch.options.valid_options.detach().to(device="cpu")
    contexts = batch.options.contexts.detach().to(device="cpu")
    for row_index, (groups, features) in enumerate(
        zip(candidate_actions, candidate_features, strict=True)
    ):
        normalized_group = tuple(
            tuple(int(option_index) for option_index in action) for action in groups
        )
        candidate_count = len(normalized_group)
        if features.shape != (candidate_count, SEARCH_EVIDENCE_FEATURE_SIZE):
            raise ValueError(
                "engine teacher candidate features must have shape "
                f"[candidates, {SEARCH_EVIDENCE_FEATURE_SIZE}]"
            )
        if not features.is_floating_point() or not bool(
            torch.isfinite(features).all().item()
        ):
            raise ValueError("engine teacher candidate features must be finite floats")
        if candidate_count == 0:
            continue
        if not bool(cpu_mask[row_index]):
            raise ValueError(
                "masked engine teacher rows cannot carry search candidates"
            )
        if len(set(normalized_group)) != candidate_count:
            raise ValueError("engine teacher candidate actions must be unique")
        target = tuple(int(index) for index in teacher_actions[row_index])
        if normalized_group.count(target) != 1:
            raise ValueError(
                "engine teacher target must occur once in search candidates"
            )
        context = int(contexts[row_index, 0].item())
        minimum = int(min_counts[row_index])
        maximum = int(max_counts[row_index])
        order_sensitive = context == int(SelectContext.SKILL_ORDER) or (
            maximum > 1
            and not is_unordered_set_selection(
                context=context,
                min_count=minimum,
                max_count=maximum,
            )
        )
        for candidate in normalized_group:
            if len(candidate) < minimum or len(candidate) > maximum:
                raise ValueError("engine teacher candidate violates count bounds")
            if len(set(candidate)) != len(candidate) or any(
                index < 0
                or index >= int(valid_options.shape[1])
                or not bool(valid_options[row_index, index])
                for index in candidate
            ):
                raise ValueError(
                    "engine teacher candidate contains an invalid option index"
                )
            if not order_sensitive and any(
                left >= right for left, right in pairwise(candidate)
            ):
                raise ValueError(
                    "unordered engine teacher candidates must be strictly increasing"
                )


def _uses_token_objective(batch: PpoBatch) -> bool:
    """Return whether the minibatch carries complete token credit evidence."""
    return batch.token_mask is not None


def _evaluate_planner_for_batch(
    model: AgentPolicyValueNet,
    batch: PpoBatch,
    behavior_evaluation: ActionEvaluation,
    config: PpoConfig,
) -> _PlannerModelEvaluation | None:
    """Replay planner rows on their immutable retained candidate supports."""
    replay = batch.planner_replay
    if replay is None:
        return None
    replay.validate(batch_size=len(batch.actions))
    policy_context = behavior_evaluation.policy_context
    if policy_context is None:
        raise RuntimeError("planner replay requires a reusable policy context")
    decision_indices = _planner_decision_indices(
        batch,
        device=policy_context.policy_global.device,
    )
    selected_options = _select_option_batch_rows(batch.options, decision_indices)
    selected_decks = (
        None
        if batch.decks is None
        else batch.decks.select(decision_indices.to(device=batch.decks.card_ids.device))
    )
    selected_context = model.select_policy_context(
        policy_context,
        decision_indices,
        decks=selected_decks,
    )
    candidate_evaluation = model.evaluate_planner_candidates_from_context(
        selected_context,
        selected_options,
        replay.candidate_actions,
        replay.candidate_features,
        decks=selected_decks,
    )
    if candidate_evaluation.candidate_counts != tuple(
        len(actions) for actions in replay.candidate_actions
    ):
        raise RuntimeError("model planner candidate groups changed during replay")
    distributions = planner_current_distributions(
        replay,
        current_base_logprobs=candidate_evaluation.base_action_logprobs,
        current_proposal_logprobs=candidate_evaluation.proposal_action_logprobs,
        current_reranker_residuals=candidate_evaluation.reranker_residuals,
    )
    selected_logprobs, entropies = planner_selected_logprobs_and_entropies(
        replay,
        distributions,
    )
    imitation = planner_imitation_loss(
        replay,
        distributions,
        config=PlannerImitationConfig(
            max_policy_age=config.planner_imitation_max_policy_age,
            target_ratio_clip=config.planner_target_ratio_clip,
        ),
    )
    return _PlannerModelEvaluation(
        distributions=distributions,
        selected_logprobs=selected_logprobs,
        entropies=entropies,
        imitation=imitation,
    )


def _root_information_value_replay_loss(
    model: AgentPolicyValueNet,
    replay: RootInformationValueReplayBatch,
    *,
    batch_size: int,
) -> Tensor:
    """Evaluate unique endpoint inputs and gather actual WDL supervision rows."""
    replay.validate(batch_size=batch_size)
    inputs = replay.model_inputs
    conditioned = model.encode_conditioned_state(inputs.states, replay.decks)
    unique_predictions = model.root_information_values_from_conditioned(
        conditioned,
        actor_relations=inputs.actor_relations,
        endpoints=inputs.endpoints,
        belief_summaries=inputs.belief_summaries,
    )
    expected_inputs = int(inputs.states.card_ids.shape[0])
    if unique_predictions.shape != (expected_inputs,):
        raise RuntimeError("model returned misaligned root-information values")
    predictions = unique_predictions.index_select(
        0,
        replay.value_input_indices.to(
            device=unique_predictions.device,
            dtype=torch.long,
        ),
    )
    return root_information_value_loss(
        predictions.float(),
        replay.final_root_outcomes,
    )


def _planner_decision_indices(
    batch: PpoBatch,
    *,
    device: torch.device,
) -> Tensor:
    replay = batch.planner_replay
    if replay is None:
        return torch.empty((0,), dtype=torch.long, device=device)
    return replay.decision_indices.to(device=device, dtype=torch.long)


def _select_option_batch_rows(options: OptionBatch, indices: Tensor) -> OptionBatch:
    """Select padded option rows without recomputing encoded features."""

    def select(values: Tensor) -> Tensor:
        return values.index_select(0, indices.to(device=values.device))

    return OptionBatch(
        option_types=select(options.option_types),
        contexts=select(options.contexts),
        entity_slots=select(options.entity_slots),
        entity_slot_mask=select(options.entity_slot_mask),
        attack_ids=select(options.attack_ids),
        card_ids=select(options.card_ids),
        scalars=select(options.scalars),
        dynamic_effect_features=select(options.dynamic_effect_features),
        dynamic_effect_masks=select(options.dynamic_effect_masks),
        valid_options=select(options.valid_options),
        min_counts=select(options.min_counts),
        max_counts=select(options.max_counts),
    )


def _decode_token_mask_for_cache(batch: PpoBatch) -> Tensor:
    """Return the actual selection/STOP steps, excluding forced termination."""
    if batch.planner_replay is not None:
        max_counts = batch.options.max_counts.detach().to(device="cpu").tolist()
        token_counts = tuple(
            len(action) + int(len(action) < int(max_count))
            for action, max_count in zip(batch.actions, max_counts, strict=True)
        )
        max_tokens = max(token_counts, default=0)
        mask = torch.zeros((len(batch.actions), max_tokens), dtype=torch.bool)
        for row_index, token_count in enumerate(token_counts):
            mask[row_index, :token_count] = True
        return mask
    if batch.integrity_validated and batch.token_counts is not None:
        max_tokens = max(batch.token_counts, default=0)
        mask = torch.zeros((len(batch.token_counts), max_tokens), dtype=torch.bool)
        for row_index, token_count in enumerate(batch.token_counts):
            mask[row_index, :token_count] = True
        return mask
    if batch.token_mask is not None:
        return batch.token_mask
    max_counts = batch.options.max_counts.detach().to(device="cpu").tolist()
    token_counts = tuple(
        len(action) + int(len(action) < int(max_count))
        for action, max_count in zip(batch.actions, max_counts, strict=True)
    )
    max_tokens = max(token_counts, default=0)
    mask = torch.zeros((len(batch.actions), max_tokens), dtype=torch.bool)
    for row_index, token_count in enumerate(token_counts):
        mask[row_index, :token_count] = True
    return mask


def _objective_unit_count(batch: PpoBatch) -> int:
    """Return decision, token, or mixed planner/fallback PPO objective units."""
    if batch.integrity_validated and batch.objective_unit_count is not None:
        return batch.objective_unit_count
    if batch.token_mask is None:
        return len(batch.actions)
    active_tokens = int(batch.token_mask.sum().item())
    planner_decisions = (
        0 if batch.planner_replay is None else batch.planner_replay.group_count
    )
    objective_units = active_tokens + planner_decisions
    if objective_units <= 0:
        raise ValueError("token-level PPO batches require active tokens")
    return objective_units


def _anchor_kl_unit_count(batch: PpoBatch) -> int:
    """Return the reduction units used by this batch's fixed-anchor KL."""
    if not _uses_token_objective(batch):
        return len(batch.actions)
    return int(_decode_token_mask_for_cache(batch).sum().item())


def _require_token_batch_tensors(
    batch: PpoBatch,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Narrow an all-or-none token batch to concrete tensors."""
    old_token_logprobs = batch.old_token_logprobs
    old_prefix_values = batch.old_prefix_values
    token_returns = batch.token_returns
    token_advantages = batch.token_advantages
    token_mask = batch.token_mask
    if (
        old_token_logprobs is None
        or old_prefix_values is None
        or token_returns is None
        or token_advantages is None
        or token_mask is None
    ):
        raise ValueError("token-level PPO tensors must be provided together")
    return (
        old_token_logprobs,
        old_prefix_values,
        token_returns,
        token_advantages,
        token_mask,
    )


def _active_token_objective_tensors(
    evaluation: ActionEvaluation,
    batch: PpoBatch,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return flattened active current/behavior token objective tensors."""
    token_logprobs = evaluation.token_logprobs
    token_entropies = evaluation.token_entropies
    prefix_values = evaluation.prefix_values
    evaluation_mask = evaluation.token_mask
    if (
        token_logprobs is None
        or token_entropies is None
        or prefix_values is None
        or evaluation_mask is None
    ):
        raise RuntimeError("model did not return token-level PPO evidence")
    if (
        token_logprobs.ndim != 2
        or token_entropies.shape != token_logprobs.shape
        or prefix_values.shape != token_logprobs.shape
        or evaluation_mask.shape != token_logprobs.shape
    ):
        raise RuntimeError("model returned misaligned token-level PPO evidence")
    if not batch.integrity_validated:
        _validate_evaluation_token_mask(evaluation_mask, batch)
        _validate_evaluation_stop_semantics(evaluation, batch)

    (
        old_token_logprobs,
        old_prefix_values,
        token_returns,
        token_advantages,
        _,
    ) = _require_token_batch_tensors(batch)
    token_width = token_logprobs.shape[1]
    device = token_logprobs.device
    mask = evaluation_mask.to(device=device)
    old_logprobs = old_token_logprobs[:, :token_width].to(
        device=device,
        dtype=token_logprobs.dtype,
    )
    advantages = token_advantages[:, :token_width].to(
        device=device,
        dtype=token_logprobs.dtype,
    )
    current_values = prefix_values.masked_select(mask)
    if current_values.dtype in (torch.float16, torch.bfloat16):
        current_values = current_values.float()
    old_values = old_prefix_values[:, :token_width].to(
        device=current_values.device,
        dtype=current_values.dtype,
    )
    returns = token_returns[:, :token_width].to(
        device=current_values.device,
        dtype=current_values.dtype,
    )
    return (
        token_logprobs.masked_select(mask),
        token_entropies.masked_select(mask),
        current_values,
        old_logprobs.masked_select(mask),
        old_values.masked_select(mask),
        returns.masked_select(mask),
        advantages.masked_select(mask),
        mask,
    )


def _hybrid_planner_token_objective_tensors(
    evaluation: ActionEvaluation,
    planner: _PlannerModelEvaluation,
    batch: PpoBatch,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Combine fallback token units with planner complete-action units."""
    replay = batch.planner_replay
    if replay is None:
        raise ValueError("hybrid planner PPO requires planner replay evidence")
    token_logprobs = evaluation.token_logprobs
    token_entropies = evaluation.token_entropies
    prefix_values = evaluation.prefix_values
    evaluation_mask = evaluation.token_mask
    if (
        token_logprobs is None
        or token_entropies is None
        or prefix_values is None
        or evaluation_mask is None
    ):
        raise RuntimeError("model did not return token-level PPO evidence")
    if (
        token_logprobs.ndim != 2
        or token_entropies.shape != token_logprobs.shape
        or prefix_values.shape != token_logprobs.shape
        or evaluation_mask.shape != token_logprobs.shape
    ):
        raise RuntimeError("model returned misaligned token-level PPO evidence")
    (
        old_token_logprobs,
        old_prefix_values,
        token_returns,
        token_advantages,
        behavior_mask,
    ) = _require_token_batch_tensors(batch)
    padded_behavior_mask = _validate_hybrid_evaluation_token_mask(
        evaluation_mask,
        behavior_mask,
        batch,
    ).to(device=token_logprobs.device)
    token_width = int(token_logprobs.shape[1])

    def pad(values: Tensor, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        selected = values.to(device=device, dtype=dtype)
        if int(selected.shape[1]) > token_width:
            return selected[:, :token_width]
        if int(selected.shape[1]) == token_width:
            return selected
        return functional.pad(selected, (0, token_width - int(selected.shape[1])))

    fallback_current_logprobs = token_logprobs.masked_select(padded_behavior_mask)
    fallback_entropies = token_entropies.masked_select(padded_behavior_mask)
    fallback_old_logprobs = pad(
        old_token_logprobs,
        dtype=token_logprobs.dtype,
        device=token_logprobs.device,
    ).masked_select(padded_behavior_mask)
    fallback_advantages = pad(
        token_advantages,
        dtype=token_logprobs.dtype,
        device=token_logprobs.device,
    ).masked_select(padded_behavior_mask)
    fallback_values = prefix_values.masked_select(padded_behavior_mask)
    if fallback_values.dtype in (torch.float16, torch.bfloat16):
        fallback_values = fallback_values.float()
    fallback_old_values = pad(
        old_prefix_values,
        dtype=fallback_values.dtype,
        device=fallback_values.device,
    ).masked_select(padded_behavior_mask)
    fallback_returns = pad(
        token_returns,
        dtype=fallback_values.dtype,
        device=fallback_values.device,
    ).masked_select(padded_behavior_mask)

    planner_indices = _planner_decision_indices(
        batch,
        device=evaluation.values.device,
    )
    planner_current_values = evaluation.values.index_select(0, planner_indices)
    planner_old_values = batch.old_values.index_select(
        0, planner_indices.to(device=batch.old_values.device)
    ).to(device=planner_current_values.device, dtype=planner_current_values.dtype)
    planner_returns = batch.returns.index_select(
        0, planner_indices.to(device=batch.returns.device)
    ).to(device=planner_current_values.device, dtype=planner_current_values.dtype)
    planner_advantages = batch.advantages.index_select(
        0, planner_indices.to(device=batch.advantages.device)
    ).to(device=token_logprobs.device, dtype=token_logprobs.dtype)
    planner_old_logprobs = batch.old_action_logprobs.index_select(
        0, planner_indices.to(device=batch.old_action_logprobs.device)
    ).to(device=token_logprobs.device, dtype=token_logprobs.dtype)
    return (
        torch.cat(
            (
                fallback_current_logprobs,
                planner.selected_logprobs.to(
                    device=token_logprobs.device,
                    dtype=token_logprobs.dtype,
                ),
            )
        ),
        torch.cat(
            (
                fallback_entropies,
                planner.entropies.to(
                    device=token_entropies.device,
                    dtype=token_entropies.dtype,
                ),
            )
        ),
        torch.cat((fallback_values, planner_current_values)),
        torch.cat((fallback_old_logprobs, planner_old_logprobs)),
        torch.cat((fallback_old_values, planner_old_values)),
        torch.cat((fallback_returns, planner_returns)),
        torch.cat((fallback_advantages, planner_advantages)),
        evaluation_mask,
    )


def _validate_hybrid_evaluation_token_mask(
    evaluation_mask: Tensor,
    behavior_mask: Tensor,
    batch: PpoBatch,
) -> Tensor:
    """Validate fallback token rows while allowing planner rows no trace."""
    if evaluation_mask.ndim != 2 or evaluation_mask.dtype != torch.bool:
        raise RuntimeError("model token_mask must be a bool [batch, steps] tensor")
    if behavior_mask.ndim != 2 or behavior_mask.dtype != torch.bool:
        raise ValueError("behavior token_mask must be a bool [batch, steps] tensor")
    if int(behavior_mask.shape[0]) != len(batch.actions):
        raise ValueError("behavior token_mask rows must align with actions")
    width = int(evaluation_mask.shape[1])
    aligned = behavior_mask.to(device=evaluation_mask.device)
    if int(aligned.shape[1]) > width:
        if bool(aligned[:, width:].any()):
            raise RuntimeError("model omitted active fallback behavior token steps")
        aligned = aligned[:, :width]
    elif int(aligned.shape[1]) < width:
        aligned = functional.pad(aligned, (0, width - int(aligned.shape[1])))
    planner_indices = _planner_decision_indices(
        batch,
        device=evaluation_mask.device,
    )
    planner_rows = torch.zeros(
        len(batch.actions),
        dtype=torch.bool,
        device=evaluation_mask.device,
    ).scatter(0, planner_indices, True)
    if bool(aligned.index_select(0, planner_indices).any()):
        raise ValueError("planner rows cannot carry autoregressive behavior tokens")
    fallback_rows = ~planner_rows
    if bool(fallback_rows.any()) and not torch.equal(
        evaluation_mask[fallback_rows],
        aligned[fallback_rows],
    ):
        raise RuntimeError(
            "teacher-forced token_mask differs from fallback behavior evidence"
        )
    return aligned


def _validate_evaluation_token_mask(
    evaluation_mask: Tensor,
    batch: PpoBatch,
) -> None:
    """Require teacher-forced active steps to match behavior evidence."""
    if evaluation_mask.ndim != 2 or evaluation_mask.dtype != torch.bool:
        raise RuntimeError("model token_mask must be a bool [batch, steps] tensor")
    batch_mask = batch.token_mask
    if batch_mask is None:
        raise ValueError("token_mask is required for a token objective")
    if evaluation_mask.shape[0] != batch_mask.shape[0]:
        raise RuntimeError("model token_mask batch dimension is misaligned")
    token_width = evaluation_mask.shape[1]
    if token_width > batch_mask.shape[1]:
        raise RuntimeError("model emitted more token steps than behavior evidence")
    aligned_batch_mask = batch_mask[:, :token_width].to(
        device=evaluation_mask.device,
        dtype=torch.bool,
    )
    if not torch.equal(evaluation_mask, aligned_batch_mask):
        raise RuntimeError("teacher-forced token_mask differs from behavior evidence")
    if bool(batch_mask[:, token_width:].any().item()):
        raise RuntimeError("model omitted active behavior token steps")


def _validate_evaluation_stop_semantics(
    evaluation: ActionEvaluation,
    batch: PpoBatch,
) -> None:
    """Distinguish voluntary STOP from max-count-forced termination."""
    stop_sampled = evaluation.stop_sampled
    if stop_sampled is None:
        raise RuntimeError("model did not report token STOP semantics")
    max_counts = batch.options.max_counts.detach().to(device="cpu").tolist()
    expected_stop = torch.tensor(
        [
            len(action) < int(max_count)
            for action, max_count in zip(batch.actions, max_counts, strict=True)
        ],
        dtype=torch.bool,
        device=stop_sampled.device,
    )
    if stop_sampled.shape != expected_stop.shape or not torch.equal(
        stop_sampled,
        expected_stop,
    ):
        raise RuntimeError(
            "teacher-forced STOP semantics differ from behavior evidence"
        )


def _engine_teacher_decision_count(batch: PpoBatch) -> int:
    if batch.integrity_validated and batch.engine_teacher_indices is not None:
        return len(batch.engine_teacher_indices)
    mask = batch.engine_teacher_mask
    return 0 if mask is None else int(mask.detach().sum().item())


def _factual_decision_count(batch: PpoBatch) -> int:
    """Return dense factual rows, rejecting partial tensor presence elsewhere."""
    return 0 if batch.factual_effect_targets is None else len(batch.actions)


def _factual_action_losses(
    evaluation: ActionEvaluation,
    batch: PpoBatch,
) -> tuple[Tensor, Tensor]:
    """Supervise one completed-action latent once per executed decision."""
    presence_logits = evaluation.factual_presence_logits
    magnitude_predictions = evaluation.factual_magnitude_predictions
    actor_relation_logits = evaluation.factual_actor_relation_logits
    next_context_logits = evaluation.factual_next_context_logits
    effect_targets = batch.factual_effect_targets
    actor_relations = batch.factual_actor_relations
    next_contexts = batch.factual_next_contexts
    if (
        presence_logits is None
        or magnitude_predictions is None
        or actor_relation_logits is None
        or next_context_logits is None
    ):
        raise RuntimeError("model did not return factual action predictions")
    if effect_targets is None or actor_relations is None or next_contexts is None:
        raise ValueError("factual batch fields must be provided together")
    batch_size = len(batch.actions)
    effect_shape = (batch_size, DYNAMIC_EFFECT_FEATURE_SIZE)
    if (
        presence_logits.shape != effect_shape
        or magnitude_predictions.shape != effect_shape
    ):
        raise RuntimeError("model factual effect predictions are misaligned")
    if actor_relation_logits.shape != (
        batch_size,
        FACTUAL_ACTOR_RELATION_COUNT,
    ) or next_context_logits.shape != (batch_size, FACTUAL_NEXT_CONTEXT_COUNT):
        raise RuntimeError("model factual successor predictions are misaligned")

    presence_values = presence_logits.float()
    magnitude_values = magnitude_predictions.float()
    target_effects = effect_targets.to(
        device=presence_values.device,
        dtype=presence_values.dtype,
    )
    presence_targets = target_effects.ne(0.0)
    presence_element_loss = functional.binary_cross_entropy_with_logits(
        presence_values,
        presence_targets.to(dtype=presence_values.dtype),
        reduction="none",
    )
    positive_count = presence_targets.sum(dim=1)
    negative_targets = ~presence_targets
    negative_count = negative_targets.sum(dim=1)
    positive_loss = presence_element_loss.masked_fill(~presence_targets, 0.0).sum(
        dim=1
    ) / positive_count.clamp_min(1).to(dtype=presence_element_loss.dtype)
    negative_loss = presence_element_loss.masked_fill(~negative_targets, 0.0).sum(
        dim=1
    ) / negative_count.clamp_min(1).to(dtype=presence_element_loss.dtype)
    present_groups = positive_count.gt(0).to(dtype=presence_element_loss.dtype)
    absent_groups = negative_count.gt(0).to(dtype=presence_element_loss.dtype)
    presence_loss = (
        (positive_loss + negative_loss) / (present_groups + absent_groups)
    ).mean()

    magnitude_indices = torch.tensor(
        DYNAMIC_EFFECT_MAGNITUDE_INDICES,
        dtype=torch.long,
        device=magnitude_values.device,
    )
    predicted_magnitudes = magnitude_values.index_select(1, magnitude_indices)
    target_magnitudes = target_effects.index_select(1, magnitude_indices)
    active_magnitudes = presence_targets.index_select(1, magnitude_indices)
    magnitude_element_loss = functional.smooth_l1_loss(
        predicted_magnitudes,
        target_magnitudes,
        reduction="none",
    )
    magnitude_counts = active_magnitudes.sum(dim=1)
    magnitude_loss = (
        magnitude_element_loss.masked_fill(~active_magnitudes, 0.0).sum(dim=1)
        / magnitude_counts.clamp_min(1).to(dtype=magnitude_element_loss.dtype)
    ).mean()
    effect_loss = presence_loss + magnitude_loss

    relation_loss = functional.cross_entropy(
        actor_relation_logits.float(),
        actor_relations.to(device=actor_relation_logits.device, dtype=torch.long),
    ) / math.log(FACTUAL_ACTOR_RELATION_COUNT)
    context_loss = functional.cross_entropy(
        next_context_logits.float(),
        next_contexts.to(device=next_context_logits.device, dtype=torch.long),
    ) / math.log(FACTUAL_NEXT_CONTEXT_COUNT)
    successor_loss = 0.5 * (relation_loss + context_loss)
    return (
        effect_loss,
        successor_loss,
    )


def _macro_root_count(batch: PpoBatch) -> int:
    mask = batch.macro_mask
    return 0 if mask is None else int(mask.detach().sum().item())


def _macro_action_losses(
    model: AgentPolicyValueNet,
    evaluation: ActionEvaluation,
    batch: PpoBatch,
) -> tuple[Tensor, Tensor]:
    """Supervise root-only realized and behavior-expected macro outcomes."""
    latents = evaluation.completed_action_latents
    effects = batch.macro_effect_targets
    endpoints = batch.macro_endpoints
    summaries = batch.macro_continuation_summaries
    mask = batch.macro_mask
    if latents is None:
        raise RuntimeError("macro loss requires completed-action latents")
    if effects is None or endpoints is None or summaries is None or mask is None:
        raise ValueError("macro batch fields must be provided together")
    predictions = model.macro_outcomes(latents, summaries)
    indices = torch.nonzero(
        mask.to(device=latents.device),
        as_tuple=False,
    ).flatten()
    if int(indices.numel()) <= 0:
        raise ValueError("macro loss requires at least one active root")
    target_effects = effects.to(
        device=latents.device,
        dtype=torch.float32,
    ).index_select(0, indices)
    target_endpoints = endpoints.to(
        device=latents.device,
        dtype=torch.long,
    ).index_select(0, indices)
    conditional = _macro_prediction_loss(
        predictions,
        indices=indices,
        effect_targets=target_effects,
        endpoint_targets=target_endpoints,
        conditional=True,
    )
    expected = _macro_prediction_loss(
        predictions,
        indices=indices,
        effect_targets=target_effects,
        endpoint_targets=target_endpoints,
        conditional=False,
    )
    return conditional, expected


def _macro_prediction_loss(
    prediction: MacroOutcomePrediction,
    *,
    indices: Tensor,
    effect_targets: Tensor,
    endpoint_targets: Tensor,
    conditional: bool,
) -> Tensor:
    """Return balanced effect plus normalized semantic-endpoint loss."""
    if conditional:
        presence = prediction.conditional_presence_logits
        magnitudes = prediction.conditional_magnitude_predictions
        endpoints = prediction.conditional_endpoint_logits
    else:
        presence = prediction.expected_presence_logits
        magnitudes = prediction.expected_magnitude_predictions
        endpoints = prediction.expected_endpoint_logits
    presence = presence.index_select(0, indices).float()
    magnitudes = magnitudes.index_select(0, indices).float()
    endpoint_logits = endpoints.index_select(0, indices).float()
    if (
        presence.shape != effect_targets.shape
        or magnitudes.shape != effect_targets.shape
    ):
        raise RuntimeError("macro effect predictions are misaligned")
    if endpoint_logits.shape != (len(indices), MACRO_ENDPOINT_COUNT):
        raise RuntimeError("macro endpoint predictions are misaligned")
    presence_targets = effect_targets.ne(0.0)
    element_loss = functional.binary_cross_entropy_with_logits(
        presence,
        presence_targets.to(dtype=presence.dtype),
        reduction="none",
    )
    positive_counts = presence_targets.sum(dim=1)
    negative_targets = ~presence_targets
    negative_counts = negative_targets.sum(dim=1)
    positive_loss = element_loss.masked_fill(~presence_targets, 0.0).sum(dim=1) / (
        positive_counts.clamp_min(1).to(dtype=element_loss.dtype)
    )
    negative_loss = element_loss.masked_fill(~negative_targets, 0.0).sum(dim=1) / (
        negative_counts.clamp_min(1).to(dtype=element_loss.dtype)
    )
    group_count = positive_counts.gt(0).to(dtype=element_loss.dtype) + (
        negative_counts.gt(0).to(dtype=element_loss.dtype)
    )
    presence_loss = ((positive_loss + negative_loss) / group_count).mean()
    magnitude_indices = torch.tensor(
        DYNAMIC_EFFECT_MAGNITUDE_INDICES,
        dtype=torch.long,
        device=magnitudes.device,
    )
    predicted_magnitudes = magnitudes.index_select(1, magnitude_indices)
    target_magnitudes = effect_targets.index_select(1, magnitude_indices)
    active = presence_targets.index_select(1, magnitude_indices)
    magnitude_elements = functional.smooth_l1_loss(
        predicted_magnitudes,
        target_magnitudes,
        reduction="none",
    )
    magnitude_loss = (
        magnitude_elements.masked_fill(~active, 0.0).sum(dim=1)
        / active.sum(dim=1).clamp_min(1).to(dtype=magnitude_elements.dtype)
    ).mean()
    endpoint_loss = functional.cross_entropy(
        endpoint_logits,
        endpoint_targets,
    ) / math.log(MACRO_ENDPOINT_COUNT)
    return presence_loss + magnitude_loss + endpoint_loss


def _engine_teacher_sequence_loss(
    model: AgentPolicyValueNet,
    batch: PpoBatch,
    behavior_evaluation: ActionEvaluation,
) -> Tensor:
    """Return weighted sequence NLL normalized by valid engine decisions."""
    actions = batch.engine_teacher_actions
    action_targets = batch.engine_teacher_action_targets
    confidences = batch.engine_teacher_confidences
    weights = batch.engine_teacher_weights
    mask = batch.engine_teacher_mask
    if (
        actions is None
        or action_targets is None
        or confidences is None
        or weights is None
        or mask is None
    ):
        raise ValueError("engine teacher tensors must be provided together")
    valid_count = _engine_teacher_decision_count(batch)
    if valid_count <= 0:
        raise ValueError("engine teacher loss requires at least one valid decision")
    if not batch.integrity_validated or batch.engine_teacher_indices is None:
        active_indices = torch.nonzero(mask, as_tuple=False).flatten()
        cpu_active_indices = tuple(
            int(index) for index in active_indices.detach().cpu().tolist()
        )
    else:
        cpu_active_indices = batch.engine_teacher_indices
        active_indices = torch.tensor(
            cpu_active_indices,
            dtype=torch.long,
        )
    teacher_batch = _select_engine_teacher_rows(
        batch,
        actions=actions,
        action_targets=action_targets,
        active_indices=active_indices,
    )
    policy_context = behavior_evaluation.policy_context
    teacher_context: PolicyEvaluationContext | None = None
    if policy_context is None:
        action_logprobs = model.evaluate_action_logprobs(
            teacher_batch.states,
            teacher_batch.options,
            teacher_batch.actions,
            decks=teacher_batch.decks,
            action_targets=teacher_batch.action_targets,
            temperature=teacher_batch.sampling_temperatures,
            validate_temperature=not batch.integrity_validated,
        )
    else:
        teacher_context = model.select_policy_context(
            policy_context,
            active_indices,
            decks=teacher_batch.decks,
        )
        action_logprobs = model.evaluate_action_logprobs_from_context(
            teacher_context,
            teacher_batch.options,
            teacher_batch.actions,
            action_targets=teacher_batch.action_targets,
            temperature=teacher_batch.sampling_temperatures,
            validate_temperature=not batch.integrity_validated,
        )
    target_device = action_logprobs.device
    target_dtype = action_logprobs.dtype
    evidence_weights = confidences.index_select(
        0,
        active_indices.to(device=confidences.device),
    ).to(device=target_device, dtype=target_dtype) * weights.index_select(
        0,
        active_indices.to(device=weights.device),
    ).to(device=target_device, dtype=target_dtype)
    sequence_nll = -action_logprobs
    combined_nll = _combine_search_candidate_loss(
        model,
        teacher_batch,
        sequence_nll=sequence_nll,
        candidate_actions=(
            None
            if batch.engine_teacher_candidate_actions is None
            else tuple(
                batch.engine_teacher_candidate_actions[index]
                for index in cpu_active_indices
            )
        ),
        candidate_features=(
            None
            if batch.engine_teacher_candidate_features is None
            else tuple(
                batch.engine_teacher_candidate_features[index]
                for index in cpu_active_indices
            )
        ),
        policy_context=teacher_context,
    )
    return (combined_nll * evidence_weights).sum() / valid_count


def _combine_search_candidate_loss(
    model: AgentPolicyValueNet,
    teacher_batch: PpoBatch,
    *,
    sequence_nll: Tensor,
    candidate_actions: Sequence[Sequence[Sequence[int]]] | None,
    candidate_features: Sequence[Tensor] | None,
    policy_context: PolicyEvaluationContext | None,
) -> Tensor:
    """Average base NLL with reranker CE only on complete evidence rows."""
    if candidate_actions is None and candidate_features is None:
        return sequence_nll
    if candidate_actions is None or candidate_features is None:
        raise ValueError("engine teacher search candidate fields must align")
    evidence_rows = tuple(
        index for index, candidates in enumerate(candidate_actions) if candidates
    )
    if not evidence_rows:
        return sequence_nll
    if teacher_batch.action_targets is None:
        raise RuntimeError("teacher action targets are unavailable")
    evidence_indices = torch.tensor(
        evidence_rows,
        dtype=torch.long,
    )
    evidence_batch = _select_engine_teacher_rows(
        teacher_batch,
        actions=teacher_batch.actions,
        action_targets=teacher_batch.action_targets,
        active_indices=evidence_indices,
    )
    selected_actions = tuple(candidate_actions[index] for index in evidence_rows)
    selected_features = tuple(candidate_features[index] for index in evidence_rows)
    if policy_context is None:
        candidate_evaluation = model.evaluate_search_candidates(
            evidence_batch.states,
            evidence_batch.options,
            selected_actions,
            selected_features,
            decks=evidence_batch.decks,
        )
    else:
        evidence_context = model.select_policy_context(
            policy_context,
            evidence_indices,
            decks=evidence_batch.decks,
        )
        candidate_evaluation = model.evaluate_search_candidates_from_context(
            evidence_context,
            evidence_batch.options,
            selected_actions,
            selected_features,
            decks=evidence_batch.decks,
        )
    target_actions = tuple(teacher_batch.actions[index] for index in evidence_rows)
    candidate_losses = torch.stack(
        tuple(
            -functional.log_softmax(logits.float(), dim=0)[
                tuple(tuple(action) for action in candidates).index(tuple(target))
            ]
            for logits, candidates, target in zip(
                torch.split(
                    candidate_evaluation.logits,
                    list(candidate_evaluation.candidate_counts),
                ),
                selected_actions,
                target_actions,
                strict=True,
            )
        )
    ).to(dtype=sequence_nll.dtype)
    dense_candidate_loss = torch.zeros_like(sequence_nll).scatter(
        0,
        evidence_indices.to(device=sequence_nll.device),
        candidate_losses.to(device=sequence_nll.device),
    )
    evidence_mask = torch.zeros_like(sequence_nll, dtype=torch.bool).scatter(
        0,
        evidence_indices.to(device=sequence_nll.device),
        True,
    )
    return torch.where(
        evidence_mask,
        0.5 * (sequence_nll + dense_candidate_loss),
        sequence_nll,
    )


def _select_engine_teacher_rows(
    batch: PpoBatch,
    *,
    actions: Sequence[Sequence[int]],
    action_targets: Tensor,
    active_indices: Tensor,
) -> PpoBatch:
    """Build the auxiliary batch from only rows carrying teacher evidence."""
    cpu_indices = tuple(int(index) for index in active_indices.detach().cpu().tolist())

    def select(tensor: Tensor) -> Tensor:
        return tensor.index_select(0, active_indices.to(device=tensor.device))

    def optional_select(tensor: Tensor | None) -> Tensor | None:
        return None if tensor is None else select(tensor)

    states = batch.states
    selected_states = StateBatch(
        card_ids=select(states.card_ids),
        areas=select(states.areas),
        owner_roles=select(states.owner_roles),
        token_kinds=select(states.token_kinds),
        scalars=select(states.scalars),
        last_attack_ids=select(states.last_attack_ids),
        padding_mask=select(states.padding_mask),
        attachment_card_ids=optional_select(states.attachment_card_ids),
        attachment_parent_indices=optional_select(states.attachment_parent_indices),
        attachment_kinds=optional_select(states.attachment_kinds),
        entity_slots=optional_select(states.entity_slots),
    )
    options = batch.options
    selected_options = OptionBatch(
        option_types=select(options.option_types),
        contexts=select(options.contexts),
        entity_slots=select(options.entity_slots),
        entity_slot_mask=select(options.entity_slot_mask),
        attack_ids=select(options.attack_ids),
        card_ids=select(options.card_ids),
        scalars=select(options.scalars),
        dynamic_effect_features=select(options.dynamic_effect_features),
        dynamic_effect_masks=select(options.dynamic_effect_masks),
        valid_options=select(options.valid_options),
        min_counts=select(options.min_counts),
        max_counts=select(options.max_counts),
    )
    return PpoBatch(
        states=selected_states,
        options=selected_options,
        actions=tuple(actions[index] for index in cpu_indices),
        decks=(
            None
            if batch.decks is None
            else batch.decks.select(
                active_indices.to(device=batch.decks.card_ids.device)
            )
        ),
        old_action_logprobs=select(batch.old_action_logprobs),
        sampling_temperatures=torch.ones_like(select(batch.sampling_temperatures)),
        old_values=select(batch.old_values),
        returns=select(batch.returns),
        advantages=select(batch.advantages),
        action_targets=select(action_targets),
    )


def _evaluate_actions_for_batch(
    model: AgentPolicyValueNet,
    batch: PpoBatch,
) -> ActionEvaluation:
    if batch.public_events is not None:
        offsets, _artifacts = _validated_recurrent_model_batch(model, batch)
        return model.evaluate_recurrent_actions(
            batch.states,
            batch.options,
            batch.actions,
            events=batch.public_events,
            sequence_offsets=offsets,
            decks=batch.decks,
            action_targets=batch.action_targets,
            temperature=batch.sampling_temperatures,
            validate_temperature=not batch.integrity_validated,
            validate_actions=not batch.integrity_validated,
            count_first_rows_present=(
                batch.count_first_rows_present if batch.integrity_validated else None
            ),
        )
    if batch.integrity_validated:
        return model.evaluate_actions(
            batch.states,
            batch.options,
            batch.actions,
            decks=batch.decks,
            action_targets=batch.action_targets,
            temperature=batch.sampling_temperatures,
            validate_temperature=False,
            validate_actions=False,
            count_first_rows_present=batch.count_first_rows_present,
        )
    if batch.action_targets is None:
        return model.evaluate_actions(
            batch.states,
            batch.options,
            batch.actions,
            decks=batch.decks,
            temperature=batch.sampling_temperatures,
        )
    return model.evaluate_actions(
        batch.states,
        batch.options,
        batch.actions,
        decks=batch.decks,
        action_targets=batch.action_targets,
        temperature=batch.sampling_temperatures,
    )


def _evaluate_action_sequences_for_batch(
    model: AgentPolicyValueNet,
    batch: PpoBatch,
) -> ActionEvaluation:
    """Run the trusted sequence-only model path for plain decision PPO."""
    if batch.public_events is not None:
        offsets, _artifacts = _validated_recurrent_model_batch(model, batch)
        return model.evaluate_recurrent_action_sequences(
            batch.states,
            batch.options,
            batch.actions,
            events=batch.public_events,
            sequence_offsets=offsets,
            decks=batch.decks,
            action_targets=batch.action_targets,
            temperature=batch.sampling_temperatures,
            validate_temperature=not batch.integrity_validated,
            validate_actions=not batch.integrity_validated,
            count_first_rows_present=(
                batch.count_first_rows_present if batch.integrity_validated else None
            ),
        )
    return model.evaluate_action_sequences(
        batch.states,
        batch.options,
        batch.actions,
        decks=batch.decks,
        action_targets=batch.action_targets,
        temperature=batch.sampling_temperatures,
        validate_temperature=not batch.integrity_validated,
        validate_actions=not batch.integrity_validated,
        count_first_rows_present=(
            batch.count_first_rows_present if batch.integrity_validated else None
        ),
    )


def _validated_recurrent_model_batch(
    model: AgentPolicyValueNet,
    batch: PpoBatch,
) -> tuple[Tensor, tuple[PolicyArtifactIdentity, ...]]:
    """Reject replay under a model with a different recurrent architecture."""
    offsets = batch.sequence_offsets
    artifacts = batch.sequence_artifacts
    if offsets is None or artifacts is None:
        raise ValueError("recurrent PPO batch is missing sequence identity")
    expected = DistributedModelCompatibility.from_model_config(model.config)
    if any(artifact.compatibility != expected for artifact in artifacts):
        raise ValueError("recurrent trajectory/model compatibility mismatch")
    return offsets, artifacts


def _uses_sequence_only_evaluation(batch: PpoBatch, config: PpoConfig) -> bool:
    """Return whether no enabled objective consumes token or auxiliary traces."""
    return bool(
        batch.token_mask is None
        and batch.planner_replay is None
        and batch.engine_teacher_actions is None
        and batch.factual_effect_targets is None
        and batch.macro_mask is None
        and config.kl_anchor_coef == 0.0
        and config.engine_teacher_coef == 0.0
        and config.factual_effect_coef == 0.0
        and config.factual_successor_coef == 0.0
        and config.macro_conditional_coef == 0.0
        and config.macro_expected_coef == 0.0
        and config.candidate_rerank_coef == 0.0
        and config.proposal_distillation_coef == 0.0
    )


def _ppo_autocast_context(
    batch: PpoBatch,
    config: PpoConfig,
) -> AbstractContextManager[None]:
    if config.autocast == "off":
        return nullcontext()
    device = batch.options.valid_options.device
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _categorical_kl(student_logits: Tensor, anchor_logits: Tensor) -> Tensor:
    if student_logits.dtype in (torch.float16, torch.bfloat16):
        student_logits = student_logits.float()
    if anchor_logits.dtype in (torch.float16, torch.bfloat16):
        anchor_logits = anchor_logits.float()
    student_logprobs = torch.log_softmax(student_logits, dim=1)
    anchor_logprobs = torch.log_softmax(anchor_logits, dim=1)
    student_probs = torch.softmax(student_logits, dim=1)
    finite = torch.isfinite(student_logprobs) & torch.isfinite(anchor_logprobs)
    safe_student_probs = torch.where(
        finite,
        student_probs,
        torch.zeros_like(student_probs),
    )
    safe_student_logprobs = torch.where(
        finite,
        student_logprobs,
        torch.zeros_like(student_logprobs),
    )
    safe_anchor_logprobs = torch.where(
        finite,
        anchor_logprobs,
        torch.zeros_like(anchor_logprobs),
    )
    terms = safe_student_probs * (safe_student_logprobs - safe_anchor_logprobs)
    return terms.sum(dim=1)


def _active_indices_for_step(
    actions: Sequence[Sequence[int]],
    *,
    action_targets: Tensor | None,
    step_index: int,
    device: torch.device,
) -> Tensor:
    if action_targets is not None:
        return _active_indices_for_step_targets(
            action_targets,
            step_index=step_index,
            device=device,
        )
    active = [
        batch_index
        for batch_index, action in enumerate(actions)
        if step_index <= len(action)
    ]
    return torch.tensor(active, dtype=torch.long, device=device)


def _active_indices_for_step_targets(
    action_targets: Tensor,
    *,
    step_index: int,
    device: torch.device,
) -> Tensor:
    if step_index >= action_targets.shape[1]:
        return torch.empty((0,), dtype=torch.long, device=device)
    step_targets = action_targets[:, step_index]
    if step_targets.device != device:
        step_targets = step_targets.to(device=device)
    return torch.nonzero(step_targets.ge(0), as_tuple=False).flatten()


def _zero_like_logits(
    student_step_logits: Sequence[Tensor],
    anchor_step_logits: Sequence[Tensor],
) -> Tensor:
    if student_step_logits:
        return student_step_logits[0].new_zeros(())
    if anchor_step_logits:
        return anchor_step_logits[0].new_zeros(())
    return torch.zeros(())


def _sample_index_values(sample_indices: Tensor) -> list[int]:
    """Return sample indices as Python ints without assuming tensor device."""
    return [int(value) for value in sample_indices.detach().cpu().tolist()]


def _deck_signature_values(decks: DeckBatch | None) -> tuple[str, ...]:
    """Return immutable deck identities for a whole-minibatch cache key."""
    return () if decks is None else decks.signatures


def _aligned_deck_signatures(
    decks: DeckBatch | None,
    batch_size: int,
) -> tuple[str, ...] | None:
    """Return row identities, using an empty legacy identity when decks are absent."""
    if decks is None:
        return ("",) * batch_size
    if len(decks) != batch_size:
        return None
    return decks.signatures


def _clear_non_value_head_gradients(model: AgentPolicyValueNet) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith(
            (
                "value_head.",
                "prefix_value_delta_head.",
                "root_perspective_value_adapter.",
            )
        ):
            parameter.grad = None


def _detached_loss_breakdown(
    breakdown: PpoLossBreakdown,
    *,
    copy_to_cpu: bool = True,
) -> PpoLossBreakdown:
    """Return logging-only loss tensors without autograd history."""
    teacher_loss = breakdown.engine_teacher_loss
    factual_effect_loss = breakdown.factual_effect_loss
    factual_successor_loss = breakdown.factual_successor_loss
    macro_conditional_loss = breakdown.macro_conditional_loss
    macro_expected_loss = breakdown.macro_expected_loss
    candidate_rerank_loss = breakdown.candidate_rerank_loss
    proposal_distillation_loss = breakdown.proposal_distillation_loss
    root_value_loss = breakdown.root_information_value_loss
    return replace(
        breakdown,
        loss=_detach_loss_scalar(breakdown.loss, copy_to_cpu=copy_to_cpu),
        policy_loss=_detach_loss_scalar(
            breakdown.policy_loss,
            copy_to_cpu=copy_to_cpu,
        ),
        value_loss=_detach_loss_scalar(
            breakdown.value_loss,
            copy_to_cpu=copy_to_cpu,
        ),
        entropy_loss=_detach_loss_scalar(
            breakdown.entropy_loss,
            copy_to_cpu=copy_to_cpu,
        ),
        kl_anchor_loss=_detach_loss_scalar(
            breakdown.kl_anchor_loss,
            copy_to_cpu=copy_to_cpu,
        ),
        engine_teacher_loss=(
            None
            if teacher_loss is None
            else _detach_loss_scalar(teacher_loss, copy_to_cpu=copy_to_cpu)
        ),
        factual_effect_loss=(
            None
            if factual_effect_loss is None
            else _detach_loss_scalar(
                factual_effect_loss,
                copy_to_cpu=copy_to_cpu,
            )
        ),
        factual_successor_loss=(
            None
            if factual_successor_loss is None
            else _detach_loss_scalar(
                factual_successor_loss,
                copy_to_cpu=copy_to_cpu,
            )
        ),
        macro_conditional_loss=(
            None
            if macro_conditional_loss is None
            else _detach_loss_scalar(
                macro_conditional_loss,
                copy_to_cpu=copy_to_cpu,
            )
        ),
        macro_expected_loss=(
            None
            if macro_expected_loss is None
            else _detach_loss_scalar(
                macro_expected_loss,
                copy_to_cpu=copy_to_cpu,
            )
        ),
        candidate_rerank_loss=(
            None
            if candidate_rerank_loss is None
            else _detach_loss_scalar(
                candidate_rerank_loss,
                copy_to_cpu=copy_to_cpu,
            )
        ),
        proposal_distillation_loss=(
            None
            if proposal_distillation_loss is None
            else _detach_loss_scalar(
                proposal_distillation_loss,
                copy_to_cpu=copy_to_cpu,
            )
        ),
        root_information_value_loss=(
            None
            if root_value_loss is None
            else _detach_loss_scalar(
                root_value_loss,
                copy_to_cpu=copy_to_cpu,
            )
        ),
    )


def _detach_loss_scalar(value: Tensor, *, copy_to_cpu: bool = True) -> Tensor:
    detached = value.detach()
    return detached.cpu() if copy_to_cpu else detached


def _target_kl_stop_threshold(config: PpoConfig) -> float | None:
    threshold = config.target_kl_early_stop
    if threshold is None:
        return None
    return threshold * config.target_kl_early_stop_multiplier


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return _strip_lightning_model_prefix(value)
        return _strip_lightning_model_prefix(checkpoint)
    raise TypeError("checkpoint must be a state_dict or contain model_state_dict")


def _strip_lightning_model_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(str(key).startswith("model.") for key in state_dict):
        return {
            str(key).removeprefix("model."): value for key, value in state_dict.items()
        }
    return state_dict


def _checkpoint_model_config(checkpoint: Any) -> AgentNetworkConfig | None:
    if not isinstance(checkpoint, dict):
        return None
    for key in ("model_config", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return AgentNetworkConfig.model_validate(value)
    full_config = checkpoint.get("config")
    if isinstance(full_config, dict):
        model_config = full_config.get("model")
        if isinstance(model_config, dict):
            return AgentNetworkConfig.model_validate(model_config)
    return None


def _tensor_float(value: Tensor) -> float:
    return float(value.detach().cpu().item())
