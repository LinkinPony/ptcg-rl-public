"""Behavior-cloning trainer for the pointer policy/value agent."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as functional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor

from ptcg_rl.model.input_schema import policy_input_schema_metadata
from ptcg_rl.model.network import (
    AgentNetworkConfig,
    AgentPolicyValueNet,
)
from ptcg_rl.training.bc_dataset import BCBatch, KaggleStepDataConfig
from ptcg_rl.training.run_config import (
    TrainingRunConfig,
    resolved_training_config_dump,
)


class CheckpointConfig(BaseModel):
    """Checkpoint retention policy for BC training."""

    model_config = ConfigDict(extra="forbid")

    save_runtime_last: bool = True
    save_runtime_best: bool = True
    save_lightning_last: bool = True
    save_lightning_best: bool = False
    lightning_last_every_n_train_steps: int | None = 2000

    @field_validator("lightning_last_every_n_train_steps")
    @classmethod
    def valid_optional_positive_int(cls, value: int | None) -> int | None:
        """Reject non-positive optional step intervals."""
        if value is not None and value <= 0:
            raise ValueError("checkpoint step intervals must be positive when set")
        return value


class LoguruConfig(BaseModel):
    """Loguru sinks for BC training diagnostics."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    level: str = "INFO"
    console_enabled: bool = True
    file_enabled: bool = True
    file_path: Path | None = None
    rotation: str | None = "256 MB"
    retention: int | str | None = 5
    enqueue: bool = True
    serialize: bool = False

    @field_validator("level")
    @classmethod
    def valid_level(cls, value: str) -> str:
        """Reject empty log levels."""
        if not value.strip():
            raise ValueError("log level must be non-empty")
        return value.upper()

    @field_validator("rotation")
    @classmethod
    def valid_optional_non_empty(cls, value: str | None) -> str | None:
        """Reject empty optional loguru strings."""
        if value is not None and not value.strip():
            raise ValueError("optional loguru strings must be non-empty")
        return value


class TensorBoardConfig(BaseModel):
    """TensorBoard logging config for BC training."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    save_dir: Path | None = None
    name: str = "tensorboard"
    version: str | int | None = ""
    default_hp_metric: bool = False

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        """Reject empty TensorBoard experiment names."""
        if not value.strip():
            raise ValueError("tensorboard name must be non-empty")
        return value


class BehaviorCloningConfig(BaseModel):
    """Hydra-backed config for step-level behavior cloning."""

    model_config = ConfigDict(extra="forbid")

    data: KaggleStepDataConfig = KaggleStepDataConfig()
    model: AgentNetworkConfig = AgentNetworkConfig()
    run: TrainingRunConfig = Field(default_factory=TrainingRunConfig)
    output_dir: Path | None = None
    device: str = "auto"
    seed: int = 0
    epochs: int = 1
    learning_rate: float = 3.0e-4
    warmup_steps: int = 0
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float | None = 1.0
    value_loss_weight: float = 0.25
    prize_diff_loss_weight: float = 0.02
    opponent_card_loss_weight: float = 0.05
    opponent_hand_loss_weight: float = 0.05
    chosen_effect_loss_weight: float = 0.05
    max_train_batches_per_epoch: int | None = None
    max_validation_batches: int | None = 128
    log_every_batches: int = 50
    initial_weights_checkpoint: Path | None = None
    evaluate_initial_weights: bool = False
    resume_from_checkpoint: Path | None = None
    auto_resume: bool = True
    accelerator: str = "auto"
    devices: int | str | list[int] = "auto"
    precision: int | str = "32-true"
    enable_progress_bar: bool = True
    checkpoints: CheckpointConfig = CheckpointConfig()
    loguru: LoguruConfig = LoguruConfig()
    tensorboard: TensorBoardConfig = TensorBoardConfig()

    @field_validator("epochs", "log_every_batches")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive counters."""
        if value <= 0:
            raise ValueError("counters must be positive")
        return value

    @field_validator("warmup_steps")
    @classmethod
    def valid_warmup_steps(cls, value: int) -> int:
        """Reject negative warmup step counts."""
        if value < 0:
            raise ValueError("warmup_steps must be non-negative")
        return value

    @field_validator(
        "learning_rate",
        "weight_decay",
        "value_loss_weight",
        "prize_diff_loss_weight",
        "opponent_card_loss_weight",
        "opponent_hand_loss_weight",
        "chosen_effect_loss_weight",
    )
    @classmethod
    def valid_non_negative_float(cls, value: float) -> float:
        """Reject negative optimizer and loss coefficients."""
        if value < 0.0:
            raise ValueError("float coefficients must be non-negative")
        return value

    @field_validator(
        "gradient_clip_norm",
        "max_train_batches_per_epoch",
        "max_validation_batches",
    )
    @classmethod
    def valid_optional_positive(cls, value: float | int | None) -> float | int | None:
        """Reject non-positive optional limits."""
        if value is not None and value <= 0:
            raise ValueError("optional limits must be positive when set")
        return value

    @field_validator("device", "accelerator")
    @classmethod
    def valid_non_empty_string(cls, value: str) -> str:
        """Reject empty trainer device strings."""
        if not value.strip():
            raise ValueError("trainer device strings must be non-empty")
        return value

    @field_validator("devices")
    @classmethod
    def valid_devices(cls, value: int | str | list[int]) -> int | str | list[int]:
        """Reject empty or invalid trainer device settings."""
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("devices must be positive")
            return value
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("devices must be non-empty")
            return value
        if not value:
            raise ValueError("devices list must be non-empty")
        if any(device < 0 for device in value):
            raise ValueError("devices list entries must be non-negative")
        return value

    @model_validator(mode="after")
    def valid_model_widths(self) -> BehaviorCloningConfig:
        """Require compatible model widths and unambiguous checkpoint semantics."""
        if self.model.state_encoder.d_model != self.model.policy.d_model:
            raise ValueError("state_encoder.d_model must match policy.d_model")
        conditioning = self.model.deck_conditioning
        if conditioning is not None and conditioning.enabled:
            raise ValueError(
                "behavior-cloning batches do not carry canonical own-deck identities; "
                "deck conditioning must be trained through the RL pipeline"
            )
        if (
            self.initial_weights_checkpoint is not None
            and self.resume_from_checkpoint is not None
        ):
            raise ValueError(
                "initial_weights_checkpoint and resume_from_checkpoint are mutually "
                "exclusive"
            )
        if self.evaluate_initial_weights and self.initial_weights_checkpoint is None:
            raise ValueError(
                "evaluate_initial_weights requires initial_weights_checkpoint"
            )
        return self


@dataclass(frozen=True)
class LossBreakdown:
    """Loss tensors for one BC batch."""

    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    prize_diff_loss: Tensor
    opponent_card_loss: Tensor
    opponent_hand_loss: Tensor
    chosen_effect_loss: Tensor
    action_accuracy: float
    first_step_accuracy: float
    action_correct: tuple[bool, ...]
    first_step_correct: tuple[bool, ...]
    value_predictions: tuple[float, ...]
    value_targets: tuple[float, ...]
    sample_weight_sum: float
    # Collapse monitors: near-zero values mean the encoder output no longer
    # depends on the input state (see full_h200_10epoch post-mortem).
    global_embedding_std: float
    value_prediction_std: float
    sequence_nll: tuple[float, ...] = ()
    sequence_token_counts: tuple[int, ...] = ()
    sequence_top3_correct: tuple[bool, ...] = ()
    sequence_top5_correct: tuple[bool, ...] = ()


@dataclass
class MetricAccumulator:
    """Running weighted means for training or validation metrics."""

    collect_details: bool = False
    batches: int = 0
    samples: int = 0
    weighted_samples: float = 0.0
    loss_sum: float = 0.0
    policy_loss_sum: float = 0.0
    value_loss_sum: float = 0.0
    prize_diff_loss_sum: float = 0.0
    opponent_card_loss_sum: float = 0.0
    opponent_hand_loss_sum: float = 0.0
    chosen_effect_loss_sum: float = 0.0
    action_correct: float = 0.0
    first_step_correct: float = 0.0
    global_embedding_std_sum: float = 0.0
    value_prediction_std_sum: float = 0.0
    sequence_nll_sum: float = 0.0
    sequence_token_count: int = 0
    sequence_top3_correct: int = 0
    sequence_top5_correct: int = 0
    group_samples: dict[tuple[int, int], int] = field(default_factory=dict)
    group_action_correct: dict[tuple[int, int], int] = field(default_factory=dict)
    group_first_step_correct: dict[tuple[int, int], int] = field(default_factory=dict)
    value_scores: list[float] = field(default_factory=list)
    value_labels: list[float] = field(default_factory=list)

    def update(self, breakdown: LossBreakdown, batch: BCBatch) -> None:
        """Accumulate one batch of scalar metrics."""
        batch_size = len(batch.actions)
        batch_weight = breakdown.sample_weight_sum
        self.batches += 1
        self.samples += batch_size
        self.weighted_samples += batch_weight
        self.loss_sum += _tensor_float(breakdown.loss) * batch_weight
        self.policy_loss_sum += _tensor_float(breakdown.policy_loss) * batch_weight
        self.value_loss_sum += _tensor_float(breakdown.value_loss) * batch_weight
        self.prize_diff_loss_sum += _tensor_float(breakdown.prize_diff_loss) * batch_weight
        self.opponent_card_loss_sum += (
            _tensor_float(breakdown.opponent_card_loss) * batch_weight
        )
        self.opponent_hand_loss_sum += (
            _tensor_float(breakdown.opponent_hand_loss) * batch_weight
        )
        self.chosen_effect_loss_sum += (
            _tensor_float(breakdown.chosen_effect_loss) * batch_weight
        )
        self.action_correct += sum(
            1.0 for correct in breakdown.action_correct if correct
        )
        self.first_step_correct += sum(
            1.0 for correct in breakdown.first_step_correct if correct
        )
        self.global_embedding_std_sum += breakdown.global_embedding_std
        self.value_prediction_std_sum += breakdown.value_prediction_std
        if breakdown.sequence_nll:
            if len(breakdown.sequence_nll) != batch_size:
                raise ValueError("sequence NLL metrics must align with the batch")
            self.sequence_nll_sum += sum(breakdown.sequence_nll)
            self.sequence_token_count += sum(breakdown.sequence_token_counts)
            self.sequence_top3_correct += sum(breakdown.sequence_top3_correct)
            self.sequence_top5_correct += sum(breakdown.sequence_top5_correct)
        if not self.collect_details:
            return
        self.value_scores.extend(breakdown.value_predictions)
        self.value_labels.extend(breakdown.value_targets)
        for select_type, select_context, correct in zip(
            batch.select_types,
            batch.select_contexts,
            breakdown.action_correct,
            strict=True,
        ):
            key = (int(select_type), int(select_context))
            self.group_samples[key] = self.group_samples.get(key, 0) + 1
            if correct:
                self.group_action_correct[key] = (
                    self.group_action_correct.get(key, 0) + 1
                )
        for select_type, select_context, correct in zip(
            batch.select_types,
            batch.select_contexts,
            breakdown.first_step_correct,
            strict=True,
        ):
            key = (int(select_type), int(select_context))
            if correct:
                self.group_first_step_correct[key] = (
                    self.group_first_step_correct.get(key, 0) + 1
                )

    def as_dict(self) -> dict[str, Any]:
        """Return average metrics."""
        loss_denominator = max(1.0, self.weighted_samples)
        sample_denominator = max(1, self.samples)
        result: dict[str, Any] = {
            "batches": self.batches,
            "samples": self.samples,
            "weighted_samples": self.weighted_samples,
            "sample_weight_mean": self.weighted_samples / sample_denominator,
            "loss": self.loss_sum / loss_denominator,
            "policy_loss": self.policy_loss_sum / loss_denominator,
            "value_loss": self.value_loss_sum / loss_denominator,
            "prize_diff_loss": self.prize_diff_loss_sum / loss_denominator,
            "opponent_card_loss": self.opponent_card_loss_sum / loss_denominator,
            "opponent_hand_loss": self.opponent_hand_loss_sum / loss_denominator,
            "chosen_effect_loss": self.chosen_effect_loss_sum / loss_denominator,
            "action_accuracy": self.action_correct / sample_denominator,
            "first_step_accuracy": self.first_step_correct / sample_denominator,
            "action_sequence_nll": self.sequence_nll_sum / sample_denominator,
            "action_token_nll": self.sequence_nll_sum
            / max(1, self.sequence_token_count),
            "action_sequence_top1_accuracy": self.action_correct
            / sample_denominator,
            "action_sequence_top3_accuracy": self.sequence_top3_correct
            / sample_denominator,
            "action_sequence_top5_accuracy": self.sequence_top5_correct
            / sample_denominator,
            "global_embedding_std": (
                self.global_embedding_std_sum / max(1, self.batches)
            ),
            "value_prediction_std": (
                self.value_prediction_std_sum / max(1, self.batches)
            ),
        }
        if self.collect_details:
            result["value_auc"] = _binary_value_auc(
                self.value_scores,
                self.value_labels,
            )
            result["action_accuracy_by_select"] = self._group_metrics()
        return result

    def _group_metrics(self) -> list[dict[str, float | int]]:
        """Return per-select-context action accuracy rows."""
        rows: list[dict[str, float | int]] = []
        for key, samples in self.group_samples.items():
            select_type, select_context = key
            action_correct = self.group_action_correct.get(key, 0)
            first_step_correct = self.group_first_step_correct.get(key, 0)
            rows.append(
                {
                    "select_type": select_type,
                    "select_context": select_context,
                    "samples": samples,
                    "action_accuracy": action_correct / max(1, samples),
                    "first_step_accuracy": first_step_correct / max(1, samples),
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                -int(row["samples"]),
                int(row["select_type"]),
                int(row["select_context"]),
            ),
        )


def run_behavior_cloning(config: BehaviorCloningConfig) -> dict[str, Any]:
    """Train the pointer policy/value model from extracted Kaggle steps."""
    from ptcg_rl.training.bc_lightning import run_lightning_behavior_cloning

    return run_lightning_behavior_cloning(config)


def behavior_cloning_loss(
    model: AgentPolicyValueNet,
    batch: BCBatch,
    *,
    value_loss_weight: float,
    prize_diff_loss_weight: float,
    opponent_card_loss_weight: float,
    opponent_hand_loss_weight: float,
    chosen_effect_loss_weight: float,
) -> LossBreakdown:
    """Return total and component losses for one collated BC batch."""
    encoded_state = model.state_encoder(
        batch.states,
        attack_embedding=model.policy_head.attack_embedding,
    )
    option_embeddings = model.policy_head.option_embeddings(
        encoded_state.token_embeddings,
        batch.options,
        model.card_encoder,
    )
    (
        policy_loss,
        first_logits,
        action_correct,
        sequence_nll,
        sequence_token_counts,
        sequence_top3_correct,
        sequence_top5_correct,
    ) = _autoregressive_policy_loss(
        model,
        batch,
        global_embedding=encoded_state.global_embedding,
        option_embeddings=option_embeddings,
    )
    value = cast(Tensor, model.value_head(encoded_state.global_embedding)).squeeze(-1)
    prize_diff = cast(
        Tensor,
        model.prize_diff_head(encoded_state.global_embedding),
    ).squeeze(-1)

    value_loss = _weighted_mse_loss(value, batch.value_targets, batch.sample_weights)
    prize_diff_loss = _masked_weighted_mse_loss(
        prize_diff,
        batch.prize_diff_targets,
        batch.prize_diff_mask,
        batch.sample_weights,
    )
    opponent_card_logits = model.opponent_card_head(encoded_state.global_embedding)
    opponent_card_loss = _masked_weighted_soft_ce_loss(
        opponent_card_logits,
        batch.opponent_card_targets,
        batch.opponent_card_target_mask,
        batch.sample_weights,
    )
    opponent_hand_logits = model.opponent_hand_head(encoded_state.global_embedding)
    opponent_hand_loss = _masked_weighted_candidate_soft_ce_loss(
        opponent_hand_logits,
        batch.opponent_hand_targets,
        batch.opponent_hand_target_mask,
        batch.opponent_hand_candidate_mask,
        batch.sample_weights,
    )
    effect_predictions = model.effect_head(option_embeddings)
    chosen_effect_prediction = _selected_action_embedding(
        effect_predictions,
        batch.actions,
        batch.options.valid_options,
    )
    chosen_effect_loss = _masked_weighted_mse_loss(
        chosen_effect_prediction,
        batch.chosen_effect_targets,
        batch.chosen_effect_mask,
        batch.sample_weights,
    )
    total = (
        policy_loss
        + value_loss * value_loss_weight
        + prize_diff_loss * prize_diff_loss_weight
        + opponent_card_loss * opponent_card_loss_weight
        + opponent_hand_loss * opponent_hand_loss_weight
        + chosen_effect_loss * chosen_effect_loss_weight
        + model.card_encoder.embedding_l2_loss()
    )
    return LossBreakdown(
        loss=total,
        policy_loss=policy_loss,
        value_loss=value_loss,
        prize_diff_loss=prize_diff_loss,
        opponent_card_loss=opponent_card_loss,
        opponent_hand_loss=opponent_hand_loss,
        chosen_effect_loss=chosen_effect_loss,
        action_accuracy=_mean_correct(action_correct),
        first_step_accuracy=_first_step_accuracy(first_logits, batch.actions),
        action_correct=action_correct,
        first_step_correct=_first_step_correct(first_logits, batch.actions),
        value_predictions=_tensor_values(value),
        value_targets=_tensor_values(batch.value_targets),
        sample_weight_sum=float(batch.sample_weights.detach().sum().cpu().item()),
        global_embedding_std=_batch_feature_std(encoded_state.global_embedding),
        value_prediction_std=_batch_feature_std(value.unsqueeze(-1)),
        sequence_nll=sequence_nll,
        sequence_token_counts=sequence_token_counts,
        sequence_top3_correct=sequence_top3_correct,
        sequence_top5_correct=sequence_top5_correct,
    )


def _batch_feature_std(features: Tensor) -> float:
    """Return the mean per-dimension standard deviation across the batch."""
    if features.shape[0] < 2:
        return 0.0
    return float(features.detach().float().std(dim=0).mean().cpu().item())


def _autoregressive_policy_loss(
    model: AgentPolicyValueNet,
    batch: BCBatch,
    *,
    global_embedding: Tensor,
    option_embeddings: Tensor,
) -> tuple[
    Tensor,
    Tensor,
    tuple[bool, ...],
    tuple[float, ...],
    tuple[int, ...],
    tuple[bool, ...],
    tuple[bool, ...],
]:
    action_correct = torch.ones(
        len(batch.actions),
        dtype=torch.bool,
        device=batch.options.valid_options.device,
    )
    losses: list[Tensor] = []
    target_weight = option_embeddings.new_zeros(())
    evaluation = model.policy_head.teacher_forced_decode(
        global_embedding,
        option_embeddings,
        batch.options,
        batch.actions,
    )
    sequence_nll = _tensor_values(-evaluation.action_logprobs)
    sequence_token_counts = [0] * len(batch.actions)
    sequence_top3 = torch.ones_like(action_correct)
    sequence_top5 = torch.ones_like(action_correct)

    for step in evaluation.steps:
        active_indices = step.active_indices
        targets = step.targets
        predictions = step.logits.argmax(dim=1).index_select(0, active_indices)
        action_correct[active_indices] &= predictions.eq(targets)
        for row_index in active_indices.detach().cpu().tolist():
            sequence_token_counts[int(row_index)] += 1
        _update_sequence_topk(
            sequence_top3,
            logits=step.logits,
            active_indices=active_indices,
            targets=targets,
            top_k=3,
        )
        _update_sequence_topk(
            sequence_top5,
            logits=step.logits,
            active_indices=active_indices,
            targets=targets,
            top_k=5,
        )
        active_weights = batch.sample_weights.index_select(0, active_indices).to(
            dtype=option_embeddings.dtype,
        )
        step_losses = functional.cross_entropy(
            step.logits.index_select(0, active_indices),
            targets,
            reduction="none",
        )
        losses.append((step_losses * active_weights).sum())
        target_weight = target_weight + active_weights.sum()

    action_correct_tuple = _bool_tensor_values(action_correct)
    if not losses or float(target_weight.detach().cpu().item()) <= 0.0:
        return (
            option_embeddings.sum() * 0.0,
            evaluation.first_logits,
            action_correct_tuple,
            sequence_nll,
            tuple(sequence_token_counts),
            _bool_tensor_values(sequence_top3),
            _bool_tensor_values(sequence_top5),
        )
    return (
        torch.stack(losses).sum() / target_weight,
        evaluation.first_logits,
        action_correct_tuple,
        sequence_nll,
        tuple(sequence_token_counts),
        _bool_tensor_values(sequence_top3),
        _bool_tensor_values(sequence_top5),
    )


def _update_sequence_topk(
    correct: Tensor,
    *,
    logits: Tensor,
    active_indices: Tensor,
    targets: Tensor,
    top_k: int,
) -> None:
    """Update per-sequence membership for one teacher-forced decode step."""
    active_logits = logits.index_select(0, active_indices)
    effective_k = min(top_k, int(active_logits.shape[1]))
    predictions = active_logits.topk(effective_k, dim=1).indices
    target_in_topk = predictions.eq(targets.unsqueeze(1)).any(dim=1)
    correct[active_indices] &= target_in_topk


def _first_step_accuracy(logits: Tensor, actions: Sequence[Sequence[int]]) -> float:
    correct = _first_step_correct(logits, actions)
    return _mean_correct(correct)


def _first_step_correct(
    logits: Tensor,
    actions: Sequence[Sequence[int]],
) -> tuple[bool, ...]:
    if not actions:
        return ()
    stop_index = logits.shape[1] - 1
    predictions = logits.argmax(dim=1).detach().cpu()
    correct: list[bool] = []
    for batch_index, action in enumerate(actions):
        target = int(action[0]) if action else stop_index
        correct.append(int(predictions[batch_index].item()) == target)
    return tuple(correct)


def _weighted_mse_loss(prediction: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    squared_error = (prediction - target).pow(2)
    sample_losses = (
        squared_error.flatten(start_dim=1).mean(dim=1)
        if squared_error.ndim > 1
        else squared_error
    )
    if tuple(sample_losses.shape) != tuple(weights.shape):
        raise ValueError(
            "weights must align with the leading prediction dimension: "
            f"{tuple(weights.shape)} vs {tuple(prediction.shape)}"
        )
    active_weights = weights.to(dtype=sample_losses.dtype)
    weight_sum = active_weights.sum()
    if float(weight_sum.detach().cpu().item()) <= 0.0:
        return prediction.sum() * 0.0
    return (sample_losses * active_weights).sum() / weight_sum


def _masked_weighted_mse_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    weights: Tensor,
) -> Tensor:
    if not bool(mask.any().item()):
        return prediction.sum() * 0.0
    return _weighted_mse_loss(prediction[mask], target[mask], weights[mask])


def _masked_weighted_soft_ce_loss(
    logits: Tensor,
    target_distribution: Tensor,
    mask: Tensor,
    weights: Tensor,
) -> Tensor:
    if not bool(mask.any().item()):
        return logits.sum() * 0.0
    log_probs = functional.log_softmax(logits[mask], dim=-1)
    losses = -(target_distribution[mask] * log_probs).sum(dim=-1)
    active_weights = weights[mask].to(dtype=losses.dtype)
    weight_sum = active_weights.sum()
    if float(weight_sum.detach().cpu().item()) <= 0.0:
        return logits.sum() * 0.0
    return (losses * active_weights).sum() / weight_sum


def _masked_weighted_candidate_soft_ce_loss(
    logits: Tensor,
    target_distribution: Tensor,
    mask: Tensor,
    candidate_mask: Tensor,
    weights: Tensor,
) -> Tensor:
    if not bool(mask.any().item()):
        return logits.sum() * 0.0
    active_logits = logits[mask]
    active_targets = target_distribution[mask]
    active_candidates = candidate_mask[mask] | active_targets.gt(0.0)
    row_has_candidate = active_candidates.any(dim=-1)
    if not bool(row_has_candidate.all().item()):
        active_candidates = torch.where(
            row_has_candidate.unsqueeze(-1),
            active_candidates,
            torch.ones_like(active_candidates),
        )
    masked_logits = active_logits.masked_fill(
        ~active_candidates,
        torch.finfo(active_logits.dtype).min,
    )
    log_probs = functional.log_softmax(masked_logits, dim=-1)
    losses = -(active_targets * log_probs).sum(dim=-1)
    active_weights = weights[mask].to(dtype=losses.dtype)
    weight_sum = active_weights.sum()
    if float(weight_sum.detach().cpu().item()) <= 0.0:
        return logits.sum() * 0.0
    return (losses * active_weights).sum() / weight_sum


def _selected_action_embedding(
    values: Tensor,
    actions: Sequence[Sequence[int]],
    valid_options: Tensor,
) -> Tensor:
    rows: list[Tensor] = []
    max_options = int(valid_options.shape[1])
    for batch_index, action in enumerate(actions):
        valid_indices = [
            int(index)
            for index in action
            if 0 <= int(index) < max_options
            and bool(valid_options[batch_index, int(index)].item())
        ]
        if not valid_indices:
            rows.append(values[batch_index].sum(dim=0) * 0.0)
            continue
        indices = torch.tensor(
            valid_indices,
            dtype=torch.long,
            device=values.device,
        )
        rows.append(values[batch_index].index_select(0, indices).mean(dim=0))
    if not rows:
        return values.new_zeros((0, values.shape[-1]))
    return torch.stack(rows)


def _save_checkpoint(
    path: Path,
    *,
    model: AgentPolicyValueNet,
    config: BehaviorCloningConfig,
    epoch: int,
    metrics: Mapping[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.model.model_dump(mode="json"),
            "policy_input_schema": policy_input_schema_metadata(),
            "training_config": resolved_training_config_dump(
                config,
                task_name="bc",
                run=config.run,
                output_dir=config.output_dir,
            ),
            "epoch": epoch,
            "metrics": dict(metrics),
        },
        path,
    )


def _metric_loss(metrics: Mapping[str, Any]) -> float | None:
    samples = int(metrics.get("samples", 0))
    if samples <= 0:
        return None
    return float(metrics["loss"])


def _binary_value_auc(scores: Sequence[float], targets: Sequence[float]) -> float | None:
    """Return ROC AUC for win/loss value scores, ignoring draw targets."""
    pairs: list[tuple[float, bool]] = []
    for score, target in zip(scores, targets, strict=True):
        if target > 0.0:
            pairs.append((float(score), True))
        elif target < 0.0:
            pairs.append((float(score), False))
    positive_count = sum(1 for _, is_positive in pairs if is_positive)
    negative_count = len(pairs) - positive_count
    if positive_count <= 0 or negative_count <= 0:
        return None

    pairs.sort(key=lambda item: item[0])
    rank_sum_positive = 0.0
    rank = 1
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (rank + rank + (end - index) - 1) / 2.0
        for _, is_positive in pairs[index:end]:
            if is_positive:
                rank_sum_positive += average_rank
        rank += end - index
        index = end

    return (
        rank_sum_positive - positive_count * (positive_count + 1) / 2.0
    ) / float(positive_count * negative_count)


def _tensor_float(value: Tensor) -> float:
    return float(value.detach().cpu().item())


def _tensor_values(value: Tensor) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().cpu().flatten().tolist())


def _bool_tensor_values(value: Tensor) -> tuple[bool, ...]:
    return tuple(bool(item) for item in value.detach().cpu().flatten().tolist())


def _mean_correct(values: Sequence[bool]) -> float:
    if not values:
        return 0.0
    return sum(1.0 for value in values if value) / float(len(values))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
