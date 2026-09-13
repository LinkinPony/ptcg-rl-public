"""Deck-macro decode-token PPO objective and route-aware microbatches."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Self, TypeAlias

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor

from ptcg_rl.model.tensor_validation import require_tensor_condition

StatelessLearnerPrecision: TypeAlias = Literal["fp32", "bf16"]


class SimpleStatelessPpoConfig(BaseModel):
    """Unique clean-lineage optimizer and PPO mechanics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    gamma: float = Field(default=1.0, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    normalize_epsilon: float = Field(default=1.0e-8, gt=0.0)
    clip_epsilon: float = Field(default=0.2, gt=0.0, lt=1.0)
    value_clip_epsilon: float = Field(default=0.2, gt=0.0)
    value_coefficient: float = Field(default=0.5, ge=0.0)
    entropy_coefficient: float = Field(default=0.01, ge=0.0)
    behavior_temperature: float = Field(default=1.0, ge=1.0, le=1.0)
    learning_rate: float = Field(gt=0.0)
    minimum_learning_rate_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    warmup_updates: int = Field(ge=0)
    total_updates: int = Field(gt=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    adam_beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, gt=0.0, lt=1.0)
    adam_epsilon: float = Field(default=1.0e-8, gt=0.0)
    maximum_gradient_norm: float = Field(default=1.0, gt=0.0)
    microbatch_decisions: int = Field(gt=0)
    homogeneous_min_decisions: int = Field(gt=0)
    logical_batch_decisions: int | None = Field(default=None, gt=0)
    ppo_epochs: int = Field(default=1, gt=0)
    target_kl: float | None = Field(default=None, gt=0.0)
    warmup_decisions: int | None = Field(default=None, ge=0)
    total_decisions: int | None = Field(default=None, gt=0)
    maximum_version_age: int = Field(ge=0)

    @field_validator(
        "gamma",
        "gae_lambda",
        "normalize_epsilon",
        "clip_epsilon",
        "value_clip_epsilon",
        "value_coefficient",
        "entropy_coefficient",
        "learning_rate",
        "minimum_learning_rate_ratio",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "maximum_gradient_norm",
        "behavior_temperature",
    )
    @classmethod
    def finite_values(cls, value: float) -> float:
        """Reject NaN and infinite optimizer values."""
        if not math.isfinite(value):
            raise ValueError("stateless PPO values must be finite")
        return value

    @field_validator("target_kl")
    @classmethod
    def finite_optional_values(cls, value: float | None) -> float | None:
        """Reject a non-finite optional KL observation target."""
        if value is not None and not math.isfinite(value):
            raise ValueError("stateless PPO target KL must be finite")
        return value

    @model_validator(mode="after")
    def coherent_microbatches(self) -> Self:
        """Keep optimizer batching and schedule choices internally coherent."""
        if self.homogeneous_min_decisions > self.microbatch_decisions:
            raise ValueError("homogeneous threshold cannot exceed microbatch decisions")
        decision_schedule = (
            self.warmup_decisions is not None,
            self.total_decisions is not None,
        )
        if decision_schedule[0] != decision_schedule[1]:
            raise ValueError(
                "warmup_decisions and total_decisions must be configured together"
            )
        if (
            self.warmup_decisions is not None
            and self.total_decisions is not None
            and self.warmup_decisions > self.total_decisions
        ):
            raise ValueError("warmup decisions cannot exceed total decisions")
        return self

    @property
    def fingerprint(self) -> str:
        """Return the exact optimizer/objective configuration identity."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(
            b"ptcg-rl/simple-stateless-ppo-config/v1\x00" + payload
        ).hexdigest()


@dataclass(frozen=True)
class StatelessMicrobatch:
    """Decision indices sharing one forward/backward accumulation unit."""

    indices: tuple[int, ...]
    deck_digests: tuple[str, ...]
    homogeneous: bool

    def __post_init__(self) -> None:
        """Reject empty, duplicate, or falsely homogeneous batches."""
        if not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError(
                "stateless microbatch indices must be non-empty and unique"
            )
        if not self.deck_digests:
            raise ValueError("stateless microbatch must name at least one deck")
        if self.homogeneous != (len(self.deck_digests) == 1):
            raise ValueError("stateless microbatch homogeneity flag is invalid")


@dataclass(frozen=True)
class StatelessLogicalBatch:
    """Global decision indices consumed by one optimizer step."""

    indices: tuple[int, ...]
    deck_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject empty, duplicate, or unnamed optimizer batches."""
        if not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError("stateless logical batch indices must be unique")
        if not self.deck_digests:
            raise ValueError("stateless logical batch must name at least one deck")


def schedule_stateless_logical_batches(
    deck_digests: Sequence[str],
    *,
    target_decisions: int | None,
) -> tuple[StatelessLogicalBatch, ...]:
    """Split one window into deterministic deck-stratified optimizer batches.

    ``None`` deliberately retains the original one-step-per-window semantics.
    Otherwise each exact deck is distributed round-robin over the minimum
    number of batches needed to respect the target on average. Starting each
    deck at the currently shortest batch prevents all deck remainders from
    accumulating in the first batch.
    """
    if not deck_digests:
        raise ValueError("logical batch scheduling requires decisions")
    if target_decisions is None:
        return (
            StatelessLogicalBatch(
                indices=tuple(range(len(deck_digests))),
                deck_digests=tuple(sorted(set(deck_digests))),
            ),
        )
    if target_decisions <= 0:
        raise ValueError("logical batch target must be positive")
    batch_count = math.ceil(len(deck_digests) / target_decisions)
    groups: list[list[int]] = [[] for _ in range(batch_count)]
    indices_by_deck: defaultdict[str, list[int]] = defaultdict(list)
    for index, digest in enumerate(deck_digests):
        indices_by_deck[digest].append(index)
    for digest in sorted(indices_by_deck):
        start = min(range(batch_count), key=lambda index: (len(groups[index]), index))
        for offset, index in enumerate(indices_by_deck[digest]):
            groups[(start + offset) % batch_count].append(index)
    batches = tuple(
        StatelessLogicalBatch(
            indices=tuple(sorted(indices)),
            deck_digests=tuple(sorted({deck_digests[index] for index in indices})),
        )
        for indices in groups
    )
    covered = tuple(index for batch in batches for index in batch.indices)
    if sorted(covered) != list(range(len(deck_digests))):
        raise RuntimeError("logical batch schedule did not cover every decision once")
    return batches


def schedule_stateless_microbatches(
    deck_digests: Sequence[str],
    *,
    maximum_decisions: int,
    homogeneous_min_decisions: int,
    family_ids: Sequence[str] | None = None,
) -> tuple[StatelessMicrobatch, ...]:
    """Pack route-contiguous deck chunks into large mixed microbatches.

    Rows are first grouped into per-deck chunks (sparse route tails mix as
    before), then whole chunks are packed first-fit into bins bounded by
    ``maximum_decisions``. The window loss is a row-weighted sum, so gradient
    accumulation is invariant to this partitioning; larger microbatches only
    reduce per-launch overhead. Rows of one deck stay contiguous inside a bin,
    which keeps the exact-route dispatch to one group per deck.
    """
    if not deck_digests:
        raise ValueError("microbatch scheduling requires decisions")
    if maximum_decisions <= 0 or not 0 < homogeneous_min_decisions <= maximum_decisions:
        raise ValueError("microbatch size and homogeneous threshold are invalid")
    if family_ids is not None and len(family_ids) != len(deck_digests):
        raise ValueError("family IDs must align with microbatch deck rows")
    if family_ids is not None:
        return _schedule_family_microbatches(
            deck_digests,
            family_ids,
            maximum_decisions=maximum_decisions,
        )
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, digest in enumerate(deck_digests):
        groups[digest].append(index)
    chunks: list[tuple[int, ...]] = []
    sparse: list[int] = []
    for digest in sorted(groups):
        indices = groups[digest]
        if len(indices) < homogeneous_min_decisions:
            sparse.extend(indices)
            continue
        for start in range(0, len(indices), maximum_decisions):
            chunks.append(tuple(indices[start : start + maximum_decisions]))
    for start in range(0, len(sparse), maximum_decisions):
        chunks.append(tuple(sparse[start : start + maximum_decisions]))
    bins: list[list[int]] = []
    for chunk in chunks:
        placed = False
        for bin_indices in bins:
            if len(bin_indices) + len(chunk) <= maximum_decisions:
                bin_indices.extend(chunk)
                placed = True
                break
        if not placed:
            bins.append(list(chunk))
    batches: list[StatelessMicrobatch] = []
    for bin_indices in bins:
        chunk_decks = tuple(sorted({deck_digests[index] for index in bin_indices}))
        batches.append(
            StatelessMicrobatch(
                indices=tuple(bin_indices),
                deck_digests=chunk_decks,
                homogeneous=len(chunk_decks) == 1,
            )
        )
    covered = tuple(index for batch in batches for index in batch.indices)
    if sorted(covered) != list(range(len(deck_digests))):
        raise RuntimeError("microbatch schedule did not cover every decision once")
    return tuple(batches)


def _schedule_family_microbatches(
    deck_digests: Sequence[str],
    family_ids: Sequence[str],
    *,
    maximum_decisions: int,
) -> tuple[StatelessMicrobatch, ...]:
    """Merge family rows first, then pack whole family chunks into VRAM bins."""
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, family_id in enumerate(family_ids):
        if not family_id:
            raise ValueError("family-aware microbatch rows require family IDs")
        groups[family_id].append(index)
    chunks: list[tuple[str, tuple[int, ...]]] = []
    for family_id in sorted(groups):
        indices = groups[family_id]
        for start in range(0, len(indices), maximum_decisions):
            chunks.append(
                (
                    family_id,
                    tuple(indices[start : start + maximum_decisions]),
                )
            )
    chunks.sort(key=lambda item: (-len(item[1]), item[0], item[1][0]))
    bins: list[list[int]] = []
    for _family_id, chunk in chunks:
        for bin_indices in bins:
            if len(bin_indices) + len(chunk) <= maximum_decisions:
                bin_indices.extend(chunk)
                break
        else:
            bins.append(list(chunk))
    batches = tuple(
        StatelessMicrobatch(
            indices=tuple(bin_indices),
            deck_digests=tuple(
                sorted({deck_digests[index] for index in bin_indices})
            ),
            homogeneous=(
                len({deck_digests[index] for index in bin_indices}) == 1
            ),
        )
        for bin_indices in bins
    )
    covered = tuple(index for batch in batches for index in batch.indices)
    if sorted(covered) != list(range(len(deck_digests))):
        raise RuntimeError("family-aware schedule did not cover every decision once")
    return batches


@dataclass(frozen=True)
class StatelessPpoLossInputs:
    """Aligned current evaluation, behavior evidence, targets, and macro weights."""

    current_token_logprobs: Tensor
    current_token_entropies: Tensor
    current_prefix_values: Tensor
    token_mask: Tensor
    old_token_logprobs: Tensor
    old_prefix_values: Tensor
    token_advantages: Tensor
    token_returns: Tensor
    current_root_values: Tensor
    old_root_values: Tensor
    root_returns: Tensor
    decision_macro_weights: Tensor
    belief_row_losses: Tensor
    belief_valid_mask: Tensor
    belief_macro_weights: Tensor


@dataclass(frozen=True)
class StatelessPpoLoss:
    """Differentiable partial-window loss and additive diagnostics."""

    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy_loss: Tensor
    belief_loss: Tensor
    ratio_sum: Tensor
    approximate_kl_sum: Tensor
    clipped_token_count: Tensor
    active_token_count: Tensor


def stateless_deck_macro_ppo_loss(
    inputs: StatelessPpoLossInputs,
    config: SimpleStatelessPpoConfig,
) -> StatelessPpoLoss:
    """Apply token clipping, decision reduction, then exact deck-macro weights."""
    _validate_loss_inputs(inputs)
    mask = inputs.token_mask
    token_count = mask.sum(dim=1).clamp_min(1)
    current_logprobs = inputs.current_token_logprobs.float()
    old_logprobs = inputs.old_token_logprobs.to(
        device=current_logprobs.device,
        dtype=current_logprobs.dtype,
    )
    advantages = inputs.token_advantages.to(
        device=current_logprobs.device,
        dtype=current_logprobs.dtype,
    )
    log_ratio = (current_logprobs - old_logprobs).clamp(
        min=-20.0,
        max=20.0,
    )
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(
        min=1.0 - config.clip_epsilon,
        max=1.0 + config.clip_epsilon,
    )
    token_policy = -torch.minimum(
        ratio * advantages,
        clipped_ratio * advantages,
    )
    policy_rows = (token_policy * mask).sum(dim=1)
    entropy_rows = -(inputs.current_token_entropies.float() * mask).sum(dim=1)

    root_losses = _clipped_value_losses(
        inputs.current_root_values,
        inputs.old_root_values,
        inputs.root_returns,
        clip_epsilon=config.value_clip_epsilon,
    )
    prefix_losses = _clipped_value_losses(
        inputs.current_prefix_values,
        inputs.old_prefix_values,
        inputs.token_returns,
        clip_epsilon=config.value_clip_epsilon,
    )
    prefix_rows = (prefix_losses * mask).sum(dim=1) / token_count
    value_rows = 0.5 * (root_losses + prefix_rows)

    decision_weights = inputs.decision_macro_weights.to(
        device=current_logprobs.device,
        dtype=current_logprobs.dtype,
    )
    belief_weights = inputs.belief_macro_weights.to(
        device=current_logprobs.device,
        dtype=current_logprobs.dtype,
    )
    policy_loss = (policy_rows * decision_weights).sum()
    value_loss = (value_rows * decision_weights).sum()
    entropy_loss = (entropy_rows * decision_weights).sum()
    belief_loss = (inputs.belief_row_losses.float() * belief_weights).sum()
    loss = (
        policy_loss
        + config.value_coefficient * value_loss
        + config.entropy_coefficient * entropy_loss
        + belief_loss
    )
    active_ratio = ratio.masked_select(mask)
    active_log_ratio = log_ratio.masked_select(mask)
    return StatelessPpoLoss(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        belief_loss=belief_loss,
        ratio_sum=active_ratio.sum(),
        approximate_kl_sum=((active_ratio - 1.0) - active_log_ratio).sum(),
        clipped_token_count=((active_ratio - 1.0).abs().gt(config.clip_epsilon).sum()),
        active_token_count=mask.sum(),
    )


def stateless_learning_rate(
    config: SimpleStatelessPpoConfig,
    *,
    update_index: int,
    lr_schedule_decisions_seen: int | None = None,
) -> float:
    """Return the warmup-plus-cosine AdamW learning rate.

    The optional decision schedule consumes an explicit schedule cursor, which
    may include a documented migration anchor. It remains distinct from the
    lineage's count of newly observed samples. With no decision schedule
    configured, the original update-index schedule is bit-for-bit unchanged.
    """
    if update_index < 0:
        raise ValueError("update index cannot be negative")
    if config.total_decisions is not None:
        if lr_schedule_decisions_seen is None:
            raise ValueError(
                "lr_schedule_decisions_seen is required by the decision LR schedule"
            )
        if lr_schedule_decisions_seen < 0:
            raise ValueError("LR schedule decision cursor cannot be negative")
        warmup = int(config.warmup_decisions or 0)
        total = config.total_decisions
        if warmup > 0 and lr_schedule_decisions_seen < warmup:
            multiplier = float(lr_schedule_decisions_seen) / float(warmup)
        else:
            decay_decisions = max(total - warmup, 1)
            progress = min(
                max(lr_schedule_decisions_seen - warmup, 0) / float(decay_decisions),
                1.0,
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            multiplier = (
                config.minimum_learning_rate_ratio
                + (1.0 - config.minimum_learning_rate_ratio) * cosine
            )
    elif config.warmup_updates > 0 and update_index < config.warmup_updates:
        multiplier = float(update_index + 1) / float(config.warmup_updates)
    else:
        decay_updates = max(config.total_updates - config.warmup_updates, 1)
        progress = min(
            max(update_index - config.warmup_updates, 0) / float(decay_updates),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        multiplier = (
            config.minimum_learning_rate_ratio
            + (1.0 - config.minimum_learning_rate_ratio) * cosine
        )
    return config.learning_rate * multiplier


def _clipped_value_losses(
    current: Tensor,
    old: Tensor,
    targets: Tensor,
    *,
    clip_epsilon: float,
) -> Tensor:
    values = current.float()
    old_values = old.to(device=values.device, dtype=values.dtype)
    returns = targets.to(device=values.device, dtype=values.dtype)
    clipped = old_values + (values - old_values).clamp(
        min=-clip_epsilon,
        max=clip_epsilon,
    )
    return torch.maximum(
        (values - returns).pow(2),
        (clipped - returns).pow(2),
    )


def _validate_loss_inputs(inputs: StatelessPpoLossInputs) -> None:
    shape = inputs.current_token_logprobs.shape
    if (
        len(shape) != 2
        or inputs.current_token_entropies.shape != shape
        or inputs.current_prefix_values.shape != shape
        or inputs.token_mask.shape != shape
        or inputs.old_token_logprobs.shape != shape
        or inputs.old_prefix_values.shape != shape
        or inputs.token_advantages.shape != shape
        or inputs.token_returns.shape != shape
    ):
        raise ValueError("stateless PPO token tensors are misaligned")
    if inputs.token_mask.dtype != torch.bool:
        raise ValueError("stateless PPO token mask must be boolean")
    require_tensor_condition(
        inputs.token_mask.any(dim=1).all(),
        "every stateless PPO decision needs active tokens",
    )
    batch_size = int(shape[0])
    vector_fields = (
        inputs.current_root_values,
        inputs.old_root_values,
        inputs.root_returns,
        inputs.decision_macro_weights,
        inputs.belief_row_losses,
        inputs.belief_valid_mask,
        inputs.belief_macro_weights,
    )
    if any(tensor.shape != (batch_size,) for tensor in vector_fields):
        raise ValueError("stateless PPO decision tensors are misaligned")
    if inputs.belief_valid_mask.dtype != torch.bool:
        raise ValueError("belief validity mask must be boolean")
    require_tensor_condition(
        ~(inputs.belief_macro_weights[~inputs.belief_valid_mask] != 0.0).any(),
        "invalid belief rows cannot carry macro weight",
    )


__all__ = [
    "SimpleStatelessPpoConfig",
    "StatelessLearnerPrecision",
    "StatelessLogicalBatch",
    "StatelessMicrobatch",
    "StatelessPpoLoss",
    "StatelessPpoLossInputs",
    "schedule_stateless_logical_batches",
    "schedule_stateless_microbatches",
    "stateless_deck_macro_ppo_loss",
    "stateless_learning_rate",
]
