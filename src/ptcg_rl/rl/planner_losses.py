"""Schema-9 planner replay distributions and learner objectives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor
from torch.nn import functional


class PlannerImitationConfig(BaseModel):
    """Fixed applicability and conservative target-projection semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_policy_age: int = Field(ge=0)
    target_ratio_clip: float = Field(default=4.0, ge=1.0)

    @field_validator("target_ratio_clip")
    @classmethod
    def finite_ratio_clip(cls, value: float) -> float:
        """Reject an unusable planner-target cap."""
        if not math.isfinite(value):
            raise ValueError("planner target ratio clip must be finite")
        return value


@dataclass(frozen=True, slots=True)
class PlannerReplayBatch:
    """Flattened candidate groups selected from schema-9 decision rows."""

    decision_indices: Tensor
    candidate_offsets: Tensor
    selected_candidate_indices: Tensor
    target_probabilities: Tensor
    old_behavior_probabilities: Tensor
    immutable_score_priors: Tensor
    rules_exact: Tensor
    scenario_grid_complete: Tensor
    identity_valid: Tensor
    policy_ages: Tensor
    support_exhaustive: Tensor
    support_censored: Tensor
    planner_temperatures: Tensor
    candidate_actions: tuple[tuple[tuple[int, ...], ...], ...]
    candidate_features: tuple[Tensor, ...]

    @property
    def group_count(self) -> int:
        """Return the number of planner-conditioned decisions."""
        return int(self.decision_indices.numel())

    @property
    def candidate_count(self) -> int:
        """Return the flattened retained-support size."""
        return int(self.target_probabilities.numel())

    def validate(self, *, batch_size: int | None = None) -> None:
        """Reject misaligned supports before any PPO ratio is computed."""
        group_count = self.group_count
        _require_integer_vector(self.decision_indices, "decision_indices", group_count)
        _require_integer_vector(
            self.selected_candidate_indices,
            "selected_candidate_indices",
            group_count,
        )
        _require_integer_vector(
            self.candidate_offsets,
            "candidate_offsets",
            group_count + 1,
        )
        offsets = self.candidate_offsets.detach().to(device="cpu", dtype=torch.long)
        if int(offsets[0]) != 0 or bool((offsets[1:] < offsets[:-1]).any()):
            raise ValueError(
                "planner candidate offsets must start at zero and increase"
            )
        if int(offsets[-1]) != self.candidate_count:
            raise ValueError("planner candidate offsets do not cover candidate rows")
        lengths = offsets[1:] - offsets[:-1]
        if bool((lengths <= 0).any()):
            raise ValueError("planner replay groups must be non-empty")
        if bool((self.selected_candidate_indices < 0).any()) or bool(
            (
                self.selected_candidate_indices
                >= lengths.to(device=self.selected_candidate_indices.device)
            ).any()
        ):
            raise ValueError("selected planner candidate index is invalid")
        if batch_size is not None and (
            bool((self.decision_indices < 0).any())
            or bool((self.decision_indices >= batch_size).any())
        ):
            raise ValueError("planner decision index is outside the PPO batch")
        ordered_decisions = self.decision_indices.sort().values
        if group_count > 1 and bool(
            (ordered_decisions[1:] == ordered_decisions[:-1]).any()
        ):
            raise ValueError("planner replay contains duplicate decision indices")
        for name, values in (
            ("target_probabilities", self.target_probabilities),
            ("old_behavior_probabilities", self.old_behavior_probabilities),
            ("immutable_score_priors", self.immutable_score_priors),
        ):
            _require_float_vector(values, name, self.candidate_count)
        if bool((self.target_probabilities < 0).any()) or bool(
            (self.old_behavior_probabilities <= 0).any()
        ):
            raise ValueError("planner replay probabilities are invalid")
        _require_bool_vector(self.rules_exact, "rules_exact", self.candidate_count)
        for name, values in (
            ("scenario_grid_complete", self.scenario_grid_complete),
            ("identity_valid", self.identity_valid),
            ("support_exhaustive", self.support_exhaustive),
            ("support_censored", self.support_censored),
        ):
            _require_bool_vector(values, name, group_count)
        if not bool(torch.equal(self.support_censored, ~self.support_exhaustive)):
            raise ValueError("planner support flags are inconsistent")
        _require_integer_vector(self.policy_ages, "policy_ages", group_count)
        if bool((self.policy_ages < 0).any()):
            raise ValueError("planner policy ages must be non-negative")
        _require_float_vector(
            self.planner_temperatures,
            "planner_temperatures",
            group_count,
        )
        if bool((self.planner_temperatures <= 0).any()):
            raise ValueError("planner temperatures must be positive")
        target_sums = _segment_sum(self.target_probabilities, lengths)
        behavior_sums = _segment_sum(self.old_behavior_probabilities, lengths)
        ones = torch.ones_like(target_sums)
        if not bool(torch.allclose(target_sums, ones, atol=1.0e-5, rtol=0.0)):
            raise ValueError("planner target probabilities must normalize per decision")
        if not bool(torch.allclose(behavior_sums, ones, atol=1.0e-5, rtol=0.0)):
            raise ValueError(
                "planner behavior probabilities must normalize per decision"
            )
        if (
            len(self.candidate_actions) != group_count
            or len(self.candidate_features) != group_count
        ):
            raise ValueError("planner candidate object rows must align with groups")
        cpu_lengths = tuple(int(value) for value in lengths.tolist())
        for actions, features, length in zip(
            self.candidate_actions,
            self.candidate_features,
            cpu_lengths,
            strict=True,
        ):
            if len(actions) != length or len(set(actions)) != length:
                raise ValueError("planner complete actions must align and be unique")
            if features.ndim != 2 or int(features.shape[0]) != length:
                raise ValueError("planner candidate features must align with actions")
            if not features.is_floating_point() or not bool(
                torch.isfinite(features).all()
            ):
                raise ValueError("planner candidate features must be finite floats")


@dataclass(frozen=True, slots=True)
class PlannerCurrentDistributions:
    """Current trainable distributions on immutable retained supports."""

    base_probabilities: Tensor
    proposal_probabilities: Tensor
    behavior_probabilities: Tensor


@dataclass(frozen=True, slots=True)
class PlannerImitationMetrics:
    """Applicability and clipping diagnostics, never control-flow gates."""

    planner_decisions: int
    applicable_decisions: int
    stale_decisions: int
    inexact_decisions: int
    incomplete_grid_decisions: int
    identity_invalid_decisions: int
    censored_support_decisions: int
    exhaustive_support_decisions: int
    clipped_candidates: int
    candidate_count: int


@dataclass(frozen=True, slots=True)
class PlannerImitationLoss:
    """Per-decision-normalized candidate, proposal, and diagnostic losses."""

    candidate_rerank_loss: Tensor
    proposal_kl_loss: Tensor
    base_kl_loss: Tensor
    behavior_kl_loss: Tensor
    metrics: PlannerImitationMetrics


def planner_current_distributions(
    replay: PlannerReplayBatch,
    *,
    current_base_logprobs: Tensor,
    current_proposal_logprobs: Tensor,
    current_reranker_residuals: Tensor,
) -> PlannerCurrentDistributions:
    """Recompute every schema-9 branch distribution on stored evidence."""
    replay.validate()
    candidate_count = replay.candidate_count
    for name, values in (
        ("current_base_logprobs", current_base_logprobs),
        ("current_proposal_logprobs", current_proposal_logprobs),
        ("current_reranker_residuals", current_reranker_residuals),
    ):
        _require_float_vector(values, name, candidate_count)
    lengths = _candidate_lengths(replay)
    temperatures = torch.repeat_interleave(
        replay.planner_temperatures.to(
            device=current_base_logprobs.device,
            dtype=current_base_logprobs.dtype,
        ),
        lengths.to(device=current_base_logprobs.device),
    )
    score_priors = replay.immutable_score_priors.to(
        device=current_base_logprobs.device,
        dtype=current_base_logprobs.dtype,
    )
    base = _segment_softmax(current_base_logprobs / temperatures, lengths)
    proposal = _segment_softmax(current_proposal_logprobs / temperatures, lengths)
    behavior = _segment_softmax(
        (current_base_logprobs + score_priors + current_reranker_residuals)
        / temperatures,
        lengths,
    )
    return PlannerCurrentDistributions(
        base_probabilities=base,
        proposal_probabilities=proposal,
        behavior_probabilities=behavior,
    )


def planner_imitation_loss(
    replay: PlannerReplayBatch,
    current: PlannerCurrentDistributions,
    *,
    config: PlannerImitationConfig,
) -> PlannerImitationLoss:
    """Return clipped planner distillation on applicable decision groups.

    Because the entire retained categorical distribution is persisted, the
    planner target can be conservatively projected relative to the collection
    behavior distribution.  The per-candidate target/behavior ratio is capped,
    then renormalized within each decision.  This is a bounded target
    projection, not an off-policy sample-importance estimator.
    """
    replay.validate()
    candidate_count = replay.candidate_count
    for name, values in (
        ("base_probabilities", current.base_probabilities),
        ("proposal_probabilities", current.proposal_probabilities),
        ("behavior_probabilities", current.behavior_probabilities),
    ):
        _require_float_vector(values, name, candidate_count)
        if bool((values <= 0).any()):
            raise ValueError("current planner probabilities must be positive")
    lengths = _candidate_lengths(replay)
    target = replay.target_probabilities.to(
        device=current.base_probabilities.device,
        dtype=current.base_probabilities.dtype,
    )
    old_behavior = replay.old_behavior_probabilities.to(
        device=target.device,
        dtype=target.dtype,
    )
    positive_target = target > 0
    ratio = torch.where(
        positive_target, target / old_behavior, torch.zeros_like(target)
    )
    inverse_clip = 1.0 / config.target_ratio_clip
    clipped_ratio = torch.where(
        positive_target,
        ratio.clamp(min=inverse_clip, max=config.target_ratio_clip),
        torch.zeros_like(ratio),
    )
    clipped_target = old_behavior * clipped_ratio
    clipped_target = clipped_target / torch.repeat_interleave(
        _segment_sum(clipped_target, lengths),
        lengths.to(device=clipped_target.device),
    )
    base_kl_by_group = _group_kl(clipped_target, current.base_probabilities, lengths)
    proposal_kl_by_group = _group_kl(
        clipped_target, current.proposal_probabilities, lengths
    )
    behavior_kl_by_group = _group_kl(
        clipped_target, current.behavior_probabilities, lengths
    )
    rules_exact_by_group = _segment_sum(
        replay.rules_exact.to(device=target.device, dtype=target.dtype), lengths
    ).eq(lengths.to(device=target.device, dtype=target.dtype))
    stale = replay.policy_ages.to(device=target.device) > config.max_policy_age
    complete = replay.scenario_grid_complete.to(device=target.device)
    identity = replay.identity_valid.to(device=target.device)
    applicable = rules_exact_by_group & ~stale & complete & identity
    zero = current.base_probabilities.sum() * 0.0
    if bool(applicable.any()):
        base_loss = base_kl_by_group[applicable].mean()
        proposal_loss = proposal_kl_by_group[applicable].mean()
        behavior_loss = behavior_kl_by_group[applicable].mean()
    else:
        base_loss = zero
        proposal_loss = zero
        behavior_loss = zero
    candidate_loss = 0.5 * (base_loss + behavior_loss)
    metrics = PlannerImitationMetrics(
        planner_decisions=replay.group_count,
        applicable_decisions=_count(applicable),
        stale_decisions=_count(stale),
        inexact_decisions=_count(~rules_exact_by_group),
        incomplete_grid_decisions=_count(~complete),
        identity_invalid_decisions=_count(~identity),
        censored_support_decisions=_count(
            replay.support_censored.to(device=target.device)
        ),
        exhaustive_support_decisions=_count(
            replay.support_exhaustive.to(device=target.device)
        ),
        clipped_candidates=_count(
            positive_target & ~torch.isclose(ratio, clipped_ratio)
        ),
        candidate_count=candidate_count,
    )
    return PlannerImitationLoss(
        candidate_rerank_loss=candidate_loss,
        proposal_kl_loss=proposal_loss,
        base_kl_loss=base_loss,
        behavior_kl_loss=behavior_loss,
        metrics=metrics,
    )


def planner_selected_logprobs_and_entropies(
    replay: PlannerReplayBatch,
    current: PlannerCurrentDistributions,
) -> tuple[Tensor, Tensor]:
    """Return current categorical PPO terms for planner-conditioned rows."""
    replay.validate()
    lengths = _candidate_lengths(replay)
    probabilities = current.behavior_probabilities
    offsets = replay.candidate_offsets[:-1].to(
        device=probabilities.device, dtype=torch.long
    )
    selected = offsets + replay.selected_candidate_indices.to(
        device=probabilities.device, dtype=torch.long
    )
    log_probabilities = probabilities.clamp_min(
        torch.finfo(probabilities.dtype).tiny
    ).log()
    selected_logprobs = log_probabilities.index_select(0, selected)
    entropy_terms = -(probabilities * log_probabilities)
    entropies = _segment_sum(entropy_terms, lengths)
    return (selected_logprobs, entropies)


def root_information_value_loss(
    predictions: Tensor,
    final_root_outcomes: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Train only actual, deduplicated endpoint rows with SmoothL1 W/D/L."""
    _require_float_vector(predictions, "predictions", int(predictions.numel()))
    if final_root_outcomes.ndim != 1 or final_root_outcomes.shape != predictions.shape:
        raise ValueError("root-information value targets must align with predictions")
    targets = final_root_outcomes.to(
        device=predictions.device,
        dtype=predictions.dtype,
    )
    if not bool(torch.isfinite(targets).all()) or not bool(
        torch.isin(targets, targets.new_tensor((-1.0, 0.0, 1.0))).all()
    ):
        raise ValueError("root-information value targets must be W/D/L")
    mask = (
        torch.ones_like(predictions, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.to(device=predictions.device, dtype=torch.bool)
    )
    if mask.shape != predictions.shape:
        raise ValueError("root-information value mask must align with predictions")
    if not bool(mask.any()):
        return predictions.sum() * 0.0
    losses = functional.smooth_l1_loss(predictions, targets, reduction="none")
    return losses[mask].mean()


def select_planner_replay(
    replay: PlannerReplayBatch | None,
    decision_indices: Sequence[int],
) -> PlannerReplayBatch | None:
    """Select and remap planner groups for one PPO minibatch."""
    if replay is None:
        return None
    replay.validate()
    target_positions = {
        int(source): target for target, source in enumerate(decision_indices)
    }
    source_decisions = tuple(
        int(value)
        for value in replay.decision_indices.detach().to(device="cpu").tolist()
    )
    source_groups = tuple(
        group
        for group, source in enumerate(source_decisions)
        if source in target_positions
    )
    if not source_groups:
        return None
    offsets = replay.candidate_offsets.detach().to(device="cpu", dtype=torch.long)
    ranges = tuple(
        torch.arange(
            int(offsets[group]),
            int(offsets[group + 1]),
            dtype=torch.long,
            device=replay.target_probabilities.device,
        )
        for group in source_groups
    )
    candidate_indices = torch.cat(ranges)
    group_indices = torch.tensor(
        source_groups,
        dtype=torch.long,
        device=replay.decision_indices.device,
    )
    new_offsets = [0]
    for values in ranges:
        new_offsets.append(new_offsets[-1] + int(values.numel()))
    selected = PlannerReplayBatch(
        decision_indices=torch.tensor(
            tuple(target_positions[source_decisions[group]] for group in source_groups),
            dtype=replay.decision_indices.dtype,
            device=replay.decision_indices.device,
        ),
        candidate_offsets=torch.tensor(
            new_offsets,
            dtype=replay.candidate_offsets.dtype,
            device=replay.candidate_offsets.device,
        ),
        selected_candidate_indices=replay.selected_candidate_indices.index_select(
            0, group_indices
        ),
        target_probabilities=replay.target_probabilities.index_select(
            0, candidate_indices
        ),
        old_behavior_probabilities=replay.old_behavior_probabilities.index_select(
            0, candidate_indices
        ),
        immutable_score_priors=replay.immutable_score_priors.index_select(
            0, candidate_indices
        ),
        rules_exact=replay.rules_exact.index_select(0, candidate_indices),
        scenario_grid_complete=replay.scenario_grid_complete.index_select(
            0, group_indices
        ),
        identity_valid=replay.identity_valid.index_select(0, group_indices),
        policy_ages=replay.policy_ages.index_select(0, group_indices),
        support_exhaustive=replay.support_exhaustive.index_select(0, group_indices),
        support_censored=replay.support_censored.index_select(0, group_indices),
        planner_temperatures=replay.planner_temperatures.index_select(0, group_indices),
        candidate_actions=tuple(
            replay.candidate_actions[group] for group in source_groups
        ),
        candidate_features=tuple(
            replay.candidate_features[group] for group in source_groups
        ),
    )
    selected.validate(batch_size=len(decision_indices))
    return selected


def move_planner_replay(
    replay: PlannerReplayBatch | None,
    *,
    device: torch.device | str,
    non_blocking: bool = False,
) -> PlannerReplayBatch | None:
    """Move replay tensors while preserving immutable Python action support."""
    if replay is None:
        return None

    def move(values: Tensor) -> Tensor:
        return values.to(device=device, non_blocking=non_blocking)

    return PlannerReplayBatch(
        decision_indices=move(replay.decision_indices),
        candidate_offsets=move(replay.candidate_offsets),
        selected_candidate_indices=move(replay.selected_candidate_indices),
        target_probabilities=move(replay.target_probabilities),
        old_behavior_probabilities=move(replay.old_behavior_probabilities),
        immutable_score_priors=move(replay.immutable_score_priors),
        rules_exact=move(replay.rules_exact),
        scenario_grid_complete=move(replay.scenario_grid_complete),
        identity_valid=move(replay.identity_valid),
        policy_ages=move(replay.policy_ages),
        support_exhaustive=move(replay.support_exhaustive),
        support_censored=move(replay.support_censored),
        planner_temperatures=move(replay.planner_temperatures),
        candidate_actions=replay.candidate_actions,
        candidate_features=tuple(move(values) for values in replay.candidate_features),
    )


def pin_planner_replay(
    replay: PlannerReplayBatch | None,
) -> PlannerReplayBatch | None:
    """Pin CPU replay tensors for asynchronous accelerator transfer."""
    if replay is None:
        return None

    def pin(values: Tensor) -> Tensor:
        if values.device.type != "cpu" or values.is_pinned():
            return values
        return values.pin_memory()

    return PlannerReplayBatch(
        decision_indices=pin(replay.decision_indices),
        candidate_offsets=pin(replay.candidate_offsets),
        selected_candidate_indices=pin(replay.selected_candidate_indices),
        target_probabilities=pin(replay.target_probabilities),
        old_behavior_probabilities=pin(replay.old_behavior_probabilities),
        immutable_score_priors=pin(replay.immutable_score_priors),
        rules_exact=pin(replay.rules_exact),
        scenario_grid_complete=pin(replay.scenario_grid_complete),
        identity_valid=pin(replay.identity_valid),
        policy_ages=pin(replay.policy_ages),
        support_exhaustive=pin(replay.support_exhaustive),
        support_censored=pin(replay.support_censored),
        planner_temperatures=pin(replay.planner_temperatures),
        candidate_actions=replay.candidate_actions,
        candidate_features=tuple(pin(values) for values in replay.candidate_features),
    )


def _group_kl(target: Tensor, current: Tensor, lengths: Tensor) -> Tensor:
    tiny = torch.finfo(current.dtype).tiny
    terms = target * (target.clamp_min(tiny).log() - current.clamp_min(tiny).log())
    return _segment_sum(terms, lengths)


def _candidate_lengths(replay: PlannerReplayBatch) -> Tensor:
    offsets = replay.candidate_offsets.to(dtype=torch.long)
    return offsets[1:] - offsets[:-1]


def _segment_softmax(values: Tensor, lengths: Tensor) -> Tensor:
    active_lengths = lengths.to(device=values.device, dtype=torch.long)
    maxima = torch.segment_reduce(values, "max", lengths=active_lengths)
    centered = values - torch.repeat_interleave(maxima, active_lengths)
    exponentials = centered.exp()
    totals = torch.segment_reduce(exponentials, "sum", lengths=active_lengths)
    return exponentials / torch.repeat_interleave(totals, active_lengths)


def _segment_sum(values: Tensor, lengths: Tensor) -> Tensor:
    return torch.segment_reduce(
        values,
        "sum",
        lengths=lengths.to(device=values.device, dtype=torch.long),
    )


def _require_float_vector(values: Tensor, name: str, length: int) -> None:
    if values.ndim != 1 or int(values.numel()) != length:
        raise ValueError(f"{name} must be an aligned vector")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain finite floating-point values")


def _require_integer_vector(values: Tensor, name: str, length: int) -> None:
    if values.ndim != 1 or int(values.numel()) != length:
        raise ValueError(f"{name} must be an aligned vector")
    if values.is_floating_point() or values.dtype is torch.bool:
        raise TypeError(f"{name} must use an integer dtype")


def _require_bool_vector(values: Tensor, name: str, length: int) -> None:
    if (
        values.dtype is not torch.bool
        or values.ndim != 1
        or int(values.numel()) != length
    ):
        raise TypeError(f"{name} must be an aligned bool vector")


def _count(values: Tensor) -> int:
    return int(values.detach().to(dtype=torch.int64).sum().item())


__all__ = [
    "PlannerCurrentDistributions",
    "PlannerImitationConfig",
    "PlannerImitationLoss",
    "PlannerImitationMetrics",
    "PlannerReplayBatch",
    "move_planner_replay",
    "pin_planner_replay",
    "planner_current_distributions",
    "planner_imitation_loss",
    "planner_selected_logprobs_and_entropies",
    "root_information_value_loss",
    "select_planner_replay",
]
