"""Proposal-corrected soft policy improvement over retained siblings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ptcg_rl.rl.amortized_policy_iteration.contracts import CmpoConfig


@dataclass(frozen=True, slots=True)
class ImprovementDistribution:
    """Flattened improved probabilities grouped by information set."""

    probabilities: Tensor
    # This compatibility name contains the configured pre-temperature
    # advantage signal. It is standardized only in the legacy transform.
    normalized_advantages: Tensor
    candidate_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ImprovementSample:
    """One sampled candidate index and its true behavior probability per root."""

    flat_indices: Tensor
    probabilities: Tensor
    log_probabilities: Tensor


def cmpo_distribution(
    expected_scores: Tensor,
    state_expected_scores: Tensor,
    old_policy_probabilities: Tensor,
    proposal_probabilities: Tensor,
    *,
    candidate_counts: tuple[int, ...],
    exhaustive_rows: tuple[bool, ...],
    config: CmpoConfig,
) -> ImprovementDistribution:
    """Build a clipped MPO distribution with sampled-support correction."""
    rows = int(expected_scores.numel())
    for name, values in (
        ("expected_scores", expected_scores),
        ("old_policy_probabilities", old_policy_probabilities),
        ("proposal_probabilities", proposal_probabilities),
    ):
        if values.ndim != 1 or int(values.shape[0]) != rows:
            raise ValueError(f"{name} must align with flattened candidates")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError(f"{name} must be finite")
    if state_expected_scores.ndim != 1 or int(state_expected_scores.shape[0]) != len(
        candidate_counts
    ):
        raise ValueError("state_expected_scores must align with information sets")
    if len(exhaustive_rows) != len(candidate_counts):
        raise ValueError("exhaustive markers must align with information sets")
    _validate_groups(candidate_counts, candidate_rows=rows)
    if bool((old_policy_probabilities < 0.0).any().item()):
        raise ValueError("old policy probabilities must be non-negative")
    if bool((proposal_probabilities <= 0.0).any().item()):
        raise ValueError("proposal probabilities must be positive")

    return _cmpo_distribution_unchecked(
        expected_scores,
        state_expected_scores,
        old_policy_probabilities,
        proposal_probabilities,
        candidate_counts=candidate_counts,
        exhaustive_rows=exhaustive_rows,
        config=config,
    )


def _cmpo_distribution_unchecked(
    expected_scores: Tensor,
    state_expected_scores: Tensor,
    old_policy_probabilities: Tensor,
    proposal_probabilities: Tensor,
    *,
    candidate_counts: tuple[int, ...],
    exhaustive_rows: tuple[bool, ...],
    config: CmpoConfig,
) -> ImprovementDistribution:
    """Build CMPO weights from locally generated, already-valid tensors.

    The actor hot path creates these inputs from softmax, exp, and validated
    proposal objects. Keeping validation in :func:`cmpo_distribution` protects
    replay and other external boundaries without forcing five CUDA host
    synchronizations for every live inference group.
    """

    layout = _ragged_layout(candidate_counts, device=expected_scores.device)
    if config.advantage_transform == "per_root_standardized":
        advantages = expected_scores - state_expected_scores.index_select(
            0,
            layout.row_indices,
        )
        group_sums = torch.zeros(
            len(candidate_counts),
            dtype=advantages.dtype,
            device=advantages.device,
        ).scatter_add_(0, layout.row_indices, advantages)
        means = group_sums / layout.counts.to(dtype=advantages.dtype)
        centered = advantages - means.index_select(0, layout.row_indices)
        group_squared_sums = torch.zeros_like(group_sums).scatter_add_(
            0,
            layout.row_indices,
            centered.square(),
        )
        scales = torch.sqrt(
            group_squared_sums / layout.counts.to(dtype=advantages.dtype)
        ).clamp_min(config.normalization_epsilon)
        transformed = advantages / scales.index_select(0, layout.row_indices)
    else:
        group_score_sums = torch.zeros(
            len(candidate_counts),
            dtype=expected_scores.dtype,
            device=expected_scores.device,
        ).scatter_add_(0, layout.row_indices, expected_scores)
        group_score_means = group_score_sums / layout.counts.to(
            dtype=expected_scores.dtype
        )
        transformed = expected_scores - group_score_means.index_select(
            0,
            layout.row_indices,
        )
    clipped = torch.clamp(
        transformed / config.temperature,
        min=-config.advantage_clip,
        max=config.advantage_clip,
    )
    prior = old_policy_probabilities.clamp_min(config.min_policy_probability)
    log_weights = config.prior_exponent * torch.log(prior) + clipped
    sampled_rows = torch.tensor(
        [not exhaustive for exhaustive in exhaustive_rows],
        dtype=torch.bool,
        device=expected_scores.device,
    ).index_select(0, layout.row_indices)
    log_weights = log_weights - torch.where(
        sampled_rows,
        torch.log(proposal_probabilities),
        torch.zeros_like(proposal_probabilities),
    )
    padded_log_weights = torch.full(
        (len(candidate_counts), layout.max_count),
        -torch.inf,
        dtype=torch.float32,
        device=expected_scores.device,
    )
    padded_log_weights[layout.row_indices, layout.column_indices] = log_weights.float()
    padded_probabilities = torch.softmax(padded_log_weights, dim=1)
    probabilities = padded_probabilities[
        layout.row_indices,
        layout.column_indices,
    ].to(dtype=expected_scores.dtype)
    return ImprovementDistribution(
        probabilities=probabilities,
        normalized_advantages=transformed,
        candidate_counts=candidate_counts,
    )


def sample_improvement_distribution(
    distribution: ImprovementDistribution,
    *,
    generator: torch.Generator | None = None,
) -> ImprovementSample:
    """Sample one action per root and retain its exact improved probability."""
    _validate_groups(
        distribution.candidate_counts,
        candidate_rows=int(distribution.probabilities.numel()),
    )
    layout = _ragged_layout(
        distribution.candidate_counts,
        device=distribution.probabilities.device,
    )
    padded = torch.zeros(
        (len(distribution.candidate_counts), layout.max_count),
        dtype=distribution.probabilities.dtype,
        device=distribution.probabilities.device,
    )
    padded[layout.row_indices, layout.column_indices] = distribution.probabilities
    local_indices = torch.multinomial(
        padded.float(),
        1,
        generator=generator,
    ).squeeze(1)
    flat_indices = layout.offsets + local_indices
    probability_tensor = distribution.probabilities.index_select(0, flat_indices)
    return ImprovementSample(
        flat_indices=flat_indices,
        probabilities=probability_tensor,
        log_probabilities=torch.log(probability_tensor),
    )


def cmpo_cross_entropy(
    model_action_logprobs: Tensor,
    target: ImprovementDistribution,
) -> Tensor:
    """Train the direct policy on each full retained sibling distribution."""
    if model_action_logprobs.shape != target.probabilities.shape:
        raise ValueError("model log-probabilities and CMPO targets must align")
    _validate_groups(
        target.candidate_counts,
        candidate_rows=int(target.probabilities.numel()),
    )
    layout = _ragged_layout(
        target.candidate_counts,
        device=model_action_logprobs.device,
    )
    weighted_losses = -(
        target.probabilities.detach().to(device=model_action_logprobs.device)
        * model_action_logprobs
    )
    row_losses = torch.zeros(
        len(target.candidate_counts),
        dtype=weighted_losses.dtype,
        device=model_action_logprobs.device,
    ).scatter_add_(
        0,
        layout.row_indices,
        weighted_losses,
    )
    return row_losses.mean()


@dataclass(frozen=True, slots=True)
class _RaggedLayout:
    counts: Tensor
    offsets: Tensor
    row_indices: Tensor
    column_indices: Tensor
    max_count: int


def _ragged_layout(
    candidate_counts: tuple[int, ...],
    *,
    device: torch.device,
) -> _RaggedLayout:
    """Build reusable tensor indices for one positive ragged grouping."""
    counts = torch.tensor(candidate_counts, dtype=torch.long, device=device)
    offsets = torch.cumsum(counts, dim=0) - counts
    row_indices = torch.repeat_interleave(
        torch.arange(len(candidate_counts), device=device),
        counts,
    )
    column_indices = torch.arange(
        sum(candidate_counts),
        dtype=torch.long,
        device=device,
    ) - offsets.index_select(0, row_indices)
    return _RaggedLayout(
        counts=counts,
        offsets=offsets,
        row_indices=row_indices,
        column_indices=column_indices,
        max_count=max(candidate_counts),
    )


def _validate_groups(
    candidate_counts: tuple[int, ...],
    *,
    candidate_rows: int,
) -> None:
    if not candidate_counts or any(count <= 0 for count in candidate_counts):
        raise ValueError("candidate groups must be non-empty")
    if sum(candidate_counts) != candidate_rows:
        raise ValueError("candidate groups do not cover flattened candidates")


__all__ = [
    "ImprovementDistribution",
    "ImprovementSample",
    "cmpo_cross_entropy",
    "cmpo_distribution",
    "sample_improvement_distribution",
]
