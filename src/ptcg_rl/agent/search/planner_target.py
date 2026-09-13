"""Detached planner targets and replayable planner-conditioned logits."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ptcg_rl.agent.search.planner_scoring import PlannerScoringConfig


@dataclass(frozen=True, slots=True)
class DetachedPlannerTarget:
    """Immutable target distribution and score prior on retained support."""

    distribution: Tensor
    score_prior: Tensor
    robust_score_center: Tensor
    scorer_fingerprint: str

    def __post_init__(self) -> None:
        """Require collection evidence to be detached from autograd."""
        if (
            self.distribution.requires_grad
            or self.score_prior.requires_grad
            or self.robust_score_center.requires_grad
        ):
            raise ValueError("planner target tensors must be detached")
        if (
            len(self.scorer_fingerprint) != 64
            or self.scorer_fingerprint != self.scorer_fingerprint.lower()
        ):
            raise ValueError("planner target scorer fingerprint must be SHA-256")
        try:
            bytes.fromhex(self.scorer_fingerprint)
        except ValueError as exc:
            raise ValueError(
                "planner target scorer fingerprint must be SHA-256"
            ) from exc


def build_detached_planner_target(
    *,
    base_logprobs_at_collection: Tensor,
    robust_scores: Tensor,
    config: PlannerScoringConfig,
) -> DetachedPlannerTarget:
    """Build ``q_planner`` on one immutable retained candidate support."""
    _validate_candidate_vector(base_logprobs_at_collection, "base log-probabilities")
    _validate_candidate_vector(robust_scores, "robust scores")
    if base_logprobs_at_collection.shape != robust_scores.shape:
        raise ValueError("base log-probabilities and robust scores must align")
    base = base_logprobs_at_collection.detach()
    scores = robust_scores.detach().to(device=base.device, dtype=base.dtype)
    center = scores.mean().detach()
    score_prior = (
        config.planner_score_weight
        * torch.tanh((scores - center) / config.planner_score_scale)
    ).detach()
    distribution = torch.softmax(
        (base + score_prior) / config.planner_temperature,
        dim=0,
    ).detach()
    return DetachedPlannerTarget(
        distribution=distribution,
        score_prior=score_prior,
        robust_score_center=center,
        scorer_fingerprint=config.scorer_fingerprint,
    )


def planner_conditioned_distribution(
    *,
    current_base_logprobs: Tensor,
    immutable_score_prior: Tensor,
    reranker_residual: Tensor,
    config: PlannerScoringConfig,
) -> Tensor:
    """Return the trainable behavior distribution on retained support."""
    _validate_candidate_vector(current_base_logprobs, "current base log-probabilities")
    _validate_candidate_vector(immutable_score_prior, "immutable score prior")
    _validate_candidate_vector(reranker_residual, "reranker residual")
    if not (
        current_base_logprobs.shape
        == immutable_score_prior.shape
        == reranker_residual.shape
    ):
        raise ValueError("planner-conditioned candidate tensors must align")
    return torch.softmax(
        (
            current_base_logprobs
            + immutable_score_prior.detach().to(
                device=current_base_logprobs.device,
                dtype=current_base_logprobs.dtype,
            )
            + reranker_residual
        )
        / config.planner_temperature,
        dim=0,
    )


def _validate_candidate_vector(values: Tensor, name: str) -> None:
    if values.ndim != 1 or int(values.numel()) <= 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError(f"{name} must be finite")


__all__ = [
    "DetachedPlannerTarget",
    "build_detached_planner_target",
    "planner_conditioned_distribution",
]
