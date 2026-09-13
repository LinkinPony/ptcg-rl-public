"""Search-conditioned scoring for complete action candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from ptcg_rl.engine.search_evidence import (
    SEARCH_EVIDENCE_FEATURE_SIZE,
    SEARCH_EVIDENCE_ROBUST_SCORE_INDEX,
)

SEARCH_ROBUST_PRIOR_LOGIT_SCALE = 1.0
SEARCH_ROBUST_PRIOR_TEMPERATURE = 0.25
SEARCH_RERANKER_ARCHITECTURE_VERSION = 1


class SearchCandidateReranker(nn.Module):
    """Add search evidence and a learned residual to base action log-probs.

    The base complete-action policy is a detached prior for this auxiliary path.
    A centered, fixed-temperature robust search score makes a newly migrated
    checkpoint consume search immediately without inflating negligible score
    margins, while the zero-initialized learned output preserves that
    deterministic cold-start rule.
    """

    def __init__(self, d_model: int, *, hidden_dim: int | None = None) -> None:
        """Initialize the candidate residual network."""
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        resolved_hidden_dim = hidden_dim or d_model
        if resolved_hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.residual_head = nn.Sequential(
            nn.Linear(2 * d_model + SEARCH_EVIDENCE_FEATURE_SIZE, resolved_hidden_dim),
            nn.GELU(),
            nn.Linear(resolved_hidden_dim, 1),
        )
        output = cast(nn.Linear, self.residual_head[-1])
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(
        self,
        base_action_logprobs: Tensor,
        state_latents: Tensor,
        action_latents: Tensor,
        evidence_features: Tensor,
        *,
        candidate_counts: Sequence[int],
    ) -> Tensor:
        """Return flattened logits for consecutive decision candidate groups."""
        normalized_counts = self._validate_candidate_inputs(
            base_action_logprobs,
            state_latents,
            action_latents,
            evidence_features,
            candidate_counts=candidate_counts,
        )
        residual = self._residual_logits(
            state_latents,
            action_latents,
            evidence_features,
        )
        robust_scores = evidence_features[:, SEARCH_EVIDENCE_ROBUST_SCORE_INDEX].to(
            dtype=base_action_logprobs.dtype
        )
        robust_prior = _temperature_scale_grouped_scores(
            robust_scores,
            normalized_counts,
        )
        return (
            base_action_logprobs.detach()
            + SEARCH_ROBUST_PRIOR_LOGIT_SCALE * robust_prior
            + residual.to(dtype=base_action_logprobs.dtype)
        )

    def residual_logits(
        self,
        base_action_logprobs: Tensor,
        state_latents: Tensor,
        action_latents: Tensor,
        evidence_features: Tensor,
        *,
        candidate_counts: Sequence[int],
    ) -> Tensor:
        """Return only the trainable candidate residual, without fixed priors."""
        self._validate_candidate_inputs(
            base_action_logprobs,
            state_latents,
            action_latents,
            evidence_features,
            candidate_counts=candidate_counts,
        )
        return self._residual_logits(
            state_latents,
            action_latents,
            evidence_features,
        ).to(dtype=base_action_logprobs.dtype)

    def _validate_candidate_inputs(
        self,
        base_action_logprobs: Tensor,
        state_latents: Tensor,
        action_latents: Tensor,
        evidence_features: Tensor,
        *,
        candidate_counts: Sequence[int],
    ) -> tuple[int, ...]:
        """Validate aligned flattened candidate tensors and group boundaries."""
        candidate_count = int(base_action_logprobs.shape[0])
        if base_action_logprobs.ndim != 1:
            raise ValueError("base_action_logprobs must have shape [candidates]")
        if state_latents.ndim != 2 or state_latents.shape[0] != candidate_count:
            raise ValueError("state_latents must align with candidates")
        if action_latents.shape != state_latents.shape:
            raise ValueError("action_latents must align with state_latents")
        if evidence_features.shape != (
            candidate_count,
            SEARCH_EVIDENCE_FEATURE_SIZE,
        ):
            raise ValueError(
                "evidence_features must have shape "
                f"[candidates, {SEARCH_EVIDENCE_FEATURE_SIZE}]"
            )
        normalized_counts = tuple(int(count) for count in candidate_counts)
        if any(count <= 0 for count in normalized_counts):
            raise ValueError("each search decision must contain a candidate")
        if sum(normalized_counts) != candidate_count:
            raise ValueError("candidate_counts must cover every candidate")
        if not bool(torch.isfinite(evidence_features).all().item()):
            raise ValueError("evidence_features must be finite")
        return normalized_counts

    def _residual_logits(
        self,
        state_latents: Tensor,
        action_latents: Tensor,
        evidence_features: Tensor,
    ) -> Tensor:
        """Apply the learned residual head to already validated candidates."""
        residual_inputs = torch.cat(
            (
                state_latents,
                action_latents,
                evidence_features.to(dtype=state_latents.dtype),
            ),
            dim=1,
        )
        return cast(Tensor, self.residual_head(residual_inputs)).squeeze(-1)


def _temperature_scale_grouped_scores(
    scores: Tensor,
    counts: Sequence[int],
) -> Tensor:
    """Center scores and bound evidence strength at a fixed teacher scale."""
    normalized: list[Tensor] = []
    start = 0
    for count in counts:
        stop = start + int(count)
        group = scores[start:stop]
        centered = group - group.mean()
        normalized.append(torch.tanh(centered / SEARCH_ROBUST_PRIOR_TEMPERATURE))
        start = stop
    return torch.cat(normalized, dim=0) if normalized else scores.new_empty((0,))


__all__ = [
    "SEARCH_RERANKER_ARCHITECTURE_VERSION",
    "SEARCH_ROBUST_PRIOR_LOGIT_SCALE",
    "SEARCH_ROBUST_PRIOR_TEMPERATURE",
    "SearchCandidateReranker",
]
