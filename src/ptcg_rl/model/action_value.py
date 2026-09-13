"""Complete-action categorical W/D/L value head.

The head is deliberately independent of engine semantics. It consumes the
same actor-visible state and complete-action latent used by the policy, then
predicts a categorical outcome from the acting player's perspective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor, nn

ACTION_VALUE_ARCHITECTURE_VERSION = 1
WDL_OUTCOME_COUNT = 3
WDL_LOSS_INDEX = 0
WDL_DRAW_INDEX = 1
WDL_WIN_INDEX = 2


class ActionValueHeadConfig(BaseModel):
    """Configuration for the training-only complete-action critic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    architecture_version: int = ACTION_VALUE_ARCHITECTURE_VERSION
    hidden_dim: int = 128
    dropout: float = 0.0

    @field_validator("architecture_version")
    @classmethod
    def supported_architecture(cls, value: int) -> int:
        """Require the implemented categorical dueling architecture."""
        if value != ACTION_VALUE_ARCHITECTURE_VERSION:
            raise ValueError("unsupported action-value architecture version")
        return value

    @field_validator("hidden_dim")
    @classmethod
    def positive_hidden_dim(cls, value: int) -> int:
        """Reject unusable hidden dimensions."""
        if value <= 0:
            raise ValueError("action-value hidden_dim must be positive")
        return value

    @field_validator("dropout")
    @classmethod
    def valid_dropout(cls, value: float) -> float:
        """Require a finite dropout probability."""
        if not math.isfinite(value) or value < 0.0 or value >= 1.0:
            raise ValueError("action-value dropout must be in [0, 1)")
        return value


@dataclass(frozen=True)
class ActionValuePrediction:
    """Flattened candidate predictions and their decision grouping."""

    logits: Tensor
    probabilities: Tensor
    expected_scores: Tensor
    state_logits: Tensor
    information_set_state_logits: Tensor
    candidate_counts: tuple[int, ...]
    action_logprobs: Tensor | None = None


class CompleteActionValueHead(nn.Module):
    """Dueling W/D/L critic over actor-visible state and complete action."""

    def __init__(
        self,
        d_model: int,
        config: ActionValueHeadConfig,
    ) -> None:
        """Build state-baseline and centered action-advantage branches."""
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.config = config
        self.state_branch = nn.Sequential(
            nn.Linear(d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, WDL_OUTCOME_COUNT),
        )
        self.advantage_branch = nn.Sequential(
            nn.Linear(2 * d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, WDL_OUTCOME_COUNT),
        )

    def state_logits(self, state_latents: Tensor) -> Tensor:
        """Return categorical state-value logits without an action input."""
        _validate_latents(state_latents, name="state_latents")
        return cast(Tensor, self.state_branch(state_latents))

    def forward(
        self,
        state_latents: Tensor,
        action_latents: Tensor,
        *,
        candidate_counts: tuple[int, ...],
    ) -> ActionValuePrediction:
        """Predict W/D/L for flat candidates grouped by information set."""
        _validate_latents(state_latents, name="state_latents")
        _validate_latents(action_latents, name="action_latents")
        if state_latents.shape != action_latents.shape:
            raise ValueError("state and action latents must have identical shapes")
        _validate_candidate_counts(
            candidate_counts,
            candidate_rows=int(state_latents.shape[0]),
        )

        state_logits = self.state_branch(state_latents)
        raw_advantages = self.advantage_branch(
            torch.cat((state_latents, action_latents), dim=-1)
        )
        # Center within the categorical outcome axis, not across retained
        # siblings.  A candidate's Q prediction must be invariant to which
        # other actions happened to be sampled beside it; cross-candidate
        # centering would also erase every action gradient for real-trajectory
        # batches containing one executed action per information set.
        centered_advantages = raw_advantages - raw_advantages.mean(
            dim=-1,
            keepdim=True,
        )
        logits = state_logits + centered_advantages
        # Probability tensors cross persistence and Retrace validation
        # boundaries. Keep the FP32 softmax result: casting it back to BF16 can
        # make an otherwise valid categorical distribution sum to 0.996 or
        # 1.004, which is not recoverable by a later ``float()`` conversion.
        probabilities = torch.softmax(logits.float(), dim=-1)
        first_candidate_rows = []
        offset = 0
        for count in candidate_counts:
            first_candidate_rows.append(offset)
            offset += count
        information_set_state_logits = state_logits.index_select(
            0,
            torch.tensor(
                first_candidate_rows,
                dtype=torch.long,
                device=state_logits.device,
            ),
        )
        return ActionValuePrediction(
            logits=logits,
            probabilities=probabilities,
            expected_scores=_wdl_expected_score_unchecked(probabilities),
            state_logits=state_logits,
            information_set_state_logits=information_set_state_logits,
            candidate_counts=candidate_counts,
        )


def wdl_expected_score(probabilities: Tensor) -> Tensor:
    """Return win plus half draw for final-axis loss/draw/win probabilities."""
    if probabilities.ndim < 1 or probabilities.shape[-1] != WDL_OUTCOME_COUNT:
        raise ValueError("W/D/L probabilities must end in three outcomes")
    if not bool(torch.isfinite(probabilities).all().item()):
        raise ValueError("W/D/L probabilities must be finite")
    return _wdl_expected_score_unchecked(probabilities)


def _wdl_expected_score_unchecked(probabilities: Tensor) -> Tensor:
    """Compute expected score for an internally generated categorical tensor."""
    return probabilities[..., WDL_WIN_INDEX] + 0.5 * probabilities[..., WDL_DRAW_INDEX]


def validate_wdl_probabilities(
    probabilities: Tensor,
    *,
    atol: float = 1.0e-5,
) -> None:
    """Validate categorical W/D/L targets at a persistence boundary."""
    if probabilities.ndim < 1 or probabilities.shape[-1] != WDL_OUTCOME_COUNT:
        raise ValueError("W/D/L probabilities must end in three outcomes")
    if not bool(torch.isfinite(probabilities).all().item()):
        raise ValueError("W/D/L probabilities must be finite")
    if bool((probabilities < 0.0).any().item()):
        raise ValueError("W/D/L probabilities must be non-negative")
    totals = probabilities.sum(dim=-1)
    if not bool(torch.allclose(totals, torch.ones_like(totals), atol=atol, rtol=0.0)):
        raise ValueError("W/D/L probabilities must sum to one")


def _validate_latents(latents: Tensor, *, name: str) -> None:
    if latents.ndim != 2 or int(latents.shape[0]) <= 0:
        raise ValueError(f"{name} must be a non-empty rank-two tensor")
    if not latents.is_floating_point():
        raise TypeError(f"{name} must be floating point")


def _validate_candidate_counts(
    candidate_counts: tuple[int, ...],
    *,
    candidate_rows: int,
) -> None:
    if not candidate_counts or any(count <= 0 for count in candidate_counts):
        raise ValueError("every information set must retain at least one candidate")
    if sum(candidate_counts) != candidate_rows:
        raise ValueError("candidate counts do not cover the flattened rows")


__all__ = [
    "ACTION_VALUE_ARCHITECTURE_VERSION",
    "ActionValueHeadConfig",
    "ActionValuePrediction",
    "CompleteActionValueHead",
    "WDL_DRAW_INDEX",
    "WDL_LOSS_INDEX",
    "WDL_OUTCOME_COUNT",
    "WDL_WIN_INDEX",
    "validate_wdl_probabilities",
    "wdl_expected_score",
]
