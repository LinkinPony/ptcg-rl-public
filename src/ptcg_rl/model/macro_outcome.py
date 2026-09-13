"""Zero-start student heads for executed-macro factual consequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor, nn

from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.macro_credit_schema import (
    MACRO_CONTINUATION_SUMMARY_SIZE,
    MACRO_ENDPOINT_COUNT,
)


class MacroOutcomeHeadConfig(BaseModel):
    """Fixed geometry for conditional and behavior-expected macro heads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hidden_dim: int = 64
    continuation_summary_size: int = MACRO_CONTINUATION_SUMMARY_SIZE

    @field_validator("hidden_dim")
    @classmethod
    def positive_hidden_dim(cls, value: int) -> int:
        """Reject an empty auxiliary projection."""
        if value <= 0:
            raise ValueError("macro outcome hidden dimension must be positive")
        return value

    @field_validator("continuation_summary_size")
    @classmethod
    def supported_summary_size(cls, value: int) -> int:
        """Bind checkpoints to the implemented compact featurizer."""
        if value != MACRO_CONTINUATION_SUMMARY_SIZE:
            raise ValueError("unsupported macro continuation summary size")
        return value


@dataclass(frozen=True)
class MacroOutcomePrediction:
    """Conditional and expected predictions aligned with selected macro roots."""

    conditional_presence_logits: Tensor
    conditional_magnitude_predictions: Tensor
    conditional_endpoint_logits: Tensor
    expected_presence_logits: Tensor
    expected_magnitude_predictions: Tensor
    expected_endpoint_logits: Tensor


class _MacroOutcomeBranch(nn.Module):
    """One completed-action projection with independent zero-start outputs."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.presence = nn.Linear(hidden_dim, DYNAMIC_EFFECT_FEATURE_SIZE)
        self.magnitude = nn.Linear(hidden_dim, DYNAMIC_EFFECT_FEATURE_SIZE)
        self.endpoint = nn.Linear(hidden_dim, MACRO_ENDPOINT_COUNT)
        self.reset_output_parameters()

    def reset_output_parameters(self) -> None:
        """Keep warm-start behavior and initial auxiliary outputs invariant."""
        for output in (self.presence, self.magnitude, self.endpoint):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = cast(Tensor, self.trunk(inputs))
        return (
            cast(Tensor, self.presence(hidden)),
            torch.tanh(cast(Tensor, self.magnitude(hidden))),
            cast(Tensor, self.endpoint(hidden)),
        )


class MacroOutcomeHeads(nn.Module):
    """Predict factual realized macros and their behavior-controller expectation."""

    def __init__(
        self,
        d_model: int,
        config: MacroOutcomeHeadConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MacroOutcomeHeadConfig()
        self.expected = _MacroOutcomeBranch(d_model, self.config.hidden_dim)
        self.conditional = _MacroOutcomeBranch(
            d_model + self.config.continuation_summary_size,
            self.config.hidden_dim,
        )

    def forward(
        self,
        completed_action_latents: Tensor,
        continuation_summaries: Tensor,
    ) -> MacroOutcomePrediction:
        """Return both heads from one already-computed action latent."""
        if completed_action_latents.ndim != 2:
            raise ValueError("macro action latents must be [batch, d_model]")
        if continuation_summaries.shape != (
            completed_action_latents.shape[0],
            self.config.continuation_summary_size,
        ):
            raise ValueError("macro continuation summaries are misaligned")
        summary = continuation_summaries.to(
            device=completed_action_latents.device,
            dtype=completed_action_latents.dtype,
        )
        expected = self.expected(completed_action_latents)
        conditional = self.conditional(
            torch.cat((completed_action_latents, summary), dim=1)
        )
        return MacroOutcomePrediction(
            conditional_presence_logits=conditional[0],
            conditional_magnitude_predictions=conditional[1],
            conditional_endpoint_logits=conditional[2],
            expected_presence_logits=expected[0],
            expected_magnitude_predictions=expected[1],
            expected_endpoint_logits=expected[2],
        )


__all__ = [
    "MacroOutcomeHeadConfig",
    "MacroOutcomeHeads",
    "MacroOutcomePrediction",
]
