"""Validated outcome multipliers for replay behavior cloning."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReplayPretrainingOutcomeWeightsConfig(BaseModel):
    """Positive acting-side outcome multipliers for episode-equal BC."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    win: float = Field(default=1.0, gt=0.0)
    draw: float = Field(default=1.0, gt=0.0)
    loss: float = Field(default=1.0, gt=0.0)

    @field_validator("win", "draw", "loss")
    @classmethod
    def finite_multiplier(cls, value: float) -> float:
        """Reject a multiplier that cannot define a finite objective."""
        if not math.isfinite(value):
            raise ValueError("pretraining outcome multiplier must be finite")
        return value

    def multiplier(self, outcome: float) -> float:
        """Resolve one validated acting-side terminal outcome."""
        if outcome == 1.0:
            return self.win
        if outcome == 0.0:
            return self.draw
        if outcome == -1.0:
            return self.loss
        raise ValueError(f"unsupported pretraining outcome: {outcome}")


__all__ = ["ReplayPretrainingOutcomeWeightsConfig"]
