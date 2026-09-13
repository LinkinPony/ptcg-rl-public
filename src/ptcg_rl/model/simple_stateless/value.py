"""Distributional win/draw/loss root-value primitives."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

WDL_CLASS_COUNT = 3


class DistributionalWdlCritic(nn.Module):
    """Independent root critic over ordered loss, draw, and win outcomes."""

    def __init__(self, *, d_model: int, hidden_dim: int) -> None:
        """Build a compact nonlinear critic independent of the policy query."""
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("WDL critic dimensions must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, WDL_CLASS_COUNT),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return ordered ``[loss, draw, win]`` logits."""
        if inputs.ndim != 2:
            raise ValueError("WDL critic inputs must have shape [batch, d_model]")
        return cast(Tensor, self.network(inputs))


class ExactWdlValueResidual(nn.Module):
    """One exact deck's zero-output WDL-logit residual."""

    def __init__(self, *, d_model: int, bottleneck_dim: int) -> None:
        """Initialize a narrow route-private distributional correction."""
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(bottleneck_dim, WDL_CLASS_COUNT)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return one route's loss/draw/win logit correction."""
        return cast(Tensor, self.output(self.activation(self.down(inputs))))

    def zero_output(self) -> None:
        """Make the initial exact route equal the shared critic."""
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


def two_hot_wdl_targets(values: Tensor) -> Tensor:
    """Project scalar values onto the ordered ``[-1, 0, 1]`` WDL support."""
    if not values.is_floating_point():
        raise TypeError("WDL target values must be floating point")
    clipped = values.float().clamp(min=-1.0, max=1.0)
    loss = (-clipped).clamp_min(0.0)
    win = clipped.clamp_min(0.0)
    draw = 1.0 - loss - win
    return torch.stack((loss, draw, win), dim=-1)


def wdl_value_from_logits(logits: Tensor) -> Tensor:
    """Return ``P(win) - P(loss)`` using stable FP32 probabilities."""
    if logits.ndim < 1 or int(logits.shape[-1]) != WDL_CLASS_COUNT:
        raise ValueError("WDL logits must end with three outcome classes")
    probabilities = torch.softmax(logits.float(), dim=-1)
    return probabilities[..., 2] - probabilities[..., 0]


__all__ = [
    "DistributionalWdlCritic",
    "ExactWdlValueResidual",
    "WDL_CLASS_COUNT",
    "two_hot_wdl_targets",
    "wdl_value_from_logits",
]
