"""Prompt-local credit assignment for autoregressive select actions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTokenCredit:
    """Advantages and value targets for one sampled prompt token trace."""

    advantages: tuple[float, ...]
    value_targets: tuple[float, ...]


def compute_prompt_token_credit(
    prefix_values: Sequence[float],
    *,
    downstream_return: float,
    gamma: float = 1.0,
    gae_lambda: float = 0.0,
) -> PromptTokenCredit:
    """Decompose one decision return over its active decode prefixes.

    Intermediate prompt transitions have zero reward. The last active token
    transitions to ``downstream_return``. With the mainline settings
    ``gamma=1`` and ``gae_lambda=0``, every token receives its one-step prefix
    value change and the contributions telescope to the decision advantage.
    """
    values = tuple(float(value) for value in prefix_values)
    if not values:
        raise ValueError("prefix_values must contain at least one active token")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("prefix_values must be finite")
    target = float(downstream_return)
    if not math.isfinite(target):
        raise ValueError("downstream_return must be finite")
    for name, value in (("gamma", gamma), ("gae_lambda", gae_lambda)):
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")

    next_values = (*values[1:], target)
    deltas = tuple(
        gamma * next_value - value
        for value, next_value in zip(values, next_values, strict=True)
    )
    advantages = _discounted_cumsum(deltas, gamma * gae_lambda)
    value_targets = tuple(
        (gamma ** (len(values) - index)) * target
        for index in range(len(values))
    )
    return PromptTokenCredit(
        advantages=advantages,
        value_targets=value_targets,
    )


def _discounted_cumsum(
    values: Sequence[float],
    discount: float,
) -> tuple[float, ...]:
    result = [0.0] * len(values)
    running = 0.0
    for index in range(len(values) - 1, -1, -1):
        running = float(values[index]) + discount * running
        result[index] = running
    return tuple(result)
