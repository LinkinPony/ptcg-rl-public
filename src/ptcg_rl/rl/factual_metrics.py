"""Aggregation helpers for dense factual learner metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ptcg_rl.rl.ppo import PpoUpdateResult


@dataclass(frozen=True)
class FactualUpdateMetrics:
    """Decision-weighted factual metrics for one learner window."""

    decisions: int
    effect_loss: float | None
    successor_loss: float | None


def aggregate_factual_updates(
    updates: Sequence[PpoUpdateResult],
) -> FactualUpdateMetrics:
    """Aggregate dense factual losses without minibatch-size bias."""
    decisions = 0
    weighted = [0.0, 0.0]
    for update in updates:
        update_decisions = int(update.factual_decisions)
        if update_decisions < 0:
            raise ValueError("factual decision count cannot be negative")
        if update_decisions == 0:
            continue
        losses = (
            update.breakdown.factual_effect_loss,
            update.breakdown.factual_successor_loss,
        )
        if any(loss is None for loss in losses):
            raise ValueError("factual decisions require all reported losses")
        scalars = tuple(
            float(loss.detach().item()) for loss in losses if loss is not None
        )
        if not all(math.isfinite(value) for value in scalars):
            raise FloatingPointError("non-finite aggregate factual loss")
        decisions += update_decisions
        for index, scalar in enumerate(scalars):
            weighted[index] += scalar * update_decisions
    if decisions <= 0:
        return FactualUpdateMetrics(0, None, None)
    return FactualUpdateMetrics(
        decisions=decisions,
        effect_loss=weighted[0] / decisions,
        successor_loss=weighted[1] / decisions,
    )


__all__ = ["FactualUpdateMetrics", "aggregate_factual_updates"]
