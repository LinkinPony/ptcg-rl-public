"""Aggregation helpers for sparse engine-teacher learner metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ptcg_rl.rl.ppo import PpoUpdateResult


@dataclass(frozen=True)
class EngineTeacherUpdateMetrics:
    """Decision-weighted engine-teacher metrics for one learner window."""

    decisions: int
    loss: float | None


def aggregate_engine_teacher_updates(
    updates: Sequence[PpoUpdateResult],
) -> EngineTeacherUpdateMetrics:
    """Aggregate sparse teacher loss without last-minibatch sampling bias."""
    decisions = 0
    weighted_loss = 0.0
    for update in updates:
        update_decisions = int(update.engine_teacher_decisions)
        if update_decisions < 0:
            raise ValueError("engine teacher decision count cannot be negative")
        if update_decisions == 0:
            continue
        loss = update.breakdown.engine_teacher_loss
        if loss is None:
            raise ValueError("engine teacher decisions require a reported loss")
        scalar = float(loss.detach().item())
        if not math.isfinite(scalar):
            raise FloatingPointError("non-finite aggregate engine teacher loss")
        decisions += update_decisions
        weighted_loss += scalar * update_decisions
    return EngineTeacherUpdateMetrics(
        decisions=decisions,
        loss=weighted_loss / decisions if decisions > 0 else None,
    )


__all__ = ["EngineTeacherUpdateMetrics", "aggregate_engine_teacher_updates"]
