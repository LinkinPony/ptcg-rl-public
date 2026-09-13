"""Aggregation helpers for root-only executed-macro learner metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ptcg_rl.rl.ppo import PpoUpdateResult


@dataclass(frozen=True)
class MacroCreditUpdateMetrics:
    """Root-weighted macro losses for one learner window."""

    roots: int
    conditional_loss: float | None
    expected_loss: float | None


def aggregate_macro_credit_updates(
    updates: Sequence[PpoUpdateResult],
) -> MacroCreditUpdateMetrics:
    """Aggregate macro losses without microbatch-size or root-density bias."""
    roots = 0
    weighted = [0.0, 0.0]
    for update in updates:
        update_roots = int(update.macro_roots)
        if update_roots < 0:
            raise ValueError("macro root count cannot be negative")
        if update_roots == 0:
            continue
        losses = (
            update.breakdown.macro_conditional_loss,
            update.breakdown.macro_expected_loss,
        )
        if any(loss is None for loss in losses):
            raise ValueError("macro roots require both reported losses")
        scalars = tuple(
            float(loss.detach().item()) for loss in losses if loss is not None
        )
        if not all(math.isfinite(value) for value in scalars):
            raise FloatingPointError("non-finite aggregate macro loss")
        roots += update_roots
        for index, scalar in enumerate(scalars):
            weighted[index] += scalar * update_roots
    if roots <= 0:
        return MacroCreditUpdateMetrics(0, None, None)
    return MacroCreditUpdateMetrics(
        roots=roots,
        conditional_loss=weighted[0] / roots,
        expected_loss=weighted[1] / roots,
    )


__all__ = ["MacroCreditUpdateMetrics", "aggregate_macro_credit_updates"]
