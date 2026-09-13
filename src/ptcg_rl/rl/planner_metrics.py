"""Aggregation helpers for schema-9 planner learner metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor

from ptcg_rl.rl.planner_losses import PlannerImitationMetrics
from ptcg_rl.rl.ppo import PpoUpdateResult


@dataclass(frozen=True)
class PlannerUpdateMetrics:
    """Support-weighted planner metrics for one learner window."""

    planner_decisions: int
    applicable_decisions: int
    candidate_rerank_loss: float | None
    proposal_distillation_loss: float | None
    root_information_value_rows: int
    root_information_value_loss: float | None
    stale_decisions: int
    inexact_decisions: int
    incomplete_grid_decisions: int
    identity_invalid_decisions: int
    censored_support_decisions: int
    exhaustive_support_decisions: int
    clipped_candidates: int
    candidate_count: int

    @property
    def applicable_fraction(self) -> float:
        """Return the fraction of processed planner decisions used by imitation."""
        if self.planner_decisions <= 0:
            return 0.0
        return self.applicable_decisions / self.planner_decisions

    @property
    def clipped_candidate_fraction(self) -> float:
        """Return the fraction of retained candidates affected by target clipping."""
        if self.candidate_count <= 0:
            return 0.0
        return self.clipped_candidates / self.candidate_count


def aggregate_planner_updates(
    updates: Sequence[PpoUpdateResult],
) -> PlannerUpdateMetrics:
    """Aggregate planner losses by their decision or endpoint-row support."""
    planner_decisions = 0
    applicable_decisions = 0
    weighted_candidate_loss = 0.0
    weighted_proposal_loss = 0.0
    root_rows = 0
    weighted_root_loss = 0.0
    diagnostic_counts = {
        "stale_decisions": 0,
        "inexact_decisions": 0,
        "incomplete_grid_decisions": 0,
        "identity_invalid_decisions": 0,
        "censored_support_decisions": 0,
        "exhaustive_support_decisions": 0,
        "clipped_candidates": 0,
        "candidate_count": 0,
    }
    for update in updates:
        update_planner_decisions = int(update.planner_decisions)
        update_applicable = int(update.planner_applicable_decisions)
        update_root_rows = int(update.root_information_value_rows)
        if update_planner_decisions < 0 or update_applicable < 0:
            raise ValueError("planner decision counts cannot be negative")
        if update_applicable > update_planner_decisions:
            raise ValueError("applicable planner decisions exceed planner decisions")
        if update_root_rows < 0:
            raise ValueError("root-information value rows cannot be negative")

        planner_decisions += update_planner_decisions
        applicable_decisions += update_applicable
        if update_applicable > 0:
            candidate_loss = _required_loss_scalar(
                update.breakdown.candidate_rerank_loss,
                label="candidate rerank",
            )
            proposal_loss = _required_loss_scalar(
                update.breakdown.proposal_distillation_loss,
                label="proposal distillation",
            )
            weighted_candidate_loss += candidate_loss * update_applicable
            weighted_proposal_loss += proposal_loss * update_applicable

        root_rows += update_root_rows
        if update_root_rows > 0:
            root_loss = _required_loss_scalar(
                update.breakdown.root_information_value_loss,
                label="root-information value",
            )
            weighted_root_loss += root_loss * update_root_rows

        imitation = update.breakdown.planner_metrics
        if imitation is not None:
            for name in diagnostic_counts:
                value = int(getattr(imitation, name))
                if value < 0:
                    raise ValueError(f"planner metric {name} cannot be negative")
                diagnostic_counts[name] += value

    return PlannerUpdateMetrics(
        planner_decisions=planner_decisions,
        applicable_decisions=applicable_decisions,
        candidate_rerank_loss=(
            weighted_candidate_loss / applicable_decisions
            if applicable_decisions > 0
            else None
        ),
        proposal_distillation_loss=(
            weighted_proposal_loss / applicable_decisions
            if applicable_decisions > 0
            else None
        ),
        root_information_value_rows=root_rows,
        root_information_value_loss=(
            weighted_root_loss / root_rows if root_rows > 0 else None
        ),
        stale_decisions=diagnostic_counts["stale_decisions"],
        inexact_decisions=diagnostic_counts["inexact_decisions"],
        incomplete_grid_decisions=diagnostic_counts["incomplete_grid_decisions"],
        identity_invalid_decisions=diagnostic_counts["identity_invalid_decisions"],
        censored_support_decisions=diagnostic_counts["censored_support_decisions"],
        exhaustive_support_decisions=diagnostic_counts["exhaustive_support_decisions"],
        clipped_candidates=diagnostic_counts["clipped_candidates"],
        candidate_count=diagnostic_counts["candidate_count"],
    )


def planner_update_metrics_summary(
    metrics: PlannerUpdateMetrics,
) -> dict[str, int | float | None]:
    """Return stable operational field names for aggregated planner metrics."""
    return {
        "planner_decisions": metrics.planner_decisions,
        "planner_applicable_decisions": metrics.applicable_decisions,
        "planner_applicable_fraction": metrics.applicable_fraction,
        "candidate_rerank_loss": metrics.candidate_rerank_loss,
        "proposal_distillation_loss": metrics.proposal_distillation_loss,
        "root_information_value_rows": metrics.root_information_value_rows,
        "root_information_value_loss": metrics.root_information_value_loss,
        "planner_stale_decisions": metrics.stale_decisions,
        "planner_inexact_decisions": metrics.inexact_decisions,
        "planner_incomplete_grid_decisions": metrics.incomplete_grid_decisions,
        "planner_identity_invalid_decisions": metrics.identity_invalid_decisions,
        "planner_censored_support_decisions": metrics.censored_support_decisions,
        "planner_exhaustive_support_decisions": metrics.exhaustive_support_decisions,
        "planner_clipped_candidates": metrics.clipped_candidates,
        "planner_candidate_count": metrics.candidate_count,
        "planner_clipped_candidate_fraction": metrics.clipped_candidate_fraction,
    }


def planner_imitation_metrics_summary(
    metrics: PlannerImitationMetrics | None,
) -> dict[str, int | float]:
    """Return per-update applicability diagnostics, including an empty update."""
    if metrics is None:
        return {
            "planner_stale_decisions": 0,
            "planner_inexact_decisions": 0,
            "planner_incomplete_grid_decisions": 0,
            "planner_identity_invalid_decisions": 0,
            "planner_censored_support_decisions": 0,
            "planner_exhaustive_support_decisions": 0,
            "planner_clipped_candidates": 0,
            "planner_candidate_count": 0,
            "planner_clipped_candidate_fraction": 0.0,
        }
    candidate_count = int(metrics.candidate_count)
    clipped_candidates = int(metrics.clipped_candidates)
    return {
        "planner_stale_decisions": int(metrics.stale_decisions),
        "planner_inexact_decisions": int(metrics.inexact_decisions),
        "planner_incomplete_grid_decisions": int(metrics.incomplete_grid_decisions),
        "planner_identity_invalid_decisions": int(metrics.identity_invalid_decisions),
        "planner_censored_support_decisions": int(metrics.censored_support_decisions),
        "planner_exhaustive_support_decisions": int(
            metrics.exhaustive_support_decisions
        ),
        "planner_clipped_candidates": clipped_candidates,
        "planner_candidate_count": candidate_count,
        "planner_clipped_candidate_fraction": (
            clipped_candidates / candidate_count if candidate_count > 0 else 0.0
        ),
    }


def _required_loss_scalar(value: Tensor | None, *, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} support requires a reported loss")
    scalar = float(value.detach().item())
    if not math.isfinite(scalar):
        raise FloatingPointError(f"non-finite aggregate {label} loss")
    return scalar


__all__ = [
    "PlannerUpdateMetrics",
    "aggregate_planner_updates",
    "planner_imitation_metrics_summary",
    "planner_update_metrics_summary",
]
