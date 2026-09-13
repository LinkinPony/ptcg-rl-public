"""Stable compact reasons for the planner's base-policy fallback branch."""

from __future__ import annotations

from enum import IntEnum


class PlannerFallbackReason(IntEnum):
    """Wire-stable primary reason that planner evidence was not consumed.

    The value is intentionally compact.  Detailed diagnostics belong in
    telemetry, while a trajectory needs only the primary branch reason.
    """

    NONE = 0
    INELIGIBLE = 1
    EVIDENCE_ABSENT = 2
    CONSTRUCTOR_INVALID = 3
    MANDATORY_ANCHOR_OVER_BUDGET = 4
    UNSUPPORTED_CHANCE = 5
    SCENARIO_GRID_INCOMPLETE = 6
    RULES_INEXACT = 7
    LEGALITY_INCONSISTENT = 8
    NONANTICIPATIVITY_VIOLATION = 9
    FINGERPRINT_MISMATCH = 10
    DEADLINE = 11
    QUEUE_FULL = 12
    ENGINE_ERROR = 13
    LEAF_VALUE_UNAVAILABLE = 14
    MODEL_VERSION_MISMATCH = 15
    BUDGET_TRUNCATED = 16
    MODEL_LEASE_CAPACITY = 17


class PlannerEvidenceError(ValueError):
    """Planner evidence error carrying one compact fallback reason."""

    def __init__(self, reason: PlannerFallbackReason, detail: str) -> None:
        if reason is PlannerFallbackReason.NONE:
            raise ValueError("planner evidence errors require a fallback reason")
        super().__init__(detail)
        self.reason = reason


__all__ = ["PlannerEvidenceError", "PlannerFallbackReason"]
