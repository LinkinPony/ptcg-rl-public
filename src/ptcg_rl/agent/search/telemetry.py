"""Compact decision-level inference-search telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SearchActTelemetry:
    """One root's budget, lifecycle, coverage, and fallback diagnostics."""

    enabled: bool = False
    shadow_only: bool = True
    remaining_overage_time: float = 0.0
    search_start_remaining_overage_time: float | None = None
    startup_seconds: float = 0.0
    probe_seconds: float = 0.0
    base_policy_seconds: float = 0.0
    planned_quota_seconds: float = 0.0
    actual_search_seconds: float = 0.0
    whole_act_seconds: float = 0.0
    bank_spent_seconds: float = 0.0
    bank_left_seconds: float = 0.0
    deadline_overshoot_seconds: float = 0.0
    stop_reason: str = "disabled"
    candidates: int = 0
    worlds_requested: int = 0
    worlds_completed: int = 0
    transitions: int = 0
    engine_sessions: int = 0
    state_pool_peak: int = 0
    state_leaks: int = 0
    same_seat_value_rows: int = 0
    handoff_value_rows: int = 0
    selection_reason: str = "not_evaluated"
    override_gate_reason: str = "not_evaluated"
    recommended_action_changed: bool = False
    selected_mean_delta: float | None = None
    selected_robust_delta: float | None = None
    selected_downside_cvar: float | None = None
    fallback_available: bool = False
    action_changed: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/Parquet-friendly representation."""
        return asdict(self)
