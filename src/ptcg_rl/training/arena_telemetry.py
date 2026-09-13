"""Bounded game-level aggregation of optional runtime callback telemetry."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeTelemetryAccumulator:
    """Aggregate compact runtime phases without retaining callback rows."""

    expected: bool
    callbacks: int = 0
    missing: int = 0
    search_roots: int = 0
    search_seconds: float = 0.0
    whole_act_seconds: float = 0.0
    startup_seconds: float = 0.0
    probe_seconds: float = 0.0
    base_policy_seconds: float = 0.0
    runner_overhead_seconds: float = 0.0
    max_whole_act_seconds: float = 0.0
    max_deadline_overshoot_seconds: float = 0.0
    max_bank_spent_seconds: float = 0.0
    min_search_start_remaining_overage: float | None = None
    action_changes: int = 0
    recommendation_changes: int = 0
    fallback_failures: int = 0
    state_leaks: int = 0
    max_state_pool_peak: int = 0
    stop_reasons: Counter[str] = field(default_factory=Counter)

    def record(self, agent: Any, callback_seconds: float) -> None:
        """Collect one callback if the agent advertises runtime telemetry."""
        if not self.expected:
            return
        telemetry_method = getattr(agent, "last_act_telemetry", None)
        telemetry = telemetry_method() if callable(telemetry_method) else None
        if not isinstance(telemetry, Mapping):
            self.missing += 1
            return
        self.callbacks += 1
        whole_act = _telemetry_float(telemetry, "whole_act_seconds")
        search = _telemetry_float(telemetry, "actual_search_seconds")
        self.whole_act_seconds += whole_act
        self.search_seconds += search
        self.startup_seconds += _telemetry_float(telemetry, "startup_seconds")
        self.probe_seconds += _telemetry_float(telemetry, "probe_seconds")
        self.base_policy_seconds += _telemetry_float(
            telemetry,
            "base_policy_seconds",
        )
        self.runner_overhead_seconds += max(0.0, callback_seconds - whole_act)
        self.max_whole_act_seconds = max(self.max_whole_act_seconds, whole_act)
        self.max_deadline_overshoot_seconds = max(
            self.max_deadline_overshoot_seconds,
            _telemetry_float(telemetry, "deadline_overshoot_seconds"),
        )
        self.max_bank_spent_seconds = max(
            self.max_bank_spent_seconds,
            _telemetry_float(telemetry, "bank_spent_seconds"),
        )
        if search > 0.0:
            self.search_roots += 1
            remaining = _telemetry_optional_float(
                telemetry,
                "search_start_remaining_overage_time",
            )
            if remaining is not None:
                current = self.min_search_start_remaining_overage
                self.min_search_start_remaining_overage = (
                    remaining if current is None else min(current, remaining)
                )
        self.action_changes += int(bool(telemetry.get("action_changed", False)))
        self.recommendation_changes += int(
            bool(telemetry.get("recommended_action_changed", False))
        )
        self.fallback_failures += int(
            not bool(telemetry.get("fallback_available", False))
        )
        self.state_leaks += _telemetry_int(telemetry, "state_leaks")
        self.max_state_pool_peak = max(
            self.max_state_pool_peak,
            _telemetry_int(telemetry, "state_pool_peak"),
        )
        self.stop_reasons[str(telemetry.get("stop_reason", "missing"))] += 1

    def row_fields(self, prefix: str) -> dict[str, Any]:
        """Return flat Parquet-safe game fields for one role."""
        return {
            f"{prefix}_runtime_telemetry_expected": self.expected,
            f"{prefix}_runtime_telemetry_callbacks": self.callbacks,
            f"{prefix}_runtime_telemetry_missing": self.missing,
            f"{prefix}_runtime_search_roots": self.search_roots,
            f"{prefix}_runtime_search_seconds": self.search_seconds,
            f"{prefix}_runtime_whole_act_seconds": self.whole_act_seconds,
            f"{prefix}_runtime_startup_seconds": self.startup_seconds,
            f"{prefix}_runtime_probe_seconds": self.probe_seconds,
            f"{prefix}_runtime_base_policy_seconds": self.base_policy_seconds,
            f"{prefix}_runtime_runner_overhead_seconds": self.runner_overhead_seconds,
            f"{prefix}_runtime_max_whole_act_seconds": self.max_whole_act_seconds,
            f"{prefix}_runtime_max_deadline_overshoot_seconds": (
                self.max_deadline_overshoot_seconds
            ),
            f"{prefix}_runtime_max_bank_spent_seconds": self.max_bank_spent_seconds,
            f"{prefix}_runtime_min_search_start_remaining_overage": (
                self.min_search_start_remaining_overage
            ),
            f"{prefix}_runtime_action_changes": self.action_changes,
            f"{prefix}_runtime_recommendation_changes": self.recommendation_changes,
            f"{prefix}_runtime_fallback_failures": self.fallback_failures,
            f"{prefix}_runtime_state_leaks": self.state_leaks,
            f"{prefix}_runtime_max_state_pool_peak": self.max_state_pool_peak,
            f"{prefix}_runtime_stop_reasons": json.dumps(
                dict(sorted(self.stop_reasons.items())),
                sort_keys=True,
            ),
        }


def _telemetry_float(telemetry: Mapping[str, Any], name: str) -> float:
    value = _telemetry_optional_float(telemetry, name)
    return value if value is not None else 0.0


def _telemetry_optional_float(
    telemetry: Mapping[str, Any],
    name: str,
) -> float | None:
    value = telemetry.get(name)
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) and normalized >= 0.0 else None


def _telemetry_int(telemetry: Mapping[str, Any], name: str) -> int:
    value = telemetry.get(name)
    if value is None:
        return 0
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, normalized)
