"""Rows and diagnostic metrics for S2 trace stress."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ptcg_rl.evaluation.search_stress_config import SearchTraceStressConfig
from ptcg_rl.evaluation.search_stress_trace import FrozenTraceObservation


@dataclass
class StressRunStats:
    """Accumulate bounded per-cell runtime and safety counters."""

    callbacks: int = 0
    illegal_actions: int = 0
    agent_errors: int = 0
    telemetry_callbacks: int = 0
    search_roots: int = 0
    search_seconds: float = 0.0
    max_bank_spent_seconds: float = 0.0
    min_search_start_remaining_overage: float | None = None
    max_deadline_overshoot_seconds: float = 0.0
    max_whole_act_seconds: float = 0.0
    fallback_failures: int = 0
    state_leaks: int = 0
    max_state_pool_peak: int = 0
    recommendation_changes: int = 0
    action_changes: int = 0
    action_digest: Any = field(default_factory=hashlib.sha256)
    seen_error_signatures: set[str] = field(default_factory=set)

    def observe(
        self,
        *,
        global_step: int,
        action: Sequence[int],
        legal: bool,
        telemetry: Mapping[str, Any],
        errors: Sequence[str],
    ) -> None:
        """Record one callback without retaining its observation."""
        self.callbacks += 1
        self.illegal_actions += int(not legal)
        for error in errors:
            if error and error not in self.seen_error_signatures:
                self.seen_error_signatures.add(error)
                self.agent_errors += 1
        self.action_digest.update(
            f"{global_step}:{','.join(str(index) for index in action)}\n".encode()
        )
        if not telemetry:
            return
        self.telemetry_callbacks += 1
        search = float_or_zero(telemetry.get("actual_search_seconds"))
        self.search_seconds += search
        if search > 0.0:
            self.search_roots += 1
            remaining = optional_nonnegative_float(
                telemetry.get("search_start_remaining_overage_time")
            )
            if remaining is not None:
                current = self.min_search_start_remaining_overage
                self.min_search_start_remaining_overage = (
                    remaining if current is None else min(current, remaining)
                )
        self.max_bank_spent_seconds = max(
            self.max_bank_spent_seconds,
            float_or_zero(telemetry.get("bank_spent_seconds")),
        )
        self.max_deadline_overshoot_seconds = max(
            self.max_deadline_overshoot_seconds,
            float_or_zero(telemetry.get("deadline_overshoot_seconds")),
        )
        self.max_whole_act_seconds = max(
            self.max_whole_act_seconds,
            float_or_zero(telemetry.get("whole_act_seconds")),
        )
        self.fallback_failures += int(
            not bool(telemetry.get("fallback_available", False))
        )
        self.state_leaks += int(telemetry.get("state_leaks", 0) or 0)
        self.max_state_pool_peak = max(
            self.max_state_pool_peak,
            int(telemetry.get("state_pool_peak", 0) or 0),
        )
        self.recommendation_changes += int(
            bool(telemetry.get("recommended_action_changed", False))
        )
        self.action_changes += int(bool(telemetry.get("action_changed", False)))


def action_row(
    identity: Mapping[str, Any],
    *,
    run_kind: str,
    cell_id: str,
    controller: str,
    seat: int,
    global_steps: int,
    slowdown_factor: float,
    source: FrozenTraceObservation,
    remaining_before: float,
    remaining_after: float,
    real_elapsed: float,
    virtual_elapsed: float,
    action: Sequence[int],
    legal: bool,
    timed_out: bool,
    error_type: str | None,
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one schema-stable callback row."""
    return {
        "campaign_fp": identity["campaign_fp"],
        "stage_fp": identity["stage_fp"],
        "run_kind": run_kind,
        "cell_id": cell_id,
        "controller": controller,
        "seat": seat,
        "global_steps_requested": global_steps,
        "slowdown_factor": slowdown_factor,
        "global_step": source.global_step,
        "callback_index": source.callback_index,
        "source_replay_index": source.source_replay_index,
        "source_episode_id": source.source_episode_id,
        "source_step": source.source_step,
        "remaining_before": remaining_before,
        "remaining_after": remaining_after,
        "real_elapsed_seconds": real_elapsed,
        "virtual_elapsed_seconds": virtual_elapsed,
        "action": list(action),
        "legal": legal,
        "timed_out": timed_out,
        "error_type": error_type,
        "telemetry_present": bool(telemetry),
        "telemetry_stop_reason": telemetry.get("stop_reason"),
        "planned_quota_seconds": telemetry.get("planned_quota_seconds"),
        "actual_search_seconds": telemetry.get("actual_search_seconds"),
        "whole_act_seconds": telemetry.get("whole_act_seconds"),
        "startup_seconds": telemetry.get("startup_seconds"),
        "probe_seconds": telemetry.get("probe_seconds"),
        "base_policy_seconds": telemetry.get("base_policy_seconds"),
        "bank_spent_seconds": telemetry.get("bank_spent_seconds"),
        "deadline_overshoot_seconds": telemetry.get("deadline_overshoot_seconds"),
        "search_start_remaining_overage": telemetry.get(
            "search_start_remaining_overage_time"
        ),
        "candidates": telemetry.get("candidates"),
        "worlds_requested": telemetry.get("worlds_requested"),
        "worlds_completed": telemetry.get("worlds_completed"),
        "state_pool_peak": telemetry.get("state_pool_peak"),
        "state_leaks": telemetry.get("state_leaks"),
        "fallback_available": telemetry.get("fallback_available"),
        "recommendation_changed": telemetry.get("recommended_action_changed"),
        "action_changed": telemetry.get("action_changed"),
    }


def run_row(
    identity: Mapping[str, Any],
    *,
    run_kind: str,
    cell_id: str,
    controller: str,
    seat: int,
    global_steps_requested: int,
    global_steps_completed: int,
    slowdown_factor: float,
    boundary_remaining: float | None,
    source_replays: int,
    real_elapsed: float,
    virtual_elapsed: float,
    final_remaining: float,
    timed_out: bool,
    stats: StressRunStats,
) -> dict[str, Any]:
    """Build one schema-stable cell summary row."""
    return {
        "campaign_fp": identity["campaign_fp"],
        "stage_fp": identity["stage_fp"],
        "run_kind": run_kind,
        "cell_id": cell_id,
        "controller": controller,
        "seat": seat,
        "global_steps_requested": global_steps_requested,
        "global_steps_completed": global_steps_completed,
        "slowdown_factor": slowdown_factor,
        "boundary_remaining_overage": boundary_remaining,
        "callbacks": stats.callbacks,
        "source_replays": source_replays,
        "real_elapsed_seconds": real_elapsed,
        "virtual_elapsed_seconds": virtual_elapsed,
        "final_remaining_overage": final_remaining,
        "timed_out": timed_out,
        "illegal_actions": stats.illegal_actions,
        "agent_errors": stats.agent_errors,
        "telemetry_callbacks": stats.telemetry_callbacks,
        "telemetry_coverage": (
            stats.telemetry_callbacks / stats.callbacks if stats.callbacks else 0.0
        ),
        "search_roots": stats.search_roots,
        "search_seconds": stats.search_seconds,
        "max_bank_spent_seconds": stats.max_bank_spent_seconds,
        "min_search_start_remaining_overage": (
            stats.min_search_start_remaining_overage
        ),
        "max_deadline_overshoot_seconds": stats.max_deadline_overshoot_seconds,
        "max_whole_act_seconds": stats.max_whole_act_seconds,
        "fallback_failures": stats.fallback_failures,
        "state_leaks": stats.state_leaks,
        "max_state_pool_peak": stats.max_state_pool_peak,
        "recommendation_changes": stats.recommendation_changes,
        "action_changes": stats.action_changes,
        "action_fingerprint": stats.action_digest.hexdigest(),
        "equivalence_match": None,
        "safety_checks_observed": False,
        "diagnostic": "2.0x diagnostic" if slowdown_factor == 2.0 else "",
    }


def attach_action_equivalence(rows: Sequence[dict[str, Any]]) -> None:
    """Attach disabled-vs-shadow action equality to matching 300-step cells."""
    for seat in (0, 1):
        selected = [
            row
            for row in rows
            if row["run_kind"] == "trace"
            and row["seat"] == seat
            and row["global_steps_requested"] == 300
            and row["slowdown_factor"] == 1.0
            and row["controller"] in {"disabled", "shadow"}
        ]
        if len(selected) != 2:
            continue
        matched = selected[0]["action_fingerprint"] == selected[1][
            "action_fingerprint"
        ]
        for row in selected:
            row["equivalence_match"] = matched


def stress_cell_checks_observed(
    row: Mapping[str, Any], config: SearchTraceStressConfig
) -> bool:
    """Report callback, timing, cutoff, fallback, and lifecycle checks."""
    if row["run_kind"] == "boundary":
        return bool(
            int(row["search_roots"]) == 0
            and int(row["illegal_actions"]) == 0
            and int(row["agent_errors"]) == 0
            and int(row["fallback_failures"]) == 0
        )
    trace_complete = int(row["global_steps_completed"]) == int(
        row["global_steps_requested"]
    )
    search_remaining = optional_nonnegative_float(
        row["min_search_start_remaining_overage"]
    )
    cutoff_ok = search_remaining is None or (
        search_remaining > config.references.hard_search_cutoff_seconds
    )
    return bool(
        trace_complete
        and not bool(row["timed_out"])
        and int(row["illegal_actions"]) == 0
        and int(row["agent_errors"]) == 0
        and float(row["telemetry_coverage"])
        >= config.references.min_telemetry_coverage
        and float(row["max_bank_spent_seconds"])
        <= config.references.max_search_bank_seconds
        and cutoff_ok
        and float(row["max_deadline_overshoot_seconds"])
        <= config.references.max_deadline_overshoot_seconds
        and int(row["fallback_failures"]) == 0
        and int(row["state_leaks"]) == 0
        and int(row["max_state_pool_peak"]) <= 128
    )


def stress_summary(
    config: SearchTraceStressConfig,
    identity: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build campaign-level S2 trace readiness and reserve evidence."""
    primary = [
        row
        for row in rows
        if row["run_kind"] == "boundary" or float(row["slowdown_factor"]) <= 1.5
    ]
    trace_rows = [row for row in rows if row["run_kind"] == "trace"]
    equivalence_rows = [
        row for row in trace_rows if row["equivalence_match"] is not None
    ]
    reserve_1x = _reserves(trace_rows, slowdown_factor=1.0)
    reserve_1_25x = _reserves(trace_rows, slowdown_factor=1.25)
    checks = {
        "primary_cells": bool(primary)
        and all(bool(row["safety_checks_observed"]) for row in primary),
        "disabled_shadow_equivalence": bool(equivalence_rows)
        and all(bool(row["equivalence_match"]) for row in equivalence_rows),
        "reserve_1x_826": bool(reserve_1x)
        and min(reserve_1x) >= config.references.min_reserve_1x_826_seconds,
        "reserve_1_25x_826": bool(reserve_1_25x)
        and min(reserve_1_25x) >= config.references.min_reserve_1_25x_826_seconds,
        "startup_charge": (
            config.references.startup_charge_seconds
            <= config.references.max_startup_p99_seconds
        ),
    }
    return {
        "protocol": "ITS-EVAL-v1-S2-TRACE",
        "experiment_id": config.experiment_id,
        "campaign_fp": identity["campaign_fp"],
        "stage_fp": identity["stage_fp"],
        "elapsed_seconds": elapsed_seconds,
        "trace_cells": len(trace_rows),
        "boundary_cells": len(rows) - len(trace_rows),
        "runner_complete": len(rows)
        == len(config.cells) + 2 * len(config.overage_boundaries),
        "decision_role": "runtime_safety_diagnostic",
        "checks": checks,
        "diagnostic_warnings": [
            name for name, observed in checks.items() if not observed
        ],
        "min_reserve_1x_826_seconds": min(reserve_1x) if reserve_1x else None,
        "min_reserve_1_25x_826_seconds": (
            min(reserve_1_25x) if reserve_1_25x else None
        ),
        "diagnostic_2x": [
            dict(row) for row in trace_rows if float(row["slowdown_factor"]) == 2.0
        ],
        "runs": [dict(row) for row in rows],
        "config": config.model_dump(mode="json"),
    }


def _reserves(
    trace_rows: Sequence[Mapping[str, Any]], *, slowdown_factor: float
) -> list[float]:
    return [
        float(row["final_remaining_overage"])
        for row in trace_rows
        if row["controller"] == "override"
        and int(row["global_steps_requested"]) == 826
        and float(row["slowdown_factor"]) == slowdown_factor
    ]


def float_or_zero(value: Any) -> float:
    """Return a finite non-negative float, otherwise zero."""
    normalized = optional_nonnegative_float(value)
    return normalized if normalized is not None else 0.0


def optional_nonnegative_float(value: Any) -> float | None:
    """Normalize optional timing fields and reject invalid values."""
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) and normalized >= 0.0 else None
