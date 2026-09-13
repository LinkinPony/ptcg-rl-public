"""Diagnostic sampling summaries for specialist RL experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.search_identity import file_sha256, write_identity_atomic

_LANES = ("target", "near", "broad")
_OPPONENT_KINDS = ("self_play", "frozen", "scripted")


class CandidateLaneTargets(BaseModel):
    """Expected target/near/broad candidate sampling mass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: float
    near: float
    broad: float

    @field_validator("target", "near", "broad")
    @classmethod
    def probability(cls, value: float) -> float:
        """Require finite non-negative lane masses."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("candidate lane targets must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def normalized(self) -> CandidateLaneTargets:
        """Require lane targets to form one distribution."""
        if not math.isclose(self.target + self.near + self.broad, 1.0, abs_tol=1e-9):
            raise ValueError("candidate lane targets must sum to one")
        return self


class OpponentMixTargets(BaseModel):
    """Expected self-play/frozen/scripted opponent sampling mass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    self_play: float
    frozen: float
    scripted: float

    @field_validator("self_play", "frozen", "scripted")
    @classmethod
    def probability(cls, value: float) -> float:
        """Require finite non-negative opponent masses."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("opponent mix targets must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def normalized(self) -> OpponentMixTargets:
        """Require opponent targets to form one distribution."""
        if not math.isclose(
            self.self_play + self.frozen + self.scripted,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("opponent mix targets must sum to one")
        return self


class RLCurriculumArmConfig(BaseModel):
    """One completed curriculum arm and its pre-registered treatment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: str
    run_dir: Path
    curriculum_summary_path: Path
    candidate_lanes_enabled: bool
    lane_targets: CandidateLaneTargets
    opponent_mix: OpponentMixTargets
    matchup_priority_enabled: bool
    anchor_max_total_probability: float | None = 0.15

    @field_validator("arm_id")
    @classmethod
    def nonempty_arm_id(cls, value: str) -> str:
        """Require a stable non-empty arm identity."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("curriculum arm_id must be non-empty")
        return normalized

    @field_validator("anchor_max_total_probability")
    @classmethod
    def optional_probability(cls, value: float | None) -> float | None:
        """Validate the pre-registered anchor cap."""
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("anchor cap must be in [0, 1]")
        return value


class RLCurriculumHealthConfig(BaseModel):
    """Hydra-facing curriculum experiment matrix and reference bands."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str
    expected_iterations: int
    arms: tuple[RLCurriculumArmConfig, ...]
    max_lane_deviation: float = 0.02
    max_opponent_mix_deviation: float = 0.02
    max_stale_fraction_p95: float = 0.05
    min_frozen_members: int = 6
    output_dir: Path

    @field_validator("experiment_id")
    @classmethod
    def immutable_experiment_id(cls, value: str) -> str:
        """Reject moving or blank experiment identities."""
        normalized = value.strip()
        if not normalized or "latest" in normalized.lower():
            raise ValueError("curriculum experiment_id must be immutable")
        return normalized

    @field_validator("expected_iterations", "min_frozen_members")
    @classmethod
    def positive_integer(cls, value: int) -> int:
        """Require positive experiment counters."""
        if value <= 0:
            raise ValueError("curriculum experiment counters must be positive")
        return value

    @field_validator(
        "max_lane_deviation",
        "max_opponent_mix_deviation",
        "max_stale_fraction_p95",
    )
    @classmethod
    def unit_interval(cls, value: float) -> float:
        """Require finite unit-interval thresholds."""
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("curriculum thresholds must be in [0, 1]")
        return value

    @field_validator("arms")
    @classmethod
    def unique_arms(
        cls,
        value: tuple[RLCurriculumArmConfig, ...],
    ) -> tuple[RLCurriculumArmConfig, ...]:
        """Require a non-empty matrix with unique IDs and artifact paths."""
        if not value:
            raise ValueError("curriculum health requires at least one arm")
        ids = [arm.arm_id for arm in value]
        paths = [records.repo_path(arm.run_dir).resolve() for arm in value]
        if len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
            raise ValueError("curriculum arms must have unique IDs and run dirs")
        return value


def score_rl_curriculum_health(
    config: RLCurriculumHealthConfig,
) -> dict[str, Any]:
    """Audit configured and observed curriculum distributions for every arm."""
    rows = [_score_arm(arm, config=config) for arm in config.arms]
    output_dir = records.repo_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "arms.parquet"
    _write_rows(rows_path, rows)
    summary = {
        "protocol": "RL-CURRICULUM-HEALTH-v1",
        "experiment_id": config.experiment_id,
        "expected_iterations": config.expected_iterations,
        "thresholds": {
            "max_lane_deviation": config.max_lane_deviation,
            "max_opponent_mix_deviation": config.max_opponent_mix_deviation,
            "max_stale_fraction_p95": config.max_stale_fraction_p95,
            "min_frozen_members": config.min_frozen_members,
        },
        "decision_role": "diagnostic_only",
        "rows_path": records.display_path(rows_path),
        "rows_sha256": file_sha256(rows_path),
        "arms": rows,
    }
    write_identity_atomic(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary)
    return summary


def _score_arm(
    arm: RLCurriculumArmConfig,
    *,
    config: RLCurriculumHealthConfig,
) -> dict[str, Any]:
    run_dir = records.repo_path(arm.run_dir)
    run_summary_path = _required_file(run_dir / "summary.json", "RL run summary")
    curriculum_path = _required_file(
        records.repo_path(arm.curriculum_summary_path),
        "curriculum summary",
    )
    run_summary = _read_object(run_summary_path)
    curriculum = _read_object(curriculum_path)
    curriculum_config = _mapping(curriculum.get("curriculum_config"))
    configured_lanes = _mapping(curriculum_config.get("candidate_lanes"))
    configured_mix = _mapping(curriculum_config.get("mix"))
    configured_matchup = _mapping(curriculum_config.get("matchup_priority"))
    observed_lanes = _mapping(curriculum.get("candidate_lanes"))
    assigned = _mapping(curriculum.get("assigned"))

    lane_targets = arm.lane_targets.model_dump(mode="python")
    actual_lanes = {
        lane: _optional_number(_mapping(observed_lanes.get(lane)).get("assigned_fraction"))
        for lane in _LANES
    }
    lane_deviations = {
        lane: _absolute_deviation(actual_lanes[lane], float(lane_targets[lane]))
        for lane in _LANES
    }
    opponent_targets = arm.opponent_mix.model_dump(mode="python")
    opponent_total = sum(_nonnegative_count(assigned.get(kind)) for kind in _OPPONENT_KINDS)
    actual_opponent_mix = {
        kind: (
            _nonnegative_count(assigned.get(kind)) / float(opponent_total)
            if opponent_total > 0
            else None
        )
        for kind in _OPPONENT_KINDS
    }
    opponent_deviations = {
        kind: _absolute_deviation(
            actual_opponent_mix[kind],
            float(opponent_targets[kind]),
        )
        for kind in _OPPONENT_KINDS
    }
    configured_anchor_id = str(
        curriculum_config.get("anchor_opponent_id", "anchor")
    ).strip()
    anchor_actual_fraction = (
        _nonnegative_count(assigned.get(f"frozen:{configured_anchor_id}"))
        / float(opponent_total)
        if configured_anchor_id and opponent_total > 0
        else None
    )
    matchups = curriculum.get("matchups")
    matchup_rows = matchups if isinstance(matchups, list) else []
    valid_matchup_rows = sum(_valid_matchup_row(row) for row in matchup_rows)
    health = _mapping(run_summary.get("health"))
    stale_p95 = _distribution_number(health, "stale_fraction", "p95")
    iterations = run_summary.get("iterations")
    iteration_count = len(iterations) if isinstance(iterations, list) else 0
    configured_anchor_cap = _optional_number(
        curriculum_config.get("anchor_max_total_probability")
    )
    checks = {
        "completed_iterations": iteration_count == config.expected_iterations,
        "curriculum_completed": curriculum.get("status") == "completed",
        "candidate_lane_mode": bool(configured_lanes.get("enabled"))
        is arm.candidate_lanes_enabled,
        "candidate_lane_config": not arm.candidate_lanes_enabled
        or all(
            _numbers_close(configured_lanes.get(f"{lane}_probability"), target)
            for lane, target in lane_targets.items()
        ),
        "candidate_lane_actual": all(
            deviation is not None and deviation <= config.max_lane_deviation
            for deviation in lane_deviations.values()
        ),
        "opponent_mix_config": all(
            _numbers_close(configured_mix.get(kind), target)
            for kind, target in opponent_targets.items()
        ),
        "opponent_mix_actual": all(
            deviation is not None
            and deviation <= config.max_opponent_mix_deviation
            for deviation in opponent_deviations.values()
        ),
        "matchup_priority_mode": bool(configured_matchup.get("enabled"))
        is arm.matchup_priority_enabled,
        "matchup_statistics_observable": bool(matchup_rows)
        and valid_matchup_rows == len(matchup_rows),
        "anchor_cap": _optional_numbers_close(
            configured_anchor_cap,
            arm.anchor_max_total_probability,
        ),
        "anchor_actual": arm.anchor_max_total_probability is None
        or (
            anchor_actual_fraction is not None
            and anchor_actual_fraction
            <= arm.anchor_max_total_probability + config.max_opponent_mix_deviation
        ),
        "repaired_frozen_pool": int(curriculum.get("frozen_members", 0))
        >= config.min_frozen_members,
        "zero_overflow": int(health.get("overflow_decisions", -1)) == 0,
        "finite_training": int(health.get("non_finite_metric_count", -1)) == 0,
        "staleness": stale_p95 is not None
        and stale_p95 <= config.max_stale_fraction_p95,
    }
    warnings = [name for name, observed in checks.items() if not observed]
    return {
        "arm_id": arm.arm_id,
        "run_dir": records.display_path(run_dir),
        "run_summary_sha256": file_sha256(run_summary_path),
        "curriculum_summary_path": records.display_path(curriculum_path),
        "curriculum_summary_sha256": file_sha256(curriculum_path),
        "iterations": iteration_count,
        "candidate_lanes_enabled": arm.candidate_lanes_enabled,
        "lane_targets": lane_targets,
        "lane_actual": actual_lanes,
        "lane_deviations": lane_deviations,
        "opponent_mix_targets": opponent_targets,
        "opponent_mix_actual": actual_opponent_mix,
        "opponent_mix_deviations": opponent_deviations,
        "anchor_actual_fraction": anchor_actual_fraction,
        "matchup_priority_enabled": arm.matchup_priority_enabled,
        "matchup_rows": len(matchup_rows),
        "matchup_games": sum(
            int(_mapping(row).get("games", 0)) for row in matchup_rows
        ),
        "frozen_members": int(curriculum.get("frozen_members", 0)),
        "stale_fraction_p95": stale_p95,
        "overflow_decisions": int(health.get("overflow_decisions", -1)),
        "non_finite_metric_count": int(health.get("non_finite_metric_count", -1)),
        "diagnostic_warnings": warnings,
        "reference_checks": checks,
    }


def _valid_matchup_row(value: Any) -> bool:
    row = _mapping(value)
    return bool(
        str(row.get("candidate_deck", "")).strip()
        and str(row.get("opponent_deck", "")).strip()
        and str(row.get("opponent_pilot", "")).strip()
        and isinstance(row.get("games"), int)
        and int(row.get("games", 0)) > 0
    )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    table_rows = [
        {
            key: json.dumps(value, sort_keys=True)
            if isinstance(value, list | dict)
            else value
            for key, value in row.items()
        }
        for row in rows
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(table_rows), temporary, compression="zstd")
    temporary.replace(path)


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        f"# RL curriculum health: {summary['experiment_id']}",
        "",
        "Decision role: **diagnostic only**",
        "",
        "| arm | target/near/broad | self/frozen/scripted | matchup rows | warnings |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in summary["arms"]:
        lane_actual = _mapping(row["lane_actual"])
        opponent_actual = _mapping(row["opponent_mix_actual"])
        lines.append(
            "| {arm} | {lanes} | {opponents} | {matchups} | {warnings} |".format(
                arm=row["arm_id"],
                lanes="/".join(_format_number(lane_actual.get(lane)) for lane in _LANES),
                opponents="/".join(
                    _format_number(opponent_actual.get(kind))
                    for kind in _OPPONENT_KINDS
                ),
                matchups=row["matchup_rows"],
                warnings=", ".join(row["diagnostic_warnings"]) or "none",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _distribution_number(
    mapping: Mapping[str, Any],
    distribution: str,
    field: str,
) -> float | None:
    return _optional_number(_mapping(mapping.get(distribution)).get(field))


def _absolute_deviation(value: float | None, target: float) -> float | None:
    return None if value is None else abs(value - target)


def _numbers_close(value: Any, expected: Any) -> bool:
    number = _optional_number(value)
    target = _optional_number(expected)
    return number is not None and target is not None and math.isclose(
        number,
        target,
        abs_tol=1e-9,
    )


def _optional_numbers_close(value: float | None, expected: float | None) -> bool:
    if value is None or expected is None:
        return value is expected
    return math.isclose(value, expected, abs_tol=1e-9)


def _optional_number(value: Any) -> float | None:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return None
    return float(value)


def _nonnegative_count(value: Any) -> int:
    return int(value) if isinstance(value, int) and value >= 0 else 0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw


def _required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _format_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


__all__ = [
    "CandidateLaneTargets",
    "OpponentMixTargets",
    "RLCurriculumArmConfig",
    "RLCurriculumHealthConfig",
    "score_rl_curriculum_health",
]
