"""Diagnostic health summaries for controlled RL sprint arms."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.search_identity import file_sha256, write_identity_atomic


class RLSprintArmConfig(BaseModel):
    """One completed training run and its KL treatment identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: str
    run_dir: Path
    summary_path: Path | None = None
    kl_stop_mode: Literal["training_batch_delta", "fixed_reference"]

    @field_validator("arm_id")
    @classmethod
    def nonempty_arm_id(cls, value: str) -> str:
        """Require a stable arm label."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("RL sprint arm_id must be non-empty")
        return normalized


class RLSprintHealthThresholds(BaseModel):
    """Reference bands frozen before any arm result is read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_early_stop_fraction: float = 0.30
    max_clip_fraction_p95: float = 0.30
    max_stale_fraction_p95: float = 0.05
    max_overflow_decisions: int = 0
    max_non_finite_metrics: int = 0
    min_data_passes_p50: float = 0.95
    max_abs_initial_reference_kl_k3: float = 1.0e-6

    @field_validator(
        "max_early_stop_fraction",
        "max_clip_fraction_p95",
        "max_stale_fraction_p95",
    )
    @classmethod
    def probability(cls, value: float) -> float:
        """Require finite unit-interval reference values."""
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("RL sprint probability thresholds must be in [0, 1]")
        return value


class RLSprintHealthConfig(BaseModel):
    """Hydra-facing arm matrix and output identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str
    arms: tuple[RLSprintArmConfig, ...]
    expected_iterations: int = 300
    thresholds: RLSprintHealthThresholds = RLSprintHealthThresholds()
    output_dir: Path

    @field_validator("experiment_id")
    @classmethod
    def immutable_experiment_id(cls, value: str) -> str:
        """Reject blank or moving experiment labels."""
        normalized = value.strip()
        if not normalized or "latest" in normalized.lower():
            raise ValueError("RL sprint experiment_id must be immutable")
        return normalized

    @field_validator("arms")
    @classmethod
    def unique_arms(cls, value: tuple[RLSprintArmConfig, ...]) -> tuple[RLSprintArmConfig, ...]:
        """Require a non-empty matrix with unique IDs and paths."""
        if not value:
            raise ValueError("RL sprint health requires at least one arm")
        ids = [arm.arm_id for arm in value]
        paths = [records.repo_path(arm.run_dir).resolve() for arm in value]
        if len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
            raise ValueError("RL sprint arms must have unique IDs and run directories")
        return value


def score_rl_sprint_health(config: RLSprintHealthConfig) -> dict[str, Any]:
    """Publish training-health observations without filtering candidates."""
    rows = [_score_arm(arm, config=config) for arm in config.arms]
    output_dir = records.repo_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "arms.parquet"
    _write_rows(rows_path, rows)
    summary = {
        "protocol": "RL-SPRINT-HEALTH-v2-DIAGNOSTIC",
        "experiment_id": config.experiment_id,
        "expected_iterations": config.expected_iterations,
        "thresholds": config.thresholds.model_dump(mode="json"),
        "decision_role": "diagnostic_only",
        "rows_path": records.display_path(rows_path),
        "rows_sha256": file_sha256(rows_path),
        "arms": rows,
    }
    write_identity_atomic(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary)
    return summary


def _score_arm(
    arm: RLSprintArmConfig,
    *,
    config: RLSprintHealthConfig,
) -> dict[str, Any]:
    run_dir = records.repo_path(arm.run_dir)
    summary_path = _required_file(
        records.repo_path(arm.summary_path)
        if arm.summary_path is not None
        else run_dir / "summary.json",
        "RL arm summary",
    )
    latest_path = _required_file(run_dir / "weights" / "latest.json", "RL latest")
    summary = _read_object(summary_path)
    latest = _read_object(latest_path)
    health = _mapping(summary.get("health"))
    supervisor = _mapping(summary.get("supervisor"))
    iterations = summary.get("iterations")
    iteration_count = len(iterations) if isinstance(iterations, list) else 0
    early_stop_fraction = _number(health, "early_stop_fraction")
    clip_p95 = _distribution_number(health, "clip_fraction", "p95")
    stale_p95 = _distribution_number(health, "stale_fraction", "p95")
    data_passes_p50 = _distribution_number(health, "data_passes", "p50")
    initial_min = _distribution_number(
        health,
        "fixed_reference_initial_kl_k3",
        "min",
    )
    initial_max = _distribution_number(
        health,
        "fixed_reference_initial_kl_k3",
        "max",
    )
    initial_count = int(
        _mapping(health.get("fixed_reference_initial_kl_k3")).get("count", 0)
    )
    max_abs_initial = (
        max(abs(initial_min), abs(initial_max))
        if initial_min is not None and initial_max is not None
        else None
    )
    overflow_decisions = _optional_int(health.get("overflow_decisions"))
    non_finite = int(health.get("non_finite_metric_count", -1))
    thresholds = config.thresholds
    checks = {
        "completed_iterations": iteration_count == config.expected_iterations,
        "learner_exit_clean": supervisor.get("learner_exitcode") in {None, 0},
        "early_stop_fraction": early_stop_fraction is not None
        and early_stop_fraction <= thresholds.max_early_stop_fraction,
        "clip_fraction_p95": clip_p95 is not None
        and clip_p95 <= thresholds.max_clip_fraction_p95,
        "stale_fraction_p95": stale_p95 is not None
        and stale_p95 <= thresholds.max_stale_fraction_p95,
        "overflow_decisions": overflow_decisions is not None
        and overflow_decisions <= thresholds.max_overflow_decisions,
        "non_finite_metrics": non_finite >= 0
        and non_finite <= thresholds.max_non_finite_metrics,
        "data_passes_p50": data_passes_p50 is not None
        and data_passes_p50 >= thresholds.min_data_passes_p50,
        "fixed_reference_initial_kl": (
            True
            if arm.kl_stop_mode == "training_batch_delta"
            else initial_count == config.expected_iterations
            and max_abs_initial is not None
            and max_abs_initial <= thresholds.max_abs_initial_reference_kl_k3
        ),
    }
    warnings = [name for name, observed in checks.items() if not observed]
    return {
        "arm_id": arm.arm_id,
        "run_dir": records.display_path(run_dir),
        "summary_sha256": file_sha256(summary_path),
        "latest_sha256": file_sha256(latest_path),
        "checkpoint_version": int(latest.get("version", -1)),
        "checkpoint_path": str(latest.get("path", "")),
        "kl_stop_mode": arm.kl_stop_mode,
        "iterations": iteration_count,
        "effective_updates": int(health.get("effective_updates", 0)),
        "early_stop_fraction": early_stop_fraction,
        "clip_fraction_p95": clip_p95,
        "stale_fraction_p95": stale_p95,
        "overflow_decisions": overflow_decisions,
        "non_finite_metric_count": non_finite,
        "data_passes_p50": data_passes_p50,
        "max_abs_initial_reference_kl_k3": max_abs_initial,
        "diagnostic_warnings": warnings,
        "reference_checks": checks,
    }


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
        f"# RL Sprint health: {summary['experiment_id']}",
        "",
        "Decision role: **diagnostic only**",
        "",
        "| arm | early stop | clip p95 | stale p95 | overflow | data pass p50 | warnings |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["arms"]:
        lines.append(
            "| {arm_id} | {early} | {clip} | {stale} | "
            "{overflow} | {passes} | {warnings} |".format(
                arm_id=row["arm_id"],
                early=_format_number(row["early_stop_fraction"]),
                clip=_format_number(row["clip_fraction_p95"]),
                stale=_format_number(row["stale_fraction_p95"]),
                overflow=row["overflow_decisions"],
                passes=_format_number(row["data_passes_p50"]),
                warnings=", ".join(row["diagnostic_warnings"]) or "none",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _distribution_number(
    mapping: Mapping[str, Any],
    distribution: str,
    field: str,
) -> float | None:
    return _number(_mapping(mapping.get(distribution)), field)


def _number(mapping: Mapping[str, Any], field: str) -> float | None:
    value = mapping.get(field)
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


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
    return "n/a" if value is None else f"{float(value):.4f}"


__all__ = [
    "RLSprintArmConfig",
    "RLSprintHealthConfig",
    "RLSprintHealthThresholds",
    "score_rl_sprint_health",
]
