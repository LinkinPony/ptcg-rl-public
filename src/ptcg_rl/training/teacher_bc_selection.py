"""Held-out diagnostics for public-pilot behavior cloning runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.search_identity import file_sha256, write_identity_atomic


class TeacherBCArmConfig(BaseModel):
    """One immutable warm-start and two-epoch teacher BC output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: str
    run_dir: Path

    @field_validator("arm_id")
    @classmethod
    def nonempty_arm_id(cls, value: str) -> str:
        """Require a stable arm identity."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("teacher BC arm_id must be non-empty")
        return normalized


class TeacherBCBenefitThresholds(BaseModel):
    """Reference bands for held-out improvements and collapse signals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_sequence_nll_improvement: float = 0.02
    min_token_nll_improvement: float = 0.01
    min_top1_accuracy_improvement: float = 0.005
    max_top3_accuracy_regression: float = 0.002
    max_top5_accuracy_regression: float = 0.002
    min_embedding_std_ratio: float = 0.50
    min_value_prediction_std_ratio: float = 0.50

    @field_validator("*")
    @classmethod
    def nonnegative_finite(cls, value: float) -> float:
        """Reject invalid benefit thresholds."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("teacher BC thresholds must be finite and non-negative")
        return value


class TeacherBCSelectionConfig(BaseModel):
    """Hydra-facing teacher BC matrix and held-out sample identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str
    expected_validation_samples: int
    expected_epochs: int = 2
    arms: tuple[TeacherBCArmConfig, ...]
    thresholds: TeacherBCBenefitThresholds = TeacherBCBenefitThresholds()
    output_dir: Path

    @field_validator("experiment_id")
    @classmethod
    def immutable_experiment_id(cls, value: str) -> str:
        """Reject moving and blank experiment identities."""
        normalized = value.strip()
        if not normalized or "latest" in normalized.lower():
            raise ValueError("teacher BC experiment_id must be immutable")
        return normalized

    @field_validator("expected_validation_samples", "expected_epochs")
    @classmethod
    def positive_count(cls, value: int) -> int:
        """Require positive evaluation counters."""
        if value <= 0:
            raise ValueError("teacher BC counters must be positive")
        return value

    @field_validator("arms")
    @classmethod
    def unique_arms(
        cls,
        value: tuple[TeacherBCArmConfig, ...],
    ) -> tuple[TeacherBCArmConfig, ...]:
        """Require non-empty unique arm IDs and directories."""
        if not value:
            raise ValueError("teacher BC selection requires at least one arm")
        ids = [arm.arm_id for arm in value]
        paths = [records.repo_path(arm.run_dir).resolve() for arm in value]
        if len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
            raise ValueError("teacher BC arms must have unique IDs and run dirs")
        return value


def score_teacher_bc_benefit(config: TeacherBCSelectionConfig) -> dict[str, Any]:
    """Compare every best trained epoch with its immutable initial validation."""
    rows = [_score_arm(arm, config=config) for arm in config.arms]
    output_dir = records.repo_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "arms.parquet"
    _write_rows(rows_path, rows)
    summary = {
        "protocol": "PUBLIC-TEACHER-BC-DIAGNOSTIC-v2",
        "experiment_id": config.experiment_id,
        "expected_validation_samples": config.expected_validation_samples,
        "expected_epochs": config.expected_epochs,
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
    arm: TeacherBCArmConfig,
    *,
    config: TeacherBCSelectionConfig,
) -> dict[str, Any]:
    run_dir = records.repo_path(arm.run_dir)
    initial_path = _required_file(run_dir / "initial_validation.json", "initial metrics")
    metrics_path = _required_file(run_dir / "metrics.json", "epoch metrics")
    summary_path = _required_file(run_dir / "summary.json", "BC summary")
    checkpoint_path = _required_file(run_dir / "checkpoint_best.pt", "best checkpoint")
    initial = _read_object(initial_path)
    metrics = _read_object(metrics_path)
    summary = _read_object(summary_path)
    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"teacher BC history is empty: {metrics_path}")
    records_by_epoch = [
        dict(record) for record in history if isinstance(record, Mapping)
    ]
    if len(records_by_epoch) != len(history):
        raise ValueError(f"teacher BC history contains non-object rows: {metrics_path}")
    best_record = min(records_by_epoch, key=_validation_loss)
    trained = _mapping(best_record.get("validation"))
    thresholds = config.thresholds
    sequence_improvement = _delta(initial, trained, "action_sequence_nll", lower=True)
    token_improvement = _delta(initial, trained, "action_token_nll", lower=True)
    top1_improvement = _delta(
        initial,
        trained,
        "action_sequence_top1_accuracy",
        lower=False,
    )
    top3_improvement = _delta(
        initial,
        trained,
        "action_sequence_top3_accuracy",
        lower=False,
    )
    top5_improvement = _delta(
        initial,
        trained,
        "action_sequence_top5_accuracy",
        lower=False,
    )
    embedding_ratio = _ratio(trained, initial, "global_embedding_std")
    value_std_ratio = _ratio(trained, initial, "value_prediction_std")
    initial_samples = _integer(initial.get("samples"))
    trained_samples = _integer(trained.get("samples"))
    checks = {
        "epoch_count": len(records_by_epoch) == config.expected_epochs,
        "initial_sample_count": initial_samples == config.expected_validation_samples,
        "trained_sample_count": trained_samples == config.expected_validation_samples,
        "sequence_nll": sequence_improvement is not None
        and sequence_improvement >= thresholds.min_sequence_nll_improvement,
        "token_nll": token_improvement is not None
        and token_improvement >= thresholds.min_token_nll_improvement,
        "top1_accuracy": top1_improvement is not None
        and top1_improvement >= thresholds.min_top1_accuracy_improvement,
        "top3_nonregression": top3_improvement is not None
        and top3_improvement >= -thresholds.max_top3_accuracy_regression,
        "top5_nonregression": top5_improvement is not None
        and top5_improvement >= -thresholds.max_top5_accuracy_regression,
        "embedding_not_collapsed": embedding_ratio is not None
        and embedding_ratio >= thresholds.min_embedding_std_ratio,
        "value_not_collapsed": value_std_ratio is not None
        and value_std_ratio >= thresholds.min_value_prediction_std_ratio,
        "summary_initial_matches": _metrics_match(
            initial,
            _mapping(summary.get("initial_validation")),
        ),
    }
    warnings = [name for name, observed in checks.items() if not observed]
    return {
        "arm_id": arm.arm_id,
        "run_dir": records.display_path(run_dir),
        "initial_metrics_sha256": file_sha256(initial_path),
        "epoch_metrics_sha256": file_sha256(metrics_path),
        "summary_sha256": file_sha256(summary_path),
        "checkpoint_best_path": records.display_path(checkpoint_path),
        "checkpoint_best_sha256": file_sha256(checkpoint_path),
        "best_epoch": int(best_record.get("epoch", -1)),
        "initial_samples": initial_samples,
        "trained_samples": trained_samples,
        "initial_sequence_nll": _number(initial, "action_sequence_nll"),
        "trained_sequence_nll": _number(trained, "action_sequence_nll"),
        "sequence_nll_improvement": sequence_improvement,
        "initial_token_nll": _number(initial, "action_token_nll"),
        "trained_token_nll": _number(trained, "action_token_nll"),
        "token_nll_improvement": token_improvement,
        "top1_accuracy_improvement": top1_improvement,
        "top3_accuracy_improvement": top3_improvement,
        "top5_accuracy_improvement": top5_improvement,
        "embedding_std_ratio": embedding_ratio,
        "value_prediction_std_ratio": value_std_ratio,
        "diagnostic_warnings": warnings,
        "reference_checks": checks,
    }


def _validation_loss(record: Mapping[str, Any]) -> float:
    validation = _mapping(record.get("validation"))
    value = _number(validation, "loss")
    return value if value is not None else float("inf")


def _delta(
    initial: Mapping[str, Any],
    trained: Mapping[str, Any],
    field: str,
    *,
    lower: bool,
) -> float | None:
    initial_value = _number(initial, field)
    trained_value = _number(trained, field)
    if initial_value is None or trained_value is None:
        return None
    return initial_value - trained_value if lower else trained_value - initial_value


def _ratio(
    numerator: Mapping[str, Any],
    denominator: Mapping[str, Any],
    field: str,
) -> float | None:
    numerator_value = _number(numerator, field)
    denominator_value = _number(denominator, field)
    if numerator_value is None or denominator_value is None or denominator_value <= 0.0:
        return None
    return numerator_value / denominator_value


def _metrics_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "samples",
        "action_sequence_nll",
        "action_token_nll",
        "action_sequence_top1_accuracy",
        "action_sequence_top3_accuracy",
        "action_sequence_top5_accuracy",
    )
    return all(_values_close(left.get(field), right.get(field)) for field in fields)


def _values_close(left: Any, right: Any) -> bool:
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if not isinstance(left, int | float) or not isinstance(right, int | float):
        return False
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)


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
        f"# Public teacher BC: {summary['experiment_id']}",
        "",
        "Decision role: **diagnostic only**",
        "",
        "| arm | seq NLL delta | token NLL delta | top-1 delta | embedding ratio | warnings |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["arms"]:
        lines.append(
            "| {arm} | {sequence} | {token} | {top1} | {embedding} | {warnings} |".format(
                arm=row["arm_id"],
                sequence=_format_number(row["sequence_nll_improvement"]),
                token=_format_number(row["token_nll_improvement"]),
                top1=_format_number(row["top1_accuracy_improvement"]),
                embedding=_format_number(row["embedding_std_ratio"]),
                warnings=", ".join(row["diagnostic_warnings"]) or "none",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _number(mapping: Mapping[str, Any], field: str) -> float | None:
    value = mapping.get(field)
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return None
    return float(value)


def _integer(value: Any) -> int | None:
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
    "TeacherBCArmConfig",
    "TeacherBCBenefitThresholds",
    "TeacherBCSelectionConfig",
    "score_teacher_bc_benefit",
]
