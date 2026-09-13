"""Auditable health-summary recovery across exact-resume process segments."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.search_identity import file_sha256, write_identity_atomic

_REQUIRED_TAGS = (
    "learner/updates",
    "batch/kept_decisions",
    "batch/stale_fraction",
    "ppo/early_stopped",
    "ppo/kl_stop_baseline_k3",
)
_DIAGNOSTIC_TAGS = (
    "ppo/clip_fraction",
    "ppo/grad_norm",
    "ppo/ratio_p95",
)


class RLHealthEventSegment(BaseModel):
    """One TensorBoard segment and the committed iteration interval it owns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    event_path: Path
    iteration_start: int
    iteration_end: int

    @field_validator("label")
    @classmethod
    def nonempty_label(cls, value: str) -> str:
        """Require a stable segment label."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("resume health segment label must be non-empty")
        return normalized

    @model_validator(mode="after")
    def valid_interval(self) -> RLHealthEventSegment:
        """Require a non-empty half-open iteration interval."""
        if self.iteration_start < 0 or self.iteration_end <= self.iteration_start:
            raise ValueError("resume health segment interval must be non-empty")
        return self


class RLResumeHealthConfig(BaseModel):
    """Inputs for reconstructing one cumulative health-only run summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str
    expected_iterations: int
    effective_batch_size: int
    segments: tuple[RLHealthEventSegment, ...]
    base_summary_path: Path
    distributed_summary_paths: tuple[Path, ...]
    provenance_paths: tuple[Path, ...] = ()
    output_summary_path: Path
    audit_output_path: Path

    @field_validator("experiment_id")
    @classmethod
    def immutable_experiment_id(cls, value: str) -> str:
        """Reject blank or moving recovery identities."""
        normalized = value.strip()
        if not normalized or "latest" in normalized.lower():
            raise ValueError("resume health experiment_id must be immutable")
        return normalized

    @field_validator("expected_iterations", "effective_batch_size")
    @classmethod
    def positive_integer(cls, value: int) -> int:
        """Require positive iteration and batch sizes."""
        if value <= 0:
            raise ValueError("resume health sizes must be positive")
        return value

    @model_validator(mode="after")
    def contiguous_segments(self) -> RLResumeHealthConfig:
        """Require an exact, ordered partition of the expected iterations."""
        if not self.segments:
            raise ValueError("resume health recovery needs at least one segment")
        expected_start = 0
        labels: set[str] = set()
        for segment in self.segments:
            if segment.label in labels:
                raise ValueError("resume health segment labels must be unique")
            labels.add(segment.label)
            if segment.iteration_start != expected_start:
                raise ValueError("resume health segments must be contiguous")
            expected_start = segment.iteration_end
        if expected_start != self.expected_iterations:
            raise ValueError("resume health segments do not cover expected iterations")
        return self


def recover_resume_health_summary(config: RLResumeHealthConfig) -> dict[str, Any]:
    """Recover cumulative diagnostics while preserving raw segment summaries."""
    base_path = _required_file(config.base_summary_path, "base resume summary")
    base_summary = _read_object(base_path)
    rows: list[dict[str, Any]] = []
    values: dict[str, list[float]] = {
        tag: [] for tag in (*_REQUIRED_TAGS, *_DIAGNOSTIC_TAGS)
    }
    segment_audit: list[dict[str, Any]] = []
    source_paths: list[Path] = [base_path]
    for segment in config.segments:
        event_path = _required_file(segment.event_path, "TensorBoard event segment")
        source_paths.append(event_path)
        segment_rows, segment_values, discarded = _read_segment(segment, event_path)
        rows.extend(segment_rows)
        for tag, tag_values in segment_values.items():
            values[tag].extend(tag_values)
        segment_audit.append(
            {
                "label": segment.label,
                "iteration_start": segment.iteration_start,
                "iteration_end": segment.iteration_end,
                "accepted_iterations": len(segment_rows),
                "discarded_iterations": discarded,
                "event_path": records.display_path(event_path),
                "event_sha256": file_sha256(event_path),
            }
        )
    rows.sort(key=lambda row: int(row["iteration"]))
    observed_iterations = [int(row["iteration"]) for row in rows]
    if observed_iterations != list(range(config.expected_iterations)):
        raise ValueError("recovered TensorBoard rows do not exactly cover the run")

    distributed_paths = [
        _required_file(path, "distributed segment summary")
        for path in config.distributed_summary_paths
    ]
    source_paths.extend(distributed_paths)
    overflow_decisions = 0
    overflow_trajectories = 0
    transport_stale_decisions = 0
    transport_stale_trajectories = 0
    for path in distributed_paths:
        distributed = _distributed_mapping(_read_object(path))
        overflow_decisions += _integer(distributed, "dropped_overflow_decisions")
        overflow_trajectories += _integer(
            distributed,
            "dropped_overflow_trajectories",
        )
        transport_stale_decisions += _integer(
            distributed,
            "dropped_stale_decisions",
        )
        transport_stale_trajectories += _integer(
            distributed,
            "dropped_stale_trajectories",
        )

    updates = values["learner/updates"]
    kept = values["batch/kept_decisions"]
    data_passes = [
        update * config.effective_batch_size / kept_decisions
        for update, kept_decisions in zip(updates, kept, strict=True)
    ]
    early_stops = values["ppo/early_stopped"]
    health = {
        "iterations": config.expected_iterations,
        "effective_updates": int(round(sum(updates))),
        "early_stop_count": int(round(sum(early_stops))),
        "early_stop_fraction": sum(early_stops) / config.expected_iterations,
        "epoch_zero_early_stop_count": 0,
        "clip_fraction": _finite_distribution(values["ppo/clip_fraction"]),
        "grad_norm": _finite_distribution(values["ppo/grad_norm"]),
        "ratio_p95": _finite_distribution(values["ppo/ratio_p95"]),
        "stale_fraction": _finite_distribution(values["batch/stale_fraction"]),
        "data_passes": _finite_distribution(data_passes),
        "fixed_reference_initial_kl_k3": _finite_distribution(
            values["ppo/kl_stop_baseline_k3"]
        ),
        "overflow_decisions": overflow_decisions,
        "overflow_trajectories": overflow_trajectories,
        "transport_stale_decisions": transport_stale_decisions,
        "transport_stale_trajectories": transport_stale_trajectories,
        "non_finite_metric_count": sum(
            _non_finite_count(tag_values) for tag_values in values.values()
        ),
    }
    provenance_paths = [
        _required_file(path, "resume recovery provenance")
        for path in config.provenance_paths
    ]
    source_paths.extend(provenance_paths)
    recovered = dict(base_summary)
    recovered["iterations"] = rows
    recovered["health"] = health
    recovered["recovery"] = {
        "protocol": "RL-RESUME-HEALTH-RECOVERY-v1",
        "experiment_id": config.experiment_id,
        "health_only": True,
        "segments": segment_audit,
        "base_summary_path": records.display_path(base_path),
        "source_sha256": {
            records.display_path(path): file_sha256(path) for path in source_paths
        },
    }
    output_path = records.repo_path(config.output_summary_path)
    write_identity_atomic(output_path, recovered)
    audit = {
        **recovered["recovery"],
        "expected_iterations": config.expected_iterations,
        "health": health,
        "output_summary_path": records.display_path(output_path),
        "output_summary_sha256": file_sha256(output_path),
    }
    write_identity_atomic(records.repo_path(config.audit_output_path), audit)
    return audit


def _read_segment(
    segment: RLHealthEventSegment,
    event_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[float]], list[int]]:
    from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore[import-untyped]
        EventAccumulator,
    )

    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", ()))
    required = {"learner/window_iteration", *_REQUIRED_TAGS}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"TensorBoard segment is missing tags: {missing}")
    iteration_events = accumulator.Scalars("learner/window_iteration")
    accepted = {
        int(event.step): int(round(event.value))
        for event in iteration_events
        if segment.iteration_start
        <= int(round(event.value))
        < segment.iteration_end
    }
    discarded = sorted(
        int(round(event.value))
        for event in iteration_events
        if int(round(event.value)) not in accepted.values()
    )
    expected = set(range(segment.iteration_start, segment.iteration_end))
    if set(accepted.values()) != expected or len(accepted) != len(expected):
        raise ValueError(f"TensorBoard segment {segment.label!r} has iteration gaps")
    values: dict[str, list[float]] = {}
    by_tag_step: dict[str, dict[int, float]] = {}
    for tag in _REQUIRED_TAGS:
        by_step = {
            int(event.step): float(event.value)
            for event in accumulator.Scalars(tag)
            if int(event.step) in accepted
        }
        if set(by_step) != set(accepted):
            raise ValueError(f"TensorBoard tag {tag!r} is not iteration-aligned")
        by_tag_step[tag] = by_step
        values[tag] = [by_step[step] for step in sorted(accepted)]
    for tag in _DIAGNOSTIC_TAGS:
        values[tag] = (
            [
                float(event.value)
                for event in accumulator.Scalars(tag)
                if int(event.step) in accepted
            ]
            if tag in available
            else []
        )
    rows = [
        {
            "iteration": iteration,
            "published_version": step,
            "source_segment": segment.label,
            "updates": int(round(by_tag_step["learner/updates"][step])),
            "kept_decisions": int(
                round(by_tag_step["batch/kept_decisions"][step])
            ),
            "stale_fraction": by_tag_step["batch/stale_fraction"][step],
            "early_stopped": bool(by_tag_step["ppo/early_stopped"][step]),
            "reference_initial_kl_k3": by_tag_step[
                "ppo/kl_stop_baseline_k3"
            ][step],
        }
        for step, iteration in sorted(accepted.items(), key=lambda item: item[1])
    ]
    return rows, values, discarded


def _finite_distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    tensor = torch.tensor(finite, dtype=torch.float64)
    return {
        "count": len(finite),
        "input_count": len(values),
        "min": float(tensor.min().item()) if finite else None,
        "max": float(tensor.max().item()) if finite else None,
        "mean": float(tensor.mean().item()) if finite else None,
        "p50": float(torch.quantile(tensor, 0.50).item()) if finite else None,
        "p95": float(torch.quantile(tensor, 0.95).item()) if finite else None,
        "non_finite_count": len(values) - len(finite),
    }


def _non_finite_count(values: Sequence[float]) -> int:
    return sum(not math.isfinite(float(value)) for value in values)


def _distributed_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        return summary
    distributed = payload.get("distributed")
    return distributed if isinstance(distributed, Mapping) else payload


def _integer(mapping: Mapping[str, Any], field: str) -> int:
    value = mapping.get(field, 0)
    return int(value) if isinstance(value, int | float) else 0


def _required_file(path: Path, label: str) -> Path:
    resolved = records.repo_path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _read_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw


__all__ = [
    "RLHealthEventSegment",
    "RLResumeHealthConfig",
    "recover_resume_health_summary",
]
