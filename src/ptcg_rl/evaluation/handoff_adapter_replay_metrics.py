"""Artifact publication and paired metrics for packaged handoff replay."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.evaluation.handoff_adapter_replay_config import (
    HandoffAdapterActTimeReplayConfig,
)


def publish_handoff_replay_artifacts(
    *,
    config: HandoffAdapterActTimeReplayConfig,
    package_identity: Mapping[str, Any],
    run_results: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Publish compact callback/run evidence and return the summary."""
    callback_rows = [
        dict(row)
        for result in run_results
        for row in _sequence_of_mappings(result.get("callbacks"))
    ]
    run_rows = [_run_row(result) for result in run_results]
    _write_parquet_atomic(
        output_dir / "callbacks.parquet",
        callback_rows,
        compression=config.compression,
    )
    _write_parquet_atomic(
        output_dir / "runs.parquet",
        run_rows,
        compression=config.compression,
    )
    resolved = config.model_dump(mode="json")
    _write_json_atomic(output_dir / "resolved_config.json", resolved)
    summary = _summary(config, package_identity, run_rows, callback_rows)
    _write_json_atomic(output_dir / "summary.json", summary)
    artifacts = {
        name: _file_sha256(output_dir / name)
        for name in (
            "callbacks.parquet",
            "runs.parquet",
            "resolved_config.json",
            "summary.json",
        )
    }
    _write_json_atomic(
        output_dir / "manifest.json",
        {
            "protocol": "HANDOFF-ADAPTER-PACKAGED-ACTTIME-v1",
            "experiment_id": config.experiment_id,
            "artifacts": artifacts,
        },
    )
    return summary


def _run_row(result: Mapping[str, Any]) -> dict[str, Any]:
    callbacks = _sequence_of_mappings(result.get("callbacks"))
    elapsed = [float(row["elapsed_seconds"]) for row in callbacks]
    return {
        "arm_id": str(result["arm_id"]),
        "handoff_score_mode": str(result["handoff_score_mode"]),
        "repetition": int(result["repetition"]),
        "replay_path": str(result["replay_path"]),
        "replay_sha256": str(result["replay_sha256"]),
        "seat": int(result["seat"]),
        "decisions": int(result["decisions"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "remaining_seconds": float(result["remaining_seconds"]),
        "exhausted": bool(result["exhausted"]),
        "legal_callbacks": sum(bool(row["legal"]) for row in callbacks),
        "policy_error_callbacks": sum(
            row.get("policy_error") is not None for row in callbacks
        ),
        "search_error_callbacks": sum(
            row.get("search_error") is not None for row in callbacks
        ),
        "completed_search_callbacks": sum(
            row.get("search_stop_reason") == "complete" for row in callbacks
        ),
        "same_seat_value_rows": sum(
            int(row.get("same_seat_value_rows", 0)) for row in callbacks
        ),
        "handoff_value_rows": sum(
            int(row.get("handoff_value_rows", 0)) for row in callbacks
        ),
        "handoff_value_callbacks": sum(
            int(row.get("handoff_value_rows", 0)) > 0 for row in callbacks
        ),
        "startup_seconds": elapsed[0] if elapsed else 0.0,
        "non_startup_seconds": sum(elapsed[1:]),
        "peak_rss_bytes": int(result["peak_rss_bytes"]),
        "torch_num_threads": int(result["torch_num_threads"]),
    }


def _summary(
    config: HandoffAdapterActTimeReplayConfig,
    package_identity: Mapping[str, Any],
    run_rows: Sequence[Mapping[str, Any]],
    callback_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in config.arms:
        selected_runs = [row for row in run_rows if row["arm_id"] == arm.arm_id]
        selected_callbacks = [
            row for row in callback_rows if row["arm_id"] == arm.arm_id
        ]
        arms[arm.arm_id] = {
            "handoff_score_mode": arm.handoff_score_mode,
            "runs": len(selected_runs),
            "callbacks": len(selected_callbacks),
            "episode_elapsed_seconds": _distribution(
                [float(row["elapsed_seconds"]) for row in selected_runs]
            ),
            "callback_elapsed_seconds": _distribution(
                [float(row["elapsed_seconds"]) for row in selected_callbacks]
            ),
            "non_startup_callback_elapsed_seconds": _distribution(
                [
                    float(row["elapsed_seconds"])
                    for row in selected_callbacks
                    if int(row["callback_index"]) > 0
                ]
            ),
            "minimum_remaining_seconds": min(
                float(row["remaining_seconds"]) for row in selected_runs
            ),
            "exhausted_runs": sum(bool(row["exhausted"]) for row in selected_runs),
            "handoff_value_rows": sum(
                int(row["handoff_value_rows"]) for row in selected_runs
            ),
            "handoff_value_callbacks": sum(
                int(row["handoff_value_callbacks"]) for row in selected_runs
            ),
            "completed_search_callbacks": sum(
                int(row["completed_search_callbacks"]) for row in selected_runs
            ),
            "search_error_callbacks": sum(
                int(row["search_error_callbacks"]) for row in selected_runs
            ),
            "policy_error_callbacks": sum(
                int(row["policy_error_callbacks"]) for row in selected_runs
            ),
            "illegal_callbacks": sum(
                int(row["decisions"]) - int(row["legal_callbacks"])
                for row in selected_runs
            ),
            "peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in selected_runs),
        }
    adapter_arm = next(
        arm.arm_id
        for arm in config.arms
        if arm.handoff_score_mode == "root_value_adapter"
    )
    control_arm = next(
        arm.arm_id for arm in config.arms if arm.handoff_score_mode == "engine_only"
    )
    episode_deltas = _paired_deltas(
        run_rows,
        adapter_arm=adapter_arm,
        control_arm=control_arm,
        value_key="elapsed_seconds",
        identity_keys=("repetition", "replay_path", "seat"),
    )
    callback_deltas = _paired_deltas(
        callback_rows,
        adapter_arm=adapter_arm,
        control_arm=control_arm,
        value_key="elapsed_seconds",
        identity_keys=("repetition", "replay_path", "seat", "callback_index"),
    )
    paired_action_differences = _paired_mismatches(
        callback_rows,
        adapter_arm=adapter_arm,
        control_arm=control_arm,
        value_key="action_json",
        identity_keys=("repetition", "replay_path", "seat", "callback_index"),
    )
    valid = all(
        int(row["policy_error_callbacks"]) == 0
        and int(row["legal_callbacks"]) == int(row["decisions"])
        for row in run_rows
    )
    adapter_exercised = int(arms[adapter_arm]["handoff_value_rows"]) > 0
    ledger_preserved = int(arms[adapter_arm]["exhausted_runs"]) == 0
    adapter_error_free = int(arms[adapter_arm]["search_error_callbacks"]) == 0
    control_mean = float(arms[control_arm]["episode_elapsed_seconds"]["mean"])
    mean_episode_delta = float(np.mean(episode_deltas))
    return {
        "protocol": "HANDOFF-ADAPTER-PACKAGED-ACTTIME-v1",
        "experiment_id": config.experiment_id,
        "environment": {
            "device": "cpu",
            "reason": "packaged Kaggle ActTime deployment-parity measurement",
        },
        "package": dict(package_identity),
        "repetitions": config.repetitions,
        "initial_overage_seconds": config.initial_overage_seconds,
        "arms": arms,
        "adapter_minus_control_episode_seconds": _distribution(episode_deltas),
        "adapter_minus_control_episode_fraction": (
            mean_episode_delta / control_mean if control_mean > 0.0 else 0.0
        ),
        "adapter_minus_control_callback_seconds": _distribution(callback_deltas),
        "paired_action_differences": paired_action_differences,
        "valid_runtime_evidence": valid,
        "adapter_handoff_path_exercised": adapter_exercised,
        "adapter_search_path_error_free": adapter_error_free,
        "adapter_episode_ledger_preserved": ledger_preserved,
        "observed_direct_integration_feasible": (
            valid and adapter_exercised and adapter_error_free and ledger_preserved
        ),
        "interpretation": (
            "ActTime feasibility only; the warm-start adapter is zero-residual and "
            "this replay does not establish policy quality."
        ),
    }


def _paired_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    adapter_arm: str,
    control_arm: str,
    value_key: str,
    identity_keys: tuple[str, ...],
) -> list[float]:
    indexed = {
        (str(row["arm_id"]), *(row[key] for key in identity_keys)): float(
            row[value_key]
        )
        for row in rows
    }
    identities = {
        tuple(row[key] for key in identity_keys)
        for row in rows
        if row["arm_id"] == adapter_arm
    }
    return [
        indexed[(adapter_arm, *identity)] - indexed[(control_arm, *identity)]
        for identity in sorted(identities, key=repr)
    ]


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _paired_mismatches(
    rows: Sequence[Mapping[str, Any]],
    *,
    adapter_arm: str,
    control_arm: str,
    value_key: str,
    identity_keys: tuple[str, ...],
) -> int:
    indexed = {
        (str(row["arm_id"]), *(row[key] for key in identity_keys)): row[value_key]
        for row in rows
    }
    identities = {
        tuple(row[key] for key in identity_keys)
        for row in rows
        if row["arm_id"] == adapter_arm
    }
    return sum(
        indexed[(adapter_arm, *identity)] != indexed[(control_arm, *identity)]
        for identity in identities
    )


def _write_parquet_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    compression: str,
) -> None:
    if not rows:
        raise ValueError("cannot publish an empty replay table")
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, temporary, compression=compression)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_of_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("replay callback rows must be a sequence")
    if not all(isinstance(item, Mapping) for item in value):
        raise TypeError("replay callback rows must be mappings")
    return tuple(value)


__all__ = ["publish_handoff_replay_artifacts"]
