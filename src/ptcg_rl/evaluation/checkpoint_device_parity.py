"""CPU-to-GPU action parity for accelerated checkpoint evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ConfigDict

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.checkpoint_asset_audit import (
    audit_checkpoint_pair,
    sample_replay_roots,
    write_parity_rows,
)
from ptcg_rl.evaluation.checkpoint_asset_models import CheckpointAssetAuditConfig
from ptcg_rl.evaluation.search_identity import (
    environment_identity,
    file_sha256,
    write_identity_atomic,
)


class CheckpointDeviceParityConfig(CheckpointAssetAuditConfig):
    """Replay sample plus frozen asset and output paths for device parity."""

    model_config = ConfigDict(extra="forbid")

    asset_manifest_path: Path
    reference_device: str = "cpu"
    accelerated_device: str = "cuda"
    device_parity_manifest_path: Path
    device_parity_parquet_path: Path


def run_checkpoint_device_parity(
    config: CheckpointDeviceParityConfig,
) -> dict[str, Any]:
    """Require CPU and accelerated inference to choose the same non-tied actions."""
    output_manifest = records.repo_path(config.device_parity_manifest_path)
    output_parquet = records.repo_path(config.device_parity_parquet_path)
    _require_absent(output_manifest)
    _require_absent(output_parquet)
    asset_manifest = _read_object(config.asset_manifest_path)
    raw_checkpoints = asset_manifest.get("checkpoints")
    if not isinstance(raw_checkpoints, Mapping):
        raise ValueError("asset manifest must contain checkpoints")
    roots, replay_manifest = sample_replay_roots(config)
    rows: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    for checkpoint in config.checkpoints:
        record = raw_checkpoints.get(checkpoint.checkpoint_tag)
        if not isinstance(record, Mapping):
            raise ValueError(
                f"asset manifest has no checkpoint {checkpoint.checkpoint_tag}"
            )
        asset_value = record.get("asset_path")
        expected_sha = record.get("asset_sha256")
        if not isinstance(asset_value, str) or not isinstance(expected_sha, str):
            raise ValueError("asset manifest checkpoint record is incomplete")
        asset_path = records.repo_path(Path(asset_value))
        if file_sha256(asset_path) != expected_sha:
            raise ValueError(
                f"device parity asset hash mismatch: {checkpoint.checkpoint_tag}"
            )
        summary, checkpoint_rows = audit_checkpoint_pair(
            checkpoint.checkpoint_tag,
            raw_path=asset_path,
            asset_path=asset_path,
            roots=roots,
            config=config,
            raw_device=config.reference_device,
            asset_device=config.accelerated_device,
        )
        rows.extend(checkpoint_rows)
        checkpoints[checkpoint.checkpoint_tag] = {
            "asset_path": records.display_path(asset_path),
            "asset_sha256": expected_sha,
            "parity": summary,
        }
    write_parity_rows(output_parquet, rows)
    result = {
        "protocol": "CHECKPOINT-SELECTION-v1-DEVICE-PARITY",
        "experiment_id": config.experiment_id,
        "reference_device": config.reference_device,
        "accelerated_device": config.accelerated_device,
        "environment": environment_identity(),
        "replay_sample": replay_manifest,
        "checkpoints": checkpoints,
        "parity_rows_path": records.display_path(output_parquet),
        "parity_rows_sha256": file_sha256(output_parquet),
        "passed": all(bool(item["parity"]["passed"]) for item in checkpoints.values()),
    }
    write_identity_atomic(output_manifest, result)
    return result


def _read_object(path: Path) -> dict[str, Any]:
    with records.repo_path(path).open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError("asset manifest must be a JSON object")
    return value


def _require_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"immutable device parity artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


__all__ = ["CheckpointDeviceParityConfig", "run_checkpoint_device_parity"]
