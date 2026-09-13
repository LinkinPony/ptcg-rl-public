"""Orchestrate an auditable weights-only policy migration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.model.network import AgentNetworkConfig, build_agent_policy_value_net
from ptcg_rl.model.weights_only_migration_classification import (
    canonical_json_sha256,
    checkpoint_model_config_payload,
    checkpoint_publish_version,
    checkpoint_state_dict,
    classification_summary,
    classify_tensors,
    comparison_contract_valid,
    stream_file_fingerprint,
    write_manifest,
)
from ptcg_rl.model.weights_only_migration_parity import run_functional_parity

_MANIFEST_SCHEMA = "weights-only-migration-audit-v1"
_SHA256_HEX_LENGTH = 64


class WeightsOnlyMigrationAuditConfig(BaseModel):
    """Immutable inputs for one checkpoint migration audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_path: Path
    expected_checkpoint_sha256: str
    output_path: Path
    initialization_seed: int = Field(default=0, ge=0)

    @field_validator("expected_checkpoint_sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require a lowercase immutable checkpoint identity."""
        normalized = value.strip().lower()
        if len(normalized) != _SHA256_HEX_LENGTH:
            raise ValueError("expected checkpoint SHA256 must contain 64 hex digits")
        try:
            digest = bytes.fromhex(normalized)
        except ValueError as exc:
            raise ValueError(
                "expected checkpoint SHA256 must contain 64 hex digits"
            ) from exc
        if len(digest) != 32:
            raise ValueError("expected checkpoint SHA256 must contain 64 hex digits")
        return normalized


def run_weights_only_migration_audit(
    config: WeightsOnlyMigrationAuditConfig,
) -> dict[str, Any]:
    """Audit raw tensor migration, cold-start outputs, and anchor isolation."""
    checkpoint_path = config.checkpoint_path
    checkpoint_size, checkpoint_sha256 = stream_file_fingerprint(checkpoint_path)
    if checkpoint_sha256 != config.expected_checkpoint_sha256:
        raise ValueError(
            "checkpoint SHA256 mismatch: "
            f"{checkpoint_sha256} != {config.expected_checkpoint_sha256}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    source_state = checkpoint_state_dict(checkpoint)
    raw_model_config = checkpoint_model_config_payload(checkpoint)
    target_config = AgentNetworkConfig.model_validate(raw_model_config)

    torch.manual_seed(config.initialization_seed)
    migrated_model = build_agent_policy_value_net(target_config).cpu().eval()
    classifications = classify_tensors(
        source_state=source_state,
        target_state=migrated_model.state_dict(),
    )
    comparison_valid = comparison_contract_valid(classifications)

    post_load: dict[str, list[str]] = {
        "migrated_missing": [],
        "migrated_unexpected": [],
        "anchor_missing": [],
        "anchor_unexpected": [],
    }
    parity: dict[str, Any]
    if comparison_valid:
        migrated_incompatible = migrated_model.load_state_dict(
            dict(source_state),
            strict=False,
        )
        torch.manual_seed(config.initialization_seed)
        anchor_model = build_agent_policy_value_net(target_config).cpu().eval()
        anchor_incompatible = anchor_model.load_state_dict(
            dict(source_state),
            strict=False,
        )
        post_load = {
            "migrated_missing": sorted(migrated_incompatible.missing_keys),
            "migrated_unexpected": sorted(migrated_incompatible.unexpected_keys),
            "anchor_missing": sorted(anchor_incompatible.missing_keys),
            "anchor_unexpected": sorted(anchor_incompatible.unexpected_keys),
        }
        if any(post_load.values()):
            parity = {
                "status": "not_run",
                "reason": "post_load_incompatible",
                "valid": False,
            }
        else:
            parity = run_functional_parity(
                migrated_model=migrated_model,
                anchor_model=anchor_model,
                config=target_config,
            )
    else:
        parity = {
            "status": "not_run",
            "reason": "raw_tensor_contract_invalid",
            "valid": False,
        }

    manifest: dict[str, Any] = {
        "schema": _MANIFEST_SCHEMA,
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_size,
            "sha256": checkpoint_sha256,
            "publish_version": checkpoint_publish_version(checkpoint),
        },
        "initialization_seed": config.initialization_seed,
        "model_config": {
            "raw_source_sha256": canonical_json_sha256(raw_model_config),
            "normalized_target_sha256": canonical_json_sha256(
                target_config.model_dump(mode="json")
            ),
            "target": target_config.model_dump(mode="json"),
        },
        "summary": classification_summary(classifications),
        "tensors": classifications,
        "raw_comparison_valid": comparison_valid,
        "post_load": post_load,
        "parity": parity,
    }
    manifest["valid"] = bool(
        comparison_valid and not any(post_load.values()) and parity.get("valid")
    )
    write_manifest(config.output_path, manifest)
    return manifest


__all__ = [
    "WeightsOnlyMigrationAuditConfig",
    "run_weights_only_migration_audit",
    "stream_file_fingerprint",
]
