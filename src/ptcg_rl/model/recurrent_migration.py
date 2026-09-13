"""Auditable stateless-policy to recurrent PPO-only checkpoint migration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor

from ptcg_rl.model.network import AgentNetworkConfig, build_agent_policy_value_net
from ptcg_rl.model.policy import LEGACY_POINTER_POLICY_MISSING_KEYS
from ptcg_rl.model.state_encoder import LEGACY_STATE_ENCODER_MISSING_KEYS
from ptcg_rl.model.weights_only_migration_classification import (
    canonical_json_sha256,
    checkpoint_model_config_payload,
    checkpoint_publish_version,
    checkpoint_state_dict,
    stream_file_fingerprint,
    tensor_descriptor,
    write_manifest,
)
from ptcg_rl.rl.learner import WeightPublisher, WeightPublisherConfig
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint

RECURRENT_MIGRATION_SCHEMA = "stateless-to-recurrent-ppo-checkpoint-v1"
RECURRENT_MIGRATION_KIND = "dccr-v4-stateless-to-recurrent-ppo-v1"
_PAIR_FORMAT = "exact_policy_learner_pair_v1"
_ACTION_VALUE_PREFIX = "action_value_head."
_RECURRENT_PREFIX = "recurrent_policy."
_ZERO_COMPATIBILITY_PARAMETERS = (
    LEGACY_STATE_ENCODER_MISSING_KEYS | LEGACY_POINTER_POLICY_MISSING_KEYS
)
_SHA256_LENGTH = 64


class RecurrentPolicyMigrationConfig(BaseModel):
    """Immutable source pair and destination for one recurrent migration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_policy_path: Path
    source_policy_sha256: str
    source_training_state_path: Path
    source_training_state_sha256: str
    source_pair_manifest_path: Path
    output_dir: Path

    @field_validator("source_policy_sha256", "source_training_state_sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require canonical immutable source identities."""
        normalized = value.strip().lower()
        if len(normalized) != _SHA256_LENGTH:
            raise ValueError("source SHA256 must contain 64 lowercase hex digits")
        try:
            digest = bytes.fromhex(normalized)
        except ValueError as exc:
            raise ValueError(
                "source SHA256 must contain 64 lowercase hex digits"
            ) from exc
        if len(digest) != 32 or normalized != value:
            raise ValueError("source SHA256 must contain 64 lowercase hex digits")
        return normalized


def migrate_recurrent_policy_checkpoint(
    target_config: AgentNetworkConfig,
    migration: RecurrentPolicyMigrationConfig,
) -> dict[str, Any]:
    """Publish a complete recurrent policy and its source-bound audit report."""
    source_policy_size, source_policy_sha256 = stream_file_fingerprint(
        migration.source_policy_path
    )
    if source_policy_sha256 != migration.source_policy_sha256:
        raise ValueError("source policy SHA256 mismatch")
    source_state_size, source_state_sha256 = stream_file_fingerprint(
        migration.source_training_state_path
    )
    if source_state_sha256 != migration.source_training_state_sha256:
        raise ValueError("source training-state SHA256 mismatch")
    pair_size, pair_sha256 = stream_file_fingerprint(
        migration.source_pair_manifest_path
    )
    pair = _load_and_validate_source_pair(
        migration.source_pair_manifest_path,
        policy_size=source_policy_size,
        policy_sha256=source_policy_sha256,
        training_state_size=source_state_size,
        training_state_sha256=source_state_sha256,
    )

    checkpoint = torch.load(
        migration.source_policy_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    source_config = AgentNetworkConfig.model_validate(
        checkpoint_model_config_payload(checkpoint)
    )
    _validate_architecture_transition(source_config, target_config)
    source_state = checkpoint_state_dict(checkpoint)
    source_version = checkpoint_publish_version(checkpoint)
    if source_version is None or source_version != int(pair["version"]):
        raise ValueError("source checkpoint and pair versions differ")
    embedded_fingerprint = checkpoint.get("model_fingerprint")
    computed_source_fingerprint = canonical_model_state_fingerprint(source_state)
    if embedded_fingerprint != computed_source_fingerprint:
        raise ValueError("source checkpoint model fingerprint mismatch")
    pair_policy = cast(Mapping[str, Any], pair["policy"])
    if pair_policy.get("model_fingerprint") != computed_source_fingerprint:
        raise ValueError("source pair model fingerprint mismatch")

    output_dir = migration.output_dir
    weights_dir = output_dir / "weights"
    report_path = output_dir / "transition" / "migration_report.json"
    target_path = weights_dir / f"policy_v{source_version}.pt"
    for path in (target_path, weights_dir / "latest.json", report_path):
        if path.exists():
            raise FileExistsError(f"recurrent migration output already exists: {path}")

    target_model = build_agent_policy_value_net(target_config).cpu().eval()
    target_initial_state = target_model.state_dict()
    classifications = _classify_transition(
        source_state=source_state,
        target_state=target_initial_state,
    )
    _validate_tensor_transition(classifications)
    migrated_state = dict(target_initial_state)
    for name in classifications["copied"]:
        migrated_state[name] = source_state[name]
    target_model.load_state_dict(migrated_state, strict=True)
    target_fingerprint = canonical_model_state_fingerprint(target_model)

    source_registry = _registry_sha256(source_config)
    target_registry = _registry_sha256(target_config)
    transition_metadata = {
        "kind": RECURRENT_MIGRATION_KIND,
        "schema": RECURRENT_MIGRATION_SCHEMA,
        "source_policy_sha256": source_policy_sha256,
        "source_training_state_sha256": source_state_sha256,
        "source_pair_manifest_sha256": pair_sha256,
        "source_model_fingerprint": computed_source_fingerprint,
        "source_registry_sha256": source_registry,
        "target_registry_sha256": target_registry,
        "optimizer_state": "reset_new_branch",
        "scheduler_state": "reset_new_branch",
        "policy_iteration_state": "dropped_retired_lane",
    }
    publisher = WeightPublisher(
        weights_dir,
        WeightPublisherConfig(keep_last=1),
    )
    published = publisher.publish(
        target_model.state_dict(),
        version=source_version,
        metadata={"architecture_transition": transition_metadata},
        checkpoint_fields={
            "model_config": target_config.model_dump(mode="json"),
            "metadata": {
                "publish_version": source_version,
                "architecture_transition": transition_metadata,
            },
        },
    )
    if published.model_fingerprint != target_fingerprint:
        raise RuntimeError("published recurrent model fingerprint changed")
    target_size, target_sha256 = stream_file_fingerprint(published.path)

    tensor_records = _tensor_records(
        classifications,
        source_state=source_state,
        target_state=target_model.state_dict(),
    )
    report: dict[str, Any] = {
        "schema": RECURRENT_MIGRATION_SCHEMA,
        "transition_kind": RECURRENT_MIGRATION_KIND,
        "valid": True,
        "source_pair": {
            "policy_path": str(migration.source_policy_path),
            "policy_size_bytes": source_policy_size,
            "policy_sha256": source_policy_sha256,
            "training_state_path": str(migration.source_training_state_path),
            "training_state_size_bytes": source_state_size,
            "training_state_sha256": source_state_sha256,
            "pair_manifest_path": str(migration.source_pair_manifest_path),
            "pair_manifest_size_bytes": pair_size,
            "pair_manifest_sha256": pair_sha256,
            "policy_version": source_version,
            "model_fingerprint": computed_source_fingerprint,
        },
        "target_policy": {
            "path": str(published.path),
            "size_bytes": target_size,
            "sha256": target_sha256,
            "policy_version": source_version,
            "model_fingerprint": target_fingerprint,
        },
        "model_config": {
            "source_sha256": canonical_json_sha256(
                source_config.model_dump(mode="json")
            ),
            "target_sha256": canonical_json_sha256(
                target_config.model_dump(mode="json")
            ),
        },
        "registry": {
            "source_sha256": source_registry,
            "target_sha256": target_registry,
            "exact_strategy_lineage": "continued_unchanged_private_topology",
        },
        "state_migration": {
            "optimizer": "reset_new_branch",
            "scheduler": "reset_new_branch",
            "completed_iterations": "reset_zero_new_branch",
            "policy_iteration": "dropped_retired_lane",
            "reanalysis_replay": "dropped_retired_lane",
            "legacy_auxiliary_state": "dropped_retired_lane",
        },
        "summary": _classification_summary(tensor_records),
        "tensors": tensor_records,
    }
    report["manifest_sha256"] = canonical_json_sha256(report)
    write_manifest(report_path, report)
    return report


def _load_and_validate_source_pair(
    path: Path,
    *,
    policy_size: int,
    policy_sha256: str,
    training_state_size: int,
    training_state_sha256: str,
) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("source pair manifest must be an object")
    if value.get("format") != _PAIR_FORMAT or value.get("schema_version") != 1:
        raise ValueError("source pair manifest has an unsupported schema")
    if not isinstance(value.get("version"), int):
        raise ValueError("source pair manifest has no integer version")
    policy = value.get("policy")
    state = value.get("training_state")
    if not isinstance(policy, Mapping) or not isinstance(state, Mapping):
        raise ValueError("source pair manifest is missing policy or training state")
    expected_policy = (policy_size, policy_sha256)
    actual_policy = (policy.get("size_bytes"), policy.get("sha256"))
    if actual_policy != expected_policy:
        raise ValueError("source pair policy identity mismatch")
    expected_state = (training_state_size, training_state_sha256, policy_sha256)
    actual_state = (
        state.get("size_bytes"),
        state.get("sha256"),
        state.get("policy_sha256"),
    )
    if actual_state != expected_state:
        raise ValueError("source pair training-state identity mismatch")
    return value


def _validate_architecture_transition(
    source: AgentNetworkConfig,
    target: AgentNetworkConfig,
) -> None:
    if source.recurrent is not None:
        raise ValueError("recurrent migration source must be stateless")
    if target.recurrent is None:
        raise ValueError("recurrent migration target must enable recurrence")
    if target.action_value.enabled:
        raise ValueError("recurrent migration target must remove action-Q")
    source_common = source.model_copy(
        update={"recurrent": None, "action_value": target.action_value}
    )
    target_common = target.model_copy(update={"recurrent": None})
    if source_common != target_common:
        raise ValueError(
            "recurrent migration may only add recurrence and remove action-Q"
        )
    if _registry_sha256(source) != _registry_sha256(target):
        raise ValueError("recurrent migration cannot change the exact deck registry")


def _registry_sha256(config: AgentNetworkConfig) -> str | None:
    conditioning = config.deck_conditioning
    if conditioning is None or not conditioning.enabled:
        return None
    return conditioning.resolved_registry_sha256


Classification = Literal["copied", "new", "dropped", "dormant", "unexpected"]


def _classify_transition(
    *,
    source_state: Mapping[str, Tensor],
    target_state: Mapping[str, Tensor],
) -> dict[Classification, list[str]]:
    result: dict[Classification, list[str]] = {
        "copied": [],
        "new": [],
        "dropped": [],
        "dormant": [],
        "unexpected": [],
    }
    for name in sorted(set(source_state) | set(target_state)):
        source = source_state.get(name)
        target = target_state.get(name)
        if source is None:
            category: Classification = (
                "new"
                if name.startswith(_RECURRENT_PREFIX)
                or name in _ZERO_COMPATIBILITY_PARAMETERS
                else "unexpected"
            )
        elif target is None:
            category = (
                "dropped" if name.startswith(_ACTION_VALUE_PREFIX) else "unexpected"
            )
        elif source.shape == target.shape and source.dtype == target.dtype:
            category = "copied"
        else:
            category = "unexpected"
        result[category].append(name)
    return result


def _validate_tensor_transition(
    classifications: Mapping[Classification, list[str]],
) -> None:
    if classifications["unexpected"]:
        raise ValueError(
            "recurrent migration found incompatible tensors: "
            + ", ".join(classifications["unexpected"][:8])
        )
    if not any(
        name.startswith(_RECURRENT_PREFIX) for name in classifications["new"]
    ):
        raise ValueError("recurrent migration did not isolate new recurrent tensors")
    compatibility_names = {
        name
        for name in classifications["new"]
        if not name.startswith(_RECURRENT_PREFIX)
    }
    if not compatibility_names.issubset(_ZERO_COMPATIBILITY_PARAMETERS):
        raise ValueError("recurrent migration found an unknown compatibility tensor")
    if not all(
        name.startswith(_ACTION_VALUE_PREFIX) for name in classifications["dropped"]
    ):
        raise ValueError("recurrent migration dropped a non-action-Q tensor")


def _tensor_records(
    classifications: Mapping[Classification, list[str]],
    *,
    source_state: Mapping[str, Tensor],
    target_state: Mapping[str, Tensor],
) -> dict[Classification, list[dict[str, Any]]]:
    records: dict[Classification, list[dict[str, Any]]] = {
        "copied": [],
        "new": [],
        "dropped": [],
        "dormant": [],
        "unexpected": [],
    }
    for category, names in classifications.items():
        for name in names:
            tensor = (
                source_state[name] if category == "dropped" else target_state[name]
            )
            descriptor = tensor_descriptor(
                name,
                tensor,
                action=cast(Any, category),
            )
            if category == "new":
                descriptor["initialization"] = (
                    "zero"
                    if bool(torch.count_nonzero(tensor).item() == 0)
                    else "deterministic"
                )
            records[category].append(descriptor)
    return records


def _classification_summary(
    records: Mapping[Classification, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        category: {
            "tensors": len(items),
            "numel": sum(int(item["numel"]) for item in items),
        }
        for category, items in records.items()
    } | {"descriptor_sha256": canonical_json_sha256(records)}


__all__ = [
    "RECURRENT_MIGRATION_KIND",
    "RECURRENT_MIGRATION_SCHEMA",
    "RecurrentPolicyMigrationConfig",
    "migrate_recurrent_policy_checkpoint",
]
