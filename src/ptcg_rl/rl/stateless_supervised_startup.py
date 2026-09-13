"""Source-bound full-model supervised initialization for stateless PPO."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor

from ptcg_rl.model.simple_stateless import SimpleStatelessModelConfig
from ptcg_rl.training.simple_stateless_pretrain_artifact import (
    SUPERVISED_POLICY_ARTIFACT_SCHEMA,
    TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA,
    WARMSTART_TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA,
    SupervisedPolicyArtifactManifest,
    SupervisedPolicyFinalEpochSelectionRecord,
    SupervisedPolicySelectionRecord,
    SupervisedPolicyTrainMonitorSelectionRecord,
    load_supervised_policy_artifact,
)
from ptcg_rl.training.simple_stateless_pretrain_data import file_sha256


class StatelessSupervisedStartupDeclaration(BaseModel):
    """Immutable identities authorized for one fresh full-model PPO lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_sha256: str
    policy_sha256: str
    model_state_fingerprint: str
    dataset_manifest_sha256: str
    dataset_fingerprint: str
    trainable_scope: Literal["full_model"] = "full_model"
    event_contract_fingerprint: str | None = None
    sequence_contract_fingerprint: str | None = None

    @field_validator(
        "manifest_sha256",
        "policy_sha256",
        "model_state_fingerprint",
        "dataset_manifest_sha256",
        "dataset_fingerprint",
        "event_contract_fingerprint",
        "sequence_contract_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str | None) -> str | None:
        """Require immutable full content identities."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("supervised startup identity must be SHA-256")
        return normalized


class StatelessSupervisedStartupAuditReport(BaseModel):
    """Compact evidence copied into the new PPO run before pair publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[
        "simple-stateless-supervised-startup-audit-v1",
        "simple-stateless-supervised-startup-audit-v2",
    ] = "simple-stateless-supervised-startup-audit-v1"
    declaration: StatelessSupervisedStartupDeclaration
    manifest_path: Path
    policy_path: Path
    policy_size_bytes: int = Field(gt=0)
    model_config_fingerprint: str
    exact_registry_fingerprint: str
    public_catalog_fingerprint: str
    input_contract_fingerprint: str
    selection_basis: Literal[
        "validation",
        "train_monitor",
        "final_epoch",
    ] = "validation"
    selection_fingerprint: str
    selection_optimizer_step: int = Field(ge=0)
    initialization_source_pair_manifest_sha256: str | None = None
    initialization_source_policy_sha256: str | None = None
    initialization_random_seed: int | None = Field(default=None, ge=0)
    initialization_model_state_fingerprint: str | None = None
    event_contract_fingerprint: str | None = None
    sequence_contract_fingerprint: str | None = None

    @field_validator("manifest_path", "policy_path")
    @classmethod
    def absolute_artifact_path(cls, value: Path) -> Path:
        """Keep the evidence independent of the process working directory."""
        if not value.is_absolute():
            raise ValueError("supervised startup artifact paths must be absolute")
        return value

    @field_validator(
        "model_config_fingerprint",
        "exact_registry_fingerprint",
        "public_catalog_fingerprint",
        "input_contract_fingerprint",
        "selection_fingerprint",
        "initialization_source_pair_manifest_sha256",
        "initialization_source_policy_sha256",
        "initialization_model_state_fingerprint",
        "event_contract_fingerprint",
        "sequence_contract_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str | None) -> str | None:
        """Require complete audit identities."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("supervised startup audit identity must be SHA-256")
        return normalized


@dataclass(frozen=True)
class StatelessSupervisedStartupPlan:
    """Validated model state and its durable startup audit."""

    manifest: SupervisedPolicyArtifactManifest
    model_state: Mapping[str, Tensor]
    audit: StatelessSupervisedStartupAuditReport


def prepare_stateless_supervised_startup(
    manifest_path: Path,
    *,
    declaration: StatelessSupervisedStartupDeclaration,
    expected_model_config: SimpleStatelessModelConfig,
    expected_exact_registry_fingerprint: str,
    expected_public_catalog_fingerprint: str,
    expected_input_contract_fingerprint: str,
    expected_event_contract_fingerprint: str | None = None,
    expected_sequence_contract_fingerprint: str | None = None,
) -> StatelessSupervisedStartupPlan:
    """Verify an explicitly selected full-model artifact against all bindings."""
    resolved_manifest = manifest_path.resolve()
    if (
        not resolved_manifest.is_file()
        or file_sha256(resolved_manifest) != declaration.manifest_sha256
    ):
        raise ValueError("supervised startup manifest identity mismatch")
    manifest, state = load_supervised_policy_artifact(
        resolved_manifest,
        expected_model_config=expected_model_config,
        expected_exact_registry_fingerprint=(
            expected_exact_registry_fingerprint
        ),
        expected_public_catalog_fingerprint=(
            expected_public_catalog_fingerprint
        ),
        expected_input_contract_fingerprint=(
            expected_input_contract_fingerprint
        ),
        expected_event_contract_fingerprint=(
            expected_event_contract_fingerprint
        ),
        expected_sequence_contract_fingerprint=(
            expected_sequence_contract_fingerprint
        ),
    )
    _require_declared_manifest(manifest, declaration=declaration)
    scope = manifest.trainable_scope
    selection = manifest.selection
    initialization = manifest.initialization
    if scope is None or scope.mode != declaration.trainable_scope:
        raise ValueError("supervised startup requires a full-model artifact")
    audit_format: Literal[
        "simple-stateless-supervised-startup-audit-v1",
        "simple-stateless-supervised-startup-audit-v2",
    ]
    selection_basis: Literal["validation", "train_monitor", "final_epoch"]
    if manifest.format == SUPERVISED_POLICY_ARTIFACT_SCHEMA:
        if (
            not isinstance(selection, SupervisedPolicySelectionRecord)
            or selection.selected_model_state_fingerprint
            != manifest.model_state_fingerprint
            or initialization.mode != "rl_pair"
            or initialization.source_pair_manifest_sha256 is None
            or initialization.source_policy_sha256 is None
        ):
            raise ValueError(
                "supervised startup requires a pair-initialized, "
                "validation-selected full-model artifact"
            )
        audit_format = "simple-stateless-supervised-startup-audit-v1"
        selection_basis = "validation"
        selection_fingerprint = selection.split_assignment_fingerprint
    elif manifest.format == TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA:
        if (
            initialization.mode != "random"
            or initialization.random_seed is None
            or initialization.initial_model_state_fingerprint is None
            or manifest.event_contract_fingerprint
            != declaration.event_contract_fingerprint
            or manifest.sequence_contract_fingerprint
            != declaration.sequence_contract_fingerprint
        ):
            raise ValueError(
                "temporal supervised startup requires a random-initialized, "
                "contract-bound full-model artifact"
            )
        audit_format = "simple-stateless-supervised-startup-audit-v2"
        if isinstance(selection, SupervisedPolicyTrainMonitorSelectionRecord):
            if (
                selection.selected_model_state_fingerprint
                != manifest.model_state_fingerprint
            ):
                raise ValueError(
                    "temporal train-monitor selection differs from artifact"
                )
            selection_basis = "train_monitor"
            selection_fingerprint = selection.monitor_fingerprint
        elif isinstance(selection, SupervisedPolicySelectionRecord):
            if (
                selection.selected_model_state_fingerprint
                != manifest.model_state_fingerprint
                or selection.selected_checkpoint_sha256 is None
                or selection.selected_checkpoint_size_bytes is None
            ):
                raise ValueError(
                    "temporal validation selection is missing its immutable binding"
                )
            selection_basis = "validation"
            selection_fingerprint = selection.split_assignment_fingerprint
        else:
            raise ValueError(
                "temporal supervised startup requires validation or train-monitor "
                "checkpoint selection"
            )
    elif manifest.format == WARMSTART_TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA:
        if (
            initialization.mode != "rl_pair"
            or initialization.source_pair_manifest_sha256 is None
            or initialization.source_policy_sha256 is None
            or manifest.event_contract_fingerprint
            != declaration.event_contract_fingerprint
            or manifest.sequence_contract_fingerprint
            != declaration.sequence_contract_fingerprint
        ):
            raise ValueError(
                "warm-start temporal supervised startup requires a "
                "pair-initialized, contract-bound full-model artifact"
            )
        audit_format = "simple-stateless-supervised-startup-audit-v2"
        if isinstance(selection, SupervisedPolicySelectionRecord):
            if (
                selection.selected_model_state_fingerprint
                != manifest.model_state_fingerprint
                or selection.selected_checkpoint_sha256 is None
                or selection.selected_checkpoint_size_bytes is None
            ):
                raise ValueError(
                    "warm-start temporal validation selection is missing its "
                    "immutable binding"
                )
            selection_basis = "validation"
            selection_fingerprint = selection.split_assignment_fingerprint
        elif isinstance(selection, SupervisedPolicyFinalEpochSelectionRecord):
            if (
                selection.selected_model_state_fingerprint
                != manifest.model_state_fingerprint
                or selection.selected_optimizer_step
                != selection.final_optimizer_steps
            ):
                raise ValueError(
                    "warm-start temporal final-epoch selection differs from "
                    "the published model"
                )
            selection_basis = "final_epoch"
            selection_fingerprint = selection.selected_model_state_fingerprint
        else:
            raise ValueError(
                "warm-start temporal supervised startup requires validation "
                "or an explicit final-epoch selection"
            )
    else:
        raise ValueError("unsupported supervised startup artifact schema")
    policy_path = (resolved_manifest.parent / manifest.policy_filename).resolve()
    audit = StatelessSupervisedStartupAuditReport(
        format=audit_format,
        declaration=declaration,
        manifest_path=resolved_manifest,
        policy_path=policy_path,
        policy_size_bytes=manifest.policy_size_bytes,
        model_config_fingerprint=manifest.model_config_fingerprint,
        exact_registry_fingerprint=manifest.exact_registry_fingerprint,
        public_catalog_fingerprint=manifest.public_catalog_fingerprint,
        input_contract_fingerprint=manifest.input_contract_fingerprint,
        selection_basis=selection_basis,
        selection_fingerprint=selection_fingerprint,
        selection_optimizer_step=selection.selected_optimizer_step,
        initialization_source_pair_manifest_sha256=(
            initialization.source_pair_manifest_sha256
        ),
        initialization_source_policy_sha256=(
            initialization.source_policy_sha256
        ),
        initialization_random_seed=getattr(initialization, "random_seed", None),
        initialization_model_state_fingerprint=(
            getattr(initialization, "initial_model_state_fingerprint", None)
        ),
        event_contract_fingerprint=getattr(
            manifest,
            "event_contract_fingerprint",
            None,
        ),
        sequence_contract_fingerprint=getattr(
            manifest,
            "sequence_contract_fingerprint",
            None,
        ),
    )
    return StatelessSupervisedStartupPlan(
        manifest=manifest,
        model_state=state,
        audit=audit,
    )


def _require_declared_manifest(
    manifest: SupervisedPolicyArtifactManifest,
    *,
    declaration: StatelessSupervisedStartupDeclaration,
) -> None:
    """Reject a self-consistent artifact other than the selected one."""
    actual = (
        manifest.policy_sha256,
        manifest.model_state_fingerprint,
        manifest.dataset_manifest_sha256,
        manifest.dataset_fingerprint,
    )
    expected = (
        declaration.policy_sha256,
        declaration.model_state_fingerprint,
        declaration.dataset_manifest_sha256,
        declaration.dataset_fingerprint,
    )
    if actual != expected:
        raise ValueError("supervised startup declaration differs from artifact")
    if manifest.format in {
        TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA,
        WARMSTART_TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA,
    } and (
        manifest.event_contract_fingerprint
        != declaration.event_contract_fingerprint
        or manifest.sequence_contract_fingerprint
        != declaration.sequence_contract_fingerprint
    ):
        raise ValueError("supervised startup temporal declaration differs from artifact")


__all__ = [
    "StatelessSupervisedStartupAuditReport",
    "StatelessSupervisedStartupDeclaration",
    "StatelessSupervisedStartupPlan",
    "prepare_stateless_supervised_startup",
]
