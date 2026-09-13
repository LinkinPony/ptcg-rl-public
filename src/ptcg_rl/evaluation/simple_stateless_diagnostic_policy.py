"""Diagnostic-only export of a supervised full-model policy for routed evaluation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor, nn

from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload
from ptcg_rl.rl.stateless_checkpoint import (
    StatelessPolicyIdentity,
    load_stateless_policy_checkpoint,
    publish_stateless_policy_checkpoint,
)
from ptcg_rl.training.simple_stateless_pretrain_artifact import (
    SUPERVISED_POLICY_ARTIFACT_SCHEMA,
    WARMSTART_TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA,
    SupervisedPolicyInitializationRecord,
    SupervisedPolicySelectionRecord,
    TrainableParameterScopeAudit,
    load_supervised_policy_artifact,
)
from ptcg_rl.training.simple_stateless_pretrain_data import file_sha256

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PROVENANCE_FILENAME = "diagnostic_policy_provenance.json"


class DiagnosticRoutedPolicyConfig(BaseModel):
    """Immutable inputs for one non-resumable routed policy export."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_policy_path: Path
    supervised_manifest_path: Path
    output_dir: Path
    version: int = Field(ge=0)

    @field_validator(
        "baseline_policy_path",
        "supervised_manifest_path",
        "output_dir",
    )
    @classmethod
    def resolved_path(cls, value: Path) -> Path:
        """Remove cwd dependence before checking identities."""
        return value.resolve()

    @model_validator(mode="after")
    def separate_output(self) -> Self:
        """Never publish diagnostic files over either immutable input."""
        if self.output_dir in {
            self.baseline_policy_path,
            self.supervised_manifest_path,
        }:
            raise ValueError("diagnostic output directory overlaps an input file")
        return self


class DiagnosticFileBinding(BaseModel):
    """Path plus complete byte identity for one immutable file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    size_bytes: int = Field(gt=0)
    sha256: str

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("diagnostic file binding path must be absolute")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        return _fingerprint(value)


class DiagnosticRoutedPolicyProvenance(BaseModel):
    """Complete evidence that this policy is for diagnostics, never RL resume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["simple-stateless-diagnostic-routed-policy-v1"] = (
        "simple-stateless-diagnostic-routed-policy-v1"
    )
    diagnostic_only: Literal[True] = True
    no_learner_state: Literal[True] = True
    rl_resume_supported: Literal[False] = False
    baseline_policy: DiagnosticFileBinding
    baseline_version: int = Field(ge=0)
    baseline_model_state_fingerprint: str
    routed_identity: StatelessPolicyIdentity
    supervised_manifest: DiagnosticFileBinding
    supervised_policy: DiagnosticFileBinding
    supervised_model_state_fingerprint: str
    supervised_model_config_fingerprint: str
    supervised_exact_registry_fingerprint: str
    supervised_dataset_manifest_sha256: str
    supervised_dataset_fingerprint: str
    supervised_initialization: SupervisedPolicyInitializationRecord
    supervised_trainable_scope: TrainableParameterScopeAudit
    supervised_selection: SupervisedPolicySelectionRecord
    output_policy: DiagnosticFileBinding
    output_version: int = Field(ge=0)
    output_model_state_fingerprint: str

    @field_validator(
        "baseline_model_state_fingerprint",
        "supervised_model_state_fingerprint",
        "supervised_model_config_fingerprint",
        "supervised_exact_registry_fingerprint",
        "supervised_dataset_manifest_sha256",
        "supervised_dataset_fingerprint",
        "output_model_state_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        return _fingerprint(value)

    @model_validator(mode="after")
    def coherent_lineage(self) -> Self:
        """Cross-bind source, selected supervised state, and routed output."""
        selection = self.supervised_selection
        initialization = self.supervised_initialization
        if self.supervised_trainable_scope.mode != "full_model":
            raise ValueError("diagnostic routed policy requires full-model scope")
        if (
            selection.selected_model_state_fingerprint
            != self.supervised_model_state_fingerprint
            or selection.selected_checkpoint_sha256 is None
            or selection.selected_checkpoint_size_bytes is None
        ):
            raise ValueError("diagnostic supervised selection binding is incomplete")
        if (
            initialization.mode != "rl_pair"
            or initialization.source_policy_sha256 != self.baseline_policy.sha256
            or initialization.source_policy_model_fingerprint
            != self.baseline_model_state_fingerprint
            or initialization.source_pair_version != self.baseline_version
        ):
            raise ValueError("diagnostic supervised policy has another baseline")
        if (
            self.supervised_model_config_fingerprint
            != self.routed_identity.model_config_fingerprint
            or self.supervised_exact_registry_fingerprint
            != self.routed_identity.exact_registry_fingerprint
        ):
            raise ValueError("diagnostic supervised topology differs from baseline")
        if (
            self.output_model_state_fingerprint
            != self.supervised_model_state_fingerprint
            or self.output_version < 0
        ):
            raise ValueError("diagnostic routed output differs from supervised state")
        return self


def publish_diagnostic_routed_policy(
    config: DiagnosticRoutedPolicyConfig,
) -> dict[str, Any]:
    """Publish or verify one routed policy without creating resumable state."""
    baseline = load_stateless_policy_checkpoint(config.baseline_policy_path)
    if baseline.model_config_value.export_mode != "routed":
        raise ValueError("diagnostic baseline must be a routed policy")
    registry = baseline.model_config_value.resolved_registry_sha256
    if registry is None:
        raise ValueError("diagnostic baseline has no exact registry")
    manifest, supervised_state = load_supervised_policy_artifact(
        config.supervised_manifest_path,
        expected_model_config=baseline.model_config_value,
        expected_exact_registry_fingerprint=registry,
        expected_public_catalog_fingerprint=(
            baseline.identity.public_deck_catalog_fingerprint
        ),
        expected_input_contract_fingerprint=(
            baseline.identity.input_contract_fingerprint
        ),
        expected_event_contract_fingerprint=(
            baseline.identity.public_context_fingerprint
        ),
        expected_sequence_contract_fingerprint=(
            baseline.identity.sequence_contract_fingerprint
        ),
    )
    _validate_supervised_source(
        manifest=manifest,
        baseline_policy_sha256=baseline.artifact.policy_sha256,
        baseline_model_fingerprint=baseline.artifact.policy_model_fingerprint,
        baseline_version=baseline.artifact.version,
    )
    scope = manifest.trainable_scope
    if scope is None:
        raise RuntimeError("validated diagnostic artifact lost its scope")
    _validate_state_transition(
        baseline.model_state,
        supervised_state,
        trainable_parameter_names=scope.trainable_parameter_names,
    )

    policy_path = (
        config.output_dir / "weights" / f"policy_v{config.version}.pt"
    ).resolve()
    provenance_path = (config.output_dir / _PROVENANCE_FILENAME).resolve()
    _validate_output_inventory(
        config.output_dir,
        policy_path=policy_path,
        provenance_path=provenance_path,
    )
    if provenance_path.is_file() and not policy_path.is_file():
        raise FileNotFoundError(
            "diagnostic provenance exists without its routed policy"
        )

    model = _state_only_module(supervised_state)
    published = publish_stateless_policy_checkpoint(
        config.output_dir,
        version=config.version,
        model=model,
        model_config=baseline.model_config_value,
        identity=baseline.identity,
    )
    output = load_stateless_policy_checkpoint(
        published.policy_path,
        expected_identity=baseline.identity,
        expected_artifact=published,
    )
    expected = _provenance(
        config=config,
        baseline=baseline,
        manifest=manifest,
        output_policy_path=output.artifact.policy_path,
        output_policy_size=output.artifact.policy_size_bytes,
        output_policy_sha256=output.artifact.policy_sha256,
        output_model_fingerprint=output.artifact.policy_model_fingerprint,
    )
    if provenance_path.is_file():
        existing = DiagnosticRoutedPolicyProvenance.model_validate_json(
            provenance_path.read_text(encoding="utf-8")
        )
        if existing != expected:
            raise FileExistsError("diagnostic provenance differs from immutable retry")
    else:
        atomic_write_bytes(
            provenance_path,
            json_payload(expected.model_dump(mode="json")),
            overwrite=False,
        )
    _validate_output_inventory(
        config.output_dir,
        policy_path=policy_path,
        provenance_path=provenance_path,
    )
    return {
        "diagnostic_only": True,
        "no_learner_state": True,
        "rl_resume_supported": False,
        "version": config.version,
        "policy_path": str(output.artifact.policy_path),
        "policy_sha256": output.artifact.policy_sha256,
        "model_state_fingerprint": output.artifact.policy_model_fingerprint,
        "provenance_path": str(provenance_path),
        "provenance_sha256": file_sha256(provenance_path),
    }


def _validate_supervised_source(
    *,
    manifest: Any,
    baseline_policy_sha256: str,
    baseline_model_fingerprint: str,
    baseline_version: int,
) -> None:
    if manifest.format not in {
        SUPERVISED_POLICY_ARTIFACT_SCHEMA,
        WARMSTART_TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA,
    }:
        raise ValueError(
            "diagnostic publisher requires a pair-initialized supervised artifact"
        )
    scope = manifest.trainable_scope
    selection = manifest.selection
    if scope is None or scope.mode != "full_model":
        raise ValueError("diagnostic publisher requires full-model scope")
    if (
        not isinstance(selection, SupervisedPolicySelectionRecord)
        or selection.selected_model_state_fingerprint
        != manifest.model_state_fingerprint
        or selection.selected_checkpoint_sha256 is None
        or selection.selected_checkpoint_size_bytes is None
    ):
        raise ValueError("diagnostic publisher requires complete selection binding")
    initialization = manifest.initialization
    if (
        initialization.mode != "rl_pair"
        or initialization.source_policy_sha256 != baseline_policy_sha256
        or initialization.source_policy_model_fingerprint != baseline_model_fingerprint
        or initialization.source_pair_version != baseline_version
    ):
        raise ValueError("supervised artifact was not initialized from baseline")


def _validate_state_transition(
    baseline: Mapping[str, Tensor],
    supervised: Mapping[str, Tensor],
    *,
    trainable_parameter_names: tuple[str, ...],
) -> None:
    """Require one finite full-parameter update with unchanged buffers."""
    if set(baseline) != set(supervised):
        raise ValueError("supervised artifact changed routed state inventory")
    trainable = frozenset(trainable_parameter_names)
    if not trainable or not trainable.issubset(baseline):
        raise ValueError("supervised full-model parameter inventory is invalid")
    changed = 0
    for name in sorted(baseline):
        source = baseline[name]
        target = supervised[name]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(f"supervised routed tensor contract changed: {name}")
        if (target.is_floating_point() or target.is_complex()) and not bool(
            torch.isfinite(target).all()
        ):
            raise ValueError(f"supervised routed tensor is non-finite: {name}")
        if torch.equal(source, target):
            continue
        changed += 1
        if name not in trainable:
            raise ValueError(f"supervised artifact changed a routed buffer: {name}")
    if changed == 0:
        raise ValueError("supervised artifact is identical to baseline")


def _provenance(
    *,
    config: DiagnosticRoutedPolicyConfig,
    baseline: Any,
    manifest: Any,
    output_policy_path: Path,
    output_policy_size: int,
    output_policy_sha256: str,
    output_model_fingerprint: str,
) -> DiagnosticRoutedPolicyProvenance:
    selection = manifest.selection
    scope = manifest.trainable_scope
    if selection is None or scope is None:
        raise RuntimeError("validated diagnostic artifact lost its bindings")
    supervised_policy_path = (
        config.supervised_manifest_path.parent / manifest.policy_filename
    ).resolve()
    return DiagnosticRoutedPolicyProvenance(
        baseline_policy=_file_binding(config.baseline_policy_path),
        baseline_version=baseline.artifact.version,
        baseline_model_state_fingerprint=(baseline.artifact.policy_model_fingerprint),
        routed_identity=baseline.identity,
        supervised_manifest=_file_binding(config.supervised_manifest_path),
        supervised_policy=_file_binding(supervised_policy_path),
        supervised_model_state_fingerprint=manifest.model_state_fingerprint,
        supervised_model_config_fingerprint=manifest.model_config_fingerprint,
        supervised_exact_registry_fingerprint=manifest.exact_registry_fingerprint,
        supervised_dataset_manifest_sha256=manifest.dataset_manifest_sha256,
        supervised_dataset_fingerprint=manifest.dataset_fingerprint,
        supervised_initialization=manifest.initialization,
        supervised_trainable_scope=scope,
        supervised_selection=selection,
        output_policy=DiagnosticFileBinding(
            path=output_policy_path,
            size_bytes=output_policy_size,
            sha256=output_policy_sha256,
        ),
        output_version=config.version,
        output_model_state_fingerprint=output_model_fingerprint,
    )


def _validate_output_inventory(
    output_dir: Path,
    *,
    policy_path: Path,
    provenance_path: Path,
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError("diagnostic output path is not a directory")
    if not output_dir.exists():
        return
    allowed = {
        (output_dir / "weights").resolve(),
        policy_path,
        provenance_path,
    }
    unexpected = tuple(
        path.resolve()
        for path in output_dir.rglob("*")
        if path.resolve() not in allowed
    )
    if unexpected:
        raise FileExistsError(
            f"diagnostic output directory has conflicting content: {unexpected[0]}"
        )


def _state_only_module(state: Mapping[str, Tensor]) -> nn.Module:
    """Build a parameter-free module whose state_dict is the validated policy."""
    if not state:
        raise ValueError("diagnostic supervised state is empty")
    root = nn.Module()
    modules: dict[tuple[str, ...], nn.Module] = {(): root}
    for name in sorted(state):
        parts = tuple(name.split("."))
        if any(not part for part in parts):
            raise ValueError("diagnostic supervised state has an invalid name")
        for index in range(1, len(parts)):
            prefix = parts[:index]
            if prefix not in modules:
                parent = modules[prefix[:-1]]
                child = nn.Module()
                parent.add_module(prefix[-1], child)
                modules[prefix] = child
        modules[parts[:-1]].register_buffer(
            parts[-1],
            state[name].detach().to(device="cpu", copy=True).contiguous(),
        )
    if set(root.state_dict()) != set(state):
        raise RuntimeError("diagnostic state-only module changed tensor names")
    return root


def _file_binding(path: Path) -> DiagnosticFileBinding:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return DiagnosticFileBinding(
        path=resolved,
        size_bytes=resolved.stat().st_size,
        sha256=file_sha256(resolved),
    )


def _fingerprint(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("diagnostic identity must be lowercase SHA-256")
    return normalized


__all__ = [
    "DiagnosticRoutedPolicyConfig",
    "DiagnosticRoutedPolicyProvenance",
    "publish_diagnostic_routed_policy",
]
