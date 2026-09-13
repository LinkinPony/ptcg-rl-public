"""Build single-checkpoint parameter ensembles with strict identity checks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from torch import Tensor

from ptcg_rl.rl.checkpoint_pair_io import publish_torch_file
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.stateless_checkpoint import (
    LoadedStatelessPolicyCheckpoint,
    load_stateless_policy_checkpoint,
)

_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RUNTIME_IDENTITY_FIELDS = (
    "model_config_fingerprint",
    "exact_registry_fingerprint",
    "active_exact_deck_digests",
    "action_schema_fingerprint",
    "public_context_fingerprint",
    "card_catalog_fingerprint",
    "belief_target_semantics_fingerprint",
    "public_deck_catalog_fingerprint",
    "input_contract_fingerprint",
    "training_roster_fingerprint",
    "sequence_contract_fingerprint",
)


class EnsembleSourceConfig(BaseModel):
    """One immutable checkpoint available to a merge campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_path: Path
    expected_sha256: str

    @field_validator("expected_sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require the complete immutable checkpoint identity."""
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("expected_sha256 must be lowercase SHA-256")
        return normalized


class EnsembleCandidateConfig(BaseModel):
    """One parameter-space ensemble produced by a campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    weights: dict[str, float]
    scope: Literal["all", "shared", "routed"] = "all"

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        """Keep artifact names portable and unambiguous."""
        normalized = value.strip().lower()
        if _NAME_PATTERN.fullmatch(normalized) is None:
            raise ValueError("candidate name contains unsupported characters")
        return normalized

    @model_validator(mode="after")
    def valid_weights(self) -> EnsembleCandidateConfig:
        """Require a finite affine combination."""
        if not self.weights:
            raise ValueError("candidate weights must not be empty")
        if any(not math.isfinite(weight) for weight in self.weights.values()):
            raise ValueError("candidate weights must be finite")
        if not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("candidate weights must sum to one")
        return self


class CheckpointEnsembleConfig(BaseModel):
    """Validated campaign for producing one or more single checkpoints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sources: dict[str, EnsembleSourceConfig]
    anchor_source: str
    candidates: tuple[EnsembleCandidateConfig, ...]
    output_dir: Path

    @model_validator(mode="after")
    def valid_campaign(self) -> CheckpointEnsembleConfig:
        """Validate source references and output names."""
        if self.anchor_source not in self.sources:
            raise ValueError("anchor_source is absent from sources")
        if len(self.sources) < 2:
            raise ValueError("checkpoint ensemble needs at least two sources")
        if not self.candidates:
            raise ValueError("checkpoint ensemble needs at least one candidate")
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("candidate names must be unique")
        for candidate in self.candidates:
            missing = sorted(set(candidate.weights) - set(self.sources))
            if missing:
                raise ValueError(
                    f"candidate {candidate.name} has unknown sources: {missing}"
                )
        return self


def build_checkpoint_ensembles(config: CheckpointEnsembleConfig) -> dict[str, object]:
    """Build and verify all configured single-checkpoint ensembles."""
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = {
        name: _load_verified_source(source)
        for name, source in config.sources.items()
    }
    anchor = loaded[config.anchor_source]
    _validate_compatibility(loaded, anchor_name=config.anchor_source)
    source_summary = {
        name: {
            "checkpoint_path": str(item.artifact.policy_path),
            "checkpoint_sha256": item.artifact.policy_sha256,
            "model_fingerprint": item.artifact.policy_model_fingerprint,
            "version": item.artifact.version,
        }
        for name, item in loaded.items()
    }
    results: list[dict[str, object]] = []
    for candidate in config.candidates:
        state, selected = merge_model_states(
            {name: item.model_state for name, item in loaded.items()},
            weights=candidate.weights,
            scope=candidate.scope,
            anchor_name=config.anchor_source,
        )
        results.append(
            _publish_candidate(
                candidate,
                state=state,
                selected=selected,
                anchor=anchor,
                sources=source_summary,
                output_dir=output_dir,
            )
        )
        del state
    summary: dict[str, object] = {
        "format": "checkpoint_ensemble_campaign_v1",
        "anchor_source": config.anchor_source,
        "sources": source_summary,
        "candidates": results,
    }
    _write_json_exclusive(output_dir / "summary.json", summary)
    return summary


def merge_model_states(
    states: Mapping[str, Mapping[str, Tensor]],
    *,
    weights: Mapping[str, float],
    scope: Literal["all", "shared", "routed"],
    anchor_name: str,
) -> tuple[dict[str, Tensor], tuple[str, ...]]:
    """Return one affine parameter merge and the selected tensor names."""
    if anchor_name not in states:
        raise ValueError("anchor state is missing")
    if not weights or set(weights) - set(states):
        raise ValueError("weights reference missing model states")
    anchor = states[anchor_name]
    selected = tuple(name for name in anchor if _selected(name, scope))
    output: dict[str, Tensor] = {}
    for name, anchor_tensor in anchor.items():
        tensors = [states[source][name] for source in weights]
        if name not in selected:
            output[name] = anchor_tensor.detach().clone()
            continue
        if not (anchor_tensor.is_floating_point() or anchor_tensor.is_complex()):
            if any(not torch.equal(anchor_tensor, tensor) for tensor in tensors):
                raise ValueError(f"selected non-floating tensor differs: {name}")
            output[name] = anchor_tensor.detach().clone()
            continue
        merged = torch.zeros_like(anchor_tensor, memory_format=torch.preserve_format)
        for source, weight in weights.items():
            merged.add_(states[source][name], alpha=weight)
        output[name] = merged
    return output, selected


def is_routed_parameter(name: str) -> bool:
    """Return whether a tensor belongs to family/exact routed capacity."""
    return (
        name.startswith("backbone.family_private.")
        or ".exact_capsules." in name
        or name.startswith("backbone.v2_adapters.prompts.")
        or name.startswith("heads.policy_residuals.")
        or name.startswith("heads.option_residuals.")
        or name.startswith("heads.value_residuals.")
    )


def _selected(name: str, scope: str) -> bool:
    if scope == "all":
        return True
    routed = is_routed_parameter(name)
    return routed if scope == "routed" else not routed


def _load_verified_source(
    source: EnsembleSourceConfig,
) -> LoadedStatelessPolicyCheckpoint:
    path = source.checkpoint_path.resolve(strict=True)
    observed = _file_sha256(path)
    if observed != source.expected_sha256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {path}")
    loaded = load_stateless_policy_checkpoint(path)
    if loaded.artifact.policy_sha256 != observed:
        raise RuntimeError("validated checkpoint artifact changed while loading")
    return loaded


def _validate_compatibility(
    loaded: Mapping[str, LoadedStatelessPolicyCheckpoint],
    *,
    anchor_name: str,
) -> None:
    anchor = loaded[anchor_name]
    anchor_state = anchor.model_state
    anchor_identity = anchor.identity.model_dump(mode="python")
    for source_name, source in loaded.items():
        identity = source.identity.model_dump(mode="python")
        mismatched = [
            field
            for field in _RUNTIME_IDENTITY_FIELDS
            if identity[field] != anchor_identity[field]
        ]
        if source.model_config_value != anchor.model_config_value:
            mismatched.append("model_config")
        if mismatched:
            raise ValueError(
                f"source {source_name} has incompatible runtime identity: {mismatched}"
            )
        if set(source.model_state) != set(anchor_state):
            raise ValueError(f"source {source_name} has different tensor names")
        for name, anchor_tensor in anchor_state.items():
            tensor = source.model_state[name]
            if tensor.shape != anchor_tensor.shape or tensor.dtype != anchor_tensor.dtype:
                raise ValueError(
                    f"source {source_name} tensor contract differs for {name}"
                )


def _publish_candidate(
    candidate: EnsembleCandidateConfig,
    *,
    state: Mapping[str, Tensor],
    selected: Sequence[str],
    anchor: LoadedStatelessPolicyCheckpoint,
    sources: Mapping[str, Mapping[str, object]],
    output_dir: Path,
) -> dict[str, object]:
    checkpoint_path = output_dir / f"{candidate.name}.pt"
    manifest_path = output_dir / f"{candidate.name}.manifest.json"
    if checkpoint_path.exists() or manifest_path.exists():
        raise FileExistsError(f"ensemble candidate already exists: {candidate.name}")
    model_fingerprint = canonical_model_state_fingerprint(state)
    payload = {
        "format": "simple_stateless_policy_v1",
        "version": anchor.artifact.version,
        "identity": anchor.identity.model_dump(mode="json"),
        "model_config": anchor.model_config_value.model_dump(mode="json"),
        "model_fingerprint": model_fingerprint,
        "model_state": dict(state),
    }
    size_bytes, checkpoint_sha256 = publish_torch_file(checkpoint_path, payload)
    verified = load_stateless_policy_checkpoint(checkpoint_path)
    if verified.artifact.policy_model_fingerprint != model_fingerprint:
        raise RuntimeError("published ensemble model fingerprint mismatch")
    provenance = {
        "name": candidate.name,
        "scope": candidate.scope,
        "weights": dict(candidate.weights),
        "anchor_version": anchor.artifact.version,
        "selected_tensor_count": len(selected),
        "total_tensor_count": len(state),
        "sources": {
            name: dict(sources[name]) for name in candidate.weights
        },
    }
    manifest: dict[str, object] = {
        "format": "diagnostic_policy_ensemble_manifest_v1",
        "version": anchor.artifact.version,
        "metadata": {
            "simple_stateless_identity": anchor.identity.model_dump(mode="json")
        },
        "policy": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "size_bytes": size_bytes,
            "model_fingerprint": model_fingerprint,
        },
        "ensemble": provenance,
    }
    _write_json_exclusive(manifest_path, manifest)
    return {
        **provenance,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": size_bytes,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "model_fingerprint": model_fingerprint,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)


__all__ = [
    "CheckpointEnsembleConfig",
    "EnsembleCandidateConfig",
    "EnsembleSourceConfig",
    "build_checkpoint_ensembles",
    "is_routed_parameter",
    "merge_model_states",
]
