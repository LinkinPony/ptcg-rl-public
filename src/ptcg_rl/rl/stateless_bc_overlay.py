"""Fail-closed supervised overlay for one settled stateless-v2 RL pair."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor

from ptcg_rl.model.simple_stateless import (
    SimpleStatelessModelConfig,
    uses_exact_v2_topology,
)
from ptcg_rl.rl.model_compatibility import model_config_fingerprint
from ptcg_rl.rl.stateless_bc_overlay_ops import (
    StatelessBcOverlayModelAudit,
    StatelessBcOverlayOptimizerAudit,
    audit_stateless_bc_overlay_state,
    transplant_stateless_bc_overlay_optimizer,
)
from ptcg_rl.rl.stateless_checkpoint import (
    LoadedStatelessCheckpointPair,
    StatelessSettledTrainingProgress,
)
from ptcg_rl.rl.stateless_curriculum import StatelessCurriculumState
from ptcg_rl.rl.stateless_deck_balance import StatelessDeckBalanceState
from ptcg_rl.training.simple_stateless_pretrain_artifact import (
    SUPERVISED_POLICY_ARTIFACT_SCHEMA,
    SupervisedPolicyArtifactManifest,
    load_supervised_policy_artifact,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class StatelessBcOverlayDeclaration(BaseModel):
    """Immutable authorization for one pair-bound private-policy overlay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["simple-stateless-bc-overlay-declaration-v1"] = (
        "simple-stateless-bc-overlay-declaration-v1"
    )
    source_pair_version: int = Field(ge=0)
    source_pair_manifest_sha256: str
    source_policy_sha256: str
    source_learner_state_sha256: str
    source_model_state_fingerprint: str
    supervised_manifest_sha256: str
    supervised_policy_sha256: str
    supervised_model_state_fingerprint: str
    target_deck_digest: str
    target_expert_id: str

    @field_validator(
        "source_pair_manifest_sha256",
        "source_policy_sha256",
        "source_learner_state_sha256",
        "source_model_state_fingerprint",
        "supervised_manifest_sha256",
        "supervised_policy_sha256",
        "supervised_model_state_fingerprint",
        "target_deck_digest",
        "target_expert_id",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require immutable full artifact and route identities."""
        return _fingerprint(value)


class StatelessBcOverlayCurriculumAudit(BaseModel):
    """Competence evidence invalidated after replacing one candidate policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_deck_digest: str
    invalidated_exact_statistics: int = Field(ge=0)
    preserved_exact_statistics: int = Field(ge=0)
    cleared_pilot_statistics: int = Field(ge=0)

    @field_validator("target_deck_digest")
    @classmethod
    def valid_target(cls, value: str) -> str:
        """Require the exact candidate identity."""
        return _fingerprint(value)


class StatelessBcOverlayAuditReport(BaseModel):
    """Complete evidence written before the new pair is published."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["simple-stateless-bc-overlay-audit-v1"] = (
        "simple-stateless-bc-overlay-audit-v1"
    )
    declaration: StatelessBcOverlayDeclaration
    source_pair_manifest_sha256: str
    supervised_manifest_sha256: str
    settled_progress: StatelessSettledTrainingProgress
    model: StatelessBcOverlayModelAudit
    optimizer: StatelessBcOverlayOptimizerAudit
    curriculum: StatelessBcOverlayCurriculumAudit
    source_fragment_parts_discarded: int = Field(ge=0)
    fragment_recovery_reset: Literal[True] = True

    @field_validator(
        "source_pair_manifest_sha256",
        "supervised_manifest_sha256",
    )
    @classmethod
    def valid_artifact(cls, value: str) -> str:
        """Require complete source artifact identities."""
        return _fingerprint(value)

    @model_validator(mode="after")
    def coherent_artifact_bindings(self) -> Self:
        """Keep the report attached to its authorized source and overlay."""
        if (
            self.source_pair_manifest_sha256
            != self.declaration.source_pair_manifest_sha256
            or self.supervised_manifest_sha256
            != self.declaration.supervised_manifest_sha256
        ):
            raise ValueError("BC overlay report differs from its declaration")
        return self


@dataclass(frozen=True)
class StatelessBcOverlayPlan:
    """Verified overlay tensors and source-derived state transitions."""

    manifest: SupervisedPolicyArtifactManifest
    model_state: dict[str, Tensor]
    model_audit: StatelessBcOverlayModelAudit
    settled_progress: StatelessSettledTrainingProgress
    curriculum_state: StatelessCurriculumState
    curriculum_audit: StatelessBcOverlayCurriculumAudit


def prepare_stateless_bc_overlay(
    *,
    source: LoadedStatelessCheckpointPair,
    artifact_manifest_path: Path,
    declaration: StatelessBcOverlayDeclaration,
    expected_model_config: SimpleStatelessModelConfig,
    expected_public_catalog_fingerprint: str,
    expected_input_contract_fingerprint: str,
) -> StatelessBcOverlayPlan:
    """Validate one exact source/artifact pair and derive its safe overlay."""
    _validate_source_pair(source, declaration, expected_model_config)
    manifest_sha256 = _file_sha256(artifact_manifest_path)
    if manifest_sha256 != declaration.supervised_manifest_sha256:
        raise ValueError("BC overlay supervised manifest is not authorized")
    manifest, loaded_state = load_supervised_policy_artifact(
        artifact_manifest_path,
        expected_model_config=expected_model_config,
        expected_exact_registry_fingerprint=str(
            expected_model_config.resolved_registry_sha256
        ),
        expected_public_catalog_fingerprint=expected_public_catalog_fingerprint,
        expected_input_contract_fingerprint=expected_input_contract_fingerprint,
    )
    if manifest.format != SUPERVISED_POLICY_ARTIFACT_SCHEMA:
        raise ValueError("BC overlay requires a v2 supervised artifact")
    if (
        manifest.policy_sha256 != declaration.supervised_policy_sha256
        or manifest.model_state_fingerprint
        != declaration.supervised_model_state_fingerprint
    ):
        raise ValueError("BC overlay supervised policy is not authorized")
    _validate_artifact_initialization(source, manifest)
    scope = manifest.trainable_scope
    if scope is None or scope.mode != "exact_actor_private_v2":
        raise ValueError("BC overlay requires an exact actor-private scope")
    if (
        scope.target_deck_digest != declaration.target_deck_digest
        or scope.target_expert_id != declaration.target_expert_id
    ):
        raise ValueError("BC overlay trainable scope changed its target route")
    model_state = dict(loaded_state)
    allowed_names = _actor_private_parameter_names(
        expected_model_config,
        model_state,
        target_deck_digest=declaration.target_deck_digest,
        target_expert_id=declaration.target_expert_id,
    )
    if scope.trainable_parameter_names != allowed_names:
        raise ValueError("BC overlay artifact whitelist is incomplete or overbroad")
    model_audit = audit_stateless_bc_overlay_state(
        source.model_state,
        model_state,
        trainable_parameter_names=allowed_names,
        target_deck_digest=declaration.target_deck_digest,
        target_expert_id=declaration.target_expert_id,
    )
    progress = require_stateless_bc_overlay_source_settled(source)
    curriculum_state, curriculum_audit = invalidate_stateless_bc_overlay_curriculum(
        source.curriculum_state,
        target_deck_digest=declaration.target_deck_digest,
    )
    require_stateless_bc_overlay_deck_balance_settled(source.deck_balance_state)
    return StatelessBcOverlayPlan(
        manifest=manifest,
        model_state=model_state,
        model_audit=model_audit,
        settled_progress=progress,
        curriculum_state=curriculum_state,
        curriculum_audit=curriculum_audit,
    )


def require_stateless_bc_overlay_source_settled(
    source: LoadedStatelessCheckpointPair,
) -> StatelessSettledTrainingProgress:
    """Require a complete optimizer boundary with no inferred cursor."""
    progress = source.settled_progress
    if (
        not source.settled_progress_recorded
        or progress is None
        or progress.rollout_window_index != source.pair.version
        or source.update_index != source.pair.version
    ):
        raise ValueError("BC overlay source is not a recorded settled pair")
    return progress


def require_stateless_bc_overlay_deck_balance_settled(
    state: StatelessDeckBalanceState,
) -> None:
    """Reject an overlay while candidate deck/seat leases are live."""
    if state.inflight_assignments:
        raise ValueError("BC overlay source has deck-balance assignments in flight")


def invalidate_stateless_bc_overlay_curriculum(
    state: StatelessCurriculumState,
    *,
    target_deck_digest: str,
) -> tuple[StatelessCurriculumState, StatelessBcOverlayCurriculumAudit]:
    """Drop target exact evidence and all non-separable pilot aggregates."""
    target = _fingerprint(target_deck_digest)
    if state.inflight:
        raise ValueError("BC overlay source has curriculum assignments in flight")
    if any(member.leases != 0 for member in state.members):
        raise ValueError("BC overlay source has PFSP member leases in flight")
    preserved: dict[str, Any] = {}
    invalidated = 0
    for key, statistic in state.exact_statistics.items():
        candidate, _pilot, _opponent = _parse_matchup_key(key)
        if candidate == target:
            invalidated += 1
        else:
            preserved[key] = statistic
    audit = StatelessBcOverlayCurriculumAudit(
        target_deck_digest=target,
        invalidated_exact_statistics=invalidated,
        preserved_exact_statistics=len(preserved),
        cleared_pilot_statistics=len(state.pilot_statistics),
    )
    return (
        state.model_copy(
            update={
                "exact_statistics": preserved,
                "pilot_statistics": {},
            }
        ),
        audit,
    )


def _validate_source_pair(
    source: LoadedStatelessCheckpointPair,
    declaration: StatelessBcOverlayDeclaration,
    expected_model_config: SimpleStatelessModelConfig,
) -> None:
    if not uses_exact_v2_topology(expected_model_config):
        raise ValueError("BC overlay requires V2 exact routes")
    expected = {
        "pair manifest": (
            source.pair.pair_manifest_sha256,
            declaration.source_pair_manifest_sha256,
        ),
        "policy": (source.pair.policy_sha256, declaration.source_policy_sha256),
        "learner state": (
            source.pair.learner_state_sha256,
            declaration.source_learner_state_sha256,
        ),
        "model state": (
            source.pair.policy_model_fingerprint,
            declaration.source_model_state_fingerprint,
        ),
    }
    for label, (observed, authorized) in expected.items():
        if observed != authorized:
            raise ValueError(f"BC overlay source {label} is not authorized")
    if source.pair.version != declaration.source_pair_version:
        raise ValueError("BC overlay source pair version is not authorized")
    if source.model_config_value != expected_model_config:
        raise ValueError("BC overlay cannot change model topology or registry")
    if source.pair.identity.model_config_fingerprint != model_config_fingerprint(
        expected_model_config
    ):
        raise ValueError("BC overlay source model-config identity changed")


def _validate_artifact_initialization(
    source: LoadedStatelessCheckpointPair,
    manifest: SupervisedPolicyArtifactManifest,
) -> None:
    initialization = manifest.initialization
    expected = (
        "rl_pair",
        source.pair.pair_manifest_sha256,
        source.pair.version,
        source.pair.policy_sha256,
        source.pair.policy_model_fingerprint,
    )
    actual = (
        initialization.mode,
        initialization.source_pair_manifest_sha256,
        initialization.source_pair_version,
        initialization.source_policy_sha256,
        initialization.source_policy_model_fingerprint,
    )
    if actual != expected:
        raise ValueError("BC overlay artifact was not trained from its source pair")


def _actor_private_parameter_names(
    model_config: SimpleStatelessModelConfig,
    state: Mapping[str, Tensor],
    *,
    target_deck_digest: str,
    target_expert_id: str,
) -> tuple[str, ...]:
    matches = tuple(
        route
        for route in model_config.exact_routes
        if route.deck_digest == target_deck_digest
    )
    if len(matches) != 1 or matches[0].expert_id != target_expert_id:
        raise ValueError("BC overlay target route is absent from the exact registry")
    module_key = matches[0].module_key
    prompt_name = f"backbone.v2_adapters.prompts.{module_key}.policy_and_scratch"
    head_prefixes = (
        f"heads.policy_residuals.{module_key}.",
        f"heads.option_residuals.{module_key}.",
    )
    capsule_marker = f".exact_capsules.{module_key}.policy_residual."
    names = tuple(
        sorted(
            name
            for name in state
            if name == prompt_name
            or name.startswith(head_prefixes)
            or capsule_marker in name
        )
    )
    if not names:
        raise ValueError("BC overlay target route has no actor-private parameters")
    return names


def _parse_matchup_key(value: str) -> tuple[str, str, str]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("BC overlay found an invalid exact-statistic key") from error
    if (
        not isinstance(decoded, list)
        or len(decoded) != 3
        or not all(isinstance(item, str) for item in decoded)
    ):
        raise ValueError("BC overlay found an invalid exact-statistic key")
    candidate, pilot, opponent = cast(list[str], decoded)
    return (
        _fingerprint(candidate),
        _fingerprint(pilot),
        _fingerprint(opponent),
    )


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("BC overlay identity must be lowercase SHA-256")
    return normalized


__all__ = [
    "StatelessBcOverlayAuditReport",
    "StatelessBcOverlayCurriculumAudit",
    "StatelessBcOverlayDeclaration",
    "StatelessBcOverlayModelAudit",
    "StatelessBcOverlayOptimizerAudit",
    "StatelessBcOverlayPlan",
    "audit_stateless_bc_overlay_state",
    "invalidate_stateless_bc_overlay_curriculum",
    "prepare_stateless_bc_overlay",
    "require_stateless_bc_overlay_deck_balance_settled",
    "require_stateless_bc_overlay_source_settled",
    "transplant_stateless_bc_overlay_optimizer",
]
