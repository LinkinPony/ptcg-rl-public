"""Exact policy/learner-state pairs for the clean stateless lineage."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor, nn

from ptcg_rl.model.simple_stateless import SimpleStatelessModelConfig
from ptcg_rl.rl.checkpoint_pair_io import (
    atomic_write_bytes,
    fsync_directory,
    json_payload,
    pair_manifest_version,
    publish_bytes_file,
    publish_torch_file,
)
from ptcg_rl.rl.durable_writer import AsyncWriteBusyError, freeze_durable_value
from ptcg_rl.rl.model_compatibility import model_config_fingerprint
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.stateless_curriculum import (
    DurablePolicyArtifact,
    StatelessCurriculumState,
    VerifiedPolicyPublication,
    compact_stateless_curriculum_state,
    load_compact_stateless_curriculum_state,
    stateless_curriculum_state_fingerprint,
)
from ptcg_rl.rl.stateless_deck_balance import StatelessDeckBalanceState
from ptcg_rl.rl.stateless_fragment_io import CompactFragmentManifest
from ptcg_rl.rl.stateless_opponent_pool_state import (
    OpponentPoolCheckpointState,
    opponent_pool_checkpoint_state,
)
from ptcg_rl.rl.stateless_ppo import SimpleStatelessPpoConfig

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMPACT_CURRICULUM_FORMAT = "simple_stateless_curriculum_msgpack_zlib_v1"


class StatelessPolicyIdentity(BaseModel):
    """All immutable train/serve identities carried by a policy checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_config_fingerprint: str
    exact_registry_fingerprint: str
    active_exact_deck_digests: tuple[str, ...]
    action_schema_fingerprint: str
    public_context_fingerprint: str
    card_catalog_fingerprint: str
    belief_target_semantics_fingerprint: str
    public_deck_catalog_fingerprint: str
    input_contract_fingerprint: str
    resolved_config_fingerprint: str
    fragment_static_contract_fingerprint: str
    curriculum_config_fingerprint: str
    pinned_manifest_fingerprint: str
    scripted_manifest_fingerprint: str
    training_roster_fingerprint: str
    sequence_contract_fingerprint: str | None = None

    @field_validator("sequence_contract_fingerprint")
    @classmethod
    def valid_optional_sequence_fingerprint(
        cls,
        value: str | None,
    ) -> str | None:
        """Validate the optional temporal architecture contract identity."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("sequence contract must be lowercase SHA-256")
        return normalized

    @field_validator(
        "model_config_fingerprint",
        "exact_registry_fingerprint",
        "action_schema_fingerprint",
        "public_context_fingerprint",
        "card_catalog_fingerprint",
        "belief_target_semantics_fingerprint",
        "public_deck_catalog_fingerprint",
        "input_contract_fingerprint",
        "resolved_config_fingerprint",
        "fragment_static_contract_fingerprint",
        "curriculum_config_fingerprint",
        "pinned_manifest_fingerprint",
        "scripted_manifest_fingerprint",
        "training_roster_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require complete content identities."""
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("policy identity values must be lowercase SHA-256")
        return normalized

    @field_validator("active_exact_deck_digests")
    @classmethod
    def valid_routes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require a sorted, unique, non-empty active roster."""
        normalized = tuple(value.strip().lower() for value in values)
        if (
            not normalized
            or tuple(sorted(set(normalized))) != normalized
            or any(_SHA256_PATTERN.fullmatch(value) is None for value in normalized)
        ):
            raise ValueError("active exact deck digests must be sorted unique SHA-256")
        return normalized


class StatelessStalenessState(BaseModel):
    """Moving behavior publication and replay acceptance cursors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    behavior_policy_version: int = Field(ge=0)
    behavior_policy_fingerprint: str
    oldest_accepted_behavior_version: int = Field(ge=0)
    fragments_seen: int = Field(default=0, ge=0)
    fragments_stale: int = Field(default=0, ge=0)

    @field_validator("behavior_policy_fingerprint")
    @classmethod
    def valid_behavior_fingerprint(cls, value: str) -> str:
        """Require an immutable full-state behavior identity."""
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("behavior policy fingerprint must be SHA-256")
        return normalized

    @model_validator(mode="after")
    def coherent_versions(self) -> Self:
        """The replay floor cannot be newer than the published behavior."""
        if self.oldest_accepted_behavior_version > self.behavior_policy_version:
            raise ValueError("staleness floor is newer than behavior policy")
        if self.fragments_stale > self.fragments_seen:
            raise ValueError("stale fragment count exceeds observed fragments")
        return self


class StatelessSettledTrainingProgress(BaseModel):
    """Durable clocks at a complete rollout-window transaction boundary.

    Exact-resume pairs intentionally cannot represent a partially consumed
    optimizer window. Supporting that would also require the immutable replay
    rows and their logical-batch plan, neither of which is part of this pair.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["stateless-settled-training-progress-v1"] = (
        "stateless-settled-training-progress-v1"
    )
    rollout_window_index: int = Field(ge=0)
    optimizer_step_index: int = Field(ge=0)
    fresh_decisions_seen: int = Field(ge=0)
    lr_schedule_decisions_seen: int = Field(ge=0)
    pending_optimizer_window: None = None


class StatelessStartupReport(BaseModel):
    """One immutable report named by a startup provenance root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[
        "bc_overlay_audit",
        "supervised_initialization",
        "topology_transition",
        "transition_source",
        "weights_only_warm_start",
    ]
    path: Path
    size_bytes: int = Field(gt=0)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require a complete report identity."""
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("startup report identity must be SHA-256")
        return normalized

    @field_validator("path")
    @classmethod
    def absolute_report_path(cls, value: Path) -> Path:
        """Avoid cwd-dependent provenance after checkpoint publication."""
        if not value.is_absolute():
            raise ValueError("startup report path must be absolute")
        return value


class StatelessStartupProvenance(BaseModel):
    """Immutable audit root for a checkpoint-lineage startup mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["stateless-startup-provenance-v1"] = (
        "stateless-startup-provenance-v1"
    )
    operation: Literal[
        "bc_overlay",
        "identity_materialization",
        "supervised_initialization",
        "topology_materialization",
        "weights_only_warm_start",
    ]
    startup_pair_version: int = Field(ge=0)
    reports: tuple[StatelessStartupReport, ...]
    source_pair_manifest_sha256: str | None = None
    target_model_state_fingerprint: str
    target_model_config_fingerprint: str
    target_exact_registry_fingerprint: str
    supervised_manifest_sha256: str | None = None
    supervised_policy_sha256: str | None = None
    supervised_dataset_manifest_sha256: str | None = None
    supervised_dataset_fingerprint: str | None = None
    target_deck_digest: str | None = None

    @field_validator(
        "source_pair_manifest_sha256",
        "target_model_state_fingerprint",
        "target_model_config_fingerprint",
        "target_exact_registry_fingerprint",
        "supervised_manifest_sha256",
        "supervised_policy_sha256",
        "supervised_dataset_manifest_sha256",
        "supervised_dataset_fingerprint",
        "target_deck_digest",
    )
    @classmethod
    def valid_fingerprint(cls, value: str | None) -> str | None:
        """Require complete content, topology, and route identities."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("startup provenance identity must be SHA-256")
        return normalized

    @model_validator(mode="after")
    def coherent_operation(self) -> Self:
        """Require the exact report and identity inventory for each mutation."""
        kinds = tuple(report.kind for report in self.reports)
        if len(kinds) != len(set(kinds)):
            raise ValueError("startup provenance report kinds must be unique")
        if self.operation == "bc_overlay":
            if (
                kinds != ("bc_overlay_audit",)
                or self.source_pair_manifest_sha256 is None
                or self.supervised_manifest_sha256 is None
                or self.supervised_policy_sha256 is not None
                or self.supervised_dataset_manifest_sha256 is not None
                or self.supervised_dataset_fingerprint is not None
                or self.target_deck_digest is None
            ):
                raise ValueError("BC overlay startup provenance is incomplete")
        elif self.operation == "supervised_initialization":
            if (
                kinds != ("supervised_initialization",)
                or self.startup_pair_version != 0
                or self.source_pair_manifest_sha256 is not None
                or self.supervised_manifest_sha256 is None
                or self.supervised_policy_sha256 is None
                or self.supervised_dataset_manifest_sha256 is None
                or self.supervised_dataset_fingerprint is None
                or self.target_deck_digest is not None
            ):
                raise ValueError("supervised initialization provenance is incomplete")
        elif self.operation == "weights_only_warm_start":
            if (
                kinds != ("weights_only_warm_start",)
                or self.startup_pair_version != 0
                or self.source_pair_manifest_sha256 is None
                or self.supervised_manifest_sha256 is not None
                or self.supervised_policy_sha256 is not None
                or self.supervised_dataset_manifest_sha256 is not None
                or self.supervised_dataset_fingerprint is not None
                or self.target_deck_digest is not None
            ):
                raise ValueError("weights-only warm-start provenance is incomplete")
        elif self.operation == "identity_materialization":
            if (
                kinds != ("transition_source",)
                or self.source_pair_manifest_sha256 is None
                or self.supervised_manifest_sha256 is not None
                or self.supervised_policy_sha256 is not None
                or self.supervised_dataset_manifest_sha256 is not None
                or self.supervised_dataset_fingerprint is not None
                or self.target_deck_digest is not None
            ):
                raise ValueError("identity materialization provenance is incomplete")
        elif (
            kinds != ("topology_transition", "transition_source")
            or self.source_pair_manifest_sha256 is None
            or self.supervised_manifest_sha256 is not None
            or self.supervised_policy_sha256 is not None
            or self.supervised_dataset_manifest_sha256 is not None
            or self.supervised_dataset_fingerprint is not None
            or self.target_deck_digest is not None
        ):
            raise ValueError("topology materialization provenance is incomplete")
        return self


class StatelessCheckpointPair(BaseModel):
    """Resolved immutable files and their manifest identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(ge=0)
    policy_path: Path
    policy_size_bytes: int = Field(gt=0)
    policy_sha256: str
    policy_model_fingerprint: str
    learner_state_path: Path
    learner_state_size_bytes: int = Field(gt=0)
    learner_state_sha256: str
    pair_manifest_path: Path
    pair_manifest_sha256: str
    identity: StatelessPolicyIdentity
    startup_provenance: StatelessStartupProvenance | None = None

    @field_validator(
        "policy_sha256",
        "policy_model_fingerprint",
        "learner_state_sha256",
        "pair_manifest_sha256",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require all artifact identities to be SHA-256."""
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("checkpoint pair fingerprint must be SHA-256")
        return normalized

    @property
    def durable_artifact(self) -> DurablePolicyArtifact:
        """Return the policy-only curriculum admission artifact."""
        return DurablePolicyArtifact(
            version=self.version,
            policy_path=self.policy_path,
            policy_size_bytes=self.policy_size_bytes,
            policy_sha256=self.policy_sha256,
            policy_model_fingerprint=self.policy_model_fingerprint,
            model_config_fingerprint=self.identity.model_config_fingerprint,
            exact_registry_fingerprint=self.identity.exact_registry_fingerprint,
            active_exact_deck_digests=self.identity.active_exact_deck_digests,
            input_contract_fingerprint=self.identity.input_contract_fingerprint,
            training_roster_fingerprint=self.identity.training_roster_fingerprint,
        )


class LoadedStatelessPolicyCheckpoint(BaseModel):
    """Validated policy-only payload used by past-self inference."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    artifact: DurablePolicyArtifact
    identity: StatelessPolicyIdentity
    model_config_value: SimpleStatelessModelConfig
    model_state: dict[str, Tensor]


class LoadedStatelessCheckpointPair(BaseModel):
    """Validated exact-resume payloads."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    pair: StatelessCheckpointPair
    model_config_value: SimpleStatelessModelConfig
    model_state: dict[str, Tensor]
    ppo_config: SimpleStatelessPpoConfig
    optimizer_state: dict[str, Any]
    update_index: int = Field(ge=0)
    settled_progress: StatelessSettledTrainingProgress | None = None
    settled_progress_recorded: bool = False
    fragment_recovery: CompactFragmentManifest
    deck_balance_state: StatelessDeckBalanceState
    curriculum_state: StatelessCurriculumState
    opponent_pool_state: OpponentPoolCheckpointState | None = None
    opponent_pool_state_recorded: bool = False
    staleness_state: StatelessStalenessState
    startup_provenance: StatelessStartupProvenance | None = None
    curriculum_predecessor_fingerprint: str | None = None
    curriculum_predecessor_recorded: bool = False


@dataclass(frozen=True, slots=True)
class PublishedStatelessCheckpointPair:
    """One background-published pair and its detached-write timings."""

    pair: StatelessCheckpointPair
    freeze_seconds: float
    background_seconds: float


class AsyncStatelessCheckpointPairPublisher:
    """Freeze optimizer state synchronously and publish one pair in background."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="stateless-checkpoint-pair",
        )
        self._lock = threading.Lock()
        self._pending: Future[PublishedStatelessCheckpointPair] | None = None
        self._last_result: PublishedStatelessCheckpointPair | None = None
        self._closed = False

    @property
    def busy(self) -> bool:
        """Return whether one publication has not crossed its barrier."""
        with self._lock:
            return self._pending is not None and not self._pending.done()

    def submit(
        self,
        output_dir: Path,
        *,
        version: int,
        model_config: SimpleStatelessModelConfig,
        identity: StatelessPolicyIdentity,
        ppo_config: SimpleStatelessPpoConfig,
        optimizer_state: Mapping[str, Any],
        update_index: int,
        settled_progress: StatelessSettledTrainingProgress,
        fragment_recovery: CompactFragmentManifest,
        deck_balance_state: StatelessDeckBalanceState,
        curriculum_state: StatelessCurriculumState,
        staleness_state: StatelessStalenessState,
        curriculum_predecessor_state: StatelessCurriculumState | None,
        policy_publication: VerifiedPolicyPublication,
        opponent_pool_state: OpponentPoolCheckpointState | None = None,
        startup_provenance: StatelessStartupProvenance | None = None,
    ) -> Future[PublishedStatelessCheckpointPair]:
        """Detach mutable optimizer tensors before handing work to the writer."""
        with self._lock:
            if self._closed:
                raise RuntimeError("stateless checkpoint publisher is closed")
            if self._pending is not None:
                if not self._pending.done():
                    raise AsyncWriteBusyError(
                        "one stateless checkpoint pair is already in flight"
                    )
                self._last_result = self._pending.result()
                self._pending = None
            freeze_started_at = time.perf_counter()
            policy_publication.verify_unchanged()
            frozen_optimizer = freeze_durable_value(optimizer_state)
            if not isinstance(frozen_optimizer, Mapping):
                raise TypeError("frozen optimizer state must be a mapping")
            freeze_seconds = time.perf_counter() - freeze_started_at
            future = self._executor.submit(
                self._publish,
                output_dir,
                version=version,
                model_config=model_config,
                identity=identity,
                ppo_config=ppo_config,
                optimizer_state=frozen_optimizer,
                update_index=update_index,
                settled_progress=settled_progress,
                fragment_recovery=fragment_recovery,
                deck_balance_state=deck_balance_state,
                curriculum_state=curriculum_state,
                staleness_state=staleness_state,
                curriculum_predecessor_state=curriculum_predecessor_state,
                policy_publication=policy_publication,
                opponent_pool_state=opponent_pool_state,
                startup_provenance=startup_provenance,
                freeze_seconds=freeze_seconds,
            )
            self._pending = future
            return future

    def barrier(self) -> PublishedStatelessCheckpointPair | None:
        """Wait for the current pair and surface any durability failure."""
        with self._lock:
            pending = self._pending
            if pending is None:
                return self._last_result
            result = pending.result()
            self._last_result = result
            self._pending = None
            return result

    def close(self) -> None:
        """Commit pending work and stop the publication thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.barrier()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)

    @staticmethod
    def _publish(
        output_dir: Path,
        *,
        freeze_seconds: float,
        **kwargs: Any,
    ) -> PublishedStatelessCheckpointPair:
        started_at = time.perf_counter()
        pair = publish_stateless_checkpoint_pair(
            output_dir,
            model=None,
            **kwargs,
        )
        return PublishedStatelessCheckpointPair(
            pair=pair,
            freeze_seconds=freeze_seconds,
            background_seconds=time.perf_counter() - started_at,
        )


def publish_verified_stateless_policy_checkpoint(
    output_dir: Path,
    *,
    version: int,
    model: nn.Module,
    model_config: SimpleStatelessModelConfig,
    identity: StatelessPolicyIdentity,
) -> VerifiedPolicyPublication:
    """Publish one policy and return a same-process verified capability."""
    if version < 0:
        raise ValueError("checkpoint version must be non-negative")
    _validate_policy_contract(model_config, identity)
    policy_path = (output_dir / "weights" / f"policy_v{version}.pt").resolve()
    state = _frozen_cpu_state(model.state_dict())
    model_fingerprint = canonical_model_state_fingerprint(state)
    if policy_path.exists():
        loaded = load_stateless_policy_checkpoint(
            policy_path,
            expected_identity=identity,
        )
        if (
            loaded.artifact.version != version
            or loaded.model_config_value != model_config
            or loaded.artifact.policy_model_fingerprint != model_fingerprint
        ):
            raise FileExistsError(
                f"immutable policy checkpoint differs from retry: {policy_path}"
            )
        return VerifiedPolicyPublication.capture(loaded.artifact)
    payload = {
        "format": "simple_stateless_policy_v1",
        "version": version,
        "identity": identity.model_dump(mode="json"),
        "model_config": model_config.model_dump(mode="json"),
        "model_fingerprint": model_fingerprint,
        "model_state": state,
    }
    policy_size, policy_sha = publish_torch_file(policy_path, payload)
    artifact = DurablePolicyArtifact(
        version=version,
        policy_path=policy_path,
        policy_size_bytes=policy_size,
        policy_sha256=policy_sha,
        policy_model_fingerprint=model_fingerprint,
        model_config_fingerprint=identity.model_config_fingerprint,
        exact_registry_fingerprint=identity.exact_registry_fingerprint,
        active_exact_deck_digests=identity.active_exact_deck_digests,
        input_contract_fingerprint=identity.input_contract_fingerprint,
        training_roster_fingerprint=identity.training_roster_fingerprint,
    )
    return VerifiedPolicyPublication.capture(artifact)


def publish_stateless_policy_checkpoint(
    output_dir: Path,
    *,
    version: int,
    model: nn.Module,
    model_config: SimpleStatelessModelConfig,
    identity: StatelessPolicyIdentity,
) -> DurablePolicyArtifact:
    """Publish or verify one immutable policy checkpoint without a sidecar."""
    return publish_verified_stateless_policy_checkpoint(
        output_dir,
        version=version,
        model=model,
        model_config=model_config,
        identity=identity,
    ).artifact


def load_stateless_policy_checkpoint(
    policy_path: Path,
    *,
    expected_identity: StatelessPolicyIdentity | None = None,
    expected_artifact: DurablePolicyArtifact | None = None,
) -> LoadedStatelessPolicyCheckpoint:
    """Load and validate one policy checkpoint independently of learner state."""
    resolved_path = policy_path.resolve()
    if expected_artifact is not None:
        expected_artifact.verify()
        if resolved_path != expected_artifact.policy_path.resolve():
            raise ValueError("policy checkpoint path differs from durable artifact")
    payload = _mapping(
        torch.load(resolved_path, map_location="cpu", weights_only=False),
        "policy payload",
    )
    version = int(payload["version"])
    _require_payload_header(
        payload,
        expected_format="simple_stateless_policy_v1",
        version=version,
    )
    identity = StatelessPolicyIdentity.model_validate(payload.get("identity"))
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("policy checkpoint identity mismatch")
    config = SimpleStatelessModelConfig.model_validate(payload["model_config"])
    _validate_policy_contract(config, identity)
    model_state = _tensor_mapping(payload.get("model_state"))
    model_fingerprint = canonical_model_state_fingerprint(model_state)
    if model_fingerprint != str(payload.get("model_fingerprint")):
        raise ValueError("checkpoint model-state fingerprint mismatch")
    artifact = DurablePolicyArtifact(
        version=version,
        policy_path=resolved_path,
        policy_size_bytes=resolved_path.stat().st_size,
        policy_sha256=_file_sha256(resolved_path),
        policy_model_fingerprint=model_fingerprint,
        model_config_fingerprint=identity.model_config_fingerprint,
        exact_registry_fingerprint=identity.exact_registry_fingerprint,
        active_exact_deck_digests=identity.active_exact_deck_digests,
        input_contract_fingerprint=identity.input_contract_fingerprint,
        training_roster_fingerprint=identity.training_roster_fingerprint,
    )
    if expected_artifact is not None and artifact != expected_artifact:
        raise ValueError("policy checkpoint differs from durable artifact")
    return LoadedStatelessPolicyCheckpoint(
        artifact=artifact,
        identity=identity,
        model_config_value=config,
        model_state=model_state,
    )


def publish_stateless_checkpoint_pair(
    output_dir: Path,
    *,
    version: int,
    model: nn.Module | None,
    model_config: SimpleStatelessModelConfig,
    identity: StatelessPolicyIdentity,
    ppo_config: SimpleStatelessPpoConfig,
    optimizer_state: Mapping[str, Any],
    update_index: int,
    settled_progress: StatelessSettledTrainingProgress,
    fragment_recovery: CompactFragmentManifest,
    deck_balance_state: StatelessDeckBalanceState,
    curriculum_state: StatelessCurriculumState,
    staleness_state: StatelessStalenessState,
    curriculum_predecessor_state: StatelessCurriculumState | None,
    opponent_pool_state: OpponentPoolCheckpointState | None = None,
    policy_artifact: DurablePolicyArtifact | None = None,
    policy_publication: VerifiedPolicyPublication | None = None,
    startup_provenance: StatelessStartupProvenance | None = None,
) -> StatelessCheckpointPair:
    """Atomically publish a mutually bound exact-resume checkpoint pair."""
    if version < 0 or update_index < 0:
        raise ValueError("checkpoint version and update index must be non-negative")
    if settled_progress.rollout_window_index != version or update_index != version:
        raise ValueError(
            "checkpoint version, update index, and settled rollout window differ"
        )
    _validate_policy_contract(model_config, identity)
    if (
        fragment_recovery.static_contract_fingerprint
        != identity.fragment_static_contract_fingerprint
    ):
        raise ValueError("fragment recovery cursor differs from policy identity")
    if curriculum_state.config_fingerprint != identity.curriculum_config_fingerprint:
        raise ValueError("curriculum state differs from policy identity")
    predecessor_fingerprint: str | None = None
    if curriculum_predecessor_state is not None:
        if (
            curriculum_predecessor_state.config_fingerprint
            != identity.curriculum_config_fingerprint
        ):
            raise ValueError("curriculum predecessor differs from policy identity")
        predecessor_fingerprint = stateless_curriculum_state_fingerprint(
            curriculum_predecessor_state
        )
    if staleness_state.behavior_policy_version != version:
        raise ValueError("behavior version differs from checkpoint version")
    if policy_artifact is not None and policy_publication is not None:
        raise ValueError("policy artifact and verified publication are exclusive")
    if policy_publication is not None:
        policy_publication.verify_unchanged()
        policy_artifact = policy_publication.artifact
        _validate_policy_artifact(
            policy_artifact,
            version=version,
            model_config=model_config,
            identity=identity,
        )
        model_fingerprint = policy_artifact.policy_model_fingerprint
    elif policy_artifact is not None:
        if model is None:
            raise ValueError("unverified policy artifact requires a live model")
        policy_artifact.verify()
        _validate_policy_artifact(
            policy_artifact,
            version=version,
            model_config=model_config,
            identity=identity,
        )
        model_fingerprint = canonical_model_state_fingerprint(model)
    else:
        if model is None:
            raise ValueError("policy publication requires a live model")
        policy_publication = publish_verified_stateless_policy_checkpoint(
            output_dir,
            version=version,
            model=model,
            model_config=model_config,
            identity=identity,
        )
        policy_artifact = policy_publication.artifact
        model_fingerprint = policy_artifact.policy_model_fingerprint
    if startup_provenance is not None:
        _verify_startup_provenance(
            startup_provenance,
            checkpoint_version=version,
            expected_model_state_fingerprint=model_fingerprint,
            expected_model_config_fingerprint=identity.model_config_fingerprint,
            expected_exact_registry_fingerprint=identity.exact_registry_fingerprint,
        )

    weights_dir = output_dir / "weights"
    learner_path = (weights_dir / f"learner_state_v{version}.pt").resolve()
    pair_path = (weights_dir / f"checkpoint_pair_v{version}.json").resolve()
    policy_path = policy_artifact.policy_path.resolve()
    policy_size = policy_artifact.policy_size_bytes
    policy_sha = policy_artifact.policy_sha256
    if model_fingerprint != policy_artifact.policy_model_fingerprint:
        raise ValueError("durable policy artifact differs from checkpoint model")
    if staleness_state.behavior_policy_fingerprint != model_fingerprint:
        raise ValueError("behavior fingerprint differs from checkpoint model state")
    learner_payload = {
        "format": "simple_stateless_learner_state_v1",
        "version": version,
        "policy_path": str(policy_path),
        "policy_size_bytes": policy_size,
        "policy_sha256": policy_sha,
        "policy_model_fingerprint": model_fingerprint,
        "identity": identity.model_dump(mode="json"),
        "ppo_config": ppo_config.model_dump(mode="json"),
        "optimizer_state": dict(optimizer_state),
        "update_index": update_index,
        "settled_progress": settled_progress.model_dump(mode="json"),
        "fragment_recovery": fragment_recovery.model_dump(mode="json"),
        "deck_balance_state": deck_balance_state.model_dump(mode="json"),
        "curriculum_state": {
            "format": _COMPACT_CURRICULUM_FORMAT,
            "state_fingerprint": stateless_curriculum_state_fingerprint(
                curriculum_state
            ),
            "payload": compact_stateless_curriculum_state(curriculum_state),
        },
        "curriculum_predecessor_fingerprint": predecessor_fingerprint,
        "staleness_state": staleness_state.model_dump(mode="json"),
        "startup_provenance": (
            None
            if startup_provenance is None
            else startup_provenance.model_dump(mode="json")
        ),
    }
    if opponent_pool_state is not None:
        learner_payload["opponent_pool_state"] = opponent_pool_state.model_dump(
            mode="json"
        )
    if pair_path.exists():
        loaded = load_stateless_checkpoint_pair(
            pair_path,
            expected_identity=identity,
        )
        _require_checkpoint_retry_matches(
            loaded,
            policy_artifact=policy_artifact,
            model_config=model_config,
            ppo_config=ppo_config,
            optimizer_state=optimizer_state,
            update_index=update_index,
            settled_progress=settled_progress,
            fragment_recovery=fragment_recovery,
            deck_balance_state=deck_balance_state,
            curriculum_state=curriculum_state,
            opponent_pool_state=opponent_pool_state,
            staleness_state=staleness_state,
            curriculum_predecessor_fingerprint=predecessor_fingerprint,
            startup_provenance=startup_provenance,
        )
        _publish_latest_pointer(output_dir, loaded.pair)
        return loaded.pair
    if learner_path.exists():
        existing_learner = _mapping(
            torch.load(learner_path, map_location="cpu", weights_only=False),
            "learner payload",
        )
        if not _same_tree(existing_learner, learner_payload):
            raise FileExistsError(
                f"immutable learner sidecar differs from retry: {learner_path}"
            )
        learner_size = learner_path.stat().st_size
        learner_sha = _file_sha256(learner_path)
    else:
        learner_size, learner_sha = publish_torch_file(
            learner_path,
            learner_payload,
        )
    pair_payload = {
        "format": "exact_policy_learner_pair_v1",
        "version": version,
        "policy": {
            "path": str(policy_path),
            "size_bytes": policy_size,
            "sha256": policy_sha,
            "model_fingerprint": model_fingerprint,
        },
        "training_state": {
            "path": str(learner_path),
            "size_bytes": learner_size,
            "sha256": learner_sha,
            "policy_sha256": policy_sha,
        },
        "metadata": {
            "simple_stateless_identity": identity.model_dump(mode="json"),
            "curriculum_predecessor_fingerprint": predecessor_fingerprint,
            "startup_provenance": (
                None
                if startup_provenance is None
                else startup_provenance.model_dump(mode="json")
            ),
        },
    }
    if opponent_pool_state is not None:
        metadata = pair_payload["metadata"]
        if not isinstance(metadata, dict):
            raise AssertionError("checkpoint pair metadata is not a mapping")
        metadata["opponent_pool_state_fingerprint"] = opponent_pool_state.fingerprint
    pair_bytes = json_payload(pair_payload)
    publish_bytes_file(pair_path, pair_bytes)
    pair_sha = hashlib.sha256(pair_bytes).hexdigest()
    pair = StatelessCheckpointPair(
        version=version,
        policy_path=policy_path,
        policy_size_bytes=policy_size,
        policy_sha256=policy_sha,
        policy_model_fingerprint=model_fingerprint,
        learner_state_path=learner_path,
        learner_state_size_bytes=learner_size,
        learner_state_sha256=learner_sha,
        pair_manifest_path=pair_path,
        pair_manifest_sha256=pair_sha,
        identity=identity,
        startup_provenance=startup_provenance,
    )
    _publish_latest_pointer(output_dir, pair)
    return pair


def load_stateless_checkpoint_pair(
    manifest_path: Path,
    *,
    expected_identity: StatelessPolicyIdentity | None = None,
) -> LoadedStatelessCheckpointPair:
    """Load only a complete pair whose files and cross-bindings all match."""
    raw_manifest = manifest_path.read_bytes()
    manifest = _mapping(json.loads(raw_manifest), "pair manifest")
    if manifest.get("format") != "exact_policy_learner_pair_v1":
        raise ValueError("unsupported checkpoint pair format")
    version = int(manifest["version"])
    policy_record = _mapping(manifest.get("policy"), "policy record")
    learner_record = _mapping(manifest.get("training_state"), "learner record")
    metadata = _mapping(manifest.get("metadata"), "pair metadata")
    identity = StatelessPolicyIdentity.model_validate(
        _mapping(metadata.get("simple_stateless_identity"), "policy identity")
    )
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("checkpoint pair policy identity mismatch")

    policy_path = Path(str(policy_record["path"])).resolve()
    learner_path = Path(str(learner_record["path"])).resolve()
    policy_size = int(policy_record["size_bytes"])
    learner_size = int(learner_record["size_bytes"])
    policy_sha = str(policy_record["sha256"])
    learner_sha = str(learner_record["sha256"])
    _verify_file(policy_path, policy_size, policy_sha, "policy")
    _verify_file(learner_path, learner_size, learner_sha, "learner state")
    if str(learner_record.get("policy_sha256")) != policy_sha:
        raise ValueError("pair manifest learner binding differs from policy")

    policy_payload = _mapping(
        torch.load(policy_path, map_location="cpu", weights_only=False),
        "policy payload",
    )
    learner_payload = _mapping(
        torch.load(learner_path, map_location="cpu", weights_only=False),
        "learner payload",
    )
    _require_payload_header(
        policy_payload,
        expected_format="simple_stateless_policy_v1",
        version=version,
    )
    _require_payload_header(
        learner_payload,
        expected_format="simple_stateless_learner_state_v1",
        version=version,
    )
    if (
        StatelessPolicyIdentity.model_validate(policy_payload.get("identity"))
        != identity
        or StatelessPolicyIdentity.model_validate(learner_payload.get("identity"))
        != identity
    ):
        raise ValueError("checkpoint payload identity differs from pair manifest")
    if str(learner_payload.get("policy_sha256")) != policy_sha:
        raise ValueError("learner payload is not bound to policy bytes")
    if str(learner_payload.get("policy_model_fingerprint")) != str(
        policy_record["model_fingerprint"]
    ):
        raise ValueError("learner payload is not bound to policy state")
    config = SimpleStatelessModelConfig.model_validate(policy_payload["model_config"])
    if model_config_fingerprint(config) != identity.model_config_fingerprint:
        raise ValueError("checkpoint model config identity changed")
    model_state = _tensor_mapping(policy_payload.get("model_state"))
    model_fingerprint = canonical_model_state_fingerprint(model_state)
    if model_fingerprint != str(
        policy_payload.get("model_fingerprint")
    ) or model_fingerprint != str(policy_record["model_fingerprint"]):
        raise ValueError("checkpoint model-state fingerprint mismatch")
    startup_provenance = _load_startup_provenance(
        learner_payload=learner_payload,
        metadata=metadata,
        expected_model_state_fingerprint=model_fingerprint,
        expected_model_config_fingerprint=identity.model_config_fingerprint,
        expected_exact_registry_fingerprint=identity.exact_registry_fingerprint,
    )
    fragment = CompactFragmentManifest.model_validate(
        learner_payload["fragment_recovery"]
    )
    if (
        fragment.static_contract_fingerprint
        != identity.fragment_static_contract_fingerprint
    ):
        raise ValueError("fragment recovery cursor contract mismatch")
    deck_balance = StatelessDeckBalanceState.model_validate(
        learner_payload["deck_balance_state"]
    )
    curriculum = _checkpoint_curriculum_state(learner_payload["curriculum_state"])
    if curriculum.config_fingerprint != identity.curriculum_config_fingerprint:
        raise ValueError("curriculum resume state identity mismatch")
    opponent_pool_recorded = "opponent_pool_state" in learner_payload
    metadata_pool_recorded = "opponent_pool_state_fingerprint" in metadata
    if opponent_pool_recorded != metadata_pool_recorded:
        raise ValueError("pair opponent-pool state metadata presence differs")
    opponent_pool_state = (
        opponent_pool_checkpoint_state(learner_payload["opponent_pool_state"])
        if opponent_pool_recorded
        else None
    )
    if opponent_pool_state is not None and (
        metadata.get("opponent_pool_state_fingerprint")
        != opponent_pool_state.fingerprint
    ):
        raise ValueError("pair opponent-pool state fingerprint differs")
    predecessor_recorded = "curriculum_predecessor_fingerprint" in learner_payload
    metadata_predecessor_recorded = "curriculum_predecessor_fingerprint" in metadata
    if predecessor_recorded != metadata_predecessor_recorded:
        raise ValueError("pair curriculum transition metadata presence differs")
    predecessor_value = learner_payload.get("curriculum_predecessor_fingerprint")
    predecessor_fingerprint = (
        None if predecessor_value is None else str(predecessor_value)
    )
    metadata_predecessor = metadata.get("curriculum_predecessor_fingerprint")
    if metadata_predecessor != predecessor_value:
        raise ValueError("pair curriculum transition metadata differs from sidecar")
    if (
        predecessor_fingerprint is not None
        and _SHA256_PATTERN.fullmatch(predecessor_fingerprint) is None
    ):
        raise ValueError("curriculum predecessor fingerprint is invalid")
    staleness = StatelessStalenessState.model_validate(
        learner_payload["staleness_state"]
    )
    if (
        staleness.behavior_policy_version != version
        or staleness.behavior_policy_fingerprint != model_fingerprint
    ):
        raise ValueError("behavior/staleness state differs from policy")
    update_index = int(learner_payload["update_index"])
    settled_progress_recorded = "settled_progress" in learner_payload
    settled_progress = (
        StatelessSettledTrainingProgress.model_validate(
            learner_payload["settled_progress"]
        )
        if settled_progress_recorded
        else None
    )
    if settled_progress is not None and (
        settled_progress.rollout_window_index != version or update_index != version
    ):
        raise ValueError(
            "checkpoint version, update index, and settled rollout window differ"
        )

    pair = StatelessCheckpointPair(
        version=version,
        policy_path=policy_path,
        policy_size_bytes=policy_size,
        policy_sha256=policy_sha,
        policy_model_fingerprint=model_fingerprint,
        learner_state_path=learner_path,
        learner_state_size_bytes=learner_size,
        learner_state_sha256=learner_sha,
        pair_manifest_path=manifest_path.resolve(),
        pair_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        identity=identity,
        startup_provenance=startup_provenance,
    )
    pair.durable_artifact.verify()
    optimizer_state = learner_payload.get("optimizer_state")
    if not isinstance(optimizer_state, dict):
        raise ValueError("learner optimizer state is missing")
    return LoadedStatelessCheckpointPair(
        pair=pair,
        model_config_value=config,
        model_state=model_state,
        ppo_config=SimpleStatelessPpoConfig.model_validate(
            learner_payload["ppo_config"]
        ),
        optimizer_state=optimizer_state,
        update_index=update_index,
        settled_progress=settled_progress,
        settled_progress_recorded=settled_progress_recorded,
        fragment_recovery=fragment,
        deck_balance_state=deck_balance,
        curriculum_state=curriculum,
        opponent_pool_state=opponent_pool_state,
        opponent_pool_state_recorded=opponent_pool_recorded,
        staleness_state=staleness,
        startup_provenance=startup_provenance,
        curriculum_predecessor_fingerprint=predecessor_fingerprint,
        curriculum_predecessor_recorded=predecessor_recorded,
    )


def load_latest_stateless_checkpoint_pair(
    output_dir: Path,
    *,
    expected_identity: StatelessPolicyIdentity | None = None,
) -> LoadedStatelessCheckpointPair:
    """Resolve the moving pointer, then verify its immutable pair target."""
    latest_path = output_dir / "weights" / "latest.json"
    latest = _mapping(json.loads(latest_path.read_bytes()), "latest pointer")
    if latest.get("format") != "simple_stateless_latest_pair_v1":
        raise ValueError("unsupported latest checkpoint pointer")
    pair_path = Path(str(latest["pair_manifest_path"])).resolve()
    pair_sha = _file_sha256(pair_path)
    if pair_sha != str(latest["pair_manifest_sha256"]):
        raise ValueError("latest pointer pair manifest fingerprint mismatch")
    loaded = load_stateless_checkpoint_pair(
        pair_path,
        expected_identity=expected_identity,
    )
    if (
        loaded.pair.version != int(latest["version"])
        or loaded.pair.policy_sha256 != str(latest["policy_sha256"])
        or loaded.pair.learner_state_sha256 != str(latest["learner_state_sha256"])
    ):
        raise ValueError("latest pointer differs from its checkpoint pair")
    return loaded


def prune_stateless_checkpoint_pairs(
    output_dir: Path,
    *,
    keep_last: int | None,
    retain_every_versions: int | None,
    protected_policy_paths: Iterable[Path] = (),
) -> tuple[int, ...]:
    """Prune old exact-resume pairs while preserving live league policies.

    Pair manifests are removed before their tensor files so an interrupted
    cleanup can leave only harmless orphans, never a manifest that names a
    deleted sidecar. A later cleanup also removes those orphans.
    """
    if keep_last is None:
        if retain_every_versions is not None:
            raise ValueError("permanent checkpoint retention requires keep_last")
        return ()
    if keep_last <= 0:
        raise ValueError("keep_last must be positive")
    if retain_every_versions is not None and retain_every_versions <= 0:
        raise ValueError("retain_every_versions must be positive when set")

    weights_dir = (output_dir / "weights").resolve()
    if not weights_dir.is_dir():
        return ()
    latest_version = _latest_pointer_version(weights_dir / "latest.json")
    protected_policies = {path.resolve() for path in protected_policy_paths}
    artifacts = _stateless_checkpoint_artifacts(weights_dir)
    complete_versions = sorted(
        (
            version
            for version, paths in artifacts.items()
            if all(paths[kind].is_file() for kind in ("manifest", "learner", "policy"))
        ),
        reverse=True,
    )
    retained = set(complete_versions[:keep_last])
    retained.add(latest_version)
    if retain_every_versions is not None:
        retained.update(
            version
            for version in complete_versions
            if version % retain_every_versions == 0
        )

    pruned: list[int] = []
    for version in sorted(artifacts):
        if version in retained:
            continue
        paths = artifacts[version]
        paths["manifest"].unlink(missing_ok=True)
        paths["learner"].unlink(missing_ok=True)
        if paths["policy"].resolve() not in protected_policies:
            paths["policy"].unlink(missing_ok=True)
        pruned.append(version)
    if pruned:
        fsync_directory(weights_dir)
    return tuple(pruned)


def reconcile_stateless_resume_artifacts(
    output_dir: Path,
    *,
    selected: LoadedStatelessCheckpointPair,
) -> tuple[int, ...]:
    """Remove only unauthenticated future files before exact replay.

    A complete pair newer than the explicitly selected resume source is settled
    progress, so callers must select it deliberately instead of silently
    rolling it forward or deleting it. A published manifest that cannot be
    validated is likewise preserved for diagnosis. Only policy/learner files
    that have no manifest can be crash leftovers from the ordered pair
    publication and are safe to remove, provided the selected curriculum does
    not reference their policy path.
    """
    weights_dir = (output_dir / "weights").resolve()
    if not weights_dir.is_dir():
        return ()
    artifacts = _stateless_checkpoint_artifacts(weights_dir)
    future = tuple(
        (version, artifacts[version])
        for version in sorted(artifacts)
        if version > selected.pair.version
    )
    if not future:
        return ()

    protected_policies = {
        member.policy_path.resolve() for member in selected.curriculum_state.members
    }
    orphan_versions: list[int] = []
    orphan_learners: list[Path] = []
    orphan_policies: list[Path] = []
    for version, paths in future:
        manifest_path = paths["manifest"]
        if manifest_path.is_file():
            try:
                load_stateless_checkpoint_pair(
                    manifest_path,
                    expected_identity=selected.pair.identity,
                )
            except Exception as error:
                raise ValueError(
                    "future checkpoint manifest is present but invalid: "
                    f"{manifest_path}"
                ) from error
            raise RuntimeError(
                "a complete checkpoint successor exists beyond the explicitly "
                f"selected resume pair: {manifest_path}; resume that pair "
                "explicitly or use a different output directory"
            )
        learner_path = paths["learner"]
        policy_path = paths["policy"]
        if policy_path.is_file() and policy_path.resolve() in protected_policies:
            raise RuntimeError(
                "selected curriculum references a manifestless future policy: "
                f"{policy_path}"
            )
        orphan_versions.append(version)
        if learner_path.is_file():
            orphan_learners.append(learner_path)
        if policy_path.is_file():
            orphan_policies.append(policy_path)

    for path in orphan_learners:
        path.unlink()
    for path in orphan_policies:
        path.unlink()
    if orphan_versions:
        fsync_directory(weights_dir)
    return tuple(orphan_versions)


def _latest_pointer_version(path: Path) -> int:
    """Read the version that must never be pruned."""
    latest = _mapping(json.loads(path.read_bytes()), "latest pointer")
    if latest.get("format") != "simple_stateless_latest_pair_v1":
        raise ValueError("unsupported latest checkpoint pointer")
    version = int(latest["version"])
    if version < 0:
        raise ValueError("latest checkpoint version must be non-negative")
    return version


def _stateless_checkpoint_artifacts(
    weights_dir: Path,
) -> dict[int, dict[str, Path]]:
    """Index owned checkpoint filenames, including crash-leftover orphans."""
    artifacts: dict[int, dict[str, Path]] = {}
    patterns = {
        "manifest": ("checkpoint_pair_v*.json", pair_manifest_version),
        "learner": (
            "learner_state_v*.pt",
            lambda path: _numbered_artifact_version(path, "learner_state_v"),
        ),
        "policy": (
            "policy_v*.pt",
            lambda path: _numbered_artifact_version(path, "policy_v"),
        ),
    }
    for pattern, parser in patterns.values():
        for path in weights_dir.glob(pattern):
            version = parser(path)
            if version is None:
                continue
            artifacts.setdefault(
                version,
                {
                    "manifest": weights_dir / f"checkpoint_pair_v{version}.json",
                    "learner": weights_dir / f"learner_state_v{version}.pt",
                    "policy": weights_dir / f"policy_v{version}.pt",
                },
            )
    return artifacts


def _numbered_artifact_version(path: Path, prefix: str) -> int | None:
    raw = path.stem.removeprefix(prefix)
    return int(raw) if raw.isdigit() else None


def _require_checkpoint_retry_matches(
    loaded: LoadedStatelessCheckpointPair,
    *,
    policy_artifact: DurablePolicyArtifact,
    model_config: SimpleStatelessModelConfig,
    ppo_config: SimpleStatelessPpoConfig,
    optimizer_state: Mapping[str, Any],
    update_index: int,
    settled_progress: StatelessSettledTrainingProgress,
    fragment_recovery: CompactFragmentManifest,
    deck_balance_state: StatelessDeckBalanceState,
    curriculum_state: StatelessCurriculumState,
    opponent_pool_state: OpponentPoolCheckpointState | None,
    staleness_state: StatelessStalenessState,
    curriculum_predecessor_fingerprint: str | None,
    startup_provenance: StatelessStartupProvenance | None,
) -> None:
    matches = (
        loaded.pair.durable_artifact == policy_artifact
        and loaded.model_config_value == model_config
        and loaded.ppo_config == ppo_config
        and loaded.update_index == update_index
        and loaded.settled_progress == settled_progress
        and loaded.settled_progress_recorded
        and loaded.fragment_recovery == fragment_recovery
        and loaded.deck_balance_state == deck_balance_state
        and loaded.curriculum_state == curriculum_state
        and loaded.opponent_pool_state == opponent_pool_state
        and loaded.opponent_pool_state_recorded == (opponent_pool_state is not None)
        and loaded.staleness_state == staleness_state
        and loaded.curriculum_predecessor_fingerprint
        == curriculum_predecessor_fingerprint
        and loaded.curriculum_predecessor_recorded
        and loaded.startup_provenance == startup_provenance
        and _same_tree(loaded.optimizer_state, optimizer_state)
    )
    if not matches:
        raise FileExistsError(
            f"immutable checkpoint pair differs from retry: "
            f"{loaded.pair.pair_manifest_path}"
        )


def _publish_latest_pointer(
    output_dir: Path,
    pair: StatelessCheckpointPair,
) -> None:
    latest_path = output_dir / "weights" / "latest.json"
    latest = {
        "format": "simple_stateless_latest_pair_v1",
        "version": pair.version,
        "pair_manifest_path": str(pair.pair_manifest_path),
        "pair_manifest_sha256": pair.pair_manifest_sha256,
        "policy_sha256": pair.policy_sha256,
        "learner_state_sha256": pair.learner_state_sha256,
    }
    if latest_path.exists():
        current = _mapping(json.loads(latest_path.read_bytes()), "latest pointer")
        current_version = int(current["version"])
        if current_version > pair.version:
            raise RuntimeError("checkpoint retry would regress the latest pointer")
        if current_version == pair.version and current != latest:
            raise FileExistsError(
                "latest pointer already names a different immutable pair version"
            )
    atomic_write_bytes(
        latest_path,
        json_payload(latest),
        overwrite=True,
    )


def _load_startup_provenance(
    *,
    learner_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    expected_model_state_fingerprint: str,
    expected_model_config_fingerprint: str,
    expected_exact_registry_fingerprint: str,
) -> StatelessStartupProvenance | None:
    """Load an optional provenance root only when both pair members bind it."""
    learner_recorded = "startup_provenance" in learner_payload
    metadata_recorded = "startup_provenance" in metadata
    if learner_recorded != metadata_recorded:
        raise ValueError(
            "pair startup provenance presence differs from learner sidecar"
        )
    if not learner_recorded:
        return None
    learner_value = learner_payload.get("startup_provenance")
    metadata_value = metadata.get("startup_provenance")
    if learner_value is None or metadata_value is None:
        if learner_value is not None or metadata_value is not None:
            raise ValueError("pair startup provenance differs from learner sidecar")
        return None
    learner_provenance = StatelessStartupProvenance.model_validate(learner_value)
    metadata_provenance = StatelessStartupProvenance.model_validate(metadata_value)
    if learner_provenance != metadata_provenance:
        raise ValueError("pair startup provenance differs from learner sidecar")
    _verify_startup_provenance(
        learner_provenance,
        checkpoint_version=int(learner_payload["version"]),
        expected_model_state_fingerprint=expected_model_state_fingerprint,
        expected_model_config_fingerprint=expected_model_config_fingerprint,
        expected_exact_registry_fingerprint=expected_exact_registry_fingerprint,
    )
    return learner_provenance


def _verify_startup_provenance(
    provenance: StatelessStartupProvenance,
    *,
    checkpoint_version: int,
    expected_model_state_fingerprint: str,
    expected_model_config_fingerprint: str,
    expected_exact_registry_fingerprint: str,
) -> None:
    """Verify report bytes, topology identities, and the initial policy state."""
    if (
        provenance.target_model_config_fingerprint != expected_model_config_fingerprint
        or provenance.target_exact_registry_fingerprint
        != expected_exact_registry_fingerprint
    ):
        raise ValueError("startup provenance target topology differs from checkpoint")
    reports: dict[str, Mapping[str, Any]] = {}
    for artifact in provenance.reports:
        _verify_file(
            artifact.path,
            artifact.size_bytes,
            artifact.sha256,
            f"{artifact.kind} startup report",
        )
        try:
            reports[artifact.kind] = _mapping(
                json.loads(artifact.path.read_bytes()),
                f"{artifact.kind} startup report",
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(
                f"{artifact.kind} startup report is invalid JSON"
            ) from error
    if provenance.operation == "bc_overlay":
        _verify_bc_overlay_startup_report(provenance, reports["bc_overlay_audit"])
    elif provenance.operation == "supervised_initialization":
        _verify_supervised_initialization_report(
            provenance,
            reports["supervised_initialization"],
        )
    elif provenance.operation == "weights_only_warm_start":
        _verify_weights_only_warm_start_report(
            provenance,
            reports["weights_only_warm_start"],
        )
    elif provenance.operation == "identity_materialization":
        _verify_identity_materialization_report(
            provenance,
            reports["transition_source"],
        )
    else:
        _verify_topology_materialization_reports(
            provenance,
            topology=reports["topology_transition"],
            transition=reports["transition_source"],
        )
    if checkpoint_version < provenance.startup_pair_version:
        raise ValueError("checkpoint predates its startup provenance")
    if (
        checkpoint_version == provenance.startup_pair_version
        and provenance.target_model_state_fingerprint
        != expected_model_state_fingerprint
    ):
        raise ValueError(
            "startup provenance target model differs from checkpoint policy"
        )


def _verify_bc_overlay_startup_report(
    provenance: StatelessStartupProvenance,
    report: Mapping[str, Any],
) -> None:
    """Cross-bind the BC report, its declaration, and the overlay policy."""
    if report.get("format") != "simple-stateless-bc-overlay-audit-v1":
        raise ValueError("BC overlay startup report format is invalid")
    declaration = _mapping(
        report.get("declaration"),
        "startup provenance declaration",
    )
    model = _mapping(report.get("model"), "startup provenance model audit")
    expected = (
        provenance.source_pair_manifest_sha256,
        provenance.supervised_manifest_sha256,
        provenance.target_deck_digest,
    )
    bindings = (
        (
            report.get("source_pair_manifest_sha256"),
            report.get("supervised_manifest_sha256"),
            model.get("target_deck_digest"),
        ),
        (
            declaration.get("source_pair_manifest_sha256"),
            declaration.get("supervised_manifest_sha256"),
            declaration.get("target_deck_digest"),
        ),
    )
    if any(binding != expected for binding in bindings):
        raise ValueError("startup provenance report bindings differ from checkpoint")
    if (
        model.get("target_model_state_fingerprint")
        != provenance.target_model_state_fingerprint
        or declaration.get("supervised_model_state_fingerprint")
        != provenance.target_model_state_fingerprint
        or declaration.get("source_pair_version") != provenance.startup_pair_version
    ):
        raise ValueError("startup provenance report model binding is invalid")


def _verify_supervised_initialization_report(
    provenance: StatelessStartupProvenance,
    report: Mapping[str, Any],
) -> None:
    """Cross-bind a selected full-model artifact to a fresh pair."""
    report_format = report.get("format")
    if report_format not in {
        "simple-stateless-supervised-startup-audit-v1",
        "simple-stateless-supervised-startup-audit-v2",
    }:
        raise ValueError("supervised startup report format is invalid")
    declaration = _mapping(
        report.get("declaration"),
        "supervised startup declaration",
    )
    expected_artifact = (
        provenance.supervised_manifest_sha256,
        provenance.supervised_policy_sha256,
        provenance.target_model_state_fingerprint,
        provenance.supervised_dataset_manifest_sha256,
        provenance.supervised_dataset_fingerprint,
    )
    declared_artifact = (
        declaration.get("manifest_sha256"),
        declaration.get("policy_sha256"),
        declaration.get("model_state_fingerprint"),
        declaration.get("dataset_manifest_sha256"),
        declaration.get("dataset_fingerprint"),
    )
    if declared_artifact != expected_artifact:
        raise ValueError("supervised startup report bindings differ from checkpoint")
    selection_fingerprint = report.get(
        "selection_fingerprint",
        report.get("selection_split_assignment_fingerprint"),
    )
    common_valid = (
        declaration.get("trainable_scope") == "full_model"
        and report.get("model_config_fingerprint")
        == provenance.target_model_config_fingerprint
        and report.get("exact_registry_fingerprint")
        == provenance.target_exact_registry_fingerprint
        and report.get("selection_optimizer_step") is not None
        and selection_fingerprint is not None
    )
    if report_format == "simple-stateless-supervised-startup-audit-v1":
        provenance_valid = (
            report.get("selection_basis") in {None, "validation"}
            and report.get("initialization_source_pair_manifest_sha256") is not None
            and report.get("initialization_source_policy_sha256") is not None
        )
    else:
        event_contract = report.get("event_contract_fingerprint")
        sequence_contract = report.get("sequence_contract_fingerprint")
        source_pair = report.get("initialization_source_pair_manifest_sha256")
        source_policy = report.get("initialization_source_policy_sha256")
        random_seed = report.get("initialization_random_seed")
        initial_model = report.get("initialization_model_state_fingerprint")
        selection_basis = report.get("selection_basis")
        pair_initial_model_valid = initial_model is None or (
            isinstance(initial_model, str)
            and _SHA256_PATTERN.fullmatch(initial_model) is not None
        )
        pair_initialized = (
            selection_basis in {"validation", "final_epoch"}
            and isinstance(source_pair, str)
            and len(source_pair) == 64
            and isinstance(source_policy, str)
            and len(source_policy) == 64
            and random_seed is None
            # A topology-migrated RL-pair initialization records the migrated
            # pre-BC state. Same-topology pair initialization leaves it null.
            and pair_initial_model_valid
            and (
                selection_basis != "final_epoch"
                or selection_fingerprint == provenance.target_model_state_fingerprint
            )
        )
        random_initialized = (
            report.get("selection_basis") in {"train_monitor", "validation"}
            and source_pair is None
            and source_policy is None
            and isinstance(random_seed, int)
            and not isinstance(random_seed, bool)
            and random_seed >= 0
            and isinstance(initial_model, str)
            and len(initial_model) == 64
        )
        provenance_valid = (
            (pair_initialized or random_initialized)
            and isinstance(event_contract, str)
            and len(event_contract) == 64
            and event_contract == declaration.get("event_contract_fingerprint")
            and isinstance(sequence_contract, str)
            and len(sequence_contract) == 64
            and sequence_contract == declaration.get("sequence_contract_fingerprint")
        )
    if not common_valid or not provenance_valid:
        raise ValueError("supervised startup report model binding is invalid")


def _verify_weights_only_warm_start_report(
    provenance: StatelessStartupProvenance,
    report: Mapping[str, Any],
) -> None:
    """Bind a model-only branch to its source pair and explicit resets."""
    expected_reset_state = (
        "adam",
        "learning_rate_schedule",
        "fragment_replay",
        "curriculum_statistics",
        "deck_balance",
        "opponent_pool_statistics",
        "optimizer_step_cursor",
        "rollout_window_cursor",
    )
    reset_state = report.get("reset_state")
    if not isinstance(reset_state, list):
        raise ValueError("weights-only warm-start reset inventory is invalid")
    target_config = report.get("target_resolved_config_fingerprint")
    source_policy = report.get("source_policy_sha256")
    if (
        report.get("format") != "simple-stateless-weights-only-warm-start-v1"
        or report.get("source_pair_manifest_sha256")
        != provenance.source_pair_manifest_sha256
        or report.get("source_model_state_fingerprint")
        != provenance.target_model_state_fingerprint
        or not isinstance(report.get("source_pair_version"), int)
        or int(report["source_pair_version"]) < 0
        or not isinstance(source_policy, str)
        or _SHA256_PATTERN.fullmatch(source_policy) is None
        or not isinstance(target_config, str)
        or _SHA256_PATTERN.fullmatch(target_config) is None
        or tuple(reset_state) != expected_reset_state
    ):
        raise ValueError(
            "weights-only warm-start report bindings differ from checkpoint"
        )


def _verify_identity_materialization_report(
    provenance: StatelessStartupProvenance,
    transition: Mapping[str, Any],
) -> None:
    """Require an identity-only publish to preserve every training contract."""
    if transition.get("format") != "simple_stateless_training_transition_v3":
        raise ValueError("identity materialization transition report format is invalid")
    source_identity = dict(
        _mapping(transition.get("source_identity"), "transition source identity")
    )
    target_identity = dict(
        _mapping(transition.get("target_identity"), "transition target identity")
    )
    for field in (
        "resolved_config_fingerprint",
        "fragment_static_contract_fingerprint",
    ):
        source_identity.pop(field, None)
        target_identity.pop(field, None)
    source_progress = transition.get("source_settled_progress")
    target_progress = transition.get("target_initial_settled_progress")
    if (
        transition.get("controller_state_mode") != "preserved_settled"
        or transition.get("source_pair_manifest_sha256")
        != provenance.source_pair_manifest_sha256
        or transition.get("version") != provenance.startup_pair_version
        or target_identity.get("model_config_fingerprint")
        != provenance.target_model_config_fingerprint
        or target_identity.get("exact_registry_fingerprint")
        != provenance.target_exact_registry_fingerprint
        or source_identity != target_identity
        or source_progress is None
        or source_progress != target_progress
        or transition.get("source_ppo_config_fingerprint")
        != transition.get("target_ppo_config_fingerprint")
        or transition.get("source_curriculum_state_fingerprint")
        != transition.get("target_curriculum_state_fingerprint")
        or "source_opponent_pool_state_fingerprint" not in transition
        or "target_opponent_pool_state_fingerprint" not in transition
        or transition.get("source_opponent_pool_state_fingerprint")
        != transition.get("target_opponent_pool_state_fingerprint")
        or any(
            transition.get(field) is not None
            for field in (
                "registry_transition",
                "topology_transition",
                "sequence_context_transition",
                "public_catalog_transition",
            )
        )
    ):
        raise ValueError("identity materialization startup report binding mismatch")


def _verify_topology_materialization_reports(
    provenance: StatelessStartupProvenance,
    *,
    topology: Mapping[str, Any],
    transition: Mapping[str, Any],
) -> None:
    """Cross-bind both v1-to-v2 reports to the materialized target policy."""
    if (
        topology.get("format") != "simple-stateless-v1-to-v2-audit-v1"
        or transition.get("format") != "simple_stateless_training_transition_v3"
    ):
        raise ValueError("topology materialization startup report format is invalid")
    declaration = _mapping(
        topology.get("declaration"),
        "topology transition declaration",
    )
    plan = _mapping(topology.get("plan"), "topology transition plan")
    model = _mapping(topology.get("model"), "topology transition model audit")
    source_identity = _mapping(
        transition.get("source_identity"),
        "transition source identity",
    )
    target_identity = _mapping(
        transition.get("target_identity"),
        "transition target identity",
    )
    transition_topology = _mapping(
        transition.get("topology_transition"),
        "transition topology summary",
    )
    source_registry = declaration.get("source_registry_sha256")
    expected = (
        provenance.source_pair_manifest_sha256,
        provenance.target_model_config_fingerprint,
        provenance.target_exact_registry_fingerprint,
        provenance.target_model_state_fingerprint,
        provenance.startup_pair_version,
    )
    observed = (
        declaration.get("source_pair_manifest_sha256"),
        declaration.get("target_model_config_fingerprint"),
        declaration.get("target_registry_sha256"),
        model.get("target_state_fingerprint"),
        transition.get("version"),
    )
    transition_observed = (
        transition.get("source_pair_manifest_sha256"),
        target_identity.get("model_config_fingerprint"),
        target_identity.get("exact_registry_fingerprint"),
        model.get("target_state_fingerprint"),
        transition.get("version"),
    )
    if observed != expected or transition_observed != expected:
        raise ValueError(
            "topology materialization reports differ from checkpoint provenance"
        )
    if (
        plan.get("source_registry_sha256") != source_registry
        or plan.get("target_registry_sha256")
        != provenance.target_exact_registry_fingerprint
        or transition_topology.get("source_registry_sha256") != source_registry
        or transition_topology.get("target_registry_sha256")
        != provenance.target_exact_registry_fingerprint
        or source_identity.get("exact_registry_fingerprint") != source_registry
    ):
        raise ValueError("topology materialization registry chain is invalid")


def _same_tree(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        if not isinstance(left, Tensor) or not isinstance(right, Tensor):
            return False
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_same_tree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(
            right,
            (list, tuple),
        ):
            return False
        return len(left) == len(right) and all(
            _same_tree(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def _validate_policy_contract(
    model_config: SimpleStatelessModelConfig,
    identity: StatelessPolicyIdentity,
) -> None:
    if model_config_fingerprint(model_config) != identity.model_config_fingerprint:
        raise ValueError("policy identity differs from the resolved model config")
    if model_config.resolved_registry_sha256 != identity.exact_registry_fingerprint:
        raise ValueError("policy identity differs from the exact registry")
    active_decks = tuple(
        sorted(route.deck_digest for route in model_config.exact_routes)
    )
    if active_decks != identity.active_exact_deck_digests:
        raise ValueError("policy identity differs from active exact routes")


def _validate_policy_artifact(
    artifact: DurablePolicyArtifact,
    *,
    version: int,
    model_config: SimpleStatelessModelConfig,
    identity: StatelessPolicyIdentity,
) -> None:
    if artifact.version != version:
        raise ValueError("durable policy version differs from checkpoint version")
    expected = (
        identity.model_config_fingerprint,
        identity.exact_registry_fingerprint,
        identity.active_exact_deck_digests,
        identity.input_contract_fingerprint,
        identity.training_roster_fingerprint,
    )
    actual = (
        artifact.model_config_fingerprint,
        artifact.exact_registry_fingerprint,
        artifact.active_exact_deck_digests,
        artifact.input_contract_fingerprint,
        artifact.training_roster_fingerprint,
    )
    if actual != expected:
        raise ValueError("durable policy artifact differs from checkpoint identity")
    if model_config_fingerprint(model_config) != artifact.model_config_fingerprint:
        raise ValueError("durable policy artifact differs from model config")


def _frozen_cpu_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        name: tensor.detach().to(device="cpu", copy=True).contiguous()
        for name, tensor in state.items()
    }


def _tensor_mapping(value: Any) -> dict[str, Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("policy state is missing")
    result: dict[str, Tensor] = {}
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise ValueError("policy state must map names to tensors")
        result[name] = tensor
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _checkpoint_curriculum_state(value: Any) -> StatelessCurriculumState:
    """Decode both legacy inline and compact curriculum checkpoint fields."""
    if not isinstance(value, Mapping) or value.get("format") != (
        _COMPACT_CURRICULUM_FORMAT
    ):
        return StatelessCurriculumState.model_validate(value)
    payload = value.get("payload")
    if not isinstance(payload, bytes):
        raise ValueError("compact checkpoint curriculum payload is missing")
    state = load_compact_stateless_curriculum_state(payload)
    if stateless_curriculum_state_fingerprint(state) != str(
        value.get("state_fingerprint", "")
    ):
        raise ValueError("compact checkpoint curriculum fingerprint mismatch")
    return state


def _require_payload_header(
    payload: Mapping[str, Any],
    *,
    expected_format: str,
    version: int,
) -> None:
    if payload.get("format") != expected_format or int(payload["version"]) != version:
        raise ValueError("checkpoint payload header mismatch")


def _verify_file(path: Path, size_bytes: int, sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if path.stat().st_size != size_bytes or _file_sha256(path) != sha256:
        raise ValueError(f"{label} content identity mismatch")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "LoadedStatelessCheckpointPair",
    "LoadedStatelessPolicyCheckpoint",
    "AsyncStatelessCheckpointPairPublisher",
    "PublishedStatelessCheckpointPair",
    "StatelessCheckpointPair",
    "StatelessPolicyIdentity",
    "StatelessSettledTrainingProgress",
    "StatelessStalenessState",
    "StatelessStartupProvenance",
    "StatelessStartupReport",
    "load_latest_stateless_checkpoint_pair",
    "load_stateless_checkpoint_pair",
    "load_stateless_policy_checkpoint",
    "prune_stateless_checkpoint_pairs",
    "publish_stateless_checkpoint_pair",
    "publish_stateless_policy_checkpoint",
    "publish_verified_stateless_policy_checkpoint",
    "reconcile_stateless_resume_artifacts",
]
