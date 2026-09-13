"""Strict identities for native rollout delivery and distributed windows."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ptcg_rl.model.simple_stateless import SimpleStatelessModelConfig
from ptcg_rl.rl.model_compatibility import model_config_fingerprint
from ptcg_rl.rl.stateless_collection import (
    StatelessAssignedGame,
    StatelessCollectionReport,
    StatelessGameOutcome,
)
from ptcg_rl.rl.stateless_curriculum import CurriculumAssignment, PfspMember
from ptcg_rl.rl.stateless_deck_balance import DeckSeatAssignment
from ptcg_rl.rl.stateless_training_config import (
    MAX_NATIVE_ENGINE_SHARDS_PER_PROCESS,
    MAX_NATIVE_PROCESS_WORKERS,
    StatelessHistoricalAnchorConfig,
)

_SHA256_LENGTH = 64
_MAX_ID_LENGTH = 256
_MAX_MODEL_CONFIG_JSON_BYTES = 1 << 20
_ASSIGNMENT_DOMAIN = b"ptcg-rl/native-distributed-assignments/v1\x00"
_ASSIGNMENT_PLAN_DOMAIN = b"ptcg-rl/native-distributed-assignment-plan/v2\x00"
_OUTCOME_DOMAIN = b"ptcg-rl/native-distributed-outcomes/v1\x00"


class _StrictIdentity(BaseModel):
    """Common fail-closed Pydantic behavior for wire identities."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class NativeRolloutWorkerIdentity(_StrictIdentity):
    """One logical worker process and its immutable native runtime."""

    worker_id: str
    session_id: str
    runtime_fingerprint: str
    source_git_commit: str
    source_snapshot_fingerprint: str
    native_library_fingerprint: str
    native_abi_version: int = Field(ge=1)
    engine_fact_contract_fingerprint: str | None
    feature_schema_fingerprint: str
    card_catalog_fingerprint: str
    static_features_fingerprint: str
    exact_registry_fingerprint: str
    scripted_opponents_fingerprint: str
    historical_opponents_fingerprint: str
    model_config_fingerprint: str
    resolved_config_fingerprint: str

    @field_validator("worker_id", "session_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Reject ambiguous or unbounded identifiers."""
        return _validate_id(value)

    @field_validator("source_git_commit")
    @classmethod
    def valid_source_commit(cls, value: str) -> str:
        """Record provenance without making commit equality a compatibility gate."""
        normalized = value.strip().lower()
        if len(normalized) not in (40, 64) or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("native worker source commit must be a full Git identity")
        return normalized

    @field_validator(
        "runtime_fingerprint",
        "source_snapshot_fingerprint",
        "native_library_fingerprint",
        "feature_schema_fingerprint",
        "card_catalog_fingerprint",
        "static_features_fingerprint",
        "exact_registry_fingerprint",
        "scripted_opponents_fingerprint",
        "historical_opponents_fingerprint",
        "model_config_fingerprint",
        "resolved_config_fingerprint",
    )
    @classmethod
    def valid_runtime_fingerprint(cls, value: str) -> str:
        """Bind the worker to one exact executable/runtime inventory."""
        return _validate_sha256(value)

    @field_validator("engine_fact_contract_fingerprint")
    @classmethod
    def valid_optional_fact_fingerprint(cls, value: str | None) -> str | None:
        """Bind exact-fact semantics when the model consumes native facts."""
        return None if value is None else _validate_sha256(value)


class NativeRolloutWindowIdentity(_StrictIdentity):
    """One learner collection window bound to one behavior publication."""

    run_id: str
    window_id: str
    sequence_id: int = Field(ge=0)
    behavior_policy_version: int = Field(ge=0)
    behavior_policy_fingerprint: str
    static_contract_fingerprint: str
    resolved_config_fingerprint: str

    @field_validator("run_id", "window_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Reject ambiguous or unbounded identifiers."""
        return _validate_id(value)

    @field_validator(
        "behavior_policy_fingerprint",
        "static_contract_fingerprint",
        "resolved_config_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require exact lower-case SHA-256 artifact identities."""
        return _validate_sha256(value)


class NativeRolloutLeaseIdentity(_StrictIdentity):
    """A bounded worker authorization for one immutable rollout window."""

    lease_id: str
    worker: NativeRolloutWorkerIdentity
    window: NativeRolloutWindowIdentity
    sequence_id: int = Field(ge=0)
    issued_at_unix_ns: int = Field(ge=0)
    expires_at_unix_ns: int = Field(gt=0)

    @field_validator("lease_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Reject ambiguous or unbounded identifiers."""
        return _validate_id(value)

    @model_validator(mode="after")
    def increasing_deadline(self) -> NativeRolloutLeaseIdentity:
        """A lease must have a non-empty validity interval."""
        if self.expires_at_unix_ns <= self.issued_at_unix_ns:
            raise ValueError("native rollout lease expiry must follow issue time")
        return self


class NativeRolloutPartIdentity(_StrictIdentity):
    """Identity and exact row counts for one memory-only trajectory part."""

    part_id: str
    lease: NativeRolloutLeaseIdentity
    sequence_id: int = Field(ge=0)
    fragment_count: int = Field(ge=1)
    decision_count: int = Field(ge=1)

    @field_validator("part_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Reject ambiguous or unbounded identifiers."""
        return _validate_id(value)


class NativeCollectionCapacityTier(_StrictIdentity):
    """One worker-advertised, independently validated collection geometry."""

    tier_id: str
    concurrent_games: int = Field(gt=0)
    native_arena_capacity: int = Field(gt=0)
    native_engine_shards: int = Field(
        ge=1,
        le=MAX_NATIVE_ENGINE_SHARDS_PER_PROCESS * MAX_NATIVE_PROCESS_WORKERS,
    )
    native_engine_fact_workers: int = Field(gt=0)
    native_policy_cohort_slots: int | None = Field(default=None, gt=0)
    native_policy_group_bank_limit: int = Field(default=2, ge=1, le=4)
    native_process_workers: int = Field(
        default=1,
        ge=1,
        le=MAX_NATIVE_PROCESS_WORKERS,
    )
    estimated_trainable_decisions: int = Field(gt=0)

    @field_validator("tier_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Reject ambiguous capacity aliases."""
        return _validate_id(value)

    @model_validator(mode="after")
    def coherent_native_geometry(self) -> NativeCollectionCapacityTier:
        """Require a banked-native geometry that can execute as one shard."""
        if (
            self.native_arena_capacity > self.concurrent_games
            or self.native_arena_capacity < self.native_engine_shards
            or self.native_arena_capacity < self.native_process_workers
            or (
                math.ceil(self.native_engine_shards / self.native_process_workers)
                > MAX_NATIVE_ENGINE_SHARDS_PER_PROCESS
            )
            or (self.native_process_workers == 1 and self.native_engine_shards % 2 != 0)
        ):
            raise ValueError(
                "native distributed tiers require a bounded live arena and an "
                "even engine-shard ring"
            )
        if (
            self.native_policy_cohort_slots is not None
            and self.native_policy_cohort_slots > self.native_arena_capacity
        ):
            raise ValueError("native policy cohort exceeds live arena capacity")
        return self


class NativeCollectionWorkerManifest(_StrictIdentity):
    """One connected worker and its complete runtime/topology inventory."""

    identity: NativeRolloutWorkerIdentity
    worker_profile: str
    cuda_device_name: str
    cuda_device_uuid: str
    cuda_total_memory_bytes: int = Field(gt=0)
    cuda_compute_capability: str
    torch_version: str
    torch_cuda_version: str
    capacity_tiers: tuple[NativeCollectionCapacityTier, ...]

    @field_validator(
        "worker_profile",
        "cuda_device_name",
        "cuda_device_uuid",
        "cuda_compute_capability",
        "torch_version",
        "torch_cuda_version",
    )
    @classmethod
    def valid_text(cls, value: str) -> str:
        """Reject empty or ambiguous runtime labels."""
        return _validate_id(value)

    @model_validator(mode="after")
    def unique_capacity_tiers(self) -> NativeCollectionWorkerManifest:
        """Each advertised geometry must have a unique stable alias."""
        tier_ids = tuple(tier.tier_id for tier in self.capacity_tiers)
        if not tier_ids or len(tier_ids) != len(set(tier_ids)):
            raise ValueError("worker capacity tiers must be non-empty and unique")
        return self


class NativeBfloat16TensorSpec(_StrictIdentity):
    """One tensor in canonical BF16 rollout-artifact wire order."""

    name: str
    logical_dtype: str
    wire_dtype: str
    shape: tuple[int, ...]
    nbytes: int = Field(ge=0)
    sha256: str

    @field_validator("name", "logical_dtype", "wire_dtype")
    @classmethod
    def valid_name(cls, value: str) -> str:
        """Reject empty or control-bearing tensor metadata."""
        return _validate_id(value)

    @field_validator("sha256")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require an exact payload fingerprint."""
        return _validate_sha256(value)

    @field_validator("shape")
    @classmethod
    def valid_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reject negative dimensions."""
        if any(dimension < 0 for dimension in value):
            raise ValueError("rollout artifact tensor shape is invalid")
        return value


class NativeBfloat16ArtifactManifest(_StrictIdentity):
    """Dual identity for one FP32 source and its exact BF16 wire artifact."""

    artifact_id: str
    kind: Literal["current", "past_self"]
    source_policy_version: int = Field(ge=0)
    source_fp32_fingerprint: str
    wire_bf16_fingerprint: str
    model_config_fingerprint: str
    exact_registry_fingerprint: str
    tensors: tuple[NativeBfloat16TensorSpec, ...]

    @field_validator("artifact_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Reject ambiguous artifact aliases."""
        return _validate_id(value)

    @field_validator(
        "source_fp32_fingerprint",
        "wire_bf16_fingerprint",
        "model_config_fingerprint",
        "exact_registry_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require immutable full artifact identities."""
        return _validate_sha256(value)

    @model_validator(mode="after")
    def canonical_tensors(self) -> NativeBfloat16ArtifactManifest:
        """Require complete, canonical tensor order."""
        names = tuple(tensor.name for tensor in self.tensors)
        if not names or names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("rollout artifact tensors must be sorted and unique")
        return self


class NativeArtifactModelConfig(_StrictIdentity):
    """Portable model topology paired with one active rollout artifact."""

    artifact_id: str
    model_config_json: str = Field(
        min_length=2,
        max_length=_MAX_MODEL_CONFIG_JSON_BYTES,
    )
    model_config_fingerprint: str
    exact_registry_fingerprint: str

    @field_validator("artifact_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Reject ambiguous artifact aliases."""
        return _validate_id(value)

    @field_validator(
        "model_config_fingerprint",
        "exact_registry_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require immutable topology identities."""
        return _validate_sha256(value)

    @model_validator(mode="after")
    def canonical_model_config(self) -> NativeArtifactModelConfig:
        """Bind canonical portable JSON to its path-independent identity."""
        config = self.to_model_config()
        if self.model_config_json != _model_config_json(config):
            raise ValueError("artifact model configuration JSON is not canonical")
        if model_config_fingerprint(config) != self.model_config_fingerprint:
            raise ValueError("artifact model configuration fingerprint differs")
        return self

    @classmethod
    def from_model_config(
        cls,
        *,
        artifact_id: str,
        model_config: SimpleStatelessModelConfig,
        exact_registry_fingerprint: str,
    ) -> NativeArtifactModelConfig:
        """Build one canonical control-plane model configuration."""
        return cls(
            artifact_id=artifact_id,
            model_config_json=_model_config_json(model_config),
            model_config_fingerprint=model_config_fingerprint(model_config),
            exact_registry_fingerprint=exact_registry_fingerprint,
        )

    def to_model_config(self) -> SimpleStatelessModelConfig:
        """Parse the validated portable topology."""
        try:
            payload = json.loads(self.model_config_json)
        except json.JSONDecodeError as exc:
            raise ValueError("artifact model configuration JSON is invalid") from exc
        return SimpleStatelessModelConfig.model_validate(payload)


class NativeCollectionWindow(_StrictIdentity):
    """One topology-frozen, globally budgeted distributed collection window.

    V2 historical resource locations are repository-relative deployment
    bindings. Every worker must receive identical content at those locations;
    the worker verifies the declared immutable identities before collection.
    """

    identity: NativeRolloutWindowIdentity
    target_trainable_decisions: int = Field(gt=0)
    active_artifacts: tuple[NativeBfloat16ArtifactManifest, ...]
    artifact_model_configs: tuple[NativeArtifactModelConfig, ...]
    active_pfsp_artifact_ids: tuple[str, ...]
    pfsp_members: tuple[PfspMember, ...]
    required_worker_ids: tuple[str, ...]
    topology_epoch: int = Field(ge=0)
    opened_at_unix_ns: int = Field(ge=0)
    shard_protocol_version: Literal[1, 2, 3] = 1
    assignment_plan_revision: str | None = None
    opponent_pool_revision: str | None = None
    historical_artifact_bindings: tuple[StatelessHistoricalAnchorConfig, ...] = ()
    immediate_whole_game_cutoff_on_drain: bool = False

    @field_validator(
        "assignment_plan_revision",
        "opponent_pool_revision",
    )
    @classmethod
    def valid_optional_revision(cls, value: str | None) -> str | None:
        """Require immutable revision identities when V2 declares them."""
        return None if value is None else _validate_sha256(value)

    @model_validator(mode="after")
    def coherent_artifacts_and_workers(self) -> NativeCollectionWindow:
        """Bind a unique active artifact set and required topology."""
        artifact_ids = tuple(item.artifact_id for item in self.active_artifacts)
        if (
            not artifact_ids
            or len(artifact_ids) != len(set(artifact_ids))
            or len(self.required_worker_ids) != len(set(self.required_worker_ids))
        ):
            raise ValueError("window artifact and worker identities must be unique")
        member_ids = tuple(member.member_id for member in self.pfsp_members)
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("window PFSP member identities must be unique")
        available = set(artifact_ids) | {
            member.policy_sha256 for member in self.pfsp_members
        }
        if any(item not in available for item in self.active_pfsp_artifact_ids):
            raise ValueError("window PFSP artifact is absent from the active set")
        current = tuple(
            item for item in self.active_artifacts if item.kind == "current"
        )
        if len(current) != 1:
            raise ValueError("distributed window requires exactly one current artifact")
        config_ids = tuple(item.artifact_id for item in self.artifact_model_configs)
        if len(config_ids) != len(set(config_ids)) or set(config_ids) != set(
            artifact_ids
        ):
            raise ValueError(
                "window artifact model configurations differ from active artifacts"
            )
        configs = {item.artifact_id: item for item in self.artifact_model_configs}
        if any(
            (
                configs[item.artifact_id].model_config_fingerprint
                != item.model_config_fingerprint
                or configs[item.artifact_id].exact_registry_fingerprint
                != item.exact_registry_fingerprint
            )
            for item in self.active_artifacts
        ):
            raise ValueError("window artifact model configuration identity differs")
        if (
            current[0].source_fp32_fingerprint
            != self.identity.behavior_policy_fingerprint
        ):
            raise ValueError("window current artifact differs from behavior identity")
        revisions = (
            self.assignment_plan_revision,
            self.opponent_pool_revision,
        )
        if self.shard_protocol_version == 1:
            if any(item is not None for item in revisions) or (
                self.historical_artifact_bindings
            ):
                raise ValueError("V1 native windows cannot declare V2 revisions")
        elif any(item is None for item in revisions):
            raise ValueError("V2 native windows require plan and pool revisions")
        if self.shard_protocol_version in {2, 3}:
            historical_members = {
                member.member_id: member
                for member in self.pfsp_members
                if member.source == "historical_anchor"
            }
            bindings = {
                binding.member_id: binding
                for binding in self.historical_artifact_bindings
            }
            if len(bindings) != len(self.historical_artifact_bindings) or set(
                bindings
            ) != set(historical_members):
                raise ValueError("native window historical binding coverage differs")
            for member_id, member in historical_members.items():
                binding = bindings[member_id]
                resource_paths = (
                    binding.checkpoint_path,
                    binding.exact_deck_path,
                    binding.belief_summary_path,
                )
                if any(
                    path is not None and (path.is_absolute() or ".." in path.parts)
                    for path in resource_paths
                ):
                    raise ValueError(
                        "native historical bindings require repo-relative paths"
                    )
                if (
                    binding.snapshot_id != member.snapshot_id
                    or binding.checkpoint_size_bytes != member.policy_size_bytes
                    or binding.checkpoint_sha256 != member.policy_sha256
                    or binding.pilot_artifact_fingerprint
                    != member.pilot_artifact_fingerprint
                    or binding.bundle_fingerprint != member.bundle_fingerprint
                    or binding.exact_deck_digest != member.exact_deck_digest
                    or binding.input_contract_fingerprint
                    != member.input_contract_fingerprint
                    or binding.exact_registry_fingerprint
                    != member.exact_registry_fingerprint
                ):
                    raise ValueError(
                        "native historical binding differs from its member"
                    )
        return self


class NativeAssignedGame(_StrictIdentity):
    """Pydantic wire projection of one central controller assignment."""

    balance: DeckSeatAssignment
    curriculum: CurriculumAssignment

    @classmethod
    def from_assignment(
        cls,
        assignment: StatelessAssignedGame,
    ) -> NativeAssignedGame:
        """Project an in-process assignment into its strict wire form."""
        return cls(
            balance=assignment.balance,
            curriculum=assignment.curriculum,
        )

    def to_assignment(self) -> StatelessAssignedGame:
        """Return the in-process immutable assignment pair."""
        return StatelessAssignedGame(
            balance=self.balance,
            curriculum=self.curriculum,
        )


class NativeCollectionShardLease(_StrictIdentity):
    """One immutable assignment shard that may have multiple attempts."""

    lease_id: str
    window: NativeCollectionWindow
    shard_sequence_id: int = Field(ge=0)
    assignments: tuple[NativeAssignedGame, ...]
    assignments_fingerprint: str
    shard_seed: int = Field(ge=0)
    capacity_tier_id: str
    capacity_tier: NativeCollectionCapacityTier
    estimated_decision_credit: int = Field(gt=0)
    issued_at_unix_ns: int = Field(ge=0)
    required_artifact_ids: tuple[str, ...] | None = None

    @field_validator("lease_id", "capacity_tier_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Reject ambiguous lease or tier aliases."""
        return _validate_id(value)

    @field_validator("assignments_fingerprint")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require exact assignment content identity."""
        return _validate_sha256(value)

    @field_validator("required_artifact_ids")
    @classmethod
    def valid_required_artifact_ids(
        cls,
        value: tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        """Reject duplicate or ambiguous lease-local artifact aliases."""
        if value is None:
            return None
        normalized = tuple(_validate_id(item) for item in value)
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("lease required artifacts must be non-empty and unique")
        return normalized

    @model_validator(mode="after")
    def coherent_assignments(self) -> NativeCollectionShardLease:
        """Reject empty, duplicate, or incorrectly fingerprinted assignments."""
        if not self.assignments:
            raise ValueError("native collection shard requires assignments")
        curriculum_ids = tuple(
            item.curriculum.assignment_id for item in self.assignments
        )
        balance_ids = tuple(item.balance.assignment_id for item in self.assignments)
        if len(curriculum_ids) != len(set(curriculum_ids)) or len(balance_ids) != len(
            set(balance_ids)
        ):
            raise ValueError("native collection shard assignments are duplicated")
        actual = native_assignment_fingerprint(self.assignments)
        if actual != self.assignments_fingerprint:
            raise ValueError("native collection shard assignment fingerprint differs")
        if (
            self.capacity_tier.tier_id != self.capacity_tier_id
            or self.capacity_tier.estimated_trainable_decisions
            != self.estimated_decision_credit
            or len(self.assignments) != self.capacity_tier.concurrent_games
        ):
            raise ValueError("native collection shard capacity tier differs")
        available_ids = {item.artifact_id for item in self.window.active_artifacts} | {
            member.policy_sha256 for member in self.window.pfsp_members
        }
        if (
            self.required_artifact_ids is not None
            and not set(self.required_artifact_ids) <= available_ids
        ):
            raise ValueError("lease requires an artifact outside its window")
        if self.window.shard_protocol_version in {2, 3}:
            expected_artifacts = native_required_artifact_ids(
                self.window,
                self.assignments,
            )
            if self.required_artifact_ids is None:
                raise ValueError(
                    "V2 native leases require explicit artifact identities"
                )
            if self.required_artifact_ids != expected_artifacts:
                raise ValueError("lease required artifacts differ from its assignments")
            frozen_artifacts = len(expected_artifacts) - 1
            if frozen_artifacts > self.capacity_tier.native_policy_group_bank_limit:
                raise ValueError("lease frozen artifacts exceed its capacity tier")
        return self

    @property
    def effective_required_artifact_ids(self) -> tuple[str, ...]:
        """Return V2's exact set or the V1 whole-window compatibility set."""
        if self.required_artifact_ids is not None:
            return self.required_artifact_ids
        return tuple(item.artifact_id for item in self.window.active_artifacts)


def native_retry_capacity_tier(
    tiers: Sequence[NativeCollectionCapacityTier],
    lease: NativeCollectionShardLease,
) -> NativeCollectionCapacityTier | None:
    """Choose a safe local geometry for retrying an immutable shard lease.

    Capacity tier aliases and tuning fields are worker-local runtime details.
    A retry preserves the lease's assignments and artifact identities while a
    different worker may use its own engine/fact parallelism and decision
    estimate.  Exact tier equality is preferred, but equal assignment geometry
    with sufficient frozen-policy capacity is also safe.
    """
    required_frozen_artifacts = (
        len(lease.effective_required_artifact_ids) - 1
        if lease.window.shard_protocol_version in {2, 3}
        else lease.capacity_tier.native_policy_group_bank_limit
    )
    compatible = tuple(
        tier
        for tier in tiers
        if tier.concurrent_games >= len(lease.assignments)
        and tier.native_arena_capacity >= lease.capacity_tier.native_arena_capacity
        and tier.native_engine_shards <= lease.capacity_tier.native_arena_capacity
        and tier.native_process_workers <= lease.capacity_tier.native_arena_capacity
        and tier.native_policy_group_bank_limit >= required_frozen_artifacts
    )
    selected = next(
        (tier for tier in compatible if tier == lease.capacity_tier),
        None,
    )
    if selected is None:
        selected = min(
            compatible,
            key=lambda tier: (
                tier.concurrent_games,
                tier.estimated_trainable_decisions,
                tier.tier_id,
            ),
            default=None,
        )
    if selected is None:
        return None
    effective = native_effective_capacity_tier(selected, len(lease.assignments))
    live_capacity = lease.capacity_tier.native_arena_capacity
    if effective.native_arena_capacity == live_capacity:
        return effective
    return effective.model_copy(
        update={
            "native_arena_capacity": live_capacity,
            "native_policy_cohort_slots": (
                None
                if effective.native_policy_cohort_slots is None
                else min(effective.native_policy_cohort_slots, live_capacity)
            ),
        }
    )


def native_effective_capacity_tier(
    tier: NativeCollectionCapacityTier,
    concurrent_games: int,
) -> NativeCollectionCapacityTier:
    """Scale an advertised tier to a smaller retry-safe tail lease."""
    minimum_games = max(tier.native_engine_shards, tier.native_process_workers)
    if not minimum_games <= concurrent_games <= tier.concurrent_games:
        raise ValueError("effective native tier game count is outside its geometry")
    if concurrent_games == tier.concurrent_games:
        return tier
    estimated_decisions = max(
        1,
        math.ceil(
            tier.estimated_trainable_decisions
            * concurrent_games
            / tier.concurrent_games
        ),
    )
    return tier.model_copy(
        update={
            "concurrent_games": concurrent_games,
            "native_arena_capacity": min(
                tier.native_arena_capacity,
                concurrent_games,
            ),
            "native_policy_cohort_slots": (
                None
                if tier.native_policy_cohort_slots is None
                else min(
                    tier.native_policy_cohort_slots,
                    tier.native_arena_capacity,
                    concurrent_games,
                )
            ),
            "estimated_trainable_decisions": estimated_decisions,
        }
    )


class NativeCollectionAttempt(_StrictIdentity):
    """One worker/session attempt for an immutable shard lease.

    The attempt ID and ownership are immutable. Its expiry is a renewable
    liveness deadline extended by authenticated active-attempt heartbeats.
    """

    attempt_id: str
    lease_id: str
    worker_id: str
    worker_session_id: str
    attempt_sequence: int = Field(ge=0)
    started_at_unix_ns: int = Field(ge=0)
    expires_at_unix_ns: int = Field(gt=0)

    @field_validator("attempt_id", "lease_id", "worker_id", "worker_session_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Reject ambiguous attempt identities."""
        return _validate_id(value)

    @model_validator(mode="after")
    def increasing_deadline(self) -> NativeCollectionAttempt:
        """Require every issued or renewed deadline to follow the start."""
        if self.expires_at_unix_ns <= self.started_at_unix_ns:
            raise ValueError("native collection attempt expiry must follow start")
        return self


class NativeCollectionPartIdentity(_StrictIdentity):
    """Identity for one ACKed compact part within one shard attempt."""

    part_id: str
    lease_id: str
    attempt_id: str
    shard_sequence_id: int = Field(ge=0)
    part_sequence_id: int = Field(ge=0)
    fragment_count: int = Field(gt=0)
    decision_count: int = Field(gt=0)

    @field_validator("part_id", "lease_id", "attempt_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Reject ambiguous part identities."""
        return _validate_id(value)


class NativeCollectionShardResult(_StrictIdentity):
    """Validated terminal evidence for one complete shard attempt."""

    lease_id: str
    attempt_id: str
    shard_sequence_id: int = Field(ge=0)
    part_count: int = Field(ge=0)
    fragment_count: int = Field(ge=0)
    decision_count: int = Field(ge=0)
    outcomes: tuple[StatelessGameOutcome, ...]
    outcomes_fingerprint: str
    report: StatelessCollectionReport
    elapsed_seconds: float = Field(gt=0.0)

    @field_validator("lease_id", "attempt_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Reject ambiguous result identities."""
        return _validate_id(value)

    @field_validator("outcomes_fingerprint")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require exact terminal-evidence identity."""
        return _validate_sha256(value)

    @model_validator(mode="after")
    def coherent_outcomes(self) -> NativeCollectionShardResult:
        """Bind the result to a unique complete outcome set."""
        ids = tuple(item.curriculum_assignment_id for item in self.outcomes)
        if len(ids) != len(set(ids)):
            raise ValueError("native shard result contains duplicate outcomes")
        if self.report.games_started != len(ids):
            raise ValueError("native shard started games differ from outcomes")
        if (
            self.report.games_finished + self.report.games_cancelled
            != self.report.games_started
        ):
            raise ValueError("native shard game settlement counters differ")
        if native_outcome_fingerprint(self.outcomes) != self.outcomes_fingerprint:
            raise ValueError("native shard outcome fingerprint differs")
        return self


class NativeArtifactExposure(_StrictIdentity):
    """Per-artifact matchup exposure derived from accepted terminal evidence."""

    artifact_id: str
    kind: Literal["current", "past_self", "historical_anchor"]
    assigned_games: int = Field(gt=0)
    engine_terminals: int = Field(ge=0)
    candidate_trainable_decisions: int = Field(ge=0)

    @field_validator("artifact_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Reject an ambiguous artifact exposure identity."""
        return _validate_id(value)

    @model_validator(mode="after")
    def coherent_counts(self) -> NativeArtifactExposure:
        """Terminal games cannot exceed the artifact's assigned matchups."""
        if self.engine_terminals > self.assigned_games:
            raise ValueError("artifact terminal exposure exceeds assigned games")
        return self


class NativeCollectionWindowReceipt(_StrictIdentity):
    """Final exactly-once evidence for one committed or aborted window."""

    window_id: str
    sequence_id: int = Field(ge=0)
    status: Literal["committed", "aborted"]
    target_trainable_decisions: int = Field(gt=0)
    accepted_trainable_decisions: int = Field(ge=0)
    overshoot_decisions: int = Field(ge=0)
    shard_lease_ids: tuple[str, ...]
    accepted_attempt_ids: tuple[str, ...]
    worker_manifests: tuple[NativeCollectionWorkerManifest, ...]
    committed_at_unix_ns: int = Field(ge=0)
    abort_reason: str | None = None
    early_commit_reason: Literal[
        "learner_ready_drain",
        "learner_clocked_high_water",
    ] | None = None
    shard_protocol_version: Literal[1, 2, 3] = 1
    assignment_plan_revision: str | None = None
    opponent_pool_revision: str | None = None
    required_exposure_artifact_ids: tuple[str, ...] = ()
    artifact_exposures: tuple[NativeArtifactExposure, ...] = ()

    @field_validator("window_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Reject ambiguous window aliases."""
        return _validate_id(value)

    @field_validator(
        "assignment_plan_revision",
        "opponent_pool_revision",
    )
    @classmethod
    def valid_optional_revision(cls, value: str | None) -> str | None:
        """Validate the optional V2 planning evidence."""
        return None if value is None else _validate_sha256(value)

    @field_validator("required_exposure_artifact_ids")
    @classmethod
    def valid_required_exposure_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Require stable unique identities for the V2 coverage gate."""
        normalized = tuple(_validate_id(value) for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("native required exposure identities must be unique")
        return normalized

    @model_validator(mode="after")
    def coherent_status(self) -> NativeCollectionWindowReceipt:
        """Require exact overshoot and status-specific evidence."""
        expected_overshoot = max(
            self.accepted_trainable_decisions - self.target_trainable_decisions,
            0,
        )
        if self.overshoot_decisions != expected_overshoot:
            raise ValueError("native window receipt overshoot differs")
        if self.status == "committed":
            if (
                (
                    self.accepted_trainable_decisions
                    < self.target_trainable_decisions
                    and self.early_commit_reason
                    not in {"learner_ready_drain", "learner_clocked_high_water"}
                )
                or self.abort_reason is not None
            ):
                raise ValueError("committed native window is incomplete")
        elif not self.abort_reason or self.early_commit_reason is not None:
            raise ValueError("aborted native window requires only an abort reason")
        if len(self.shard_lease_ids) != len(set(self.shard_lease_ids)) or len(
            self.accepted_attempt_ids
        ) != len(set(self.accepted_attempt_ids)):
            raise ValueError("native window receipt identities must be unique")
        revisions = (
            self.assignment_plan_revision,
            self.opponent_pool_revision,
        )
        if self.shard_protocol_version == 1:
            if any(value is not None for value in revisions) or (
                self.artifact_exposures or self.required_exposure_artifact_ids
            ):
                raise ValueError("V1 native receipt contains V2 evidence")
        else:
            if any(value is None for value in revisions):
                raise ValueError("V2 native receipt omitted planning evidence")
            exposure_ids = tuple(item.artifact_id for item in self.artifact_exposures)
            if exposure_ids != tuple(sorted(exposure_ids)) or len(exposure_ids) != len(
                set(exposure_ids)
            ):
                raise ValueError("V2 receipt artifact exposure is not canonical")
            current = tuple(
                item for item in self.artifact_exposures if item.kind == "current"
            )
            if (
                (self.status == "committed" or self.artifact_exposures)
                and len(current) != 1
            ) or (
                current
                and current[0].candidate_trainable_decisions
                != self.accepted_trainable_decisions
            ):
                raise ValueError("V2 receipt artifact decision evidence differs")
            covered_artifacts = {
                artifact.artifact_id
                for artifact in self.artifact_exposures
                if (
                    artifact.kind != "current"
                    and artifact.candidate_trainable_decisions > 0
                )
            }
            if (
                self.status == "committed"
                and not set(self.required_exposure_artifact_ids) <= covered_artifacts
            ):
                raise ValueError("V2 receipt omitted required artifact exposure")
        return self


def native_assignment_fingerprint(
    assignments: tuple[NativeAssignedGame, ...],
) -> str:
    """Return the canonical content identity of one assignment shard."""
    return _canonical_fingerprint(
        _ASSIGNMENT_DOMAIN,
        [item.model_dump(mode="json") for item in assignments],
    )


def native_assignment_plan_revision(
    assignments: Sequence[NativeAssignedGame],
) -> str:
    """Return the ordered immutable identity of one controller plan."""
    return _canonical_fingerprint(
        _ASSIGNMENT_PLAN_DOMAIN,
        [item.model_dump(mode="json") for item in assignments],
    )


def native_required_artifact_ids(
    window: NativeCollectionWindow,
    assignments: Sequence[NativeAssignedGame],
) -> tuple[str, ...]:
    """Derive the exact BF16 artifacts needed by an assignment cohort."""
    current = tuple(
        item.artifact_id for item in window.active_artifacts if item.kind == "current"
    )
    if len(current) != 1:
        raise ValueError("native artifact resolution requires one current policy")
    required = {current[0]}
    members = {member.member_id: member for member in window.pfsp_members}
    for assignment in assignments:
        member_id = assignment.curriculum.member_id
        if not member_id:
            continue
        try:
            member = members[member_id]
        except KeyError as exc:
            raise ValueError(
                "native assignment references an absent opponent-pool member"
            ) from exc
        curriculum = assignment.curriculum
        if (
            curriculum.opponent_artifact_fingerprint != member.bundle_fingerprint
            or curriculum.opponent_pilot_fingerprint
            != member.pilot_artifact_fingerprint
            or curriculum.opponent_deck_digest != member.exact_deck_digest
        ):
            raise ValueError("native assignment differs from its opponent-pool member")
        required.add(member.policy_sha256)
    available = {item.artifact_id for item in window.active_artifacts} | {
        member.policy_sha256 for member in window.pfsp_members
    }
    if not required <= available:
        raise ValueError("native assignment requires an unavailable BF16 artifact")
    return (current[0], *sorted(required - {current[0]}))


def native_outcome_fingerprint(
    outcomes: tuple[StatelessGameOutcome, ...],
) -> str:
    """Return the canonical identity of ordered terminal evidence."""
    return _canonical_fingerprint(
        _OUTCOME_DOMAIN,
        [item.model_dump(mode="json") for item in outcomes],
    )


def _validate_id(value: str) -> str:
    if (
        not value
        or len(value) > _MAX_ID_LENGTH
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("native rollout identity is invalid")
    return value


def _validate_sha256(value: str) -> str:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("native rollout fingerprint must be lower-case SHA-256")
    return value


def _canonical_fingerprint(domain: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


def _model_config_json(config: SimpleStatelessModelConfig) -> str:
    return json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
