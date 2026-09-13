"""Hydra-facing configuration for the clean single-H200 stateless trainer."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from ipaddress import ip_address
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.belief.public_catalog import PublicDeckCatalogConfig
from ptcg_rl.decks.registry import PrivateDeckRegistrySourceConfig
from ptcg_rl.model.simple_stateless import SimpleStatelessModelConfig
from ptcg_rl.model.simple_stateless.config import (
    uses_family_private_topology,
    uses_generalist_sequence,
)
from ptcg_rl.rl.opponent_pool import PlannerPolicy, StratumPolicy
from ptcg_rl.rl.opponent_pool.adaptive import (
    AdaptiveOpponentAllocationConfig,
    RoleBudgetOpponentAllocationConfig,
)
from ptcg_rl.rl.stateless_bc_overlay import StatelessBcOverlayDeclaration
from ptcg_rl.rl.stateless_curriculum import (
    CurriculumLane,
    MatchupPfspConfig,
    OpponentLaneMix,
    PastSelfRetentionConfig,
    apportion_neutral_deficit_lane_coverage,
)
from ptcg_rl.rl.stateless_deck_balance import DeckTargetShare
from ptcg_rl.rl.stateless_ppo import (
    SimpleStatelessPpoConfig,
    StatelessLearnerPrecision,
)
from ptcg_rl.rl.stateless_supervised_startup import (
    StatelessSupervisedStartupDeclaration,
)
from ptcg_rl.rl.stateless_topology_transition import (
    StatelessTopologyTransitionDeclaration,
)
from ptcg_rl.training.run_config import TrainingRunConfig

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
MAX_NATIVE_PROCESS_WORKERS = 64
MAX_NATIVE_ENGINE_SHARDS_PER_PROCESS = 8


class StatelessExactStrategyInitialization(BaseModel):
    """One explicitly authorized target-only exact strategy lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_deck_digest: str
    target_expert_id: str
    mode: Literal["zero", "clone"] = "zero"
    source_expert_id: str | None = None

    @field_validator("target_deck_digest", "target_expert_id", "source_expert_id")
    @classmethod
    def valid_identity(cls, value: str | None) -> str | None:
        """Require immutable full identities for both deck and lineage."""
        return None if value is None else _fingerprint(value)

    @model_validator(mode="after")
    def coherent_clone_source(self) -> Self:
        """Bind exact clones to one explicit retained source lineage."""
        if self.mode == "clone" and self.source_expert_id is None:
            raise ValueError("exact clone initialization requires source_expert_id")
        if self.mode == "zero" and self.source_expert_id is not None:
            raise ValueError(
                "zero exact initialization cannot declare source_expert_id"
            )
        if self.source_expert_id == self.target_expert_id:
            raise ValueError("exact initialization cannot clone itself")
        return self


class StatelessFamilyStrategyInitialization(BaseModel):
    """One explicitly authorized target-only family strategy lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_family_id: str
    mode: Literal["clone_generic", "clone"] = "clone_generic"
    source_family_id: str | None = None

    @field_validator("target_family_id", "source_family_id")
    @classmethod
    def valid_identity(cls, value: str | None) -> str | None:
        """Require immutable full family-lineage identities."""
        return None if value is None else _fingerprint(value)

    @model_validator(mode="after")
    def coherent_clone_source(self) -> Self:
        """Bind learned-family clones to one explicit source lineage."""
        if self.mode == "clone" and self.source_family_id is None:
            raise ValueError("family clone initialization requires source_family_id")
        if self.mode == "clone_generic" and self.source_family_id is not None:
            raise ValueError(
                "generic family initialization cannot declare source_family_id"
            )
        if self.source_family_id == self.target_family_id:
            raise ValueError("family initialization cannot clone itself")
        return self


class StatelessFamilyStrategyReassignment(BaseModel):
    """One explicitly authorized exact-deck move to a new family lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_digest: str
    source_family_id: str
    target_family_id: str

    @field_validator("deck_digest", "source_family_id", "target_family_id")
    @classmethod
    def valid_identity(cls, value: str) -> str:
        """Require immutable full identities for the route and both families."""
        return _fingerprint(value)

    @model_validator(mode="after")
    def changes_family(self) -> Self:
        """Reject a declaration that leaves the route in the same family."""
        if self.source_family_id == self.target_family_id:
            raise ValueError("family reassignment must change the family lineage")
        return self


def _default_opponent_pool_v2_policy() -> PlannerPolicy:
    """Return the initial three-stratum formal scheduling policy."""
    return PlannerPolicy(
        strata=(
            StratumPolicy(
                stratum="protected",
                artifact_slots=1,
                target_fraction=0.25,
            ),
            StratumPolicy(
                stratum="recent",
                artifact_slots=2,
                target_fraction=0.45,
            ),
            StratumPolicy(
                stratum="age_diverse",
                artifact_slots=2,
                target_fraction=0.30,
            ),
        )
    )


class StatelessOpponentPoolV2Config(BaseModel):
    """Formal Historical Opponent Pool V2 controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    behavior_version: Literal[1, 2, 3, 4, 5] = 1
    maximum_active_artifacts: int = Field(default=8, ge=3)
    past_self_artifacts: int = Field(default=6, ge=2)
    recent_artifacts: int = Field(default=2, ge=1)
    planner_policy: PlannerPolicy = Field(
        default_factory=_default_opponent_pool_v2_policy
    )
    adaptive_allocation: AdaptiveOpponentAllocationConfig | None = None
    role_budget_allocation: RoleBudgetOpponentAllocationConfig | None = None

    @model_validator(mode="after")
    def coherent_archive_partition(self) -> Self:
        """Keep recent and age-diverse membership non-empty and bounded."""
        if self.behavior_version == 3:
            if self.adaptive_allocation is None:
                raise ValueError(
                    "joint adaptive opponent pool requires allocation settings"
                )
            if self.maximum_active_artifacts < 3:
                raise ValueError(
                    "joint adaptive opponent pool requires an active artifact set"
                )
            if self.role_budget_allocation is not None:
                raise ValueError("V3 adaptive allocation cannot use V4 role settings")
            return self
        if self.behavior_version == 4:
            if self.role_budget_allocation is None:
                raise ValueError("role-budget opponent pool requires evidence settings")
            if self.adaptive_allocation is not None:
                raise ValueError("V4 role allocation cannot use V3 weighted settings")
            if self.maximum_active_artifacts < 3:
                raise ValueError("role-budget opponent pool requires active artifacts")
            if (
                self.role_budget_allocation.matchup_game_batch_size != 1
                or self.role_budget_allocation.matchup_coverage_windows is not None
            ):
                raise ValueError("V4 role allocation cannot enable sparse execution")
            return self
        if self.behavior_version == 5:
            if self.role_budget_allocation is None:
                raise ValueError("sparse role-budget pool requires evidence settings")
            if self.adaptive_allocation is not None:
                raise ValueError("V5 role allocation cannot use V3 weighted settings")
            if self.maximum_active_artifacts < 3:
                raise ValueError("sparse role-budget pool requires active artifacts")
            if (
                self.role_budget_allocation.matchup_game_batch_size <= 1
                or self.role_budget_allocation.matchup_coverage_windows is None
            ):
                raise ValueError(
                    "V5 role allocation requires batched execution and coverage"
                )
            return self
        if self.adaptive_allocation is not None:
            raise ValueError(
                "adaptive allocation settings require behavior version three"
            )
        if self.role_budget_allocation is not None:
            raise ValueError(
                "role-budget settings require behavior version four or five"
            )
        if self.behavior_version == 2:
            strata = tuple(item.stratum for item in self.planner_policy.strata)
            expected = (
                "protected",
                "counter_frontier",
                "recent",
                "age_diverse",
            )
            if strata != expected:
                raise ValueError(
                    "lineage opponent pool requires protected, counter, recent, "
                    "and age strata"
                )
            if self.maximum_active_artifacts < 8:
                raise ValueError(
                    "lineage opponent pool requires at least eight active artifacts"
                )
            return self
        if self.recent_artifacts >= self.past_self_artifacts:
            raise ValueError(
                "opponent-pool recent artifacts must leave age-diverse artifacts"
            )
        return self


class StatelessRegistryTransitionConfig(BaseModel):
    """Explicit lifecycle declaration for one stateless exact-roster change."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_registry_sha256: str
    source_family_registry_sha256: str | None = None
    source_weighted_balance_config_fingerprint: str | None = None
    source_weighted_target_deck_shares: tuple[DeckTargetShare, ...] = ()
    initializations: tuple[StatelessExactStrategyInitialization, ...] = ()
    family_initializations: tuple[StatelessFamilyStrategyInitialization, ...] = ()
    family_reassignments: tuple[StatelessFamilyStrategyReassignment, ...] = ()
    retired_expert_ids: tuple[str, ...] = ()
    retired_family_ids: tuple[str, ...] = ()

    @field_validator(
        "source_registry_sha256",
        "source_family_registry_sha256",
        "source_weighted_balance_config_fingerprint",
    )
    @classmethod
    def valid_source_registry(cls, value: str | None) -> str | None:
        """Bind the transition to one immutable source registry."""
        return None if value is None else _fingerprint(value)

    @field_validator("retired_expert_ids", "retired_family_ids")
    @classmethod
    def valid_retired_lineages(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject ambiguous duplicate retirement declarations."""
        normalized = tuple(_fingerprint(value) for value in values)
        if len(set(normalized)) != len(normalized):
            raise ValueError("retired stateless lineage IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def unique_initializations(self) -> Self:
        """Require one declaration for each target-only deck and lineage."""
        decks = tuple(item.target_deck_digest for item in self.initializations)
        experts = tuple(item.target_expert_id for item in self.initializations)
        families = tuple(item.target_family_id for item in self.family_initializations)
        reassigned_decks = tuple(
            item.deck_digest for item in self.family_reassignments
        )
        if len(set(decks)) != len(decks):
            raise ValueError("stateless initialized deck digests must be unique")
        if len(set(experts)) != len(experts):
            raise ValueError("stateless initialized expert IDs must be unique")
        if len(set(families)) != len(families):
            raise ValueError("stateless initialized family IDs must be unique")
        if len(set(reassigned_decks)) != len(reassigned_decks):
            raise ValueError("stateless reassigned deck digests must be unique")
        has_source_balance_fingerprint = (
            self.source_weighted_balance_config_fingerprint is not None
        )
        has_source_balance_shares = bool(self.source_weighted_target_deck_shares)
        if has_source_balance_fingerprint != has_source_balance_shares:
            raise ValueError(
                "weighted registry source balance requires both its fingerprint "
                "and target shares"
            )
        source_balance_digests = tuple(
            item.deck_digest for item in self.source_weighted_target_deck_shares
        )
        if source_balance_digests and source_balance_digests != tuple(
            sorted(set(source_balance_digests))
        ):
            raise ValueError(
                "weighted registry source balance deck targets must be sorted "
                "and unique"
            )
        return self


class StatelessCollectionConfig(BaseModel):
    """Fixed-horizon synchronous collection and publication budgets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal[
        "python_parallel",
        "native",
        "native_banked",
        "native_distributed",
        "hybrid",
    ] = "python_parallel"
    training_updates: int = Field(gt=0)
    concurrent_games: int = Field(gt=0)
    native_arena_capacity: int | None = Field(default=None, gt=0)
    native_engine_shards: int = Field(default=1, ge=1, le=8)
    native_policy_cohort_slots: int | None = Field(default=None, gt=0)
    native_policy_group_bank_limit: int = Field(default=2, ge=1, le=4)
    native_policy_cohort_wait_ms: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )
    native_trainable_decision_budget: int | None = Field(default=None, gt=0)
    native_shard_protocol_version: Literal[2, 3] = 2
    native_frozen_batch_min_rows: int = Field(default=1, ge=1)
    native_frozen_batch_max_wait_waves: int = Field(default=1, ge=1)
    native_sequence_rollout_precision: Literal["fp32", "bf16"] = "fp32"
    native_engine_fact_workers: int | None = Field(default=None, gt=0)
    native_process_workers: int = Field(
        default=1,
        ge=1,
        le=MAX_NATIVE_PROCESS_WORKERS,
    )
    integrate_scripted_current_inference: bool = False
    pipeline_mode: Literal["synchronous", "one_version_lag"] = "synchronous"
    fragment_horizon: int = Field(gt=0)
    fragments_per_part: int = Field(gt=0)
    mirror_bilateral_trajectories: bool = False
    maximum_engine_steps: int = Field(gt=0)
    checkpoint_interval_updates: int = Field(gt=0)
    checkpoint_keep_last: int | None = Field(default=None, gt=0)
    checkpoint_retain_every_versions: int | None = Field(default=None, gt=0)
    status_interval_seconds: float = Field(default=30.0, gt=0.0)
    actor_workers: int = Field(default=1, ge=1, le=32)
    actor_game_chunk_size: int = Field(default=32, gt=0)
    inference_max_batch_rows: int = Field(default=1024, gt=0)
    inference_batch_wait_ms: float = Field(default=2.0, ge=0.0, le=100.0)
    inference_timeout_seconds: float = Field(default=120.0, gt=0.0)
    collection_timeout_seconds: float = Field(default=900.0, gt=0.0)
    seed: int = Field(ge=0)

    @field_validator(
        "status_interval_seconds",
        "inference_batch_wait_ms",
        "inference_timeout_seconds",
        "collection_timeout_seconds",
    )
    @classmethod
    def finite_interval(cls, value: float) -> float:
        """Reject a non-finite monitoring interval."""
        if not math.isfinite(value):
            raise ValueError("status interval must be finite")
        return value

    @model_validator(mode="after")
    def coherent_checkpoint_retention(self) -> Self:
        """Require rolling retention when permanent milestones are enabled."""
        if (
            self.checkpoint_retain_every_versions is not None
            and self.checkpoint_keep_last is None
        ):
            raise ValueError(
                "checkpoint_retain_every_versions requires checkpoint_keep_last"
            )
        return self

    @model_validator(mode="after")
    def coherent_native_decision_budget(self) -> Self:
        """Restrict native rollout boundaries to native-capable backends."""
        if (
            self.native_trainable_decision_budget is not None
            and self.backend == "python_parallel"
        ):
            raise ValueError(
                "native_trainable_decision_budget requires native or hybrid backend"
            )
        if self.native_frozen_batch_min_rows > 1 and self.backend == "python_parallel":
            raise ValueError(
                "native_frozen_batch_min_rows requires native or hybrid backend"
            )
        if self.native_process_workers > 1 and self.backend == "python_parallel":
            raise ValueError("native_process_workers requires native or hybrid backend")
        if (
            self.native_process_workers > 1
            and self.native_arena_capacity is not None
            and self.native_arena_capacity < self.native_process_workers
        ):
            raise ValueError(
                "native arena capacity must cover every native process worker"
            )
        if self.integrate_scripted_current_inference and (
            self.backend != "hybrid"
            or self.native_process_workers <= 1
            or self.actor_workers <= 1
        ):
            raise ValueError(
                "integrated scripted inference requires hybrid multi-process "
                "native and actor workers"
            )
        if self.backend == "native_banked" and (
            self.native_process_workers != 1
            or self.native_engine_shards < 4
            or self.native_engine_shards % 2 != 0
            or self.integrate_scripted_current_inference
            or self.native_arena_capacity is None
            or self.native_arena_capacity != self.concurrent_games
            or self.concurrent_games < self.native_engine_shards
            or (
                self.native_policy_cohort_slots is not None
                and (
                    self.native_engine_shards != 4
                    or self.native_policy_cohort_slots != self.concurrent_games
                )
            )
            or self.native_policy_cohort_wait_ms != 0.0
            or self.native_frozen_batch_min_rows != 1
            or self.native_frozen_batch_max_wait_waves != 1
        ):
            raise ValueError(
                "banked native collection requires one in-process worker, an "
                "even fixed ring of at least four engine arenas, full-window "
                "capacity, optional full-window two-bank coalescing only, and "
                "disabled frozen-row parking"
            )
        return self


class StatelessNativeDistributedTransportConfig(BaseModel):
    """Trusted-private-network endpoints for three independent ZMQ channels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bind_host: str = "127.0.0.1"
    control_port: int = Field(default=47670, ge=1, le=65535)
    artifact_port: int = Field(default=47671, ge=1, le=65535)
    data_port: int = Field(default=47672, ge=1, le=65535)
    io_threads: int = Field(default=1, ge=1, le=8)
    socket_high_watermark: int = Field(default=8, ge=1)
    worker_part_queue_capacity: int = Field(default=2, ge=1, le=8)
    trusted_private_network: Literal[True] = True

    @field_validator("bind_host")
    @classmethod
    def private_bind_host(cls, value: str) -> str:
        """Forbid wildcard/public binds in the unauthenticated first version."""
        cleaned = value.strip()
        try:
            address = ip_address(cleaned)
        except ValueError as exc:
            raise ValueError(
                "native distributed bind_host must be a private IP literal"
            ) from exc
        if address.is_unspecified or not (address.is_private or address.is_loopback):
            raise ValueError(
                "native distributed transport may bind only a private address"
            )
        return cleaned

    @model_validator(mode="after")
    def distinct_ports(self) -> Self:
        """Keep control, artifact, and data backpressure independent."""
        ports = (self.control_port, self.artifact_port, self.data_port)
        if len(set(ports)) != len(ports):
            raise ValueError("native distributed channel ports must be distinct")
        return self


class StatelessNativeDistributedQuorumConfig(BaseModel):
    """Minimum worker availability without making accelerators fatal gates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    required_worker_ids: tuple[str, ...] = ()
    heartbeat_interval_seconds: float = Field(default=5.0, gt=0.0)
    heartbeat_timeout_seconds: float = Field(default=30.0, gt=0.0)
    degrade_grace_seconds: float = Field(default=120.0, gt=0.0)
    # Kept under its established config name for profile compatibility.  When
    # enabled, the minimum quorum is valid at startup as well as after startup.
    allow_degraded_after_startup: bool = True
    minimum_degraded_workers: int = Field(default=1, ge=1)

    @field_validator("required_worker_ids")
    @classmethod
    def unique_workers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require stable, unique formal worker identities."""
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized) or len(normalized) != len(
            set(normalized)
        ):
            raise ValueError("native distributed worker IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def coherent_deadlines(self) -> Self:
        """Allow multiple heartbeats before expiry and grace."""
        if (
            self.heartbeat_timeout_seconds <= self.heartbeat_interval_seconds
            or self.degrade_grace_seconds < self.heartbeat_timeout_seconds
        ):
            raise ValueError("native distributed heartbeat timeout/grace is incoherent")
        if self.required_worker_ids and self.minimum_degraded_workers > len(
            self.required_worker_ids
        ):
            raise ValueError(
                "native distributed degraded quorum exceeds required workers"
            )
        return self


class StatelessNativeDistributedRetryConfig(BaseModel):
    """Bounded attempt retry and channel deadline policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_attempts_per_shard: int = Field(default=3, ge=1)
    lease_timeout_seconds: float = Field(default=900.0, gt=0.0)
    part_ack_timeout_seconds: float = Field(default=120.0, gt=0.0)
    control_poll_interval_seconds: float = Field(default=0.05, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def coherent_timeouts(self) -> Self:
        """A part ACK must fit inside its shard lease."""
        if self.part_ack_timeout_seconds >= self.lease_timeout_seconds:
            raise ValueError(
                "native distributed part ACK timeout must precede lease timeout"
            )
        return self


class StatelessNativeDistributedSchedulingConfig(BaseModel):
    """Latency-first speculative scheduling for a global rollout window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tail_start_fraction: float = Field(
        default=0.65,
        gt=0.0,
        lt=1.0,
        allow_inf_nan=False,
    )
    early_inflight_reservation_fraction: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    tail_inflight_reservation_fraction: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    tail_target_seconds: float = Field(
        default=45.0,
        gt=0.0,
        allow_inf_nan=False,
    )
    worker_yield_ewma_alpha: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    learner_clocked_primary_only: bool = False
    learner_clocked_immediate_whole_game_cutoff: bool = False
    learner_clocked_max_trainable_decisions: int | None = Field(
        default=None,
        gt=0,
    )
    learner_ready_drain_grace_seconds: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )
    learner_ready_drain_minimum_collection_seconds: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def coherent_reservations(self) -> Self:
        """Reserve less speculative credit in the latency-sensitive tail."""
        if (
            self.tail_inflight_reservation_fraction
            > self.early_inflight_reservation_fraction
        ):
            raise ValueError(
                "native tail inflight reservation must not exceed early reservation"
            )
        if (
            self.learner_ready_drain_grace_seconds is None
            and self.learner_ready_drain_minimum_collection_seconds != 0.0
        ):
            raise ValueError(
                "native learner-ready minimum collection time requires its drain grace"
            )
        return self


class StatelessNativeDistributedStatusConfig(BaseModel):
    """Atomic coordinator/worker status publication cadence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    interval_seconds: float = Field(default=5.0, gt=0.0)
    include_worker_gpu_metrics: bool = True


class StatelessNativeWorkerCudaCacheConfig(BaseModel):
    """Pressure-aware cache policy for workers sharing one physical GPU."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    check_interval_seconds: float = Field(default=15.0, gt=0.0)
    force_attempt_settlement_trim: bool = False
    force_window_settlement_trim: bool = False
    clear_cublas_workspaces_at_window_settlement: bool = False
    settled_model_cache_limit: int | None = Field(default=None, ge=0)
    device_free_floor_fraction: float = Field(
        default=0.25,
        gt=0.0,
        lt=1.0,
        allow_inf_nan=False,
    )
    minimum_reclaimable_device_fraction: float = Field(
        default=0.005,
        gt=0.0,
        lt=1.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def coherent_pressure_thresholds(self) -> Self:
        """Require a reclaimable block smaller than the protected headroom."""
        if self.minimum_reclaimable_device_fraction >= self.device_free_floor_fraction:
            raise ValueError(
                "native worker CUDA reclaimable fraction must be below its "
                "device free floor"
            )
        return self


class StatelessNativeWorkerHostMemoryConfig(BaseModel):
    """Host-memory pressure policy for one long-lived collection worker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    check_interval_seconds: float = Field(default=15.0, gt=0.0)
    release_arena_resources_after_attempt: bool = False
    release_arena_resources_after_window: bool = False
    trim_after_attempt: bool = False
    trim_after_window: bool = False
    process_rss_recycle_limit_bytes: int | None = Field(default=None, gt=0)
    process_rss_soft_limit_bytes: int | None = Field(default=None, gt=0)
    process_rss_hard_limit_bytes: int | None = Field(default=None, gt=0)
    system_available_memory_floor_bytes: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def coherent_process_limits(self) -> Self:
        """Keep the drain threshold below the post-trim rebuild threshold."""
        if (
            self.process_rss_recycle_limit_bytes is not None
            and self.process_rss_soft_limit_bytes is not None
            and self.process_rss_recycle_limit_bytes
            >= self.process_rss_soft_limit_bytes
        ):
            raise ValueError(
                "native worker RSS recycle limit must be below its soft limit"
            )
        if (
            self.process_rss_soft_limit_bytes is not None
            and self.process_rss_hard_limit_bytes is not None
            and self.process_rss_soft_limit_bytes >= self.process_rss_hard_limit_bytes
        ):
            raise ValueError(
                "native worker RSS soft limit must be below its hard limit"
            )
        return self


class StatelessNativeWorkerCapacityTierConfig(BaseModel):
    """Hydra-owned runtime geometry advertised by one worker profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

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
    def non_empty_id(cls, value: str) -> str:
        """Require a stable tier alias."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("native distributed capacity tier ID is empty")
        return cleaned

    @model_validator(mode="after")
    def coherent_geometry(self) -> Self:
        """Mirror the banked-native fixed-ring execution constraints."""
        if (
            self.native_arena_capacity > self.concurrent_games
            or self.native_arena_capacity < self.native_engine_shards
            or self.native_arena_capacity < self.native_process_workers
            or (
                math.ceil(self.native_engine_shards / self.native_process_workers)
                > MAX_NATIVE_ENGINE_SHARDS_PER_PROCESS
            )
            or (
                self.native_process_workers == 1
                and (
                    self.native_engine_shards < 4 or self.native_engine_shards % 2 != 0
                )
            )
            or (
                self.native_policy_cohort_slots is not None
                and self.native_policy_cohort_slots > self.native_arena_capacity
            )
        ):
            raise ValueError("native distributed worker tier geometry is invalid")
        return self


class StatelessNativeWorkerProfileConfig(BaseModel):
    """Independent non-semantic runtime inventory for a collection host."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cuda_device_index: int = Field(default=0, ge=0)
    expected_cuda_device_name: str
    minimum_cuda_memory_bytes: int = Field(gt=0)
    require_bfloat16: Literal[True] = True
    maximum_cuda_memory_fraction: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    capacity_tiers: tuple[StatelessNativeWorkerCapacityTierConfig, ...]

    @field_validator("expected_cuda_device_name")
    @classmethod
    def non_empty_device(cls, value: str) -> str:
        """Require an explicit GPU inventory gate."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("native distributed expected CUDA device is empty")
        return cleaned

    @model_validator(mode="after")
    def unique_tiers(self) -> Self:
        """Require at least one uniquely named geometry."""
        ids = tuple(item.tier_id for item in self.capacity_tiers)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("native distributed worker capacity tiers must be unique")
        return self


class StatelessNativeDistributedConfig(BaseModel):
    """Coordinator policy and worker profiles for native scale-out."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transport: StatelessNativeDistributedTransportConfig = Field(
        default_factory=StatelessNativeDistributedTransportConfig
    )
    quorum: StatelessNativeDistributedQuorumConfig = Field(
        default_factory=StatelessNativeDistributedQuorumConfig
    )
    retry: StatelessNativeDistributedRetryConfig = Field(
        default_factory=StatelessNativeDistributedRetryConfig
    )
    scheduling: StatelessNativeDistributedSchedulingConfig = Field(
        default_factory=StatelessNativeDistributedSchedulingConfig
    )
    status: StatelessNativeDistributedStatusConfig = Field(
        default_factory=StatelessNativeDistributedStatusConfig
    )
    cuda_cache: StatelessNativeWorkerCudaCacheConfig = Field(
        default_factory=StatelessNativeWorkerCudaCacheConfig
    )
    host_memory: StatelessNativeWorkerHostMemoryConfig = Field(
        default_factory=StatelessNativeWorkerHostMemoryConfig
    )
    worker_profiles: dict[str, StatelessNativeWorkerProfileConfig] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def coherent_profiles(self) -> Self:
        """Require worker/profile identities without duplicated mutable aliases."""
        normalized = tuple(key.strip() for key in self.worker_profiles)
        if (
            any(not key for key in normalized)
            or len(normalized) != len(set(normalized))
            or tuple(self.worker_profiles) != normalized
        ):
            raise ValueError("native distributed worker profile names are invalid")
        tiers = tuple(
            tier
            for profile in self.worker_profiles.values()
            for tier in profile.capacity_tiers
        )
        if tiers:
            smallest_tier = min(tier.concurrent_games for tier in tiers)
            if any(tier.concurrent_games % smallest_tier != 0 for tier in tiers):
                raise ValueError(
                    "native distributed capacity tiers must share the smallest "
                    "assignment quantum"
                )
        return self


class StatelessLearnerRuntimeConfig(BaseModel):
    """Non-semantic execution choices for the stateless learner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    precision: StatelessLearnerPrecision = "bf16"
    fused_adamw: bool = False
    compile_shared_backbone: bool = False
    host_prepare_workers: int = Field(default=1, gt=0)
    host_prepare_prefetch_batches: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def coherent_host_preparation(self) -> Self:
        """Keep enough bounded work queued to use every preparation worker."""
        if self.host_prepare_prefetch_batches < self.host_prepare_workers:
            raise ValueError(
                "host preparation prefetch must cover every preparation worker"
            )
        return self


class StatelessOptimizerScopeConfig(BaseModel):
    """Semantic optimizer ownership and scope-specific learning rates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["full_model", "private_only", "hybrid"] = "full_model"
    private_learning_rate: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
    )
    shared_learning_rate_initial: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
    )
    shared_learning_rate_target: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
    )
    shared_learning_rate_warmup_start_update_index: int | None = Field(
        default=None,
        ge=0,
    )
    shared_learning_rate_warmup_updates: int | None = Field(
        default=None,
        ge=2,
    )

    @model_validator(mode="after")
    def coherent_private_learning_rate(self) -> Self:
        """Require only the learning-rate fields used by the selected scope."""
        shared_values = (
            self.shared_learning_rate_initial,
            self.shared_learning_rate_target,
            self.shared_learning_rate_warmup_start_update_index,
            self.shared_learning_rate_warmup_updates,
        )
        if self.mode == "full_model" and (
            self.private_learning_rate is not None
            or any(value is not None for value in shared_values)
        ):
            raise ValueError(
                "full-model optimizer scope uses only the PPO learning-rate schedule"
            )
        if self.mode == "private_only" and (
            self.private_learning_rate is None
            or any(value is not None for value in shared_values)
        ):
            raise ValueError(
                "private-only optimizer scope requires exactly one private "
                "learning rate"
            )
        if self.mode == "hybrid":
            if self.private_learning_rate is None or any(
                value is None for value in shared_values
            ):
                raise ValueError(
                    "hybrid optimizer scope requires private and shared learning "
                    "rate declarations"
                )
            assert self.shared_learning_rate_initial is not None
            assert self.shared_learning_rate_target is not None
            if self.shared_learning_rate_initial > self.shared_learning_rate_target:
                raise ValueError(
                    "hybrid shared learning-rate warmup cannot decrease its rate"
                )
        return self


class StatelessOptimizerScopeTransitionConfig(BaseModel):
    """Authorize one audited AdamW optimizer-scope transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_mode: Literal["full_model", "private_only"] = "full_model"
    target_mode: Literal["private_only", "hybrid"] = "private_only"
    target_private_learning_rate: float = Field(
        gt=0.0,
        allow_inf_nan=False,
    )
    target_shared_learning_rate_initial: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
    )
    target_shared_learning_rate_target: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
    )
    target_shared_learning_rate_warmup_start_update_index: int | None = Field(
        default=None,
        ge=0,
    )
    target_shared_learning_rate_warmup_updates: int | None = Field(
        default=None,
        ge=2,
    )
    preserve_private_optimizer_state: Literal[True] = True
    reset_shared_optimizer_state: Literal[True] | None = None

    @model_validator(mode="after")
    def coherent_target_scope(self) -> Self:
        """Bind shared optimizer reset and LR fields exactly to hybrid thaw."""
        shared_values = (
            self.target_shared_learning_rate_initial,
            self.target_shared_learning_rate_target,
            self.target_shared_learning_rate_warmup_start_update_index,
            self.target_shared_learning_rate_warmup_updates,
        )
        if self.target_mode == "private_only" and (
            any(value is not None for value in shared_values)
            or self.reset_shared_optimizer_state is not None
        ):
            raise ValueError(
                "private-only optimizer transition cannot declare shared state"
            )
        if self.target_mode == "hybrid":
            if self.source_mode != "private_only":
                raise ValueError(
                    "hybrid thaw currently requires a private-only source optimizer"
                )
            if any(value is None for value in shared_values):
                raise ValueError(
                    "hybrid optimizer transition requires every shared LR field"
                )
            if self.reset_shared_optimizer_state is not True:
                raise ValueError(
                    "hybrid optimizer transition must reset shared optimizer state"
                )
            assert self.target_shared_learning_rate_initial is not None
            assert self.target_shared_learning_rate_target is not None
            if (
                self.target_shared_learning_rate_initial
                > self.target_shared_learning_rate_target
            ):
                raise ValueError(
                    "hybrid shared learning-rate warmup cannot decrease its rate"
                )
        return self


class StatelessDeckBalanceTransitionConfig(BaseModel):
    """Authorize one settled dynamic-to-weighted scheduler migration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_config_fingerprint: str
    target_config_fingerprint: str
    source_schema_version: Literal[2, 3]
    target_schema_version: Literal[2] = 2
    preserve_rolling_history: Literal[True] = True
    discard_terminal_score_events: Literal[True] = True

    @field_validator("source_config_fingerprint", "target_config_fingerprint")
    @classmethod
    def valid_config_fingerprint(cls, value: str) -> str:
        """Bind both scheduler identities to immutable full fingerprints."""
        return _fingerprint(value)

    @model_validator(mode="after")
    def changed_scheduler(self) -> Self:
        """Reject a no-op scheduler migration declaration."""
        if self.source_config_fingerprint == self.target_config_fingerprint:
            raise ValueError("deck-balance transition must change its configuration")
        return self


class StatelessCurriculumLaneCoverageRebaseConfig(BaseModel):
    """Authorize one settled, mass-preserving lane-ledger epoch change."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_config_fingerprint: str
    target_config_fingerprint: str
    source_assignment_cursor: int = Field(ge=0)
    source_lane_coverage: dict[CurriculumLane, int]
    target_lane_mix: OpponentLaneMix
    target_lane_coverage: dict[CurriculumLane, int]
    apportionment: Literal["hamilton_neutral_deficit_v1"] = (
        "hamilton_neutral_deficit_v1"
    )
    require_settled: Literal[True] = True
    preserve_assignment_cursor: Literal[True] = True
    preserve_non_lane_state: Literal[True] = True

    @field_validator("source_config_fingerprint", "target_config_fingerprint")
    @classmethod
    def valid_config_fingerprint(cls, value: str) -> str:
        """Bind both curriculum epochs to immutable full fingerprints."""
        return _fingerprint(value)

    @field_validator(
        "source_lane_coverage",
        "target_lane_coverage",
        mode="before",
    )
    @classmethod
    def exact_lane_coverage(cls, value: object) -> object:
        """Require exact, non-negative integer counters for all three lanes."""
        if not isinstance(value, Mapping):
            raise ValueError("lane coverage must be a mapping")
        lanes = ("mirror", "pfsp", "scripted")
        if set(value) != set(lanes) or any(
            not isinstance(value[lane], int)
            or isinstance(value[lane], bool)
            or value[lane] < 0
            for lane in lanes
        ):
            raise ValueError(
                "lane coverage must contain exact non-negative integer counters"
            )
        return {lane: value[lane] for lane in lanes}

    @model_validator(mode="after")
    def coherent_rebase(self) -> Self:
        """Bind the declared target to the neutral-deficit Hamilton result."""
        if self.source_config_fingerprint == self.target_config_fingerprint:
            raise ValueError("lane-coverage rebase must change curriculum identity")
        source_total = sum(self.source_lane_coverage.values())
        if source_total > self.source_assignment_cursor:
            raise ValueError("source lane coverage cannot exceed assignment cursor")
        expected = apportion_neutral_deficit_lane_coverage(
            assignment_cursor=self.source_assignment_cursor,
            coverage_total=source_total,
            lane_mix=self.target_lane_mix,
        )
        if self.target_lane_coverage != expected:
            raise ValueError(
                "target lane coverage differs from the neutral-deficit "
                "Hamilton allocation"
            )
        return self


class StatelessDeckBalanceSettings(BaseModel):
    """Rolling deficit scheduler settings resolved against the active roster."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rolling_window_decisions: int = Field(gt=0)
    assignment_seed: int = Field(ge=0)
    inflight_decision_credit: float = Field(default=1.0, ge=0.0)
    deficit_exponent: float = Field(default=1.0, gt=0.0)


class StatelessDeckAllocationConfig(BaseModel):
    """Optional primary/auxiliary collection and learner allocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["primary_auxiliary"] = "primary_auxiliary"
    primary_deck_labels: tuple[str, ...]
    primary_probability: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    learner_weighting: Literal["match_collection"] = "match_collection"

    @field_validator("primary_deck_labels")
    @classmethod
    def valid_primary_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require canonical non-empty unique source labels."""
        normalized = tuple(value.strip() for value in values)
        if (
            not normalized
            or any(not value for value in normalized)
            or normalized != tuple(sorted(set(normalized)))
        ):
            raise ValueError(
                "primary deck labels must be sorted, unique, and non-empty"
            )
        return normalized


class StatelessDynamicDeckAllocationConfig(BaseModel):
    """Bounded adaptive allocation driven by recent engine terminal scores."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["dynamic_difficulty"] = "dynamic_difficulty"
    performance_window_games: int = Field(gt=0)
    evidence_prior_games: float = Field(ge=0.0, allow_inf_nan=False)
    uniform_mix: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    difficulty_temperature: float = Field(gt=0.0, allow_inf_nan=False)
    maximum_share_ratio: float = Field(ge=1.0, allow_inf_nan=False)
    learner_weighting: Literal["match_collection"] = "match_collection"


StatelessDeckAllocationConfigValue = (
    StatelessDeckAllocationConfig | StatelessDynamicDeckAllocationConfig
)


class StatelessPerformanceConfig(BaseModel):
    """Non-resume-critical exact terminal diagnostics for the WebUI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    interval_seconds: float = Field(default=60.0, gt=0.0)
    rolling_window_minutes: int = Field(default=15, gt=0)
    recent_window_limit: int = Field(default=60, gt=0)
    parquet_shard_windows: int = Field(default=15, gt=0)
    learner_metric_history_enabled: bool = False
    tensorboard_enabled: bool = True
    tensorboard_flush_seconds: float = Field(default=30.0, gt=0.0)
    stationary_opponent_kinds: tuple[
        Literal["self_play", "frozen", "scripted"], ...
    ] = ("frozen", "scripted")

    @field_validator("interval_seconds", "tensorboard_flush_seconds")
    @classmethod
    def finite_positive_interval(cls, value: float) -> float:
        """Reject non-finite reporter intervals."""
        if not math.isfinite(value):
            raise ValueError("performance intervals must be finite")
        return value

    @field_validator("stationary_opponent_kinds")
    @classmethod
    def unique_stationary_kinds(
        cls,
        value: tuple[Literal["self_play", "frozen", "scripted"], ...],
    ) -> tuple[Literal["self_play", "frozen", "scripted"], ...]:
        """Require a non-empty unique stationary controller set."""
        if not value or len(value) != len(set(value)):
            raise ValueError("stationary opponent kinds must be non-empty and unique")
        return value

    @model_validator(mode="after")
    def aligned_windows(self) -> Self:
        """Keep rolling and persisted reporter windows mechanically coherent."""
        intervals = self.rolling_window_minutes * 60.0 / self.interval_seconds
        if intervals <= 1.0 or not math.isclose(intervals, round(intervals)):
            raise ValueError(
                "rolling window must contain more than one whole reporting interval"
            )
        if self.recent_window_limit < round(intervals):
            raise ValueError("recent window limit must cover the rolling window")
        return self


class StatelessHistoricalAnchorConfig(BaseModel):
    """One immutable archive-native policy/deck PFSP member."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    member_id: str
    snapshot_id: str
    checkpoint_path: Path
    checkpoint_size_bytes: int = Field(gt=0)
    checkpoint_sha256: str
    pilot_artifact_fingerprint: str
    bundle_fingerprint: str
    exact_deck_path: Path
    exact_deck_digest: str
    input_contract_fingerprint: str
    exact_registry_fingerprint: str
    runtime_kind: Literal["legacy_resident", "fixed_stateless_wire"] = "legacy_resident"
    belief_summary_path: Path | None = None
    belief_summary_sha256: str | None = None
    public_catalog_manifest_path: Path | None = None
    base_weight: float = Field(default=1.0, gt=0.0)
    sampling_floor: float = Field(default=0.001, ge=0.0, lt=1.0)
    device: Literal["cpu", "cuda"] = "cuda"

    @field_validator("member_id", "snapshot_id")
    @classmethod
    def non_empty_id(cls, value: str) -> str:
        """Reject ambiguous anchor aliases."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("historical anchor IDs must be non-empty")
        return normalized

    @field_validator(
        "checkpoint_sha256",
        "pilot_artifact_fingerprint",
        "bundle_fingerprint",
        "exact_deck_digest",
        "input_contract_fingerprint",
        "exact_registry_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require immutable historical bundle identity."""
        return _fingerprint(value)

    @field_validator("belief_summary_sha256")
    @classmethod
    def valid_optional_fingerprint(cls, value: str | None) -> str | None:
        """Bind an optional archive-native prior."""
        return None if value is None else _fingerprint(value)

    @model_validator(mode="after")
    def complete_belief_binding(self) -> Self:
        """Require runtime resources supported by the declared anchor kind."""
        if (self.belief_summary_path is None) != (self.belief_summary_sha256 is None):
            raise ValueError("historical belief path and fingerprint must pair")
        if (
            self.public_catalog_manifest_path is not None
            and self.runtime_kind != "fixed_stateless_wire"
        ):
            raise ValueError(
                "route-specific public catalogs require a fixed stateless anchor"
            )
        return self


class StatelessAnchorTransitionConfig(BaseModel):
    """Explicit immutable membership change for the protected anchor pool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_member_ids: tuple[str, ...]
    target_member_ids: tuple[str, ...]

    @field_validator("source_member_ids", "target_member_ids")
    @classmethod
    def valid_member_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require stable, non-empty, unique anchor membership declarations."""
        normalized = tuple(value.strip() for value in values)
        if not normalized or any(not value for value in normalized):
            raise ValueError("anchor transition member IDs must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("anchor transition member IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def changes_membership(self) -> Self:
        """Reject a declaration that authorizes no protected-pool change."""
        if set(self.source_member_ids) == set(self.target_member_ids):
            raise ValueError("anchor transition must change protected membership")
        return self


class StatelessPublicCatalogTransitionConfig(BaseModel):
    """Authorize one weight-preserving public belief catalog migration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_catalog_fingerprint: str
    target_catalog_fingerprint: str
    source_scripted_manifest_fingerprint: str | None = None
    target_scripted_manifest_fingerprint: str | None = None

    @field_validator(
        "source_catalog_fingerprint",
        "target_catalog_fingerprint",
        "source_scripted_manifest_fingerprint",
        "target_scripted_manifest_fingerprint",
    )
    @classmethod
    def valid_catalog_fingerprint(cls, value: str | None) -> str | None:
        """Bind both sides of the semantic input transition."""
        if value is None:
            return None
        return _fingerprint(value)

    @model_validator(mode="after")
    def changed_catalog(self) -> Self:
        """Reject a declaration that does not change catalog identity."""
        if self.source_catalog_fingerprint == self.target_catalog_fingerprint:
            raise ValueError("public catalog transition requires distinct identities")
        source_scripted = self.source_scripted_manifest_fingerprint
        target_scripted = self.target_scripted_manifest_fingerprint
        if (source_scripted is None) != (target_scripted is None):
            raise ValueError(
                "scripted manifest transition requires both source and target"
            )
        if source_scripted is not None and source_scripted == target_scripted:
            raise ValueError(
                "scripted manifest transition requires distinct identities"
            )
        return self


class StatelessScriptedOpponentConfig(BaseModel):
    """One versioned fixed-behavior scripted lane bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_id: str
    opponent_name: str
    artifact_fingerprint: str
    exact_deck_path: Path
    exact_deck_digest: str
    base_weight: float = Field(default=1.0, gt=0.0)

    @field_validator("opponent_id", "opponent_name")
    @classmethod
    def non_empty_id(cls, value: str) -> str:
        """Require explicit runtime and sampling names."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("scripted opponent names must be non-empty")
        return normalized

    @field_validator("artifact_fingerprint", "exact_deck_digest")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Bind script code/runtime and exact deck identities."""
        return _fingerprint(value)


class StatelessCurriculumSourceConfig(BaseModel):
    """Resolved three-lane sources and bounded past-self lifecycle settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lane_mix: OpponentLaneMix = Field(default_factory=OpponentLaneMix)
    pfsp: MatchupPfspConfig = Field(default_factory=MatchupPfspConfig)
    past_self_retention: PastSelfRetentionConfig = Field(
        default_factory=PastSelfRetentionConfig
    )
    replaceable_snapshot_capacity: int = Field(default=4, ge=1)
    assignment_seed: int = Field(ge=0)
    anchors: tuple[StatelessHistoricalAnchorConfig, ...]
    scripted_manifest_path: Path
    scripted_manifest_fingerprint: str
    scripted: tuple[StatelessScriptedOpponentConfig, ...]
    scripted_weight_overrides: dict[str, float] = Field(default_factory=dict)
    admit_past_self_after_updates: int = Field(default=1, ge=1)
    past_self_admission_interval_updates: int = Field(default=1, ge=1)
    past_self_reentry_manifest_paths: tuple[Path, ...] = ()
    retain_all_past_self: bool = False

    @model_validator(mode="after")
    def complete_lanes(self) -> Self:
        """Positive PFSP/scripted mass requires immutable lane resources."""
        if self.lane_mix.pfsp > 0.0 and not self.anchors:
            raise ValueError("PFSP lane requires at least one historical anchor")
        if self.lane_mix.scripted > 0.0 and not self.scripted:
            raise ValueError("scripted lane requires at least one bundle")
        member_ids = tuple(anchor.member_id for anchor in self.anchors)
        bundle_ids = tuple(anchor.bundle_fingerprint for anchor in self.anchors)
        scripted_ids = tuple(item.opponent_id for item in self.scripted)
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("historical anchor member IDs must be unique")
        if len(set(bundle_ids)) != len(bundle_ids):
            raise ValueError("historical anchor bundles must be unique")
        if len(set(scripted_ids)) != len(scripted_ids):
            raise ValueError("scripted opponent IDs must be unique")
        if set(self.scripted_weight_overrides) - set(scripted_ids):
            raise ValueError("scripted weight override names an unknown opponent")
        if any(
            not math.isfinite(weight) or weight <= 0.0
            for weight in self.scripted_weight_overrides.values()
        ):
            raise ValueError("scripted weight overrides must be finite and positive")
        reentry_paths = tuple(
            str(path.resolve()) for path in self.past_self_reentry_manifest_paths
        )
        if len(set(reentry_paths)) != len(reentry_paths):
            raise ValueError("past-self reentry manifests must be unique")
        return self

    @field_validator("scripted_manifest_fingerprint")
    @classmethod
    def valid_scripted_manifest_fingerprint(cls, value: str) -> str:
        """Bind the full code/deck/runtime scripted manifest."""
        return _fingerprint(value)


class StatelessResumeConfig(BaseModel):
    """Fresh, supervised, exact-resume, or explicit pair transition startup."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal[
        "fresh",
        "weights_only",
        "supervised",
        "transition",
        "bc_overlay",
        "resume",
    ] = "fresh"
    transition_action: Literal["continue_training", "materialize_only"] = (
        "continue_training"
    )
    startup_action: Literal["continue_training", "materialize_only"] = (
        "continue_training"
    )
    pair_manifest_path: Path | None = None
    supervised_artifact_manifest_path: Path | None = None
    preserve_controller_state: bool = False
    expected_source_pair_version: int | None = Field(default=None, ge=0)
    expected_source_pair_manifest_sha256: str | None = None
    expected_source_curriculum_fingerprint: str | None = None
    expected_target_curriculum_fingerprint: str | None = None
    gae_lambda_transition_from: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    optimizer_scope_transition: StatelessOptimizerScopeTransitionConfig | None = None
    deck_balance_transition: StatelessDeckBalanceTransitionConfig | None = None
    curriculum_lane_coverage_rebase: (
        StatelessCurriculumLaneCoverageRebaseConfig | None
    ) = None
    registry_transition: StatelessRegistryTransitionConfig | None = None
    topology_transition: StatelessTopologyTransitionDeclaration | None = None
    public_catalog_transition: StatelessPublicCatalogTransitionConfig | None = None
    anchor_transition: StatelessAnchorTransitionConfig | None = None
    bc_overlay: StatelessBcOverlayDeclaration | None = None
    supervised: StatelessSupervisedStartupDeclaration | None = None

    @field_validator(
        "expected_source_pair_manifest_sha256",
        "expected_source_curriculum_fingerprint",
        "expected_target_curriculum_fingerprint",
    )
    @classmethod
    def valid_transition_fingerprint(cls, value: str | None) -> str | None:
        """Bind transitions to complete immutable source identities."""
        return None if value is None else _fingerprint(value)

    @model_validator(mode="after")
    def coherent_mode(self) -> Self:
        """Keep model-only initialization distinct from exact recovery."""
        if self.mode == "fresh" and (
            self.pair_manifest_path is not None
            or self.supervised_artifact_manifest_path is not None
        ):
            raise ValueError("fresh stateless training cannot name a source artifact")
        if self.mode == "supervised" and (
            self.supervised_artifact_manifest_path is None
            or self.pair_manifest_path is not None
            or self.supervised is None
        ):
            raise ValueError(
                "supervised stateless initialization requires its manifest "
                "and immutable declaration"
            )
        if self.mode in {"weights_only", "resume", "transition"} and (
            self.pair_manifest_path is None
            or self.supervised_artifact_manifest_path is not None
        ):
            raise ValueError("pair-based stateless startup requires a pair manifest")
        if self.mode == "bc_overlay" and (
            self.pair_manifest_path is None
            or self.supervised_artifact_manifest_path is None
            or self.bc_overlay is None
            or not self.preserve_controller_state
        ):
            raise ValueError(
                "BC overlay requires a pair, supervised artifact, declaration, "
                "and controller-state preservation"
            )
        if self.mode != "bc_overlay" and self.bc_overlay is not None:
            raise ValueError("BC overlay declaration is available only in overlay mode")
        if self.mode != "supervised" and self.supervised is not None:
            raise ValueError(
                "supervised startup declaration is available only in supervised mode"
            )
        if self.preserve_controller_state and self.mode not in {
            "transition",
            "bc_overlay",
        }:
            raise ValueError(
                "controller-state migration requires a transition or BC overlay"
            )
        if self.gae_lambda_transition_from is not None and (
            self.mode != "transition" or not self.preserve_controller_state
        ):
            raise ValueError(
                "GAE lambda migration requires a controller-preserving transition"
            )
        if self.optimizer_scope_transition is not None and (
            self.mode != "transition"
            or not self.preserve_controller_state
            or self.transition_action != "continue_training"
        ):
            raise ValueError(
                "optimizer-scope migration requires a controller-preserving "
                "training transition"
            )
        if self.deck_balance_transition is not None and (
            self.mode != "transition"
            or not self.preserve_controller_state
            or self.transition_action != "continue_training"
        ):
            raise ValueError(
                "deck-balance migration requires a controller-preserving "
                "training transition"
            )
        if self.curriculum_lane_coverage_rebase is not None and (
            self.mode != "transition"
            or not self.preserve_controller_state
            or self.transition_action != "continue_training"
        ):
            raise ValueError(
                "lane-coverage rebase requires a controller-preserving "
                "training transition"
            )
        if self.registry_transition is not None and self.mode != "transition":
            raise ValueError(
                "stateless registry migration is available only for transitions"
            )
        if self.topology_transition is not None and self.mode != "transition":
            raise ValueError(
                "stateless topology migration is available only for transitions"
            )
        if self.public_catalog_transition is not None and (
            self.mode != "transition" or not self.preserve_controller_state
        ):
            raise ValueError(
                "public catalog migration requires a controller-preserving transition"
            )
        if self.anchor_transition is not None and (
            self.mode != "transition" or not self.preserve_controller_state
        ):
            raise ValueError(
                "anchor migration requires a controller-preserving transition"
            )
        identity_only_materialization = (
            self.topology_transition is None
            and self.registry_transition is None
            and self.public_catalog_transition is None
            and self.anchor_transition is None
            and self.gae_lambda_transition_from is None
            and self.optimizer_scope_transition is None
            and self.deck_balance_transition is None
            and self.curriculum_lane_coverage_rebase is None
        )
        if self.transition_action == "materialize_only" and (
            self.mode != "transition"
            or not self.preserve_controller_state
            or (self.topology_transition is None and not identity_only_materialization)
        ):
            raise ValueError(
                "materialize-only startup requires a controller-preserving "
                "topology or identity-only transition"
            )
        if self.startup_action == "materialize_only" and self.mode != "supervised":
            raise ValueError(
                "startup materialization is available only in supervised mode"
            )
        if self.mode == "supervised" and self.transition_action != "continue_training":
            raise ValueError(
                "supervised startup uses startup_action, not transition_action"
            )
        if self.mode != "supervised" and self.startup_action != "continue_training":
            raise ValueError("startup_action is available only in supervised mode")
        exclusive_transitions = (
            self.registry_transition,
            self.topology_transition,
            self.public_catalog_transition,
            self.optimizer_scope_transition,
        )
        if sum(item is not None for item in exclusive_transitions) > 1:
            raise ValueError(
                "stateless registry, topology, public catalog, and optimizer "
                "scope transitions are mutually exclusive"
            )
        source_bindings = (
            self.expected_source_pair_version,
            self.expected_source_pair_manifest_sha256,
            self.expected_source_curriculum_fingerprint,
            self.expected_target_curriculum_fingerprint,
        )
        source_bound_modes = {"transition", "bc_overlay"}
        missing_source_binding = any(binding is None for binding in source_bindings)
        if self.mode == "transition" and missing_source_binding:
            raise ValueError(
                "transition startup requires exact source pair and curriculum bindings"
            )
        if self.mode == "bc_overlay" and missing_source_binding:
            raise ValueError(
                "BC overlay requires exact source pair and curriculum bindings"
            )
        lane_rebase = self.curriculum_lane_coverage_rebase
        if lane_rebase is not None and (
            lane_rebase.source_config_fingerprint
            != self.expected_source_curriculum_fingerprint
            or lane_rebase.target_config_fingerprint
            != self.expected_target_curriculum_fingerprint
        ):
            raise ValueError(
                "lane-coverage rebase differs from its curriculum bindings"
            )
        if (
            self.mode == "bc_overlay"
            and self.bc_overlay is not None
            and (
                self.expected_source_pair_version != self.bc_overlay.source_pair_version
                or self.expected_source_pair_manifest_sha256
                != self.bc_overlay.source_pair_manifest_sha256
            )
        ):
            raise ValueError("BC overlay declaration differs from its source binding")
        if self.mode not in source_bound_modes and any(
            binding is not None for binding in source_bindings
        ):
            raise ValueError(
                "source pair and curriculum bindings require a transition or BC overlay"
            )
        return self


class SimpleStatelessTrainingConfig(BaseModel):
    """Unique clean-lineage trainer profile validated after Hydra compose."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trainer: Literal["simple_stateless"] = "simple_stateless"
    run: TrainingRunConfig
    output_dir: Path | None = None
    device: Literal["cuda"] = "cuda"
    validate_only: bool = False
    model: SimpleStatelessModelConfig
    private_deck_registry: PrivateDeckRegistrySourceConfig
    public_deck_catalog: PublicDeckCatalogConfig
    collection: StatelessCollectionConfig
    learner_runtime: StatelessLearnerRuntimeConfig = Field(
        default_factory=StatelessLearnerRuntimeConfig
    )
    optimizer_scope: StatelessOptimizerScopeConfig = Field(
        default_factory=StatelessOptimizerScopeConfig
    )
    deck_balance: StatelessDeckBalanceSettings
    deck_allocation: StatelessDeckAllocationConfigValue | None = None
    performance: StatelessPerformanceConfig = Field(
        default_factory=StatelessPerformanceConfig
    )
    native_distributed: StatelessNativeDistributedConfig = Field(
        default_factory=StatelessNativeDistributedConfig
    )
    opponent_pool_v2: StatelessOpponentPoolV2Config = Field(
        default_factory=StatelessOpponentPoolV2Config
    )
    curriculum: StatelessCurriculumSourceConfig
    ppo: SimpleStatelessPpoConfig
    resume: StatelessResumeConfig = Field(default_factory=StatelessResumeConfig)
    hydra: dict[str, object] | None = None

    @model_validator(mode="after")
    def optimizer_scope_matches_transition(self) -> Self:
        """Cross-bind a scope migration to the target optimizer declaration."""
        declaration = self.resume.optimizer_scope_transition
        if declaration is None:
            return self
        if (
            self.optimizer_scope.mode != declaration.target_mode
            or self.optimizer_scope.private_learning_rate
            != declaration.target_private_learning_rate
            or self.optimizer_scope.shared_learning_rate_initial
            != declaration.target_shared_learning_rate_initial
            or self.optimizer_scope.shared_learning_rate_target
            != declaration.target_shared_learning_rate_target
            or self.optimizer_scope.shared_learning_rate_warmup_start_update_index
            != declaration.target_shared_learning_rate_warmup_start_update_index
            or self.optimizer_scope.shared_learning_rate_warmup_updates
            != declaration.target_shared_learning_rate_warmup_updates
        ):
            raise ValueError(
                "optimizer-scope transition differs from the target optimizer scope"
            )
        return self

    @model_validator(mode="after")
    def curriculum_lane_mix_matches_rebase(self) -> Self:
        """Cross-bind a lane-ledger rebase to the target lane mixture."""
        declaration = self.resume.curriculum_lane_coverage_rebase
        if (
            declaration is not None
            and self.curriculum.lane_mix != declaration.target_lane_mix
        ):
            raise ValueError(
                "lane-coverage rebase differs from the target curriculum lane mix"
            )
        return self

    @model_validator(mode="after")
    def unresolved_model_has_no_routes(self) -> Self:
        """Hydra declares deck sources; runtime resolves their path-free routes."""
        if (
            self.model.exact_routes
            or self.model.resolved_registry_sha256 is not None
            or self.model.family_routes
            or self.model.resolved_family_registry_sha256 is not None
        ):
            raise ValueError(
                "training profile must not hand-copy resolved exact routes"
            )
        if self.model.export_mode != "routed":
            raise ValueError("training model must use routed export mode")
        if not self.private_deck_registry.decks:
            raise ValueError("stateless training requires active exact deck sources")
        if any(entry.expert_id is None for entry in self.private_deck_registry.decks):
            raise ValueError("every active deck needs a new explicit expert lineage")
        has_family_ids = tuple(
            entry.family_id is not None for entry in self.private_deck_registry.decks
        )
        if uses_family_private_topology(self.model):
            if not all(has_family_ids):
                raise ValueError(
                    "every family-private deck needs an explicit family lineage"
                )
        elif any(has_family_ids):
            raise ValueError("family lineages require the family-private architecture")
        if uses_generalist_sequence(self.model):
            if self.collection.backend not in {
                "python_parallel",
                "native",
                "native_banked",
                "native_distributed",
            } or (
                self.collection.pipeline_mode == "one_version_lag"
                and self.collection.backend
                not in {"native_banked", "native_distributed"}
            ):
                raise ValueError(
                    "generalist sequence V1 requires synchronous Python/native "
                    "collection or one-version-lag banked/distributed native "
                    "collection"
                )
        elif self.collection.native_sequence_rollout_precision != "fp32":
            raise ValueError(
                "native BF16 sequence rollout requires a generalist sequence model"
            )
        if (
            self.collection.native_sequence_rollout_precision == "bf16"
            and self.collection.backend
            not in {"native", "native_banked", "native_distributed"}
        ):
            raise ValueError("native BF16 sequence rollout requires the native backend")
        if self.collection.backend == "native_banked" and (
            self.curriculum.pfsp.active_artifacts_per_window is None
            or self.curriculum.pfsp.active_artifacts_per_window
            > self.collection.native_engine_shards // 2
        ):
            raise ValueError(
                "banked native collection cannot schedule more active frozen "
                "artifacts than fixed execution banks"
            )
        if self.collection.pipeline_mode == "one_version_lag":
            integrated_hybrid = (
                self.collection.backend == "hybrid"
                and self.collection.native_process_workers > 1
                and self.collection.integrate_scripted_current_inference
            )
            banked_generalist = (
                self.collection.backend in {"native_banked", "native_distributed"}
                and uses_generalist_sequence(self.model)
                and self.collection.native_process_workers == 1
                and self.collection.native_sequence_rollout_precision == "bf16"
            )
            if not (integrated_hybrid or banked_generalist):
                raise ValueError(
                    "one-version-lag pipeline requires integrated hybrid "
                    "multi-process collection or one-worker BF16 banked "
                    "generalist sequence collection"
                )
            if self.ppo.maximum_version_age < 1:
                raise ValueError(
                    "one-version-lag pipeline requires maximum_version_age >= 1"
                )
        if self.collection.backend == "native_distributed" and not (
            self.native_distributed.quorum.required_worker_ids
            and self.native_distributed.worker_profiles
            and self.collection.native_trainable_decision_budget is not None
            and self.collection.pipeline_mode == "one_version_lag"
            and self.collection.native_sequence_rollout_precision == "bf16"
            and uses_generalist_sequence(self.model)
        ):
            raise ValueError(
                "native distributed collection requires explicit workers and "
                "profiles, a global decision target, and one-version-lag BF16 "
                "generalist sequence rollout"
            )
        if (
            any(
                anchor.runtime_kind == "fixed_stateless_wire"
                for anchor in self.curriculum.anchors
            )
            and self.collection.backend != "native_distributed"
        ):
            raise ValueError(
                "fixed stateless anchors require native distributed collection"
            )
        if any(
            anchor.public_catalog_manifest_path is not None
            for anchor in self.curriculum.anchors
        ) and any(
            tier.native_process_workers > 1
            for profile in self.native_distributed.worker_profiles.values()
            for tier in profile.capacity_tiers
        ):
            raise ValueError(
                "route-specific public catalogs require direct native worker tiers"
            )
        if self.collection.native_shard_protocol_version == 3 and not (
            self.collection.backend == "native_distributed"
            and self.opponent_pool_v2.enabled
            and self.opponent_pool_v2.behavior_version in {2, 3, 4, 5}
        ):
            raise ValueError(
                "native shard protocol V3 requires the lineage opponent pool"
            )
        if (
            self.opponent_pool_v2.enabled
            and self.opponent_pool_v2.behavior_version in {3, 4, 5}
            and self.collection.native_shard_protocol_version != 3
        ):
            raise ValueError(
                "joint adaptive opponent allocation requires aggregate quota shards"
            )
        if self.opponent_pool_v2.enabled:
            if self.collection.backend != "native_distributed":
                raise ValueError(
                    "opponent-pool V2 requires native distributed collection"
                )
            if self.opponent_pool_v2.behavior_version in {2, 3, 4, 5}:
                if not self.curriculum.retain_all_past_self:
                    raise ValueError(
                        "lineage opponent pool requires durable full-history retention"
                    )
                if self.opponent_pool_v2.behavior_version in {3, 4, 5}:
                    return self
                slots = {
                    item.stratum: item.artifact_slots
                    for item in self.opponent_pool_v2.planner_policy.strata
                }
                if slots != {
                    "protected": 2,
                    "counter_frontier": 1,
                    "recent": 1,
                    "age_diverse": 1,
                }:
                    raise ValueError(
                        "lineage opponent pool must select exactly five artifacts"
                    )
                return self
            protected = len(self.curriculum.anchors)
            expected_active = protected + self.opponent_pool_v2.past_self_artifacts
            if expected_active > self.opponent_pool_v2.maximum_active_artifacts:
                raise ValueError(
                    "opponent-pool V2 archive partition exceeds its active bound"
                )
            slots = {
                item.stratum: item.artifact_slots
                for item in self.opponent_pool_v2.planner_policy.strata
            }
            available: dict[Literal["protected", "recent", "age_diverse"], int] = {
                "protected": protected,
                "recent": self.opponent_pool_v2.recent_artifacts,
                "age_diverse": (
                    self.opponent_pool_v2.past_self_artifacts
                    - self.opponent_pool_v2.recent_artifacts
                ),
            }
            if set(slots) != set(available) or any(
                slots[stratum] > count for stratum, count in available.items()
            ):
                raise ValueError(
                    "opponent-pool V2 planner strata differ from archive partition"
                )
        return self


def _fingerprint(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("artifact identity must be lowercase SHA-256")
    return normalized


__all__ = [
    "MAX_NATIVE_ENGINE_SHARDS_PER_PROCESS",
    "MAX_NATIVE_PROCESS_WORKERS",
    "SimpleStatelessTrainingConfig",
    "StatelessAnchorTransitionConfig",
    "StatelessBcOverlayDeclaration",
    "StatelessCollectionConfig",
    "StatelessCurriculumSourceConfig",
    "StatelessCurriculumLaneCoverageRebaseConfig",
    "StatelessDeckAllocationConfig",
    "StatelessDeckAllocationConfigValue",
    "StatelessDeckBalanceSettings",
    "StatelessDeckBalanceTransitionConfig",
    "StatelessDynamicDeckAllocationConfig",
    "StatelessExactStrategyInitialization",
    "StatelessHistoricalAnchorConfig",
    "StatelessLearnerRuntimeConfig",
    "StatelessOptimizerScopeConfig",
    "StatelessOptimizerScopeTransitionConfig",
    "StatelessOpponentPoolV2Config",
    "StatelessNativeDistributedConfig",
    "StatelessNativeDistributedQuorumConfig",
    "StatelessNativeDistributedRetryConfig",
    "StatelessNativeDistributedStatusConfig",
    "StatelessNativeWorkerCudaCacheConfig",
    "StatelessNativeWorkerHostMemoryConfig",
    "StatelessNativeDistributedTransportConfig",
    "StatelessNativeWorkerCapacityTierConfig",
    "StatelessNativeWorkerProfileConfig",
    "StatelessPerformanceConfig",
    "StatelessPublicCatalogTransitionConfig",
    "StatelessRegistryTransitionConfig",
    "StatelessResumeConfig",
    "StatelessScriptedOpponentConfig",
    "StatelessSupervisedStartupDeclaration",
    "StatelessTopologyTransitionDeclaration",
]
