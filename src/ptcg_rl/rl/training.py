"""Hydra-backed synchronous PPO training runner."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import importlib
import json
import math
import os
import queue
import random
import shutil
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, closing, nullcontext, suppress
from dataclasses import asdict, dataclass, field, replace
from functools import cache, partial
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast

import torch
import torch.multiprocessing as torch_mp
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ptcg_rl.context import OpponentBeliefFeatureProducer
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.decks.registry import (
    PrivateDeckRegistrySourceConfig,
    resolve_deck_expert_registry,
    resolve_private_registry,
    validate_active_exact_strategy_routes,
)
from ptcg_rl.engine.vector_battle import (
    DeckPair,
    DeckPairSampler,
    FinishedGame,
    VectorBattlePool,
    VectorGame,
)
from ptcg_rl.evaluation.search_identity import file_sha256
from ptcg_rl.model import (
    DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
    DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
    DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION,
    LEGACY_STATE_ENCODER_MISSING_KEYS,
    AgentNetworkConfig,
    AgentPolicyValueNet,
    build_agent_policy_value_net,
    load_agent_policy_value_state_dict,
    policy_input_schema_metadata,
)
from ptcg_rl.opponents import BattleAgent, OpponentSpec, build_opponent
from ptcg_rl.profiling import StageTimer, time_stage
from ptcg_rl.rl.actor import (
    ActorLoopConfig,
    ActorLoopStats,
    RecurrentStaleGameWatcher,
    TrajectoryQueueProducerConfig,
    WeightPollingLoader,
    run_actor_loop,
)
from ptcg_rl.rl.advantage import GaeConfig
from ptcg_rl.rl.amortized_policy_iteration.belief_reanalysis import (
    NativeReanalysisJob,
    NativeReanalysisResult,
    ReanalysisRoot,
    native_reanalysis_worker,
    rebind_reanalysis_game_id,
)
from ptcg_rl.rl.amortized_policy_iteration.contracts import (
    AmortizedPolicyIterationConfig,
)
from ptcg_rl.rl.amortized_policy_iteration.improvement_actor import (
    ImprovementRolloutPolicy,
)
from ptcg_rl.rl.amortized_policy_iteration.learner_update import (
    AmortizedPolicyIterationLearner,
    PolicyIterationUpdate,
)
from ptcg_rl.rl.amortized_policy_iteration.reanalysis_coordinator import (
    build_native_reanalysis_jobs,
)
from ptcg_rl.rl.amortized_policy_iteration.replay import ReplayIdentity
from ptcg_rl.rl.amortized_policy_iteration.replay_store import (
    PolicyIterationReplayStore,
)
from ptcg_rl.rl.async_runtime import (
    ActorRecycleRequest,
    AsyncActorLearnerSupervisorConfig,
    ManagedProcess,
    stop_managed_processes,
    supervise_actor_group_learner,
)
from ptcg_rl.rl.background_engine_teacher import BackgroundEngineTeacherProducer
from ptcg_rl.rl.background_macro_teacher import BackgroundMacroTeacherProducer
from ptcg_rl.rl.checkpoint_pair import (
    AsyncCheckpointPairPublisher,
    PublishedCheckpointPair,
)
from ptcg_rl.rl.collection import (
    MinCountRolloutPolicy,
    ModelRolloutPolicy,
    RolloutDeckConfig,
    RolloutPolicyKind,
)
from ptcg_rl.rl.curriculum import (
    AssignmentScheduleConfig,
    CurriculumConfig,
    CurriculumOutcome,
    CurriculumSampler,
    FrozenPoolMember,
    GameAssignment,
    add_frozen_pool_member,
    consume_frozen_pool_additions,
    fingerprint_checkpoint,
    read_frozen_pool_state,
    write_frozen_pool_state,
)
from ptcg_rl.rl.deck_transition import (
    DeckRegistryTransitionConfig,
    DeckRegistryTransitionPlan,
    build_deck_registry_transition_plan,
    migrate_deck_registry_weights,
    private_state_expert_id,
)
from ptcg_rl.rl.deck_transition_optimizer import (
    transplant_deck_registry_optimizer_state,
    validate_deck_registry_source_state,
)
from ptcg_rl.rl.engine_teacher_policy import DecodePolicy
from ptcg_rl.rl.experience import (
    GameTrajectory,
    TensorTrajectoryRecorder,
    compact_game_trajectory,
    rebind_game_trajectory_id,
)
from ptcg_rl.rl.factual import FactualLaneConfig
from ptcg_rl.rl.factual_metrics import aggregate_factual_updates
from ptcg_rl.rl.frozen_league import (
    FrozenLeagueConfig,
    queue_frozen_league_candidate,
)
from ptcg_rl.rl.frozen_pool import (
    FrozenPolicyPool,
    FrozenPolicyPoolConfig,
    FrozenPolicyPoolUpdate,
    load_frozen_rollout_policy,
)
from ptcg_rl.rl.inference_server import (
    InferenceClientConfig,
    InferencePolicyRegistry,
    InferenceServerConfig,
    RemoteInferenceClientState,
    RemoteInferencePolicy,
    run_inference_server_step,
)
from ptcg_rl.rl.league_promotion import (
    LeaguePromotionController,
    LeaguePromotionState,
)
from ptcg_rl.rl.learner import (
    FixedKlReference,
    LearnerBatchConfig,
    LearnerBatchResult,
    LearnerIterationResult,
    PublishedWeights,
    Schema9LearnerBatchConfig,
    WeightPublisher,
    WeightPublisherConfig,
    build_ppo_minibatches,
    capture_fixed_kl_reference,
    reshuffle_ppo_minibatches,
    run_ppo_iteration,
    summarize_learner_window_staleness,
)
from ptcg_rl.rl.macro_credit import (
    MacroCreditConfig,
    MacroTeacherIdentityConfig,
)
from ptcg_rl.rl.macro_credit_metrics import aggregate_macro_credit_updates
from ptcg_rl.rl.model_publication import PreparedModelState, prepare_model_state
from ptcg_rl.rl.online_engine_teacher import (
    OnlineEngineTeacherConfig,
    OnlineEngineTeacherProducer,
)
from ptcg_rl.rl.planner_metrics import (
    aggregate_planner_updates,
    planner_imitation_metrics_summary,
    planner_update_metrics_summary,
)
from ptcg_rl.rl.planner_runtime_factory import (
    PlannerBehaviorRuntime,
    create_planner_behavior_runtime,
)
from ptcg_rl.rl.planner_runtime_identity import ResolvedPlannerRuntimeConfig
from ptcg_rl.rl.ppo import (
    AnchorStepLogitsCache,
    PpoConfig,
    PpoUpdateResult,
    ensure_finite_model,
    load_frozen_anchor_model,
)
from ptcg_rl.rl.rollout import (
    RolloutActors,
    RolloutBeliefConfig,
    RolloutPolicy,
    RolloutProbeConfig,
    RolloutStepper,
    VectorPoolLike,
)
from ptcg_rl.rl.shared_weights import (
    SharedMemoryWeightLoader,
    SharedMemoryWeightPublisher,
    SharedMemoryWeightPublisherConfig,
    SharedMemoryWeights,
)
from ptcg_rl.rl.teacher_metrics import aggregate_engine_teacher_updates
from ptcg_rl.rl.tensorboard import (
    TensorboardMetricWriter,
    tensorboard_root_dir,
    write_learner_tensorboard,
    write_runtime_tensorboard,
)
from ptcg_rl.rl.training_resume import (
    TrainingProgress,
    TrainingResumeConfig,
    adopt_training_state_pointer,
    restore_training_progress,
)
from ptcg_rl.rl.trajectory import (
    CompletedTrajectory,
    TrajectoryRecorder,
    TrajectoryShardWriter,
    TrajectoryWriteResult,
)
from ptcg_rl.rl.trajectory_transport import SharedTrajectoryRing
from ptcg_rl.training.run_config import (
    TrainingRunConfig,
    resolve_training_output_dir,
    resolved_training_config_dump,
)

if TYPE_CHECKING:
    from ptcg_rl.rl.performance import TrainingPerformanceReporter
    from ptcg_rl.rl.performance_state import PerformanceReporterConfig

ExecutionMode = Literal["sync", "async", "distributed_async"]
RLTrainOpponentMode = Literal["self_play", "curriculum"]

RLTrainPoolFactory = Callable[
    [int, DeckPairSampler],
    AbstractContextManager[VectorPoolLike],
]

_LEGACY_TRAINING_CHECKPOINT_MISSING_KEYS = frozenset(
    {
        "opponent_hand_head.weight",
        "opponent_hand_head.bias",
        *LEGACY_STATE_ENCODER_MISSING_KEYS,
    }
)
_RUNTIME_MONITOR_TAIL_CAPACITY = 4096
_OLDEST_LIVE_GAMES_LIMIT = 4


class RLTrainExecutionConfig(BaseModel):
    """Execution mode for RL training."""

    model_config = ConfigDict(extra="forbid")

    mode: ExecutionMode = "sync"
    actors: int = 1
    trajectory_queue_maxsize: int = 2048
    trajectory_transport: Literal["queue", "shared_memory_ring"] = "queue"
    trajectory_ring_slots: int = 64
    trajectory_ring_slot_bytes: int = 8 * 1024 * 1024
    trajectory_ring_allow_inline_oversize: bool = False
    collection_flow_control: bool = False
    collection_high_watermark_decisions: int | None = None
    collection_resume_watermark_decisions: int | None = None
    gpu_service_mode: Literal["separate_process", "learner_thread"] = "separate_process"
    inference_request_queue_maxsize: int = 256
    inference_client_max_inflight_batch_decisions: int = 0
    curriculum_bootstrap_state_path: Path | None = None
    curriculum_bootstrap_state_sha256: str | None = None
    league_promotion_bootstrap_state_path: Path | None = None
    league_promotion_bootstrap_state_sha256: str | None = None
    learner_queue_get_timeout_seconds: float = 60.0
    inference_server: bool = False
    gpu_worker_torch_threads: int | None = None
    profile_actor_queue_pickle: bool = False
    archive_trajectories: bool = False
    archive_rows_per_shard: int = 65_536
    archive_compression: str = "zstd"
    inference_cuda_memory_trim_interval_steps: int | None = 64
    learner_cuda_memory_trim_interval_versions: int | None = 1
    async_supervisor: AsyncActorLearnerSupervisorConfig = Field(
        default_factory=AsyncActorLearnerSupervisorConfig
    )

    @field_validator(
        "trajectory_queue_maxsize",
        "trajectory_ring_slots",
        "trajectory_ring_slot_bytes",
        "inference_request_queue_maxsize",
        "archive_rows_per_shard",
    )
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject invalid positive integer settings."""
        if value <= 0:
            raise ValueError("execution integer settings must be positive")
        return value

    @field_validator(
        "collection_high_watermark_decisions",
        "collection_resume_watermark_decisions",
    )
    @classmethod
    def valid_optional_collection_watermark(cls, value: int | None) -> int | None:
        """Reject invalid optional decision-credit watermarks."""
        if value is not None and value <= 0:
            raise ValueError("collection watermarks must be positive when set")
        return value

    @model_validator(mode="after")
    def valid_collection_flow_control(self) -> Self:
        """Require a complete ordered high/low watermark pair when enabled."""
        high = self.collection_high_watermark_decisions
        resume = self.collection_resume_watermark_decisions
        if not self.collection_flow_control:
            if high is not None or resume is not None:
                raise ValueError(
                    "collection watermarks require collection_flow_control"
                )
            return self
        if high is None or resume is None:
            raise ValueError(
                "collection_flow_control requires both decision watermarks"
            )
        if resume >= high:
            raise ValueError(
                "collection resume watermark must be below the high watermark"
            )
        return self

    @model_validator(mode="after")
    def valid_curriculum_bootstrap(self) -> Self:
        """Bind optional run-local curriculum bootstraps to exact bytes."""
        pairs = (
            (
                "curriculum",
                self.curriculum_bootstrap_state_path,
                self.curriculum_bootstrap_state_sha256,
            ),
            (
                "league promotion",
                self.league_promotion_bootstrap_state_path,
                self.league_promotion_bootstrap_state_sha256,
            ),
        )
        for label, path, fingerprint in pairs:
            if (path is None) != (fingerprint is None):
                raise ValueError(
                    f"{label} bootstrap path and SHA256 must be configured together"
                )
            if fingerprint is not None and (
                len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
            ):
                raise ValueError(f"{label} bootstrap SHA256 must be lowercase hex")
        return self

    @field_validator("actors")
    @classmethod
    def valid_non_negative_actors(cls, value: int) -> int:
        """Reject invalid actor counts."""
        if value < 0:
            raise ValueError("execution actors must be non-negative")
        return value

    @field_validator("inference_client_max_inflight_batch_decisions")
    @classmethod
    def valid_non_negative_inflight_decisions(cls, value: int) -> int:
        """Allow zero to retain the legacy one-request-at-a-time actor path."""
        if value < 0:
            raise ValueError(
                "inference_client_max_inflight_batch_decisions must be non-negative"
            )
        return value

    @field_validator("learner_queue_get_timeout_seconds")
    @classmethod
    def valid_queue_get_timeout(cls, value: float) -> float:
        """Reject invalid learner queue get timeouts."""
        if value <= 0.0:
            raise ValueError("learner_queue_get_timeout_seconds must be positive")
        return value

    @field_validator("gpu_worker_torch_threads")
    @classmethod
    def valid_optional_positive_int(cls, value: int | None) -> int | None:
        """Reject invalid optional GPU worker thread caps."""
        if value is not None and value <= 0:
            raise ValueError("gpu_worker_torch_threads must be positive when set")
        return value

    @field_validator(
        "inference_cuda_memory_trim_interval_steps",
        "learner_cuda_memory_trim_interval_versions",
    )
    @classmethod
    def valid_optional_positive_trim_interval(
        cls,
        value: int | None,
    ) -> int | None:
        """Reject invalid optional CUDA memory trim intervals."""
        if value is not None and value <= 0:
            raise ValueError("CUDA memory trim intervals must be positive when set")
        return value

    @field_validator("archive_compression")
    @classmethod
    def valid_archive_compression(cls, value: str) -> str:
        """Reject empty archive compression names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("archive_compression must be non-empty")
        return cleaned


class DistributedTransportConfig(BaseModel):
    """Network transport settings for distributed async rollout."""

    model_config = ConfigDict(extra="forbid")

    bind_host: str = "0.0.0.0"
    coordinator_host: str = "127.0.0.1"
    trajectory_port: int = 47_300
    weight_port: int = 47_301
    trajectory_batch_decisions: int = 2048
    trajectory_flush_interval_seconds: float = 0.25
    weight_poll_interval_seconds: float = 5.0
    weight_publish_interval_versions: int = 5
    queue_high_watermark_ratio: float = 0.85
    queue_resume_watermark_ratio: float = 0.50
    flow_control_sleep_seconds: float = 0.25

    @field_validator("bind_host", "coordinator_host")
    @classmethod
    def valid_non_empty_host(cls, value: str) -> str:
        """Reject empty host strings."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("distributed host strings must be non-empty")
        return cleaned

    @field_validator("trajectory_port", "weight_port")
    @classmethod
    def valid_port(cls, value: int) -> int:
        """Reject invalid TCP ports."""
        if value <= 0 or value > 65_535:
            raise ValueError("distributed ports must be in [1, 65535]")
        return value

    @field_validator("trajectory_batch_decisions", "weight_publish_interval_versions")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject invalid positive integer settings."""
        if value <= 0:
            raise ValueError("distributed integer settings must be positive")
        return value

    @field_validator(
        "trajectory_flush_interval_seconds",
        "weight_poll_interval_seconds",
        "flow_control_sleep_seconds",
    )
    @classmethod
    def valid_positive_float(cls, value: float) -> float:
        """Reject invalid positive float settings."""
        if value <= 0.0:
            raise ValueError("distributed interval settings must be positive")
        return value

    @model_validator(mode="after")
    def valid_flow_control_watermarks(self) -> Self:
        """Reject invalid distributed queue watermark ratios."""
        if not 0.0 <= self.queue_resume_watermark_ratio <= 1.0:
            raise ValueError("queue_resume_watermark_ratio must be in [0, 1]")
        if not 0.0 <= self.queue_high_watermark_ratio <= 1.0:
            raise ValueError("queue_high_watermark_ratio must be in [0, 1]")
        if self.queue_resume_watermark_ratio > self.queue_high_watermark_ratio:
            raise ValueError(
                "queue_resume_watermark_ratio must be <= queue_high_watermark_ratio"
            )
        return self


class DistributedWorkerConfig(BaseModel):
    """Static metadata for an optional remote actor worker."""

    model_config = ConfigDict(extra="forbid")

    worker_id: str
    host: str
    repo_path: Path
    python_executable: str = "python3"
    actors: int | None = None
    num_concurrent_games: int | None = None
    device: str = "auto"

    @field_validator("worker_id", "host", "python_executable", "device")
    @classmethod
    def valid_non_empty_string(cls, value: str) -> str:
        """Reject empty worker strings."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("distributed worker strings must be non-empty")
        return cleaned

    @field_validator("actors", "num_concurrent_games")
    @classmethod
    def valid_optional_positive_int(cls, value: int | None) -> int | None:
        """Reject invalid optional worker overrides."""
        if value is not None and value <= 0:
            raise ValueError("distributed worker overrides must be positive")
        return value


class DistributedTrainConfig(BaseModel):
    """Distributed async rollout settings."""

    model_config = ConfigDict(extra="forbid")

    coordinator_enabled: bool = False
    worker_enabled: bool = False
    worker_id: str = "worker-0"
    remote_max_staleness: int = 8
    transport: DistributedTransportConfig = Field(
        default_factory=DistributedTransportConfig
    )
    workers: tuple[DistributedWorkerConfig, ...] = ()

    @field_validator("worker_id")
    @classmethod
    def valid_worker_id(cls, value: str) -> str:
        """Reject empty worker ids."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("worker_id must be non-empty")
        return cleaned

    @field_validator("remote_max_staleness")
    @classmethod
    def valid_remote_max_staleness(cls, value: int) -> int:
        """Reject invalid remote staleness limits."""
        if value < 0:
            raise ValueError("remote_max_staleness must be non-negative")
        return value


class RLTrainCollectionConfig(BaseModel):
    """Rollout collection settings for synchronous PPO training."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    training_iterations: int = 1
    iteration_decisions: int = 49_152
    ppo_epochs: int = 2
    microbatch_size: int = Field(
        default=1024,
        validation_alias=AliasChoices("microbatch_size", "minibatch_size"),
    )
    gradient_accumulation_steps: int = 1
    sampling_temperature: float = 1.0
    num_concurrent_games: int = 64
    max_collect_iterations: int = 100_000
    policy_kind: RolloutPolicyKind = "model"
    opponent_mode: RLTrainOpponentMode = "self_play"
    decks: RolloutDeckConfig = Field(default_factory=RolloutDeckConfig)

    @field_validator(
        "iteration_decisions",
        "training_iterations",
        "ppo_epochs",
        "microbatch_size",
        "gradient_accumulation_steps",
        "num_concurrent_games",
        "max_collect_iterations",
    )
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive collection counters."""
        if value <= 0:
            raise ValueError("collection counters must be positive")
        return value

    @property
    def minibatch_size(self) -> int:
        """Return the microbatch size under the legacy attribute name."""
        return self.microbatch_size

    @property
    def effective_batch_size(self) -> int:
        """Return the nominal number of rows per optimizer update."""
        return self.microbatch_size * self.gradient_accumulation_steps

    @field_validator("sampling_temperature")
    @classmethod
    def valid_sampling_temperature(cls, value: float) -> float:
        """Reject invalid sampling temperatures."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("sampling_temperature must be finite and positive")
        return value


class RLTrainLearnerConfig(BaseModel):
    """Learner-side PPO batch assembly settings."""

    model_config = ConfigDict(extra="forbid")

    max_staleness: int = 2
    staleness_scope: Literal["decision", "seat_trajectory"] = "decision"
    pin_memory: bool = False
    non_blocking_transfer: bool = True
    copy_stream: bool = True
    shape_bucket_accumulation: bool = False
    route_bucket_minibatches: bool = False
    shuffle_each_epoch: bool = False
    compile_evaluate_actions: bool = False
    frozen_encoder_prefix_layers: int = 0
    fused_adamw: bool = True
    drain_buffer_decisions: int | None = None
    max_update_decisions: int | None = None
    shared_memory_publish: bool = False
    shared_memory_keep_last: int = 1
    async_checkpoint_pairs: bool = False
    disk_checkpoint_interval_versions: int = 1
    disk_checkpoint_keep_last: int = 2
    disk_checkpoint_retain_every_versions: int | None = None
    legacy_sampling_temperature: float | None = None
    schema9: Schema9LearnerBatchConfig | None = None

    @field_validator("max_staleness")
    @classmethod
    def valid_non_negative_int(cls, value: int) -> int:
        """Reject invalid staleness limits."""
        if value < 0:
            raise ValueError("max_staleness must be non-negative")
        return value

    @field_validator("frozen_encoder_prefix_layers")
    @classmethod
    def valid_frozen_encoder_prefix_layers(cls, value: int) -> int:
        """Reject a negative frozen-prefix boundary."""
        if value < 0:
            raise ValueError("frozen_encoder_prefix_layers must be non-negative")
        return value

    @field_validator("shared_memory_keep_last")
    @classmethod
    def valid_shared_memory_keep_last(cls, value: int) -> int:
        """Reject invalid shared policy retention counts."""
        if value <= 0:
            raise ValueError("shared_memory_keep_last must be positive")
        return value

    @field_validator("legacy_sampling_temperature")
    @classmethod
    def valid_legacy_sampling_temperature(
        cls,
        value: float | None,
    ) -> float | None:
        """Validate an explicit schema-1 trajectory temperature fallback."""
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError("legacy_sampling_temperature must be finite and positive")
        return value

    @field_validator("drain_buffer_decisions")
    @classmethod
    def valid_optional_drain_buffer(cls, value: int | None) -> int | None:
        """Reject invalid optional drain buffer sizes."""
        if value is not None and value <= 0:
            raise ValueError("drain_buffer_decisions must be positive when set")
        return value

    @field_validator("max_update_decisions")
    @classmethod
    def valid_optional_max_update_decisions(cls, value: int | None) -> int | None:
        """Reject invalid optional learner-window caps."""
        if value is not None and value <= 0:
            raise ValueError("max_update_decisions must be positive when set")
        return value

    @field_validator(
        "disk_checkpoint_interval_versions",
        "disk_checkpoint_keep_last",
    )
    @classmethod
    def valid_disk_checkpoint_interval(cls, value: int) -> int:
        """Reject invalid disk checkpoint intervals and retention counts."""
        if value <= 0:
            raise ValueError("disk checkpoint values must be positive")
        return value

    @field_validator("disk_checkpoint_retain_every_versions")
    @classmethod
    def valid_disk_checkpoint_retention(cls, value: int | None) -> int | None:
        """Reject invalid optional milestone retention intervals."""
        if value is not None and value <= 0:
            raise ValueError(
                "disk_checkpoint_retain_every_versions must be positive when set"
            )
        return value

    @model_validator(mode="after")
    def valid_checkpoint_retention_alignment(self) -> Self:
        """Require retained milestones to coincide with written checkpoints."""
        retain_every = self.disk_checkpoint_retain_every_versions
        if (
            retain_every is not None
            and retain_every % self.disk_checkpoint_interval_versions != 0
        ):
            raise ValueError(
                "disk checkpoint milestone interval must be divisible by the "
                "write interval"
            )
        return self


class RLTrainOptimizerConfig(BaseModel):
    """Optimizer settings for PPO training."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 1.0e-5
    final_learning_rate: float = 1.0e-6
    weight_decay: float = 1.0e-4
    deck_encoder_lr_multiplier: float = 1.0
    private_lr_multiplier: float = 1.0
    private_weight_decay: float | None = None
    scheduler_total_updates: int | None = None

    @field_validator("learning_rate", "final_learning_rate")
    @classmethod
    def valid_positive_float(cls, value: float) -> float:
        """Reject non-positive learning rates."""
        if value <= 0.0:
            raise ValueError("learning rates must be positive")
        return value

    @field_validator("deck_encoder_lr_multiplier", "private_lr_multiplier")
    @classmethod
    def valid_positive_multiplier(cls, value: float) -> float:
        """Reject non-positive optimizer group multipliers."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("optimizer LR multipliers must be finite and positive")
        return value

    @field_validator("weight_decay", "private_weight_decay")
    @classmethod
    def valid_weight_decay(cls, value: float | None) -> float | None:
        """Reject invalid weight decay."""
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise ValueError("weight_decay must be non-negative")
        return value

    @field_validator("scheduler_total_updates")
    @classmethod
    def valid_scheduler_total_updates(cls, value: int | None) -> int | None:
        """Reject invalid explicit effective-update budgets."""
        if value is not None and value <= 0:
            raise ValueError("scheduler_total_updates must be positive when set")
        return value


class RLTrainPerformanceDiagnosticsConfig(BaseModel):
    """Minute-level game-outcome diagnostics for distributed training."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    interval_seconds: float = 60.0
    rolling_window_minutes: int = 15
    recent_window_limit: int = 60
    parquet_shard_windows: int = 15
    primary_opponent_kinds: tuple[Literal["self_play", "frozen", "scripted"], ...] = (
        "frozen",
        "scripted",
    )

    @field_validator("interval_seconds")
    @classmethod
    def valid_interval_seconds(cls, value: float) -> float:
        """Require a positive performance reporting interval."""
        if value <= 0.0:
            raise ValueError("performance interval_seconds must be positive")
        return value

    @field_validator("rolling_window_minutes")
    @classmethod
    def valid_rolling_window_minutes(cls, value: int) -> int:
        """Require a positive rolling performance window."""
        if value <= 0:
            raise ValueError("rolling_window_minutes must be positive")
        return value

    @field_validator("recent_window_limit", "parquet_shard_windows")
    @classmethod
    def valid_positive_window_count(cls, value: int) -> int:
        """Require positive bounded-history and shard sizes."""
        if value <= 0:
            raise ValueError("performance window counts must be positive")
        return value

    @field_validator("primary_opponent_kinds")
    @classmethod
    def valid_primary_opponent_kinds(
        cls,
        value: tuple[Literal["self_play", "frozen", "scripted"], ...],
    ) -> tuple[Literal["self_play", "frozen", "scripted"], ...]:
        """Require unique non-empty primary opponent categories."""
        if not value:
            raise ValueError("primary_opponent_kinds must be non-empty")
        if len(value) != len(set(value)):
            raise ValueError("primary_opponent_kinds must be unique")
        return value

    @model_validator(mode="after")
    def valid_window_alignment(self) -> Self:
        """Require the rolling duration to contain whole reporting intervals."""
        intervals = self.rolling_window_minutes * 60.0 / self.interval_seconds
        if intervals <= 1.0 or not math.isclose(intervals, round(intervals)):
            raise ValueError(
                "rolling_window_minutes must contain more than one and an integer "
                "number of performance intervals"
            )
        if self.recent_window_limit < round(intervals):
            raise ValueError(
                "recent_window_limit must cover the configured rolling window"
            )
        return self


class RLTrainDiagnosticsConfig(BaseModel):
    """Diagnostics settings for the RL training runner."""

    model_config = ConfigDict(extra="forbid")

    log_every_updates: int = 10
    tensorboard: bool = True
    tensorboard_flush_seconds: float = 30.0
    tensorboard_runtime_interval_seconds: float = 10.0
    inference_summary_interval_seconds: float = 2.0
    inference_registry_sync_interval_seconds: float = 1.0
    inference_latency_window: int = 4096
    inference_shape_histogram_max_keys: int = 512
    curriculum_persist_interval_seconds: float = 5.0
    learner_status_interval_versions: int = 1
    ppo_diagnostics_interval_versions: int = 1
    throughput_warmup_iterations: int = 0
    performance: RLTrainPerformanceDiagnosticsConfig = Field(
        default_factory=RLTrainPerformanceDiagnosticsConfig
    )

    @field_validator(
        "log_every_updates",
        "learner_status_interval_versions",
        "ppo_diagnostics_interval_versions",
    )
    @classmethod
    def valid_log_interval(cls, value: int) -> int:
        """Reject invalid logging intervals."""
        if value <= 0:
            raise ValueError("diagnostic log intervals must be positive")
        return value

    @field_validator(
        "tensorboard_flush_seconds",
        "tensorboard_runtime_interval_seconds",
        "inference_summary_interval_seconds",
        "inference_registry_sync_interval_seconds",
        "curriculum_persist_interval_seconds",
    )
    @classmethod
    def valid_positive_interval(cls, value: float) -> float:
        """Reject invalid diagnostics intervals."""
        if value <= 0.0:
            raise ValueError("diagnostics intervals must be positive")
        return value

    @field_validator("inference_latency_window")
    @classmethod
    def valid_inference_latency_window(cls, value: int) -> int:
        """Reject invalid inference latency summary windows."""
        if value <= 0:
            raise ValueError("inference_latency_window must be positive")
        return value

    @field_validator("inference_shape_histogram_max_keys")
    @classmethod
    def valid_shape_histogram_max_keys(cls, value: int) -> int:
        """Reject invalid exact-shape diagnostic bounds."""
        if value <= 0:
            raise ValueError("inference_shape_histogram_max_keys must be positive")
        return value

    @field_validator("throughput_warmup_iterations")
    @classmethod
    def valid_throughput_warmup_iterations(cls, value: int) -> int:
        """Reject invalid profile warmup window counts."""
        if value < 0:
            raise ValueError("throughput_warmup_iterations must be non-negative")
        return value


class RLTrainConfig(BaseModel):
    """Hydra-backed config for PPO RL training."""

    model_config = ConfigDict(extra="forbid")

    run: TrainingRunConfig = Field(default_factory=TrainingRunConfig)
    execution: RLTrainExecutionConfig = Field(default_factory=RLTrainExecutionConfig)
    model: AgentNetworkConfig = Field(default_factory=AgentNetworkConfig)
    private_deck_registry: PrivateDeckRegistrySourceConfig | None = None
    registry_transition: DeckRegistryTransitionConfig | None = None
    checkpoint_path: Path | None = None
    anchor_checkpoint_path: Path | None = None
    resume: TrainingResumeConfig = Field(default_factory=TrainingResumeConfig)
    output_dir: Path | None = None
    device: str = "auto"
    seed: int = 0
    collection: RLTrainCollectionConfig = Field(default_factory=RLTrainCollectionConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    frozen_league: FrozenLeagueConfig = Field(default_factory=FrozenLeagueConfig)
    frozen_policy_pool: FrozenPolicyPoolConfig = Field(
        default_factory=FrozenPolicyPoolConfig
    )
    inference: InferenceServerConfig = Field(default_factory=InferenceServerConfig)
    planner: ResolvedPlannerRuntimeConfig | None = None
    rollout_probe: RolloutProbeConfig = Field(default_factory=RolloutProbeConfig)
    rollout_belief: RolloutBeliefConfig = Field(default_factory=RolloutBeliefConfig)
    factual: FactualLaneConfig = Field(default_factory=FactualLaneConfig)
    macro_credit: MacroCreditConfig = Field(default_factory=MacroCreditConfig)
    engine_teacher: OnlineEngineTeacherConfig = Field(
        default_factory=OnlineEngineTeacherConfig
    )
    amortized_policy_iteration: AmortizedPolicyIterationConfig = Field(
        default_factory=AmortizedPolicyIterationConfig
    )
    distributed: DistributedTrainConfig = Field(default_factory=DistributedTrainConfig)
    gae: GaeConfig = Field(default_factory=GaeConfig)
    learner: RLTrainLearnerConfig = Field(default_factory=RLTrainLearnerConfig)
    ppo: PpoConfig = Field(default_factory=PpoConfig)
    optimizer: RLTrainOptimizerConfig = Field(default_factory=RLTrainOptimizerConfig)
    diagnostics: RLTrainDiagnosticsConfig = Field(
        default_factory=RLTrainDiagnosticsConfig
    )
    hydra: Mapping[str, Any] | None = None

    @field_validator("device")
    @classmethod
    def valid_device(cls, value: str) -> str:
        """Reject empty device strings."""
        if not value.strip():
            raise ValueError("device must be non-empty")
        return value

    @model_validator(mode="after")
    def valid_actor_topology(self) -> Self:
        """Restrict actor-free runs to distributed learner coordinators."""
        if self.execution.actors > 0:
            return self
        if (
            self.execution.mode == "distributed_async"
            and self.distributed.coordinator_enabled
            and not self.distributed.worker_enabled
        ):
            return self
        raise ValueError(
            "execution.actors=0 is only supported for distributed async "
            "coordinator runs"
        )

    @model_validator(mode="after")
    def valid_gpu_service_mode(self) -> Self:
        """Restrict learner-owned inference to local asynchronous training."""
        if self.execution.gpu_service_mode == "separate_process":
            if self.learner.async_checkpoint_pairs:
                raise ValueError(
                    "async checkpoint pairs currently require learner_thread serving"
                )
            return self
        if self.execution.mode != "async":
            raise ValueError("learner_thread GPU service requires async execution")
        if not self.execution.inference_server:
            raise ValueError("learner_thread GPU service requires inference_server")
        if self.collection.policy_kind != "model":
            raise ValueError("learner_thread GPU service requires a model policy")
        if self.learner.shared_memory_publish:
            raise ValueError(
                "learner_thread GPU service must not duplicate shared-memory weights"
            )
        if not self.learner.async_checkpoint_pairs:
            raise ValueError(
                "learner_thread GPU service requires async checkpoint pairs"
            )
        if (
            self.execution.inference_cuda_memory_trim_interval_steps is not None
            or self.execution.learner_cuda_memory_trim_interval_versions is not None
        ):
            raise ValueError(
                "learner_thread GPU service requires CUDA memory trimming disabled"
            )
        if self.planner is not None:
            raise ValueError(
                "learner_thread GPU service does not yet support planner snapshots"
            )
        if self.macro_credit.native_teacher_enabled:
            raise ValueError(
                "learner_thread GPU service does not yet support the native macro "
                "teacher"
            )
        conditioning = self.model.deck_conditioning
        if any(
            dropout > 0.0
            for dropout in (
                self.model.state_encoder.dropout,
                self.model.policy.dropout,
                self.model.action_value.dropout,
                0.0 if conditioning is None else conditioning.adapter_dropout,
            )
        ):
            raise ValueError(
                "learner_thread GPU service requires dropout-free learner models"
            )
        return self

    @model_validator(mode="after")
    def valid_recurrent_runtime_topology(self) -> Self:
        """Fail closed on serving paths that cannot preserve sequence leases."""
        if self.model.recurrent is None:
            return self
        if self.collection.policy_kind != "model":
            raise ValueError("recurrent training requires a model rollout policy")
        if self.planner is not None:
            raise ValueError("recurrent training does not support runtime planning")
        if self.inference.graph_decode:
            raise ValueError("recurrent training does not support graph decode")
        if self.inference.compile_model or self.inference.bucketize:
            raise ValueError(
                "recurrent training requires exact eager inference execution"
            )
        if self.execution.gpu_service_mode == "learner_thread":
            raise ValueError(
                "recurrent training requires immutable process-served snapshots"
            )
        if (
            self.inference.recurrent_snapshot_min_version_gap
            > self.learner.max_staleness + 1
        ):
            raise ValueError(
                "recurrent snapshot version gap cannot exceed the learner "
                "staleness horizon plus one"
            )
        return self

    @model_validator(mode="after")
    def valid_recurrent_stale_game_recycling(self) -> Self:
        """Restrict stale-game recycling to its immutable local serving path."""
        recycling = self.inference.recurrent_stale_game_recycling
        if not recycling.enabled:
            return self
        if self.execution.mode != "async":
            raise ValueError(
                "recurrent stale-game recycling requires local async execution"
            )
        if not self.execution.inference_server:
            raise ValueError("recurrent stale-game recycling requires inference_server")
        if self.model.recurrent is None:
            raise ValueError(
                "recurrent stale-game recycling requires a recurrent model"
            )
        if not self.learner.shared_memory_publish:
            raise ValueError(
                "recurrent stale-game recycling requires shared-memory publish"
            )
        return self

    @model_validator(mode="after")
    def valid_recurrent_ppo_only_training(self) -> Self:
        """Keep recurrent training structurally free of retired auxiliary loops."""
        if self.model.recurrent is None:
            return self
        if self.model.action_value.enabled:
            raise ValueError("recurrent PPO must not instantiate the action-Q head")
        if self.amortized_policy_iteration.enabled:
            raise ValueError("recurrent PPO cannot enable CMPO or reanalysis")
        if self.engine_teacher.enabled or self.factual.enabled:
            raise ValueError("recurrent PPO cannot enable auxiliary engine targets")
        if self.macro_credit.enabled or self.macro_credit.native_teacher_enabled:
            raise ValueError("recurrent PPO cannot enable macro-credit teaching")
        if self.learner.schema9 is not None:
            raise ValueError("recurrent PPO cannot enable planner replay")
        if self.rollout_probe.enabled or self.rollout_belief.enabled:
            raise ValueError(
                "the first recurrent PPO branch requires probe and belief inputs off"
            )
        if self.execution.archive_trajectories:
            raise ValueError(
                "legacy trajectory archives cannot persist recurrent sequences"
            )
        if self.learner.staleness_scope != "seat_trajectory":
            raise ValueError("recurrent PPO requires seat-trajectory staleness")
        if (
            self.learner.shape_bucket_accumulation
            or self.learner.route_bucket_minibatches
        ):
            raise ValueError("recurrent PPO cannot reorder rows for learner bucketing")
        if self.learner.compile_evaluate_actions:
            raise ValueError(
                "recurrent PPO evaluate-actions compilation is unsupported"
            )
        if self.gae.credit_unit != "decision":
            raise ValueError("recurrent PPO requires decision-level complete actions")
        if self.anchor_checkpoint_path is not None or self.ppo.kl_anchor_coef != 0.0:
            raise ValueError("recurrent PPO cannot construct a legacy anchor policy")
        if self.ppo.transition_distillation.optimizer_updates != 0:
            raise ValueError("recurrent PPO cannot run transition distillation")
        auxiliary_coefficients = {
            "engine_teacher_coef": self.ppo.engine_teacher_coef,
            "factual_effect_coef": self.ppo.factual_effect_coef,
            "factual_successor_coef": self.ppo.factual_successor_coef,
            "macro_conditional_coef": self.ppo.macro_conditional_coef,
            "macro_expected_coef": self.ppo.macro_expected_coef,
            "candidate_rerank_coef": self.ppo.candidate_rerank_coef,
            "proposal_distillation_coef": self.ppo.proposal_distillation_coef,
            "root_information_value_coef": self.ppo.root_information_value_coef,
        }
        active = tuple(
            name for name, value in auxiliary_coefficients.items() if value != 0.0
        )
        if active:
            raise ValueError(
                "recurrent PPO has active auxiliary coefficients: " + ", ".join(active)
            )
        return self

    @model_validator(mode="after")
    def valid_collection_flow_control_window(self) -> Self:
        """Ensure producer credits cannot stop below one learner window."""
        if not self.execution.collection_flow_control:
            return self
        high = self.execution.collection_high_watermark_decisions
        if high is None:
            raise ValueError("collection flow control has no high watermark")
        learner_window = min(
            self.collection.iteration_decisions,
            self.learner.max_update_decisions or self.collection.iteration_decisions,
        )
        if high < learner_window:
            raise ValueError(
                "collection high watermark must cover one learner decision window"
            )
        return self

    @model_validator(mode="after")
    def valid_shared_ring_topology(self) -> Self:
        """Require one crash-recoverable slot lane per local actor."""
        if self.execution.trajectory_transport != "shared_memory_ring":
            return self
        if self.execution.mode != "async":
            raise ValueError(
                "shared trajectory ring supports local async actor lanes only"
            )
        if self.execution.trajectory_ring_slots < self.execution.actors:
            raise ValueError("shared trajectory ring needs one slot per actor")
        if self.execution.trajectory_ring_allow_inline_oversize:
            raise ValueError(
                "shared trajectory ring cannot inline oversized payloads; "
                "increase slot bytes"
            )
        return self

    @model_validator(mode="after")
    def valid_curriculum_execution_lanes(self) -> Self:
        """Bind equal-width curriculum lanes to the local actor topology."""
        lane_count = self.curriculum.assignment_schedule.execution_lanes
        if lane_count <= 0:
            return self
        actor_count = self.execution.actors
        if actor_count <= 0 or lane_count > actor_count:
            raise ValueError(
                "curriculum execution lanes cannot exceed local training actors"
            )
        if actor_count % lane_count != 0:
            raise ValueError(
                "training actors must divide evenly across curriculum execution lanes"
            )
        return self

    @model_validator(mode="after")
    def valid_schema9_planner_identity(self) -> Self:
        """Bind schema-9 learner semantics to one explicit static planner."""
        schema9 = self.learner.schema9
        if schema9 is None:
            return self
        if self.planner is None:
            raise ValueError("schema9 learner requires a static planner config")
        expected = self.planner.resolve_static().schema9_learner_config()
        if schema9 != expected:
            raise ValueError("schema9 learner differs from static planner identity")
        planner_belief_dim = self.planner.tensorizer.belief_summary_dim
        model_belief_dim = self.model.root_perspective_value.belief_summary_dim
        if planner_belief_dim != model_belief_dim:
            raise ValueError(
                "planner tensorizer belief summary width differs from the "
                "root-perspective value model"
            )
        return self

    @model_validator(mode="after")
    def valid_planner_actor_geometry(self) -> Self:
        """Bind planner lease rows to the actual actor rollout topology."""
        if self.planner is None:
            return self
        batching = self.planner.batching
        if batching.actor_count != self.execution.actors:
            raise ValueError("planner actor count differs from training execution")
        if batching.max_root_rows_per_request < self.collection.num_concurrent_games:
            raise ValueError(
                "planner root-row capacity is below concurrent games per actor"
            )
        required_context_rows = (
            self.execution.actors * self.collection.num_concurrent_games
        )
        if self.planner.contexts.retained_root_rows < required_context_rows:
            raise ValueError(
                "planner retained-root capacity cannot cover the actor topology"
            )
        return self

    @model_validator(mode="after")
    def valid_planner_context_liveness(self) -> Self:
        """Keep server TTL beyond foreground planning and cleanup traffic."""
        if self.planner is None:
            return self
        if self.engine_teacher.enabled:
            raise ValueError(
                "schema-9 planner behavior cannot run with the legacy engine teacher"
            )
        if self.collection.policy_kind == "min_count":
            raise ValueError("planner behavior requires a learned rollout policy")
        if not self.execution.inference_server:
            return self
        deadlines = self.planner.deadlines
        foreground_seconds = min(
            deadlines.request_timeout_seconds - deadlines.return_guard_seconds,
            (
                self.planner.planner_behavior.constructor.work_budget.wall_clock_limit_ms
                / 1_000.0
            ),
        )
        cleanup_and_ipc_seconds = (
            deadlines.cleanup_timeout_seconds
            + self.planner.batching.inference_batch_wait_ms / 1_000.0
        )
        minimum_ttl = foreground_seconds + cleanup_and_ipc_seconds
        if self.inference.planner_context_ttl_seconds <= minimum_ttl:
            raise ValueError(
                "inference planner-context TTL must exceed the complete planner "
                "foreground deadline plus cleanup and IPC margin"
            )
        batching = self.planner.batching
        actual_geometry = (
            self.inference.max_planner_proposal_rows,
            self.inference.max_planner_candidate_rows,
            self.inference.max_root_information_rows,
            self.inference.max_wait_ms,
            self.execution.inference_request_queue_maxsize,
        )
        expected_geometry = (
            batching.proposal_microbatch_rows,
            batching.candidate_microbatch_rows,
            batching.root_value_microbatch_rows,
            batching.inference_batch_wait_ms,
            batching.inference_queue_capacity,
        )
        if actual_geometry != expected_geometry:
            raise ValueError(
                "inference server queue/microbatch geometry differs from planner"
            )
        return self

    @model_validator(mode="after")
    def valid_registry_transition(self) -> Self:
        """Restrict expert transitions to explicit routed warm-start branches."""
        if self.registry_transition is None:
            return self
        if self.resume.mode != "warm_start" or self.checkpoint_path is None:
            raise ValueError(
                "registry_transition requires warm_start with checkpoint_path"
            )
        conditioning = self.model.deck_conditioning
        routed_v2 = (
            conditioning is not None
            and conditioning.enabled
            and conditioning.architecture_version
            == DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION
            and conditioning.lora is not None
            and conditioning.lora.export_mode == "routed"
        )
        routed_dense_v3 = (
            conditioning is not None
            and conditioning.enabled
            and conditioning.architecture_version
            == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
            and conditioning.dense_private is not None
            and conditioning.dense_private.export_mode == "routed"
        )
        routed_compositional_v4 = (
            conditioning is not None
            and conditioning.enabled
            and conditioning.architecture_version
            == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
            and conditioning.compositional is not None
            and conditioning.compositional.export_mode == "routed"
        )
        if not routed_v2 and not routed_dense_v3 and not routed_compositional_v4:
            raise ValueError(
                "registry_transition requires a routed expert architecture"
            )
        return self

    @model_validator(mode="after")
    def valid_transition_distillation(self) -> Self:
        """Bind a fixed teacher budget to a routed DCCR-v4 training branch."""
        transition = self.ppo.transition_distillation
        if transition.optimizer_updates == 0:
            return self
        if self.anchor_checkpoint_path is None:
            raise ValueError("transition distillation requires anchor_checkpoint_path")
        conditioning = self.model.deck_conditioning
        if (
            conditioning is None
            or conditioning.architecture_version
            != DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
            or conditioning.compositional is None
            or conditioning.compositional.export_mode != "routed"
        ):
            raise ValueError("transition distillation requires routed DCCR-v4")
        if transition.action_wdl_coef > 0.0 and not self.model.action_value.enabled:
            raise ValueError(
                "action-WDL transition distillation requires action_value.enabled"
            )
        return self

    @model_validator(mode="after")
    def valid_frozen_league(self) -> Self:
        """Keep scheduled snapshots aligned with durable policy/state pairs."""
        league = self.frozen_league
        if not league.enabled:
            return self
        if self.collection.opponent_mode != "curriculum":
            raise ValueError("frozen_league requires curriculum opponent mode")
        if self.curriculum.mix.frozen <= 0.0:
            raise ValueError("frozen_league requires positive frozen curriculum mass")
        if (
            self.learner.disk_checkpoint_retain_every_versions
            != league.snapshot_interval_versions
        ):
            raise ValueError(
                "frozen_league cadence must equal permanent checkpoint retention"
            )
        configured_seed_count = (
            len(self.curriculum.fixed_frozen_bundles)
            + len(self.curriculum.fixed_frozen_policies)
            + int(
                self.curriculum.seed_anchor_in_frozen_pool
                and self.anchor_checkpoint_path is not None
            )
        )
        if league.max_opponents < configured_seed_count:
            raise ValueError(
                "frozen_league max_opponents cannot be smaller than configured "
                "pinned opponents"
            )
        if configured_seed_count + self.curriculum.frozen_capacity > (
            league.max_opponents
        ):
            raise ValueError(
                "configured pinned opponents plus non-pinned frozen capacity "
                "exceed frozen_league max_opponents"
            )
        return self

    @model_validator(mode="after")
    def valid_engine_teacher(self) -> Self:
        """Reject configurations that silently disable or randomize teaching."""
        if not self.engine_teacher.enabled:
            return self
        if self.collection.policy_kind != "model":
            raise ValueError("engine_teacher requires collection.policy_kind=model")
        if self.ppo.engine_teacher_coef <= 0.0:
            raise ValueError("enabled engine_teacher requires engine_teacher_coef > 0")
        if not self.engine_teacher.manual_coin:
            raise ValueError("engine_teacher requires paired manual-coin expectation")
        if self.engine_teacher.async_actor and (
            self.execution.mode != "async" or not self.execution.inference_server
        ):
            raise ValueError(
                "async engine_teacher requires local async inference-server execution"
            )
        sampler = self.engine_teacher.sampler
        if sampler.mode != "archetype":
            raise ValueError("engine_teacher requires archetype belief sampling")
        if not sampler.strict_own_deck_counts:
            raise ValueError("engine_teacher requires strict own-deck counts")
        if (
            sampler.prior_deck_signature_summary_path is None
            or sampler.prior_deck_signature_summary_sha256 is None
        ):
            raise ValueError(
                "engine_teacher archetype sampling requires a fingerprinted prior"
            )
        probe_sampler = self.rollout_probe.sampler
        if (
            self.rollout_probe.enabled
            and probe_sampler.mode
            in {
                "archetype",
                "model",
            }
            and (
                probe_sampler.prior_deck_signature_summary_path is None
                or probe_sampler.prior_deck_signature_summary_sha256 is None
            )
        ):
            raise ValueError("engine_teacher requires a fingerprinted probe prior")
        if self.rollout_belief.enabled and (
            self.rollout_belief.deck_signature_summary_path is None
            or self.rollout_belief.deck_signature_summary_sha256 is None
        ):
            raise ValueError("engine_teacher requires a fingerprinted belief prior")
        return self

    @model_validator(mode="after")
    def valid_count_first_policy(self) -> Self:
        """Keep count-first likelihoods on their supported decision objective."""
        if self.model.policy.unordered_set_policy != "count_first":
            return self
        if self.gae.credit_unit != "decision":
            raise ValueError("count-first policy requires decision-level PPO credit")
        if self.ppo.kl_anchor_coef != 0.0:
            raise ValueError(
                "count-first policy cannot use a STOP-policy stepwise KL anchor"
            )
        if self.planner is not None or self.learner.schema9 is not None:
            raise ValueError(
                "count-first policy is incompatible with legacy prefix proposals"
            )
        return self

    @model_validator(mode="after")
    def valid_amortized_policy_iteration(self) -> Self:
        """Bind the selected API architecture to one non-gated async mainline."""
        policy_iteration = self.amortized_policy_iteration
        if not policy_iteration.enabled:
            return self
        if not self.model.action_value.enabled:
            raise ValueError(
                "amortized policy iteration requires the complete-action value head"
            )
        if self.model.policy.unordered_set_policy != "count_first":
            raise ValueError("amortized policy iteration requires count-first policy")
        if self.execution.mode != "async" or not self.execution.inference_server:
            raise ValueError(
                "amortized policy iteration requires local async inference serving"
            )
        if policy_iteration.improvement.actor_count >= self.execution.actors:
            raise ValueError("improvement actor allocation must leave ordinary actors")
        if self.planner is not None or self.engine_teacher.enabled:
            raise ValueError(
                "amortized policy iteration cannot reactivate planner/teacher lanes"
            )
        if self.macro_credit.enabled:
            raise ValueError(
                "amortized policy iteration cannot reactivate retired macro credit"
            )
        maximum_candidates = max(
            policy_iteration.candidate_proposal.max_candidates,
            policy_iteration.candidate_proposal.exhaustive_action_cap,
        )
        maximum_cells = policy_iteration.belief_worlds * maximum_candidates
        if maximum_cells > policy_iteration.native.max_cells_per_root:
            raise ValueError(
                "candidate-by-world geometry exceeds native max_cells_per_root"
            )
        return self

    @model_validator(mode="after")
    def valid_macro_credit(self) -> Self:
        """Bind schema-10 hindsight lanes to factual and native runtime inputs."""
        macro = self.macro_credit
        if not macro.enabled:
            if (
                self.ppo.macro_conditional_coef > 0.0
                or self.ppo.macro_expected_coef > 0.0
            ):
                raise ValueError("macro PPO coefficients require macro credit")
            return self
        if not self.factual.enabled:
            raise ValueError("macro credit requires factual rollout targets")
        if self.collection.policy_kind != "model":
            raise ValueError("macro credit requires learned policy collection")
        if self.learner.schema9 is not None:
            raise ValueError("schema 10 does not activate schema-9 behavior replay")
        if (
            self.ppo.macro_conditional_coef <= 0.0
            or self.ppo.macro_expected_coef <= 0.0
        ):
            raise ValueError("macro credit requires both root-only losses")
        if self.ppo.root_information_value_coef <= 0.0:
            raise ValueError("macro credit requires endpoint value learning")
        tensorizer = macro.root_information_tensorizer
        if tensorizer is None:
            raise ValueError("macro credit requires root-information tensorization")
        if (
            tensorizer.belief_summary_dim
            != self.model.root_perspective_value.belief_summary_dim
        ):
            raise ValueError("macro and model belief-summary widths differ")
        if macro.native_teacher_enabled:
            if self.planner is None:
                raise ValueError("native macro teacher requires planner runtime")
            if tensorizer != self.planner.tensorizer:
                raise ValueError(
                    "macro root-information tensorizer differs from the planner"
                )
            if not self.planner.planner_behavior.emit_macro_teacher:
                raise ValueError(
                    "native macro teacher planner must emit macro evidence"
                )
            static = self.planner.resolve_static()
            expected_identity = MacroTeacherIdentityConfig(
                constructor_fingerprint=static.constructor_fingerprint,
                scorer_fingerprint=static.scorer_fingerprint,
                controller_fingerprint=static.controller.controller_fingerprint,
                planner_fingerprint=static.planner_fingerprint,
            )
            if macro.native_teacher_identity != expected_identity:
                raise ValueError(
                    "native macro teacher identity differs from static planner"
                )
            if self.engine_teacher.enabled:
                raise ValueError("native macro teacher replaces the legacy teacher")
            if self.execution.mode != "async" or not self.execution.inference_server:
                raise ValueError(
                    "native macro teacher requires local async inference execution"
                )
        elif self.planner is not None:
            raise ValueError(
                "macro credit planner runtime is reserved for native teaching"
            )
        return self


@dataclass
class _MutableSyncCounters:
    collect_iterations: int = 0
    policy_actions: int = 0
    forced_actions: int = 0
    scripted_actions: int = 0
    recorded_decisions: int = 0
    finished_games: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return JSON-friendly counters."""
        return {
            "collect_iterations": self.collect_iterations,
            "policy_actions": self.policy_actions,
            "forced_actions": self.forced_actions,
            "scripted_actions": self.scripted_actions,
            "recorded_decisions": self.recorded_decisions,
            "finished_games": self.finished_games,
        }


@dataclass
class _SyncIteration:
    iteration: int
    policy_version: int
    trajectories: tuple[GameTrajectory, ...]
    deferred_trajectories: tuple[GameTrajectory, ...] = ()
    counters: _MutableSyncCounters = field(default_factory=_MutableSyncCounters)
    curriculum_summary: dict[str, Any] | None = None
    rollout_features: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _TrainingRolloutActors:
    actors: RolloutActors
    frozen_update: FrozenPolicyPoolUpdate | None = None


@dataclass(frozen=True)
class _RemoteInferenceRoutes:
    actor_id: str
    actor_incarnation: int
    request_queue: Any
    response_queue: Any
    client_config: InferenceClientConfig
    client_state: RemoteInferenceClientState
    request_id_base: int
    recurrent_model_config: AgentNetworkConfig | None
    frozen_state_path: Path


class _RemoteFrozenPolicyMapping(Mapping[str, RolloutPolicy]):
    """Create remote frozen policy clients lazily by opponent id."""

    def __init__(self, routes: _RemoteInferenceRoutes) -> None:
        self._routes = routes
        self._policies: dict[str, RolloutPolicy] = {}

    def __getitem__(self, key: str) -> RolloutPolicy:
        policy = self._policies.get(key)
        if policy is None:
            frozen_state = read_frozen_pool_state(self._routes.frozen_state_path)
            member = next(
                (item for item in frozen_state.members if item.opponent_id == key),
                None,
            )
            if member is None:
                raise KeyError(f"frozen policy metadata is unavailable: {key}")
            policy = RemoteInferencePolicy(
                actor_id=self._routes.actor_id,
                actor_incarnation=self._routes.actor_incarnation,
                policy_id=key,
                request_queue=self._routes.request_queue,
                response_queue=self._routes.response_queue,
                config=self._routes.client_config,
                client_state=self._routes.client_state,
                initial_request_id=self._routes.request_id_base,
                recurrent_model_config=(
                    self._routes.recurrent_model_config if member.recurrent else None
                ),
            )
            self._policies[key] = policy
        return policy

    def __iter__(self) -> Iterator[str]:
        return iter(self._policies)

    def __len__(self) -> int:
        return len(self._policies)


class _CurriculumAssignmentManager:
    """Bridge curriculum deck sampling to engine-assigned game IDs."""

    def __init__(
        self,
        sampler: CurriculumSampler,
        *,
        state_path: Path,
        additions_dir: Path,
        candidate_seat: int = 0,
    ) -> None:
        self.sampler = sampler
        self.state_path = state_path
        self.additions_dir = additions_dir
        self.candidate_seat = candidate_seat
        self.assignments: dict[str, GameAssignment] = {}
        self._pending_assignments: list[GameAssignment] = []
        self._assigned_counts: Counter[str] = Counter()
        self._finished_counts: Counter[str] = Counter()
        self._canceled_counts: Counter[str] = Counter()
        self._deck_outcome_counts: dict[str, Counter[str]] = {}
        self._deck_winrate_ema: dict[str, float] = {}
        self._candidate_policy_versions: dict[str, set[int]] = {}

    @property
    def frozen_members(self) -> tuple[FrozenPoolMember, ...]:
        """Return frozen members currently known to the sampler."""
        return self.sampler.frozen_members

    def sample_deck_pair(self) -> DeckPair:
        """Sample a curriculum assignment and return its deck pair."""
        self._consume_frozen_additions()
        lane_index = (
            0 if self.sampler.config.assignment_schedule.execution_lanes > 0 else None
        )
        assignment = self.sampler.assign(lane_index=lane_index)
        self._pending_assignments.append(assignment)
        _record_assignment_counts(assignment, self._assigned_counts)
        return (assignment.candidate_deck, assignment.opponent_deck)

    def sync_live_games(self, live_games: Sequence[VectorGame]) -> None:
        """Attach pending assignments to newly created live engine games."""
        for game in live_games:
            if game.game_id in self.assignments:
                continue
            if not self._pending_assignments:
                raise RuntimeError(
                    "curriculum mode requires the pool factory to use the provided "
                    "deck_pair_sampler"
                )
            assignment = self._pending_assignments.pop(0)
            expected = (assignment.candidate_deck, assignment.opponent_deck)
            if _normalize_deck_pair(game.deck_pair) != expected:
                raise ValueError(
                    f"curriculum assignment deck pair does not match game {game.game_id}"
                )
            self.assignments[game.game_id] = assignment

    def observe_decision(self, decision: Any) -> None:
        """Bind a game to the exact candidate policy version that acted in it."""
        if int(decision.seat) != self.candidate_seat:
            return
        self._candidate_policy_versions.setdefault(decision.game_id, set()).add(
            int(decision.policy_version)
        )

    def finalize(self, finished: FinishedGame) -> None:
        """Observe one terminal game and update PFSP state if needed."""
        self._consume_frozen_additions()
        assignment = self.assignments.pop(finished.game_id, None)
        candidate_policy_version = _single_candidate_policy_version(
            self._candidate_policy_versions.pop(finished.game_id, set())
        )
        if assignment is None:
            self._finished_counts["missing_assignment"] += 1
            return
        candidate_reward = _reward_for_seat(
            self.candidate_seat,
            finished.winner_index,
        )
        _record_finished_counts(
            assignment,
            candidate_reward=candidate_reward,
            finished_counts=self._finished_counts,
            deck_outcome_counts=self._deck_outcome_counts,
            deck_winrate_ema=self._deck_winrate_ema,
            winrate_ema_alpha=self.sampler.config.winrate_ema_alpha,
        )
        self.sampler.observe(
            CurriculumOutcome(
                opponent_kind=assignment.opponent_kind,
                opponent_id=assignment.opponent_id,
                candidate_reward=candidate_reward,
                candidate_deck_label=assignment.candidate_deck_label,
                opponent_deck_label=assignment.opponent_deck_label,
                candidate_policy_version=candidate_policy_version,
            )
        )

    def cancel(self, game_ids: Sequence[str], reason: str) -> None:
        """Release unfinished local assignments without observing outcomes."""
        concrete = tuple(str(game_id) for game_id in game_ids)
        if len(concrete) != len(set(concrete)):
            raise ValueError("curriculum cancellation contains duplicate game IDs")
        missing = [game_id for game_id in concrete if game_id not in self.assignments]
        if missing:
            raise RuntimeError(
                "curriculum cancellation is missing assignments: "
                + ", ".join(sorted(missing))
            )
        for game_id in concrete:
            del self.assignments[game_id]
            self._candidate_policy_versions.pop(game_id, None)
        self._canceled_counts["games"] += len(concrete)
        self._canceled_counts[f"reason:{reason}"] += len(concrete)

    def save_state(self) -> None:
        """Persist the latest frozen-pool state."""
        self._consume_frozen_additions()
        self.sampler.save_state(self.state_path)

    def _consume_frozen_additions(self) -> int:
        consumed = consume_frozen_pool_additions(
            self.sampler,
            self.additions_dir,
            state_path=self.state_path,
        )
        return len(consumed)

    def summary(
        self,
        frozen_update: FrozenPolicyPoolUpdate | None,
    ) -> dict[str, Any]:
        """Return JSON-friendly curriculum collection diagnostics."""
        data: dict[str, Any] = {
            "mode": "curriculum",
            "state_path": deck_records.display_path(self.state_path),
            "curriculum_config": self.sampler.config.model_dump(mode="json"),
            "assigned": dict(sorted(self._assigned_counts.items())),
            "finished": dict(sorted(self._finished_counts.items())),
            "canceled": dict(sorted(self._canceled_counts.items())),
            "decks": _curriculum_deck_summary(
                sampler=self.sampler,
                assigned_counts=self._assigned_counts,
                finished_counts=self._finished_counts,
                deck_outcome_counts=self._deck_outcome_counts,
                deck_winrate_ema=self.sampler.state().candidate_deck_winrate_ema,
            ),
            "lanes": _curriculum_lane_summary(
                self.sampler,
                assigned_counts=self._assigned_counts,
                finished_counts=self._finished_counts,
            ),
            "matchups": list(self.sampler.matchup_statistics),
            "active_assignments": len(self.assignments),
            "pending_unmatched_assignments": len(self._pending_assignments),
            "frozen_members": len(self.sampler.frozen_members),
        }
        if frozen_update is not None:
            data["frozen_policy_pool"] = {
                "loaded": list(frozen_update.loaded),
                "unloaded": list(frozen_update.unloaded),
                "kept": list(frozen_update.kept),
            }
        return data


def _remote_curriculum_request_id_base() -> int:
    """Return a process-incarnation-specific curriculum request namespace."""
    return uuid.uuid4().int << 64


def _matching_curriculum_response(
    response_queue: Any,
    *,
    request_id: int,
    actor_incarnation: int,
) -> Mapping[str, Any]:
    """Discard responses from an older actor incarnation."""
    while True:
        response = response_queue.get()
        if not isinstance(response, Mapping):
            raise ValueError("curriculum assignment response must be a mapping")
        if int(response.get("request_id", -1)) != request_id:
            continue
        if int(response.get("actor_incarnation", -1)) != actor_incarnation:
            continue
        error = response.get("error")
        if error is not None:
            raise RuntimeError(f"curriculum assignment request failed: {error}")
        return cast(Mapping[str, Any], response)


class _RemoteCurriculumAssignmentManager:
    """Actor-side proxy for a central curriculum sampler process."""

    def __init__(
        self,
        config: RLTrainConfig,
        *,
        state_path: Path,
        actor_index: int,
        actor_incarnation: int,
        request_queue: Any,
        response_queue: Any,
        candidate_seat: int = 0,
    ) -> None:
        self.state_path = state_path
        self.actor_index = actor_index
        self.actor_incarnation = actor_incarnation
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.candidate_seat = candidate_seat
        self.sampler = CurriculumSampler(
            config.curriculum.model_copy(update={"frozen_state_path": state_path}),
            rng=random.Random(config.seed + 31 + actor_index),
        )
        self.assignments: dict[str, GameAssignment] = {}
        self._assignment_request_ids: dict[str, int] = {}
        self._pending_assignments: list[tuple[int, GameAssignment]] = []
        self._assigned_counts: Counter[str] = Counter()
        self._finished_counts: Counter[str] = Counter()
        self._canceled_counts: Counter[str] = Counter()
        self._deck_outcome_counts: dict[str, Counter[str]] = {}
        self._deck_winrate_ema: dict[str, float] = {}
        self._candidate_policy_versions: dict[str, set[int]] = {}
        self._request_id = _remote_curriculum_request_id_base()

    @property
    def frozen_members(self) -> tuple[FrozenPoolMember, ...]:
        """Return frozen members known when this actor started."""
        return self.sampler.frozen_members

    def sample_deck_pair(self) -> DeckPair:
        """Request one central curriculum assignment and return its deck pair."""
        request_id = self._request_id
        self._request_id += 1
        self.request_queue.put(
            {
                "type": "sample",
                "actor_index": self.actor_index,
                "actor_incarnation": self.actor_incarnation,
                "request_id": request_id,
            }
        )
        response = _matching_curriculum_response(
            self.response_queue,
            request_id=request_id,
            actor_incarnation=self.actor_incarnation,
        )
        assignment = response.get("assignment")
        if not isinstance(assignment, GameAssignment):
            raise ValueError("curriculum assignment response missing assignment")
        self._pending_assignments.append((request_id, assignment))
        _record_assignment_counts(assignment, self._assigned_counts)
        return (assignment.candidate_deck, assignment.opponent_deck)

    def sync_live_games(self, live_games: Sequence[VectorGame]) -> None:
        """Attach central assignments to newly created live engine games."""
        for game in live_games:
            if game.game_id in self.assignments:
                continue
            if not self._pending_assignments:
                raise RuntimeError(
                    "curriculum mode requires the pool factory to use the provided "
                    "deck_pair_sampler"
                )
            request_id, assignment = self._pending_assignments.pop(0)
            expected = (assignment.candidate_deck, assignment.opponent_deck)
            if _normalize_deck_pair(game.deck_pair) != expected:
                raise ValueError(
                    f"curriculum assignment deck pair does not match game {game.game_id}"
                )
            self.assignments[game.game_id] = assignment
            self._assignment_request_ids[game.game_id] = request_id

    def observe_decision(self, decision: Any) -> None:
        """Bind a game to the exact candidate policy version that acted in it."""
        if int(decision.seat) != self.candidate_seat:
            return
        self._candidate_policy_versions.setdefault(decision.game_id, set()).add(
            int(decision.policy_version)
        )

    def finalize(self, finished: FinishedGame) -> None:
        """Send one terminal curriculum outcome to the central sampler."""
        assignment = self.assignments.pop(finished.game_id, None)
        assignment_request_id = self._assignment_request_ids.pop(
            finished.game_id,
            None,
        )
        candidate_policy_version = _single_candidate_policy_version(
            self._candidate_policy_versions.pop(finished.game_id, set())
        )
        if assignment is None:
            self._finished_counts["missing_assignment"] += 1
            return
        if assignment_request_id is None:
            raise RuntimeError(
                "central curriculum assignment lost its request lease: "
                f"{finished.game_id}"
            )
        candidate_reward = _reward_for_seat(
            self.candidate_seat,
            finished.winner_index,
        )
        _record_finished_counts(
            assignment,
            candidate_reward=candidate_reward,
            finished_counts=self._finished_counts,
            deck_outcome_counts=self._deck_outcome_counts,
            deck_winrate_ema=self._deck_winrate_ema,
            winrate_ema_alpha=self.sampler.config.winrate_ema_alpha,
        )
        self.request_queue.put(
            {
                "type": "outcome",
                "actor_index": self.actor_index,
                "actor_incarnation": self.actor_incarnation,
                "assignment_request_id": assignment_request_id,
                "candidate_reward": candidate_reward,
                "candidate_policy_version": candidate_policy_version,
            }
        )

    def cancel(self, game_ids: Sequence[str], reason: str) -> None:
        """Cancel central assignment leases and wait for one batch ack."""
        concrete = tuple(str(game_id) for game_id in game_ids)
        if len(concrete) != len(set(concrete)):
            raise ValueError("curriculum cancellation contains duplicate game IDs")
        missing = [
            game_id
            for game_id in concrete
            if game_id not in self.assignments
            or game_id not in self._assignment_request_ids
        ]
        if missing:
            raise RuntimeError(
                "central curriculum cancellation is missing assignments: "
                + ", ".join(sorted(missing))
            )
        assignment_request_ids = tuple(
            self._assignment_request_ids[game_id] for game_id in concrete
        )
        request_id = self._request_id
        self._request_id += 1
        self.request_queue.put(
            {
                "type": "cancel",
                "actor_index": self.actor_index,
                "actor_incarnation": self.actor_incarnation,
                "request_id": request_id,
                "assignment_request_ids": list(assignment_request_ids),
                "reason": str(reason),
            }
        )
        response = _matching_curriculum_response(
            self.response_queue,
            request_id=request_id,
            actor_incarnation=self.actor_incarnation,
        )
        acknowledged = tuple(
            int(value)
            for value in response.get("acknowledged_assignment_request_ids", ())
        )
        if acknowledged != assignment_request_ids:
            raise RuntimeError("central curriculum cancellation ack is misaligned")
        for game_id in concrete:
            del self.assignments[game_id]
            del self._assignment_request_ids[game_id]
            self._candidate_policy_versions.pop(game_id, None)
        self._canceled_counts["games"] += len(concrete)
        self._canceled_counts[f"reason:{reason}"] += len(concrete)

    def save_state(self) -> None:
        """State is owned and persisted by the central curriculum process."""

    def summary(
        self,
        frozen_update: FrozenPolicyPoolUpdate | None,
    ) -> dict[str, Any]:
        """Return JSON-friendly actor-side curriculum diagnostics."""
        data: dict[str, Any] = {
            "mode": "curriculum",
            "assignment_source": "central",
            "actor_index": self.actor_index,
            "actor_incarnation": self.actor_incarnation,
            "state_path": deck_records.display_path(self.state_path),
            "assigned": dict(sorted(self._assigned_counts.items())),
            "finished": dict(sorted(self._finished_counts.items())),
            "canceled": dict(sorted(self._canceled_counts.items())),
            "decks": _curriculum_deck_summary(
                sampler=self.sampler,
                assigned_counts=self._assigned_counts,
                finished_counts=self._finished_counts,
                deck_outcome_counts=self._deck_outcome_counts,
                deck_winrate_ema=self._deck_winrate_ema,
            ),
            "lanes": _curriculum_lane_summary(
                self.sampler,
                assigned_counts=self._assigned_counts,
                finished_counts=self._finished_counts,
            ),
            "active_assignments": len(self.assignments),
            "pending_unmatched_assignments": len(self._pending_assignments),
            "frozen_members": len(self.sampler.frozen_members),
        }
        if frozen_update is not None:
            data["frozen_policy_pool"] = {
                "loaded": list(frozen_update.loaded),
                "unloaded": list(frozen_update.unloaded),
                "kept": list(frozen_update.kept),
            }
        return data


class _CurriculumRolloutRecorder:
    """Forward rollout events and observe curriculum outcomes on finalize."""

    def __init__(
        self,
        recorder: TensorTrajectoryRecorder,
        manager: _CurriculumAssignmentManager | _RemoteCurriculumAssignmentManager,
    ) -> None:
        self._recorder = recorder
        self._manager = manager

    def record(self, decision: Any) -> None:
        self._recorder.record(decision)
        self._manager.observe_decision(decision)

    def finalize(self, finished: FinishedGame) -> None:
        self._recorder.finalize(finished)
        self._manager.finalize(finished)

    def discard(self, game_id: str, *, reason: str) -> None:
        self._recorder.discard(game_id, reason=reason)
        self._manager.cancel((game_id,), reason)


def run_rl_training(
    config: RLTrainConfig,
    *,
    pool_factory: RLTrainPoolFactory | None = None,
) -> dict[str, Any]:
    """Run RL training according to config."""
    config = _resolve_training_model_config(config)
    if config.execution.mode in ("async", "distributed_async"):
        return _run_async_training(
            config,
            pool_factory=pool_factory,
        )
    return _run_sync_training(
        config,
        pool_factory=pool_factory,
    )


def resolve_rl_train_output_dir(config: RLTrainConfig) -> Path:
    """Return the resolved output directory for RL training artifacts."""
    return resolve_training_output_dir(
        task_name="rl",
        run=config.run,
        output_dir=config.output_dir,
    )


def _validate_distributed_warm_start_output(
    config: RLTrainConfig,
    *,
    output_dir: Path,
    include_mutable_artifacts: bool,
) -> None:
    """Reject artifacts from a collided warm-start run identity."""
    if config.resume.mode != "warm_start" or (
        config.registry_transition is None
        and (
            config.execution.mode != "distributed_async"
            or not config.distributed.coordinator_enabled
        )
    ):
        return
    patterns = [
        "weights/policy_v*.pt",
        "weights/shared_policy_v*.pt",
        "weights/latest.json",
        "weights/shared_latest.json",
        "resume/training_state_v*.pt",
        "resume/latest.json",
        ".checkpoint_mirror/latest.json",
    ]
    if include_mutable_artifacts:
        patterns.extend(
            (
                "distributed_summary.json",
                "learner_status.json",
                "runtime_monitor.json",
                "runtime_status.json",
                "summary.json",
                "curriculum/central_summary.json",
                "curriculum/frozen_pool_state.json",
                "performance/training_performance.json",
            )
        )
    collisions = sorted(
        {
            path
            for pattern in patterns
            for path in output_dir.glob(pattern)
            if path.exists()
        },
        key=str,
    )
    if collisions:
        rendered = ", ".join(str(path) for path in collisions[:8])
        raise RuntimeError(
            "distributed warm start requires a fresh run identity; existing "
            f"training artifacts: {rendered}"
        )


def _claim_fresh_local_training_output(
    config: RLTrainConfig,
    *,
    output_dir: Path,
) -> None:
    """Atomically reserve one immutable output directory for a local run."""
    if config.execution.mode not in ("sync", "async"):
        return
    output_identity = output_dir.resolve()
    frozen_state_identity = deck_records.repo_path(
        _resolved_frozen_state_path(config, output_dir=output_dir)
    ).resolve()
    try:
        frozen_state_identity.relative_to(output_identity)
    except ValueError as exc:
        raise ValueError(
            "local training mutable curriculum state must live inside its fresh "
            f"output directory: {frozen_state_identity}"
        ) from exc
    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"training output is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    claim_path = output_dir / ".training_attempt.json"
    if claim_path.exists():
        raise RuntimeError(
            "local training requires a fresh run identity; output is already "
            f"claimed: {claim_path}"
        )
    preexisting = sorted(output_dir.iterdir(), key=str)
    if preexisting:
        rendered = ", ".join(str(path) for path in preexisting[:8])
        raise RuntimeError(
            "local training requires a fresh run identity; existing output "
            f"entries: {rendered}"
        )
    claim = {
        "schema_version": 1,
        "attempt_id": str(uuid.uuid4()),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "run_version": config.run.version,
        "execution_mode": config.execution.mode,
        "resume_mode": config.resume.mode,
    }
    try:
        descriptor = os.open(
            claim_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "local training requires a fresh run identity; output is already "
            f"claimed: {claim_path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(claim, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_descriptor = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # Keep a partial claim fail-closed. Its exclusive existence prevents a
        # second process from adopting an output whose first attempt is unclear.
        raise
    collisions = sorted(
        (path for path in output_dir.iterdir() if path != claim_path),
        key=str,
    )
    if collisions:
        rendered = ", ".join(str(path) for path in collisions[:8])
        raise RuntimeError(
            "local training requires a fresh run identity; existing output "
            f"entries: {rendered}"
        )


def _distributed_adjusted_config(config: RLTrainConfig) -> RLTrainConfig:
    """Return config adjusted for distributed async collection."""
    if (
        config.execution.mode != "distributed_async"
        or not config.distributed.coordinator_enabled
    ):
        return config
    if config.distributed.remote_max_staleness <= config.learner.max_staleness:
        return config
    return config.model_copy(
        update={
            "learner": config.learner.model_copy(
                update={"max_staleness": config.distributed.remote_max_staleness}
            )
        }
    )


def _maybe_start_distributed_coordinator(
    config: RLTrainConfig,
    *,
    output_dir: Path,
    trajectory_queue: Any,
) -> Any | None:
    """Start distributed services when configured for a coordinator run."""
    if (
        config.execution.mode != "distributed_async"
        or not config.distributed.coordinator_enabled
    ):
        return None
    from ptcg_rl.rl.distributed.compatibility import DistributedModelCompatibility
    from ptcg_rl.rl.distributed.coordinator import (
        DistributedCoordinator,
        DistributedCoordinatorConfig,
    )

    transport = config.distributed.transport
    coordinator = DistributedCoordinator(
        trajectory_queue=trajectory_queue,
        config=DistributedCoordinatorConfig(
            bind_host=transport.bind_host,
            trajectory_port=transport.trajectory_port,
            weight_port=transport.weight_port,
            summary_path=output_dir / "distributed_summary.json",
            weights_dir=output_dir / "weights",
            weight_poll_interval_versions=transport.weight_publish_interval_versions,
            max_staleness=config.learner.max_staleness,
            queue_maxsize=config.execution.trajectory_queue_maxsize,
            queue_high_watermark_ratio=transport.queue_high_watermark_ratio,
            queue_resume_watermark_ratio=transport.queue_resume_watermark_ratio,
            flow_control_sleep_seconds=transport.flow_control_sleep_seconds,
            performance=_performance_reporter_config(config, output_dir=output_dir),
            compatibility=DistributedModelCompatibility.from_model_config(config.model),
        ),
    )
    coordinator.start()
    return coordinator


def _performance_reporter_config(
    config: RLTrainConfig,
    *,
    output_dir: Path,
) -> PerformanceReporterConfig | None:
    """Build the shared local or distributed performance reporter config."""
    from ptcg_rl.rl.performance_state import PerformanceReporterConfig

    performance = config.diagnostics.performance
    if not performance.enabled:
        return None
    return PerformanceReporterConfig(
        summary_path=output_dir / "performance" / "training_performance.json",
        tensorboard_dir=tensorboard_root_dir(output_dir) / "performance",
        interval_seconds=performance.interval_seconds,
        rolling_window_minutes=performance.rolling_window_minutes,
        target_deck_labels=(
            tuple(config.curriculum.candidate_lanes.target_labels)
            if config.curriculum.candidate_lanes.enabled
            else config.curriculum.candidate_deck_labels()
        ),
        stationary_opponent_kinds=tuple(performance.primary_opponent_kinds),
        tensorboard_enabled=config.diagnostics.tensorboard,
        tensorboard_flush_seconds=config.diagnostics.tensorboard_flush_seconds,
        recent_window_limit=performance.recent_window_limit,
        parquet_shard_windows=performance.parquet_shard_windows,
    )


def _run_sync_training(
    config: RLTrainConfig,
    *,
    pool_factory: RLTrainPoolFactory | None,
) -> dict[str, Any]:
    _configure_gpu_worker_runtime(config)
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    output_dir = deck_records.repo_path(resolve_rl_train_output_dir(config))
    _validate_distributed_warm_start_output(
        config,
        output_dir=output_dir,
        include_mutable_artifacts=True,
    )
    _claim_fresh_local_training_output(config, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _maybe_seed_curriculum_anchor(config, output_dir=output_dir)
    tensorboard_writer = _learner_tensorboard_writer(config, output_dir)
    model = _load_training_model(config, device=device)
    _maybe_compile_learner_evaluate_actions(
        model,
        enabled=config.learner.compile_evaluate_actions,
    )
    anchor_model = _load_anchor_model(config, device=device)
    _reset_torch_random_stream(config.seed)
    optimizer = _build_ppo_optimizer(config, model=model, device=device)
    _apply_registry_transition_optimizer_state(
        config,
        model=model,
        optimizer=optimizer,
    )
    lr_scheduler_steps = _planned_lr_scheduler_steps(config)
    lr_scheduler = _build_lr_scheduler(
        optimizer,
        config=config,
        planned_steps=lr_scheduler_steps,
    )
    current_policy_version = _initial_policy_version(config)
    training_progress = _restore_training_progress(
        config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        policy_version=current_policy_version,
        planned_scheduler_steps=lr_scheduler_steps,
    )
    if config.learner.frozen_encoder_prefix_layers > 0:
        model.freeze_conditioned_encoder_prefix(
            config.learner.frozen_encoder_prefix_layers
        )
    _validate_remaining_training_iterations(config, training_progress)
    publisher = WeightPublisher(
        output_dir / "weights",
        _disk_weight_publisher_config(config),
    )
    policy = _sync_rollout_policy(config, model=model)
    deck_pair = _deck_pair(config.collection.decks)
    factory = pool_factory or _training_pool_factory(
        include_search_input=_training_include_search_input(config)
    )

    start = time.perf_counter()
    learner_timer = StageTimer()
    iterations: list[dict[str, Any]] = []
    total_updates = training_progress.total_optimizer_updates
    ppo_updates = _restored_ppo_optimizer_updates(config, training_progress)
    latest_batch_stats: dict[str, Any] | None = None
    last_sync_iteration: _SyncIteration | None = None
    _maybe_write_learner_status(
        config,
        output_dir,
        {
            "phase": "starting",
            "elapsed_seconds": 0.0,
            "iteration": None,
            "current_policy_version": current_policy_version,
            "resume": _training_progress_summary(config, training_progress),
        },
    )
    deferred_trajectories: tuple[GameTrajectory, ...] = ()
    learner_window_decisions = _learner_window_decision_budget(config)
    for iteration_index in range(
        training_progress.completed_iterations,
        config.collection.training_iterations,
    ):
        iteration_started = time.perf_counter() - start
        _maybe_write_learner_status(
            config,
            output_dir,
            {
                "phase": "collecting",
                "elapsed_seconds": time.perf_counter() - start,
                "iteration": iteration_index,
                "current_policy_version": current_policy_version,
                "publish_version": current_policy_version + 1,
                "target_decisions": learner_window_decisions,
                "configured_iteration_decisions": (
                    config.collection.iteration_decisions
                ),
                "deferred_trajectories": len(deferred_trajectories),
                "deferred_decisions": _decision_count(deferred_trajectories),
            },
        )
        sync_iteration = _collect_sync_iteration(
            config,
            factory=factory,
            deck_pair=deck_pair,
            policy=policy,
            device=device,
            policy_version=current_policy_version,
            output_dir=output_dir,
            initial_trajectories=deferred_trajectories,
            target_decisions=learner_window_decisions,
            retain_boundary=(
                iteration_index + 1 < config.collection.training_iterations
            ),
        )
        if (
            iteration_index + 1 == config.collection.training_iterations
            and sync_iteration.deferred_trajectories
        ):
            # There is no later learner window in which to retain a synchronous
            # collection overshoot. Consume the remaining completed games whole
            # in the final update instead of silently abandoning them.
            sync_iteration = replace(
                sync_iteration,
                trajectories=(
                    sync_iteration.trajectories + sync_iteration.deferred_trajectories
                ),
                deferred_trajectories=(),
            )
        deferred_trajectories = sync_iteration.deferred_trajectories
        last_sync_iteration = sync_iteration
        batch_config = LearnerBatchConfig(
            microbatch_size=config.collection.microbatch_size,
            gradient_accumulation_steps=(config.collection.gradient_accumulation_steps),
            max_decisions=config.learner.max_update_decisions,
            max_staleness=config.learner.max_staleness,
            staleness_scope=config.learner.staleness_scope,
            shuffle=True,
            shuffle_each_epoch=config.learner.shuffle_each_epoch,
            drop_last=False,
            pin_memory=config.learner.pin_memory,
            non_blocking_transfer=config.learner.non_blocking_transfer,
            copy_stream=config.learner.copy_stream,
            shape_bucket_accumulation=(config.learner.shape_bucket_accumulation),
            route_bucket_minibatches=config.learner.route_bucket_minibatches,
            seed=config.seed + iteration_index * config.collection.ppo_epochs,
            legacy_sampling_temperature=(config.learner.legacy_sampling_temperature),
            gae=config.gae,
            require_deck_context=_deck_conditioning_enabled(config.model),
            schema9=config.learner.schema9,
            macro_credit=(config.macro_credit if config.macro_credit.enabled else None),
        )
        _maybe_write_learner_status(
            config,
            output_dir,
            {
                "phase": "building_minibatches",
                "elapsed_seconds": time.perf_counter() - start,
                "iteration": iteration_index,
                "current_policy_version": current_policy_version,
                "publish_version": current_policy_version + 1,
                "target_decisions": learner_window_decisions,
                "configured_iteration_decisions": (
                    config.collection.iteration_decisions
                ),
                "trajectories": len(sync_iteration.trajectories),
                "trajectory_decisions": _decision_count(sync_iteration.trajectories),
                "deferred_trajectories": len(deferred_trajectories),
                "deferred_decisions": _decision_count(deferred_trajectories),
            },
        )
        batch_result = build_ppo_minibatches(
            sync_iteration.trajectories,
            current_policy_version=current_policy_version,
            config=batch_config,
            device=None,
            timer=learner_timer,
        )
        update_results = _run_ppo_epochs_and_publish(
            config=config,
            model=model,
            optimizer=optimizer,
            trajectories=sync_iteration.trajectories,
            current_policy_version=current_policy_version,
            batch_config=batch_config,
            batch_result=batch_result,
            anchor_model=anchor_model,
            publisher=publisher,
            shared_publisher=None,
            publish_version=current_policy_version + 1,
            first_update_index=total_updates,
            first_ppo_update_index=ppo_updates,
            device=device,
            lr_scheduler=lr_scheduler,
            timer=learner_timer,
            output_path=output_dir,
            started_at=start,
            iteration_index=iteration_index,
        )
        published_version = _effective_published_version(
            config,
            update_results,
            logical_version=current_policy_version + 1,
        )
        if published_version is not None:
            current_policy_version = published_version
            _set_rollout_policy_version(policy, current_policy_version)
        iteration_finished = time.perf_counter() - start
        iteration_summary = _sync_iteration_summary(
            iteration_index=iteration_index,
            sync_iteration=sync_iteration,
            update_results=update_results,
            published_version=published_version,
            started_elapsed_seconds=iteration_started,
            finished_elapsed_seconds=iteration_finished,
        )
        iterations.append(iteration_summary)
        write_learner_tensorboard(
            tensorboard_writer,
            step=published_version
            if published_version is not None
            else iteration_index,
            iteration=iteration_summary,
            update_results=update_results,
            optimizer=optimizer,
            timer=learner_timer,
        )
        completed_ppo_updates = sum(len(result.updates) for result in update_results)
        total_updates += completed_ppo_updates
        ppo_updates += completed_ppo_updates
        if update_results:
            latest_batch_stats = dict(update_results[-1].batch_result.stats.__dict__)
        del batch_result, update_results
        _maybe_trim_learner_memory(config, published_version=published_version)
    elapsed_seconds = time.perf_counter() - start
    if last_sync_iteration is None:
        raise RuntimeError("training loop did not run any iterations")
    summary = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": config.execution.mode,
        "device": str(device),
        "output_dir": deck_records.display_path(output_dir),
        "elapsed_seconds": elapsed_seconds,
        "iterations": iterations,
        "collect": _sum_collect_counters(iterations),
        "trajectories": sum(int(iteration["trajectories"]) for iteration in iterations),
        "trajectory_decisions": sum(
            int(iteration["trajectory_decisions"]) for iteration in iterations
        ),
        "learner": {
            "updates": total_updates,
            "effective_updates": total_updates,
            "ppo_updates": ppo_updates,
            "published_version": current_policy_version,
            "batching": {
                "microbatch_size": config.collection.microbatch_size,
                "gradient_accumulation_steps": (
                    config.collection.gradient_accumulation_steps
                ),
                "effective_batch_size": config.collection.effective_batch_size,
            },
            "batch_stats": latest_batch_stats,
            "timings": _learner_timing_summary(
                learner_timer,
                elapsed_seconds=elapsed_seconds,
            ),
            "last_iteration_ppo": iterations[-1]["learner"]["ppo"],
            "last_iteration_early_stop": iterations[-1]["learner"]["early_stop"],
        },
        "optimizer": _optimizer_summary(
            config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            planned_steps=lr_scheduler_steps,
        ),
        "resume": _training_progress_summary(config, training_progress),
        "throughput": _training_throughput_summary(
            config,
            iterations=iterations,
            elapsed_seconds=elapsed_seconds,
        ),
        "curriculum": last_sync_iteration.curriculum_summary,
        "health": _training_health_summary(iterations),
    }
    _add_tensorboard_log_dir(summary, tensorboard_writer, output_dir=output_dir)
    _write_summary(output_dir / "summary.json", summary)
    tensorboard_writer.close()
    return summary


def _run_async_training(
    config: RLTrainConfig,
    *,
    pool_factory: RLTrainPoolFactory | None,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    config = _distributed_adjusted_config(config)
    if (
        config.execution.mode in ("async", "distributed_async")
        and config.execution.archive_trajectories
        and config.collection.training_iterations > 1
    ):
        raise ValueError("async archive writing supports one training iteration")
    output_dir = deck_records.repo_path(resolve_rl_train_output_dir(config))
    _validate_distributed_warm_start_output(
        config,
        output_dir=output_dir,
        include_mutable_artifacts=True,
    )
    _claim_fresh_local_training_output(config, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curriculum_bootstrap_summary = _maybe_bootstrap_curriculum_state(
        config,
        output_dir=output_dir,
    )
    league_promotion_bootstrap_summary = _maybe_bootstrap_league_promotion_state(
        config,
        output_dir=output_dir,
    )
    _maybe_seed_curriculum_anchor(config, output_dir=output_dir)
    config_data = config.model_dump(mode="python")
    context = torch_mp.get_context("spawn")
    trajectory_queue = _new_local_trajectory_transport(config, context=context)
    collection_gate = (
        context.Event() if config.execution.collection_flow_control else None
    )
    if collection_gate is not None:
        collection_gate.set()
    learner_ready_event = (
        context.Event()
        if config.execution.gpu_service_mode == "learner_thread"
        else None
    )
    policy_iteration = config.amortized_policy_iteration
    reanalysis_root_queue = (
        context.Queue(maxsize=policy_iteration.native.root_queue_capacity)
        if policy_iteration.enabled
        else None
    )
    reanalysis_job_queue = (
        context.Queue(maxsize=policy_iteration.native.job_queue_capacity)
        if policy_iteration.enabled
        else None
    )
    reanalysis_result_queue = (
        context.Queue(maxsize=policy_iteration.native.result_queue_capacity)
        if policy_iteration.enabled
        else None
    )
    distributed_coordinator: Any | None = None
    inference_request_queue = (
        context.Queue(maxsize=config.execution.inference_request_queue_maxsize)
        if config.execution.inference_server
        else None
    )
    inference_response_queues = None
    if config.execution.inference_server:
        inference_response_queues = [
            context.Queue() for _index in range(config.execution.actors)
        ]
    teacher_inference_response_queues = None
    if (
        config.execution.inference_server
        and config.engine_teacher.enabled
        and config.engine_teacher.async_actor
    ):
        teacher_inference_response_queues = [
            context.Queue() for _index in range(config.execution.actors)
        ]
    central_curriculum = _uses_central_curriculum(config)
    curriculum_request_queue = context.Queue() if central_curriculum else None
    curriculum_response_queues = (
        [context.Queue() for _index in range(config.execution.actors)]
        if central_curriculum
        else None
    )
    multiprocessing_queues = (
        trajectory_queue,
        reanalysis_root_queue,
        reanalysis_job_queue,
        reanalysis_result_queue,
        inference_request_queue,
        *(inference_response_queues or ()),
        *(teacher_inference_response_queues or ()),
        curriculum_request_queue,
        *(curriculum_response_queues or ()),
    )

    actor_incarnations = [-1] * config.execution.actors

    def actor_factory(actor_index: int) -> ManagedProcess:
        actor_incarnations[actor_index] += 1
        actor_incarnation = actor_incarnations[actor_index]
        actor_inference_response_queue = (
            None
            if inference_response_queues is None
            else inference_response_queues[actor_index]
        )
        actor_curriculum_response_queue = (
            None
            if curriculum_response_queues is None
            else curriculum_response_queues[actor_index]
        )
        actor_teacher_inference_response_queue = (
            None
            if teacher_inference_response_queues is None
            else teacher_inference_response_queues[actor_index]
        )
        _drain_stale_actor_response_queues(
            inference_response_queue=actor_inference_response_queue,
            teacher_inference_response_queue=(actor_teacher_inference_response_queue),
            curriculum_response_queue=actor_curriculum_response_queue,
        )
        return cast(
            ManagedProcess,
            context.Process(
                target=_async_actor_worker,
                kwargs={
                    "config_data": config_data,
                    "output_dir": str(output_dir),
                    "trajectory_queue": trajectory_queue,
                    "reanalysis_root_queue": reanalysis_root_queue,
                    "pool_factory": pool_factory,
                    "actor_index": actor_index,
                    "actor_incarnation": actor_incarnation,
                    "inference_request_queue": inference_request_queue,
                    "inference_response_queue": actor_inference_response_queue,
                    "teacher_inference_response_queue": (
                        actor_teacher_inference_response_queue
                    ),
                    "curriculum_request_queue": curriculum_request_queue,
                    "curriculum_response_queue": actor_curriculum_response_queue,
                    "collection_gate": collection_gate,
                    "learner_ready_event": learner_ready_event,
                },
            ),
        )

    def learner_factory() -> ManagedProcess:
        return cast(
            ManagedProcess,
            context.Process(
                target=_async_learner_worker,
                kwargs={
                    "config_data": config_data,
                    "output_dir": str(output_dir),
                    "trajectory_queue": trajectory_queue,
                    "reanalysis_root_queue": reanalysis_root_queue,
                    "reanalysis_job_queue": reanalysis_job_queue,
                    "reanalysis_result_queue": reanalysis_result_queue,
                    "inference_request_queue": (
                        inference_request_queue
                        if config.execution.gpu_service_mode == "learner_thread"
                        else None
                    ),
                    "inference_response_queues": (
                        inference_response_queues
                        if config.execution.gpu_service_mode == "learner_thread"
                        else None
                    ),
                    "teacher_inference_response_queues": (
                        teacher_inference_response_queues
                        if config.execution.gpu_service_mode == "learner_thread"
                        else None
                    ),
                    "collection_gate": collection_gate,
                    "learner_ready_event": learner_ready_event,
                },
            ),
        )

    reanalysis_processes: list[ManagedProcess] = []
    curriculum_process: ManagedProcess | None = None
    inference_process: ManagedProcess | None = None
    join_timeout_seconds = (
        config.execution.async_supervisor.process_join_timeout_seconds
    )
    runtime_monitor: _AsyncRuntimeMonitor | None = None
    runtime_tensorboard_writer: TensorboardMetricWriter | None = None
    try:
        distributed_coordinator = _maybe_start_distributed_coordinator(
            config,
            output_dir=output_dir,
            trajectory_queue=trajectory_queue,
        )
        if policy_iteration.enabled:
            if reanalysis_job_queue is None or reanalysis_result_queue is None:
                raise RuntimeError("native reanalysis queues were not initialized")
            for worker_index in range(policy_iteration.native.worker_count):
                process = cast(
                    ManagedProcess,
                    context.Process(
                        target=native_reanalysis_worker,
                        kwargs={
                            "config_data": policy_iteration.model_dump(mode="python"),
                            "rollout_belief_data": config.rollout_belief.model_dump(
                                mode="python"
                            ),
                            "job_queue": reanalysis_job_queue,
                            "result_queue": reanalysis_result_queue,
                            "library_path": None,
                        },
                        name=f"native-reanalysis-{worker_index}",
                    ),
                )
                reanalysis_processes.append(process)
                process.start()

        if central_curriculum:
            if curriculum_request_queue is None or curriculum_response_queues is None:
                raise RuntimeError("curriculum queues were not initialized")
            process = cast(
                ManagedProcess,
                context.Process(
                    target=_async_curriculum_worker,
                    kwargs={
                        "config_data": config_data,
                        "output_dir": str(output_dir),
                        "request_queue": curriculum_request_queue,
                        "response_queues": curriculum_response_queues,
                    },
                ),
            )
            curriculum_process = process
            process.start()

        if (
            config.execution.inference_server
            and config.execution.gpu_service_mode == "separate_process"
        ):
            if inference_request_queue is None or inference_response_queues is None:
                raise RuntimeError("inference queues were not initialized")
            process = cast(
                ManagedProcess,
                context.Process(
                    target=_async_inference_worker,
                    kwargs={
                        "config_data": config_data,
                        "output_dir": str(output_dir),
                        "request_queue": inference_request_queue,
                        "response_queues": inference_response_queues,
                        "teacher_response_queues": (teacher_inference_response_queues),
                    },
                ),
            )
            inference_process = process
            process.start()

        start = time.perf_counter()
        runtime_tensorboard_writer = _runtime_tensorboard_writer(config, output_dir)
        runtime_monitor = _AsyncRuntimeMonitor(
            output_path=output_dir / "runtime_monitor.json",
            live_output_path=output_dir / "runtime_status.json",
            tensorboard_writer=runtime_tensorboard_writer,
            trajectory_queue=trajectory_queue,
            collection_gate=collection_gate,
            inference_request_queue=inference_request_queue,
            inference_response_queues=(
                None
                if inference_response_queues is None
                else (
                    *inference_response_queues,
                    *(teacher_inference_response_queues or ()),
                )
            ),
            curriculum_request_queue=curriculum_request_queue,
            curriculum_response_queues=curriculum_response_queues,
            inference_process=inference_process,
            live_write_interval_seconds=(
                config.diagnostics.tensorboard_runtime_interval_seconds
            ),
            actor_heartbeat_timeout_seconds=(
                config.execution.async_supervisor.actor_heartbeat_timeout_seconds
            ),
        )
        if runtime_monitor is None:
            raise RuntimeError("async runtime monitor construction returned no monitor")
    except BaseException:
        if runtime_monitor is not None:
            with suppress(Exception):
                runtime_monitor.close()
            with suppress(Exception):
                runtime_monitor.close_tensorboard()
        elif runtime_tensorboard_writer is not None:
            with suppress(Exception):
                runtime_tensorboard_writer.close()
        if distributed_coordinator is not None:
            with suppress(Exception):
                distributed_coordinator.close()
        cleanup_processes = tuple(
            process
            for process in (
                inference_process,
                curriculum_process,
                *reanalysis_processes,
            )
            if process is not None
        )
        with suppress(Exception):
            stop_managed_processes(
                cleanup_processes,
                join_timeout_seconds=join_timeout_seconds,
            )
        _close_multiprocessing_queues(multiprocessing_queues)
        raise

    actor_factories = tuple(
        lambda actor_index=actor_index: actor_factory(actor_index)
        for actor_index in range(config.execution.actors)
    )
    required_processes: list[tuple[str, ManagedProcess]] = []
    if inference_process is not None:
        required_processes.append(("inference", inference_process))
    if curriculum_process is not None:
        required_processes.append(("curriculum", curriculum_process))
    required_processes.extend(
        (f"native_reanalysis_{index}", process)
        for index, process in enumerate(reanalysis_processes)
    )
    supervisor_completed = False
    try:
        supervisor_result = supervise_actor_group_learner(
            actor_factories=actor_factories,
            learner_factory=learner_factory,
            required_processes=required_processes,
            config=config.execution.async_supervisor,
            poll_callback=runtime_monitor.sample,
        )
        runtime_monitor.record_actor_recycles(supervisor_result.actor_recycles_by_actor)
        supervisor_completed = True
    except BaseException as exc:
        runtime_monitor.record_failure(exc)
        raise
    finally:
        if distributed_coordinator is not None:
            distributed_coordinator.close()
        service_processes = tuple(
            process
            for process in (inference_process, *reanalysis_processes)
            if process is not None
        )
        stop_managed_processes(
            service_processes,
            join_timeout_seconds=(
                config.execution.async_supervisor.process_join_timeout_seconds
            ),
        )
        if (
            not supervisor_completed
            and curriculum_process is not None
            and curriculum_request_queue is not None
        ):
            # Give the debounced curriculum service a chance to flush its last
            # state before falling back to termination on timeout.
            try:
                _finish_async_curriculum_worker(
                    config,
                    output_dir=output_dir,
                    process=curriculum_process,
                    request_queue=curriculum_request_queue,
                )
            except Exception:
                with suppress(Exception):
                    _stop_managed_process(
                        curriculum_process,
                        join_timeout_seconds=(
                            config.execution.async_supervisor.process_join_timeout_seconds
                        ),
                    )
        if not supervisor_completed:
            runtime_monitor.close()
            with suppress(Exception):
                runtime_monitor.write()
            with suppress(Exception):
                runtime_monitor.close_tensorboard()
            _close_multiprocessing_queues(multiprocessing_queues)
    elapsed_seconds = time.perf_counter() - start
    summary_path = output_dir / "summary.json"
    summary = _read_summary(summary_path)
    curriculum_service_summary = None
    if curriculum_process is not None and curriculum_request_queue is not None:
        curriculum_service_summary = _finish_async_curriculum_worker(
            config,
            output_dir=output_dir,
            process=curriculum_process,
            request_queue=curriculum_request_queue,
        )
    inference_summary = None
    if config.execution.inference_server:
        inference_summary = _read_optional_summary(_inference_summary_path(output_dir))
        if inference_summary is not None and inference_process is not None:
            inference_summary["process_exitcode"] = inference_process.exitcode
    summary.update(
        {
            "mode": config.execution.mode,
            "output_dir": deck_records.display_path(output_dir),
            "elapsed_seconds": elapsed_seconds,
            "actor_queue": _actor_queue_summary(
                output_dir,
                actor_count=config.execution.actors,
            ),
            "trajectory_transport": {
                "kind": config.execution.trajectory_transport,
                "shared_slots": (
                    config.execution.trajectory_ring_slots
                    if config.execution.trajectory_transport == "shared_memory_ring"
                    else None
                ),
                "shared_slot_bytes": (
                    config.execution.trajectory_ring_slot_bytes
                    if config.execution.trajectory_transport == "shared_memory_ring"
                    else None
                ),
                "collection_flow_control": (config.execution.collection_flow_control),
            },
            "curriculum_bootstrap": curriculum_bootstrap_summary,
            "league_promotion_bootstrap": league_promotion_bootstrap_summary,
            "supervisor": {
                "actor_restarts": supervisor_result.actor_restarts,
                "actor_restarts_by_actor": list(
                    supervisor_result.actor_restarts_by_actor
                ),
                "actor_recycles": supervisor_result.actor_recycles,
                "actor_recycles_by_actor": list(
                    supervisor_result.actor_recycles_by_actor
                ),
                "polls": supervisor_result.polls,
                "actor_exitcode": supervisor_result.actor_exitcode,
                "actor_exitcodes": list(supervisor_result.actor_exitcodes),
                "learner_exitcode": supervisor_result.learner_exitcode,
                "inference_server": config.execution.inference_server,
                "inference_exitcode": (
                    None if inference_process is None else inference_process.exitcode
                ),
            },
        }
    )
    if curriculum_service_summary is not None:
        summary["curriculum_service"] = curriculum_service_summary
    if inference_summary is not None:
        summary["inference"] = inference_summary
    if distributed_coordinator is not None:
        distributed_summary = distributed_coordinator.summary()
        summary["distributed"] = distributed_summary
        summary["health"] = _health_with_distributed_transport(
            summary.get("health"),
            distributed_summary,
        )
    runtime_monitor.close()
    runtime_monitor.write()
    runtime_monitor.close_tensorboard()
    summary["runtime_monitor"] = runtime_monitor.summary()
    _write_summary(summary_path, summary)
    _close_multiprocessing_queues(multiprocessing_queues)
    return summary


def _new_local_trajectory_transport(
    config: RLTrainConfig,
    *,
    context: Any,
) -> Any:
    """Create the configured bounded actor-to-learner data plane."""
    execution = config.execution
    if execution.trajectory_transport == "queue":
        return context.Queue(maxsize=execution.trajectory_queue_maxsize)
    return SharedTrajectoryRing(
        context,
        slots=execution.trajectory_ring_slots,
        slot_bytes=execution.trajectory_ring_slot_bytes,
        producer_count=execution.actors,
        allow_inline_oversize=execution.trajectory_ring_allow_inline_oversize,
    )


def _close_multiprocessing_queues(queues: Iterable[Any | None]) -> None:
    """Release queue feeder threads without flushing abandoned trajectories.

    Async actors can leave the bounded trajectory queue nonempty after the
    learner reaches its terminal iteration.  Joining that queue's feeder at
    interpreter shutdown can then block forever because no consumer remains.
    All child processes have stopped before this helper is called, so any
    buffered messages are deliberately discarded.
    """
    for queue_obj in queues:
        if queue_obj is None:
            continue
        with suppress(Exception):
            queue_obj.cancel_join_thread()
        with suppress(Exception):
            queue_obj.close()


def _drain_queue_batch(
    queue_obj: Any,
    *,
    limit: int,
    idle_timeout_ms: int = 0,
) -> list[Any]:
    """Drain a bounded batch until its limit or one producer-idle timeout.

    ``multiprocessing.Queue`` publishes through a feeder thread, so a message can
    already contribute to ``qsize()`` while a concurrent ``get_nowait()`` still
    observes an empty pipe. A positive idle timeout closes that visibility race
    without turning the learner drain into an unbounded wait. Zero preserves the
    original nonblocking behavior for compatibility profiles.
    """
    if queue_obj is None:
        raise ValueError("cannot drain an uninitialized queue")
    if limit <= 0:
        raise ValueError("queue drain limit must be positive")
    if idle_timeout_ms < 0:
        raise ValueError("queue drain idle timeout must be non-negative")
    items = []
    for _index in range(limit):
        try:
            if idle_timeout_ms == 0:
                item = queue_obj.get_nowait()
            else:
                item = queue_obj.get(timeout=idle_timeout_ms / 1_000.0)
        except queue.Empty:
            break
        items.append(item)
    return items


@dataclass(frozen=True, slots=True)
class _PolicyIterationCompletion:
    """Learner work completed while and after native jobs execute."""

    real_update: PolicyIterationUpdate | None
    counterfactual_updates: tuple[PolicyIterationUpdate, ...]
    ingest_summary: dict[str, int]
    native_result_root_ids: frozenset[str]


def _complete_policy_iteration_updates(
    *,
    policy_iteration_learner: AmortizedPolicyIterationLearner,
    policy_iteration_replay: PolicyIterationReplayStore,
    trajectories: Sequence[GameTrajectory],
    reanalysis_result_queue: Any,
    result_drain_limit: int,
    queue_drain_idle_timeout_ms: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    timer: StageTimer | None = None,
    wait_for_result_root_ids: frozenset[str] = frozenset(),
    result_barrier_timeout_seconds: float = 300.0,
    real_retrace_enabled: bool = True,
) -> _PolicyIterationCompletion:
    """Complete configured Q updates while submitted native jobs execute.

    The caller has already persisted immutable jobs and their proposal policy
    versions. Native workers therefore do not observe optimizer mutations.
    Results are recorded before student-safe target construction. When enabled,
    real Retrace precedes counterfactual updates so leaf bootstrap observes any
    target-network refresh scheduled by that real step.
    """
    real_update = (
        policy_iteration_learner.update_real_trajectories(
            trajectories,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            timer=timer,
        )
        if real_retrace_enabled
        else None
    )
    native_results = _drain_queue_batch(
        reanalysis_result_queue,
        limit=result_drain_limit,
        idle_timeout_ms=queue_drain_idle_timeout_ms,
    )
    if wait_for_result_root_ids:
        native_results.extend(
            _await_native_reanalysis_results(
                reanalysis_result_queue,
                already_received=native_results,
                expected_root_ids=wait_for_result_root_ids,
                timeout_seconds=result_barrier_timeout_seconds,
            )
        )
    typed_results = tuple(
        result
        for result in native_results
        if isinstance(result, NativeReanalysisResult)
    )
    policy_iteration_replay.record_results(typed_results)
    ingest_summary, ingested_targets = policy_iteration_learner.ingest_native_results(
        native_results,
        timer=timer,
    )
    policy_iteration_replay.record_targets(ingested_targets)
    counterfactual_updates = policy_iteration_learner.update_counterfactuals(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        timer=timer,
    )
    return _PolicyIterationCompletion(
        real_update=real_update,
        counterfactual_updates=counterfactual_updates,
        ingest_summary=ingest_summary,
        native_result_root_ids=frozenset(
            result.root.student.root_id for result in typed_results
        ),
    )


def _await_native_reanalysis_results(
    reanalysis_result_queue: Any,
    *,
    already_received: Sequence[Any],
    expected_root_ids: frozenset[str],
    timeout_seconds: float = 300.0,
) -> list[Any]:
    """Drain until every already-submitted native root has one result.

    Steady-state windows retain asynchronous overlap. Durable checkpoint and
    final boundaries quiesce accepted jobs so neither a sidecar nor clean
    shutdown can strand engine work. A wall-clock deadline converts an ABI hang
    into a resumable run failure instead of a permanent learner deadlock.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("native result barrier timeout must be finite and positive")
    received_ids = {
        result.root.student.root_id
        for result in already_received
        if isinstance(result, NativeReanalysisResult)
    }
    remaining = set(expected_root_ids) - received_ids
    drained: list[Any] = []
    deadline = time.monotonic() + timeout_seconds
    while remaining:
        wait_seconds = min(1.0, deadline - time.monotonic())
        if wait_seconds <= 0.0:
            sample = ", ".join(sorted(remaining)[:8])
            raise RuntimeError(
                "native reanalysis result barrier timed out with "
                f"{len(remaining)} roots pending: {sample}"
            )
        try:
            result = reanalysis_result_queue.get(timeout=wait_seconds)
        except queue.Empty:
            continue
        drained.append(result)
        if isinstance(result, NativeReanalysisResult):
            remaining.discard(result.root.student.root_id)
    return drained


def _requires_native_reanalysis_quiescence(
    config: RLTrainConfig,
    *,
    iteration_index: int,
    current_policy_version: int,
) -> bool:
    """Return whether the next sidecar/final boundary must own no live jobs."""
    return bool(
        iteration_index + 1 == config.collection.training_iterations
        or _should_write_disk_checkpoint(config, current_policy_version + 1)
    )


def run_distributed_actor_worker(config: RLTrainConfig) -> dict[str, Any]:
    """Run a remote actor+inference worker for distributed async training."""
    from ptcg_rl.rl.distributed.worker import run_distributed_actor_worker as run_worker

    return run_worker(config)


@dataclass(frozen=True)
class _InferenceRequestPrefetchFailure:
    """Failure raised by the background request queue reader."""

    error: BaseException


@dataclass(frozen=True, slots=True)
class _LearnerOwnedInferenceRuntime:
    """Live serving objects owned by the learner process."""

    policy: RolloutPolicy
    candidate_model: AgentPolicyValueNet | None
    stop_event: threading.Event
    model_lock: threading.Lock
    graph_capture_allowed: threading.Event


class _PrefetchedInferenceRequestQueue:
    """Bounded local queue populated from multiprocessing IPC in one thread."""

    _POLL_SECONDS = 0.1
    _JOIN_SECONDS = 1.0

    def __init__(self, source_queue: Any, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("inference request prefetch capacity must be positive")
        self._source_queue = source_queue
        self._items: queue.Queue[Any] = queue.Queue(maxsize=capacity)
        self._stop_event = threading.Event()
        self._failure: BaseException | None = None
        self._started = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="inference-request-prefetch",
            daemon=True,
        )

    def __enter__(self) -> Self:
        """Return the lazily started queue wrapper."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the background reader on normal or exceptional worker exit."""
        del exc_type, exc_value, traceback
        self.close()

    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> Any:
        """Return one prefetched request or propagate the producer failure."""
        self._start()
        failure = self._failure
        if failure is not None and self._items.empty():
            raise RuntimeError("inference request prefetch failed") from failure
        item = self._items.get(block=block, timeout=timeout)
        if isinstance(item, _InferenceRequestPrefetchFailure):
            raise RuntimeError("inference request prefetch failed") from item.error
        return item

    def close(self) -> None:
        """Stop and join the background reader."""
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._started:
            # A multiprocessing queue can remain blocked while reconstructing a
            # tensor after its producer disappears. The reader is a daemon, so
            # never let that partial IPC frame block inference-process teardown.
            self._thread.join(timeout=self._JOIN_SECONDS)

    def _start(self) -> None:
        if self._closed:
            raise RuntimeError("inference request prefetch queue is closed")
        if self._started:
            return
        self._started = True
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._source_queue.get(
                        block=True,
                        timeout=self._POLL_SECONDS,
                    )
                except queue.Empty:
                    continue
                except (EOFError, ConnectionError):
                    # Actor restart and orderly teardown can invalidate a tensor
                    # resource-sharer connection while the request queue remains
                    # usable for other producers.
                    continue
                if not self._put(item):
                    return
        except BaseException as exc:
            if self._stop_event.is_set():
                return
            self._failure = exc
            self._put(_InferenceRequestPrefetchFailure(exc))

    def _put(self, item: Any) -> bool:
        while not self._stop_event.is_set():
            try:
                self._items.put(item, timeout=self._POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False


def _async_inference_worker(
    *,
    config_data: Mapping[str, Any],
    output_dir: str,
    request_queue: Any,
    response_queues: Sequence[Any],
    teacher_response_queues: Sequence[Any] | None = None,
) -> None:
    config = RLTrainConfig.model_validate(config_data)
    prefetch_capacity = config.inference.request_prefetch_capacity
    if prefetch_capacity == 0:
        _run_async_inference_worker(
            config=config,
            output_dir=output_dir,
            request_queue=request_queue,
            response_queues=response_queues,
            teacher_response_queues=teacher_response_queues,
        )
        return
    with _PrefetchedInferenceRequestQueue(
        request_queue,
        capacity=prefetch_capacity,
    ) as prefetched_request_queue:
        _run_async_inference_worker(
            config=config,
            output_dir=output_dir,
            request_queue=prefetched_request_queue,
            response_queues=response_queues,
            teacher_response_queues=teacher_response_queues,
        )


def _run_async_inference_worker(
    *,
    config: RLTrainConfig,
    output_dir: str,
    request_queue: Any,
    response_queues: Sequence[Any],
    teacher_response_queues: Sequence[Any] | None = None,
    learner_runtime: _LearnerOwnedInferenceRuntime | None = None,
) -> None:
    if learner_runtime is None:
        _configure_gpu_worker_runtime(config)
        torch.manual_seed(config.seed + 3)
    device = _resolve_device(config.device)
    output_path = Path(output_dir)
    if learner_runtime is None:
        policy, candidate_model = _inference_server_policy(config, device=device)
    else:
        policy = learner_runtime.policy
        candidate_model = learner_runtime.candidate_model
    candidate_policy_aliases = None
    if config.amortized_policy_iteration.enabled:
        if not isinstance(candidate_model, AgentPolicyValueNet):
            raise RuntimeError("improvement inference requires a trainable model")
        candidate_policy_aliases = {
            "candidate_improvement": ImprovementRolloutPolicy(
                candidate_model,
                publication_source=policy,
                proposal_config=(config.amortized_policy_iteration.candidate_proposal),
                cmpo_config=config.amortized_policy_iteration.cmpo,
                seed=config.seed + 4,
                autocast=config.ppo.autocast,
                generator=(
                    None
                    if learner_runtime is None
                    else _seeded_torch_generator(device, seed=config.seed + 4)
                ),
            )
        }
    registry = InferencePolicyRegistry(
        candidate_policy=policy,
        candidate_model=candidate_model,
        weights_dir=(
            output_path / "weights"
            if candidate_model is not None and learner_runtime is None
            else None
        ),
        shared_weight_loader=(
            SharedMemoryWeightLoader(output_path / "weights")
            if candidate_model is not None
            and learner_runtime is None
            and config.learner.shared_memory_publish
            else None
        ),
        frozen_pool=FrozenPolicyPool(
            model_config=config.model,
            config=config.frozen_policy_pool,
            loader=(
                None
                if learner_runtime is None
                else partial(
                    load_frozen_rollout_policy,
                    sampling_seed=config.seed + 5,
                )
            ),
        ),
        frozen_state_path=_resolved_frozen_state_path(config, output_dir=output_path),
        registry_sync_interval_seconds=(
            config.diagnostics.inference_registry_sync_interval_seconds
        ),
        candidate_snapshot_factory=(
            None
            if config.planner is None and config.model.recurrent is None
            else partial(
                _build_inference_snapshot_policy,
                config=config,
                device=device,
                sampling_seed=(None if learner_runtime is None else config.seed + 6),
            )
        ),
        max_resident_snapshots=(
            config.planner.leases.max_resident_snapshots
            if config.planner is not None
            else (
                config.inference.recurrent_max_resident_snapshots
                if config.model.recurrent is not None
                else None
            )
        ),
        max_in_flight_leases=(
            config.planner.leases.max_in_flight_leases
            if config.planner is not None
            else (
                config.inference.recurrent_max_sequence_leases
                if config.model.recurrent is not None
                else None
            )
        ),
        max_recurrent_sequence_leases=(
            config.inference.recurrent_max_sequence_leases
            if config.model.recurrent is not None
            else None
        ),
        recurrent_replay_cache_capacity=(
            config.inference.recurrent_replay_cache_capacity
        ),
        candidate_snapshot_min_version_gap=(
            config.inference.recurrent_snapshot_min_version_gap
            if config.model.recurrent is not None
            else 1
        ),
        candidate_policy_aliases=candidate_policy_aliases,
    )
    if learner_runtime is None:
        _reset_torch_random_stream(config.seed + 3)
    response_queue_map = {
        f"actor-{actor_index}": response_queue
        for actor_index, response_queue in enumerate(response_queues)
    }
    response_queue_map.update(
        {
            f"teacher-{actor_index}": response_queue
            for actor_index, response_queue in enumerate(teacher_response_queues or ())
        }
    )
    summary_path = _inference_summary_path(output_path)
    started = time.perf_counter()
    served_steps = 0
    idle_steps = 0
    requests = 0
    decisions = 0
    responses = 0
    policy_batches = 0
    expired_teacher_requests = 0
    expired_teacher_decisions = 0
    planner_request_outcomes: Counter[str] = Counter()
    drained_by_policy: Counter[str] = Counter()
    request_batch_histogram: Counter[int] = Counter()
    policy_batch_histogram: Counter[int] = Counter()
    shape_histogram: Counter[str] = Counter()
    decode_step_histogram: Counter[int] = Counter()
    service_time_histogram_ms: Counter[str] = Counter()
    bucket_histogram: Counter[str] = Counter()
    bucket_fallback_histogram: Counter[str] = Counter()
    bucket_slot_totals: Counter[str] = Counter()
    serving_path_histogram: Counter[str] = Counter()
    graph_decode_totals: Counter[str] = Counter()
    graph_decode_resident_captures = 0
    graph_decode_peak_resident_captures = 0
    coalescing_totals: dict[str, float] = {}
    coalescing_latest: dict[str, float | int | bool] = {}
    latency_window_size = config.diagnostics.inference_latency_window
    request_latencies_ms = _LatencyAccumulator(latency_window_size)
    ipc_latency_segments_ms = _new_latency_segments(latency_window_size)
    phase_decisions: Counter[str] = Counter()
    phase_requests: Counter[str] = Counter()
    phase_request_latencies_ms: dict[str, _LatencyAccumulator] = {}
    phase_ipc_latency_segments_ms: dict[
        str,
        dict[str, _LatencyAccumulator],
    ] = {}
    server_stage_seconds = {
        "drain": 0.0,
        "sample": 0.0,
        "response": 0.0,
        "step": 0.0,
    }
    current_weight_version: int | None = int(getattr(policy, "policy_version", 0))
    last_sync: dict[str, Any] | None = None
    last_summary_write = time.perf_counter()
    _write_summary(
        summary_path,
        _inference_runtime_summary(
            started_at=started,
            served_steps=served_steps,
            idle_steps=idle_steps,
            requests=requests,
            decisions=decisions,
            responses=responses,
            policy_batches=policy_batches,
            expired_teacher_requests=expired_teacher_requests,
            expired_teacher_decisions=expired_teacher_decisions,
            planner_request_outcomes=planner_request_outcomes,
            drained_by_policy=drained_by_policy,
            request_batch_histogram=request_batch_histogram,
            policy_batch_histogram=policy_batch_histogram,
            shape_histogram=shape_histogram,
            decode_step_histogram=decode_step_histogram,
            service_time_histogram_ms=service_time_histogram_ms,
            bucket_histogram=bucket_histogram,
            bucket_fallback_histogram=bucket_fallback_histogram,
            bucket_slot_totals=bucket_slot_totals,
            request_latencies_ms=request_latencies_ms,
            ipc_latency_segments_ms=ipc_latency_segments_ms,
            phase_decisions=phase_decisions,
            phase_requests=phase_requests,
            phase_request_latencies_ms=phase_request_latencies_ms,
            phase_ipc_latency_segments_ms=phase_ipc_latency_segments_ms,
            server_stage_seconds=server_stage_seconds,
            current_weight_version=current_weight_version,
            last_sync=last_sync,
            serving_path_histogram=serving_path_histogram,
            graph_decode_totals=graph_decode_totals,
            graph_decode_resident_captures=graph_decode_resident_captures,
            graph_decode_peak_resident_captures=(graph_decode_peak_resident_captures),
            coalescing_totals=coalescing_totals,
            coalescing_latest=coalescing_latest,
        ),
    )
    while learner_runtime is None or not learner_runtime.stop_event.is_set():
        previous_policy_ids = (
            ()
            if last_sync is None
            else tuple(str(value) for value in last_sync.get("policy_ids", ()))
        )
        previous_frozen_load = (
            None if last_sync is None else last_sync.get("frozen_load")
        )
        sync_result = registry.sync()
        if sync_result.loaded_weight_version is not None:
            current_weight_version = sync_result.loaded_weight_version
        last_sync = {
            "loaded_weight_version": sync_result.loaded_weight_version,
            "current_weight_version": current_weight_version,
            "policy_ids": list(sync_result.policy_ids),
            "deferred_weight_version": sync_result.deferred_weight_version,
            "deferred_weight_reason": sync_result.deferred_weight_reason,
            "coalesced_weight_versions": sync_result.coalesced_weight_versions,
            "snapshot_min_version_gap": sync_result.snapshot_min_version_gap,
            "snapshot_pool": sync_result.snapshot_pool,
            "frozen_load": (
                None
                if sync_result.frozen_load is None
                else dict(sync_result.frozen_load)
            ),
            "frozen_update": (
                None
                if sync_result.frozen_update is None
                else {
                    "loaded": list(sync_result.frozen_update.loaded),
                    "unloaded": list(sync_result.frozen_update.unloaded),
                    "kept": list(sync_result.frozen_update.kept),
                }
            ),
        }
        if (
            sync_result.loaded_weight_version is not None
            or sync_result.policy_ids != previous_policy_ids
            or sync_result.frozen_load != previous_frozen_load
        ):
            _write_inference_registry_sync(
                summary_path,
                last_sync=last_sync,
                current_weight_version=current_weight_version,
            )
        if learner_runtime is None:
            stats = run_inference_server_step(
                request_queue=request_queue,
                response_queues=response_queue_map,
                policies=registry.routing_policies,
                config=config.inference,
            )
        else:
            # A hot D2D publication swaps all serving tensors under this same
            # lock, so no request can observe a partially copied generation.
            with learner_runtime.model_lock:
                stats = run_inference_server_step(
                    request_queue=request_queue,
                    response_queues=response_queue_map,
                    policies=registry.routing_policies,
                    config=config.inference,
                )
                current_weight_version = int(
                    getattr(policy, "policy_version", current_weight_version or 0)
                )
        expired_teacher_requests += stats.expired_teacher_requests
        expired_teacher_decisions += stats.expired_teacher_decisions
        planner_request_outcomes.update(
            {
                "expired_requests": stats.expired_planner_requests,
                "expired_decisions": stats.expired_planner_decisions,
                "lease_rejected_requests": (stats.rejected_planner_lease_requests),
                "lease_rejected_decisions": (stats.rejected_planner_lease_decisions),
                "oversized_rejected_requests": (stats.rejected_oversized_requests),
                "oversized_rejected_decisions": (stats.rejected_oversized_decisions),
            }
        )
        rejected_or_expired = (
            stats.expired_teacher_requests
            + stats.expired_planner_requests
            + stats.rejected_planner_lease_requests
            + stats.rejected_oversized_requests
        )
        if stats.requests == 0 and rejected_or_expired == 0:
            idle_steps += 1
            continue
        learner_phase = _current_inference_phase(output_path)
        served_steps += 1
        requests += stats.requests
        decisions += stats.decisions
        responses += stats.responses
        policy_batches += stats.policy_batches
        drained_by_policy.update(
            {str(key): int(value) for key, value in stats.drained_by_policy.items()}
        )
        request_batch_histogram.update(
            {
                int(key): int(value)
                for key, value in stats.request_batch_histogram.items()
            }
        )
        policy_batch_histogram.update(
            {
                int(key): int(value)
                for key, value in stats.policy_batch_histogram.items()
            }
        )
        _update_bounded_histogram(
            shape_histogram,
            stats.shape_histogram,
            max_keys=config.diagnostics.inference_shape_histogram_max_keys,
        )
        decode_step_histogram.update(
            {int(key): int(value) for key, value in stats.decode_step_histogram.items()}
        )
        service_time_histogram_ms.update(
            {
                str(key): int(value)
                for key, value in stats.service_time_histogram_ms.items()
            }
        )
        bucket_histogram.update(
            {str(key): int(value) for key, value in stats.bucket_histogram.items()}
        )
        bucket_fallback_histogram.update(
            {
                str(key): int(value)
                for key, value in stats.bucket_fallback_histogram.items()
            }
        )
        bucket_slot_totals.update(
            {str(key): int(value) for key, value in stats.bucket_slot_totals.items()}
        )
        serving_path_histogram.update(
            {
                str(key): int(value)
                for key, value in stats.serving_path_histogram.items()
            }
        )
        graph_decode_totals.update(
            {
                str(key): int(value)
                for key, value in stats.graph_decode_stats.items()
                if str(key) != "resident_captures"
            }
        )
        graph_decode_resident_captures = int(
            stats.graph_decode_stats.get(
                "resident_captures",
                graph_decode_resident_captures,
            )
        )
        graph_decode_peak_resident_captures = max(
            graph_decode_peak_resident_captures,
            graph_decode_resident_captures,
        )
        coalescing_latest = dict(stats.coalescing_stats)
        for key, value in stats.coalescing_stats.items():
            if isinstance(value, bool):
                total_key = f"{key}_steps"
                coalescing_totals[total_key] = coalescing_totals.get(
                    total_key, 0.0
                ) + float(value)
            elif key != "min_batch_target_decisions":
                total_key = str(key)
                coalescing_totals[total_key] = coalescing_totals.get(
                    total_key, 0.0
                ) + float(value)
        request_latencies_ms.extend(
            float(value) for value in stats.request_latencies_ms
        )
        _extend_latency_segments(
            ipc_latency_segments_ms,
            {
                "server_queue": stats.server_queue_latencies_ms,
                "server_forward": stats.server_forward_latencies_ms,
                "response_put": stats.response_put_latencies_ms,
            },
        )
        phase_requests[learner_phase] += stats.requests
        phase_decisions[learner_phase] += stats.decisions
        phase_request_latencies_ms.setdefault(
            learner_phase,
            _LatencyAccumulator(latency_window_size),
        ).extend(stats.request_latencies_ms)
        _extend_latency_segments(
            phase_ipc_latency_segments_ms.setdefault(
                learner_phase,
                _new_latency_segments(latency_window_size),
            ),
            {
                "server_queue": stats.server_queue_latencies_ms,
                "server_forward": stats.server_forward_latencies_ms,
                "response_put": stats.response_put_latencies_ms,
            },
        )
        server_stage_seconds["drain"] += stats.drain_seconds
        server_stage_seconds["sample"] += stats.sample_seconds
        server_stage_seconds["response"] += stats.response_seconds
        server_stage_seconds["step"] += stats.step_seconds
        _maybe_trim_inference_worker_memory(config, served_steps)
        now = time.perf_counter()
        if not _should_write_inference_summary(
            served_steps=served_steps,
            now=now,
            last_write=last_summary_write,
            interval_seconds=config.diagnostics.inference_summary_interval_seconds,
        ):
            continue
        last_summary_write = now
        _write_summary(
            summary_path,
            _inference_runtime_summary(
                started_at=started,
                served_steps=served_steps,
                idle_steps=idle_steps,
                requests=requests,
                decisions=decisions,
                responses=responses,
                policy_batches=policy_batches,
                expired_teacher_requests=expired_teacher_requests,
                expired_teacher_decisions=expired_teacher_decisions,
                planner_request_outcomes=planner_request_outcomes,
                drained_by_policy=drained_by_policy,
                request_batch_histogram=request_batch_histogram,
                policy_batch_histogram=policy_batch_histogram,
                shape_histogram=shape_histogram,
                decode_step_histogram=decode_step_histogram,
                service_time_histogram_ms=service_time_histogram_ms,
                bucket_histogram=bucket_histogram,
                bucket_fallback_histogram=bucket_fallback_histogram,
                bucket_slot_totals=bucket_slot_totals,
                request_latencies_ms=request_latencies_ms,
                ipc_latency_segments_ms=ipc_latency_segments_ms,
                phase_decisions=phase_decisions,
                phase_requests=phase_requests,
                phase_request_latencies_ms=phase_request_latencies_ms,
                phase_ipc_latency_segments_ms=phase_ipc_latency_segments_ms,
                server_stage_seconds=server_stage_seconds,
                current_weight_version=current_weight_version,
                last_sync=last_sync,
                serving_path_histogram=serving_path_histogram,
                graph_decode_totals=graph_decode_totals,
                graph_decode_resident_captures=graph_decode_resident_captures,
                graph_decode_peak_resident_captures=(
                    graph_decode_peak_resident_captures
                ),
                coalescing_totals=coalescing_totals,
                coalescing_latest=coalescing_latest,
            ),
        )


def _maybe_trim_inference_worker_memory(
    config: RLTrainConfig,
    served_steps: int,
) -> None:
    """Return freed allocator pages to the OS from the inference worker."""
    interval = config.execution.inference_cuda_memory_trim_interval_steps
    if interval is None or served_steps % interval != 0:
        return
    _trim_worker_memory(cuda=True)


def _maybe_trim_learner_memory(
    config: RLTrainConfig,
    *,
    published_version: int | None,
) -> None:
    """Return freed allocator pages to the OS from the learner process."""
    interval = config.execution.learner_cuda_memory_trim_interval_versions
    if interval is None:
        return
    if published_version is not None and published_version % interval != 0:
        return
    _trim_worker_memory(cuda=True)


def _trim_worker_memory(*, cuda: bool = False) -> None:
    """Return freed device, pinned-host, and libc allocator pages to the OS."""
    gc.collect()
    if cuda and torch.cuda.is_available():
        with suppress(RuntimeError):
            torch.cuda.empty_cache()
        # Learner minibatches use page-locked host tensors for asynchronous
        # copies. Their allocator is independent of both the CUDA device cache
        # and libc, so shape-varying windows otherwise retain another cache
        # generation after every policy version.
        empty_host_cache = getattr(torch.accelerator, "empty_host_cache", None)
        if callable(empty_host_cache):
            with suppress(RuntimeError):
                empty_host_cache()
    trim = _malloc_trim()
    if trim is None:
        return
    with suppress(OSError, RuntimeError):
        trim(0)


@dataclass
class _LatencyAccumulator:
    """Bounded latency window with cumulative count and mean."""

    maxlen: int
    values: deque[float] = field(init=False)
    total_count: int = 0
    total_sum: float = 0.0

    def __post_init__(self) -> None:
        """Initialize the bounded latency window."""
        self.values = deque(maxlen=self.maxlen)

    def extend(self, values: Iterable[float]) -> None:
        """Append latency observations while preserving cumulative totals."""
        for raw_value in values:
            value = float(raw_value)
            self.values.append(value)
            self.total_count += 1
            self.total_sum += value

    def summary(self) -> dict[str, float | int]:
        """Return cumulative mean plus bounded-window tail percentiles."""
        return {
            "count": self.total_count,
            "window_count": len(self.values),
            "mean": (
                self.total_sum / float(self.total_count)
                if self.total_count > 0
                else 0.0
            ),
            "p95": _sequence_percentile(tuple(self.values), 0.95),
        }


def _new_latency_segments(window_size: int) -> dict[str, _LatencyAccumulator]:
    """Return latency accumulators for inference IPC segments."""
    return {
        "server_queue": _LatencyAccumulator(window_size),
        "server_forward": _LatencyAccumulator(window_size),
        "response_put": _LatencyAccumulator(window_size),
    }


def _should_write_inference_summary(
    *,
    served_steps: int,
    now: float,
    last_write: float,
    interval_seconds: float,
) -> bool:
    """Return whether the inference worker should refresh its JSON summary."""
    return served_steps <= 3 or now - last_write >= interval_seconds


@cache
def _malloc_trim() -> Callable[[int], int] | None:
    """Return libc malloc_trim on Linux-like systems, when available."""
    if os.name != "posix":
        return None
    with suppress(OSError, AttributeError):
        libc = ctypes.CDLL("libc.so.6")
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return cast(Callable[[int], int], trim)
    return None


def _uses_central_curriculum(config: RLTrainConfig) -> bool:
    """Return whether async actors should use a central curriculum sampler."""
    return config.collection.opponent_mode == "curriculum" and (
        config.execution.actors > 1 or config.execution.inference_server
    )


def _curriculum_execution_lane(
    *,
    actor_index: int,
    actor_count: int,
    lane_count: int,
) -> int | None:
    """Map adjacent actors onto equal-width route-local execution lanes."""
    if lane_count <= 0:
        return None
    if actor_count <= 0 or actor_count % lane_count != 0:
        raise ValueError("actors must divide evenly across curriculum lanes")
    if actor_index < 0 or actor_index >= actor_count:
        raise ValueError("actor_index is outside the configured actor topology")
    actors_per_lane = actor_count // lane_count
    return actor_index // actors_per_lane


def _inference_summary_path(output_dir: Path) -> Path:
    return output_dir / "inference_summary.json"


def _write_inference_registry_sync(
    summary_path: Path,
    *,
    last_sync: Mapping[str, Any],
    current_weight_version: int | None,
) -> None:
    """Publish policy IDs immediately after a registry membership change."""
    summary = _read_optional_summary(summary_path) or {"status": "running"}
    summary.update(
        {
            "current_weight_version": current_weight_version,
            "last_sync": dict(last_sync),
            "registry_sync_updated_at_unix": time.time(),
        }
    )
    _write_summary(summary_path, summary)


def _inference_runtime_summary(
    *,
    started_at: float,
    served_steps: int,
    idle_steps: int,
    requests: int,
    decisions: int,
    responses: int,
    policy_batches: int,
    expired_teacher_requests: int,
    expired_teacher_decisions: int,
    planner_request_outcomes: Mapping[str, int],
    drained_by_policy: Counter[str],
    request_batch_histogram: Counter[int],
    policy_batch_histogram: Counter[int],
    shape_histogram: Counter[str],
    decode_step_histogram: Counter[int],
    service_time_histogram_ms: Counter[str],
    bucket_histogram: Counter[str],
    bucket_fallback_histogram: Counter[str],
    bucket_slot_totals: Counter[str],
    request_latencies_ms: Sequence[float] | _LatencyAccumulator,
    ipc_latency_segments_ms: Mapping[str, Sequence[float] | _LatencyAccumulator],
    phase_decisions: Counter[str],
    phase_requests: Counter[str],
    phase_request_latencies_ms: Mapping[str, Sequence[float] | _LatencyAccumulator],
    phase_ipc_latency_segments_ms: Mapping[
        str,
        Mapping[str, Sequence[float] | _LatencyAccumulator],
    ],
    server_stage_seconds: Mapping[str, float],
    current_weight_version: int | None,
    last_sync: Mapping[str, Any] | None,
    serving_path_histogram: Mapping[str, int] | None = None,
    graph_decode_totals: Mapping[str, int] | None = None,
    graph_decode_resident_captures: int = 0,
    graph_decode_peak_resident_captures: int = 0,
    coalescing_totals: Mapping[str, float] | None = None,
    coalescing_latest: Mapping[str, float | int | bool] | None = None,
) -> dict[str, Any]:
    elapsed_seconds = time.perf_counter() - started_at
    stage_seconds = {
        str(stage): float(seconds)
        for stage, seconds in sorted(server_stage_seconds.items())
    }
    request_latency_summary = _latency_values_summary(request_latencies_ms)
    return {
        "status": "running",
        "elapsed_seconds": elapsed_seconds,
        "served_steps": served_steps,
        "idle_steps": idle_steps,
        "requests": requests,
        "decisions": decisions,
        "responses": responses,
        "policy_batches": policy_batches,
        "expired_teacher_requests": expired_teacher_requests,
        "expired_teacher_decisions": expired_teacher_decisions,
        "planner_request_outcomes": dict(sorted(planner_request_outcomes.items())),
        "decisions_per_second": (
            decisions / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        ),
        "drained_by_policy": dict(sorted(drained_by_policy.items())),
        "request_batch_histogram": _json_int_keyed_counts(request_batch_histogram),
        "policy_batch_histogram": _json_int_keyed_counts(policy_batch_histogram),
        "shape_histogram": dict(sorted(shape_histogram.items())),
        "decode_step_histogram": _json_int_keyed_counts(decode_step_histogram),
        "service_time_histogram_ms": dict(sorted(service_time_histogram_ms.items())),
        "bucket_histogram": dict(sorted(bucket_histogram.items())),
        "bucket_fallback_histogram": dict(sorted(bucket_fallback_histogram.items())),
        "bucket_slot_totals": dict(sorted(bucket_slot_totals.items())),
        "serving_path_histogram": dict(sorted((serving_path_histogram or {}).items())),
        "graph_decode": {
            "totals": dict(sorted((graph_decode_totals or {}).items())),
            "resident_captures": graph_decode_resident_captures,
            "peak_resident_captures": graph_decode_peak_resident_captures,
        },
        "coalescing": {
            "totals": dict(sorted((coalescing_totals or {}).items())),
            "latest": dict(sorted((coalescing_latest or {}).items())),
        },
        "bucket_padding_waste_fraction": _bucket_padding_waste_fraction(
            bucket_slot_totals
        ),
        "request_batch_p50": _weighted_histogram_percentile(
            request_batch_histogram,
            0.50,
        ),
        "policy_batch_p50": _weighted_histogram_percentile(
            policy_batch_histogram,
            0.50,
        ),
        "server_stage_seconds": stage_seconds,
        "server_stage_fraction": {
            stage: seconds / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
            for stage, seconds in stage_seconds.items()
        },
        "request_latency_mean_ms": request_latency_summary["mean"],
        "request_latency_p95_ms": request_latency_summary["p95"],
        "request_latency_window_count": request_latency_summary.get(
            "window_count",
            request_latency_summary["count"],
        ),
        "ipc_latency_segments_ms": _latency_segment_summary(ipc_latency_segments_ms),
        "phase_latency": _inference_phase_latency_summary(
            phase_decisions=phase_decisions,
            phase_requests=phase_requests,
            phase_request_latencies_ms=phase_request_latencies_ms,
            phase_ipc_latency_segments_ms=phase_ipc_latency_segments_ms,
        ),
        "current_weight_version": current_weight_version,
        "last_sync": last_sync,
    }


def _json_int_keyed_counts(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _bucket_padding_waste_fraction(slot_totals: Mapping[str, int]) -> dict[str, float]:
    return {
        "decision": _slot_waste_fraction(slot_totals, "decision"),
        "token": _slot_waste_fraction(slot_totals, "token"),
        "option": _slot_waste_fraction(slot_totals, "option"),
        "cross": _slot_waste_fraction(slot_totals, "cross"),
    }


def _slot_waste_fraction(slot_totals: Mapping[str, int], prefix: str) -> float:
    actual = int(slot_totals.get(f"{prefix}_actual", 0))
    padded = int(slot_totals.get(f"{prefix}_padded", 0))
    if padded <= 0:
        return 0.0
    return max(0.0, float(padded - actual) / float(padded))


def _weighted_histogram_percentile(
    histogram: Mapping[int, int],
    percentile: float,
) -> float:
    total = sum(int(count) for count in histogram.values())
    if total <= 0:
        return 0.0
    threshold = max(1, math.ceil(percentile * total))
    cumulative = 0
    for value, count in sorted(histogram.items()):
        cumulative += int(count)
        if cumulative >= threshold:
            return float(value)
    return float(max(histogram))


def _update_bounded_histogram(
    target: Counter[str],
    values: Mapping[str, int],
    *,
    max_keys: int,
) -> None:
    """Accumulate exact keys up to a bound and merge new ones into overflow."""
    overflow_key = "__other__"
    for raw_key, raw_count in values.items():
        key = str(raw_key)
        count = int(raw_count)
        exact_key_count = len(target) - int(overflow_key in target)
        if key in target or (key != overflow_key and exact_key_count < max_keys - 1):
            target[key] += count
        else:
            target[overflow_key] += count


def _sequence_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _sequence_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(percentile * len(ordered)) - 1),
    )
    return ordered[index]


def _extend_latency_segments(
    target: dict[str, _LatencyAccumulator],
    source: Mapping[str, Sequence[float]],
) -> None:
    for segment, values in source.items():
        key = str(segment)
        if key not in target:
            window_size = next(iter(target.values())).maxlen if target else 1
            target[key] = _LatencyAccumulator(window_size)
        target[key].extend(values)


def _latency_segment_summary(
    segments: Mapping[str, Sequence[float] | _LatencyAccumulator],
) -> dict[str, dict[str, float | int]]:
    return {
        str(segment): _latency_values_summary(values)
        for segment, values in sorted(segments.items())
    }


def _latency_values_summary(
    values: Sequence[float] | _LatencyAccumulator,
) -> dict[str, float | int]:
    if isinstance(values, _LatencyAccumulator):
        return values.summary()
    return {
        "count": len(values),
        "mean": _sequence_mean(values),
        "p95": _sequence_percentile(values, 0.95),
    }


def _inference_phase_latency_summary(
    *,
    phase_decisions: Counter[str],
    phase_requests: Counter[str],
    phase_request_latencies_ms: Mapping[str, Sequence[float] | _LatencyAccumulator],
    phase_ipc_latency_segments_ms: Mapping[
        str,
        Mapping[str, Sequence[float] | _LatencyAccumulator],
    ],
) -> dict[str, dict[str, Any]]:
    phases = (
        set(phase_decisions)
        | set(phase_requests)
        | set(phase_request_latencies_ms)
        | set(phase_ipc_latency_segments_ms)
    )
    return {
        phase: {
            "requests": int(phase_requests.get(phase, 0)),
            "decisions": int(phase_decisions.get(phase, 0)),
            "request_latency_ms": _latency_values_summary(
                phase_request_latencies_ms.get(phase, ())
            ),
            "ipc_latency_segments_ms": _latency_segment_summary(
                phase_ipc_latency_segments_ms.get(phase, {})
            ),
        }
        for phase in sorted(phases)
    }


def _current_inference_phase(output_path: Path) -> str:
    phase = _learner_monitor_status(output_path).get("phase")
    if isinstance(phase, str) and phase:
        return phase
    return "unknown"


def _admit_curriculum_actor_incarnation(
    *,
    actor_index: int,
    actor_incarnation: int,
    current_actor_incarnations: dict[int, int],
    active_assignment_leases: dict[tuple[int, int, int], GameAssignment],
) -> tuple[bool, int]:
    """Admit one actor incarnation and revoke leases held by predecessors."""
    if actor_incarnation < 0:
        raise ValueError("curriculum actor_incarnation must be nonnegative")
    current = current_actor_incarnations.get(actor_index)
    if current is not None and actor_incarnation < current:
        return False, 0
    if current == actor_incarnation:
        return True, 0

    stale_keys = tuple(
        key
        for key in active_assignment_leases
        if key[0] == actor_index and key[1] != actor_incarnation
    )
    for key in stale_keys:
        del active_assignment_leases[key]
    current_actor_incarnations[actor_index] = actor_incarnation
    return True, len(stale_keys)


def _async_curriculum_worker(
    *,
    config_data: Mapping[str, Any],
    output_dir: str,
    request_queue: Any,
    response_queues: Sequence[Any],
) -> None:
    """Own curriculum sampling and outcome updates for async actors."""
    config = RLTrainConfig.model_validate(config_data)
    output_path = Path(output_dir)
    _maybe_seed_curriculum_anchor(config, output_dir=output_path)
    state_path = _resolved_frozen_state_path(config, output_dir=output_path)
    sampler = CurriculumSampler(
        config.curriculum.model_copy(update={"frozen_state_path": state_path}),
        rng=random.Random(config.seed + 31),
    )
    assigned_counts: Counter[str] = Counter()
    finished_counts: Counter[str] = Counter()
    deck_outcome_counts: dict[str, Counter[str]] = {}
    deck_winrate_ema: dict[str, float] = {}
    current_actor_incarnations: dict[int, int] = {}
    active_assignment_leases: dict[
        tuple[int, int, int],
        GameAssignment,
    ] = {}
    lease_protocol_counts: Counter[str] = Counter()
    additions_dir = _frozen_pool_addition_dir(output_path)
    consume_frozen_pool_additions(
        sampler,
        additions_dir,
        state_path=state_path,
    )
    league_controller = (
        LeaguePromotionController(
            config.frozen_league,
            sampler=sampler,
            pool_state_path=state_path,
            promotion_state_path=_league_promotion_state_path(output_path),
            candidates_dir=_frozen_league_candidate_dir(output_path),
        )
        if config.frozen_league.enabled
        else None
    )
    if league_controller is not None:
        league_controller.poll_candidates()
    summary_path = _async_curriculum_summary_path(output_path)
    _write_summary(
        summary_path,
        _async_curriculum_summary(
            sampler=sampler,
            state_path=state_path,
            assigned_counts=assigned_counts,
            finished_counts=finished_counts,
            deck_outcome_counts=deck_outcome_counts,
            deck_winrate_ema=deck_winrate_ema,
            league_controller=league_controller,
            status="running",
            active_assignment_leases=active_assignment_leases,
            current_actor_incarnations=current_actor_incarnations,
            lease_protocol_counts=lease_protocol_counts,
        ),
    )
    persist_interval = config.diagnostics.curriculum_persist_interval_seconds
    last_state_write = time.monotonic()
    last_summary_write = last_state_write
    state_dirty = False
    promotion_state_dirty = False
    summary_dirty = False

    def persist_dirty_state(*, force: bool = False) -> None:
        """Debounce growing curriculum JSON artifacts off the message hot path."""
        nonlocal last_state_write, last_summary_write
        nonlocal state_dirty, promotion_state_dirty, summary_dirty
        now = time.monotonic()
        if (state_dirty or promotion_state_dirty) and (
            force or now - last_state_write >= persist_interval
        ):
            if state_dirty:
                sampler.save_state(state_path)
            if promotion_state_dirty and league_controller is not None:
                league_controller.save_state()
            state_dirty = False
            promotion_state_dirty = False
            last_state_write = now
        if summary_dirty and (force or now - last_summary_write >= persist_interval):
            _write_summary(
                summary_path,
                _async_curriculum_summary(
                    sampler=sampler,
                    state_path=state_path,
                    assigned_counts=assigned_counts,
                    finished_counts=finished_counts,
                    deck_outcome_counts=deck_outcome_counts,
                    deck_winrate_ema=deck_winrate_ema,
                    league_controller=league_controller,
                    status="running",
                    active_assignment_leases=active_assignment_leases,
                    current_actor_incarnations=current_actor_incarnations,
                    lease_protocol_counts=lease_protocol_counts,
                ),
            )
            summary_dirty = False
            last_summary_write = now

    while True:
        if league_controller is not None and league_controller.poll_candidates():
            state_dirty = False
            promotion_state_dirty = False
            last_state_write = time.monotonic()
            summary_dirty = True
        consumed_additions = consume_frozen_pool_additions(
            sampler,
            additions_dir,
            state_path=state_path,
        )
        if consumed_additions:
            state_dirty = False
            last_state_write = time.monotonic()
            summary_dirty = True
        try:
            message = request_queue.get(timeout=1.0)
        except queue.Empty:
            persist_dirty_state()
            continue
        if message is None:
            break
        if not isinstance(message, Mapping):
            raise ValueError("curriculum message must be a mapping")
        message_type = message.get("type")
        if message_type == "sample":
            actor_index = int(message["actor_index"])
            if not 0 <= actor_index < len(response_queues):
                raise ValueError(
                    f"curriculum actor_index is out of range: {actor_index}"
                )
            actor_incarnation = int(message["actor_incarnation"])
            request_id = int(message["request_id"])
            admitted, canceled_leases = _admit_curriculum_actor_incarnation(
                actor_index=actor_index,
                actor_incarnation=actor_incarnation,
                current_actor_incarnations=current_actor_incarnations,
                active_assignment_leases=active_assignment_leases,
            )
            if not admitted:
                lease_protocol_counts["stale_sample_requests"] += 1
                summary_dirty = True
                response_queues[actor_index].put(
                    {
                        "request_id": request_id,
                        "actor_incarnation": actor_incarnation,
                        "error": (
                            "stale actor incarnation; current="
                            f"{current_actor_incarnations[actor_index]}"
                        ),
                    }
                )
                persist_dirty_state()
                continue
            if canceled_leases:
                lease_protocol_counts["canceled_replaced_actor_leases"] += (
                    canceled_leases
                )
            lane_index = _curriculum_execution_lane(
                actor_index=actor_index,
                actor_count=config.execution.actors,
                lane_count=(config.curriculum.assignment_schedule.execution_lanes),
            )
            lease_key = (actor_index, actor_incarnation, request_id)
            assignment = active_assignment_leases.get(lease_key)
            if assignment is None:
                assignment = sampler.assign(lane_index=lane_index)
                active_assignment_leases[lease_key] = assignment
                _record_assignment_counts(assignment, assigned_counts)
                if config.curriculum.assignment_schedule.mode == "stratified":
                    state_dirty = True
            else:
                lease_protocol_counts["replayed_sample_requests"] += 1
            summary_dirty = True
            response_queues[actor_index].put(
                {
                    "request_id": request_id,
                    "actor_incarnation": actor_incarnation,
                    "assignment": assignment,
                }
            )
        elif message_type == "cancel":
            actor_index = int(message["actor_index"])
            if not 0 <= actor_index < len(response_queues):
                raise ValueError(
                    f"curriculum actor_index is out of range: {actor_index}"
                )
            actor_incarnation = int(message["actor_incarnation"])
            request_id = int(message["request_id"])
            raw_assignment_request_ids = message.get("assignment_request_ids")
            if not isinstance(raw_assignment_request_ids, Sequence) or isinstance(
                raw_assignment_request_ids,
                str,
            ):
                raise ValueError(
                    "curriculum cancellation request IDs must be a sequence"
                )
            assignment_request_ids = tuple(
                int(value) for value in raw_assignment_request_ids
            )
            if len(assignment_request_ids) != len(set(assignment_request_ids)):
                raise ValueError("curriculum cancellation request IDs must be unique")
            if current_actor_incarnations.get(actor_index) != actor_incarnation:
                lease_protocol_counts["stale_cancellations"] += 1
                summary_dirty = True
                response_queues[actor_index].put(
                    {
                        "request_id": request_id,
                        "actor_incarnation": actor_incarnation,
                        "error": (
                            "stale actor incarnation; current="
                            f"{current_actor_incarnations.get(actor_index)}"
                        ),
                    }
                )
                persist_dirty_state()
                continue
            canceled = 0
            duplicates = 0
            for assignment_request_id in assignment_request_ids:
                lease_key = (
                    actor_index,
                    actor_incarnation,
                    assignment_request_id,
                )
                if active_assignment_leases.pop(lease_key, None) is None:
                    duplicates += 1
                else:
                    canceled += 1
            if canceled:
                lease_protocol_counts["canceled_stale_recurrent_leases"] += canceled
            if duplicates:
                lease_protocol_counts["duplicate_cancellations"] += duplicates
            summary_dirty = True
            response_queues[actor_index].put(
                {
                    "request_id": request_id,
                    "actor_incarnation": actor_incarnation,
                    "acknowledged_assignment_request_ids": list(assignment_request_ids),
                }
            )
        elif message_type == "outcome":
            actor_index = int(message["actor_index"])
            actor_incarnation = int(message["actor_incarnation"])
            request_id = int(message["assignment_request_id"])
            lease_key = (actor_index, actor_incarnation, request_id)
            if current_actor_incarnations.get(actor_index) != actor_incarnation:
                lease_protocol_counts["stale_outcomes"] += 1
                summary_dirty = True
                persist_dirty_state()
                continue
            assignment = active_assignment_leases.get(lease_key)
            if assignment is None:
                lease_protocol_counts["unknown_or_duplicate_outcomes"] += 1
                summary_dirty = True
                persist_dirty_state()
                continue
            candidate_reward = float(message["candidate_reward"])
            raw_candidate_policy_version = message.get("candidate_policy_version")
            candidate_policy_version = (
                None
                if raw_candidate_policy_version is None
                else int(raw_candidate_policy_version)
            )
            _record_finished_counts(
                assignment,
                candidate_reward=candidate_reward,
                finished_counts=finished_counts,
                deck_outcome_counts=deck_outcome_counts,
                deck_winrate_ema=deck_winrate_ema,
                winrate_ema_alpha=config.curriculum.winrate_ema_alpha,
            )
            outcome = CurriculumOutcome(
                opponent_kind=assignment.opponent_kind,
                opponent_id=assignment.opponent_id,
                candidate_reward=candidate_reward,
                candidate_deck_label=assignment.candidate_deck_label,
                opponent_deck_label=assignment.opponent_deck_label,
                candidate_policy_version=candidate_policy_version,
            )
            sampler.observe(outcome)
            if league_controller is not None and league_controller.observe(outcome):
                promotion_state_dirty = True
                decision = league_controller.maybe_decide()
                if decision is not None:
                    state_dirty = False
                    promotion_state_dirty = False
                    last_state_write = time.monotonic()
            state_dirty = True
            summary_dirty = True
            consumed_additions = consume_frozen_pool_additions(
                sampler,
                additions_dir,
                state_path=state_path,
            )
            if consumed_additions:
                state_dirty = False
                last_state_write = time.monotonic()
            del active_assignment_leases[lease_key]
        else:
            raise ValueError(f"unknown curriculum message type: {message_type}")
        drained_retirements = _remove_drained_retired_frozen_members(
            sampler,
            active_assignments=active_assignment_leases.values(),
        )
        if drained_retirements:
            state_dirty = True
            summary_dirty = True
        persist_dirty_state()
    consume_frozen_pool_additions(
        sampler,
        additions_dir,
        state_path=state_path,
    )
    if active_assignment_leases:
        lease_protocol_counts["canceled_shutdown_leases"] += len(
            active_assignment_leases
        )
        active_assignment_leases.clear()
    if _remove_drained_retired_frozen_members(sampler, active_assignments=()):
        state_dirty = True
    sampler.finalize_assignment_state()
    state_dirty = True
    promotion_state_dirty = league_controller is not None
    summary_dirty = True
    persist_dirty_state(force=True)
    _write_summary(
        summary_path,
        _async_curriculum_summary(
            sampler=sampler,
            state_path=state_path,
            assigned_counts=assigned_counts,
            finished_counts=finished_counts,
            deck_outcome_counts=deck_outcome_counts,
            deck_winrate_ema=deck_winrate_ema,
            league_controller=league_controller,
            status="completed",
            active_assignment_leases=active_assignment_leases,
            current_actor_incarnations=current_actor_incarnations,
            lease_protocol_counts=lease_protocol_counts,
        ),
    )


def _finish_async_curriculum_worker(
    config: RLTrainConfig,
    *,
    output_dir: Path,
    process: ManagedProcess,
    request_queue: Any,
) -> dict[str, Any]:
    """Stop the central curriculum worker and return its persisted summary."""
    if process.is_alive():
        put_nowait = getattr(request_queue, "put_nowait", None)
        try:
            if callable(put_nowait):
                put_nowait(None)
            else:
                request_queue.put(None, block=False)
        except (TypeError, queue.Full, BrokenPipeError, EOFError, OSError, ValueError):
            # Cleanup must remain bounded when a wedged consumer leaves its
            # request queue full or disconnected. Terminate/kill below is the
            # authoritative fallback.
            pass
    process.join(timeout=config.execution.async_supervisor.process_join_timeout_seconds)
    terminated = False
    if process.is_alive():
        terminated = True
        stop_managed_processes(
            (process,),
            join_timeout_seconds=(
                config.execution.async_supervisor.process_join_timeout_seconds
            ),
        )
    summary = _read_optional_summary(_async_curriculum_summary_path(output_dir)) or {
        "status": "missing",
        "mode": "central_curriculum",
        "assigned": {},
        "finished": {},
    }
    summary["process_exitcode"] = process.exitcode
    if terminated:
        summary["status"] = "terminated"
    elif process.exitcode not in (0, None):
        summary["status"] = "process_failed"
    return summary


def _async_curriculum_summary(
    *,
    sampler: CurriculumSampler,
    state_path: Path,
    assigned_counts: Counter[str],
    finished_counts: Counter[str],
    deck_outcome_counts: Mapping[str, Counter[str]],
    deck_winrate_ema: Mapping[str, float],
    league_controller: LeaguePromotionController | None,
    status: str,
    active_assignment_leases: Mapping[
        tuple[int, int, int],
        GameAssignment,
    ],
    current_actor_incarnations: Mapping[int, int],
    lease_protocol_counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "status": status,
        "mode": "central_curriculum",
        "state_path": deck_records.display_path(state_path),
        "curriculum_config": sampler.config.model_dump(mode="json"),
        "assigned": dict(sorted(assigned_counts.items())),
        "finished": dict(sorted(finished_counts.items())),
        "candidate_deck_pool": list(sampler.candidate_deck_distribution),
        "candidate_lanes": _curriculum_lane_summary(
            sampler,
            assigned_counts=assigned_counts,
            finished_counts=finished_counts,
        ),
        "matchups": list(sampler.matchup_statistics),
        "decks": _curriculum_deck_summary(
            sampler=sampler,
            assigned_counts=assigned_counts,
            finished_counts=finished_counts,
            deck_outcome_counts=deck_outcome_counts,
            deck_winrate_ema=sampler.state().candidate_deck_winrate_ema
            or deck_winrate_ema,
        ),
        "frozen_members": len(sampler.frozen_members),
        "assignment_leases": {
            "active": len(active_assignment_leases),
            "actor_incarnations": {
                str(actor_index): actor_incarnation
                for actor_index, actor_incarnation in sorted(
                    current_actor_incarnations.items()
                )
            },
            "protocol_counts": dict(sorted(lease_protocol_counts.items())),
        },
        "league_promotion": (
            None if league_controller is None else league_controller.summary()
        ),
        "assignment_stream": {
            "mode": sampler.config.assignment_schedule.mode,
            "cursor": sampler.state().assignment_cursor,
            "reserved_until": sampler.state().assignment_reserved_until,
            "schedule_fingerprint": (sampler.state().assignment_schedule_fingerprint),
            "layout_fingerprint": sampler.state().assignment_layout_fingerprint,
            "next_template_index": (sampler.state().assignment_next_template_index),
            "pending_template_indices": list(
                sampler.state().assignment_pending_template_indices
            ),
            "lanes": {
                key: lane.model_dump(mode="json")
                for key, lane in sorted(sampler.state().assignment_lanes.items())
            },
        },
    }


def _async_curriculum_summary_path(output_dir: Path) -> Path:
    return output_dir / "curriculum" / "central_summary.json"


def _inference_server_policy(
    config: RLTrainConfig,
    *,
    device: torch.device,
) -> tuple[RolloutPolicy, torch.nn.Module | None]:
    if config.collection.policy_kind == "min_count":
        return (MinCountRolloutPolicy(), None)
    model = _load_training_model(config, device=device)
    model.eval()
    serving_model = _maybe_compile_inference_model(
        model,
        enabled=(
            config.planner is None
            and config.inference.compile_model
            and not config.inference.graph_decode
        ),
    )
    return (
        ModelRolloutPolicy(
            serving_model,
            policy_version=_initial_policy_version(config),
            autocast=config.ppo.autocast,
            graph_decode=config.inference.graph_decode,
            graph_warmup_steps=config.inference.graph_warmup_steps,
            graph_max_captures=config.inference.graph_max_captures,
            graph_capture_idle_replays=(config.inference.graph_capture_idle_replays),
            planner_context_capacity=(
                0
                if config.planner is None
                else config.planner.contexts.retained_root_rows
            ),
        ),
        model,
    )


def _build_inference_snapshot_policy(
    state_dict: Mapping[str, Any],
    version: int,
    fingerprint: str,
    *,
    config: RLTrainConfig,
    device: torch.device,
    sampling_seed: int | None = None,
) -> ModelRolloutPolicy:
    """Strictly construct one fresh immutable leased-serving snapshot."""
    if config.planner is None and config.model.recurrent is None:
        raise ValueError("snapshot construction requires a leased runtime")
    model = build_agent_policy_value_net(
        _training_model_config(config, checkpoint=None)
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return ModelRolloutPolicy(
        model,
        policy_version=int(version),
        autocast=config.ppo.autocast,
        graph_decode=config.inference.graph_decode,
        graph_warmup_steps=config.inference.graph_warmup_steps,
        graph_max_captures=config.inference.graph_max_captures,
        graph_capture_idle_replays=config.inference.graph_capture_idle_replays,
        planner_context_capacity=(
            0 if config.planner is None else config.planner.contexts.retained_root_rows
        ),
        verified_model_fingerprint=fingerprint,
        generator=(
            None
            if sampling_seed is None
            else _seeded_torch_generator(
                device,
                seed=sampling_seed + int(version),
            )
        ),
    )


def _maybe_compile_inference_model(
    model: AgentPolicyValueNet,
    *,
    enabled: bool,
) -> AgentPolicyValueNet:
    if not enabled:
        return model
    return cast(
        AgentPolicyValueNet,
        torch.compile(model, mode="reduce-overhead"),
    )


def _stop_managed_process(
    process: ManagedProcess,
    *,
    join_timeout_seconds: float,
) -> None:
    stop_managed_processes(
        (process,),
        join_timeout_seconds=join_timeout_seconds,
    )


def _wait_for_async_learner_readiness(learner_ready_event: Any | None) -> None:
    """Keep actors off inference queues until the learner is ready to serve."""
    if learner_ready_event is not None:
        learner_ready_event.wait()


def _async_actor_worker(
    *,
    config_data: Mapping[str, Any],
    output_dir: str,
    trajectory_queue: Any,
    reanalysis_root_queue: Any | None = None,
    pool_factory: RLTrainPoolFactory | None,
    actor_index: int = 0,
    actor_incarnation: int = 0,
    inference_request_queue: Any | None = None,
    inference_response_queue: Any | None = None,
    teacher_inference_response_queue: Any | None = None,
    curriculum_request_queue: Any | None = None,
    curriculum_response_queue: Any | None = None,
    collection_gate: Any | None = None,
    learner_ready_event: Any | None = None,
) -> None:
    config = RLTrainConfig.model_validate(config_data)
    if isinstance(trajectory_queue, SharedTrajectoryRing):
        trajectory_queue.bind_producer(actor_index)
    _wait_for_async_learner_readiness(learner_ready_event)
    _configure_async_actor_threads()
    torch.manual_seed(config.seed + 1 + actor_index)
    device = (
        torch.device("cpu")
        if config.execution.inference_server
        else _resolve_device(config.device)
    )
    output_path = Path(output_dir)
    deck_pair = _deck_pair(config.collection.decks)
    factory = pool_factory or _training_pool_factory(
        include_search_input=_training_include_search_input(config)
    )
    curriculum_manager = _async_actor_curriculum_manager(
        config,
        output_dir=output_path,
        actor_index=actor_index,
        actor_incarnation=actor_incarnation,
        curriculum_request_queue=curriculum_request_queue,
        curriculum_response_queue=curriculum_response_queue,
    )
    deck_pair_sampler = (
        curriculum_manager.sample_deck_pair
        if curriculum_manager is not None
        else lambda: deck_pair
    )
    remote_client_state = (
        RemoteInferenceClientState() if config.execution.inference_server else None
    )

    policy, weight_model = _async_actor_policy(
        config,
        device=device,
        actor_index=actor_index,
        actor_incarnation=actor_incarnation,
        inference_request_queue=inference_request_queue,
        inference_response_queue=inference_response_queue,
        remote_client_state=remote_client_state,
    )
    engine_teacher_producer = _training_engine_teacher_producer(
        config,
        policy=policy,
        device=device,
        actor_index=actor_index,
        actor_incarnation=actor_incarnation,
        inference_request_queue=inference_request_queue,
        inference_response_queue=inference_response_queue,
        teacher_inference_response_queue=teacher_inference_response_queue,
        remote_client_state=remote_client_state,
    )
    remote_inference_routes = (
        None
        if not config.execution.inference_server
        else _remote_inference_routes(
            config,
            output_dir=output_path,
            actor_index=actor_index,
            actor_incarnation=actor_incarnation,
            inference_request_queue=inference_request_queue,
            inference_response_queue=inference_response_queue,
            remote_client_state=remote_client_state,
        )
    )
    weight_loader = (
        WeightPollingLoader(output_path / "weights")
        if weight_model is not None
        else None
    )
    recurrent_stale_game_watcher = (
        RecurrentStaleGameWatcher(
            output_path / "weights",
            poll_interval_seconds=(
                config.inference.recurrent_stale_game_recycling.poll_interval_seconds
            ),
        )
        if config.inference.recurrent_stale_game_recycling.enabled
        else None
    )
    # Remote recurrent policies roll immutable snapshots per game-seat sequence.
    # The inference server owns publication and model leases, so actors must keep
    # admitting refill games while older sequences finish on their bound artifact.
    archive_recorder = (
        _async_archive_recorder() if config.execution.archive_trajectories else None
    )
    summary_path = _actor_summary_path(
        output_path,
        actor_index=actor_index,
        actor_count=config.execution.actors,
    )
    last_progress_summary_write = 0.0
    reanalysis_counters: Counter[str] = Counter()

    def persist_progress(stats: ActorLoopStats) -> None:
        nonlocal last_progress_summary_write
        now = time.perf_counter()
        if (
            last_progress_summary_write > 0.0
            and now - last_progress_summary_write < 1.0
        ):
            return
        last_progress_summary_write = now
        _write_actor_summary(
            summary_path,
            actor_index=actor_index,
            stats=stats,
            curriculum_summary=None,
            reanalysis_summary=reanalysis_counters,
        )

    planner_runtime_instance = (
        None
        if config.planner is None
        else create_planner_behavior_runtime(
            runtime_config=config.planner,
            sampler_config=config.rollout_probe.sampler,
            belief_config=config.rollout_belief,
            stochastic_seed=config.seed + 200_003 * (actor_index + 1),
        )
    )
    planner_lifecycle: AbstractContextManager[PlannerBehaviorRuntime | None] = (
        nullcontext(None)
        if planner_runtime_instance is None
        else planner_runtime_instance
    )
    active_teacher_producer: (
        OnlineEngineTeacherProducer
        | BackgroundEngineTeacherProducer
        | BackgroundMacroTeacherProducer
        | None
    ) = engine_teacher_producer
    if config.macro_credit.native_teacher_enabled:
        if planner_runtime_instance is None:
            raise RuntimeError("native macro teacher planner runtime is unavailable")
        active_teacher_producer = BackgroundMacroTeacherProducer(
            planner_runtime_instance.service,
            queue_batches=config.macro_credit.async_queue_batches,
        )
    teacher_lifecycle = (
        nullcontext()
        if active_teacher_producer is None
        else closing(active_teacher_producer)
    )
    with (
        planner_lifecycle as planner_runtime,
        teacher_lifecycle,
        factory(config.collection.num_concurrent_games, deck_pair_sampler) as pool,
    ):
        if curriculum_manager is not None:
            curriculum_manager.sync_live_games(_live_games(pool))
        actor_setup = _training_rollout_actors(
            config,
            candidate_policy=policy,
            curriculum_manager=curriculum_manager,
            remote_inference_routes=remote_inference_routes,
        )
        stats = run_actor_loop(
            pool=pool,
            actors=actor_setup.actors,
            trajectory_queue=trajectory_queue,
            config=ActorLoopConfig(
                max_iterations=config.collection.max_collect_iterations,
                sampling_temperature=config.collection.sampling_temperature,
                queue=TrajectoryQueueProducerConfig(
                    profile_pickle=config.execution.profile_actor_queue_pickle,
                ),
                rollout_probe=config.rollout_probe,
                rollout_belief=config.rollout_belief,
                factual=config.factual,
                macro_credit=config.macro_credit,
                reanalysis_root_probability=(
                    config.amortized_policy_iteration.root_sample_probability
                    if config.amortized_policy_iteration.enabled
                    else 0.0
                ),
                stats_callback_interval_seconds=1.0,
                seed=config.seed + 100_003 * (actor_index + 1),
                record_public_event_deltas=config.model.recurrent is not None,
            ),
            weight_loader=weight_loader,
            weight_model=weight_model,
            recurrent_stale_game_watcher=recurrent_stale_game_watcher,
            recurrent_max_staleness=(
                config.learner.max_staleness
                if recurrent_stale_game_watcher is not None
                else None
            ),
            device=device,
            archive_recorder=archive_recorder,
            decision_callback=(
                None
                if curriculum_manager is None
                else curriculum_manager.observe_decision
            ),
            finished_callback=(
                None if curriculum_manager is None else curriculum_manager.finalize
            ),
            discard_callback=(
                None if curriculum_manager is None else curriculum_manager.cancel
            ),
            post_step_callback=(
                None
                if curriculum_manager is None
                else lambda: curriculum_manager.sync_live_games(_live_games(pool))
            ),
            stats_callback=persist_progress,
            trajectory_transform=(
                lambda trajectory: _async_actor_queue_trajectory(
                    trajectory,
                    actor_index=actor_index,
                    actor_count=config.execution.actors,
                    reanalysis_root_queue=reanalysis_root_queue,
                    reanalysis_counters=reanalysis_counters,
                )
            ),
            engine_teacher_producer=active_teacher_producer,
            planner_behavior_service=(
                None
                if planner_runtime is None or config.macro_credit.native_teacher_enabled
                else planner_runtime.service
            ),
            collection_gate=collection_gate,
            reanalysis_root_callback=(
                None
                if reanalysis_root_queue is None
                else partial(
                    _publish_async_reanalysis_root,
                    actor_index=actor_index,
                    actor_count=config.execution.actors,
                    reanalysis_root_queue=reanalysis_root_queue,
                    reanalysis_counters=reanalysis_counters,
                )
            ),
        )
    curriculum_summary = None
    if curriculum_manager is not None:
        curriculum_manager.save_state()
        curriculum_summary = curriculum_manager.summary(actor_setup.frozen_update)
    _write_actor_summary(
        summary_path,
        actor_index=actor_index,
        stats=stats,
        curriculum_summary=curriculum_summary,
        reanalysis_summary=reanalysis_counters,
    )


def _configure_async_actor_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _configure_gpu_worker_matmul() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _configure_gpu_worker_runtime(config: RLTrainConfig) -> None:
    thread_count = config.execution.gpu_worker_torch_threads
    if thread_count is not None:
        os.environ["OMP_NUM_THREADS"] = str(thread_count)
        os.environ["MKL_NUM_THREADS"] = str(thread_count)
        torch.set_num_threads(thread_count)
        with suppress(RuntimeError):
            torch.set_num_interop_threads(thread_count)
    _configure_gpu_worker_matmul()


def _reset_torch_random_stream(seed: int) -> None:
    """Restart sampling RNG after architecture-dependent module construction."""
    torch.manual_seed(seed)


def _seeded_torch_generator(
    device: torch.device,
    *,
    seed: int,
) -> torch.Generator:
    """Create a device-local RNG without advancing the process-global stream."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


class _LearnerOwnedInferenceService:
    """Serve rollout requests from the learner's single CUDA process.

    Training and serving keep distinct model tensors, but share one CUDA
    context and allocator. Generation publication is a direct device-to-device
    copy guarded against concurrent inference; no checkpoint serialization,
    IPC deserialization, or second model hash is needed on the hot path.
    """

    _JOIN_TIMEOUT_SECONDS = 30.0

    def __init__(
        self,
        *,
        config: RLTrainConfig,
        output_path: Path,
        learner_model: AgentPolicyValueNet,
        current_policy_version: int,
        device: torch.device,
        request_queue: Any,
        response_queues: Sequence[Any],
        teacher_response_queues: Sequence[Any] | None,
    ) -> None:
        if config.execution.gpu_service_mode != "learner_thread":
            raise ValueError("learner-owned service requires learner_thread mode")
        # The shadow model's random initialization is overwritten immediately.
        # Preserve the learner's process-global RNG so adding this operational
        # serving copy cannot change any later stochastic optimizer work.
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            serving_model = build_agent_policy_value_net(learner_model.config).to(
                device
            )
        serving_model.load_state_dict(learner_model.state_dict(), strict=True)
        serving_model.eval()
        graph_capture_allowed = threading.Event()
        serving_execution_model = _maybe_compile_inference_model(
            serving_model,
            enabled=(
                config.inference.compile_model and not config.inference.graph_decode
            ),
        )
        self._policy = ModelRolloutPolicy(
            serving_execution_model,
            policy_version=current_policy_version,
            autocast=config.ppo.autocast,
            graph_decode=config.inference.graph_decode,
            graph_warmup_steps=config.inference.graph_warmup_steps,
            graph_max_captures=config.inference.graph_max_captures,
            graph_capture_idle_replays=(config.inference.graph_capture_idle_replays),
            graph_capture_allowed=graph_capture_allowed.is_set,
            # New capture is unsafe while learner kernels are launching, but
            # resident graphs only read the distinct serving model and can
            # replay concurrently on the inference stream.
            graph_replay_allowed=lambda: True,
            generator=_seeded_torch_generator(device, seed=config.seed + 3),
        )
        self._serving_model = serving_model
        self._config = config
        self._output_path = output_path
        self._request_queue = request_queue
        self._response_queues = response_queues
        self._teacher_response_queues = teacher_response_queues
        self._runtime = _LearnerOwnedInferenceRuntime(
            policy=self._policy,
            candidate_model=serving_model,
            stop_event=threading.Event(),
            model_lock=threading.Lock(),
            graph_capture_allowed=graph_capture_allowed,
        )
        self._thread = threading.Thread(
            target=self._thread_main,
            name="learner-owned-inference",
            daemon=True,
        )
        self._failure: BaseException | None = None
        self._publication_count = 0
        self._publication_copy_seconds = 0.0
        self._last_publication_copy_seconds = 0.0
        # A resumed process may publish the same logical version after its last
        # durable boundary. Bind non-artifact leases to this process attempt so
        # such generations can never collide despite sharing run/version data.
        self._runtime_attempt_id = uuid.uuid4().hex

    @property
    def runtime_attempt_id(self) -> str:
        """Return the unique identity namespace for this serving process."""
        return self._runtime_attempt_id

    def start(self) -> None:
        """Start the bounded request-serving thread."""
        summary_path = _inference_summary_path(self._output_path)
        summary = _read_optional_summary(summary_path) or {}
        summary.update(
            {
                "status": "starting",
                "service_mode": "learner_thread",
                "runtime_attempt_id": self._runtime_attempt_id,
            }
        )
        _write_summary(summary_path, summary)
        self._thread.start()

    def publish(
        self,
        model: AgentPolicyValueNet,
        version: int,
        model_fingerprint: str,
    ) -> None:
        """Atomically expose one exact learner generation to rollout actors."""
        self.raise_if_failed()
        started_at = time.perf_counter()
        with self._runtime.model_lock:
            self._serving_model.load_state_dict(model.state_dict(), strict=True)
            if next(self._serving_model.parameters()).device.type == "cuda":
                torch.cuda.synchronize(next(self._serving_model.parameters()).device)
            self._policy.reset_planner_contexts(
                policy_version=version,
                model_fingerprint=model_fingerprint,
            )
        elapsed = time.perf_counter() - started_at
        self._publication_count += 1
        self._publication_copy_seconds += elapsed
        self._last_publication_copy_seconds = elapsed

    def allow_graph_capture(self) -> None:
        """Open graph capture and replay after learner CUDA work is quiescent."""
        self.raise_if_failed()
        if not self._policy.graph_decode_enabled:
            return
        with self._runtime.model_lock:
            parameter = next(self._serving_model.parameters())
            if parameter.device.type == "cuda":
                torch.cuda.synchronize(parameter.device)
            self._runtime.graph_capture_allowed.set()

    def suspend_graph_capture(self) -> None:
        """Close graph capture/replay and await in-flight serving completion."""
        self.raise_if_failed()
        if not self._policy.graph_decode_enabled:
            return
        # Clear first so a busy inference loop cannot admit another capture
        # while the learner is waiting to acquire the serving lock.
        self._runtime.graph_capture_allowed.clear()
        with self._runtime.model_lock:
            parameter = next(self._serving_model.parameters())
            if parameter.device.type == "cuda":
                torch.cuda.synchronize(parameter.device)

    def raise_if_failed(self) -> None:
        """Surface an asynchronous serving failure in the learner control flow."""
        if self._failure is not None:
            raise RuntimeError(
                "learner-owned inference service failed"
            ) from self._failure

    def summary(self) -> dict[str, Any]:
        """Return hot-publication diagnostics for the learner summary."""
        return {
            "mode": "learner_thread",
            "runtime_attempt_id": self._runtime_attempt_id,
            "thread_alive": self._thread.is_alive(),
            "publication_count": self._publication_count,
            "publication_copy_seconds": self._publication_copy_seconds,
            "last_publication_copy_seconds": (self._last_publication_copy_seconds),
            "failed": self._failure is not None,
        }

    def close(self) -> None:
        """Stop serving, join the thread, and surface any background failure."""
        self._runtime.stop_event.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=self._JOIN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError("learner-owned inference thread did not stop")
        self.raise_if_failed()

    def _thread_main(self) -> None:
        try:
            capacity = self._config.inference.request_prefetch_capacity
            if capacity == 0:
                self._serve(self._request_queue)
            else:
                with _PrefetchedInferenceRequestQueue(
                    self._request_queue,
                    capacity=capacity,
                ) as prefetched_request_queue:
                    self._serve(prefetched_request_queue)
        except BaseException as exc:
            self._failure = exc
            summary_path = _inference_summary_path(self._output_path)
            summary = _read_optional_summary(summary_path) or {}
            summary.update(
                {
                    "status": "failed",
                    "service_mode": "learner_thread",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            _write_summary(summary_path, summary)
        else:
            summary_path = _inference_summary_path(self._output_path)
            summary = _read_optional_summary(summary_path) or {}
            summary.update(
                {
                    "status": "stopped",
                    "service_mode": "learner_thread",
                }
            )
            _write_summary(summary_path, summary)

    def _serve(self, request_queue: Any) -> None:
        _run_async_inference_worker(
            config=self._config,
            output_dir=str(self._output_path),
            request_queue=request_queue,
            response_queues=self._response_queues,
            teacher_response_queues=self._teacher_response_queues,
            learner_runtime=self._runtime,
        )


class _AsyncLearnerCheckpointService:
    """Own one bounded exact policy/learner-state persistence lane."""

    def __init__(
        self,
        *,
        config: RLTrainConfig,
        output_path: Path,
        weight_publisher: WeightPublisher,
    ) -> None:
        if not config.learner.async_checkpoint_pairs:
            raise ValueError("async checkpoint service is not enabled")
        self._config = config
        self._output_path = output_path
        self._publisher = AsyncCheckpointPairPublisher(
            weight_publisher,
            output_path / "resume",
            keep_last=config.learner.disk_checkpoint_keep_last,
            retain_every_versions=(
                config.learner.disk_checkpoint_retain_every_versions
            ),
        )
        self._finalized_versions: set[int] = set()
        self._submitted_versions: list[int] = []
        self._last_result: PublishedCheckpointPair | None = None

    def submit(
        self,
        state_dict: PreparedModelState,
        *,
        version: int,
        metadata: Mapping[str, Any],
        checkpoint_fields: Mapping[str, Any],
        completed_iterations: int,
        total_optimizer_updates: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        planned_scheduler_steps: int,
        resume_config_sha256: str,
        optimizer_parameter_names: Sequence[Sequence[str]],
        auxiliary_state: Mapping[str, Any] | None,
    ) -> None:
        """Apply bounded backpressure, then submit one immutable pair."""
        self._finalize(self._publisher.barrier())
        self._publisher.submit(
            state_dict,
            version=version,
            metadata=metadata,
            checkpoint_fields=checkpoint_fields,
            completed_iterations=completed_iterations,
            total_optimizer_updates=total_optimizer_updates,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            planned_scheduler_steps=planned_scheduler_steps,
            resume_config_sha256=resume_config_sha256,
            optimizer_parameter_names=optimizer_parameter_names,
            auxiliary_state=auxiliary_state,
            take_prepared_ownership=True,
            take_auxiliary_ownership=auxiliary_state is not None,
        )
        self._submitted_versions.append(version)

    def poll(self) -> None:
        """Finalize a completed pair without blocking learner progress."""
        if self._publisher.busy:
            return
        self._finalize(self._publisher.barrier())

    def close(self) -> None:
        """Wait for the final pair and propagate background failures."""
        try:
            self._finalize(self._publisher.barrier())
        finally:
            self._publisher.close()

    def summary(self) -> dict[str, Any]:
        """Return bounded persistence progress and the latest timing."""
        result = self._last_result
        return {
            "enabled": True,
            "busy": self._publisher.busy,
            "submitted_versions": list(self._submitted_versions),
            "durable_versions": sorted(self._finalized_versions),
            "latest": (
                None
                if result is None
                else {
                    "version": result.version,
                    "policy_path": deck_records.display_path(result.policy.path),
                    "training_state_path": deck_records.display_path(
                        result.training_state_path
                    ),
                    "pair_manifest_path": deck_records.display_path(
                        result.pair_manifest_path
                    ),
                    "timing": asdict(result.timing),
                    "post_commit_errors": list(result.post_commit_errors),
                }
            ),
        }

    def _finalize(self, result: PublishedCheckpointPair | None) -> None:
        if result is None or result.version in self._finalized_versions:
            return
        queue_frozen_league_candidate(
            self._config.frozen_league,
            policy_version=result.version,
            checkpoint_path=result.policy.path,
            checkpoint_size_bytes=result.policy_size_bytes,
            checkpoint_sha256=result.policy_sha256,
            candidates_dir=_frozen_league_candidate_dir(self._output_path),
            recurrent=self._config.model.recurrent is not None,
        )
        self._finalized_versions.add(result.version)
        self._last_result = result


def _async_learner_worker(
    *,
    config_data: Mapping[str, Any],
    output_dir: str,
    trajectory_queue: Any,
    reanalysis_root_queue: Any | None = None,
    reanalysis_job_queue: Any | None = None,
    reanalysis_result_queue: Any | None = None,
    inference_request_queue: Any | None = None,
    inference_response_queues: Sequence[Any] | None = None,
    teacher_inference_response_queues: Sequence[Any] | None = None,
    collection_gate: Any | None = None,
    learner_ready_event: Any | None = None,
) -> None:
    config = RLTrainConfig.model_validate(config_data)
    _configure_gpu_worker_runtime(config)
    torch.manual_seed(config.seed + 2)
    device = _resolve_device(config.device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tensorboard_writer = _learner_tensorboard_writer(config, output_path)
    model = _load_training_model(config, device=device)
    _maybe_compile_learner_evaluate_actions(
        model,
        enabled=config.learner.compile_evaluate_actions,
    )
    anchor_model = _load_anchor_model(config, device=device)
    _reset_torch_random_stream(config.seed + 2)
    optimizer = _build_ppo_optimizer(config, model=model, device=device)
    _apply_registry_transition_optimizer_state(
        config,
        model=model,
        optimizer=optimizer,
    )
    lr_scheduler_steps = _planned_lr_scheduler_steps(config)
    lr_scheduler = _build_lr_scheduler(
        optimizer,
        config=config,
        planned_steps=lr_scheduler_steps,
    )
    current_policy_version = _initial_policy_version(config)
    training_progress = _restore_training_progress(
        config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        policy_version=current_policy_version,
        planned_scheduler_steps=lr_scheduler_steps,
    )
    if config.learner.frozen_encoder_prefix_layers > 0:
        model.freeze_conditioned_encoder_prefix(
            config.learner.frozen_encoder_prefix_layers
        )
    _validate_remaining_training_iterations(config, training_progress)
    publisher = WeightPublisher(
        output_path / "weights",
        _disk_weight_publisher_config(config),
    )
    checkpoint_service = (
        _AsyncLearnerCheckpointService(
            config=config,
            output_path=output_path,
            weight_publisher=publisher,
        )
        if config.learner.async_checkpoint_pairs
        else None
    )
    shared_publisher = (
        SharedMemoryWeightPublisher(
            output_path / "weights",
            SharedMemoryWeightPublisherConfig(
                keep_last=config.learner.shared_memory_keep_last,
            ),
        )
        if config.learner.shared_memory_publish
        else None
    )
    _ensure_distributed_bootstrap_weights(
        config,
        model=model,
        publisher=publisher,
        current_policy_version=current_policy_version,
        training_progress=training_progress,
    )
    policy_iteration_learner = (
        AmortizedPolicyIterationLearner(
            model,
            config=config.amortized_policy_iteration,
            device=device,
            autocast=config.ppo.autocast,
            grad_clip_norm=config.ppo.grad_clip_norm,
        )
        if config.amortized_policy_iteration.enabled
        else None
    )
    if policy_iteration_learner is not None and any(
        queue_obj is None
        for queue_obj in (
            reanalysis_root_queue,
            reanalysis_job_queue,
            reanalysis_result_queue,
        )
    ):
        raise RuntimeError("policy-iteration learner queues were not initialized")
    if (
        policy_iteration_learner is not None
        and config.resume.mode == "resume"
        and training_progress.state_path is not None
    ):
        resume_payload = torch.load(
            training_progress.state_path,
            map_location="cpu",
            weights_only=False,
        )
        auxiliary = (
            resume_payload.get("auxiliary_state")
            if isinstance(resume_payload, dict)
            else None
        )
        policy_iteration_state = (
            auxiliary.get("amortized_policy_iteration")
            if isinstance(auxiliary, Mapping)
            else None
        )
        if policy_iteration_state is None:
            raise ValueError(
                "exact policy-iteration resume is missing its target-network state"
            )
        policy_iteration_learner.load_training_state_dict(policy_iteration_state)
        del auxiliary, policy_iteration_state, resume_payload
        _trim_worker_memory()
    policy_iteration_replay = (
        _policy_iteration_replay_store(
            config,
            output_path=output_path,
        )
        if policy_iteration_learner is not None
        else None
    )
    if policy_iteration_replay is not None:
        replay_summary = policy_iteration_replay.summary()
        if config.resume.mode != "resume" and any(
            replay_summary.committed_rows.values()
        ):
            raise ValueError(
                "a new policy-iteration branch cannot append existing replay"
            )
        if config.resume.mode == "resume":
            if policy_iteration_learner is None:
                raise RuntimeError("policy-iteration learner was not initialized")
            _validate_policy_iteration_pending_resume(
                restored_schema=(
                    policy_iteration_learner.restored_training_schema_version
                ),
                pending_migration=(
                    config.resume.policy_iteration_pending_targets_migration
                ),
            )

    inference_service: _LearnerOwnedInferenceService | None = None
    if config.execution.gpu_service_mode == "learner_thread":
        if inference_request_queue is None or inference_response_queues is None:
            raise RuntimeError("learner-owned inference queues were not initialized")
        inference_service = _LearnerOwnedInferenceService(
            config=config,
            output_path=output_path,
            learner_model=model,
            current_policy_version=current_policy_version,
            device=device,
            request_queue=inference_request_queue,
            response_queues=inference_response_queues,
            teacher_response_queues=teacher_inference_response_queues,
        )

    start = time.perf_counter()
    learner_timer = StageTimer()
    iterations: list[dict[str, Any]] = []
    total_updates = training_progress.total_optimizer_updates
    ppo_updates = _restored_ppo_optimizer_updates(config, training_progress)
    latest_batch_stats: dict[str, Any] | None = None
    archive_result: TrajectoryWriteResult | None = None
    outstanding_reanalysis_root_ids: set[str] = set()
    _maybe_write_learner_status(
        config,
        output_path,
        {
            "phase": "starting",
            "elapsed_seconds": 0.0,
            "iteration": None,
            "current_policy_version": current_policy_version,
            "resume": _training_progress_summary(config, training_progress),
        },
    )
    learner_window_decisions = _learner_window_decision_budget(config)
    performance_config = (
        _performance_reporter_config(config, output_dir=output_path)
        if config.execution.mode == "async"
        else None
    )
    performance_reporter: TrainingPerformanceReporter | None = None
    if performance_config is not None:
        from ptcg_rl.rl.performance import TrainingPerformanceReporter

        performance_reporter = TrainingPerformanceReporter(performance_config)

    def observe_trajectory(trajectory: GameTrajectory) -> None:
        if performance_reporter is None:
            return
        performance_reporter.observe_decoded(
            worker_id=_local_trajectory_worker_id(trajectory),
            trajectories=(trajectory,),
        )
        performance_reporter.record_delivery(
            stale_excluded_games=0,
            queued_games=1,
        )

    drainer = _AsyncTrajectoryWindowDrainer(
        trajectory_queue,
        get_timeout_seconds=config.execution.learner_queue_get_timeout_seconds,
        max_buffer_decisions=_async_drainer_buffer_limit(config),
        trajectory_observer=(
            observe_trajectory if performance_reporter is not None else None
        ),
        wait_health_check=(
            inference_service.raise_if_failed if inference_service is not None else None
        ),
        collection_gate=collection_gate,
        high_watermark_decisions=(config.execution.collection_high_watermark_decisions),
        resume_watermark_decisions=(
            config.execution.collection_resume_watermark_decisions
        ),
    )
    performance_reporter_started = False
    drainer_started = False
    inference_service_started = False
    inference_service_summary: dict[str, Any] | None = None
    checkpoint_service_summary: dict[str, Any] | None = None
    try:
        if inference_service is not None:
            inference_service.start()
            inference_service_started = True
        if performance_reporter is not None:
            performance_reporter.start()
            performance_reporter_started = True
        drainer.start()
        drainer_started = True
        if learner_ready_event is not None:
            learner_ready_event.set()
        for iteration_index in range(
            training_progress.completed_iterations,
            config.collection.training_iterations,
        ):
            iteration_started = time.perf_counter() - start
            staleness_refill_status: dict[str, int] = {}
            persist_draining_status = partial(
                _write_async_draining_status,
                config,
                output_path,
                started_at=start,
                iteration=iteration_index,
                current_policy_version=current_policy_version,
                publish_version=current_policy_version + 1,
                target_decisions=learner_window_decisions,
                configured_iteration_decisions=(config.collection.iteration_decisions),
                staleness_refill_status=staleness_refill_status,
            )
            if inference_service is not None:
                inference_service.allow_graph_capture()
            try:
                persist_draining_status(drainer.snapshot())
                trajectories = _take_async_learner_window(
                    drainer,
                    target_decisions=learner_window_decisions,
                    current_policy_version=current_policy_version,
                    max_staleness=config.learner.max_staleness,
                    staleness_scope=config.learner.staleness_scope,
                    final_window=(
                        iteration_index + 1 == config.collection.training_iterations
                    ),
                    progress_callback=persist_draining_status,
                    staleness_refill_status=staleness_refill_status,
                )
            finally:
                if inference_service is not None:
                    inference_service.suspend_graph_capture()
            if inference_service is not None:
                inference_service.raise_if_failed()
            _maybe_write_learner_status(
                config,
                output_path,
                {
                    "phase": "building_minibatches",
                    "elapsed_seconds": time.perf_counter() - start,
                    "iteration": iteration_index,
                    "current_policy_version": current_policy_version,
                    "publish_version": current_policy_version + 1,
                    "target_decisions": learner_window_decisions,
                    "configured_iteration_decisions": (
                        config.collection.iteration_decisions
                    ),
                    "trajectories": len(trajectories),
                    "trajectory_decisions": _decision_count(trajectories),
                    "staleness_refill": dict(staleness_refill_status),
                    "drainer": _async_drainer_status(drainer),
                },
            )
            archive_result = (
                _write_async_archive(
                    config,
                    output_path=output_path,
                    trajectories=trajectories,
                )
                if config.execution.archive_trajectories
                else None
            )
            policy_iteration_summary: dict[str, Any] | None = None
            if policy_iteration_learner is not None:
                if policy_iteration_replay is None:
                    raise RuntimeError("policy-iteration replay was not initialized")
                if reanalysis_root_queue is None:
                    raise RuntimeError(
                        "policy-iteration root queue was not initialized"
                    )
                if reanalysis_job_queue is None:
                    raise RuntimeError("policy-iteration job queue was not initialized")
                if reanalysis_result_queue is None:
                    raise RuntimeError("policy-iteration queues were not initialized")
                policy_iteration_learner_config = (
                    config.amortized_policy_iteration.learner
                )
                queue_drain_idle_timeout_ms = (
                    policy_iteration_learner_config.queue_drain_idle_timeout_ms
                )
                roots_raw = _drain_queue_batch(
                    reanalysis_root_queue,
                    limit=(
                        config.amortized_policy_iteration.learner.roots_per_iteration
                    ),
                    idle_timeout_ms=queue_drain_idle_timeout_ms,
                )
                roots = tuple(
                    root for root in roots_raw if isinstance(root, ReanalysisRoot)
                )
                policy_iteration_replay.record_roots(roots)
                _maybe_write_learner_status(
                    config,
                    output_path,
                    {
                        "phase": "policy_iteration_proposal",
                        "elapsed_seconds": time.perf_counter() - start,
                        "iteration": iteration_index,
                        "current_policy_version": current_policy_version,
                        "publish_version": current_policy_version + 1,
                        "roots": len(roots),
                    },
                )
                with time_stage(learner_timer, "learner_api_native_proposal"):
                    proposal_batch = build_native_reanalysis_jobs(
                        model,
                        roots,
                        current_policy_version=current_policy_version,
                        config=config.amortized_policy_iteration,
                        device=device,
                        seed=config.seed + 1_000_003 * (iteration_index + 1),
                        autocast=config.ppo.autocast,
                    )
                queue_counters: Counter[str] = Counter(proposal_batch.counters)
                queue_counters["invalid_root_type"] += len(roots_raw) - len(roots)
                submitted_jobs: list[NativeReanalysisJob] = []
                for job_batch in proposal_batch.wire_batches:
                    try:
                        reanalysis_job_queue.put_nowait(job_batch)
                    except queue.Full:
                        queue_counters["job_batch_queue_full"] += 1
                        queue_counters["jobs_queue_full"] += len(job_batch.jobs)
                    else:
                        queue_counters["job_batches_submitted"] += 1
                        queue_counters["jobs_submitted"] += len(job_batch.jobs)
                        submitted_jobs.extend(job_batch.jobs)
                        outstanding_reanalysis_root_ids.update(
                            job.root.student.root_id for job in job_batch.jobs
                        )
                policy_iteration_replay.record_jobs(submitted_jobs)
                requires_native_quiescence = _requires_native_reanalysis_quiescence(
                    config,
                    iteration_index=iteration_index,
                    current_policy_version=current_policy_version,
                )
                _maybe_write_learner_status(
                    config,
                    output_path,
                    {
                        "phase": "policy_iteration_updates",
                        "elapsed_seconds": time.perf_counter() - start,
                        "iteration": iteration_index,
                        "current_policy_version": current_policy_version,
                        "publish_version": current_policy_version + 1,
                        "submitted_jobs": len(submitted_jobs),
                    },
                )
                completion = _complete_policy_iteration_updates(
                    policy_iteration_learner=policy_iteration_learner,
                    policy_iteration_replay=policy_iteration_replay,
                    trajectories=trajectories,
                    reanalysis_result_queue=reanalysis_result_queue,
                    result_drain_limit=(
                        policy_iteration_learner_config.result_drain_limit
                    ),
                    queue_drain_idle_timeout_ms=queue_drain_idle_timeout_ms,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    timer=learner_timer,
                    wait_for_result_root_ids=(
                        frozenset(outstanding_reanalysis_root_ids)
                        if requires_native_quiescence
                        else frozenset()
                    ),
                    result_barrier_timeout_seconds=(
                        config.amortized_policy_iteration.native.result_barrier_timeout_seconds
                    ),
                    real_retrace_enabled=(
                        policy_iteration_learner_config.real_retrace_enabled
                    ),
                )
                outstanding_reanalysis_root_ids.difference_update(
                    completion.native_result_root_ids
                )
                if requires_native_quiescence and outstanding_reanalysis_root_ids:
                    raise RuntimeError(
                        "native checkpoint barrier left submitted roots pending"
                    )
                real_update = completion.real_update
                counterfactual_updates = completion.counterfactual_updates
                ingest_summary = completion.ingest_summary
                policy_updates = (
                    () if real_update is None else (real_update,)
                ) + counterfactual_updates
                total_updates += len(policy_updates)
                policy_iteration_replay.flush_async()
                replay_summary = policy_iteration_replay.summary()
                queue_counters["root_queue_size"] = (
                    _queue_size(reanalysis_root_queue) or 0
                )
                queue_counters["job_queue_size"] = (
                    _queue_size(reanalysis_job_queue) or 0
                )
                queue_counters["result_queue_size"] = (
                    _queue_size(reanalysis_result_queue) or 0
                )
                queue_counters["outstanding_jobs"] = len(
                    outstanding_reanalysis_root_ids
                )
                policy_iteration_summary = {
                    "queue": dict(queue_counters),
                    "ingest": ingest_summary,
                    "updates": [asdict(update) for update in policy_updates],
                    "update_count": len(policy_updates),
                    "pending_counterfactual_roots": (
                        policy_iteration_learner.pending_counterfactual_roots
                    ),
                    "replay": asdict(replay_summary),
                }
                del (
                    completion,
                    counterfactual_updates,
                    ingest_summary,
                    policy_updates,
                    proposal_batch,
                    real_update,
                    roots,
                    roots_raw,
                    submitted_jobs,
                )
            batch_config = LearnerBatchConfig(
                microbatch_size=config.collection.microbatch_size,
                gradient_accumulation_steps=(
                    config.collection.gradient_accumulation_steps
                ),
                max_decisions=config.learner.max_update_decisions,
                max_staleness=config.learner.max_staleness,
                staleness_scope=config.learner.staleness_scope,
                shuffle=True,
                shuffle_each_epoch=config.learner.shuffle_each_epoch,
                drop_last=False,
                pin_memory=config.learner.pin_memory,
                non_blocking_transfer=config.learner.non_blocking_transfer,
                copy_stream=config.learner.copy_stream,
                shape_bucket_accumulation=(config.learner.shape_bucket_accumulation),
                route_bucket_minibatches=config.learner.route_bucket_minibatches,
                seed=config.seed + iteration_index * config.collection.ppo_epochs,
                legacy_sampling_temperature=(
                    config.learner.legacy_sampling_temperature
                ),
                gae=config.gae,
                require_deck_context=_deck_conditioning_enabled(config.model),
                schema9=config.learner.schema9,
                macro_credit=(
                    config.macro_credit if config.macro_credit.enabled else None
                ),
            )
            batch_result = build_ppo_minibatches(
                trajectories,
                current_policy_version=current_policy_version,
                config=batch_config,
                device=None,
                timer=learner_timer,
            )
            update_results = _run_ppo_epochs_and_publish(
                config=config,
                model=model,
                optimizer=optimizer,
                trajectories=trajectories,
                current_policy_version=current_policy_version,
                batch_config=batch_config,
                batch_result=batch_result,
                anchor_model=anchor_model,
                publisher=publisher,
                shared_publisher=shared_publisher,
                publish_version=current_policy_version + 1,
                first_update_index=total_updates,
                first_ppo_update_index=ppo_updates,
                device=device,
                lr_scheduler=lr_scheduler,
                timer=learner_timer,
                output_path=output_path,
                started_at=start,
                iteration_index=iteration_index,
                policy_iteration_learner=policy_iteration_learner,
                policy_iteration_replay=policy_iteration_replay,
                hot_policy_publisher=(
                    None if inference_service is None else inference_service.publish
                ),
                runtime_policy_attempt_id=(
                    None
                    if inference_service is None
                    else inference_service.runtime_attempt_id
                ),
                checkpoint_service=checkpoint_service,
            )
            published_version = _effective_published_version(
                config,
                update_results,
                logical_version=current_policy_version + 1,
            )
            iteration_finished = time.perf_counter() - start
            iteration_summary = _async_iteration_summary(
                iteration_index=iteration_index,
                policy_version=current_policy_version,
                trajectories=trajectories,
                update_results=update_results,
                published_version=published_version,
                archive_result=archive_result,
                started_elapsed_seconds=iteration_started,
                finished_elapsed_seconds=iteration_finished,
                policy_iteration_summary=policy_iteration_summary,
            )
            iterations.append(iteration_summary)
            write_learner_tensorboard(
                tensorboard_writer,
                step=(
                    published_version
                    if published_version is not None
                    else iteration_index
                ),
                iteration=iteration_summary,
                update_results=update_results,
                optimizer=optimizer,
                timer=learner_timer,
            )
            if published_version is not None:
                current_policy_version = published_version
            completed_ppo_updates = sum(
                len(result.updates) for result in update_results
            )
            total_updates += completed_ppo_updates
            ppo_updates += completed_ppo_updates
            if update_results:
                latest_batch_stats = dict(
                    update_results[-1].batch_result.stats.__dict__
                )
            del batch_result, trajectories, update_results
            if checkpoint_service is not None:
                checkpoint_service.poll()
            _maybe_trim_learner_memory(config, published_version=published_version)
    finally:
        if learner_ready_event is not None:
            learner_ready_event.clear()
        try:
            if drainer_started:
                drainer.close()
        finally:
            try:
                if performance_reporter_started and performance_reporter is not None:
                    performance_reporter.close()
            finally:
                try:
                    if inference_service_started and inference_service is not None:
                        inference_service.close()
                finally:
                    if inference_service is not None:
                        inference_service_summary = inference_service.summary()
                    try:
                        if checkpoint_service is not None:
                            checkpoint_service.close()
                    finally:
                        if checkpoint_service is not None:
                            checkpoint_service_summary = checkpoint_service.summary()
                        final_drainer_status = _async_drainer_status(drainer)
                        try:
                            if policy_iteration_replay is not None:
                                policy_iteration_replay.close()
                        finally:
                            try:
                                if shared_publisher is not None:
                                    shared_publisher.close()
                            finally:
                                tensorboard_writer.close()

    elapsed_seconds = time.perf_counter() - start
    summary = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": config.execution.mode,
        "device": str(device),
        "output_dir": deck_records.display_path(output_path),
        "elapsed_seconds": elapsed_seconds,
        "iterations": iterations,
        "trajectories": sum(int(iteration["trajectories"]) for iteration in iterations),
        "trajectory_decisions": sum(
            int(iteration["trajectory_decisions"]) for iteration in iterations
        ),
        "learner": {
            "updates": total_updates,
            "effective_updates": total_updates,
            "ppo_updates": ppo_updates,
            "published_version": current_policy_version,
            "batching": {
                "microbatch_size": config.collection.microbatch_size,
                "gradient_accumulation_steps": (
                    config.collection.gradient_accumulation_steps
                ),
                "effective_batch_size": config.collection.effective_batch_size,
            },
            "batch_stats": latest_batch_stats,
            "timings": _learner_timing_summary(
                learner_timer,
                elapsed_seconds=elapsed_seconds,
            ),
            "last_iteration_ppo": iterations[-1]["learner"]["ppo"],
            "last_iteration_early_stop": iterations[-1]["learner"]["early_stop"],
        },
        "optimizer": _optimizer_summary(
            config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            planned_steps=lr_scheduler_steps,
        ),
        "resume": _training_progress_summary(config, training_progress),
        "throughput": _training_throughput_summary(
            config,
            iterations=iterations,
            elapsed_seconds=elapsed_seconds,
        ),
        "archive": _archive_summary(archive_result),
        "health": _training_health_summary(iterations),
        "trajectory_drainer": final_drainer_status,
        "inference_service": inference_service_summary,
        "checkpoint_service": checkpoint_service_summary,
    }
    _add_tensorboard_log_dir(summary, tensorboard_writer, output_dir=output_path)
    _write_summary(output_path / "summary.json", summary)


def _async_actor_policy(
    config: RLTrainConfig,
    *,
    device: torch.device,
    actor_index: int = 0,
    actor_incarnation: int = 0,
    inference_request_queue: Any | None = None,
    inference_response_queue: Any | None = None,
    remote_client_state: RemoteInferenceClientState | None = None,
) -> tuple[RolloutPolicy, torch.nn.Module | None]:
    if config.execution.inference_server:
        if inference_request_queue is None or inference_response_queue is None:
            raise ValueError("inference queues are required for remote actor policy")
        if remote_client_state is None:
            raise ValueError("remote client state is required for remote actor policy")
        planner = config.planner
        improvement_actor = bool(
            config.amortized_policy_iteration.enabled
            and actor_index < config.amortized_policy_iteration.improvement.actor_count
        )
        return (
            RemoteInferencePolicy(
                actor_id=f"actor-{actor_index}",
                actor_incarnation=actor_incarnation,
                policy_id=(
                    "candidate_improvement" if improvement_actor else "candidate"
                ),
                request_queue=inference_request_queue,
                response_queue=inference_response_queue,
                config=(
                    _inference_client_config(config)
                    if planner is None
                    else InferenceClientConfig(
                        response_timeout_seconds=(
                            _inference_client_config(config).response_timeout_seconds
                            if config.macro_credit.native_teacher_enabled
                            else planner.deadlines.inference_timeout_seconds
                        ),
                        response_retries=0,
                    )
                ),
                client_state=remote_client_state,
                initial_request_id=_remote_request_id_base(actor_index),
                request_purpose=("behavior" if planner is None else "planner_behavior"),
                planner_tensor_schema_fingerprint=(
                    ""
                    if planner is None
                    else planner.resolve_static().tensor_schema_fingerprint
                ),
                planner_inference_device_type=(None if planner is None else "cuda"),
                behavior_kind=("improvement" if improvement_actor else "policy_sample"),
                recurrent_model_config=(
                    config.model if config.model.recurrent is not None else None
                ),
            ),
            None,
        )
    if config.collection.policy_kind == "min_count":
        return (MinCountRolloutPolicy(), None)
    model = _load_training_model(config, device=device)
    model.eval()
    return (
        ModelRolloutPolicy(
            model,
            policy_version=_initial_policy_version(config),
            autocast=config.ppo.autocast,
            planner_context_capacity=(
                0
                if config.planner is None
                else config.planner.contexts.retained_root_rows
            ),
        ),
        model,
    )


def _training_engine_teacher_producer(
    config: RLTrainConfig,
    *,
    policy: RolloutPolicy,
    device: torch.device | str | None,
    actor_index: int,
    actor_incarnation: int = 0,
    inference_request_queue: Any | None = None,
    inference_response_queue: Any | None = None,
    teacher_inference_response_queue: Any | None = None,
    remote_client_state: RemoteInferenceClientState | None = None,
) -> OnlineEngineTeacherProducer | BackgroundEngineTeacherProducer | None:
    """Build one actor-local online teacher with immutable run semantics."""
    if not config.engine_teacher.enabled:
        return None
    teacher_policy: DecodePolicy
    if config.execution.inference_server:
        if inference_request_queue is None or inference_response_queue is None:
            raise ValueError("inference queues are required for remote teaching")
        if remote_client_state is None:
            raise ValueError("remote client state is required for remote teaching")
        async_teacher = config.engine_teacher.async_actor
        active_response_queue = (
            teacher_inference_response_queue
            if async_teacher
            else inference_response_queue
        )
        if active_response_queue is None:
            raise ValueError("teacher inference response queue is required")
        teacher_client_state = (
            RemoteInferenceClientState() if async_teacher else remote_client_state
        )
        teacher_policy = RemoteInferencePolicy(
            actor_id=(
                f"teacher-{actor_index}" if async_teacher else f"actor-{actor_index}"
            ),
            actor_incarnation=actor_incarnation,
            policy_id="candidate",
            request_queue=inference_request_queue,
            response_queue=active_response_queue,
            config=InferenceClientConfig(
                response_timeout_seconds=(
                    config.engine_teacher.inference_timeout_seconds
                ),
                response_retries=0,
            ),
            client_state=teacher_client_state,
            initial_request_id=_remote_request_id_base(actor_index),
            request_purpose="teacher",
        )
    else:
        if not callable(getattr(policy, "predict_values", None)):
            raise TypeError("engine teacher policy requires value-only inference")
        teacher_policy = cast(DecodePolicy, policy)
    belief_producer = (
        OpponentBeliefFeatureProducer.from_config(config.rollout_belief)
        if config.rollout_belief.enabled
        else None
    )
    producer = OnlineEngineTeacherProducer(
        policy=teacher_policy,
        device=device,
        config=config.engine_teacher,
        seed=config.seed + 1_000_003 * (actor_index + 1),
        belief_producer=belief_producer,
    )
    if not config.engine_teacher.async_actor:
        return producer
    return BackgroundEngineTeacherProducer(
        producer,
        queue_batches=config.engine_teacher.async_queue_batches,
        batch_deadline_seconds=(
            config.engine_teacher.max_teacher_seconds_per_actor_step
        ),
    )


def _async_archive_recorder() -> TrajectoryRecorder:
    return TrajectoryRecorder(
        policy_name="policy",
        policy_version="0",
        opponent_name="self_play",
        opponent_tier=-1,
    )


def _write_async_archive(
    config: RLTrainConfig,
    *,
    output_path: Path,
    trajectories: tuple[GameTrajectory, ...],
) -> TrajectoryWriteResult:
    completed = tuple(_require_archive(trajectory) for trajectory in trajectories)
    writer = TrajectoryShardWriter(
        output_dir=output_path / "archive",
        rows_per_shard=config.execution.archive_rows_per_shard,
        compression=config.execution.archive_compression,
        config=_rl_train_config_dump(config, output_dir=output_path),
        metadata={
            "mode": "async",
            "source": "actor_queue",
        },
    )
    writer.add_completed(completed)
    return writer.close()


def _require_archive(trajectory: GameTrajectory) -> CompletedTrajectory:
    if trajectory.archive is None:
        raise ValueError(f"trajectory is missing archive rows: {trajectory.game_id}")
    return trajectory.archive


def _archive_summary(result: TrajectoryWriteResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "manifest_path": deck_records.display_path(result.manifest_path),
        "games_path": deck_records.display_path(result.games_path),
        "summary": result.manifest.get("summary", {}),
    }


def _published_version_from_results(
    results: Sequence[LearnerIterationResult],
) -> int | None:
    for result in reversed(results):
        if result.published_weights is not None:
            return result.published_weights.version
    return None


def _effective_published_version(
    config: RLTrainConfig,
    results: Sequence[LearnerIterationResult],
    *,
    logical_version: int,
) -> int | None:
    """Return the new policy version, including metadata-only publications."""
    artifact_version = _published_version_from_results(results)
    if artifact_version is not None:
        return artifact_version
    if _allows_logical_only_publication(config) and _learner_update_count(results) > 0:
        return logical_version
    return None


def _run_ppo_epochs_and_publish(
    *,
    config: RLTrainConfig,
    model: AgentPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[GameTrajectory],
    current_policy_version: int,
    batch_config: LearnerBatchConfig,
    batch_result: LearnerBatchResult,
    anchor_model: AgentPolicyValueNet | None,
    publisher: WeightPublisher,
    shared_publisher: SharedMemoryWeightPublisher | None,
    publish_version: int,
    first_update_index: int,
    first_ppo_update_index: int,
    device: torch.device,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    timer: StageTimer,
    output_path: Path,
    started_at: float,
    iteration_index: int,
    policy_iteration_learner: AmortizedPolicyIterationLearner | None = None,
    policy_iteration_replay: PolicyIterationReplayStore | None = None,
    hot_policy_publisher: Callable[[AgentPolicyValueNet, int, str], None] | None = None,
    runtime_policy_attempt_id: str | None = None,
    checkpoint_service: _AsyncLearnerCheckpointService | None = None,
) -> list[LearnerIterationResult]:
    """Run PPO epochs and publish once if any optimizer update ran."""
    update_results: list[LearnerIterationResult] = []
    anchor_cache = AnchorStepLogitsCache() if anchor_model is not None else None
    kl_reference: FixedKlReference | None = None
    kl_stop_baseline_k3: float | None = None
    if (
        config.ppo.target_kl_early_stop is not None
        and config.ppo.kl_stop_mode == "fixed_reference"
    ):
        kl_reference = capture_fixed_kl_reference(
            model,
            batch_result,
            batch_config=batch_config,
            ppo_config=config.ppo,
            device=device,
            timer=timer,
        )
    collect_ppo_diagnostics = _should_collect_ppo_diagnostics(
        config,
        publish_version=publish_version,
    )
    _maybe_write_learner_status(
        config,
        output_path,
        {
            "phase": "updating",
            "elapsed_seconds": time.perf_counter() - started_at,
            "iteration": iteration_index,
            "current_policy_version": current_policy_version,
            "publish_version": publish_version,
            "ppo_epochs": config.collection.ppo_epochs,
            "updates": 0,
            "trajectory_decisions": sum(
                trajectory.decision_count for trajectory in trajectories
            ),
            "batch_stats": _learner_batch_stats_dict(batch_result),
        },
    )
    for epoch in range(config.collection.ppo_epochs):
        epoch_batch_result = batch_result
        if epoch > 0 and batch_config.shuffle_each_epoch:
            epoch_batch_result = reshuffle_ppo_minibatches(
                batch_result,
                config=batch_config,
                seed=batch_config.seed + epoch,
                device=None,
            )
        epoch_first_update_index = first_update_index + sum(
            len(result.updates) for result in update_results
        )
        epoch_first_ppo_update_index = first_ppo_update_index + sum(
            len(result.updates) for result in update_results
        )

        def progress_callback(
            update_index: int,
            update: PpoUpdateResult,
            *,
            epoch_index: int = epoch,
            callback_batch_result: LearnerBatchResult = epoch_batch_result,
        ) -> None:
            iteration_update = update_index - first_update_index + 1
            if (
                not update.should_stop
                and iteration_update % config.diagnostics.log_every_updates != 0
            ):
                return
            _maybe_write_learner_status(
                config,
                output_path,
                {
                    "phase": "updating",
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "iteration": iteration_index,
                    "current_policy_version": current_policy_version,
                    "publish_version": publish_version,
                    "ppo_epochs": config.collection.ppo_epochs,
                    "epoch": epoch_index,
                    "global_update_index": update_index,
                    "ppo_update_index": (
                        first_ppo_update_index + update_index - first_update_index
                    ),
                    "iteration_updates": iteration_update,
                    "batch_stats": _learner_batch_stats_dict(callback_batch_result),
                    "last_update": _ppo_update_summary(update),
                },
            )

        update_results.append(
            run_ppo_iteration(
                model=model,
                optimizer=optimizer,
                trajectories=trajectories,
                current_policy_version=current_policy_version,
                batch_config=batch_config,
                ppo_config=config.ppo,
                anchor_model=anchor_model,
                anchor_cache=anchor_cache,
                publisher=None,
                publish_version=publish_version,
                first_update_index=epoch_first_update_index,
                first_ppo_update_index=epoch_first_ppo_update_index,
                device=device,
                lr_scheduler=lr_scheduler,
                timer=timer,
                batch_result=epoch_batch_result,
                progress_callback=progress_callback,
                anchor_cache_key_offset=epoch * max(1, batch_result.stats.minibatches),
                kl_stop_baseline_k3=kl_stop_baseline_k3,
                kl_reference=kl_reference,
                collect_diagnostics=collect_ppo_diagnostics,
            )
        )
        if (
            config.ppo.kl_stop_mode == "training_batch_delta"
            and kl_stop_baseline_k3 is None
            and update_results[-1].updates
        ):
            kl_stop_baseline_k3 = update_results[-1].updates[0].kl_stop_baseline_k3
        if update_results[-1].updates and update_results[-1].updates[-1].should_stop:
            break
    if _learner_update_count(update_results) <= 0:
        _maybe_write_learner_status(
            config,
            output_path,
            {
                "phase": "publish_skipped",
                "skip_reason": "no_updates",
                "elapsed_seconds": time.perf_counter() - started_at,
                "iteration": iteration_index,
                "current_policy_version": current_policy_version,
                "publish_version": None,
                "ppo_epochs": config.collection.ppo_epochs,
                "updates": 0,
                "batch_stats": _learner_batch_stats_dict(batch_result),
            },
        )
        return update_results
    teacher_metrics = aggregate_engine_teacher_updates(_ppo_updates(update_results))
    factual_metrics = aggregate_factual_updates(_ppo_updates(update_results))
    planner_metrics = aggregate_planner_updates(_ppo_updates(update_results))
    planner_summary = planner_update_metrics_summary(planner_metrics)
    _maybe_write_learner_status(
        config,
        output_path,
        {
            "phase": "publishing",
            "elapsed_seconds": time.perf_counter() - started_at,
            "iteration": iteration_index,
            "current_policy_version": current_policy_version,
            "publish_version": publish_version,
            "ppo_epochs": config.collection.ppo_epochs,
            "updates": _learner_update_count(update_results),
            "batch_stats": _learner_batch_stats_dict(batch_result),
            "early_stop": _learner_early_stop_summary(update_results),
            "engine_teacher_decisions": teacher_metrics.decisions,
            "engine_teacher_loss": teacher_metrics.loss,
            "factual_decisions": factual_metrics.decisions,
            "factual_effect_loss": factual_metrics.effect_loss,
            "factual_successor_loss": factual_metrics.successor_loss,
            **planner_summary,
            "last_update": _last_ppo_update_summary(update_results),
        },
    )
    published = _publish_learner_iteration(
        config=config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        publisher=publisher,
        shared_publisher=shared_publisher,
        output_path=output_path,
        update_results=update_results,
        publish_version=publish_version,
        current_policy_version=current_policy_version,
        completed_iterations=iteration_index + 1,
        total_optimizer_updates=(
            first_update_index + _learner_update_count(update_results)
        ),
        ppo_optimizer_updates=(
            first_ppo_update_index + _learner_update_count(update_results)
        ),
        configured_ppo_epochs=config.collection.ppo_epochs,
        target_kl_early_stop=config.ppo.target_kl_early_stop,
        target_kl_early_stop_multiplier=config.ppo.target_kl_early_stop_multiplier,
        target_kl_stop_mode=config.ppo.kl_stop_mode,
        policy_iteration_learner=policy_iteration_learner,
        policy_iteration_replay=policy_iteration_replay,
        hot_policy_publisher=hot_policy_publisher,
        runtime_policy_attempt_id=runtime_policy_attempt_id,
        checkpoint_service=checkpoint_service,
    )
    _maybe_write_learner_status(
        config,
        output_path,
        {
            "phase": "published",
            "elapsed_seconds": time.perf_counter() - started_at,
            "iteration": iteration_index,
            "current_policy_version": current_policy_version,
            "published_version": publish_version,
            "published_path": (
                None if published is None else deck_records.display_path(published.path)
            ),
            "publication_kind": (
                "async_checkpoint_pair"
                if published is None
                and checkpoint_service is not None
                and _should_write_disk_checkpoint(config, publish_version)
                else "logical"
                if published is None
                else "artifact"
            ),
            "ppo_epochs": config.collection.ppo_epochs,
            "updates": _learner_update_count(update_results),
            "batch_stats": _learner_batch_stats_dict(batch_result),
            "early_stop": _learner_early_stop_summary(update_results),
            "engine_teacher_decisions": teacher_metrics.decisions,
            "engine_teacher_loss": teacher_metrics.loss,
            "factual_decisions": factual_metrics.decisions,
            "factual_effect_loss": factual_metrics.effect_loss,
            "factual_successor_loss": factual_metrics.successor_loss,
            **planner_summary,
            "last_update": _last_ppo_update_summary(update_results),
        },
    )
    return update_results


def _ensure_distributed_bootstrap_weights(
    config: RLTrainConfig,
    *,
    model: torch.nn.Module,
    publisher: WeightPublisher,
    current_policy_version: int,
    training_progress: TrainingProgress,
) -> PublishedWeights | None:
    """Publish the learner's initial policy before remote actors request it."""
    if (
        config.execution.mode != "distributed_async"
        or not config.distributed.coordinator_enabled
    ):
        return None
    latest = publisher.read_latest()
    _validate_distributed_warm_start_output(
        config,
        output_dir=publisher.directory.parent,
        include_mutable_artifacts=False,
    )
    if latest is not None and latest.version > current_policy_version:
        raise RuntimeError(
            "distributed output contains weights newer than the learner's "
            f"initial policy: {latest.version} > {current_policy_version}"
        )
    metadata = {
        "initial_distributed_bootstrap": True,
        "source_policy_version": current_policy_version,
        "completed_training_iterations": training_progress.completed_iterations,
        "total_optimizer_updates": training_progress.total_optimizer_updates,
        "resume": _training_progress_summary(config, training_progress),
    }
    registry_transition = _registry_transition_summary(model)
    if registry_transition is not None:
        metadata["registry_transition"] = dict(registry_transition)
    if config.resume.mode == "resume":
        if config.checkpoint_path is None or training_progress.state_path is None:
            raise RuntimeError(
                "exact distributed resume requires its validated policy/state pair"
            )
        checkpoint_path = deck_records.repo_path(config.checkpoint_path)
        if (
            latest is not None
            and latest.version == current_policy_version
            and latest.path.exists()
            and latest.path.resolve() != checkpoint_path.resolve()
        ):
            raise RuntimeError(
                "distributed latest pointer conflicts with the selected exact-"
                f"resume checkpoint: {latest.path} != {checkpoint_path}"
            )
        adopt_training_state_pointer(
            publisher.directory.parent / "resume",
            state_path=training_progress.state_path,
            policy_version=current_policy_version,
        )
        adopted = publisher.adopt_existing(
            checkpoint_path,
            version=current_policy_version,
            metadata={**metadata, "adopted_exact_resume_checkpoint": True},
        )
        return adopted
    if (
        latest is not None
        and latest.version == current_policy_version
        and latest.path.exists()
    ):
        return latest
    return publisher.publish(
        model.state_dict(),
        version=current_policy_version,
        metadata=metadata,
        checkpoint_fields=_learner_checkpoint_fields(config, metadata=metadata),
    )


def _publish_learner_iteration(
    *,
    config: RLTrainConfig,
    model: AgentPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    publisher: WeightPublisher,
    shared_publisher: SharedMemoryWeightPublisher | None,
    output_path: Path,
    update_results: list[LearnerIterationResult],
    publish_version: int,
    current_policy_version: int,
    completed_iterations: int,
    total_optimizer_updates: int,
    ppo_optimizer_updates: int,
    configured_ppo_epochs: int,
    target_kl_early_stop: float | None,
    target_kl_early_stop_multiplier: float,
    target_kl_stop_mode: str,
    policy_iteration_learner: AmortizedPolicyIterationLearner | None = None,
    policy_iteration_replay: PolicyIterationReplayStore | None = None,
    hot_policy_publisher: Callable[[AgentPolicyValueNet, int, str], None] | None = None,
    runtime_policy_attempt_id: str | None = None,
    checkpoint_service: _AsyncLearnerCheckpointService | None = None,
) -> PublishedWeights | None:
    if not update_results or _learner_update_count(update_results) <= 0:
        raise ValueError("cannot publish learner iteration without PPO updates")
    ensure_finite_model(model)
    metadata = _learner_publish_metadata(
        update_results=update_results,
        publish_version=publish_version,
        current_policy_version=current_policy_version,
        completed_iterations=completed_iterations,
        total_optimizer_updates=total_optimizer_updates,
        ppo_optimizer_updates=ppo_optimizer_updates,
        transition_distillation_optimizer_updates=(
            config.ppo.transition_distillation.optimizer_updates
        ),
        configured_ppo_epochs=configured_ppo_epochs,
        target_kl_early_stop=target_kl_early_stop,
        target_kl_early_stop_multiplier=target_kl_early_stop_multiplier,
        target_kl_stop_mode=target_kl_stop_mode,
    )
    disk_checkpoint = _should_write_disk_checkpoint(config, publish_version)
    if disk_checkpoint and policy_iteration_replay is not None:
        # A durable learner pair must never point beyond replay evidence that
        # was still queued in the asynchronous shard writer.
        policy_iteration_replay.barrier()
    shared_weights: SharedMemoryWeights | None = None
    # Only artifact publications need a full CPU snapshot and canonical content
    # hash. Learner-owned hot generations otherwise stay device-to-device; a
    # model-sized D2H copy on every logical version would erase that benefit.
    state_dict: PreparedModelState | None = None
    if disk_checkpoint or shared_publisher is not None:
        state_dict = (
            prepare_model_state(model.state_dict())
            if shared_publisher is None
            else shared_publisher.prepare_state(model.state_dict())
        )
    hot_policy_identity_kind: str | None = None
    if hot_policy_publisher is not None:
        if state_dict is None and not runtime_policy_attempt_id:
            raise ValueError(
                "logical hot publication requires a runtime attempt identity"
            )
        hot_policy_identity = (
            state_dict.model_fingerprint
            if state_dict is not None
            else _runtime_policy_generation_fingerprint(
                config,
                publish_version=publish_version,
                runtime_attempt_id=cast(str, runtime_policy_attempt_id),
            )
        )
        hot_policy_publisher(
            model,
            publish_version,
            hot_policy_identity,
        )
        hot_policy_identity_kind = (
            "content_sha256" if state_dict is not None else "runtime_generation_sha256"
        )
    metadata = {
        **metadata,
        "disk_checkpoint": disk_checkpoint,
        "checkpoint_pair": disk_checkpoint,
        "async_checkpoint_pair": (disk_checkpoint and checkpoint_service is not None),
        "shared_memory": shared_publisher is not None,
        "logical_only": not disk_checkpoint and shared_publisher is None,
        "hot_policy_identity_kind": hot_policy_identity_kind,
        "hot_policy_runtime_attempt_id": runtime_policy_attempt_id,
        "resume_state_path": (
            deck_records.display_path(
                output_path / "resume" / f"training_state_v{publish_version}.pt"
            )
            if disk_checkpoint
            else None
        ),
    }
    registry_transition = _registry_transition_summary(model)
    if registry_transition is not None:
        metadata["registry_transition"] = dict(registry_transition)
    if shared_publisher is not None:
        if state_dict is None:
            raise AssertionError("shared publication requires a prepared model state")
        shared_weights = shared_publisher.publish(
            state_dict,
            version=publish_version,
            metadata=metadata,
        )
    committed_pair: PublishedCheckpointPair | None = None
    if disk_checkpoint and checkpoint_service is not None:
        if state_dict is None:
            raise AssertionError("checkpoint publication requires a model snapshot")
        checkpoint_service.submit(
            state_dict,
            version=publish_version,
            metadata=metadata,
            checkpoint_fields=_learner_checkpoint_fields(
                config,
                metadata=metadata,
            ),
            completed_iterations=completed_iterations,
            total_optimizer_updates=total_optimizer_updates,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            planned_scheduler_steps=_planned_lr_scheduler_steps(config),
            resume_config_sha256=_resume_relevant_config_sha256(config),
            optimizer_parameter_names=_optimizer_parameter_names(
                model,
                optimizer,
            ),
            auxiliary_state=_learner_auxiliary_state(
                ppo_optimizer_updates=ppo_optimizer_updates,
                policy_iteration_learner=policy_iteration_learner,
            ),
        )
        published = None
    elif disk_checkpoint:
        if state_dict is None:
            raise AssertionError("checkpoint publication requires a model snapshot")
        auxiliary_state = _learner_auxiliary_state(
            ppo_optimizer_updates=ppo_optimizer_updates,
            policy_iteration_learner=policy_iteration_learner,
        )
        with AsyncCheckpointPairPublisher(
            publisher,
            output_path / "resume",
            keep_last=config.learner.disk_checkpoint_keep_last,
            retain_every_versions=(
                config.learner.disk_checkpoint_retain_every_versions
            ),
        ) as pair_publisher:
            committed_pair = pair_publisher.publish(
                state_dict,
                version=publish_version,
                metadata=metadata,
                checkpoint_fields=_learner_checkpoint_fields(
                    config,
                    metadata=metadata,
                ),
                completed_iterations=completed_iterations,
                total_optimizer_updates=total_optimizer_updates,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                planned_scheduler_steps=_planned_lr_scheduler_steps(config),
                resume_config_sha256=_resume_relevant_config_sha256(config),
                optimizer_parameter_names=_optimizer_parameter_names(
                    model,
                    optimizer,
                ),
                auxiliary_state=auxiliary_state,
                take_prepared_ownership=True,
                take_auxiliary_ownership=True,
            )
        published = committed_pair.policy
    elif shared_weights is not None:
        published = PublishedWeights(
            version=publish_version,
            path=shared_weights.latest_path,
            latest_path=shared_weights.latest_path,
            published_at=shared_weights.published_at,
            metadata=metadata,
            model_fingerprint=shared_weights.model_fingerprint,
        )
    elif _allows_logical_only_publication(config):
        published = None
    else:
        if state_dict is None:
            raise AssertionError("artifact publication requires a model snapshot")
        published = publisher.publish(
            state_dict,
            version=publish_version,
            metadata=metadata,
        )
    if committed_pair is not None:
        queue_frozen_league_candidate(
            config.frozen_league,
            policy_version=publish_version,
            checkpoint_path=committed_pair.policy.path,
            checkpoint_size_bytes=committed_pair.policy_size_bytes,
            checkpoint_sha256=committed_pair.policy_sha256,
            candidates_dir=_frozen_league_candidate_dir(output_path),
            recurrent=config.model.recurrent is not None,
        )
    if published is not None:
        update_results[-1] = replace(
            update_results[-1],
            published_weights=published,
        )
    return published


def _learner_auxiliary_state(
    *,
    ppo_optimizer_updates: int,
    policy_iteration_learner: AmortizedPolicyIterationLearner | None,
) -> dict[str, Any]:
    """Snapshot independent learner clocks and optional target-network state."""
    if ppo_optimizer_updates < 0:
        raise ValueError("ppo_optimizer_updates must be non-negative")
    auxiliary: dict[str, Any] = {
        "ppo_optimizer_updates": ppo_optimizer_updates,
    }
    if policy_iteration_learner is not None:
        auxiliary["amortized_policy_iteration"] = (
            policy_iteration_learner.training_state_dict()
        )
    return auxiliary


def _runtime_policy_generation_fingerprint(
    config: RLTrainConfig,
    *,
    publish_version: int,
    runtime_attempt_id: str,
) -> str:
    """Return a collision-resistant identity for a non-artifact hot generation.

    This is deliberately not advertised as a content hash. It lets rollout
    leases distinguish exact in-process generations without forcing a full
    device-to-host model copy solely to feed the wire-format SHA-256 field.
    Learner-thread mode forbids planner snapshots, whose evidence contract
    requires a durable content fingerprint.
    """
    if publish_version < 0:
        raise ValueError("publish_version must be non-negative")
    if not runtime_attempt_id:
        raise ValueError("runtime_attempt_id must be non-empty")
    payload = json.dumps(
        {
            "resume_config_sha256": _resume_relevant_config_sha256(config),
            "run_version": config.run.version,
            "publish_version": publish_version,
            "runtime_attempt_id": runtime_attempt_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        b"ptcg-rl/runtime-policy-generation/v2\x00" + payload
    ).hexdigest()


def _should_write_disk_checkpoint(config: RLTrainConfig, publish_version: int) -> bool:
    interval = config.learner.disk_checkpoint_interval_versions
    if interval <= 1 or publish_version % interval == 0:
        return True
    if _allows_logical_only_publication(config):
        return False
    return not config.learner.shared_memory_publish


def _disk_weight_publisher_config(config: RLTrainConfig) -> WeightPublisherConfig:
    """Build the policy retention contract shared with resume sidecars."""
    return WeightPublisherConfig(
        keep_last=config.learner.disk_checkpoint_keep_last,
        retain_every_versions=config.learner.disk_checkpoint_retain_every_versions,
    )


def _allows_logical_only_publication(config: RLTrainConfig) -> bool:
    """Return whether hot actors can advance without an intermediate artifact."""
    learner_owned_hot_path = (
        config.execution.mode == "async"
        and config.execution.inference_server
        and config.execution.gpu_service_mode == "learner_thread"
        and not config.learner.shared_memory_publish
    )
    actor_free_coordinator = (
        config.execution.mode == "distributed_async"
        and config.execution.actors == 0
        and not config.execution.inference_server
        and config.distributed.coordinator_enabled
        and not config.distributed.worker_enabled
        and not config.learner.shared_memory_publish
    )
    return learner_owned_hot_path or actor_free_coordinator


def _learner_checkpoint_fields(
    config: RLTrainConfig,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    model_config = config.model.model_dump(mode="json")
    training_config = resolved_training_config_dump(
        config,
        task_name="rl",
        run=config.run,
        output_dir=config.output_dir,
    )
    if _is_dense_private_model(config.model):
        _remove_lora_runtime_schema(model_config)
        raw_training_model = training_config.get("model")
        if isinstance(raw_training_model, dict):
            _remove_lora_runtime_schema(raw_training_model)
    return {
        "model_config": model_config,
        "policy_input_schema": policy_input_schema_metadata(),
        "training_config": training_config,
        "metadata": dict(metadata),
    }


def _is_dense_private_model(config: AgentNetworkConfig) -> bool:
    conditioning = config.deck_conditioning
    return (
        conditioning is not None
        and conditioning.enabled
        and conditioning.architecture_version
        == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
    )


def _remove_lora_runtime_schema(model_payload: dict[str, Any]) -> None:
    """Remove the retired LoRA field from a dense-private artifact config."""
    conditioning = model_payload.get("deck_conditioning")
    if isinstance(conditioning, dict):
        conditioning.pop("lora", None)


def _learner_publish_metadata(
    *,
    update_results: Sequence[LearnerIterationResult],
    publish_version: int,
    current_policy_version: int,
    completed_iterations: int,
    total_optimizer_updates: int,
    ppo_optimizer_updates: int,
    transition_distillation_optimizer_updates: int,
    configured_ppo_epochs: int,
    target_kl_early_stop: float | None,
    target_kl_early_stop_multiplier: float,
    target_kl_stop_mode: str,
) -> dict[str, Any]:
    batch_stats = update_results[-1].batch_result.stats
    last_update = _last_ppo_update(update_results)
    updates = _ppo_updates(update_results)
    teacher_metrics = aggregate_engine_teacher_updates(updates)
    factual_metrics = aggregate_factual_updates(updates)
    macro_metrics = aggregate_macro_credit_updates(updates)
    planner_metrics = aggregate_planner_updates(updates)
    metadata: dict[str, Any] = {
        "updates": len(updates),
        "effective_updates": len(updates),
        "microbatches": sum(update.microbatch_count for update in updates),
        "updated_decisions": sum(update.sample_count for update in updates),
        "updated_objective_units": sum(
            update.objective_unit_count for update in updates
        ),
        "source_policy_version": current_policy_version,
        "publish_version": publish_version,
        "completed_training_iterations": completed_iterations,
        "total_optimizer_updates": total_optimizer_updates,
        "ppo_optimizer_updates": ppo_optimizer_updates,
        "transition_distillation": {
            "ppo_optimizer_update_budget": (transition_distillation_optimizer_updates),
            "ppo_optimizer_updates_completed": min(
                ppo_optimizer_updates,
                transition_distillation_optimizer_updates,
            ),
            "active_updates_in_window": sum(
                int(update.breakdown.transition_distillation) for update in updates
            ),
            "complete": (
                ppo_optimizer_updates >= transition_distillation_optimizer_updates
            ),
        },
        "ppo_epochs_configured": configured_ppo_epochs,
        "ppo_epochs_started": len(update_results),
        "target_kl_early_stop": target_kl_early_stop,
        "target_kl_early_stop_multiplier": target_kl_early_stop_multiplier,
        "target_kl_stop_mode": target_kl_stop_mode,
        "target_kl_stop_threshold": (
            target_kl_early_stop * target_kl_early_stop_multiplier
            if target_kl_early_stop is not None
            else None
        ),
        "trajectories": batch_stats.trajectories,
        "total_decisions": batch_stats.total_decisions,
        "kept_decisions": batch_stats.kept_decisions,
        "stale_decisions": batch_stats.stale_decisions,
        "intrinsically_stale_decisions": (batch_stats.intrinsically_stale_decisions),
        "propagated_stale_decisions": batch_stats.propagated_stale_decisions,
        "stale_seat_trajectories": batch_stats.stale_seat_trajectories,
        "retained_seat_trajectories": batch_stats.retained_seat_trajectories,
        "active_tokens": batch_stats.active_tokens,
        "voluntary_stop_tokens": batch_stats.voluntary_stop_tokens,
        "forced_max_decisions": batch_stats.forced_max_decisions,
        "min_policy_version": batch_stats.min_policy_version,
        "max_policy_version": batch_stats.max_policy_version,
        "policy_version_span": batch_stats.policy_version_span,
        "mixed_policy_games": batch_stats.mixed_policy_games,
        "max_game_policy_version_span": batch_stats.max_game_policy_version_span,
        "policy_age_histogram": dict(batch_stats.policy_age_histogram),
        "selection_count_histogram": dict(batch_stats.selection_count_histogram),
        "token_count_histogram": dict(batch_stats.token_count_histogram),
        "prompt_context_histogram": dict(batch_stats.prompt_context_histogram),
        "curriculum_mass": {
            dimension: {label: dict(counts) for label, counts in labels.items()}
            for dimension, labels in batch_stats.curriculum_mass.items()
        },
        "behavior_filtered_decisions": batch_stats.behavior_filtered_decisions,
        "trimmed_decisions": batch_stats.trimmed_decisions,
        "budget_overshoot_decisions": batch_stats.budget_overshoot_decisions,
        "legacy_temperature_inferred_decisions": (
            batch_stats.legacy_temperature_inferred_decisions
        ),
        "minibatches": batch_stats.minibatches,
        "early_stop": _learner_early_stop_summary(update_results),
        "engine_teacher_decisions": teacher_metrics.decisions,
        "engine_teacher_loss": teacher_metrics.loss,
        "factual_decisions": factual_metrics.decisions,
        "factual_effect_loss": factual_metrics.effect_loss,
        "factual_successor_loss": factual_metrics.successor_loss,
        "macro_roots": macro_metrics.roots,
        "macro_conditional_loss": macro_metrics.conditional_loss,
        "macro_expected_loss": macro_metrics.expected_loss,
        **planner_update_metrics_summary(planner_metrics),
        "ppo_update_distributions": _ppo_update_distributions(updates),
    }
    if last_update is not None:
        metadata["last_approx_kl"] = last_update.breakdown.approx_kl
        metadata["last_approx_kl_k3"] = last_update.breakdown.approx_kl_k3
        metadata["last_reference_kl"] = last_update.reference_kl
        metadata["last_reference_kl_k3"] = last_update.reference_kl_k3
        metadata["initial_reference_kl_k3"] = last_update.reference_initial_kl_k3
        metadata["last_anchor_kl"] = last_update.breakdown.anchor_kl
        metadata["last_engine_teacher_loss"] = _optional_loss_scalar(
            last_update.breakdown.engine_teacher_loss
        )
        metadata["last_engine_teacher_decisions"] = last_update.engine_teacher_decisions
        metadata["last_factual_decisions"] = last_update.factual_decisions
        metadata["last_factual_effect_loss"] = _optional_loss_scalar(
            last_update.breakdown.factual_effect_loss
        )
        metadata["last_factual_successor_loss"] = _optional_loss_scalar(
            last_update.breakdown.factual_successor_loss
        )
        metadata["last_macro_roots"] = last_update.macro_roots
        metadata["last_macro_conditional_loss"] = _optional_loss_scalar(
            last_update.breakdown.macro_conditional_loss
        )
        metadata["last_macro_expected_loss"] = _optional_loss_scalar(
            last_update.breakdown.macro_expected_loss
        )
        metadata["last_planner_decisions"] = last_update.planner_decisions
        metadata["last_planner_applicable_decisions"] = (
            last_update.planner_applicable_decisions
        )
        metadata["last_candidate_rerank_loss"] = _optional_loss_scalar(
            last_update.breakdown.candidate_rerank_loss
        )
        metadata["last_proposal_distillation_loss"] = _optional_loss_scalar(
            last_update.breakdown.proposal_distillation_loss
        )
        metadata["last_root_information_value_rows"] = (
            last_update.root_information_value_rows
        )
        metadata["last_root_information_value_loss"] = _optional_loss_scalar(
            last_update.breakdown.root_information_value_loss
        )
        metadata["last_update"] = _ppo_update_summary(last_update)
    return metadata


def _learner_update_count(
    update_results: Sequence[LearnerIterationResult],
) -> int:
    return sum(len(result.updates) for result in update_results)


def _ppo_updates(
    update_results: Sequence[LearnerIterationResult],
) -> tuple[PpoUpdateResult, ...]:
    return tuple(update for result in update_results for update in result.updates)


def _last_ppo_update(
    update_results: Sequence[LearnerIterationResult],
) -> PpoUpdateResult | None:
    for result in reversed(update_results):
        if result.updates:
            return result.updates[-1]
    return None


def _last_ppo_update_summary(
    update_results: Sequence[LearnerIterationResult],
) -> dict[str, Any] | None:
    update = _last_ppo_update(update_results)
    if update is None:
        return None
    return _ppo_update_summary(update)


def _learner_early_stop_summary(
    update_results: Sequence[LearnerIterationResult],
) -> dict[str, Any]:
    update_index = 0
    for epoch, result in enumerate(update_results):
        for minibatch, update in enumerate(result.updates):
            if update.should_stop:
                return {
                    "early_stopped": True,
                    "epoch": epoch,
                    "minibatch": minibatch,
                    "effective_update": update_index,
                    "iteration_update_index": update_index,
                    "approx_kl": update.breakdown.approx_kl,
                    "approx_kl_k3": update.breakdown.approx_kl_k3,
                    "kl_stop_baseline_k3": update.kl_stop_baseline_k3,
                    "kl_stop_delta_k3": update.kl_stop_delta_k3,
                    "kl_stop_threshold": update.kl_stop_threshold,
                    "reference_kl": update.reference_kl,
                    "reference_kl_k3": update.reference_kl_k3,
                    "reference_initial_kl_k3": update.reference_initial_kl_k3,
                    "reference_decisions": update.reference_decisions,
                }
            update_index += 1
    last_update = _last_ppo_update(update_results)
    return {
        "early_stopped": False,
        "effective_updates": update_index,
        "reference_kl": (None if last_update is None else last_update.reference_kl),
        "reference_kl_k3": (
            None if last_update is None else last_update.reference_kl_k3
        ),
        "reference_initial_kl_k3": (
            None if last_update is None else last_update.reference_initial_kl_k3
        ),
        "reference_decisions": (
            0 if last_update is None else last_update.reference_decisions
        ),
    }


def _ppo_update_summary(update: PpoUpdateResult) -> dict[str, Any]:
    breakdown = update.breakdown
    planner_applicable_fraction = (
        update.planner_applicable_decisions / update.planner_decisions
        if update.planner_decisions > 0
        else 0.0
    )
    return {
        "approx_kl": breakdown.approx_kl,
        "approx_kl_k3": breakdown.approx_kl_k3,
        "kl_stop_baseline_k3": update.kl_stop_baseline_k3,
        "kl_stop_delta_k3": update.kl_stop_delta_k3,
        "kl_stop_threshold": update.kl_stop_threshold,
        "reference_kl": update.reference_kl,
        "reference_kl_k3": update.reference_kl_k3,
        "reference_initial_kl_k3": update.reference_initial_kl_k3,
        "reference_decisions": update.reference_decisions,
        "anchor_kl": breakdown.anchor_kl,
        "entropy": breakdown.entropy,
        "clip_fraction": breakdown.clip_fraction,
        "ratio_mean": breakdown.ratio_mean,
        "ratio_p95": breakdown.ratio_p95,
        "value_mean": breakdown.value_mean,
        "grad_norm": update.grad_norm,
        "microbatch_count": update.microbatch_count,
        "sample_count": update.sample_count,
        "objective_unit_count": update.objective_unit_count,
        "engine_teacher_loss": _optional_loss_scalar(breakdown.engine_teacher_loss),
        "engine_teacher_decisions": update.engine_teacher_decisions,
        "factual_effect_loss": _optional_loss_scalar(breakdown.factual_effect_loss),
        "factual_successor_loss": _optional_loss_scalar(
            breakdown.factual_successor_loss
        ),
        "factual_decisions": update.factual_decisions,
        "macro_conditional_loss": _optional_loss_scalar(
            breakdown.macro_conditional_loss
        ),
        "macro_expected_loss": _optional_loss_scalar(breakdown.macro_expected_loss),
        "macro_roots": update.macro_roots,
        "candidate_rerank_loss": _optional_loss_scalar(breakdown.candidate_rerank_loss),
        "proposal_distillation_loss": _optional_loss_scalar(
            breakdown.proposal_distillation_loss
        ),
        "planner_decisions": update.planner_decisions,
        "planner_applicable_decisions": update.planner_applicable_decisions,
        "planner_applicable_fraction": planner_applicable_fraction,
        "root_information_value_loss": _optional_loss_scalar(
            breakdown.root_information_value_loss
        ),
        "root_information_value_rows": update.root_information_value_rows,
        **planner_imitation_metrics_summary(breakdown.planner_metrics),
        "should_stop": update.should_stop,
        "critic_warmup": breakdown.critic_warmup,
        "transition_distillation": breakdown.transition_distillation,
        "transition_sequence_kl": _optional_loss_scalar(
            breakdown.transition_sequence_kl
        ),
        "transition_prefix_kl": _optional_loss_scalar(breakdown.transition_prefix_kl),
        "transition_count_kl": _optional_loss_scalar(breakdown.transition_count_kl),
        "transition_root_value_loss": _optional_loss_scalar(
            breakdown.transition_root_value_loss
        ),
        "transition_prefix_value_loss": _optional_loss_scalar(
            breakdown.transition_prefix_value_loss
        ),
        "transition_action_wdl_loss": _optional_loss_scalar(
            breakdown.transition_action_wdl_loss
        ),
        "transition_engine_return_loss": _optional_loss_scalar(
            breakdown.transition_engine_return_loss
        ),
    }


def _ppo_update_distributions(
    updates: Sequence[PpoUpdateResult],
) -> dict[str, Any]:
    """Return acceptance-oriented distributions over effective updates."""
    teacher_metrics = aggregate_engine_teacher_updates(updates)
    factual_metrics = aggregate_factual_updates(updates)
    macro_metrics = aggregate_macro_credit_updates(updates)
    planner_metrics = aggregate_planner_updates(updates)
    return {
        "effective_updates": len(updates),
        "microbatches": sum(update.microbatch_count for update in updates),
        "updated_decisions": sum(update.sample_count for update in updates),
        "updated_objective_units": sum(
            update.objective_unit_count for update in updates
        ),
        "engine_teacher_decisions": teacher_metrics.decisions,
        "macro_roots": macro_metrics.roots,
        "engine_teacher_decision_weighted_loss": teacher_metrics.loss,
        "factual_decisions": factual_metrics.decisions,
        "factual_decision_weighted_effect_loss": factual_metrics.effect_loss,
        "factual_decision_weighted_successor_loss": (factual_metrics.successor_loss),
        "planner_decisions": planner_metrics.planner_decisions,
        "planner_applicable_decisions": planner_metrics.applicable_decisions,
        "planner_applicable_fraction": planner_metrics.applicable_fraction,
        "planner_applicable_decision_weighted_candidate_rerank_loss": (
            planner_metrics.candidate_rerank_loss
        ),
        "planner_applicable_decision_weighted_proposal_distillation_loss": (
            planner_metrics.proposal_distillation_loss
        ),
        "root_information_value_rows": (planner_metrics.root_information_value_rows),
        "root_information_value_row_weighted_loss": (
            planner_metrics.root_information_value_loss
        ),
        "planner_stale_decisions": planner_metrics.stale_decisions,
        "planner_inexact_decisions": planner_metrics.inexact_decisions,
        "planner_incomplete_grid_decisions": (
            planner_metrics.incomplete_grid_decisions
        ),
        "planner_identity_invalid_decisions": (
            planner_metrics.identity_invalid_decisions
        ),
        "planner_censored_support_decisions": (
            planner_metrics.censored_support_decisions
        ),
        "planner_exhaustive_support_decisions": (
            planner_metrics.exhaustive_support_decisions
        ),
        "planner_clipped_candidates": planner_metrics.clipped_candidates,
        "planner_candidate_count": planner_metrics.candidate_count,
        "planner_clipped_candidate_fraction": (
            planner_metrics.clipped_candidate_fraction
        ),
        "engine_teacher_loss": _finite_value_distribution(
            _optional_loss_scalar(update.breakdown.engine_teacher_loss)
            for update in updates
        ),
        "factual_effect_loss": _finite_value_distribution(
            _optional_loss_scalar(update.breakdown.factual_effect_loss)
            for update in updates
        ),
        "factual_successor_loss": _finite_value_distribution(
            _optional_loss_scalar(update.breakdown.factual_successor_loss)
            for update in updates
        ),
        "macro_conditional_loss": _finite_value_distribution(
            _optional_loss_scalar(update.breakdown.macro_conditional_loss)
            for update in updates
        ),
        "macro_expected_loss": _finite_value_distribution(
            _optional_loss_scalar(update.breakdown.macro_expected_loss)
            for update in updates
        ),
        "candidate_rerank_loss": _finite_value_distribution(
            _optional_loss_scalar(update.breakdown.candidate_rerank_loss)
            for update in updates
        ),
        "proposal_distillation_loss": _finite_value_distribution(
            _optional_loss_scalar(update.breakdown.proposal_distillation_loss)
            for update in updates
        ),
        "root_information_value_loss": _finite_value_distribution(
            _optional_loss_scalar(update.breakdown.root_information_value_loss)
            for update in updates
        ),
        "clip_fraction": _finite_value_distribution(
            update.breakdown.clip_fraction for update in updates
        ),
        "grad_norm": _finite_value_distribution(update.grad_norm for update in updates),
        "reference_kl_k3": _finite_value_distribution(
            update.reference_kl_k3 for update in updates
        ),
        "ratio_p95": _finite_value_distribution(
            update.breakdown.ratio_p95 for update in updates
        ),
    }


def _ppo_update_samples(
    updates: Sequence[PpoUpdateResult],
) -> dict[str, Any]:
    """Persist the small exact sample set needed by Sprint acceptance audits."""
    fields: dict[str, list[float | int | None]] = {
        "clip_fraction": [],
        "grad_norm": [],
        "reference_kl_k3": [],
        "reference_initial_kl_k3": [],
        "ratio_p95": [],
        "sample_count": [],
        "microbatch_count": [],
        "objective_unit_count": [],
        "engine_teacher_loss": [],
        "engine_teacher_decisions": [],
        "factual_effect_loss": [],
        "factual_successor_loss": [],
        "factual_decisions": [],
        "macro_conditional_loss": [],
        "macro_expected_loss": [],
        "macro_roots": [],
        "candidate_rerank_loss": [],
        "proposal_distillation_loss": [],
        "planner_decisions": [],
        "planner_applicable_decisions": [],
        "root_information_value_loss": [],
        "root_information_value_rows": [],
    }
    non_finite_metric_count = 0
    for update in updates:
        raw_values: dict[str, float | int | None] = {
            "clip_fraction": update.breakdown.clip_fraction,
            "grad_norm": update.grad_norm,
            "reference_kl_k3": update.reference_kl_k3,
            "reference_initial_kl_k3": update.reference_initial_kl_k3,
            "ratio_p95": update.breakdown.ratio_p95,
            "sample_count": update.sample_count,
            "microbatch_count": update.microbatch_count,
            "objective_unit_count": update.objective_unit_count,
            "engine_teacher_loss": _optional_loss_scalar(
                update.breakdown.engine_teacher_loss
            ),
            "engine_teacher_decisions": update.engine_teacher_decisions,
            "factual_effect_loss": _optional_loss_scalar(
                update.breakdown.factual_effect_loss
            ),
            "factual_successor_loss": _optional_loss_scalar(
                update.breakdown.factual_successor_loss
            ),
            "factual_decisions": update.factual_decisions,
            "macro_conditional_loss": _optional_loss_scalar(
                update.breakdown.macro_conditional_loss
            ),
            "macro_expected_loss": _optional_loss_scalar(
                update.breakdown.macro_expected_loss
            ),
            "macro_roots": update.macro_roots,
            "candidate_rerank_loss": _optional_loss_scalar(
                update.breakdown.candidate_rerank_loss
            ),
            "proposal_distillation_loss": _optional_loss_scalar(
                update.breakdown.proposal_distillation_loss
            ),
            "planner_decisions": update.planner_decisions,
            "planner_applicable_decisions": update.planner_applicable_decisions,
            "root_information_value_loss": _optional_loss_scalar(
                update.breakdown.root_information_value_loss
            ),
            "root_information_value_rows": update.root_information_value_rows,
        }
        for name, value in raw_values.items():
            if isinstance(value, float) and not math.isfinite(value):
                non_finite_metric_count += 1
                fields[name].append(None)
            else:
                fields[name].append(value)
    return {
        **fields,
        "non_finite_metric_count": non_finite_metric_count,
    }


def _training_health_summary(
    iterations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate exact per-window health metrics for diagnostics."""
    early_stop_count = 0
    epoch_zero_early_stop_count = 0
    stale_fractions: list[float] = []
    data_passes: list[float] = []
    clip_fractions: list[float | None] = []
    grad_norms: list[float | None] = []
    reference_initial_k3: list[float | None] = []
    ratio_p95_values: list[float | None] = []
    effective_updates = 0
    non_finite_metric_count = 0
    for iteration in iterations:
        learner = iteration.get("learner")
        if not isinstance(learner, Mapping):
            continue
        early_stop = learner.get("early_stop")
        if isinstance(early_stop, Mapping) and bool(
            early_stop.get("early_stopped", False)
        ):
            early_stop_count += 1
            if int(early_stop.get("epoch", -1)) == 0:
                epoch_zero_early_stop_count += 1
        batch_stats = learner.get("batch_stats")
        kept_decisions = 0
        if isinstance(batch_stats, Mapping):
            total_decisions = int(batch_stats.get("total_decisions", 0))
            stale_decisions = int(batch_stats.get("stale_decisions", 0))
            kept_decisions = int(batch_stats.get("kept_decisions", 0))
            stale_fractions.append(
                stale_decisions / total_decisions if total_decisions > 0 else 0.0
            )
        samples = learner.get("ppo_samples")
        if not isinstance(samples, Mapping):
            continue
        clip_fractions.extend(_optional_float_sequence(samples.get("clip_fraction")))
        grad_norms.extend(_optional_float_sequence(samples.get("grad_norm")))
        ratio_p95_values.extend(_optional_float_sequence(samples.get("ratio_p95")))
        initial_values = _optional_float_sequence(
            samples.get("reference_initial_kl_k3")
        )
        reference_initial_k3.append(
            next((value for value in initial_values if value is not None), None)
        )
        sample_counts = _int_sequence(samples.get("sample_count"))
        effective_updates += len(sample_counts)
        if kept_decisions > 0:
            data_passes.append(sum(sample_counts) / kept_decisions)
        non_finite_metric_count += int(samples.get("non_finite_metric_count", 0))
    iteration_count = len(iterations)
    return {
        "iterations": iteration_count,
        "early_stop_count": early_stop_count,
        "early_stop_fraction": (
            early_stop_count / iteration_count if iteration_count > 0 else 0.0
        ),
        "epoch_zero_early_stop_count": epoch_zero_early_stop_count,
        "effective_updates": effective_updates,
        "clip_fraction": _finite_value_distribution(clip_fractions),
        "grad_norm": _finite_value_distribution(grad_norms),
        "ratio_p95": _finite_value_distribution(ratio_p95_values),
        "fixed_reference_initial_kl_k3": _finite_value_distribution(
            reference_initial_k3
        ),
        "stale_fraction": _finite_value_distribution(stale_fractions),
        "data_passes": _finite_value_distribution(data_passes),
        "non_finite_metric_count": non_finite_metric_count,
        "overflow_decisions": None,
        "overflow_trajectories": None,
    }


def _health_with_distributed_transport(
    raw_health: Any,
    distributed_summary: Mapping[str, Any],
) -> dict[str, Any]:
    health = dict(raw_health) if isinstance(raw_health, Mapping) else {}
    counters = distributed_summary.get("summary", distributed_summary)
    if not isinstance(counters, Mapping):
        counters = {}
    health["overflow_decisions"] = int(counters.get("dropped_overflow_decisions", 0))
    health["overflow_trajectories"] = int(
        counters.get("dropped_overflow_trajectories", 0)
    )
    health["transport_stale_decisions"] = int(
        counters.get("dropped_stale_decisions", 0)
    )
    health["transport_stale_trajectories"] = int(
        counters.get("dropped_stale_trajectories", 0)
    )
    return health


def _optional_float_sequence(value: Any) -> list[float | None]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    output: list[float | None] = []
    for item in value:
        if item is None:
            output.append(None)
        elif isinstance(item, int | float):
            output.append(float(item))
    return output


def _optional_loss_scalar(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    scalar = float(value.detach().item())
    return scalar if math.isfinite(scalar) else None


def _int_sequence(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [int(item) for item in value if isinstance(item, int)]


def _finite_value_distribution(
    values: Iterable[float | None],
) -> dict[str, float | int | None]:
    raw_values = list(values)
    finite_values = [
        float(value)
        for value in raw_values
        if value is not None and math.isfinite(value)
    ]
    non_finite_count = sum(
        1 for value in raw_values if value is not None and not math.isfinite(value)
    )
    if not finite_values:
        return {
            "count": 0,
            "input_count": len(raw_values),
            "non_finite_count": non_finite_count,
            "mean": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    tensor = torch.tensor(finite_values, dtype=torch.float64)
    return {
        "count": len(finite_values),
        "input_count": len(raw_values),
        "non_finite_count": non_finite_count,
        "mean": float(tensor.mean().item()),
        "min": min(finite_values),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "max": max(finite_values),
    }


def _learner_batch_stats_dict(batch_result: LearnerBatchResult) -> dict[str, Any]:
    return {
        "trajectories": batch_result.stats.trajectories,
        "total_decisions": batch_result.stats.total_decisions,
        "kept_decisions": batch_result.stats.kept_decisions,
        "stale_decisions": batch_result.stats.stale_decisions,
        "intrinsically_stale_decisions": (
            batch_result.stats.intrinsically_stale_decisions
        ),
        "propagated_stale_decisions": (batch_result.stats.propagated_stale_decisions),
        "stale_seat_trajectories": (batch_result.stats.stale_seat_trajectories),
        "retained_seat_trajectories": (batch_result.stats.retained_seat_trajectories),
        "active_tokens": batch_result.stats.active_tokens,
        "voluntary_stop_tokens": batch_result.stats.voluntary_stop_tokens,
        "forced_max_decisions": batch_result.stats.forced_max_decisions,
        "min_policy_version": batch_result.stats.min_policy_version,
        "max_policy_version": batch_result.stats.max_policy_version,
        "policy_version_span": batch_result.stats.policy_version_span,
        "mixed_policy_games": batch_result.stats.mixed_policy_games,
        "max_game_policy_version_span": (
            batch_result.stats.max_game_policy_version_span
        ),
        "policy_age_histogram": dict(batch_result.stats.policy_age_histogram),
        "selection_count_histogram": dict(batch_result.stats.selection_count_histogram),
        "token_count_histogram": dict(batch_result.stats.token_count_histogram),
        "prompt_context_histogram": dict(batch_result.stats.prompt_context_histogram),
        "curriculum_mass": {
            dimension: {label: dict(counts) for label, counts in labels.items()}
            for dimension, labels in batch_result.stats.curriculum_mass.items()
        },
        "deck_mass": {
            signature: dict(counts)
            for signature, counts in batch_result.stats.deck_mass.items()
        },
        "behavior_filtered_decisions": (batch_result.stats.behavior_filtered_decisions),
        "trimmed_decisions": batch_result.stats.trimmed_decisions,
        "budget_overshoot_decisions": (batch_result.stats.budget_overshoot_decisions),
        "legacy_temperature_inferred_decisions": (
            batch_result.stats.legacy_temperature_inferred_decisions
        ),
        "minibatches": batch_result.stats.minibatches,
    }


def _maybe_write_learner_status(
    config: RLTrainConfig,
    output_path: Path,
    payload: Mapping[str, Any],
) -> None:
    if _should_write_learner_status(config, payload):
        _write_learner_status(output_path, payload)


def _should_write_learner_status(
    config: RLTrainConfig,
    payload: Mapping[str, Any],
) -> bool:
    interval = config.diagnostics.learner_status_interval_versions
    if interval <= 1:
        return True
    if payload.get("phase") == "starting":
        return True
    version = _learner_status_payload_version(payload)
    return version is not None and version % interval == 0


def _should_collect_ppo_diagnostics(
    config: RLTrainConfig,
    *,
    publish_version: int,
) -> bool:
    interval = config.diagnostics.ppo_diagnostics_interval_versions
    return interval <= 1 or publish_version % interval == 0


def _learner_status_payload_version(payload: Mapping[str, Any]) -> int | None:
    for key in ("published_version", "publish_version"):
        value = payload.get(key)
        if value is None:
            continue
        return int(value)
    return None


def _write_learner_status(
    output_path: Path,
    payload: Mapping[str, Any],
) -> None:
    status = {
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    _write_summary(_learner_status_path(output_path), status)


def _learner_status_path(output_path: Path) -> Path:
    return output_path / "learner_status.json"


def _learner_tensorboard_writer(
    config: RLTrainConfig,
    output_path: Path,
) -> TensorboardMetricWriter:
    return TensorboardMetricWriter.create(
        tensorboard_root_dir(output_path) / "learner",
        enabled=config.diagnostics.tensorboard,
        flush_seconds=config.diagnostics.tensorboard_flush_seconds,
    )


def _runtime_tensorboard_writer(
    config: RLTrainConfig,
    output_path: Path,
) -> TensorboardMetricWriter:
    return TensorboardMetricWriter.create(
        tensorboard_root_dir(output_path) / "runtime",
        enabled=config.diagnostics.tensorboard,
        flush_seconds=config.diagnostics.tensorboard_flush_seconds,
    )


def _add_tensorboard_log_dir(
    summary: dict[str, Any],
    writer: TensorboardMetricWriter,
    *,
    output_dir: Path,
) -> None:
    if writer.enabled:
        summary["tensorboard_log_dir"] = deck_records.display_path(
            tensorboard_root_dir(output_dir)
        )


def _runtime_actor_queue_summary(
    output_dir: Path,
    runtime_summary: Mapping[str, Any],
) -> dict[str, Any] | None:
    last_sample = runtime_summary.get("last_sample")
    if not isinstance(last_sample, Mapping):
        return None
    actors = last_sample.get("actors")
    if not isinstance(actors, Sequence) or isinstance(actors, str):
        return None
    if not actors:
        return None
    with suppress(OSError, ValueError, json.JSONDecodeError):
        return _actor_queue_summary(output_dir, actor_count=len(actors))
    return None


def _sync_iteration_summary(
    *,
    iteration_index: int,
    sync_iteration: _SyncIteration,
    update_results: Sequence[LearnerIterationResult],
    published_version: int | None,
    started_elapsed_seconds: float,
    finished_elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "iteration": iteration_index,
        "started_elapsed_seconds": started_elapsed_seconds,
        "finished_elapsed_seconds": finished_elapsed_seconds,
        "policy_version": sync_iteration.policy_version,
        "collect": sync_iteration.counters.as_dict(),
        "rollout_features": dict(sync_iteration.rollout_features),
        "trajectories": len(sync_iteration.trajectories),
        "trajectory_decisions": sum(
            trajectory.decision_count for trajectory in sync_iteration.trajectories
        ),
        "deferred_trajectories": len(sync_iteration.deferred_trajectories),
        "deferred_decisions": _decision_count(sync_iteration.deferred_trajectories),
        "learner": {
            "updates": sum(len(result.updates) for result in update_results),
            "effective_updates": sum(len(result.updates) for result in update_results),
            "published_version": published_version,
            "batch_stats": (
                update_results[-1].batch_result.stats.__dict__
                if update_results
                else None
            ),
            "ppo": _ppo_update_distributions(_ppo_updates(update_results)),
            "ppo_samples": _ppo_update_samples(_ppo_updates(update_results)),
            "early_stop": _learner_early_stop_summary(update_results),
        },
        "curriculum": sync_iteration.curriculum_summary,
    }


def _async_iteration_summary(
    *,
    iteration_index: int,
    policy_version: int,
    trajectories: Sequence[GameTrajectory],
    update_results: Sequence[LearnerIterationResult],
    published_version: int | None,
    archive_result: TrajectoryWriteResult | None,
    started_elapsed_seconds: float,
    finished_elapsed_seconds: float,
    policy_iteration_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy_iteration_updates = int(
        (policy_iteration_summary or {}).get("update_count", 0)
    )
    return {
        "iteration": iteration_index,
        "started_elapsed_seconds": started_elapsed_seconds,
        "finished_elapsed_seconds": finished_elapsed_seconds,
        "policy_version": policy_version,
        "trajectories": len(trajectories),
        "trajectory_decisions": sum(
            trajectory.decision_count for trajectory in trajectories
        ),
        "learner": {
            "updates": (
                sum(len(result.updates) for result in update_results)
                + policy_iteration_updates
            ),
            "effective_updates": (
                sum(len(result.updates) for result in update_results)
                + policy_iteration_updates
            ),
            "published_version": published_version,
            "batch_stats": (
                update_results[-1].batch_result.stats.__dict__
                if update_results
                else None
            ),
            "ppo": _ppo_update_distributions(_ppo_updates(update_results)),
            "ppo_samples": _ppo_update_samples(_ppo_updates(update_results)),
            "early_stop": _learner_early_stop_summary(update_results),
            "amortized_policy_iteration": policy_iteration_summary,
        },
        "archive": _archive_summary(archive_result),
    }


def _training_throughput_summary(
    config: RLTrainConfig,
    *,
    iterations: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Return all-window and post-warmup learner throughput diagnostics."""
    warmup_iterations = min(
        config.diagnostics.throughput_warmup_iterations,
        len(iterations),
    )
    return {
        "warmup_iterations": warmup_iterations,
        "all": _training_throughput_window(
            iterations,
            start_index=0,
            elapsed_seconds=elapsed_seconds,
        ),
        "measured": _training_throughput_window(
            iterations,
            start_index=warmup_iterations,
            elapsed_seconds=elapsed_seconds,
        ),
    }


def _training_throughput_window(
    iterations: Sequence[Mapping[str, Any]],
    *,
    start_index: int,
    elapsed_seconds: float,
) -> dict[str, float | int | None]:
    selected = list(iterations[start_index:])
    if not selected:
        return {
            "start_iteration": None,
            "end_iteration": None,
            "iterations": 0,
            "elapsed_seconds": 0.0,
            "trajectory_decisions": 0,
            "kept_decisions": 0,
            "stale_decisions": 0,
            "trajectory_decisions_per_second": 0.0,
            "kept_decisions_per_second": 0.0,
        }
    window_start = (
        0.0
        if start_index <= 0
        else _float_mapping_value(selected[0], "started_elapsed_seconds", 0.0)
    )
    window_end = _float_mapping_value(
        selected[-1],
        "finished_elapsed_seconds",
        elapsed_seconds,
    )
    window_seconds = max(0.0, window_end - window_start)
    trajectory_decisions = sum(
        int(iteration.get("trajectory_decisions", 0)) for iteration in selected
    )
    kept_decisions = sum(
        _iteration_batch_stat(iteration, "kept_decisions") for iteration in selected
    )
    stale_decisions = sum(
        _iteration_batch_stat(iteration, "stale_decisions") for iteration in selected
    )
    return {
        "start_iteration": int(selected[0].get("iteration", start_index)),
        "end_iteration": int(selected[-1].get("iteration", start_index)),
        "iterations": len(selected),
        "elapsed_seconds": window_seconds,
        "trajectory_decisions": trajectory_decisions,
        "kept_decisions": kept_decisions,
        "stale_decisions": stale_decisions,
        "trajectory_decisions_per_second": _safe_rate(
            trajectory_decisions,
            window_seconds,
        ),
        "kept_decisions_per_second": _safe_rate(kept_decisions, window_seconds),
    }


def _iteration_batch_stat(iteration: Mapping[str, Any], key: str) -> int:
    learner = iteration.get("learner")
    if not isinstance(learner, Mapping):
        return 0
    batch_stats = learner.get("batch_stats")
    if not isinstance(batch_stats, Mapping):
        return 0
    return int(batch_stats.get(key, 0))


def _float_mapping_value(
    mapping: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    value = mapping.get(key, default)
    return float(value) if value is not None else default


def _safe_rate(count: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0.0:
        return 0.0
    return float(count) / elapsed_seconds


def _sum_collect_counters(
    iterations: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    totals = _MutableSyncCounters()
    for iteration in iterations:
        collect = iteration.get("collect", {})
        if not isinstance(collect, Mapping):
            continue
        totals.collect_iterations += int(collect.get("collect_iterations", 0))
        totals.policy_actions += int(collect.get("policy_actions", 0))
        totals.forced_actions += int(collect.get("forced_actions", 0))
        totals.scripted_actions += int(collect.get("scripted_actions", 0))
        totals.recorded_decisions += int(collect.get("recorded_decisions", 0))
        totals.finished_games += int(collect.get("finished_games", 0))
    return totals.as_dict()


def _learner_timing_summary(
    timer: StageTimer,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    stage_timings = timer.summary()
    data_prep_seconds = sum(
        float(stage_timings.get(stage, {}).get("seconds", 0.0))
        for stage in (
            "learner_index_decisions",
            "learner_gae",
            "learner_filter_shuffle",
            "learner_collate",
        )
    )
    return {
        "stage_timings": stage_timings,
        "data_prep_seconds": data_prep_seconds,
        "data_prep_fraction": (
            data_prep_seconds / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        ),
    }


def _actor_queue_summary(
    output_path: Path,
    *,
    actor_count: int,
) -> dict[str, Any]:
    actor_summaries = tuple(
        _read_summary(
            _actor_summary_path(
                output_path,
                actor_index=actor_index,
                actor_count=actor_count,
            )
        )
        for actor_index in range(actor_count)
    )
    totals = Counter[str]()
    queue_put_seconds = 0.0
    pickle_probe_seconds = 0.0
    pickle_probe_bytes = 0
    actor_elapsed_seconds = 0.0
    rollout_stage_seconds: dict[str, float] = {}
    rollout_stage_counts: dict[str, int] = {}
    rollout_feature_totals = Counter[str]()
    rollout_probe_backend_counts = Counter[str]()
    rollout_teacher_totals: dict[str, float] = {}
    rollout_teacher_max_step_seconds = 0.0
    rollout_teacher_reason_counts = Counter[str]()
    rollout_teacher_error_counts = Counter[str]()
    rollout_teacher_context_counts = Counter[str]()
    saw_live_game_steps = False
    live_game_count = 0
    live_game_steps_max = 0
    max_live_game_steps_seen = 0
    oldest_live_games: list[dict[str, Any]] = []
    stale_recurrent_last_learner_version = -1
    stale_recurrent_last_served_version = -1
    stale_recurrent_oldest_candidate_policy_version: int | None = None
    stale_recurrent_max_candidate_version_age = 0
    for fallback_actor_index, summary in enumerate(actor_summaries):
        stats = summary.get("stats", {})
        if not isinstance(stats, Mapping):
            continue
        totals["queued_trajectories"] += int(stats.get("queued_trajectories", 0))
        totals["queued_trajectory_decisions"] += int(
            stats.get("queued_trajectory_decisions", 0)
        )
        totals["queue_full_retries"] += int(stats.get("queue_full_retries", 0))
        queue_put_seconds += float(stats.get("queue_put_seconds", 0.0))
        pickle_probe_seconds += float(stats.get("queue_pickle_probe_seconds", 0.0))
        pickle_probe_bytes += int(stats.get("queue_pickle_probe_bytes", 0))
        actor_elapsed_seconds += float(stats.get("elapsed_seconds", 0.0))
        stage_timings = stats.get("stage_timings", {})
        if isinstance(stage_timings, Mapping):
            _accumulate_stage_timings(
                stage_timings,
                seconds_by_stage=rollout_stage_seconds,
                counts_by_stage=rollout_stage_counts,
            )
        rollout_features = stats.get("rollout_features", {})
        if isinstance(rollout_features, Mapping):
            if "live_game_count" in rollout_features:
                saw_live_game_steps = True
                live_game_count += max(
                    0,
                    int(rollout_features.get("live_game_count", 0)),
                )
                live_game_steps_max = max(
                    live_game_steps_max,
                    int(rollout_features.get("live_game_steps_max", 0)),
                )
                max_live_game_steps_seen = max(
                    max_live_game_steps_seen,
                    int(rollout_features.get("max_live_game_steps_seen", 0)),
                )
                actor_index = int(summary.get("actor_index", fallback_actor_index))
                raw_oldest = rollout_features.get("oldest_live_games", ())
                if isinstance(raw_oldest, Sequence) and not isinstance(
                    raw_oldest,
                    str,
                ):
                    for raw_game in raw_oldest:
                        if not isinstance(raw_game, Mapping):
                            continue
                        oldest_live_games.append(
                            {
                                "actor_index": actor_index,
                                "game_id": str(raw_game.get("game_id", "")),
                                "steps": max(0, int(raw_game.get("steps", 0))),
                            }
                        )
            for key in (
                "probe_calls",
                "probe_eligible_options",
                "probe_probed_options",
                "probe_worlds",
                "probe_native_batch_calls",
                "probe_native_transitions",
                "probe_native_errors",
                "probe_unresolved_options",
                "probe_unresolved_worlds",
                "belief_cache_entries",
                "stale_recurrent_recycle_polls",
                "stale_recurrent_candidate_sequences_examined",
                "stale_recurrent_games_recycled",
                "stale_recurrent_games_deferred_pending_evidence",
                "stale_recurrent_sequences_released",
            ):
                rollout_feature_totals[key] += int(rollout_features.get(key, 0))
            stale_recurrent_last_learner_version = max(
                stale_recurrent_last_learner_version,
                int(
                    rollout_features.get(
                        "stale_recurrent_last_learner_version",
                        -1,
                    )
                ),
            )
            stale_recurrent_last_served_version = max(
                stale_recurrent_last_served_version,
                int(
                    rollout_features.get(
                        "stale_recurrent_last_served_version",
                        -1,
                    )
                ),
            )
            raw_oldest_candidate_version = int(
                rollout_features.get(
                    "stale_recurrent_oldest_candidate_policy_version",
                    -1,
                )
            )
            if raw_oldest_candidate_version >= 0:
                stale_recurrent_oldest_candidate_policy_version = min(
                    raw_oldest_candidate_version,
                    (
                        raw_oldest_candidate_version
                        if stale_recurrent_oldest_candidate_policy_version is None
                        else stale_recurrent_oldest_candidate_policy_version
                    ),
                )
            stale_recurrent_max_candidate_version_age = max(
                stale_recurrent_max_candidate_version_age,
                int(
                    rollout_features.get(
                        "stale_recurrent_max_candidate_version_age",
                        0,
                    )
                ),
            )
            backend_counts = rollout_features.get("probe_backend_counts", {})
            if isinstance(backend_counts, Mapping):
                for backend, count in backend_counts.items():
                    rollout_probe_backend_counts[str(backend)] += int(count)
            for key in (
                "engine_teacher_requests",
                "engine_teacher_batches",
                "engine_teacher_targets",
                "engine_teacher_search_targets",
                "engine_teacher_behavior_matches",
                "engine_teacher_eligible",
                "engine_teacher_attempted",
                "engine_teacher_emitted",
                "engine_teacher_behavior_matches_online",
                "engine_teacher_worlds",
                "engine_teacher_nodes",
                "engine_teacher_coverage_sum",
                "engine_teacher_confidence_sum_online",
                "engine_teacher_seconds",
                "engine_teacher_step_calls",
                "engine_teacher_probability_skips",
                "engine_teacher_attempt_cap_skips",
                "engine_teacher_step_budget_skips",
                "engine_teacher_deadline_expiries",
                "engine_teacher_isolated_process",
                "engine_teacher_worker_starts",
                "engine_teacher_worker_restarts",
                "engine_teacher_worker_startup_failures",
                "engine_teacher_worker_hard_timeouts",
                "engine_teacher_worker_crashes",
                "engine_teacher_worker_forced_terminations",
            ):
                rollout_teacher_totals[key] = rollout_teacher_totals.get(
                    key, 0.0
                ) + float(rollout_features.get(key, 0.0))
            rollout_teacher_max_step_seconds = max(
                rollout_teacher_max_step_seconds,
                float(rollout_features.get("engine_teacher_max_step_seconds", 0.0)),
            )
            for source_key, target in (
                ("engine_teacher_reason_counts", rollout_teacher_reason_counts),
                ("engine_teacher_error_counts", rollout_teacher_error_counts),
                ("engine_teacher_context_counts", rollout_teacher_context_counts),
            ):
                counts = rollout_features.get(source_key, {})
                if isinstance(counts, Mapping):
                    for label, count in counts.items():
                        target[str(label)] += int(count)
    total_seconds = queue_put_seconds + pickle_probe_seconds
    policy_wait_seconds = rollout_stage_seconds.get("rollout_policy_wait", 0.0)
    teacher_eligible = int(rollout_teacher_totals.get("engine_teacher_eligible", 0.0))
    teacher_attempted = int(rollout_teacher_totals.get("engine_teacher_attempted", 0.0))
    teacher_emitted = int(rollout_teacher_totals.get("engine_teacher_emitted", 0.0))
    teacher_targets = int(rollout_teacher_totals.get("engine_teacher_targets", 0.0))
    teacher_search_targets = int(
        rollout_teacher_totals.get("engine_teacher_search_targets", 0.0)
    )
    oldest_live_games.sort(
        key=lambda row: (
            -int(row["steps"]),
            int(row["actor_index"]),
            str(row["game_id"]),
        )
    )
    live_game_step_features = (
        {
            "live_game_count": live_game_count,
            "live_game_steps_max": live_game_steps_max,
            "max_live_game_steps_seen": max_live_game_steps_seen,
            "oldest_live_games": oldest_live_games[:_OLDEST_LIVE_GAMES_LIMIT],
        }
        if saw_live_game_steps
        else {}
    )
    stale_recurrent_features = {
        "stale_recurrent_last_learner_version": (stale_recurrent_last_learner_version),
        "stale_recurrent_last_served_version": stale_recurrent_last_served_version,
        "stale_recurrent_oldest_candidate_policy_version": (
            -1
            if stale_recurrent_oldest_candidate_policy_version is None
            else stale_recurrent_oldest_candidate_policy_version
        ),
        "stale_recurrent_max_candidate_version_age": (
            stale_recurrent_max_candidate_version_age
        ),
    }
    return {
        "actors": actor_count,
        "queued_trajectories": totals["queued_trajectories"],
        "queued_trajectory_decisions": totals["queued_trajectory_decisions"],
        "queue_full_retries": totals["queue_full_retries"],
        "queue_put_seconds": queue_put_seconds,
        "queue_pickle_probe_seconds": pickle_probe_seconds,
        "queue_transfer_seconds": total_seconds,
        "queue_pickle_probe_bytes": pickle_probe_bytes,
        "mean_queue_transfer_seconds_per_trajectory": (
            total_seconds / totals["queued_trajectories"]
            if totals["queued_trajectories"] > 0
            else 0.0
        ),
        "actor_elapsed_seconds": actor_elapsed_seconds,
        "rollout_stage_timings": _stage_timing_summary(
            rollout_stage_seconds,
            rollout_stage_counts,
        ),
        "rollout_policy_wait_seconds": policy_wait_seconds,
        "rollout_policy_wait_fraction": (
            policy_wait_seconds / actor_elapsed_seconds
            if actor_elapsed_seconds > 0.0
            else 0.0
        ),
        "rollout_features": {
            **dict(sorted(rollout_feature_totals.items())),
            **live_game_step_features,
            **stale_recurrent_features,
            "probe_backend_counts": dict(sorted(rollout_probe_backend_counts.items())),
            "engine_teacher": {
                **{
                    key: int(value)
                    for key, value in rollout_teacher_totals.items()
                    if key
                    not in {
                        "engine_teacher_coverage_sum",
                        "engine_teacher_confidence_sum_online",
                        "engine_teacher_seconds",
                    }
                },
                "seconds": rollout_teacher_totals.get("engine_teacher_seconds", 0.0),
                "max_step_seconds": rollout_teacher_max_step_seconds,
                "attempt_rate": (
                    teacher_attempted / teacher_eligible if teacher_eligible else 0.0
                ),
                "emit_rate": (
                    teacher_emitted / teacher_attempted if teacher_attempted else 0.0
                ),
                "search_target_rate": (
                    teacher_search_targets / teacher_targets if teacher_targets else 0.0
                ),
                "mean_attempt_coverage": (
                    rollout_teacher_totals.get("engine_teacher_coverage_sum", 0.0)
                    / teacher_attempted
                    if teacher_attempted
                    else 0.0
                ),
                "mean_emitted_confidence": (
                    rollout_teacher_totals.get(
                        "engine_teacher_confidence_sum_online", 0.0
                    )
                    / teacher_emitted
                    if teacher_emitted
                    else 0.0
                ),
                "reason_counts": dict(sorted(rollout_teacher_reason_counts.items())),
                "error_counts": dict(sorted(rollout_teacher_error_counts.items())),
                "context_counts": dict(sorted(rollout_teacher_context_counts.items())),
            },
        },
    }


@dataclass
class _ProcessCpuTracker:
    """Compute per-process CPU percent from cumulative ``/proc`` counters."""

    previous: dict[int, tuple[float, float]] = field(default_factory=dict)

    def sample_cpu_percent(self, pid: int | None, *, now: float) -> float | None:
        """Return CPU percent since the last sample for this process."""
        if pid is None:
            return None
        cpu_seconds = _process_cpu_seconds(pid)
        if cpu_seconds is None:
            return None
        previous = self.previous.get(pid)
        self.previous[pid] = (now, cpu_seconds)
        if previous is None:
            return 0.0
        previous_time, previous_cpu_seconds = previous
        elapsed_seconds = now - previous_time
        cpu_delta = cpu_seconds - previous_cpu_seconds
        if elapsed_seconds <= 0.0 or cpu_delta < 0.0:
            return 0.0
        return 100.0 * cpu_delta / elapsed_seconds


class _NvmlUtilization(ctypes.Structure):
    """ctypes mirror of NVML utilization counters."""

    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]


@dataclass
class _CtypesNvmlDevice:
    """Minimal direct NVML binding used when ``pynvml`` is unavailable."""

    library: Any
    handle: ctypes.c_void_p

    @classmethod
    def open(cls, device_index: int) -> _CtypesNvmlDevice:
        """Open one NVML device handle through ``libnvidia-ml``."""
        library = ctypes.CDLL("libnvidia-ml.so.1")
        init = _nvml_symbol(library, "nvmlInit_v2", "nvmlInit")
        _check_nvml(init())
        handle = ctypes.c_void_p()
        get_handle = _nvml_symbol(
            library,
            "nvmlDeviceGetHandleByIndex_v2",
            "nvmlDeviceGetHandleByIndex",
        )
        _check_nvml(get_handle(ctypes.c_uint(device_index), ctypes.byref(handle)))
        return cls(library=library, handle=handle)

    def sample(self) -> dict[str, Any]:
        """Return one GPU sample from direct NVML calls."""
        sample: dict[str, Any] = {}
        try:
            utilization = _NvmlUtilization()
            get_utilization = _nvml_symbol(
                self.library,
                "nvmlDeviceGetUtilizationRates",
            )
            _check_nvml(get_utilization(self.handle, ctypes.byref(utilization)))
            sample["utilization_percent"] = float(utilization.gpu)
            sample["memory_utilization_percent"] = float(utilization.memory)
        except (AttributeError, RuntimeError) as exc:
            sample["utilization_error"] = f"{type(exc).__name__}: {exc}"
        try:
            power_mw = ctypes.c_uint()
            get_power = _nvml_symbol(self.library, "nvmlDeviceGetPowerUsage")
            _check_nvml(get_power(self.handle, ctypes.byref(power_mw)))
            sample["power_watts"] = float(power_mw.value) / 1000.0
        except (AttributeError, RuntimeError) as exc:
            sample["power_error"] = f"{type(exc).__name__}: {exc}"
        try:
            temperature = ctypes.c_uint()
            get_temperature = _nvml_symbol(self.library, "nvmlDeviceGetTemperature")
            _check_nvml(
                get_temperature(
                    self.handle,
                    ctypes.c_uint(0),
                    ctypes.byref(temperature),
                )
            )
            sample["temperature_celsius"] = float(temperature.value)
        except (AttributeError, RuntimeError) as exc:
            sample["temperature_error"] = f"{type(exc).__name__}: {exc}"
        return sample


def _nvml_symbol(library: Any, *names: str) -> Any:
    for name in names:
        symbol = getattr(library, name, None)
        if symbol is not None:
            return symbol
    raise AttributeError(f"missing NVML symbol: {'/'.join(names)}")


def _check_nvml(result: int) -> None:
    if int(result) != 0:
        raise RuntimeError(f"NVML error code {int(result)}")


@dataclass
class _NvmlGpuSampler:
    """Best-effort NVML sampler for GPU utilization and power draw."""

    device_index: int = 0
    _initialized: bool = False
    _nvml: Any | None = None
    _handle: Any | None = None
    _ctypes_device: _CtypesNvmlDevice | None = None
    _error: str | None = None

    def sample(self) -> dict[str, Any]:
        """Return one NVML sample, or an error field when unavailable."""
        if not self._initialized:
            self._initialize()
        if self._error is not None:
            return {"error": self._error}
        if self._nvml is None or self._handle is None:
            if self._ctypes_device is None:
                return {"error": "nvml_not_initialized"}
            return self._ctypes_device.sample()
        sample: dict[str, Any] = {}
        try:
            utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            sample["utilization_percent"] = float(utilization.gpu)
            sample["memory_utilization_percent"] = float(utilization.memory)
        except Exception as exc:  # pragma: no cover - depends on host NVML state
            sample["utilization_error"] = f"{type(exc).__name__}: {exc}"
        try:
            sample["power_watts"] = (
                float(self._nvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
            )
        except Exception as exc:  # pragma: no cover - depends on host NVML state
            sample["power_error"] = f"{type(exc).__name__}: {exc}"
        try:
            sample["temperature_celsius"] = float(
                self._nvml.nvmlDeviceGetTemperature(self._handle, 0)
            )
        except Exception as exc:  # pragma: no cover - depends on host NVML state
            sample["temperature_error"] = f"{type(exc).__name__}: {exc}"
        return sample

    def _initialize(self) -> None:
        self._initialized = True
        try:
            nvml = cast(Any, importlib.import_module("pynvml"))
        except ModuleNotFoundError:
            self._initialize_ctypes()
            return
        try:
            nvml.nvmlInit()
            self._handle = nvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._nvml = nvml
        except Exception as exc:  # pragma: no cover - depends on host NVML state
            self._initialize_ctypes(f"{type(exc).__name__}: {exc}")

    def _initialize_ctypes(self, pynvml_error: str | None = None) -> None:
        with suppress(OSError, RuntimeError, AttributeError):
            self._ctypes_device = _CtypesNvmlDevice.open(self.device_index)
            return
        if pynvml_error is not None:
            self._error = f"pynvml failed ({pynvml_error}); ctypes_nvml_unavailable"
        else:
            self._error = "ctypes_nvml_unavailable"


@dataclass
class _ActorHeartbeatState:
    """Progress observed for one actor process incarnation."""

    pid: int | None
    summary_mtime_ns: int | None
    last_progress_at: float
    recycle_requested: bool = False


@dataclass
class _ActorCohortStallWatchdog:
    """Recycle isolated stale actors and fail closed on a full data-plane stall."""

    output_dir: Path
    timeout_seconds: float | None
    _actors: dict[int, _ActorHeartbeatState] = field(default_factory=dict)
    _inference_decisions: int | None = None
    _last_inference_progress_at: float | None = None

    def observe(
        self,
        *,
        now: float,
        actor_samples: Sequence[Mapping[str, Any]],
        inference_decisions: int | None,
        inference_request_depth: int | None,
        collection_paused: bool,
    ) -> ActorRecycleRequest | None:
        """Return isolated stale actor slots or raise on a full idle stall."""
        timeout = self.timeout_seconds
        if timeout is None or not actor_samples:
            return None
        if collection_paused:
            self._actors.clear()
            self._inference_decisions = inference_decisions
            self._last_inference_progress_at = now
            return None
        if any(sample.get("exitcode") is not None for sample in actor_samples):
            return None
        actor_count = len(actor_samples)
        for actor_index, sample in enumerate(actor_samples):
            raw_pid = sample.get("pid")
            pid = None if raw_pid is None else int(raw_pid)
            summary_path = _actor_summary_path(
                self.output_dir,
                actor_index=actor_index,
                actor_count=actor_count,
            )
            try:
                summary_mtime_ns = summary_path.stat().st_mtime_ns
            except OSError:
                summary_mtime_ns = None
            state = self._actors.get(actor_index)
            if state is None:
                self._actors[actor_index] = _ActorHeartbeatState(
                    pid=pid,
                    summary_mtime_ns=summary_mtime_ns,
                    last_progress_at=now,
                )
                continue
            if state.pid != pid:
                state.pid = pid
                state.summary_mtime_ns = summary_mtime_ns
                state.last_progress_at = now
                state.recycle_requested = False
                continue
            if (
                summary_mtime_ns is not None
                and summary_mtime_ns != state.summary_mtime_ns
            ):
                state.summary_mtime_ns = summary_mtime_ns
                state.last_progress_at = now
                state.recycle_requested = False

        if (
            inference_decisions is not None
            and self._inference_decisions != inference_decisions
        ):
            self._inference_decisions = inference_decisions
            self._last_inference_progress_at = now
        states = tuple(self._actors.get(index) for index in range(actor_count))
        if any(state is None for state in states):
            return None
        typed_states = cast(tuple[_ActorHeartbeatState, ...], states)
        stale_indices = tuple(
            index
            for index, state in enumerate(typed_states)
            if now - state.last_progress_at > timeout
        )
        if not stale_indices:
            return None

        inference_progress_at = self._last_inference_progress_at
        if (
            len(stale_indices) == actor_count
            and inference_progress_at is not None
            and now - inference_progress_at > timeout
        ):
            oldest_actor_age = max(
                now - state.last_progress_at for state in typed_states
            )
            raise RuntimeError(
                "actor cohort heartbeat stalled while inference was idle: "
                f"actors={actor_count}, oldest_actor_age={oldest_actor_age:.1f}s, "
                f"inference_age={now - inference_progress_at:.1f}s, "
                f"inference_request_depth={inference_request_depth}"
            )

        requested_indices = tuple(
            index
            for index in stale_indices
            if not typed_states[index].recycle_requested
        )
        if not requested_indices:
            return None
        for index in requested_indices:
            typed_states[index].recycle_requested = True
        return ActorRecycleRequest(actor_indices=requested_indices)


@dataclass
class _AsyncRuntimeMonitor:
    output_path: Path
    live_output_path: Path
    tensorboard_writer: TensorboardMetricWriter
    trajectory_queue: Any
    collection_gate: Any | None
    inference_request_queue: Any | None
    inference_response_queues: Sequence[Any] | None
    curriculum_request_queue: Any | None
    curriculum_response_queues: Sequence[Any] | None
    inference_process: ManagedProcess | None = None
    started_at: float = field(default_factory=time.perf_counter)
    samples: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_RUNTIME_MONITOR_TAIL_CAPACITY)
    )
    gpu_samples: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_RUNTIME_MONITOR_TAIL_CAPACITY)
    )
    live_write_interval_seconds: float = 10.0
    gpu_sample_interval_seconds: float = 1.0
    actor_heartbeat_timeout_seconds: float | None = 300.0
    _process_cpu_tracker: _ProcessCpuTracker = field(
        default_factory=_ProcessCpuTracker,
    )
    _gpu_sampler: _NvmlGpuSampler = field(default_factory=_NvmlGpuSampler)
    _gpu_sample_lock: Any = field(default_factory=threading.Lock, init=False)
    _gpu_stop_event: threading.Event = field(
        default_factory=threading.Event,
        init=False,
    )
    _gpu_thread: threading.Thread | None = field(default=None, init=False)
    _last_live_write: float = 0.0
    _sample_count: int = 0
    _gpu_sample_count: int = 0
    _max_queue_depths_seen: dict[str, int] = field(default_factory=dict)
    _max_process_rss_mb_seen: dict[str, float] = field(default_factory=dict)
    _max_process_cpu_percent_seen: dict[str, float] = field(default_factory=dict)
    _max_gpu_memory_used_mb_seen: float | None = None
    _runtime_phase_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    _runtime_previous_elapsed: float | None = None
    _runtime_previous_decisions: int | None = None
    _gpu_phase_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    _gpu_previous_elapsed: float | None = None
    _gpu_previous_decisions: int | None = None
    _actor_recycle_requests_by_actor: Counter[int] = field(
        default_factory=Counter,
    )
    _confirmed_actor_recycles_by_actor: tuple[int, ...] | None = None
    _actor_watchdog: _ActorCohortStallWatchdog = field(init=False)
    _terminal_failure: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Start the independent GPU sampler when CUDA is visible."""
        self._actor_watchdog = _ActorCohortStallWatchdog(
            output_dir=self.output_path.parent,
            timeout_seconds=self.actor_heartbeat_timeout_seconds,
        )
        if not torch.cuda.is_available():
            return
        self._gpu_thread = threading.Thread(
            target=self._gpu_sample_loop,
            name="rl-runtime-gpu-monitor",
            daemon=True,
        )
        self._gpu_thread.start()

    def close(self) -> None:
        """Stop the background GPU sampler."""
        self._gpu_stop_event.set()
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout=2.0)

    def close_tensorboard(self) -> None:
        """Close runtime TensorBoard events after the final status write."""
        self.tensorboard_writer.close()

    def record_failure(self, error: BaseException) -> None:
        """Record the terminal supervisor error before cleanup changes PIDs."""
        self._terminal_failure = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "recorded_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        }

    def record_actor_recycles(self, counts: Sequence[int]) -> None:
        """Attach the supervisor's authoritative successful recycle counts."""
        concrete = tuple(int(count) for count in counts)
        if any(count < 0 for count in concrete):
            raise ValueError("actor recycle counts must be non-negative")
        self._confirmed_actor_recycles_by_actor = concrete

    def sample(
        self,
        poll: int,
        actors: Sequence[ManagedProcess],
        learner: ManagedProcess,
    ) -> ActorRecycleRequest | None:
        """Record one async supervisor runtime sample."""
        now = time.perf_counter()
        output_dir = self.output_path.parent
        learner_status = _learner_monitor_status(output_dir)
        gpu_sample = self._latest_gpu_sample()
        actor_samples = [
            _process_monitor_sample(
                actor,
                index=index,
                cpu_tracker=self._process_cpu_tracker,
                now=now,
            )
            for index, actor in enumerate(actors)
        ]
        inference_decisions = _inference_monitor_decisions(output_dir)
        inference_request_depth = _queue_size(self.inference_request_queue)
        collection_paused = bool(
            self.collection_gate is not None and not self.collection_gate.is_set()
        )
        sample = {
            "poll": poll,
            "elapsed_seconds": now - self.started_at,
            "actors": actor_samples,
            "learner": _process_monitor_sample(
                learner,
                index=None,
                cpu_tracker=self._process_cpu_tracker,
                now=now,
            ),
            "inference": (
                None
                if self.inference_process is None
                else _process_monitor_sample(
                    self.inference_process,
                    index=None,
                    cpu_tracker=self._process_cpu_tracker,
                    now=now,
                )
            ),
            "learner_status": learner_status,
            "learner_phase": learner_status.get("phase"),
            "inference_decisions": inference_decisions,
            "collection_paused": collection_paused,
            "queues": {
                "trajectory": _queue_size(self.trajectory_queue),
                "inference_request": inference_request_depth,
                "inference_responses_total": _queue_size_total(
                    self.inference_response_queues
                ),
                "curriculum_request": _queue_size(self.curriculum_request_queue),
                "curriculum_responses_total": _queue_size_total(
                    self.curriculum_response_queues
                ),
            },
            "gpu": None if gpu_sample is None else gpu_sample.get("gpu"),
        }
        self._retain_runtime_sample(sample)
        recycle_request = self._actor_watchdog.observe(
            now=now,
            actor_samples=actor_samples,
            inference_decisions=inference_decisions,
            inference_request_depth=inference_request_depth,
            collection_paused=collection_paused,
        )
        if recycle_request is not None:
            sample["actor_recycle_request"] = list(recycle_request.actor_indices)
            self._actor_recycle_requests_by_actor.update(recycle_request.actor_indices)
        if now - self._last_live_write >= self.live_write_interval_seconds:
            self.write_status()
            self._last_live_write = now
        return recycle_request

    def write(self) -> None:
        """Persist monitor samples for long-run acceptance audits."""
        payload = {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": self.summary(),
            "samples": list(self.samples),
            "gpu_samples": self._retained_gpu_samples(),
        }
        _write_summary(self.output_path, payload)
        self.write_status()

    def write_status(self) -> None:
        """Persist compact live status while async training is still running."""
        output_dir = self.output_path.parent
        summary = self.summary()
        _write_summary(
            self.live_output_path,
            {
                "updated_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),
                "summary": summary,
            },
        )
        write_runtime_tensorboard(
            self.tensorboard_writer,
            summary=summary,
            inference=_read_optional_summary(_inference_summary_path(output_dir)),
            actor_queue=_runtime_actor_queue_summary(output_dir, summary),
            step=self._sample_count,
        )

    def summary(self) -> dict[str, Any]:
        """Return compact monitor summary for ``summary.json``."""
        with self._gpu_sample_lock:
            gpu_sample_count = self._gpu_sample_count
            gpu_phase_summary = _summarize_gpu_phase_stats(self._gpu_phase_stats)
            max_gpu_memory_used_mb = self._max_gpu_memory_used_mb_seen
        if gpu_sample_count == 0:
            gpu_phase_summary = _summarize_gpu_phase_stats(self._runtime_phase_stats)
        return {
            "samples_path": deck_records.display_path(self.output_path),
            "live_status_path": deck_records.display_path(self.live_output_path),
            "sample_count": self._sample_count,
            "retained_sample_count": len(self.samples),
            "live_write_interval_seconds": self.live_write_interval_seconds,
            "gpu_sample_count": gpu_sample_count,
            "retained_gpu_sample_count": len(self.gpu_samples),
            "last_sample": self.samples[-1] if self.samples else None,
            "terminal_failure": self._terminal_failure,
            "actor_recycle_requests": sum(
                self._actor_recycle_requests_by_actor.values()
            ),
            "actor_recycle_requests_by_actor": {
                str(actor_index): count
                for actor_index, count in sorted(
                    self._actor_recycle_requests_by_actor.items()
                )
            },
            "actor_recycles": (
                None
                if self._confirmed_actor_recycles_by_actor is None
                else sum(self._confirmed_actor_recycles_by_actor)
            ),
            "actor_recycles_by_actor": (
                None
                if self._confirmed_actor_recycles_by_actor is None
                else list(self._confirmed_actor_recycles_by_actor)
            ),
            "max_queue_depths": dict(sorted(self._max_queue_depths_seen.items())),
            "max_process_rss_mb": dict(sorted(self._max_process_rss_mb_seen.items())),
            "max_process_cpu_percent": dict(
                sorted(self._max_process_cpu_percent_seen.items())
            ),
            "max_gpu_memory_used_mb": max_gpu_memory_used_mb,
            "gpu_phase_summary": gpu_phase_summary,
        }

    def _gpu_sample_loop(self) -> None:
        while not self._gpu_stop_event.is_set():
            self._record_gpu_sample(time.perf_counter())
            self._gpu_stop_event.wait(self.gpu_sample_interval_seconds)

    def _record_gpu_sample(self, now: float) -> None:
        output_dir = self.output_path.parent
        sample = {
            "elapsed_seconds": now - self.started_at,
            "learner_status": _learner_monitor_status(output_dir),
            "inference_decisions": _inference_monitor_decisions(output_dir),
            "gpu": _gpu_runtime_sample(self._gpu_sampler),
        }
        self._retain_gpu_sample(sample)

    def _retain_gpu_sample(self, sample: dict[str, Any]) -> None:
        """Retain one bounded GPU tail sample and lifetime aggregates."""
        with self._gpu_sample_lock:
            self.gpu_samples.append(sample)
            self._gpu_sample_count += 1
            gpu_max = _max_gpu_memory_used_mb((sample,))
            if gpu_max is not None:
                self._max_gpu_memory_used_mb_seen = max(
                    self._max_gpu_memory_used_mb_seen or 0.0,
                    gpu_max,
                )
            (
                self._gpu_previous_elapsed,
                self._gpu_previous_decisions,
            ) = _accumulate_gpu_phase_sample(
                sample,
                phase_stats=self._gpu_phase_stats,
                previous_elapsed=self._gpu_previous_elapsed,
                previous_decisions=self._gpu_previous_decisions,
            )

    def _retain_runtime_sample(self, sample: dict[str, Any]) -> None:
        """Retain a bounded tail while preserving lifetime aggregates."""
        self.samples.append(sample)
        self._sample_count += 1
        _merge_int_maxima(
            self._max_queue_depths_seen,
            _max_queue_depths((sample,)),
        )
        _merge_float_maxima(
            self._max_process_rss_mb_seen,
            _max_process_rss((sample,)),
        )
        _merge_float_maxima(
            self._max_process_cpu_percent_seen,
            _max_process_cpu_percent((sample,)),
        )
        (
            self._runtime_previous_elapsed,
            self._runtime_previous_decisions,
        ) = _accumulate_gpu_phase_sample(
            sample,
            phase_stats=self._runtime_phase_stats,
            previous_elapsed=self._runtime_previous_elapsed,
            previous_decisions=self._runtime_previous_decisions,
        )

    def _retained_gpu_samples(self) -> list[dict[str, Any]]:
        with self._gpu_sample_lock:
            return list(self.gpu_samples)

    def _latest_gpu_sample(self) -> Mapping[str, Any] | None:
        with self._gpu_sample_lock:
            if self.gpu_samples:
                return dict(self.gpu_samples[-1])
        if not torch.cuda.is_available():
            return None
        self._record_gpu_sample(time.perf_counter())
        with self._gpu_sample_lock:
            return dict(self.gpu_samples[-1]) if self.gpu_samples else None


def _process_monitor_sample(
    process: ManagedProcess,
    *,
    index: int | None,
    cpu_tracker: _ProcessCpuTracker,
    now: float,
) -> dict[str, Any]:
    pid = getattr(process, "pid", None)
    pid_int = int(pid) if pid is not None else None
    return {
        "index": index,
        "pid": pid_int,
        "exitcode": process.exitcode,
        "rss_mb": _process_rss_mb(pid_int),
        "cpu_percent": cpu_tracker.sample_cpu_percent(pid_int, now=now),
    }


def _process_cpu_seconds(pid: int | None) -> float | None:
    if pid is None:
        return None
    stat_path = Path("/proc") / str(pid) / "stat"
    with suppress(OSError, ValueError, IndexError):
        content = stat_path.read_text(encoding="utf-8")
        fields = content.rsplit(")", maxsplit=1)[1].split()
        user_ticks = int(fields[11])
        system_ticks = int(fields[12])
        return float(user_ticks + system_ticks) / _clock_ticks_per_second()
    return None


@cache
def _clock_ticks_per_second() -> float:
    return float(os.sysconf("SC_CLK_TCK"))


def _process_rss_mb(pid: int | None) -> float | None:
    if pid is None:
        return None
    statm_path = Path("/proc") / str(pid) / "statm"
    with suppress(OSError, ValueError):
        fields = statm_path.read_text(encoding="utf-8").split()
        if len(fields) >= 2:
            return int(fields[1]) * float(os.sysconf("SC_PAGE_SIZE")) / 1_000_000.0
    return None


def _queue_size(queue_obj: Any | None) -> int | None:
    if queue_obj is None:
        return None
    qsize = getattr(queue_obj, "qsize", None)
    if not callable(qsize):
        return None
    with suppress(NotImplementedError, OSError, AttributeError):
        return int(qsize())
    return None


def _queue_size_total(queues: Sequence[Any] | None) -> int | None:
    if queues is None:
        return None
    total = 0
    observed = False
    for queue_obj in queues:
        size = _queue_size(queue_obj)
        if size is None:
            continue
        total += size
        observed = True
    return total if observed else None


def _gpu_memory_sample() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    with suppress(RuntimeError, AssertionError):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        total_mb = total_bytes / 1_000_000.0
        free_mb = free_bytes / 1_000_000.0
        return {
            "memory_total_mb": total_mb,
            "memory_free_mb": free_mb,
            "memory_used_mb": total_mb - free_mb,
        }
    return None


def _gpu_runtime_sample(sampler: _NvmlGpuSampler) -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        return None
    sample: dict[str, Any] = {}
    memory_sample = _gpu_memory_sample()
    if memory_sample is not None:
        sample.update(memory_sample)
    sample.update(sampler.sample())
    return sample


def _learner_monitor_status(output_dir: Path) -> dict[str, Any]:
    path = _learner_status_path(output_dir)
    status = _read_optional_summary(path)
    if status is None:
        return {}
    return {
        "phase": status.get("phase"),
        "iteration": status.get("iteration"),
        "current_policy_version": status.get("current_policy_version"),
        "published_version": (
            status.get("published_version") or status.get("publish_version")
        ),
        "age_seconds": _file_age_seconds(path),
    }


def _inference_monitor_decisions(output_dir: Path) -> int | None:
    status = _read_optional_summary(_inference_summary_path(output_dir))
    if status is None:
        return None
    decisions = status.get("decisions")
    if decisions is None:
        return None
    with suppress(TypeError, ValueError):
        return int(decisions)
    return None


def _file_age_seconds(path: Path) -> float | None:
    with suppress(OSError):
        return max(0.0, time.time() - path.stat().st_mtime)
    return None


def _max_queue_depths(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    maxima: dict[str, int] = {}
    for sample in samples:
        queues = sample.get("queues", {})
        if not isinstance(queues, Mapping):
            continue
        for name, value in queues.items():
            if value is None:
                continue
            maxima[str(name)] = max(maxima.get(str(name), 0), int(value))
    return dict(sorted(maxima.items()))


def _merge_int_maxima(
    target: dict[str, int],
    observed: Mapping[str, int],
) -> None:
    """Merge integer lifetime maxima without retaining their source samples."""
    for name, value in observed.items():
        target[name] = max(target.get(name, value), value)


def _merge_float_maxima(
    target: dict[str, float],
    observed: Mapping[str, float],
) -> None:
    """Merge float lifetime maxima without retaining their source samples."""
    for name, value in observed.items():
        target[name] = max(target.get(name, value), value)


def _max_process_rss(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    maxima: dict[str, float] = {}
    for sample in samples:
        learner = sample.get("learner")
        if isinstance(learner, Mapping):
            _update_rss_max(maxima, "learner", learner)
        inference = sample.get("inference")
        if isinstance(inference, Mapping):
            _update_rss_max(maxima, "inference", inference)
        actors = sample.get("actors", ())
        if not isinstance(actors, Sequence):
            continue
        for actor in actors:
            if not isinstance(actor, Mapping):
                continue
            index = actor.get("index")
            _update_rss_max(maxima, f"actor_{index}", actor)
    return dict(sorted(maxima.items()))


def _max_process_cpu_percent(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    maxima: dict[str, float] = {}
    for sample in samples:
        learner = sample.get("learner")
        if isinstance(learner, Mapping):
            _update_cpu_max(maxima, "learner", learner)
        inference = sample.get("inference")
        if isinstance(inference, Mapping):
            _update_cpu_max(maxima, "inference", inference)
        actors = sample.get("actors", ())
        if not isinstance(actors, Sequence):
            continue
        for actor in actors:
            if not isinstance(actor, Mapping):
                continue
            index = actor.get("index")
            _update_cpu_max(maxima, f"actor_{index}", actor)
    return dict(sorted(maxima.items()))


def _update_rss_max(
    maxima: dict[str, float],
    name: str,
    sample: Mapping[str, Any],
) -> None:
    value = sample.get("rss_mb")
    if value is None:
        return
    maxima[name] = max(maxima.get(name, 0.0), float(value))


def _update_cpu_max(
    maxima: dict[str, float],
    name: str,
    sample: Mapping[str, Any],
) -> None:
    value = sample.get("cpu_percent")
    if value is None:
        return
    maxima[name] = max(maxima.get(name, 0.0), float(value))


def _max_gpu_memory_used_mb(samples: Sequence[Mapping[str, Any]]) -> float | None:
    values: list[float] = []
    for sample in samples:
        gpu = sample.get("gpu")
        if not isinstance(gpu, Mapping):
            continue
        used = gpu.get("memory_used_mb")
        if used is not None:
            values.append(float(used))
    return max(values) if values else None


def _gpu_phase_summary(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate phase telemetry for a finite sample sequence."""
    phase_stats: dict[str, dict[str, float]] = {}
    previous_elapsed: float | None = None
    previous_decisions: int | None = None
    for sample in samples:
        previous_elapsed, previous_decisions = _accumulate_gpu_phase_sample(
            sample,
            phase_stats=phase_stats,
            previous_elapsed=previous_elapsed,
            previous_decisions=previous_decisions,
        )
    return _summarize_gpu_phase_stats(phase_stats)


def _accumulate_gpu_phase_sample(
    sample: Mapping[str, Any],
    *,
    phase_stats: dict[str, dict[str, float]],
    previous_elapsed: float | None,
    previous_decisions: int | None,
) -> tuple[float | None, int | None]:
    """Update lifetime phase aggregates from one sample."""
    phase = _runtime_sample_phase(sample)
    stats = phase_stats.setdefault(
        phase,
        {
            "samples": 0.0,
            "power_samples": 0.0,
            "powered_samples": 0.0,
            "power_watts_total": 0.0,
            "utilization_samples": 0.0,
            "utilization_percent_total": 0.0,
            "inference_seconds": 0.0,
            "inference_decisions": 0.0,
        },
    )
    stats["samples"] += 1.0
    gpu = sample.get("gpu")
    if isinstance(gpu, Mapping):
        power_watts = _optional_float(gpu.get("power_watts"))
        if power_watts is not None:
            stats["power_samples"] += 1.0
            stats["power_watts_total"] += power_watts
            if power_watts > 300.0:
                stats["powered_samples"] += 1.0
        utilization_percent = _optional_float(gpu.get("utilization_percent"))
        if utilization_percent is not None:
            stats["utilization_samples"] += 1.0
            stats["utilization_percent_total"] += utilization_percent

    elapsed = _optional_float(sample.get("elapsed_seconds"))
    decisions = _optional_int(sample.get("inference_decisions"))
    if (
        elapsed is not None
        and previous_elapsed is not None
        and elapsed > previous_elapsed
        and decisions is not None
        and previous_decisions is not None
    ):
        decision_delta = decisions - previous_decisions
        if decision_delta >= 0:
            stats["inference_seconds"] += elapsed - previous_elapsed
            stats["inference_decisions"] += float(decision_delta)
    if elapsed is not None:
        previous_elapsed = elapsed
    if decisions is not None:
        previous_decisions = decisions
    return previous_elapsed, previous_decisions


def _summarize_gpu_phase_stats(
    phase_stats: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float | int]]:
    """Render accumulated phase telemetry without requiring sample history."""

    summary: dict[str, dict[str, float | int]] = {}
    for phase, stats in sorted(phase_stats.items()):
        power_samples = stats["power_samples"]
        utilization_samples = stats["utilization_samples"]
        inference_seconds = stats["inference_seconds"]
        summary[phase] = {
            "samples": int(stats["samples"]),
            "power_samples": int(power_samples),
            "gpu_power_occupancy_fraction": (
                stats["powered_samples"] / power_samples if power_samples > 0.0 else 0.0
            ),
            "mean_power_watts": (
                stats["power_watts_total"] / power_samples
                if power_samples > 0.0
                else 0.0
            ),
            "mean_utilization_percent": (
                stats["utilization_percent_total"] / utilization_samples
                if utilization_samples > 0.0
                else 0.0
            ),
            "inference_decisions_per_second": (
                stats["inference_decisions"] / inference_seconds
                if inference_seconds > 0.0
                else 0.0
            ),
        }
    return summary


def _runtime_sample_phase(sample: Mapping[str, Any]) -> str:
    learner_status = sample.get("learner_status")
    if isinstance(learner_status, Mapping):
        phase = learner_status.get("phase")
        if phase is not None:
            return str(phase)
    phase = sample.get("learner_phase")
    if phase is not None:
        return str(phase)
    return "unknown"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    with suppress(TypeError, ValueError):
        return float(value)
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    with suppress(TypeError, ValueError):
        return int(value)
    return None


def _accumulate_stage_timings(
    stage_timings: Mapping[str, Any],
    *,
    seconds_by_stage: dict[str, float],
    counts_by_stage: dict[str, int],
) -> None:
    for raw_stage, raw_timing in stage_timings.items():
        if not isinstance(raw_stage, str) or not isinstance(raw_timing, Mapping):
            continue
        seconds_by_stage[raw_stage] = seconds_by_stage.get(raw_stage, 0.0) + float(
            raw_timing.get("seconds", 0.0)
        )
        counts_by_stage[raw_stage] = counts_by_stage.get(raw_stage, 0) + int(
            raw_timing.get("count", 0)
        )


def _stage_timing_summary(
    seconds_by_stage: Mapping[str, float],
    counts_by_stage: Mapping[str, int],
) -> dict[str, dict[str, float | int]]:
    return {
        stage: {
            "seconds": seconds,
            "count": counts_by_stage.get(stage, 0),
            "mean_ms": (
                1000.0 * seconds / float(counts_by_stage.get(stage, 0))
                if counts_by_stage.get(stage, 0) > 0
                else 0.0
            ),
        }
        for stage, seconds in sorted(seconds_by_stage.items())
    }


def _set_rollout_policy_version(policy: object, version: int) -> None:
    current = getattr(policy, "policy_version", None)
    if current is None or not callable(current):
        cast(Any, policy).policy_version = int(version)


def _actor_summary_path(
    output_path: Path,
    *,
    actor_index: int,
    actor_count: int,
) -> Path:
    if actor_count == 1:
        return output_path / "actor_summary.json"
    return output_path / f"actor_summary_{actor_index}.json"


def _write_actor_summary(
    path: Path,
    *,
    actor_index: int,
    stats: ActorLoopStats,
    curriculum_summary: Mapping[str, Any] | None,
    reanalysis_summary: Mapping[str, int] | None = None,
) -> None:
    _write_summary(
        path,
        {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "async_actor",
            "actor_index": actor_index,
            "stats": stats.__dict__,
            "curriculum": curriculum_summary,
            "reanalysis_roots": dict(reanalysis_summary or {}),
        },
    )


def _namespace_actor_trajectory(
    trajectory: GameTrajectory,
    *,
    actor_index: int,
) -> GameTrajectory:
    namespaced_game_id = f"actor-{actor_index}:{trajectory.game_id}"
    archive = (
        None
        if trajectory.archive is None
        else _namespace_archive_trajectory(
            trajectory.archive,
            game_id=namespaced_game_id,
        )
    )
    namespaced = rebind_game_trajectory_id(
        trajectory,
        game_id=namespaced_game_id,
    )
    return replace(namespaced, archive=archive)


def _async_actor_queue_trajectory(
    trajectory: GameTrajectory,
    *,
    actor_index: int,
    actor_count: int,
    reanalysis_root_queue: Any | None = None,
    reanalysis_counters: Counter[str] | None = None,
) -> GameTrajectory:
    if actor_count > 1:
        trajectory = _namespace_actor_trajectory(
            trajectory,
            actor_index=actor_index,
        )
    if reanalysis_root_queue is not None:
        for decision in trajectory.decisions:
            root = decision.reanalysis_root
            if root is None:
                continue
            if reanalysis_counters is not None:
                reanalysis_counters["eligible"] += 1
            root = rebind_reanalysis_game_id(root, game_id=trajectory.game_id)
            try:
                reanalysis_root_queue.put_nowait(root)
            except queue.Full:
                if reanalysis_counters is not None:
                    reanalysis_counters["dropped_full"] += 1
            else:
                if reanalysis_counters is not None:
                    reanalysis_counters["admitted"] += 1
    return compact_game_trajectory(trajectory)


def _publish_async_reanalysis_root(
    root: ReanalysisRoot,
    *,
    actor_index: int,
    actor_count: int,
    reanalysis_root_queue: Any,
    reanalysis_counters: Counter[str],
) -> bool:
    """Publish a sampled root at its decision boundary, before the game ends."""
    reanalysis_counters["eligible"] += 1
    if actor_count > 1:
        root = rebind_reanalysis_game_id(
            root,
            game_id=f"actor-{actor_index}:{root.student.game_id}",
        )
    try:
        reanalysis_root_queue.put_nowait(root)
    except queue.Full:
        reanalysis_counters["dropped_full"] += 1
        return False
    else:
        reanalysis_counters["admitted"] += 1
        return True


def _namespace_archive_trajectory(
    trajectory: CompletedTrajectory,
    *,
    game_id: str,
) -> CompletedTrajectory:
    rows = tuple({**dict(row), "game_id": game_id} for row in trajectory.rows)
    game_row = {**dict(trajectory.game_row), "game_id": game_id}
    return CompletedTrajectory(game_id=game_id, rows=rows, game_row=game_row)


def _resolved_frozen_state_path(config: RLTrainConfig, *, output_dir: Path) -> Path:
    return (
        config.curriculum.frozen_state_path
        or output_dir / "curriculum" / "frozen_pool_state.json"
    )


def _maybe_bootstrap_curriculum_state(
    config: RLTrainConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Atomically seed a new run from one immutable curriculum-state file."""
    configured_source = config.execution.curriculum_bootstrap_state_path
    expected_sha256 = config.execution.curriculum_bootstrap_state_sha256
    if configured_source is None or expected_sha256 is None:
        return {"status": "skipped", "reason": "not_configured"}
    target = _resolved_frozen_state_path(config, output_dir=output_dir)
    if target.exists():
        # A restarted run owns its progressed state. Never overwrite it with the
        # original seed merely because its bytes have legitimately advanced.
        read_frozen_pool_state(target)
        return {
            "status": "kept_existing",
            "path": deck_records.display_path(target),
        }
    source = deck_records.repo_path(configured_source)
    if not source.is_file():
        raise FileNotFoundError(f"curriculum bootstrap state is missing: {source}")
    if source.resolve() == target.resolve():
        raise ValueError("curriculum bootstrap source cannot equal its target")
    actual_sha256 = file_sha256(source)
    if actual_sha256 != expected_sha256:
        raise ValueError("curriculum bootstrap state SHA256 mismatch")
    read_frozen_pool_state(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f".{target.name}.bootstrap-{os.getpid()}")
    try:
        shutil.copyfile(source, pending)
        with pending.open("rb+") as handle:
            os.fsync(handle.fileno())
        if file_sha256(pending) != expected_sha256:
            raise RuntimeError("copied curriculum bootstrap state changed in transit")
        os.replace(pending, target)
        with suppress(OSError):
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        pending.unlink(missing_ok=True)
    return {
        "status": "bootstrapped",
        "path": deck_records.display_path(target),
        "source": deck_records.display_path(source),
        "sha256": expected_sha256,
    }


def _maybe_bootstrap_league_promotion_state(
    config: RLTrainConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Seed a new run from one fingerprinted promotion-role state."""
    configured_source = config.execution.league_promotion_bootstrap_state_path
    expected_sha256 = config.execution.league_promotion_bootstrap_state_sha256
    if configured_source is None or expected_sha256 is None:
        return {"status": "skipped", "reason": "not_configured"}
    if not config.frozen_league.enabled:
        raise ValueError("league promotion bootstrap requires frozen_league.enabled")
    target = _league_promotion_state_path(output_dir)
    if target.exists():
        LeaguePromotionState.model_validate_json(target.read_text(encoding="utf-8"))
        return {
            "status": "kept_existing",
            "path": deck_records.display_path(target),
        }
    source = deck_records.repo_path(configured_source)
    if not source.is_file():
        raise FileNotFoundError(
            f"league promotion bootstrap state is missing: {source}"
        )
    if source.resolve() == target.resolve():
        raise ValueError("league promotion bootstrap source cannot equal its target")
    if file_sha256(source) != expected_sha256:
        raise ValueError("league promotion bootstrap state SHA256 mismatch")
    LeaguePromotionState.model_validate_json(source.read_text(encoding="utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f".{target.name}.bootstrap-{os.getpid()}")
    try:
        shutil.copyfile(source, pending)
        with pending.open("rb+") as handle:
            os.fsync(handle.fileno())
        if file_sha256(pending) != expected_sha256:
            raise RuntimeError(
                "copied league promotion bootstrap state changed in transit"
            )
        os.replace(pending, target)
        with suppress(OSError):
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        pending.unlink(missing_ok=True)
    return {
        "status": "bootstrapped",
        "path": deck_records.display_path(target),
        "source": deck_records.display_path(source),
        "sha256": expected_sha256,
    }


def _frozen_pool_addition_dir(output_dir: Path) -> Path:
    return output_dir / "curriculum" / "frozen_pool_additions"


def _frozen_league_candidate_dir(output_dir: Path) -> Path:
    return output_dir / "curriculum" / "league_candidates"


def _league_promotion_state_path(output_dir: Path) -> Path:
    return output_dir / "curriculum" / "league_promotion_state.json"


def _maybe_seed_curriculum_anchor(
    config: RLTrainConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Seed configured pinned frozen policies before any rollout process starts."""
    if config.collection.opponent_mode != "curriculum":
        return {"status": "skipped", "reason": "opponent_mode_not_curriculum"}
    state_path = _resolved_frozen_state_path(config, output_dir=output_dir)
    state = read_frozen_pool_state(state_path)
    seeds: list[tuple[str, Path]] = []
    if (
        config.curriculum.seed_anchor_in_frozen_pool
        and config.anchor_checkpoint_path is not None
    ):
        seeds.append(
            (
                config.curriculum.anchor_opponent_id,
                config.anchor_checkpoint_path,
            )
        )
    fixed_seeds = tuple(config.curriculum.fixed_frozen_bundles) + tuple(
        config.curriculum.fixed_frozen_policies
    )
    for seed in fixed_seeds:
        if seed.checkpoint_size_bytes is not None:
            fingerprint = fingerprint_checkpoint(seed.checkpoint_path)
            if fingerprint.size_bytes != seed.checkpoint_size_bytes:
                raise ValueError(
                    "fixed frozen checkpoint size mismatch: "
                    f"{seed.checkpoint_path} "
                    f"expected={seed.checkpoint_size_bytes} "
                    f"actual={fingerprint.size_bytes}"
                )
            if fingerprint.sha256 != seed.checkpoint_sha256:
                raise ValueError(
                    "fixed frozen checkpoint SHA256 mismatch: "
                    f"{seed.checkpoint_path} "
                    f"expected={seed.checkpoint_sha256} "
                    f"actual={fingerprint.sha256}"
                )
        seeds.append((seed.opponent_id, seed.checkpoint_path))
    if not seeds:
        return {"status": "skipped", "reason": "no_frozen_pool_seeds"}

    duplicate_ids = [opponent_id for opponent_id, _ in seeds]
    if len(duplicate_ids) != len(set(duplicate_ids)):
        raise ValueError("anchor and fixed frozen opponent_id values collide")

    updated = state
    for opponent_id, checkpoint_path in seeds:
        updated = add_frozen_pool_member(
            updated,
            opponent_id=opponent_id,
            checkpoint_path=checkpoint_path,
            winrate_ema=0.5,
            pinned=True,
            capacity=config.curriculum.frozen_capacity,
        )
    if updated == state:
        if len(seeds) == 1 and not fixed_seeds:
            reason = "anchor_already_seeded"
        elif config.curriculum.fixed_frozen_policies:
            reason = "fixed_frozen_opponents_already_seeded"
        else:
            reason = "fixed_frozen_bundles_already_seeded"
        return {
            "status": "skipped",
            "reason": reason,
            "state_path": deck_records.display_path(state_path),
        }
    write_frozen_pool_state(state_path, updated)
    return {
        "status": "updated",
        "state_path": deck_records.display_path(state_path),
        "opponent_ids": [opponent_id for opponent_id, _ in seeds],
        "members": len(updated.members),
    }


def _curriculum_manager(
    config: RLTrainConfig,
    *,
    output_dir: Path,
) -> _CurriculumAssignmentManager | None:
    if config.collection.opponent_mode == "self_play":
        return None
    state_path = _resolved_frozen_state_path(config, output_dir=output_dir)
    sampler = CurriculumSampler(
        config.curriculum.model_copy(update={"frozen_state_path": state_path}),
        rng=random.Random(config.seed + 31),
    )
    return _CurriculumAssignmentManager(
        sampler,
        state_path=state_path,
        additions_dir=_frozen_pool_addition_dir(output_dir),
    )


def _async_actor_curriculum_manager(
    config: RLTrainConfig,
    *,
    output_dir: Path,
    actor_index: int,
    actor_incarnation: int,
    curriculum_request_queue: Any | None,
    curriculum_response_queue: Any | None,
) -> _CurriculumAssignmentManager | _RemoteCurriculumAssignmentManager | None:
    if config.collection.opponent_mode == "self_play":
        return None
    state_path = _resolved_frozen_state_path(config, output_dir=output_dir)
    if curriculum_request_queue is None and curriculum_response_queue is None:
        return _curriculum_manager(config, output_dir=output_dir)
    if curriculum_request_queue is None or curriculum_response_queue is None:
        raise ValueError("central curriculum requires request and response queues")
    return _RemoteCurriculumAssignmentManager(
        config,
        state_path=state_path,
        actor_index=actor_index,
        actor_incarnation=actor_incarnation,
        request_queue=curriculum_request_queue,
        response_queue=curriculum_response_queue,
    )


def _remote_inference_routes(
    config: RLTrainConfig,
    *,
    output_dir: Path,
    actor_index: int,
    actor_incarnation: int,
    inference_request_queue: Any | None,
    inference_response_queue: Any | None,
    remote_client_state: RemoteInferenceClientState | None,
) -> _RemoteInferenceRoutes:
    if inference_request_queue is None or inference_response_queue is None:
        raise ValueError("inference queues are required for remote routing")
    if remote_client_state is None:
        raise ValueError("remote client state is required for remote routing")
    return _RemoteInferenceRoutes(
        actor_id=f"actor-{actor_index}",
        actor_incarnation=actor_incarnation,
        request_queue=inference_request_queue,
        response_queue=inference_response_queue,
        client_config=_inference_client_config(config),
        client_state=remote_client_state,
        request_id_base=_remote_request_id_base(actor_index),
        recurrent_model_config=(
            config.model if config.model.recurrent is not None else None
        ),
        frozen_state_path=_resolved_frozen_state_path(config, output_dir=output_dir),
    )


def _inference_client_config(config: RLTrainConfig) -> InferenceClientConfig:
    return InferenceClientConfig(
        max_inflight_batch_decisions=(
            config.execution.inference_client_max_inflight_batch_decisions
        )
    )


def _remote_request_id_base(actor_index: int) -> int:
    return (time.time_ns() & ((1 << 62) - 1)) + actor_index


def _drain_stale_actor_response_queues(
    *,
    inference_response_queue: Any | None,
    teacher_inference_response_queue: Any | None = None,
    curriculum_response_queue: Any | None,
) -> None:
    _drain_queue_nowait(inference_response_queue)
    _drain_queue_nowait(teacher_inference_response_queue)
    _drain_queue_nowait(curriculum_response_queue)


def _drain_queue_nowait(queue_obj: Any | None) -> None:
    if queue_obj is None:
        return
    while True:
        try:
            queue_obj.get_nowait()
        except queue.Empty:
            return
        except (EOFError, OSError, AttributeError):
            return


def _training_rollout_actors(
    config: RLTrainConfig,
    *,
    candidate_policy: RolloutPolicy,
    curriculum_manager: (
        _CurriculumAssignmentManager | _RemoteCurriculumAssignmentManager | None
    ),
    remote_inference_routes: _RemoteInferenceRoutes | None = None,
) -> _TrainingRolloutActors:
    if curriculum_manager is None:
        return _TrainingRolloutActors(
            actors=RolloutActors(mode="self_play", candidate_policy=candidate_policy)
        )

    frozen_update: FrozenPolicyPoolUpdate | None = None
    frozen_policies: Mapping[str, RolloutPolicy] = {}
    if config.curriculum.mix.frozen > 0.0 and curriculum_manager.frozen_members:
        if remote_inference_routes is None:
            frozen_pool = FrozenPolicyPool(
                model_config=config.model,
                config=config.frozen_policy_pool,
            )
            frozen_update = frozen_pool.sync(curriculum_manager.frozen_members)
            frozen_policies = frozen_pool.policies
        else:
            frozen_ids = tuple(
                member.opponent_id for member in curriculum_manager.frozen_members
            )
            frozen_update = FrozenPolicyPoolUpdate(
                loaded=frozen_ids,
                unloaded=(),
                kept=(),
            )
            frozen_policies = _RemoteFrozenPolicyMapping(remote_inference_routes)

    scripted_agents: Mapping[str, BattleAgent] = {}
    scripted_agent_factories: Mapping[str, Callable[[], BattleAgent]] = {}
    scripted_tiers: Mapping[str, int] = {}
    if config.curriculum.mix.scripted > 0.0:
        scripted_agents = _scripted_agents(curriculum_manager, seed=config.seed)
        scripted_agent_factories = _scripted_agent_factories(
            curriculum_manager,
            seed=config.seed,
        )
        scripted_tiers = _scripted_tiers(curriculum_manager)

    return _TrainingRolloutActors(
        actors=RolloutActors(
            mode="curriculum",
            candidate_policy=candidate_policy,
            frozen_policies=frozen_policies,
            scripted_agents=scripted_agents,
            scripted_agent_factories=scripted_agent_factories,
            scripted_tiers=scripted_tiers,
            curriculum_assignments=curriculum_manager.assignments,
        ),
        frozen_update=frozen_update,
    )


def _scripted_agents(
    curriculum_manager: _CurriculumAssignmentManager
    | _RemoteCurriculumAssignmentManager,
    *,
    seed: int,
) -> Mapping[str, BattleAgent]:
    return {
        spec.name: build_opponent(spec, seed=seed + 10_000 + index)
        for index, spec in enumerate(curriculum_manager.sampler.scripted_specs)
        if spec.vector_safe
    }


def _scripted_agent_factories(
    curriculum_manager: _CurriculumAssignmentManager
    | _RemoteCurriculumAssignmentManager,
    *,
    seed: int,
) -> Mapping[str, Callable[[], BattleAgent]]:
    """Return per-game factories for stateful (non-vector-safe) opponents.

    Public notebook opponents keep module-level per-game state, so the
    interleaved vectorized rollout must build one instance per game instead of
    sharing a single agent across concurrent games.
    """

    def _factory(spec: OpponentSpec, spec_seed: int) -> Callable[[], BattleAgent]:
        return lambda: build_opponent(spec, seed=spec_seed)

    return {
        spec.name: _factory(spec, seed + 20_000 + index)
        for index, spec in enumerate(curriculum_manager.sampler.scripted_specs)
        if not spec.vector_safe
    }


def _scripted_tiers(
    curriculum_manager: _CurriculumAssignmentManager
    | _RemoteCurriculumAssignmentManager,
) -> Mapping[str, int]:
    return {
        spec.name: int(spec.tier) for spec in curriculum_manager.sampler.scripted_specs
    }


def _live_games(pool: VectorPoolLike) -> tuple[VectorGame, ...]:
    live_games = getattr(pool, "live_games", None)
    if live_games is not None:
        return tuple(cast(Sequence[VectorGame], live_games))
    return tuple(cast(Sequence[VectorGame], pool.pending()))


def _drain_async_iteration_trajectories(
    trajectory_queue: Any,
    *,
    target_decisions: int,
    get_timeout_seconds: float,
) -> tuple[GameTrajectory, ...]:
    trajectories: list[GameTrajectory] = []
    while _decision_count(trajectories) < target_decisions:
        try:
            trajectory = trajectory_queue.get(timeout=get_timeout_seconds)
        except queue.Empty:
            continue
        if not isinstance(trajectory, GameTrajectory):
            raise TypeError("async learner queue yielded a non-trajectory item")
        trajectories.append(trajectory)
    return tuple(trajectories)


def _take_async_learner_window(
    drainer: _AsyncTrajectoryWindowDrainer,
    *,
    target_decisions: int,
    current_policy_version: int,
    max_staleness: int,
    staleness_scope: Literal["decision", "seat_trajectory"],
    final_window: bool,
    progress_callback: Callable[[_TrajectoryDrainerSnapshot], None] | None = None,
    staleness_refill_status: dict[str, int] | None = None,
) -> tuple[GameTrajectory, ...]:
    """Take complete trajectories until staleness leaves one full window.

    Each raw chunk is removed from the bounded drainer before the next chunk is
    awaited. This lets actors refill a buffer whose preceding contents were all
    stale, while keeping the learner version fixed for the entire accumulated
    window. Stale raw trajectories remain in the accumulated evidence for the
    archive and other generic trajectory consumers; formal PPO assembly applies
    the configured exclusion once to the final combined window.
    """
    if target_decisions <= 0:
        raise ValueError("target_decisions must be positive")
    trajectories: list[GameTrajectory] = []
    remaining_decisions = target_decisions
    chunk_count = 0
    if staleness_refill_status is not None:
        staleness_refill_status.clear()
        staleness_refill_status.update(
            {
                "chunks": 0,
                "accumulated_raw_decisions": 0,
                "stale_decisions": 0,
                "retained_budget_decisions": 0,
                "remaining_decisions": target_decisions,
                "topup_chunks": 0,
            }
        )
    while remaining_decisions > 0:
        chunk = drainer.take_window(
            target_decisions=remaining_decisions,
            retain_boundary=True,
            progress_callback=progress_callback,
        )
        trajectories.extend(chunk)
        chunk_count += 1
        staleness = summarize_learner_window_staleness(
            trajectories,
            current_policy_version=current_policy_version,
            max_staleness=max_staleness,
            staleness_scope=staleness_scope,
        )
        remaining_decisions = max(
            0,
            target_decisions - staleness.retained_budget_decisions,
        )
        if staleness_refill_status is not None:
            staleness_refill_status.update(
                {
                    "chunks": chunk_count,
                    "accumulated_raw_decisions": staleness.total_decisions,
                    "stale_decisions": staleness.stale_decisions,
                    "retained_budget_decisions": (staleness.retained_budget_decisions),
                    "remaining_decisions": remaining_decisions,
                    "topup_chunks": max(0, chunk_count - 1),
                }
            )
        if progress_callback is not None:
            progress_callback(drainer.snapshot())
    if final_window:
        trajectories.extend(drainer.finish_and_flush())
    return tuple(trajectories)


@dataclass(frozen=True)
class _TrajectoryDrainerSnapshot:
    buffered_trajectories: int
    buffered_decisions: int
    drained_trajectories: int
    drained_decisions: int
    windowed_trajectories: int
    windowed_decisions: int
    boundary_deferrals: int
    budget_overshoot_decisions: int


class _AsyncTrajectoryWindowDrainer:
    """Background learner-side drain thread for rolling trajectory windows."""

    def __init__(
        self,
        trajectory_queue: Any,
        *,
        get_timeout_seconds: float,
        max_buffer_decisions: int,
        trajectory_observer: Callable[[GameTrajectory], None] | None = None,
        wait_health_check: Callable[[], None] | None = None,
        collection_gate: Any | None = None,
        high_watermark_decisions: int | None = None,
        resume_watermark_decisions: int | None = None,
    ) -> None:
        if collection_gate is None and (
            high_watermark_decisions is not None
            or resume_watermark_decisions is not None
        ):
            raise ValueError("collection watermarks require a collection gate")
        if collection_gate is not None and (
            high_watermark_decisions is None or resume_watermark_decisions is None
        ):
            raise ValueError("collection gate requires both decision watermarks")
        if (
            high_watermark_decisions is not None
            and resume_watermark_decisions is not None
            and resume_watermark_decisions >= high_watermark_decisions
        ):
            raise ValueError("collection resume watermark must be below high watermark")
        self._queue = trajectory_queue
        self._queue_poll_seconds = min(get_timeout_seconds, 0.1)
        self._max_buffer_decisions = max_buffer_decisions
        self._trajectory_observer = trajectory_observer
        self._wait_health_check = wait_health_check
        self._collection_gate = collection_gate
        self._high_watermark_decisions = high_watermark_decisions
        self._resume_watermark_decisions = resume_watermark_decisions
        self._condition = threading.Condition()
        self._buffer: deque[GameTrajectory] = deque()
        self._buffered_decisions = 0
        self._drained_trajectories = 0
        self._drained_decisions = 0
        self._windowed_trajectories = 0
        self._windowed_decisions = 0
        self._boundary_deferrals = 0
        self._budget_overshoot_decisions = 0
        self._stopping = False
        self._exception: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="async-trajectory-drainer",
            daemon=True,
        )

    def start(self) -> None:
        """Start background queue draining."""
        self._update_collection_gate_unlocked()
        self._thread.start()

    def close(self) -> None:
        """Stop the drainer and wait briefly for its thread."""
        with self._condition:
            self._stopping = True
            if self._collection_gate is not None:
                # Shutdown must not let actors start another engine/inference
                # step while learner-owned inference and durable writers close.
                self._collection_gate.clear()
            self._condition.notify_all()
        self._thread.join(timeout=max(1.0, self._queue_poll_seconds + 0.5))

    def take_window(
        self,
        *,
        target_decisions: int,
        retain_boundary: bool = False,
        progress_callback: Callable[[_TrajectoryDrainerSnapshot], None] | None = None,
        progress_interval_seconds: float = 1.0,
    ) -> tuple[GameTrajectory, ...]:
        """Take a complete-trajectory window, retaining a crossing game."""
        if target_decisions <= 0:
            raise ValueError("target_decisions must be positive")
        self._wait_for_target_decisions(
            target_decisions,
            progress_callback=progress_callback,
            progress_interval_seconds=progress_interval_seconds,
        )
        with self._condition:
            self._raise_if_failed()
            trajectories, deferred = _partition_trajectory_window(
                tuple(self._buffer),
                decision_budget=target_decisions,
                retain_boundary=retain_boundary,
            )
            if not trajectories:
                raise RuntimeError("trajectory drainer produced an empty window")
            for _trajectory in trajectories:
                self._buffer.popleft()
            decisions = _decision_count(trajectories)
            self._buffered_decisions -= decisions
            self._windowed_trajectories += len(trajectories)
            self._windowed_decisions += decisions
            self._budget_overshoot_decisions += max(
                0,
                decisions - target_decisions,
            )
            if deferred and decisions < target_decisions:
                self._boundary_deferrals += 1
            self._update_collection_gate_unlocked()
            self._condition.notify_all()
            return trajectories

    def take_final_window(
        self,
        *,
        target_decisions: int,
        progress_callback: Callable[[_TrajectoryDrainerSnapshot], None] | None = None,
        progress_interval_seconds: float = 1.0,
    ) -> tuple[GameTrajectory, ...]:
        """Freeze intake at a bounded cutoff and flush all completed games.

        The final learner update first waits for its normal decision target.
        It then stops the background consumer instead of waiting for a
        continuously produced queue to become empty. Once the consumer has
        stopped, every trajectory it accepted is returned whole in one final
        window.
        """
        if target_decisions <= 0:
            raise ValueError("target_decisions must be positive")
        self._wait_for_target_decisions(
            target_decisions,
            progress_callback=progress_callback,
            progress_interval_seconds=progress_interval_seconds,
        )
        return self._finish_and_flush(
            target_decisions=target_decisions,
            require_nonempty=True,
        )

    def finish_and_flush(self) -> tuple[GameTrajectory, ...]:
        """Atomically stop intake and return the already accepted suffix."""
        return self._finish_and_flush(
            target_decisions=0,
            require_nonempty=False,
        )

    def _finish_and_flush(
        self,
        *,
        target_decisions: int,
        require_nonempty: bool,
    ) -> tuple[GameTrajectory, ...]:
        with self._condition:
            self._raise_if_failed()
            self._stopping = True
            if self._collection_gate is not None:
                self._collection_gate.clear()
            self._condition.notify_all()

        join_timeout_seconds = max(1.0, self._queue_poll_seconds + 0.5)
        self._thread.join(timeout=join_timeout_seconds)
        if self._thread.is_alive():
            raise RuntimeError("async trajectory drainer did not stop for final flush")

        with self._condition:
            self._raise_if_failed()
            trajectories = tuple(self._buffer)
            if require_nonempty and not trajectories:
                raise RuntimeError("trajectory drainer produced an empty final window")
            self._buffer.clear()
            decisions = self._buffered_decisions
            self._buffered_decisions = 0
            self._windowed_trajectories += len(trajectories)
            self._windowed_decisions += decisions
            self._budget_overshoot_decisions += max(
                0,
                decisions - target_decisions,
            )
            self._condition.notify_all()
            return trajectories

    def snapshot(self) -> _TrajectoryDrainerSnapshot:
        """Return a thread-safe drain/buffer snapshot."""
        with self._condition:
            return self._snapshot_unlocked()

    def _wait_for_target_decisions(
        self,
        target_decisions: int,
        *,
        progress_callback: Callable[[_TrajectoryDrainerSnapshot], None] | None,
        progress_interval_seconds: float,
    ) -> None:
        if progress_interval_seconds <= 0.0:
            raise ValueError("progress_interval_seconds must be positive")
        next_progress_at = time.monotonic() + progress_interval_seconds
        while True:
            if self._wait_health_check is not None:
                self._wait_health_check()
            progress_snapshot: _TrajectoryDrainerSnapshot | None = None
            with self._condition:
                self._raise_if_failed()
                if self._buffered_decisions >= target_decisions:
                    return
                wait_seconds = 0.1
                if progress_callback is not None:
                    now = time.monotonic()
                    until_progress = next_progress_at - now
                    if until_progress <= 0.0:
                        progress_snapshot = self._snapshot_unlocked()
                        next_progress_at = now + progress_interval_seconds
                    else:
                        wait_seconds = min(wait_seconds, until_progress)
                if progress_snapshot is None:
                    self._condition.wait(timeout=wait_seconds)
            if progress_snapshot is not None and progress_callback is not None:
                progress_callback(progress_snapshot)

    def _snapshot_unlocked(self) -> _TrajectoryDrainerSnapshot:
        return _TrajectoryDrainerSnapshot(
            buffered_trajectories=len(self._buffer),
            buffered_decisions=self._buffered_decisions,
            drained_trajectories=self._drained_trajectories,
            drained_decisions=self._drained_decisions,
            windowed_trajectories=self._windowed_trajectories,
            windowed_decisions=self._windowed_decisions,
            boundary_deferrals=self._boundary_deferrals,
            budget_overshoot_decisions=self._budget_overshoot_decisions,
        )

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._stopping:
                    return
                while (
                    not self._stopping
                    and self._buffered_decisions >= self._max_buffer_decisions
                ):
                    self._update_collection_gate_unlocked()
                    self._condition.wait(timeout=0.1)
                if self._stopping:
                    return
            try:
                trajectory = self._queue.get(timeout=self._queue_poll_seconds)
            except queue.Empty:
                continue
            except BaseException as exc:
                self._record_exception(exc)
                return
            if not isinstance(trajectory, GameTrajectory):
                self._record_exception(
                    TypeError("async learner queue yielded a non-trajectory item")
                )
                return
            if self._trajectory_observer is not None:
                try:
                    self._trajectory_observer(trajectory)
                except BaseException as exc:
                    self._record_exception(exc)
                    return
            decisions = trajectory.decision_count
            with self._condition:
                self._buffer.append(trajectory)
                self._buffered_decisions += decisions
                self._drained_trajectories += 1
                self._drained_decisions += decisions
                self._update_collection_gate_unlocked()
                self._condition.notify_all()

    def _update_collection_gate_unlocked(self) -> None:
        """Apply hysteretic collection credits from the buffered decision count."""
        gate = self._collection_gate
        high = self._high_watermark_decisions
        resume = self._resume_watermark_decisions
        if gate is None or high is None or resume is None:
            return
        if self._buffered_decisions >= high:
            gate.clear()
        elif self._buffered_decisions <= resume:
            gate.set()

    def _record_exception(self, exc: BaseException) -> None:
        with self._condition:
            self._exception = exc
            self._condition.notify_all()

    def _raise_if_failed(self) -> None:
        if self._exception is not None:
            raise RuntimeError("async trajectory drainer failed") from self._exception


def _local_trajectory_worker_id(trajectory: GameTrajectory) -> str:
    """Recover the local actor identity from namespaced trajectory IDs."""
    prefix, separator, _suffix = trajectory.game_id.partition(":")
    if separator and prefix.startswith("actor-"):
        return prefix
    return "local"


def _async_drainer_status(
    drainer: _AsyncTrajectoryWindowDrainer,
) -> dict[str, int]:
    return _async_drainer_snapshot_status(drainer.snapshot())


def _async_drainer_snapshot_status(
    snapshot: _TrajectoryDrainerSnapshot,
) -> dict[str, int]:
    return {
        "buffered_trajectories": snapshot.buffered_trajectories,
        "buffered_decisions": snapshot.buffered_decisions,
        "drained_trajectories": snapshot.drained_trajectories,
        "drained_decisions": snapshot.drained_decisions,
        "windowed_trajectories": snapshot.windowed_trajectories,
        "windowed_decisions": snapshot.windowed_decisions,
        "boundary_deferrals": snapshot.boundary_deferrals,
        "budget_overshoot_decisions": snapshot.budget_overshoot_decisions,
    }


def _write_async_draining_status(
    config: RLTrainConfig,
    output_path: Path,
    snapshot: _TrajectoryDrainerSnapshot,
    *,
    started_at: float,
    iteration: int,
    current_policy_version: int,
    publish_version: int,
    target_decisions: int,
    configured_iteration_decisions: int,
    staleness_refill_status: Mapping[str, int] | None = None,
) -> None:
    _maybe_write_learner_status(
        config,
        output_path,
        {
            "phase": "draining_trajectories",
            "elapsed_seconds": time.perf_counter() - started_at,
            "iteration": iteration,
            "current_policy_version": current_policy_version,
            "publish_version": publish_version,
            "target_decisions": target_decisions,
            "configured_iteration_decisions": configured_iteration_decisions,
            "staleness_refill": dict(staleness_refill_status or {}),
            "drainer": _async_drainer_snapshot_status(snapshot),
        },
    )


def _async_drainer_buffer_limit(config: RLTrainConfig) -> int:
    configured = config.learner.drain_buffer_decisions
    if configured is not None:
        return max(configured, _learner_window_decision_budget(config))
    return max(
        config.collection.iteration_decisions * 2,
        config.collection.iteration_decisions + config.collection.effective_batch_size,
    )


def _collect_sync_iteration(
    config: RLTrainConfig,
    *,
    factory: RLTrainPoolFactory,
    deck_pair: DeckPair,
    policy: RolloutPolicy,
    device: torch.device,
    policy_version: int,
    output_dir: Path,
    initial_trajectories: Sequence[GameTrajectory] = (),
    target_decisions: int | None = None,
    retain_boundary: bool = False,
) -> _SyncIteration:
    decision_target = target_decisions or config.collection.iteration_decisions
    if decision_target <= 0:
        raise ValueError("target_decisions must be positive")
    if _decision_count(initial_trajectories) >= decision_target:
        window, deferred = _partition_trajectory_window(
            initial_trajectories,
            decision_budget=decision_target,
            retain_boundary=retain_boundary,
        )
        return _SyncIteration(
            iteration=0,
            policy_version=policy_version,
            trajectories=window,
            deferred_trajectories=deferred,
        )
    recorder = TensorTrajectoryRecorder()
    curriculum_manager = _curriculum_manager(config, output_dir=output_dir)
    deck_pair_sampler = (
        curriculum_manager.sample_deck_pair
        if curriculum_manager is not None
        else lambda: deck_pair
    )
    counters = _MutableSyncCounters()
    engine_teacher_producer = _training_engine_teacher_producer(
        config,
        policy=policy,
        device=device,
        actor_index=0,
    )

    teacher_lifecycle = (
        nullcontext()
        if engine_teacher_producer is None
        else closing(engine_teacher_producer)
    )
    planner_lifecycle: AbstractContextManager[PlannerBehaviorRuntime | None] = (
        nullcontext(None)
        if config.planner is None
        else create_planner_behavior_runtime(
            runtime_config=config.planner,
            sampler_config=config.rollout_probe.sampler,
            belief_config=config.rollout_belief,
            stochastic_seed=config.seed,
        )
    )
    with (
        teacher_lifecycle,
        planner_lifecycle as planner_runtime,
        factory(config.collection.num_concurrent_games, deck_pair_sampler) as pool,
    ):
        if curriculum_manager is not None:
            curriculum_manager.sync_live_games(_live_games(pool))
        actor_setup = _training_rollout_actors(
            config,
            candidate_policy=policy,
            curriculum_manager=curriculum_manager,
        )
        rollout_recorder = (
            _CurriculumRolloutRecorder(recorder, curriculum_manager)
            if curriculum_manager is not None
            else recorder
        )
        stepper = RolloutStepper(
            pool=pool,
            actors=actor_setup.actors,
            recorder=rollout_recorder,
            temperature=config.collection.sampling_temperature,
            device=device,
            probe_config=config.rollout_probe,
            belief_config=config.rollout_belief,
            factual_config=config.factual,
            macro_credit_config=config.macro_credit,
            engine_teacher_producer=engine_teacher_producer,
            planner_behavior_service=(
                None if planner_runtime is None else planner_runtime.service
            ),
            seed=config.seed,
            record_public_event_deltas=config.model.recurrent is not None,
        )
        trajectories = list(initial_trajectories)
        while counters.collect_iterations < config.collection.max_collect_iterations:
            if _decision_count(trajectories) >= decision_target:
                break
            stats = stepper.step()
            counters.collect_iterations += 1
            counters.policy_actions += stats.policy_actions
            counters.forced_actions += stats.forced_actions
            counters.scripted_actions += stats.scripted_actions
            counters.recorded_decisions += stats.recorded_decisions
            counters.finished_games += stats.finished_games
            if curriculum_manager is not None:
                curriculum_manager.sync_live_games(_live_games(pool))
            trajectories.extend(recorder.pop_completed())
            if stats.pending_games == 0 and stats.finished_games == 0:
                break
        trajectories.extend(recorder.pop_completed())
    curriculum_summary = None
    if curriculum_manager is not None:
        curriculum_manager.save_state()
        curriculum_summary = curriculum_manager.summary(actor_setup.frozen_update)
    window, deferred = _partition_trajectory_window(
        trajectories,
        decision_budget=decision_target,
        retain_boundary=retain_boundary,
    )
    return _SyncIteration(
        iteration=0,
        policy_version=policy_version,
        trajectories=window,
        deferred_trajectories=deferred,
        counters=counters,
        curriculum_summary=curriculum_summary,
        rollout_features=dict(stepper.feature_summary),
    )


def _sync_rollout_policy(
    config: RLTrainConfig,
    *,
    model: AgentPolicyValueNet,
) -> RolloutPolicy:
    if config.collection.policy_kind == "min_count":
        return MinCountRolloutPolicy()
    model.eval()
    return ModelRolloutPolicy(
        model,
        policy_version=_initial_policy_version(config),
        autocast=config.ppo.autocast,
        planner_context_capacity=(
            0 if config.planner is None else config.planner.contexts.retained_root_rows
        ),
    )


def _load_training_model(
    config: RLTrainConfig,
    *,
    device: torch.device,
) -> AgentPolicyValueNet:
    checkpoint_path = config.checkpoint_path
    checkpoint = _load_checkpoint(checkpoint_path)
    model_config = _training_model_config(config, checkpoint=checkpoint)
    model = build_agent_policy_value_net(model_config).to(device)
    if checkpoint is not None:
        state_dict = _checkpoint_state_dict(checkpoint)
        source_config = _checkpoint_model_config(checkpoint)
        if config.registry_transition is not None:
            if source_config is None or checkpoint_path is None:
                raise RuntimeError(
                    "registry transition requires a checkpoint model configuration"
                )
            resolved_checkpoint_path = deck_records.repo_path(checkpoint_path)
            actual_checkpoint_sha256 = file_sha256(resolved_checkpoint_path)
            if (
                actual_checkpoint_sha256
                != config.registry_transition.source_checkpoint_sha256
            ):
                raise ValueError(
                    "registry transition source checkpoint SHA256 mismatch"
                )
            transition_plan = build_deck_registry_transition_plan(
                source_config,
                model_config,
                config.registry_transition,
            )
            weight_summary = migrate_deck_registry_weights(
                model,
                state_dict,
                transition_plan,
                source_config=source_config,
            )
            cast(Any, model)._deck_registry_transition_plan = transition_plan
            cast(Any, model)._deck_registry_transition_summary = {
                **transition_plan.summary(),
                "weights": weight_summary,
            }
            return model
        conditioning = model_config.deck_conditioning
        if (
            config.resume.mode == "warm_start"
            and conditioning is not None
            and conditioning.enabled
        ):
            load_agent_policy_value_state_dict(
                model,
                cast(Mapping[str, torch.Tensor], state_dict),
                source_config=source_config,
                additional_allowed_missing=(
                    tuple(_LEGACY_TRAINING_CHECKPOINT_MISSING_KEYS)
                    if source_config is None
                    or not _deck_conditioning_enabled(source_config)
                    else ()
                ),
            )
            _attach_checkpoint_transition_lineage(model, checkpoint)
            return model
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing = (
            set(_LEGACY_TRAINING_CHECKPOINT_MISSING_KEYS)
            if config.resume.mode != "resume"
            or config.resume.allow_legacy_unbound_state
            else set()
        )
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        disallowed_missing = sorted(missing - allowed_missing)
        if disallowed_missing or unexpected:
            raise RuntimeError(
                "checkpoint state dict is incompatible with training model: "
                f"missing={disallowed_missing}, unexpected={sorted(unexpected)}"
            )
        _attach_checkpoint_transition_lineage(model, checkpoint)
    return model


def _resolve_training_model_config(config: RLTrainConfig) -> RLTrainConfig:
    """Return a run config carrying the portable model architecture actually used."""
    checkpoint = _load_checkpoint(config.checkpoint_path)
    model_config = _training_model_config(config, checkpoint=checkpoint)
    return config.model_copy(update={"model": model_config})


def _training_model_config(
    config: RLTrainConfig,
    *,
    checkpoint: Any | None,
) -> AgentNetworkConfig:
    """Select resolved target or checkpoint config according to resume semantics."""
    target = _resolved_target_model_config(config)
    source = _checkpoint_model_config(checkpoint)
    if checkpoint is None:
        return target
    if config.resume.mode == "warm_start":
        _validate_routed_expert_warm_start_boundary(
            source=source,
            target=target,
            has_registry_transition=config.registry_transition is not None,
        )
        return target
    if source is None:
        return target
    if config.resume.mode == "resume" and source != target:
        raise RuntimeError(
            "exact resume requires identical target and checkpoint model configs"
        )
    if config.resume.mode == "legacy_resume" and _deck_conditioning_enabled(
        source
    ) != _deck_conditioning_enabled(target):
        raise RuntimeError("legacy_resume cannot change deck conditioning architecture")
    return source


def _validate_routed_expert_warm_start_boundary(
    *,
    source: AgentNetworkConfig | None,
    target: AgentNetworkConfig,
    has_registry_transition: bool,
) -> None:
    """Require an explicit lifecycle transition for routed model changes."""
    if source is None or has_registry_transition:
        return
    if _is_count_first_policy_transition(source, target):
        return
    source_conditioning = source.deck_conditioning
    target_conditioning = target.deck_conditioning
    source_is_routed = (
        source_conditioning is not None
        and source_conditioning.enabled
        and (
            (
                source_conditioning.architecture_version
                == DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION
                and source_conditioning.lora is not None
                and source_conditioning.lora.export_mode == "routed"
            )
            or (
                source_conditioning.architecture_version
                == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
                and source_conditioning.dense_private is not None
                and source_conditioning.dense_private.export_mode == "routed"
            )
            or (
                source_conditioning.architecture_version
                == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
                and source_conditioning.compositional is not None
                and source_conditioning.compositional.export_mode == "routed"
            )
        )
    )
    target_is_routed = (
        target_conditioning is not None
        and target_conditioning.enabled
        and (
            (
                target_conditioning.architecture_version
                == DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION
                and target_conditioning.lora is not None
                and target_conditioning.lora.export_mode == "routed"
            )
            or (
                target_conditioning.architecture_version
                == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
                and target_conditioning.dense_private is not None
                and target_conditioning.dense_private.export_mode == "routed"
            )
            or (
                target_conditioning.architecture_version
                == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
                and target_conditioning.compositional is not None
                and target_conditioning.compositional.export_mode == "routed"
            )
        )
    )
    if source_is_routed and target_is_routed and source != target:
        raise RuntimeError(
            "routed expert warm start cannot change the model config or deck "
            "registry without registry_transition"
        )


def _is_count_first_policy_transition(
    source: AgentNetworkConfig,
    target: AgentNetworkConfig,
) -> bool:
    """Return whether only the explicit unordered-set decoder mode changes."""
    if (
        source.policy.unordered_set_policy != "autoregressive_stop"
        or target.policy.unordered_set_policy != "count_first"
    ):
        return False
    source_with_target_policy = source.model_copy(update={"policy": target.policy})
    return source_with_target_policy == target


def _resolved_target_model_config(config: RLTrainConfig) -> AgentNetworkConfig:
    """Resolve Hydra deck source paths into a portable AgentNetworkConfig."""
    model_config = config.model
    conditioning = model_config.deck_conditioning
    source_registry = config.private_deck_registry
    if conditioning is None or not conditioning.enabled:
        if source_registry is not None and source_registry.decks:
            raise ValueError(
                "private_deck_registry requires enabled model.deck_conditioning"
            )
        return model_config
    if source_registry is None:
        return model_config
    base_dir = deck_records.repo_path(Path("."))
    if conditioning.architecture_version in {
        DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION,
        DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
        DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
    }:
        resolved_experts = resolve_deck_expert_registry(
            source_registry,
            base_dir=base_dir,
        )
        validate_active_exact_strategy_routes(resolved_experts.routes)
        if conditioning.expert_routes and (
            conditioning.expert_routes != resolved_experts.routes
            or conditioning.resolved_registry_sha256
            != resolved_experts.resolved_registry_sha256
        ):
            raise ValueError(
                "resolved model registry disagrees with private_deck_registry sources"
            )
        resolved_conditioning = conditioning.model_copy(
            update={
                "expert_routes": resolved_experts.routes,
                "resolved_registry_sha256": (resolved_experts.resolved_registry_sha256),
            }
        )
    else:
        resolved = resolve_private_registry(source_registry, base_dir=base_dir)
        if conditioning.private_profiles and (
            conditioning.private_profiles != resolved.profiles
            or conditioning.resolved_registry_sha256
            != resolved.resolved_registry_sha256
        ):
            raise ValueError(
                "resolved model registry disagrees with private_deck_registry sources"
            )
        resolved_conditioning = conditioning.model_copy(
            update={
                "private_profiles": resolved.profiles,
                "resolved_registry_sha256": resolved.resolved_registry_sha256,
            }
        )
    resolved_model = model_config.model_copy(
        update={"deck_conditioning": resolved_conditioning}
    )
    _validate_active_dense_registry(config, resolved_model)
    return resolved_model


def _validate_active_dense_registry(
    config: RLTrainConfig,
    model_config: AgentNetworkConfig,
) -> None:
    """Require exact one-to-one routes for every active v3/v4 rollout deck."""
    conditioning = model_config.deck_conditioning
    if conditioning is None or not (
        (
            conditioning.architecture_version
            == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
            and conditioning.dense_private is not None
            and conditioning.dense_private.export_mode == "routed"
        )
        or (
            conditioning.architecture_version
            == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
            and conditioning.compositional is not None
            and conditioning.compositional.export_mode == "routed"
        )
    ):
        return
    candidate_entries = (
        config.curriculum.candidate_deck_pool
        + config.curriculum.additional_candidate_deck_pool
    )
    opponent_entries = (
        config.curriculum.opponent_deck_pool
        + config.curriculum.additional_opponent_deck_pool
    )
    if not candidate_entries or not opponent_entries:
        raise ValueError(
            "routed exact-strategy training requires explicit candidate and "
            "opponent deck pools"
        )
    active_paths = {entry.path for entry in (*candidate_entries, *opponent_entries)}
    active_digests = {
        canonicalize_deck(
            tuple(deck_records.read_deck(deck_records.repo_path(path)))
        ).deck_digest
        for path in active_paths
    }
    route_digests = {route.deck_digest for route in conditioning.expert_routes}
    if active_digests != route_digests:
        raise ValueError(
            "active deck pools and exact-strategy registry must cover the same "
            "exact deck identities"
        )


def _deck_conditioning_enabled(config: AgentNetworkConfig) -> bool:
    conditioning = config.deck_conditioning
    return conditioning is not None and conditioning.enabled


def _initial_policy_version(config: RLTrainConfig) -> int:
    """Return the policy version represented by the configured checkpoint."""
    checkpoint_path = config.checkpoint_path
    if checkpoint_path is None:
        return 0
    path_version = _policy_version_from_checkpoint_filename(checkpoint_path)
    if path_version is not None:
        return path_version
    checkpoint = _load_checkpoint(checkpoint_path)
    return _checkpoint_publish_version(checkpoint) or 0


def _checkpoint_publish_version(checkpoint: Any | None) -> int | None:
    if not isinstance(checkpoint, Mapping):
        return None
    metadata = checkpoint.get("metadata")
    if isinstance(metadata, Mapping):
        version = _optional_int(metadata.get("publish_version"))
        if version is not None:
            return version
    return _optional_int(checkpoint.get("version"))


def _policy_version_from_checkpoint_filename(path: Path) -> int | None:
    stem = path.name.removesuffix(".pt")
    prefix = "policy_v"
    if not stem.startswith(prefix):
        return None
    raw_version = stem.removeprefix(prefix)
    if not raw_version.isdigit():
        return None
    return int(raw_version)


def _maybe_compile_learner_evaluate_actions(
    model: AgentPolicyValueNet,
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    compiled = torch.compile(model.evaluate_actions, mode="reduce-overhead")
    cast(Any, model).evaluate_actions = compiled


def _build_ppo_optimizer(
    config: RLTrainConfig,
    *,
    model: AgentPolicyValueNet,
    device: torch.device,
) -> torch.optim.Optimizer:
    parameter_groups = _ppo_optimizer_parameter_groups(config, model=model)
    kwargs: dict[str, Any] = {}
    if config.learner.fused_adamw and device.type == "cuda":
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(parameter_groups, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(parameter_groups, **kwargs)


def _ppo_optimizer_parameter_groups(
    config: RLTrainConfig,
    *,
    model: AgentPolicyValueNet,
) -> list[dict[str, Any]]:
    """Build disjoint AdamW groups for the active learner contract."""
    optimizer_config = config.optimizer
    named_parameters = tuple(_ppo_optimizer_named_parameters(config, model=model))
    split_groups = (
        optimizer_config.deck_encoder_lr_multiplier != 1.0
        or optimizer_config.private_lr_multiplier != 1.0
        or optimizer_config.private_weight_decay is not None
        or (
            model.config.deck_conditioning is not None
            and model.config.deck_conditioning.enabled
            and model.config.deck_conditioning.architecture_version
            in {
                DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION,
                DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
                DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
            }
        )
    )
    if not split_groups:
        return [
            {
                "name": "model",
                "params": [parameter for _name, parameter in named_parameters],
                "lr": optimizer_config.learning_rate,
                "weight_decay": optimizer_config.weight_decay,
            }
        ]

    groups: dict[str, list[torch.nn.Parameter]] = {
        "shared": [],
        "deck_encoder": [],
    }
    for name, parameter in named_parameters:
        if _is_private_parameter_name(name):
            expert_id = private_state_expert_id(name)
            if expert_id is None:
                raise ValueError(f"private parameter has no expert identity: {name}")
            groups.setdefault(f"private.deck_{expert_id}", []).append(parameter)
        elif name.startswith(("deck_encoder.", "deck_input_projection.")):
            groups["deck_encoder"].append(parameter)
        else:
            groups["shared"].append(parameter)

    private_weight_decay = (
        optimizer_config.weight_decay
        if optimizer_config.private_weight_decay is None
        else optimizer_config.private_weight_decay
    )
    parameter_groups: list[dict[str, Any]] = []
    for name, parameters in groups.items():
        if not parameters:
            continue
        if name == "shared":
            multiplier = 1.0
            weight_decay = optimizer_config.weight_decay
            expert_id = None
        elif name == "deck_encoder":
            multiplier = optimizer_config.deck_encoder_lr_multiplier
            weight_decay = optimizer_config.weight_decay
            expert_id = None
        else:
            multiplier = optimizer_config.private_lr_multiplier
            weight_decay = private_weight_decay
            expert_id = name.removeprefix("private.deck_")
        parameter_groups.append(
            {
                "name": name,
                "params": parameters,
                "lr": optimizer_config.learning_rate * multiplier,
                "weight_decay": weight_decay,
                "expert_id": expert_id,
                "scheduler_active_updates": 0,
                "scheduler_horizon_updates": _planned_lr_scheduler_steps(config),
            }
        )
    return parameter_groups


def _ppo_optimizer_named_parameters(
    config: RLTrainConfig,
    *,
    model: AgentPolicyValueNet,
) -> Iterator[tuple[str, torch.nn.Parameter]]:
    """Omit inert schema-9 parameters from historical exact-resume optimizers."""
    planner_learning = _planner_learning_enabled(config)
    root_value_learning = config.ppo.root_information_value_coef > 0.0
    macro_learning = (
        config.ppo.macro_conditional_coef > 0.0 or config.ppo.macro_expected_coef > 0.0
    )
    for name, parameter in model.named_parameters():
        if not planner_learning and (
            name == "policy_head.proposal_query_projection.weight"
            or name.startswith("planner_reranker.")
        ):
            continue
        if not root_value_learning and name.startswith(
            "root_perspective_value_adapter."
        ):
            continue
        if not macro_learning and name.startswith("macro_outcome_heads."):
            continue
        yield name, parameter


def _is_private_parameter_name(name: str) -> bool:
    """Return whether a model parameter belongs to a routed expert bank."""
    return name.startswith(
        (
            "state_encoder.private_adapters.",
            "state_encoder.private_lora.",
            "policy_head.private_lora.",
            "private_policy_adapters.",
            "private_root_value_heads.",
            "private_prefix_value_heads.",
            "state_encoder.private_strategy_stacks.",
            "policy_head.private_strategies.",
            "dense_private_root_value_heads.",
            "dense_private_prefix_value_heads.",
            "exact_capsules.",
        )
    )


def _apply_registry_transition_optimizer_state(
    config: RLTrainConfig,
    *,
    model: AgentPolicyValueNet,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Apply a bound name-based optimizer transplant for a transition run."""
    declaration = config.registry_transition
    if declaration is None:
        return
    plan = getattr(model, "_deck_registry_transition_plan", None)
    if not isinstance(plan, DeckRegistryTransitionPlan):
        raise RuntimeError("registry transition model has no validated plan")
    if config.checkpoint_path is None:
        raise RuntimeError("registry transition has no source checkpoint")
    checkpoint_path = deck_records.repo_path(config.checkpoint_path)
    if declaration.optimizer_mode == "fresh":
        optimizer_summary = validate_deck_registry_source_state(
            state_path=deck_records.repo_path(declaration.optimizer_state_path),
            source_checkpoint_path=checkpoint_path,
            source_checkpoint_sha256=declaration.source_checkpoint_sha256,
            optimizer_state_sha256=declaration.optimizer_state_sha256,
            optimizer_state_size_bytes=declaration.optimizer_state_size_bytes,
            plan=plan,
        )
    else:
        optimizer_summary = transplant_deck_registry_optimizer_state(
            optimizer=optimizer,
            model=model,
            optimizer_parameter_names=_optimizer_parameter_names(model, optimizer),
            state_path=deck_records.repo_path(declaration.optimizer_state_path),
            source_checkpoint_path=checkpoint_path,
            source_checkpoint_sha256=declaration.source_checkpoint_sha256,
            optimizer_state_sha256=declaration.optimizer_state_sha256,
            optimizer_state_size_bytes=declaration.optimizer_state_size_bytes,
            plan=plan,
        )
        _configure_transition_optimizer_group_clocks(
            config,
            optimizer=optimizer,
            plan=plan,
            optimizer_summary=optimizer_summary,
        )
    transition_summary = dict(_registry_transition_summary(model) or {})
    transition_summary["optimizer"] = optimizer_summary
    cast(Any, model)._deck_registry_transition_summary = transition_summary


def _configure_transition_optimizer_group_clocks(
    config: RLTrainConfig,
    *,
    optimizer: torch.optim.Optimizer,
    plan: DeckRegistryTransitionPlan,
    optimizer_summary: Mapping[str, Any],
) -> None:
    """Give new/remapped experts a fresh active-update LR clock."""
    source_updates = int(optimizer_summary["source_total_optimizer_updates"])
    planned_steps = _planned_lr_scheduler_steps(config)
    continue_schedule = plan.scheduler_mode == "continue"
    raw_source_clocks = optimizer_summary.get("source_expert_scheduler_clocks")
    if continue_schedule and not isinstance(raw_source_clocks, Mapping):
        raise ValueError("continued transition has no source expert scheduler clocks")
    source_clocks = cast(Mapping[str, Any], raw_source_clocks or {})
    target_expert_ids = set(plan.by_target_expert)
    configured_expert_ids: set[str] = set()
    for group in optimizer.param_groups:
        expert_id = group.get("expert_id")
        if expert_id is not None:
            if not isinstance(expert_id, str) or expert_id not in target_expert_ids:
                raise ValueError("target optimizer group has an unknown expert ID")
            if group.get("name") != f"private.deck_{expert_id}":
                raise ValueError(
                    "target optimizer group name does not match its expert"
                )
            if expert_id in configured_expert_ids:
                raise ValueError("target optimizer contains duplicate expert groups")
            configured_expert_ids.add(expert_id)
        preserved = (
            expert_id is None
            or plan.by_target_expert[expert_id].preserve_optimizer_state
        )
        if continue_schedule and expert_id is None:
            group["scheduler_active_updates"] = source_updates
            group["scheduler_horizon_updates"] = planned_steps
        elif continue_schedule and preserved:
            if expert_id is None:
                raise RuntimeError("continued private scheduler group has no expert ID")
            active_updates, horizon_updates = _source_expert_scheduler_clock(
                source_clocks,
                expert_id,
            )
            group["scheduler_active_updates"] = active_updates
            group["scheduler_horizon_updates"] = horizon_updates
        elif continue_schedule:
            group["scheduler_active_updates"] = 0
            group["scheduler_horizon_updates"] = planned_steps - source_updates
        else:
            group["scheduler_active_updates"] = 0
            group["scheduler_horizon_updates"] = planned_steps
        if int(group["scheduler_horizon_updates"]) <= 0:
            raise ValueError("optimizer group has no remaining scheduler horizon")
    if configured_expert_ids != target_expert_ids:
        raise ValueError(
            "target optimizer expert groups do not match the target registry"
        )


def _source_expert_scheduler_clock(
    clocks: Mapping[str, Any],
    expert_id: str,
) -> tuple[int, int]:
    """Validate one continued expert's source active-update clock."""
    raw_clock = clocks.get(expert_id)
    if not isinstance(raw_clock, Mapping):
        raise ValueError("continued expert has no source scheduler clock")
    active_updates = raw_clock.get("active_updates")
    horizon_updates = raw_clock.get("horizon_updates")
    if (
        not isinstance(active_updates, int)
        or isinstance(active_updates, bool)
        or active_updates < 0
        or not isinstance(horizon_updates, int)
        or isinstance(horizon_updates, bool)
        or horizon_updates <= 0
        or active_updates > horizon_updates
    ):
        raise ValueError("continued expert has an invalid source scheduler clock")
    return active_updates, horizon_updates


def _registry_transition_summary(
    model: torch.nn.Module,
) -> Mapping[str, Any] | None:
    """Return derived transition provenance attached during model loading."""
    summary = getattr(model, "_deck_registry_transition_summary", None)
    return cast(Mapping[str, Any], summary) if isinstance(summary, Mapping) else None


def _attach_checkpoint_transition_lineage(
    model: torch.nn.Module,
    checkpoint: Any,
) -> None:
    """Carry validated transition ancestry through exact descendant resumes."""
    if not isinstance(checkpoint, Mapping):
        return
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        return
    summary = metadata.get("registry_transition")
    if not isinstance(summary, Mapping):
        return
    manifest_sha256 = summary.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise ValueError("checkpoint registry transition lineage is malformed")
    cast(Any, model)._deck_registry_transition_summary = dict(summary)


def _load_anchor_model(
    config: RLTrainConfig,
    *,
    device: torch.device,
) -> AgentPolicyValueNet | None:
    transition_updates = config.ppo.transition_distillation.optimizer_updates
    if config.anchor_checkpoint_path is None or (
        config.ppo.kl_anchor_coef == 0.0 and transition_updates == 0
    ):
        return None
    if config.registry_transition is not None and transition_updates == 0:
        declaration = config.registry_transition
        anchor_path = deck_records.repo_path(config.anchor_checkpoint_path)
        if not anchor_path.is_file():
            raise FileNotFoundError(
                f"registry transition anchor checkpoint is missing: {anchor_path}"
            )
        if file_sha256(anchor_path) != declaration.source_checkpoint_sha256:
            raise ValueError(
                "registry transition anchor must bind the declared source checkpoint"
            )
        anchor = _load_training_model(config, device=device)
        anchor.eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
        return anchor
    return load_frozen_anchor_model(
        config.anchor_checkpoint_path,
        fallback_config=config.model,
        device=device,
    )


def _load_checkpoint(checkpoint_path: Path | None) -> Any | None:
    if checkpoint_path is None:
        return None
    return torch.load(deck_records.repo_path(checkpoint_path), map_location="cpu")


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return _strip_lightning_model_prefix(cast(Mapping[str, Any], value))
        return _strip_lightning_model_prefix(cast(Mapping[str, Any], checkpoint))
    raise TypeError("checkpoint must be a state_dict or contain model_state_dict")


def _strip_lightning_model_prefix(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    if not state_dict:
        return state_dict
    if all(str(key).startswith("model.") for key in state_dict):
        return {
            str(key).removeprefix("model."): value for key, value in state_dict.items()
        }
    return state_dict


def _checkpoint_model_config(checkpoint: Any | None) -> AgentNetworkConfig | None:
    if not isinstance(checkpoint, Mapping):
        return None
    for key in ("model_config", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return AgentNetworkConfig.model_validate(value)
    full_config = checkpoint.get("config")
    if isinstance(full_config, Mapping):
        model_config = full_config.get("model")
        if isinstance(model_config, Mapping):
            return AgentNetworkConfig.model_validate(model_config)
    return None


def _deck_pair(config: RolloutDeckConfig) -> DeckPair:
    candidate = tuple(deck_records.read_deck(deck_records.repo_path(config.candidate)))
    opponent_path = config.opponent or config.candidate
    opponent = tuple(deck_records.read_deck(deck_records.repo_path(opponent_path)))
    return (candidate, opponent)


def _normalize_deck_pair(deck_pair: DeckPair) -> DeckPair:
    return (
        tuple(int(card_id) for card_id in deck_pair[0]),
        tuple(int(card_id) for card_id in deck_pair[1]),
    )


def _record_assignment_counts(
    assignment: GameAssignment,
    assigned_counts: Counter[str],
) -> None:
    assigned_counts[assignment.opponent_kind] += 1
    if assignment.frozen_sampling_lane is not None:
        assigned_counts[f"frozen_lane:{assignment.frozen_sampling_lane}"] += 1
    assigned_counts[_opponent_count_key(assignment)] += 1
    assigned_counts[_deck_count_key(assignment)] += 1
    assigned_counts[f"lane:{assignment.candidate_lane}"] += 1


def _remove_drained_retired_frozen_members(
    sampler: CurriculumSampler,
    *,
    active_assignments: Iterable[GameAssignment],
) -> tuple[str, ...]:
    """Remove retired runtimes only after their active leases have drained."""
    active_frozen_ids = {
        assignment.opponent_id
        for assignment in active_assignments
        if assignment.opponent_kind == "frozen"
    }
    removed: list[str] = []
    for member in tuple(sampler.frozen_members):
        if not member.retired:
            continue
        if member.opponent_id in active_frozen_ids:
            continue
        sampler.remove_frozen_member(member.opponent_id)
        removed.append(member.opponent_id)
    return tuple(removed)


def _record_finished_counts(
    assignment: GameAssignment,
    *,
    candidate_reward: float,
    finished_counts: Counter[str],
    deck_outcome_counts: dict[str, Counter[str]],
    deck_winrate_ema: dict[str, float],
    winrate_ema_alpha: float,
) -> None:
    finished_counts[assignment.opponent_kind] += 1
    if assignment.frozen_sampling_lane is not None:
        finished_counts[f"frozen_lane:{assignment.frozen_sampling_lane}"] += 1
    finished_counts[_opponent_count_key(assignment)] += 1
    finished_counts[f"lane:{assignment.candidate_lane}"] += 1
    _record_finished_deck_counts(
        _candidate_deck_label(assignment),
        candidate_reward=candidate_reward,
        finished_counts=finished_counts,
        deck_outcome_counts=deck_outcome_counts,
        deck_winrate_ema=deck_winrate_ema,
        winrate_ema_alpha=winrate_ema_alpha,
    )


def _record_finished_deck_counts(
    candidate_deck_label: str,
    *,
    candidate_reward: float,
    finished_counts: Counter[str],
    deck_outcome_counts: dict[str, Counter[str]],
    deck_winrate_ema: dict[str, float],
    winrate_ema_alpha: float,
) -> None:
    label = candidate_deck_label or "candidate"
    deck_key = _deck_label_count_key(label)
    outcome = _outcome_label_from_reward(candidate_reward)
    finished_counts[deck_key] += 1
    finished_counts[f"{deck_key}:{outcome}"] += 1
    outcomes = deck_outcome_counts.setdefault(label, Counter())
    outcomes[outcome] += 1
    previous = deck_winrate_ema.get(label, 0.5)
    deck_winrate_ema[label] = (
        1.0 - winrate_ema_alpha
    ) * previous + winrate_ema_alpha * _score_from_candidate_reward(candidate_reward)


def _curriculum_deck_summary(
    *,
    sampler: CurriculumSampler,
    assigned_counts: Mapping[str, int],
    finished_counts: Mapping[str, int],
    deck_outcome_counts: Mapping[str, Counter[str]],
    deck_winrate_ema: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    configured = {
        str(entry["label"]): entry for entry in sampler.candidate_deck_distribution
    }
    labels = set(configured) | set(deck_outcome_counts)
    labels.update(_labels_from_deck_counts(assigned_counts))
    labels.update(_labels_from_deck_counts(finished_counts))
    total_assigned = sum(
        int(assigned_counts.get(_deck_label_count_key(label), 0)) for label in labels
    )
    decks: dict[str, dict[str, Any]] = {}
    for label in sorted(labels):
        outcomes = deck_outcome_counts.get(label, Counter())
        wins = int(outcomes.get("win", 0))
        losses = int(outcomes.get("loss", 0))
        draws = int(outcomes.get("draw", 0))
        games = wins + losses + draws
        assigned = int(assigned_counts.get(_deck_label_count_key(label), 0))
        configured_entry = configured.get(label, {})
        decks[label] = {
            "label": label,
            "path": configured_entry.get("path"),
            "configured_weight": configured_entry.get("weight"),
            "configured_probability": configured_entry.get("probability"),
            "target_probability": configured_entry.get(
                "target_probability",
                configured_entry.get("probability"),
            ),
            "assigned": assigned,
            "assigned_fraction": (
                float(assigned) / float(total_assigned) if total_assigned > 0 else 0.0
            ),
            "finished": int(finished_counts.get(_deck_label_count_key(label), 0)),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "winrate": (
                (float(wins) + 0.5 * float(draws)) / float(games) if games > 0 else None
            ),
            "winrate_ema": deck_winrate_ema.get(label),
        }
    return decks


def _curriculum_lane_summary(
    sampler: CurriculumSampler,
    *,
    assigned_counts: Mapping[str, int],
    finished_counts: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    """Return configured and observed target/near/broad lane fractions."""
    rows = sampler.candidate_lane_distribution
    total_assigned = sum(
        int(assigned_counts.get(f"lane:{row['lane']}", 0)) for row in rows
    )
    total_finished = sum(
        int(finished_counts.get(f"lane:{row['lane']}", 0)) for row in rows
    )
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        lane = str(row["lane"])
        assigned = int(assigned_counts.get(f"lane:{lane}", 0))
        finished = int(finished_counts.get(f"lane:{lane}", 0))
        summary[lane] = {
            **row,
            "assigned": assigned,
            "assigned_fraction": (
                assigned / float(total_assigned) if total_assigned > 0 else 0.0
            ),
            "finished": finished,
            "finished_fraction": (
                finished / float(total_finished) if total_finished > 0 else 0.0
            ),
        }
    return summary


def _labels_from_deck_counts(counts: Mapping[str, int]) -> set[str]:
    labels: set[str] = set()
    for key in counts:
        if key.startswith("deck:") and key.count(":") == 1:
            labels.add(key.removeprefix("deck:"))
    return labels


def _opponent_count_key(assignment: GameAssignment) -> str:
    return f"{assignment.opponent_kind}:{assignment.opponent_id}"


def _deck_count_key(assignment: GameAssignment) -> str:
    return _deck_label_count_key(_candidate_deck_label(assignment))


def _deck_label_count_key(label: str) -> str:
    return f"deck:{label or 'candidate'}"


def _candidate_deck_label(assignment: GameAssignment) -> str:
    return assignment.candidate_deck_label or "candidate"


def _outcome_label_from_reward(candidate_reward: float) -> str:
    if candidate_reward > 0.0:
        return "win"
    if candidate_reward < 0.0:
        return "loss"
    return "draw"


def _score_from_candidate_reward(candidate_reward: float) -> float:
    if candidate_reward > 0.0:
        return 1.0
    if candidate_reward < 0.0:
        return 0.0
    return 0.5


def _reward_for_seat(seat: int, winner_index: int) -> float:
    if winner_index == 2:
        return 0.0
    return 1.0 if winner_index == seat else -1.0


def _single_candidate_policy_version(versions: set[int]) -> int | None:
    """Return the sole acting version, excluding missing or mixed-version games."""
    if len(versions) != 1:
        return None
    return next(iter(versions))


def _rl_train_config_dump(
    config: RLTrainConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    data["output_dir"] = deck_records.display_path(output_dir)
    return data


def _default_pool_factory(
    num_games: int,
    deck_pair_sampler: DeckPairSampler,
) -> AbstractContextManager[VectorPoolLike]:
    return VectorBattlePool(num_games, deck_pair_sampler)


def _training_pool_factory(
    *,
    include_search_input: bool,
) -> RLTrainPoolFactory:
    def factory(
        num_games: int,
        deck_pair_sampler: DeckPairSampler,
    ) -> AbstractContextManager[VectorPoolLike]:
        return VectorBattlePool(
            num_games,
            deck_pair_sampler,
            include_search_input=include_search_input,
        )

    return factory


def _training_include_search_input(config: RLTrainConfig) -> bool:
    """Return whether rollout observations need engine search state tokens."""
    policy_iteration = getattr(config, "amortized_policy_iteration", None)
    sequence_config = getattr(getattr(config, "model", None), "sequence", None)
    return bool(
        config.execution.archive_trajectories
        or config.rollout_probe.enabled
        or config.engine_teacher.enabled
        or config.planner is not None
        or bool(getattr(policy_iteration, "enabled", False))
        or bool(
            sequence_config is not None
            and sequence_config.engine_facts.enabled
        )
    )


def _decision_count(trajectories: Sequence[GameTrajectory]) -> int:
    return sum(trajectory.decision_count for trajectory in trajectories)


def _learner_window_decision_budget(config: RLTrainConfig) -> int:
    """Return the complete-trajectory decision target for one learner window."""
    return min(
        config.collection.iteration_decisions,
        config.learner.max_update_decisions or config.collection.iteration_decisions,
    )


def _partition_trajectory_window(
    trajectories: Sequence[GameTrajectory],
    *,
    decision_budget: int,
    retain_boundary: bool,
) -> tuple[tuple[GameTrajectory, ...], tuple[GameTrajectory, ...]]:
    """Partition an ordered buffer without ever splitting a completed game.

    When ``retain_boundary`` is true, the first trajectory that would cross the
    soft decision budget remains in the deferred suffix. A single trajectory
    larger than the budget is always selected so it cannot become permanently
    stuck. For a final window, ``retain_boundary`` is false and the crossing
    trajectory is included whole.
    """
    if decision_budget <= 0:
        raise ValueError("decision_budget must be positive")
    selected_count = 0
    selected_decisions = 0
    for trajectory in trajectories:
        trajectory_decisions = trajectory.decision_count
        crosses_budget = selected_decisions + trajectory_decisions > decision_budget
        if retain_boundary and selected_decisions > 0 and crosses_budget:
            break
        selected_count += 1
        selected_decisions += trajectory_decisions
        if selected_decisions >= decision_budget:
            break
    return (
        tuple(trajectories[:selected_count]),
        tuple(trajectories[selected_count:]),
    )


def _planned_lr_scheduler_steps(config: RLTrainConfig) -> int:
    explicit_budget = config.optimizer.scheduler_total_updates
    if explicit_budget is not None:
        return explicit_budget
    planned_decisions = min(
        config.collection.iteration_decisions,
        config.learner.max_update_decisions or config.collection.iteration_decisions,
    )
    microbatches_per_epoch = math.ceil(
        planned_decisions / config.collection.microbatch_size
    )
    effective_updates_per_epoch = math.ceil(
        microbatches_per_epoch / config.collection.gradient_accumulation_steps
    )
    ppo_updates = (
        config.collection.training_iterations
        * config.collection.ppo_epochs
        * effective_updates_per_epoch
    )
    policy_iteration_updates = 0
    if config.amortized_policy_iteration.enabled:
        policy_iteration_updates = config.collection.training_iterations * (
            int(config.amortized_policy_iteration.learner.real_retrace_enabled)
            + config.amortized_policy_iteration.learner.counterfactual_updates_per_iteration
        )
    return max(1, ppo_updates + policy_iteration_updates)


def _policy_iteration_replay_store(
    config: RLTrainConfig,
    *,
    output_path: Path,
) -> PolicyIterationReplayStore:
    policy_iteration = config.amortized_policy_iteration
    belief_fingerprint = (
        policy_iteration.belief_sampler.prior_deck_signature_summary_sha256
    )
    if belief_fingerprint is None:
        raise ValueError("policy-iteration replay requires a belief fingerprint")
    configured_library = os.environ.get("PTCG_RL_CG_PROBE_LIB")
    engine_library = (
        Path(configured_library)
        if configured_library
        else deck_records.repo_path(Path("src/native/cg_probe/libcg_probe.so"))
    )
    if not engine_library.is_file():
        raise FileNotFoundError(
            f"policy-iteration native engine library is missing: {engine_library}"
        )
    identity = ReplayIdentity(
        run_id=config.run.version,
        model_schema_fingerprint=_canonical_json_fingerprint(
            config.model.model_dump(mode="json")
        ),
        belief_fingerprint=belief_fingerprint,
        proposal_fingerprint=_canonical_json_fingerprint(
            {
                "candidate_proposal": policy_iteration.candidate_proposal.model_dump(
                    mode="json"
                ),
                "belief_worlds": policy_iteration.belief_worlds,
                "native": policy_iteration.native.model_dump(mode="json"),
                "target_network": policy_iteration.target_network.model_dump(
                    mode="json"
                ),
                "retrace": policy_iteration.retrace.model_dump(mode="json"),
                "cmpo": policy_iteration.cmpo.model_dump(mode="json"),
            }
        ),
        engine_fingerprint=file_sha256(engine_library),
        schema_version=policy_iteration.replay.schema_version,
    )
    replay = policy_iteration.replay
    return PolicyIterationReplayStore(
        output_path / replay.relative_dir,
        identity=identity,
        rows_per_shard=replay.rows_per_shard,
        compress=replay.compression,
        max_committed_shards=replay.max_committed_shards,
        async_writes=replay.async_writes,
    )


def _validate_policy_iteration_pending_resume(
    *,
    restored_schema: int | None,
    pending_migration: Literal["empty"] | None,
) -> None:
    """Reject replay guesses or redundant declarations at an exact boundary."""
    if restored_schema == 1 and pending_migration != "empty":
        raise ValueError(
            "schema-v1 policy-iteration state has no pending-target cursor; "
            "exact resume requires an audited "
            "policy_iteration_pending_targets_migration=empty declaration"
        )
    if restored_schema == 2 and pending_migration is not None:
        raise ValueError(
            "policy-iteration pending-target migration is only valid for a "
            "schema-v1 auxiliary state"
        )


def _canonical_json_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _restore_training_progress(
    config: RLTrainConfig,
    *,
    model: AgentPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    policy_version: int,
    planned_scheduler_steps: int,
) -> TrainingProgress:
    if config.registry_transition is not None:
        return _registry_transition_training_progress(
            config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            planned_scheduler_steps=planned_scheduler_steps,
        )
    checkpoint_path = (
        deck_records.repo_path(config.checkpoint_path)
        if config.checkpoint_path is not None
        else None
    )
    resume_config = config.resume
    if resume_config.state_path is not None:
        resume_config = resume_config.model_copy(
            update={"state_path": deck_records.repo_path(resume_config.state_path)}
        )
    return restore_training_progress(
        config=resume_config,
        checkpoint_path=checkpoint_path,
        policy_version=policy_version,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        planned_scheduler_steps=planned_scheduler_steps,
        resume_config_sha256=_resume_relevant_config_sha256(config),
        approved_previous_resume_config_sha256=(
            _resume_config_migration_source_sha256(config)
        ),
        optimizer_parameter_names=_optimizer_parameter_names(model, optimizer),
        new_optimizer_parameters=(
            model.state_encoder.global_context_projection.weight,
            model.state_encoder.public_state_projection.weight,
            model.policy_head.attachment_identity_projection.weight,
        ),
    )


def _registry_transition_training_progress(
    config: RLTrainConfig,
    *,
    model: AgentPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    planned_scheduler_steps: int,
) -> TrainingProgress:
    """Initialize a new run while applying its declared scheduler semantics."""
    declaration = config.registry_transition
    if declaration is None:
        raise RuntimeError("registry transition progress requires a declaration")
    summary = _registry_transition_summary(model)
    optimizer_summary = None if summary is None else summary.get("optimizer")
    if not isinstance(optimizer_summary, Mapping):
        raise RuntimeError("registry transition optimizer provenance is missing")
    state_path = deck_records.repo_path(declaration.optimizer_state_path)
    if declaration.scheduler_mode == "restart":
        return TrainingProgress(0, 0, state_path, 0)
    source_updates = int(optimizer_summary["source_total_optimizer_updates"])
    if source_updates >= planned_scheduler_steps:
        raise ValueError(
            "continued registry-transition schedule requires "
            "scheduler_total_updates above the source update clock"
        )
    _set_lr_scheduler_progress(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        completed_updates=source_updates,
    )
    return TrainingProgress(0, source_updates, state_path, 0)


def _set_lr_scheduler_progress(
    *,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_updates: int,
) -> None:
    """Set a freshly built cosine scheduler to an explicit cumulative clock."""
    scheduler_state = lr_scheduler.state_dict()
    base_lrs = [float(value) for value in scheduler_state["base_lrs"]]
    if isinstance(lr_scheduler, torch.optim.lr_scheduler.LambdaLR):
        current_lrs = [
            base_lr * float(schedule(completed_updates))
            for base_lr, schedule in zip(
                base_lrs,
                lr_scheduler.lr_lambdas,
                strict=True,
            )
        ]
    elif isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        total_steps = int(scheduler_state["T_max"])
        eta_min = float(scheduler_state["eta_min"])
        bounded_updates = min(completed_updates, total_steps)
        multiplier = 0.5 * (1.0 + math.cos(math.pi * bounded_updates / total_steps))
        current_lrs = [
            eta_min + (base_lr - eta_min) * multiplier for base_lr in base_lrs
        ]
    else:
        raise TypeError("registry transition supports only cosine LR schedulers")
    scheduler_state["last_epoch"] = completed_updates
    scheduler_state["_step_count"] = completed_updates + 1
    scheduler_state["_last_lr"] = current_lrs
    lr_scheduler.load_state_dict(scheduler_state)
    for parameter_group, learning_rate in zip(
        optimizer.param_groups,
        current_lrs,
        strict=True,
    ):
        parameter_group["lr"] = learning_rate


def _resume_relevant_config_sha256(config: RLTrainConfig) -> str:
    """Fingerprint stateful learning semantics, excluding operational settings."""
    collection = config.collection
    learner = config.learner
    ppo_payload = config.ppo.model_dump(mode="json")
    if config.ppo.engine_teacher_coef == 0.0:
        # Preserve exact-resume identity for checkpoints written before the
        # optional auxiliary objective existed. A disabled loss has no stateful
        # learning semantics and must not invalidate those sidecars.
        ppo_payload.pop("engine_teacher_coef", None)
    planner_resume_semantics = _planner_learning_enabled(config)
    macro_resume_semantics = config.macro_credit.enabled
    model_payload = config.model.model_dump(mode="json")
    if not planner_resume_semantics and not macro_resume_semantics:
        # These zero-start heads were added after schema-v2 sidecars had been
        # published.  When schema 9 and all of its losses are disabled they do
        # not participate in the behavior or learner contract.  Preserve the
        # exact v1 payload so historical recovery pairs remain resumable.
        model_payload.pop("planner_reranker_architecture_version", None)
        model_payload.pop("root_perspective_value", None)
        for field_name in (
            "candidate_rerank_coef",
            "proposal_distillation_coef",
            "root_information_value_coef",
            "planner_imitation_max_policy_age",
            "planner_target_ratio_clip",
        ):
            ppo_payload.pop(field_name, None)
    if not macro_resume_semantics:
        model_payload.pop("macro_outcome", None)
        ppo_payload.pop("macro_conditional_coef", None)
        ppo_payload.pop("macro_expected_coef", None)
    payload: dict[str, Any] = {
        "schema": (
            "rl-resume-relevant-config-v3"
            if macro_resume_semantics
            else "rl-resume-relevant-config-v2"
            if planner_resume_semantics
            else "rl-resume-relevant-config-v1"
        ),
        "model": model_payload,
        "anchor_checkpoint_path": (
            str(config.anchor_checkpoint_path)
            if config.anchor_checkpoint_path is not None
            else None
        ),
        "seed": config.seed,
        "collection": {
            "iteration_decisions": collection.iteration_decisions,
            "ppo_epochs": collection.ppo_epochs,
            "effective_batch_size": collection.effective_batch_size,
            "sampling_temperature": collection.sampling_temperature,
            "policy_kind": collection.policy_kind,
            "opponent_mode": collection.opponent_mode,
            "decks": collection.decks.model_dump(mode="json"),
        },
        "curriculum": _resume_curriculum_payload(config.curriculum),
        "rollout_probe": config.rollout_probe.model_dump(mode="json"),
        "rollout_belief": config.rollout_belief.model_dump(mode="json"),
        "gae": config.gae.model_dump(mode="json"),
        "ppo": ppo_payload,
        "optimizer": config.optimizer.model_dump(mode="json"),
        "learner": {
            "max_staleness": learner.max_staleness,
            "staleness_scope": learner.staleness_scope,
            "shuffle_each_epoch": learner.shuffle_each_epoch,
            "drain_buffer_decisions": learner.drain_buffer_decisions,
            "max_update_decisions": learner.max_update_decisions,
            "legacy_sampling_temperature": learner.legacy_sampling_temperature,
        },
        "remote_max_staleness": config.distributed.remote_max_staleness,
    }
    if learner.frozen_encoder_prefix_layers > 0:
        payload["learner"]["frozen_encoder_prefix_layers"] = (
            learner.frozen_encoder_prefix_layers
        )
    if config.amortized_policy_iteration.enabled:
        payload["amortized_policy_iteration"] = (
            config.amortized_policy_iteration.model_dump(mode="json")
        )
    if planner_resume_semantics or macro_resume_semantics:
        payload["factual"] = config.factual.model_dump(mode="json")
        payload["planner"] = (
            None if config.planner is None else config.planner.model_dump(mode="json")
        )
        payload["learner"]["schema9"] = (
            None if learner.schema9 is None else learner.schema9.model_dump(mode="json")
        )
    if macro_resume_semantics:
        payload["macro_credit"] = config.macro_credit.model_dump(mode="json")
    if config.frozen_league.enabled:
        payload["frozen_league"] = config.frozen_league.model_dump(mode="json")
    if config.engine_teacher.enabled:
        engine_teacher_payload = config.engine_teacher.model_dump(mode="json")
        # These fields change scheduling/transport only and were introduced
        # after schema-v2 sidecars were already published. Excluding them keeps
        # unchanged historical learning semantics exactly resumable.
        for operational_field in (
            "minimum_inference_budget_seconds",
            "async_actor",
            "async_queue_batches",
        ):
            engine_teacher_payload.pop(operational_field, None)
        payload["engine_teacher"] = engine_teacher_payload
    serialized = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _planner_learning_enabled(config: RLTrainConfig) -> bool:
    """Return whether schema-9 parameters belong to the learner contract."""
    return bool(
        config.learner.schema9 is not None
        or config.ppo.candidate_rerank_coef > 0.0
        or config.ppo.proposal_distillation_coef > 0.0
    )


def _resume_curriculum_payload(config: CurriculumConfig) -> dict[str, Any]:
    """Preserve legacy hashes when the new assignment stream is inactive."""
    payload = config.model_dump(mode="json")
    if config.assignment_schedule.mode == "random":
        payload.pop("assignment_schedule", None)
    return payload


def _resume_config_migration_source_sha256(
    config: RLTrainConfig,
) -> str | None:
    """Fingerprint the exact prior config allowed by declared migrations."""
    previous_max_staleness = config.resume.max_staleness_migration_from
    previous_effective_batch_size = config.resume.effective_batch_size_migration_from
    previous_assignment_block_size = (
        config.resume.curriculum_assignment_block_size_migration_from
    )
    assignment_lanes_from_disabled = (
        config.resume.curriculum_assignment_lanes_migration_from_disabled
    )
    previous_shuffle_each_epoch = (
        config.resume.learner_shuffle_each_epoch_migration_from
    )
    frozen_league_from_disabled = config.resume.frozen_league_migration_from_disabled
    previous_engine_teacher_values = config.resume.engine_teacher_migration_from
    approved_previous_sha256 = config.resume.approved_previous_resume_config_sha256
    if (
        previous_max_staleness is None
        and previous_effective_batch_size is None
        and previous_assignment_block_size is None
        and not assignment_lanes_from_disabled
        and previous_shuffle_each_epoch is None
        and not frozen_league_from_disabled
        and previous_engine_teacher_values is None
        and approved_previous_sha256 is None
    ):
        return None
    collection = config.collection
    if previous_effective_batch_size is not None:
        collection = collection.model_copy(
            update={
                "microbatch_size": previous_effective_batch_size,
                "gradient_accumulation_steps": 1,
            }
        )
    learner = config.learner
    distributed = config.distributed
    if previous_max_staleness is not None:
        learner = learner.model_copy(update={"max_staleness": previous_max_staleness})
        distributed = distributed.model_copy(
            update={"remote_max_staleness": previous_max_staleness}
        )
    if previous_shuffle_each_epoch is not None:
        learner = learner.model_copy(
            update={"shuffle_each_epoch": previous_shuffle_each_epoch}
        )
    curriculum = config.curriculum
    if previous_assignment_block_size is not None:
        curriculum = curriculum.model_copy(
            update={
                "assignment_block_size": previous_assignment_block_size,
                "assignment_schedule": AssignmentScheduleConfig(),
            }
        )
    if assignment_lanes_from_disabled:
        assignment_schedule = curriculum.assignment_schedule.model_copy(
            update={"execution_lanes": 0, "lane_block_size": 0}
        )
        curriculum = curriculum.model_copy(
            update={"assignment_schedule": assignment_schedule}
        )
    previous_engine_teacher = config.engine_teacher
    if previous_engine_teacher_values is not None:
        allowed_fields = {
            "deadline_seconds",
            "max_teacher_seconds_per_actor_step",
            "inference_timeout_seconds",
        }
        unknown_fields = set(previous_engine_teacher_values).difference(allowed_fields)
        if unknown_fields:
            unknown = ", ".join(sorted(unknown_fields))
            raise ValueError(
                f"unsupported engine teacher resume migration fields: {unknown}"
            )
        previous_payload = config.engine_teacher.model_dump(mode="python")
        previous_payload.update(dict(previous_engine_teacher_values))
        previous_engine_teacher = OnlineEngineTeacherConfig.model_validate(
            previous_payload
        )
    previous_config = config.model_copy(
        update={
            "collection": collection,
            "curriculum": curriculum,
            "learner": learner,
            "distributed": distributed,
            "engine_teacher": previous_engine_teacher,
            "frozen_league": (
                config.frozen_league.model_copy(update={"enabled": False})
                if frozen_league_from_disabled
                else config.frozen_league
            ),
        }
    )
    previous_sha256 = _resume_relevant_config_sha256(previous_config)
    current_sha256 = _resume_relevant_config_sha256(config)
    if approved_previous_sha256 is not None:
        if approved_previous_sha256 == current_sha256:
            raise ValueError(
                "approved previous resume config must differ from the current config"
            )
        return approved_previous_sha256
    if previous_sha256 == current_sha256:
        raise ValueError(
            "declared resume config migration source must differ from the "
            "current resume-relevant config"
        )
    return previous_sha256


def _optimizer_parameter_names(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[tuple[str, ...], ...]:
    """Return model parameter names in the optimizer's exact positional layout."""
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    groups: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            name = names_by_id.get(id(parameter))
            if name is None:
                raise ValueError(
                    "optimizer contains a parameter not owned by the model"
                )
            if name in seen:
                raise ValueError("optimizer contains a duplicate model parameter")
            seen.add(name)
            names.append(name)
        groups.append(tuple(names))
    return tuple(groups)


def _validate_remaining_training_iterations(
    config: RLTrainConfig,
    progress: TrainingProgress,
) -> None:
    if progress.completed_iterations >= config.collection.training_iterations:
        raise ValueError(
            "checkpoint has already reached the configured training iteration "
            "budget: "
            f"{progress.completed_iterations} >= "
            f"{config.collection.training_iterations}"
        )


def _training_progress_summary(
    config: RLTrainConfig,
    progress: TrainingProgress,
) -> dict[str, Any]:
    return {
        "mode": config.resume.mode,
        "completed_iterations": progress.completed_iterations,
        "total_optimizer_updates": progress.total_optimizer_updates,
        "ppo_optimizer_updates": progress.ppo_optimizer_updates,
        "state_path": (
            deck_records.display_path(progress.state_path)
            if progress.state_path is not None
            else None
        ),
        "max_staleness_migration_from": (config.resume.max_staleness_migration_from),
        "effective_batch_size_migration_from": (
            config.resume.effective_batch_size_migration_from
        ),
        "curriculum_assignment_block_size_migration_from": (
            config.resume.curriculum_assignment_block_size_migration_from
        ),
        "curriculum_assignment_lanes_migration_from_disabled": (
            config.resume.curriculum_assignment_lanes_migration_from_disabled
        ),
        "learner_shuffle_each_epoch_migration_from": (
            config.resume.learner_shuffle_each_epoch_migration_from
        ),
    }


def _restored_ppo_optimizer_updates(
    config: RLTrainConfig,
    progress: TrainingProgress,
) -> int:
    """Resolve the dedicated PPO clock used by transition distillation."""
    restored = progress.ppo_optimizer_updates
    if restored is None:
        if config.ppo.transition_distillation.optimizer_updates > 0:
            raise ValueError(
                "transition distillation exact resume is missing the dedicated "
                "ppo_optimizer_updates clock"
            )
        return progress.total_optimizer_updates
    if restored > progress.total_optimizer_updates:
        raise ValueError(
            "restored PPO optimizer clock exceeds the aggregate optimizer clock: "
            f"{restored} > {progress.total_optimizer_updates}"
        )
    return restored


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    config: RLTrainConfig,
    planned_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    if len(optimizer.param_groups) > 1:
        final_ratio = (
            config.optimizer.final_learning_rate / config.optimizer.learning_rate
        )

        def group_cosine_multiplier(group_index: int) -> Callable[[int], float]:
            def cosine_multiplier(_step: int) -> float:
                group = optimizer.param_groups[group_index]
                active_updates = int(group.get("scheduler_active_updates", _step))
                horizon = int(group.get("scheduler_horizon_updates", planned_steps))
                progress = min(1.0, max(0.0, active_updates / max(1, horizon)))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return final_ratio + (1.0 - final_ratio) * cosine

            return cosine_multiplier

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                group_cosine_multiplier(index)
                for index in range(len(optimizer.param_groups))
            ],
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, planned_steps),
        eta_min=config.optimizer.final_learning_rate,
    )


def _optimizer_summary(
    config: RLTrainConfig,
    *,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    planned_steps: int,
) -> dict[str, Any]:
    return {
        "learning_rate": config.optimizer.learning_rate,
        "final_learning_rate": config.optimizer.final_learning_rate,
        "current_learning_rate": _current_learning_rate(optimizer),
        "parameter_groups": [
            {
                "name": group.get("name", f"group_{index}"),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameters": len(group["params"]),
                "expert_id": group.get("expert_id"),
                "scheduler_active_updates": group.get("scheduler_active_updates"),
                "scheduler_horizon_updates": group.get("scheduler_horizon_updates"),
            }
            for index, group in enumerate(optimizer.param_groups)
        ],
        "lr_scheduler": {
            "type": "cosine",
            "planned_steps": planned_steps,
            "budget_source": (
                "explicit_effective_updates"
                if config.optimizer.scheduler_total_updates is not None
                else "computed_effective_updates"
            ),
            "steps": int(lr_scheduler.last_epoch),
        },
    }


def _current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def _resolve_device(raw_device: str) -> torch.device:
    normalized = raw_device.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    elif normalized == "gpu":
        normalized = "cuda"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {raw_device}")
    return device


def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return cast(dict[str, Any], raw)


def _read_optional_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_summary(path)


def _load_hydra_config(config_name: str) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir.resolve()),
    ):
        hydra_config = compose(config_name=config_name)
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = cast(dict[str, Any], raw_config)
    config.pop("hydra", None)
    return config
