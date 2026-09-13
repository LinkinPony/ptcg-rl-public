"""Centralized batched rollout inference service primitives."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from torch import Tensor

from ptcg_rl.agent.search.proposal_generation import (
    PlannerProposalBatchResult,
    PlannerProposalDeadlineError,
    PlannerProposalDecisionResult,
    PlannerProposalSearchLimits,
)
from ptcg_rl.context import (
    PublicEventBatch,
    concatenate_public_event_batches,
    validate_public_event_batch,
)
from ptcg_rl.decks.batch import (
    DeckBatch,
    concatenate_deck_batches,
    pad_deck_batch,
)
from ptcg_rl.model import (
    LEGACY_STATE_ENCODER_MISSING_KEYS,
    AgentNetworkConfig,
    OptionBatch,
    RecurrentPolicyState,
    StateBatch,
    concatenate_recurrent_policy_states,
)
from ptcg_rl.model.network import (
    PlannerCandidateEvaluation,
    SampleDecodeTensorTrace,
    SampleDecodeTrace,
)
from ptcg_rl.model.policy import actions_from_decode_tensors
from ptcg_rl.rl.amortized_policy_iteration.contracts import BehaviorKind
from ptcg_rl.rl.curriculum import FrozenPoolMember, read_frozen_pool_state
from ptcg_rl.rl.frozen_pool import (
    FrozenPolicyPool,
    FrozenPolicyPoolUpdate,
    PreparedFrozenPolicyPoolUpdate,
)
from ptcg_rl.rl.inference_planner_proposals import (
    PlannerProposalRequestPayload,
    PlannerProposalResponsePayload,
    evaluate_planner_proposal_group,
    planner_proposal_predictor,
)
from ptcg_rl.rl.inference_publication import write_inference_served_policy
from ptcg_rl.rl.inference_scheduling import (
    fair_schedule_requests,
    inference_stage_row_cap,
    scheduled_stage_batches,
)
from ptcg_rl.rl.inference_snapshot_router import InferencePolicySnapshotRouter
from ptcg_rl.rl.learner import PublishedWeights, read_latest_published_weights
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.model_publication import validate_model_fingerprint
from ptcg_rl.rl.recurrent_runtime import (
    PolicyArtifactIdentity,
    RecurrentDecodeResult,
    RecurrentInferenceBatch,
    RecurrentSequenceIdentity,
    validate_policy_artifact_identity,
)
from ptcg_rl.rl.shared_weights import SharedMemoryWeightLoader

InferenceRequestType = Literal[
    "decode",
    "value",
    "root_information_value",
    "planner_candidates",
    "planner_proposals",
    "planner_context_release",
    "recurrent_release",
]
InferenceRequestPurpose = Literal["behavior", "planner_behavior", "teacher"]


class RecurrentStaleGameRecyclingConfig(BaseModel):
    """Operational polling controls for retiring learner-stale live games."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    poll_interval_seconds: float = 1.0

    @field_validator("poll_interval_seconds")
    @classmethod
    def valid_poll_interval_seconds(cls, value: float) -> float:
        """Require a finite interval so actors never poll storage per step."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("recurrent stale-game poll interval must be positive")
        return value


class InferenceServerConfig(BaseModel):
    """Config for draining and aggregating rollout inference requests."""

    model_config = ConfigDict(extra="forbid")

    max_batch: int = 512
    max_wait_ms: float = 2.0
    min_batch_decisions: int = 0
    request_prefetch_capacity: int = 32
    concurrent_policy_streams: bool = True
    compile_model: bool = False
    bucketize: bool = False
    graph_decode: bool = False
    graph_warmup_steps: int = 2
    graph_max_captures: int = 2
    graph_capture_idle_replays: int = 64
    packed_mixed_route: bool = False
    packed_mixed_route_policy_ids: tuple[str, ...] = ("candidate",)
    graph_roster_layout: bool = False
    graph_roster_slots_per_route: int = 0
    graph_roster_layout_min_rows: int = 0
    graph_roster_layout_policy_ids: tuple[str, ...] = ("candidate",)
    graph_sparse_eager_policy_ids: tuple[str, ...] = ()
    bucket_batch_sizes: tuple[int, ...] = (256, 512, 1024)
    bucket_token_sizes: tuple[int, ...] = (128, 192, 256)
    bucket_option_sizes: tuple[int, ...] = (32, 64, 128)
    bucket_attachment_sizes: tuple[int, ...] = (32, 64, 128)
    bucket_max_select_steps: int = 6
    max_planner_candidate_rows: int = 4_096
    max_planner_proposal_rows: int = 256
    max_root_information_rows: int = 256
    planner_context_ttl_seconds: float = 2.0
    recurrent_max_resident_snapshots: int = 2
    recurrent_snapshot_min_version_gap: int = 1
    recurrent_max_sequence_leases: int = 4096
    recurrent_replay_cache_capacity: int = 4096
    recurrent_stale_game_recycling: RecurrentStaleGameRecyclingConfig = (
        RecurrentStaleGameRecyclingConfig()
    )
    purpose_schedule_weights: tuple[int, int, int] = (4, 2, 1)
    purpose_aging_seconds: float = 0.05

    @field_validator("max_batch")
    @classmethod
    def valid_max_batch(cls, value: int) -> int:
        """Reject invalid aggregate batch sizes."""
        if value <= 0:
            raise ValueError("max_batch must be positive")
        return value

    @field_validator(
        "max_planner_candidate_rows",
        "max_planner_proposal_rows",
        "max_root_information_rows",
        "recurrent_max_sequence_leases",
        "recurrent_replay_cache_capacity",
    )
    @classmethod
    def valid_planner_microbatch_rows(cls, value: int) -> int:
        """Reject unusable planner-stage microbatch caps."""
        if value <= 0:
            raise ValueError("planner microbatch row limits must be positive")
        return value

    @field_validator("recurrent_snapshot_min_version_gap")
    @classmethod
    def valid_recurrent_snapshot_gap(cls, value: int) -> int:
        """Require recurrent candidate publication to remain reachable."""
        if value <= 0:
            raise ValueError("recurrent_snapshot_min_version_gap must be positive")
        return value

    @field_validator("recurrent_max_resident_snapshots")
    @classmethod
    def valid_recurrent_snapshot_capacity(cls, value: int) -> int:
        """Keep one serving and one draining recurrent generation available."""
        if value < 2:
            raise ValueError("recurrent_max_resident_snapshots must be at least 2")
        return value

    @field_validator("max_wait_ms")
    @classmethod
    def valid_max_wait_ms(cls, value: float) -> float:
        """Reject invalid wait windows."""
        if value < 0.0:
            raise ValueError("max_wait_ms must be non-negative")
        return value

    @field_validator("planner_context_ttl_seconds")
    @classmethod
    def valid_planner_context_ttl_seconds(cls, value: float) -> float:
        """Require a finite positive abandoned-root cleanup interval."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("planner_context_ttl_seconds must be finite and positive")
        return value

    @field_validator("purpose_schedule_weights")
    @classmethod
    def valid_purpose_schedule_weights(
        cls,
        value: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """Require positive planner/behavior/teacher scheduling shares."""
        if any(weight <= 0 for weight in value):
            raise ValueError("purpose schedule weights must be positive")
        return value

    @field_validator("purpose_aging_seconds")
    @classmethod
    def valid_purpose_aging_seconds(cls, value: float) -> float:
        """Require a finite positive wait bound for aging promotion."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("purpose_aging_seconds must be finite and positive")
        return value

    @field_validator("min_batch_decisions")
    @classmethod
    def valid_min_batch_decisions(cls, value: int) -> int:
        """Reject invalid minimum drain batch sizes."""
        if value < 0:
            raise ValueError("min_batch_decisions must be non-negative")
        return value

    @field_validator("request_prefetch_capacity")
    @classmethod
    def valid_request_prefetch_capacity(cls, value: int) -> int:
        """Reject invalid async-worker request prefetch capacities."""
        if value < 0:
            raise ValueError("request_prefetch_capacity must be non-negative")
        return value

    @field_validator("graph_warmup_steps", "graph_max_captures")
    @classmethod
    def valid_graph_warmup_steps(cls, value: int) -> int:
        """Reject invalid graph warmup counts."""
        if value < 0:
            raise ValueError("graph limits must be non-negative")
        return value

    @field_validator("graph_capture_idle_replays")
    @classmethod
    def valid_graph_capture_idle_replays(cls, value: int) -> int:
        """Require a positive idle interval before replacing a graph pool."""
        if value <= 0:
            raise ValueError("graph_capture_idle_replays must be positive")
        return value

    @field_validator("graph_roster_slots_per_route", "graph_roster_layout_min_rows")
    @classmethod
    def valid_graph_roster_sizes(cls, value: int) -> int:
        """Reject negative fixed-layout geometry."""
        if value < 0:
            raise ValueError("graph roster layout sizes must be non-negative")
        return value

    @field_validator(
        "packed_mixed_route_policy_ids",
        "graph_roster_layout_policy_ids",
        "graph_sparse_eager_policy_ids",
    )
    @classmethod
    def valid_graph_roster_policy_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Normalize and reject ambiguous graph-serving policy identities."""
        cleaned = tuple(policy_id.strip() for policy_id in value)
        if any(not policy_id for policy_id in cleaned):
            raise ValueError("graph-serving policy IDs must be non-empty")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("graph-serving policy IDs must be unique")
        return cleaned

    @model_validator(mode="after")
    def valid_graph_roster_layout(self) -> InferenceServerConfig:
        """Require graph/bucket support and fixed slots for roster layouts."""
        if self.packed_mixed_route:
            if not self.graph_decode:
                raise ValueError("packed mixed-route serving requires graph_decode")
            if not self.packed_mixed_route_policy_ids:
                raise ValueError(
                    "packed mixed-route serving requires at least one policy ID"
                )
        if not self.graph_roster_layout:
            return self
        if not self.graph_decode or not self.bucketize:
            raise ValueError("graph roster layout requires graph_decode and bucketize")
        if self.graph_roster_slots_per_route <= 0:
            raise ValueError("graph roster layout requires positive route slots")
        if not self.graph_roster_layout_policy_ids:
            raise ValueError("graph roster layout requires at least one policy ID")
        return self

    @field_validator(
        "bucket_batch_sizes",
        "bucket_token_sizes",
        "bucket_option_sizes",
        "bucket_attachment_sizes",
    )
    @classmethod
    def valid_bucket_sizes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reject empty, non-positive, or unsorted bucket size lists."""
        if not value:
            raise ValueError("bucket sizes must be non-empty")
        if any(size <= 0 for size in value):
            raise ValueError("bucket sizes must be positive")
        if tuple(sorted(set(value))) != value:
            raise ValueError("bucket sizes must be strictly increasing")
        return value

    @field_validator("bucket_max_select_steps")
    @classmethod
    def valid_bucket_max_select_steps(cls, value: int) -> int:
        """Reject invalid static decode caps."""
        if value < 0:
            raise ValueError("bucket_max_select_steps must be non-negative")
        return value


class InferenceClientConfig(BaseModel):
    """Config for actor-side request/response inference clients."""

    model_config = ConfigDict(extra="forbid")

    response_timeout_seconds: float = 30.0
    response_retries: int = 1
    max_inflight_batch_decisions: int = 0

    @field_validator("response_timeout_seconds")
    @classmethod
    def valid_response_timeout(cls, value: float) -> float:
        """Reject invalid response wait timeouts."""
        if value <= 0.0:
            raise ValueError("response_timeout_seconds must be positive")
        return value

    @field_validator("response_retries")
    @classmethod
    def valid_response_retries(cls, value: int) -> int:
        """Reject invalid response retry counts."""
        if value < 0:
            raise ValueError("response_retries must be non-negative")
        return value

    @field_validator("max_inflight_batch_decisions")
    @classmethod
    def valid_max_inflight_batch_decisions(cls, value: int) -> int:
        """Allow zero to preserve one-request-per-route actor behavior."""
        if value < 0:
            raise ValueError("max_inflight_batch_decisions must be non-negative")
        return value


@dataclass(frozen=True)
class InferenceRequest:
    """One actor-to-server batched policy inference request."""

    actor_id: str
    request_id: int
    policy_id: str
    states: StateBatch
    options: OptionBatch | None
    decks: DeckBatch
    temperature: float = 1.0
    created_at: float = 0.0
    server_received_at: float = 0.0
    request_type: InferenceRequestType = "decode"
    purpose: InferenceRequestPurpose = "behavior"
    deadline_monotonic: float = 0.0
    model_version_lease: int | None = None
    tensor_schema_fingerprint: str = ""
    actor_relations: Tensor | None = None
    endpoints: Tensor | None = None
    belief_summaries: Tensor | None = None
    candidate_actions: tuple[tuple[tuple[int, ...], ...], ...] | None = None
    candidate_features: tuple[Tensor, ...] | None = None
    ordered_rows: Tensor | None = None
    planner_proposal_request: PlannerProposalRequestPayload | None = None
    planner_context_handles: tuple[str, ...] = ()
    retain_planner_context: bool | None = None
    recurrent: RecurrentInferenceBatch | None = None
    actor_incarnation: int = 0

    def __post_init__(self) -> None:
        """Reject actor requests whose persistent inputs are row-misaligned."""
        if self.actor_incarnation < 0:
            raise ValueError("actor incarnation must be non-negative")
        if self.request_type not in (
            "decode",
            "value",
            "root_information_value",
            "planner_candidates",
            "planner_proposals",
            "planner_context_release",
        ):
            raise ValueError(f"unsupported inference request type: {self.request_type}")
        if self.purpose not in ("behavior", "planner_behavior", "teacher"):
            raise ValueError(f"unsupported inference request purpose: {self.purpose}")
        if not math.isfinite(self.deadline_monotonic):
            raise ValueError("inference request deadline must be finite")
        if (
            self.purpose in ("teacher", "planner_behavior")
            and self.deadline_monotonic <= 0.0
        ):
            raise ValueError("deadline-aware inference requests require a deadline")
        if self.purpose == "behavior" and self.deadline_monotonic < 0.0:
            raise ValueError("behavior inference request deadline must be non-negative")
        if self.purpose == "planner_behavior":
            if not _is_sha256(self.tensor_schema_fingerprint):
                raise ValueError(
                    "planner_behavior requires a tensor schema fingerprint"
                )
            if self.model_version_lease is None:
                if self.request_type != "decode":
                    raise ValueError(
                        "only planner root decode may acquire an unbound lease"
                    )
            elif self.model_version_lease < 0:
                raise ValueError("planner model-version lease must be non-negative")
        elif self.model_version_lease is not None or self.tensor_schema_fingerprint:
            raise ValueError("only planner_behavior may bind lease/schema identity")
        state_batch_size = int(self.states.card_ids.shape[0])
        uses_options = self.request_type in (
            "decode",
            "planner_candidates",
            "planner_proposals",
        )
        if uses_options and self.options is None:
            raise ValueError("decode and planner requests require options")
        if not uses_options and self.options is not None:
            raise ValueError("value inference requests must not include options")
        if (
            self.options is not None
            and int(self.options.valid_options.shape[0]) != state_batch_size
        ):
            raise ValueError("inference request states and options must align")
        if len(self.decks) != state_batch_size:
            raise ValueError("inference request decks and states must align")
        if self.recurrent is not None:
            if self.request_type != "decode":
                raise ValueError("recurrent inputs require decode request type")
            validate_public_event_batch(self.recurrent.public_events)
            if self.recurrent.batch_size != state_batch_size:
                raise ValueError("recurrent inputs and inference states must align")
            if (
                self.recurrent.public_events.event_types.device
                != self.states.card_ids.device
            ):
                raise ValueError(
                    "recurrent inputs and inference states use different devices"
                )
            signatures = tuple(
                sequence.exact_deck_signature for sequence in self.recurrent.sequences
            )
            if signatures != self.decks.signatures:
                raise ValueError("recurrent sequence decks and inference decks differ")
        root_value_tensors = (
            self.actor_relations,
            self.endpoints,
            self.belief_summaries,
        )
        if self.request_type == "root_information_value":
            if self.purpose != "planner_behavior":
                raise ValueError(
                    "root-information values require planner_behavior purpose"
                )
            if any(tensor is None for tensor in root_value_tensors):
                raise ValueError("root-information value inputs are incomplete")
            relations = cast(Tensor, self.actor_relations)
            endpoints = cast(Tensor, self.endpoints)
            beliefs = cast(Tensor, self.belief_summaries)
            if relations.shape != (state_batch_size,) or relations.dtype != torch.long:
                raise ValueError("actor_relations has the wrong shape or dtype")
            if endpoints.shape != (state_batch_size,) or endpoints.dtype != torch.long:
                raise ValueError("endpoints has the wrong shape or dtype")
            if (
                beliefs.ndim != 2
                or int(beliefs.shape[0]) != state_batch_size
                or beliefs.dtype != torch.float32
                or not bool(torch.isfinite(beliefs).all().item())
            ):
                raise ValueError("belief_summaries has the wrong shape or dtype")
        elif any(tensor is not None for tensor in root_value_tensors):
            raise ValueError(
                "root-information tensors require root_information_value request type"
            )
        candidate_payload = (self.candidate_actions, self.candidate_features)
        if self.request_type == "planner_candidates":
            if self.purpose != "planner_behavior":
                raise ValueError(
                    "planner candidate evaluation requires planner_behavior purpose"
                )
            if any(item is None for item in candidate_payload):
                raise ValueError("planner candidate inputs are incomplete")
            action_groups = cast(
                tuple[tuple[tuple[int, ...], ...], ...],
                self.candidate_actions,
            )
            feature_groups = cast(tuple[Tensor, ...], self.candidate_features)
            if self.ordered_rows is None:
                raise ValueError("planner candidate ordering semantics are required")
            if (
                self.ordered_rows.shape != (state_batch_size,)
                or self.ordered_rows.dtype != torch.bool
            ):
                raise ValueError("planner candidate ordered_rows are misaligned")
            if (
                len(self.planner_context_handles) != state_batch_size
                or len(set(self.planner_context_handles)) != state_batch_size
            ):
                raise ValueError("planner candidate context handles are misaligned")
            if (
                len(action_groups) != state_batch_size
                or len(feature_groups) != state_batch_size
            ):
                raise ValueError("planner candidate groups must align with states")
            for actions, features in zip(
                action_groups,
                feature_groups,
                strict=True,
            ):
                if not actions:
                    raise ValueError("planner candidate groups must not be empty")
                if (
                    features.ndim != 2
                    or int(features.shape[0]) != len(actions)
                    or not bool(torch.isfinite(features).all().item())
                ):
                    raise ValueError("planner candidate features are misaligned")
        elif any(item is not None for item in candidate_payload):
            raise ValueError(
                "planner candidate inputs require planner_candidates request type"
            )
        elif self.ordered_rows is not None:
            raise ValueError("ordered_rows require planner_candidates request type")
        elif (
            self.planner_context_handles
            and self.request_type != "planner_context_release"
        ):
            raise ValueError(
                "planner context handles require planner_candidates request type"
            )
        if self.retain_planner_context is not None:
            if self.request_type != "decode" or self.purpose != "planner_behavior":
                raise ValueError(
                    "planner context retention applies only to planner root decode"
                )
            if self.retain_planner_context and self.model_version_lease is not None:
                raise ValueError(
                    "lease-bound continuation decode cannot retain root context"
                )
        if self.request_type == "planner_proposals":
            if self.purpose != "planner_behavior":
                raise ValueError(
                    "planner proposal generation requires planner_behavior purpose"
                )
            if self.planner_proposal_request is None:
                raise ValueError("planner proposal request payload is missing")
            self.planner_proposal_request.validate(batch_size=state_batch_size)
        elif self.planner_proposal_request is not None:
            raise ValueError(
                "planner proposal payload requires planner_proposals request type"
            )
        if self.request_type == "planner_context_release":
            if self.purpose != "planner_behavior":
                raise ValueError("planner context release requires planner purpose")
            if len(self.planner_context_handles) != state_batch_size:
                raise ValueError("planner release context handles are misaligned")

    @property
    def batch_size(self) -> int:
        """Return decisions contained in this request."""
        return int(self.states.card_ids.shape[0])


@dataclass(frozen=True)
class RecurrentReleaseRequest:
    """Release immutable server-side sequence leases without fake model inputs."""

    actor_id: str
    request_id: int
    policy_id: str
    sequences: tuple[RecurrentSequenceIdentity, ...]
    expected_artifact: PolicyArtifactIdentity | None
    allow_missing: bool = False
    created_at: float = 0.0
    server_received_at: float = 0.0
    request_type: Literal["recurrent_release"] = "recurrent_release"
    purpose: Literal["behavior"] = "behavior"
    deadline_monotonic: float = 0.0
    actor_incarnation: int = 0

    def __post_init__(self) -> None:
        """Require one exact, non-duplicated release owner set."""
        if self.actor_incarnation < 0:
            raise ValueError("actor incarnation must be non-negative")
        if self.request_id < 0:
            raise ValueError("recurrent release request_id must be non-negative")
        if not self.actor_id.strip() or not self.policy_id.strip():
            raise ValueError("recurrent release routing identities must be non-empty")
        if not self.sequences:
            raise ValueError("recurrent release must contain at least one sequence")
        if len(set(self.sequences)) != len(self.sequences):
            raise ValueError("recurrent release contains duplicate sequences")
        if self.expected_artifact is None and not self.allow_missing:
            raise ValueError("strict recurrent release requires an artifact")
        if self.expected_artifact is not None and not (
            self.expected_artifact.compatibility.public_event_schema_fingerprint
        ):
            raise ValueError("recurrent release requires a recurrent artifact")

    @property
    def batch_size(self) -> int:
        """Return sequence leases named by this release."""
        return len(self.sequences)


InferenceQueueRequest = InferenceRequest | RecurrentReleaseRequest


@dataclass(frozen=True)
class InferenceResponse:
    """One server-to-actor inference response."""

    actor_id: str
    request_id: int
    policy_id: str
    policy_version: int
    actions: tuple[tuple[int, ...], ...]
    action_logprobs: Tensor
    values: Tensor
    token_logprobs: Tensor | None = None
    prefix_values: Tensor | None = None
    token_mask: Tensor | None = None
    stop_sampled: Tensor | None = None
    server_received_at: float = 0.0
    server_sample_started_at: float = 0.0
    server_sample_finished_at: float = 0.0
    server_put_at: float = 0.0
    request_type: InferenceRequestType = "decode"
    error_type: str | None = None
    error_message: str = ""
    requested_batch_size: int = 0
    planner_base_action_logprobs: Tensor | None = None
    planner_proposal_action_logprobs: Tensor | None = None
    planner_reranker_residuals: Tensor | None = None
    planner_candidate_counts: tuple[int, ...] = ()
    planner_proposal_response: PlannerProposalResponsePayload | None = None
    planner_context_handles: tuple[str, ...] = ()
    model_fingerprint: str = ""
    proposal_version: int = 0
    planner_fallback_reason: str = ""
    recurrent_result: RecurrentDecodeResult | None = None
    released_recurrent_sequences: int = 0
    released_sequence_identities: tuple[RecurrentSequenceIdentity, ...] = ()
    served_policy_artifact: PolicyArtifactIdentity | None = None
    actor_incarnation: int = 0

    @property
    def batch_size(self) -> int:
        """Return decisions contained in this response."""
        if self.error_type is not None:
            return int(self.requested_batch_size)
        if self.request_type == "planner_candidates":
            return len(self.planner_candidate_counts)
        if self.request_type == "planner_proposals":
            payload = self.planner_proposal_response
            return 0 if payload is None else len(payload.decisions)
        if self.request_type == "recurrent_release":
            return len(self.released_sequence_identities)
        if self.request_type != "decode":
            return int(self.values.shape[0])
        return len(self.actions)


@dataclass(frozen=True)
class RemoteInferenceHandle:
    """Actor-side handle for one in-flight remote inference request."""

    actor_id: str
    request_id: int
    policy_id: str
    batch_size: int
    attempts: int = 0
    request_type: InferenceRequestType = "decode"
    deadline_monotonic: float = 0.0
    recurrent_request: RecurrentInferenceBatch | None = None
    recurrent_release_request: RecurrentReleaseRequest | None = None
    actor_incarnation: int = 0


@dataclass(frozen=True)
class InferenceStepStats:
    """Counters from one inference server drain/serve step."""

    requests: int
    decisions: int
    policy_batches: int
    responses: int
    expired_teacher_requests: int
    expired_teacher_decisions: int
    expired_planner_requests: int
    expired_planner_decisions: int
    rejected_planner_lease_requests: int
    rejected_planner_lease_decisions: int
    rejected_oversized_requests: int
    rejected_oversized_decisions: int
    drained_by_policy: Mapping[str, int]
    request_batch_histogram: Mapping[int, int]
    policy_batch_histogram: Mapping[int, int]
    shape_histogram: Mapping[str, int]
    decode_step_histogram: Mapping[int, int]
    service_time_histogram_ms: Mapping[str, int]
    bucket_histogram: Mapping[str, int]
    bucket_fallback_histogram: Mapping[str, int]
    bucket_slot_totals: Mapping[str, int]
    serving_path_histogram: Mapping[str, int]
    graph_decode_stats: Mapping[str, int]
    coalescing_stats: Mapping[str, float | int | bool]
    mean_request_latency_ms: float
    p95_request_latency_ms: float
    request_latencies_ms: tuple[float, ...]
    server_queue_latencies_ms: tuple[float, ...]
    server_forward_latencies_ms: tuple[float, ...]
    response_put_latencies_ms: tuple[float, ...]
    drain_seconds: float
    sample_seconds: float
    response_seconds: float
    step_seconds: float


@dataclass(frozen=True)
class _DrainedInferenceRequests:
    """Live requests and stale auxiliary work observed during one drain."""

    requests: tuple[InferenceQueueRequest, ...]
    expired_teacher: tuple[InferenceRequest, ...]
    expired_planner: tuple[InferenceRequest, ...]
    oversized: tuple[InferenceQueueRequest, ...]
    coalescing_stats: Mapping[str, float | int | bool]


_DRAIN_LOOKAHEAD_LOCK = threading.Lock()
_DRAIN_LOOKAHEAD: dict[int, tuple[object, deque[InferenceQueueRequest]]] = {}


@dataclass(frozen=True)
class InferencePolicyRegistrySync:
    """Summary from syncing candidate and frozen policies."""

    loaded_weight_version: int | None
    frozen_update: FrozenPolicyPoolUpdate | None
    policy_ids: tuple[str, ...]
    deferred_weight_version: int | None = None
    deferred_weight_reason: str | None = None
    coalesced_weight_versions: int = 0
    snapshot_min_version_gap: int = 1
    snapshot_pool: Mapping[str, Any] | None = None
    frozen_load: Mapping[str, Any] | None = None


@dataclass
class _FrozenPolicyLoadTask:
    """One background frozen-pool generation build."""

    signature: tuple[int, int, int]
    member_signature: tuple[tuple[str, str, bool], ...]
    policy_ids: tuple[str, ...]
    started_at: float
    started_at_utc: str
    thread: threading.Thread | None = None
    prepared: PreparedFrozenPolicyPoolUpdate | None = None
    error: Exception | None = None
    finished_at: float | None = None


def _utc_timestamp() -> str:
    """Return one compact UTC timestamp for runtime telemetry."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _frozen_member_runtime_signature(
    members: Sequence[FrozenPoolMember],
) -> tuple[tuple[str, str, bool], ...]:
    """Return only the frozen fields that can change inference routes."""
    return tuple(
        sorted(
            (
                str(member.opponent_id),
                str(member.checkpoint_path),
                bool(member.recurrent),
            )
            for member in members
        )
    )


class InferencePolicy(Protocol):
    """Policy interface used by the inference server."""

    def sample_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        """Sample actions, log-probs, and value predictions."""


class _ValueInferencePolicy(Protocol):
    """Optional value-only policy surface used by engine-teacher leaves."""

    def predict_values(
        self,
        states: StateBatch,
        decks: DeckBatch,
    ) -> Tensor:
        """Predict root values without autoregressively decoding actions."""


class _RootInformationValueInferencePolicy(Protocol):
    """Optional root-perspective semantic leaf value surface."""

    def predict_root_information_values(
        self,
        states: StateBatch,
        decks: DeckBatch,
        *,
        actor_relations: Tensor,
        endpoints: Tensor,
        belief_summaries: Tensor,
    ) -> Tensor:
        """Predict root-perspective values for one unique-leaf microbatch."""


class _PlannerCandidateInferencePolicy(Protocol):
    """Optional dedicated schema-9 planner candidate evaluation surface."""

    def evaluate_planner_candidates(
        self,
        states: StateBatch,
        options: OptionBatch,
        candidate_actions: Sequence[Sequence[Sequence[int]]],
        candidate_features: Sequence[Tensor],
        *,
        ordered_rows: Tensor,
        decks: DeckBatch,
        planner_context_handles: Sequence[str],
        model_version_lease: int,
        tensor_schema_fingerprint: str,
    ) -> PlannerCandidateEvaluation:
        """Evaluate ragged candidate supports without schema-8 weights."""


class _TensorDecodePolicy(Protocol):
    """Optional tensor-only decode interface for stream scheduling."""

    def sample_decode_tensors(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
        max_select_steps: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample actions as tensor traces without materializing Python tuples."""


class _TraceDecodePolicy(Protocol):
    """Optional materialized decode interface carrying token-level evidence."""

    def sample_decode_with_trace(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
    ) -> SampleDecodeTrace:
        """Sample actions and retain token behavior evidence."""


class _TensorTraceDecodePolicy(Protocol):
    """Optional tensor decode interface carrying token-level evidence."""

    def sample_decode_tensors_with_trace(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
        max_select_steps: int,
    ) -> SampleDecodeTensorTrace:
        """Sample tensor actions and retain token behavior evidence."""


@dataclass(frozen=True)
class _BucketPlan:
    """Static padded shape selected for one policy group."""

    batch_size: int
    tokens: int
    options: int
    attachments: int
    max_select_steps: int

    @property
    def key(self) -> str:
        """Return a compact histogram key for this bucket."""
        return (
            f"B{self.batch_size}/T{self.tokens}/O{self.options}"
            f"/S{self.max_select_steps}/A{self.attachments}"
        )


@dataclass(frozen=True)
class _PolicyGroupSample:
    """Result and routing metadata from sampling one policy group."""

    actions: tuple[tuple[int, ...], ...]
    logprobs: Tensor
    values: Tensor
    token_logprobs: Tensor | None
    prefix_values: Tensor | None
    token_mask: Tensor | None
    stop_sampled: Tensor | None
    planner_context_handles: tuple[str, ...]
    served_policy_version: int | None
    served_model_fingerprint: str
    served_proposal_version: int | None
    planner_fallback_reason: str
    bucket_key: str | None
    bucket_fallback_reason: str | None
    bucket_slot_totals: Mapping[str, int]


@dataclass(frozen=True)
class _PreparedPolicyGroup:
    """Policy group with concatenated tensors ready for sampling."""

    policy_id: str
    rows: tuple[_InferenceRequestRow, ...]
    policy: InferencePolicy
    states: StateBatch
    options: OptionBatch
    decks: DeckBatch
    shape: tuple[int, int, int, int]
    max_select_steps: int
    temperature: float
    bucket_key: str | None
    bucket_fallback_reason: str | None
    bucket_slot_totals: Mapping[str, int]
    sample_indices: tuple[int, ...]
    serving_path: str = "mixed_eager"


@dataclass(frozen=True)
class _RosterLayoutRoute:
    """One fixed private route and the canonical deck used for padding."""

    signature: str
    module_key: str
    canonical_card_ids: tuple[int, ...]


@dataclass(frozen=True)
class _TensorPolicyGroupSample:
    """Tensor decode traces before Python action materialization."""

    choice_indices: Tensor
    append_masks: Tensor
    logprobs: Tensor
    values: Tensor
    token_logprobs: Tensor | None = None
    prefix_values: Tensor | None = None
    token_mask: Tensor | None = None
    stop_sampled: Tensor | None = None


@dataclass(frozen=True)
class _CudaStreamSlot:
    """Reusable CUDA stream timing resources for one policy-group slot."""

    stream: Any
    started: Any
    finished: Any


@dataclass(frozen=True)
class _ServedPolicyGroup:
    """One sampled policy group plus timing and routing metadata."""

    policy_id: str
    rows: tuple[_InferenceRequestRow, ...]
    policy_version: int
    shape: tuple[int, int, int, int]
    sample: _PolicyGroupSample
    sample_seconds: float
    sample_started_at: float
    sample_finished_at: float
    serving_path: str


@dataclass(frozen=True)
class _ServedValueGroup:
    """One value-only policy group plus timing and routing metadata."""

    policy_id: str
    rows: tuple[_InferenceRequestRow, ...]
    policy_version: int
    values: Tensor
    state_token_width: int
    sample_seconds: float
    sample_started_at: float
    sample_finished_at: float


@dataclass(frozen=True)
class _ServedPlannerCandidateGroup:
    """One candidate-evaluation microbatch with decision-row scatter data."""

    policy_id: str
    rows: tuple[_InferenceRequestRow, ...]
    policy_version: int
    evaluation: PlannerCandidateEvaluation
    state_token_width: int
    sample_seconds: float
    sample_started_at: float
    sample_finished_at: float


@dataclass(frozen=True)
class _ServedPlannerProposalGroup:
    """One cross-actor proposal campaign plus request-level scatter payloads."""

    policy_id: str
    requests: tuple[InferenceRequest, ...]
    policy_version: int
    payloads: tuple[PlannerProposalResponsePayload, ...]
    state_token_width: int
    sample_seconds: float
    sample_started_at: float
    sample_finished_at: float


@dataclass(frozen=True)
class _ServedRecurrentRequest:
    """One completed recurrent decode/release response ready for delivery."""

    request: InferenceQueueRequest
    response: InferenceResponse
    sample_seconds: float
    policy_batch: bool
    policy_batch_size: int = 0
    serving_path: str = "recurrent_eager"


@dataclass(frozen=True)
class _BoundRecurrentRequest:
    """One uncached request bound to an immutable recurrent snapshot."""

    request: InferenceRequest
    router: InferencePolicySnapshotRouter
    policy: Any
    policy_version: int
    artifact: PolicyArtifactIdentity
    first_bind: bool


@dataclass(frozen=True)
class _ServedInferenceStages:
    """Stage results produced in strict request scheduling order."""

    decode: tuple[_ServedPolicyGroup, ...] = ()
    value: tuple[_ServedValueGroup, ...] = ()
    root_value: tuple[_ServedValueGroup, ...] = ()
    context_release: tuple[_ServedValueGroup, ...] = ()
    candidates: tuple[_ServedPlannerCandidateGroup, ...] = ()
    proposals: tuple[_ServedPlannerProposalGroup, ...] = ()
    recurrent: tuple[_ServedRecurrentRequest, ...] = ()
    expired: tuple[InferenceRequest, ...] = ()


@dataclass(frozen=True)
class _InferenceRequestRow:
    """One original request row retained for grouped-output scatter."""

    request: InferenceRequest
    row_index: int


@dataclass(frozen=True)
class _InferenceRowResult:
    """One sampled row plus its route-group timing."""

    policy_version: int
    action: tuple[int, ...]
    action_logprob: Tensor
    value: Tensor
    token_logprobs: Tensor | None
    prefix_values: Tensor | None
    token_mask: Tensor | None
    stop_sampled: Tensor | None
    planner_base_action_logprobs: Tensor | None
    planner_proposal_action_logprobs: Tensor | None
    planner_reranker_residuals: Tensor | None
    planner_proposal_decision: PlannerProposalDecisionResult | None
    planner_base_greedy_action: tuple[int, ...] | None
    planner_context_handle: str | None
    model_fingerprint: str
    proposal_version: int
    planner_fallback_reason: str
    sample_started_at: float
    sample_finished_at: float


@dataclass(frozen=True)
class _PendingRequestTiming:
    """Actor-side timing metadata retained until a response arrives."""

    created_at: float
    client_put_finished_at: float
    request_queue_put_seconds: float
    batch_size: int


@dataclass(frozen=True)
class _PendingInferencePayload:
    """Inputs retained by a remote client for timeout retries."""

    states: StateBatch
    options: OptionBatch | None
    decks: DeckBatch
    temperature: float
    request_type: InferenceRequestType
    actor_relations: Tensor | None = None
    endpoints: Tensor | None = None
    belief_summaries: Tensor | None = None
    model_version_lease: int | None = None
    tensor_schema_fingerprint: str = ""
    deadline_monotonic: float = 0.0
    candidate_actions: tuple[tuple[tuple[int, ...], ...], ...] | None = None
    candidate_features: tuple[Tensor, ...] | None = None
    ordered_rows: Tensor | None = None
    planner_proposal_request: PlannerProposalRequestPayload | None = None
    planner_context_handles: tuple[str, ...] = ()
    retain_planner_context: bool | None = None
    recurrent: RecurrentInferenceBatch | None = None
    original_request: InferenceRequest | None = None


@dataclass(frozen=True)
class _PendingRecurrentReleasePayload:
    """Exact logical release request retained for same-ID retry."""

    original_request: RecurrentReleaseRequest


_PendingRemotePayload = _PendingInferencePayload | _PendingRecurrentReleasePayload


_REMOTE_LATENCY_WINDOW_SIZE = 4096
_REMOTE_ABANDONED_REQUEST_LIMIT = 4096
_NOT_ABANDONED = object()
_RemoteResponseKey = tuple[int, str, int]


@dataclass
class _RemoteLatencyWindow:
    """Keep lifetime mean/count while bounding percentile samples."""

    count: int = 0
    total_ms: float = 0.0
    values: deque[float] = field(
        default_factory=lambda: deque(maxlen=_REMOTE_LATENCY_WINDOW_SIZE)
    )

    def append(self, value: float) -> None:
        """Record one latency sample."""
        normalized = float(value)
        self.count += 1
        self.total_ms += normalized
        self.values.append(normalized)

    def summary(self) -> dict[str, float | int]:
        """Return lifetime mean/count and a bounded-window percentile."""
        return {
            "count": self.count,
            "mean": self.total_ms / float(self.count) if self.count > 0 else 0.0,
            "p95": _percentile(self.values, 0.95),
            "window_count": len(self.values),
        }

    def stage_timing(self) -> dict[str, float | int]:
        """Return lifetime totals in the actor stage-timing shape."""
        return {
            "seconds": self.total_ms / 1000.0,
            "count": self.count,
            "mean_ms": self.total_ms / float(self.count) if self.count > 0 else 0.0,
        }


@dataclass
class _RemoteInferenceClientStats:
    """Aggregate actor-side remote inference IPC latency measurements."""

    submitted_requests: int = 0
    submitted_decisions: int = 0
    received_responses: int = 0
    received_decisions: int = 0
    buffered_responses: int = 0
    abandoned_responses: int = 0
    missing_timing_responses: int = 0
    request_enqueue_ms: _RemoteLatencyWindow = field(
        default_factory=_RemoteLatencyWindow
    )
    server_queue_ms: _RemoteLatencyWindow = field(default_factory=_RemoteLatencyWindow)
    server_forward_ms: _RemoteLatencyWindow = field(
        default_factory=_RemoteLatencyWindow
    )
    response_return_ms: _RemoteLatencyWindow = field(
        default_factory=_RemoteLatencyWindow
    )
    actor_pickup_ms: _RemoteLatencyWindow = field(default_factory=_RemoteLatencyWindow)
    client_response_get_ms: _RemoteLatencyWindow = field(
        default_factory=_RemoteLatencyWindow
    )
    end_to_end_ms: _RemoteLatencyWindow = field(default_factory=_RemoteLatencyWindow)

    def record_submit(self, *, batch_size: int) -> None:
        """Record one submitted request."""
        self.submitted_requests += 1
        self.submitted_decisions += int(batch_size)

    def record_arrival(
        self,
        *,
        response: InferenceResponse,
        timing: _PendingRequestTiming | None,
        client_received_at: float,
        receive_started_at: float | None,
        buffered: bool,
        abandoned: bool = False,
    ) -> None:
        """Record one response arrival at the actor process."""
        self.received_responses += 1
        self.received_decisions += response.batch_size
        if abandoned:
            self.abandoned_responses += 1
            return
        if buffered:
            self.buffered_responses += 1
        if timing is None:
            self.missing_timing_responses += 1
            return
        self.request_enqueue_ms.append(timing.request_queue_put_seconds * 1000.0)
        if response.server_sample_started_at > 0.0:
            self.server_queue_ms.append(
                max(
                    0.0,
                    (response.server_sample_started_at - timing.client_put_finished_at)
                    * 1000.0,
                )
            )
        if (
            response.server_sample_started_at > 0.0
            and response.server_sample_finished_at >= response.server_sample_started_at
        ):
            self.server_forward_ms.append(
                (response.server_sample_finished_at - response.server_sample_started_at)
                * 1000.0
            )
        if response.server_put_at > 0.0:
            response_window_start = response.server_put_at
            if receive_started_at is not None:
                self.actor_pickup_ms.append(
                    max(0.0, (receive_started_at - response.server_put_at) * 1000.0)
                )
                response_window_start = max(response_window_start, receive_started_at)
                self.client_response_get_ms.append(
                    max(0.0, (client_received_at - receive_started_at) * 1000.0)
                )
            self.response_return_ms.append(
                max(0.0, (client_received_at - response_window_start) * 1000.0)
            )
        self.end_to_end_ms.append(
            max(0.0, (client_received_at - timing.created_at) * 1000.0)
        )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly latency summary."""
        return {
            "submitted_requests": self.submitted_requests,
            "submitted_decisions": self.submitted_decisions,
            "received_responses": self.received_responses,
            "received_decisions": self.received_decisions,
            "buffered_responses": self.buffered_responses,
            "abandoned_responses": self.abandoned_responses,
            "missing_timing_responses": self.missing_timing_responses,
            "latency_ms": {
                "request_enqueue": self.request_enqueue_ms.summary(),
                "server_queue": self.server_queue_ms.summary(),
                "server_forward": self.server_forward_ms.summary(),
                "response_return": self.response_return_ms.summary(),
                "actor_pickup": self.actor_pickup_ms.summary(),
                "client_response_get": self.client_response_get_ms.summary(),
                "end_to_end": self.end_to_end_ms.summary(),
            },
        }

    def stage_timings(self) -> dict[str, dict[str, float | int]]:
        """Return latency segments in the actor ``stage_timings`` shape."""
        return {
            "inference_request_enqueue": self.request_enqueue_ms.stage_timing(),
            "inference_server_queue": self.server_queue_ms.stage_timing(),
            "inference_forward": self.server_forward_ms.stage_timing(),
            "inference_response_return": self.response_return_ms.stage_timing(),
            "inference_actor_pickup": self.actor_pickup_ms.stage_timing(),
            "inference_client_response_get": (
                self.client_response_get_ms.stage_timing()
            ),
            "inference_end_to_end": self.end_to_end_ms.stage_timing(),
        }


@dataclass
class RemoteInferenceClientState:
    """Actor-local shared state for remote inference policy clients."""

    response_buffer: MutableMapping[_RemoteResponseKey, InferenceResponse] = field(
        default_factory=dict
    )
    pending_payloads: dict[_RemoteResponseKey, _PendingRemotePayload] = field(
        default_factory=dict
    )
    pending_timings: dict[_RemoteResponseKey, _PendingRequestTiming] = field(
        default_factory=dict
    )
    stats_by_policy: dict[str, _RemoteInferenceClientStats] = field(
        default_factory=dict
    )
    next_request_ids: dict[str, int] = field(default_factory=dict)
    abandoned_request_keys: dict[_RemoteResponseKey, None] = field(default_factory=dict)
    _condition: threading.Condition = field(
        default_factory=threading.Condition,
        init=False,
        repr=False,
    )
    _active_response_readers: set[int] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _arrived_request_keys: set[_RemoteResponseKey] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _completed_request_keys: OrderedDict[_RemoteResponseKey, None] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )

    def allocate_request_id(self, policy_id: str, *, minimum: int) -> int:
        """Allocate a collision-free id across clients sharing one response queue."""
        with self._condition:
            request_id = max(minimum, self.next_request_ids.get(policy_id, minimum))
            self.next_request_ids[policy_id] = request_id + 1
            return request_id

    def register_request(
        self,
        key: _RemoteResponseKey,
        *,
        payload: _PendingRemotePayload,
        timing: _PendingRequestTiming,
    ) -> None:
        """Register retry and timing state before making a request visible."""
        with self._condition:
            if key in self.pending_payloads or key in self.pending_timings:
                raise RuntimeError("duplicate pending inference request")
            self.abandoned_request_keys.pop(key, None)
            self.pending_payloads[key] = payload
            self.pending_timings[key] = timing
            self._completed_request_keys.pop(key, None)

    def finish_request_submit(
        self,
        key: _RemoteResponseKey,
        *,
        timing: _PendingRequestTiming,
        policy_id: str,
        batch_size: int,
    ) -> None:
        """Publish final enqueue timing and count one visible request."""
        with self._condition:
            if key in self.pending_timings:
                self.pending_timings[key] = timing
            self.stats_by_policy.setdefault(
                policy_id,
                _RemoteInferenceClientStats(),
            ).record_submit(batch_size=batch_size)

    def discard_unsubmitted_request(self, key: _RemoteResponseKey) -> None:
        """Remove local state for a request that never reached the queue."""
        with self._condition:
            self.pending_payloads.pop(key, None)
            self.pending_timings.pop(key, None)
            self.response_buffer.pop(key, None)
            self._arrived_request_keys.discard(key)

    def pop_buffered_response(
        self,
        key: _RemoteResponseKey,
    ) -> InferenceResponse | None:
        """Atomically take one response already demultiplexed for ``key``."""
        with self._condition:
            return self.response_buffer.pop(key, None)

    def pop_response_or_claim_reader(
        self,
        key: _RemoteResponseKey,
        *,
        response_queue_id: int,
        timeout: float,
    ) -> tuple[InferenceResponse | None, bool]:
        """Take a response or elect this caller as the queue's sole reader."""
        with self._condition:
            response = self.response_buffer.pop(key, None)
            if response is not None:
                return response, False
            if response_queue_id not in self._active_response_readers:
                self._active_response_readers.add(response_queue_id)
                return None, True
            self._condition.wait(timeout=max(0.0, timeout))
            return self.response_buffer.pop(key, None), False

    def release_response_reader(self, response_queue_id: int) -> None:
        """Release one response-queue reader election and wake all waiters."""
        with self._condition:
            self._active_response_readers.discard(response_queue_id)
            self._condition.notify_all()

    def route_response(
        self,
        response: InferenceResponse,
        *,
        expected_key: _RemoteResponseKey,
        receiving_policy_id: str,
        client_received_at: float,
        receive_started_at: float,
    ) -> None:
        """Record and atomically demultiplex one response for shared waiters."""
        key = _response_buffer_key(
            response.actor_incarnation,
            response.policy_id,
            response.request_id,
        )
        with self._condition:
            if key in self._completed_request_keys:
                self.stats_by_policy.setdefault(
                    response.policy_id,
                    _RemoteInferenceClientStats(),
                ).record_arrival(
                    response=response,
                    timing=None,
                    client_received_at=client_received_at,
                    receive_started_at=None,
                    buffered=False,
                    abandoned=True,
                )
                self._condition.notify_all()
                return
            explicitly_abandoned = (
                self.abandoned_request_keys.pop(key, _NOT_ABANDONED) is None
            )
            # A response can race actor restart or arrive after its bounded
            # abandonment/completion marker was evicted.  Never retain a
            # response for which this process owns no live request; otherwise
            # the shared demultiplexer becomes an unbounded stale-response
            # cache and a later request-ID reuse can consume foreign work.
            locally_pending = key in self.pending_payloads
            duplicate_arrival = key in self._arrived_request_keys
            abandoned = explicitly_abandoned or not locally_pending or duplicate_arrival
            buffered = key != expected_key and response.policy_id == receiving_policy_id
            timing = self.pending_timings.pop(key, None)
            self.stats_by_policy.setdefault(
                response.policy_id,
                _RemoteInferenceClientStats(),
            ).record_arrival(
                response=response,
                timing=timing,
                client_received_at=client_received_at,
                receive_started_at=(
                    receive_started_at
                    if key == expected_key and not abandoned
                    else None
                ),
                buffered=buffered,
                abandoned=abandoned,
            )
            if not abandoned:
                self._arrived_request_keys.add(key)
                self.response_buffer[key] = response
            self._condition.notify_all()

    def complete_request(self, key: _RemoteResponseKey) -> None:
        """Forget retry state after a matching response has been consumed."""
        with self._condition:
            self.pending_payloads.pop(key, None)
            self.pending_timings.pop(key, None)
            self._arrived_request_keys.discard(key)
            self._completed_request_keys[key] = None
            while len(self._completed_request_keys) > _REMOTE_ABANDONED_REQUEST_LIMIT:
                self._completed_request_keys.popitem(last=False)

    def payload_and_abandon(
        self,
        key: _RemoteResponseKey,
    ) -> _PendingRemotePayload | None:
        """Atomically retain retry inputs while abandoning their old request."""
        with self._condition:
            payload = self.pending_payloads.get(key)
            self._abandon_request_locked(key)
            return payload

    def pending_payload(
        self,
        key: _RemoteResponseKey,
    ) -> _PendingRemotePayload | None:
        """Return one live retry payload without changing logical ownership."""
        with self._condition:
            return self.pending_payloads.get(key)

    def summary_for_policy(self, policy_id: str) -> dict[str, Any]:
        """Return a consistent snapshot of one policy's client statistics."""
        with self._condition:
            return self.stats_by_policy.setdefault(
                policy_id,
                _RemoteInferenceClientStats(),
            ).summary()

    def stage_timings_for_policy(
        self,
        policy_id: str,
    ) -> dict[str, dict[str, float | int]]:
        """Return a consistent stage-timing snapshot for one policy."""
        with self._condition:
            return self.stats_by_policy.setdefault(
                policy_id,
                _RemoteInferenceClientStats(),
            ).stage_timings()

    def abandon_request(self, key: _RemoteResponseKey) -> None:
        """Forget timed-out payloads and retain a bounded late-response marker."""
        with self._condition:
            self._abandon_request_locked(key)

    def _abandon_request_locked(self, key: _RemoteResponseKey) -> None:
        """Abandon one request while ``_condition`` is held."""
        self.pending_payloads.pop(key, None)
        self.pending_timings.pop(key, None)
        self.response_buffer.pop(key, None)
        self._arrived_request_keys.discard(key)
        self.abandoned_request_keys[key] = None
        while len(self.abandoned_request_keys) > _REMOTE_ABANDONED_REQUEST_LIMIT:
            oldest = next(iter(self.abandoned_request_keys))
            del self.abandoned_request_keys[oldest]
        self._condition.notify_all()


class RequestQueue(Protocol):
    """Queue-like input for inference requests."""

    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> InferenceQueueRequest:
        """Return one request or raise ``queue.Empty``."""


class ClientRequestQueue(Protocol):
    """Queue-like output used by actor inference clients."""

    def put(
        self,
        item: InferenceQueueRequest,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Put one request."""


class ResponseQueue(Protocol):
    """Queue-like output for inference responses."""

    def put(self, item: InferenceResponse) -> None:
        """Put one response."""


class ClientResponseQueue(Protocol):
    """Queue-like input used by actor inference clients."""

    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> InferenceResponse:
        """Return one response or raise ``queue.Empty``."""


class _InferencePolicyRegistryMapping(Mapping[str, InferencePolicy]):
    """Mapping that performs one state resync before rejecting a new policy."""

    def __init__(self, registry: InferencePolicyRegistry) -> None:
        self._registry = registry

    def __getitem__(self, policy_id: str) -> InferencePolicy:
        return self._registry.policy_for_id(policy_id)

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry.policies)

    def __len__(self) -> int:
        return len(self._registry.policies)

    @property
    def policy_load_pending(self) -> bool:
        """Return whether a missing route may still be loading."""
        return self._registry.frozen_policy_load_pending


class InferencePolicyRegistry:
    """Maintain candidate and frozen policies for inference-server routing."""

    def __init__(
        self,
        *,
        candidate_policy: InferencePolicy,
        candidate_model: torch.nn.Module | None = None,
        weights_dir: Path | None = None,
        weight_map_location: str = "cpu",
        weight_strict: bool = True,
        shared_weight_loader: SharedMemoryWeightLoader | None = None,
        frozen_pool: FrozenPolicyPool | None = None,
        frozen_state_path: Path | None = None,
        registry_sync_interval_seconds: float = 0.0,
        candidate_snapshot_factory: Callable[
            [Mapping[str, Any], int, str], InferencePolicy
        ]
        | None = None,
        max_resident_snapshots: int | None = None,
        max_in_flight_leases: int | None = None,
        max_recurrent_sequence_leases: int | None = None,
        recurrent_replay_cache_capacity: int = 4096,
        candidate_snapshot_min_version_gap: int = 1,
        candidate_policy_aliases: Mapping[str, InferencePolicy] | None = None,
    ) -> None:
        """Initialize registry state."""
        if registry_sync_interval_seconds < 0.0:
            raise ValueError("registry_sync_interval_seconds must be non-negative")
        if candidate_snapshot_min_version_gap <= 0:
            raise ValueError("candidate_snapshot_min_version_gap must be positive")
        snapshot_values = (
            candidate_snapshot_factory,
            max_resident_snapshots,
            max_in_flight_leases,
        )
        snapshot_enabled = candidate_snapshot_factory is not None
        aliases = dict(candidate_policy_aliases or {})
        if "candidate" in aliases:
            raise ValueError("candidate policy aliases cannot replace candidate")
        if snapshot_enabled and aliases:
            raise ValueError("candidate policy aliases do not support snapshot leases")
        if any(value is not None for value in snapshot_values) and not all(
            value is not None for value in snapshot_values
        ):
            raise ValueError(
                "snapshot factory and both lease capacities are required together"
            )
        self._snapshot_router = (
            InferencePolicySnapshotRouter(
                candidate_policy,
                max_resident_snapshots=cast(int, max_resident_snapshots),
                max_in_flight_leases=cast(int, max_in_flight_leases),
                max_recurrent_sequence_leases=max_recurrent_sequence_leases,
                recurrent_replay_cache_capacity=recurrent_replay_cache_capacity,
            )
            if snapshot_enabled
            else None
        )
        self._candidate_snapshot_factory = candidate_snapshot_factory
        self._candidate_snapshot_min_version_gap = int(
            candidate_snapshot_min_version_gap
        )
        self._candidate_policy_aliases = aliases
        self.candidate_policy = cast(
            InferencePolicy,
            self._snapshot_router or candidate_policy,
        )
        self.candidate_model = candidate_model
        self.weights_dir = weights_dir
        self.weight_map_location = weight_map_location
        self.weight_strict = bool(weight_strict)
        self.shared_weight_loader = shared_weight_loader
        self.frozen_pool = frozen_pool
        self.frozen_state_path = frozen_state_path
        self.registry_sync_interval_seconds = float(registry_sync_interval_seconds)
        self._loaded_weight_version: int | None = (
            self._snapshot_router.policy_version
            if self._snapshot_router is not None
            else None
        )
        self._deferred_weight_version: int | None = None
        self._deferred_weight_reason: str | None = None
        self._coalesced_weight_versions = 0
        self._last_registry_sync_at: float | None = None
        self._frozen_state_signature: tuple[int, int, int] | None = None
        self._frozen_member_signature: tuple[tuple[str, str, bool], ...] | None = None
        self._frozen_load_task: _FrozenPolicyLoadTask | None = None
        self._frozen_load_failure: RuntimeError | None = None
        self._frozen_discarded_loads = 0
        self._frozen_load_status: dict[str, Any] = {
            "status": (
                "disabled"
                if frozen_pool is None or frozen_state_path is None
                else "idle"
            ),
            "discarded_loads": 0,
        }
        self._routing_policies = _InferencePolicyRegistryMapping(self)
        if self._snapshot_router is not None and self.weights_dir is not None:
            write_inference_served_policy(
                self.weights_dir,
                version=self._snapshot_router.policy_version,
                model_fingerprint=self._snapshot_router.model_fingerprint,
            )

    @property
    def policies(self) -> Mapping[str, InferencePolicy]:
        """Return policies keyed by inference ``policy_id``."""
        policies: dict[str, InferencePolicy] = {"candidate": self.candidate_policy}
        policies.update(self._candidate_policy_aliases)
        if self.frozen_pool is not None:
            frozen = self.frozen_pool.policies
            overlap = policies.keys() & frozen.keys()
            if overlap:
                raise RuntimeError(
                    f"frozen policy IDs collide with candidate routes: {sorted(overlap)}"
                )
            policies.update(frozen)
        return policies

    @property
    def routing_policies(self) -> Mapping[str, InferencePolicy]:
        """Return a mapping that hot-loads newly published frozen policies."""
        return self._routing_policies

    @property
    def snapshot_router(self) -> InferencePolicySnapshotRouter | None:
        """Return the bounded candidate router when planner leases are enabled."""
        return self._snapshot_router

    def policy_for_id(self, policy_id: str) -> InferencePolicy:
        """Resolve one policy, resyncing worker state once on a cache miss."""
        policy = self.policies.get(policy_id)
        if policy is not None:
            return policy
        self.sync_frozen_pool()
        policy = self.policies.get(policy_id)
        if policy is None:
            raise KeyError(f"inference policy is not loaded: {policy_id}")
        return policy

    def sync(self) -> InferencePolicyRegistrySync:
        """Poll candidate weights and frozen state once."""
        now = time.monotonic()
        if (
            self._last_registry_sync_at is not None
            and now - self._last_registry_sync_at < self.registry_sync_interval_seconds
        ):
            return InferencePolicyRegistrySync(
                loaded_weight_version=None,
                frozen_update=None,
                policy_ids=tuple(sorted(self.policies)),
                frozen_load=self.frozen_load_status,
                **self._snapshot_sync_fields(),
            )
        latest = self.poll_latest_weights()
        frozen_update = self.sync_frozen_pool()
        self._last_registry_sync_at = now
        return InferencePolicyRegistrySync(
            loaded_weight_version=None if latest is None else latest.version,
            frozen_update=frozen_update,
            policy_ids=tuple(sorted(self.policies)),
            frozen_load=self.frozen_load_status,
            **self._snapshot_sync_fields(),
        )

    @property
    def frozen_load_status(self) -> Mapping[str, Any]:
        """Return JSON-safe background frozen-pool load telemetry."""
        return dict(self._frozen_load_status)

    @property
    def frozen_policy_load_pending(self) -> bool:
        """Return whether a staged frozen generation is still loading."""
        task = self._frozen_load_task
        return task is not None and task.thread is not None

    def poll_latest_weights(self) -> PublishedWeights | None:
        """Load a newer candidate ``latest.json`` checkpoint if configured."""
        if self.weights_dir is None or (
            self.candidate_model is None and self._snapshot_router is None
        ):
            return None
        if self._snapshot_router is not None:
            return self._poll_latest_snapshot()
        if self.candidate_model is None:
            raise RuntimeError("mutable weight polling has no candidate model")
        if self.shared_weight_loader is not None:
            shared_latest_version = self.shared_weight_loader.latest_version()
            try:
                shared = self.shared_weight_loader.poll(self.candidate_model)
            except FileNotFoundError:
                shared = None
                shared_latest_version = None
            if shared is not None:
                self._loaded_weight_version = shared.version
                if shared.model_fingerprint is None:
                    raise RuntimeError(
                        "shared policy publication has no model fingerprint"
                    )
                _set_policy_publication(
                    self.candidate_policy,
                    version=shared.version,
                    model_fingerprint=shared.model_fingerprint,
                )
                return PublishedWeights(
                    version=shared.version,
                    path=shared.latest_path,
                    latest_path=shared.latest_path,
                    published_at=shared.published_at,
                    metadata=shared.metadata,
                    model_fingerprint=shared.model_fingerprint,
                )
            if shared_latest_version is not None:
                return None
        latest = read_latest_published_weights(self.weights_dir)
        if latest is None or (
            self._loaded_weight_version is not None
            and latest.version <= self._loaded_weight_version
        ):
            return None
        checkpoint = torch.load(latest.path, map_location=self.weight_map_location)
        checkpoint_state = _state_dict_from_checkpoint(checkpoint)
        checkpoint_fingerprint = canonical_model_state_fingerprint(checkpoint_state)
        if (
            latest.model_fingerprint is not None
            and latest.model_fingerprint != checkpoint_fingerprint
        ):
            raise RuntimeError(
                "candidate checkpoint differs from its published model fingerprint"
            )
        incompatible = self.candidate_model.load_state_dict(
            checkpoint_state,
            strict=False,
        )
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if self.weight_strict and (
            missing - LEGACY_STATE_ENCODER_MISSING_KEYS or unexpected
        ):
            raise RuntimeError(
                "candidate weights are incompatible with inference model"
            )
        model_fingerprint = checkpoint_fingerprint
        if missing or unexpected:
            # A legacy partial state identifies the stored asset, not the fully
            # constructed serving model after compatibility initialization.
            model_fingerprint = canonical_model_state_fingerprint(self.candidate_model)
        self._loaded_weight_version = latest.version
        _set_policy_publication(
            self.candidate_policy,
            version=latest.version,
            model_fingerprint=model_fingerprint,
        )
        return latest

    def _poll_latest_snapshot(self) -> PublishedWeights | None:
        """Publish the newest complete state as a fresh immutable policy object."""
        router = cast(InferencePolicySnapshotRouter, self._snapshot_router)
        factory = self._candidate_snapshot_factory
        if factory is None or self.weights_dir is None:
            raise RuntimeError("snapshot registry is incompletely configured")

        if self.shared_weight_loader is not None:
            shared_latest_version = self.shared_weight_loader.latest_version()
            if (
                shared_latest_version is not None
                and shared_latest_version > router.policy_version
            ):
                if not self._snapshot_version_is_ready(shared_latest_version):
                    return None
                try:
                    shared_payload = self.shared_weight_loader.poll_state_dict()
                except FileNotFoundError:
                    shared_payload = None
                    shared_latest_version = None
                if shared_payload is not None:
                    shared, checkpoint = shared_payload
                    fingerprint = shared.model_fingerprint
                    if fingerprint is None:
                        raise RuntimeError(
                            "shared snapshot publication has no model fingerprint"
                        )
                    state_dict = _state_dict_from_checkpoint(checkpoint)
                    policy = factory(state_dict, shared.version, fingerprint)
                    self._publish_snapshot_policy(
                        policy,
                        version=shared.version,
                        model_fingerprint=fingerprint,
                    )
                    self.shared_weight_loader.mark_loaded(shared)
                    return PublishedWeights(
                        version=shared.version,
                        path=shared.latest_path,
                        latest_path=shared.latest_path,
                        published_at=shared.published_at,
                        metadata=shared.metadata,
                        model_fingerprint=fingerprint,
                    )
            if shared_latest_version is not None:
                return None

        latest = read_latest_published_weights(self.weights_dir)
        if latest is None or latest.version <= router.policy_version:
            return None
        if not self._snapshot_version_is_ready(latest.version):
            return None
        checkpoint = torch.load(latest.path, map_location=self.weight_map_location)
        state_dict = _state_dict_from_checkpoint(checkpoint)
        fingerprint = canonical_model_state_fingerprint(state_dict)
        if (
            latest.model_fingerprint is not None
            and latest.model_fingerprint != fingerprint
        ):
            raise RuntimeError(
                "candidate checkpoint differs from its published model fingerprint"
            )
        policy = factory(state_dict, latest.version, fingerprint)
        self._publish_snapshot_policy(
            policy,
            version=latest.version,
            model_fingerprint=fingerprint,
        )
        return latest

    def _snapshot_version_is_ready(self, version: int) -> bool:
        """Defer early or capacity-blocked versions without building a model."""
        router = cast(InferencePolicySnapshotRouter, self._snapshot_router)
        self._deferred_weight_version = int(version)
        if version - router.policy_version < self._candidate_snapshot_min_version_gap:
            self._deferred_weight_reason = "version_gap"
            return False
        if not router.can_publish(version):
            self._deferred_weight_reason = "capacity"
            return False
        return True

    def _publish_snapshot_policy(
        self,
        policy: InferencePolicy,
        *,
        version: int,
        model_fingerprint: str,
    ) -> None:
        router = cast(InferencePolicySnapshotRouter, self._snapshot_router)
        served_fingerprint = getattr(policy, "model_fingerprint", None)
        if served_fingerprint != model_fingerprint:
            raise RuntimeError(
                "constructed policy differs from its full-state publication"
            )
        previous = router.policy_version
        router.publish_snapshot(version, policy)
        if self.weights_dir is not None:
            write_inference_served_policy(
                self.weights_dir,
                version=version,
                model_fingerprint=model_fingerprint,
            )
        self._loaded_weight_version = int(version)
        self._coalesced_weight_versions += max(0, int(version) - previous - 1)
        self._deferred_weight_version = None
        self._deferred_weight_reason = None

    def _snapshot_sync_fields(self) -> dict[str, Any]:
        router = self._snapshot_router
        if router is None:
            return {}
        stats = router.stats()
        return {
            "deferred_weight_version": self._deferred_weight_version,
            "deferred_weight_reason": self._deferred_weight_reason,
            "coalesced_weight_versions": self._coalesced_weight_versions,
            "snapshot_min_version_gap": self._candidate_snapshot_min_version_gap,
            "snapshot_pool": {
                "current_version": stats.current_version,
                "resident_versions": list(stats.resident_versions),
                "leases_by_version": dict(stats.leases_by_version),
                "in_flight_leases": stats.in_flight_leases,
                "max_resident_snapshots": stats.max_resident_snapshots,
                "max_in_flight_leases": stats.max_in_flight_leases,
                "rejected_acquires": stats.rejected_acquires,
                "rejected_publishes": stats.rejected_publishes,
            },
        }

    def sync_frozen_pool(self) -> FrozenPolicyPoolUpdate | None:
        """Advance a non-blocking frozen-policy hot load when state changes."""
        if self.frozen_pool is None or self.frozen_state_path is None:
            return None
        if self._frozen_load_failure is not None:
            raise self._frozen_load_failure
        completed = self._poll_frozen_pool_load()
        if completed is not None:
            return completed
        if self._frozen_load_failure is not None:
            # Return one failed telemetry snapshot before the next sync raises.
            return None
        signature = self._frozen_pool_state_signature()
        if signature == self._frozen_state_signature:
            return None
        if self._frozen_load_task is not None:
            return None
        state = read_frozen_pool_state(self.frozen_state_path)
        member_signature = _frozen_member_runtime_signature(state.members)
        if (
            self._frozen_member_signature is not None
            and member_signature == self._frozen_member_signature
        ):
            # Curriculum EMAs, counters, and assignment cursors update this
            # file continuously. They do not change an inference route and
            # must not trigger a model-generation rebuild.
            self._frozen_state_signature = signature
            return None
        if self._frozen_state_signature is None:
            return self._load_initial_frozen_pool(
                signature=signature,
                members=state.members,
            )
        self._start_frozen_pool_load(
            signature=signature,
            member_signature=member_signature,
            members=state.members,
        )
        return None

    def _frozen_pool_state_signature(self) -> tuple[int, int, int]:
        """Return the identity of the latest complete frozen-state file."""
        if self.frozen_state_path is None:
            return (-1, -1, -1)
        try:
            stat = self.frozen_state_path.stat()
            return (int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size))
        except FileNotFoundError:
            return (-1, -1, -1)

    def _load_initial_frozen_pool(
        self,
        *,
        signature: tuple[int, int, int],
        members: Sequence[FrozenPoolMember],
    ) -> FrozenPolicyPoolUpdate:
        """Synchronously establish the baseline roster before hot serving."""
        pool = self.frozen_pool
        if pool is None:
            raise RuntimeError("initial frozen load has no pool")
        started_at = time.perf_counter()
        policy_ids = tuple(str(member.opponent_id) for member in members)
        self._frozen_load_status = self._frozen_loading_status(
            signature=signature,
            policy_ids=policy_ids,
            started_at_utc=_utc_timestamp(),
            background=False,
        )
        try:
            prepared = pool.prepare_sync(members)
        except Exception as exc:
            self._record_frozen_load_failure(
                exc,
                signature=signature,
                policy_ids=policy_ids,
                started_at=started_at,
                started_at_utc=str(self._frozen_load_status["started_at_utc"]),
                background=False,
            )
            raise
        update = pool.commit_prepared(prepared)
        self._frozen_state_signature = signature
        self._frozen_member_signature = _frozen_member_runtime_signature(members)
        self._record_frozen_load_success(
            update,
            signature=signature,
            policy_ids=policy_ids,
            started_at=started_at,
            started_at_utc=str(self._frozen_load_status["started_at_utc"]),
            background=False,
        )
        return update

    def _start_frozen_pool_load(
        self,
        *,
        signature: tuple[int, int, int],
        member_signature: tuple[tuple[str, str, bool], ...],
        members: Sequence[FrozenPoolMember],
    ) -> None:
        """Prepare one complete replacement generation on a daemon thread."""
        pool = self.frozen_pool
        if pool is None:
            raise RuntimeError("background frozen load has no pool")
        member_snapshot = tuple(members)
        task = _FrozenPolicyLoadTask(
            signature=signature,
            member_signature=member_signature,
            policy_ids=tuple(str(member.opponent_id) for member in member_snapshot),
            started_at=time.perf_counter(),
            started_at_utc=_utc_timestamp(),
        )

        def prepare() -> None:
            try:
                task.prepared = pool.prepare_sync(member_snapshot)
            except Exception as exc:  # pragma: no cover - asserted via poll result.
                task.error = exc
            finally:
                task.finished_at = time.perf_counter()

        task.thread = threading.Thread(
            target=prepare,
            name="frozen-policy-hot-load",
            daemon=True,
        )
        self._frozen_load_task = task
        self._frozen_load_status = self._frozen_loading_status(
            signature=signature,
            policy_ids=task.policy_ids,
            started_at_utc=task.started_at_utc,
            background=True,
        )
        task.thread.start()

    def _poll_frozen_pool_load(self) -> FrozenPolicyPoolUpdate | None:
        """Commit one complete background generation without partial exposure."""
        task = self._frozen_load_task
        if task is None or task.thread is None:
            return None
        if task.thread.is_alive():
            pool = self.frozen_pool
            if pool is None:
                raise RuntimeError("pending frozen load has no pool")
            now = time.perf_counter()
            timeout_seconds = pool.config.load_timeout_seconds
            if now - task.started_at <= timeout_seconds:
                return None
            self._frozen_load_task = None
            self._record_frozen_load_failure(
                TimeoutError(
                    "staged frozen-policy load exceeded its deadline of "
                    f"{timeout_seconds:g} seconds"
                ),
                signature=task.signature,
                policy_ids=task.policy_ids,
                started_at=task.started_at,
                started_at_utc=task.started_at_utc,
                background=True,
                finished_at=now,
            )
            return None
        task.thread.join(timeout=0.0)
        self._frozen_load_task = None
        finished_at = task.finished_at or time.perf_counter()
        if task.error is not None:
            self._record_frozen_load_failure(
                task.error,
                signature=task.signature,
                policy_ids=task.policy_ids,
                started_at=task.started_at,
                started_at_utc=task.started_at_utc,
                background=True,
                finished_at=finished_at,
            )
            return None
        prepared = task.prepared
        if prepared is None:
            error = RuntimeError("background frozen-policy load returned no generation")
            self._record_frozen_load_failure(
                error,
                signature=task.signature,
                policy_ids=task.policy_ids,
                started_at=task.started_at,
                started_at_utc=task.started_at_utc,
                background=True,
                finished_at=finished_at,
            )
            return None
        state_path = self.frozen_state_path
        if state_path is None:
            raise RuntimeError("completed frozen load has no state path")
        current_state = read_frozen_pool_state(state_path)
        current_member_signature = _frozen_member_runtime_signature(
            current_state.members
        )
        if current_member_signature != task.member_signature:
            self._frozen_discarded_loads += 1
            self._frozen_load_status = {
                "status": "stale",
                "background": True,
                "policy_ids": list(task.policy_ids),
                "load_seconds": max(0.0, finished_at - task.started_at),
                "discarded_loads": self._frozen_discarded_loads,
            }
            return None
        pool = self.frozen_pool
        if pool is None:
            raise RuntimeError("completed frozen load has no pool")
        update = pool.commit_prepared(prepared)
        current_signature = self._frozen_pool_state_signature()
        self._frozen_state_signature = current_signature
        self._frozen_member_signature = current_member_signature
        self._record_frozen_load_success(
            update,
            signature=current_signature,
            policy_ids=task.policy_ids,
            started_at=task.started_at,
            started_at_utc=task.started_at_utc,
            background=True,
            finished_at=finished_at,
        )
        return update

    def _frozen_loading_status(
        self,
        *,
        signature: tuple[int, int, int],
        policy_ids: tuple[str, ...],
        started_at_utc: str,
        background: bool,
    ) -> dict[str, Any]:
        return {
            "status": "loading",
            "background": background,
            "state_signature": list(signature),
            "policy_ids": list(policy_ids),
            "started_at_utc": started_at_utc,
            "discarded_loads": self._frozen_discarded_loads,
        }

    def _record_frozen_load_success(
        self,
        update: FrozenPolicyPoolUpdate,
        *,
        signature: tuple[int, int, int],
        policy_ids: tuple[str, ...],
        started_at: float,
        started_at_utc: str,
        background: bool,
        finished_at: float | None = None,
    ) -> None:
        completed_at = finished_at or time.perf_counter()
        self._frozen_load_status = {
            "status": "ready",
            "background": background,
            "state_signature": list(signature),
            "policy_ids": list(policy_ids),
            "started_at_utc": started_at_utc,
            "completed_at_utc": _utc_timestamp(),
            "load_seconds": max(0.0, completed_at - started_at),
            "loaded": list(update.loaded),
            "unloaded": list(update.unloaded),
            "kept": list(update.kept),
            "discarded_loads": self._frozen_discarded_loads,
        }

    def _record_frozen_load_failure(
        self,
        error: Exception,
        *,
        signature: tuple[int, int, int],
        policy_ids: tuple[str, ...],
        started_at: float,
        started_at_utc: str,
        background: bool,
        finished_at: float | None = None,
    ) -> None:
        completed_at = finished_at or time.perf_counter()
        self._frozen_load_status = {
            "status": "failed",
            "background": background,
            "state_signature": list(signature),
            "policy_ids": list(policy_ids),
            "started_at_utc": started_at_utc,
            "completed_at_utc": _utc_timestamp(),
            "load_seconds": max(0.0, completed_at - started_at),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "discarded_loads": self._frozen_discarded_loads,
        }
        self._frozen_load_failure = RuntimeError(
            "background frozen-policy load failed without changing live routes: "
            f"{type(error).__name__}: {error}"
        )


class RemoteInferencePolicy:
    """Rollout policy that delegates sampling to an inference server queue."""

    def __init__(
        self,
        *,
        actor_id: str,
        policy_id: str,
        request_queue: ClientRequestQueue,
        response_queue: ClientResponseQueue,
        config: InferenceClientConfig | None = None,
        client_state: RemoteInferenceClientState | None = None,
        response_buffer: MutableMapping[_RemoteResponseKey, InferenceResponse]
        | None = None,
        initial_request_id: int = 0,
        actor_incarnation: int = 0,
        request_purpose: InferenceRequestPurpose = "behavior",
        planner_model_version_lease: int | None = None,
        planner_tensor_schema_fingerprint: str = "",
        planner_inference_device_type: Literal["cpu", "cuda"] | None = None,
        behavior_kind: BehaviorKind = "policy_sample",
        recurrent_model_config: AgentNetworkConfig | None = None,
    ) -> None:
        """Initialize remote inference client state."""
        if initial_request_id < 0:
            raise ValueError("initial_request_id must be non-negative")
        if actor_incarnation < 0:
            raise ValueError("actor_incarnation must be non-negative")
        if behavior_kind not in ("policy_sample", "improvement"):
            raise ValueError("remote inference behavior_kind is invalid")
        if request_purpose not in ("behavior", "planner_behavior", "teacher"):
            raise ValueError(
                f"unsupported inference request purpose: {request_purpose}"
            )
        resolved_config = config or InferenceClientConfig()
        if request_purpose == "planner_behavior":
            if not _is_sha256(planner_tensor_schema_fingerprint):
                raise ValueError("planner client requires a tensor schema identity")
            if (
                planner_model_version_lease is not None
                and planner_model_version_lease < 0
            ):
                raise ValueError("planner model-version lease must be non-negative")
            if resolved_config.response_retries != 0:
                raise ValueError(
                    "planner inference disables retries to preserve root ownership"
                )
            if planner_inference_device_type not in ("cpu", "cuda"):
                raise ValueError(
                    "planner client requires its server inference device type"
                )
        elif (
            planner_model_version_lease is not None
            or planner_tensor_schema_fingerprint
            or planner_inference_device_type is not None
        ):
            raise ValueError("only planner clients may prebind model identity")
        if client_state is not None and response_buffer is not None:
            raise ValueError("client_state and response_buffer are mutually exclusive")
        self.actor_id = actor_id
        self.actor_incarnation = int(actor_incarnation)
        self.policy_id = policy_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.config = resolved_config
        self.request_purpose = request_purpose
        self.behavior_kind = behavior_kind
        self.planner_inference_device_type = planner_inference_device_type
        self._planner_tensor_schema_fingerprint = planner_tensor_schema_fingerprint
        self._metadata_lock = threading.RLock()
        self._last_response_policy_version: int | None = None
        self._model_fingerprint = ""
        self._proposal_version = 0
        self._policy_version = (
            0
            if planner_model_version_lease is None
            else int(planner_model_version_lease)
        )
        self._initial_request_id = int(initial_request_id)
        self._recurrent_model_config = recurrent_model_config
        self._recurrent_bindings: dict[
            RecurrentSequenceIdentity,
            PolicyArtifactIdentity,
        ] = {}
        self._recurrent_release_inflight: set[RecurrentSequenceIdentity] = set()
        self._recurrent_decode_inflight: set[RecurrentSequenceIdentity] = set()
        self._client_state = client_state or RemoteInferenceClientState(
            response_buffer={} if response_buffer is None else response_buffer
        )

    @property
    def recurrent_enabled(self) -> bool:
        """Return whether this client has an explicit recurrent wire contract."""
        config = self._recurrent_model_config
        return config is not None and config.recurrent is not None

    def initial_recurrent_state(self, batch_size: int) -> RecurrentPolicyState:
        """Create the canonical actor-side CPU FP32 reset state."""
        if batch_size <= 0:
            raise ValueError("recurrent batch_size must be positive")
        config = self._recurrent_model_config
        if config is None or config.recurrent is None:
            raise RuntimeError("remote recurrent inference is disabled")
        recurrent = config.recurrent
        shape = (
            recurrent.num_layers,
            batch_size,
            recurrent.resolved_hidden_size(config.state_encoder.d_model),
        )
        return RecurrentPolicyState(
            hidden=torch.zeros(shape, dtype=torch.float32),
            cell=torch.zeros(shape, dtype=torch.float32),
        )

    @property
    def preferred_inflight_batch_decisions(self) -> int:
        """Return the actor-side chunk size for concurrent request submission."""
        return self.config.max_inflight_batch_decisions

    @property
    def policy_version(self) -> int:
        """Return the newest decode version observed by this client."""
        with self._metadata_lock:
            return self._policy_version

    @policy_version.setter
    def policy_version(self, value: int) -> None:
        """Set the externally published policy version."""
        with self._metadata_lock:
            self._policy_version = int(value)

    @property
    def planner_tensor_schema_fingerprint(self) -> str:
        """Return the immutable tensor schema bound to planner requests."""
        return self._planner_tensor_schema_fingerprint

    @property
    def last_response_policy_version(self) -> int | None:
        """Return the newest policy version seen on any response type."""
        with self._metadata_lock:
            return self._last_response_policy_version

    @last_response_policy_version.setter
    def last_response_policy_version(self, value: int | None) -> None:
        """Set the last-response version used by compatibility callers."""
        with self._metadata_lock:
            self._last_response_policy_version = None if value is None else int(value)

    @property
    def model_fingerprint(self) -> str:
        """Return the fingerprint paired with the newest planner decode."""
        with self._metadata_lock:
            return self._model_fingerprint

    @model_fingerprint.setter
    def model_fingerprint(self, value: str) -> None:
        """Set the externally published model fingerprint."""
        with self._metadata_lock:
            self._model_fingerprint = str(value)

    @property
    def proposal_version(self) -> int:
        """Return the proposal version paired with the newest planner decode."""
        with self._metadata_lock:
            return self._proposal_version

    @proposal_version.setter
    def proposal_version(self, value: int) -> None:
        """Set the externally published proposal version."""
        with self._metadata_lock:
            self._proposal_version = int(value)

    def sample_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        """Send one encoded batch and wait for its matching server response."""
        handle = self.submit_decode(states, options, decks, temperature=temperature)
        return self.receive_decode(handle)

    def sample_decode_with_trace(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
    ) -> SampleDecodeTrace:
        """Send one encoded batch and return its token-level behavior trace."""
        handle = self.submit_decode(states, options, decks, temperature=temperature)
        return self.receive_decode_with_trace(handle)

    def sample_recurrent_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        recurrent: RecurrentInferenceBatch,
        *,
        temperature: float,
        retain_planner_context: bool = False,
    ) -> RecurrentDecodeResult:
        """Run one strict recurrent request through the remote server."""
        return self.receive_recurrent_decode(
            self.submit_recurrent_decode(
                states,
                options,
                decks,
                recurrent,
                temperature=temperature,
                retain_planner_context=retain_planner_context,
            )
        )

    def submit_recurrent_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        recurrent: RecurrentInferenceBatch,
        *,
        temperature: float,
        retain_planner_context: bool,
    ) -> RemoteInferenceHandle:
        """Submit a recurrent behavior decode without committing actor state."""
        if not self.recurrent_enabled:
            raise RuntimeError("remote recurrent inference is disabled")
        if self.request_purpose != "behavior":
            raise RuntimeError("recurrent wire transport supports behavior decode only")
        if retain_planner_context:
            raise ValueError("recurrent behavior decode cannot retain planner context")
        with self._metadata_lock:
            for sequence in recurrent.sequences:
                if sequence in self._recurrent_decode_inflight:
                    raise RuntimeError(
                        "recurrent sequence already has an in-flight decode"
                    )
                if sequence in self._recurrent_release_inflight:
                    raise RuntimeError("recurrent sequence release is in flight")
                bound = self._recurrent_bindings.get(sequence)
                if recurrent.expected_artifact is None:
                    if bound is not None:
                        raise RuntimeError(
                            "bound recurrent sequence omitted its artifact"
                        )
                elif bound != recurrent.expected_artifact:
                    raise RuntimeError(
                        "recurrent request differs from its local binding"
                    )
            self._recurrent_decode_inflight.update(recurrent.sequences)
        try:
            handle = self._submit_request(
                states,
                options,
                decks,
                temperature=temperature,
                request_type="decode",
                recurrent=recurrent,
            )
        except BaseException:
            with self._metadata_lock:
                self._recurrent_decode_inflight.difference_update(recurrent.sequences)
            raise
        return replace(handle, recurrent_request=recurrent)

    def receive_recurrent_decode(
        self,
        handle: RemoteInferenceHandle,
    ) -> RecurrentDecodeResult:
        """Validate, bind, and return an uncommitted recurrent proposal."""
        request = handle.recurrent_request
        if handle.request_type != "decode" or request is None:
            raise RuntimeError("recurrent receive requires a recurrent decode handle")
        try:
            response = self._receive_response(handle)
            result = response.recurrent_result
            if result is None or response.served_policy_artifact is None:
                raise RuntimeError(
                    "recurrent response omitted its full result identity"
                )
            if result.event_generations != request.event_generations:
                raise RuntimeError("recurrent response changed event generations")
            if result.sequences != request.sequences:
                raise RuntimeError("recurrent response changed sequence ownership")
            validate_policy_artifact_identity(
                result.served_artifact,
                response.served_policy_artifact,
            )
            if request.expected_artifact is not None:
                validate_policy_artifact_identity(
                    request.expected_artifact,
                    result.served_artifact,
                )
            if (
                result.trace.served_model_fingerprint
                != result.served_artifact.model_fingerprint
            ):
                raise RuntimeError("recurrent trace and artifact identities differ")
            with self._metadata_lock:
                for sequence in request.sequences:
                    bound = self._recurrent_bindings.get(sequence)
                    if request.expected_artifact is None:
                        if bound is not None:
                            raise RuntimeError("recurrent first bind raced local state")
                    elif bound != request.expected_artifact:
                        raise RuntimeError("recurrent continuation lost its binding")
                for sequence in request.sequences:
                    self._recurrent_bindings[sequence] = result.served_artifact
            return result
        finally:
            with self._metadata_lock:
                self._recurrent_decode_inflight.difference_update(request.sequences)

    def release_recurrent_sequences(
        self,
        sequences: Sequence[RecurrentSequenceIdentity],
        expected_artifact: PolicyArtifactIdentity,
    ) -> int:
        """Release exact remote sequence leases and wait for acknowledgement."""
        frozen = tuple(sequences)
        if not frozen or len(set(frozen)) != len(frozen):
            raise ValueError("recurrent release sequences must be non-empty and unique")
        with self._metadata_lock:
            for sequence in frozen:
                if self._recurrent_bindings.get(sequence) != expected_artifact:
                    raise RuntimeError("recurrent release differs from local binding")
                if sequence in self._recurrent_decode_inflight:
                    raise RuntimeError("cannot release an in-flight recurrent decode")
                if sequence in self._recurrent_release_inflight:
                    raise RuntimeError("recurrent release is already in flight")
            self._recurrent_release_inflight.update(frozen)
        try:
            handle = self._submit_recurrent_release(frozen, expected_artifact)
            response = self._receive_response(handle)
            validate_policy_artifact_identity(
                expected_artifact,
                response.served_policy_artifact,
            )
            if response.released_sequence_identities != frozen:
                raise RuntimeError("server changed recurrent release ownership")
            if not 0 <= response.released_recurrent_sequences <= len(frozen):
                raise RuntimeError("server returned an invalid recurrent release count")
            with self._metadata_lock:
                for sequence in frozen:
                    if self._recurrent_bindings.get(sequence) != expected_artifact:
                        raise RuntimeError("recurrent binding changed during release")
                for sequence in frozen:
                    del self._recurrent_bindings[sequence]
            return response.released_recurrent_sequences
        finally:
            with self._metadata_lock:
                self._recurrent_release_inflight.difference_update(frozen)

    def abort_recurrent_sequences(
        self,
        sequences: Sequence[RecurrentSequenceIdentity],
    ) -> int:
        """Clean up a first bind when no authoritative response was observed."""
        frozen = tuple(sequences)
        if not frozen or len(set(frozen)) != len(frozen):
            raise ValueError("recurrent abort sequences must be non-empty and unique")
        with self._metadata_lock:
            for sequence in frozen:
                if sequence in self._recurrent_bindings:
                    raise RuntimeError(
                        "bound recurrent sequence requires strict release"
                    )
                if sequence in self._recurrent_decode_inflight:
                    raise RuntimeError("cannot abort an in-flight recurrent decode")
                if sequence in self._recurrent_release_inflight:
                    raise RuntimeError("recurrent release is already in flight")
            self._recurrent_release_inflight.update(frozen)
        try:
            handle = self._submit_recurrent_release(
                frozen,
                expected_artifact=None,
                allow_missing=True,
            )
            response = self._receive_response(handle)
            if response.released_sequence_identities != frozen:
                raise RuntimeError("server changed recurrent abort ownership")
            if not 0 <= response.released_recurrent_sequences <= len(frozen):
                raise RuntimeError("server returned an invalid recurrent abort count")
            return response.released_recurrent_sequences
        finally:
            with self._metadata_lock:
                self._recurrent_release_inflight.difference_update(frozen)

    def _submit_recurrent_release(
        self,
        sequences: tuple[RecurrentSequenceIdentity, ...],
        expected_artifact: PolicyArtifactIdentity | None,
        *,
        allow_missing: bool = False,
    ) -> RemoteInferenceHandle:
        """Enqueue one standalone release request with retry ownership."""
        request_id = self._client_state.allocate_request_id(
            self.policy_id,
            minimum=self._initial_request_id,
        )
        created_at = time.perf_counter()
        request = RecurrentReleaseRequest(
            actor_id=self.actor_id,
            request_id=request_id,
            policy_id=self.policy_id,
            sequences=sequences,
            expected_artifact=expected_artifact,
            allow_missing=allow_missing,
            created_at=created_at,
            actor_incarnation=self.actor_incarnation,
        )
        key = _response_buffer_key(
            self.actor_incarnation,
            self.policy_id,
            request_id,
        )
        self._client_state.register_request(
            key,
            payload=_PendingRecurrentReleasePayload(original_request=request),
            timing=_PendingRequestTiming(
                created_at=created_at,
                client_put_finished_at=created_at,
                request_queue_put_seconds=0.0,
                batch_size=request.batch_size,
            ),
        )
        try:
            self.request_queue.put(request)
        except BaseException:
            self._client_state.discard_unsubmitted_request(key)
            raise
        put_seconds = time.perf_counter() - created_at
        self._client_state.finish_request_submit(
            key,
            timing=_PendingRequestTiming(
                created_at=created_at,
                client_put_finished_at=created_at + put_seconds,
                request_queue_put_seconds=put_seconds,
                batch_size=request.batch_size,
            ),
            policy_id=self.policy_id,
            batch_size=request.batch_size,
        )
        return RemoteInferenceHandle(
            actor_id=self.actor_id,
            request_id=request_id,
            policy_id=self.policy_id,
            batch_size=request.batch_size,
            request_type="recurrent_release",
            recurrent_release_request=request,
            actor_incarnation=self.actor_incarnation,
        )

    def predict_values(
        self,
        states: StateBatch,
        decks: DeckBatch,
    ) -> Tensor:
        """Send one value-only batch and wait for its critic predictions."""
        handle = self._submit_request(
            states,
            None,
            decks,
            temperature=1.0,
            request_type="value",
        )
        response = self._receive_response(handle)
        return response.values

    def predict_values_until(
        self,
        states: StateBatch,
        decks: DeckBatch,
        *,
        deadline_monotonic: float,
        model_version_lease: int | None = None,
        tensor_schema_fingerprint: str = "",
    ) -> Tensor:
        """Predict values under the caller's existing absolute deadline."""
        handle = self._submit_request(
            states,
            None,
            decks,
            temperature=1.0,
            request_type="value",
            deadline_monotonic=deadline_monotonic,
            model_version_lease=model_version_lease,
            tensor_schema_fingerprint=tensor_schema_fingerprint,
        )
        return self._receive_response(handle).values

    def predict_root_information_values_until(
        self,
        inputs: Any,
        decks: DeckBatch,
        *,
        deadline_monotonic: float,
        model_version_lease: int,
        tensor_schema_fingerprint: str,
    ) -> Tensor:
        """Schedule one action-critical unique-leaf value microbatch."""
        if self.request_purpose != "planner_behavior":
            raise ValueError(
                "root-information values require a planner_behavior client"
            )
        states = getattr(inputs, "states", None)
        actor_relations = getattr(inputs, "actor_relations", None)
        endpoints = getattr(inputs, "endpoints", None)
        belief_summaries = getattr(inputs, "belief_summaries", None)
        if not isinstance(states, StateBatch):
            raise TypeError("root-information inputs must contain StateBatch states")
        handle = self._submit_request(
            states,
            None,
            decks,
            temperature=1.0,
            request_type="root_information_value",
            deadline_monotonic=deadline_monotonic,
            actor_relations=actor_relations,
            endpoints=endpoints,
            belief_summaries=belief_summaries,
            model_version_lease=model_version_lease,
            tensor_schema_fingerprint=tensor_schema_fingerprint,
        )
        return self._receive_response(handle).values

    def evaluate_planner_candidates_until(
        self,
        states: StateBatch,
        options: OptionBatch,
        candidate_actions: Sequence[Sequence[Sequence[int]]],
        candidate_features: Sequence[Tensor],
        decks: DeckBatch,
        *,
        ordered_rows: Tensor,
        planner_context_handles: Sequence[str],
        deadline_monotonic: float,
        model_version_lease: int,
        tensor_schema_fingerprint: str,
    ) -> PlannerCandidateEvaluation:
        """Schedule one cross-actor schema-9 candidate evaluation batch."""
        if self.request_purpose != "planner_behavior":
            raise ValueError(
                "planner candidate evaluation requires a planner_behavior client"
            )
        frozen_actions = tuple(
            tuple(tuple(int(index) for index in action) for action in group)
            for group in candidate_actions
        )
        frozen_features = tuple(candidate_features)
        handle = self._submit_request(
            states,
            options,
            decks,
            temperature=1.0,
            request_type="planner_candidates",
            deadline_monotonic=deadline_monotonic,
            model_version_lease=model_version_lease,
            tensor_schema_fingerprint=tensor_schema_fingerprint,
            candidate_actions=frozen_actions,
            candidate_features=frozen_features,
            ordered_rows=ordered_rows,
            planner_context_handles=tuple(planner_context_handles),
        )
        response = self._receive_response(handle)
        base = response.planner_base_action_logprobs
        proposal = response.planner_proposal_action_logprobs
        residuals = response.planner_reranker_residuals
        if base is None or proposal is None or residuals is None:
            raise RuntimeError("planner candidate response is incomplete")
        return PlannerCandidateEvaluation(
            base_action_logprobs=base,
            proposal_action_logprobs=proposal,
            reranker_residuals=residuals,
            candidate_counts=response.planner_candidate_counts,
        )

    def generate_planner_proposals_until(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        ordered_rows: Tensor,
        limits: PlannerProposalSearchLimits,
        planner_context_handles: Sequence[str],
        deadline_monotonic: float,
        model_version_lease: int,
        tensor_schema_fingerprint: str,
    ) -> PlannerProposalBatchResult:
        """Schedule one cross-actor learned proposal prefix campaign."""
        if self.request_purpose != "planner_behavior":
            raise ValueError(
                "planner proposal generation requires a planner_behavior client"
            )
        handle = self._submit_request(
            states,
            options,
            decks,
            temperature=1.0,
            request_type="planner_proposals",
            deadline_monotonic=deadline_monotonic,
            model_version_lease=model_version_lease,
            tensor_schema_fingerprint=tensor_schema_fingerprint,
            planner_proposal_request=PlannerProposalRequestPayload(
                ordered_rows=ordered_rows,
                limits=limits,
                context_handles=tuple(planner_context_handles),
            ),
        )
        payload = self._receive_response(handle).planner_proposal_response
        if payload is None:
            raise RuntimeError("planner proposal response is incomplete")
        return payload.as_batch_result()

    def release_planner_contexts_until(
        self,
        states: StateBatch,
        decks: DeckBatch,
        *,
        planner_context_handles: Sequence[str],
        deadline_monotonic: float,
        model_version_lease: int,
        tensor_schema_fingerprint: str,
    ) -> int:
        """Explicitly release root GPU contexts on every planner exit path."""
        if self.request_purpose != "planner_behavior":
            raise ValueError("planner context release requires a planner client")
        handle = self._submit_request(
            states,
            None,
            decks,
            temperature=1.0,
            request_type="planner_context_release",
            deadline_monotonic=deadline_monotonic,
            model_version_lease=model_version_lease,
            tensor_schema_fingerprint=tensor_schema_fingerprint,
            planner_context_handles=tuple(planner_context_handles),
        )
        response = self._receive_response(handle)
        return int(response.values.sum().item())

    def submit_planner_context_release_until(
        self,
        states: StateBatch,
        decks: DeckBatch,
        *,
        planner_context_handles: Sequence[str],
        deadline_monotonic: float,
        model_version_lease: int,
        tensor_schema_fingerprint: str,
    ) -> None:
        """Enqueue best-effort cleanup without blocking the rollout thread."""
        if self.request_purpose != "planner_behavior":
            raise ValueError("planner context release requires a planner client")
        handle = self._submit_request(
            states,
            None,
            decks,
            temperature=1.0,
            request_type="planner_context_release",
            deadline_monotonic=deadline_monotonic,
            model_version_lease=model_version_lease,
            tensor_schema_fingerprint=tensor_schema_fingerprint,
            planner_context_handles=tuple(planner_context_handles),
        )
        self._client_state.abandon_request(
            _response_buffer_key(
                handle.actor_incarnation,
                handle.policy_id,
                handle.request_id,
            )
        )

    def submit_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float,
    ) -> RemoteInferenceHandle:
        """Send one encoded batch and return a handle for a later receive."""
        return self._submit_request(
            states,
            options,
            decks,
            temperature=temperature,
            request_type="decode",
            model_version_lease=None,
            tensor_schema_fingerprint=(
                self._planner_tensor_schema_fingerprint
                if self.request_purpose == "planner_behavior"
                else ""
            ),
            retain_planner_context=(
                True if self.request_purpose == "planner_behavior" else None
            ),
        )

    def submit_decode_for_request(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float,
        retain_planner_context: bool,
    ) -> RemoteInferenceHandle:
        """Submit a root decode with explicit retained-context admission."""
        if self.request_purpose != "planner_behavior":
            if retain_planner_context:
                raise ValueError("only planner decode may retain root contexts")
            return self.submit_decode(
                states,
                options,
                decks,
                temperature=temperature,
            )
        return self._submit_request(
            states,
            options,
            decks,
            temperature=temperature,
            request_type="decode",
            model_version_lease=None,
            tensor_schema_fingerprint=self._planner_tensor_schema_fingerprint,
            retain_planner_context=bool(retain_planner_context),
        )

    def sample_decode_until(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float,
        deadline_monotonic: float,
        model_version_lease: int | None = None,
        tensor_schema_fingerprint: str = "",
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        """Decode under the caller's existing absolute deadline."""
        if self.request_purpose == "planner_behavior" and model_version_lease is None:
            raise ValueError("planner continuation decode requires a root lease")
        handle = self._submit_request(
            states,
            options,
            decks,
            temperature=temperature,
            request_type="decode",
            deadline_monotonic=deadline_monotonic,
            model_version_lease=model_version_lease,
            tensor_schema_fingerprint=tensor_schema_fingerprint,
        )
        return self.receive_decode(handle)

    def _submit_request(
        self,
        states: StateBatch,
        options: OptionBatch | None,
        decks: DeckBatch,
        *,
        temperature: float,
        request_type: InferenceRequestType,
        deadline_monotonic: float | None = None,
        actor_relations: Tensor | None = None,
        endpoints: Tensor | None = None,
        belief_summaries: Tensor | None = None,
        model_version_lease: int | None = None,
        tensor_schema_fingerprint: str = "",
        candidate_actions: tuple[tuple[tuple[int, ...], ...], ...] | None = None,
        candidate_features: tuple[Tensor, ...] | None = None,
        ordered_rows: Tensor | None = None,
        planner_proposal_request: PlannerProposalRequestPayload | None = None,
        planner_context_handles: tuple[str, ...] = (),
        retain_planner_context: bool | None = None,
        recurrent: RecurrentInferenceBatch | None = None,
    ) -> RemoteInferenceHandle:
        """Submit one decode or value request and retain its retry payload."""
        request_id = self._client_state.allocate_request_id(
            self.policy_id,
            minimum=self._initial_request_id,
        )
        put_started_at = time.perf_counter()
        request_deadline = 0.0
        if self.request_purpose in ("teacher", "planner_behavior"):
            request_deadline = (
                time.monotonic() + self.config.response_timeout_seconds
                if deadline_monotonic is None
                else float(deadline_monotonic)
            )
            if not math.isfinite(request_deadline):
                raise ValueError("deadline-aware inference deadline must be finite")
            if request_deadline <= time.monotonic():
                raise TimeoutError("inference deadline already expired")
        elif deadline_monotonic is not None:
            raise ValueError("behavior inference cannot override its deadline")
        request = InferenceRequest(
            actor_id=self.actor_id,
            request_id=request_id,
            policy_id=self.policy_id,
            states=states,
            options=options,
            decks=decks,
            temperature=temperature,
            created_at=put_started_at,
            request_type=request_type,
            purpose=self.request_purpose,
            deadline_monotonic=request_deadline,
            model_version_lease=model_version_lease,
            tensor_schema_fingerprint=tensor_schema_fingerprint,
            actor_relations=actor_relations,
            endpoints=endpoints,
            belief_summaries=belief_summaries,
            candidate_actions=candidate_actions,
            candidate_features=candidate_features,
            ordered_rows=ordered_rows,
            planner_proposal_request=planner_proposal_request,
            planner_context_handles=planner_context_handles,
            retain_planner_context=retain_planner_context,
            recurrent=recurrent,
            actor_incarnation=self.actor_incarnation,
        )
        key = _response_buffer_key(
            self.actor_incarnation,
            self.policy_id,
            request_id,
        )
        pending_payload = _PendingInferencePayload(
            states=states,
            options=options,
            decks=decks,
            temperature=float(temperature),
            request_type=request_type,
            actor_relations=actor_relations,
            endpoints=endpoints,
            belief_summaries=belief_summaries,
            model_version_lease=model_version_lease,
            tensor_schema_fingerprint=tensor_schema_fingerprint,
            deadline_monotonic=request_deadline,
            candidate_actions=candidate_actions,
            candidate_features=candidate_features,
            ordered_rows=ordered_rows,
            planner_proposal_request=planner_proposal_request,
            planner_context_handles=planner_context_handles,
            retain_planner_context=retain_planner_context,
            recurrent=recurrent,
            original_request=request,
        )
        self._client_state.register_request(
            key,
            payload=pending_payload,
            timing=_PendingRequestTiming(
                created_at=put_started_at,
                client_put_finished_at=put_started_at,
                request_queue_put_seconds=0.0,
                batch_size=request.batch_size,
            ),
        )
        try:
            if self.request_purpose in ("teacher", "planner_behavior"):
                self.request_queue.put(
                    request,
                    timeout=max(0.0, request_deadline - time.monotonic()),
                )
            else:
                self.request_queue.put(request)
        except queue.Full as exc:
            self._client_state.discard_unsubmitted_request(key)
            raise TimeoutError("inference request enqueue timed out") from exc
        except BaseException:
            self._client_state.discard_unsubmitted_request(key)
            raise
        put_seconds = time.perf_counter() - put_started_at
        self._client_state.finish_request_submit(
            key,
            timing=_PendingRequestTiming(
                created_at=put_started_at,
                client_put_finished_at=put_started_at + put_seconds,
                request_queue_put_seconds=put_seconds,
                batch_size=request.batch_size,
            ),
            policy_id=self.policy_id,
            batch_size=request.batch_size,
        )
        if (
            self.request_purpose in ("teacher", "planner_behavior")
            and time.monotonic() >= request_deadline
        ):
            self._client_state.abandon_request(key)
            raise TimeoutError("inference request enqueue timed out")
        return RemoteInferenceHandle(
            actor_id=self.actor_id,
            request_id=request_id,
            policy_id=self.policy_id,
            batch_size=request.batch_size,
            request_type=request_type,
            deadline_monotonic=request_deadline,
            recurrent_request=recurrent,
            actor_incarnation=self.actor_incarnation,
        )

    def receive_decode(
        self,
        handle: RemoteInferenceHandle,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        """Wait for and return the server response for a submitted request."""
        if handle.request_type != "decode" or handle.recurrent_request is not None:
            raise RuntimeError("decode receive requires a decode inference handle")
        response = self._receive_response(handle)
        return (response.actions, response.action_logprobs, response.values)

    def receive_decode_with_trace(
        self,
        handle: RemoteInferenceHandle,
    ) -> SampleDecodeTrace:
        """Wait for and return a submitted request's token-level trace."""
        if handle.request_type != "decode" or handle.recurrent_request is not None:
            raise RuntimeError("decode receive requires a decode inference handle")
        response = self._receive_response(handle)
        _validate_response_token_trace(response, handle.batch_size)
        return SampleDecodeTrace(
            actions=response.actions,
            action_logprobs=response.action_logprobs,
            values=response.values,
            token_logprobs=cast(Tensor, response.token_logprobs),
            prefix_values=cast(Tensor, response.prefix_values),
            token_mask=cast(Tensor, response.token_mask),
            stop_sampled=cast(Tensor, response.stop_sampled),
            planner_context_handles=response.planner_context_handles,
            served_policy_version=response.policy_version,
            served_model_fingerprint=response.model_fingerprint,
            served_proposal_version=response.proposal_version,
            planner_fallback_reason=response.planner_fallback_reason,
        )

    def _receive_response(
        self,
        handle: RemoteInferenceHandle,
    ) -> InferenceResponse:
        if handle.actor_id != self.actor_id:
            raise RuntimeError("inference handle actor_id mismatch")
        if handle.actor_incarnation != self.actor_incarnation:
            raise RuntimeError("inference handle actor incarnation mismatch")
        if handle.policy_id != self.policy_id:
            raise RuntimeError("inference handle policy_id mismatch")
        receive_started_at = time.perf_counter()
        key = _response_buffer_key(
            handle.actor_incarnation,
            handle.policy_id,
            handle.request_id,
        )
        response = self._client_state.pop_buffered_response(key)
        if response is None:
            try:
                response = self._get_matching_response(
                    handle,
                    receive_started_at=receive_started_at,
                )
            except TimeoutError:
                return self._retry_response(handle)
        self._client_state.complete_request(key)
        _validate_response(
            response,
            self.actor_id,
            self.actor_incarnation,
            handle.request_id,
            self.policy_id,
            handle.request_type,
        )
        if response.error_type is not None:
            if response.batch_size != handle.batch_size:
                raise RuntimeError("inference error response batch size mismatch")
            if response.error_type in (
                "TeacherDeadlineExpired",
                "PlannerDeadlineExpired",
            ):
                raise TimeoutError(response.error_message)
            raise RuntimeError(
                f"remote inference failed: {response.error_type}: "
                f"{response.error_message}"
            )
        _validate_response_batch(response, handle.batch_size)
        recurrent_response = response.recurrent_result is not None
        if recurrent_response != (handle.recurrent_request is not None):
            raise RuntimeError("inference response changed recurrent wire semantics")
        if handle.request_type == "recurrent_release":
            release = handle.recurrent_release_request
            if release is None:
                raise RuntimeError("recurrent release handle lost its request")
            if response.released_sequence_identities != release.sequences:
                raise RuntimeError("recurrent release response changed ownership")
            if response.served_policy_artifact is None and not release.allow_missing:
                raise RuntimeError("recurrent release omitted its artifact identity")
        elif handle.recurrent_request is None and (
            response.served_policy_artifact is not None
            or response.released_recurrent_sequences
        ):
            raise RuntimeError("stateless inference returned recurrent metadata")
        if (
            handle.request_type == "decode"
            and self.request_purpose == "planner_behavior"
        ):
            if not _is_sha256(response.model_fingerprint):
                raise RuntimeError("planner decode response has no model fingerprint")
            if response.proposal_version < 0:
                raise RuntimeError(
                    "planner decode response has an invalid proposal version"
                )
        with self._metadata_lock:
            current_last = self._last_response_policy_version
            if current_last is None or response.policy_version >= current_last:
                self._last_response_policy_version = response.policy_version
            if (
                handle.request_type == "decode"
                and response.policy_version >= self._policy_version
            ):
                self._policy_version = response.policy_version
                if self.request_purpose == "planner_behavior":
                    self._model_fingerprint = response.model_fingerprint
                    self._proposal_version = response.proposal_version
        return response

    def _retry_response(
        self,
        handle: RemoteInferenceHandle,
    ) -> InferenceResponse:
        key = _response_buffer_key(
            handle.actor_incarnation,
            handle.policy_id,
            handle.request_id,
        )
        if handle.attempts >= self.config.response_retries:
            self._client_state.abandon_request(key)
            raise TimeoutError("inference response timed out")
        exact_payload = self._client_state.pending_payload(key)
        if isinstance(
            exact_payload,
            (_PendingRecurrentReleasePayload, _PendingInferencePayload),
        ) and (
            isinstance(exact_payload, _PendingRecurrentReleasePayload)
            or exact_payload.recurrent is not None
        ):
            original = exact_payload.original_request
            if original is None:
                raise RuntimeError("recurrent retry lost its exact request payload")
            self.request_queue.put(original)
            return self._receive_response(replace(handle, attempts=handle.attempts + 1))
        payload = self._client_state.payload_and_abandon(key)
        if payload is None:
            raise TimeoutError("inference response timed out")
        if not isinstance(payload, _PendingInferencePayload):
            raise RuntimeError("stateless retry received a release payload")
        retry = self._submit_request(
            payload.states,
            payload.options,
            payload.decks,
            temperature=payload.temperature,
            request_type=payload.request_type,
            deadline_monotonic=(
                payload.deadline_monotonic if payload.deadline_monotonic > 0.0 else None
            ),
            actor_relations=payload.actor_relations,
            endpoints=payload.endpoints,
            belief_summaries=payload.belief_summaries,
            model_version_lease=payload.model_version_lease,
            tensor_schema_fingerprint=payload.tensor_schema_fingerprint,
            candidate_actions=payload.candidate_actions,
            candidate_features=payload.candidate_features,
            ordered_rows=payload.ordered_rows,
            planner_proposal_request=payload.planner_proposal_request,
            planner_context_handles=payload.planner_context_handles,
            retain_planner_context=payload.retain_planner_context,
        )
        retry = RemoteInferenceHandle(
            actor_id=retry.actor_id,
            request_id=retry.request_id,
            policy_id=retry.policy_id,
            batch_size=retry.batch_size,
            attempts=handle.attempts + 1,
            request_type=retry.request_type,
            deadline_monotonic=retry.deadline_monotonic,
            actor_incarnation=self.actor_incarnation,
        )
        return self._receive_response(retry)

    def _get_matching_response(
        self,
        handle: RemoteInferenceHandle,
        *,
        receive_started_at: float,
    ) -> InferenceResponse:
        """Drain actor responses until the requested handle arrives."""
        deadline = (
            handle.deadline_monotonic
            if handle.deadline_monotonic > 0.0
            else time.monotonic() + self.config.response_timeout_seconds
        )
        expected_key = _response_buffer_key(
            handle.actor_incarnation,
            handle.policy_id,
            handle.request_id,
        )
        response_queue_id = id(self.response_queue)
        while True:
            timeout = deadline - time.monotonic()
            if timeout <= 0.0:
                raise TimeoutError("inference response timed out")
            response, is_reader = self._client_state.pop_response_or_claim_reader(
                expected_key,
                response_queue_id=response_queue_id,
                timeout=timeout,
            )
            if response is not None:
                return response
            if not is_reader:
                continue
            try:
                response = self.response_queue.get(timeout=timeout)
            except queue.Empty as exc:
                self._client_state.release_response_reader(response_queue_id)
                response = self._client_state.pop_buffered_response(expected_key)
                if response is not None:
                    return response
                raise TimeoutError("inference response timed out") from exc
            except BaseException:
                self._client_state.release_response_reader(response_queue_id)
                raise
            try:
                client_received_at = time.perf_counter()
                if response.actor_id != self.actor_id:
                    raise RuntimeError("inference response actor_id mismatch")
                if response.actor_incarnation != self.actor_incarnation:
                    continue
                self._client_state.route_response(
                    response,
                    expected_key=expected_key,
                    receiving_policy_id=self.policy_id,
                    client_received_at=client_received_at,
                    receive_started_at=receive_started_at,
                )
            finally:
                self._client_state.release_response_reader(response_queue_id)

    def inference_client_summary(self) -> dict[str, Any]:
        """Return actor-side IPC latency counters for this remote policy."""
        return {
            "actor_id": self.actor_id,
            "actor_incarnation": self.actor_incarnation,
            "policy_id": self.policy_id,
            **self._client_state.summary_for_policy(self.policy_id),
        }

    def inference_client_stage_timings(self) -> dict[str, dict[str, float | int]]:
        """Return IPC latency segments using the actor stage timing schema."""
        return self._client_state.stage_timings_for_policy(self.policy_id)


def _expire_abandoned_planner_contexts(
    policies: Mapping[str, InferencePolicy],
) -> int:
    """Sweep caller-abandoned root leases before admitting another step."""
    released = 0
    now = time.monotonic()
    for policy in policies.values():
        expire = getattr(policy, "expire_planner_context_handles", None)
        if callable(expire):
            released += int(expire(now))
    return released


def _graph_decode_counter_snapshot(
    policies: Mapping[str, InferencePolicy],
) -> dict[str, int]:
    """Aggregate graph runner counters across unique resident policy objects."""
    counters: Counter[str] = Counter()
    seen: set[int] = set()
    for policy in policies.values():
        if id(policy) in seen:
            continue
        seen.add(id(policy))
        stats = getattr(policy, "graph_decode_stats", None)
        if not callable(stats):
            continue
        snapshot = cast(Any, stats)()
        if not isinstance(snapshot, Mapping):
            raise TypeError("graph_decode_stats must return a mapping")
        counters.update({str(key): int(value) for key, value in snapshot.items()})
    return dict(counters)


def _graph_decode_step_stats(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    """Convert lifetime graph counters to step deltas plus resident gauge."""
    result = {
        key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
        for key in set(before) | set(after)
        if key != "resident_captures"
    }
    if "resident_captures" in after:
        result["resident_captures"] = int(after["resident_captures"])
    return dict(sorted(result.items()))


def _defer_requests_for_pending_policies(
    request_queue: RequestQueue,
    requests: Sequence[InferenceQueueRequest],
    *,
    policies: Mapping[str, InferencePolicy],
) -> tuple[
    tuple[InferenceQueueRequest, ...],
    int,
    tuple[InferenceQueueRequest, ...],
]:
    """Keep requests for a staged frozen route without blocking old routes."""
    ready: list[InferenceQueueRequest] = []
    unavailable: list[InferenceQueueRequest] = []
    deferred = 0
    for request in requests:
        try:
            _policy_for_id(policies, request.policy_id)
        except KeyError:
            if not bool(getattr(policies, "policy_load_pending", False)):
                unavailable.append(request)
                continue
            _defer_drain_request(request_queue, request)
            deferred += 1
            continue
        ready.append(request)
    return tuple(ready), deferred, tuple(unavailable)


def run_inference_server_step(
    *,
    request_queue: RequestQueue,
    response_queues: Mapping[str, ResponseQueue],
    policies: Mapping[str, InferencePolicy],
    config: InferenceServerConfig | None = None,
) -> InferenceStepStats:
    """Drain requests once, run grouped policy batches, and emit responses."""
    cfg = config or InferenceServerConfig()
    _expire_abandoned_planner_contexts(policies)
    step_start = time.perf_counter()
    drain_start = time.perf_counter()
    drained = _drain_requests(request_queue, cfg)
    graph_counters_before = _graph_decode_counter_snapshot(policies)
    requests = drained.requests
    (
        requests,
        pending_policy_requests,
        unavailable_policy_requests,
    ) = _defer_requests_for_pending_policies(
        request_queue,
        requests,
        policies=policies,
    )
    _reject_unavailable_policy_requests(
        unavailable_policy_requests,
        response_queues=response_queues,
    )
    coalescing_stats = {
        **drained.coalescing_stats,
        "pending_policy_requests": pending_policy_requests,
        "unavailable_policy_requests": len(unavailable_policy_requests),
    }
    expired_teacher = drained.expired_teacher
    expired_planner = drained.expired_planner
    oversized = drained.oversized
    expired_teacher_requests = len(expired_teacher)
    expired_teacher_decisions = sum(request.batch_size for request in expired_teacher)
    expired_planner_requests = len(expired_planner)
    expired_planner_decisions = sum(request.batch_size for request in expired_planner)
    _reject_deadline_requests(
        expired_teacher,
        response_queues=response_queues,
    )
    _reject_deadline_requests(
        expired_planner,
        response_queues=response_queues,
    )
    _reject_oversized_requests(
        oversized,
        response_queues=response_queues,
    )
    rejected_oversized_requests = len(oversized)
    rejected_oversized_decisions = sum(request.batch_size for request in oversized)
    requests, rejected_leases = _drop_unavailable_planner_leases(
        requests,
        policies=policies,
    )
    rejected_planner_lease_requests = len(rejected_leases)
    rejected_planner_lease_decisions = sum(
        request.batch_size for request in rejected_leases
    )
    _reject_unavailable_planner_leases(
        rejected_leases,
        response_queues=response_queues,
    )
    _admit_latest_recurrent_actor_incarnations(requests, policies=policies)
    drain_seconds = time.perf_counter() - drain_start
    if not requests:
        return InferenceStepStats(
            requests=0,
            decisions=0,
            policy_batches=0,
            responses=0,
            expired_teacher_requests=expired_teacher_requests,
            expired_teacher_decisions=expired_teacher_decisions,
            expired_planner_requests=expired_planner_requests,
            expired_planner_decisions=expired_planner_decisions,
            rejected_planner_lease_requests=rejected_planner_lease_requests,
            rejected_planner_lease_decisions=rejected_planner_lease_decisions,
            rejected_oversized_requests=rejected_oversized_requests,
            rejected_oversized_decisions=rejected_oversized_decisions,
            drained_by_policy={},
            request_batch_histogram={},
            policy_batch_histogram={},
            shape_histogram={},
            decode_step_histogram={},
            service_time_histogram_ms={},
            bucket_histogram={},
            bucket_fallback_histogram={},
            bucket_slot_totals={},
            serving_path_histogram={},
            graph_decode_stats=_graph_decode_step_stats(
                graph_counters_before,
                _graph_decode_counter_snapshot(policies),
            ),
            coalescing_stats=coalescing_stats,
            mean_request_latency_ms=0.0,
            p95_request_latency_ms=0.0,
            request_latencies_ms=(),
            server_queue_latencies_ms=(),
            server_forward_latencies_ms=(),
            response_put_latencies_ms=(),
            drain_seconds=drain_seconds,
            sample_seconds=0.0,
            response_seconds=0.0,
            step_seconds=time.perf_counter() - step_start,
        )

    responses = 0
    policy_counts: Counter[str] = Counter()
    policy_batch_histogram: Counter[int] = Counter()
    shape_histogram: Counter[str] = Counter()
    decode_step_histogram: Counter[int] = Counter()
    service_time_histogram_ms: Counter[str] = Counter()
    bucket_histogram: Counter[str] = Counter()
    bucket_fallback_histogram: Counter[str] = Counter()
    bucket_slot_totals: Counter[str] = Counter()
    serving_path_histogram: Counter[str] = Counter()
    latencies_ms: list[float] = []
    server_queue_latencies_ms: list[float] = []
    server_forward_latencies_ms: list[float] = []
    response_put_latencies_ms: list[float] = []
    sample_start = time.perf_counter()
    served = _serve_scheduled_requests(
        requests,
        policies=policies,
        config=cfg,
    )
    if served.expired:
        for request in served.expired:
            if request.purpose == "teacher":
                expired_teacher_requests += 1
                expired_teacher_decisions += request.batch_size
            else:
                expired_planner_requests += 1
                expired_planner_decisions += request.batch_size
        _reject_deadline_requests(
            served.expired,
            response_queues=response_queues,
        )
        expired_keys = {_inference_request_key(request) for request in served.expired}
        requests = tuple(
            request
            for request in requests
            if _inference_request_key(request) not in expired_keys
        )
    served_groups = list(served.decode)
    served_value_groups = list(served.value)
    served_root_value_groups = list(served.root_value)
    served_context_release_groups = list(served.context_release)
    served_planner_candidate_groups = list(served.candidates)
    served_planner_proposal_groups = list(served.proposals)
    served_recurrent_requests = list(served.recurrent)
    served_value_groups.extend(served_context_release_groups)
    sample_seconds = time.perf_counter() - sample_start
    response_seconds = 0.0
    direct_responses = {
        _inference_request_key(item.request): item for item in served_recurrent_requests
    }
    row_results: dict[
        tuple[str, int, str, int],
        list[_InferenceRowResult | None],
    ] = {
        _inference_request_key(request): [None] * request.batch_size
        for request in requests
        if _inference_request_key(request) not in direct_responses
    }
    for served_group in served_groups:
        policy_id = served_group.policy_id
        sample = _policy_group_sample_to_cpu(served_group.sample)
        batch_sample_seconds = served_group.sample_seconds
        policy_version = served_group.policy_version
        group_decisions = len(served_group.rows)
        planner_rows = tuple(
            row
            for row in served_group.rows
            if row.request.purpose == "planner_behavior"
        )
        if planner_rows:
            if len(planner_rows) != group_decisions:
                raise RuntimeError("decode group mixed planner and base purposes")
            root_acquisition = all(
                row.request.model_version_lease is None for row in planner_rows
            )
            retain_root_context = all(
                _request_retains_planner_context(row.request) for row in planner_rows
            )
            if any(
                _request_retains_planner_context(row.request) != retain_root_context
                for row in planner_rows
            ):
                raise RuntimeError("planner decode mixed context ownership")
            capacity_fallback = sample.planner_fallback_reason == "model_lease_capacity"
            if sample.planner_fallback_reason not in (
                "",
                "model_lease_capacity",
            ):
                raise RuntimeError("planner decode returned an unknown fallback")
            if root_acquisition and retain_root_context:
                if capacity_fallback and sample.planner_context_handles:
                    raise RuntimeError(
                        "lease-capacity fallback unexpectedly retained contexts"
                    )
                if not capacity_fallback and (
                    len(sample.planner_context_handles) != group_decisions
                ):
                    raise RuntimeError("planner root decode did not issue contexts")
            elif root_acquisition:
                if sample.planner_context_handles or capacity_fallback:
                    raise RuntimeError(
                        "planner base-fast-path decode retained root metadata"
                    )
            elif sample.planner_context_handles or sample.planner_fallback_reason:
                raise RuntimeError(
                    "planner continuation decode returned root-only metadata"
                )
            if sample.served_policy_version != policy_version:
                raise RuntimeError(
                    "planner decode returned a mismatched policy version"
                )
            if not _is_sha256(sample.served_model_fingerprint):
                raise RuntimeError("planner decode omitted its model fingerprint")
            if sample.served_proposal_version is None or (
                sample.served_proposal_version < 0
            ):
                raise RuntimeError("planner decode omitted its proposal version")
            schemas = {row.request.tensor_schema_fingerprint for row in planner_rows}
            if len(schemas) != 1:
                raise RuntimeError("planner decode group mixed tensor schemas")
            if root_acquisition and retain_root_context and not capacity_fallback:
                bound_policy = _policy_for_id(policies, policy_id)
                binder = getattr(
                    bound_policy,
                    "bind_planner_context_handles",
                    None,
                )
                if not callable(binder):
                    raise RuntimeError("planner policy cannot bind context handles")
                bind_kwargs: dict[str, Any] = {
                    "policy_version": policy_version,
                    "tensor_schema_fingerprint": schemas.pop(),
                }
                if callable(
                    getattr(bound_policy, "expire_planner_context_handles", None)
                ):
                    bind_kwargs["deadline_monotonic"] = (
                        time.monotonic() + cfg.planner_context_ttl_seconds
                    )
                cast(Any, binder)(
                    sample.planner_context_handles,
                    **bind_kwargs,
                )
        policy_batch_histogram[group_decisions] += 1
        serving_path_histogram[served_group.serving_path] += 1
        shape = served_group.shape
        shape_histogram[_shape_histogram_key(shape)] += 1
        decode_step_histogram[shape[3]] += 1
        if sample.bucket_key is not None:
            bucket_histogram[sample.bucket_key] += 1
        if sample.bucket_fallback_reason is not None:
            bucket_fallback_histogram[sample.bucket_fallback_reason] += 1
        bucket_slot_totals.update(
            {str(key): int(value) for key, value in sample.bucket_slot_totals.items()}
        )
        service_time_histogram_ms[
            _service_time_histogram_bucket(batch_sample_seconds * 1000.0)
        ] += 1
        for sample_index, row in enumerate(served_group.rows):
            result_rows = row_results[_inference_request_key(row.request)]
            if result_rows[row.row_index] is not None:
                raise RuntimeError("inference request row was sampled more than once")
            result_rows[row.row_index] = _inference_row_result(
                sample,
                sample_index=sample_index,
                policy_version=policy_version,
                sample_started_at=served_group.sample_started_at,
                sample_finished_at=served_group.sample_finished_at,
            )
        policy_counts[policy_id] += group_decisions

    for value_group in served_value_groups:
        values = value_group.values.detach().cpu()
        group_decisions = len(value_group.rows)
        policy_batch_histogram[group_decisions] += 1
        shape_histogram[
            _value_shape_histogram_key(
                group_decisions,
                value_group.state_token_width,
            )
        ] += 1
        service_time_histogram_ms[
            _service_time_histogram_bucket(value_group.sample_seconds * 1000.0)
        ] += 1
        for sample_index, row in enumerate(value_group.rows):
            result_rows = row_results[_inference_request_key(row.request)]
            if result_rows[row.row_index] is not None:
                raise RuntimeError("inference request row was sampled more than once")
            result_rows[row.row_index] = _value_inference_row_result(
                values[sample_index],
                policy_version=value_group.policy_version,
                sample_started_at=value_group.sample_started_at,
                sample_finished_at=value_group.sample_finished_at,
            )
        policy_counts[value_group.policy_id] += group_decisions

    for value_group in served_root_value_groups:
        values = value_group.values.detach().cpu()
        group_decisions = len(value_group.rows)
        policy_batch_histogram[group_decisions] += 1
        shape_histogram[
            "root/"
            + _value_shape_histogram_key(
                group_decisions,
                value_group.state_token_width,
            )
        ] += 1
        service_time_histogram_ms[
            _service_time_histogram_bucket(value_group.sample_seconds * 1000.0)
        ] += 1
        for sample_index, row in enumerate(value_group.rows):
            result_rows = row_results[_inference_request_key(row.request)]
            if result_rows[row.row_index] is not None:
                raise RuntimeError("inference request row was sampled more than once")
            result_rows[row.row_index] = _value_inference_row_result(
                values[sample_index],
                policy_version=value_group.policy_version,
                sample_started_at=value_group.sample_started_at,
                sample_finished_at=value_group.sample_finished_at,
            )
        policy_counts[value_group.policy_id] += group_decisions

    for candidate_group in served_planner_candidate_groups:
        evaluation = candidate_group.evaluation
        base = evaluation.base_action_logprobs.detach().cpu()
        proposal = evaluation.proposal_action_logprobs.detach().cpu()
        residuals = evaluation.reranker_residuals.detach().cpu()
        group_decisions = len(candidate_group.rows)
        candidate_count = sum(evaluation.candidate_counts)
        policy_batch_histogram[group_decisions] += 1
        shape_histogram[
            "planner/"
            f"B{group_decisions}/T{candidate_group.state_token_width}"
            f"/C{candidate_count}"
        ] += 1
        service_time_histogram_ms[
            _service_time_histogram_bucket(candidate_group.sample_seconds * 1000.0)
        ] += 1
        offset = 0
        for row, count in zip(
            candidate_group.rows,
            evaluation.candidate_counts,
            strict=True,
        ):
            result_rows = row_results[_inference_request_key(row.request)]
            if result_rows[row.row_index] is not None:
                raise RuntimeError("inference request row was sampled more than once")
            stop = offset + count
            result_rows[row.row_index] = _planner_candidate_inference_row_result(
                base_action_logprobs=base[offset:stop],
                proposal_action_logprobs=proposal[offset:stop],
                reranker_residuals=residuals[offset:stop],
                policy_version=candidate_group.policy_version,
                sample_started_at=candidate_group.sample_started_at,
                sample_finished_at=candidate_group.sample_finished_at,
            )
            offset = stop
        if offset != candidate_count:
            raise RuntimeError("planner candidate scatter did not consume outputs")
        policy_counts[candidate_group.policy_id] += group_decisions

    for proposal_group in served_planner_proposal_groups:
        group_decisions = sum(
            len(payload.decisions) for payload in proposal_group.payloads
        )
        scoring_rows = sum(payload.scoring_rows for payload in proposal_group.payloads)
        policy_batch_histogram[group_decisions] += 1
        shape_histogram[
            "proposal/"
            f"B{group_decisions}/T{proposal_group.state_token_width}"
            f"/P{scoring_rows}"
        ] += 1
        service_time_histogram_ms[
            _service_time_histogram_bucket(proposal_group.sample_seconds * 1000.0)
        ] += 1
        for request, payload in zip(
            proposal_group.requests,
            proposal_group.payloads,
            strict=True,
        ):
            result_rows = row_results[_inference_request_key(request)]
            for row_index, decision in enumerate(payload.decisions):
                if result_rows[row_index] is not None:
                    raise RuntimeError(
                        "inference request row was sampled more than once"
                    )
                result_rows[row_index] = _planner_proposal_inference_row_result(
                    decision=decision,
                    base_greedy_action=payload.base_greedy_actions[row_index],
                    policy_version=proposal_group.policy_version,
                    sample_started_at=proposal_group.sample_started_at,
                    sample_finished_at=proposal_group.sample_finished_at,
                )
        policy_counts[proposal_group.policy_id] += group_decisions

    for recurrent_request in served_recurrent_requests:
        queued_request = recurrent_request.request
        decisions = queued_request.batch_size
        policy_counts[queued_request.policy_id] += decisions
        if recurrent_request.policy_batch:
            policy_batch_decisions = recurrent_request.policy_batch_size
            if policy_batch_decisions <= 0:
                raise RuntimeError("recurrent policy batch size must be positive")
            policy_batch_histogram[policy_batch_decisions] += 1
            service_time_histogram_ms[
                _service_time_histogram_bucket(
                    recurrent_request.sample_seconds * 1000.0
                )
            ] += 1
            serving_path_histogram[recurrent_request.serving_path] += 1
            shape_histogram[
                (
                    f"recurrent/B{policy_batch_decisions}"
                    if queued_request.request_type == "decode"
                    else f"recurrent_release/B{policy_batch_decisions}"
                )
            ] += 1

    for queued_request in requests:
        direct = direct_responses.get(_inference_request_key(queued_request))
        if direct is not None:
            response_start = time.perf_counter()
            response = replace(direct.response, server_put_at=response_start)
            _response_queue_for_actor(response_queues, queued_request.actor_id).put(
                response
            )
            response_put_seconds = time.perf_counter() - response_start
            response_seconds += response_put_seconds
            response_put_latencies_ms.append(response_put_seconds * 1000.0)
            if queued_request.server_received_at > 0.0:
                server_queue_latencies_ms.append(
                    max(
                        0.0,
                        (
                            response.server_sample_started_at
                            - queued_request.server_received_at
                        )
                        * 1000.0,
                    )
                )
            server_forward_latencies_ms.append(
                max(
                    0.0,
                    (
                        response.server_sample_finished_at
                        - response.server_sample_started_at
                    )
                    * 1000.0,
                )
            )
            if queued_request.created_at > 0.0:
                latencies_ms.append(
                    max(
                        0.0,
                        (time.perf_counter() - queued_request.created_at) * 1000.0,
                    )
                )
            responses += 1
            continue
        request = cast(InferenceRequest, queued_request)
        if _is_expired_deadline_request(request):
            if request.purpose == "teacher":
                expired_teacher_requests += 1
                expired_teacher_decisions += request.batch_size
            else:
                expired_planner_requests += 1
                expired_planner_decisions += request.batch_size
                _release_expired_planner_root_contexts(
                    request,
                    row_results[_inference_request_key(request)],
                    policies=policies,
                )
            _reject_deadline_requests(
                (request,),
                response_queues=response_queues,
            )
            continue
        response_start = time.perf_counter()
        results = row_results[_inference_request_key(request)]
        if any(result is None for result in results):
            raise RuntimeError("inference request has unsampled rows")
        completed = tuple(cast(_InferenceRowResult, result) for result in results)
        response = _inference_response_from_rows(
            request,
            completed,
            server_put_at=response_start,
        )
        try:
            _response_queue_for_actor(response_queues, request.actor_id).put(response)
        except BaseException:
            if request.purpose == "planner_behavior":
                _release_expired_planner_root_contexts(
                    request,
                    completed,
                    policies=policies,
                )
            raise
        response_put_seconds = time.perf_counter() - response_start
        response_seconds += response_put_seconds
        response_put_latencies_ms.append(response_put_seconds * 1000.0)
        if request.server_received_at > 0.0:
            server_queue_latencies_ms.append(
                max(
                    0.0,
                    (response.server_sample_started_at - request.server_received_at)
                    * 1000.0,
                )
            )
        server_forward_latencies_ms.append(
            max(
                0.0,
                (response.server_sample_finished_at - response.server_sample_started_at)
                * 1000.0,
            )
        )
        if request.created_at > 0.0:
            latencies_ms.append(
                max(0.0, (time.perf_counter() - request.created_at) * 1000.0)
            )
        responses += 1

    return InferenceStepStats(
        requests=len(requests),
        decisions=sum(request.batch_size for request in requests),
        policy_batches=(
            len(served_groups)
            + len(served_value_groups)
            + len(served_root_value_groups)
            + len(served_planner_candidate_groups)
            + len(served_planner_proposal_groups)
            + sum(item.policy_batch for item in served_recurrent_requests)
        ),
        responses=responses,
        expired_teacher_requests=expired_teacher_requests,
        expired_teacher_decisions=expired_teacher_decisions,
        expired_planner_requests=expired_planner_requests,
        expired_planner_decisions=expired_planner_decisions,
        rejected_planner_lease_requests=rejected_planner_lease_requests,
        rejected_planner_lease_decisions=rejected_planner_lease_decisions,
        rejected_oversized_requests=rejected_oversized_requests,
        rejected_oversized_decisions=rejected_oversized_decisions,
        drained_by_policy=dict(sorted(policy_counts.items())),
        request_batch_histogram=_batch_size_histogram(requests),
        policy_batch_histogram=dict(sorted(policy_batch_histogram.items())),
        shape_histogram=dict(sorted(shape_histogram.items())),
        decode_step_histogram=dict(sorted(decode_step_histogram.items())),
        service_time_histogram_ms=dict(sorted(service_time_histogram_ms.items())),
        bucket_histogram=dict(sorted(bucket_histogram.items())),
        bucket_fallback_histogram=dict(sorted(bucket_fallback_histogram.items())),
        bucket_slot_totals=dict(sorted(bucket_slot_totals.items())),
        serving_path_histogram=dict(sorted(serving_path_histogram.items())),
        graph_decode_stats=_graph_decode_step_stats(
            graph_counters_before,
            _graph_decode_counter_snapshot(policies),
        ),
        coalescing_stats=coalescing_stats,
        mean_request_latency_ms=_mean(latencies_ms),
        p95_request_latency_ms=_percentile(latencies_ms, 0.95),
        request_latencies_ms=tuple(latencies_ms),
        server_queue_latencies_ms=tuple(server_queue_latencies_ms),
        server_forward_latencies_ms=tuple(server_forward_latencies_ms),
        response_put_latencies_ms=tuple(response_put_latencies_ms),
        drain_seconds=drain_seconds,
        sample_seconds=sample_seconds,
        response_seconds=response_seconds,
        step_seconds=time.perf_counter() - step_start,
    )


def _release_expired_planner_root_contexts(
    request: InferenceRequest,
    results: Sequence[_InferenceRowResult | None],
    *,
    policies: Mapping[str, InferencePolicy],
) -> None:
    """Release root handles whose sampled response missed its caller deadline."""
    if request.request_type != "decode" or request.model_version_lease is not None:
        return
    handles = tuple(
        result.planner_context_handle
        for result in results
        if result is not None and result.planner_context_handle is not None
    )
    if not handles:
        return
    releaser = getattr(
        _policy_for_id(policies, request.policy_id),
        "release_planner_context_handles",
        None,
    )
    if not callable(releaser):
        raise RuntimeError("expired planner root policy cannot release contexts")
    released = int(cast(Any, releaser)(handles))
    if released != len(handles):
        raise RuntimeError("expired planner root released incomplete contexts")


def _serve_scheduled_requests(
    requests: Sequence[InferenceQueueRequest],
    *,
    policies: Mapping[str, InferencePolicy],
    config: InferenceServerConfig,
) -> _ServedInferenceStages:
    """Execute bounded contiguous stage batches without defeating EDF order."""
    decode: list[_ServedPolicyGroup] = []
    value: list[_ServedValueGroup] = []
    root_value: list[_ServedValueGroup] = []
    context_release: list[_ServedValueGroup] = []
    candidates: list[_ServedPlannerCandidateGroup] = []
    proposals: list[_ServedPlannerProposalGroup] = []
    recurrent: list[_ServedRecurrentRequest] = []
    expired: list[InferenceRequest] = []
    for batch in scheduled_stage_batches(requests, config=config):
        stale = tuple(
            request
            for request in batch
            if isinstance(request, InferenceRequest)
            and _is_expired_deadline_request(request)
        )
        expired.extend(stale)
        stale_keys = {_inference_request_key(request) for request in stale}
        live_batch = tuple(
            request
            for request in batch
            if _inference_request_key(request) not in stale_keys
        )
        if not live_batch:
            continue
        request_type = live_batch[0].request_type
        if request_type == "decode":
            recurrent_batch = tuple(
                request
                for request in live_batch
                if isinstance(request, InferenceRequest)
                and request.recurrent is not None
            )
            ordinary_batch = tuple(
                request
                for request in live_batch
                if isinstance(request, InferenceRequest) and request.recurrent is None
            )
            recurrent.extend(
                _serve_recurrent_decode_requests(
                    recurrent_batch,
                    policies=policies,
                )
            )
            decode.extend(
                _sample_policy_groups(
                    requests=ordinary_batch,
                    policies=policies,
                    config=config,
                )
                if ordinary_batch
                else ()
            )
        elif request_type == "recurrent_release":
            recurrent.extend(
                _serve_recurrent_release_requests(
                    tuple(
                        cast(RecurrentReleaseRequest, request) for request in live_batch
                    ),
                    policies=policies,
                )
            )
        elif request_type == "value":
            value.extend(
                _predict_value_groups(
                    requests=cast(tuple[InferenceRequest, ...], live_batch),
                    policies=policies,
                )
            )
        elif request_type == "root_information_value":
            root_value.extend(
                _predict_root_information_value_groups(
                    requests=cast(tuple[InferenceRequest, ...], live_batch),
                    policies=policies,
                )
            )
        elif request_type == "planner_context_release":
            try:
                context_release.extend(
                    _release_planner_context_groups(
                        requests=cast(tuple[InferenceRequest, ...], live_batch),
                        policies=policies,
                    )
                )
            except KeyError:
                expired.extend(cast(tuple[InferenceRequest, ...], live_batch))
        elif request_type == "planner_candidates":
            try:
                candidates.extend(
                    _predict_planner_candidate_groups(
                        requests=cast(tuple[InferenceRequest, ...], live_batch),
                        policies=policies,
                        config=config,
                    )
                )
            except KeyError:
                # A caller may time out and enqueue cleanup while an older
                # auxiliary request is still in flight. Missing retained
                # context is therefore an explicit absent target, not an
                # inference-process failure.
                expired.extend(cast(tuple[InferenceRequest, ...], live_batch))
        elif request_type == "planner_proposals":
            for proposal_batch in _planner_proposal_stage_groups(
                cast(tuple[InferenceRequest, ...], live_batch)
            ):
                try:
                    proposals.extend(
                        _predict_planner_proposal_groups(
                            requests=proposal_batch,
                            policies=policies,
                        )
                    )
                except (KeyError, PlannerProposalDeadlineError, ValueError):
                    expired.extend(proposal_batch)
        else:
            raise AssertionError(f"unsupported inference stage: {request_type}")
    return _ServedInferenceStages(
        decode=tuple(decode),
        value=tuple(value),
        root_value=tuple(root_value),
        context_release=tuple(context_release),
        candidates=tuple(candidates),
        proposals=tuple(proposals),
        recurrent=tuple(recurrent),
        expired=tuple(expired),
    )


def _serve_recurrent_decode_requests(
    requests: Sequence[InferenceRequest],
    *,
    policies: Mapping[str, InferencePolicy],
) -> tuple[_ServedRecurrentRequest, ...]:
    """Merge compatible actor batches under immutable sequence leases."""
    served: list[_ServedRecurrentRequest] = []
    bound: list[_BoundRecurrentRequest] = []
    # An exact request is re-enqueued after an acknowledgement timeout.  Both
    # queue copies can be drained in one step before either has populated the
    # replay cache, so collapse them before acquiring sequence leases or
    # sampling stochastic actions.
    unique_requests = tuple(
        OrderedDict(
            (_inference_request_key(request), request) for request in requests
        ).values()
    )
    for request in unique_requests:
        recurrent = request.recurrent
        if recurrent is None:
            raise ValueError("recurrent serving received a stateless request")
        router = _recurrent_snapshot_router(policies, request.policy_id)
        cached = router.recurrent_replay_lookup(
            actor_id=request.actor_id,
            actor_incarnation=request.actor_incarnation,
            policy_id=request.policy_id,
            request_id=request.request_id,
        )
        if cached is not None:
            response = cast(InferenceResponse, cached)
            served.append(
                _ServedRecurrentRequest(
                    request=request,
                    response=response,
                    sample_seconds=0.0,
                    policy_batch=False,
                )
            )
            continue
        try:
            _admit_recurrent_actor_incarnation(
                policies,
                actor_id=request.actor_id,
                actor_incarnation=request.actor_incarnation,
            )
            binding = router.bind_recurrent_sequences(
                actor_id=request.actor_id,
                actor_incarnation=request.actor_incarnation,
                policy_id=request.policy_id,
                sequences=recurrent.sequences,
                expected_artifact=recurrent.expected_artifact,
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            error = _served_recurrent_error(request, exc)
            router.recurrent_replay_store(
                error.response,
                actor_id=request.actor_id,
                actor_incarnation=request.actor_incarnation,
                policy_id=request.policy_id,
                request_id=request.request_id,
            )
            served.append(error)
            continue
        bound.append(
            _BoundRecurrentRequest(
                request=request,
                router=router,
                policy=binding.policy,
                policy_version=binding.policy_version,
                artifact=binding.artifact,
                first_bind=recurrent.expected_artifact is None,
            )
        )

    grouped: OrderedDict[
        tuple[int, int, int, str, float],
        list[_BoundRecurrentRequest],
    ] = OrderedDict()
    for item in bound:
        key = (
            id(item.router),
            id(item.policy),
            item.policy_version,
            item.artifact.fingerprint,
            item.request.temperature,
        )
        grouped.setdefault(key, []).append(item)

    for group in grouped.values():
        served.extend(_serve_bound_recurrent_group(group))
    return tuple(served)


def _serve_bound_recurrent_group(
    group: Sequence[_BoundRecurrentRequest],
) -> tuple[_ServedRecurrentRequest, ...]:
    """Run one compatible cross-actor recurrent batch and scatter its rows."""
    if not group:
        raise ValueError("recurrent policy group must be non-empty")
    policy = group[0].policy
    sampler = getattr(policy, "sample_decode_with_recurrent_state", None)
    if not callable(sampler):
        _release_first_recurrent_bindings(group)
        raise RuntimeError("recurrent snapshot lacks its state-aware decode surface")
    requests = tuple(item.request for item in group)
    recurrent_batches = tuple(
        cast(RecurrentInferenceBatch, request.recurrent) for request in requests
    )
    device = _policy_device(policy, fallback=requests[0].states.card_ids.device)
    started = time.perf_counter()
    try:
        trace, proposed_state = cast(Any, sampler)(
            _move_state_batch(
                _concat_state_batches(tuple(request.states for request in requests)),
                device=device,
            ),
            _move_option_batch(
                _concat_option_batches(
                    tuple(_decode_options(request) for request in requests)
                ),
                device=device,
            ),
            concatenate_deck_batches(tuple(request.decks for request in requests)).to(
                device
            ),
            _move_public_event_batch(
                concatenate_public_event_batches(
                    tuple(batch.public_events for batch in recurrent_batches)
                ),
                device=device,
            ),
            _move_recurrent_policy_state(
                concatenate_recurrent_policy_states(
                    tuple(batch.previous_state for batch in recurrent_batches)
                ),
                device=device,
            ),
            temperature=requests[0].temperature,
            retain_planner_context=False,
        )
        finished = time.perf_counter()
        cpu_trace = _recurrent_trace_to_cpu(
            cast(SampleDecodeTrace, trace),
            policy_version=group[0].policy_version,
            artifact=group[0].artifact,
        )
        cpu_state = _recurrent_state_to_cpu(cast(RecurrentPolicyState, proposed_state))
        total_rows = sum(request.batch_size for request in requests)
        if len(cpu_trace.actions) != total_rows or cpu_state.batch_size != total_rows:
            raise RuntimeError("recurrent merged output has a mismatched batch size")

        responses: list[tuple[_BoundRecurrentRequest, InferenceResponse]] = []
        offset = 0
        for item, recurrent in zip(group, recurrent_batches, strict=True):
            request = item.request
            stop = offset + request.batch_size
            request_trace = _slice_recurrent_trace(cpu_trace, offset, stop)
            request_state = RecurrentPolicyState(
                hidden=cpu_state.hidden[:, offset:stop],
                cell=cpu_state.cell[:, offset:stop],
            )
            result = RecurrentDecodeResult(
                trace=request_trace,
                proposed_state=request_state,
                event_generations=recurrent.event_generations,
                sequences=recurrent.sequences,
                served_artifact=item.artifact,
            )
            responses.append(
                (
                    item,
                    InferenceResponse(
                        actor_id=request.actor_id,
                        actor_incarnation=request.actor_incarnation,
                        request_id=request.request_id,
                        policy_id=request.policy_id,
                        policy_version=item.policy_version,
                        actions=request_trace.actions,
                        action_logprobs=request_trace.action_logprobs,
                        values=request_trace.values,
                        token_logprobs=request_trace.token_logprobs,
                        prefix_values=request_trace.prefix_values,
                        token_mask=request_trace.token_mask,
                        stop_sampled=request_trace.stop_sampled,
                        server_received_at=request.server_received_at,
                        server_sample_started_at=started,
                        server_sample_finished_at=finished,
                        request_type="decode",
                        model_fingerprint=item.artifact.model_fingerprint,
                        proposal_version=int(getattr(policy, "proposal_version", 0)),
                        recurrent_result=result,
                        served_policy_artifact=item.artifact,
                    ),
                )
            )
            offset = stop
        if offset != total_rows:
            raise RuntimeError("recurrent merged output scatter was incomplete")
    except Exception:
        _release_first_recurrent_bindings(group)
        raise

    for item, response in responses:
        item.router.recurrent_replay_store(
            response,
            actor_id=item.request.actor_id,
            actor_incarnation=item.request.actor_incarnation,
            policy_id=item.request.policy_id,
            request_id=item.request.request_id,
        )
    group_rows = sum(item.request.batch_size for item in group)
    serving_path = "recurrent_merged_eager" if len(group) > 1 else "recurrent_eager"
    return tuple(
        _ServedRecurrentRequest(
            request=item.request,
            response=response,
            sample_seconds=(finished - started if index == 0 else 0.0),
            policy_batch=index == 0,
            policy_batch_size=(group_rows if index == 0 else 0),
            serving_path=serving_path,
        )
        for index, (item, response) in enumerate(responses)
    )


def _release_first_recurrent_bindings(
    bound: Sequence[_BoundRecurrentRequest],
) -> None:
    """Roll back newly acquired sequence leases after an unserved failure."""
    for item in reversed(bound):
        if not item.first_bind:
            continue
        recurrent = cast(RecurrentInferenceBatch, item.request.recurrent)
        item.router.release_recurrent_sequences(
            actor_id=item.request.actor_id,
            actor_incarnation=item.request.actor_incarnation,
            policy_id=item.request.policy_id,
            sequences=recurrent.sequences,
            expected_artifact=item.artifact,
        )


def _slice_recurrent_trace(
    trace: SampleDecodeTrace,
    start: int,
    stop: int,
) -> SampleDecodeTrace:
    """Slice a merged recurrent trace without changing artifact metadata."""
    return replace(
        trace,
        actions=trace.actions[start:stop],
        action_logprobs=trace.action_logprobs[start:stop],
        values=trace.values[start:stop],
        token_logprobs=trace.token_logprobs[start:stop],
        prefix_values=trace.prefix_values[start:stop],
        token_mask=trace.token_mask[start:stop],
        stop_sampled=trace.stop_sampled[start:stop],
    )


def _serve_recurrent_release_requests(
    requests: Sequence[RecurrentReleaseRequest],
    *,
    policies: Mapping[str, InferencePolicy],
) -> tuple[_ServedRecurrentRequest, ...]:
    """Release or abort exact sequence leases with same-ID replay semantics."""
    served: list[_ServedRecurrentRequest] = []
    unique_requests = tuple(
        OrderedDict(
            (_inference_request_key(request), request) for request in requests
        ).values()
    )
    for request in unique_requests:
        router = _recurrent_snapshot_router(policies, request.policy_id)
        cached = router.recurrent_replay_lookup(
            actor_id=request.actor_id,
            actor_incarnation=request.actor_incarnation,
            policy_id=request.policy_id,
            request_id=request.request_id,
        )
        if cached is not None:
            served.append(
                _ServedRecurrentRequest(
                    request=request,
                    response=cast(InferenceResponse, cached),
                    sample_seconds=0.0,
                    policy_batch=False,
                )
            )
            continue
        started = time.perf_counter()
        try:
            _admit_recurrent_actor_incarnation(
                policies,
                actor_id=request.actor_id,
                actor_incarnation=request.actor_incarnation,
            )
            if request.expected_artifact is None:
                if not request.allow_missing:
                    raise RuntimeError(
                        "artifact-free recurrent release is not an abort"
                    )
                released, served_artifact = router.abort_recurrent_sequences(
                    actor_id=request.actor_id,
                    actor_incarnation=request.actor_incarnation,
                    policy_id=request.policy_id,
                    sequences=request.sequences,
                )
            else:
                released = router.release_recurrent_sequences(
                    actor_id=request.actor_id,
                    actor_incarnation=request.actor_incarnation,
                    policy_id=request.policy_id,
                    sequences=request.sequences,
                    expected_artifact=request.expected_artifact,
                )
                served_artifact = request.expected_artifact
        except (KeyError, RuntimeError, ValueError) as exc:
            error = _served_recurrent_error(request, exc)
            router.recurrent_replay_store(
                error.response,
                actor_id=request.actor_id,
                actor_incarnation=request.actor_incarnation,
                policy_id=request.policy_id,
                request_id=request.request_id,
            )
            served.append(error)
            continue
        finished = time.perf_counter()
        response = InferenceResponse(
            actor_id=request.actor_id,
            actor_incarnation=request.actor_incarnation,
            request_id=request.request_id,
            policy_id=request.policy_id,
            policy_version=(0 if served_artifact is None else router.policy_version),
            actions=(),
            action_logprobs=torch.empty(0),
            values=torch.empty(0),
            server_received_at=request.server_received_at,
            server_sample_started_at=started,
            server_sample_finished_at=finished,
            request_type="recurrent_release",
            released_recurrent_sequences=released,
            released_sequence_identities=request.sequences,
            served_policy_artifact=served_artifact,
        )
        router.recurrent_replay_store(
            response,
            actor_id=request.actor_id,
            actor_incarnation=request.actor_incarnation,
            policy_id=request.policy_id,
            request_id=request.request_id,
        )
        served.append(
            _ServedRecurrentRequest(
                request=request,
                response=response,
                sample_seconds=finished - started,
                policy_batch=False,
            )
        )
    return tuple(served)


def _served_recurrent_error(
    request: InferenceRequest | RecurrentReleaseRequest,
    error: Exception,
) -> _ServedRecurrentRequest:
    """Turn one actor-owned recurrent protocol failure into an RPC error."""
    now = time.perf_counter()
    response = InferenceResponse(
        actor_id=request.actor_id,
        actor_incarnation=request.actor_incarnation,
        request_id=request.request_id,
        policy_id=request.policy_id,
        policy_version=0,
        actions=(),
        action_logprobs=torch.empty(0),
        values=torch.empty(0),
        server_received_at=request.server_received_at,
        server_sample_started_at=now,
        server_sample_finished_at=now,
        request_type=request.request_type,
        error_type=type(error).__name__,
        error_message=str(error),
        requested_batch_size=request.batch_size,
    )
    return _ServedRecurrentRequest(
        request=request,
        response=response,
        sample_seconds=0.0,
        policy_batch=False,
    )


def _recurrent_snapshot_router(
    policies: Mapping[str, InferencePolicy],
    policy_id: str,
) -> InferencePolicySnapshotRouter:
    """Require the immutable router used by recurrent serving."""
    policy = _policy_for_id(policies, policy_id)
    if not isinstance(policy, InferencePolicySnapshotRouter):
        raise RuntimeError("recurrent inference requires a snapshot router")
    if not policy.recurrent_enabled:
        raise RuntimeError("recurrent request targeted a stateless snapshot")
    return policy


def _admit_recurrent_actor_incarnation(
    policies: Mapping[str, InferencePolicy],
    *,
    actor_id: str,
    actor_incarnation: int,
) -> int:
    """Reclaim one replaced actor's leases atomically across loaded policies."""
    routers: list[InferencePolicySnapshotRouter] = []
    seen: set[int] = set()
    for policy in policies.values():
        if not isinstance(policy, InferencePolicySnapshotRouter):
            continue
        identity = id(policy)
        if identity in seen:
            continue
        seen.add(identity)
        routers.append(policy)

    # Validate every router before mutating any of them.  A newly hot-loaded
    # frozen route may not have observed the replacement actor yet while the
    # candidate route already has; a delayed predecessor request must not move
    # only that new route backwards and leave the registry inconsistent.
    stale_currents = tuple(
        current
        for policy in routers
        if (current := policy.recurrent_actor_incarnation(actor_id)) is not None
        and actor_incarnation < current
    )
    if stale_currents:
        raise RuntimeError(
            "stale recurrent actor incarnation: "
            f"actor={actor_id}, request={actor_incarnation}, "
            f"current={max(stale_currents)}"
        )

    released = 0
    for policy in routers:
        released += policy.admit_recurrent_actor_incarnation(
            actor_id=actor_id,
            actor_incarnation=actor_incarnation,
        )
    return released


def _admit_latest_recurrent_actor_incarnations(
    requests: Sequence[InferenceQueueRequest],
    *,
    policies: Mapping[str, InferencePolicy],
) -> None:
    """Advance every actor generation before any request in the step runs."""
    latest: dict[str, int] = {}
    for request in requests:
        is_recurrent = isinstance(request, RecurrentReleaseRequest) or (
            isinstance(request, InferenceRequest) and request.recurrent is not None
        )
        if not is_recurrent:
            continue
        latest[request.actor_id] = max(
            request.actor_incarnation,
            latest.get(request.actor_id, -1),
        )
    for actor_id, actor_incarnation in latest.items():
        try:
            _admit_recurrent_actor_incarnation(
                policies,
                actor_id=actor_id,
                actor_incarnation=actor_incarnation,
            )
        except RuntimeError:
            # A batch containing only delayed predecessor requests is rejected
            # request-by-request below; it must not terminate the server step.
            continue


def _move_public_event_batch(
    batch: PublicEventBatch,
    *,
    device: torch.device,
) -> PublicEventBatch:
    """Move every event tensor together across the RPC device boundary."""
    return PublicEventBatch(
        **{
            name: cast(Tensor, getattr(batch, name)).to(device=device)
            for name in PublicEventBatch.__dataclass_fields__
        }
    )


def _move_recurrent_policy_state(
    state: RecurrentPolicyState,
    *,
    device: torch.device,
) -> RecurrentPolicyState:
    """Move one actor-owned state without changing its declared dtype."""
    return RecurrentPolicyState(
        hidden=state.hidden.to(device=device),
        cell=state.cell.to(device=device),
    )


def _recurrent_state_to_cpu(state: RecurrentPolicyState) -> RecurrentPolicyState:
    """Detach a proposed state before crossing the process boundary."""
    return RecurrentPolicyState(
        hidden=state.hidden.detach().cpu(),
        cell=state.cell.detach().cpu(),
    )


def _recurrent_trace_to_cpu(
    trace: SampleDecodeTrace,
    *,
    policy_version: int,
    artifact: PolicyArtifactIdentity,
) -> SampleDecodeTrace:
    """Detach behavior evidence and bind the snapshot that produced it."""
    if trace.planner_context_handles:
        raise RuntimeError("recurrent behavior decode retained planner contexts")
    return replace(
        trace,
        action_logprobs=trace.action_logprobs.detach().cpu(),
        values=trace.values.detach().cpu(),
        token_logprobs=trace.token_logprobs.detach().cpu(),
        prefix_values=trace.prefix_values.detach().cpu(),
        token_mask=trace.token_mask.detach().cpu(),
        stop_sampled=trace.stop_sampled.detach().cpu(),
        served_policy_version=policy_version,
        served_model_fingerprint=artifact.model_fingerprint,
    )


def _planner_proposal_stage_groups(
    requests: Sequence[InferenceRequest],
) -> tuple[tuple[InferenceRequest, ...], ...]:
    """Keep an iterative timeout local to one proposal batching identity."""
    grouped: dict[
        tuple[str, str, int | None, str, bool, str],
        list[InferenceRequest],
    ] = {}
    for request in requests:
        payload = request.planner_proposal_request
        if payload is None:
            raise ValueError("planner proposal request payload is missing")
        identity = (*_inference_batch_identity(request), payload.batch_identity)
        grouped.setdefault(identity, []).append(request)
    return tuple(tuple(group) for group in grouped.values())


def _drain_requests(
    request_queue: RequestQueue,
    config: InferenceServerConfig,
) -> _DrainedInferenceRequests:
    requests: list[InferenceQueueRequest] = []
    expired_teacher: list[InferenceRequest] = []
    expired_planner: list[InferenceRequest] = []
    oversized: list[InferenceQueueRequest] = []
    scanned_requests = 0
    deadline: float | None = None
    first_request_wait_seconds = 0.0
    coalescing_wait_seconds = 0.0
    blocking_wait_calls = 0
    nonblocking_drain_calls = 0
    deferred_requests = 0
    wait_budget_seconds = config.max_wait_ms / 1000.0
    while (
        _drained_decisions(requests) < config.max_batch
        and scanned_requests < config.max_batch
    ):
        block = False
        timeout = 0.0
        waiting_for_first_valid = not requests
        if waiting_for_first_valid:
            block = True
            timeout = wait_budget_seconds
        elif _needs_min_batch_wait(requests, config):
            if deadline is None:
                raise RuntimeError("inference coalescing deadline was not initialized")
            timeout = max(0.0, deadline - time.perf_counter())
            if timeout <= 0.0:
                break
            block = True
        get_started_at = time.perf_counter()
        try:
            request = _next_drain_request(
                request_queue,
                block=block,
                timeout=timeout,
            )
        except queue.Empty:
            elapsed = time.perf_counter() - get_started_at
            if block:
                blocking_wait_calls += 1
                if waiting_for_first_valid:
                    first_request_wait_seconds += elapsed
                else:
                    coalescing_wait_seconds += elapsed
            else:
                nonblocking_drain_calls += 1
            break
        except (EOFError, ConnectionError):
            continue
        elapsed = time.perf_counter() - get_started_at
        if block:
            blocking_wait_calls += 1
            if waiting_for_first_valid:
                first_request_wait_seconds += elapsed
            else:
                coalescing_wait_seconds += elapsed
        else:
            nonblocking_drain_calls += 1
        scanned_requests += 1
        received = (
            request
            if request.server_received_at > 0.0
            else replace(request, server_received_at=time.perf_counter())
        )
        if isinstance(received, InferenceRequest) and (
            _is_expired_deadline_request(received)
        ):
            destination = (
                expired_teacher if received.purpose == "teacher" else expired_planner
            )
            destination.append(received)
            continue
        if _oversized_request_reason(received, config=config) is not None:
            oversized.append(received)
            continue
        if requests and (
            _drained_decisions(requests) + received.batch_size > config.max_batch
        ):
            _defer_drain_request(request_queue, received)
            deferred_requests += 1
            break
        requests.append(received)
        if len(requests) == 1:
            # The configured coalescing window is useful only after a request
            # that can actually enter this batch has arrived. Starting it before
            # the initial blocking get silently consumes the budget while idle.
            deadline = time.perf_counter() + wait_budget_seconds
    requests = list(fair_schedule_requests(requests, config=config))
    min_batch_target = (
        min(config.min_batch_decisions, config.max_batch)
        if config.min_batch_decisions > 0
        else 0
    )
    return _DrainedInferenceRequests(
        requests=tuple(requests),
        expired_teacher=tuple(expired_teacher),
        expired_planner=tuple(expired_planner),
        oversized=tuple(oversized),
        coalescing_stats={
            "scanned_requests": scanned_requests,
            "accepted_requests": len(requests),
            "coalesced_requests": max(0, len(requests) - 1),
            "accepted_decisions": _drained_decisions(requests),
            "min_batch_target_decisions": min_batch_target,
            "min_batch_target_reached": bool(
                min_batch_target > 0
                and _drained_decisions(requests) >= min_batch_target
            ),
            "blocking_wait_calls": blocking_wait_calls,
            "nonblocking_drain_calls": nonblocking_drain_calls,
            "first_request_wait_seconds": first_request_wait_seconds,
            "coalescing_wait_seconds": coalescing_wait_seconds,
            "deferred_requests": deferred_requests,
        },
    )


def _next_drain_request(
    request_queue: RequestQueue,
    *,
    block: bool,
    timeout: float,
) -> InferenceQueueRequest:
    """Read a previously deferred whole request before touching the queue."""
    key = id(request_queue)
    with _DRAIN_LOOKAHEAD_LOCK:
        entry = _DRAIN_LOOKAHEAD.get(key)
        if entry is not None and entry[0] is request_queue and entry[1]:
            request = entry[1].popleft()
            if not entry[1]:
                _DRAIN_LOOKAHEAD.pop(key, None)
            return request
    return request_queue.get(block=block, timeout=timeout)


def _defer_drain_request(
    request_queue: RequestQueue,
    request: InferenceQueueRequest,
) -> None:
    """Retain one whole lookahead request for the next bounded server step."""
    key = id(request_queue)
    with _DRAIN_LOOKAHEAD_LOCK:
        entry = _DRAIN_LOOKAHEAD.get(key)
        if entry is None or entry[0] is not request_queue:
            pending: deque[InferenceQueueRequest] = deque()
            _DRAIN_LOOKAHEAD[key] = (request_queue, pending)
        else:
            pending = entry[1]
        pending.append(request)


def _oversized_request_reason(
    request: InferenceQueueRequest,
    *,
    config: InferenceServerConfig,
) -> str | None:
    """Return the violated whole-request or categorical microbatch bound."""
    stage_cap = inference_stage_row_cap(request.request_type, config=config)
    if request.batch_size > stage_cap:
        return f"{request.request_type} request exceeds its stage row cap"
    if request.request_type != "planner_candidates":
        return None
    groups = request.candidate_actions or ()
    if any(len(group) > config.max_planner_candidate_rows for group in groups):
        return "planner categorical support exceeds its candidate row cap"
    return None


def _is_expired_deadline_request(request: InferenceQueueRequest) -> bool:
    """Return whether deadline-bound work can no longer reach its caller."""
    return (
        request.purpose in ("teacher", "planner_behavior")
        and time.monotonic() >= request.deadline_monotonic
    )


def _reject_deadline_requests(
    requests: Sequence[InferenceRequest],
    *,
    response_queues: Mapping[str, ResponseQueue],
) -> None:
    """Notify callers that action-critical or auxiliary work is stale."""
    for request in requests:
        now = time.perf_counter()
        planner = request.purpose == "planner_behavior"
        response = InferenceResponse(
            actor_id=request.actor_id,
            actor_incarnation=request.actor_incarnation,
            request_id=request.request_id,
            policy_id=request.policy_id,
            policy_version=0,
            actions=(),
            action_logprobs=torch.empty(0),
            values=torch.empty(0),
            server_received_at=request.server_received_at,
            server_sample_started_at=now,
            server_sample_finished_at=now,
            server_put_at=now,
            request_type=request.request_type,
            error_type=(
                "PlannerDeadlineExpired" if planner else "TeacherDeadlineExpired"
            ),
            error_message=(
                "planner inference request expired before service"
                if planner
                else "teacher inference request expired before service"
            ),
            requested_batch_size=request.batch_size,
        )
        _response_queue_for_actor(response_queues, request.actor_id).put(response)


def _reject_oversized_requests(
    requests: Sequence[InferenceQueueRequest],
    *,
    response_queues: Mapping[str, ResponseQueue],
) -> None:
    """Return explicit errors for whole requests that cannot fit a hard cap."""
    for request in requests:
        now = time.perf_counter()
        response = InferenceResponse(
            actor_id=request.actor_id,
            actor_incarnation=request.actor_incarnation,
            request_id=request.request_id,
            policy_id=request.policy_id,
            policy_version=0,
            actions=(),
            action_logprobs=torch.empty(0),
            values=torch.empty(0),
            server_received_at=request.server_received_at,
            server_sample_started_at=now,
            server_sample_finished_at=now,
            server_put_at=now,
            request_type=request.request_type,
            error_type="InferenceBatchTooLarge",
            error_message="inference request exceeds a configured hard row cap",
            requested_batch_size=request.batch_size,
        )
        _response_queue_for_actor(response_queues, request.actor_id).put(response)


def _reject_unavailable_policy_requests(
    requests: Sequence[InferenceQueueRequest],
    *,
    response_queues: Mapping[str, ResponseQueue],
) -> None:
    """Reject one stale/unknown route without terminating shared inference."""
    for request in requests:
        now = time.perf_counter()
        response = InferenceResponse(
            actor_id=request.actor_id,
            actor_incarnation=request.actor_incarnation,
            request_id=request.request_id,
            policy_id=request.policy_id,
            policy_version=0,
            actions=(),
            action_logprobs=torch.empty(0),
            values=torch.empty(0),
            server_received_at=request.server_received_at,
            server_sample_started_at=now,
            server_sample_finished_at=now,
            server_put_at=now,
            request_type=request.request_type,
            error_type="InferencePolicyUnavailable",
            error_message=(
                f"inference policy route is not loaded: {request.policy_id}"
            ),
            requested_batch_size=request.batch_size,
        )
        _response_queue_for_actor(response_queues, request.actor_id).put(response)


def _drop_unavailable_planner_leases(
    requests: Sequence[InferenceQueueRequest],
    *,
    policies: Mapping[str, InferencePolicy],
) -> tuple[tuple[InferenceQueueRequest, ...], tuple[InferenceRequest, ...]]:
    """Reject stale planner versions before mixing rows into a GPU batch."""
    live: list[InferenceQueueRequest] = []
    rejected: list[InferenceRequest] = []
    for request in requests:
        if not isinstance(request, InferenceRequest):
            live.append(request)
            continue
        if request.purpose != "planner_behavior":
            live.append(request)
            continue
        if request.model_version_lease is None:
            live.append(request)
            continue
        policy = _policy_for_id(policies, request.policy_id)
        if not _supports_model_version_lease(
            policy,
            request.model_version_lease,
        ):
            rejected.append(request)
        else:
            live.append(request)
    return (tuple(live), tuple(rejected))


def _reject_unavailable_planner_leases(
    requests: Sequence[InferenceRequest],
    *,
    response_queues: Mapping[str, ResponseQueue],
) -> None:
    """Return a compact explicit-fallback signal for unavailable snapshots."""
    for request in requests:
        now = time.perf_counter()
        response = InferenceResponse(
            actor_id=request.actor_id,
            actor_incarnation=request.actor_incarnation,
            request_id=request.request_id,
            policy_id=request.policy_id,
            policy_version=0,
            actions=(),
            action_logprobs=torch.empty(0),
            values=torch.empty(0),
            server_received_at=request.server_received_at,
            server_sample_started_at=now,
            server_sample_finished_at=now,
            server_put_at=now,
            request_type=request.request_type,
            error_type="PlannerModelLeaseUnavailable",
            error_message="planner model-version lease is no longer resident",
            requested_batch_size=request.batch_size,
        )
        _response_queue_for_actor(response_queues, request.actor_id).put(response)


def _needs_min_batch_wait(
    requests: Sequence[InferenceQueueRequest],
    config: InferenceServerConfig,
) -> bool:
    if config.min_batch_decisions <= 0 or config.max_wait_ms <= 0.0:
        return False
    min_decisions = min(config.min_batch_decisions, config.max_batch)
    return _drained_decisions(requests) < min_decisions


def _drained_decisions(requests: Sequence[InferenceQueueRequest]) -> int:
    return sum(request.batch_size for request in requests)


def _inference_request_key(
    request: InferenceQueueRequest,
) -> tuple[str, int, str, int]:
    return (
        request.actor_id,
        request.actor_incarnation,
        request.policy_id,
        request.request_id,
    )


def _decode_options(request: InferenceRequest) -> OptionBatch:
    """Return options after enforcing decode-request semantics."""
    if request.request_type != "decode" or request.options is None:
        raise ValueError("decode inference requires an option batch")
    return request.options


def _inference_row_result(
    sample: _PolicyGroupSample,
    *,
    sample_index: int,
    policy_version: int,
    sample_started_at: float,
    sample_finished_at: float,
) -> _InferenceRowResult:
    return _InferenceRowResult(
        policy_version=policy_version,
        action=sample.actions[sample_index],
        action_logprob=sample.logprobs[sample_index],
        value=sample.values[sample_index],
        token_logprobs=_optional_tensor_row(sample.token_logprobs, sample_index),
        prefix_values=_optional_tensor_row(sample.prefix_values, sample_index),
        token_mask=_optional_tensor_row(sample.token_mask, sample_index),
        stop_sampled=_optional_tensor_row(sample.stop_sampled, sample_index),
        planner_base_action_logprobs=None,
        planner_proposal_action_logprobs=None,
        planner_reranker_residuals=None,
        planner_proposal_decision=None,
        planner_base_greedy_action=None,
        planner_context_handle=(
            None
            if not sample.planner_context_handles
            else sample.planner_context_handles[sample_index]
        ),
        model_fingerprint=sample.served_model_fingerprint,
        proposal_version=(
            0
            if sample.served_proposal_version is None
            else sample.served_proposal_version
        ),
        planner_fallback_reason=sample.planner_fallback_reason,
        sample_started_at=sample_started_at,
        sample_finished_at=sample_finished_at,
    )


def _value_inference_row_result(
    value: Tensor,
    *,
    policy_version: int,
    sample_started_at: float,
    sample_finished_at: float,
) -> _InferenceRowResult:
    """Build one value-only row without any decode evidence."""
    return _InferenceRowResult(
        policy_version=policy_version,
        action=(),
        action_logprob=value.new_zeros(()),
        value=value,
        token_logprobs=None,
        prefix_values=None,
        token_mask=None,
        stop_sampled=None,
        planner_base_action_logprobs=None,
        planner_proposal_action_logprobs=None,
        planner_reranker_residuals=None,
        planner_proposal_decision=None,
        planner_base_greedy_action=None,
        planner_context_handle=None,
        model_fingerprint="",
        proposal_version=0,
        planner_fallback_reason="",
        sample_started_at=sample_started_at,
        sample_finished_at=sample_finished_at,
    )


def _planner_candidate_inference_row_result(
    *,
    base_action_logprobs: Tensor,
    proposal_action_logprobs: Tensor,
    reranker_residuals: Tensor,
    policy_version: int,
    sample_started_at: float,
    sample_finished_at: float,
) -> _InferenceRowResult:
    """Build one ragged candidate row without decode or value evidence."""
    if (
        base_action_logprobs.ndim != 1
        or proposal_action_logprobs.shape != base_action_logprobs.shape
        or reranker_residuals.shape != base_action_logprobs.shape
        or int(base_action_logprobs.shape[0]) <= 0
    ):
        raise RuntimeError("planner candidate result row is misaligned")
    zero = base_action_logprobs.new_zeros(())
    return _InferenceRowResult(
        policy_version=policy_version,
        action=(),
        action_logprob=zero,
        value=zero,
        token_logprobs=None,
        prefix_values=None,
        token_mask=None,
        stop_sampled=None,
        planner_base_action_logprobs=base_action_logprobs,
        planner_proposal_action_logprobs=proposal_action_logprobs,
        planner_reranker_residuals=reranker_residuals,
        planner_proposal_decision=None,
        planner_base_greedy_action=None,
        planner_context_handle=None,
        model_fingerprint="",
        proposal_version=0,
        planner_fallback_reason="",
        sample_started_at=sample_started_at,
        sample_finished_at=sample_finished_at,
    )


def _planner_proposal_inference_row_result(
    *,
    decision: PlannerProposalDecisionResult,
    base_greedy_action: tuple[int, ...],
    policy_version: int,
    sample_started_at: float,
    sample_finished_at: float,
) -> _InferenceRowResult:
    """Build one proposal row without copying group-only batch telemetry."""
    zero = torch.zeros(())
    return _InferenceRowResult(
        policy_version=policy_version,
        action=(),
        action_logprob=zero,
        value=zero,
        token_logprobs=None,
        prefix_values=None,
        token_mask=None,
        stop_sampled=None,
        planner_base_action_logprobs=None,
        planner_proposal_action_logprobs=None,
        planner_reranker_residuals=None,
        planner_proposal_decision=decision,
        planner_base_greedy_action=base_greedy_action,
        planner_context_handle=None,
        model_fingerprint="",
        proposal_version=0,
        planner_fallback_reason="",
        sample_started_at=sample_started_at,
        sample_finished_at=sample_finished_at,
    )


def _policy_group_sample_to_cpu(sample: _PolicyGroupSample) -> _PolicyGroupSample:
    """Transfer sampled tensors once per policy group instead of once per row."""
    return replace(
        sample,
        logprobs=sample.logprobs.detach().cpu(),
        values=sample.values.detach().cpu(),
        token_logprobs=_optional_cpu_tensor(sample.token_logprobs),
        prefix_values=_optional_cpu_tensor(sample.prefix_values),
        token_mask=_optional_cpu_tensor(sample.token_mask),
        stop_sampled=_optional_cpu_tensor(sample.stop_sampled),
    )


def _optional_cpu_tensor(tensor: Tensor | None) -> Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().cpu()


def _optional_tensor_row(tensor: Tensor | None, row_index: int) -> Tensor | None:
    if tensor is None:
        return None
    return tensor[row_index]


def _inference_response_from_rows(
    request: InferenceRequest,
    rows: tuple[_InferenceRowResult, ...],
    *,
    server_put_at: float,
) -> InferenceResponse:
    versions = {row.policy_version for row in rows}
    if len(versions) != 1:
        raise RuntimeError("inference request rows returned different policy versions")
    model_fingerprints = {row.model_fingerprint for row in rows}
    proposal_versions = {row.proposal_version for row in rows}
    planner_fallback_reasons = {row.planner_fallback_reason for row in rows}
    if (
        len(model_fingerprints) != 1
        or len(proposal_versions) != 1
        or len(planner_fallback_reasons) != 1
    ):
        raise RuntimeError("inference request rows returned different model identities")
    if request.request_type == "planner_proposals":
        decisions = tuple(row.planner_proposal_decision for row in rows)
        if any(decision is None for decision in decisions):
            raise RuntimeError("planner proposal rows are incomplete")
        proposal_request = request.planner_proposal_request
        if proposal_request is None:
            raise RuntimeError("planner proposal request payload was lost")
        required = tuple(
            cast(PlannerProposalDecisionResult, decision) for decision in decisions
        )
        if any(row.planner_base_greedy_action is None for row in rows):
            raise RuntimeError("planner proposal base greedy rows are incomplete")
        payload = PlannerProposalResponsePayload(
            decisions=required,
            base_greedy_actions=tuple(
                cast(tuple[int, ...], row.planner_base_greedy_action) for row in rows
            ),
            search_fingerprint=proposal_request.limits.fingerprint,
            scoring_rows=sum(item.nodes_expanded for item in required),
        )
        empty = torch.empty(0)
        return InferenceResponse(
            actor_id=request.actor_id,
            actor_incarnation=request.actor_incarnation,
            request_id=request.request_id,
            policy_id=request.policy_id,
            policy_version=versions.pop(),
            actions=(),
            action_logprobs=empty,
            values=empty,
            server_received_at=request.server_received_at,
            server_sample_started_at=min(row.sample_started_at for row in rows),
            server_sample_finished_at=max(row.sample_finished_at for row in rows),
            server_put_at=server_put_at,
            request_type="planner_proposals",
            planner_proposal_response=payload,
        )
    if request.request_type == "planner_candidates":
        planner_fields = tuple(
            (
                row.planner_base_action_logprobs,
                row.planner_proposal_action_logprobs,
                row.planner_reranker_residuals,
            )
            for row in rows
        )
        if any(any(item is None for item in fields) for fields in planner_fields):
            raise RuntimeError("planner candidate rows are incomplete")
        base = torch.cat([cast(Tensor, fields[0]) for fields in planner_fields])
        proposal = torch.cat([cast(Tensor, fields[1]) for fields in planner_fields])
        residuals = torch.cat([cast(Tensor, fields[2]) for fields in planner_fields])
        counts = tuple(
            int(cast(Tensor, fields[0]).shape[0]) for fields in planner_fields
        )
        return InferenceResponse(
            actor_id=request.actor_id,
            actor_incarnation=request.actor_incarnation,
            request_id=request.request_id,
            policy_id=request.policy_id,
            policy_version=versions.pop(),
            actions=(),
            action_logprobs=base.new_empty((0,)),
            values=base.new_empty((0,)),
            server_received_at=request.server_received_at,
            server_sample_started_at=min(row.sample_started_at for row in rows),
            server_sample_finished_at=max(row.sample_finished_at for row in rows),
            server_put_at=server_put_at,
            request_type="planner_candidates",
            planner_base_action_logprobs=base,
            planner_proposal_action_logprobs=proposal,
            planner_reranker_residuals=residuals,
            planner_candidate_counts=counts,
        )
    values = torch.stack([row.value for row in rows])
    if request.request_type != "decode":
        return InferenceResponse(
            actor_id=request.actor_id,
            actor_incarnation=request.actor_incarnation,
            request_id=request.request_id,
            policy_id=request.policy_id,
            policy_version=versions.pop(),
            actions=(),
            action_logprobs=values.new_empty((0,)),
            values=values,
            server_received_at=request.server_received_at,
            server_sample_started_at=min(row.sample_started_at for row in rows),
            server_sample_finished_at=max(row.sample_finished_at for row in rows),
            server_put_at=server_put_at,
            request_type=request.request_type,
        )
    raw_handles = tuple(row.planner_context_handle for row in rows)
    if any(handle is None for handle in raw_handles) and any(
        handle is not None for handle in raw_handles
    ):
        raise RuntimeError("decode rows mixed planner context handle presence")
    context_handles = (
        ()
        if all(handle is None for handle in raw_handles)
        else tuple(cast(str, handle) for handle in raw_handles)
    )
    return InferenceResponse(
        actor_id=request.actor_id,
        actor_incarnation=request.actor_incarnation,
        request_id=request.request_id,
        policy_id=request.policy_id,
        policy_version=versions.pop(),
        actions=tuple(row.action for row in rows),
        action_logprobs=torch.stack([row.action_logprob for row in rows]),
        values=values,
        token_logprobs=_stack_optional_trace_rows(
            tuple(row.token_logprobs for row in rows),
            fill_value=0.0,
        ),
        prefix_values=_stack_optional_trace_rows(
            tuple(row.prefix_values for row in rows),
            fill_value=0.0,
        ),
        token_mask=_stack_optional_trace_rows(
            tuple(row.token_mask for row in rows),
            fill_value=False,
        ),
        stop_sampled=_stack_optional_scalar_rows(
            tuple(row.stop_sampled for row in rows)
        ),
        server_received_at=request.server_received_at,
        server_sample_started_at=min(row.sample_started_at for row in rows),
        server_sample_finished_at=max(row.sample_finished_at for row in rows),
        server_put_at=server_put_at,
        request_type="decode",
        planner_context_handles=context_handles,
        model_fingerprint=model_fingerprints.pop(),
        proposal_version=proposal_versions.pop(),
        planner_fallback_reason=planner_fallback_reasons.pop(),
    )


def _stack_optional_trace_rows(
    rows: tuple[Tensor | None, ...],
    *,
    fill_value: float | bool,
) -> Tensor | None:
    if all(row is None for row in rows):
        return None
    if any(row is None for row in rows):
        raise RuntimeError("route groups returned inconsistent token trace presence")
    tensors = tuple(cast(Tensor, row) for row in rows)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise RuntimeError("token trace rows must be one-dimensional")
    width = max(int(tensor.shape[0]) for tensor in tensors)
    output = tensors[0].new_full((len(tensors), width), fill_value)
    for row_index, tensor in enumerate(tensors):
        output[row_index, : tensor.shape[0]] = tensor
    return output


def _stack_optional_scalar_rows(
    rows: tuple[Tensor | None, ...],
) -> Tensor | None:
    if all(row is None for row in rows):
        return None
    if any(row is None for row in rows):
        raise RuntimeError("route groups returned inconsistent STOP trace presence")
    tensors = tuple(cast(Tensor, row) for row in rows)
    if any(tensor.ndim != 0 for tensor in tensors):
        raise RuntimeError("STOP trace rows must be scalar")
    return torch.stack(tensors)


def _batch_size_histogram(
    requests: Sequence[InferenceQueueRequest],
) -> dict[int, int]:
    counts: Counter[int] = Counter(request.batch_size for request in requests)
    return dict(sorted(counts.items()))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile < 0.0 or percentile > 1.0:
        raise ValueError("percentile must be in [0, 1]")
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(percentile * len(ordered)) - 1),
    )
    return ordered[index]


def _group_request_rows(
    requests: Sequence[InferenceRequest],
    *,
    policies: Mapping[str, InferencePolicy],
    group_by_exact_deck: bool,
) -> dict[
    tuple[str, str, str, float, str, int | None, str, bool],
    list[_InferenceRequestRow],
]:
    """Group rows for one model call, isolating decks only for graph replay."""
    grouped: dict[
        tuple[str, str, str, float, str, int | None, str, bool],
        list[_InferenceRequestRow],
    ] = {}
    for request in requests:
        policy = _policy_for_id(policies, request.policy_id)
        for row_index, signature in enumerate(request.decks.signatures):
            deck_key = signature if group_by_exact_deck else "mixed"
            route_key = (
                _policy_private_route_key(policy, signature)
                if group_by_exact_deck
                else "mixed"
            )
            key = (
                request.policy_id,
                deck_key,
                route_key,
                float(request.temperature),
                request.purpose,
                request.model_version_lease,
                request.tensor_schema_fingerprint,
                _request_retains_planner_context(request),
            )
            grouped.setdefault(key, []).append(
                _InferenceRequestRow(request=request, row_index=row_index)
            )
    return grouped


def _group_whole_requests(
    requests: Sequence[InferenceRequest],
    *,
    policies: Mapping[str, InferencePolicy],
) -> dict[
    tuple[str, float, str, int | None, str, bool],
    list[InferenceRequest],
]:
    """Group eager requests without expanding their already-batched rows."""
    grouped: dict[
        tuple[str, float, str, int | None, str, bool],
        list[InferenceRequest],
    ] = {}
    for request in requests:
        _policy_for_id(policies, request.policy_id)
        key = (
            request.policy_id,
            float(request.temperature),
            request.purpose,
            request.model_version_lease,
            request.tensor_schema_fingerprint,
            _request_retains_planner_context(request),
        )
        grouped.setdefault(key, []).append(request)
    return grouped


def _inference_batch_identity(
    request: InferenceRequest,
) -> tuple[str, str, int | None, str, bool]:
    """Return identities that must never be mixed in one model call."""
    return (
        request.policy_id,
        request.purpose,
        request.model_version_lease,
        request.tensor_schema_fingerprint,
        _request_retains_planner_context(request),
    )


def _request_retains_planner_context(request: InferenceRequest) -> bool:
    """Resolve legacy root-decode ownership to an explicit boolean."""
    if request.retain_planner_context is not None:
        return bool(request.retain_planner_context)
    return bool(
        request.purpose == "planner_behavior"
        and request.request_type == "decode"
        and request.model_version_lease is None
    )


def _require_group_model_lease(
    policy: InferencePolicy,
    requests: Sequence[InferenceRequest],
) -> None:
    """Recheck immutable planner version immediately before GPU execution."""
    planner = tuple(
        request for request in requests if request.purpose == "planner_behavior"
    )
    if not planner:
        return
    identities = {_inference_batch_identity(request) for request in planner}
    if len(identities) != 1 or len(planner) != len(requests):
        raise RuntimeError("planner inference group mixed lease/schema identities")
    (identity,) = identities
    lease = identity[2]
    if lease is None:
        if any(request.request_type != "decode" for request in planner):
            raise RuntimeError("only planner root decode may be unbound")
        return
    if not _supports_model_version_lease(policy, lease):
        raise RuntimeError("planner model-version lease changed before inference")


def _supports_model_version_lease(
    policy: InferencePolicy,
    version: int,
) -> bool:
    """Return whether an exact immutable snapshot remains routable."""
    supports = getattr(policy, "supports_model_version_lease", None)
    if callable(supports):
        return bool(cast(Any, supports)(int(version)))
    return int(getattr(policy, "policy_version", 0)) == int(version)


def _leased_policy_for_group(
    policy: InferencePolicy,
    requests: Sequence[InferenceRequest],
) -> InferencePolicy:
    """Resolve the resident snapshot after validating a bound request group."""
    _require_group_model_lease(policy, requests)
    leases = {
        request.model_version_lease
        for request in requests
        if request.purpose == "planner_behavior"
    }
    if not leases or leases == {None}:
        return policy
    if len(leases) != 1 or None in leases:
        raise RuntimeError("planner inference group mixed model leases")
    version = cast(int, next(iter(leases)))
    resolver = getattr(policy, "policy_for_lease", None)
    if not callable(resolver):
        return policy
    return cast(InferencePolicy, cast(Any, resolver)(version))


def _policy_private_route_key(policy: InferencePolicy, signature: str) -> str:
    """Return the model-relative private key used only for serving grouping."""
    model = getattr(policy, "_model", policy)
    config = getattr(model, "config", None)
    conditioning = getattr(config, "deck_conditioning", None)
    if conditioning is None or not bool(getattr(conditioning, "enabled", False)):
        return "legacy"
    profile_by_signature = getattr(conditioning, "profile_by_signature", {})
    if not isinstance(profile_by_signature, Mapping):
        return "generic"
    profile = profile_by_signature.get(signature)
    return "generic" if profile is None else str(profile.module_key)


def _sample_policy_groups(
    *,
    requests: Sequence[InferenceRequest],
    policies: Mapping[str, InferencePolicy],
    config: InferenceServerConfig,
) -> tuple[_ServedPolicyGroup, ...]:
    if any(request.request_type != "decode" for request in requests):
        raise ValueError("decode policy groups cannot contain value requests")
    if config.graph_decode:
        prepared = _prepare_graph_decode_policy_groups(
            requests=requests,
            policies=policies,
            config=config,
        )
    else:
        grouped_requests = _group_whole_requests(requests, policies=policies)
        prepared = tuple(
            _prepare_whole_request_policy_group(
                policy_id=group_key[0],
                policy=_policy_for_id(policies, group_key[0]),
                requests=tuple(group),
                config=config,
            )
            for group_key, group in grouped_requests.items()
        )
    if config.graph_decode:
        return _sample_graph_policy_groups_on_one_stream(prepared)
    if _can_sample_policy_groups_on_cuda_streams(prepared, config=config):
        return _sample_policy_groups_on_cuda_streams(prepared)
    return tuple(_sample_prepared_policy_group(group) for group in prepared)


def _sample_graph_policy_groups_on_one_stream(
    prepared_groups: Sequence[_PreparedPolicyGroup],
) -> tuple[_ServedPolicyGroup, ...]:
    """Submit adjacent static replays before one shared-stream synchronization."""
    served: list[_ServedPolicyGroup] = []
    pending: list[
        tuple[
            _PreparedPolicyGroup,
            _CudaStreamSlot,
            _TensorPolicyGroupSample,
            float,
        ]
    ] = []
    pending_graph_keys: set[tuple[object, ...]] = set()

    def flush() -> None:
        if not pending:
            return
        device = pending[0][0].states.card_ids.device
        cast(Any, torch.cuda.current_stream(device=device)).synchronize()
        for prepared, slot, tensor_sample, sample_start in pending:
            sample_seconds = (
                float(cast(Any, slot.started).elapsed_time(slot.finished)) / 1000.0
            )
            sample = _policy_group_sample_from_tensors(
                tensor_sample,
                prepared=prepared,
            )
            served.append(
                _ServedPolicyGroup(
                    policy_id=prepared.policy_id,
                    rows=prepared.rows,
                    policy_version=int(getattr(prepared.policy, "policy_version", 0)),
                    shape=prepared.shape,
                    sample=sample,
                    sample_seconds=sample_seconds,
                    sample_started_at=sample_start,
                    sample_finished_at=sample_start + sample_seconds,
                    serving_path=prepared.serving_path,
                )
            )
        pending.clear()
        pending_graph_keys.clear()

    for slot_index, prepared in enumerate(prepared_groups):
        trace_sampler = _tensor_trace_decode_sampler(prepared.policy)
        tensor_sampler = _tensor_decode_sampler(prepared.policy)
        eligible = (
            prepared.bucket_key is not None
            and prepared.states.card_ids.device.type == "cuda"
            and (trace_sampler is not None or tensor_sampler is not None)
            and all(row.request.purpose != "planner_behavior" for row in prepared.rows)
        )
        if not eligible:
            flush()
            served.append(_sample_prepared_policy_group(prepared))
            continue
        graph_key = _prepared_graph_replay_key(prepared)
        if graph_key in pending_graph_keys:
            # One captured graph owns one static output buffer. Materialize its
            # previous result before another replay can overwrite that buffer.
            flush()
        device = prepared.states.card_ids.device
        slot = _cuda_stream_slot(device, slot_index)
        sample_start = time.perf_counter()
        cast(Any, slot.started).record()
        tensor_sample = _sample_prepared_policy_group_tensors(prepared)
        cast(Any, slot.finished).record()
        pending.append((prepared, slot, tensor_sample, sample_start))
        pending_graph_keys.add(graph_key)
    flush()
    return tuple(served)


def _prepared_graph_replay_key(
    prepared: _PreparedPolicyGroup,
) -> tuple[object, ...]:
    """Return the static-buffer identity used by one graph decode replay."""
    states = prepared.states
    options = prepared.options
    return (
        id(prepared.policy),
        tuple(states.card_ids.shape),
        int(states.scalars.shape[2]),
        0
        if states.attachment_card_ids is None
        else int(states.attachment_card_ids.shape[1]),
        tuple(options.valid_options.shape),
        int(options.entity_slots.shape[2]),
        int(options.scalars.shape[2]),
        int(options.dynamic_effect_features.shape[2]),
        prepared.max_select_steps,
        str(states.card_ids.device),
        tuple(prepared.decks.signatures),
    )


def _prepare_graph_decode_policy_groups(
    *,
    requests: Sequence[InferenceRequest],
    policies: Mapping[str, InferencePolicy],
    config: InferenceServerConfig,
) -> tuple[_PreparedPolicyGroup, ...]:
    """Prepare packed eager or fixed graph layouts for decode groups."""
    graph_row_cap = (
        min(config.max_batch, max(config.bucket_batch_sizes))
        if config.bucketize
        else config.max_batch
    )
    prepared: list[_PreparedPolicyGroup] = []
    graph_requests: list[InferenceRequest] = []
    if config.packed_mixed_route:
        for request_group_key, group in _group_whole_requests(
            requests,
            policies=policies,
        ).items():
            policy_id = request_group_key[0]
            policy_requests = tuple(group)
            if policy_id in config.packed_mixed_route_policy_ids and all(
                request.purpose != "planner_behavior" for request in policy_requests
            ):
                packed = _prepare_whole_request_policy_group(
                    policy_id=policy_id,
                    policy=_policy_for_id(policies, policy_id),
                    requests=policy_requests,
                    # A changing exact-deck layout would create one CUDA graph
                    # cache identity per actor wave. Keep the mixed batch eager
                    # so the model can run its shared trunk once for all routes.
                    config=config.model_copy(update={"bucketize": False}),
                )
                prepared.append(replace(packed, serving_path="packed_mixed_eager"))
            elif not _policy_graph_decode_enabled(_policy_for_id(policies, policy_id)):
                eager = _prepare_whole_request_policy_group(
                    policy_id=policy_id,
                    policy=_policy_for_id(policies, policy_id),
                    requests=policy_requests,
                    config=config.model_copy(update={"bucketize": False}),
                )
                prepared.append(replace(eager, serving_path="mixed_eager_no_graph"))
            else:
                graph_requests.extend(policy_requests)
    else:
        graph_requests.extend(requests)
    if not graph_requests:
        return tuple(prepared)

    if not config.graph_roster_layout:
        grouped_rows = _group_request_rows(
            graph_requests,
            policies=policies,
            group_by_exact_deck=True,
        )
        prepared.extend(
            replace(
                group_prepared,
                serving_path=(
                    "exact_route_graph"
                    if group_prepared.bucket_key is not None
                    else "exact_route_eager_fallback"
                ),
            )
            for group_key, group in grouped_rows.items()
            for offset in range(0, len(group), graph_row_cap)
            for group_prepared in (
                _prepare_policy_group(
                    policy_id=group_key[0],
                    policy=_policy_for_id(policies, group_key[0]),
                    rows=tuple(group[offset : offset + graph_row_cap]),
                    config=config,
                ),
            )
        )
        return tuple(prepared)

    grouped_rows = _group_request_rows(
        graph_requests,
        policies=policies,
        group_by_exact_deck=False,
    )
    for row_group_key, raw_rows in grouped_rows.items():
        policy_id = row_group_key[0]
        policy = _policy_for_id(policies, policy_id)
        rows = tuple(raw_rows)
        routes = _policy_roster_layout_routes(policy)
        route_row_counts = {route.signature: 0 for route in routes}
        for row in rows:
            signature = row.request.decks.signatures[row.row_index]
            if signature in route_row_counts:
                route_row_counts[signature] += 1
        eligible = (
            policy_id in config.graph_roster_layout_policy_ids
            and len(rows) >= config.graph_roster_layout_min_rows
            and bool(routes)
            and len(routes) * config.graph_roster_slots_per_route <= config.max_batch
            and all(row.request.purpose != "planner_behavior" for row in rows)
        )
        route_signatures = {route.signature for route in routes}
        if (
            eligible
            and all(
                row.request.decks.signatures[row.row_index] in route_signatures
                for row in rows
            )
            and all(
                count <= config.graph_roster_slots_per_route
                for count in route_row_counts.values()
            )
        ):
            prepared.append(
                _prepare_roster_layout_policy_group(
                    policy_id=policy_id,
                    policy=policy,
                    rows=rows,
                    routes=routes,
                    config=config,
                )
            )
            continue
        if policy_id in config.graph_sparse_eager_policy_ids and all(
            row.request.purpose != "planner_behavior" for row in rows
        ):
            # Sparse frozen pilots are cheaper as one actual-size mixed-route
            # call than as many mostly-empty exact-route graph buckets.
            sparse = _prepare_policy_group(
                policy_id=row_group_key[0],
                policy=policy,
                rows=rows,
                config=config.model_copy(update={"bucketize": False}),
            )
            prepared.append(replace(sparse, serving_path="sparse_mixed_eager"))
            continue
        prepared.extend(
            _prepare_exact_route_policy_groups(
                policy_id=policy_id,
                policy=policy,
                rows=rows,
                graph_row_cap=graph_row_cap,
                config=config,
            )
        )
    return tuple(prepared)


def _policy_graph_decode_enabled(policy: InferencePolicy) -> bool:
    """Return whether padding this policy can reach a real CUDA graph runner."""
    return bool(getattr(policy, "graph_decode_enabled", False))


def _prepare_exact_route_policy_groups(
    *,
    policy_id: str,
    policy: InferencePolicy,
    rows: tuple[_InferenceRequestRow, ...],
    graph_row_cap: int,
    config: InferenceServerConfig,
) -> tuple[_PreparedPolicyGroup, ...]:
    """Prepare legacy exact-deck graph groups from an identity-safe row set."""
    grouped: dict[tuple[str, str], list[_InferenceRequestRow]] = {}
    for row in rows:
        signature = row.request.decks.signatures[row.row_index]
        key = (signature, _policy_private_route_key(policy, signature))
        grouped.setdefault(key, []).append(row)
    return tuple(
        replace(
            group_prepared,
            serving_path=(
                "exact_route_graph"
                if group_prepared.bucket_key is not None
                else "exact_route_eager_fallback"
            ),
        )
        for group in grouped.values()
        for offset in range(0, len(group), graph_row_cap)
        for group_prepared in (
            _prepare_policy_group(
                policy_id=policy_id,
                policy=policy,
                rows=tuple(group[offset : offset + graph_row_cap]),
                config=config,
            ),
        )
    )


def _policy_roster_layout_routes(
    policy: InferencePolicy,
) -> tuple[_RosterLayoutRoute, ...]:
    """Resolve an immutable dense-private roster from one serving model."""
    model = getattr(policy, "_model", None)
    model_config = getattr(model, "config", None)
    conditioning = getattr(model_config, "deck_conditioning", None)
    if conditioning is None or not bool(getattr(conditioning, "enabled", False)):
        return ()
    active_routes = getattr(conditioning, "active_routes", ())
    routes = tuple(
        _RosterLayoutRoute(
            signature=str(route.signature),
            module_key=str(route.module_key),
            canonical_card_ids=tuple(
                int(card_id) for card_id in route.canonical_card_ids
            ),
        )
        for route in active_routes
    )
    if not routes:
        return ()
    if len({route.signature for route in routes}) != len(routes):
        raise RuntimeError("serving roster contains duplicate exact deck signatures")
    if len({route.module_key for route in routes}) != len(routes):
        raise RuntimeError("serving roster contains shared private route keys")
    return tuple(sorted(routes, key=lambda route: route.module_key))


def _predict_value_groups(
    *,
    requests: Sequence[InferenceRequest],
    policies: Mapping[str, InferencePolicy],
) -> tuple[_ServedValueGroup, ...]:
    """Batch value requests by policy without entering any decode path."""
    grouped: dict[tuple[str, str, int | None, str, bool], list[InferenceRequest]] = {}
    for request in requests:
        if request.request_type != "value":
            raise ValueError("value policy groups cannot contain decode requests")
        _policy_for_id(policies, request.policy_id)
        grouped.setdefault(_inference_batch_identity(request), []).append(request)

    served: list[_ServedValueGroup] = []
    for identity, policy_requests in grouped.items():
        policy_id = identity[0]
        policy = _leased_policy_for_group(
            _policy_for_id(policies, policy_id),
            policy_requests,
        )
        predictor = _value_predictor(policy)
        if predictor is None:
            raise RuntimeError(
                f"inference policy does not expose value-only prediction: {policy_id}"
            )
        rows = tuple(
            _InferenceRequestRow(request=request, row_index=row_index)
            for request in policy_requests
            for row_index in range(request.batch_size)
        )
        states = _concat_state_batches(
            tuple(request.states for request in policy_requests)
        )
        decks = concatenate_deck_batches(
            tuple(request.decks for request in policy_requests)
        )
        device = _policy_device(policy, fallback=states.card_ids.device)
        states = _move_state_batch(states, device=device)
        decks = decks.to(device)
        sample_started_at = time.perf_counter()
        with torch.inference_mode():
            values = predictor.predict_values(states, decks)
        sample_finished_at = time.perf_counter()
        if values.ndim != 1 or int(values.shape[0]) != len(rows):
            raise RuntimeError("policy returned the wrong value-only batch shape")
        served.append(
            _ServedValueGroup(
                policy_id=policy_id,
                rows=rows,
                policy_version=int(getattr(policy, "policy_version", 0)),
                values=values,
                state_token_width=int(states.card_ids.shape[1]),
                sample_seconds=sample_finished_at - sample_started_at,
                sample_started_at=sample_started_at,
                sample_finished_at=sample_finished_at,
            )
        )
    return tuple(served)


def _predict_root_information_value_groups(
    *,
    requests: Sequence[InferenceRequest],
    policies: Mapping[str, InferencePolicy],
) -> tuple[_ServedValueGroup, ...]:
    """Batch semantic leaves only across identical lease/schema identities."""
    grouped: dict[tuple[str, str, int | None, str, bool], list[InferenceRequest]] = {}
    for request in requests:
        if request.request_type != "root_information_value":
            raise ValueError(
                "root-information groups cannot contain another request type"
            )
        _policy_for_id(policies, request.policy_id)
        grouped.setdefault(_inference_batch_identity(request), []).append(request)

    served: list[_ServedValueGroup] = []
    for identity, policy_requests in grouped.items():
        policy_id = identity[0]
        policy = _leased_policy_for_group(
            _policy_for_id(policies, policy_id),
            policy_requests,
        )
        predictor = _root_information_value_predictor(policy)
        if predictor is None:
            raise RuntimeError(
                f"inference policy does not expose root-information values: {policy_id}"
            )
        rows = tuple(
            _InferenceRequestRow(request=request, row_index=row_index)
            for request in policy_requests
            for row_index in range(request.batch_size)
        )
        states = _concat_state_batches(
            tuple(request.states for request in policy_requests)
        )
        decks = concatenate_deck_batches(
            tuple(request.decks for request in policy_requests)
        )
        actor_relations = torch.cat(
            [cast(Tensor, request.actor_relations) for request in policy_requests]
        )
        endpoints = torch.cat(
            [cast(Tensor, request.endpoints) for request in policy_requests]
        )
        belief_summaries = torch.cat(
            [cast(Tensor, request.belief_summaries) for request in policy_requests]
        )
        device = _policy_device(policy, fallback=states.card_ids.device)
        states = _move_state_batch(states, device=device)
        decks = decks.to(device)
        actor_relations = actor_relations.to(device=device)
        endpoints = endpoints.to(device=device)
        belief_summaries = belief_summaries.to(device=device)
        sample_started_at = time.perf_counter()
        with torch.inference_mode():
            values = predictor.predict_root_information_values(
                states,
                decks,
                actor_relations=actor_relations,
                endpoints=endpoints,
                belief_summaries=belief_summaries,
            )
        sample_finished_at = time.perf_counter()
        if values.ndim != 1 or int(values.shape[0]) != len(rows):
            raise RuntimeError(
                "policy returned the wrong root-information value batch shape"
            )
        served.append(
            _ServedValueGroup(
                policy_id=policy_id,
                rows=rows,
                policy_version=cast(
                    int,
                    policy_requests[0].model_version_lease,
                ),
                values=values,
                state_token_width=int(states.card_ids.shape[1]),
                sample_seconds=sample_finished_at - sample_started_at,
                sample_started_at=sample_started_at,
                sample_finished_at=sample_finished_at,
            )
        )
    return tuple(served)


def _release_planner_context_groups(
    *,
    requests: Sequence[InferenceRequest],
    policies: Mapping[str, InferencePolicy],
) -> tuple[_ServedValueGroup, ...]:
    """Release bounded root contexts without entering model inference."""
    served: list[_ServedValueGroup] = []
    for request in requests:
        if request.request_type != "planner_context_release":
            raise ValueError("context release group contains another request type")
        policy = _policy_for_id(policies, request.policy_id)
        _require_group_model_lease(policy, (request,))
        releaser = getattr(policy, "release_planner_context_handles", None)
        if not callable(releaser):
            raise RuntimeError("planner policy cannot release root contexts")
        sample_started_at = time.perf_counter()
        released = int(cast(Any, releaser)(request.planner_context_handles))
        sample_finished_at = time.perf_counter()
        if released < 0 or released > request.batch_size:
            raise RuntimeError("planner context release count is invalid")
        values = torch.zeros(request.batch_size, dtype=torch.float32)
        values[:released] = 1.0
        rows = tuple(
            _InferenceRequestRow(request=request, row_index=row_index)
            for row_index in range(request.batch_size)
        )
        served.append(
            _ServedValueGroup(
                policy_id=request.policy_id,
                rows=rows,
                policy_version=cast(int, request.model_version_lease),
                values=values,
                state_token_width=int(request.states.card_ids.shape[1]),
                sample_seconds=sample_finished_at - sample_started_at,
                sample_started_at=sample_started_at,
                sample_finished_at=sample_finished_at,
            )
        )
    return tuple(served)


def _predict_planner_candidate_groups(
    *,
    requests: Sequence[InferenceRequest],
    policies: Mapping[str, InferencePolicy],
    config: InferenceServerConfig,
) -> tuple[_ServedPlannerCandidateGroup, ...]:
    """Batch ragged schema-9 supports across actors under one model lease."""
    grouped: dict[
        tuple[str, str, int | None, str, bool],
        list[_InferenceRequestRow],
    ] = {}
    for request in requests:
        if request.request_type != "planner_candidates":
            raise ValueError(
                "planner candidate groups cannot contain another request type"
            )
        _policy_for_id(policies, request.policy_id)
        grouped.setdefault(_inference_batch_identity(request), []).extend(
            _InferenceRequestRow(request=request, row_index=row_index)
            for row_index in range(request.batch_size)
        )

    served: list[_ServedPlannerCandidateGroup] = []
    for identity, identity_rows in grouped.items():
        policy_id = identity[0]
        policy = _policy_for_id(policies, policy_id)
        predictor = _planner_candidate_predictor(policy)
        if predictor is None:
            raise RuntimeError(
                "inference policy does not expose planner candidate evaluation: "
                f"{policy_id}"
            )
        for rows in _planner_candidate_microbatches(
            identity_rows,
            max_candidates=config.max_planner_candidate_rows,
        ):
            policy_requests = tuple(row.request for row in rows)
            _require_group_model_lease(policy, policy_requests)
            states = _concat_state_batches(
                tuple(
                    _select_state_row(row.request.states, row.row_index) for row in rows
                )
            )
            options = _concat_option_batches(
                tuple(_planner_candidate_option_row(row) for row in rows)
            )
            decks = concatenate_deck_batches(
                tuple(row.request.decks.select((row.row_index,)) for row in rows)
            )
            candidate_actions = tuple(
                _planner_candidate_action_group(row) for row in rows
            )
            candidate_features = tuple(
                _planner_candidate_feature_group(row) for row in rows
            )
            ordered_rows = torch.stack(
                tuple(_planner_candidate_ordered_row(row) for row in rows)
            )
            expected_counts = tuple(len(group) for group in candidate_actions)
            device = _policy_device(policy, fallback=states.card_ids.device)
            states = _move_state_batch(states, device=device)
            options = _move_option_batch(options, device=device)
            decks = decks.to(device)
            candidate_features = tuple(
                features.to(device=device) for features in candidate_features
            )
            ordered_rows = ordered_rows.to(device=device)
            sample_started_at = time.perf_counter()
            with torch.inference_mode():
                evaluation = predictor.evaluate_planner_candidates(
                    states,
                    options,
                    candidate_actions,
                    candidate_features,
                    ordered_rows=ordered_rows,
                    decks=decks,
                    planner_context_handles=tuple(
                        handle
                        for row in rows
                        for handle in _planner_candidate_context_handle(row)
                    ),
                    model_version_lease=cast(
                        int,
                        rows[0].request.model_version_lease,
                    ),
                    tensor_schema_fingerprint=(
                        rows[0].request.tensor_schema_fingerprint
                    ),
                )
            sample_finished_at = time.perf_counter()
            if evaluation.candidate_counts != expected_counts:
                raise RuntimeError("policy returned another planner candidate grouping")
            flattened = sum(expected_counts)
            tensors = (
                evaluation.base_action_logprobs,
                evaluation.proposal_action_logprobs,
                evaluation.reranker_residuals,
            )
            if any(
                tensor.ndim != 1 or int(tensor.shape[0]) != flattened
                for tensor in tensors
            ):
                raise RuntimeError(
                    "policy returned misaligned planner candidate tensors"
                )
            served.append(
                _ServedPlannerCandidateGroup(
                    policy_id=policy_id,
                    rows=rows,
                    policy_version=cast(
                        int,
                        rows[0].request.model_version_lease,
                    ),
                    evaluation=evaluation,
                    state_token_width=int(states.card_ids.shape[1]),
                    sample_seconds=sample_finished_at - sample_started_at,
                    sample_started_at=sample_started_at,
                    sample_finished_at=sample_finished_at,
                )
            )
    return tuple(served)


def _predict_planner_proposal_groups(
    *,
    requests: Sequence[InferenceRequest],
    policies: Mapping[str, InferencePolicy],
) -> tuple[_ServedPlannerProposalGroup, ...]:
    """Group learned proposal searches by model lease and search identity."""
    grouped: dict[
        tuple[str, str, int | None, str, bool, str],
        list[InferenceRequest],
    ] = {}
    for request in requests:
        if request.request_type != "planner_proposals":
            raise ValueError(
                "planner proposal groups cannot contain another request type"
            )
        payload = request.planner_proposal_request
        if payload is None:
            raise ValueError("planner proposal request payload is missing")
        identity = (*_inference_batch_identity(request), payload.batch_identity)
        grouped.setdefault(identity, []).append(request)

    served: list[_ServedPlannerProposalGroup] = []
    for identity, grouped_requests in grouped.items():
        policy_id = identity[0]
        policy = _policy_for_id(policies, policy_id)
        predictor = planner_proposal_predictor(policy)
        if predictor is None:
            raise RuntimeError(
                "inference policy does not expose planner proposal generation: "
                f"{policy_id}"
            )
        policy_requests = tuple(grouped_requests)
        _require_group_model_lease(policy, policy_requests)
        states = _concat_state_batches(
            tuple(request.states for request in policy_requests)
        )
        options = _concat_option_batches(
            tuple(_planner_request_options(request) for request in policy_requests)
        )
        decks = concatenate_deck_batches(
            tuple(request.decks for request in policy_requests)
        )
        payloads = tuple(
            cast(PlannerProposalRequestPayload, request.planner_proposal_request)
            for request in policy_requests
        )
        device = _policy_device(policy, fallback=states.card_ids.device)
        states = _move_state_batch(states, device=device)
        options = _move_option_batch(options, device=device)
        decks = decks.to(device)
        sample_started_at = time.perf_counter()
        with torch.inference_mode():
            _result, response_payloads = evaluate_planner_proposal_group(
                predictor,
                states=states,
                options=options,
                decks=decks,
                payloads=payloads,
                request_batch_sizes=tuple(
                    request.batch_size for request in policy_requests
                ),
                model_version_lease=cast(
                    int,
                    policy_requests[0].model_version_lease,
                ),
                tensor_schema_fingerprint=(
                    policy_requests[0].tensor_schema_fingerprint
                ),
                deadline_monotonic=min(
                    request.deadline_monotonic for request in policy_requests
                ),
            )
        sample_finished_at = time.perf_counter()
        served.append(
            _ServedPlannerProposalGroup(
                policy_id=policy_id,
                requests=policy_requests,
                policy_version=cast(
                    int,
                    policy_requests[0].model_version_lease,
                ),
                payloads=response_payloads,
                state_token_width=int(states.card_ids.shape[1]),
                sample_seconds=sample_finished_at - sample_started_at,
                sample_started_at=sample_started_at,
                sample_finished_at=sample_finished_at,
            )
        )
    return tuple(served)


def _planner_request_options(request: InferenceRequest) -> OptionBatch:
    options = request.options
    if options is None:
        raise ValueError("planner inference request has no options")
    return options


def _planner_candidate_microbatches(
    rows: Sequence[_InferenceRequestRow],
    *,
    max_candidates: int,
) -> tuple[tuple[_InferenceRequestRow, ...], ...]:
    """Pack whole decision supports without splitting categorical groups."""
    batches: list[tuple[_InferenceRequestRow, ...]] = []
    current: list[_InferenceRequestRow] = []
    current_candidates = 0
    for row in rows:
        count = len(_planner_candidate_action_group(row))
        if count > max_candidates:
            raise RuntimeError("one planner support exceeds max_planner_candidate_rows")
        if current and current_candidates + count > max_candidates:
            batches.append(tuple(current))
            current = []
            current_candidates = 0
        current.append(row)
        current_candidates += count
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _planner_candidate_action_group(
    row: _InferenceRequestRow,
) -> tuple[tuple[int, ...], ...]:
    groups = row.request.candidate_actions
    if groups is None:
        raise ValueError("planner candidate request has no actions")
    return groups[row.row_index]


def _planner_candidate_feature_group(row: _InferenceRequestRow) -> Tensor:
    groups = row.request.candidate_features
    if groups is None:
        raise ValueError("planner candidate request has no features")
    return groups[row.row_index]


def _planner_candidate_ordered_row(row: _InferenceRequestRow) -> Tensor:
    ordered_rows = row.request.ordered_rows
    if ordered_rows is None:
        raise ValueError("planner candidate request has no ordering semantics")
    return ordered_rows[row.row_index]


def _planner_candidate_context_handle(
    row: _InferenceRequestRow,
) -> tuple[str]:
    return (row.request.planner_context_handles[row.row_index],)


def _planner_candidate_option_row(row: _InferenceRequestRow) -> OptionBatch:
    options = row.request.options
    if options is None:
        raise ValueError("planner candidate request has no options")
    return _select_option_row(options, row.row_index)


def _select_state_row(states: StateBatch, row_index: int) -> StateBatch:
    """Select one state row without losing optional feature tensors."""
    row = slice(row_index, row_index + 1)
    return StateBatch(
        card_ids=states.card_ids[row],
        areas=states.areas[row],
        owner_roles=states.owner_roles[row],
        token_kinds=states.token_kinds[row],
        scalars=states.scalars[row],
        last_attack_ids=states.last_attack_ids[row],
        padding_mask=states.padding_mask[row],
        attachment_card_ids=_optional_row(states.attachment_card_ids, row),
        attachment_parent_indices=_optional_row(
            states.attachment_parent_indices,
            row,
        ),
        attachment_kinds=_optional_row(states.attachment_kinds, row),
        entity_slots=_optional_row(states.entity_slots, row),
        root_input_fingerprints=(
            ()
            if not states.root_input_fingerprints
            else (states.root_input_fingerprints[row_index],)
        ),
    )


def _select_option_row(options: OptionBatch, row_index: int) -> OptionBatch:
    """Select one option row for exact-route serving collation."""
    row = slice(row_index, row_index + 1)
    return OptionBatch(
        option_types=options.option_types[row],
        contexts=options.contexts[row],
        entity_slots=options.entity_slots[row],
        entity_slot_mask=options.entity_slot_mask[row],
        attack_ids=options.attack_ids[row],
        card_ids=options.card_ids[row],
        scalars=options.scalars[row],
        dynamic_effect_features=options.dynamic_effect_features[row],
        dynamic_effect_masks=options.dynamic_effect_masks[row],
        valid_options=options.valid_options[row],
        min_counts=options.min_counts[row],
        max_counts=options.max_counts[row],
    )


def _optional_row(tensor: Tensor | None, row: slice) -> Tensor | None:
    return None if tensor is None else tensor[row]


def _prepare_policy_group(
    *,
    policy_id: str,
    policy: InferencePolicy,
    rows: tuple[_InferenceRequestRow, ...],
    config: InferenceServerConfig,
) -> _PreparedPolicyGroup:
    if not rows:
        raise ValueError("inference policy groups must contain at least one row")
    _require_group_model_lease(policy, tuple(row.request for row in rows))
    states = _concat_state_batches(
        tuple(_select_state_row(row.request.states, row.row_index) for row in rows)
    )
    options = _concat_option_batches(
        tuple(
            _select_option_row(_decode_options(row.request), row.row_index)
            for row in rows
        )
    )
    decks = concatenate_deck_batches(
        tuple(row.request.decks.select((row.row_index,)) for row in rows)
    )
    return _prepare_policy_group_batches(
        policy_id=policy_id,
        policy=policy,
        rows=rows,
        states=states,
        options=options,
        decks=decks,
        config=config,
    )


def _prepare_roster_layout_policy_group(
    *,
    policy_id: str,
    policy: InferencePolicy,
    rows: tuple[_InferenceRequestRow, ...],
    routes: tuple[_RosterLayoutRoute, ...],
    config: InferenceServerConfig,
) -> _PreparedPolicyGroup:
    """Pack real rows into fixed contiguous slots for every private route."""
    if not rows:
        raise ValueError("roster layout requires at least one real row")
    _require_group_model_lease(policy, tuple(row.request for row in rows))
    slots_per_route = config.graph_roster_slots_per_route
    layout_batch_size = len(routes) * slots_per_route
    if layout_batch_size > config.max_batch:
        raise ValueError(
            "roster layout batch exceeds inference max_batch: "
            f"{layout_batch_size} > {config.max_batch}"
        )
    rows_by_signature: dict[str, list[_InferenceRequestRow]] = {
        route.signature: [] for route in routes
    }
    for row in rows:
        signature = row.request.decks.signatures[row.row_index]
        if signature not in rows_by_signature:
            raise ValueError("roster layout received an unregistered deck")
        rows_by_signature[signature].append(row)
    if any(
        len(route_rows) > slots_per_route for route_rows in rows_by_signature.values()
    ):
        raise ValueError("roster layout route exceeds its fixed slot capacity")

    filler = rows[0]
    route_state_batches: list[StateBatch] = []
    route_option_batches: list[OptionBatch] = []
    route_deck_batches: list[DeckBatch] = []
    ordered_rows: list[_InferenceRequestRow] = []
    sample_indices: list[int] = []
    for route_index, route in enumerate(routes):
        route_rows = tuple(rows_by_signature[route.signature])
        source_rows = route_rows or (filler,)
        states = _concat_state_batches(
            tuple(
                _select_state_row(row.request.states, row.row_index)
                for row in source_rows
            )
        )
        options = _concat_option_batches(
            tuple(
                _select_option_row(_decode_options(row.request), row.row_index)
                for row in source_rows
            )
        )
        states = _pad_state_batch_to_bucket(
            states,
            batch_size=slots_per_route,
            tokens=int(states.card_ids.shape[1]),
            attachments=_state_attachment_width(states),
        )
        options = _pad_option_batch_to_bucket(
            options,
            batch_size=slots_per_route,
            option_width=int(options.valid_options.shape[1]),
        )
        if route_rows:
            decks = concatenate_deck_batches(
                tuple(row.request.decks.select((row.row_index,)) for row in route_rows)
            )
        else:
            decks = DeckBatch.from_card_ids(
                (route.canonical_card_ids,),
                device=filler.request.decks.card_ids.device,
            )
        route_state_batches.append(states)
        route_option_batches.append(options)
        route_deck_batches.append(pad_deck_batch(decks, slots_per_route))
        ordered_rows.extend(route_rows)
        route_offset = route_index * slots_per_route
        sample_indices.extend(route_offset + index for index in range(len(route_rows)))

    layout_states = _concat_state_batches(tuple(route_state_batches))
    layout_options = _concat_option_batches(tuple(route_option_batches))
    layout_decks = concatenate_deck_batches(tuple(route_deck_batches))
    actual_shape = _policy_group_shape_for_rows(tuple(ordered_rows))
    layout_shape = (
        layout_batch_size,
        actual_shape[1],
        actual_shape[2],
        actual_shape[3],
    )
    actual_attachment_width = _state_attachment_width(layout_states)
    bucket_plan, fallback_reason = _bucket_plan_for_shape(
        layout_shape,
        config,
        attachment_width=actual_attachment_width,
    )
    if bucket_plan is None:
        eager = _prepare_policy_group(
            policy_id=policy_id,
            policy=policy,
            rows=rows,
            config=config.model_copy(update={"bucketize": False}),
        )
        return replace(
            eager,
            bucket_fallback_reason=f"roster_layout:{fallback_reason}",
            serving_path="roster_eager_fallback",
        )
    layout_states = _pad_state_batch_to_bucket(
        layout_states,
        batch_size=bucket_plan.batch_size,
        tokens=bucket_plan.tokens,
        attachments=bucket_plan.attachments,
    )
    layout_options = _pad_option_batch_to_bucket(
        layout_options,
        batch_size=bucket_plan.batch_size,
        option_width=bucket_plan.options,
    )
    layout_decks = pad_deck_batch(layout_decks, bucket_plan.batch_size)
    device = _policy_device(policy, fallback=layout_states.card_ids.device)
    return _PreparedPolicyGroup(
        policy_id=policy_id,
        rows=tuple(ordered_rows),
        policy=policy,
        states=_move_state_batch(layout_states, device=device),
        options=_move_option_batch(layout_options, device=device),
        decks=layout_decks.to(device),
        shape=actual_shape,
        max_select_steps=bucket_plan.max_select_steps,
        temperature=_group_row_temperature(ordered_rows),
        bucket_key=(f"R{len(routes)}x{slots_per_route}/{bucket_plan.key}"),
        bucket_fallback_reason=None,
        bucket_slot_totals=_bucket_slot_totals(
            actual_shape,
            bucket_plan,
            attachment_width=actual_attachment_width,
        ),
        sample_indices=tuple(sample_indices),
        serving_path="roster_graph",
    )


def _policy_group_shape_for_rows(
    rows: tuple[_InferenceRequestRow, ...],
) -> tuple[int, int, int, int]:
    """Return the unpadded decode shape of an arbitrary selected row set."""
    if not rows:
        raise ValueError("policy group shape requires at least one row")
    tokens = max(int(row.request.states.card_ids.shape[1]) for row in rows)
    options = max(
        int(_decode_options(row.request).valid_options.shape[1]) for row in rows
    )
    max_select_steps = max(
        int(_decode_options(row.request).max_counts[row.row_index].item())
        for row in rows
    )
    decode_steps = min(options, max_select_steps) + 1
    return (len(rows), tokens, options, decode_steps)


def _prepare_whole_request_policy_group(
    *,
    policy_id: str,
    policy: InferencePolicy,
    requests: tuple[InferenceRequest, ...],
    config: InferenceServerConfig,
) -> _PreparedPolicyGroup:
    """Prepare an eager policy group from complete request batches."""
    if not requests:
        raise ValueError("inference policy groups must contain at least one request")
    _require_group_model_lease(policy, requests)
    rows = tuple(
        _InferenceRequestRow(request=request, row_index=row_index)
        for request in requests
        for row_index in range(request.batch_size)
    )
    return _prepare_policy_group_batches(
        policy_id=policy_id,
        policy=policy,
        rows=rows,
        states=_concat_state_batches(tuple(request.states for request in requests)),
        options=_concat_option_batches(
            tuple(_decode_options(request) for request in requests)
        ),
        decks=concatenate_deck_batches(tuple(request.decks for request in requests)),
        config=config,
    )


def _prepare_policy_group_batches(
    *,
    policy_id: str,
    policy: InferencePolicy,
    rows: tuple[_InferenceRequestRow, ...],
    states: StateBatch,
    options: OptionBatch,
    decks: DeckBatch,
    config: InferenceServerConfig,
) -> _PreparedPolicyGroup:
    """Finish policy-group preparation after input collation."""
    actual_shape = _policy_group_shape_from_batches(states, options)
    actual_attachment_width = _state_attachment_width(states)
    bucket_plan, fallback_reason = _bucket_plan_for_shape(
        actual_shape,
        config,
        attachment_width=actual_attachment_width,
    )
    if any(row.request.purpose == "planner_behavior" for row in rows):
        # Planner decode retains one immutable root context per real request
        # row. Bucket padding would create unowned synthetic-row handles.
        bucket_plan = None
        fallback_reason = "planner_context_exact_batch"
    bucket_slot_totals: Mapping[str, int] = {}
    max_select_steps = max(0, actual_shape[3] - 1)
    if bucket_plan is not None:
        states = _pad_state_batch_to_bucket(
            states,
            batch_size=bucket_plan.batch_size,
            tokens=bucket_plan.tokens,
            attachments=bucket_plan.attachments,
        )
        options = _pad_option_batch_to_bucket(
            options,
            batch_size=bucket_plan.batch_size,
            option_width=bucket_plan.options,
        )
        decks = pad_deck_batch(decks, bucket_plan.batch_size)
        bucket_slot_totals = _bucket_slot_totals(
            actual_shape,
            bucket_plan,
            attachment_width=actual_attachment_width,
        )
        max_select_steps = bucket_plan.max_select_steps
    device = _policy_device(policy, fallback=states.card_ids.device)
    return _PreparedPolicyGroup(
        policy_id=policy_id,
        rows=rows,
        policy=policy,
        states=_move_state_batch(states, device=device),
        options=_move_option_batch(options, device=device),
        decks=decks.to(device),
        shape=actual_shape,
        max_select_steps=max_select_steps,
        temperature=_group_row_temperature(rows),
        bucket_key=None if bucket_plan is None else bucket_plan.key,
        bucket_fallback_reason=fallback_reason,
        bucket_slot_totals=bucket_slot_totals,
        sample_indices=tuple(range(len(rows))),
    )


def _sample_prepared_policy_group(
    prepared: _PreparedPolicyGroup,
) -> _ServedPolicyGroup:
    sample_start = time.perf_counter()
    sample = _sample_prepared_policy_group_eager(prepared)
    sample_finished = time.perf_counter()
    sample_seconds = sample_finished - sample_start
    policy_version = sample.served_policy_version
    if policy_version is None:
        policy_version = int(getattr(prepared.policy, "policy_version", 0))
    return _ServedPolicyGroup(
        policy_id=prepared.policy_id,
        rows=prepared.rows,
        policy_version=policy_version,
        shape=prepared.shape,
        sample=sample,
        sample_seconds=sample_seconds,
        sample_started_at=sample_start,
        sample_finished_at=sample_finished,
        serving_path=prepared.serving_path,
    )


def _sample_prepared_policy_group_eager(
    prepared: _PreparedPolicyGroup,
) -> _PolicyGroupSample:
    planner_rows = tuple(
        row for row in prepared.rows if row.request.purpose == "planner_behavior"
    )
    if planner_rows:
        if len(planner_rows) != len(prepared.rows):
            raise RuntimeError("decode group mixed planner and base purposes")
        leases = {row.request.model_version_lease for row in planner_rows}
        if len(leases) != 1:
            raise RuntimeError("planner decode group mixed model leases")
        (model_version_lease,) = leases
        retention = {
            _request_retains_planner_context(row.request) for row in planner_rows
        }
        if len(retention) != 1:
            raise RuntimeError("planner decode group mixed context ownership")
        (retain_planner_context,) = retention
        request_sampler = getattr(
            prepared.policy,
            "sample_decode_with_trace_for_request",
            None,
        )
        if not callable(request_sampler):
            raise RuntimeError("planner policy has no lease-aware decode surface")
        decoded = cast(Any, request_sampler)(
            prepared.states,
            prepared.options,
            prepared.decks,
            temperature=prepared.temperature,
            model_version_lease=model_version_lease,
            retain_planner_context=retain_planner_context,
        )
    elif prepared.bucket_key is None:
        decoded = _sample_policy_materialized(
            prepared.policy,
            prepared.states,
            prepared.options,
            prepared.decks,
            temperature=prepared.temperature,
        )
    else:
        decoded = _sample_policy_materialized(
            prepared.policy,
            prepared.states,
            prepared.options,
            prepared.decks,
            temperature=prepared.temperature,
            max_select_steps=prepared.max_select_steps,
        )
    if isinstance(decoded, SampleDecodeTrace):
        return _policy_group_sample(
            actions=decoded.actions,
            logprobs=decoded.action_logprobs,
            values=decoded.values,
            token_logprobs=decoded.token_logprobs,
            prefix_values=decoded.prefix_values,
            token_mask=decoded.token_mask,
            stop_sampled=decoded.stop_sampled,
            planner_context_handles=decoded.planner_context_handles,
            served_policy_version=decoded.served_policy_version,
            served_model_fingerprint=decoded.served_model_fingerprint,
            served_proposal_version=decoded.served_proposal_version,
            planner_fallback_reason=decoded.planner_fallback_reason,
            prepared=prepared,
        )
    actions, logprobs, values = decoded
    return _policy_group_sample(
        actions=actions,
        logprobs=logprobs,
        values=values,
        prepared=prepared,
    )


def _can_sample_policy_groups_on_cuda_streams(
    prepared: Sequence[_PreparedPolicyGroup],
    *,
    config: InferenceServerConfig,
) -> bool:
    if (
        not config.concurrent_policy_streams
        or config.graph_decode
        or len(prepared) < 2
        or not torch.cuda.is_available()
    ):
        return False
    if any(
        row.request.purpose == "planner_behavior"
        for group in prepared
        for row in group.rows
    ):
        return False
    return all(
        (
            _tensor_trace_decode_sampler(group.policy) is not None
            or _tensor_decode_sampler(group.policy) is not None
        )
        and group.states.card_ids.device.type == "cuda"
        and group.options.valid_options.device == group.states.card_ids.device
        and group.decks.card_ids.device == group.states.card_ids.device
        for group in prepared
    )


def _sample_policy_groups_on_cuda_streams(
    prepared_groups: Sequence[_PreparedPolicyGroup],
) -> tuple[_ServedPolicyGroup, ...]:
    sample_start = time.perf_counter()
    pending: list[
        tuple[
            _PreparedPolicyGroup,
            _CudaStreamSlot,
            _TensorPolicyGroupSample,
        ]
    ] = []
    for slot_index, prepared in enumerate(prepared_groups):
        device = prepared.states.card_ids.device
        slot = _cuda_stream_slot(device, slot_index)
        with torch.cuda.stream(slot.stream):
            cast(Any, slot.started).record()
            tensor_sample = _sample_prepared_policy_group_tensors(prepared)
            cast(Any, slot.finished).record()
        pending.append((prepared, slot, tensor_sample))

    served_groups: list[_ServedPolicyGroup] = []
    for prepared, slot, tensor_sample in pending:
        cast(Any, slot.stream).synchronize()
        sample_seconds = (
            float(cast(Any, slot.started).elapsed_time(slot.finished)) / 1000.0
        )
        sample = _policy_group_sample_from_tensors(
            tensor_sample,
            prepared=prepared,
        )
        served_groups.append(
            _ServedPolicyGroup(
                policy_id=prepared.policy_id,
                rows=prepared.rows,
                policy_version=int(getattr(prepared.policy, "policy_version", 0)),
                shape=prepared.shape,
                sample=sample,
                sample_seconds=sample_seconds,
                sample_started_at=sample_start,
                sample_finished_at=sample_start + sample_seconds,
                serving_path=prepared.serving_path,
            )
        )
    return tuple(served_groups)


_CUDA_STREAM_SLOTS: dict[tuple[int, int], _CudaStreamSlot] = {}


def _cuda_stream_slot(device: torch.device, slot_index: int) -> _CudaStreamSlot:
    """Return reusable CUDA stream/event resources for one group slot."""
    device_index = _cuda_device_index(device)
    key = (device_index, int(slot_index))
    slot = _CUDA_STREAM_SLOTS.get(key)
    if slot is not None:
        return slot
    with torch.cuda.device(device_index):
        slot = _CudaStreamSlot(
            stream=cast(Any, torch.cuda.Stream)(device=device),
            started=cast(Any, torch.cuda.Event)(enable_timing=True),
            finished=cast(Any, torch.cuda.Event)(enable_timing=True),
        )
    _CUDA_STREAM_SLOTS[key] = slot
    return slot


def _cuda_device_index(device: torch.device) -> int:
    """Return a concrete CUDA device index."""
    if device.type != "cuda":
        raise ValueError(f"expected CUDA device, got {device}")
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _sample_prepared_policy_group_tensors(
    prepared: _PreparedPolicyGroup,
) -> _TensorPolicyGroupSample:
    trace_sampler = _tensor_trace_decode_sampler(prepared.policy)
    if trace_sampler is not None:
        trace = trace_sampler.sample_decode_tensors_with_trace(
            prepared.states,
            prepared.options,
            prepared.decks,
            temperature=prepared.temperature,
            max_select_steps=prepared.max_select_steps,
        )
        return _TensorPolicyGroupSample(
            choice_indices=trace.choice_indices,
            append_masks=trace.append_masks,
            logprobs=trace.action_logprobs,
            values=trace.values,
            token_logprobs=trace.token_logprobs,
            prefix_values=trace.prefix_values,
            token_mask=trace.token_mask,
            stop_sampled=trace.stop_sampled,
        )
    sampler = _tensor_decode_sampler(prepared.policy)
    if sampler is None:
        raise RuntimeError("policy does not expose tensor decode")
    choice_indices, append_masks, logprobs, values = sampler.sample_decode_tensors(
        prepared.states,
        prepared.options,
        prepared.decks,
        temperature=prepared.temperature,
        max_select_steps=prepared.max_select_steps,
    )
    return _TensorPolicyGroupSample(
        choice_indices=choice_indices,
        append_masks=append_masks,
        logprobs=logprobs,
        values=values,
    )


def _policy_group_sample_from_tensors(
    tensor_sample: _TensorPolicyGroupSample,
    *,
    prepared: _PreparedPolicyGroup,
) -> _PolicyGroupSample:
    return _policy_group_sample(
        actions=actions_from_decode_tensors(
            tensor_sample.choice_indices,
            tensor_sample.append_masks,
        ),
        logprobs=tensor_sample.logprobs,
        values=tensor_sample.values,
        token_logprobs=tensor_sample.token_logprobs,
        prefix_values=tensor_sample.prefix_values,
        token_mask=tensor_sample.token_mask,
        stop_sampled=tensor_sample.stop_sampled,
        prepared=prepared,
    )


def _policy_group_sample(
    *,
    actions: tuple[tuple[int, ...], ...],
    logprobs: Tensor,
    values: Tensor,
    prepared: _PreparedPolicyGroup,
    token_logprobs: Tensor | None = None,
    prefix_values: Tensor | None = None,
    token_mask: Tensor | None = None,
    stop_sampled: Tensor | None = None,
    planner_context_handles: tuple[str, ...] = (),
    served_policy_version: int | None = None,
    served_model_fingerprint: str = "",
    served_proposal_version: int | None = None,
    planner_fallback_reason: str = "",
) -> _PolicyGroupSample:
    expected_batch = int(prepared.options.valid_options.shape[0])
    if len(actions) != expected_batch:
        raise RuntimeError("policy returned the wrong action batch size")
    if planner_context_handles and len(planner_context_handles) != expected_batch:
        raise RuntimeError("policy returned misaligned planner context handles")
    _validate_policy_sample_tensors(
        logprobs=logprobs,
        values=values,
        token_logprobs=token_logprobs,
        prefix_values=prefix_values,
        token_mask=token_mask,
        stop_sampled=stop_sampled,
        batch_size=expected_batch,
    )
    sample_indices = prepared.sample_indices
    if len(sample_indices) != len(prepared.rows):
        raise RuntimeError("prepared sample indices must align with real rows")
    if any(index < 0 or index >= expected_batch for index in sample_indices):
        raise RuntimeError("prepared sample index is outside the static batch")
    actions = tuple(actions[index] for index in sample_indices)
    logprobs = _select_sample_tensor_rows(logprobs, sample_indices)
    values = _select_sample_tensor_rows(values, sample_indices)
    token_logprobs = _select_optional_sample_tensor_rows(
        token_logprobs,
        sample_indices,
    )
    prefix_values = _select_optional_sample_tensor_rows(
        prefix_values,
        sample_indices,
    )
    token_mask = _select_optional_sample_tensor_rows(token_mask, sample_indices)
    stop_sampled = _select_optional_sample_tensor_rows(stop_sampled, sample_indices)
    if planner_context_handles:
        planner_context_handles = tuple(
            planner_context_handles[index] for index in sample_indices
        )
    return _PolicyGroupSample(
        actions=actions,
        logprobs=logprobs,
        values=values,
        token_logprobs=token_logprobs,
        prefix_values=prefix_values,
        token_mask=token_mask,
        stop_sampled=stop_sampled,
        planner_context_handles=planner_context_handles,
        served_policy_version=served_policy_version,
        served_model_fingerprint=served_model_fingerprint,
        served_proposal_version=served_proposal_version,
        planner_fallback_reason=planner_fallback_reason,
        bucket_key=prepared.bucket_key,
        bucket_fallback_reason=prepared.bucket_fallback_reason,
        bucket_slot_totals=prepared.bucket_slot_totals,
    )


def _select_sample_tensor_rows(tensor: Tensor, indices: tuple[int, ...]) -> Tensor:
    """Select real rows from a static serving layout on the tensor's device."""
    if indices == tuple(range(int(tensor.shape[0]))):
        return tensor
    index_tensor = torch.tensor(indices, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, index_tensor)


def _select_optional_sample_tensor_rows(
    tensor: Tensor | None,
    indices: tuple[int, ...],
) -> Tensor | None:
    """Select static-layout rows from an optional decode trace tensor."""
    if tensor is None:
        return None
    return _select_sample_tensor_rows(tensor, indices)


def _validate_policy_sample_tensors(
    *,
    logprobs: Tensor,
    values: Tensor,
    token_logprobs: Tensor | None,
    prefix_values: Tensor | None,
    token_mask: Tensor | None,
    stop_sampled: Tensor | None,
    batch_size: int,
) -> None:
    if logprobs.ndim != 1 or int(logprobs.shape[0]) != batch_size:
        raise RuntimeError("policy returned the wrong logprob batch shape")
    if values.ndim != 1 or int(values.shape[0]) != batch_size:
        raise RuntimeError("policy returned the wrong value batch shape")
    fields = (token_logprobs, prefix_values, token_mask, stop_sampled)
    if all(field is None for field in fields):
        return
    if any(field is None for field in fields):
        raise RuntimeError("policy returned an incomplete token trace")
    active_logprobs = cast(Tensor, token_logprobs)
    active_prefix_values = cast(Tensor, prefix_values)
    active_mask = cast(Tensor, token_mask)
    active_stop = cast(Tensor, stop_sampled)
    if (
        active_logprobs.ndim != 2
        or active_prefix_values.shape != active_logprobs.shape
        or active_mask.shape != active_logprobs.shape
        or int(active_logprobs.shape[0]) != batch_size
    ):
        raise RuntimeError("policy returned misaligned token trace tensors")
    if active_stop.ndim != 1 or int(active_stop.shape[0]) != batch_size:
        raise RuntimeError("policy returned the wrong STOP trace batch shape")
    if active_mask.dtype != torch.bool or active_stop.dtype != torch.bool:
        raise RuntimeError("policy token masks must be boolean")


def _tensor_decode_sampler(policy: InferencePolicy) -> _TensorDecodePolicy | None:
    if bool(getattr(policy, "planner_context_enabled", False)):
        return None
    sampler = getattr(policy, "sample_decode_tensors", None)
    if callable(sampler):
        return cast(_TensorDecodePolicy, policy)
    return None


def _tensor_trace_decode_sampler(
    policy: InferencePolicy,
) -> _TensorTraceDecodePolicy | None:
    if bool(getattr(policy, "planner_context_enabled", False)):
        return None
    sampler = getattr(policy, "sample_decode_tensors_with_trace", None)
    if callable(sampler):
        return cast(_TensorTraceDecodePolicy, policy)
    return None


def _value_predictor(policy: InferencePolicy) -> _ValueInferencePolicy | None:
    predictor = getattr(policy, "predict_values", None)
    if callable(predictor):
        return cast(_ValueInferencePolicy, policy)
    return None


def _root_information_value_predictor(
    policy: InferencePolicy,
) -> _RootInformationValueInferencePolicy | None:
    predictor = getattr(policy, "predict_root_information_values", None)
    if callable(predictor):
        return cast(_RootInformationValueInferencePolicy, policy)
    return None


def _planner_candidate_predictor(
    policy: InferencePolicy,
) -> _PlannerCandidateInferencePolicy | None:
    predictor = getattr(policy, "evaluate_planner_candidates", None)
    if callable(predictor):
        return cast(_PlannerCandidateInferencePolicy, policy)
    return None


def _sample_policy_group(
    policy: InferencePolicy,
    requests: Sequence[InferenceRequest],
    *,
    config: InferenceServerConfig,
) -> _PolicyGroupSample:
    rows = tuple(
        _InferenceRequestRow(request=request, row_index=row_index)
        for request in requests
        for row_index in range(request.batch_size)
    )
    prepared = _prepare_policy_group(
        policy_id="",
        policy=policy,
        rows=rows,
        config=config,
    )
    return _sample_prepared_policy_group_eager(prepared)


def _policy_group_shape(
    requests: Sequence[InferenceRequest],
) -> tuple[int, int, int, int]:
    batch_size = sum(request.batch_size for request in requests)
    max_tokens = max(int(request.states.card_ids.shape[1]) for request in requests)
    max_options = max(
        int(_decode_options(request).valid_options.shape[1]) for request in requests
    )
    max_steps = max(
        int(_decode_options(request).max_counts.max().item()) for request in requests
    )
    max_steps = min(max_options, max_steps) + 1
    return (batch_size, max_tokens, max_options, max_steps)


def _policy_group_shape_from_batches(
    states: StateBatch,
    options: OptionBatch,
) -> tuple[int, int, int, int]:
    batch_size = int(options.valid_options.shape[0])
    max_tokens = int(states.card_ids.shape[1])
    max_options = int(options.valid_options.shape[1])
    max_steps = int(options.max_counts.max().item())
    max_steps = min(max_options, max_steps) + 1
    return (batch_size, max_tokens, max_options, max_steps)


def _bucket_plan_for_shape(
    shape: tuple[int, int, int, int],
    config: InferenceServerConfig,
    *,
    attachment_width: int,
) -> tuple[_BucketPlan | None, str | None]:
    if not config.bucketize:
        return (None, None)
    batch_size, tokens, options, decode_steps = shape
    if attachment_width < 0:
        raise ValueError("attachment width must be non-negative")
    bucket_attachments = (
        0
        if attachment_width == 0
        else _ceil_bucket(attachment_width, config.bucket_attachment_sizes)
    )
    if bucket_attachments is None:
        return (None, "attachments_oversize")
    max_select_steps = max(0, decode_steps - 1)
    if max_select_steps > config.bucket_max_select_steps:
        return (None, "decode_steps")
    bucket_batch = _ceil_bucket(batch_size, config.bucket_batch_sizes)
    bucket_tokens = _ceil_bucket(tokens, config.bucket_token_sizes)
    bucket_options = _ceil_bucket(options, config.bucket_option_sizes)
    if bucket_batch is None or bucket_tokens is None or bucket_options is None:
        return (None, "shape_oversize")
    return (
        _BucketPlan(
            batch_size=bucket_batch,
            tokens=bucket_tokens,
            options=bucket_options,
            attachments=bucket_attachments,
            max_select_steps=config.bucket_max_select_steps,
        ),
        None,
    )


def _ceil_bucket(value: int, buckets: Sequence[int]) -> int | None:
    for bucket in buckets:
        if value <= bucket:
            return int(bucket)
    return None


def _bucket_slot_totals(
    actual_shape: tuple[int, int, int, int],
    bucket: _BucketPlan,
    *,
    attachment_width: int,
) -> dict[str, int]:
    batch_size, tokens, options, _decode_steps = actual_shape
    return {
        "decision_actual": batch_size,
        "decision_padded": bucket.batch_size,
        "token_actual": batch_size * tokens,
        "token_padded": bucket.batch_size * bucket.tokens,
        "option_actual": batch_size * options,
        "option_padded": bucket.batch_size * bucket.options,
        "attachment_actual": batch_size * attachment_width,
        "attachment_padded": bucket.batch_size * bucket.attachments,
        "cross_actual": batch_size * tokens * options,
        "cross_padded": bucket.batch_size * bucket.tokens * bucket.options,
    }


def _shape_histogram_key(shape: tuple[int, int, int, int]) -> str:
    batch_size, tokens, options, steps = shape
    return f"B{batch_size}/T{tokens}/O{options}/S{steps}"


def _value_shape_histogram_key(batch_size: int, tokens: int) -> str:
    """Return a shape key that cannot be confused with decode batches."""
    return f"B{batch_size}/T{tokens}/VALUE"


def _service_time_histogram_bucket(milliseconds: float) -> str:
    if milliseconds < 1.0:
        return "<1"
    bucket = 1
    while bucket < 512 and milliseconds >= float(bucket * 2):
        bucket *= 2
    lower = bucket
    upper = bucket * 2
    return f"{lower}-{upper}"


def _group_row_temperature(rows: Sequence[_InferenceRequestRow]) -> float:
    temperatures = {float(row.request.temperature) for row in rows}
    if len(temperatures) != 1:
        raise ValueError("one inference route group must share temperature")
    return temperatures.pop()


def _sample_policy_materialized(
    policy: InferencePolicy,
    states: StateBatch,
    options: OptionBatch,
    decks: DeckBatch,
    *,
    temperature: float,
    max_select_steps: int | None = None,
) -> SampleDecodeTrace | tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
    if max_select_steps is not None:
        static_trace_sampler = getattr(
            policy,
            "sample_decode_static_with_trace",
            None,
        )
        if callable(static_trace_sampler):
            return cast(
                SampleDecodeTrace,
                static_trace_sampler(
                    states,
                    options,
                    decks,
                    temperature=temperature,
                    max_select_steps=max_select_steps,
                ),
            )
        static_sampler = getattr(policy, "sample_decode_static", None)
        if callable(static_sampler):
            return cast(
                tuple[tuple[tuple[int, ...], ...], Tensor, Tensor],
                cast(Any, static_sampler)(
                    states,
                    options,
                    decks,
                    temperature=temperature,
                    max_select_steps=max_select_steps,
                ),
            )
    trace_sampler = _trace_decode_sampler(policy)
    if trace_sampler is not None:
        return trace_sampler.sample_decode_with_trace(
            states,
            options,
            decks,
            temperature=temperature,
        )
    return policy.sample_decode(states, options, decks, temperature=temperature)


def _trace_decode_sampler(policy: InferencePolicy) -> _TraceDecodePolicy | None:
    sampler = getattr(policy, "sample_decode_with_trace", None)
    if callable(sampler):
        return cast(_TraceDecodePolicy, policy)
    return None


def _policy_device(policy: InferencePolicy, *, fallback: torch.device) -> torch.device:
    model = getattr(policy, "_model", None)
    if isinstance(model, torch.nn.Module):
        for parameter in model.parameters():
            return parameter.device
        for buffer in model.buffers():
            return buffer.device
    return fallback


def _pad_state_batch_to_bucket(
    states: StateBatch,
    *,
    batch_size: int,
    tokens: int,
    attachments: int,
) -> StateBatch:
    padding_mask = _pad_tensor(
        states.padding_mask,
        shape=(batch_size, tokens),
        fill_value=True,
    )
    actual_batch_size = int(states.card_ids.shape[0])
    if batch_size > actual_batch_size and tokens > 0:
        padding_mask[actual_batch_size:, 0] = False
    return StateBatch(
        card_ids=_pad_tensor(
            states.card_ids,
            shape=(batch_size, tokens),
            fill_value=0,
        ),
        areas=_pad_tensor(states.areas, shape=(batch_size, tokens), fill_value=0),
        owner_roles=_pad_tensor(
            states.owner_roles,
            shape=(batch_size, tokens),
            fill_value=0,
        ),
        token_kinds=_pad_tensor(
            states.token_kinds,
            shape=(batch_size, tokens),
            fill_value=0,
        ),
        scalars=_pad_tensor(
            states.scalars,
            shape=(batch_size, tokens, int(states.scalars.shape[2])),
            fill_value=0.0,
        ),
        last_attack_ids=_pad_tensor(
            states.last_attack_ids,
            shape=(batch_size, tokens),
            fill_value=0,
        ),
        padding_mask=padding_mask,
        attachment_card_ids=_pad_optional_tensor(
            states.attachment_card_ids,
            shape=(batch_size, attachments),
            fill_value=0,
        ),
        attachment_parent_indices=_pad_optional_tensor(
            states.attachment_parent_indices,
            shape=(batch_size, attachments),
            fill_value=0,
        ),
        attachment_kinds=_pad_optional_tensor(
            states.attachment_kinds,
            shape=(batch_size, attachments),
            fill_value=0,
        ),
        entity_slots=_pad_optional_tensor(
            states.entity_slots,
            shape=(batch_size, tokens),
            fill_value=0,
        ),
        root_input_fingerprints=(
            ()
            if not states.root_input_fingerprints
            else states.root_input_fingerprints
            + ("0" * 64,) * (batch_size - actual_batch_size)
        ),
    )


def _pad_option_batch_to_bucket(
    batch: OptionBatch,
    *,
    batch_size: int,
    option_width: int,
) -> OptionBatch:
    return OptionBatch(
        option_types=_pad_tensor(
            batch.option_types,
            shape=(batch_size, option_width),
            fill_value=0,
        ),
        contexts=_pad_tensor(
            batch.contexts,
            shape=(batch_size, option_width),
            fill_value=0,
        ),
        entity_slots=_pad_tensor(
            batch.entity_slots,
            shape=(batch_size, option_width, int(batch.entity_slots.shape[2])),
            fill_value=0,
        ),
        entity_slot_mask=_pad_tensor(
            batch.entity_slot_mask,
            shape=(batch_size, option_width, int(batch.entity_slot_mask.shape[2])),
            fill_value=False,
        ),
        attack_ids=_pad_tensor(
            batch.attack_ids,
            shape=(batch_size, option_width),
            fill_value=0,
        ),
        card_ids=_pad_tensor(
            batch.card_ids,
            shape=(batch_size, option_width),
            fill_value=0,
        ),
        scalars=_pad_tensor(
            batch.scalars,
            shape=(batch_size, option_width, int(batch.scalars.shape[2])),
            fill_value=0.0,
        ),
        dynamic_effect_features=_pad_tensor(
            batch.dynamic_effect_features,
            shape=(
                batch_size,
                option_width,
                int(batch.dynamic_effect_features.shape[2]),
            ),
            fill_value=0.0,
        ),
        dynamic_effect_masks=_pad_tensor(
            batch.dynamic_effect_masks,
            shape=(batch_size, option_width),
            fill_value=False,
        ),
        valid_options=_pad_tensor(
            batch.valid_options,
            shape=(batch_size, option_width),
            fill_value=False,
        ),
        min_counts=_pad_tensor(
            batch.min_counts,
            shape=(batch_size,),
            fill_value=0,
        ),
        max_counts=_pad_tensor(
            batch.max_counts,
            shape=(batch_size,),
            fill_value=0,
        ),
    )


def _move_state_batch(states: StateBatch, *, device: torch.device) -> StateBatch:
    if states.card_ids.device == device:
        return states
    return StateBatch(
        card_ids=states.card_ids.to(device=device),
        areas=states.areas.to(device=device),
        owner_roles=states.owner_roles.to(device=device),
        token_kinds=states.token_kinds.to(device=device),
        scalars=states.scalars.to(device=device),
        last_attack_ids=states.last_attack_ids.to(device=device),
        padding_mask=states.padding_mask.to(device=device),
        attachment_card_ids=_move_optional_tensor(
            states.attachment_card_ids,
            device=device,
        ),
        attachment_parent_indices=_move_optional_tensor(
            states.attachment_parent_indices,
            device=device,
        ),
        attachment_kinds=_move_optional_tensor(
            states.attachment_kinds,
            device=device,
        ),
        entity_slots=_move_optional_tensor(states.entity_slots, device=device),
        root_input_fingerprints=states.root_input_fingerprints,
    )


def _move_option_batch(options: OptionBatch, *, device: torch.device) -> OptionBatch:
    if options.valid_options.device == device:
        return options
    return OptionBatch(
        option_types=options.option_types.to(device=device),
        contexts=options.contexts.to(device=device),
        entity_slots=options.entity_slots.to(device=device),
        entity_slot_mask=options.entity_slot_mask.to(device=device),
        attack_ids=options.attack_ids.to(device=device),
        card_ids=options.card_ids.to(device=device),
        scalars=options.scalars.to(device=device),
        dynamic_effect_features=options.dynamic_effect_features.to(device=device),
        dynamic_effect_masks=options.dynamic_effect_masks.to(device=device),
        valid_options=options.valid_options.to(device=device),
        min_counts=options.min_counts.to(device=device),
        max_counts=options.max_counts.to(device=device),
    )


def _concat_state_batches(batches: Sequence[StateBatch]) -> StateBatch:
    if not batches:
        raise ValueError("state batches must be non-empty")
    max_tokens = max(int(batch.card_ids.shape[1]) for batch in batches)
    attachment_width = max(
        _optional_width(batch.attachment_card_ids) for batch in batches
    )
    return StateBatch(
        card_ids=_pad_cat_2d(
            tuple(batch.card_ids for batch in batches),
            width=max_tokens,
            fill_value=0,
        ),
        areas=_pad_cat_2d(
            tuple(batch.areas for batch in batches),
            width=max_tokens,
            fill_value=0,
        ),
        owner_roles=_pad_cat_2d(
            tuple(batch.owner_roles for batch in batches),
            width=max_tokens,
            fill_value=0,
        ),
        token_kinds=_pad_cat_2d(
            tuple(batch.token_kinds for batch in batches),
            width=max_tokens,
            fill_value=0,
        ),
        scalars=_pad_cat_3d(
            tuple(batch.scalars for batch in batches),
            width=max_tokens,
            fill_value=0.0,
        ),
        last_attack_ids=_pad_cat_2d(
            tuple(batch.last_attack_ids for batch in batches),
            width=max_tokens,
            fill_value=0,
        ),
        padding_mask=_pad_cat_2d(
            tuple(batch.padding_mask for batch in batches),
            width=max_tokens,
            fill_value=True,
        ),
        attachment_card_ids=_pad_cat_2d(
            tuple(
                _attachment_tensor(batch, "attachment_card_ids") for batch in batches
            ),
            width=attachment_width,
            fill_value=0,
        ),
        attachment_parent_indices=_pad_cat_2d(
            tuple(
                _attachment_tensor(batch, "attachment_parent_indices")
                for batch in batches
            ),
            width=attachment_width,
            fill_value=0,
        ),
        attachment_kinds=_pad_cat_2d(
            tuple(_attachment_tensor(batch, "attachment_kinds") for batch in batches),
            width=attachment_width,
            fill_value=0,
        ),
        entity_slots=_pad_cat_2d(
            tuple(_entity_slot_tensor(batch) for batch in batches),
            width=max_tokens,
            fill_value=0,
        ),
        root_input_fingerprints=_concat_root_input_fingerprints(batches),
    )


def _concat_root_input_fingerprints(
    batches: Sequence[StateBatch],
) -> tuple[str, ...]:
    """Concatenate optional CPU root identities without partial alignment."""
    if not any(batch.root_input_fingerprints for batch in batches):
        return ()
    for batch in batches:
        if len(batch.root_input_fingerprints) != int(batch.card_ids.shape[0]):
            raise ValueError(
                "state batches mix missing or misaligned root fingerprints"
            )
    return tuple(
        fingerprint
        for batch in batches
        for fingerprint in batch.root_input_fingerprints
    )


def _attachment_tensor(batch: StateBatch, name: str) -> Tensor:
    value = getattr(batch, name)
    if isinstance(value, Tensor):
        return value
    dtype = torch.uint8 if name == "attachment_kinds" else torch.uint16
    return torch.zeros(
        (int(batch.card_ids.shape[0]), 1),
        dtype=dtype,
        device=batch.card_ids.device,
    )


def _entity_slot_tensor(batch: StateBatch) -> Tensor:
    if batch.entity_slots is not None:
        return batch.entity_slots
    return torch.zeros_like(batch.card_ids, dtype=torch.uint8)


def _optional_width(tensor: Tensor | None) -> int:
    if tensor is None:
        return 1
    return int(tensor.shape[1])


def _state_attachment_width(states: StateBatch) -> int:
    """Return one aligned attachment width, preserving the absent branch."""
    tensors = (
        states.attachment_card_ids,
        states.attachment_parent_indices,
        states.attachment_kinds,
    )
    present = tuple(tensor for tensor in tensors if tensor is not None)
    if not present:
        return 0
    if len(present) != len(tensors):
        raise ValueError("attachment state tensors must be provided together")
    widths = {int(tensor.shape[1]) for tensor in present}
    if len(widths) != 1:
        raise ValueError("attachment state tensors must share one width")
    return widths.pop()


def _pad_optional_tensor(
    tensor: Tensor | None,
    *,
    shape: tuple[int, ...],
    fill_value: int | bool | float,
) -> Tensor | None:
    if tensor is None:
        return None
    return _pad_tensor(tensor, shape=shape, fill_value=fill_value)


def _move_optional_tensor(
    tensor: Tensor | None,
    *,
    device: torch.device,
) -> Tensor | None:
    if tensor is None:
        return None
    return tensor.to(device=device)


def _concat_option_batches(batches: Sequence[OptionBatch]) -> OptionBatch:
    if not batches:
        raise ValueError("option batches must be non-empty")
    max_options = max(int(batch.valid_options.shape[1]) for batch in batches)
    return OptionBatch(
        option_types=_pad_cat_2d(
            tuple(batch.option_types for batch in batches),
            width=max_options,
            fill_value=0,
        ),
        contexts=_pad_cat_2d(
            tuple(batch.contexts for batch in batches),
            width=max_options,
            fill_value=0,
        ),
        entity_slots=_pad_cat_3d(
            tuple(batch.entity_slots for batch in batches),
            width=max_options,
            fill_value=0,
        ),
        entity_slot_mask=_pad_cat_3d(
            tuple(batch.entity_slot_mask for batch in batches),
            width=max_options,
            fill_value=False,
        ),
        attack_ids=_pad_cat_2d(
            tuple(batch.attack_ids for batch in batches),
            width=max_options,
            fill_value=0,
        ),
        card_ids=_pad_cat_2d(
            tuple(batch.card_ids for batch in batches),
            width=max_options,
            fill_value=0,
        ),
        scalars=_pad_cat_3d(
            tuple(batch.scalars for batch in batches),
            width=max_options,
            fill_value=0.0,
        ),
        dynamic_effect_features=_pad_cat_3d(
            tuple(batch.dynamic_effect_features for batch in batches),
            width=max_options,
            fill_value=0.0,
        ),
        dynamic_effect_masks=_pad_cat_2d(
            tuple(batch.dynamic_effect_masks for batch in batches),
            width=max_options,
            fill_value=False,
        ),
        valid_options=_pad_cat_2d(
            tuple(batch.valid_options for batch in batches),
            width=max_options,
            fill_value=False,
        ),
        min_counts=torch.cat(tuple(batch.min_counts for batch in batches), dim=0),
        max_counts=torch.cat(tuple(batch.max_counts for batch in batches), dim=0),
    )


def _pad_cat_2d(
    tensors: Sequence[Tensor],
    *,
    width: int,
    fill_value: int | bool | float,
) -> Tensor:
    rows = [
        _pad_tensor(tensor, shape=(int(tensor.shape[0]), width), fill_value=fill_value)
        for tensor in tensors
    ]
    return torch.cat(tuple(rows), dim=0)


def _pad_cat_3d(
    tensors: Sequence[Tensor],
    *,
    width: int,
    fill_value: int | bool | float,
) -> Tensor:
    rows = [
        _pad_tensor(
            tensor,
            shape=(int(tensor.shape[0]), width, int(tensor.shape[2])),
            fill_value=fill_value,
        )
        for tensor in tensors
    ]
    return torch.cat(tuple(rows), dim=0)


def _pad_tensor(
    tensor: Tensor,
    *,
    shape: tuple[int, ...],
    fill_value: int | bool | float,
) -> Tensor:
    if tuple(tensor.shape) == shape:
        return tensor
    if len(tensor.shape) != len(shape):
        raise ValueError("padded tensor rank mismatch")
    if any(
        int(current) > target
        for current, target in zip(tensor.shape, shape, strict=True)
    ):
        raise ValueError("padded tensor target shape is smaller than source")
    padded = torch.full(
        shape,
        fill_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    slices = tuple(slice(0, int(size)) for size in tensor.shape)
    padded[slices] = tensor
    return padded


def _policy_for_id(
    policies: Mapping[str, InferencePolicy],
    policy_id: str,
) -> InferencePolicy:
    try:
        return policies[policy_id]
    except KeyError as exc:
        raise KeyError(f"inference policy is not loaded: {policy_id}") from exc


def _response_queue_for_actor(
    response_queues: Mapping[str, ResponseQueue],
    actor_id: str,
) -> ResponseQueue:
    try:
        return response_queues[actor_id]
    except KeyError as exc:
        raise KeyError(
            f"inference response queue is not registered: {actor_id}"
        ) from exc


def _validate_response(
    response: InferenceResponse,
    actor_id: str,
    actor_incarnation: int,
    request_id: int,
    policy_id: str,
    request_type: InferenceRequestType,
) -> None:
    if response.actor_id != actor_id:
        raise RuntimeError("inference response actor_id mismatch")
    if response.actor_incarnation != actor_incarnation:
        raise RuntimeError("inference response actor incarnation mismatch")
    if response.request_id != request_id:
        raise RuntimeError("inference response request_id mismatch")
    if response.policy_id != policy_id:
        raise RuntimeError("inference response policy_id mismatch")
    if response.request_type != request_type:
        raise RuntimeError("inference response request_type mismatch")


def _validate_response_batch(response: InferenceResponse, batch_size: int) -> None:
    if response.batch_size != batch_size:
        raise RuntimeError("inference response batch size mismatch")
    if response.request_type == "recurrent_release":
        if (
            response.actions
            or response.action_logprobs.numel() != 0
            or response.values.numel() != 0
        ):
            raise RuntimeError("recurrent release returned model evidence")
        if not 0 <= response.released_recurrent_sequences <= batch_size:
            raise RuntimeError("recurrent release count is invalid")
        if any(
            field is not None
            for field in (
                response.token_logprobs,
                response.prefix_values,
                response.token_mask,
                response.stop_sampled,
                response.recurrent_result,
            )
        ):
            raise RuntimeError("recurrent release returned decode evidence")
        return
    if response.planner_fallback_reason and (
        response.planner_fallback_reason != "model_lease_capacity"
        or response.request_type != "decode"
        or response.planner_context_handles
    ):
        raise RuntimeError("inference response has invalid planner fallback")
    planner_fields = (
        response.planner_base_action_logprobs,
        response.planner_proposal_action_logprobs,
        response.planner_reranker_residuals,
    )
    if response.request_type == "planner_proposals":
        payload = response.planner_proposal_response
        if response.actions or response.action_logprobs.numel() != 0:
            raise RuntimeError("proposal response must not contain decode evidence")
        if response.values.numel() != 0 or payload is None:
            raise RuntimeError("proposal response payload is incomplete")
        if len(payload.decisions) != batch_size:
            raise RuntimeError("proposal response decision count is misaligned")
        if any(field is not None for field in planner_fields) or (
            response.planner_candidate_counts
        ):
            raise RuntimeError("proposal response contains candidate evidence")
        if any(
            field is not None
            for field in (
                response.token_logprobs,
                response.prefix_values,
                response.token_mask,
                response.stop_sampled,
            )
        ):
            raise RuntimeError("proposal response must not contain a token trace")
        if response.planner_context_handles:
            raise RuntimeError("proposal response must not issue new root contexts")
        return
    if response.request_type == "planner_candidates":
        if response.actions or response.action_logprobs.numel() != 0:
            raise RuntimeError("planner response must not contain decode evidence")
        if response.values.numel() != 0:
            raise RuntimeError("planner response must not contain value evidence")
        if any(field is None for field in planner_fields):
            raise RuntimeError("planner response is missing candidate evidence")
        counts = response.planner_candidate_counts
        if len(counts) != batch_size or any(count <= 0 for count in counts):
            raise RuntimeError("planner response candidate counts are invalid")
        candidate_count = sum(counts)
        if any(
            cast(Tensor, field).ndim != 1
            or int(cast(Tensor, field).shape[0]) != candidate_count
            or not bool(torch.isfinite(cast(Tensor, field)).all().item())
            for field in planner_fields
        ):
            raise RuntimeError("planner response candidate tensors are misaligned")
        if any(
            field is not None
            for field in (
                response.token_logprobs,
                response.prefix_values,
                response.token_mask,
                response.stop_sampled,
            )
        ):
            raise RuntimeError("planner response must not contain a token trace")
        return
    if response.planner_proposal_response is not None:
        raise RuntimeError("non-proposal response contains proposal evidence")
    if any(field is not None for field in planner_fields) or (
        response.planner_candidate_counts
    ):
        raise RuntimeError("non-planner response contains candidate evidence")
    if int(response.values.shape[0]) != batch_size:
        raise RuntimeError("inference response value batch size mismatch")
    if response.request_type != "decode":
        if response.actions or response.action_logprobs.numel() != 0:
            raise RuntimeError("value response must not contain decode evidence")
        if any(
            field is not None
            for field in (
                response.token_logprobs,
                response.prefix_values,
                response.token_mask,
                response.stop_sampled,
            )
        ):
            raise RuntimeError("value response must not contain a token trace")
        if response.planner_context_handles:
            raise RuntimeError("non-decode response contains planner contexts")
        return
    if int(response.action_logprobs.shape[0]) != batch_size:
        raise RuntimeError("inference response logprob batch size mismatch")
    if response.planner_context_handles and (
        len(response.planner_context_handles) != batch_size
        or len(set(response.planner_context_handles)) != batch_size
    ):
        raise RuntimeError("decode planner context handles are misaligned")


def _validate_response_token_trace(
    response: InferenceResponse,
    batch_size: int,
) -> None:
    fields = (
        response.token_logprobs,
        response.prefix_values,
        response.token_mask,
        response.stop_sampled,
    )
    if any(field is None for field in fields):
        raise RuntimeError("inference response is missing its token trace")
    _validate_policy_sample_tensors(
        logprobs=response.action_logprobs,
        values=response.values,
        token_logprobs=response.token_logprobs,
        prefix_values=response.prefix_values,
        token_mask=response.token_mask,
        stop_sampled=response.stop_sampled,
        batch_size=batch_size,
    )


def _response_buffer_key(
    actor_incarnation: int,
    policy_id: str,
    request_id: int,
) -> _RemoteResponseKey:
    return (int(actor_incarnation), policy_id, request_id)


def _state_dict_from_checkpoint(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return cast(Mapping[str, Any], value)
        return cast(Mapping[str, Any], checkpoint)
    raise TypeError("published weights must be a state_dict or checkpoint mapping")


def _set_policy_publication(
    policy: object,
    *,
    version: int,
    model_fingerprint: str,
) -> None:
    """Expose an already-loaded model version and its exact content atomically."""
    validate_model_fingerprint(model_fingerprint)
    reset = getattr(policy, "reset_planner_contexts", None)
    if callable(reset):
        cast(Any, reset)(
            policy_version=int(version),
            model_fingerprint=model_fingerprint,
        )
        return
    current = getattr(policy, "policy_version", None)
    if current is not None and not callable(current):
        cast(Any, policy).policy_version = int(version)
    current_fingerprint = getattr(policy, "model_fingerprint", None)
    if current_fingerprint is not None and not callable(current_fingerprint):
        cast(Any, policy).model_fingerprint = model_fingerprint


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
