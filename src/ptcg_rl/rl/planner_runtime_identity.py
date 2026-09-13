"""Canonical static and per-lease identities for one formal planner runtime.

Hydra owns only stable planner semantics.  Mutable publication data (model
bytes, policy version, and proposal-head version) is supplied when a model
lease is acquired.  Keeping those layers separate prevents a static profile
from silently becoming stale after a learner snapshot publication.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.agent.search.hierarchical_contract import (
    StableContinuationControllerIdentity,
)
from ptcg_rl.agent.search.planner_scoring import PlannerScoringConfig
from ptcg_rl.agent.search.planning_session_contract import (
    HierarchicalSearchConfig,
)
from ptcg_rl.agent.search.proposal_generation import PlannerProposalSearchLimits
from ptcg_rl.agent.search.root_information_tensorizer import (
    ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT,
    RootInformationTensorizerConfig,
)
from ptcg_rl.engine.native_planning_session_pool_contract import (
    NativePlanningSessionPoolConfig,
)
from ptcg_rl.rl.planner_behavior_policy_contract import (
    PlannerBehaviorPolicyConfig,
    PlannerRuntimeIdentity,
)
from ptcg_rl.rl.planner_service_inputs import PlannerRequestCostConfig
from ptcg_rl.runtime.work_ledger import PlannerWorkLimits

if TYPE_CHECKING:
    from ptcg_rl.rl.learner import Schema9LearnerBatchConfig


PLANNER_RUNTIME_IDENTITY_SCHEMA_VERSION: Final[Literal[2]] = 2

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_POLICY_DOMAIN = b"ptcg-rl/planner-model-policy-binding/v2\x00"
_CONSTRUCTOR_DOMAIN = b"ptcg-rl/resolved-planner-constructor/v2\x00"
_TENSORIZER_DOMAIN = b"ptcg-rl/resolved-root-information-tensorizer/v2\x00"
_CONTINUATION_DOMAIN = b"ptcg-rl/resolved-continuation-semantics/v2\x00"
_PLANNER_DOMAIN = b"ptcg-rl/resolved-planner-semantics/v2\x00"
_RUNTIME_LEASE_DOMAIN = b"ptcg-rl/resolved-planner-runtime-lease/v2\x00"


class PlannerModelPolicyBinding(BaseModel):
    """The exact model artifact and policy version held by one lease."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_fingerprint: str
    policy_version: int = Field(ge=0)

    @field_validator("model_fingerprint")
    @classmethod
    def valid_model_fingerprint(cls, value: str) -> str:
        """Require an exact lowercase artifact digest."""
        if _SHA256.fullmatch(value) is None:
            raise ValueError("model_fingerprint must be lowercase SHA-256 hex")
        return value

    @property
    def fingerprint(self) -> str:
        """Bind the model bytes and their wire policy version as one pair."""
        return _canonical_fingerprint(
            _MODEL_POLICY_DOMAIN,
            {
                "identity_schema_version": PLANNER_RUNTIME_IDENTITY_SCHEMA_VERSION,
                "model_fingerprint": self.model_fingerprint,
                "policy_version": self.policy_version,
            },
        )


class PlannerScenarioSemanticsConfig(BaseModel):
    """Fixed belief and engine-exposed chance support semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_version: Literal[1] = 1
    support_mode: Literal["belief_sampled_chance_enumerated"] = (
        "belief_sampled_chance_enumerated"
    )
    belief_sampler_fingerprint: str
    belief_world_count: int = Field(gt=0)
    manual_coin_semantics: Literal["binary_yes_no_equal_probability"] = (
        "binary_yes_no_equal_probability"
    )
    unsupported_randomness_policy: Literal["whole_decision_base_fallback"] = (
        "whole_decision_base_fallback"
    )
    paired_support: Literal[True] = True

    @field_validator("belief_sampler_fingerprint")
    @classmethod
    def valid_belief_sampler_fingerprint(cls, value: str) -> str:
        """Require a content identity for the resolved belief sampler."""
        if _SHA256.fullmatch(value) is None:
            raise ValueError("belief_sampler_fingerprint must be lowercase SHA-256 hex")
        return value


class PlannerEngineSemanticsConfig(BaseModel):
    """Exact native library, v5 ABI, and Python payload schema identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    library_fingerprint: str
    native_abi_fingerprint: str
    native_schema_fingerprint: str

    @field_validator(
        "library_fingerprint",
        "native_abi_fingerprint",
        "native_schema_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require exact lowercase SHA-256 content identities."""
        if _SHA256.fullmatch(value) is None:
            raise ValueError("engine identities must be lowercase SHA-256 hex")
        return value


class PlannerDeadlineLimits(BaseModel):
    """Bounded action-critical deadline and cleanup intervals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_timeout_seconds: float
    return_guard_seconds: float
    inference_timeout_seconds: float
    cleanup_timeout_seconds: float

    @field_validator(
        "request_timeout_seconds",
        "inference_timeout_seconds",
        "cleanup_timeout_seconds",
    )
    @classmethod
    def positive_finite_seconds(cls, value: float) -> float:
        """Require usable finite timeout intervals."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("planner timeout intervals must be finite and positive")
        return value

    @field_validator("return_guard_seconds")
    @classmethod
    def nonnegative_finite_guard(cls, value: float) -> float:
        """Allow a zero guard while rejecting non-replayable values."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("return_guard_seconds must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def leave_time_to_return(self) -> Self:
        """Keep every foreground inference deadline inside the request."""
        if self.return_guard_seconds >= self.request_timeout_seconds:
            raise ValueError("return guard must be smaller than request timeout")
        if (
            self.inference_timeout_seconds + self.return_guard_seconds
            > self.request_timeout_seconds
        ):
            raise ValueError("inference timeout leaves no configured return guard")
        return self


class PlannerBatchingLimits(BaseModel):
    """Fixed actor, queue, and GPU microbatch geometry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_count: int
    max_root_rows_per_request: int
    planner_queue_capacity: int
    inference_queue_capacity: int
    proposal_microbatch_rows: int
    candidate_microbatch_rows: int
    root_value_microbatch_rows: int
    inference_batch_wait_ms: float

    @field_validator(
        "actor_count",
        "max_root_rows_per_request",
        "planner_queue_capacity",
        "inference_queue_capacity",
        "proposal_microbatch_rows",
        "candidate_microbatch_rows",
        "root_value_microbatch_rows",
    )
    @classmethod
    def positive_capacity(cls, value: int) -> int:
        """Require bounded non-empty queues and batches."""
        if value <= 0:
            raise ValueError("planner batching capacities must be positive")
        return value

    @field_validator("inference_batch_wait_ms")
    @classmethod
    def finite_batch_wait(cls, value: float) -> float:
        """Require a finite non-negative batching window."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("inference_batch_wait_ms must be finite and non-negative")
        return value


class PlannerContextLimits(BaseModel):
    """Bound retained root rows owned by active immutable model leases.

    This is a lifecycle store, not a cross-request result cache. Engine-prefix
    and unique-leaf reuse remain request-local and are reported as telemetry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    retained_root_rows: int = Field(gt=0)


class PlannerModelLeaseLimits(BaseModel):
    """Cap snapshot residency and concurrent immutable model leases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_resident_snapshots: int = Field(gt=0)
    max_in_flight_leases: int = Field(gt=0)


class PlannerBufferLimits(BaseModel):
    """Fixed admission-ledger capacities for planner staging memory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_inflight_requests: int = Field(gt=0)
    host_bytes_per_cell: int = Field(gt=0)
    ipc_bytes_per_cell: int = Field(gt=0)
    gpu_bytes_per_unique_leaf: int = Field(ge=0)
    host_pool_bytes: int = Field(gt=0)
    ipc_pool_bytes: int = Field(gt=0)
    gpu_staging_bytes: int = Field(ge=0)

    def validate_runtime(self, runtime: ResolvedPlannerRuntimeConfig) -> None:
        """Reject geometry that can overrun a declared staging ledger."""
        candidate_count = runtime.planner_behavior.constructor.k_total
        scenario_count = runtime.scenario.belief_world_count
        cell_count = self.max_inflight_requests * candidate_count * scenario_count
        if self.host_pool_bytes < cell_count * self.host_bytes_per_cell:
            raise ValueError("host pool cannot cover the worst-case planner grid")
        if self.ipc_pool_bytes < cell_count * self.ipc_bytes_per_cell:
            raise ValueError("IPC pool cannot cover the worst-case planner grid")
        unique_leaf_count = (
            self.max_inflight_requests * runtime.tensorizer.max_unique_leaves
        )
        if self.gpu_staging_bytes < (
            unique_leaf_count * self.gpu_bytes_per_unique_leaf
        ):
            raise ValueError("GPU staging cannot cover the worst-case planner grid")
        active_and_queued = (
            runtime.batching.actor_count
            + runtime.batching.planner_queue_capacity
        )
        if self.max_inflight_requests > active_and_queued:
            raise ValueError(
                "inflight capacity exceeds actors plus bounded planner queue"
            )


@dataclass(frozen=True, slots=True)
class ResolvedPlannerStaticIdentity:
    """Derived identities that are invariant across model publications."""

    identity_schema_version: int
    controller: StableContinuationControllerIdentity
    constructor_fingerprint: str
    scorer_fingerprint: str
    planner_fingerprint: str
    tensor_schema_fingerprint: str
    tensorizer_fingerprint: str
    continuation_semantics_fingerprint: str
    root_information_tensorizer: RootInformationTensorizerConfig

    def schema9_learner_config(self) -> Schema9LearnerBatchConfig:
        """Build the learner contract from the same resolved static identity."""
        from ptcg_rl.rl.learner import Schema9LearnerBatchConfig

        return Schema9LearnerBatchConfig(
            root_information_tensorizer=self.root_information_tensorizer,
            expected_constructor_fingerprint=self.constructor_fingerprint,
            expected_scorer_fingerprint=self.scorer_fingerprint,
            expected_controller_fingerprint=self.controller.controller_fingerprint,
            expected_planner_fingerprint=self.planner_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class ResolvedPlannerRuntimeIdentity:
    """Static semantics bound to one immutable model-version lease."""

    static: ResolvedPlannerStaticIdentity
    runtime_identity: PlannerRuntimeIdentity
    model_policy_fingerprint: str
    runtime_fingerprint: str

    @property
    def identity_schema_version(self) -> int:
        """Return the static identity schema version."""
        return self.static.identity_schema_version

    @property
    def controller(self) -> StableContinuationControllerIdentity:
        """Return the stable controller shared by all snapshot leases."""
        return self.static.controller

    @property
    def tensor_schema_fingerprint(self) -> str:
        """Return the root-information tensor wire contract."""
        return self.static.tensor_schema_fingerprint

    @property
    def tensorizer_fingerprint(self) -> str:
        """Return the static root-information tensorizer identity."""
        return self.static.tensorizer_fingerprint

    @property
    def continuation_semantics_fingerprint(self) -> str:
        """Return the stable engine-continuation semantics identity."""
        return self.static.continuation_semantics_fingerprint

    @property
    def root_information_tensorizer(self) -> RootInformationTensorizerConfig:
        """Return the static root-information tensorizer configuration."""
        return self.static.root_information_tensorizer

    def schema9_learner_config(self) -> Schema9LearnerBatchConfig:
        """Build the learner contract without depending on lease data."""
        return self.static.schema9_learner_config()


class ResolvedPlannerRuntimeConfig(BaseModel):
    """Static source configuration for canonical planner identities.

    Composite fingerprints are deliberately not accepted as fields.  A caller
    must resolve the stable semantics, then explicitly bind publication data
    through :meth:`resolve_for_lease`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity_schema_version: Literal[2] = PLANNER_RUNTIME_IDENTITY_SCHEMA_VERSION
    controller_version: str
    planner_behavior: PlannerBehaviorPolicyConfig
    request_costs: PlannerRequestCostConfig
    proposal_search: PlannerProposalSearchLimits
    hierarchical_search: HierarchicalSearchConfig
    session_pool: NativePlanningSessionPoolConfig
    scoring: PlannerScoringConfig
    tensorizer: RootInformationTensorizerConfig
    scenario: PlannerScenarioSemanticsConfig
    engine: PlannerEngineSemanticsConfig
    work_limits: PlannerWorkLimits
    deadlines: PlannerDeadlineLimits
    batching: PlannerBatchingLimits
    contexts: PlannerContextLimits
    leases: PlannerModelLeaseLimits
    buffers: PlannerBufferLimits

    @field_validator("controller_version")
    @classmethod
    def nonempty_controller_version(cls, value: str) -> str:
        """Canonicalize the stable controller implementation version."""
        canonical = value.strip()
        if not canonical:
            raise ValueError("controller_version must not be empty")
        return canonical

    @model_validator(mode="after")
    def coherent_runtime_geometry(self) -> Self:
        """Reject source configurations that cannot share one runtime ledger."""
        constructor = self.planner_behavior.constructor
        budget = constructor.work_budget
        probe_cap = self.planner_behavior.eligibility.single_select_probe_cap
        if probe_cap > constructor.exhaustive_action_cap:
            raise ValueError(
                "fixed-select probe cap must fit exhaustive constructor support"
            )
        if probe_cap > constructor.k_seed:
            raise ValueError("fixed-select probe cap must fit the seed support")
        if budget.engine_transition_limit != self.work_limits.max_transitions:
            raise ValueError(
                "constructor transition budget must equal the request ledger limit"
            )
        if budget.prefix_node_limit != self.work_limits.max_nodes:
            raise ValueError(
                "constructor prefix-node budget must equal the request ledger limit"
            )
        if constructor.k_total > self.work_limits.max_candidates:
            raise ValueError("constructor support exceeds the request candidate limit")
        if constructor.k_total > self.batching.candidate_microbatch_rows:
            raise ValueError("one candidate support exceeds its GPU microbatch limit")
        if (
            self.batching.max_root_rows_per_request
            > self.batching.proposal_microbatch_rows
        ):
            raise ValueError(
                "one root request exceeds its proposal GPU microbatch limit"
            )
        root_rows = constructor.k_total * self.scenario.belief_world_count
        root_capacities = {
            "hierarchical state slots": self.hierarchical_search.max_state_slots,
            "native engine steps": (self.hierarchical_search.max_engine_steps_per_call),
            "session-pool chunk": self.session_pool.max_transitions_per_call,
            "request native-call chunk": (
                self.work_limits.max_native_transitions_per_call
            ),
            "request transitions": self.work_limits.max_transitions,
            "request nodes": self.work_limits.max_nodes,
        }
        for label, capacity in root_capacities.items():
            if root_rows > capacity:
                raise ValueError(f"maximum root grid exceeds {label} capacity")
        if (
            self.hierarchical_search.max_continue_rows_per_call
            > self.session_pool.max_transitions_per_call
        ):
            raise ValueError("hierarchical continuation chunks exceed pool capacity")
        if (
            self.hierarchical_search.max_continue_rows_per_call
            > self.work_limits.max_native_transitions_per_call
        ):
            raise ValueError(
                "hierarchical continuation chunks exceed request call capacity"
            )
        if self.leases.max_in_flight_leases < self.batching.actor_count:
            raise ValueError("model lease capacity cannot cover configured actors")
        minimum_context_rows = (
            self.batching.actor_count
            * self.batching.max_root_rows_per_request
        )
        if self.contexts.retained_root_rows < minimum_context_rows:
            raise ValueError(
                "retained root context capacity cannot cover active model leases"
            )
        self.buffers.validate_runtime(self)
        return self

    def resolve_static(self) -> ResolvedPlannerStaticIdentity:
        """Derive identities that remain stable across snapshot publications."""
        constructor_fingerprint = _canonical_fingerprint(
            _CONSTRUCTOR_DOMAIN,
            self._constructor_payload(),
        )
        tensorizer_fingerprint = _canonical_fingerprint(
            _TENSORIZER_DOMAIN,
            {
                "identity_schema_version": self.identity_schema_version,
                "tensor_schema_fingerprint": (
                    ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT
                ),
                "tensorizer": self.tensorizer.model_dump(mode="json"),
            },
        )
        continuation_fingerprint = _canonical_fingerprint(
            _CONTINUATION_DOMAIN,
            {
                "identity_schema_version": self.identity_schema_version,
                "hierarchical_search": self.hierarchical_search.model_dump(mode="json"),
                "native_session_caps": _native_caps_payload(self.hierarchical_search),
                "scenario": self.scenario.model_dump(mode="json"),
                "engine": self.engine.model_dump(mode="json"),
                "tensor_schema_fingerprint": (
                    ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT
                ),
                "tensorizer_fingerprint": tensorizer_fingerprint,
            },
        )
        scorer_fingerprint = self.scoring.scorer_fingerprint
        controller = StableContinuationControllerIdentity.create(
            controller_version=self.controller_version,
            constructor_fingerprint=constructor_fingerprint,
            scorer_fingerprint=scorer_fingerprint,
            continuation_semantics_fingerprint=continuation_fingerprint,
        )
        controller_fingerprint = controller.controller_fingerprint
        planner_fingerprint = _canonical_fingerprint(
            _PLANNER_DOMAIN,
            self._planner_payload(
                constructor_fingerprint=constructor_fingerprint,
                scorer_fingerprint=scorer_fingerprint,
                tensorizer_fingerprint=tensorizer_fingerprint,
                continuation_fingerprint=continuation_fingerprint,
                controller_fingerprint=controller_fingerprint,
            ),
        )
        return ResolvedPlannerStaticIdentity(
            identity_schema_version=self.identity_schema_version,
            controller=controller,
            constructor_fingerprint=constructor_fingerprint,
            scorer_fingerprint=scorer_fingerprint,
            planner_fingerprint=planner_fingerprint,
            tensor_schema_fingerprint=ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT,
            tensorizer_fingerprint=tensorizer_fingerprint,
            continuation_semantics_fingerprint=continuation_fingerprint,
            root_information_tensorizer=self.tensorizer,
        )

    def resolve_for_lease(
        self,
        *,
        model_fingerprint: str,
        policy_version: int,
        proposal_version: int,
    ) -> ResolvedPlannerRuntimeIdentity:
        """Bind stable semantics to one exact immutable serving lease."""
        static = self.resolve_static()
        model_policy = PlannerModelPolicyBinding(
            model_fingerprint=model_fingerprint,
            policy_version=policy_version,
        )
        runtime_identity = PlannerRuntimeIdentity(
            model_fingerprint=model_policy.model_fingerprint,
            constructor_fingerprint=static.constructor_fingerprint,
            scorer_fingerprint=static.scorer_fingerprint,
            controller_fingerprint=static.controller.controller_fingerprint,
            planner_fingerprint=static.planner_fingerprint,
            policy_version=model_policy.policy_version,
            proposal_version=proposal_version,
            constructor_version=self.planner_behavior.constructor.architecture_version,
            planner_version=self.planner_behavior.architecture_version,
        )
        runtime_fingerprint = _canonical_fingerprint(
            _RUNTIME_LEASE_DOMAIN,
            {
                "identity_schema_version": self.identity_schema_version,
                "planner_fingerprint": static.planner_fingerprint,
                "model_policy_fingerprint": model_policy.fingerprint,
                "model_fingerprint": runtime_identity.model_fingerprint,
                "policy_version": runtime_identity.policy_version,
                "proposal_version": runtime_identity.proposal_version,
                "constructor_version": runtime_identity.constructor_version,
                "planner_version": runtime_identity.planner_version,
            },
        )
        return ResolvedPlannerRuntimeIdentity(
            static=static,
            runtime_identity=runtime_identity,
            model_policy_fingerprint=model_policy.fingerprint,
            runtime_fingerprint=runtime_fingerprint,
        )

    def _constructor_payload(self) -> dict[str, Any]:
        return {
            "identity_schema_version": self.identity_schema_version,
            "planner_behavior_architecture_version": (
                self.planner_behavior.architecture_version
            ),
            "constructor": self.planner_behavior.constructor.model_dump(mode="json"),
            "eligibility": self.planner_behavior.eligibility.model_dump(mode="json"),
            "max_mutation_parents": self.planner_behavior.max_mutation_parents,
            "request_costs": self.request_costs.model_dump(mode="json"),
            "proposal_search": self.proposal_search.model_dump(mode="json"),
        }

    def _planner_payload(
        self,
        *,
        constructor_fingerprint: str,
        scorer_fingerprint: str,
        tensorizer_fingerprint: str,
        continuation_fingerprint: str,
        controller_fingerprint: str,
    ) -> dict[str, Any]:
        return {
            "identity_schema_version": self.identity_schema_version,
            "planner_version": self.planner_behavior.architecture_version,
            "controller_version": self.controller_version,
            "constructor_fingerprint": constructor_fingerprint,
            "scorer_fingerprint": scorer_fingerprint,
            "tensorizer_fingerprint": tensorizer_fingerprint,
            "continuation_semantics_fingerprint": continuation_fingerprint,
            "controller_fingerprint": controller_fingerprint,
            "planner_behavior": self.planner_behavior.model_dump(mode="json"),
            "request_costs": self.request_costs.model_dump(mode="json"),
            "proposal_search": self.proposal_search.model_dump(mode="json"),
            "hierarchical_search": self.hierarchical_search.model_dump(mode="json"),
            "native_session_caps": _native_caps_payload(self.hierarchical_search),
            "session_pool": self.session_pool.model_dump(mode="json"),
            "scoring": self.scoring.model_dump(mode="json"),
            "tensor_schema_fingerprint": ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT,
            "tensorizer": self.tensorizer.model_dump(mode="json"),
            "scenario": self.scenario.model_dump(mode="json"),
            "engine": self.engine.model_dump(mode="json"),
            "work_limits": self.work_limits.model_dump(mode="json"),
            "deadlines": self.deadlines.model_dump(mode="json"),
            "batching": self.batching.model_dump(mode="json"),
            "contexts": self.contexts.model_dump(mode="json"),
            "leases": self.leases.model_dump(mode="json"),
            "buffers": self.buffers.model_dump(mode="json"),
        }


def resolve_planner_runtime(
    config: ResolvedPlannerRuntimeConfig,
    *,
    model_fingerprint: str,
    policy_version: int,
    proposal_version: int,
) -> ResolvedPlannerRuntimeIdentity:
    """Bind a validated static config to one immutable model lease."""
    return config.resolve_for_lease(
        model_fingerprint=model_fingerprint,
        policy_version=policy_version,
        proposal_version=proposal_version,
    )


def _canonical_fingerprint(domain: bytes, payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


def _native_caps_payload(config: HierarchicalSearchConfig) -> dict[str, int]:
    caps = config.native_caps
    return {
        "max_engine_steps": caps.max_engine_steps,
        "max_forced_steps": caps.max_forced_steps,
        "max_observation_bytes": caps.max_observation_bytes,
    }


__all__ = [
    "PLANNER_RUNTIME_IDENTITY_SCHEMA_VERSION",
    "PlannerBatchingLimits",
    "PlannerBufferLimits",
    "PlannerContextLimits",
    "PlannerDeadlineLimits",
    "PlannerEngineSemanticsConfig",
    "PlannerModelLeaseLimits",
    "PlannerModelPolicyBinding",
    "PlannerScenarioSemanticsConfig",
    "ResolvedPlannerRuntimeConfig",
    "ResolvedPlannerRuntimeIdentity",
    "ResolvedPlannerStaticIdentity",
    "StableContinuationControllerIdentity",
    "resolve_planner_runtime",
]
