"""Validated decision and run records for integrated planner profiling."""

from __future__ import annotations

import math
import re
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ptcg_rl.evaluation.planner_profile_config import (
    PlannerDecisionShape,
    PlannerProfileEnvironment,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PlannerProfileDecisionRecord(BaseModel):
    """One privacy-safe behavior decision and integrated work measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    campaign_id: str
    point_id: str
    budget_id: str
    environment: PlannerProfileEnvironment
    corpus_row_id: str
    corpus_repetition: int = Field(ge=0)
    decision_shapes: tuple[PlannerDecisionShape, ...]
    decision_index: int = Field(ge=0)
    batch_row_position: int = Field(ge=0)
    model_fingerprint: str
    runtime_fingerprint: str
    planner_fingerprint: str
    scenario_support_fingerprint: str | None = None
    policy_version: int = Field(ge=0)
    proposal_version: int = Field(ge=0)
    planner_eligible: bool
    equivalence_probe_used: bool
    planner_used: bool
    policy_runtime_agent_used: bool
    fallback_reason: str | None = None
    base_action_fingerprint: str
    selected_action_fingerprint: str
    legal_action_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    scenario_count: int = Field(ge=0)
    belief_world_count: int = Field(ge=0)
    chance_outcome_count: int = Field(ge=0)
    engine_transitions: int = Field(ge=0)
    prefix_nodes: int = Field(ge=0)
    prefix_reuse_count: int = Field(ge=0)
    unique_leaf_count: int = Field(ge=0)
    consequence_cell_count: int = Field(ge=0)
    native_chunk_size: int = Field(ge=0)
    gpu_rows: int = Field(ge=0)
    gpu_batch_capacity: int = Field(ge=0)
    native_lane_occupancy: int = Field(ge=0)
    ipc_bytes: int | None = Field(default=None, ge=0)
    ipc_measurement_unavailable_reason: str | None = None
    model_lease_lifetime_ms: float
    actor_policy_wait_ms: float
    planner_queue_wait_ms: float
    total_latency_ms: float
    base_planner_agree: bool | None = None
    support_exhaustive: bool
    scenario_grid_complete: bool
    rules_exact: bool
    leaf_bootstrapped: bool
    failed: bool = Field(
        default=False,
        description=(
            "Unusable/corrupt measurement only; safe production fallback is not failure"
        ),
    )
    deadline_exceeded: bool = False
    oracle_compatible: bool
    oracle_exclusion_reason: str | None = None
    oracle_provenance_fingerprint: str | None = None
    oracle_executed: bool = False
    oracle_candidate_count: int = Field(ge=0)
    oracle_scenario_count: int = Field(ge=0)
    oracle_engine_transitions: int = Field(ge=0)
    oracle_scenario_grid_complete: bool
    oracle_rules_exact: bool
    oracle_latency_ms: float | None = None
    base_action_regret: float | None = None
    served_action_regret: float | None = None
    candidate_best_regret: float | None = None
    served_epsilon_optimal: bool | None = None
    candidate_epsilon_recall: bool | None = None
    value_calibration_error: float | None = None

    @field_validator(
        "model_fingerprint",
        "runtime_fingerprint",
        "planner_fingerprint",
        "base_action_fingerprint",
        "selected_action_fingerprint",
    )
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("planner profile identities must be SHA-256")
        return value

    @field_validator(
        "scenario_support_fingerprint",
        "oracle_provenance_fingerprint",
    )
    @classmethod
    def optional_sha256(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("scenario support identity must be SHA-256")
        return value

    @field_validator(
        "model_lease_lifetime_ms",
        "actor_policy_wait_ms",
        "planner_queue_wait_ms",
        "total_latency_ms",
    )
    @classmethod
    def nonnegative_time(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("planner profile times must be finite and non-negative")
        return value

    @field_validator(
        "oracle_latency_ms",
        "base_action_regret",
        "served_action_regret",
        "candidate_best_regret",
        "value_calibration_error",
    )
    @classmethod
    def optional_nonnegative_finite(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise ValueError(
                "planner diagnostics must be finite and non-negative when present"
            )
        return value

    @model_validator(mode="after")
    def coherent_oracle_evidence(self) -> Self:
        if (self.ipc_bytes is None) != bool(self.ipc_measurement_unavailable_reason):
            raise ValueError(
                "IPC bytes must be measured or carry an unavailable reason"
            )
        if self.planner_used and not self.planner_eligible:
            raise ValueError("planner work requires an eligible decision")
        if self.planner_used and self.failed:
            raise ValueError("failed decisions cannot claim valid planner evidence")
        if self.planner_used and (
            self.candidate_count <= 0
            or self.scenario_count <= 0
            or not self.scenario_grid_complete
            or not self.rules_exact
        ):
            raise ValueError("planner work requires complete exact candidate evidence")
        if self.planner_used and self.fallback_reason is not None:
            raise ValueError("planner evidence cannot also claim a fallback")
        if not self.planner_used and not self.fallback_reason:
            raise ValueError("a non-planner branch requires an explicit reason")
        if self.candidate_count > self.legal_action_count:
            raise ValueError("candidate support cannot exceed the legal action support")
        if self.support_exhaustive and self.candidate_count != self.legal_action_count:
            raise ValueError("exhaustive support must retain every legal action")
        if not self.planner_used and (
            self.selected_action_fingerprint != self.base_action_fingerprint
        ):
            raise ValueError("fallback/control action must preserve the base action")
        if self.gpu_batch_capacity == 0 and self.gpu_rows != 0:
            raise ValueError("GPU rows require a declared batch capacity")
        if self.gpu_batch_capacity and self.gpu_rows > self.gpu_batch_capacity:
            raise ValueError("GPU rows exceed the declared batch capacity")
        if self.oracle_compatible:
            if self.oracle_exclusion_reason is not None:
                raise ValueError("compatible oracle rows cannot have exclusion reasons")
        elif not self.oracle_exclusion_reason:
            raise ValueError("oracle incompatibility requires an explicit reason")
        has_oracle = self.served_action_regret is not None
        required_oracle_values = (
            self.base_action_regret,
            self.served_epsilon_optimal,
            self.oracle_latency_ms,
        )
        if has_oracle and any(value is None for value in required_oracle_values):
            raise ValueError("base/served regret and oracle timing must be paired")
        if not has_oracle and any(
            value is not None for value in required_oracle_values
        ):
            raise ValueError("partial oracle quality evidence is invalid")
        has_candidate_quality = self.candidate_best_regret is not None
        if (self.candidate_epsilon_recall is not None) != has_candidate_quality:
            raise ValueError("candidate regret and epsilon recall must be paired")
        if has_candidate_quality and not self.planner_used:
            raise ValueError("candidate regret requires served planner evidence")
        if not self.oracle_compatible and (has_oracle or has_candidate_quality):
            raise ValueError("oracle-incompatible rows cannot claim exact quality")
        if has_oracle:
            self._validate_complete_oracle(has_candidate_quality)
        elif (
            self.oracle_executed
            or self.oracle_provenance_fingerprint is not None
            or self.oracle_candidate_count
            or self.oracle_scenario_count
            or self.oracle_engine_transitions
            or self.oracle_scenario_grid_complete
            or self.oracle_rules_exact
        ):
            raise ValueError("excluded oracle rows cannot claim reference work")
        return self

    def _validate_complete_oracle(self, has_candidate_quality: bool) -> None:
        if (
            self.oracle_candidate_count != self.legal_action_count
            or self.oracle_scenario_count <= 0
            or self.oracle_engine_transitions <= 0
            or not self.oracle_scenario_grid_complete
            or not self.oracle_rules_exact
            or self.oracle_provenance_fingerprint is None
            or self.oracle_latency_ms is None
        ):
            raise ValueError("oracle quality requires measured full-support work")
        if self.oracle_executed != (self.oracle_latency_ms > 0.0):
            raise ValueError(
                "oracle execution must carry time; reused evidence must not"
            )
        if self.planner_used and not has_candidate_quality:
            raise ValueError("served planner oracle rows require candidate regret")
        assert self.base_action_regret is not None
        assert self.served_action_regret is not None
        if not self.planner_used and not math.isclose(
            self.served_action_regret,
            self.base_action_regret,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("fallback/control served regret must equal base regret")
        if (
            self.candidate_best_regret is not None
            and self.candidate_best_regret > self.served_action_regret + 1e-9
        ):
            raise ValueError("candidate-best regret cannot exceed served regret")


class PlannerProfileRunRecord(BaseModel):
    """One point's measured rates, identities, and bounded resource usage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    campaign_id: str
    point_id: str
    budget_id: str
    environment: PlannerProfileEnvironment
    planner_enabled: bool
    checkpoint_sha256: str
    decision_corpus_sha256: str
    workload_fingerprint: str
    machine_fingerprint: str
    model_fingerprint: str
    runtime_fingerprint: str
    planner_fingerprint: str
    controller_fingerprint: str
    constructor_fingerprint: str
    scorer_fingerprint: str
    tensor_schema_fingerprint: str
    native_abi_fingerprint: str
    native_schema_fingerprint: str
    native_library_fingerprint: str
    packaged_archive_fingerprint: str | None = None
    packaged_required_files_fingerprint: str | None = None
    policy_version: int = Field(ge=0)
    proposal_version: int = Field(ge=0)
    decisions: int = Field(ge=0)
    valid_planner_evidence_decisions: int = Field(ge=0)
    learner_warmup_updates: int = Field(ge=0)
    learner_updates: int = Field(ge=0)
    learner_optimizer_steps: int = Field(ge=0)
    learner_kernel_rows: int = Field(ge=0)
    policy_runtime_agent_warmup_calls: int = Field(ge=0)
    policy_runtime_agent_calls: int = Field(ge=0)
    packaged_isolated_processes: int = Field(ge=0)
    act_time_replay_episodes: int = Field(ge=0)
    act_time_replay_decisions: int = Field(ge=0)
    act_time_replay_exhausted_episodes: int = Field(ge=0)
    act_time_replay_elapsed_seconds: float
    act_time_replay_min_remaining_seconds: float
    oracle_decisions: int = Field(ge=0)
    oracle_elapsed_seconds: float
    elapsed_seconds: float
    inference_decisions_per_second: float
    learner_kernel_rows_per_second: float
    valid_evidence_decisions_per_second: float
    checkpoint_cadence_seconds: float
    cpu_lane_utilization: float
    gil_utilization: float | None
    gil_measurement_unavailable_reason: str | None = None
    actor_idle_fraction: float
    peak_vram_bytes: int = Field(ge=0)
    peak_host_bytes: int = Field(ge=0)
    fallback_decisions: int = Field(ge=0)
    failed_decisions: int = Field(ge=0)
    deadline_exceeded_decisions: int = Field(ge=0)

    @field_validator(
        "checkpoint_sha256",
        "decision_corpus_sha256",
        "workload_fingerprint",
        "machine_fingerprint",
        "model_fingerprint",
        "runtime_fingerprint",
        "planner_fingerprint",
        "controller_fingerprint",
        "constructor_fingerprint",
        "scorer_fingerprint",
        "tensor_schema_fingerprint",
        "native_abi_fingerprint",
        "native_schema_fingerprint",
        "native_library_fingerprint",
    )
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("planner run identities must be SHA-256")
        return value

    @field_validator(
        "packaged_archive_fingerprint",
        "packaged_required_files_fingerprint",
    )
    @classmethod
    def optional_run_sha256(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("packaged artifact identities must be SHA-256")
        return value

    @field_validator(
        "oracle_elapsed_seconds",
        "act_time_replay_elapsed_seconds",
        "elapsed_seconds",
        "inference_decisions_per_second",
        "learner_kernel_rows_per_second",
        "valid_evidence_decisions_per_second",
        "checkpoint_cadence_seconds",
    )
    @classmethod
    def finite_rate(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("planner run rates must be finite and non-negative")
        return value

    @field_validator("cpu_lane_utilization", "actor_idle_fraction")
    @classmethod
    def fraction(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("planner utilization must be in [0, 1]")
        return value

    @field_validator("gil_utilization")
    @classmethod
    def optional_fraction(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
            raise ValueError("planner GIL utilization must be in [0, 1]")
        return value

    @model_validator(mode="after")
    def coherent_oracle_run(self) -> Self:
        if (self.gil_utilization is None) != bool(
            self.gil_measurement_unavailable_reason
        ):
            raise ValueError(
                "GIL utilization must be measured or carry an unavailable reason"
            )
        if not math.isfinite(self.act_time_replay_min_remaining_seconds):
            raise ValueError("ActTime remaining ledger must be finite")
        packaged_identities = (
            self.packaged_archive_fingerprint,
            self.packaged_required_files_fingerprint,
        )
        if self.environment == "packaged_cpu_acttime":
            if any(value is None for value in packaged_identities):
                raise ValueError("packaged runs require exact archive identities")
        elif any(value is not None for value in packaged_identities):
            raise ValueError("H200 runs cannot claim packaged archive identities")
        if self.oracle_decisions == 0 and self.oracle_elapsed_seconds != 0.0:
            raise ValueError("oracle time requires oracle decisions")
        if self.oracle_decisions > 0 and self.oracle_elapsed_seconds <= 0.0:
            raise ValueError("oracle decisions require separately reported time")
        if self.act_time_replay_exhausted_episodes > self.act_time_replay_episodes:
            raise ValueError("exhausted ActTime episodes exceed replay episodes")
        if self.act_time_replay_episodes == 0:
            if (
                self.act_time_replay_decisions
                or self.act_time_replay_elapsed_seconds
                or self.act_time_replay_min_remaining_seconds
            ):
                raise ValueError("ActTime metrics require ordered replay episodes")
        elif (
            self.act_time_replay_decisions <= 0
            or self.act_time_replay_elapsed_seconds <= 0.0
            or self.packaged_isolated_processes < self.act_time_replay_episodes
        ):
            raise ValueError("ActTime episodes require isolated measured agent work")
        elif self.act_time_replay_exhausted_episodes == 0 and (
            self.act_time_replay_min_remaining_seconds <= 0.0
        ):
            raise ValueError("non-exhausted ActTime workload requires positive reserve")
        elif self.act_time_replay_exhausted_episodes > 0 and (
            self.act_time_replay_min_remaining_seconds > 0.0
        ):
            raise ValueError("ActTime exhaustion requires a depleted seat ledger")
        return self


__all__ = ["PlannerProfileDecisionRecord", "PlannerProfileRunRecord"]
