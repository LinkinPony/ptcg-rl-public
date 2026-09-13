"""Strict configuration and persistence contracts for policy iteration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.belief.sampling import BeliefSamplerConfig

BehaviorKind = Literal["policy_sample", "improvement"]
ProposalSource = Literal[
    "exhaustive",
    "behavior",
    "structural",
    "exploration",
]
ConsequenceKind = Literal[
    "terminal",
    "same_seat",
    "handoff",
    "infrastructure_error",
]


class CandidateProposalConfig(BaseModel):
    """Known-density generic complete-action proposal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exhaustive_action_cap: int = 64
    max_candidates: int = 10
    behavior_samples: int = 6
    structural_samples: int = 3
    exploration_samples: int = 1
    behavior_weight: float = 0.60
    structural_weight: float = 0.30
    exploration_weight: float = 0.10

    @field_validator(
        "exhaustive_action_cap",
        "max_candidates",
        "behavior_samples",
        "structural_samples",
        "exploration_samples",
    )
    @classmethod
    def positive_counts(cls, value: int) -> int:
        """Require usable candidate geometry."""
        if value <= 0:
            raise ValueError("candidate proposal counts must be positive")
        return value

    @field_validator(
        "behavior_weight",
        "structural_weight",
        "exploration_weight",
    )
    @classmethod
    def nonnegative_weight(cls, value: float) -> float:
        """Require finite mixture weights."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("candidate proposal weights must be non-negative")
        return value

    @model_validator(mode="after")
    def valid_mixture(self) -> Self:
        """Keep a normalized proposal with nonzero global support."""
        total = self.behavior_weight + self.structural_weight + self.exploration_weight
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
            raise ValueError("candidate proposal weights must sum to one")
        if self.exploration_weight <= 0.0:
            raise ValueError("exploration proposal weight must be positive")
        sample_count = (
            self.behavior_samples + self.structural_samples + self.exploration_samples
        )
        if self.max_candidates < sample_count:
            raise ValueError(
                "max_candidates cannot be below the declared source sample count"
            )
        declared = (
            self.behavior_weight,
            self.structural_weight,
            self.exploration_weight,
        )
        empirical = (
            self.behavior_samples / sample_count,
            self.structural_samples / sample_count,
            self.exploration_samples / sample_count,
        )
        if any(
            not math.isclose(weight, frequency, rel_tol=0.0, abs_tol=1.0e-8)
            for weight, frequency in zip(declared, empirical, strict=True)
        ):
            raise ValueError(
                "proposal weights must equal declared source sample frequencies"
            )
        return self


class NativeReanalysisConfig(BaseModel):
    """Bounded CPU engine-worker resources for one-action reanalysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_count: int = 4
    root_queue_capacity: int = 512
    job_queue_capacity: int = 128
    result_queue_capacity: int = 128
    roots_per_job: int = 8
    max_cells_per_root: int = 256
    max_engine_steps_per_root: int = 4_096
    max_forced_steps: int = 64
    max_observation_bytes: int = 1 << 24
    result_barrier_timeout_seconds: float = 300.0

    @field_validator(
        "worker_count",
        "root_queue_capacity",
        "job_queue_capacity",
        "result_queue_capacity",
        "roots_per_job",
        "max_cells_per_root",
        "max_engine_steps_per_root",
        "max_observation_bytes",
    )
    @classmethod
    def positive_capacity(cls, value: int) -> int:
        """Reject empty worker and wire capacities."""
        if value <= 0:
            raise ValueError("native reanalysis capacities must be positive")
        return value

    @field_validator("max_forced_steps")
    @classmethod
    def nonnegative_forced_steps(cls, value: int) -> int:
        """Allow zero forced closure while rejecting negative caps."""
        if value < 0:
            raise ValueError("max_forced_steps must be non-negative")
        return value

    @field_validator("result_barrier_timeout_seconds")
    @classmethod
    def positive_result_barrier_timeout(cls, value: float) -> float:
        """Fail a stuck native engine instead of hanging a checkpoint forever."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("result barrier timeout must be finite and positive")
        return value


class ReplayShardConfig(BaseModel):
    """Atomic compact-shard persistence and bounded replay retention."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_dir: Path = Path("reanalysis_replay")
    rows_per_shard: int = 4_096
    max_committed_shards: int = 256
    compression: bool = True
    async_writes: bool = False
    schema_version: int = 1

    @field_validator("rows_per_shard", "max_committed_shards", "schema_version")
    @classmethod
    def positive_values(cls, value: int) -> int:
        """Require positive shard geometry and schema identity."""
        if value <= 0:
            raise ValueError("replay shard values must be positive")
        return value

    @field_validator("relative_dir")
    @classmethod
    def relative_storage(cls, value: Path) -> Path:
        """Keep run-local evidence beneath its training output directory."""
        if value.is_absolute() or ".." in value.parts or value == Path("."):
            raise ValueError("replay relative_dir must be a safe relative path")
        return value


class TargetNetworkConfig(BaseModel):
    """Frozen-target update schedule used by successor bootstrapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    update_interval: int = 8
    polyak: float = 1.0

    @field_validator("update_interval")
    @classmethod
    def positive_interval(cls, value: int) -> int:
        """Require an explicit positive update cadence."""
        if value <= 0:
            raise ValueError("target update_interval must be positive")
        return value

    @field_validator("polyak")
    @classmethod
    def valid_polyak(cls, value: float) -> float:
        """Require a valid hard/soft target interpolation coefficient."""
        if not math.isfinite(value) or value <= 0.0 or value > 1.0:
            raise ValueError("target polyak must be in (0, 1]")
        return value


class RetraceConfig(BaseModel):
    """Distributional Retrace recurrence for real behavior sequences."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gamma: float = 1.0
    lambda_: float = 1.0
    rho_clip: float = 1.0
    c_clip: float = 1.0
    horizon: int = 64

    @field_validator("gamma")
    @classmethod
    def undiscounted_outcome(cls, value: float) -> float:
        """Categorical terminal W/D/L is an undiscounted outcome contract."""
        if not math.isfinite(value) or not math.isclose(value, 1.0):
            raise ValueError("categorical W/D/L Retrace requires gamma=1")
        return value

    @field_validator("lambda_")
    @classmethod
    def unit_interval(cls, value: float) -> float:
        """Require a finite trace coefficient."""
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("Retrace lambda must be in [0, 1]")
        return value

    @field_validator("rho_clip", "c_clip")
    @classmethod
    def positive_clip(cls, value: float) -> float:
        """Require positive finite importance caps."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Retrace importance clips must be positive")
        return value

    @field_validator("horizon")
    @classmethod
    def positive_horizon(cls, value: int) -> int:
        """Require at least one real transition."""
        if value <= 0:
            raise ValueError("Retrace horizon must be positive")
        return value


class CmpoConfig(BaseModel):
    """Proposal-corrected clipped MPO-style policy improvement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    advantage_transform: Literal[
        "per_root_standardized",
        "centered_wdl",
    ] = "per_root_standardized"
    temperature: float = 0.25
    advantage_clip: float = 4.0
    prior_exponent: float = 1.0
    min_policy_probability: float = 1.0e-8
    normalization_epsilon: float = 1.0e-6

    @field_validator(
        "temperature",
        "advantage_clip",
        "prior_exponent",
        "min_policy_probability",
        "normalization_epsilon",
    )
    @classmethod
    def positive_finite(cls, value: float) -> float:
        """Reject degenerate improvement distributions."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("CMPO parameters must be finite and positive")
        return value


class PolicyIterationLossConfig(BaseModel):
    """Explicit weights for independent learner lanes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state_wdl: float = 0.25
    retrace_wdl: float = 1.0
    counterfactual_wdl: float = 1.0
    cmpo: float = 0.25

    @field_validator("state_wdl", "retrace_wdl", "counterfactual_wdl", "cmpo")
    @classmethod
    def nonnegative_finite(cls, value: float) -> float:
        """Allow an explicit zero while rejecting implicit invalid weights."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("policy-iteration loss weights must be non-negative")
        return value

    @model_validator(mode="after")
    def retains_required_lanes(self) -> Self:
        """The selected design requires Q grounding and amortization."""
        if self.retrace_wdl <= 0.0 or self.counterfactual_wdl <= 0.0:
            raise ValueError("real and counterfactual W/D/L losses are required")
        if self.cmpo <= 0.0:
            raise ValueError("CMPO policy amortization is required")
        return self


class PolicyIterationLearnerConfig(BaseModel):
    """Fixed learner work budgets for real and counterfactual value lanes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    roots_per_iteration: int = 128
    proposal_batch_size: int = 32
    real_retrace_enabled: bool = True
    real_max_decisions: int = 4_096
    real_microbatch_size: int = 512
    counterfactual_roots_per_update: int = 32
    counterfactual_updates_per_iteration: int = 1
    result_drain_limit: int = 256
    max_root_policy_lag: int = 0
    queue_drain_idle_timeout_ms: int = 0

    @field_validator(
        "roots_per_iteration",
        "proposal_batch_size",
        "real_max_decisions",
        "real_microbatch_size",
        "counterfactual_roots_per_update",
        "counterfactual_updates_per_iteration",
        "result_drain_limit",
    )
    @classmethod
    def positive_budget(cls, value: int) -> int:
        """Require every selected learner lane to have bounded positive work."""
        if value <= 0:
            raise ValueError("policy-iteration learner budgets must be positive")
        return value

    @field_validator("max_root_policy_lag", "queue_drain_idle_timeout_ms")
    @classmethod
    def nonnegative_async_queue_setting(cls, value: int) -> int:
        """Allow exact-version roots and compatibility-mode nonblocking drains."""
        if value < 0:
            raise ValueError(
                "policy-iteration async queue settings must be non-negative"
            )
        return value


class ImprovementActorConfig(BaseModel):
    """Fixed, non-adaptive actor allocation for real improved trajectories."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_count: int = 2

    @field_validator("actor_count")
    @classmethod
    def nonnegative_actor_count(cls, value: int) -> int:
        """Allow counterfactual-only improvement without online Q actors."""
        if value < 0:
            raise ValueError("improvement actor_count must be non-negative")
        return value


class DeploymentStripConfig(BaseModel):
    """Training-component stripping contract for the first release bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strip_action_value: bool = True
    strip_reanalysis_runtime: bool = True
    runtime_q_rerank: bool = False

    @model_validator(mode="after")
    def direct_actor_only(self) -> Self:
        """Keep this migration separate from a runtime-reranker decision."""
        if self.runtime_q_rerank:
            raise ValueError("the first deployment contract forbids runtime Q rerank")
        if not self.strip_reanalysis_runtime:
            raise ValueError("release bundles must strip engine reanalysis services")
        return self


class AmortizedPolicyIterationConfig(BaseModel):
    """Top-level Hydra schema for the selected training architecture."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    root_sample_probability: float = 0.10
    belief_worlds: int = 8
    candidate_proposal: CandidateProposalConfig = CandidateProposalConfig()
    belief_sampler: BeliefSamplerConfig = BeliefSamplerConfig()
    native: NativeReanalysisConfig = NativeReanalysisConfig()
    replay: ReplayShardConfig = ReplayShardConfig()
    target_network: TargetNetworkConfig = TargetNetworkConfig()
    retrace: RetraceConfig = RetraceConfig()
    cmpo: CmpoConfig = CmpoConfig()
    losses: PolicyIterationLossConfig = PolicyIterationLossConfig()
    learner: PolicyIterationLearnerConfig = PolicyIterationLearnerConfig()
    improvement: ImprovementActorConfig = ImprovementActorConfig()
    deployment: DeploymentStripConfig = DeploymentStripConfig()

    @field_validator("root_sample_probability")
    @classmethod
    def valid_sample_probability(cls, value: float) -> float:
        """Require a fixed nonzero auxiliary sampling contract."""
        if not math.isfinite(value) or value <= 0.0 or value > 1.0:
            raise ValueError("root_sample_probability must be in (0, 1]")
        return value

    @field_validator("belief_worlds")
    @classmethod
    def positive_worlds(cls, value: int) -> int:
        """Require at least one shared belief particle."""
        if value <= 0:
            raise ValueError("belief_worlds must be positive")
        return value

    @model_validator(mode="after")
    def valid_enabled_contract(self) -> Self:
        """Bind enabled reanalysis to a fingerprinted strict belief model."""
        if not self.enabled:
            return self
        sampler = self.belief_sampler
        if sampler.mode != "archetype" or not sampler.strict_own_deck_counts:
            raise ValueError(
                "enabled reanalysis requires strict archetype belief sampling"
            )
        if (
            sampler.prior_deck_signature_summary_path is None
            or sampler.prior_deck_signature_summary_sha256 is None
        ):
            raise ValueError("enabled reanalysis requires a fingerprinted belief prior")
        return self


__all__ = [
    "AmortizedPolicyIterationConfig",
    "BehaviorKind",
    "CandidateProposalConfig",
    "CmpoConfig",
    "ConsequenceKind",
    "DeploymentStripConfig",
    "ImprovementActorConfig",
    "NativeReanalysisConfig",
    "PolicyIterationLearnerConfig",
    "PolicyIterationLossConfig",
    "ProposalSource",
    "ReplayShardConfig",
    "RetraceConfig",
    "TargetNetworkConfig",
]
