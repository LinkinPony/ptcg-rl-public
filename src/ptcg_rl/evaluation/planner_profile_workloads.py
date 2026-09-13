"""Reusable workload schemas for the integrated planner profile."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.agent.runtime import ActTimeConfig
from ptcg_rl.belief.sampling import BeliefSamplerConfig
from ptcg_rl.context.belief import OpponentBeliefFeatureConfig
from ptcg_rl.engine.native_planning_session_pool_contract import (
    NativePlanningSessionPoolConfig,
)
from ptcg_rl.evaluation.planner_profile_package_config import PlannerPackageAssetId
from ptcg_rl.rl.planner_runtime_identity import ResolvedPlannerRuntimeConfig
from ptcg_rl.runtime.work_ledger import PlannerWorkLimits

PlannerProfileEnvironment = Literal["h200_mps", "packaged_cpu_acttime"]
PlannerDecisionShape = Literal[
    "direct",
    "subset",
    "ordered",
    "multi_prompt",
    "handoff",
    "engine_chance",
]
PlannerProfileOperation = Literal[
    "build_corpus",
    "build_package_assets",
    "run_campaign",
    "validate",
]

REQUIRED_PLANNER_PROFILE_SHAPES: frozenset[str] = frozenset(
    {
        "direct",
        "subset",
        "ordered",
        "multi_prompt",
        "handoff",
        "engine_chance",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PlannerProfileCorpusBuildConfig(BaseModel):
    """Streaming source and immutable output contract for the decision corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_globs: tuple[str, ...]
    chance_evidence_globs: tuple[str, ...]
    replay_root: Path
    replay_archive_root: Path
    output_path: Path
    manifest_path: Path
    rows_per_shape: int = Field(gt=0)
    scan_batch_size: int = Field(gt=0)
    seed: str
    compression: str = "zstd"

    @field_validator("source_globs", "chance_evidence_globs")
    @classmethod
    def nonempty_globs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("planner corpus globs must be non-empty")
        return value

    @field_validator("seed", "compression")
    @classmethod
    def nonempty_string(cls, value: str) -> str:
        canonical = value.strip()
        if not canonical:
            raise ValueError("planner corpus strings must be non-empty")
        return canonical

    @model_validator(mode="after")
    def distinct_outputs(self) -> Self:
        if self.output_path == self.manifest_path:
            raise ValueError("planner corpus and manifest paths must differ")
        return self


class PlannerBeliefWorkloadConfig(BaseModel):
    """Immutable deployment belief input used by every campaign point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factory: str
    sampler: BeliefSamplerConfig
    producer: OpponentBeliefFeatureConfig
    stochastic_seed: int

    @field_validator("factory")
    @classmethod
    def factory_path(cls, value: str) -> str:
        canonical = value.strip()
        if ":" not in canonical:
            raise ValueError("belief factory must be module:function")
        return canonical

    @model_validator(mode="after")
    def immutable_prior(self) -> Self:
        if self.sampler.prior_deck_signature_summary_path is None:
            raise ValueError("integrated profile requires a fixed belief prior")
        if self.sampler.prior_deck_signature_summary_sha256 is None:
            raise ValueError("integrated profile belief prior requires SHA-256")
        if not self.producer.enabled:
            raise ValueError("integrated profile requires public posterior features")
        if self.producer.deck_signature_summary_path is None:
            raise ValueError("integrated profile producer requires a fixed prior")
        if self.producer.deck_signature_summary_sha256 is None:
            raise ValueError("integrated profile producer prior requires SHA-256")
        if (
            self.sampler.prior_deck_signature_summary_path
            != self.producer.deck_signature_summary_path
            or self.sampler.prior_deck_signature_summary_sha256
            != self.producer.deck_signature_summary_sha256
        ):
            raise ValueError("sampler and public producer must share one prior asset")
        return self


class PlannerLearnerWorkloadConfig(BaseModel):
    """Fixed synthetic learner-kernel contention work for H200 points."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factory: str
    updates: int = Field(gt=0)
    kernel_rows_per_update: int = Field(gt=0)
    warmup_updates: int = Field(ge=0)
    microbatch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    ppo_epochs: int = Field(gt=0)
    max_policy_age: int = Field(ge=0)
    engine_teacher_coefficient: float
    factual_effect_coefficient: float = Field(ge=0.0)
    factual_successor_coefficient: float = Field(ge=0.0)
    root_information_value_coefficient: float = Field(ge=0.0)
    candidate_rerank_coefficient: float = Field(ge=0.0)
    proposal_distillation_coefficient: float = Field(ge=0.0)
    anchor_coefficient: float = Field(ge=0.0)
    planner_target_ratio_clip: float = Field(ge=1.0)
    anchor_checkpoint_path: Path
    anchor_checkpoint_sha256: str

    @field_validator("factory")
    @classmethod
    def factory_path(cls, value: str) -> str:
        canonical = value.strip()
        if ":" not in canonical:
            raise ValueError("learner factory must be module:function")
        return canonical

    @field_validator("anchor_checkpoint_sha256")
    @classmethod
    def anchor_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None or value == "0" * 64:
            raise ValueError("learner anchor fingerprint must be SHA-256")
        return value

    @field_validator("engine_teacher_coefficient")
    @classmethod
    def disabled_engine_teacher(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("profile learner engine-teacher coefficient must be zero")
        return value

    @model_validator(mode="after")
    def exact_update_geometry(self) -> Self:
        if self.kernel_rows_per_update != (
            self.microbatch_size * self.gradient_accumulation_steps
        ):
            raise ValueError(
                "learner kernel rows/update must equal microbatch times accumulation"
            )
        return self


class PlannerPackagedWorkloadConfig(BaseModel):
    """Kaggle-like packaged PolicyRuntimeAgent execution contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    submission_profile: str
    package_asset_id: PlannerPackageAssetId
    planner_enabled_by_default: bool
    device: Literal["cpu"]
    act_time: ActTimeConfig
    warmup_decisions: int = Field(ge=0)
    isolate_working_directory: Literal[True] = True
    remove_agent_directory_from_sys_path: Literal[True] = True

    @field_validator("submission_profile")
    @classmethod
    def immutable_submission_profile(cls, value: str) -> str:
        canonical = value.strip()
        if not canonical or "latest" in canonical.lower():
            raise ValueError("submission profile must be immutable and non-empty")
        return canonical

    @model_validator(mode="after")
    def shared_v5_only(self) -> Self:
        expected_planner_default = self.package_asset_id != "control"
        if self.planner_enabled_by_default != expected_planner_default:
            raise ValueError("only the control package may default the planner off")
        if self.act_time.search.enabled:
            raise ValueError("packaged profile must disable the legacy runtime probe")
        if self.act_time.search.macro.mode != "disabled":
            raise ValueError("packaged profile must disable legacy macro search")
        return self


class PlannerActTimeReplayAsset(BaseModel):
    """One immutable ordered Kaggle episode replay used for ActTime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    sha256: str
    active_callbacks_by_seat: tuple[int, int]

    @field_validator("active_callbacks_by_seat")
    @classmethod
    def positive_callback_counts(cls, value: tuple[int, int]) -> tuple[int, int]:
        if any(count <= 0 for count in value):
            raise ValueError("ActTime replay callback counts must be positive")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None or value == "0" * 64:
            raise ValueError("ActTime replay fingerprints must be SHA-256")
        return value


class PlannerActTimeReplayConfig(BaseModel):
    """Ordered episode workload with an isolated per-seat overage ledger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assets: tuple[PlannerActTimeReplayAsset, ...]
    seats: tuple[Literal[0, 1], ...] = (0, 1)
    initial_overage_seconds: float
    isolated_subprocess_per_seat_episode: Literal[True] = True
    bind_recorded_deck_registration: Literal[True] = True
    runtime_stress_only: Literal[True] = True

    @field_validator("initial_overage_seconds")
    @classmethod
    def positive_finite_overage(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("ActTime overage must be finite and positive")
        return value

    @model_validator(mode="after")
    def complete_assets(self) -> Self:
        if not self.assets:
            raise ValueError("ActTime profile requires ordered episode replays")
        paths = tuple(asset.path for asset in self.assets)
        if len(set(paths)) != len(paths):
            raise ValueError("ActTime replay paths must be unique")
        if not self.seats or len(set(self.seats)) != len(self.seats):
            raise ValueError("ActTime profile seats must be non-empty and unique")
        return self


class PlannerProfileRuntimeConfig(BaseModel):
    """One exact semantic runtime plus environment-specific execution load."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    planner: ResolvedPlannerRuntimeConfig
    belief: PlannerBeliefWorkloadConfig
    learner: PlannerLearnerWorkloadConfig | None = None
    packaged: PlannerPackagedWorkloadConfig | None = None


class PlannerProfilePointConfig(BaseModel):
    """One control or planner point executed against the same fixed corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    point_id: str
    budget_id: str
    environment: PlannerProfileEnvironment
    planner_enabled: bool
    runtime_id: str
    decision_repetitions: int = Field(gt=0)
    request_batch_size: int = Field(gt=0)

    @field_validator("point_id", "budget_id", "runtime_id")
    @classmethod
    def immutable_label(cls, value: str) -> str:
        canonical = value.strip()
        if not canonical or "latest" in canonical.lower():
            raise ValueError("profile point labels must be immutable and non-empty")
        return canonical


class PlannerProfileResourceContract(BaseModel):
    """Only physical memory bounds whose breach invalidates deployment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    packaged_hard_peak_host_bytes: int = Field(gt=0)
    h200_hard_peak_vram_bytes: int = Field(gt=0)
    h200_hard_peak_host_bytes: int = Field(gt=0)


class PlannerProfileOracleConfig(BaseModel):
    """Separately timed full-support reference for direct regret measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_legal_actions: int = Field(gt=0)
    epsilon_regret: float
    timeout_seconds: float
    work_limits: PlannerWorkLimits
    session_pool: NativePlanningSessionPoolConfig
    root_value_microbatch_rows: int = Field(gt=0)

    @field_validator("epsilon_regret")
    @classmethod
    def nonnegative_finite_epsilon(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("oracle epsilon regret must be finite and non-negative")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def positive_finite_timeout(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("oracle timeout must be finite and positive")
        return value

    @model_validator(mode="after")
    def exhaustive_capacity(self) -> Self:
        if self.work_limits.max_candidates < self.max_legal_actions:
            raise ValueError("oracle work limits cannot truncate legal support")
        if (
            self.work_limits.max_native_transitions_per_call
            != self.session_pool.max_transitions_per_call
        ):
            raise ValueError("oracle ledger and native pool chunk caps must agree")
        return self


__all__ = [
    "PlannerActTimeReplayAsset",
    "PlannerActTimeReplayConfig",
    "PlannerBeliefWorkloadConfig",
    "PlannerDecisionShape",
    "PlannerLearnerWorkloadConfig",
    "PlannerPackagedWorkloadConfig",
    "PlannerProfileCorpusBuildConfig",
    "PlannerProfileEnvironment",
    "PlannerProfileOperation",
    "PlannerProfileOracleConfig",
    "PlannerProfilePointConfig",
    "PlannerProfileResourceContract",
    "PlannerProfileRuntimeConfig",
    "REQUIRED_PLANNER_PROFILE_SHAPES",
]
