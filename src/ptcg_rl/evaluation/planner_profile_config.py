"""Validated configuration for one integrated planner performance campaign."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.evaluation.planner_profile_package_config import (
    PlannerPackageAssetId,
    PlannerProfileModelIdentity,
    PlannerProfilePackageAssetsConfig,
)
from ptcg_rl.evaluation.planner_profile_workloads import (
    REQUIRED_PLANNER_PROFILE_SHAPES,
    PlannerActTimeReplayAsset,
    PlannerActTimeReplayConfig,
    PlannerBeliefWorkloadConfig,
    PlannerDecisionShape,
    PlannerLearnerWorkloadConfig,
    PlannerPackagedWorkloadConfig,
    PlannerProfileCorpusBuildConfig,
    PlannerProfileEnvironment,
    PlannerProfileOperation,
    PlannerProfileOracleConfig,
    PlannerProfilePointConfig,
    PlannerProfileResourceContract,
    PlannerProfileRuntimeConfig,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENVIRONMENTS: tuple[PlannerProfileEnvironment, ...] = (
    "h200_mps",
    "packaged_cpu_acttime",
)
_FIXED_POLICY_VERSION = 27_470
_FIXED_CHECKPOINT_FILENAME = "policy_v27470.pt"
_FIXED_CHECKPOINT_SHA256 = (
    "c3ca43fc516c71928146c63caad731cdd74d85d98c063ad695649e6c91b5d246"
)


class IntegratedPlannerProfileConfig(BaseModel):
    """Hydra-facing single-campaign configuration."""

    model_config = ConfigDict(extra="forbid")

    operation: PlannerProfileOperation
    campaign_id: str
    checkpoint_path: Path
    expected_checkpoint_sha256: str
    expected_model_fingerprint: str
    model_migration_seed: int = Field(ge=0)
    policy_version: int = Field(ge=0)
    proposal_version: int = Field(ge=0)
    native_library_path: Path
    expected_native_library_sha256: str
    expected_native_abi_fingerprint: str
    expected_native_schema_fingerprint: str
    corpus_build: PlannerProfileCorpusBuildConfig
    decision_corpus_path: Path
    expected_decision_corpus_sha256: str
    expected_decision_corpus_manifest_sha256: str
    package_assets: PlannerProfilePackageAssetsConfig
    runtime_profiles: Mapping[str, PlannerProfileRuntimeConfig]
    learner_kernel_runtime_id: str
    points: tuple[PlannerProfilePointConfig, ...]
    oracle: PlannerProfileOracleConfig
    act_time_replay: PlannerActTimeReplayConfig
    resource_contract: PlannerProfileResourceContract
    output_dir: Path
    rows_per_shard: int = Field(gt=0)
    compression: str = "zstd"
    require_h200: Literal[True] = True
    require_cuda_mps: Literal[True] = True
    formal_collection_num_concurrent_games_per_actor: int = Field(gt=0)
    production_backend_factory: str
    hydra: Mapping[str, Any] | None = None

    @field_validator("campaign_id", "compression", "learner_kernel_runtime_id")
    @classmethod
    def immutable_string(cls, value: str) -> str:
        canonical = value.strip()
        if not canonical or "latest" in canonical.lower():
            raise ValueError("campaign identities must be immutable and non-empty")
        return canonical

    @field_validator(
        "expected_checkpoint_sha256",
        "expected_model_fingerprint",
        "expected_native_library_sha256",
        "expected_native_abi_fingerprint",
        "expected_native_schema_fingerprint",
        "expected_decision_corpus_sha256",
        "expected_decision_corpus_manifest_sha256",
    )
    @classmethod
    def sha256_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None or value == "0" * 64:
            raise ValueError("profile fingerprints must be lowercase SHA-256")
        return value

    @field_validator("production_backend_factory")
    @classmethod
    def backend_factory_path(cls, value: str) -> str:
        canonical = value.strip()
        if ":" not in canonical:
            raise ValueError("production backend factory must be module:function")
        return canonical

    @model_validator(mode="after")
    def complete_campaign(self) -> Self:
        self._validate_fixed_checkpoint()
        if self.decision_corpus_path != self.corpus_build.output_path:
            raise ValueError("campaign and builder must name the same corpus artifact")
        if self.package_assets.source_checkpoint_sha256 != (
            self.expected_checkpoint_sha256
        ):
            raise ValueError("package source lineage differs from campaign checkpoint")
        if (
            self.operation in {"run_campaign", "validate"}
            and self.package_assets.expected_manifest_sha256 is None
        ):
            raise ValueError("validate/run require an immutable package manifest")
        runtime_ids = set(self.runtime_profiles)
        if not runtime_ids:
            raise ValueError("integrated profile requires runtime profiles")
        if self.learner_kernel_runtime_id not in runtime_ids:
            raise ValueError("learner kernel runtime id is not configured")
        kernel_runtime = self.runtime_profiles[self.learner_kernel_runtime_id]
        if kernel_runtime.learner is None or kernel_runtime.packaged is not None:
            raise ValueError("learner kernel runtime must be an H200 runtime")
        if any(
            runtime.learner is not None and runtime.learner != kernel_runtime.learner
            for runtime in self.runtime_profiles.values()
        ):
            raise ValueError("H200 points change the fixed learner kernel workload")
        point_ids = tuple(point.point_id for point in self.points)
        if not point_ids or len(set(point_ids)) != len(point_ids):
            raise ValueError("profile point ids must be non-empty and unique")
        for point in self.points:
            if point.runtime_id not in runtime_ids:
                raise ValueError(f"unknown point runtime_id: {point.runtime_id}")
            runtime = self.runtime_profiles[point.runtime_id]
            self._validate_point_runtime(point, runtime)
        if len({point.decision_repetitions for point in self.points}) != 1:
            raise ValueError(
                "every point must replay the same fixed corpus repetitions"
            )
        self._validate_shared_inputs()
        packaged_assets = {
            runtime.packaged.package_asset_id
            for runtime in self.runtime_profiles.values()
            if runtime.packaged is not None
        }
        if packaged_assets != set(self.package_assets.submission_profiles):
            raise ValueError("packaged runtimes differ from package asset matrix")
        self._validate_point_matrix()
        self._validate_environment_invariants()
        self._validate_paired_budgets()
        return self

    def runtime_for(
        self, point: PlannerProfilePointConfig
    ) -> PlannerProfileRuntimeConfig:
        """Return the exact validated runtime referenced by a point."""
        return self.runtime_profiles[point.runtime_id]

    def model_identity_for(
        self,
        point: PlannerProfilePointConfig,
    ) -> PlannerProfileModelIdentity:
        """Resolve the exact raw or exported serving identity for one point."""
        if point.environment == "h200_mps":
            return PlannerProfileModelIdentity(
                checkpoint_path=self.checkpoint_path,
                checkpoint_sha256=self.expected_checkpoint_sha256,
                model_fingerprint=self.expected_model_fingerprint,
                source_checkpoint_sha256=self.expected_checkpoint_sha256,
                policy_version=self.policy_version,
                proposal_version=self.proposal_version,
            )
        return PlannerProfileModelIdentity(
            checkpoint_path=self.package_assets.deployment_checkpoint_path,
            checkpoint_sha256=(
                self.package_assets.expected_deployment_checkpoint_sha256
            ),
            model_fingerprint=(
                self.package_assets.expected_deployment_model_fingerprint
            ),
            source_checkpoint_sha256=self.package_assets.source_checkpoint_sha256,
            policy_version=self.policy_version,
            proposal_version=self.proposal_version,
        )

    def _validate_fixed_checkpoint(self) -> None:
        """Reject retired warm starts and learner anchors for this campaign."""
        if self.policy_version != _FIXED_POLICY_VERSION:
            raise ValueError("integrated profile is fixed to policy version 27470")
        if self.checkpoint_path.name != _FIXED_CHECKPOINT_FILENAME:
            raise ValueError("integrated profile warm start must be policy_v27470.pt")
        if self.expected_checkpoint_sha256 != _FIXED_CHECKPOINT_SHA256:
            raise ValueError("integrated profile warm-start fingerprint is not v27470")

    def _validate_point_runtime(
        self,
        point: PlannerProfilePointConfig,
        runtime: PlannerProfileRuntimeConfig,
    ) -> None:
        if point.environment == "h200_mps":
            if runtime.learner is None or runtime.packaged is not None:
                raise ValueError("H200 runtime requires only a learner workload")
            if runtime.planner.batching.max_root_rows_per_request != (
                self.formal_collection_num_concurrent_games_per_actor
            ):
                raise ValueError(
                    "H200 root-request capacity must equal the selected formal "
                    "concurrent-games-per-actor topology"
                )
            if (
                runtime.learner.anchor_checkpoint_path != self.checkpoint_path
                or runtime.learner.anchor_checkpoint_sha256
                != self.expected_checkpoint_sha256
            ):
                raise ValueError(
                    "H200 learner anchor must be the fixed v27470 warm start"
                )
            if (
                runtime.planner.buffers.gpu_bytes_per_unique_leaf <= 0
                or runtime.planner.buffers.gpu_staging_bytes <= 0
            ):
                raise ValueError("H200 runtime requires fixed GPU staging")
        elif runtime.packaged is None or runtime.learner is not None:
            raise ValueError("packaged runtime requires only an agent workload")
        elif point.planner_enabled != runtime.packaged.planner_enabled_by_default:
            raise ValueError(
                "packaged point planner state differs from its immutable default"
            )
        elif (
            runtime.planner.buffers.gpu_bytes_per_unique_leaf != 0
            or runtime.planner.buffers.gpu_staging_bytes != 0
        ):
            raise ValueError("packaged CPU runtime cannot reserve GPU staging")
        if runtime.packaged is not None:
            self._validate_packaged_runtime(runtime)
        maximum_root_wave = (
            runtime.planner.batching.actor_count
            * runtime.planner.batching.max_root_rows_per_request
        )
        if point.request_batch_size > maximum_root_wave:
            raise ValueError("profile batch exceeds the production root wave")
        engine = runtime.planner.engine
        if (
            engine.library_fingerprint != self.expected_native_library_sha256
            or engine.native_abi_fingerprint != self.expected_native_abi_fingerprint
            or engine.native_schema_fingerprint
            != self.expected_native_schema_fingerprint
        ):
            raise ValueError("runtime engine identity differs from campaign inputs")
        if runtime.planner.scenario.belief_sampler_fingerprint == "0" * 64:
            raise ValueError("runtime belief identity cannot be a placeholder")
        if runtime.planner.tensorizer.belief_summary_dim != 0:
            raise ValueError(
                "profile public posterior already enters producer context; "
                "a tensorizer belief summary would duplicate card-id features"
            )

    def _validate_packaged_runtime(
        self,
        runtime: PlannerProfileRuntimeConfig,
    ) -> None:
        packaged = runtime.packaged
        if packaged is None:
            raise AssertionError("packaged runtime validator requires a workload")
        act_time = packaged.act_time
        if not math.isclose(
            act_time.default_remaining_overage_time,
            self.act_time_replay.initial_overage_seconds,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("packaged ActTime ledger differs from ordered replay")
        if act_time.checkpoint_path != self.package_assets.deployment_checkpoint_path:
            raise ValueError(
                "packaged ActTime checkpoint differs from deployment export"
            )
        if act_time.belief != runtime.belief.producer:
            raise ValueError("packaged ActTime belief producer differs from runtime")
        if act_time.search.sampler != runtime.belief.sampler:
            raise ValueError("packaged ActTime sampler differs from runtime")

    def _validate_shared_inputs(self) -> None:
        shared_inputs = {
            (
                json.dumps(
                    runtime.planner.request_costs.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                json.dumps(
                    runtime.belief.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for runtime in self.runtime_profiles.values()
        }
        if len(shared_inputs) != 1:
            raise ValueError(
                "all profile points must share request costs and belief semantics"
            )

    def _validate_point_matrix(self) -> None:
        by_environment = Counter(point.environment for point in self.points)
        if any(by_environment[environment] < 2 for environment in _ENVIRONMENTS):
            raise ValueError("each environment requires a control and planner point")
        for environment in _ENVIRONMENTS:
            environment_points = tuple(
                point for point in self.points if point.environment == environment
            )
            if sum(not point.planner_enabled for point in environment_points) != 1:
                raise ValueError(
                    "each environment requires exactly one planner-off control"
                )
            if not any(point.planner_enabled for point in environment_points):
                raise ValueError("each environment requires a planner-on point")

    def _validate_paired_budgets(self) -> None:
        h200_points = tuple(
            point
            for point in self.points
            if point.environment == "h200_mps" and point.planner_enabled
        )
        packaged_points = tuple(
            point
            for point in self.points
            if point.environment == "packaged_cpu_acttime" and point.planner_enabled
        )
        h200 = {point.budget_id: point for point in h200_points}
        packaged = {point.budget_id: point for point in packaged_points}
        if len(h200) != len(h200_points) or len(packaged) != len(packaged_points):
            raise ValueError(
                "planner budget ids must be unique within each environment"
            )
        if set(h200) != set(packaged):
            raise ValueError(
                "H200 and packaged profiles must compare identical budgets"
            )
        for budget_id in sorted(h200):
            left = self.runtime_for(h200[budget_id])
            right = self.runtime_for(packaged[budget_id])
            if _deployment_semantics(left) != _deployment_semantics(right):
                raise ValueError(
                    f"paired budget {budget_id} changes deployment semantics"
                )
        controls = {
            point.environment: point
            for point in self.points
            if not point.planner_enabled
        }
        if _deployment_semantics(
            self.runtime_for(controls["h200_mps"])
        ) != _deployment_semantics(self.runtime_for(controls["packaged_cpu_acttime"])):
            raise ValueError("paired planner-off controls change deployment semantics")

    def _validate_environment_invariants(self) -> None:
        """Allow candidate capacity, and only candidate capacity, to vary."""
        for environment in _ENVIRONMENTS:
            semantics = {
                _budget_invariant_semantics(self.runtime_for(point))
                for point in self.points
                if point.environment == environment
            }
            if len(semantics) != 1:
                raise ValueError(
                    f"profile budgets change non-budget semantics in {environment}"
                )


def _deployment_semantics(config: PlannerProfileRuntimeConfig) -> tuple[Any, ...]:
    """Return fields that must remain identical between train and serve."""
    planner = config.planner
    return (
        planner.controller_version,
        planner.planner_behavior,
        planner.proposal_search,
        planner.hierarchical_search,
        planner.scoring,
        planner.tensorizer,
        planner.scenario,
        planner.engine,
        planner.work_limits,
        planner.request_costs,
        config.belief,
    )


def _budget_invariant_semantics(config: PlannerProfileRuntimeConfig) -> str:
    """Canonicalize one environment after removing candidate-capacity fields."""
    payload = config.model_dump(mode="json")
    planner = payload["planner"]
    constructor = planner["planner_behavior"]["constructor"]
    constructor.pop("k_total")
    constructor.pop("expansion_quotas")
    planner["work_limits"].pop("max_candidates")
    packaged = payload.get("packaged")
    if isinstance(packaged, dict):
        packaged.pop("submission_profile")
        packaged.pop("package_asset_id")
        packaged.pop("planner_enabled_by_default")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


__all__ = [
    "IntegratedPlannerProfileConfig",
    "PlannerActTimeReplayAsset",
    "PlannerActTimeReplayConfig",
    "PlannerBeliefWorkloadConfig",
    "PlannerDecisionShape",
    "PlannerLearnerWorkloadConfig",
    "PlannerPackageAssetId",
    "PlannerPackagedWorkloadConfig",
    "PlannerProfileCorpusBuildConfig",
    "PlannerProfileEnvironment",
    "PlannerProfileModelIdentity",
    "PlannerProfileOperation",
    "PlannerProfileOracleConfig",
    "PlannerProfilePackageAssetsConfig",
    "PlannerProfilePointConfig",
    "PlannerProfileResourceContract",
    "PlannerProfileRuntimeConfig",
    "REQUIRED_PLANNER_PROFILE_SHAPES",
]
