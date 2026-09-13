"""Fresh/exact-resume single-H200 trainer for the clean stateless policy."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import multiprocessing
import os
import random
import shutil
import threading
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, ParamSpec, TypeVar

import numpy as np
import torch

from ptcg_rl.belief.public_catalog import (
    PublicDeckCatalog,
    PublicDeckCatalogManifest,
    load_public_deck_catalog,
)
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import PUBLIC_EVENT_SCHEMA_FINGERPRINT
from ptcg_rl.data.kaggle_deck.records import read_deck
from ptcg_rl.decks.identity import CanonicalDeck, canonicalize_deck
from ptcg_rl.decks.registry import (
    resolve_deck_expert_registry,
    resolve_deck_family_registry,
    validate_active_exact_strategy_routes,
)
from ptcg_rl.engine.native_training import (
    NATIVE_TRAINING_ABI_VERSION,
    resolve_native_training_library,
)
from ptcg_rl.engine.prospective_facts import ProspectiveEngineFactProducer
from ptcg_rl.model.input_schema import POLICY_INPUT_SCHEMA_FINGERPRINT
from ptcg_rl.model.sequence.config import (
    GENERALIST_SEQUENCE_CONTRACT_FINGERPRINT,
)
from ptcg_rl.model.simple_stateless import (
    SIMPLE_BELIEF_TARGET_SEMANTICS,
    SimpleStatelessModelConfig,
    SimpleStatelessPolicyValueNet,
    simple_stateless_parameter_report,
    validate_simple_stateless_parameter_report,
)
from ptcg_rl.model.simple_stateless.config import (
    uses_family_private_topology,
    uses_generalist_sequence,
)
from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload
from ptcg_rl.rl.learner_metric_history import (
    LearnerMetricRecord,
    NonBlockingLearnerMetricWriter,
)
from ptcg_rl.rl.model_compatibility import model_config_fingerprint
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.native_banked_route_collection import (
    collect_native_banked_assigned,
)
from ptcg_rl.rl.native_distributed.artifact import (
    NATIVE_BFLOAT16_ARTIFACT_SEMANTICS_FINGERPRINT,
    EncodedBfloat16RolloutArtifact,
)
from ptcg_rl.rl.native_distributed.catalog import MetadataPastSelfRouteCatalog
from ptcg_rl.rl.native_distributed.contracts import NativeRolloutWorkerIdentity
from ptcg_rl.rl.native_distributed.training_backend import (
    NativeDistributedTrainingBackend,
)
from ptcg_rl.rl.native_historical_inference import NativeHistoricalPolicyPool
from ptcg_rl.rl.native_policy_bank import NativePolicyCudaStreamPool
from ptcg_rl.rl.native_policy_inference import NativePolicyInferenceExecutor
from ptcg_rl.rl.native_process_collection import (
    NativeProcessCollector,
    NativeProcessWorkerSpec,
)
from ptcg_rl.rl.native_scripted_mixed75 import NativeMixed75Policy
from ptcg_rl.rl.native_scripted_policy import (
    NativePublicScriptedPolicy,
    NativeScriptedPolicy,
)
from ptcg_rl.rl.native_stateless_collection import NativeStatelessCollector
from ptcg_rl.rl.performance import TrainingPerformanceReporter
from ptcg_rl.rl.performance_state import PerformanceReporterConfig
from ptcg_rl.rl.policy_inputs import (
    SIMPLE_STATELESS_WRAPPER_RUNTIME_FINGERPRINT,
    PolicyInputContract,
    simple_stateless_input_contract,
)
from ptcg_rl.rl.scripted_manifest import (
    ResolvedScriptedOpponent,
    builtin_scripted_implementations,
    load_scripted_manifest,
    resolve_scripted_manifest,
)
from ptcg_rl.rl.sequence_actor import GeneralistSequenceActorPolicy
from ptcg_rl.rl.sequence_context_transition import (
    StatelessSequenceContextTransitionPlan,
    build_stateless_sequence_context_transition_plan,
)
from ptcg_rl.rl.stateless_actor import SimpleStatelessActorPolicy
from ptcg_rl.rl.stateless_array_replay import (
    StatelessArrayOptimizerWindow,
    prepare_stateless_array_optimizer_window,
)
from ptcg_rl.rl.stateless_bc_overlay import (
    StatelessBcOverlayAuditReport,
    StatelessBcOverlayPlan,
    prepare_stateless_bc_overlay,
    transplant_stateless_bc_overlay_optimizer,
)
from ptcg_rl.rl.stateless_checkpoint import (
    AsyncStatelessCheckpointPairPublisher,
    LoadedStatelessCheckpointPair,
    PublishedStatelessCheckpointPair,
    StatelessCheckpointPair,
    StatelessPolicyIdentity,
    StatelessSettledTrainingProgress,
    StatelessStalenessState,
    StatelessStartupProvenance,
    StatelessStartupReport,
    load_stateless_checkpoint_pair,
    prune_stateless_checkpoint_pairs,
    publish_stateless_checkpoint_pair,
    publish_stateless_policy_checkpoint,
    publish_verified_stateless_policy_checkpoint,
    reconcile_stateless_resume_artifacts,
)
from ptcg_rl.rl.stateless_collection import (
    StatelessAssignedGame,
    StatelessCollectionReport,
    StatelessCollectionResult,
    StatelessEngineCollector,
    StatelessGameOutcome,
    assign_stateless_games,
    cancel_stateless_assignments,
    commit_stateless_outcomes,
)
from ptcg_rl.rl.stateless_curriculum import (
    DurablePolicyArtifact,
    PfspMember,
    ScriptedCurriculumBundle,
    StatelessCurriculumConfig,
    StatelessCurriculumController,
    StatelessCurriculumLaneCoverageRebaseAudit,
    StatelessCurriculumState,
    VerifiedPolicyPublication,
    historical_anchor_member,
    load_external_stateless_curriculum_state,
    prune_external_stateless_curriculum_states,
    publish_external_stateless_curriculum_state,
    rebase_settled_stateless_curriculum_lane_coverage,
    stateless_curriculum_state_fingerprint,
)
from ptcg_rl.rl.stateless_deck_balance import (
    DeckTargetShare,
    StatelessDeckBalanceConfig,
    StatelessDeckBalanceConfigValue,
    StatelessDeckBalanceSampler,
    StatelessDeckBalanceState,
    StatelessDeckBalanceTransitionAudit,
    StatelessDynamicDeckBalanceConfig,
    StatelessWeightedDeckBalanceConfig,
    migrate_settled_deck_balance_to_weighted,
)
from ptcg_rl.rl.stateless_export import load_fixed_deck_checkpoint
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity
from ptcg_rl.rl.stateless_fragment_io import (
    CompactFragmentManifest,
    CompactFragmentPart,
    CompactFragmentShardWriter,
)
from ptcg_rl.rl.stateless_hybrid import (
    merge_hybrid_collection_results,
    partition_hybrid_assignments,
)
from ptcg_rl.rl.stateless_learner import (
    SimpleStatelessLearner,
    SimpleStatelessLearnerUpdate,
)
from ptcg_rl.rl.stateless_opponent_pool_v2 import (
    BoundOpponentPoolWindow,
    BoundOpponentQuotaWindow,
    OpponentPoolCheckpointState,
    StatelessOpponentPoolAdaptiveState,
    StatelessOpponentPoolLineageState,
    StatelessOpponentPoolV2,
    assign_stateless_games_v2,
)
from ptcg_rl.rl.stateless_opponents import (
    HistoricalPolicyPool,
    PastSelfPolicyPool,
)
from ptcg_rl.rl.stateless_parallel import (
    StatelessParallelCollector,
    StatelessParallelWorkerSpec,
    cleanup_parallel_collection_parts,
)
from ptcg_rl.rl.stateless_performance import (
    stateless_opponent_strata,
    stateless_performance_outcomes,
)
from ptcg_rl.rl.stateless_ppo import SimpleStatelessPpoConfig
from ptcg_rl.rl.stateless_private_optimizer import (
    StatelessHybridOptimizerTransitionAudit,
    StatelessPrivateOptimizerTransitionAudit,
    retarget_private_optimizer_learning_rate,
    transplant_full_optimizer_to_private,
    transplant_private_optimizer_to_hybrid,
)
from ptcg_rl.rl.stateless_quota_assignments import (
    StatelessQuotaAssignmentPlan,
    plan_stateless_assignment_quotas,
)
from ptcg_rl.rl.stateless_registry_transition import (
    StatelessRegistryTransitionPlan,
    build_stateless_registry_transition_plan,
    migrate_stateless_deck_balance_state,
    migrate_stateless_registry_model,
    source_weighted_balance_for_registry_transition,
    transplant_stateless_registry_optimizer,
)
from ptcg_rl.rl.stateless_replay import prepare_stateless_optimizer_window
from ptcg_rl.rl.stateless_status_writer import (
    NonBlockingStatelessStatusWriter,
    StatelessStatusSnapshot,
)
from ptcg_rl.rl.stateless_supervised_startup import (
    StatelessSupervisedStartupPlan,
    prepare_stateless_supervised_startup,
)
from ptcg_rl.rl.stateless_topology_transition import (
    StatelessTopologyModelAudit,
    StatelessTopologyOptimizerAudit,
    StatelessTopologyTransitionAuditReport,
    StatelessTopologyTransitionPlan,
    build_stateless_topology_transition_plan,
    migrate_stateless_topology_model,
    transplant_stateless_topology_optimizer,
    validate_stateless_topology_source_pair,
)
from ptcg_rl.rl.stateless_training_config import (
    SimpleStatelessTrainingConfig,
    StatelessAnchorTransitionConfig,
    StatelessCurriculumLaneCoverageRebaseConfig,
    StatelessDeckBalanceTransitionConfig,
    StatelessDynamicDeckAllocationConfig,
    StatelessOptimizerScopeConfig,
    StatelessOptimizerScopeTransitionConfig,
    StatelessPublicCatalogTransitionConfig,
    StatelessResumeConfig,
)
from ptcg_rl.training.run_config import resolve_training_output_dir
from ptcg_rl.training.source_identity import (
    bind_run_source_identity,
    resolve_training_source_identity,
    source_adoption_requested,
)

_LOGGER = logging.getLogger(__name__)
_P = ParamSpec("_P")
_BackgroundValue = TypeVar("_BackgroundValue")
_ForegroundValue = TypeVar("_ForegroundValue")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RESOLVED_CONFIG_DOMAIN = b"ptcg-rl/simple-stateless-resolved-training/v1\x00"
_DISTRIBUTED_RESOLVED_CONFIG_DOMAIN = (
    b"ptcg-rl/simple-stateless-distributed-training/v2\x00"
)
_ROSTER_DOMAIN = b"ptcg-rl/simple-stateless-training-roster/v1\x00"
_PINNED_DOMAIN = b"ptcg-rl/simple-stateless-pinned-manifest/v1\x00"
_BELIEF_SEMANTICS_DOMAIN = b"ptcg-rl/simple-stateless-belief-semantics/v1\x00"
_SCRIPTED_SAMPLING_DOMAIN = b"ptcg-rl/stateless-scripted-sampling/v1\x00"
_IN_PROCESS_NATIVE_BACKENDS = frozenset(("native", "native_banked"))
_NATIVE_BACKENDS = _IN_PROCESS_NATIVE_BACKENDS | {"hybrid"}


@dataclass(frozen=True)
class _ResolvedResources:
    model_config: SimpleStatelessModelConfig
    catalog: PublicDeckCatalog
    catalog_manifest: PublicDeckCatalogManifest
    active_decks: dict[str, CanonicalDeck]
    active_deck_labels: dict[str, str]
    opponent_decks: dict[str, CanonicalDeck]
    route_input_contracts: dict[str, tuple[PublicDeckCatalog, str]]
    input_contract: PolicyInputContract
    curriculum_config: StatelessCurriculumConfig
    deck_balance_config: StatelessDeckBalanceConfigValue
    deck_target_shares: dict[str, float]
    policy_identity: StatelessPolicyIdentity
    resolved_config_fingerprint: str
    pinned_manifest_fingerprint: str
    scripted_manifest_fingerprint: str
    scripted_opponents: dict[str, ResolvedScriptedOpponent]
    training_roster_fingerprint: str
    native_library_path: Path | None
    native_library_sha256: str | None


@dataclass(frozen=True)
class _PreparedHybridCollection:
    """One behavior-bound cohort ready for synchronous or background execution."""

    behavior_version: int
    identity_seconds: float
    assignments: tuple[StatelessAssignedGame, ...]
    collect: Callable[[], StatelessCollectionResult]


@dataclass(frozen=True)
class _TimedCollection:
    """One completed collection with its independent service interval."""

    result: StatelessCollectionResult
    started_at: float
    finished_at: float

    @property
    def elapsed_seconds(self) -> float:
        """Return collection service wall time independent of learner overlap."""
        return max(self.finished_at - self.started_at, 1.0e-9)


@dataclass(frozen=True)
class _OverlapTiming:
    """Wall-time accounting for two deliberately overlapping operations."""

    background_seconds: float
    foreground_seconds: float
    background_wait_seconds: float
    overlap_seconds: float


@dataclass(frozen=True)
class _PendingCollection:
    """One speculative cohort whose controller leases remain in memory only."""

    prepared: _PreparedHybridCollection
    future: Future[_TimedCollection]
    member_sources: Mapping[str, str]


@dataclass(frozen=True)
class _PendingDistributedCollection:
    """One age-one remote window collecting while the H200 updates."""

    behavior_version: int
    identity_seconds: float
    identity_stage_seconds: Mapping[str, float]
    assignments: tuple[StatelessAssignedGame, ...]
    settled_curriculum_state: StatelessCurriculumState
    settled_balance_state: StatelessDeckBalanceState
    opponent_pool_window: BoundOpponentPoolWindow | BoundOpponentQuotaWindow | None
    settled_opponent_pool_state: OpponentPoolCheckpointState | None
    started_at: float
    deck_target_shares: Mapping[str, float]
    preparation_future: Future[_PreparedDistributedWindow] | None = None
    prepublication_wait_seconds: float = 0.0


@dataclass(frozen=True)
class _PendingCheckpoint:
    """One detached pair that must commit before the next state mutation."""

    future: Future[PublishedStatelessCheckpointPair]
    published_curriculum_state: StatelessCurriculumState
    predecessor_fingerprint: str
    timing: dict[str, Any]
    update: SimpleStatelessLearnerUpdate


@dataclass(frozen=True)
class _PreparedDistributedWindow:
    """A remote window whose CPU replay preparation overlapped the learner."""

    collected: StatelessCollectionResult
    array_window: StatelessArrayOptimizerWindow | None
    collection_finished_at: float
    preparation_seconds: float


@dataclass(frozen=True)
class _PreparedCurrentRolloutArtifact:
    """Behavior identity and BF16 bytes prepared before controller settlement."""

    identity: StatelessFragmentIdentity
    artifact: EncodedBfloat16RolloutArtifact
    fingerprint_seconds: float
    started_at: float
    finished_at: float

    @property
    def preparation_seconds(self) -> float:
        """Return background wall time for fingerprint, conversion, and hash."""
        return max(self.finished_at - self.started_at, 0.0)


@dataclass(slots=True)
class _OwnedNativePreparedCollect:
    """Own and release one speculative in-process native collection window."""

    collector: NativeStatelessCollector | None
    actor: GeneralistSequenceActorPolicy | None
    behavior_model: SimpleStatelessPolicyValueNet | None
    assignments: tuple[StatelessAssignedGame, ...]
    cuda_stream: Any | None

    def __call__(self) -> StatelessCollectionResult:
        """Collect exactly once and release models, KV pools, and CUDA streams."""
        collector = self.collector
        actor = self.actor
        if collector is None or actor is None or self.behavior_model is None:
            raise RuntimeError("prepared native collection resources were released")
        failure: BaseException | None = None
        try:
            if self.cuda_stream is not None:
                with torch.cuda.stream(self.cuda_stream):
                    return collect_native_banked_assigned(
                        collector,
                        self.assignments,
                    )
            return collect_native_banked_assigned(
                collector,
                self.assignments,
            )
        except BaseException as error:
            failure = error
            raise
        finally:
            try:
                self.close()
            except BaseException as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(
                    "prepared native collection teardown also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )

    def close(self) -> None:
        """Release an unstarted or completed speculative native window."""
        collector = self.collector
        actor = self.actor
        if collector is None and actor is None and self.behavior_model is None:
            return
        _close_native_collection_window(
            collector=collector,
            actor=actor,
        )
        self.collector = None
        self.actor = None
        self.behavior_model = None
        self.cuda_stream = None


def validate_simple_stateless_training_config(
    config: SimpleStatelessTrainingConfig,
) -> dict[str, Any]:
    """Resolve and fingerprint every local resource without constructing a model."""
    resources = _resolve_resources(config)
    return _validation_report(config, resources)


def resolve_simple_stateless_training_resources(
    config: SimpleStatelessTrainingConfig,
) -> _ResolvedResources:
    """Resolve the exact topology and public input contract for sibling trainers."""
    return _resolve_resources(config)


def _validation_report(
    config: SimpleStatelessTrainingConfig,
    resources: _ResolvedResources,
) -> dict[str, Any]:
    family_by_digest = {
        route.deck_digest: route.family_id
        for route in resources.model_config.family_routes
    }
    return {
        "trainer": config.trainer,
        "run_version": config.run.version,
        "resolved_config_fingerprint": resources.resolved_config_fingerprint,
        "model_config_fingerprint": model_config_fingerprint(resources.model_config),
        "exact_registry_fingerprint": (resources.model_config.resolved_registry_sha256),
        "active_exact_decks": len(resources.active_decks),
        "active_deck_routes": [
            {
                "label": label,
                "deck_digest": digest,
                "family_id": family_by_digest.get(digest),
            }
            for digest, label in sorted(
                resources.active_deck_labels.items(),
                key=lambda item: item[1],
            )
        ],
        "learner_precision": config.learner_runtime.precision,
        "optimizer_scope": config.optimizer_scope.model_dump(mode="json"),
        "training_roster_fingerprint": resources.training_roster_fingerprint,
        "curriculum_config_fingerprint": resources.curriculum_config.fingerprint,
        "deck_balance_config_fingerprint": resources.deck_balance_config.fingerprint,
        "deck_target_shares": resources.deck_target_shares,
        "public_deck_catalog_fingerprint": resources.catalog.fingerprint,
        "public_deck_catalog_entries": len(resources.catalog.entries),
        "route_input_contracts": {
            artifact_sha256: {
                "public_deck_catalog_fingerprint": route_catalog.fingerprint,
                "input_contract_fingerprint": route_contract,
            }
            for artifact_sha256, (
                route_catalog,
                route_contract,
            ) in sorted(resources.route_input_contracts.items())
        },
        "unknown_prior_mass": resources.catalog.unknown_prior_mass,
        "pinned_manifest_fingerprint": resources.pinned_manifest_fingerprint,
        "scripted_manifest_fingerprint": resources.scripted_manifest_fingerprint,
        "native_library_path": (
            None
            if resources.native_library_path is None
            else str(resources.native_library_path)
        ),
        "native_library_sha256": resources.native_library_sha256,
        "cuda_allocator_config": os.environ.get("PYTORCH_ALLOC_CONF"),
        "resume_mode": config.resume.mode,
        "transition_action": config.resume.transition_action,
        "startup_action": config.resume.startup_action,
        "opponent_pool_v2": _opponent_pool_config_payload(config),
        "supervised_startup": (
            None
            if config.resume.mode != "supervised"
            or config.resume.supervised_artifact_manifest_path is None
            or config.resume.supervised is None
            else {
                "manifest_path": str(
                    _path(config.resume.supervised_artifact_manifest_path)
                ),
                "declaration": config.resume.supervised.model_dump(mode="json"),
            }
        ),
    }


def _prepare_supervised_startup(
    config: SimpleStatelessTrainingConfig,
    resources: _ResolvedResources,
) -> StatelessSupervisedStartupPlan | None:
    """Resolve and verify the only artifact authorized for supervised mode."""
    if config.resume.mode != "supervised":
        return None
    declaration = config.resume.supervised
    manifest_path = config.resume.supervised_artifact_manifest_path
    if declaration is None or manifest_path is None:
        raise RuntimeError("supervised startup is missing its immutable declaration")
    return prepare_stateless_supervised_startup(
        _path(manifest_path),
        declaration=declaration,
        expected_model_config=resources.model_config,
        expected_exact_registry_fingerprint=(
            str(resources.model_config.resolved_registry_sha256)
        ),
        expected_public_catalog_fingerprint=resources.catalog.fingerprint,
        expected_input_contract_fingerprint=resources.input_contract.fingerprint,
        expected_event_contract_fingerprint=(
            resources.policy_identity.public_context_fingerprint
        ),
        expected_sequence_contract_fingerprint=(
            resources.policy_identity.sequence_contract_fingerprint
        ),
    )


def run_simple_stateless_training(
    config: SimpleStatelessTrainingConfig,
) -> dict[str, Any]:
    """Train from fresh, supervised, exact-resume, or transition state."""
    resources = _resolve_resources(config)
    if config.validate_only:
        report = _validation_report(config, resources)
        supervised_plan = _prepare_supervised_startup(config, resources)
        if supervised_plan is not None:
            report["supervised_artifact_audit"] = supervised_plan.audit.model_dump(
                mode="json"
            )
        return report
    output_dir = resolve_training_output_dir(
        task_name="rl",
        run=config.run,
        output_dir=config.output_dir,
    ).resolve()
    _prepare_output_dir(output_dir, fresh=config.resume.mode != "resume")
    distributed_backend = config.collection.backend == "native_distributed"
    if distributed_backend:
        bind_run_source_identity(
            output_dir,
            identity=resolve_training_source_identity(_REPO_ROOT),
            exact_resume=config.resume.mode == "resume",
            allow_legacy_adoption=source_adoption_requested(),
        )
    atomic_write_bytes(
        output_dir / "resolved_config.json",
        json_payload(
            {
                **_validation_report(config, resources),
                "output_dir": str(output_dir),
                "resolved_model_config": resources.model_config.model_dump(mode="json"),
                "performance": config.performance.model_dump(mode="json"),
            }
        ),
        overwrite=config.resume.mode == "resume",
    )

    past_self: Any
    if distributed_backend:
        past_self = MetadataPastSelfRouteCatalog()
    else:
        past_self = PastSelfPolicyPool(
            device=config.device,
            fragment_horizon=config.collection.fragment_horizon,
            non_sequence_rollout_precision=(
                config.collection.native_sequence_rollout_precision
                if config.collection.backend in _IN_PROCESS_NATIVE_BACKENDS
                else "fp32"
            ),
            archive_sequence_models_on_cpu=(
                config.collection.backend in _IN_PROCESS_NATIVE_BACKENDS
                and config.collection.native_process_workers == 1
                and config.collection.native_sequence_rollout_precision == "bf16"
            ),
        )
    anchors = tuple(
        historical_anchor_member(
            member_id=anchor.member_id,
            snapshot_id=anchor.snapshot_id,
            pilot_artifact_fingerprint=anchor.pilot_artifact_fingerprint,
            bundle_fingerprint=anchor.bundle_fingerprint,
            exact_deck_digest=anchor.exact_deck_digest,
            policy_path=_path(anchor.checkpoint_path),
            policy_size_bytes=anchor.checkpoint_size_bytes,
            policy_sha256=anchor.checkpoint_sha256,
            input_contract_fingerprint=anchor.input_contract_fingerprint,
            exact_registry_fingerprint=anchor.exact_registry_fingerprint,
            base_weight=anchor.base_weight,
            sampling_floor=anchor.sampling_floor,
            source=(
                "fixed_stateless_anchor"
                if anchor.runtime_kind == "fixed_stateless_wire"
                else "historical_anchor"
            ),
        )
        for anchor in config.curriculum.anchors
    )
    scripted_bundles = tuple(
        ScriptedCurriculumBundle(
            opponent_id=item.opponent_id,
            artifact_fingerprint=item.artifact_fingerprint,
            exact_deck_digest=item.exact_deck_digest,
            base_weight=config.curriculum.scripted_weight_overrides.get(
                item.opponent_id,
                item.base_weight,
            ),
        )
        for item in config.curriculum.scripted
    )
    pair_source = (
        None
        if config.resume.pair_manifest_path is None
        else load_stateless_checkpoint_pair(
            _path(config.resume.pair_manifest_path),
            expected_identity=(
                resources.policy_identity if config.resume.mode == "resume" else None
            ),
        )
    )
    exact_resume = config.resume.mode == "resume"
    weights_only = config.resume.mode == "weights_only"
    transition = config.resume.mode == "transition"
    bc_overlay = config.resume.mode == "bc_overlay"
    supervised = config.resume.mode == "supervised"
    preserve_transition_state = transition and config.resume.preserve_controller_state
    preserve_pair_state = preserve_transition_state or bc_overlay
    checkpoint_startup_provenance = (
        pair_source.startup_provenance
        if exact_resume and pair_source is not None
        else None
    )
    migrated_curriculum_state: StatelessCurriculumState | None = None
    migrated_balance_state: StatelessDeckBalanceState | None = None
    deck_balance_transition_audit: StatelessDeckBalanceTransitionAudit | None = None
    curriculum_lane_coverage_rebase_audit: (
        StatelessCurriculumLaneCoverageRebaseAudit | None
    ) = None
    transition_progress: StatelessSettledTrainingProgress | None = None
    registry_transition_plan: StatelessRegistryTransitionPlan | None = None
    topology_transition_plan: StatelessTopologyTransitionPlan | None = None
    sequence_context_transition_plan: StatelessSequenceContextTransitionPlan | None = (
        None
    )
    public_catalog_transition: StatelessPublicCatalogTransitionConfig | None = None
    bc_overlay_plan: StatelessBcOverlayPlan | None = None
    supervised_startup_plan: StatelessSupervisedStartupPlan | None = None
    transition_source_report: StatelessStartupReport | None = None
    topology_transition_report: StatelessStartupReport | None = None
    if weights_only:
        if pair_source is None:
            raise RuntimeError("weights-only startup did not load its source pair")
        _validate_weights_only_source(pair_source, resources=resources)
        warm_start_report = _write_weights_only_report(
            output_dir,
            source=pair_source,
            resources=resources,
        )
        checkpoint_startup_provenance = StatelessStartupProvenance(
            operation="weights_only_warm_start",
            startup_pair_version=0,
            reports=(warm_start_report,),
            source_pair_manifest_sha256=pair_source.pair.pair_manifest_sha256,
            target_model_state_fingerprint=pair_source.pair.policy_model_fingerprint,
            target_model_config_fingerprint=(
                resources.policy_identity.model_config_fingerprint
            ),
            target_exact_registry_fingerprint=(
                resources.policy_identity.exact_registry_fingerprint
            ),
        )
    if transition:
        if pair_source is None:
            raise RuntimeError("transition startup did not load its source pair")
        source_manifest_path = config.resume.pair_manifest_path
        if source_manifest_path is None:
            raise RuntimeError("transition startup omitted its source manifest")
        (
            registry_transition_plan,
            topology_transition_plan,
            sequence_context_transition_plan,
            public_catalog_transition,
        ) = _validate_transition_source(
            pair_source,
            resources=resources,
            ppo=config.ppo,
            optimizer_scope=config.optimizer_scope,
            resume=config.resume,
        )
        transition_progress = (
            _materialization_settled_progress(
                pair_source,
                target_ppo=config.ppo,
            )
            if config.resume.transition_action == "materialize_only"
            else _transition_settled_progress(
                pair_source,
                target_ppo=config.ppo,
            )
        )
        if preserve_transition_state:
            (
                curriculum_transition_source,
                curriculum_lane_coverage_rebase_audit,
            ) = _apply_transition_curriculum_lane_coverage_rebase(
                pair_source.curriculum_state,
                target_config=resources.curriculum_config,
                declaration=config.resume.curriculum_lane_coverage_rebase,
                expected_source_config_fingerprint=(
                    config.resume.expected_source_curriculum_fingerprint
                ),
                expected_target_config_fingerprint=(
                    config.resume.expected_target_curriculum_fingerprint
                ),
            )
            migrated_curriculum_state = _migrate_transition_curriculum_state(
                curriculum_transition_source,
                target_config=resources.curriculum_config,
                target_anchors=anchors,
                anchor_transition=config.resume.anchor_transition,
                drop_replaceable_members=public_catalog_transition is not None,
                target_active_deck_digests=(
                    resources.policy_identity.active_exact_deck_digests
                ),
                registered_opponent_deck_digests=frozenset(resources.opponent_decks),
                expected_source_config_fingerprint=(
                    config.resume.expected_source_curriculum_fingerprint
                ),
                expected_target_config_fingerprint=(
                    config.resume.expected_target_curriculum_fingerprint
                ),
            )
            if registry_transition_plan is not None:
                registry_declaration = config.resume.registry_transition
                if registry_declaration is None:
                    raise ValueError(
                        "registry transition plan has no lifecycle declaration"
                    )
                if config.resume.deck_balance_transition is not None:
                    raise ValueError(
                        "registry and deck-balance transitions cannot be combined"
                    )
                target_balance_config = resources.deck_balance_config
                if not isinstance(
                    target_balance_config,
                    (
                        StatelessDeckBalanceConfig,
                        StatelessDynamicDeckBalanceConfig,
                        StatelessWeightedDeckBalanceConfig,
                    ),
                ):
                    raise ValueError(
                        "registry transition requires a supported deck-balance "
                        "state migration"
                    )
                source_deck_digests = tuple(
                    route.deck_digest
                    for route in pair_source.model_config_value.exact_routes
                )
                source_balance_config: (
                    StatelessDeckBalanceConfig
                    | StatelessDynamicDeckBalanceConfig
                    | StatelessWeightedDeckBalanceConfig
                )
                if isinstance(
                    target_balance_config,
                    StatelessWeightedDeckBalanceConfig,
                ):
                    source_balance_config = (
                        target_balance_config
                        if set(source_deck_digests)
                        == set(target_balance_config.active_deck_digests)
                        else source_weighted_balance_for_registry_transition(
                            target_balance_config,
                            source_deck_digests=source_deck_digests,
                            source_target_deck_shares=(
                                registry_declaration.source_weighted_target_deck_shares
                            ),
                            expected_source_config_fingerprint=(
                                registry_declaration.source_weighted_balance_config_fingerprint
                            ),
                        )
                    )
                else:
                    source_balance_config = target_balance_config.model_copy(
                        update={"active_deck_digests": source_deck_digests}
                    )
                _require_settled_transition_balance_state(
                    pair_source.deck_balance_state,
                    target_config=source_balance_config,
                )
                migrated_balance_state = migrate_stateless_deck_balance_state(
                    pair_source.deck_balance_state,
                    source_config=source_balance_config,
                    target_config=target_balance_config,
                    plan=registry_transition_plan,
                )
            elif config.resume.deck_balance_transition is not None:
                balance_declaration = config.resume.deck_balance_transition
                target_balance_config = resources.deck_balance_config
                if not isinstance(
                    target_balance_config,
                    StatelessWeightedDeckBalanceConfig,
                ):
                    raise ValueError(
                        "deck-balance transition target must be weighted schema two"
                    )
                if (
                    pair_source.deck_balance_state.schema_version
                    != balance_declaration.source_schema_version
                    or target_balance_config.schema_version
                    != balance_declaration.target_schema_version
                    or target_balance_config.fingerprint
                    != balance_declaration.target_config_fingerprint
                ):
                    raise ValueError(
                        "deck-balance transition differs from its bound schemas or "
                        "target fingerprint"
                    )
                (
                    migrated_balance_state,
                    deck_balance_transition_audit,
                ) = migrate_settled_deck_balance_to_weighted(
                    pair_source.deck_balance_state,
                    target_config=target_balance_config,
                    expected_source_config_fingerprint=(
                        balance_declaration.source_config_fingerprint
                    ),
                )
            else:
                _require_settled_transition_balance_state(
                    pair_source.deck_balance_state,
                    target_config=resources.deck_balance_config,
                )
        transition_source_report = _write_transition_source(
            output_dir,
            manifest_path=_path(source_manifest_path),
            source=pair_source,
            target_identity=resources.policy_identity,
            target_ppo=config.ppo,
            transition_progress=transition_progress,
            migrated_curriculum_state=migrated_curriculum_state,
            registry_transition_plan=registry_transition_plan,
            topology_transition_plan=topology_transition_plan,
            sequence_context_transition_plan=(sequence_context_transition_plan),
            public_catalog_transition=public_catalog_transition,
            optimizer_scope_transition=(config.resume.optimizer_scope_transition),
            deck_balance_transition=(config.resume.deck_balance_transition),
            deck_balance_transition_audit=deck_balance_transition_audit,
            curriculum_lane_coverage_rebase=(
                config.resume.curriculum_lane_coverage_rebase
            ),
            curriculum_lane_coverage_rebase_audit=(
                curriculum_lane_coverage_rebase_audit
            ),
            target_optimizer_scope=config.optimizer_scope,
            preserve_opponent_pool_state=(
                (
                    config.resume.transition_action == "materialize_only"
                    or config.resume.optimizer_scope_transition is not None
                    or config.resume.deck_balance_transition is not None
                    or config.resume.curriculum_lane_coverage_rebase is not None
                )
                and registry_transition_plan is None
                and topology_transition_plan is None
                and sequence_context_transition_plan is None
                and public_catalog_transition is None
            ),
        )
    elif bc_overlay:
        if pair_source is None:
            raise RuntimeError("BC overlay startup did not load its source pair")
        overlay_declaration = config.resume.bc_overlay
        artifact_manifest_path = config.resume.supervised_artifact_manifest_path
        if overlay_declaration is None or artifact_manifest_path is None:
            raise RuntimeError("BC overlay startup is missing its authorization")
        _validate_bc_overlay_source(
            pair_source,
            resources=resources,
            ppo=config.ppo,
            resume=config.resume,
        )
        bc_overlay_plan = prepare_stateless_bc_overlay(
            source=pair_source,
            artifact_manifest_path=_path(artifact_manifest_path),
            declaration=overlay_declaration,
            expected_model_config=resources.model_config,
            expected_public_catalog_fingerprint=resources.catalog.fingerprint,
            expected_input_contract_fingerprint=resources.input_contract.fingerprint,
        )
        migrated_curriculum_state = bc_overlay_plan.curriculum_state
        migrated_balance_state = pair_source.deck_balance_state
    elif supervised:
        supervised_startup_plan = _prepare_supervised_startup(
            config,
            resources,
        )
        if supervised_startup_plan is None:
            raise RuntimeError("supervised startup plan is missing")
        supervised_declaration = supervised_startup_plan.audit.declaration
        supervised_startup_report = _write_supervised_startup_report(
            output_dir,
            supervised_startup_plan,
        )
        checkpoint_startup_provenance = StatelessStartupProvenance(
            operation="supervised_initialization",
            startup_pair_version=0,
            reports=(supervised_startup_report,),
            source_pair_manifest_sha256=None,
            target_model_state_fingerprint=(
                supervised_startup_plan.manifest.model_state_fingerprint
            ),
            target_model_config_fingerprint=(
                resources.policy_identity.model_config_fingerprint
            ),
            target_exact_registry_fingerprint=(
                resources.policy_identity.exact_registry_fingerprint
            ),
            supervised_manifest_sha256=supervised_declaration.manifest_sha256,
            supervised_policy_sha256=supervised_declaration.policy_sha256,
            supervised_dataset_manifest_sha256=(
                supervised_declaration.dataset_manifest_sha256
            ),
            supervised_dataset_fingerprint=(supervised_declaration.dataset_fingerprint),
        )
    curriculum_state_path = output_dir / "control" / "curriculum_state.json"
    exact_pair: LoadedStatelessCheckpointPair | None = None
    if exact_resume:
        if pair_source is None:
            raise RuntimeError("exact resume did not load its source pair")
        exact_pair = pair_source
        _require_external_curriculum_state(
            curriculum_state_path,
            exact_pair,
        )
        reconcile_stateless_resume_artifacts(
            output_dir,
            selected=exact_pair,
        )
    curriculum = StatelessCurriculumController(
        resources.curriculum_config,
        anchors=anchors,
        scripted=scripted_bundles,
        state_path=curriculum_state_path,
        route_preparer=past_self.prepare,
        route_unloader=past_self.unload,
        resume=exact_resume,
        initial_state=migrated_curriculum_state,
        persist_mutations=False,
    )
    founder_artifact: DurablePolicyArtifact | None = None
    starts_new_lineage_pool = weights_only or (
        transition and not preserve_transition_state
    )
    if starts_new_lineage_pool and config.opponent_pool_v2.behavior_version in {
        2,
        3,
        4,
        5,
    }:
        if pair_source is None:
            raise RuntimeError("lineage founder source is missing")
        founder_artifact = _create_pair_lineage_founder(
            output_dir,
            source=pair_source,
            curriculum=curriculum,
        )
    if exact_pair is not None and curriculum.state != exact_pair.curriculum_state:
        raise ValueError("external curriculum state differs from exact sidecar")
    if (
        migrated_curriculum_state is not None
        and curriculum.state != migrated_curriculum_state
    ):
        raise ValueError("external curriculum state differs from transition migration")

    model, initial_version, model_transition_report = _model_from_startup(
        resources,
        pair_source,
        supervised_state=(
            None
            if supervised_startup_plan is None
            else supervised_startup_plan.model_state
        ),
        bc_overlay_state=(
            None if bc_overlay_plan is None else bc_overlay_plan.model_state
        ),
        seed=config.collection.seed,
        registry_transition_plan=registry_transition_plan,
        topology_transition_plan=topology_transition_plan,
        sequence_context_transition_plan=sequence_context_transition_plan,
        public_catalog_transition=public_catalog_transition,
        weights_only=weights_only,
    )
    supervised_startup_plan = None
    learner = SimpleStatelessLearner(
        model,
        config.ppo,
        device=config.device,
        require_h200=True,
        precision=config.learner_runtime.precision,
        fused_adamw=config.learner_runtime.fused_adamw,
        compile_shared_backbone=(config.learner_runtime.compile_shared_backbone),
        host_prepare_workers=config.learner_runtime.host_prepare_workers,
        host_prepare_prefetch_batches=(
            config.learner_runtime.host_prepare_prefetch_batches
        ),
        trainable_scope=config.optimizer_scope.mode,
        private_learning_rate=config.optimizer_scope.private_learning_rate,
        shared_initial_learning_rate=(
            config.optimizer_scope.shared_learning_rate_initial
        ),
        shared_learning_rate=config.optimizer_scope.shared_learning_rate_target,
        shared_warmup_start_update=(
            config.optimizer_scope.shared_learning_rate_warmup_start_update_index
        ),
        shared_warmup_updates=(
            config.optimizer_scope.shared_learning_rate_warmup_updates
        ),
        deck_macro_target_shares=resources.deck_target_shares,
    )
    if exact_pair is not None:
        if exact_pair.ppo_config != config.ppo:
            raise ValueError("exact-resume PPO config changed")
        if (
            config.ppo.total_decisions is not None
            and not exact_pair.settled_progress_recorded
        ):
            raise ValueError(
                "decision-scheduled exact resume requires recorded settled progress"
            )
        exact_progress = exact_pair.settled_progress
        learner.restore_exact(
            optimizer_state=exact_pair.optimizer_state,
            update_index=exact_pair.update_index,
            optimizer_step_index=(
                None if exact_progress is None else exact_progress.optimizer_step_index
            ),
            fresh_decisions_seen=(
                0 if exact_progress is None else exact_progress.fresh_decisions_seen
            ),
            lr_schedule_decisions_seen=(
                0
                if exact_progress is None
                else exact_progress.lr_schedule_decisions_seen
            ),
        )
        if learner.update_index != initial_version:
            raise ValueError("learner update cursor differs from policy version")
    elif transition:
        if pair_source is None or transition_progress is None:
            raise RuntimeError("transition startup omitted its settled progress")
        optimizer_state = pair_source.optimizer_state
        optimizer_transition_report: dict[str, int] | None = None
        topology_optimizer_report: StatelessTopologyOptimizerAudit | None = None
        optimizer_scope_report: (
            StatelessPrivateOptimizerTransitionAudit
            | StatelessHybridOptimizerTransitionAudit
            | None
        ) = None
        if config.resume.optimizer_scope_transition is not None:
            private_learning_rate = config.optimizer_scope.private_learning_rate
            if private_learning_rate is None:
                raise RuntimeError(
                    "private optimizer transition omitted its target learning rate"
                )
            optimizer_transition = config.resume.optimizer_scope_transition
            if optimizer_transition.target_mode == "hybrid":
                shared_learning_rate = (
                    config.optimizer_scope.shared_learning_rate_initial
                )
                if shared_learning_rate is None:
                    raise RuntimeError(
                        "hybrid optimizer transition omitted its initial shared LR"
                    )
                optimizer_scope_report = transplant_private_optimizer_to_hybrid(
                    optimizer=learner.optimizer,
                    target_model=learner.model,
                    source_optimizer_state=pair_source.optimizer_state,
                    scope=learner.trainable_scope,
                    private_learning_rate=private_learning_rate,
                    shared_learning_rate=shared_learning_rate,
                )
            elif optimizer_transition.source_mode == "full_model":
                optimizer_scope_report = transplant_full_optimizer_to_private(
                    optimizer=learner.optimizer,
                    target_model=learner.model,
                    source_optimizer_state=pair_source.optimizer_state,
                    scope=learner.trainable_scope,
                    private_learning_rate=private_learning_rate,
                )
            else:
                optimizer_scope_report = retarget_private_optimizer_learning_rate(
                    optimizer=learner.optimizer,
                    target_model=learner.model,
                    source_optimizer_state=pair_source.optimizer_state,
                    scope=learner.trainable_scope,
                    private_learning_rate=private_learning_rate,
                )
            optimizer_state = learner.optimizer.state_dict()
        elif registry_transition_plan is not None:
            optimizer_transition_report = transplant_stateless_registry_optimizer(
                optimizer=learner.optimizer,
                target_model=learner.model,
                source_model_config=pair_source.model_config_value,
                source_optimizer_state=pair_source.optimizer_state,
                plan=registry_transition_plan,
            )
            optimizer_state = learner.optimizer.state_dict()
        elif topology_transition_plan is not None:
            topology_optimizer_report = transplant_stateless_topology_optimizer(
                optimizer=learner.optimizer,
                target_model=learner.model,
                source_model_config=pair_source.model_config_value,
                source_optimizer_state=pair_source.optimizer_state,
                plan=topology_transition_plan,
            )
            optimizer_state = learner.optimizer.state_dict()
        learner.restore_exact(
            optimizer_state=optimizer_state,
            update_index=transition_progress.rollout_window_index,
            optimizer_step_index=transition_progress.optimizer_step_index,
            fresh_decisions_seen=transition_progress.fresh_decisions_seen,
            lr_schedule_decisions_seen=(transition_progress.lr_schedule_decisions_seen),
        )
        if learner.update_index != initial_version:
            raise ValueError("transition learner cursor differs from policy version")
        if config.resume.optimizer_scope_transition is not None:
            if optimizer_scope_report is None:
                raise RuntimeError("optimizer scope transition report is incomplete")
            _write_optimizer_scope_transition_report(
                output_dir,
                source=pair_source,
                target_identity=resources.policy_identity,
                declaration=config.resume.optimizer_scope_transition,
                audit=optimizer_scope_report,
            )
        if registry_transition_plan is not None:
            if (
                not isinstance(model_transition_report, dict)
                or optimizer_transition_report is None
            ):
                raise RuntimeError("stateless registry transition report is incomplete")
            _write_registry_transition_report(
                output_dir,
                plan=registry_transition_plan,
                model_report=model_transition_report,
                optimizer_report=optimizer_transition_report,
                source_balance=pair_source.deck_balance_state,
                target_balance=migrated_balance_state,
            )
        if topology_transition_plan is not None:
            if (
                not isinstance(model_transition_report, StatelessTopologyModelAudit)
                or topology_optimizer_report is None
                or config.resume.topology_transition is None
            ):
                raise RuntimeError("stateless topology transition report is incomplete")
            topology_transition_report = _write_topology_transition_report(
                output_dir,
                StatelessTopologyTransitionAuditReport(
                    declaration=config.resume.topology_transition,
                    plan=topology_transition_plan,
                    model=model_transition_report,
                    optimizer=topology_optimizer_report,
                ),
            )
            if config.resume.transition_action == "materialize_only":
                if transition_source_report is None or pair_source is None:
                    raise RuntimeError(
                        "materialize-only transition provenance is incomplete"
                    )
                checkpoint_startup_provenance = StatelessStartupProvenance(
                    operation="topology_materialization",
                    startup_pair_version=pair_source.pair.version,
                    reports=(
                        topology_transition_report,
                        transition_source_report,
                    ),
                    source_pair_manifest_sha256=(pair_source.pair.pair_manifest_sha256),
                    target_model_state_fingerprint=(
                        canonical_model_state_fingerprint(learner.model)
                    ),
                    target_model_config_fingerprint=(
                        resources.policy_identity.model_config_fingerprint
                    ),
                    target_exact_registry_fingerprint=(
                        resources.policy_identity.exact_registry_fingerprint
                    ),
                )
        elif config.resume.transition_action == "materialize_only":
            if (
                transition_source_report is None
                or pair_source is None
                or registry_transition_plan is not None
                or sequence_context_transition_plan is not None
                or public_catalog_transition is not None
                or config.resume.anchor_transition is not None
                or config.resume.gae_lambda_transition_from is not None
            ):
                raise RuntimeError(
                    "identity-only materialization changed a training contract"
                )
            checkpoint_startup_provenance = StatelessStartupProvenance(
                operation="identity_materialization",
                startup_pair_version=pair_source.pair.version,
                reports=(transition_source_report,),
                source_pair_manifest_sha256=(pair_source.pair.pair_manifest_sha256),
                target_model_state_fingerprint=(
                    canonical_model_state_fingerprint(learner.model)
                ),
                target_model_config_fingerprint=(
                    resources.policy_identity.model_config_fingerprint
                ),
                target_exact_registry_fingerprint=(
                    resources.policy_identity.exact_registry_fingerprint
                ),
            )
    elif bc_overlay:
        if pair_source is None or bc_overlay_plan is None:
            raise RuntimeError("BC overlay startup plan is incomplete")
        overlay_declaration = config.resume.bc_overlay
        if overlay_declaration is None:
            raise RuntimeError("BC overlay declaration disappeared after validation")
        optimizer_overlay_report = transplant_stateless_bc_overlay_optimizer(
            optimizer=learner.optimizer,
            target_model=learner.model,
            source_optimizer_state=pair_source.optimizer_state,
            reset_parameter_names=(
                bc_overlay_plan.model_audit.trainable_parameter_names
            ),
        )
        overlay_progress = bc_overlay_plan.settled_progress
        learner.restore_exact(
            optimizer_state=learner.optimizer.state_dict(),
            update_index=overlay_progress.rollout_window_index,
            optimizer_step_index=overlay_progress.optimizer_step_index,
            fresh_decisions_seen=overlay_progress.fresh_decisions_seen,
            lr_schedule_decisions_seen=overlay_progress.lr_schedule_decisions_seen,
        )
        if (
            learner.update_index != initial_version
            or learner.config != pair_source.ppo_config
            or _learner_settled_progress(learner) != overlay_progress
        ):
            raise ValueError("BC overlay changed its PPO config or settled cursors")
        if (
            canonical_model_state_fingerprint(learner.model)
            != bc_overlay_plan.model_audit.target_model_state_fingerprint
        ):
            raise RuntimeError("BC overlay model changed after its tensor audit")
        bc_overlay_report = _write_bc_overlay_report(
            output_dir,
            StatelessBcOverlayAuditReport(
                declaration=overlay_declaration,
                source_pair_manifest_sha256=(pair_source.pair.pair_manifest_sha256),
                supervised_manifest_sha256=(
                    overlay_declaration.supervised_manifest_sha256
                ),
                settled_progress=overlay_progress,
                model=bc_overlay_plan.model_audit,
                optimizer=optimizer_overlay_report,
                curriculum=bc_overlay_plan.curriculum_audit,
                source_fragment_parts_discarded=len(
                    pair_source.fragment_recovery.parts
                ),
            ),
        )
        checkpoint_startup_provenance = StatelessStartupProvenance(
            operation="bc_overlay",
            startup_pair_version=overlay_declaration.source_pair_version,
            reports=(bc_overlay_report,),
            source_pair_manifest_sha256=(
                overlay_declaration.source_pair_manifest_sha256
            ),
            target_model_state_fingerprint=(
                bc_overlay_plan.model_audit.target_model_state_fingerprint
            ),
            target_model_config_fingerprint=(
                resources.policy_identity.model_config_fingerprint
            ),
            target_exact_registry_fingerprint=(
                resources.policy_identity.exact_registry_fingerprint
            ),
            supervised_manifest_sha256=(overlay_declaration.supervised_manifest_sha256),
            target_deck_digest=overlay_declaration.target_deck_digest,
        )
    if (
        public_catalog_transition is not None
        and config.opponent_pool_v2.enabled
        and config.opponent_pool_v2.behavior_version in {2, 3, 4, 5}
    ):
        if founder_artifact is not None:
            raise RuntimeError("catalog transition founder was already initialized")
        founder_artifact = _create_public_catalog_transition_founder(
            output_dir,
            learner=learner,
            resources=resources,
            curriculum=curriculum,
        )
    if (
        supervised
        and config.opponent_pool_v2.enabled
        and config.opponent_pool_v2.behavior_version in {2, 3, 4, 5}
    ):
        if founder_artifact is not None:
            raise RuntimeError("supervised lineage founder was already initialized")
        founder_artifact = _create_supervised_lineage_founder(
            output_dir,
            learner=learner,
            resources=resources,
            curriculum=curriculum,
        )
    recovered_balance_state = (
        (
            migrated_balance_state
            if migrated_balance_state is not None
            else pair_source.deck_balance_state
        )
        if preserve_pair_state and pair_source is not None
        else exact_pair.deck_balance_state
        if exact_pair is not None
        else None
    )
    balance = StatelessDeckBalanceSampler(
        resources.deck_balance_config,
        state=recovered_balance_state,
    )
    opponent_pool_v2: StatelessOpponentPoolV2 | None = None
    if config.opponent_pool_v2.enabled:
        initial_opponent_pool_state = (
            exact_pair.opponent_pool_state
            if exact_pair is not None
            else None
            if public_catalog_transition is not None
            else pair_source.opponent_pool_state
            if preserve_pair_state and pair_source is not None
            else None
        )
        if exact_pair is not None and initial_opponent_pool_state is None:
            raise ValueError("opponent-pool V2 exact resume omitted its state")
        opponent_pool_v2 = StatelessOpponentPoolV2(
            config.opponent_pool_v2,
            active_deck_digests=(resources.policy_identity.active_exact_deck_digests),
            curriculum=curriculum,
            initial_state=initial_opponent_pool_state,
            founder_policy_sha256=(
                founder_artifact.policy_sha256
                if founder_artifact is not None
                and config.opponent_pool_v2.behavior_version in {2, 3, 4, 5}
                else None
            ),
            # A dynamic balance state owns the targets for the next cohort.
            # Reusing static config targets here can mutate the active pool
            # revision before an exact-resume process has collected anything.
            deck_target_shares=balance.target_share_mapping,
            # Preserve a settled checkpoint verbatim. The first assignment
            # plan synchronizes the successor revision using the recovered
            # next-cohort targets, just as uninterrupted training does.
            synchronize_revision=not (
                exact_pair is not None
                or (
                    checkpoint_startup_provenance is not None
                    and checkpoint_startup_provenance.operation
                    == "identity_materialization"
                )
            ),
        )
    elif exact_pair is not None and exact_pair.opponent_pool_state is not None:
        raise ValueError("exact resume disabled its recorded opponent-pool V2 state")
    if (
        transition_source_report is not None
        and config.resume.deck_balance_transition is not None
    ):
        if checkpoint_startup_provenance is not None:
            raise RuntimeError(
                "deck-balance transition report cannot precede startup provenance"
            )
        transition_source_report = _retarget_transition_opponent_pool_report(
            transition_source_report,
            target_state=(None if opponent_pool_v2 is None else opponent_pool_v2.state),
        )
    writer = CompactFragmentShardWriter(
        output_dir / "fragments",
        static_contract_fingerprint=(
            resources.policy_identity.fragment_static_contract_fingerprint
        ),
        horizon=config.collection.fragment_horizon,
        fragments_per_part=config.collection.fragments_per_part,
        resume=exact_resume,
        sequence=uses_generalist_sequence(resources.model_config),
    )
    if exact_pair is not None:
        _require_exact_fragment_recovery(
            writer.manifest,
            exact_pair.fragment_recovery,
        )
    if config.resume.transition_action == "materialize_only":
        if pair_source is None or transition_progress is None:
            raise RuntimeError(
                "materialize-only transition omitted its source progress"
            )
        materialization_started_at = time.perf_counter()
        try:
            if (
                checkpoint_startup_provenance is not None
                and checkpoint_startup_provenance.operation
                == "identity_materialization"
            ):
                _require_identity_materialization_controller_state(
                    pair_source,
                    balance=balance,
                    curriculum=curriculum,
                    opponent_pool_state=(
                        None if opponent_pool_v2 is None else opponent_pool_v2.state
                    ),
                )
            _require_transition_materialization_state(
                learner,
                source_progress=pair_source.settled_progress,
                expected_progress=transition_progress,
                source_ppo=pair_source.ppo_config,
                expected_ppo=config.ppo,
            )
            materialized_pair = _publish_initial_pair(
                output_dir,
                learner=learner,
                resources=resources,
                writer=writer,
                balance=balance,
                curriculum=curriculum,
                opponent_pool_state=(
                    None if opponent_pool_v2 is None else opponent_pool_v2.state
                ),
                config=config,
                fragments_seen=pair_source.staleness_state.fragments_seen,
                fragments_stale=pair_source.staleness_state.fragments_stale,
                startup_provenance=checkpoint_startup_provenance,
            )
        finally:
            writer.close()
        return {
            "operation": "transition_materialization",
            "run_version": config.run.version,
            "output_dir": str(output_dir),
            "updates": learner.update_index,
            "optimizer_steps": learner.optimizer_step_index,
            "fresh_decisions_seen": learner.fresh_decisions_seen,
            "lr_schedule_decisions_seen": learner.lr_schedule_decisions_seen,
            "policy_path": str(materialized_pair.policy_path),
            "learner_state_path": str(materialized_pair.learner_state_path),
            "pair_manifest_path": str(materialized_pair.pair_manifest_path),
            "pair_manifest_sha256": materialized_pair.pair_manifest_sha256,
            "past_self_admissions": sum(
                event.kind == "admitted" for event in curriculum.state.events
            ),
            "elapsed_seconds": time.perf_counter() - materialization_started_at,
        }
    if supervised and config.resume.startup_action == "materialize_only":
        materialization_started_at = time.perf_counter()
        if (
            learner.update_index != 0
            or learner.optimizer_step_index != 0
            or learner.fresh_decisions_seen != 0
            or learner.lr_schedule_decisions_seen != 0
        ):
            raise RuntimeError("supervised v0 materialization is not fresh")
        try:
            materialized_pair = _publish_initial_pair(
                output_dir,
                learner=learner,
                resources=resources,
                writer=writer,
                balance=balance,
                curriculum=curriculum,
                opponent_pool_state=(
                    None if opponent_pool_v2 is None else opponent_pool_v2.state
                ),
                config=config,
                fragments_seen=0,
                fragments_stale=0,
                startup_provenance=checkpoint_startup_provenance,
            )
        finally:
            writer.close()
        return {
            "operation": "supervised_materialization",
            "run_version": config.run.version,
            "output_dir": str(output_dir),
            "updates": learner.update_index,
            "optimizer_steps": learner.optimizer_step_index,
            "fresh_decisions_seen": learner.fresh_decisions_seen,
            "lr_schedule_decisions_seen": learner.lr_schedule_decisions_seen,
            "policy_path": str(materialized_pair.policy_path),
            "learner_state_path": str(materialized_pair.learner_state_path),
            "pair_manifest_path": str(materialized_pair.pair_manifest_path),
            "pair_manifest_sha256": materialized_pair.pair_manifest_sha256,
            "elapsed_seconds": time.perf_counter() - materialization_started_at,
        }
    if bc_overlay:
        if pair_source is None or bc_overlay_plan is None:
            raise RuntimeError("BC overlay startup plan is incomplete")
        if (
            writer.manifest.parts
            or writer.manifest.fragments_committed != 0
            or writer.manifest.decisions_committed != 0
        ):
            raise RuntimeError("BC overlay fragment recovery was not reset")
        overlay_started_at = time.perf_counter()
        try:
            overlay_pair = _publish_initial_pair(
                output_dir,
                learner=learner,
                resources=resources,
                writer=writer,
                balance=balance,
                curriculum=curriculum,
                opponent_pool_state=(
                    None if opponent_pool_v2 is None else opponent_pool_v2.state
                ),
                config=config,
                fragments_seen=pair_source.staleness_state.fragments_seen,
                fragments_stale=pair_source.staleness_state.fragments_stale,
                startup_provenance=checkpoint_startup_provenance,
            )
        finally:
            writer.close()
        return {
            "operation": "bc_overlay",
            "run_version": config.run.version,
            "output_dir": str(output_dir),
            "updates": learner.update_index,
            "optimizer_steps": learner.optimizer_step_index,
            "fresh_decisions_seen": learner.fresh_decisions_seen,
            "lr_schedule_decisions_seen": learner.lr_schedule_decisions_seen,
            "policy_path": str(overlay_pair.policy_path),
            "learner_state_path": str(overlay_pair.learner_state_path),
            "pair_manifest_path": str(overlay_pair.pair_manifest_path),
            "pair_manifest_sha256": overlay_pair.pair_manifest_sha256,
            "target_deck_digest": (bc_overlay_plan.model_audit.target_deck_digest),
            "changed_tensors": bc_overlay_plan.model_audit.changed_tensor_count,
            "reset_optimizer_parameters": (
                bc_overlay_plan.model_audit.trainable_tensor_count
            ),
            "elapsed_seconds": time.perf_counter() - overlay_started_at,
        }
    performance_reporter: TrainingPerformanceReporter | None = None
    try:
        performance_reporter = _performance_reporter(
            config,
            resources=resources,
            output_dir=output_dir,
        )
        if performance_reporter is not None:
            performance_reporter.start()
    except Exception:
        _LOGGER.exception(
            "performance telemetry failed to start; training will continue"
        )
        if performance_reporter is not None:
            _run_noncritical(
                "partial performance telemetry shutdown",
                performance_reporter.close,
            )
        performance_reporter = None
    historical_resources = {
        anchor.member_id: (
            anchor,
            resources.opponent_decks[anchor.exact_deck_digest].card_ids,
        )
        for anchor in config.curriculum.anchors
    }
    historical: HistoricalPolicyPool | None = None
    native_historical: NativeHistoricalPolicyPool | None = None
    native_scripted: dict[str, NativeScriptedPolicy] = {}
    native_scripted_bindings: dict[str, tuple[str, str]] = {}
    if distributed_backend:
        # The H200 coordinator owns metadata and wire artifacts only. Engine,
        # frozen-policy inference, and scripted-policy runtimes stay remote.
        pass
    elif config.collection.backend in _NATIVE_BACKENDS:
        native_historical = NativeHistoricalPolicyPool(historical_resources)
        native_historical.preload_artifacts()
        if config.collection.backend in _IN_PROCESS_NATIVE_BACKENDS:
            static_features = np.load(
                _path(resources.model_config.card_encoder.feature_table_path),
                mmap_mode="r",
                allow_pickle=False,
            )
            for item in config.curriculum.scripted:
                if item.opponent_name == "mixed75":
                    native_scripted[item.opponent_id] = NativeMixed75Policy(
                        scripted_deck=resources.opponent_decks[
                            item.exact_deck_digest
                        ].card_ids,
                        static_features=static_features,
                    )
                else:
                    native_scripted[item.opponent_id] = NativePublicScriptedPolicy(
                        resources.scripted_opponents[item.opponent_id]
                    )
                native_scripted_bindings[item.opponent_id] = (
                    item.artifact_fingerprint,
                    item.exact_deck_digest,
                )
    else:
        historical = HistoricalPolicyPool(historical_resources)
    parallel_collector: StatelessParallelCollector | None = None
    parallel_temporary_root: Path | None = None
    native_process_collector: NativeProcessCollector | None = None
    native_process_temporary_root: Path | None = None
    rollout_model: SimpleStatelessPolicyValueNet | None = None
    integrated_process_inference = (
        config.collection.integrate_scripted_current_inference
    )
    pipeline_enabled = (
        config.collection.pipeline_mode == "one_version_lag" and not distributed_backend
    )
    native_policy_stream_pool = (
        NativePolicyCudaStreamPool()
        if config.collection.backend in _IN_PROCESS_NATIVE_BACKENDS
        else None
    )
    pipeline_collection_stream = (
        torch.cuda.Stream(priority=-1)  # type: ignore[no-untyped-call]
        if pipeline_enabled
        else None
    )
    parallel_inference_request_queue: Any | None = None
    parallel_inference_response_queues: dict[int, Any] = {}
    if (
        config.collection.backend in {"python_parallel", "hybrid"}
        and config.collection.actor_workers > 1
    ):
        process_context = multiprocessing.get_context("spawn")
        parallel_inference_request_queue = process_context.Queue(
            maxsize=config.collection.actor_workers * 4
        )
        parallel_inference_response_queues = {
            actor_index: process_context.Queue(maxsize=2)
            for actor_index in range(config.collection.actor_workers)
        }
        parallel_historical_resources = (
            {}
            if config.collection.backend == "hybrid"
            else {
                anchor.member_id: (
                    anchor,
                    tuple(resources.opponent_decks[anchor.exact_deck_digest].card_ids),
                )
                for anchor in config.curriculum.anchors
            }
        )
        worker_specs = tuple(
            StatelessParallelWorkerSpec(
                actor_index=actor_index,
                candidate_contract=resources.input_contract,
                catalog=resources.catalog,
                active_decks=resources.active_decks,
                opponent_decks=resources.opponent_decks,
                historical_resources=parallel_historical_resources,
                scripted={
                    item.opponent_id: item for item in config.curriculum.scripted
                },
                fragment_horizon=config.collection.fragment_horizon,
                maximum_engine_steps=(config.collection.maximum_engine_steps),
                seed=config.collection.seed,
                inference_timeout_seconds=(config.collection.inference_timeout_seconds),
                mirror_bilateral_trajectories=(
                    config.collection.mirror_bilateral_trajectories
                ),
                engine_fact_config=(
                    None
                    if resources.model_config.sequence is None
                    else resources.model_config.sequence.engine_facts
                ),
            )
            for actor_index in range(config.collection.actor_workers)
        )
        parallel_collector = StatelessParallelCollector(
            context=process_context,
            worker_specs=worker_specs,
            request_queue=parallel_inference_request_queue,
            response_queues=parallel_inference_response_queues,
            past_self_pool=past_self,
            max_batch_rows=config.collection.inference_max_batch_rows,
            batch_wait_seconds=(config.collection.inference_batch_wait_ms / 1000.0),
            result_timeout_seconds=(config.collection.collection_timeout_seconds),
            game_chunk_size=config.collection.actor_game_chunk_size,
        )
        parallel_temporary_root = (
            _REPO_ROOT
            / "tmp"
            / "stateless_parallel"
            / f"{config.run.version}-{uuid.uuid4().hex}"
        )
        rollout_model = SimpleStatelessPolicyValueNet(
            resources.model_config,
            load_static_features=False,
            initialize=False,
        )
        rollout_model.load_state_dict(learner.model.state_dict(), strict=True)
        rollout_model.to(device=config.device, dtype=torch.bfloat16).eval()
        if pipeline_collection_stream is not None:
            pipeline_collection_stream.wait_stream(
                torch.cuda.current_stream(device=learner.device)
            )
    if (
        config.collection.backend in _NATIVE_BACKENDS
        and config.collection.native_process_workers > 1
    ):
        if native_historical is None:
            raise RuntimeError("native process historical resources are absent")
        process_context = multiprocessing.get_context("spawn")
        inference_request_queue = process_context.Queue(
            maxsize=config.collection.native_process_workers * 2
        )
        if integrated_process_inference and parallel_inference_request_queue is None:
            raise RuntimeError("integrated scripted inference queue is absent")
        inference_response_queues = {
            worker_index: process_context.Queue(maxsize=2)
            for worker_index in range(config.collection.native_process_workers)
        }
        native_process_collector = NativeProcessCollector(
            context=process_context,
            worker_specs=tuple(
                NativeProcessWorkerSpec(
                    worker_index=worker_index,
                    catalog=resources.catalog,
                    active_decks=resources.active_decks,
                    opponent_decks=resources.opponent_decks,
                    scripted=(
                        config.curriculum.scripted
                        if config.collection.backend in _IN_PROCESS_NATIVE_BACKENDS
                        else ()
                    ),
                    static_feature_path=_path(
                        resources.model_config.card_encoder.feature_table_path
                    ),
                    maximum_engine_steps=(config.collection.maximum_engine_steps),
                    fragments_per_part=(config.collection.fragments_per_part),
                    mirror_bilateral_trajectories=(
                        config.collection.mirror_bilateral_trajectories
                    ),
                    options_per_lane=256,
                    library_path=resources.native_library_path,
                    inference_timeout_seconds=(
                        config.collection.inference_timeout_seconds
                    ),
                    sequence_rollout_precision=(
                        config.collection.native_sequence_rollout_precision
                    ),
                    engine_shards=config.collection.native_engine_shards,
                    policy_cohort_wait_ms=(
                        config.collection.native_policy_cohort_wait_ms
                    ),
                    engine_fact_config=(
                        resources.model_config.sequence.engine_facts
                        if resources.model_config.sequence is not None
                        and resources.model_config.sequence.engine_facts.enabled
                        else None
                    ),
                    engine_fact_workers=(
                        None
                        if config.collection.native_engine_fact_workers is None
                        else max(
                            1,
                            config.collection.native_engine_fact_workers
                            // config.collection.native_process_workers,
                        )
                    ),
                    scripted_artifacts={
                        opponent_id: resolved.artifact
                        for opponent_id, resolved in (
                            resources.scripted_opponents.items()
                        )
                    },
                )
                for worker_index in range(config.collection.native_process_workers)
            ),
            request_queue=inference_request_queue,
            scripted_request_queue=(
                parallel_inference_request_queue
                if integrated_process_inference
                else None
            ),
            response_queues=inference_response_queues,
            past_self_pool=past_self,
            historical_pool=native_historical,
            max_batch_rows=config.collection.inference_max_batch_rows,
            batch_wait_seconds=(config.collection.inference_batch_wait_ms / 1000.0),
            result_timeout_seconds=(config.collection.collection_timeout_seconds),
            scripted_response_queues=(
                parallel_inference_response_queues
                if integrated_process_inference
                else {}
            ),
            scripted_sampling_seed=config.collection.seed,
            cuda_stream=pipeline_collection_stream,
        )
        native_process_temporary_root = (
            _REPO_ROOT
            / "tmp"
            / "native_process"
            / f"{config.run.version}-{uuid.uuid4().hex}"
        )
    pipeline_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline-collection")
        if pipeline_enabled
        else None
    )
    distributed_prepare_executor = (
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="distributed-replay-prepare",
        )
        if distributed_backend and config.collection.pipeline_mode == "one_version_lag"
        else None
    )
    distributed_artifact_executor = (
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="distributed-artifact-prepare",
        )
        if distributed_backend and config.collection.pipeline_mode == "one_version_lag"
        else None
    )
    pending_collection: _PendingCollection | None = None
    pending_distributed: _PendingDistributedCollection | None = None
    rollout_model_version = initial_version
    collection_reports: list[StatelessCollectionReport] = []
    update_reports: list[SimpleStatelessLearnerUpdate] = []
    recovered_state_pair = (
        pair_source if exact_resume or preserve_transition_state else None
    )
    fragments_seen = (
        recovered_state_pair.staleness_state.fragments_seen
        if recovered_state_pair is not None
        else 0
    )
    fragments_stale = (
        recovered_state_pair.staleness_state.fragments_stale
        if recovered_state_pair is not None
        else 0
    )
    last_pair: StatelessCheckpointPair | None = (
        exact_pair.pair if exact_pair is not None else None
    )
    if not exact_resume:
        last_pair = _publish_initial_pair(
            output_dir,
            learner=learner,
            resources=resources,
            writer=writer,
            balance=balance,
            curriculum=curriculum,
            opponent_pool_state=(
                None if opponent_pool_v2 is None else opponent_pool_v2.state
            ),
            config=config,
            fragments_seen=fragments_seen,
            fragments_stale=fragments_stale,
            startup_provenance=checkpoint_startup_provenance,
        )
    distributed_collection: NativeDistributedTrainingBackend | None = None
    if distributed_backend:
        distributed_collection = NativeDistributedTrainingBackend(
            config,
            expected_worker_contract=_distributed_worker_contract(resources),
            output_dir=output_dir,
        )
    started_at = time.perf_counter()
    timing_reports: list[dict[str, Any]] = []
    checkpoint_publisher = AsyncStatelessCheckpointPairPublisher()
    pending_checkpoint: _PendingCheckpoint | None = None
    status_writer = NonBlockingStatelessStatusWriter(output_dir)
    learner_metric_writer = (
        NonBlockingLearnerMetricWriter(output_dir)
        if config.performance.learner_metric_history_enabled
        else None
    )

    native_collector: NativeStatelessCollector | None = None
    native_actor: (
        NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy | None
    ) = None
    training_failed = True
    try:
        while not _training_budget_reached(learner, config=config):
            if learner.update_index >= config.collection.training_updates:
                raise RuntimeError(
                    "rollout-window safety cap was reached before the decision budget"
                )
            window_started_at = time.perf_counter()
            _reset_cuda_peak_memory(learner.device)
            cuda_memory = _cuda_memory_snapshot(
                learner.device,
                phase="window_start",
            )
            version = learner.update_index
            collection_prefetched = (
                pending_collection is not None or pending_distributed is not None
            )
            collection_wait_seconds = 0.0
            collection_overlap_seconds = 0.0
            admission_prefetch_wait_seconds = 0.0
            policy_prefetch_publication_seconds = 0.0
            policy_prefetch_wait_seconds = 0.0
            policy_prefetch_overlap_seconds = 0.0
            prefetched_policy_artifact: VerifiedPolicyPublication | None = None
            behavior_version = version
            identity_stage_seconds: dict[str, float] = {
                "fingerprint_seconds": 0.0,
                "assignment_planning_seconds": 0.0,
                "artifact_assignment_overlap_seconds": 0.0,
                "artifact_preparation_task_seconds": 0.0,
                "artifact_preparation_prefetch_overlap_seconds": 0.0,
                "artifact_preparation_wait_seconds": 0.0,
                "artifact_source_fingerprint_seconds": 0.0,
                "bf16_conversion_seconds": 0.0,
                "wire_hashing_seconds": 0.0,
                "window_open_seconds": 0.0,
            }
            behavior_identity: StatelessFragmentIdentity | None = None
            prefetched_array_window: StatelessArrayOptimizerWindow | None = None
            actor: SimpleStatelessActorPolicy | GeneralistSequenceActorPolicy | None = (
                None
            )
            native_actor = None
            native_collector = None
            native_assignments: tuple[StatelessAssignedGame, ...] | None = None
            distributed_assignment_pool: tuple[StatelessAssignedGame, ...] = ()
            distributed_adopted_assignments: tuple[StatelessAssignedGame, ...] = ()
            distributed_assignment_plan: (
                tuple[StatelessAssignedGame, ...] | StatelessQuotaAssignmentPlan
            ) = ()
            active_opponent_pool_window: (
                BoundOpponentPoolWindow | BoundOpponentQuotaWindow | None
            ) = None
            window_deck_target_shares = balance.target_share_mapping
            performance_member_sources: Mapping[str, str] = _performance_member_sources(
                curriculum.state
            )
            if pending_distributed is not None:
                if distributed_collection is None:
                    raise RuntimeError(
                        "pending distributed collection lost its backend"
                    )
                pending_remote = pending_distributed
                pending_distributed = None
                window_deck_target_shares = dict(pending_remote.deck_target_shares)
                performance_member_sources = _performance_member_sources(
                    pending_remote.settled_curriculum_state
                )
                distributed_assignment_pool = pending_remote.assignments
                active_opponent_pool_window = pending_remote.opponent_pool_window
                wait_started_at = time.perf_counter()
                try:
                    prepared_remote = _prepared_distributed_window(
                        pending_remote,
                        distributed_collection=distributed_collection,
                        config=config,
                    )
                    collected = prepared_remote.collected
                    prefetched_array_window = prepared_remote.array_window
                    preparation_seconds = prepared_remote.preparation_seconds
                    distributed_adopted_assignments = (
                        distributed_collection.adopt_quota_assignments()
                    )
                    unused_assignments = distributed_collection.unused_assignments()
                    if unused_assignments:
                        cancel_stateless_assignments(
                            unused_assignments,
                            curriculum=curriculum,
                            deck_balance=balance,
                        )
                except BaseException:
                    cancel_stateless_assignments(
                        (pending_remote.assignments + distributed_adopted_assignments),
                        curriculum=curriculum,
                        deck_balance=balance,
                    )
                    if (
                        opponent_pool_v2 is not None
                        and active_opponent_pool_window is not None
                    ):
                        opponent_pool_v2.abort(active_opponent_pool_window)
                    raise
                collection_wait_seconds = (
                    time.perf_counter()
                    - wait_started_at
                    + pending_remote.prepublication_wait_seconds
                )
                collection_seconds = max(
                    prepared_remote.collection_finished_at - pending_remote.started_at,
                    1.0e-9,
                )
                collection_overlap_seconds = max(
                    collection_seconds - collection_wait_seconds,
                    0.0,
                )
                native_assignments = collected.assignments
                behavior_version = pending_remote.behavior_version
                identity_seconds = pending_remote.identity_seconds
                identity_stage_seconds.update(pending_remote.identity_stage_seconds)
                if behavior_version != version - 1:
                    distributed_collection.abort(
                        reason="one-version-lag behavior age mismatch"
                    )
                    cancel_stateless_assignments(
                        native_assignments,
                        curriculum=curriculum,
                        deck_balance=balance,
                    )
                    if (
                        opponent_pool_v2 is not None
                        and active_opponent_pool_window is not None
                    ):
                        opponent_pool_v2.abort(active_opponent_pool_window)
                    raise RuntimeError(
                        "distributed one-version-lag collection has unexpected "
                        "behavior age"
                    )
            elif pending_collection is not None:
                pending = pending_collection
                pending_collection = None
                performance_member_sources = pending.member_sources
                wait_started_at = time.perf_counter()
                try:
                    timed_collection = pending.future.result()
                except BaseException as error:
                    try:
                        _release_prepared_collection(pending.prepared)
                    except BaseException as cleanup_error:
                        error.add_note(
                            "pending collection release also failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                    cancel_stateless_assignments(
                        pending.prepared.assignments,
                        curriculum=curriculum,
                        deck_balance=balance,
                    )
                    if parallel_collector is not None:
                        parallel_collector.close()
                    if native_process_collector is not None:
                        native_process_collector.close()
                    raise
                collection_wait_seconds = time.perf_counter() - wait_started_at
                collection_overlap_seconds = max(
                    timed_collection.elapsed_seconds - collection_wait_seconds,
                    0.0,
                )
                collected = timed_collection.result
                collection_seconds = timed_collection.elapsed_seconds
                native_assignments = pending.prepared.assignments
                behavior_version = pending.prepared.behavior_version
                identity_seconds = pending.prepared.identity_seconds
                if pipeline_enabled and behavior_version != version - 1:
                    if collected.compact_part_paths:
                        cleanup_parallel_collection_parts(collected.compact_part_paths)
                    cancel_stateless_assignments(
                        pending.prepared.assignments,
                        curriculum=curriculum,
                        deck_balance=balance,
                    )
                    raise RuntimeError(
                        "one-version-lag collection has unexpected behavior age"
                    )
            else:
                identity_started_at = time.perf_counter()
                fingerprint_started_at = time.perf_counter()
                model_fingerprint = canonical_model_state_fingerprint(learner.model)
                identity_stage_seconds["fingerprint_seconds"] = (
                    time.perf_counter() - fingerprint_started_at
                )
                behavior_identity = _fragment_identity(
                    resources,
                    behavior_version=version,
                    behavior_fingerprint=model_fingerprint,
                    horizon=config.collection.fragment_horizon,
                )
                if rollout_model is not None and rollout_model_version != version:
                    rollout_model.load_state_dict(
                        learner.model.state_dict(),
                        strict=True,
                    )
                    rollout_model_version = version
                if config.collection.backend == "native":
                    actor_model = (
                        learner.model if rollout_model is None else rollout_model
                    )
                    if actor_model.sequence is None:
                        native_actor = NativePolicyInferenceExecutor(
                            actor_model,
                            identity=behavior_identity,
                            device=config.device,
                            verify_model_state=False,
                        )
                    else:
                        native_actor = GeneralistSequenceActorPolicy(
                            actor_model,
                            identity=behavior_identity,
                            device=config.device,
                            verify_model_state=False,
                            retain_raw_blocks=False,
                            rollout_precision=(
                                config.collection.native_sequence_rollout_precision
                            ),
                            temporal_cache_slots=2
                            * (
                                config.collection.native_arena_capacity
                                or config.collection.concurrent_games
                            ),
                        )
                if config.collection.backend == "python_parallel":
                    actor_model = (
                        learner.model if rollout_model is None else rollout_model
                    )
                    if actor_model.sequence is None:
                        actor = SimpleStatelessActorPolicy(
                            actor_model,
                            identity=behavior_identity,
                            device=config.device,
                            verify_model_state=False,
                        )
                    else:
                        actor = GeneralistSequenceActorPolicy(
                            actor_model,
                            identity=behavior_identity,
                            device=config.device,
                            verify_model_state=False,
                        )
                identity_seconds = time.perf_counter() - identity_started_at
            cuda_memory.update(_cuda_memory_snapshot(learner.device, phase="identity"))
            collection_started_at = time.perf_counter()
            if collection_prefetched:
                pass
            elif distributed_backend:
                if behavior_identity is None or distributed_collection is None:
                    raise RuntimeError(
                        "distributed collection resources are not initialized"
                    )
                assignment_started_at = time.perf_counter()
                if opponent_pool_v2 is None:
                    distributed_assignment_pool = assign_stateless_games(
                        games=distributed_collection.assignment_pool_games(),
                        curriculum=curriculum,
                        deck_balance=balance,
                        active_decks=resources.active_decks,
                        opponent_decks=resources.opponent_decks,
                        mirror_policy_fingerprint=(
                            behavior_identity.behavior_policy_fingerprint
                        ),
                    )
                    distributed_assignment_plan = distributed_assignment_pool
                elif config.collection.native_shard_protocol_version == 3:
                    quota_plan = plan_stateless_assignment_quotas(
                        games=distributed_collection.assignment_pool_games(),
                        opponent_pool=opponent_pool_v2,
                        curriculum=curriculum,
                        deck_balance=balance,
                        active_decks=resources.active_decks,
                        opponent_decks=resources.opponent_decks,
                        mirror_policy_fingerprint=(
                            behavior_identity.behavior_policy_fingerprint
                        ),
                        minimum_cohort_games=(
                            distributed_collection.quota_exposure_games()
                        ),
                    )
                    distributed_assignment_plan = quota_plan
                    active_opponent_pool_window = quota_plan.opponent_window
                    window_deck_target_shares = dict(quota_plan.candidate_target_shares)
                else:
                    planned_assignments = assign_stateless_games_v2(
                        games=distributed_collection.assignment_pool_games(),
                        opponent_pool=opponent_pool_v2,
                        curriculum=curriculum,
                        deck_balance=balance,
                        active_decks=resources.active_decks,
                        opponent_decks=resources.opponent_decks,
                        mirror_policy_fingerprint=(
                            behavior_identity.behavior_policy_fingerprint
                        ),
                    )
                    distributed_assignment_pool = planned_assignments.assignments
                    active_opponent_pool_window = (
                        planned_assignments.opponent_pool_window
                    )
                    distributed_assignment_plan = distributed_assignment_pool
                identity_stage_seconds["assignment_planning_seconds"] = (
                    time.perf_counter() - assignment_started_at
                )
                try:
                    collected = distributed_collection.collect(
                        model_state=learner.model.state_dict(),
                        model_config=resources.model_config,
                        behavior_identity=behavior_identity,
                        assignments=distributed_assignment_plan,
                        curriculum_members=curriculum.state.members,
                        opponent_pool_revision=(
                            None
                            if active_opponent_pool_window is None
                            else active_opponent_pool_window.plan.revision_fingerprint
                        ),
                    )
                    distributed_adopted_assignments = (
                        distributed_collection.adopt_quota_assignments()
                    )
                    unused_assignments = distributed_collection.unused_assignments()
                    if unused_assignments:
                        cancel_stateless_assignments(
                            unused_assignments,
                            curriculum=curriculum,
                            deck_balance=balance,
                        )
                    native_assignments = collected.assignments
                    backend_timing = distributed_collection.last_preparation_timing
                    identity_stage_seconds["assignment_planning_seconds"] += (
                        backend_timing.assignment_planning_seconds
                    )
                    identity_stage_seconds["artifact_source_fingerprint_seconds"] = (
                        backend_timing.source_fingerprint_seconds
                    )
                    identity_stage_seconds["bf16_conversion_seconds"] = (
                        backend_timing.bf16_conversion_seconds
                    )
                    identity_stage_seconds["wire_hashing_seconds"] = (
                        backend_timing.wire_hashing_seconds
                    )
                    identity_stage_seconds["window_open_seconds"] = (
                        backend_timing.window_open_seconds
                    )
                    identity_seconds = sum(identity_stage_seconds.values())
                except BaseException:
                    cancel_stateless_assignments(
                        (distributed_assignment_pool + distributed_adopted_assignments),
                        curriculum=curriculum,
                        deck_balance=balance,
                    )
                    if (
                        opponent_pool_v2 is not None
                        and active_opponent_pool_window is not None
                    ):
                        opponent_pool_v2.abort(active_opponent_pool_window)
                    raise
            elif config.collection.backend == "hybrid":
                if (
                    behavior_identity is None
                    or native_historical is None
                    or parallel_collector is None
                    or parallel_temporary_root is None
                    or rollout_model is None
                ):
                    raise RuntimeError("hybrid collector resources are not initialized")
                prepared_collection = _prepare_hybrid_collection(
                    config=config,
                    resources=resources,
                    curriculum=curriculum,
                    balance=balance,
                    past_self=past_self,
                    native_historical=native_historical,
                    parallel_collector=parallel_collector,
                    native_process_collector=native_process_collector,
                    parallel_temporary_root=parallel_temporary_root,
                    behavior_model=rollout_model,
                    behavior_identity=behavior_identity,
                    identity_started_at=identity_started_at,
                    policy_stream_pool=native_policy_stream_pool,
                )
                native_assignments = prepared_collection.assignments
                try:
                    timed_collection = _timed_collection(prepared_collection.collect)
                    collected = timed_collection.result
                    collection_seconds = timed_collection.elapsed_seconds
                except BaseException:
                    cancel_stateless_assignments(
                        prepared_collection.assignments,
                        curriculum=curriculum,
                        deck_balance=balance,
                    )
                    parallel_collector.close()
                    if native_process_collector is not None:
                        native_process_collector.close()
                    raise
            elif config.collection.backend in _IN_PROCESS_NATIVE_BACKENDS:
                if behavior_identity is None or native_historical is None:
                    raise RuntimeError("native collector resources are not initialized")
                assignments = assign_stateless_games(
                    games=config.collection.concurrent_games,
                    curriculum=curriculum,
                    deck_balance=balance,
                    active_decks=resources.active_decks,
                    opponent_decks=resources.opponent_decks,
                    mirror_policy_fingerprint=(
                        behavior_identity.behavior_policy_fingerprint
                    ),
                )
                native_assignments = assignments
                try:
                    if native_actor is None:
                        actor_model = (
                            learner.model if rollout_model is None else rollout_model
                        )
                        if actor_model.sequence is None:
                            native_actor = NativePolicyInferenceExecutor(
                                actor_model,
                                identity=behavior_identity,
                                device=config.device,
                                verify_model_state=False,
                            )
                        else:
                            temporal_cache_slots = sum(
                                1 + int(assignment.curriculum.lane == "mirror")
                                for assignment in assignments
                            )
                            native_actor = GeneralistSequenceActorPolicy(
                                actor_model,
                                identity=behavior_identity,
                                device=config.device,
                                verify_model_state=False,
                                retain_raw_blocks=False,
                                rollout_precision=(
                                    config.collection.native_sequence_rollout_precision
                                ),
                                temporal_cache_slots=temporal_cache_slots,
                            )
                    native_collector = _build_in_process_native_collector(
                        config=config,
                        resources=resources,
                        curriculum_state=curriculum.state,
                        past_self=past_self,
                        native_historical=native_historical,
                        native_scripted=native_scripted,
                        native_scripted_bindings=native_scripted_bindings,
                        actor=native_actor,
                        behavior_identity=behavior_identity,
                        seed=config.collection.seed + version,
                        policy_stream_pool=native_policy_stream_pool,
                    )
                    if native_process_collector is None:
                        if config.collection.backend == "native_banked":
                            collected = collect_native_banked_assigned(
                                native_collector,
                                assignments,
                            )
                        else:
                            collected = native_collector.collect_assigned(assignments)
                    else:
                        if native_process_temporary_root is None:
                            raise RuntimeError(
                                "native process temporary root is absent"
                            )
                        collected = native_process_collector.collect(
                            actor=native_actor,
                            identity=behavior_identity,
                            assignments=assignments,
                            members=curriculum.state.members,
                            temporary_root=(
                                native_process_temporary_root / f"update-{version:08d}"
                            ),
                            seed=config.collection.seed + version,
                            arena_capacity=(
                                config.collection.native_arena_capacity
                                or len(assignments)
                            ),
                            trainable_decision_budget=(
                                config.collection.native_trainable_decision_budget
                            ),
                            frozen_batch_min_rows=(
                                config.collection.native_frozen_batch_min_rows
                            ),
                            frozen_batch_max_wait_waves=(
                                config.collection.native_frozen_batch_max_wait_waves
                            ),
                        )
                except BaseException as error:
                    cancel_stateless_assignments(
                        assignments,
                        curriculum=curriculum,
                        deck_balance=balance,
                    )
                    if native_process_collector is not None:
                        native_process_collector.close()
                    try:
                        _close_native_collection_window(
                            collector=native_collector,
                            actor=native_actor,
                        )
                    except BaseException as cleanup_error:
                        error.add_note(
                            "native collection window teardown also failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                    finally:
                        native_collector = None
                        native_actor = None
                    raise
            elif parallel_collector is None:
                if behavior_identity is None or actor is None or historical is None:
                    raise RuntimeError("Python collector resources are not initialized")
                collector = StatelessEngineCollector(
                    actor=actor,
                    identity=behavior_identity,
                    candidate_contract=resources.input_contract,
                    catalog=resources.catalog,
                    active_decks=resources.active_decks,
                    opponent_decks=resources.opponent_decks,
                    curriculum=curriculum,
                    deck_balance=balance,
                    past_self_pool=past_self,
                    historical_pool=historical,
                    scripted={
                        item.opponent_id: item for item in config.curriculum.scripted
                    },
                    maximum_engine_steps=config.collection.maximum_engine_steps,
                    seed=config.collection.seed + version,
                    mirror_bilateral_trajectories=(
                        config.collection.mirror_bilateral_trajectories
                    ),
                    engine_fact_producer=(
                        ProspectiveEngineFactProducer(
                            sampler=BeliefSampler(
                                config=learner.model.config.sequence.engine_facts.sampler
                            ),
                            config=(learner.model.config.sequence.engine_facts),
                        )
                        if learner.model.config.sequence is not None
                        and learner.model.config.sequence.engine_facts.enabled
                        else None
                    ),
                )
                collected = collector.collect(games=config.collection.concurrent_games)
                _record_performance_outcomes(
                    performance_reporter,
                    collected=collected,
                    active_deck_labels=resources.active_deck_labels,
                    policy_version=version,
                    member_sources=performance_member_sources,
                )
            else:
                if behavior_identity is None or actor is None:
                    raise RuntimeError("parallel collector actor is not initialized")
                if parallel_temporary_root is None:
                    raise RuntimeError("parallel temporary root is not initialized")
                assignments = assign_stateless_games(
                    games=config.collection.concurrent_games,
                    curriculum=curriculum,
                    deck_balance=balance,
                    active_decks=resources.active_decks,
                    opponent_decks=resources.opponent_decks,
                    mirror_policy_fingerprint=(
                        behavior_identity.behavior_policy_fingerprint
                    ),
                )
                try:
                    collected = parallel_collector.collect(
                        actor=actor,
                        identity=behavior_identity,
                        assignments=assignments,
                        members=curriculum.state.members,
                        temporary_root=(
                            parallel_temporary_root / f"update-{version:08d}"
                        ),
                        fragments_per_part=(config.collection.fragments_per_part),
                    )
                except BaseException:
                    cancel_stateless_assignments(
                        assignments,
                        curriculum=curriculum,
                        deck_balance=balance,
                    )
                    parallel_collector.close()
                    raise
                commit_stateless_outcomes(
                    assignments,
                    collected.outcomes,
                    curriculum=curriculum,
                    deck_balance=balance,
                )
                _record_performance_outcomes(
                    performance_reporter,
                    collected=collected,
                    active_deck_labels=resources.active_deck_labels,
                    policy_version=version,
                    member_sources=performance_member_sources,
                )
            if not collection_prefetched and config.collection.backend != "hybrid":
                collection_seconds = time.perf_counter() - collection_started_at
            try:
                cuda_memory.update(
                    _cuda_memory_snapshot(learner.device, phase="collection")
                )
                if not collected.fragments and not collected.compact_parts:
                    if native_assignments is not None:
                        cancel_stateless_assignments(
                            native_assignments,
                            curriculum=curriculum,
                            deck_balance=balance,
                        )
                    raise RuntimeError(
                        "stateless collection produced no trainable fragments"
                    )
                if collected.fragments and collected.compact_parts:
                    if native_assignments is not None:
                        cancel_stateless_assignments(
                            native_assignments,
                            curriculum=curriculum,
                            deck_balance=balance,
                        )
                    raise RuntimeError(
                        "stateless collection mixed object and array fragment payloads"
                    )
                if distributed_backend and not collected.compact_parts:
                    raise RuntimeError(
                        "distributed collection requires compact array parts"
                    )
                collection_reports.append(collected.report)
                _run_noncritical(
                    "collection status publication",
                    _write_collection_status,
                    output_dir,
                    run_version=config.run.version,
                    behavior_policy_version=behavior_version,
                    report=collected.report,
                )
                fragment_count = (
                    sum(part.fragment_count for part in collected.compact_parts)
                    if collected.compact_parts
                    else len(collected.fragments)
                )
                fragments_seen += fragment_count
            finally:
                try:
                    _close_native_collection_window(
                        collector=native_collector,
                        actor=native_actor,
                    )
                finally:
                    native_collector = None
                    native_actor = None
            behavior_age_fragments: dict[str, int] = {}
            outcomes_committed = False
            launched_pending: _PendingCollection | None = None
            launched_distributed: _PendingDistributedCollection | None = None
            current_artifact_future: Future[_PreparedCurrentRolloutArtifact] | None = (
                None
            )
            if pending_checkpoint is not None and not distributed_backend:
                last_pair = _settle_pending_checkpoint(
                    pending_checkpoint,
                    publisher=checkpoint_publisher,
                    output_dir=output_dir,
                    config=config,
                    curriculum=curriculum,
                    learner_metric_writer=learner_metric_writer,
                )
                pending_checkpoint = None
            if collected.compact_parts:
                try:
                    retained_decisions = _retained_compact_decision_count(
                        collected.compact_parts,
                        current_policy_version=version,
                        maximum_version_age=config.ppo.maximum_version_age,
                    )
                    artifact_prefetch_candidate = (
                        distributed_backend
                        and _pipeline_can_prefetch(
                            learner,
                            decisions=retained_decisions,
                            config=config,
                            curriculum_state=curriculum.state,
                            allow_checkpoint_boundary=True,
                            allow_admission_boundary=True,
                        )
                    )
                    if artifact_prefetch_candidate:
                        if (
                            distributed_collection is None
                            or distributed_artifact_executor is None
                        ):
                            raise RuntimeError(
                                "distributed artifact prefetch has no executor"
                            )
                        current_artifact_future = distributed_artifact_executor.submit(
                            _prepare_current_rollout_artifact,
                            learner=learner,
                            resources=resources,
                            distributed_collection=distributed_collection,
                        )
                    if distributed_backend and not outcomes_committed:
                        if native_assignments is None or distributed_collection is None:
                            raise RuntimeError(
                                "distributed collection omitted central assignments"
                            )
                        distributed_receipt = distributed_collection.commit(
                            partial(
                                _commit_expected_stateless_outcomes,
                                expected_assignments=native_assignments,
                                curriculum=curriculum,
                                balance=balance,
                            )
                        )
                        if opponent_pool_v2 is not None:
                            if active_opponent_pool_window is None:
                                raise RuntimeError(
                                    "V2 distributed collection omitted its pool plan"
                                )
                            if isinstance(
                                active_opponent_pool_window,
                                BoundOpponentQuotaWindow,
                            ):
                                opponent_pool_v2.commit_quota(
                                    active_opponent_pool_window,
                                    collected.outcomes,
                                )
                            else:
                                opponent_pool_v2.commit(
                                    active_opponent_pool_window,
                                    collected.outcomes,
                                )
                        elif active_opponent_pool_window is not None:
                            raise RuntimeError(
                                "legacy distributed collection carried a V2 plan"
                            )
                        outcomes_committed = True
                        distributed_collection.settle_receipt(distributed_receipt)
                        _record_performance_outcomes(
                            performance_reporter,
                            collected=collected,
                            active_deck_labels=resources.active_deck_labels,
                            policy_version=behavior_version,
                            member_sources=performance_member_sources,
                        )
                        distributed_collection.release()
                        distributed_prefetch_allowed = _pipeline_can_prefetch(
                            learner,
                            decisions=retained_decisions,
                            config=config,
                            curriculum_state=curriculum.state,
                            allow_checkpoint_boundary=True,
                            allow_admission_boundary=True,
                        )
                        if distributed_prefetch_allowed:
                            launched_distributed = _begin_distributed_prefetch(
                                learner=learner,
                                config=config,
                                resources=resources,
                                curriculum=curriculum,
                                balance=balance,
                                opponent_pool_v2=opponent_pool_v2,
                                distributed_collection=distributed_collection,
                                current_artifact_future=current_artifact_future,
                            )
                            current_artifact_future = None
                            if distributed_prepare_executor is None:
                                raise RuntimeError(
                                    "distributed prefetch has no preparation executor"
                                )
                            launched_distributed = replace(
                                launched_distributed,
                                preparation_future=distributed_prepare_executor.submit(
                                    _wait_and_prepare_distributed_window,
                                    distributed_collection,
                                    current_policy_version=(
                                        launched_distributed.behavior_version + 1
                                    ),
                                    maximum_version_age=(
                                        config.ppo.maximum_version_age
                                    ),
                                    gamma=config.ppo.gamma,
                                    gae_lambda=config.ppo.gae_lambda,
                                    normalize_epsilon=config.ppo.normalize_epsilon,
                                    deck_target_shares=(
                                        launched_distributed.deck_target_shares
                                    ),
                                ),
                            )
                            pending_distributed = launched_distributed
                        elif current_artifact_future is not None:
                            current_artifact_future.cancel()
                    # The submitted pair owns detached optimizer tensors and
                    # immutable controller snapshots. Let its durable write
                    # overlap outcome settlement, assignment planning, and
                    # opening the next remote window. The barrier remains
                    # before the next optimizer mutation, and this try block
                    # aborts the speculative window if publication failed.
                    if pending_checkpoint is not None:
                        last_pair = _settle_pending_checkpoint(
                            pending_checkpoint,
                            publisher=checkpoint_publisher,
                            output_dir=output_dir,
                            config=config,
                            curriculum=curriculum,
                            learner_metric_writer=learner_metric_writer,
                        )
                        pending_checkpoint = None
                    if prefetched_array_window is None:
                        preparation_started_at = time.perf_counter()
                        array_window = prepare_stateless_array_optimizer_window(
                            collected.compact_parts,
                            current_policy_version=version,
                            maximum_version_age=config.ppo.maximum_version_age,
                            gamma=config.ppo.gamma,
                            gae_lambda=config.ppo.gae_lambda,
                            normalize_epsilon=config.ppo.normalize_epsilon,
                            deck_target_shares=window_deck_target_shares,
                        )
                        preparation_seconds = (
                            time.perf_counter() - preparation_started_at
                        )
                    else:
                        array_window = prefetched_array_window
                    cuda_memory.update(
                        _cuda_memory_snapshot(
                            learner.device,
                            phase="preparation",
                        )
                    )
                    if (
                        config.collection.backend in _IN_PROCESS_NATIVE_BACKENDS
                        or distributed_backend
                    ):
                        persistence_seconds = 0.0
                    else:
                        disk_backed_parts = tuple(
                            part
                            for part in collected.compact_parts
                            if part.path is not None
                        )
                        disk_backed_paths = tuple(
                            part.path
                            for part in disk_backed_parts
                            if part.path is not None
                        )
                        if len(disk_backed_paths) != len(
                            collected.compact_part_paths
                        ) or set(disk_backed_paths) != set(
                            collected.compact_part_paths
                        ):
                            raise RuntimeError(
                                "disk-backed compact parts differ from worker paths"
                            )
                        if config.collection.backend == "hybrid":
                            persistence_seconds = 0.0
                        else:
                            persistence_started_at = time.perf_counter()
                            for part in disk_backed_parts:
                                writer.import_validated_part(
                                    part,
                                    static_contract_fingerprint=(
                                        array_window.static_contract_fingerprint
                                    ),
                                )
                            cleanup_parallel_collection_parts(
                                collected.compact_part_paths
                            )
                            persistence_seconds = (
                                time.perf_counter() - persistence_started_at
                            )
                    fragments_stale += array_window.fragments_stale
                    behavior_age_fragments = {
                        str(age): count
                        for age, count in sorted(
                            Counter(
                                int(value)
                                for value in array_window.behavior_version_ages
                            ).items()
                        )
                    }
                    if pipeline_enabled and not distributed_backend:
                        if native_assignments is None:
                            raise RuntimeError(
                                "pipelined collection omitted central assignments"
                            )
                        commit_stateless_outcomes(
                            native_assignments,
                            collected.outcomes,
                            curriculum=curriculum,
                            deck_balance=balance,
                        )
                        outcomes_committed = True
                        _record_performance_outcomes(
                            performance_reporter,
                            collected=collected,
                            active_deck_labels=resources.active_deck_labels,
                            policy_version=behavior_version,
                            member_sources=performance_member_sources,
                        )
                        if _pipeline_can_prefetch(
                            learner,
                            decisions=array_window.decision_count,
                            config=config,
                            curriculum_state=curriculum.state,
                        ):
                            if native_historical is None or pipeline_executor is None:
                                raise RuntimeError(
                                    "pipeline collection resources are not initialized"
                                )
                            next_identity_started_at = time.perf_counter()
                            next_fingerprint = canonical_model_state_fingerprint(
                                learner.model
                            )
                            next_identity = _fragment_identity(
                                resources,
                                behavior_version=learner.update_index,
                                behavior_fingerprint=next_fingerprint,
                                horizon=config.collection.fragment_horizon,
                            )
                            if config.collection.backend == "native_banked":
                                if pipeline_collection_stream is None:
                                    raise RuntimeError(
                                        "pipeline collection CUDA stream is absent"
                                    )
                                next_collection = _prepare_in_process_native_collection(
                                    config=config,
                                    resources=resources,
                                    curriculum=curriculum,
                                    balance=balance,
                                    past_self=past_self,
                                    native_historical=native_historical,
                                    native_scripted=native_scripted,
                                    native_scripted_bindings=(native_scripted_bindings),
                                    learner_model=learner.model,
                                    behavior_identity=next_identity,
                                    identity_started_at=(next_identity_started_at),
                                    cuda_stream=pipeline_collection_stream,
                                    policy_stream_pool=(native_policy_stream_pool),
                                )
                            elif config.collection.backend == "hybrid":
                                if (
                                    rollout_model is None
                                    or parallel_collector is None
                                    or parallel_temporary_root is None
                                ):
                                    raise RuntimeError(
                                        "hybrid pipeline resources are not initialized"
                                    )
                                if rollout_model_version != learner.update_index:
                                    rollout_model.load_state_dict(
                                        learner.model.state_dict(),
                                        strict=True,
                                    )
                                    rollout_model_version = learner.update_index
                                    if pipeline_collection_stream is None:
                                        raise RuntimeError(
                                            "pipeline collection CUDA stream is absent"
                                        )
                                    pipeline_collection_stream.wait_stream(
                                        torch.cuda.current_stream(device=learner.device)
                                    )
                                next_collection = _prepare_hybrid_collection(
                                    config=config,
                                    resources=resources,
                                    curriculum=curriculum,
                                    balance=balance,
                                    past_self=past_self,
                                    native_historical=native_historical,
                                    parallel_collector=parallel_collector,
                                    native_process_collector=(native_process_collector),
                                    parallel_temporary_root=(parallel_temporary_root),
                                    behavior_model=rollout_model,
                                    behavior_identity=next_identity,
                                    identity_started_at=(next_identity_started_at),
                                    policy_stream_pool=(native_policy_stream_pool),
                                )
                            else:
                                raise RuntimeError(
                                    "unsupported one-version-lag backend"
                                )
                            try:
                                future = pipeline_executor.submit(
                                    _timed_collection,
                                    next_collection.collect,
                                )
                            except BaseException as error:
                                try:
                                    _release_prepared_collection(next_collection)
                                except BaseException as cleanup_error:
                                    error.add_note(
                                        "unstarted collection release also failed: "
                                        f"{type(cleanup_error).__name__}: "
                                        f"{cleanup_error}"
                                    )
                                cancel_stateless_assignments(
                                    next_collection.assignments,
                                    curriculum=curriculum,
                                    deck_balance=balance,
                                )
                                raise
                            launched_pending = _PendingCollection(
                                prepared=next_collection,
                                future=future,
                                member_sources=_performance_member_sources(
                                    curriculum.state
                                ),
                            )
                            pending_collection = launched_pending
                    learner_started_at = time.perf_counter()
                    learner.set_deck_macro_target_shares(window_deck_target_shares)
                    update = learner.update_array(array_window)
                    learner_seconds = time.perf_counter() - learner_started_at
                    if collected.compact_part_paths:
                        persistence_started_at = time.perf_counter()
                        cleanup_parallel_collection_parts(collected.compact_part_paths)
                        persistence_seconds = (
                            time.perf_counter() - persistence_started_at
                        )
                    cuda_memory.update(
                        _cuda_memory_snapshot(learner.device, phase="learner")
                    )
                except BaseException as error:
                    if collected.compact_part_paths:
                        try:
                            cleanup_parallel_collection_parts(
                                collected.compact_part_paths
                            )
                        except BaseException as cleanup_error:
                            error.add_note(
                                "hybrid fragment transport cleanup also failed: "
                                f"{type(cleanup_error).__name__}: {cleanup_error}"
                            )
                    if (
                        launched_distributed is not None
                        and distributed_collection is not None
                    ):
                        distributed_collection.abort(
                            reason=f"{type(error).__name__}: {error}"
                        )
                        cancel_stateless_assignments(
                            launched_distributed.assignments,
                            curriculum=curriculum,
                            deck_balance=balance,
                        )
                        if (
                            opponent_pool_v2 is not None
                            and launched_distributed.opponent_pool_window is not None
                        ):
                            opponent_pool_v2.abort(
                                launched_distributed.opponent_pool_window
                            )
                        pending_distributed = None
                    elif distributed_backend and distributed_collection is not None:
                        if not outcomes_committed:
                            distributed_collection.abort(
                                reason=f"{type(error).__name__}: {error}"
                            )
                        else:
                            distributed_collection.release()
                    if native_assignments is not None and not outcomes_committed:
                        cancel_stateless_assignments(
                            native_assignments,
                            curriculum=curriculum,
                            deck_balance=balance,
                        )
                    if (
                        opponent_pool_v2 is not None
                        and active_opponent_pool_window is not None
                        and not outcomes_committed
                    ):
                        opponent_pool_v2.abort(active_opponent_pool_window)
                    if launched_pending is not None:
                        _discard_pending_collection(
                            launched_pending,
                            curriculum=curriculum,
                            balance=balance,
                        )
                        pending_collection = None
                    if parallel_collector is not None:
                        parallel_collector.close()
                    if native_process_collector is not None:
                        native_process_collector.close()
                    raise
                if native_assignments is not None and not outcomes_committed:
                    commit_stateless_outcomes(
                        native_assignments,
                        collected.outcomes,
                        curriculum=curriculum,
                        deck_balance=balance,
                    )
                    _record_performance_outcomes(
                        performance_reporter,
                        collected=collected,
                        active_deck_labels=resources.active_deck_labels,
                        policy_version=behavior_version,
                        member_sources=performance_member_sources,
                    )
                del array_window
                del collected
            else:
                persistence_started_at = time.perf_counter()
                for fragment in collected.fragments:
                    writer.add(fragment)
                writer.flush()
                persistence_seconds = time.perf_counter() - persistence_started_at
                preparation_started_at = time.perf_counter()
                object_window = prepare_stateless_optimizer_window(
                    collected.fragments,
                    current_policy_version=version,
                    maximum_version_age=config.ppo.maximum_version_age,
                    gamma=config.ppo.gamma,
                    gae_lambda=config.ppo.gae_lambda,
                    normalize_epsilon=config.ppo.normalize_epsilon,
                    deck_target_shares=window_deck_target_shares,
                )
                preparation_seconds = time.perf_counter() - preparation_started_at
                fragments_stale += object_window.fragments_stale
                learner_started_at = time.perf_counter()
                learner.set_deck_macro_target_shares(window_deck_target_shares)
                update = learner.update(object_window)
                learner_seconds = time.perf_counter() - learner_started_at
                cuda_memory.update(
                    _cuda_memory_snapshot(learner.device, phase="learner")
                )
            update_reports.append(update)
            budget_reached = _training_budget_reached(learner, config=config)
            checkpoint_due = _should_publish_checkpoint(
                learner.update_index,
                final=budget_reached,
                interval=config.collection.checkpoint_interval_updates,
                retain_every_versions=(
                    config.collection.checkpoint_retain_every_versions
                ),
            )
            admission_due = _past_self_admission_due(
                learner.update_index,
                config=config,
                final=budget_reached,
                curriculum_state=curriculum.state,
            )
            if (
                admission_due
                and pending_distributed is not None
                and config.collection.native_shard_protocol_version == 3
            ):
                if distributed_collection is None:
                    raise RuntimeError(
                        "admission prefetch lost its distributed backend"
                    )

                (
                    prefetched_policy_artifact,
                    adopted,
                    overlap_timing,
                ) = _run_overlapping_calls(
                    background=partial(
                        publish_verified_stateless_policy_checkpoint,
                        output_dir,
                        version=learner.update_index,
                        model=learner.model,
                        model_config=resources.model_config,
                        identity=resources.policy_identity,
                    ),
                    foreground=partial(
                        _settle_distributed_admission_prefetch,
                        pending_distributed,
                        distributed_collection=distributed_collection,
                        config=config,
                    ),
                    thread_name_prefix="policy-checkpoint",
                )
                admission_prefetch_wait_seconds = overlap_timing.foreground_seconds
                policy_prefetch_publication_seconds = overlap_timing.background_seconds
                policy_prefetch_wait_seconds = overlap_timing.background_wait_seconds
                policy_prefetch_overlap_seconds = overlap_timing.overlap_seconds
                pending_distributed = replace(
                    pending_distributed,
                    assignments=adopted,
                    prepublication_wait_seconds=(admission_prefetch_wait_seconds),
                )
                if launched_distributed is not None:
                    launched_distributed = pending_distributed
            should_publish = checkpoint_due or admission_due
            if should_publish and pending_collection is not None:
                raise RuntimeError(
                    "local publication cannot include speculative collection leases"
                )
            publication_pending = pending_distributed if should_publish else None
            if (
                publication_pending is not None
                and publication_pending is not launched_distributed
            ):
                raise RuntimeError(
                    "publication lost ownership of its distributed prefetch"
                )
            publication_curriculum_state = (
                curriculum.state
                if publication_pending is None
                else publication_pending.settled_curriculum_state
            )
            publication_balance_state = (
                balance.state
                if publication_pending is None
                else publication_pending.settled_balance_state
            )
            publication_opponent_pool_state = (
                None
                if opponent_pool_v2 is None
                else opponent_pool_v2.state
                if publication_pending is None
                else publication_pending.settled_opponent_pool_state
            )
            allocator_trim_seconds = 0.0
            if distributed_backend or (should_publish and pipeline_enabled):
                allocator_trim_started_at = time.perf_counter()
                cuda_memory.update(
                    _trim_cuda_allocator_at_pipeline_drain(learner.device)
                )
                allocator_trim_seconds = time.perf_counter() - allocator_trim_started_at
            checkpoint_started_at = time.perf_counter()
            submitted_checkpoint: (
                tuple[
                    Future[PublishedStatelessCheckpointPair],
                    StatelessCurriculumState,
                    str,
                ]
                | None
            ) = None
            if should_publish:
                external_predecessor = curriculum.persisted_state
                if external_predecessor is None:
                    raise RuntimeError(
                        "checkpointed training has no external curriculum predecessor"
                    )
                external_predecessor_fingerprint = (
                    stateless_curriculum_state_fingerprint(external_predecessor)
                )
                if admission_due:
                    policy_artifact = prefetched_policy_artifact
                    if policy_artifact is None:
                        policy_artifact = publish_verified_stateless_policy_checkpoint(
                            output_dir,
                            version=learner.update_index,
                            model=learner.model,
                            model_config=resources.model_config,
                            identity=resources.policy_identity,
                        )
                    reentry_artifact = _past_self_reentry_artifact(
                        output_dir,
                        config=config,
                        curriculum=curriculum,
                        current_version=learner.update_index,
                    )
                    admission_artifact = reentry_artifact or policy_artifact
                    snapshot_id = (
                        f"past-self-v{learner.update_index}"
                        if reentry_artifact is None
                        else curriculum.next_past_self_reentry_snapshot_id(
                            reentry_artifact
                        )
                    )
                    prepared = curriculum.prepare_past_self_admission(
                        admission_artifact,
                        snapshot_id=snapshot_id,
                        current_version=learner.update_index,
                        sampling_floor=config.curriculum.pfsp.probability_floor,
                    )
                    live_admitted_state = curriculum.preview_past_self_admission(
                        prepared
                    )
                    published_admitted_state = (
                        live_admitted_state
                        if publication_pending is None
                        else curriculum.preview_settled_past_self_admission(
                            prepared,
                            publication_curriculum_state,
                        )
                    )
                    try:
                        checkpoint_future = _submit_pair(
                            output_dir,
                            publisher=checkpoint_publisher,
                            learner=learner,
                            resources=resources,
                            writer=writer,
                            balance=balance,
                            balance_state=publication_balance_state,
                            curriculum_state=published_admitted_state,
                            opponent_pool_state=(publication_opponent_pool_state),
                            curriculum_predecessor_state=external_predecessor,
                            policy_publication=policy_artifact,
                            fragments_seen=fragments_seen,
                            fragments_stale=fragments_stale,
                            startup_provenance=checkpoint_startup_provenance,
                        )
                    except BaseException:
                        prepared.routes.abort()
                        raise
                    curriculum.commit_past_self_admission(prepared)
                    if curriculum.state != live_admitted_state:
                        raise RuntimeError(
                            "committed curriculum differs from its live preview"
                        )
                    published_curriculum_state = published_admitted_state
                else:
                    checkpoint_future = _submit_pair(
                        output_dir,
                        publisher=checkpoint_publisher,
                        learner=learner,
                        resources=resources,
                        writer=writer,
                        balance=balance,
                        balance_state=publication_balance_state,
                        curriculum_state=publication_curriculum_state,
                        opponent_pool_state=publication_opponent_pool_state,
                        curriculum_predecessor_state=external_predecessor,
                        fragments_seen=fragments_seen,
                        fragments_stale=fragments_stale,
                        startup_provenance=checkpoint_startup_provenance,
                    )
                    published_curriculum_state = publication_curriculum_state
                submitted_checkpoint = (
                    checkpoint_future,
                    published_curriculum_state,
                    external_predecessor_fingerprint,
                )
            checkpoint_seconds = time.perf_counter() - checkpoint_started_at
            cuda_memory.update(
                _cuda_memory_snapshot(learner.device, phase="checkpoint")
            )
            total_seconds = time.perf_counter() - window_started_at
            timing = {
                "update_index": learner.update_index,
                "kept_decisions": update.decisions,
                "optimizer_steps": update.optimizer_steps,
                "optimizer_step_index": learner.optimizer_step_index,
                "fresh_decisions_seen": learner.fresh_decisions_seen,
                "lr_schedule_decisions_seen": (learner.lr_schedule_decisions_seen),
                "identity_seconds": identity_seconds,
                **identity_stage_seconds,
                "collection_seconds": collection_seconds,
                "collection_prefetched": collection_prefetched,
                "collection_wait_seconds": collection_wait_seconds,
                "collection_overlap_seconds": collection_overlap_seconds,
                "admission_prefetch_wait_seconds": (admission_prefetch_wait_seconds),
                "policy_prefetch_publication_seconds": (
                    policy_prefetch_publication_seconds
                ),
                "policy_prefetch_wait_seconds": policy_prefetch_wait_seconds,
                "policy_prefetch_overlap_seconds": (policy_prefetch_overlap_seconds),
                "next_collection_prefetched": (
                    launched_pending is not None or launched_distributed is not None
                ),
                "learner_policy_version": version,
                "behavior_policy_version": behavior_version,
                "behavior_policy_version_age": version - behavior_version,
                "behavior_version_age_fragments": behavior_age_fragments,
                "persistence_seconds": persistence_seconds,
                "preparation_seconds": preparation_seconds,
                "learner_seconds": learner_seconds,
                "host_prepare_task_seconds": update.host_prepare_task_seconds,
                "host_prepare_wait_seconds": update.host_prepare_wait_seconds,
                "allocator_trim_seconds": allocator_trim_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "total_seconds": total_seconds,
                "kept_decisions_per_second": update.decisions / total_seconds,
                **cuda_memory,
            }
            timing_reports.append(timing)
            if submitted_checkpoint is not None:
                checkpoint_future, published_state, predecessor_fingerprint = (
                    submitted_checkpoint
                )
                pending_checkpoint = _PendingCheckpoint(
                    future=checkpoint_future,
                    published_curriculum_state=published_state,
                    predecessor_fingerprint=predecessor_fingerprint,
                    timing=timing,
                    update=update,
                )
            _run_noncritical(
                "learner status publication",
                _write_status,
                output_dir,
                config=config,
                learner=learner,
                collection_reports=collection_reports,
                update_reports=update_reports,
                timing_reports=timing_reports,
                balance=balance,
                curriculum=curriculum,
                opponent_pool_v2=opponent_pool_v2,
                started_at=started_at,
                latest_pair=last_pair,
                learner_metric_writer=learner_metric_writer,
                status_writer=status_writer,
            )
        if pending_checkpoint is not None:
            last_pair = _settle_pending_checkpoint(
                pending_checkpoint,
                publisher=checkpoint_publisher,
                output_dir=output_dir,
                config=config,
                curriculum=curriculum,
                learner_metric_writer=learner_metric_writer,
            )
            pending_checkpoint = None
            _run_noncritical(
                "learner status publication",
                _write_status,
                output_dir,
                config=config,
                learner=learner,
                collection_reports=collection_reports,
                update_reports=update_reports,
                timing_reports=timing_reports,
                balance=balance,
                curriculum=curriculum,
                opponent_pool_v2=opponent_pool_v2,
                started_at=started_at,
                latest_pair=last_pair,
                learner_metric_writer=learner_metric_writer,
                status_writer=status_writer,
            )
        if pending_collection is not None:
            _discard_pending_collection(
                pending_collection,
                curriculum=curriculum,
                balance=balance,
            )
            pending_collection = None
            raise RuntimeError("training ended with a speculative collection")
        if pending_distributed is not None:
            if distributed_collection is None:
                raise RuntimeError("training ended with a lost distributed backend")
            distributed_collection.abort(
                reason="training ended with a speculative collection"
            )
            cancel_stateless_assignments(
                pending_distributed.assignments,
                curriculum=curriculum,
                deck_balance=balance,
            )
            if (
                opponent_pool_v2 is not None
                and pending_distributed.opponent_pool_window is not None
            ):
                opponent_pool_v2.abort(pending_distributed.opponent_pool_window)
            pending_distributed = None
            raise RuntimeError(
                "training ended with a speculative distributed collection"
            )
        training_failed = False
    finally:
        try:
            _close_native_collection_window(
                collector=native_collector,
                actor=native_actor,
            )
        except BaseException:
            # The explicit per-window boundary reports teardown failures. This
            # fallback must preserve any exception already unwinding training.
            if not training_failed:
                raise
        finally:
            native_collector = None
            native_actor = None
        if training_failed:
            if pending_distributed is not None and distributed_collection is not None:
                distributed_collection.abort(
                    reason="learner failed with a speculative collection"
                )
                cancel_stateless_assignments(
                    pending_distributed.assignments,
                    curriculum=curriculum,
                    deck_balance=balance,
                )
                if (
                    opponent_pool_v2 is not None
                    and pending_distributed.opponent_pool_window is not None
                ):
                    opponent_pool_v2.abort(pending_distributed.opponent_pool_window)
                pending_distributed = None
            if pending_collection is not None:
                pending_collection.future.cancel()
            if parallel_collector is not None:
                parallel_collector.close()
            if native_process_collector is not None:
                native_process_collector.close()
        if pipeline_executor is not None:
            pipeline_executor.shutdown(
                wait=True,
                cancel_futures=training_failed,
            )
        if distributed_prepare_executor is not None:
            distributed_prepare_executor.shutdown(
                wait=True,
                cancel_futures=training_failed,
            )
        if distributed_artifact_executor is not None:
            distributed_artifact_executor.shutdown(
                wait=True,
                cancel_futures=training_failed,
            )
        if training_failed and pending_collection is not None:
            _discard_pending_collection(
                pending_collection,
                curriculum=curriculum,
                balance=balance,
            )
        if not training_failed:
            if parallel_collector is not None:
                parallel_collector.close()
            if native_process_collector is not None:
                native_process_collector.close()
        if native_policy_stream_pool is not None:
            native_policy_stream_pool.close()
        if distributed_collection is not None:
            distributed_collection.close()
        if parallel_temporary_root is not None and parallel_temporary_root.exists():
            parallel_temporary_root.rmdir()
        if (
            native_process_temporary_root is not None
            and native_process_temporary_root.exists()
        ):
            native_process_temporary_root.rmdir()
        if performance_reporter is not None:
            _run_noncritical(
                "performance telemetry shutdown", performance_reporter.close
            )
        checkpoint_publisher.close()
        status_writer.close()
        if learner_metric_writer is not None:
            _run_noncritical("learner metric shutdown", learner_metric_writer.close)
        writer.close()
    if last_pair is None:
        raise RuntimeError("stateless training completed without a checkpoint pair")
    return {
        "run_version": config.run.version,
        "output_dir": str(output_dir),
        "updates": learner.update_index,
        "optimizer_steps": learner.optimizer_step_index,
        "fresh_decisions_seen": learner.fresh_decisions_seen,
        "lr_schedule_decisions_seen": learner.lr_schedule_decisions_seen,
        "policy_path": str(last_pair.policy_path),
        "learner_state_path": str(last_pair.learner_state_path),
        "pair_manifest_path": str(last_pair.pair_manifest_path),
        "pair_manifest_sha256": last_pair.pair_manifest_sha256,
        "past_self_admissions": sum(
            event.kind == "admitted" for event in curriculum.state.events
        ),
        "elapsed_seconds": time.perf_counter() - started_at,
    }


def _training_budget_reached(
    learner: SimpleStatelessLearner,
    *,
    config: SimpleStatelessTrainingConfig,
) -> bool:
    """Use cumulative decisions when configured, otherwise legacy windows."""
    if config.ppo.total_decisions is not None:
        return learner.lr_schedule_decisions_seen >= config.ppo.total_decisions
    return learner.update_index >= config.collection.training_updates


def _commit_expected_stateless_outcomes(
    assignments: tuple[StatelessAssignedGame, ...],
    outcomes: tuple[StatelessGameOutcome, ...],
    *,
    expected_assignments: tuple[StatelessAssignedGame, ...],
    curriculum: StatelessCurriculumController,
    balance: StatelessDeckBalanceSampler,
) -> None:
    """Commit the coordinator-ordered result only for the returned window."""
    if assignments != expected_assignments:
        raise RuntimeError(
            "distributed coordinator assignments differ from collection result"
        )
    commit_stateless_outcomes(
        assignments,
        outcomes,
        curriculum=curriculum,
        deck_balance=balance,
    )


def _retained_compact_decision_count(
    parts: Sequence[CompactFragmentPart],
    *,
    current_policy_version: int,
    maximum_version_age: int,
) -> int:
    """Count trainable decisions without constructing CPU optimizer targets."""
    if current_policy_version < 0 or maximum_version_age < 0:
        raise ValueError("compact decision-count policy versions are invalid")
    decisions = 0
    for part in parts:
        versions = np.asarray(
            part.arrays["behavior_policy_versions"],
            dtype=np.int64,
        )
        offsets = np.asarray(
            part.arrays["fragment_decision_offsets"],
            dtype=np.int64,
        )
        if offsets.shape != (versions.shape[0] + 1,):
            raise ValueError("compact fragment decision offsets are misaligned")
        if np.any(versions > current_policy_version):
            raise ValueError("current policy version cannot trail behavior")
        retained = current_policy_version - versions <= maximum_version_age
        decisions += int(np.diff(offsets)[retained].sum(dtype=np.int64))
    return decisions


def _prepare_current_rollout_artifact(
    *,
    learner: SimpleStatelessLearner,
    resources: _ResolvedResources,
    distributed_collection: NativeDistributedTrainingBackend,
) -> _PreparedCurrentRolloutArtifact:
    """Freeze the next current-policy wire artifact while outcomes settle."""
    started_at = time.perf_counter()
    behavior_version = learner.update_index
    fingerprint_started_at = time.perf_counter()
    fingerprint = canonical_model_state_fingerprint(learner.model)
    fingerprint_seconds = time.perf_counter() - fingerprint_started_at
    identity = _fragment_identity(
        resources,
        behavior_version=behavior_version,
        behavior_fingerprint=fingerprint,
        horizon=distributed_collection.config.collection.fragment_horizon,
    )
    artifact = distributed_collection.prepare_current_artifact(
        model_state=learner.model.state_dict(),
        behavior_identity=identity,
    )
    if learner.update_index != behavior_version:
        raise RuntimeError("learner mutated during rollout artifact preparation")
    return _PreparedCurrentRolloutArtifact(
        identity=identity,
        artifact=artifact,
        fingerprint_seconds=fingerprint_seconds,
        started_at=started_at,
        finished_at=time.perf_counter(),
    )


def _begin_distributed_prefetch(
    *,
    learner: SimpleStatelessLearner,
    config: SimpleStatelessTrainingConfig,
    resources: _ResolvedResources,
    curriculum: StatelessCurriculumController,
    balance: StatelessDeckBalanceSampler,
    opponent_pool_v2: StatelessOpponentPoolV2 | None,
    distributed_collection: NativeDistributedTrainingBackend,
    current_artifact_future: Future[_PreparedCurrentRolloutArtifact] | None = None,
) -> _PendingDistributedCollection:
    """Open the next remote window from the just-settled controller state."""
    learner_clocked = config.native_distributed.scheduling.learner_clocked_primary_only
    identity_started_at = time.perf_counter()
    prefetched_current: _PreparedCurrentRolloutArtifact | None = None
    artifact_prefetch_wait_seconds = 0.0
    if current_artifact_future is None:
        fingerprint_started_at = time.perf_counter()
        fingerprint = canonical_model_state_fingerprint(learner.model)
        fingerprint_seconds = time.perf_counter() - fingerprint_started_at
        identity = _fragment_identity(
            resources,
            behavior_version=learner.update_index,
            behavior_fingerprint=fingerprint,
            horizon=config.collection.fragment_horizon,
        )
    else:
        artifact_wait_started_at = time.perf_counter()
        prefetched_current = current_artifact_future.result()
        artifact_prefetch_wait_seconds = time.perf_counter() - artifact_wait_started_at
        identity = prefetched_current.identity
        fingerprint_seconds = prefetched_current.fingerprint_seconds
        if identity.behavior_policy_version != learner.update_index:
            raise RuntimeError("prefetched rollout artifact has stale learner version")
    stage_seconds = {
        "fingerprint_seconds": fingerprint_seconds,
        "assignment_planning_seconds": 0.0,
        "artifact_assignment_overlap_seconds": 0.0,
        "artifact_preparation_task_seconds": (
            0.0
            if prefetched_current is None
            else prefetched_current.preparation_seconds
        ),
        "artifact_preparation_prefetch_overlap_seconds": (
            0.0
            if prefetched_current is None
            else max(
                prefetched_current.preparation_seconds - artifact_prefetch_wait_seconds,
                0.0,
            )
        ),
        "artifact_preparation_wait_seconds": artifact_prefetch_wait_seconds,
        "artifact_source_fingerprint_seconds": 0.0,
        "bf16_conversion_seconds": 0.0,
        "wire_hashing_seconds": 0.0,
        "window_open_seconds": 0.0,
    }
    settled_curriculum_state = curriculum.state
    settled_balance_state = balance.state
    settled_opponent_pool_state = (
        None if opponent_pool_v2 is None else opponent_pool_v2.state
    )
    opponent_pool_window: BoundOpponentPoolWindow | BoundOpponentQuotaWindow | None = (
        None
    )
    assignments: tuple[StatelessAssignedGame, ...] = ()

    def plan_assignments() -> tuple[
        tuple[StatelessAssignedGame, ...],
        tuple[StatelessAssignedGame, ...] | StatelessQuotaAssignmentPlan,
        BoundOpponentPoolWindow | BoundOpponentQuotaWindow | None,
    ]:
        if opponent_pool_v2 is None:
            direct = assign_stateless_games(
                games=distributed_collection.assignment_pool_games(
                    learner_clocked=learner_clocked
                ),
                curriculum=curriculum,
                deck_balance=balance,
                active_decks=resources.active_decks,
                opponent_decks=resources.opponent_decks,
                mirror_policy_fingerprint=identity.behavior_policy_fingerprint,
            )
            return direct, direct, None
        if config.collection.native_shard_protocol_version == 3:
            quota = plan_stateless_assignment_quotas(
                games=distributed_collection.assignment_pool_games(
                    learner_clocked=learner_clocked
                ),
                opponent_pool=opponent_pool_v2,
                curriculum=curriculum,
                deck_balance=balance,
                active_decks=resources.active_decks,
                opponent_decks=resources.opponent_decks,
                mirror_policy_fingerprint=identity.behavior_policy_fingerprint,
                minimum_cohort_games=(distributed_collection.quota_exposure_games()),
            )
            return (), quota, quota.opponent_window
        planned = assign_stateless_games_v2(
            games=distributed_collection.assignment_pool_games(
                learner_clocked=learner_clocked
            ),
            opponent_pool=opponent_pool_v2,
            curriculum=curriculum,
            deck_balance=balance,
            active_decks=resources.active_decks,
            opponent_decks=resources.opponent_decks,
            mirror_policy_fingerprint=identity.behavior_policy_fingerprint,
        )
        return planned.assignments, planned.assignments, planned.opponent_pool_window

    try:
        if prefetched_current is None:
            current_artifact, planned_values, overlap_timing = _run_overlapping_calls(
                background=partial(
                    distributed_collection.prepare_current_artifact,
                    model_state=learner.model.state_dict(),
                    behavior_identity=identity,
                ),
                foreground=plan_assignments,
                thread_name_prefix="rollout-artifact",
            )
            stage_seconds["artifact_preparation_task_seconds"] = (
                overlap_timing.background_seconds
            )
            stage_seconds["artifact_assignment_overlap_seconds"] = (
                overlap_timing.overlap_seconds
            )
            stage_seconds["artifact_preparation_wait_seconds"] = (
                overlap_timing.background_wait_seconds
            )
            assignment_planning_seconds = overlap_timing.foreground_seconds
        else:
            current_artifact = prefetched_current.artifact
            assignment_started_at = time.perf_counter()
            planned_values = plan_assignments()
            assignment_planning_seconds = time.perf_counter() - assignment_started_at
        assignments, assignment_plan, opponent_pool_window = planned_values
        stage_seconds["assignment_planning_seconds"] = assignment_planning_seconds
        current_artifact_timing = current_artifact.preparation_timing
        stage_seconds["artifact_source_fingerprint_seconds"] = (
            current_artifact_timing.source_fingerprint_seconds
        )
        stage_seconds["bf16_conversion_seconds"] = (
            current_artifact_timing.bf16_conversion_seconds
        )
        stage_seconds["wire_hashing_seconds"] = (
            current_artifact_timing.wire_hashing_seconds
        )
        started_at = time.perf_counter()
        backend_timing = distributed_collection.begin_collection(
            model_state=learner.model.state_dict(),
            model_config=resources.model_config,
            behavior_identity=identity,
            assignments=assignment_plan,
            curriculum_members=curriculum.state.members,
            opponent_pool_revision=(
                None
                if opponent_pool_window is None
                else opponent_pool_window.plan.revision_fingerprint
            ),
            learner_clocked_primary_only=learner_clocked,
            prepared_current_artifact=current_artifact,
        )
    except BaseException:
        cancel_stateless_assignments(
            assignments,
            curriculum=curriculum,
            deck_balance=balance,
        )
        if opponent_pool_v2 is not None and opponent_pool_window is not None:
            opponent_pool_v2.abort(opponent_pool_window)
        raise
    stage_seconds["assignment_planning_seconds"] += (
        backend_timing.assignment_planning_seconds
    )
    stage_seconds["artifact_source_fingerprint_seconds"] += (
        backend_timing.source_fingerprint_seconds
    )
    stage_seconds["bf16_conversion_seconds"] += backend_timing.bf16_conversion_seconds
    stage_seconds["wire_hashing_seconds"] += backend_timing.wire_hashing_seconds
    stage_seconds["window_open_seconds"] = backend_timing.window_open_seconds
    return _PendingDistributedCollection(
        behavior_version=identity.behavior_policy_version,
        identity_seconds=time.perf_counter() - identity_started_at,
        identity_stage_seconds=stage_seconds,
        assignments=assignments,
        settled_curriculum_state=settled_curriculum_state,
        settled_balance_state=settled_balance_state,
        opponent_pool_window=opponent_pool_window,
        settled_opponent_pool_state=settled_opponent_pool_state,
        started_at=started_at,
        deck_target_shares=(
            dict(assignment_plan.candidate_target_shares)
            if isinstance(assignment_plan, StatelessQuotaAssignmentPlan)
            else dict(balance.target_share_mapping)
        ),
    )


def _prepared_distributed_window(
    pending: _PendingDistributedCollection,
    *,
    distributed_collection: NativeDistributedTrainingBackend,
    config: SimpleStatelessTrainingConfig,
) -> _PreparedDistributedWindow:
    """Resolve an overlapped remote preparation, or execute it synchronously."""
    future = pending.preparation_future
    if future is not None:
        scheduling = config.native_distributed.scheduling
        grace = scheduling.learner_ready_drain_grace_seconds
        if grace is not None and not future.done():
            elapsed = max(time.perf_counter() - pending.started_at, 0.0)
            minimum_remaining = max(
                scheduling.learner_ready_drain_minimum_collection_seconds - elapsed,
                0.0,
            )
            try:
                return future.result(timeout=max(grace, minimum_remaining))
            except TimeoutError:
                if future.done():
                    return future.result()
                drain_requested = distributed_collection.request_learner_ready_drain()
                if scheduling.learner_clocked_primary_only:
                    poll_seconds = min(
                        max(
                            config.native_distributed.retry.control_poll_interval_seconds,
                            0.01,
                        ),
                        1.0,
                    )
                    while not drain_requested and not future.done():
                        try:
                            return future.result(timeout=poll_seconds)
                        except TimeoutError:
                            drain_requested = (
                                distributed_collection.request_learner_ready_drain()
                            )
        return future.result()
    return _wait_and_prepare_distributed_window(
        distributed_collection,
        current_policy_version=pending.behavior_version + 1,
        maximum_version_age=config.ppo.maximum_version_age,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        normalize_epsilon=config.ppo.normalize_epsilon,
        deck_target_shares=pending.deck_target_shares,
    )


def _settle_distributed_admission_prefetch(
    pending: _PendingDistributedCollection,
    *,
    distributed_collection: NativeDistributedTrainingBackend,
    config: SimpleStatelessTrainingConfig,
) -> tuple[StatelessAssignedGame, ...]:
    """Wait for a speculative admission window before adopting its leases."""
    _prepared_distributed_window(
        pending,
        distributed_collection=distributed_collection,
        config=config,
    )
    return distributed_collection.adopt_quota_assignments()


def _wait_and_prepare_distributed_window(
    distributed_collection: NativeDistributedTrainingBackend,
    *,
    current_policy_version: int,
    maximum_version_age: int,
    gamma: float,
    gae_lambda: float,
    normalize_epsilon: float,
    deck_target_shares: Mapping[str, float],
) -> _PreparedDistributedWindow:
    """Wait for remote parts, then build CPU replay while the GPU learns."""
    collected = distributed_collection.wait_collection()
    collection_finished_at = time.perf_counter()
    array_window: StatelessArrayOptimizerWindow | None = None
    preparation_seconds = 0.0
    if collected.compact_parts and not collected.fragments:
        preparation_started_at = time.perf_counter()
        array_window = prepare_stateless_array_optimizer_window(
            collected.compact_parts,
            current_policy_version=current_policy_version,
            maximum_version_age=maximum_version_age,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_epsilon=normalize_epsilon,
            deck_target_shares=deck_target_shares,
        )
        preparation_seconds = time.perf_counter() - preparation_started_at
    return _PreparedDistributedWindow(
        collected=collected,
        array_window=array_window,
        collection_finished_at=collection_finished_at,
        preparation_seconds=preparation_seconds,
    )


def _pipeline_can_prefetch(
    learner: SimpleStatelessLearner,
    *,
    decisions: int,
    config: SimpleStatelessTrainingConfig,
    curriculum_state: StatelessCurriculumState,
    allow_checkpoint_boundary: bool = False,
    allow_admission_boundary: bool = False,
) -> bool:
    """Predict whether one-version-lag work can cross the next update.

    A backend may checkpoint from a settled sidecar while collection remains in
    flight. V3 admission boundaries finish and adopt their already-overlapped
    quota window immediately before publication, so they do not require a full
    pre-update drain.
    """
    if config.collection.pipeline_mode != "one_version_lag":
        return False
    if decisions <= 0:
        raise ValueError("pipeline prediction requires fresh decisions")
    next_version = learner.update_index + 1
    if config.ppo.total_decisions is None:
        final = next_version >= config.collection.training_updates
    else:
        final = (
            learner.lr_schedule_decisions_seen + decisions >= config.ppo.total_decisions
        )
        if next_version >= config.collection.training_updates and not final:
            return False
    if final:
        return False
    checkpoint_due = _should_publish_checkpoint(
        next_version,
        final=final,
        interval=config.collection.checkpoint_interval_updates,
        retain_every_versions=config.collection.checkpoint_retain_every_versions,
    )
    admission_due = _past_self_admission_due(
        next_version,
        config=config,
        final=final,
        curriculum_state=curriculum_state,
    )
    return (allow_checkpoint_boundary or not checkpoint_due) and (
        allow_admission_boundary or not admission_due
    )


def _past_self_admission_due(
    version: int,
    *,
    config: SimpleStatelessTrainingConfig,
    final: bool,
    curriculum_state: StatelessCurriculumState,
) -> bool:
    """Schedule admission independently, but only at a settled publication."""
    snapshot_id = f"past-self-v{version}"
    return (
        not final
        and version >= config.curriculum.admit_past_self_after_updates
        and (version % config.curriculum.past_self_admission_interval_updates == 0)
        and not any(
            event.kind == "admitted" and event.snapshot_id == snapshot_id
            for event in curriculum_state.events
        )
    )


def _past_self_reentry_artifact(
    output_dir: Path,
    *,
    config: SimpleStatelessTrainingConfig,
    curriculum: StatelessCurriculumController,
    current_version: int,
) -> DurablePolicyArtifact | None:
    """Load immutable archive candidates when the configured reentry probe is due."""
    interval = config.curriculum.past_self_retention.reentry_interval_admissions
    if not _past_self_reentry_probe_due(curriculum.state, interval=interval):
        return None

    manifest_paths = {
        _path(path) for path in config.curriculum.past_self_reentry_manifest_paths
    }
    manifest_paths.update((output_dir / "weights").glob("checkpoint_pair_v*.json"))
    candidates = tuple(
        DurablePolicyArtifact.from_manifest(path)
        for path in sorted(manifest_paths, key=lambda item: str(item.resolve()))
    )
    return curriculum.select_past_self_reentry(
        candidates,
        current_version=current_version,
    )


def _past_self_reentry_probe_due(
    state: StatelessCurriculumState,
    *,
    interval: int | None,
) -> bool:
    """Return whether this regular-admission boundary gets one reentry probe."""
    if interval is None:
        return False
    if interval <= 0:
        raise ValueError("past-self reentry interval must be positive")
    admission_events = tuple(
        event
        for event in state.events
        if event.kind == "admitted"
        and (
            (
                event.snapshot_id.startswith("past-self-v")
                and event.snapshot_id.removeprefix("past-self-v").isdigit()
            )
            or event.snapshot_id.startswith("past-self-reentry-v")
        )
    )
    regular_admissions = sum(
        event.snapshot_id.startswith("past-self-v")
        and event.snapshot_id.removeprefix("past-self-v").isdigit()
        for event in admission_events
    )
    return bool(
        admission_events
        and not admission_events[-1].snapshot_id.startswith("past-self-reentry-v")
        and regular_admissions > 0
        and regular_admissions % interval == 0
    )


def _should_publish_checkpoint(
    version: int,
    *,
    final: bool,
    interval: int,
    retain_every_versions: int | None,
) -> bool:
    """Select rolling, permanent-milestone, and final checkpoint versions."""
    if version < 0:
        raise ValueError("checkpoint version must be non-negative")
    if interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    if retain_every_versions is not None and retain_every_versions <= 0:
        raise ValueError("permanent checkpoint interval must be positive")
    return (
        final
        or version % interval == 0
        or (retain_every_versions is not None and version % retain_every_versions == 0)
    )


def _prune_checkpoint_history(
    output_dir: Path,
    *,
    config: SimpleStatelessTrainingConfig,
    curriculum_states: Sequence[StatelessCurriculumState],
) -> tuple[int, ...]:
    """Retain policies referenced by both live and just-published state."""
    protected = {
        member.policy_path
        for state in curriculum_states
        for member in state.members
        if member.pair is not None
    }
    removed = prune_stateless_checkpoint_pairs(
        output_dir,
        keep_last=config.collection.checkpoint_keep_last,
        retain_every_versions=(config.collection.checkpoint_retain_every_versions),
        protected_policy_paths=protected,
    )
    if config.collection.checkpoint_keep_last is not None:
        prune_external_stateless_curriculum_states(
            output_dir / "control" / "curriculum_state.json",
            keep_last=config.collection.checkpoint_keep_last,
        )
    return removed


def _build_in_process_native_collector(
    *,
    config: SimpleStatelessTrainingConfig,
    resources: _ResolvedResources,
    curriculum_state: StatelessCurriculumState,
    past_self: PastSelfPolicyPool,
    native_historical: NativeHistoricalPolicyPool,
    native_scripted: Mapping[str, NativeScriptedPolicy],
    native_scripted_bindings: Mapping[str, tuple[str, str]],
    actor: NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
    behavior_identity: StatelessFragmentIdentity,
    seed: int,
    policy_stream_pool: NativePolicyCudaStreamPool | None,
) -> NativeStatelessCollector:
    """Construct one window-local in-process native collector."""
    sequence = resources.model_config.sequence
    return NativeStatelessCollector(
        actor=actor,
        identity=behavior_identity,
        catalog=resources.catalog,
        active_decks=resources.active_decks,
        opponent_decks=resources.opponent_decks,
        members=curriculum_state.members,
        past_self_pool=past_self,
        historical_pool=native_historical,
        scripted_policies=native_scripted,
        scripted_bindings=native_scripted_bindings,
        maximum_engine_steps=config.collection.maximum_engine_steps,
        seed=seed,
        fragments_per_part=config.collection.fragments_per_part,
        mirror_bilateral_trajectories=(config.collection.mirror_bilateral_trajectories),
        arena_capacity=config.collection.native_arena_capacity,
        engine_shards=config.collection.native_engine_shards,
        policy_cohort_slots=config.collection.native_policy_cohort_slots,
        policy_group_bank_limit=(config.collection.native_policy_group_bank_limit),
        policy_cohort_wait_ms=config.collection.native_policy_cohort_wait_ms,
        trainable_decision_budget=(config.collection.native_trainable_decision_budget),
        frozen_batch_min_rows=config.collection.native_frozen_batch_min_rows,
        frozen_batch_max_wait_waves=(
            config.collection.native_frozen_batch_max_wait_waves
        ),
        sequence_rollout_precision=(
            config.collection.native_sequence_rollout_precision
        ),
        library_path=resources.native_library_path,
        engine_fact_producer=(
            ProspectiveEngineFactProducer(
                sampler=BeliefSampler(config=sequence.engine_facts.sampler),
                config=sequence.engine_facts,
            )
            if sequence is not None and sequence.engine_facts.enabled
            else None
        ),
        engine_fact_workers=config.collection.native_engine_fact_workers,
        policy_stream_pool=policy_stream_pool,
    )


def _prepare_in_process_native_collection(
    *,
    config: SimpleStatelessTrainingConfig,
    resources: _ResolvedResources,
    curriculum: StatelessCurriculumController,
    balance: StatelessDeckBalanceSampler,
    past_self: PastSelfPolicyPool,
    native_historical: NativeHistoricalPolicyPool,
    native_scripted: Mapping[str, NativeScriptedPolicy],
    native_scripted_bindings: Mapping[str, tuple[str, str]],
    learner_model: SimpleStatelessPolicyValueNet,
    behavior_identity: StatelessFragmentIdentity,
    identity_started_at: float,
    cuda_stream: Any,
    policy_stream_pool: NativePolicyCudaStreamPool | None,
) -> _PreparedHybridCollection:
    """Freeze and own one BF16 banked sequence collection before learner update."""
    if (
        config.collection.backend != "native_banked"
        or config.collection.native_process_workers != 1
        or config.collection.native_sequence_rollout_precision != "bf16"
        or learner_model.sequence is None
    ):
        raise ValueError(
            "prepared native pipeline requires one-worker BF16 banked sequence "
            "collection"
        )
    behavior_model: SimpleStatelessPolicyValueNet | None = None
    actor: GeneralistSequenceActorPolicy | None = None
    collector: NativeStatelessCollector | None = None
    assignments: tuple[StatelessAssignedGame, ...] = ()
    try:
        behavior_model = SimpleStatelessPolicyValueNet(
            resources.model_config,
            load_static_features=False,
            initialize=False,
        )
        behavior_model.load_state_dict(learner_model.state_dict(), strict=True)
        behavior_model.eval().requires_grad_(False)
        assignments = assign_stateless_games(
            games=config.collection.concurrent_games,
            curriculum=curriculum,
            deck_balance=balance,
            active_decks=resources.active_decks,
            opponent_decks=resources.opponent_decks,
            mirror_policy_fingerprint=(behavior_identity.behavior_policy_fingerprint),
        )
        temporal_cache_slots = sum(
            1 + int(assignment.curriculum.lane == "mirror")
            for assignment in assignments
        )
        actor = GeneralistSequenceActorPolicy(
            behavior_model,
            identity=behavior_identity,
            device=config.device,
            verify_model_state=False,
            retain_raw_blocks=False,
            rollout_precision="bf16",
            temporal_cache_slots=temporal_cache_slots,
        )
        # The FP32 clone was needed only to bind an exact behavior identity and
        # construct the immutable BF16 shadow. This prepared actor is never
        # published or reused as a master, so keep the shadow as its config
        # owner and release the redundant FP32 CUDA parameters before overlap.
        behavior_model = actor.rollout_model
        actor.model = behavior_model
        collector = _build_in_process_native_collector(
            config=config,
            resources=resources,
            curriculum_state=curriculum.state,
            past_self=past_self,
            native_historical=native_historical,
            native_scripted=native_scripted,
            native_scripted_bindings=native_scripted_bindings,
            actor=actor,
            behavior_identity=behavior_identity,
            seed=_native_seed_for_assignments(config, assignments),
            policy_stream_pool=policy_stream_pool,
        )
        cuda_stream.wait_stream(torch.cuda.current_stream(device=actor.device))
        return _PreparedHybridCollection(
            behavior_version=behavior_identity.behavior_policy_version,
            identity_seconds=time.perf_counter() - identity_started_at,
            assignments=assignments,
            collect=_OwnedNativePreparedCollect(
                collector=collector,
                actor=actor,
                behavior_model=behavior_model,
                assignments=assignments,
                cuda_stream=cuda_stream,
            ),
        )
    except BaseException as error:
        try:
            _close_native_collection_window(
                collector=collector,
                actor=actor,
            )
        except BaseException as cleanup_error:
            error.add_note(
                "prepared native construction teardown also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        if assignments:
            cancel_stateless_assignments(
                assignments,
                curriculum=curriculum,
                deck_balance=balance,
            )
        behavior_model = None
        raise


def _prepare_hybrid_collection(
    *,
    config: SimpleStatelessTrainingConfig,
    resources: _ResolvedResources,
    curriculum: StatelessCurriculumController,
    balance: StatelessDeckBalanceSampler,
    past_self: PastSelfPolicyPool,
    native_historical: NativeHistoricalPolicyPool,
    parallel_collector: StatelessParallelCollector,
    native_process_collector: NativeProcessCollector | None,
    parallel_temporary_root: Path,
    behavior_model: SimpleStatelessPolicyValueNet,
    behavior_identity: StatelessFragmentIdentity,
    identity_started_at: float,
    policy_stream_pool: NativePolicyCudaStreamPool | None,
) -> _PreparedHybridCollection:
    """Lease and bind one hybrid cohort without starting its engine workers."""
    native_actor = NativePolicyInferenceExecutor(
        behavior_model,
        identity=behavior_identity,
        device=config.device,
        verify_model_state=False,
    )
    actor = SimpleStatelessActorPolicy(
        behavior_model,
        identity=behavior_identity,
        device=config.device,
        verify_model_state=False,
    )
    identity_seconds = time.perf_counter() - identity_started_at
    assignments: tuple[StatelessAssignedGame, ...] = ()
    try:
        assignments = assign_stateless_games(
            games=config.collection.concurrent_games,
            curriculum=curriculum,
            deck_balance=balance,
            active_decks=resources.active_decks,
            opponent_decks=resources.opponent_decks,
            mirror_policy_fingerprint=(behavior_identity.behavior_policy_fingerprint),
        )
        temporary_root = _parallel_temporary_window(
            parallel_temporary_root,
            assignments,
        )
        members = curriculum.state.members
        native_collector = NativeStatelessCollector(
            actor=native_actor,
            identity=behavior_identity,
            catalog=resources.catalog,
            active_decks=resources.active_decks,
            opponent_decks=resources.opponent_decks,
            members=members,
            past_self_pool=past_self,
            historical_pool=native_historical,
            scripted_policies={},
            scripted_bindings={},
            maximum_engine_steps=config.collection.maximum_engine_steps,
            seed=_native_seed_for_assignments(config, assignments),
            fragments_per_part=config.collection.fragments_per_part,
            mirror_bilateral_trajectories=(
                config.collection.mirror_bilateral_trajectories
            ),
            arena_capacity=config.collection.native_arena_capacity,
            engine_shards=config.collection.native_engine_shards,
            policy_cohort_slots=config.collection.native_policy_cohort_slots,
            policy_group_bank_limit=(config.collection.native_policy_group_bank_limit),
            policy_cohort_wait_ms=(config.collection.native_policy_cohort_wait_ms),
            trainable_decision_budget=(
                config.collection.native_trainable_decision_budget
            ),
            frozen_batch_min_rows=(config.collection.native_frozen_batch_min_rows),
            frozen_batch_max_wait_waves=(
                config.collection.native_frozen_batch_max_wait_waves
            ),
            sequence_rollout_precision=(
                config.collection.native_sequence_rollout_precision
            ),
            library_path=resources.native_library_path,
            engine_fact_workers=config.collection.native_engine_fact_workers,
            policy_stream_pool=policy_stream_pool,
        )
        native_collect: Callable[..., StatelessCollectionResult]
        if native_process_collector is None:
            native_collect = native_collector.collect_assigned
        else:
            native_collect = partial(
                native_process_collector.collect,
                actor=native_actor,
                identity=behavior_identity,
                members=members,
                temporary_root=temporary_root,
                seed=_native_seed_for_assignments(config, assignments),
                arena_capacity=(
                    config.collection.native_arena_capacity or len(assignments)
                ),
                trainable_decision_budget=(
                    config.collection.native_trainable_decision_budget
                ),
                frozen_batch_min_rows=(config.collection.native_frozen_batch_min_rows),
                frozen_batch_max_wait_waves=(
                    config.collection.native_frozen_batch_max_wait_waves
                ),
            )
        collect = partial(
            _collect_hybrid_window,
            native_collect=native_collect,
            parallel_collector=parallel_collector,
            actor=actor,
            identity=behavior_identity,
            assignments=assignments,
            temporary_root=temporary_root,
            fragments_per_part=config.collection.fragments_per_part,
            integrated_process_inference=(
                config.collection.integrate_scripted_current_inference
            ),
        )
        return _PreparedHybridCollection(
            behavior_version=behavior_identity.behavior_policy_version,
            identity_seconds=identity_seconds,
            assignments=assignments,
            collect=collect,
        )
    except BaseException:
        if assignments:
            cancel_stateless_assignments(
                assignments,
                curriculum=curriculum,
                deck_balance=balance,
            )
        raise


def _timed_collection(
    collect: Callable[[], StatelessCollectionResult],
) -> _TimedCollection:
    """Execute one collection while preserving its independent wall interval."""
    started_at = time.perf_counter()
    result = collect()
    return _TimedCollection(
        result=result,
        started_at=started_at,
        finished_at=time.perf_counter(),
    )


def _run_overlapping_calls(
    *,
    background: Callable[[], _BackgroundValue],
    foreground: Callable[[], _ForegroundValue],
    thread_name_prefix: str,
) -> tuple[_BackgroundValue, _ForegroundValue, _OverlapTiming]:
    """Run one bounded background operation while the foreground makes progress."""

    def timed_background() -> tuple[_BackgroundValue, float, float]:
        started_at = time.perf_counter()
        result = background()
        return result, started_at, time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix=thread_name_prefix,
    ) as executor:
        future = executor.submit(timed_background)
        foreground_started_at = time.perf_counter()
        foreground_result = foreground()
        foreground_finished_at = time.perf_counter()
        wait_started_at = time.perf_counter()
        background_result, background_started_at, background_finished_at = (
            future.result()
        )
        wait_finished_at = time.perf_counter()

    overlap_seconds = max(
        0.0,
        min(background_finished_at, foreground_finished_at)
        - max(background_started_at, foreground_started_at),
    )
    return (
        background_result,
        foreground_result,
        _OverlapTiming(
            background_seconds=max(
                0.0,
                background_finished_at - background_started_at,
            ),
            foreground_seconds=max(
                0.0,
                foreground_finished_at - foreground_started_at,
            ),
            background_wait_seconds=max(0.0, wait_finished_at - wait_started_at),
            overlap_seconds=overlap_seconds,
        ),
    )


def _close_native_collection_window(
    *,
    collector: NativeStatelessCollector | None,
    actor: NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy | None,
) -> None:
    """Release one native window without touching shared policy models."""
    try:
        if collector is not None:
            collector.close()
    finally:
        if isinstance(actor, GeneralistSequenceActorPolicy):
            actor.close()


def _release_prepared_collection(
    prepared: _PreparedHybridCollection,
) -> None:
    """Release resources retained by a native prepared callable, if present."""
    collect = prepared.collect
    if isinstance(collect, _OwnedNativePreparedCollect):
        collect.close()


def _discard_pending_collection(
    pending: _PendingCollection,
    *,
    curriculum: StatelessCurriculumController,
    balance: StatelessDeckBalanceSampler,
) -> None:
    """Drain a speculative cohort and release every non-durable lease."""
    release_error: BaseException | None = None
    try:
        completed = pending.future.result()
    except BaseException:
        completed = None
    finally:
        try:
            _release_prepared_collection(pending.prepared)
        except BaseException as error:
            release_error = error
    if completed is not None and completed.result.compact_part_paths:
        cleanup_parallel_collection_parts(completed.result.compact_part_paths)
    cancel_stateless_assignments(
        pending.prepared.assignments,
        curriculum=curriculum,
        deck_balance=balance,
    )
    if release_error is not None:
        raise release_error


def _collect_hybrid_window(
    *,
    native_collect: Callable[..., StatelessCollectionResult],
    parallel_collector: StatelessParallelCollector,
    actor: SimpleStatelessActorPolicy,
    identity: StatelessFragmentIdentity,
    assignments: tuple[StatelessAssignedGame, ...],
    temporary_root: Path,
    fragments_per_part: int,
    integrated_process_inference: bool = False,
) -> StatelessCollectionResult:
    """Collect native and exact-script lanes under one frozen behavior."""
    native_assignments, scripted_assignments = partition_hybrid_assignments(assignments)
    started_at = time.perf_counter()
    native_phase_seconds = 0.0
    scripted_phase_seconds = 0.0
    results: tuple[StatelessCollectionResult, ...]
    if native_assignments and scripted_assignments:
        if integrated_process_inference:
            native_result, scripted_result = _collect_integrated_hybrid_phases(
                native_collect=native_collect,
                parallel_collector=parallel_collector,
                actor=actor,
                identity=identity,
                native_assignments=native_assignments,
                scripted_assignments=scripted_assignments,
                temporary_root=temporary_root,
                fragments_per_part=fragments_per_part,
            )
            native_phase_seconds = native_result.report.elapsed_seconds
            scripted_phase_seconds = scripted_result.report.elapsed_seconds
        else:
            phase_started_at = time.perf_counter()
            native_result = native_collect(assignments=native_assignments)
            native_phase_seconds = time.perf_counter() - phase_started_at
            phase_started_at = time.perf_counter()
            scripted_result = parallel_collector.collect(
                actor=actor,
                identity=identity,
                assignments=scripted_assignments,
                members=(),
                temporary_root=temporary_root,
                fragments_per_part=fragments_per_part,
            )
            scripted_phase_seconds = time.perf_counter() - phase_started_at
        results = (native_result, scripted_result)
    elif native_assignments:
        phase_started_at = time.perf_counter()
        native_result = native_collect(assignments=native_assignments)
        native_phase_seconds = time.perf_counter() - phase_started_at
        results = (native_result,)
    elif scripted_assignments:
        phase_started_at = time.perf_counter()
        scripted_result = parallel_collector.collect(
            actor=actor,
            identity=identity,
            assignments=scripted_assignments,
            members=(),
            temporary_root=temporary_root,
            fragments_per_part=fragments_per_part,
        )
        scripted_phase_seconds = time.perf_counter() - phase_started_at
        results = (scripted_result,)
    else:
        raise ValueError("hybrid collection received no assignments")
    return merge_hybrid_collection_results(
        results,
        assignments=assignments,
        elapsed_seconds=max(time.perf_counter() - started_at, 1.0e-9),
        native_phase_seconds=native_phase_seconds,
        scripted_phase_seconds=scripted_phase_seconds,
    )


def _collect_integrated_hybrid_phases(
    *,
    native_collect: Callable[..., StatelessCollectionResult],
    parallel_collector: StatelessParallelCollector,
    actor: SimpleStatelessActorPolicy,
    identity: StatelessFragmentIdentity,
    native_assignments: tuple[StatelessAssignedGame, ...],
    scripted_assignments: tuple[StatelessAssignedGame, ...],
    temporary_root: Path,
    fragments_per_part: int,
) -> tuple[StatelessCollectionResult, StatelessCollectionResult]:
    """Run both engine lanes while one native broker owns current inference."""
    broker_ready = threading.Event()
    broker_hold = threading.Event()
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="integrated-hybrid",
    ) as executor:
        native_future = executor.submit(
            native_collect,
            assignments=native_assignments,
            broker_ready_event=broker_ready,
            broker_hold_event=broker_hold,
        )
        while not broker_ready.wait(timeout=0.05):
            if native_future.done():
                native_future.result()
        scripted_future = executor.submit(
            parallel_collector.collect,
            actor=actor,
            identity=identity,
            assignments=scripted_assignments,
            members=(),
            temporary_root=temporary_root,
            fragments_per_part=fragments_per_part,
            serve_inference=False,
        )
        try:
            scripted_result = scripted_future.result()
        finally:
            broker_hold.set()
        native_result = native_future.result()
    return native_result, scripted_result


def _native_seed_for_assignments(
    config: SimpleStatelessTrainingConfig,
    assignments: tuple[StatelessAssignedGame, ...],
) -> int:
    """Derive a unique deterministic native seed from the cohort cursor."""
    if not assignments:
        raise ValueError("native seed requires a non-empty assignment cohort")
    return config.collection.seed + int(assignments[0].balance.assignment_cursor)


def _parallel_temporary_window(
    root: Path,
    assignments: tuple[StatelessAssignedGame, ...],
) -> Path:
    """Return a collision-free worker path for one assignment cohort."""
    if not assignments:
        raise ValueError("parallel collection requires assignment identities")
    cursor = int(assignments[0].balance.assignment_cursor)
    return root / f"assignment-{cursor:012d}"


def _resolve_resources(
    config: SimpleStatelessTrainingConfig,
) -> _ResolvedResources:
    registry = resolve_deck_expert_registry(
        config.private_deck_registry,
        base_dir=_REPO_ROOT,
    )
    validate_active_exact_strategy_routes(registry.routes)
    family_registry = (
        resolve_deck_family_registry(
            config.private_deck_registry,
            base_dir=_REPO_ROOT,
        )
        if uses_family_private_topology(config.model)
        else None
    )
    catalog_path = _path(config.public_deck_catalog.manifest_path)
    catalog, catalog_manifest = load_public_deck_catalog(catalog_path)
    if catalog.fingerprint != config.public_deck_catalog.catalog_fingerprint:
        raise ValueError("configured public catalog fingerprint mismatch")
    model_payload = config.model.model_dump(mode="python")
    model_payload.update(
        {
            "exact_routes": registry.routes,
            "resolved_registry_sha256": registry.resolved_registry_sha256,
            "public_deck_catalog_fingerprint": catalog.fingerprint,
        }
    )
    if family_registry is not None:
        model_payload.update(
            {
                "family_routes": family_registry.routes,
                "resolved_family_registry_sha256": (
                    family_registry.resolved_registry_sha256
                ),
            }
        )
    model_config = SimpleStatelessModelConfig.model_validate(model_payload)
    active_decks = {
        route.deck_digest: canonicalize_deck(route.canonical_card_ids)
        for route in registry.routes
    }
    active_deck_labels: dict[str, str] = {}
    for source in config.private_deck_registry.decks:
        deck = canonicalize_deck(read_deck(_path(source.path)))
        if deck.deck_digest in active_deck_labels:
            raise ValueError("active performance deck labels are ambiguous")
        active_deck_labels[deck.deck_digest] = source.label
    if set(active_deck_labels) != set(active_decks):
        raise ValueError("active performance labels differ from resolved registry")
    deck_target_shares = _resolve_deck_target_shares(
        config,
        active_deck_labels=active_deck_labels,
    )
    roster_fingerprint = _fingerprint(
        _ROSTER_DOMAIN,
        {
            "routes": [
                {
                    "deck_digest": route.deck_digest,
                    "expert_id": route.expert_id,
                    "family_id": (
                        None
                        if family_registry is None
                        else next(
                            family_route.family_id
                            for family_route in family_registry.routes
                            if family_route.deck_digest == route.deck_digest
                        )
                    ),
                }
                for route in registry.routes
            ]
        },
    )
    pinned_fingerprint = _fingerprint(
        _PINNED_DOMAIN,
        {
            "anchors": [
                anchor.model_dump(
                    mode="json",
                    exclude={
                        "checkpoint_path",
                        "exact_deck_path",
                        "belief_summary_path",
                        "public_catalog_manifest_path",
                    },
                )
                for anchor in config.curriculum.anchors
            ]
        },
    )
    scripted_manifest = load_scripted_manifest(
        _path(config.curriculum.scripted_manifest_path)
    )
    if scripted_manifest.fingerprint != config.curriculum.scripted_manifest_fingerprint:
        raise ValueError("configured scripted manifest fingerprint mismatch")
    resolved_scripted = resolve_scripted_manifest(
        scripted_manifest,
        implementations=builtin_scripted_implementations(
            tuple(item.opponent_name for item in config.curriculum.scripted)
        ),
        source_root=_REPO_ROOT,
    )
    scripted_by_id = {
        item.artifact.opponent_id: item.artifact for item in resolved_scripted
    }
    if set(scripted_by_id) != {item.opponent_id for item in config.curriculum.scripted}:
        raise ValueError("scripted runtime entries differ from immutable manifest")
    for item in config.curriculum.scripted:
        artifact = scripted_by_id[item.opponent_id]
        if (
            artifact.script_name != item.opponent_name
            or artifact.fingerprint != item.artifact_fingerprint
            or artifact.exact_deck_digest != item.exact_deck_digest
        ):
            raise ValueError("scripted runtime binding differs from manifest")
    scripted_fingerprint = scripted_manifest.fingerprint
    curriculum_config = StatelessCurriculumConfig(
        lane_mix=config.curriculum.lane_mix,
        pfsp=config.curriculum.pfsp,
        past_self_retention=config.curriculum.past_self_retention,
        replaceable_snapshot_capacity=(config.curriculum.replaceable_snapshot_capacity),
        past_self_admission_interval_versions=(
            config.curriculum.past_self_admission_interval_updates
        ),
        retain_all_past_self=config.curriculum.retain_all_past_self,
        assignment_seed=config.curriculum.assignment_seed,
        pinned_manifest_fingerprint=pinned_fingerprint,
        scripted_manifest_fingerprint=scripted_fingerprint,
        scripted_sampling_fingerprint=(
            _fingerprint(
                _SCRIPTED_SAMPLING_DOMAIN,
                dict(sorted(config.curriculum.scripted_weight_overrides.items())),
            )
            if config.curriculum.scripted_weight_overrides
            else None
        ),
        training_roster_fingerprint=roster_fingerprint,
    )
    if config.deck_allocation is None:
        deck_balance_config: StatelessDeckBalanceConfigValue = (
            StatelessDeckBalanceConfig(
                active_deck_digests=tuple(sorted(active_decks)),
                rolling_window_decisions=config.deck_balance.rolling_window_decisions,
                assignment_seed=config.deck_balance.assignment_seed,
                inflight_decision_credit=config.deck_balance.inflight_decision_credit,
                deficit_exponent=config.deck_balance.deficit_exponent,
            )
        )
    elif isinstance(config.deck_allocation, StatelessDynamicDeckAllocationConfig):
        deck_balance_config = StatelessDynamicDeckBalanceConfig(
            active_deck_digests=tuple(sorted(active_decks)),
            rolling_window_decisions=config.deck_balance.rolling_window_decisions,
            assignment_seed=config.deck_balance.assignment_seed,
            inflight_decision_credit=config.deck_balance.inflight_decision_credit,
            deficit_exponent=config.deck_balance.deficit_exponent,
            performance_window_games=(config.deck_allocation.performance_window_games),
            evidence_prior_games=config.deck_allocation.evidence_prior_games,
            uniform_mix=config.deck_allocation.uniform_mix,
            difficulty_temperature=(config.deck_allocation.difficulty_temperature),
            maximum_share_ratio=config.deck_allocation.maximum_share_ratio,
        )
    else:
        deck_balance_config = StatelessWeightedDeckBalanceConfig(
            target_deck_shares=tuple(
                DeckTargetShare(deck_digest=digest, target_share=share)
                for digest, share in sorted(deck_target_shares.items())
            ),
            rolling_window_decisions=config.deck_balance.rolling_window_decisions,
            assignment_seed=config.deck_balance.assignment_seed,
            inflight_decision_credit=config.deck_balance.inflight_decision_credit,
            deficit_exponent=config.deck_balance.deficit_exponent,
        )
    input_contract = simple_stateless_input_contract(
        public_catalog_fingerprint=catalog.fingerprint,
        card_catalog_fingerprint=catalog_manifest.card_catalog_fingerprint,
        public_context_fingerprint=PUBLIC_EVENT_SCHEMA_FINGERPRINT,
        wrapper_runtime_fingerprint=(SIMPLE_STATELESS_WRAPPER_RUNTIME_FINGERPRINT),
    )
    opponent_decks = dict(active_decks)
    route_input_contracts: dict[str, tuple[PublicDeckCatalog, str]] = {}
    for anchor in config.curriculum.anchors:
        _verify_file(
            _path(anchor.checkpoint_path),
            size_bytes=anchor.checkpoint_size_bytes,
            sha256=anchor.checkpoint_sha256,
            label=f"historical checkpoint {anchor.member_id}",
        )
        deck = canonicalize_deck(read_deck(_path(anchor.exact_deck_path)))
        if deck.deck_digest != anchor.exact_deck_digest:
            raise ValueError("historical exact deck fingerprint mismatch")
        if anchor.runtime_kind == "fixed_stateless_wire":
            if anchor.belief_summary_path is not None:
                raise ValueError(
                    "fixed stateless anchors use the native sequence belief path"
                )
            fixed_model, fixed_payload = load_fixed_deck_checkpoint(
                _path(anchor.checkpoint_path)
            )
            fixed_model.float()
            runtime_model_fingerprint = canonical_model_state_fingerprint(
                fixed_model.state_dict()
            )
            asset_fingerprints = fixed_payload.get("asset_manifest_fingerprints")
            if not isinstance(asset_fingerprints, dict):
                raise ValueError("fixed stateless anchor omitted asset identities")
            if (
                runtime_model_fingerprint != anchor.pilot_artifact_fingerprint
                or fixed_payload.get("target_deck_digest") != anchor.exact_deck_digest
                or asset_fingerprints.get("input_contract")
                != anchor.input_contract_fingerprint
                or fixed_model.config.resolved_registry_sha256
                != anchor.exact_registry_fingerprint
            ):
                raise ValueError("fixed stateless anchor identity mismatch")
            expected_catalog_fingerprint = (
                fixed_model.config.public_deck_catalog_fingerprint
            )
            if expected_catalog_fingerprint is None:
                raise ValueError("fixed stateless anchor omitted its public catalog")
            route_catalog = catalog
            if anchor.public_catalog_manifest_path is not None:
                route_catalog, route_catalog_manifest = load_public_deck_catalog(
                    _path(anchor.public_catalog_manifest_path)
                )
                if (
                    route_catalog_manifest.card_catalog_fingerprint
                    != catalog_manifest.card_catalog_fingerprint
                ):
                    raise ValueError(
                        "fixed stateless anchor public catalog uses a different "
                        "card catalog"
                    )
            if route_catalog.fingerprint != expected_catalog_fingerprint:
                raise ValueError(
                    "fixed stateless anchor public catalog mismatch; declare "
                    "public_catalog_manifest_path for its immutable catalog"
                )
            if (
                route_catalog.fingerprint != catalog.fingerprint
                or anchor.input_contract_fingerprint != input_contract.fingerprint
            ):
                route_binding = (
                    route_catalog,
                    anchor.input_contract_fingerprint,
                )
                existing_binding = route_input_contracts.get(anchor.checkpoint_sha256)
                if existing_binding is not None and (
                    existing_binding[0].fingerprint != route_catalog.fingerprint
                    or existing_binding[1] != anchor.input_contract_fingerprint
                ):
                    raise ValueError(
                        "one fixed stateless artifact has conflicting input routes"
                    )
                route_input_contracts[anchor.checkpoint_sha256] = route_binding
        opponent_decks[deck.deck_digest] = deck
        if anchor.belief_summary_path is not None:
            _verify_file(
                _path(anchor.belief_summary_path),
                size_bytes=_path(anchor.belief_summary_path).stat().st_size,
                sha256=str(anchor.belief_summary_sha256),
                label=f"historical belief {anchor.member_id}",
            )
    for item in config.curriculum.scripted:
        deck = canonicalize_deck(read_deck(_path(item.exact_deck_path)))
        if deck.deck_digest != item.exact_deck_digest:
            raise ValueError("scripted exact deck fingerprint mismatch")
        opponent_decks[deck.deck_digest] = deck
    card_feature_path = _path(model_config.card_encoder.feature_table_path)
    if _file_sha256(card_feature_path) != catalog_manifest.card_catalog_fingerprint:
        raise ValueError("model static-card table differs from public catalog")
    native_library_path: Path | None = None
    native_library_sha256: str | None = None
    if config.collection.backend in _NATIVE_BACKENDS:
        native_library_path, _native_abi = resolve_native_training_library()
        native_library_sha256 = _file_sha256(native_library_path)
    if config.collection.backend == "native_distributed":
        resolved_fingerprint = _fingerprint(
            _DISTRIBUTED_RESOLVED_CONFIG_DOMAIN,
            {
                "run_version": config.run.version,
                "model_config_fingerprint": model_config_fingerprint(model_config),
                "training_roster_fingerprint": roster_fingerprint,
                "public_deck_catalog_fingerprint": catalog.fingerprint,
                "collection": _distributed_semantic_collection(config),
                **(
                    {"opponent_pool_v2": _opponent_pool_config_payload(config)}
                    if config.opponent_pool_v2.enabled
                    else {}
                ),
                "deck_balance": config.deck_balance.model_dump(mode="json"),
                **(
                    {
                        "deck_allocation": {
                            "declaration": config.deck_allocation.model_dump(
                                mode="json"
                            ),
                            "resolved_target_shares": deck_target_shares,
                        }
                    }
                    if config.deck_allocation is not None
                    else {}
                ),
                "curriculum_config_fingerprint": curriculum_config.fingerprint,
                "ppo_config_fingerprint": config.ppo.fingerprint,
                **(
                    {
                        "optimizer_scope": config.optimizer_scope.model_dump(
                            mode="json",
                            exclude_none=True,
                        )
                    }
                    if config.optimizer_scope.mode != "full_model"
                    else {}
                ),
            },
        )
    else:
        # Preserve the v1 identity byte-for-byte for every historical local
        # backend. The new distributed runtime config is deliberately absent.
        resolved_fingerprint = _fingerprint(
            _RESOLVED_CONFIG_DOMAIN,
            {
                "run_version": config.run.version,
                "model_config_fingerprint": model_config_fingerprint(model_config),
                "training_roster_fingerprint": roster_fingerprint,
                "public_deck_catalog_fingerprint": catalog.fingerprint,
                "collection": config.collection.model_dump(mode="json"),
                "deck_balance": config.deck_balance.model_dump(mode="json"),
                **(
                    {
                        "deck_allocation": {
                            "declaration": config.deck_allocation.model_dump(
                                mode="json"
                            ),
                            "resolved_target_shares": deck_target_shares,
                        }
                    }
                    if config.deck_allocation is not None
                    else {}
                ),
                "curriculum_config_fingerprint": curriculum_config.fingerprint,
                "ppo_config_fingerprint": config.ppo.fingerprint,
                "native_library_sha256": native_library_sha256,
                **(
                    {
                        "optimizer_scope": config.optimizer_scope.model_dump(
                            mode="json",
                            exclude_none=True,
                        )
                    }
                    if config.optimizer_scope.mode != "full_model"
                    else {}
                ),
            },
        )
    placeholder = StatelessFragmentIdentity(
        schema_version=(2 if uses_generalist_sequence(model_config) else 1),
        horizon=config.collection.fragment_horizon,
        behavior_policy_version=0,
        behavior_policy_fingerprint="0" * 64,
        model_config_fingerprint=model_config_fingerprint(model_config),
        action_schema_fingerprint=POLICY_INPUT_SCHEMA_FINGERPRINT,
        public_context_fingerprint=PUBLIC_EVENT_SCHEMA_FINGERPRINT,
        card_catalog_fingerprint=catalog_manifest.card_catalog_fingerprint,
        public_deck_catalog_fingerprint=catalog.fingerprint,
        exact_registry_fingerprint=registry.resolved_registry_sha256,
        belief_target_semantics_fingerprint=_belief_semantics_fingerprint(),
        input_contract_fingerprint=input_contract.fingerprint,
        resolved_config_fingerprint=resolved_fingerprint,
        sequence_contract_fingerprint=(
            GENERALIST_SEQUENCE_CONTRACT_FINGERPRINT
            if uses_generalist_sequence(model_config)
            else None
        ),
    )
    policy_identity = StatelessPolicyIdentity(
        model_config_fingerprint=model_config_fingerprint(model_config),
        exact_registry_fingerprint=registry.resolved_registry_sha256,
        active_exact_deck_digests=tuple(sorted(active_decks)),
        action_schema_fingerprint=POLICY_INPUT_SCHEMA_FINGERPRINT,
        public_context_fingerprint=PUBLIC_EVENT_SCHEMA_FINGERPRINT,
        card_catalog_fingerprint=catalog_manifest.card_catalog_fingerprint,
        belief_target_semantics_fingerprint=_belief_semantics_fingerprint(),
        public_deck_catalog_fingerprint=catalog.fingerprint,
        input_contract_fingerprint=input_contract.fingerprint,
        resolved_config_fingerprint=resolved_fingerprint,
        fragment_static_contract_fingerprint=(placeholder.static_contract_fingerprint),
        curriculum_config_fingerprint=curriculum_config.fingerprint,
        pinned_manifest_fingerprint=pinned_fingerprint,
        scripted_manifest_fingerprint=scripted_fingerprint,
        training_roster_fingerprint=roster_fingerprint,
        sequence_contract_fingerprint=(
            GENERALIST_SEQUENCE_CONTRACT_FINGERPRINT
            if uses_generalist_sequence(model_config)
            else None
        ),
    )
    return _ResolvedResources(
        model_config=model_config,
        catalog=catalog,
        catalog_manifest=catalog_manifest,
        active_decks=active_decks,
        active_deck_labels=active_deck_labels,
        opponent_decks=opponent_decks,
        route_input_contracts=route_input_contracts,
        input_contract=input_contract,
        curriculum_config=curriculum_config,
        deck_balance_config=deck_balance_config,
        deck_target_shares=deck_target_shares,
        policy_identity=policy_identity,
        resolved_config_fingerprint=resolved_fingerprint,
        pinned_manifest_fingerprint=pinned_fingerprint,
        scripted_manifest_fingerprint=scripted_fingerprint,
        scripted_opponents={
            item.artifact.opponent_id: item for item in resolved_scripted
        },
        training_roster_fingerprint=roster_fingerprint,
        native_library_path=native_library_path,
        native_library_sha256=native_library_sha256,
    )


def _resolve_deck_target_shares(
    config: SimpleStatelessTrainingConfig,
    *,
    active_deck_labels: Mapping[str, str],
) -> dict[str, float]:
    """Resolve optional source labels into one complete digest probability map."""
    declaration = config.deck_allocation
    digests = tuple(sorted(active_deck_labels))
    if declaration is None or isinstance(
        declaration,
        StatelessDynamicDeckAllocationConfig,
    ):
        share = 1.0 / float(len(digests))
        return dict.fromkeys(digests, share)
    labels_to_digest = {label: digest for digest, label in active_deck_labels.items()}
    if len(labels_to_digest) != len(active_deck_labels):
        raise ValueError("active deck allocation labels are ambiguous")
    unknown = set(declaration.primary_deck_labels) - set(labels_to_digest)
    if unknown:
        raise ValueError(
            "primary deck labels are absent from the active registry: "
            + ", ".join(sorted(unknown))
        )
    primary = {labels_to_digest[label] for label in declaration.primary_deck_labels}
    auxiliary = set(digests) - primary
    if not auxiliary:
        raise ValueError("primary deck allocation requires auxiliary decks")
    primary_share = declaration.primary_probability / float(len(primary))
    auxiliary_share = (1.0 - declaration.primary_probability) / float(len(auxiliary))
    return {
        digest: primary_share if digest in primary else auxiliary_share
        for digest in digests
    }


def _opponent_pool_config_payload(
    config: SimpleStatelessTrainingConfig,
) -> dict[str, object]:
    """Serialize pool behavior without changing historical v1 identities."""
    payload = config.opponent_pool_v2.model_dump(mode="json")
    if config.opponent_pool_v2.behavior_version == 1:
        payload.pop("behavior_version")
        payload.pop("adaptive_allocation")
        payload.pop("role_budget_allocation")
    elif config.opponent_pool_v2.behavior_version == 2:
        payload.pop("adaptive_allocation")
        payload.pop("role_budget_allocation")
    elif config.opponent_pool_v2.behavior_version == 3:
        payload.pop("role_budget_allocation")
        payload.pop("planner_policy")
        payload.pop("past_self_artifacts")
        payload.pop("recent_artifacts")
    elif config.opponent_pool_v2.behavior_version in {4, 5}:
        payload.pop("adaptive_allocation")
        payload.pop("planner_policy")
        payload.pop("past_self_artifacts")
        payload.pop("recent_artifacts")
    return payload


def _distributed_semantic_collection(
    config: SimpleStatelessTrainingConfig,
) -> dict[str, object]:
    """Project collection semantics without host/runtime geometry."""
    collection = config.collection
    return {
        "backend": collection.backend,
        "native_trainable_decision_budget": (
            collection.native_trainable_decision_budget
        ),
        "native_sequence_rollout_precision": (
            collection.native_sequence_rollout_precision
        ),
        "rollout_artifact_semantics_fingerprint": (
            NATIVE_BFLOAT16_ARTIFACT_SEMANTICS_FINGERPRINT
        ),
        "pipeline_mode": collection.pipeline_mode,
        "fragment_horizon": collection.fragment_horizon,
        "mirror_bilateral_trajectories": (collection.mirror_bilateral_trajectories),
        "maximum_engine_steps": collection.maximum_engine_steps,
        "checkpoint_interval_updates": collection.checkpoint_interval_updates,
        "checkpoint_keep_last": collection.checkpoint_keep_last,
        "checkpoint_retain_every_versions": (
            collection.checkpoint_retain_every_versions
        ),
        "seed": collection.seed,
    }


def _model_from_startup(
    resources: _ResolvedResources,
    resume: LoadedStatelessCheckpointPair | None,
    *,
    supervised_state: Mapping[str, torch.Tensor] | None,
    bc_overlay_state: Mapping[str, torch.Tensor] | None,
    seed: int,
    registry_transition_plan: StatelessRegistryTransitionPlan | None = None,
    topology_transition_plan: StatelessTopologyTransitionPlan | None = None,
    sequence_context_transition_plan: (
        StatelessSequenceContextTransitionPlan | None
    ) = None,
    public_catalog_transition: StatelessPublicCatalogTransitionConfig | None = None,
    weights_only: bool = False,
) -> tuple[
    SimpleStatelessPolicyValueNet,
    int,
    dict[str, int] | StatelessTopologyModelAudit | None,
]:
    transition_plans = (
        registry_transition_plan,
        topology_transition_plan,
        sequence_context_transition_plan,
        public_catalog_transition,
    )
    if sum(plan is not None for plan in transition_plans) > 1:
        raise ValueError("stateless startup cannot apply multiple transition plans")
    if bc_overlay_state is not None and (
        resume is None
        or supervised_state is not None
        or registry_transition_plan is not None
        or topology_transition_plan is not None
        or sequence_context_transition_plan is not None
        or public_catalog_transition is not None
    ):
        raise ValueError("BC overlay cannot stack with another startup mutation")
    if supervised_state is not None and resume is not None:
        raise ValueError("supervised startup cannot stack with a checkpoint pair")
    if weights_only and (
        resume is None
        or supervised_state is not None
        or bc_overlay_state is not None
        or registry_transition_plan is not None
        or topology_transition_plan is not None
        or sequence_context_transition_plan is not None
        or public_catalog_transition is not None
    ):
        raise ValueError("weights-only startup cannot stack with another mutation")
    if resume is None and supervised_state is None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = SimpleStatelessPolicyValueNet(resources.model_config)
        report = simple_stateless_parameter_report(model, resources.model_config)
        validate_simple_stateless_parameter_report(report)
        return model, 0, None
    if resume is None:
        if supervised_state is None:
            raise RuntimeError("supervised startup state was not provided")
        model = SimpleStatelessPolicyValueNet(
            resources.model_config,
            load_static_features=False,
            initialize=False,
        )
        model.load_state_dict(dict(supervised_state), strict=True)
        return model, 0, None
    if weights_only:
        if resume.model_config_value != resources.model_config:
            raise ValueError("weights-only source model topology changed")
        model = SimpleStatelessPolicyValueNet(
            resources.model_config,
            load_static_features=False,
            initialize=False,
        )
        model.load_state_dict(resume.model_state, strict=True)
        return model, 0, None
    if bc_overlay_state is not None:
        model = SimpleStatelessPolicyValueNet(
            resources.model_config,
            load_static_features=False,
            initialize=False,
        )
        model.load_state_dict(dict(bc_overlay_state), strict=True)
        return model, resume.pair.version, None
    if registry_transition_plan is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = SimpleStatelessPolicyValueNet(
            resources.model_config,
            load_static_features=False,
            initialize=True,
        )
        registry_report = migrate_stateless_registry_model(
            target=model,
            source_state=resume.model_state,
            plan=registry_transition_plan,
        )
        parameter_report = simple_stateless_parameter_report(
            model,
            resources.model_config,
        )
        validate_simple_stateless_parameter_report(parameter_report)
        return model, resume.pair.version, registry_report
    if topology_transition_plan is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = SimpleStatelessPolicyValueNet(
            resources.model_config,
            load_static_features=False,
            initialize=True,
        )
        model.assert_v2_additions_output_inert()
        copied_target_names = {
            topology_transition_plan.target_tensor_name(source_name)
            for source_name in resume.model_state
        }
        inert_output_tensor_names = tuple(
            sorted(
                name
                for name, _parameter in model.v2_inert_output_named_parameters()
                if name not in copied_target_names
            )
        )
        if not inert_output_tensor_names:
            raise ValueError("topology transition introduced no inert output tensors")
        topology_report = migrate_stateless_topology_model(
            target=model,
            source_state=resume.model_state,
            plan=topology_transition_plan,
            inert_output_tensor_names=inert_output_tensor_names,
        )
        if (
            topology_report.source_state_fingerprint
            != resume.pair.policy_model_fingerprint
        ):
            raise ValueError("stateless topology source model state changed")
        model.assert_named_outputs_inert(inert_output_tensor_names)
        parameter_report = simple_stateless_parameter_report(
            model,
            resources.model_config,
        )
        validate_simple_stateless_parameter_report(parameter_report)
        return model, resume.pair.version, topology_report
    if sequence_context_transition_plan is not None:
        model = SimpleStatelessPolicyValueNet(
            resources.model_config,
            load_static_features=False,
            initialize=False,
        )
        model.load_state_dict(resume.model_state, strict=True)
        if (
            canonical_model_state_fingerprint(model)
            != resume.pair.policy_model_fingerprint
        ):
            raise ValueError("sequence context transition changed model state")
        return model, resume.pair.version, None
    if public_catalog_transition is not None:
        model = SimpleStatelessPolicyValueNet(
            resources.model_config,
            load_static_features=False,
            initialize=False,
        )
        model.load_state_dict(resume.model_state, strict=True)
        if (
            canonical_model_state_fingerprint(model)
            != resume.pair.policy_model_fingerprint
        ):
            raise ValueError("public catalog transition changed model state")
        return model, resume.pair.version, None
    if resume.model_config_value != resources.model_config:
        raise ValueError("exact resume resolved model topology changed")
    model = SimpleStatelessPolicyValueNet(
        resources.model_config,
        load_static_features=False,
        initialize=False,
    )
    model.load_state_dict(resume.model_state, strict=True)
    return model, resume.pair.version, None


def _validate_transition_source(
    source: LoadedStatelessCheckpointPair,
    *,
    resources: _ResolvedResources,
    ppo: SimpleStatelessPpoConfig,
    optimizer_scope: StatelessOptimizerScopeConfig,
    resume: StatelessResumeConfig,
) -> tuple[
    StatelessRegistryTransitionPlan | None,
    StatelessTopologyTransitionPlan | None,
    StatelessSequenceContextTransitionPlan | None,
    StatelessPublicCatalogTransitionConfig | None,
]:
    """Allow only explicitly bound schedule, execution, and GAE migrations."""
    _validate_transition_source_binding(
        observed_pair_version=source.pair.version,
        observed_pair_manifest_sha256=source.pair.pair_manifest_sha256,
        observed_curriculum_fingerprint=source.curriculum_state.config_fingerprint,
        resume=resume,
    )
    registry_transition_plan: StatelessRegistryTransitionPlan | None = None
    topology_transition_plan: StatelessTopologyTransitionPlan | None = None
    sequence_context_transition_plan: StatelessSequenceContextTransitionPlan | None = (
        None
    )
    public_catalog_transition = resume.public_catalog_transition
    optimizer_scope_transition = resume.optimizer_scope_transition
    if optimizer_scope_transition is not None and (
        source.model_config_value != resources.model_config
        or optimizer_scope.mode != optimizer_scope_transition.target_mode
        or optimizer_scope.private_learning_rate
        != optimizer_scope_transition.target_private_learning_rate
        or optimizer_scope.shared_learning_rate_initial
        != optimizer_scope_transition.target_shared_learning_rate_initial
        or optimizer_scope.shared_learning_rate_target
        != optimizer_scope_transition.target_shared_learning_rate_target
        or optimizer_scope.shared_learning_rate_warmup_start_update_index
        != optimizer_scope_transition.target_shared_learning_rate_warmup_start_update_index
        or optimizer_scope.shared_learning_rate_warmup_updates
        != optimizer_scope_transition.target_shared_learning_rate_warmup_updates
    ):
        raise ValueError(
            "optimizer-scope transition changed topology or differs from its target"
        )
    if (
        optimizer_scope_transition is not None
        and optimizer_scope_transition.target_mode == "hybrid"
        and optimizer_scope_transition.target_shared_learning_rate_warmup_start_update_index
        != source.pair.version
    ):
        raise ValueError(
            "hybrid shared learning-rate warmup must start at the source pair version"
        )
    if source.model_config_value != resources.model_config:
        registry_declaration = resume.registry_transition
        topology_declaration = resume.topology_transition
        if public_catalog_transition is not None:
            _validate_public_catalog_model_transition(
                source.model_config_value,
                resources.model_config,
                public_catalog_transition,
            )
        elif registry_declaration is not None:
            registry_transition_plan = build_stateless_registry_transition_plan(
                source.model_config_value,
                resources.model_config,
                registry_declaration,
            )
        elif topology_declaration is not None:
            validate_stateless_topology_source_pair(source, topology_declaration)
            topology_transition_plan = build_stateless_topology_transition_plan(
                source.model_config_value,
                resources.model_config,
                topology_declaration,
            )
        else:
            sequence_context_transition_plan = (
                build_stateless_sequence_context_transition_plan(
                    source.model_config_value,
                    resources.model_config,
                )
            )
    elif (
        resume.registry_transition is not None
        or resume.topology_transition is not None
        or public_catalog_transition is not None
    ):
        raise ValueError("stateless transition declaration is unnecessary")
    disallowed_ppo_changes = _disallowed_transition_ppo_changes(
        source.ppo_config,
        ppo,
        resume=resume,
    )
    if disallowed_ppo_changes:
        raise ValueError(
            "transition source changed PPO objective/optimizer fields: "
            + ", ".join(sorted(disallowed_ppo_changes))
        )
    if ppo.total_decisions is not None and ppo.logical_batch_decisions is None:
        raise ValueError(
            "decision-scheduled transition requires bounded logical batches"
        )
    target = resources.policy_identity
    invariant_fields = [
        "action_schema_fingerprint",
        "public_context_fingerprint",
        "card_catalog_fingerprint",
        "belief_target_semantics_fingerprint",
    ]
    source_scripted = (
        None
        if public_catalog_transition is None
        else public_catalog_transition.source_scripted_manifest_fingerprint
    )
    target_scripted = (
        None
        if public_catalog_transition is None
        else public_catalog_transition.target_scripted_manifest_fingerprint
    )
    if source_scripted is None:
        invariant_fields.append("scripted_manifest_fingerprint")
    elif (
        source.pair.identity.scripted_manifest_fingerprint != source_scripted
        or target.scripted_manifest_fingerprint != target_scripted
    ):
        raise ValueError(
            "scripted manifest transition declaration differs from resources"
        )
    if public_catalog_transition is None:
        invariant_fields.extend(
            ("public_deck_catalog_fingerprint", "input_contract_fingerprint")
        )
    if all(
        plan is None
        for plan in (
            registry_transition_plan,
            topology_transition_plan,
            sequence_context_transition_plan,
            public_catalog_transition,
        )
    ):
        invariant_fields.extend(
            (
                "model_config_fingerprint",
                "exact_registry_fingerprint",
                "active_exact_deck_digests",
                "training_roster_fingerprint",
            )
        )
    elif topology_transition_plan is not None:
        invariant_fields.append("active_exact_deck_digests")
    elif (
        sequence_context_transition_plan is not None
        or public_catalog_transition is not None
    ):
        invariant_fields.extend(
            (
                "exact_registry_fingerprint",
                "active_exact_deck_digests",
                "training_roster_fingerprint",
                "sequence_contract_fingerprint",
            )
        )
    for field in invariant_fields:
        if getattr(source.pair.identity, field) != getattr(target, field):
            raise ValueError(f"transition source changed invariant {field}")
    pinned_changed = (
        source.pair.identity.pinned_manifest_fingerprint
        != target.pinned_manifest_fingerprint
    )
    if pinned_changed != (resume.anchor_transition is not None):
        raise ValueError(
            "transition pinned-manifest change requires exactly one anchor declaration"
        )
    return (
        registry_transition_plan,
        topology_transition_plan,
        sequence_context_transition_plan,
        public_catalog_transition,
    )


def _validate_public_catalog_model_transition(
    source: SimpleStatelessModelConfig,
    target: SimpleStatelessModelConfig,
    declaration: StatelessPublicCatalogTransitionConfig,
) -> None:
    """Require the public catalog identity to be the only model-config change."""
    if (
        source.public_deck_catalog_fingerprint != declaration.source_catalog_fingerprint
        or target.public_deck_catalog_fingerprint
        != declaration.target_catalog_fingerprint
    ):
        raise ValueError("public catalog transition declaration differs from models")
    source_payload = source.model_dump(mode="python")
    source_payload["public_deck_catalog_fingerprint"] = (
        declaration.target_catalog_fingerprint
    )
    if source_payload != target.model_dump(mode="python"):
        raise ValueError("public catalog transition changed model topology")


def _validate_weights_only_source(
    source: LoadedStatelessCheckpointPair,
    *,
    resources: _ResolvedResources,
) -> None:
    """Require identical topology, exact routes, and inference contracts."""
    if source.model_config_value != resources.model_config:
        raise ValueError("weights-only source model topology changed")
    target = resources.policy_identity
    invariant_fields = (
        "model_config_fingerprint",
        "exact_registry_fingerprint",
        "active_exact_deck_digests",
        "action_schema_fingerprint",
        "public_context_fingerprint",
        "card_catalog_fingerprint",
        "belief_target_semantics_fingerprint",
        "public_deck_catalog_fingerprint",
        "input_contract_fingerprint",
        "training_roster_fingerprint",
        "sequence_contract_fingerprint",
    )
    for field in invariant_fields:
        if getattr(source.pair.identity, field) != getattr(target, field):
            raise ValueError(f"weights-only source changed invariant {field}")


def _import_founder_policy_artifact(
    output_dir: Path,
    source: DurablePolicyArtifact,
) -> DurablePolicyArtifact:
    """Durably retain the founding policy inside the new lineage."""
    source.verify()
    archive_dir = output_dir / "weights" / "opponent_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = (archive_dir / f"founder_{source.policy_sha256}.pt").resolve()
    imported = source.model_copy(update={"policy_path": destination})
    if destination.exists():
        imported.verify()
        return imported
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(source.policy_path, temporary)
        except OSError:
            shutil.copyfile(source.policy_path, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    imported.verify()
    return imported


def _create_supervised_lineage_founder(
    output_dir: Path,
    *,
    learner: SimpleStatelessLearner,
    resources: _ResolvedResources,
    curriculum: StatelessCurriculumController,
) -> DurablePolicyArtifact:
    """Publish and admit the supervised v0 policy as its lineage founder."""
    progress = _learner_settled_progress(learner)
    if progress != StatelessSettledTrainingProgress(
        rollout_window_index=0,
        optimizer_step_index=0,
        fresh_decisions_seen=0,
        lr_schedule_decisions_seen=0,
        pending_optimizer_window=None,
    ):
        raise RuntimeError("supervised lineage founder requires fresh learner state")
    policy_artifact = publish_stateless_policy_checkpoint(
        output_dir,
        version=0,
        model=learner.model,
        model_config=resources.model_config,
        identity=resources.policy_identity,
    )
    founder = _import_founder_policy_artifact(output_dir, policy_artifact)
    prepared = curriculum.prepare_past_self_admission(
        founder,
        snapshot_id="lineage-founder-v0",
        current_version=0,
        sampling_floor=curriculum.config.pfsp.probability_floor,
    )
    curriculum.commit_past_self_admission(prepared)
    return founder


def _create_pair_lineage_founder(
    output_dir: Path,
    *,
    source: LoadedStatelessCheckpointPair,
    curriculum: StatelessCurriculumController,
) -> DurablePolicyArtifact:
    """Import a pair policy as the founder of a reset lineage controller."""
    founder = _import_founder_policy_artifact(
        output_dir,
        source.pair.durable_artifact,
    )
    prepared = curriculum.prepare_past_self_admission(
        founder,
        snapshot_id=f"lineage-founder-v{source.pair.version}",
        current_version=source.pair.version,
        sampling_floor=curriculum.config.pfsp.probability_floor,
    )
    curriculum.commit_past_self_admission(prepared)
    return founder


def _create_public_catalog_transition_founder(
    output_dir: Path,
    *,
    learner: SimpleStatelessLearner,
    resources: _ResolvedResources,
    curriculum: StatelessCurriculumController,
) -> DurablePolicyArtifact:
    """Publish the preserved weights under the new catalog input identity."""
    version = learner.update_index
    policy_artifact = publish_stateless_policy_checkpoint(
        output_dir,
        version=version,
        model=learner.model,
        model_config=resources.model_config,
        identity=resources.policy_identity,
    )
    founder = _import_founder_policy_artifact(output_dir, policy_artifact)
    prepared = curriculum.prepare_past_self_admission(
        founder,
        snapshot_id=f"public-catalog-transition-founder-v{version}",
        current_version=version,
        sampling_floor=curriculum.config.pfsp.probability_floor,
    )
    curriculum.commit_past_self_admission(prepared)
    return founder


def _validate_bc_overlay_source(
    source: LoadedStatelessCheckpointPair,
    *,
    resources: _ResolvedResources,
    ppo: SimpleStatelessPpoConfig,
    resume: StatelessResumeConfig,
) -> None:
    """Require one same-topology pair with unchanged RL/controller contracts."""
    _validate_transition_source_binding(
        observed_pair_version=source.pair.version,
        observed_pair_manifest_sha256=source.pair.pair_manifest_sha256,
        observed_curriculum_fingerprint=source.curriculum_state.config_fingerprint,
        resume=resume,
    )
    expected_target_curriculum = resume.expected_target_curriculum_fingerprint
    if (
        expected_target_curriculum is None
        or resources.curriculum_config.fingerprint != expected_target_curriculum
        or source.curriculum_state.config_fingerprint
        != resources.curriculum_config.fingerprint
    ):
        raise ValueError("BC overlay changed its curriculum configuration")
    if source.ppo_config != ppo:
        raise ValueError("BC overlay changed its PPO configuration")
    target = resources.policy_identity
    invariant_fields = (
        "model_config_fingerprint",
        "exact_registry_fingerprint",
        "active_exact_deck_digests",
        "action_schema_fingerprint",
        "public_context_fingerprint",
        "card_catalog_fingerprint",
        "belief_target_semantics_fingerprint",
        "public_deck_catalog_fingerprint",
        "input_contract_fingerprint",
        "curriculum_config_fingerprint",
        "pinned_manifest_fingerprint",
        "scripted_manifest_fingerprint",
        "training_roster_fingerprint",
    )
    for field in invariant_fields:
        if getattr(source.pair.identity, field) != getattr(target, field):
            raise ValueError(f"BC overlay source changed invariant {field}")


def _disallowed_transition_ppo_changes(
    source: SimpleStatelessPpoConfig,
    target: SimpleStatelessPpoConfig,
    *,
    resume: StatelessResumeConfig | None = None,
) -> set[str]:
    """Separate authorized migrations from accidental PPO objective drift."""
    source_values = source.model_dump(mode="python")
    target_values = target.model_dump(mode="python")
    changed_fields = {
        field for field in source_values if source_values[field] != target_values[field]
    }
    allowed_changes = {
        "logical_batch_decisions",
        "microbatch_decisions",
        "warmup_decisions",
        "total_decisions",
    }
    gae_lambda_transition_from = (
        None if resume is None else resume.gae_lambda_transition_from
    )
    if gae_lambda_transition_from is not None:
        if source.gae_lambda != gae_lambda_transition_from:
            raise ValueError("declared source GAE lambda differs from transition pair")
        if target.gae_lambda == source.gae_lambda:
            raise ValueError("GAE lambda transition declaration is unnecessary")
        allowed_changes.add("gae_lambda")
    return changed_fields - allowed_changes


def _validate_transition_source_binding(
    *,
    observed_pair_version: int,
    observed_pair_manifest_sha256: str,
    observed_curriculum_fingerprint: str,
    resume: StatelessResumeConfig,
) -> None:
    """Reject a valid but unauthorized pair before applying a transition."""
    expected_version = resume.expected_source_pair_version
    expected_manifest = resume.expected_source_pair_manifest_sha256
    expected_curriculum = resume.expected_source_curriculum_fingerprint
    expected_target_curriculum = resume.expected_target_curriculum_fingerprint
    if (
        resume.mode not in {"transition", "bc_overlay"}
        or expected_version is None
        or expected_manifest is None
        or expected_curriculum is None
        or expected_target_curriculum is None
    ):
        raise ValueError("transition source authorization is incomplete")
    if observed_pair_version != expected_version:
        raise ValueError("transition source pair version is not authorized")
    if observed_pair_manifest_sha256 != expected_manifest:
        raise ValueError("transition source pair manifest is not authorized")
    if observed_curriculum_fingerprint != expected_curriculum:
        raise ValueError("transition source curriculum is not authorized")


def _materialization_settled_progress(
    source: LoadedStatelessCheckpointPair,
    *,
    target_ppo: SimpleStatelessPpoConfig,
) -> StatelessSettledTrainingProgress:
    """Retain every recorded PPO clock for a publish-only topology conversion."""
    progress = source.settled_progress
    if source.ppo_config != target_ppo:
        raise ValueError("materialize-only transition changed its PPO config")
    if not source.settled_progress_recorded or progress is None:
        raise ValueError("materialize-only transition requires settled source cursors")
    return progress


def _transition_settled_progress(
    source: LoadedStatelessCheckpointPair,
    *,
    target_ppo: SimpleStatelessPpoConfig,
) -> StatelessSettledTrainingProgress:
    """Create explicit new-lineage cursors from one settled source pair."""
    source_progress = source.settled_progress
    optimizer_step_index = (
        source.update_index
        if source_progress is None
        else source_progress.optimizer_step_index
    )
    if target_ppo.total_decisions is None:
        lr_schedule_decisions_seen = (
            0 if source_progress is None else source_progress.lr_schedule_decisions_seen
        )
    elif source.ppo_config.total_decisions is None:
        lr_schedule_decisions_seen = _legacy_decision_schedule_anchor(
            source.ppo_config,
            completed_updates=source.update_index,
            target_ppo=target_ppo,
        )
    elif (
        source_progress is not None
        and source.ppo_config.warmup_decisions == target_ppo.warmup_decisions
        and source.ppo_config.total_decisions == target_ppo.total_decisions
    ):
        lr_schedule_decisions_seen = source_progress.lr_schedule_decisions_seen
    else:
        raise ValueError(
            "cannot infer a decision-schedule cursor from this transition source"
        )
    return StatelessSettledTrainingProgress(
        rollout_window_index=source.update_index,
        optimizer_step_index=optimizer_step_index,
        fresh_decisions_seen=0,
        lr_schedule_decisions_seen=lr_schedule_decisions_seen,
        pending_optimizer_window=None,
    )


def _legacy_decision_schedule_anchor(
    source_ppo: SimpleStatelessPpoConfig,
    *,
    completed_updates: int,
    target_ppo: SimpleStatelessPpoConfig,
) -> int:
    """Map completed legacy Adam steps onto the target sample schedule."""
    if completed_updates < 0:
        raise ValueError("completed update count must be non-negative")
    if target_ppo.warmup_decisions is None or target_ppo.total_decisions is None:
        raise ValueError("target PPO config has no decision schedule")
    if (
        source_ppo.warmup_decisions is not None
        or source_ppo.total_decisions is not None
    ):
        raise ValueError("source PPO config is not a legacy update schedule")
    if source_ppo.warmup_updates > 0 and completed_updates <= source_ppo.warmup_updates:
        progress = completed_updates / float(source_ppo.warmup_updates)
        return round(progress * target_ppo.warmup_decisions)
    completed_step_index = max(completed_updates - 1, 0)
    source_decay = max(
        source_ppo.total_updates - source_ppo.warmup_updates,
        1,
    )
    decay_progress = min(
        max(
            completed_step_index - source_ppo.warmup_updates,
            0,
        )
        / float(source_decay),
        1.0,
    )
    target_decay = target_ppo.total_decisions - target_ppo.warmup_decisions
    return target_ppo.warmup_decisions + round(decay_progress * target_decay)


def _write_transition_source(
    output_dir: Path,
    *,
    manifest_path: Path,
    source: LoadedStatelessCheckpointPair,
    target_identity: StatelessPolicyIdentity,
    target_ppo: SimpleStatelessPpoConfig,
    transition_progress: StatelessSettledTrainingProgress,
    migrated_curriculum_state: StatelessCurriculumState | None,
    registry_transition_plan: StatelessRegistryTransitionPlan | None,
    topology_transition_plan: StatelessTopologyTransitionPlan | None,
    sequence_context_transition_plan: (StatelessSequenceContextTransitionPlan | None),
    public_catalog_transition: StatelessPublicCatalogTransitionConfig | None,
    optimizer_scope_transition: StatelessOptimizerScopeTransitionConfig | None,
    deck_balance_transition: StatelessDeckBalanceTransitionConfig | None,
    deck_balance_transition_audit: StatelessDeckBalanceTransitionAudit | None,
    curriculum_lane_coverage_rebase: (
        StatelessCurriculumLaneCoverageRebaseConfig | None
    ),
    curriculum_lane_coverage_rebase_audit: (
        StatelessCurriculumLaneCoverageRebaseAudit | None
    ),
    target_optimizer_scope: StatelessOptimizerScopeConfig,
    preserve_opponent_pool_state: bool,
) -> StatelessStartupReport:
    """Persist the immutable source binding and controller migration mode."""
    source_curriculum_fingerprint = stateless_curriculum_state_fingerprint(
        source.curriculum_state
    )
    source_opponent_pool_fingerprint = (
        None
        if source.opponent_pool_state is None
        else source.opponent_pool_state.fingerprint
    )
    if (curriculum_lane_coverage_rebase is None) != (
        curriculum_lane_coverage_rebase_audit is None
    ):
        raise RuntimeError("lane-coverage rebase declaration/audit mismatch")
    if curriculum_lane_coverage_rebase_audit is not None:
        assert curriculum_lane_coverage_rebase is not None
        lane_audit = curriculum_lane_coverage_rebase_audit
        lane_declaration = curriculum_lane_coverage_rebase
        if (
            lane_audit.source_state_fingerprint
            != source_curriculum_fingerprint
            or lane_audit.source_config_fingerprint
            != lane_declaration.source_config_fingerprint
            or lane_audit.target_config_fingerprint
            != lane_declaration.target_config_fingerprint
            or lane_audit.assignment_cursor
            != lane_declaration.source_assignment_cursor
            or lane_audit.source_lane_coverage
            != lane_declaration.source_lane_coverage
            or lane_audit.target_lane_mix != lane_declaration.target_lane_mix
            or lane_audit.target_lane_coverage
            != lane_declaration.target_lane_coverage
            or migrated_curriculum_state is None
            or migrated_curriculum_state.assignment_cursor
            != lane_audit.assignment_cursor
            or migrated_curriculum_state.config_fingerprint
            != lane_audit.target_config_fingerprint
            or migrated_curriculum_state.lane_coverage
            != lane_audit.target_lane_coverage
        ):
            raise RuntimeError("lane-coverage rebase audit differs from declaration")
    return _write_startup_report(
        output_dir / "transition_source.json",
        kind="transition_source",
        payload={
            "format": "simple_stateless_training_transition_v3",
            "controller_state_mode": (
                "fresh" if migrated_curriculum_state is None else "preserved_settled"
            ),
            "source_pair_manifest_path": str(manifest_path.resolve()),
            "source_pair_manifest_sha256": (source.pair.pair_manifest_sha256),
            "source_policy_sha256": source.pair.policy_sha256,
            "source_learner_state_sha256": (source.pair.learner_state_sha256),
            "source_curriculum_state_fingerprint": (source_curriculum_fingerprint),
            "target_curriculum_state_fingerprint": (
                None
                if migrated_curriculum_state is None
                else stateless_curriculum_state_fingerprint(migrated_curriculum_state)
            ),
            "source_opponent_pool_state_fingerprint": (
                source_opponent_pool_fingerprint
            ),
            "target_opponent_pool_state_fingerprint": (
                source_opponent_pool_fingerprint
                if preserve_opponent_pool_state
                else None
            ),
            "curriculum_member_transition": (
                None
                if migrated_curriculum_state is None
                else _curriculum_member_transition_summary(
                    source.curriculum_state,
                    migrated_curriculum_state,
                )
            ),
            "source_identity": source.pair.identity.model_dump(mode="json"),
            "target_identity": target_identity.model_dump(mode="json"),
            "registry_transition": (
                None
                if registry_transition_plan is None
                else registry_transition_plan.summary
            ),
            "optimizer_scope_transition": (
                None
                if optimizer_scope_transition is None
                else optimizer_scope_transition.model_dump(mode="json")
            ),
            "deck_balance_transition": (
                None
                if deck_balance_transition is None
                else deck_balance_transition.model_dump(mode="json")
            ),
            "deck_balance_transition_audit": (
                None
                if deck_balance_transition_audit is None
                else deck_balance_transition_audit.model_dump(mode="json")
            ),
            "curriculum_lane_coverage_rebase": (
                None
                if curriculum_lane_coverage_rebase is None
                else curriculum_lane_coverage_rebase.model_dump(mode="json")
            ),
            "curriculum_lane_coverage_rebase_audit": (
                None
                if curriculum_lane_coverage_rebase_audit is None
                else curriculum_lane_coverage_rebase_audit.model_dump(mode="json")
            ),
            "target_optimizer_scope": target_optimizer_scope.model_dump(mode="json"),
            **(
                {}
                if topology_transition_plan is None
                else {"topology_transition": topology_transition_plan.summary}
            ),
            **(
                {}
                if sequence_context_transition_plan is None
                else {
                    "sequence_context_transition": (
                        sequence_context_transition_plan.summary
                    )
                }
            ),
            **(
                {}
                if public_catalog_transition is None
                else {
                    "public_catalog_transition": {
                        **public_catalog_transition.model_dump(mode="json"),
                        "model_state_preserved": True,
                        "optimizer_state_preserved": True,
                        "past_self_members_retired": True,
                        "opponent_pool_state_reset": True,
                    }
                }
            ),
            "source_ppo_config": source.ppo_config.model_dump(mode="json"),
            "source_ppo_config_fingerprint": source.ppo_config.fingerprint,
            "target_ppo_config": target_ppo.model_dump(mode="json"),
            "target_ppo_config_fingerprint": target_ppo.fingerprint,
            "source_settled_progress_recorded": (source.settled_progress_recorded),
            "source_settled_progress": (
                None
                if source.settled_progress is None
                else source.settled_progress.model_dump(mode="json")
            ),
            "target_initial_settled_progress": (
                transition_progress.model_dump(mode="json")
            ),
            "lr_schedule_anchor_derivation": (
                "legacy_completed_optimizer_progress"
                if source.ppo_config.total_decisions is None
                and target_ppo.total_decisions is not None
                else "preserved"
            ),
            "version": source.pair.version,
        },
    )


def _curriculum_member_transition_summary(
    source: StatelessCurriculumState,
    target: StatelessCurriculumState,
) -> dict[str, object]:
    """Describe one settled PFSP membership transition without rewriting history."""
    target_member_ids = {member.member_id for member in target.members}
    source_member_ids = {member.member_id for member in source.members}
    removed = tuple(
        member for member in source.members if member.member_id not in target_member_ids
    )
    added = tuple(
        member for member in target.members if member.member_id not in source_member_ids
    )
    return {
        "source_members": len(source.members),
        "target_members": len(target.members),
        "removed_members": len(removed),
        "added_members": len(added),
        "removed_member_ids": sorted(member.member_id for member in removed),
        "added_member_ids": sorted(member.member_id for member in added),
        "source_past_self_members": sum(
            member.source == "past_self" for member in source.members
        ),
        "target_past_self_members": sum(
            member.source == "past_self" for member in target.members
        ),
        "source_historical_anchors": sum(member.pinned for member in source.members),
        "target_historical_anchors": sum(member.pinned for member in target.members),
        "removed_past_self_deck_digests": sorted(
            {
                member.exact_deck_digest
                for member in removed
                if member.source == "past_self"
            }
        ),
    }


def _write_topology_transition_report(
    output_dir: Path,
    report: StatelessTopologyTransitionAuditReport,
) -> StatelessStartupReport:
    """Atomically persist the validated topology and optimizer transplant."""
    return _write_startup_report(
        output_dir / "topology_transition_report.json",
        kind="topology_transition",
        payload=report.model_dump(mode="json"),
    )


def _write_bc_overlay_report(
    output_dir: Path,
    report: StatelessBcOverlayAuditReport,
) -> StatelessStartupReport:
    return _write_startup_report(
        output_dir / "bc_overlay_report.json",
        kind="bc_overlay_audit",
        payload=report.model_dump(mode="json"),
    )


def _write_supervised_startup_report(
    output_dir: Path,
    plan: StatelessSupervisedStartupPlan,
) -> StatelessStartupReport:
    """Persist the selected full-model artifact identities before v0."""
    return _write_startup_report(
        output_dir / "supervised_startup_report.json",
        kind="supervised_initialization",
        payload=plan.audit.model_dump(mode="json"),
    )


def _write_weights_only_report(
    output_dir: Path,
    *,
    source: LoadedStatelessCheckpointPair,
    resources: _ResolvedResources,
) -> StatelessStartupReport:
    """Record the immutable source and every intentionally reset state family."""
    return _write_startup_report(
        output_dir / "weights_only_warm_start_report.json",
        kind="weights_only_warm_start",
        payload={
            "format": "simple-stateless-weights-only-warm-start-v1",
            "source_pair_version": source.pair.version,
            "source_pair_manifest_path": str(source.pair.pair_manifest_path),
            "source_pair_manifest_sha256": source.pair.pair_manifest_sha256,
            "source_policy_sha256": source.pair.policy_sha256,
            "source_model_state_fingerprint": (source.pair.policy_model_fingerprint),
            "target_resolved_config_fingerprint": (
                resources.resolved_config_fingerprint
            ),
            "reset_state": [
                "adam",
                "learning_rate_schedule",
                "fragment_replay",
                "curriculum_statistics",
                "deck_balance",
                "opponent_pool_statistics",
                "optimizer_step_cursor",
                "rollout_window_cursor",
            ],
        },
    )


def _write_startup_report(
    path: Path,
    *,
    kind: Literal[
        "bc_overlay_audit",
        "supervised_initialization",
        "topology_transition",
        "transition_source",
        "weights_only_warm_start",
    ],
    payload: Mapping[str, Any],
) -> StatelessStartupReport:
    """Atomically write and identify one report before pair publication."""
    report_path = path.resolve()
    atomic_write_bytes(
        report_path,
        json_payload(payload),
        overwrite=False,
    )
    return StatelessStartupReport(
        kind=kind,
        path=report_path,
        size_bytes=report_path.stat().st_size,
        sha256=_file_sha256(report_path),
    )


def _retarget_transition_opponent_pool_report(
    report: StatelessStartupReport,
    *,
    target_state: OpponentPoolCheckpointState | None,
) -> StatelessStartupReport:
    """Bind a settled deck-share migration to its constructed pool revision."""
    if report.kind != "transition_source":
        raise ValueError("opponent-pool target requires a transition-source report")
    try:
        payload = json.loads(report.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("transition-source report is not readable JSON") from error
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "simple_stateless_training_transition_v3"
        or "source_opponent_pool_state_fingerprint" not in payload
        or "target_opponent_pool_state_fingerprint" not in payload
    ):
        raise ValueError("transition-source opponent-pool binding is malformed")
    payload["target_opponent_pool_state_fingerprint"] = (
        None if target_state is None else target_state.fingerprint
    )
    atomic_write_bytes(
        report.path,
        json_payload(payload),
        overwrite=True,
    )
    return StatelessStartupReport(
        kind=report.kind,
        path=report.path,
        size_bytes=report.path.stat().st_size,
        sha256=_file_sha256(report.path),
    )


def _write_registry_transition_report(
    output_dir: Path,
    *,
    plan: StatelessRegistryTransitionPlan,
    model_report: dict[str, int],
    optimizer_report: dict[str, int],
    source_balance: StatelessDeckBalanceState,
    target_balance: StatelessDeckBalanceState | None,
) -> None:
    """Persist the applied model, optimizer, and settled-controller evidence."""
    if target_balance is None:
        raise ValueError("stateless registry transition omitted deck-balance migration")
    atomic_write_bytes(
        output_dir / "registry_transition_report.json",
        json_payload(
            {
                "format": "simple_stateless_registry_transition_report_v1",
                "plan": plan.summary,
                "model": model_report,
                "optimizer": optimizer_report,
                "deck_balance": {
                    "source_config_fingerprint": source_balance.config_fingerprint,
                    "target_config_fingerprint": target_balance.config_fingerprint,
                    "source_history_cells": len(source_balance.history),
                    "target_history_cells": len(target_balance.history),
                    "source_rolling_events": len(source_balance.rolling_events),
                    "target_rolling_events": len(target_balance.rolling_events),
                    "assignment_cursor": target_balance.assignment_cursor,
                    "decision_cursor": target_balance.decision_cursor,
                },
            }
        ),
        overwrite=False,
    )


def _write_optimizer_scope_transition_report(
    output_dir: Path,
    *,
    source: LoadedStatelessCheckpointPair,
    target_identity: StatelessPolicyIdentity,
    declaration: StatelessOptimizerScopeTransitionConfig,
    audit: (
        StatelessPrivateOptimizerTransitionAudit
        | StatelessHybridOptimizerTransitionAudit
    ),
) -> None:
    """Persist the exact source binding and applied optimizer state migration."""
    hybrid = isinstance(audit, StatelessHybridOptimizerTransitionAudit)
    atomic_write_bytes(
        output_dir
        / (
            "hybrid_optimizer_transition_report.json"
            if hybrid
            else "private_optimizer_transition_report.json"
        ),
        json_payload(
            {
                "format": (
                    "simple-stateless-hybrid-optimizer-transition-report-v1"
                    if hybrid
                    else "simple-stateless-private-optimizer-transition-report-v1"
                ),
                "source_pair_manifest_sha256": (source.pair.pair_manifest_sha256),
                "source_policy_sha256": source.pair.policy_sha256,
                "source_learner_state_sha256": source.pair.learner_state_sha256,
                "target_resolved_config_fingerprint": (
                    target_identity.resolved_config_fingerprint
                ),
                "declaration": declaration.model_dump(mode="json"),
                "audit": audit.model_dump(mode="json"),
            }
        ),
        overwrite=False,
    )


def _apply_transition_curriculum_lane_coverage_rebase(
    source: StatelessCurriculumState,
    *,
    target_config: StatelessCurriculumConfig,
    declaration: StatelessCurriculumLaneCoverageRebaseConfig | None,
    expected_source_config_fingerprint: str | None,
    expected_target_config_fingerprint: str | None,
) -> tuple[
    StatelessCurriculumState,
    StatelessCurriculumLaneCoverageRebaseAudit | None,
]:
    """Apply one explicitly bound lane-only rebase before epoch migration."""
    if declaration is None:
        return source, None
    if (
        expected_source_config_fingerprint is None
        or expected_target_config_fingerprint is None
        or declaration.source_config_fingerprint
        != expected_source_config_fingerprint
        or declaration.target_config_fingerprint
        != expected_target_config_fingerprint
    ):
        raise ValueError("lane-coverage rebase curriculum binding mismatch")
    if (
        target_config.fingerprint != declaration.target_config_fingerprint
        or target_config.lane_mix != declaration.target_lane_mix
    ):
        raise ValueError("lane-coverage rebase differs from target curriculum")
    return rebase_settled_stateless_curriculum_lane_coverage(
        source,
        source_config_fingerprint=declaration.source_config_fingerprint,
        target_config_fingerprint=declaration.target_config_fingerprint,
        target_lane_mix=declaration.target_lane_mix,
        expected_assignment_cursor=declaration.source_assignment_cursor,
        expected_source_lane_coverage=declaration.source_lane_coverage,
        expected_target_lane_coverage=declaration.target_lane_coverage,
    )


def _migrate_transition_curriculum_state(
    source: StatelessCurriculumState,
    *,
    target_config: StatelessCurriculumConfig,
    target_active_deck_digests: tuple[str, ...],
    registered_opponent_deck_digests: frozenset[str],
    expected_source_config_fingerprint: str | None,
    expected_target_config_fingerprint: str | None,
    target_anchors: tuple[PfspMember, ...] | None = None,
    anchor_transition: StatelessAnchorTransitionConfig | None = None,
    drop_replaceable_members: bool = False,
) -> StatelessCurriculumState:
    """Rebind one settled curriculum while pruning retired past-self routes."""
    if (
        expected_source_config_fingerprint is None
        or source.config_fingerprint != expected_source_config_fingerprint
    ):
        raise ValueError("controller-state transition source is not authorized")
    if (
        expected_target_config_fingerprint is None
        or target_config.fingerprint != expected_target_config_fingerprint
    ):
        raise ValueError("controller-state transition target is not authorized")
    if source.inflight:
        raise ValueError(
            "controller-state transition requires zero curriculum assignments in flight"
        )
    leased_members = tuple(
        member.member_id for member in source.members if member.leases != 0
    )
    if leased_members:
        raise ValueError("controller-state transition requires zero PFSP member leases")
    target_active_decks = frozenset(target_active_deck_digests)
    if (
        not target_active_decks
        or tuple(sorted(target_active_decks)) != target_active_deck_digests
    ):
        raise ValueError(
            "controller-state transition target active decks must be sorted and unique"
        )
    if not target_active_decks.issubset(registered_opponent_deck_digests):
        raise ValueError(
            "controller-state transition target active deck is not registered"
        )
    source_anchors = tuple(member for member in source.members if member.pinned)
    resolved_target_anchors = (
        source_anchors if target_anchors is None else target_anchors
    )
    source_anchor_ids = tuple(member.member_id for member in source_anchors)
    target_anchor_ids = tuple(member.member_id for member in resolved_target_anchors)
    if anchor_transition is None:
        if {
            member.member_id: member.bundle_fingerprint for member in source_anchors
        } != {
            member.member_id: member.bundle_fingerprint
            for member in resolved_target_anchors
        }:
            raise ValueError(
                "controller-state transition changed anchors without a declaration"
            )
        migrated_anchors = source_anchors
    else:
        if set(source_anchor_ids) != set(anchor_transition.source_member_ids):
            raise ValueError("anchor transition source membership is not authorized")
        if set(target_anchor_ids) != set(anchor_transition.target_member_ids):
            raise ValueError("anchor transition target membership is not authorized")
        migrated_anchors = resolved_target_anchors
    retained_replaceable = (
        ()
        if drop_replaceable_members
        else tuple(
            member
            for member in source.members
            if not member.pinned and member.exact_deck_digest in target_active_decks
        )
    )
    migrated_members = (*migrated_anchors, *retained_replaceable)
    member_ids = tuple(member.member_id for member in migrated_members)
    bundle_ids = tuple(member.bundle_fingerprint for member in migrated_members)
    if len(set(member_ids)) != len(member_ids) or len(set(bundle_ids)) != len(
        bundle_ids
    ):
        raise ValueError("controller-state transition produced duplicate PFSP identity")
    if any(
        member.exact_deck_digest not in registered_opponent_deck_digests
        for member in migrated_members
    ):
        raise ValueError(
            "controller-state transition retained an unregistered opponent deck"
        )
    active_replaceable_snapshots = {
        member.snapshot_id
        for member in migrated_members
        if not member.pinned and member.status == "active"
    }
    if (
        not target_config.retain_all_past_self
        and len(active_replaceable_snapshots)
        > target_config.replaceable_snapshot_capacity
    ):
        raise ValueError(
            "controller-state transition exceeds target PFSP snapshot capacity"
        )
    # ``active_artifacts_per_window`` limits the cohort-local working set, not
    # the resident member archive. The controller deterministically chooses
    # that subset when planning each window, so a settled transition may retain
    # more resident artifacts than the per-window limit.
    return source.model_copy(
        update={
            "config_fingerprint": target_config.fingerprint,
            "members": migrated_members,
        }
    )


def _require_settled_transition_balance_state(
    state: StatelessDeckBalanceState,
    *,
    target_config: StatelessDeckBalanceConfigValue,
) -> None:
    """Reject migrating deck-balance leases without their unfinished games."""
    if state.config_fingerprint != target_config.fingerprint:
        raise ValueError(
            "controller-state transition changed deck-balance configuration"
        )
    if state.inflight_assignments:
        raise ValueError(
            "controller-state transition requires zero deck-balance assignments "
            "in flight"
        )


def _fragment_identity(
    resources: _ResolvedResources,
    *,
    behavior_version: int,
    behavior_fingerprint: str,
    horizon: int,
) -> StatelessFragmentIdentity:
    identity = resources.policy_identity
    result = StatelessFragmentIdentity(
        schema_version=(2 if identity.sequence_contract_fingerprint is not None else 1),
        horizon=horizon,
        behavior_policy_version=behavior_version,
        behavior_policy_fingerprint=behavior_fingerprint,
        model_config_fingerprint=identity.model_config_fingerprint,
        action_schema_fingerprint=identity.action_schema_fingerprint,
        public_context_fingerprint=identity.public_context_fingerprint,
        card_catalog_fingerprint=identity.card_catalog_fingerprint,
        public_deck_catalog_fingerprint=identity.public_deck_catalog_fingerprint,
        exact_registry_fingerprint=identity.exact_registry_fingerprint,
        belief_target_semantics_fingerprint=(
            identity.belief_target_semantics_fingerprint
        ),
        input_contract_fingerprint=identity.input_contract_fingerprint,
        resolved_config_fingerprint=identity.resolved_config_fingerprint,
        sequence_contract_fingerprint=(identity.sequence_contract_fingerprint),
    )
    if (
        result.static_contract_fingerprint
        != identity.fragment_static_contract_fingerprint
    ):
        raise RuntimeError("runtime fragment contract differs from policy identity")
    return result


def _distributed_worker_contract(
    resources: _ResolvedResources,
) -> NativeRolloutWorkerIdentity:
    """Build the host-independent compatibility gate for remote workers."""
    identity = resources.policy_identity
    native_library_path, _native_abi = resolve_native_training_library()
    sequence = resources.model_config.sequence
    source_identity = resolve_training_source_identity(_REPO_ROOT)
    return NativeRolloutWorkerIdentity(
        worker_id="coordinator-contract",
        session_id="coordinator-contract",
        runtime_fingerprint="0" * 64,
        source_git_commit=source_identity.source_git_commit,
        source_snapshot_fingerprint=(source_identity.training_source_fingerprint),
        native_library_fingerprint=_file_sha256(native_library_path),
        native_abi_version=NATIVE_TRAINING_ABI_VERSION,
        engine_fact_contract_fingerprint=(
            ProspectiveEngineFactProducer(
                sampler=BeliefSampler(config=sequence.engine_facts.sampler),
                config=sequence.engine_facts,
            ).contract_fingerprint
            if sequence is not None and sequence.engine_facts.enabled
            else None
        ),
        feature_schema_fingerprint=(identity.fragment_static_contract_fingerprint),
        card_catalog_fingerprint=identity.card_catalog_fingerprint,
        static_features_fingerprint=_file_sha256(
            _path(resources.model_config.card_encoder.feature_table_path)
        ),
        exact_registry_fingerprint=identity.exact_registry_fingerprint,
        scripted_opponents_fingerprint=(resources.scripted_manifest_fingerprint),
        historical_opponents_fingerprint=(resources.pinned_manifest_fingerprint),
        model_config_fingerprint=identity.model_config_fingerprint,
        resolved_config_fingerprint=identity.resolved_config_fingerprint,
    )


def _publish_pair(
    output_dir: Path,
    *,
    learner: SimpleStatelessLearner,
    resources: _ResolvedResources,
    writer: CompactFragmentShardWriter,
    balance: StatelessDeckBalanceSampler,
    balance_state: StatelessDeckBalanceState | None = None,
    curriculum_state: StatelessCurriculumState,
    opponent_pool_state: OpponentPoolCheckpointState | None = None,
    curriculum_predecessor_state: StatelessCurriculumState | None = None,
    policy_artifact: DurablePolicyArtifact | None = None,
    policy_publication: VerifiedPolicyPublication | None = None,
    fragments_seen: int,
    fragments_stale: int,
    startup_provenance: StatelessStartupProvenance | None = None,
) -> StatelessCheckpointPair:
    version = learner.update_index
    if policy_artifact is not None and policy_publication is not None:
        raise ValueError("policy artifact and verified publication are exclusive")
    if policy_publication is None and policy_artifact is None:
        policy_publication = publish_verified_stateless_policy_checkpoint(
            output_dir,
            version=version,
            model=learner.model,
            model_config=resources.model_config,
            identity=resources.policy_identity,
        )
    if policy_publication is not None:
        fingerprint = policy_publication.artifact.policy_model_fingerprint
    else:
        if policy_artifact is None:
            raise AssertionError("checkpoint policy artifact is unavailable")
        fingerprint = policy_artifact.policy_model_fingerprint
    return publish_stateless_checkpoint_pair(
        output_dir,
        version=version,
        model=learner.model,
        model_config=resources.model_config,
        identity=resources.policy_identity,
        ppo_config=learner.config,
        optimizer_state=learner.optimizer.state_dict(),
        update_index=version,
        settled_progress=_learner_settled_progress(learner),
        fragment_recovery=writer.manifest,
        deck_balance_state=balance.state if balance_state is None else balance_state,
        curriculum_state=curriculum_state,
        opponent_pool_state=opponent_pool_state,
        staleness_state=StatelessStalenessState(
            behavior_policy_version=version,
            behavior_policy_fingerprint=fingerprint,
            oldest_accepted_behavior_version=max(
                0,
                version - learner.config.maximum_version_age,
            ),
            fragments_seen=fragments_seen,
            fragments_stale=fragments_stale,
        ),
        policy_artifact=policy_artifact,
        policy_publication=policy_publication,
        curriculum_predecessor_state=curriculum_predecessor_state,
        startup_provenance=startup_provenance,
    )


def _submit_pair(
    output_dir: Path,
    *,
    publisher: AsyncStatelessCheckpointPairPublisher,
    learner: SimpleStatelessLearner,
    resources: _ResolvedResources,
    writer: CompactFragmentShardWriter,
    balance: StatelessDeckBalanceSampler,
    balance_state: StatelessDeckBalanceState | None = None,
    curriculum_state: StatelessCurriculumState,
    opponent_pool_state: OpponentPoolCheckpointState | None = None,
    curriculum_predecessor_state: StatelessCurriculumState | None = None,
    policy_publication: VerifiedPolicyPublication | None = None,
    fragments_seen: int,
    fragments_stale: int,
    startup_provenance: StatelessStartupProvenance | None = None,
) -> Future[PublishedStatelessCheckpointPair]:
    """Detach a settled pair and hand its durable write to the background."""
    version = learner.update_index
    if policy_publication is None:
        policy_publication = publish_verified_stateless_policy_checkpoint(
            output_dir,
            version=version,
            model=learner.model,
            model_config=resources.model_config,
            identity=resources.policy_identity,
        )
    fingerprint = policy_publication.artifact.policy_model_fingerprint
    return publisher.submit(
        output_dir,
        version=version,
        model_config=resources.model_config,
        identity=resources.policy_identity,
        ppo_config=learner.config,
        optimizer_state=learner.optimizer.state_dict(),
        update_index=version,
        settled_progress=_learner_settled_progress(learner),
        fragment_recovery=writer.manifest,
        deck_balance_state=balance.state if balance_state is None else balance_state,
        curriculum_state=curriculum_state,
        opponent_pool_state=opponent_pool_state,
        staleness_state=StatelessStalenessState(
            behavior_policy_version=version,
            behavior_policy_fingerprint=fingerprint,
            oldest_accepted_behavior_version=max(
                0,
                version - learner.config.maximum_version_age,
            ),
            fragments_seen=fragments_seen,
            fragments_stale=fragments_stale,
        ),
        policy_publication=policy_publication,
        curriculum_predecessor_state=curriculum_predecessor_state,
        startup_provenance=startup_provenance,
    )


def _settle_pending_checkpoint(
    pending: _PendingCheckpoint,
    *,
    publisher: AsyncStatelessCheckpointPairPublisher,
    output_dir: Path,
    config: SimpleStatelessTrainingConfig,
    curriculum: StatelessCurriculumController,
    learner_metric_writer: NonBlockingLearnerMetricWriter | None,
) -> StatelessCheckpointPair:
    """Cross the durability barrier before the next learner mutation."""
    wait_started_at = time.perf_counter()
    published = pending.future.result()
    barrier_result = publisher.barrier()
    wait_seconds = time.perf_counter() - wait_started_at
    if barrier_result != published:
        raise RuntimeError("checkpoint publisher barrier returned a different pair")
    pending.timing.update(
        {
            "checkpoint_freeze_seconds": published.freeze_seconds,
            "checkpoint_background_seconds": published.background_seconds,
            "checkpoint_wait_seconds": wait_seconds,
            "checkpoint_overlap_seconds": max(
                published.background_seconds - wait_seconds,
                0.0,
            ),
            "checkpoint_seconds": (
                published.freeze_seconds + published.background_seconds
            ),
        }
    )
    curriculum.persist_settled_state(
        pending.published_curriculum_state,
        expected_predecessor_fingerprint=pending.predecessor_fingerprint,
    )
    _prune_checkpoint_history(
        output_dir,
        config=config,
        curriculum_states=(
            curriculum.state,
            pending.published_curriculum_state,
        ),
    )
    if learner_metric_writer is not None:
        _run_noncritical(
            "learner metric publication",
            _publish_learner_metric,
            learner_metric_writer,
            run_version=config.run.version,
            update=pending.update,
            timing=pending.timing,
            pair=published.pair,
        )
    return published.pair


def _learner_settled_progress(
    learner: SimpleStatelessLearner,
) -> StatelessSettledTrainingProgress:
    """Snapshot every durable learner cursor without advancing the learner."""
    return StatelessSettledTrainingProgress(
        rollout_window_index=learner.update_index,
        optimizer_step_index=learner.optimizer_step_index,
        fresh_decisions_seen=learner.fresh_decisions_seen,
        lr_schedule_decisions_seen=learner.lr_schedule_decisions_seen,
        pending_optimizer_window=None,
    )


def _require_transition_materialization_state(
    learner: SimpleStatelessLearner,
    *,
    source_progress: StatelessSettledTrainingProgress | None,
    expected_progress: StatelessSettledTrainingProgress,
    source_ppo: SimpleStatelessPpoConfig,
    expected_ppo: SimpleStatelessPpoConfig,
) -> None:
    """Reject a materialization that would publish altered PPO state."""
    if source_ppo != expected_ppo:
        raise ValueError("materialize-only transition changed its PPO config")
    if source_progress is None or source_progress != expected_progress:
        raise ValueError("materialize-only transition remapped its source cursors")
    if learner.config != expected_ppo:
        raise ValueError("materialize-only transition changed its PPO config")
    if _learner_settled_progress(learner) != expected_progress:
        raise ValueError("materialize-only transition changed its learner cursors")


def _require_identity_materialization_controller_state(
    source: LoadedStatelessCheckpointPair,
    *,
    balance: StatelessDeckBalanceSampler,
    curriculum: StatelessCurriculumController,
    opponent_pool_state: OpponentPoolCheckpointState | None,
) -> None:
    """Reject identity rebinding that changes any sampling controller state."""
    if balance.state != source.deck_balance_state:
        raise ValueError("identity materialization changed deck-balance state")
    if curriculum.state != source.curriculum_state:
        raise ValueError("identity materialization changed curriculum state")
    if opponent_pool_state != source.opponent_pool_state:
        raise ValueError("identity materialization changed opponent-pool state")


def _publish_initial_pair(
    output_dir: Path,
    *,
    learner: SimpleStatelessLearner,
    resources: _ResolvedResources,
    writer: CompactFragmentShardWriter,
    balance: StatelessDeckBalanceSampler,
    curriculum: StatelessCurriculumController,
    opponent_pool_state: OpponentPoolCheckpointState | None,
    config: SimpleStatelessTrainingConfig,
    fragments_seen: int,
    fragments_stale: int,
    startup_provenance: StatelessStartupProvenance | None = None,
) -> StatelessCheckpointPair:
    """Publish the startup transaction before collection can acquire leases."""
    initial_predecessor = curriculum.persisted_state
    pair = _publish_pair(
        output_dir,
        learner=learner,
        resources=resources,
        writer=writer,
        balance=balance,
        curriculum_state=curriculum.state,
        opponent_pool_state=opponent_pool_state,
        curriculum_predecessor_state=initial_predecessor,
        fragments_seen=fragments_seen,
        fragments_stale=fragments_stale,
        startup_provenance=startup_provenance,
    )
    curriculum.persist_current_state(
        expected_predecessor_fingerprint=(
            None
            if initial_predecessor is None
            else stateless_curriculum_state_fingerprint(initial_predecessor)
        ),
    )
    _prune_checkpoint_history(
        output_dir,
        config=config,
        curriculum_states=(curriculum.state,),
    )
    return pair


def _reset_cuda_peak_memory(device: torch.device) -> None:
    """Start one window-local allocator peak measurement."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_memory_snapshot(
    device: torch.device,
    *,
    phase: str,
) -> dict[str, int]:
    """Return allocator state without synchronizing or mutating its cache."""
    if device.type != "cuda":
        return {}
    prefix = f"cuda_{phase}"
    stats = torch.cuda.memory_stats(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        f"{prefix}_allocated_bytes": torch.cuda.memory_allocated(device),
        f"{prefix}_reserved_bytes": torch.cuda.memory_reserved(device),
        f"{prefix}_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        f"{prefix}_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        f"{prefix}_device_free_bytes": free_bytes,
        f"{prefix}_device_used_bytes": total_bytes - free_bytes,
        f"{prefix}_device_total_bytes": total_bytes,
        f"{prefix}_inactive_split_bytes": int(
            stats.get("inactive_split_bytes.all.current", 0)
        ),
        f"{prefix}_allocation_retries": int(stats.get("num_alloc_retries", 0)),
        f"{prefix}_ooms": int(stats.get("num_ooms", 0)),
    }


def _trim_cuda_allocator_at_pipeline_drain(
    device: torch.device,
) -> dict[str, int]:
    """Return idle overlap high-water pages at a settled publication boundary."""
    if device.type != "cuda":
        return {}
    collected_objects = gc.collect()
    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    torch.cuda.empty_cache()
    allocated_after = torch.cuda.memory_allocated(device)
    reserved_after = torch.cuda.memory_reserved(device)
    free_after, total_after = torch.cuda.mem_get_info(device)
    if total_after != total_bytes:
        raise RuntimeError("CUDA device capacity changed while trimming allocator")
    return {
        "cuda_pipeline_drain_collected_objects": collected_objects,
        "cuda_pipeline_drain_allocated_before_bytes": allocated_before,
        "cuda_pipeline_drain_reserved_before_bytes": reserved_before,
        "cuda_pipeline_drain_device_free_before_bytes": free_before,
        "cuda_pipeline_drain_allocated_after_bytes": allocated_after,
        "cuda_pipeline_drain_reserved_after_bytes": reserved_after,
        "cuda_pipeline_drain_device_free_after_bytes": free_after,
        "cuda_pipeline_drain_released_bytes": max(
            reserved_before - reserved_after,
            0,
        ),
        "cuda_pipeline_drain_device_released_bytes": max(
            free_after - free_before,
            0,
        ),
    }


def _performance_reporter(
    config: SimpleStatelessTrainingConfig,
    *,
    resources: _ResolvedResources,
    output_dir: Path,
) -> TrainingPerformanceReporter | None:
    settings = config.performance
    if not settings.enabled:
        return None
    return TrainingPerformanceReporter(
        PerformanceReporterConfig(
            summary_path=output_dir / "performance" / "training_performance.json",
            tensorboard_dir=output_dir / "tensorboard" / "performance",
            count_stage="scored_terminal_single_writer",
            interval_seconds=settings.interval_seconds,
            rolling_window_minutes=settings.rolling_window_minutes,
            target_deck_labels=tuple(resources.active_deck_labels.values()),
            stationary_opponent_kinds=settings.stationary_opponent_kinds,
            tensorboard_enabled=settings.tensorboard_enabled,
            tensorboard_flush_seconds=settings.tensorboard_flush_seconds,
            recent_window_limit=settings.recent_window_limit,
            parquet_shard_windows=settings.parquet_shard_windows,
        )
    )


def _record_performance_outcomes(
    reporter: TrainingPerformanceReporter | None,
    *,
    collected: StatelessCollectionResult,
    active_deck_labels: dict[str, str],
    policy_version: int,
    member_sources: Mapping[str, str],
) -> None:
    if reporter is None:
        return
    try:
        reporter.observe_outcomes(
            worker_id="stateless-single-writer",
            outcomes=stateless_performance_outcomes(
                collected.assignments,
                collected.outcomes,
                active_deck_labels=active_deck_labels,
                policy_version=policy_version,
                opponent_strata_by_assignment=stateless_opponent_strata(
                    collected.assignments,
                    member_sources=member_sources,
                ),
            ),
        )
    except Exception:
        _LOGGER.exception(
            "performance outcome recording failed; training will continue"
        )


def _run_noncritical(
    operation: str,
    action: Callable[_P, object],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> None:
    """Run an observational side effect without giving it learner ownership."""
    try:
        action(*args, **kwargs)
    except Exception:
        _LOGGER.exception("%s failed; training will continue", operation)


def _performance_member_sources(
    state: StatelessCurriculumState,
) -> dict[str, str]:
    """Snapshot authoritative PFSP source semantics before leases can retire."""
    return {member.member_id: member.source for member in state.members}


def _write_collection_status(
    output_dir: Path,
    *,
    run_version: str,
    behavior_policy_version: int,
    report: StatelessCollectionReport,
) -> None:
    """Publish a valid Collection report before the learner starts."""
    atomic_write_bytes(
        output_dir / "collection_status.json",
        json_payload(
            {
                "format": "simple-stateless-collection-status-v1",
                "run_version": run_version,
                "behavior_policy_version": behavior_policy_version,
                "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "collection": report.model_dump(mode="json"),
            }
        ),
        overwrite=True,
    )


def _write_status(
    _output_dir: Path,
    *,
    config: SimpleStatelessTrainingConfig,
    learner: SimpleStatelessLearner,
    collection_reports: list[StatelessCollectionReport],
    update_reports: list[SimpleStatelessLearnerUpdate],
    timing_reports: list[dict[str, Any]],
    balance: StatelessDeckBalanceSampler,
    curriculum: StatelessCurriculumController,
    opponent_pool_v2: StatelessOpponentPoolV2 | None,
    started_at: float,
    latest_pair: StatelessCheckpointPair | None,
    learner_metric_writer: NonBlockingLearnerMetricWriter | None,
    status_writer: NonBlockingStatelessStatusWriter,
) -> None:
    elapsed_seconds = max(time.perf_counter() - started_at, 1.0e-9)
    curriculum_state = curriculum.state
    curriculum_state_fingerprint = stateless_curriculum_state_fingerprint(
        curriculum_state
    )
    member_probabilities = {
        deck_digest: curriculum.pfsp_probabilities(deck_digest)
        for deck_digest in balance.config.active_deck_digests
    }
    kept_decisions = sum(int(report["kept_decisions"]) for report in timing_reports)
    optimizer_steps = sum(int(report["optimizer_steps"]) for report in timing_reports)
    behavior_ages: Counter[int] = Counter()
    for timing in timing_reports:
        behavior_ages.update(
            {
                int(age): int(count)
                for age, count in dict(
                    timing.get("behavior_version_age_fragments", {})
                ).items()
            }
        )
    latest_checkpoint_timing = next(
        (
            timing
            for timing in reversed(timing_reports)
            if "checkpoint_background_seconds" in timing
        ),
        None,
    )
    payload = {
        "format": "simple-stateless-training-status-v1",
        "run_version": config.run.version,
        "update_index": learner.update_index,
        "target_updates": config.collection.training_updates,
        "optimizer_step_index": learner.optimizer_step_index,
        "fresh_decisions_seen": learner.fresh_decisions_seen,
        "lr_schedule_decisions_seen": learner.lr_schedule_decisions_seen,
        "target_decisions": config.ppo.total_decisions,
        "decision_budget_reached": _training_budget_reached(
            learner,
            config=config,
        ),
        "collection_pipeline": {
            "mode": config.collection.pipeline_mode,
            "prefetched_windows": sum(
                bool(report.get("collection_prefetched")) for report in timing_reports
            ),
            "behavior_version_age_fragments": {
                str(age): count for age, count in sorted(behavior_ages.items())
            },
            "fragments_stale": sum(report.fragments_stale for report in update_reports),
            "windows": [
                {
                    **{
                        key: timing.get(key)
                        for key in (
                            "update_index",
                            "collection_prefetched",
                            "collection_seconds",
                            "collection_wait_seconds",
                            "collection_overlap_seconds",
                            "next_collection_prefetched",
                            "learner_policy_version",
                            "behavior_policy_version",
                            "behavior_policy_version_age",
                            "behavior_version_age_fragments",
                            "preparation_seconds",
                            "learner_seconds",
                            "host_prepare_task_seconds",
                            "host_prepare_wait_seconds",
                            "checkpoint_seconds",
                            "total_seconds",
                            "cuda_learner_peak_allocated_bytes",
                            "cuda_learner_peak_reserved_bytes",
                            "cuda_learner_allocation_retries",
                            "cuda_learner_ooms",
                        )
                    },
                    "games_started": collection.games_started,
                    "games_finished": collection.games_finished,
                    "games_window_cutoff": collection.games_window_cutoff,
                    "games_immediate_window_cutoff": (
                        collection.games_immediate_window_cutoff
                    ),
                    "fragments": collection.fragments,
                    "kept_decisions": update.decisions,
                    "approximate_kl": update.approximate_kl,
                    "clip_fraction": update.clip_fraction,
                }
                for timing, collection, update in zip(
                    timing_reports,
                    collection_reports,
                    update_reports,
                    strict=True,
                )
            ],
        },
        "elapsed_seconds": elapsed_seconds,
        "throughput": {
            "windows": len(timing_reports),
            "optimizer_steps": optimizer_steps,
            "kept_decisions": kept_decisions,
            "kept_decisions_per_second": kept_decisions / elapsed_seconds,
        },
        "latest_timing": (None if not timing_reports else dict(timing_reports[-1])),
        "latest_checkpoint_timing": (
            None
            if latest_checkpoint_timing is None
            else {
                key: latest_checkpoint_timing.get(key)
                for key in (
                    "update_index",
                    "checkpoint_freeze_seconds",
                    "checkpoint_background_seconds",
                    "checkpoint_wait_seconds",
                    "checkpoint_overlap_seconds",
                    "checkpoint_seconds",
                )
            }
        ),
        "latest_collection": (
            None
            if not collection_reports
            else collection_reports[-1].model_dump(mode="json")
        ),
        "latest_update": (
            None if not update_reports else update_reports[-1].model_dump(mode="json")
        ),
        "deck_balance": balance.report().model_dump(mode="json"),
        "deck_allocation": _deck_allocation_status(
            balance,
            None if not update_reports else update_reports[-1],
            adaptive_target_shares=(
                opponent_pool_v2.state.candidate_target_shares
                if opponent_pool_v2 is not None
                and isinstance(
                    opponent_pool_v2.state,
                    StatelessOpponentPoolAdaptiveState,
                )
                and opponent_pool_v2.state.candidate_target_shares
                else None
            ),
        ),
        "curriculum": {
            "lane_targets": curriculum.config.lane_mix.model_dump(mode="json"),
            "state_summary": {
                "state_fingerprint": curriculum_state_fingerprint,
                "generation": curriculum_state.generation,
                "assignment_cursor": curriculum_state.assignment_cursor,
                "members": len(curriculum_state.members),
                "active_members": sum(
                    member.status == "active" for member in curriculum_state.members
                ),
                "exact_statistics": len(curriculum_state.exact_statistics),
                "inflight_assignments": len(curriculum_state.inflight),
            },
            "detail_artifact": None,
            "execution_selection_owner": (
                "curriculum_member_pfsp"
                if opponent_pool_v2 is None
                else "opponent_pool_joint_adaptive"
                if opponent_pool_v2.config.behavior_version == 3
                else "opponent_pool_role_budget"
                if opponent_pool_v2.config.behavior_version in {4, 5}
                else "opponent_pool_v2_stratified"
            ),
        },
        "opponent_pool_v2": (
            {"enabled": False}
            if opponent_pool_v2 is None
            else _opponent_pool_v2_status(opponent_pool_v2)
        ),
        "latest_pair": (
            None
            if latest_pair is None
            else {
                "version": latest_pair.version,
                "manifest_path": str(latest_pair.pair_manifest_path),
                "manifest_sha256": latest_pair.pair_manifest_sha256,
            }
        ),
        "learner_metric_history": {
            "enabled": learner_metric_writer is not None,
            **(
                {}
                if learner_metric_writer is None
                else learner_metric_writer.status.model_dump(mode="json")
            ),
        },
    }
    adaptive_report = (
        None
        if opponent_pool_v2 is None
        else opponent_pool_v2.adaptive_allocation_report
    )
    allocation_payload = (
        None
        if adaptive_report is None
        else {
            "format": (
                "role-budget-opponent-allocation-envelope-v1"
                if adaptive_report.format == "role-budget-opponent-allocation-report-v1"
                else "adaptive-opponent-allocation-envelope-v1"
            ),
            "run_version": config.run.version,
            "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "report": adaptive_report.model_dump(mode="json"),
        }
    )
    status_writer.publish(
        StatelessStatusSnapshot(
            learner=payload,
            allocation=allocation_payload,
            curriculum_detail={
                "format": "simple-stateless-curriculum-observability-v1",
                "state_fingerprint": curriculum_state_fingerprint,
                "state": curriculum_state.model_dump(mode="json"),
                "member_pfsp_prior_probabilities": member_probabilities,
            },
        )
    )


def _deck_allocation_status(
    balance: StatelessDeckBalanceSampler,
    latest_update: SimpleStatelessLearnerUpdate | None,
    *,
    adaptive_target_shares: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Join collection targets and observed learner mass by exact deck."""
    balance_rows = balance.report().cells
    terminal_counts: Counter[str] = Counter()
    terminal_scores: defaultdict[str, float] = defaultdict(float)
    for event in balance.state.terminal_score_events:
        terminal_counts[event.deck_digest] += 1
        terminal_scores[event.deck_digest] += event.candidate_score
    learner_rows = (
        {}
        if latest_update is None
        else {item.deck_digest: item for item in latest_update.decks}
    )
    return {
        deck_digest: {
            "sampling_share": sum(
                item.rolling_share
                for item in balance_rows
                if item.deck_digest == deck_digest
            ),
            "target_share": sum(
                item.target_share
                for item in balance_rows
                if item.deck_digest == deck_digest
            ),
            "adaptive_target_share": (
                None
                if adaptive_target_shares is None
                else adaptive_target_shares[deck_digest]
            ),
            "recent_terminal_games": terminal_counts[deck_digest],
            "recent_terminal_score": (
                None
                if terminal_counts[deck_digest] == 0
                else terminal_scores[deck_digest] / terminal_counts[deck_digest]
            ),
            "learner_decisions": (
                None
                if deck_digest not in learner_rows
                else learner_rows[deck_digest].decisions
            ),
            "effective_learner_macro_weight": (
                None
                if deck_digest not in learner_rows
                else learner_rows[deck_digest].effective_macro_weight
            ),
            "effective_belief_macro_weight": (
                None
                if deck_digest not in learner_rows
                else learner_rows[deck_digest].belief_macro_weight
            ),
        }
        for deck_digest in balance.config.active_deck_digests
    }


def _opponent_pool_v2_status(
    opponent_pool: StatelessOpponentPoolV2,
) -> dict[str, Any]:
    """Build a bounded status view from the normalized committed V2 state."""
    checkpoint_state = opponent_pool.state
    state = opponent_pool.league_state
    entries = {entry.artifact_id: entry for entry in state.revision.entries}
    lineage = (
        checkpoint_state
        if isinstance(checkpoint_state, StatelessOpponentPoolLineageState)
        else None
    )
    adaptive = (
        checkpoint_state
        if isinstance(checkpoint_state, StatelessOpponentPoolAdaptiveState)
        else None
    )
    targets = (
        {
            item.stratum: item.target_fraction
            for item in opponent_pool.effective_planner_policy().strata
        }
        if lineage is not None
        else {}
    )
    decision_totals = (
        {item.stratum: item.decisions for item in lineage.stratum_decisions}
        if lineage is not None
        else {}
    )
    total_decisions = sum(decision_totals.values())
    adaptive_report = opponent_pool.adaptive_allocation_report
    return {
        "enabled": True,
        "behavior_version": opponent_pool.config.behavior_version,
        "state_fingerprint": checkpoint_state.fingerprint,
        "revision_fingerprint": state.revision.fingerprint,
        "revision_sequence": state.revision.revision_sequence,
        "generation": state.generation,
        "next_window_sequence": state.next_window_sequence,
        "last_committed_plan_id": state.last_committed_plan_id,
        "active_artifacts": len(state.revision.artifacts),
        "active_routes": len(state.revision.routes),
        "active_matchups": len(state.revision.active_matchups),
        "archive_artifacts": len(
            {
                member.policy_sha256
                for member in opponent_pool.curriculum.state.members
                if member.status == "active"
            }
        ),
        "founder_policy_sha256": (
            lineage.founder_policy_sha256
            if lineage is not None
            else None
            if adaptive is None
            else adaptive.founder_policy_sha256
        ),
        "adaptive_allocation": (
            None
            if adaptive is None
            else {
                "last_target_fingerprint": adaptive.last_target_fingerprint,
                "candidate_target_shares": adaptive.candidate_target_shares,
                "portfolio_target_mass": adaptive.portfolio_target_mass,
                "portfolio_decisions": adaptive.portfolio_decisions,
                "role_target_mass": adaptive.role_target_mass,
                "role_decisions": adaptive.role_decisions,
                "allocation_decision_clock": adaptive.allocation_decision_clock,
                "evidence_cells": len(adaptive.evidence),
                "terminal_games": sum(
                    item.total_terminal_games for item in adaptive.evidence
                ),
                "trainable_decisions": sum(
                    item.total_trainable_decisions for item in adaptive.evidence
                ),
                "last_window": (
                    None
                    if adaptive_report is None
                    else {
                        "window_sequence": adaptive_report.window_sequence,
                        "plan_id": adaptive_report.plan_id,
                        "target_fingerprint": adaptive_report.target_fingerprint,
                        "planned_games": sum(
                            item.planned_games for item in adaptive_report.candidates
                        ),
                        "actual_games": sum(
                            item.actual_games for item in adaptive_report.candidates
                        ),
                        "actual_decisions": sum(
                            item.actual_decisions for item in adaptive_report.candidates
                        ),
                    }
                ),
            }
        ),
        "strata": {
            stratum: {
                "artifacts": sum(
                    entry.stratum == stratum for entry in entries.values()
                ),
                "executed_games": sum(
                    state.artifact_executed_games(artifact_id)
                    for artifact_id, entry in entries.items()
                    if entry.stratum == stratum
                ),
                "learning_exposures": sum(
                    state.artifact_learning_exposures(artifact_id)
                    for artifact_id, entry in entries.items()
                    if entry.stratum == stratum
                ),
                "target_decision_share": targets.get(stratum),
                "accepted_trainable_decisions": decision_totals.get(stratum, 0),
                "actual_decision_share": (
                    None
                    if total_decisions == 0 or stratum not in decision_totals
                    else decision_totals[stratum] / float(total_decisions)
                ),
            }
            for stratum in (
                "protected",
                "counter_frontier",
                "recent",
                "age_diverse",
                "probe_reentry",
            )
        },
    }


def _learner_metric_record(
    *,
    run_version: str,
    update: SimpleStatelessLearnerUpdate,
    timing: dict[str, Any],
    pair: StatelessCheckpointPair,
) -> LearnerMetricRecord:
    """Build one scalar-only checkpoint observation without device access."""
    return LearnerMetricRecord(
        recorded_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        run_version=run_version,
        update_index=update.update_index,
        optimizer_step_index=int(timing["optimizer_step_index"]),
        checkpoint_version=pair.version,
        pair_manifest_sha256=pair.pair_manifest_sha256,
        policy_sha256=pair.policy_sha256,
        learner_state_sha256=pair.learner_state_sha256,
        decisions=update.decisions,
        fragments_seen=update.fragments_seen,
        fragments_stale=update.fragments_stale,
        loss=update.loss,
        policy_loss=update.policy_loss,
        value_loss=update.value_loss,
        belief_loss=update.belief_loss,
        entropy=update.entropy,
        ratio_mean=update.ratio_mean,
        approximate_kl=update.approximate_kl,
        clip_fraction=update.clip_fraction,
        gradient_norm=update.gradient_norm,
        learning_rate=update.learning_rate,
        kept_decisions_per_second=float(timing["kept_decisions_per_second"]),
        collection_seconds=float(timing["collection_seconds"]),
        learner_seconds=float(timing["learner_seconds"]),
        checkpoint_seconds=float(timing["checkpoint_seconds"]),
        total_seconds=float(timing["total_seconds"]),
        cuda_peak_allocated_bytes=timing.get("cuda_checkpoint_peak_allocated_bytes"),
        cuda_peak_reserved_bytes=timing.get("cuda_checkpoint_peak_reserved_bytes"),
        cuda_allocation_retries=timing.get("cuda_checkpoint_allocation_retries"),
        cuda_ooms=timing.get("cuda_checkpoint_ooms"),
    )


def _publish_learner_metric(
    writer: NonBlockingLearnerMetricWriter,
    *,
    run_version: str,
    update: SimpleStatelessLearnerUpdate,
    timing: dict[str, Any],
    pair: StatelessCheckpointPair,
) -> None:
    """Build and enqueue one non-critical learner metric record."""
    writer.publish(
        _learner_metric_record(
            run_version=run_version,
            update=update,
            timing=timing,
            pair=pair,
        )
    )


def _require_external_curriculum_state(
    path: Path,
    resume: LoadedStatelessCheckpointPair,
) -> None:
    if not path.is_file():
        if (
            resume.curriculum_predecessor_recorded
            and resume.curriculum_predecessor_fingerprint is None
        ):
            publish_external_stateless_curriculum_state(
                path,
                resume.curriculum_state,
            )
            return
        raise FileNotFoundError(
            "exact resume curriculum state file is missing from the run workspace"
        )
    actual = load_external_stateless_curriculum_state(path)
    if actual == resume.curriculum_state:
        return
    predecessor = resume.curriculum_predecessor_fingerprint
    if (
        resume.curriculum_predecessor_recorded
        and predecessor is not None
        and stateless_curriculum_state_fingerprint(actual) == predecessor
    ):
        publish_external_stateless_curriculum_state(
            path,
            resume.curriculum_state,
        )
        return
    raise ValueError("run curriculum state is ahead of or differs from sidecar")


def _require_exact_fragment_recovery(
    live: CompactFragmentManifest,
    checkpoint: CompactFragmentManifest,
) -> None:
    """Reject any live fragment cursor not authenticated by the exact sidecar."""
    if live != checkpoint:
        raise ValueError(
            "live fragment recovery manifest differs from exact checkpoint sidecar"
        )


def _prepare_output_dir(path: Path, *, fresh: bool) -> None:
    if fresh and path.exists() and _contains_fresh_training_artifacts(path):
        raise FileExistsError(f"fresh stateless output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _contains_fresh_training_artifacts(path: Path) -> bool:
    """Ignore only the outer supervisor's pre-launch ownership artifacts."""
    allowed_control_artifacts = {
        "learner_stderr.log",
        "learner_supervisor.json",
    }
    for entry in path.iterdir():
        if entry.name != "control" or not entry.is_dir():
            return True
        if any(item.name not in allowed_control_artifacts for item in entry.iterdir()):
            return True
    return False


def _belief_semantics_fingerprint() -> str:
    return hashlib.sha256(
        _BELIEF_SEMANTICS_DOMAIN + SIMPLE_BELIEF_TARGET_SEMANTICS.encode()
    ).hexdigest()


def _fingerprint(domain: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


def _verify_file(
    path: Path,
    *,
    size_bytes: int,
    sha256: str,
    label: str,
) -> None:
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


def _path(path: Path) -> Path:
    return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


__all__ = [
    "resolve_simple_stateless_training_resources",
    "run_simple_stateless_training",
    "validate_simple_stateless_training_config",
]
