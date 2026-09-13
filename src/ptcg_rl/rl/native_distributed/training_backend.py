"""H200-side owner for globally transactional native collection windows."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from torch import Tensor

from ptcg_rl.model.simple_stateless import SimpleStatelessModelConfig
from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.native_distributed.artifact import (
    EncodedBfloat16RolloutArtifact,
    prepare_bfloat16_rollout_artifact,
)
from ptcg_rl.rl.native_distributed.contracts import (
    NativeArtifactModelConfig,
    NativeAssignedGame,
    NativeCollectionWindow,
    NativeCollectionWindowReceipt,
    NativeRolloutWindowIdentity,
    NativeRolloutWorkerIdentity,
    native_assignment_plan_revision,
)
from ptcg_rl.rl.native_distributed.coordinator import (
    CommitCallback,
    NativeCollectionCoordinator,
    NativeCoordinatorQuorumError,
)
from ptcg_rl.rl.native_distributed.exposure_schedule import (
    select_v3_exposure_games,
)
from ptcg_rl.rl.native_distributed.server import NativeCoordinatorServer
from ptcg_rl.rl.native_distributed.transport import NativeDistributedEndpoints
from ptcg_rl.rl.stateless_checkpoint import load_stateless_policy_checkpoint
from ptcg_rl.rl.stateless_collection import (
    StatelessAssignedGame,
    StatelessCollectionResult,
)
from ptcg_rl.rl.stateless_curriculum import PfspMember
from ptcg_rl.rl.stateless_export import load_fixed_deck_checkpoint
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity
from ptcg_rl.rl.stateless_quota_assignments import StatelessQuotaAssignmentPlan
from ptcg_rl.rl.stateless_training_config import (
    SimpleStatelessTrainingConfig,
    StatelessHistoricalAnchorConfig,
)


@dataclass(frozen=True, slots=True)
class _PreparedArtifacts:
    """Window-local wire artifacts and their portable model topologies."""

    artifacts: tuple[EncodedBfloat16RolloutArtifact, ...]
    model_configs: tuple[NativeArtifactModelConfig, ...]
    prepared_now: tuple[EncodedBfloat16RolloutArtifact, ...]


@dataclass(frozen=True, slots=True)
class NativeCollectionPreparationTiming:
    """Model and assignment preparation work before a collection window."""

    assignment_planning_seconds: float
    source_fingerprint_seconds: float
    bf16_conversion_seconds: float
    wire_hashing_seconds: float
    window_open_seconds: float


class NativeDistributedTrainingBackend:
    """Prepare artifacts and own remote frames without learner inference state."""

    def __init__(
        self,
        config: SimpleStatelessTrainingConfig,
        *,
        expected_worker_contract: NativeRolloutWorkerIdentity,
        output_dir: Path,
    ) -> None:
        """Bind the trusted-LAN coordinator and initialize an idle window."""
        if config.collection.backend != "native_distributed":
            raise ValueError(
                "native distributed training backend requires its Hydra backend"
            )
        self.config = config
        self.output_dir = output_dir
        distributed = config.native_distributed
        quorum = distributed.quorum
        retry = distributed.retry
        coordinator = NativeCollectionCoordinator(
            expected_contract=expected_worker_contract,
            required_worker_ids=quorum.required_worker_ids,
            maximum_attempts_per_shard=retry.maximum_attempts_per_shard,
            lease_timeout_seconds=retry.lease_timeout_seconds,
            heartbeat_timeout_seconds=quorum.heartbeat_timeout_seconds,
            degrade_grace_seconds=quorum.degrade_grace_seconds,
            allow_degraded_after_startup=quorum.allow_degraded_after_startup,
            minimum_degraded_workers=quorum.minimum_degraded_workers,
            tail_start_fraction=distributed.scheduling.tail_start_fraction,
            early_inflight_reservation_fraction=(
                distributed.scheduling.early_inflight_reservation_fraction
            ),
            tail_inflight_reservation_fraction=(
                distributed.scheduling.tail_inflight_reservation_fraction
            ),
            tail_target_seconds=distributed.scheduling.tail_target_seconds,
            worker_yield_ewma_alpha=(distributed.scheduling.worker_yield_ewma_alpha),
            learner_clocked_max_trainable_decisions=(
                distributed.scheduling.learner_clocked_max_trainable_decisions
            ),
        )
        transport = distributed.transport
        endpoints = NativeDistributedEndpoints.tcp(
            transport.bind_host,
            control=transport.control_port,
            artifact=transport.artifact_port,
            data=transport.data_port,
        )
        self.server = NativeCoordinatorServer(
            coordinator,
            endpoints,
            io_threads=transport.io_threads,
            high_watermark=transport.socket_high_watermark,
            control_poll_interval_seconds=retry.control_poll_interval_seconds,
            status_path=output_dir / "control" / "native_distributed_status.json",
            status_interval_seconds=distributed.status.interval_seconds,
        )
        self._past_artifact_cache: dict[str, EncodedBfloat16RolloutArtifact] = {}
        self._past_model_config_cache: dict[str, NativeArtifactModelConfig] = {}
        self._past_artifact_cache_limit = (
            config.opponent_pool_v2.maximum_active_artifacts
        )
        self._active_assignments: tuple[StatelessAssignedGame, ...] = ()
        self._active_quota_plan: StatelessQuotaAssignmentPlan | None = None
        self._quota_assignments_adopted = False
        self._window_open = False
        self._committed_windows = 0
        self.last_preparation_timing = NativeCollectionPreparationTiming(
            assignment_planning_seconds=0.0,
            source_fingerprint_seconds=0.0,
            bf16_conversion_seconds=0.0,
            wire_hashing_seconds=0.0,
            window_open_seconds=0.0,
        )

    def assignment_pool_games(self, *, learner_clocked: bool = False) -> int:
        """Return a safe pool size while preserving one frozen PFSP plan."""
        target = self.config.collection.native_trainable_decision_budget
        if target is None:
            raise RuntimeError("native distributed global target is absent")
        profiles = self.config.native_distributed.worker_profiles.values()
        tiers = tuple(tier for profile in profiles for tier in profile.capacity_tiers)
        if not tiers:
            raise RuntimeError("native distributed capacity inventory is empty")
        minimum_decisions_per_game = min(
            tier.estimated_trainable_decisions / tier.concurrent_games for tier in tiers
        )
        required_workers = len(
            self.config.native_distributed.quorum.required_worker_ids
        )
        if required_workers <= 0:
            raise RuntimeError("native distributed worker inventory is empty")
        # Worker profiles describe compatible geometries, not the number of
        # processes using each profile.  Reserve a complete largest-tier wave
        # for every required worker so fast workers can claim follow-up work
        # while slower shards are still in flight.
        topology_tail = required_workers * max(tier.concurrent_games for tier in tiers)
        smallest_tier = min(tier.concurrent_games for tier in tiers)
        # Reserve only the estimated target plus one complete topology wave.
        # Shards are materialized lazily by the service, so the controller does
        # not construct the old two-target pool whose second half was normally
        # cancelled without ever reaching a worker.
        pool = math.ceil(target / minimum_decisions_per_game) + topology_tail
        if learner_clocked:
            # The overlap window is closed by the learner, not this planning
            # bound. Three lazy primary waves keep fast workers supplied across
            # a normal update; V3 materializes assignments only when leased.
            pool = max(pool, 3 * topology_tail)
        return math.ceil(pool / smallest_tier) * smallest_tier

    def quota_exposure_games(self) -> int:
        """Return the capacity-aware V3 exposure-wave shard size."""
        distributed = self.config.native_distributed
        quorum = distributed.quorum
        if quorum.allow_degraded_after_startup:
            self.server.wait_for_worker_quorum(
                quorum.required_worker_ids,
                minimum_workers=quorum.minimum_degraded_workers,
                timeout_seconds=self.config.collection.collection_timeout_seconds,
            )
        else:
            self.server.wait_for_workers(
                quorum.required_worker_ids,
                timeout_seconds=self.config.collection.collection_timeout_seconds,
            )
        manifests = self.server.coordinator.ready_worker_manifests(
            quorum.required_worker_ids
        )
        if len(manifests) < quorum.minimum_degraded_workers or any(
            not manifest.capacity_tiers for manifest in manifests
        ):
            raise RuntimeError("native distributed READY capacity inventory is empty")
        target = self.config.collection.native_trainable_decision_budget
        if target is None:
            raise RuntimeError("native distributed global target is absent")
        return select_v3_exposure_games(
            tuple(manifest.capacity_tiers for manifest in manifests),
            target_trainable_decisions=target,
            required_artifact_count=(
                self.config.opponent_pool_v2.maximum_active_artifacts
            ),
            topology_worker_count=len(manifests),
        )

    def collect(
        self,
        *,
        model_state: Mapping[str, Tensor],
        model_config: SimpleStatelessModelConfig,
        behavior_identity: StatelessFragmentIdentity,
        assignments: Sequence[StatelessAssignedGame] | StatelessQuotaAssignmentPlan,
        curriculum_members: Sequence[PfspMember],
        opponent_pool_revision: str | None = None,
        historical_artifact_bindings: Sequence[StatelessHistoricalAnchorConfig]
        | None = None,
    ) -> StatelessCollectionResult:
        """Open, serve, and validate one complete global array window."""
        self.begin_collection(
            model_state=model_state,
            model_config=model_config,
            behavior_identity=behavior_identity,
            assignments=assignments,
            curriculum_members=curriculum_members,
            opponent_pool_revision=opponent_pool_revision,
            historical_artifact_bindings=historical_artifact_bindings,
        )
        return self.wait_collection()

    def begin_collection(
        self,
        *,
        model_state: Mapping[str, Tensor],
        model_config: SimpleStatelessModelConfig,
        behavior_identity: StatelessFragmentIdentity,
        assignments: Sequence[StatelessAssignedGame] | StatelessQuotaAssignmentPlan,
        curriculum_members: Sequence[PfspMember],
        opponent_pool_revision: str | None = None,
        historical_artifact_bindings: Sequence[StatelessHistoricalAnchorConfig]
        | None = None,
        learner_clocked_primary_only: bool = False,
        prepared_current_artifact: EncodedBfloat16RolloutArtifact | None = None,
    ) -> NativeCollectionPreparationTiming:
        """Open a window and return while remote workers collect asynchronously."""
        if self._window_open:
            raise RuntimeError("native distributed window is already open")
        planning_started_at = time.perf_counter()
        quota_plan: StatelessQuotaAssignmentPlan | None
        if isinstance(assignments, StatelessQuotaAssignmentPlan):
            quota_plan = assignments
            assignment_pool: (
                tuple[StatelessAssignedGame, ...] | StatelessQuotaAssignmentPlan
            ) = assignments
            concrete_assignments: tuple[StatelessAssignedGame, ...] = ()
            relevant_members = _relevant_quota_pfsp_members(
                assignments,
                curriculum_members,
            )
            wire_assignments: tuple[NativeAssignedGame, ...] = ()
        else:
            quota_plan = None
            concrete_assignments = tuple(assignments)
            if not concrete_assignments:
                raise ValueError("native distributed assignment pool is empty")
            assignment_pool = concrete_assignments
            relevant_members = _relevant_pfsp_members(
                concrete_assignments,
                curriculum_members,
            )
            wire_assignments = tuple(
                NativeAssignedGame.from_assignment(item)
                for item in concrete_assignments
            )
        assignment_projection_seconds = time.perf_counter() - planning_started_at
        relevant_historical_ids = {
            member.member_id
            for member in relevant_members
            if member.source == "historical_anchor"
        }
        binding_source = (
            tuple(self.config.curriculum.anchors)
            if historical_artifact_bindings is None
            else tuple(historical_artifact_bindings)
        )
        relevant_historical_bindings = tuple(
            binding
            for binding in binding_source
            if binding.member_id in relevant_historical_ids
        )
        prepared = self._prepare_artifacts(
            model_state=model_state,
            model_config=model_config,
            behavior_identity=behavior_identity,
            members=relevant_members,
            prepared_current_artifact=prepared_current_artifact,
        )
        target = self.config.collection.native_trainable_decision_budget
        if target is None:
            raise RuntimeError("native distributed decision target is absent")
        now = time.time_ns()
        sequence_id = behavior_identity.behavior_policy_version
        shard_protocol_version = (
            1
            if opponent_pool_revision is None
            else self.config.collection.native_shard_protocol_version
        )
        if (shard_protocol_version == 3) != (quota_plan is not None):
            raise ValueError(
                "native V3 requires an aggregate quota plan and V2 forbids one"
            )
        window = NativeCollectionWindow(
            identity=NativeRolloutWindowIdentity(
                run_id=self.config.run.version,
                window_id=(f"w{sequence_id}-{uuid.uuid4().hex}"),
                sequence_id=sequence_id,
                behavior_policy_version=sequence_id,
                behavior_policy_fingerprint=(
                    behavior_identity.behavior_policy_fingerprint
                ),
                static_contract_fingerprint=(
                    behavior_identity.static_contract_fingerprint
                ),
                resolved_config_fingerprint=(
                    behavior_identity.resolved_config_fingerprint
                ),
            ),
            target_trainable_decisions=target,
            active_artifacts=tuple(item.manifest for item in prepared.artifacts),
            artifact_model_configs=prepared.model_configs,
            active_pfsp_artifact_ids=tuple(
                sorted({member.policy_sha256 for member in relevant_members})
            ),
            pfsp_members=relevant_members,
            required_worker_ids=(),
            topology_epoch=sequence_id,
            opened_at_unix_ns=now,
            shard_protocol_version=shard_protocol_version,
            assignment_plan_revision=(
                None
                if opponent_pool_revision is None
                else (
                    quota_plan.revision
                    if quota_plan is not None
                    else native_assignment_plan_revision(wire_assignments)
                )
            ),
            opponent_pool_revision=opponent_pool_revision,
            historical_artifact_bindings=(
                () if opponent_pool_revision is None else relevant_historical_bindings
            ),
            immediate_whole_game_cutoff_on_drain=(
                learner_clocked_primary_only
                and self.config.native_distributed.scheduling.learner_clocked_immediate_whole_game_cutoff
            ),
        )
        if self._committed_windows == 0:
            quorum = self.config.native_distributed.quorum
            if quorum.allow_degraded_after_startup:
                self.server.wait_for_worker_quorum(
                    quorum.required_worker_ids,
                    minimum_workers=quorum.minimum_degraded_workers,
                    timeout_seconds=(self.config.collection.collection_timeout_seconds),
                )
            else:
                self.server.wait_for_workers(
                    quorum.required_worker_ids,
                    timeout_seconds=(self.config.collection.collection_timeout_seconds),
                )
        open_started_at = time.perf_counter()
        self._begin_with_quorum(
            window,
            artifacts=prepared.artifacts,
            assignment_pool=assignment_pool,
            learner_clocked_primary_only=learner_clocked_primary_only,
        )
        self._window_open = True
        self._active_assignments = concrete_assignments
        self._active_quota_plan = quota_plan
        self._quota_assignments_adopted = False
        artifact_timings = tuple(
            item.preparation_timing for item in prepared.prepared_now
        )
        timing = NativeCollectionPreparationTiming(
            assignment_planning_seconds=(
                assignment_projection_seconds
                + self.server.assignment_planning_seconds()
            ),
            source_fingerprint_seconds=sum(
                item.source_fingerprint_seconds for item in artifact_timings
            ),
            bf16_conversion_seconds=sum(
                item.bf16_conversion_seconds for item in artifact_timings
            ),
            wire_hashing_seconds=sum(
                item.wire_hashing_seconds for item in artifact_timings
            ),
            window_open_seconds=time.perf_counter() - open_started_at,
        )
        self.last_preparation_timing = timing
        return timing

    def prepare_current_artifact(
        self,
        *,
        model_state: Mapping[str, Tensor],
        behavior_identity: StatelessFragmentIdentity,
    ) -> EncodedBfloat16RolloutArtifact:
        """Encode the immutable current policy independently of PFSP planning."""
        return prepare_bfloat16_rollout_artifact(
            model_state,
            artifact_id=_current_artifact_id(behavior_identity),
            kind="current",
            source_policy_version=behavior_identity.behavior_policy_version,
            expected_source_fp32_fingerprint=(
                behavior_identity.behavior_policy_fingerprint
            ),
            model_config_fingerprint=(behavior_identity.model_config_fingerprint),
            exact_registry_fingerprint=(behavior_identity.exact_registry_fingerprint),
            precomputed_source_fp32_fingerprint=(
                behavior_identity.behavior_policy_fingerprint
            ),
        )

    def wait_collection(self) -> StatelessCollectionResult:
        """Wait for and return the currently open global array window."""
        if not self._window_open:
            raise RuntimeError("native distributed window is not open")
        try:
            self.server.wait_until_complete(
                timeout_seconds=self.config.collection.collection_timeout_seconds
            )
            return self.server.collection_result()
        except BaseException as exc:
            self.abort(reason=f"{type(exc).__name__}: {exc}")
            raise

    def request_learner_ready_drain(self) -> bool:
        """Ask an overlapped window to settle without reaching its soft cap."""
        if not self._window_open:
            return False
        return self.server.coordinator.request_learner_ready_drain()

    def unused_assignments(self) -> tuple[StatelessAssignedGame, ...]:
        """Return preplanned controller leases not used by the finished window."""
        if not self._window_open:
            return ()
        return self.server.unused_assignments()

    def adopt_quota_assignments(self) -> tuple[StatelessAssignedGame, ...]:
        """Adopt only V3 reservations that actually reached engine start."""
        if not self._window_open or self._active_quota_plan is None:
            return ()
        if self._quota_assignments_adopted:
            return ()
        self._active_assignments = self._active_quota_plan.adopt_issued(
            self.server.accepted_assignment_ids()
        )
        self._quota_assignments_adopted = True
        return self._active_assignments

    def commit(
        self,
        callback: CommitCallback,
    ) -> NativeCollectionWindowReceipt:
        """Apply all accepted outcomes once and return the final receipt."""
        if not self._window_open:
            raise RuntimeError("native distributed window is not open")
        receipt = self.server.commit(callback)
        self._committed_windows += 1
        return receipt

    def settle_receipt(
        self,
        receipt: NativeCollectionWindowReceipt,
    ) -> None:
        """Persist a committed receipt and wait for every worker to observe it."""
        if not self._window_open:
            raise RuntimeError("native distributed window is not open")
        if receipt.status != "committed":
            raise ValueError("only a committed receipt may be settled")
        if self.server.coordinator.receipt != receipt:
            raise ValueError("receipt does not identify the active committed window")
        _write_receipt(self.output_dir, receipt)
        self.server.wait_for_receipt_delivery(
            timeout_seconds=self.config.collection.collection_timeout_seconds
        )

    def release(self) -> None:
        """Release network frames after optimizer arrays are no longer live."""
        if not self._window_open:
            return
        self.server.release_parts()
        self._active_assignments = ()
        self._active_quota_plan = None
        self._quota_assignments_adopted = False
        self._window_open = False

    def abort(self, *, reason: str) -> NativeCollectionWindowReceipt | None:
        """Abort a half-window; caller remains responsible for assignment cancel."""
        if not self._window_open:
            return None
        receipt = self.server.abort(reason=reason)
        _write_receipt(self.output_dir, receipt)
        self.server.release_parts()
        self._window_open = False
        self._active_assignments = ()
        self._active_quota_plan = None
        self._quota_assignments_adopted = False
        return receipt

    def close(self) -> None:
        """Fail closed on an unsettled window and stop coordinator I/O."""
        if self._window_open:
            self.abort(reason="learner shutdown with an unsettled window")
        self._past_artifact_cache.clear()
        self._past_model_config_cache.clear()
        self.server.close()

    def _begin_with_quorum(
        self,
        window: NativeCollectionWindow,
        *,
        artifacts: Sequence[EncodedBfloat16RolloutArtifact],
        assignment_pool: Sequence[StatelessAssignedGame] | StatelessQuotaAssignmentPlan,
        learner_clocked_primary_only: bool,
    ) -> None:
        deadline = time.monotonic() + (
            self.config.collection.collection_timeout_seconds
        )
        while True:
            try:
                self.server.begin_window(
                    window,
                    artifacts=artifacts,
                    assignment_pool=assignment_pool,
                    learner_clocked_primary_only=learner_clocked_primary_only,
                )
                return
            except NativeCoordinatorQuorumError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(
                    self.config.native_distributed.retry.control_poll_interval_seconds
                )

    def _prepare_artifacts(
        self,
        *,
        model_state: Mapping[str, Tensor],
        model_config: SimpleStatelessModelConfig,
        behavior_identity: StatelessFragmentIdentity,
        members: Sequence[PfspMember],
        prepared_current_artifact: EncodedBfloat16RolloutArtifact | None = None,
    ) -> _PreparedArtifacts:
        active_past_ids = {
            member.policy_sha256
            for member in members
            if member.source in {"past_self", "fixed_stateless_anchor"}
        }
        if set(self._past_artifact_cache) != set(self._past_model_config_cache):
            raise RuntimeError("past-self artifact and model caches diverged")
        current = (
            self.prepare_current_artifact(
                model_state=model_state,
                behavior_identity=behavior_identity,
            )
            if prepared_current_artifact is None
            else prepared_current_artifact
        )
        _require_current_artifact(current, behavior_identity=behavior_identity)
        current_model_config = NativeArtifactModelConfig.from_model_config(
            artifact_id=current.manifest.artifact_id,
            model_config=model_config,
            exact_registry_fingerprint=(behavior_identity.exact_registry_fingerprint),
        )
        past_artifacts: dict[str, EncodedBfloat16RolloutArtifact] = {}
        past_model_configs: dict[str, NativeArtifactModelConfig] = {}
        prepared_now = [] if prepared_current_artifact is not None else [current]
        for member in members:
            if member.source not in {"past_self", "fixed_stateless_anchor"}:
                continue
            artifact = self._past_artifact_cache.get(member.policy_sha256)
            artifact_model_config = self._past_model_config_cache.get(
                member.policy_sha256
            )
            if (artifact is None) != (artifact_model_config is None):
                raise RuntimeError("past-self artifact and model caches diverged")
            if artifact is None:
                if member.source == "past_self":
                    if member.pair is None:
                        raise ValueError("past-self member has no durable policy")
                    loaded = load_stateless_policy_checkpoint(
                        member.policy_path,
                        expected_artifact=member.pair,
                    )
                    source_state = loaded.model_state
                    source_version = loaded.artifact.version
                    source_model_config = loaded.model_config_value
                    source_model_config_fingerprint = (
                        loaded.artifact.model_config_fingerprint
                    )
                    source_registry_fingerprint = (
                        loaded.artifact.exact_registry_fingerprint
                    )
                else:
                    fixed_model, fixed_payload = load_fixed_deck_checkpoint(
                        member.policy_path
                    )
                    fixed_model.float()
                    runtime_model_fingerprint = canonical_model_state_fingerprint(
                        fixed_model.state_dict()
                    )
                    if (
                        runtime_model_fingerprint != member.pilot_artifact_fingerprint
                        or fixed_payload.get("target_deck_digest")
                        != member.exact_deck_digest
                        or fixed_model.config.resolved_registry_sha256
                        != member.exact_registry_fingerprint
                    ):
                        raise ValueError("fixed stateless anchor identity changed")
                    source_state = fixed_model.state_dict()
                    suffix = member.snapshot_id.rsplit("v", maxsplit=1)[-1]
                    source_version = int(suffix) if suffix.isdigit() else 0
                    source_model_config = fixed_model.config
                    source_model_config_fingerprint = str(
                        fixed_payload["model_config_fingerprint"]
                    )
                    source_registry_fingerprint = (
                        fixed_model.config.resolved_registry_sha256
                    )
                artifact = prepare_bfloat16_rollout_artifact(
                    source_state,
                    artifact_id=member.policy_sha256,
                    kind="past_self",
                    source_policy_version=source_version,
                    expected_source_fp32_fingerprint=(
                        member.pilot_artifact_fingerprint
                    ),
                    model_config_fingerprint=source_model_config_fingerprint,
                    exact_registry_fingerprint=source_registry_fingerprint,
                    precomputed_source_fp32_fingerprint=(
                        member.pilot_artifact_fingerprint
                    ),
                )
                artifact_model_config = NativeArtifactModelConfig.from_model_config(
                    artifact_id=member.policy_sha256,
                    model_config=source_model_config,
                    exact_registry_fingerprint=source_registry_fingerprint,
                )
                self._past_artifact_cache[member.policy_sha256] = artifact
                self._past_model_config_cache[member.policy_sha256] = (
                    artifact_model_config
                )
                prepared_now.append(artifact)
            else:
                if artifact_model_config is None:
                    raise RuntimeError("past-self model configuration is absent")
                # Dict insertion order is the bounded host-cache LRU. Refresh a
                # hit so a sparse PFSP rotation does not evict the artifact it
                # is actively reusing.
                self._past_artifact_cache.pop(member.policy_sha256)
                self._past_artifact_cache[member.policy_sha256] = artifact
                self._past_model_config_cache.pop(member.policy_sha256)
                self._past_model_config_cache[member.policy_sha256] = (
                    artifact_model_config
                )
            past_artifacts[member.policy_sha256] = artifact
            past_model_configs[member.policy_sha256] = artifact_model_config
        while len(self._past_artifact_cache) > self._past_artifact_cache_limit:
            evicted_id = next(
                (
                    artifact_id
                    for artifact_id in self._past_artifact_cache
                    if artifact_id not in active_past_ids
                ),
                None,
            )
            if evicted_id is None:
                raise RuntimeError("active past-self artifacts exceed cache capacity")
            self._past_artifact_cache.pop(evicted_id)
            self._past_model_config_cache.pop(evicted_id)
        keys = sorted(past_artifacts)
        return _PreparedArtifacts(
            artifacts=(current, *(past_artifacts[key] for key in keys)),
            model_configs=(
                current_model_config,
                *(past_model_configs[key] for key in keys),
            ),
            prepared_now=tuple(prepared_now),
        )


def _current_artifact_id(identity: StatelessFragmentIdentity) -> str:
    """Return the behavior-bound current-policy wire identity."""
    return (
        f"current-v{identity.behavior_policy_version}-"
        f"{identity.behavior_policy_fingerprint[:16]}"
    )


def _require_current_artifact(
    artifact: EncodedBfloat16RolloutArtifact,
    *,
    behavior_identity: StatelessFragmentIdentity,
) -> None:
    """Reject a prefetched current artifact from another behavior identity."""
    manifest = artifact.manifest
    expected = (
        _current_artifact_id(behavior_identity),
        "current",
        behavior_identity.behavior_policy_version,
        behavior_identity.behavior_policy_fingerprint,
        behavior_identity.model_config_fingerprint,
        behavior_identity.exact_registry_fingerprint,
    )
    actual = (
        manifest.artifact_id,
        manifest.kind,
        manifest.source_policy_version,
        manifest.source_fp32_fingerprint,
        manifest.model_config_fingerprint,
        manifest.exact_registry_fingerprint,
    )
    if actual != expected:
        raise ValueError("prefetched current artifact identity differs")


def _relevant_pfsp_members(
    assignments: Sequence[StatelessAssignedGame],
    members: Sequence[PfspMember],
) -> tuple[PfspMember, ...]:
    member_ids = {
        assignment.curriculum.member_id
        for assignment in assignments
        if assignment.curriculum.member_id
    }
    by_id = {member.member_id: member for member in members}
    missing = member_ids - set(by_id)
    if missing:
        raise ValueError(
            "native distributed assignment references absent PFSP members: "
            + ", ".join(sorted(missing))
        )
    return tuple(member for member in members if member.member_id in member_ids)


def _relevant_quota_pfsp_members(
    plan: StatelessQuotaAssignmentPlan,
    members: Sequence[PfspMember],
) -> tuple[PfspMember, ...]:
    """Resolve only members named by compact PFSP quota rows."""
    member_ids = {row.member_id for row in plan.rows if row.member_id}
    by_id = {member.member_id: member for member in members}
    missing = member_ids - set(by_id)
    if missing:
        raise ValueError(
            "native quota plan references absent PFSP members: "
            + ", ".join(sorted(missing))
        )
    return tuple(member for member in members if member.member_id in member_ids)


def _write_receipt(
    output_dir: Path,
    receipt: NativeCollectionWindowReceipt,
) -> None:
    path = (
        output_dir
        / "control"
        / "native_distributed_receipts"
        / f"{receipt.window_id}.json"
    )
    atomic_write_bytes(
        path,
        json_payload(
            {
                "format": "native_distributed_window_receipt_v1",
                "receipt": receipt.model_dump(mode="json"),
            }
        ),
        overwrite=False,
    )


__all__ = ["NativeDistributedTrainingBackend"]
