"""Production backend for the deployment-aligned integrated planner profile."""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ptcg_rl.agent.planner_select_policy import PlannerSelectPolicy
from ptcg_rl.agent.runtime import PolicyRuntimeAgent
from ptcg_rl.agent.search.planner_fallback import PlannerFallbackReason
from ptcg_rl.agent.search.prompt_actions import describe_prompt_action_space
from ptcg_rl.evaluation.planner_profile_config import (
    IntegratedPlannerProfileConfig,
    PlannerProfilePointConfig,
    PlannerProfileRuntimeConfig,
)
from ptcg_rl.evaluation.planner_profile_corpus_reader import (
    PlannerProfileCorpusRecord,
)
from ptcg_rl.evaluation.planner_profile_package_models import (
    PlannerPackageAssetManifestEntry,
    PlannerProfilePackageAssetsValidation,
)
from ptcg_rl.evaluation.planner_profile_records import PlannerProfileRunRecord
from ptcg_rl.evaluation.planner_profile_runner import (
    PlannerProfileMeasuredDecision,
)
from ptcg_rl.rl.planner_behavior_policy_contract import PlannerPolicyDecision
from ptcg_rl.rl.planner_profile_actor_runtime import (
    ProfileActor,
    ProfileActorWave,
    create_h200_actor_resources,
    create_packaged_actor_resources,
)
from ptcg_rl.rl.planner_profile_archive import ExtractedPlannerPackage
from ptcg_rl.rl.planner_profile_backend_factories import (
    create_concurrent_learner_workload,
)
from ptcg_rl.rl.planner_profile_inference import ProfileInferenceServer
from ptcg_rl.rl.planner_profile_inputs import (
    CollatedProfileRoots,
    collate_profile_roots,
)
from ptcg_rl.rl.planner_profile_learner import ConcurrentProfileLearner
from ptcg_rl.rl.planner_profile_measurement import (
    ProfileDecisionTiming,
    build_profile_decision_record,
)
from ptcg_rl.rl.planner_profile_oracle import ProfileOracleCache
from ptcg_rl.rl.planner_profile_oracle_store import ProfileOracleReferenceStore
from ptcg_rl.rl.planner_profile_packaged import run_packaged_act_time_replay
from ptcg_rl.rl.planner_profile_resources import ProfileResourceSampler
from ptcg_rl.rl.planner_profile_run_record import (
    ProfileRunMeasurements,
    build_profile_run_record,
)
from ptcg_rl.rl.rollout import RolloutPlannerBatch, RolloutPlannerRowContext
from ptcg_rl.runtime.planner_telemetry import PlannerStage


class PlannerProfileBackend:
    """Own exact model, actor-local services, oracle cache, and workload."""

    def __init__(
        self,
        *,
        config: IntegratedPlannerProfileConfig,
        point: PlannerProfilePointConfig,
        runtime: PlannerProfileRuntimeConfig,
        oracle_cache: ProfileOracleCache,
        package_validation: PlannerProfilePackageAssetsValidation,
        oracle_precompute_only: bool,
    ) -> None:
        self._config = config
        self._point = point
        self._runtime = runtime
        self._oracle_precompute_only = oracle_precompute_only
        self._model_identity = config.model_identity_for(point)
        self._resolved = runtime.planner.resolve_for_lease(
            model_fingerprint=self._model_identity.model_fingerprint,
            policy_version=self._model_identity.policy_version,
            proposal_version=self._model_identity.proposal_version,
        )
        self._package_asset: PlannerPackageAssetManifestEntry | None = None
        self._actors: list[ProfileActor] = []
        self._inference_server: ProfileInferenceServer | None = None
        self._packaged_agent: PolicyRuntimeAgent | None = None
        self._packaged_policy: PlannerSelectPolicy | None = None
        self._packaged_workspace: ExtractedPlannerPackage | None = None
        self._model_deck: tuple[int, ...] | None = None
        self._learner: ConcurrentProfileLearner | None = None
        self._oracle_store: ProfileOracleReferenceStore | None = None
        self._resource_sampler = ProfileResourceSampler(
            include_vram=point.environment == "h200_mps"
        )
        self._measurement_started_at: float | None = None
        self._deployment_elapsed_seconds = 0.0
        self._actor_busy_seconds = 0.0
        self._native_seconds = 0.0
        self._valid_planner_decisions = 0
        self._fallback_decisions = 0
        self._failed_decisions = 0
        self._deadline_decisions = 0
        self._closed = False
        try:
            if point.environment == "h200_mps":
                self._initialize_h200()
            else:
                self._package_asset = package_validation.asset_for_runtime(
                    point.runtime_id
                )
                self._initialize_packaged()
            actor = self._actors[0]
            self._oracle_store = ProfileOracleReferenceStore(
                config=config,
                point=point,
                runtime=runtime,
                resolved=self._resolved,
                cache=oracle_cache,
                policy=actor.policy,
                sampler=actor.belief.sampler,
                producer=actor.belief.producer,
                model_deck=self._model_deck,
            )
            self._oracle_store.prepare()
            oracle_executions = self._oracle_store.executions
            if oracle_precompute_only and not oracle_executions:
                raise RuntimeError("oracle precompute backend performed no exact work")
            if not oracle_precompute_only and self._oracle_store.cache_misses:
                raise RuntimeError("measured backend performed oracle warmup work")
            if point.environment == "h200_mps" and not oracle_precompute_only:
                self._learner = create_concurrent_learner_workload(
                    config=config,
                    runtime=runtime,
                )
        except Exception:
            self.close()
            raise

    @property
    def planner_service(self) -> Any:
        """Expose the exact service instance used by actor zero."""
        if not self._actors:
            raise RuntimeError("planner profile backend has no actor runtime")
        return self._actors[0].planner_runtime.service

    @property
    def packaged_agent(self) -> PolicyRuntimeAgent | None:
        return self._packaged_agent

    def execute_batch(
        self,
        records: Sequence[PlannerProfileCorpusRecord],
        *,
        decision_index: int,
        planner_enabled: bool,
    ) -> Sequence[PlannerProfileMeasuredDecision]:
        """Execute one cross-actor root wave through production services."""
        if self._oracle_precompute_only:
            raise RuntimeError("oracle precompute backend cannot measure decisions")
        if planner_enabled != self._point.planner_enabled:
            raise ValueError("profile backend planner branch changed within a point")
        if not records:
            return ()
        self._begin_measurement()
        started = time.perf_counter()
        waves = self._decode_actor_waves(
            tuple(records), planner_enabled=planner_enabled
        )
        decisions = self._plan_actor_waves(waves, planner_enabled=planner_enabled)
        elapsed = max(0.0, time.perf_counter() - started)
        self._deployment_elapsed_seconds += elapsed
        per_row_total_ms = elapsed * 1_000.0
        measured: list[PlannerProfileMeasuredDecision] = []
        offset = 0
        corpus_rows = _corpus_row_count(self._config)
        for wave, wave_decisions in zip(waves, decisions, strict=True):
            occupancy = wave.actor.planner_runtime.service.stats().peak_active_rows
            for row_position, (source, planner_decision) in enumerate(
                zip(wave.records, wave_decisions, strict=True)
            ):
                base_action = tuple(wave.trace.actions[row_position])
                selected_action = (
                    base_action
                    if planner_decision is None
                    else tuple(planner_decision.action)
                )
                absolute_index = decision_index + offset
                repetition = absolute_index // corpus_rows
                if self._oracle_store is None:
                    raise RuntimeError("profile backend has no oracle store")
                reference = self._oracle_store.reference_for(
                    repetition,
                    source.row_id,
                )
                materialized = self._oracle_store.materialize(
                    source,
                    base_action=base_action,
                    base_old_logprob=float(
                        wave.trace.action_logprobs[row_position].item()
                    ),
                )
                if (
                    materialized.support_fingerprint
                    != reference.materialized.support_fingerprint
                ):
                    raise RuntimeError("deployment belief support differs from oracle")
                record = build_profile_decision_record(
                    config=self._config,
                    point=self._point,
                    runtime=self._runtime,
                    resolved=self._resolved,
                    source=source,
                    decision_index=absolute_index,
                    corpus_repetition=repetition,
                    batch_row_position=row_position,
                    base_action=base_action,
                    selected_action=selected_action,
                    planner_decision=planner_decision,
                    materialized=materialized,
                    oracle=reference.cache_value,
                    timing=ProfileDecisionTiming(
                        actor_policy_wait_ms=wave.actor_policy_wait_ms,
                        total_latency_ms=per_row_total_ms,
                    ),
                    native_lane_occupancy=occupancy,
                )
                events = (
                    ()
                    if planner_decision is None
                    else planner_decision.telemetry_events
                )
                self._native_seconds += sum(
                    event.seconds
                    for event in events
                    if event.stage is PlannerStage.NATIVE_ENGINE
                )
                self._valid_planner_decisions += int(record.planner_used)
                self._fallback_decisions += int(record.fallback_reason is not None)
                self._failed_decisions += int(record.failed)
                self._deadline_decisions += int(record.deadline_exceeded)
                measured.append(
                    PlannerProfileMeasuredDecision(record=record, events=events)
                )
                offset += 1
        return tuple(measured)

    def finish_run(self, *, decisions: int) -> PlannerProfileRunRecord:
        """Finish concurrent learner/ActTime work and report bounded resources."""
        if self._oracle_precompute_only:
            raise RuntimeError("oracle precompute backend cannot finish a run")
        if self._measurement_started_at is None:
            raise RuntimeError("profile backend cannot finish before deployment")
        learner_result = None if self._learner is None else self._learner.finish()
        replay_result = None
        archive_sha256: str | None = None
        required_sha256: str | None = None
        if self._point.environment == "packaged_cpu_acttime":
            if self._package_asset is None:
                raise RuntimeError("packaged backend has no validated package asset")
            replay_result = run_packaged_act_time_replay(
                config=self._config,
                runtime=self._runtime,
                planner_enabled=self._point.planner_enabled,
                asset=self._package_asset,
            )
            archive_sha256 = self._package_asset.submission_archive_sha256
            required_sha256 = self._package_asset.required_files_fingerprint
        self._resource_sampler.stop()
        elapsed = max(
            time.perf_counter() - self._measurement_started_at,
            1.0e-9,
        )
        oracle_executions = (
            () if self._oracle_store is None else self._oracle_store.executions
        )
        return build_profile_run_record(
            config=self._config,
            point=self._point,
            runtime=self._runtime,
            model_identity=self._model_identity,
            resolved=self._resolved,
            measured=ProfileRunMeasurements(
                decisions=decisions,
                elapsed_seconds=elapsed,
                actor_busy_seconds=self._actor_busy_seconds,
                native_seconds=self._native_seconds,
                valid_planner_decisions=self._valid_planner_decisions,
                fallback_decisions=self._fallback_decisions,
                failed_decisions=self._failed_decisions,
                deadline_decisions=self._deadline_decisions,
                resource_peak=self._resource_sampler.peak,
                learner=learner_result,
                replay=replay_result,
                oracle_executions=oracle_executions,
                archive_sha256=archive_sha256,
                required_files_fingerprint=required_sha256,
            ),
        )

    def close(self) -> None:
        """Release profile resources exactly once, including error paths."""
        if self._closed:
            return
        self._closed = True
        self._resource_sampler.stop()
        if self._learner is not None:
            self._learner.close()
            self._learner = None
        if self._oracle_store is not None:
            self._oracle_store.close()
            self._oracle_store = None
        if self._packaged_agent is not None:
            self._packaged_agent.close()
            self._packaged_agent = None
        if self._packaged_policy is not None:
            self._packaged_policy.close()
            self._packaged_policy = None
        for actor in self._actors:
            actor.planner_runtime.close()
        self._actors.clear()
        if self._inference_server is not None:
            self._inference_server.close()
            self._inference_server = None
        if self._packaged_workspace is not None:
            self._packaged_workspace.close()
            self._packaged_workspace = None

    def _initialize_h200(self) -> None:
        resources = create_h200_actor_resources(
            config=self._config,
            runtime=self._runtime,
            identity=self._model_identity,
        )
        self._inference_server = resources.inference_server
        self._actors.extend(resources.actors)

    def _initialize_packaged(self) -> None:
        if self._package_asset is None:
            raise RuntimeError("packaged profile has no validated manifest asset")
        resources = create_packaged_actor_resources(
            config=self._config,
            point=self._point,
            runtime=self._runtime,
            identity=self._model_identity,
            resolved=self._resolved,
            asset=self._package_asset,
        )
        self._packaged_agent = resources.agent
        self._packaged_policy = resources.planner_policy
        self._packaged_workspace = resources.workspace
        self._model_deck = resources.deployment_deck
        self._actors.append(resources.actor)

    def _begin_measurement(self) -> None:
        if self._measurement_started_at is not None:
            return
        self._measurement_started_at = time.perf_counter()
        self._resource_sampler.start()
        if self._learner is not None:
            self._learner.begin()

    def _decode_actor_waves(
        self,
        records: tuple[PlannerProfileCorpusRecord, ...],
        *,
        planner_enabled: bool,
    ) -> tuple[ProfileActorWave, ...]:
        maximum = self._runtime.planner.batching.max_root_rows_per_request
        chunks = tuple(
            records[start : start + maximum]
            for start in range(0, len(records), maximum)
        )
        if len(chunks) > len(self._actors):
            raise ValueError("profile root wave exceeds actor count")
        pending: list[
            tuple[
                ProfileActor,
                tuple[PlannerProfileCorpusRecord, ...],
                CollatedProfileRoots,
                Any,
                float,
            ]
        ] = []
        local: list[ProfileActorWave] = []
        for actor, chunk in zip(self._actors, chunks, strict=False):
            collated = collate_profile_roots(
                chunk,
                producer=actor.belief.producer,
                device="cpu",
                model_deck=self._model_deck,
            )
            retain = planner_enabled and any(
                describe_prompt_action_space(
                    record.observation.get("select")
                ).legal_action_count
                > 1
                for record in chunk
            )
            submitted_at = time.perf_counter()
            submit = getattr(actor.policy, "submit_decode_for_request", None)
            if callable(submit):
                handle = submit(
                    collated.states,
                    collated.options,
                    collated.decks,
                    temperature=0.0,
                    retain_planner_context=retain,
                )
                pending.append((actor, chunk, collated, handle, submitted_at))
                continue
            trace = actor.policy.sample_decode_with_trace_for_request(
                collated.states,
                collated.options,
                collated.decks,
                temperature=0.0,
                model_version_lease=None,
                retain_planner_context=retain,
            )
            wait_ms = (time.perf_counter() - submitted_at) * 1_000.0
            local.append(
                ProfileActorWave(
                    actor=actor,
                    records=chunk,
                    collated=collated,
                    trace=trace,
                    actor_policy_wait_ms=wait_ms,
                )
            )
        remote: list[ProfileActorWave] = []
        for actor, chunk, collated, handle, submitted_at in pending:
            trace = actor.policy.receive_decode_with_trace(handle)
            wait_ms = (time.perf_counter() - submitted_at) * 1_000.0
            remote.append(
                ProfileActorWave(
                    actor=actor,
                    records=chunk,
                    collated=collated,
                    trace=trace,
                    actor_policy_wait_ms=wait_ms,
                )
            )
        waves_by_actor = {wave.actor.actor_id: wave for wave in (*local, *remote)}
        return tuple(
            waves_by_actor[actor.actor_id] for actor in self._actors[: len(chunks)]
        )

    def _plan_actor_waves(
        self,
        waves: tuple[ProfileActorWave, ...],
        *,
        planner_enabled: bool,
    ) -> tuple[tuple[PlannerPolicyDecision | None, ...], ...]:
        if not planner_enabled:
            self._actor_busy_seconds += sum(
                wave.actor_policy_wait_ms / 1_000.0 for wave in waves
            )
            return tuple(tuple(None for _ in wave.records) for wave in waves)

        def plan(
            wave: ProfileActorWave,
        ) -> tuple[PlannerPolicyDecision | None, ...]:
            started = time.perf_counter()
            trace = wave.trace
            root_fallback = (
                None
                if not trace.planner_fallback_reason
                else PlannerFallbackReason.MODEL_LEASE_CAPACITY
            )
            result = tuple(
                wave.actor.planner_runtime.service.plan_batch(
                    RolloutPlannerBatch(
                        policy=wave.actor.policy,
                        rows=tuple(
                            RolloutPlannerRowContext(
                                game_id=f"profile-{record.row_id}",
                                seat=_record_seat(record),
                                policy_role="candidate",
                                should_record=True,
                                observation=wave.collated.prepared[index].observation,
                                context_features=(
                                    wave.collated.prepared[index].context_features
                                ),
                                context_snapshot=record.context_snapshot,
                                deck_pair=(record.own_deck, record.own_deck),
                                model_deck=self._model_deck,
                            )
                            for index, record in enumerate(wave.records)
                        ),
                        states=wave.collated.states,
                        options=wave.collated.options,
                        decks=wave.collated.decks,
                        base_actions=trace.actions,
                        base_logprobs=tuple(
                            float(value) for value in trace.action_logprobs
                        ),
                        base_values=tuple(float(value) for value in trace.values),
                        policy_version=self._resolved.runtime_identity.policy_version,
                        model_fingerprint=(
                            self._resolved.runtime_identity.model_fingerprint
                        ),
                        proposal_version=(
                            self._resolved.runtime_identity.proposal_version
                        ),
                        planner_context_handles=trace.planner_context_handles,
                        root_fallback_reason=root_fallback,
                    )
                )
            )
            self._actor_busy_seconds += (
                time.perf_counter() - started + wave.actor_policy_wait_ms / 1_000.0
            )
            return result

        with ThreadPoolExecutor(
            max_workers=len(waves),
            thread_name_prefix="planner-profile-actor",
        ) as executor:
            futures = tuple(executor.submit(plan, wave) for wave in waves)
            results = tuple(future.result() for future in futures)
        for wave, result in zip(waves, results, strict=True):
            if len(result) != len(wave.records) or any(item is None for item in result):
                raise RuntimeError("planner service returned a misaligned actor wave")
        return results


def create_planner_profile_backend(
    *,
    config: IntegratedPlannerProfileConfig,
    point: PlannerProfilePointConfig,
    runtime: PlannerProfileRuntimeConfig,
    oracle_cache: ProfileOracleCache,
    package_validation: PlannerProfilePackageAssetsValidation,
    oracle_precompute_only: bool,
) -> PlannerProfileBackend:
    """Return the production backend named by the immutable profile config."""
    return PlannerProfileBackend(
        config=config,
        point=point,
        runtime=runtime,
        oracle_cache=oracle_cache,
        package_validation=package_validation,
        oracle_precompute_only=oracle_precompute_only,
    )


def _record_seat(record: PlannerProfileCorpusRecord) -> int:
    seat = record.context_snapshot.player_index
    if seat not in (0, 1):
        raise ValueError("planner profile root has no exact player index")
    return int(seat)


def _corpus_row_count(config: IntegratedPlannerProfileConfig) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(config.decision_corpus_path).metadata.num_rows)


__all__ = [
    "PlannerProfileBackend",
    "create_planner_profile_backend",
]
