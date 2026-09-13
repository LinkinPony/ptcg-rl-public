"""Multiprocess native rollout workers with one parent-owned CUDA service."""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ptcg_rl.belief.public_catalog import PublicDeckCatalog
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.decks.identity import CanonicalDeck
from ptcg_rl.engine.native_prospective_facts import (
    NativeProspectiveEngineFactProducer,
)
from ptcg_rl.engine.prospective_facts import (
    ProspectiveEngineFactConfig,
    ProspectiveEngineFactProducer,
)
from ptcg_rl.rl.native_historical_inference import NativeHistoricalPolicyPool
from ptcg_rl.rl.native_policy_inference import NativePolicyInferenceExecutor
from ptcg_rl.rl.native_process_client import (
    RemoteNativeHistoricalPolicyPool,
    RemoteNativePolicyExecutor,
)
from ptcg_rl.rl.native_process_inference import NativeProcessInferenceBroker
from ptcg_rl.rl.native_process_results import (
    load_native_worker_parts,
    merge_native_process_reports,
    order_native_worker_outcomes,
    persist_native_worker_parts,
)
from ptcg_rl.rl.native_process_sequence import (
    RemoteGeneralistSequenceActorPolicy,
)
from ptcg_rl.rl.native_process_shared_batch import NativeSharedBatchWriter
from ptcg_rl.rl.native_route_arena import NativeArenaResourcePool
from ptcg_rl.rl.native_scripted_catalog import MIXED75_71EB_DECK_DIGEST
from ptcg_rl.rl.native_scripted_mixed75 import NativeMixed75Policy
from ptcg_rl.rl.native_scripted_policy import NativePublicScriptedPolicy
from ptcg_rl.rl.native_stateless_collection import NativeStatelessCollector
from ptcg_rl.rl.scripted_manifest import (
    ResolvedScriptedOpponent,
    ScriptedOpponentArtifact,
    builtin_scripted_implementations,
)
from ptcg_rl.rl.sequence_actor import (
    GeneralistSequenceActorPolicy,
    SequenceRolloutPrecision,
)
from ptcg_rl.rl.stateless_collection import (
    StatelessAssignedGame,
    StatelessCollectionResult,
)
from ptcg_rl.rl.stateless_curriculum import PfspMember
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity
from ptcg_rl.rl.stateless_fragment_io import CompactFragmentPart
from ptcg_rl.rl.stateless_opponents import PastSelfPolicyPool
from ptcg_rl.rl.stateless_parallel import StatelessParallelWorkerResult
from ptcg_rl.rl.stateless_training_config import StatelessScriptedOpponentConfig


@dataclass(frozen=True)
class NativeProcessWorkerSpec:
    """Immutable CPU resources loaded once by a rollout worker."""

    worker_index: int
    catalog: PublicDeckCatalog
    active_decks: Mapping[str, CanonicalDeck]
    opponent_decks: Mapping[str, CanonicalDeck]
    scripted: tuple[StatelessScriptedOpponentConfig, ...]
    static_feature_path: Path
    maximum_engine_steps: int
    fragments_per_part: int
    mirror_bilateral_trajectories: bool
    options_per_lane: int
    library_path: Path | None
    inference_timeout_seconds: float
    sequence_rollout_precision: SequenceRolloutPrecision = "fp32"
    # Every stable process owns this many independently advancing engine
    # arenas. The parent still remains the sole CUDA inference owner.
    engine_shards: int = 1
    policy_cohort_wait_ms: float = 0.0
    engine_fact_config: ProspectiveEngineFactConfig | None = None
    engine_fact_workers: int | None = None
    maximum_arena_capacity: int | None = None
    # Keep every stable feeder and all of its descendant native/fact workers
    # on a disjoint CPU partition. Without this boundary each spawned feeder
    # observes the full launcher affinity and independently creates a full-size
    # C++ worker pool, multiplying runnable threads on the same cores.
    cpu_affinity: tuple[int, ...] | None = None
    scripted_artifacts: Mapping[str, ScriptedOpponentArtifact] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class NativeProcessCollectCommand:
    """One immutable behavior-bound assignment shard."""

    shard_index: int
    identity: StatelessFragmentIdentity
    assignments: tuple[StatelessAssignedGame, ...]
    members: tuple[PfspMember, ...]
    past_identities: Mapping[str, StatelessFragmentIdentity]
    seed: int
    arena_capacity: int
    trainable_decision_budget: int | None
    frozen_batch_min_rows: int
    frozen_batch_max_wait_waves: int
    shard_dir: Path
    memory_only_parts: bool = False
    stream_memory_parts: bool = False


@dataclass(frozen=True)
class _NativeProcessCompactPart:
    """One process-local part published before its shard is terminal."""

    shard_index: int
    worker_index: int
    part_sequence: int
    part: CompactFragmentPart


class _RemoteNativeCollector(NativeStatelessCollector):
    """Native collector whose learned routes are parent-process proxies."""

    def __init__(
        self,
        *,
        remote_past: Mapping[
            str,
            RemoteNativePolicyExecutor | RemoteGeneralistSequenceActorPolicy,
        ],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._remote_past = dict(remote_past)
        # CUDA stream overlap belongs to the central owner. A worker performs
        # one synchronous shared-memory request per bulk route wave.
        self.policy_bank = cast(Any, None)

    def _past_executor(
        self,
        artifact_sha256: str,
    ) -> NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy:
        try:
            return cast(
                NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
                self._remote_past[artifact_sha256],
            )
        except KeyError as error:
            raise KeyError("remote native past-self route is absent") from error


class NativeProcessCollector:
    """Own persistent native CPU processes and one CUDA batching boundary."""

    def __init__(
        self,
        *,
        context: Any,
        worker_specs: Sequence[NativeProcessWorkerSpec],
        request_queue: Any,
        scripted_request_queue: Any | None = None,
        response_queues: Mapping[int, Any],
        past_self_pool: PastSelfPolicyPool,
        historical_pool: NativeHistoricalPolicyPool,
        max_batch_rows: int,
        batch_wait_seconds: float,
        result_timeout_seconds: float,
        scripted_response_queues: Mapping[int, Any] | None = None,
        scripted_sampling_seed: int = 0,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        if len(worker_specs) <= 1:
            raise ValueError("native process collection requires multiple workers")
        if result_timeout_seconds <= 0.0:
            raise ValueError("native process result timeout must be positive")
        self.worker_specs = _bind_worker_cpu_affinities(worker_specs)
        self.request_queue = request_queue
        self.scripted_request_queue = scripted_request_queue
        self.response_queues = dict(response_queues)
        self.past_self_pool = past_self_pool
        self.historical_pool = historical_pool
        self.max_batch_rows = int(max_batch_rows)
        self.batch_wait_seconds = float(batch_wait_seconds)
        self.result_timeout_seconds = float(result_timeout_seconds)
        self.scripted_response_queues = dict(scripted_response_queues or {})
        self.scripted_sampling_seed = int(scripted_sampling_seed)
        self.cuda_stream = cuda_stream
        self.command_queues = {
            spec.worker_index: context.Queue(maxsize=1) for spec in self.worker_specs
        }
        self.result_queue = context.Queue()
        self.processes = {
            spec.worker_index: context.Process(
                target=_native_process_worker_main,
                args=(
                    spec,
                    self.command_queues[spec.worker_index],
                    self.result_queue,
                    self.request_queue,
                    self.response_queues[spec.worker_index],
                ),
                name=f"native-rollout-worker-{spec.worker_index}",
            )
            for spec in self.worker_specs
        }
        self._closed = False
        for process in self.processes.values():
            process.start()

    def collect(
        self,
        *,
        actor: NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
        identity: StatelessFragmentIdentity,
        assignments: Sequence[StatelessAssignedGame],
        members: Sequence[PfspMember],
        temporary_root: Path,
        seed: int,
        arena_capacity: int,
        trainable_decision_budget: int | None,
        frozen_batch_min_rows: int,
        frozen_batch_max_wait_waves: int,
        broker_ready_event: threading.Event | None = None,
        broker_hold_event: threading.Event | None = None,
        memory_only_parts: bool = False,
        past_executors: Mapping[
            str,
            NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
        ]
        | None = None,
        compact_part_sink: Callable[[CompactFragmentPart], None] | None = None,
    ) -> StatelessCollectionResult:
        """Collect one window through static worker shards and merged inference."""
        if not assignments:
            raise ValueError("native process collection requires assignments")
        worker_count = min(len(self.worker_specs), len(assignments), arena_capacity)
        if worker_count != len(self.worker_specs):
            raise ValueError(
                "native process window cannot leave configured workers idle"
            )
        assignment_shards = _partition_assignments(
            assignments,
            members=members,
            workers=worker_count,
        )
        capacities = _allocate_worker_capacities(
            arena_capacity,
            tuple(len(shard) for shard in assignment_shards),
        )
        budgets = _split_optional_weighted(
            trainable_decision_budget,
            capacities,
        )
        past: dict[
            str,
            NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
        ]
        if past_executors is None:
            stateless_past, past_identities = _central_past_executors(
                self.past_self_pool,
                members,
            )
            past = {}
            past.update(stateless_past)
        else:
            past = dict(past_executors)
            past_identities = {
                artifact: executor.identity for artifact, executor in past.items()
            }
        broker = NativeProcessInferenceBroker(
            current=actor,
            past=past,
            historical=self.historical_pool,
            request_queue=self.request_queue,
            scripted_request_queue=self.scripted_request_queue,
            response_queues=self.response_queues,
            scripted_response_queues=self.scripted_response_queues,
            max_batch_rows=self.max_batch_rows,
            batch_wait_seconds=self.batch_wait_seconds,
            scripted_sampling_seed=self.scripted_sampling_seed,
            cuda_stream=self.cuda_stream,
        )
        broker.start()
        if broker_ready_event is not None:
            broker_ready_event.set()
        started_at = time.perf_counter()
        collection_elapsed: float | None = None
        results: dict[int, StatelessParallelWorkerResult] = {}
        streamed_parts: dict[int, list[CompactFragmentPart]] = {
            index: [] for index in range(worker_count)
        }
        try:
            for shard_index, spec in enumerate(self.worker_specs):
                shard_dir = temporary_root / (
                    f"shard-{shard_index:04d}-worker-{spec.worker_index:02d}"
                )
                self.command_queues[spec.worker_index].put(
                    NativeProcessCollectCommand(
                        shard_index=shard_index,
                        identity=identity,
                        assignments=assignment_shards[shard_index],
                        members=tuple(members),
                        past_identities=past_identities,
                        seed=_worker_seed(seed, spec.worker_index),
                        arena_capacity=min(
                            capacities[shard_index],
                            len(assignment_shards[shard_index]),
                        ),
                        trainable_decision_budget=budgets[shard_index],
                        frozen_batch_min_rows=frozen_batch_min_rows,
                        frozen_batch_max_wait_waves=frozen_batch_max_wait_waves,
                        shard_dir=shard_dir,
                        memory_only_parts=memory_only_parts,
                        stream_memory_parts=(
                            memory_only_parts and compact_part_sink is not None
                        ),
                    )
                )
            deadline = time.monotonic() + self.result_timeout_seconds
            while len(results) < worker_count:
                self._raise_failed_worker()
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("native process collection timed out")
                try:
                    result = self.result_queue.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    if broker.fatal_error is not None:
                        raise RuntimeError(
                            "native process inference broker failed"
                        ) from broker.fatal_error
                    continue
                if isinstance(result, _NativeProcessCompactPart):
                    shard_parts = streamed_parts[result.shard_index]
                    if result.part_sequence != len(shard_parts):
                        raise RuntimeError(
                            "native rollout worker part sequence is not contiguous"
                        )
                    shard_parts.append(result.part)
                    if compact_part_sink is not None:
                        compact_part_sink(result.part)
                    continue
                if not isinstance(result, StatelessParallelWorkerResult):
                    raise TypeError("native rollout worker returned an invalid result")
                if result.shard_index in results:
                    raise RuntimeError("native rollout worker duplicated a shard")
                if result.error is not None:
                    raise RuntimeError(
                        f"native rollout worker {result.actor_index} failed:\n"
                        f"{result.error}"
                    )
                results[result.shard_index] = result
            collection_elapsed = max(
                time.perf_counter() - started_at,
                1.0e-9,
            )
        finally:
            if broker_hold_event is not None:
                while not broker_hold_event.wait(timeout=1.0):
                    if broker.fatal_error is not None:
                        break
            broker.close()
        if broker.fatal_error is not None:
            raise RuntimeError("native process inference broker failed") from (
                broker.fatal_error
            )
        ordered = tuple(results[index] for index in range(worker_count))
        if compact_part_sink is None:
            parts = load_native_worker_parts(ordered, expected_identity=identity)
        else:
            parts = tuple(
                part
                for shard_index in range(worker_count)
                for part in streamed_parts[shard_index]
            )
        paths = tuple(path for result in ordered for path in result.part_paths)
        if collection_elapsed is None:
            raise RuntimeError("native process collection elapsed time is absent")
        outcomes = tuple(outcome for result in ordered for outcome in result.outcomes)
        ordered_outcomes = order_native_worker_outcomes(assignments, outcomes)
        started_ids = {outcome.curriculum_assignment_id for outcome in ordered_outcomes}
        started_assignments = tuple(
            assignment
            for assignment in assignments
            if assignment.curriculum.assignment_id in started_ids
        )
        return StatelessCollectionResult(
            fragments=(),
            compact_parts=parts,
            compact_part_paths=paths,
            assignments=started_assignments,
            outcomes=ordered_outcomes,
            report=merge_native_process_reports(
                ordered,
                elapsed_seconds=collection_elapsed,
                decision_budget=trainable_decision_budget,
                broker=broker,
            ),
        )

    def close(self) -> None:
        """Stop and reap every owned worker process."""
        if self._closed:
            return
        self._closed = True
        for command_queue in self.command_queues.values():
            with suppress(queue.Full):
                command_queue.put_nowait(None)
        deadline = time.monotonic() + 30.0
        for process in self.processes.values():
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self.processes.values():
            if process.is_alive():
                process.terminate()
        for process in self.processes.values():
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
        for command_queue in self.command_queues.values():
            command_queue.close()
        self.result_queue.close()
        self.request_queue.close()
        for response_queue in self.response_queues.values():
            response_queue.close()

    def _raise_failed_worker(self) -> None:
        failed = {
            index: process.exitcode
            for index, process in self.processes.items()
            if process.exitcode not in (None, 0)
        }
        if failed:
            raise RuntimeError(f"native rollout worker exited: {failed}")


def _native_process_worker_main(
    spec: NativeProcessWorkerSpec,
    command_queue: Any,
    result_queue: Any,
    request_queue: Any,
    response_queue: Any,
) -> None:
    _apply_worker_cpu_affinity(spec.cpu_affinity)
    # One feeder registers dozens of persistent shared tensor slabs with the
    # parent. The default file-descriptor strategy exhausts the parent's
    # per-process FD limit at server-scale worker counts and can strand queued
    # SCM_RIGHTS handles when a resource sharer closes. Filename-backed shared
    # storage keeps the same zero-copy tensors without one retained FD each.
    cast(Any, torch.multiprocessing).set_sharing_strategy("file_system")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    static_features = np.load(
        spec.static_feature_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    scripted_policies = _build_worker_scripted_policies(
        spec,
        static_features=static_features,
    )
    scripted_bindings = {
        item.opponent_id: (item.artifact_fingerprint, item.exact_deck_digest)
        for item in spec.scripted
    }
    engine_fact_producer = (
        NativeProspectiveEngineFactProducer(
            ProspectiveEngineFactProducer(
                sampler=BeliefSampler(config=spec.engine_fact_config.sampler),
                config=spec.engine_fact_config,
            ),
            maximum_workers=spec.engine_fact_workers,
            pin_memory=False,
        )
        if spec.engine_fact_config is not None
        else None
    )
    arena_resource_pool: NativeArenaResourcePool | None = None
    try:
        while True:
            command = command_queue.get()
            if command is None:
                return
            if not isinstance(command, NativeProcessCollectCommand):
                result_queue.put(
                    StatelessParallelWorkerResult(
                        actor_index=spec.worker_index,
                        shard_index=-1,
                        error="invalid native process collection command",
                    )
                )
                continue
            try:
                required_lane_capacity = (
                    command.arena_capacity + spec.engine_shards - 1
                ) // spec.engine_shards
                if (
                    arena_resource_pool is not None
                    and required_lane_capacity > arena_resource_pool.lane_capacity
                ):
                    arena_resource_pool.close()
                    arena_resource_pool = None
                if arena_resource_pool is None:
                    maximum_capacity = max(
                        command.arena_capacity,
                        spec.maximum_arena_capacity or 0,
                    )
                    lane_capacity = (
                        maximum_capacity + spec.engine_shards - 1
                    ) // spec.engine_shards
                    arena_resource_pool = NativeArenaResourcePool(
                        resource_count=spec.engine_shards,
                        lane_capacity=lane_capacity,
                        options_per_lane=spec.options_per_lane,
                        library_path=spec.library_path,
                        catalog=spec.catalog,
                        input_contract_fingerprint=(
                            command.identity.input_contract_fingerprint
                        ),
                    )
                _run_native_process_command(
                    spec,
                    command,
                    result_queue=result_queue,
                    request_queue=request_queue,
                    response_queue=response_queue,
                    scripted_policies=scripted_policies,
                    scripted_bindings=scripted_bindings,
                    engine_fact_producer=engine_fact_producer,
                    arena_resource_pool=arena_resource_pool,
                )
            except BaseException:
                if arena_resource_pool is not None:
                    arena_resource_pool.close()
                    arena_resource_pool = None
                result_queue.put(
                    StatelessParallelWorkerResult(
                        actor_index=spec.worker_index,
                        shard_index=command.shard_index,
                        error=traceback.format_exc(),
                    )
                )
    finally:
        if arena_resource_pool is not None:
            arena_resource_pool.close()
        if engine_fact_producer is not None:
            engine_fact_producer.close()


def _run_native_process_command(
    spec: NativeProcessWorkerSpec,
    command: NativeProcessCollectCommand,
    *,
    result_queue: Any,
    request_queue: Any,
    response_queue: Any,
    scripted_policies: Mapping[str, NativePublicScriptedPolicy | NativeMixed75Policy],
    scripted_bindings: Mapping[str, tuple[str, str]],
    engine_fact_producer: NativeProspectiveEngineFactProducer | None,
    arena_resource_pool: NativeArenaResourcePool,
) -> None:
    """Collect one command while retaining process-owned static resources."""
    shared_batch_writer = NativeSharedBatchWriter()
    historical = RemoteNativeHistoricalPolicyPool(
        worker_index=spec.worker_index,
        request_queue=request_queue,
        response_queue=response_queue,
        timeout_seconds=spec.inference_timeout_seconds,
        member_artifacts={
            member.member_id: member.policy_sha256
            for member in command.members
            if member.source == "historical_anchor"
        },
        shared_batch_writer=shared_batch_writer,
    )
    temporal_slots = sum(
        1 + int(item.curriculum.lane == "mirror") for item in command.assignments
    )
    if command.identity.schema_version == 2:
        current: RemoteNativePolicyExecutor | RemoteGeneralistSequenceActorPolicy = (
            RemoteGeneralistSequenceActorPolicy(
                worker_index=spec.worker_index,
                identity=command.identity,
                route_kind="current",
                artifact_sha256=command.identity.behavior_policy_fingerprint,
                request_queue=request_queue,
                response_queue=response_queue,
                timeout_seconds=spec.inference_timeout_seconds,
                shared_batch_writer=shared_batch_writer,
                rollout_precision=spec.sequence_rollout_precision,
                temporal_cache_slots=temporal_slots,
            )
        )
    else:
        current = RemoteNativePolicyExecutor(
            worker_index=spec.worker_index,
            identity=command.identity,
            route_kind="current",
            artifact_sha256=command.identity.behavior_policy_fingerprint,
            request_queue=request_queue,
            response_queue=response_queue,
            timeout_seconds=spec.inference_timeout_seconds,
            shared_batch_writer=shared_batch_writer,
        )
    remote_past = {
        artifact_sha256: (
            RemoteGeneralistSequenceActorPolicy(
                worker_index=spec.worker_index,
                identity=identity,
                route_kind="past_self",
                artifact_sha256=artifact_sha256,
                request_queue=request_queue,
                response_queue=response_queue,
                timeout_seconds=spec.inference_timeout_seconds,
                shared_batch_writer=shared_batch_writer,
                rollout_precision=spec.sequence_rollout_precision,
                temporal_cache_slots=temporal_slots,
            )
            if identity.schema_version == 2
            else RemoteNativePolicyExecutor(
                worker_index=spec.worker_index,
                identity=identity,
                route_kind="past_self",
                artifact_sha256=artifact_sha256,
                request_queue=request_queue,
                response_queue=response_queue,
                timeout_seconds=spec.inference_timeout_seconds,
                shared_batch_writer=shared_batch_writer,
            )
        )
        for artifact_sha256, identity in command.past_identities.items()
    }
    part_sequence = 0

    def publish_part(part: CompactFragmentPart) -> None:
        nonlocal part_sequence
        result_queue.put(
            _NativeProcessCompactPart(
                shard_index=command.shard_index,
                worker_index=spec.worker_index,
                part_sequence=part_sequence,
                part=part,
            )
        )
        part_sequence += 1

    collector = _RemoteNativeCollector(
        remote_past=remote_past,
        actor=cast(Any, current),
        identity=command.identity,
        catalog=spec.catalog,
        active_decks=spec.active_decks,
        opponent_decks=spec.opponent_decks,
        members=command.members,
        past_self_pool=cast(PastSelfPolicyPool, None),
        historical_pool=cast(NativeHistoricalPolicyPool, historical),
        scripted_policies=scripted_policies,
        scripted_bindings=scripted_bindings,
        maximum_engine_steps=spec.maximum_engine_steps,
        seed=command.seed,
        fragments_per_part=spec.fragments_per_part,
        mirror_bilateral_trajectories=spec.mirror_bilateral_trajectories,
        arena_capacity=command.arena_capacity,
        engine_shards=spec.engine_shards,
        policy_cohort_wait_ms=spec.policy_cohort_wait_ms,
        trainable_decision_budget=command.trainable_decision_budget,
        frozen_batch_min_rows=command.frozen_batch_min_rows,
        frozen_batch_max_wait_waves=command.frozen_batch_max_wait_waves,
        sequence_rollout_precision=spec.sequence_rollout_precision,
        options_per_lane=spec.options_per_lane,
        library_path=spec.library_path,
        engine_fact_producer=engine_fact_producer,
        owns_engine_fact_producer=False,
        arena_resource_pool=arena_resource_pool,
        compact_part_sink=(publish_part if command.stream_memory_parts else None),
    )
    try:
        collected = collector.collect_assigned(command.assignments)
    finally:
        collector.close()
    if command.memory_only_parts:
        paths: tuple[Path, ...] = ()
        compact_parts = () if command.stream_memory_parts else collected.compact_parts
    else:
        paths = persist_native_worker_parts(
            collected.compact_parts,
            shard_dir=command.shard_dir,
        )
        compact_parts = ()
    result_queue.put(
        StatelessParallelWorkerResult(
            actor_index=spec.worker_index,
            shard_index=command.shard_index,
            report=collected.report,
            outcomes=collected.outcomes,
            part_paths=paths,
            compact_parts=compact_parts,
        )
    )


def _build_worker_scripted_policies(
    spec: NativeProcessWorkerSpec,
    *,
    static_features: np.ndarray,
) -> dict[str, NativeMixed75Policy | NativePublicScriptedPolicy]:
    """Rebuild verified public factories inside one spawned worker."""
    artifacts = {
        item.opponent_id: spec.scripted_artifacts[item.opponent_id]
        for item in spec.scripted
    }
    implementations = {
        implementation.script_name: implementation
        for implementation in builtin_scripted_implementations(
            tuple(artifact.script_name for artifact in artifacts.values())
        )
    }
    policies: dict[str, NativeMixed75Policy | NativePublicScriptedPolicy] = {}
    for item in spec.scripted:
        artifact = artifacts[item.opponent_id]
        if (
            artifact.script_name == "mixed75"
            and item.exact_deck_digest == MIXED75_71EB_DECK_DIGEST
        ):
            policies[item.opponent_id] = NativeMixed75Policy(
                scripted_deck=(spec.opponent_decks[item.exact_deck_digest].card_ids),
                static_features=static_features,
            )
            continue
        try:
            implementation = implementations[artifact.script_name]
        except KeyError as error:
            raise RuntimeError("scripted worker implementation is absent") from error
        if (
            implementation.implementation_files != artifact.implementation_files
            or implementation.vector_safe != artifact.vector_safe
            or implementation.requires_search != artifact.requires_search
        ):
            raise ValueError("scripted worker implementation identity changed")
        policies[item.opponent_id] = NativePublicScriptedPolicy(
            ResolvedScriptedOpponent(
                artifact=artifact,
                factory=implementation.factory,
            )
        )
    return policies


def _central_past_executors(
    pool: PastSelfPolicyPool,
    members: Sequence[PfspMember],
) -> tuple[
    dict[
        str,
        NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
    ],
    dict[str, StatelessFragmentIdentity],
]:
    executors: dict[
        str,
        NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
    ] = {}
    identities: dict[str, StatelessFragmentIdentity] = {}
    for member in members:
        if member.source not in {"past_self", "fixed_stateless_anchor"}:
            continue
        actor = pool.actor(member.member_id)
        identity = actor.identity
        existing = identities.get(member.policy_sha256)
        if existing is not None and existing != identity:
            raise ValueError("one past artifact resolves to conflicting identities")
        identities[member.policy_sha256] = identity
        if member.policy_sha256 not in executors:
            executors[member.policy_sha256] = (
                actor
                if isinstance(actor, GeneralistSequenceActorPolicy)
                else NativePolicyInferenceExecutor(
                    actor.model,
                    identity=identity,
                    device=actor.device,
                    verify_model_state=False,
                )
            )
    return executors, identities


def _partition_assignments(
    assignments: Sequence[StatelessAssignedGame],
    *,
    members: Sequence[PfspMember],
    workers: int,
) -> tuple[tuple[StatelessAssignedGame, ...], ...]:
    """Balance games while keeping each frozen artifact on one worker."""
    if workers <= 0:
        raise ValueError("native assignment partition requires workers")
    artifact_by_member = {member.member_id: member.policy_sha256 for member in members}
    frozen: defaultdict[str, list[StatelessAssignedGame]] = defaultdict(list)
    current: list[StatelessAssignedGame] = []
    for assignment in assignments:
        curriculum = assignment.curriculum
        if curriculum.lane != "pfsp":
            current.append(assignment)
            continue
        try:
            artifact = artifact_by_member[curriculum.member_id]
        except KeyError as error:
            raise ValueError("PFSP assignment member is absent") from error
        frozen[artifact].append(assignment)

    frozen_groups = tuple(
        group
        for _artifact, group in sorted(
            frozen.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
    )
    shards: list[list[StatelessAssignedGame]] = [[] for _index in range(workers)]
    if frozen_groups and len(frozen_groups) < workers and current:
        frozen_replicas = max(1, (workers - 1) // len(frozen_groups))
        frozen_worker_count = frozen_replicas * len(frozen_groups)
        next_worker = 0
        for group in frozen_groups:
            route_workers = tuple(range(next_worker, next_worker + frozen_replicas))
            next_worker += frozen_replicas
            for assignment in group:
                destination = min(
                    route_workers,
                    key=lambda index: (len(shards[index]), index),
                )
                shards[destination].append(assignment)
        current_workers = tuple(range(frozen_worker_count, workers))
        for assignment in current:
            destination = min(
                current_workers,
                key=lambda index: (len(shards[index]), index),
            )
            shards[destination].append(assignment)
    else:
        for group in frozen_groups:
            destination = min(
                range(workers),
                key=lambda index: (len(shards[index]), index),
            )
            shards[destination].extend(group)
        for assignment in current:
            destination = min(
                range(workers),
                key=lambda index: (len(shards[index]), index),
            )
            shards[destination].append(assignment)
    if any(not shard for shard in shards):
        # Small smoke cohorts may not have enough work to dedicate route roles.
        # Retain the original balanced placement rather than idling a process.
        shards = [[] for _index in range(workers)]
        for group in frozen_groups:
            destination = min(
                range(workers),
                key=lambda index: (len(shards[index]), index),
            )
            shards[destination].extend(group)
        for assignment in current:
            destination = min(
                range(workers),
                key=lambda index: (len(shards[index]), index),
            )
            shards[destination].append(assignment)
    if any(not shard for shard in shards):
        raise ValueError("native assignment affinity left a worker idle")
    for shard in shards:
        shard.sort(key=lambda assignment: assignment.curriculum.cursor)
    return tuple(tuple(shard) for shard in shards)


def _allocate_worker_capacities(
    total: int,
    assignment_counts: Sequence[int],
) -> tuple[int, ...]:
    """Allocate live slots in proportion to route-dedicated worker demand."""
    counts = tuple(int(value) for value in assignment_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("native worker assignment counts must be positive")
    if total < len(counts) or total > sum(counts):
        raise ValueError("native arena capacity cannot cover worker assignments")
    if total == sum(counts):
        return counts
    exact = tuple(total * value / sum(counts) for value in counts)
    capacities = [
        max(1, min(counts[index], int(value))) for index, value in enumerate(exact)
    ]
    add_order = sorted(
        range(len(counts)),
        key=lambda index: (-(exact[index] - int(exact[index])), index),
    )
    while sum(capacities) < total:
        changed = False
        for index in add_order:
            if capacities[index] < counts[index]:
                capacities[index] += 1
                changed = True
                if sum(capacities) == total:
                    break
        if not changed:
            raise RuntimeError("native arena capacity allocation could not grow")
    remove_order = tuple(reversed(add_order))
    while sum(capacities) > total:
        changed = False
        for index in remove_order:
            if capacities[index] > 1:
                capacities[index] -= 1
                changed = True
                if sum(capacities) == total:
                    break
        if not changed:
            raise RuntimeError("native arena capacity allocation could not shrink")
    return tuple(capacities)


def _split_optional_weighted(
    total: int | None,
    weights: Sequence[int],
) -> tuple[int | None, ...]:
    """Split an optional budget without starving a larger worker shard."""
    if total is None:
        return tuple(None for _weight in weights)
    normalized = tuple(int(value) for value in weights)
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("native worker budget weights must be positive")
    if total < len(normalized):
        raise ValueError("native worker budget cannot cover every worker")
    budgets = [1] * len(normalized)
    remaining = total - len(normalized)
    weight_total = sum(normalized)
    exact = tuple(remaining * value / weight_total for value in normalized)
    for index, value in enumerate(exact):
        budgets[index] += int(value)
    leftover = total - sum(budgets)
    order = sorted(
        range(len(normalized)),
        key=lambda index: (-(exact[index] - int(exact[index])), index),
    )
    for index in order[:leftover]:
        budgets[index] += 1
    return cast(tuple[int | None, ...], tuple(budgets))


def _worker_seed(seed: int, worker_index: int) -> int:
    return int((seed + 0x9E3779B1 * (worker_index + 1)) & 0x7FFF_FFFF)


def _bind_worker_cpu_affinities(
    worker_specs: Sequence[NativeProcessWorkerSpec],
) -> tuple[NativeProcessWorkerSpec, ...]:
    """Assign stable feeder process trees to deterministic CPU partitions."""
    specs = tuple(worker_specs)
    if not specs:
        raise ValueError("native process collection requires worker specs")
    explicit = tuple(spec.cpu_affinity for spec in specs)
    if all(affinity is not None for affinity in explicit):
        return specs
    if any(affinity is not None for affinity in explicit):
        raise ValueError(
            "native process CPU affinities must be all explicit or all auto"
        )
    if not hasattr(os, "sched_getaffinity"):
        return specs
    partitions = _partition_worker_cpu_affinity(
        tuple(sorted(os.sched_getaffinity(0))),
        workers=len(specs),
    )
    return tuple(
        replace(spec, cpu_affinity=partitions[index])
        for index, spec in enumerate(specs)
    )


def _partition_worker_cpu_affinity(
    cpus: Sequence[int],
    *,
    workers: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition launcher CPUs completely, using explicit sharing if needed."""
    normalized = tuple(sorted({int(cpu) for cpu in cpus}))
    if workers <= 0 or not normalized or normalized[0] < 0:
        raise ValueError("native process CPU partition inputs are invalid")
    if len(normalized) < workers:
        return tuple((normalized[index % len(normalized)],) for index in range(workers))
    base, remainder = divmod(len(normalized), workers)
    partitions: list[tuple[int, ...]] = []
    cursor = 0
    for index in range(workers):
        width = base + int(index < remainder)
        partitions.append(normalized[cursor : cursor + width])
        cursor += width
    return tuple(partitions)


def _apply_worker_cpu_affinity(affinity: tuple[int, ...] | None) -> None:
    """Narrow one feeder before native libraries or fact children initialize."""
    if affinity is None or not hasattr(os, "sched_setaffinity"):
        return
    requested = set(affinity)
    available = set(os.sched_getaffinity(0))
    if not requested or not requested.issubset(available):
        raise ValueError("native process worker CPU affinity is unavailable")
    os.sched_setaffinity(0, requested)


__all__ = [
    "NativeProcessCollector",
    "NativeProcessWorkerSpec",
]
