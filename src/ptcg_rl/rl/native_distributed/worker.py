"""Remote CUDA worker for banked native collection shards."""

from __future__ import annotations

import logging
import multiprocessing
import queue
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import zmq

from ptcg_rl.belief.identity import file_sha256
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.decks import canonicalize_deck
from ptcg_rl.engine.native_prospective_facts import (
    NativeProspectiveEngineFactProducer,
)
from ptcg_rl.engine.prospective_facts import ProspectiveEngineFactProducer
from ptcg_rl.model.simple_stateless import (
    SimpleStatelessPolicyValueNet,
    materialize_simple_stateless_checkpoint_model,
)
from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload
from ptcg_rl.rl.native_banked_route_collection import (
    collect_native_banked_assigned,
)
from ptcg_rl.rl.native_distributed.artifact import (
    DecodedBfloat16RolloutArtifact,
    decode_bfloat16_rollout_artifact,
)
from ptcg_rl.rl.native_distributed.contracts import (
    NativeArtifactModelConfig,
    NativeBfloat16ArtifactManifest,
    NativeCollectionAttempt,
    NativeCollectionCapacityTier,
    NativeCollectionShardLease,
    NativeCollectionShardResult,
    NativeCollectionWindowReceipt,
    native_outcome_fingerprint,
    native_retry_capacity_tier,
)
from ptcg_rl.rl.native_distributed.cuda_memory import (
    cuda_memory_snapshot,
    trim_cuda_cache,
    trim_cuda_cache_if_needed,
)
from ptcg_rl.rl.native_distributed.data_plane import (
    encode_native_collection_part,
)
from ptcg_rl.rl.native_distributed.host_memory import (
    HostMemorySnapshot,
    host_memory_snapshot,
    trim_process_heap,
)
from ptcg_rl.rl.native_distributed.lifecycle import (
    NativeWorkerProcessRecycleError,
)
from ptcg_rl.rl.native_distributed.messages import (
    NativeArtifactRequest,
    NativeAttemptFailedRequest,
    NativeDrainResponse,
    NativeHeartbeatRequest,
    NativeLeaseResponse,
    NativePartAck,
    NativeProtocolAbort,
    NativeReadyRequest,
    NativeRegisterRequest,
    NativeShardCompleteRequest,
    NativeWaitResponse,
    NativeWindowReceiptMessage,
    NativeWorkerMetrics,
    NativeWorkRequest,
    decode_message,
    encode_message,
    message_model_name,
)
from ptcg_rl.rl.native_distributed.runtime import build_worker_manifest
from ptcg_rl.rl.native_distributed.transport import (
    NativeDistributedEndpoints,
    NativeWorkerSockets,
)
from ptcg_rl.rl.native_historical_inference import NativeHistoricalPolicyPool
from ptcg_rl.rl.native_policy_bank import NativePolicyCudaStreamPool
from ptcg_rl.rl.native_process_collection import (
    NativeProcessCollector,
    NativeProcessWorkerSpec,
)
from ptcg_rl.rl.native_route_arena import NativeArenaResourcePool
from ptcg_rl.rl.native_scripted_mixed75 import NativeMixed75Policy
from ptcg_rl.rl.native_scripted_policy import (
    NativePublicScriptedPolicy,
    NativeScriptedPolicy,
)
from ptcg_rl.rl.native_stateless_collection import NativeStatelessCollector
from ptcg_rl.rl.sequence_actor import (
    GeneralistSequenceActorPolicy,
    SequenceRolloutPrecision,
)
from ptcg_rl.rl.stateless_actor import SimpleStatelessActorPolicy
from ptcg_rl.rl.stateless_collection import StatelessCollectionResult
from ptcg_rl.rl.stateless_curriculum import PfspMember
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity
from ptcg_rl.rl.stateless_fragment_io import CompactFragmentPart
from ptcg_rl.rl.stateless_opponents import PastSelfNativeSource
from ptcg_rl.rl.stateless_training import (
    resolve_simple_stateless_training_resources,
)
from ptcg_rl.rl.stateless_training_config import (
    SimpleStatelessTrainingConfig,
    StatelessHistoricalAnchorConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LOGGER = logging.getLogger(__name__)

_ArtifactCacheState = DecodedBfloat16RolloutArtifact | Mapping[str, torch.Tensor]


class _LoadedPastSelfPool:
    """Minimal immutable native-source lookup for active past-self artifacts."""

    def __init__(self) -> None:
        self._actors: dict[str, PastSelfNativeSource] = {}

    def bind(
        self,
        member_ids: tuple[str, ...],
        actor: PastSelfNativeSource,
    ) -> None:
        for member_id in member_ids:
            existing = self._actors.get(member_id)
            if existing is not None and existing is not actor:
                raise ValueError("past-self member has conflicting BF16 artifacts")
            self._actors[member_id] = actor

    def native_source(self, member_id: str) -> PastSelfNativeSource:
        """Return the verified artifact source for one PFSP member."""
        try:
            return self._actors[member_id]
        except KeyError as exc:
            raise KeyError(f"past-self member is not loaded: {member_id}") from exc

    def executors_by_artifact(
        self,
        members: Sequence[PfspMember],
    ) -> dict[str, PastSelfNativeSource]:
        """Return one central actor per immutable past-self artifact."""
        result: dict[str, PastSelfNativeSource] = {}
        for member in members:
            if member.source not in {"past_self", "fixed_stateless_anchor"}:
                continue
            actor = self.native_source(member.member_id)
            existing = result.get(member.policy_sha256)
            if existing is not None and existing is not actor:
                raise ValueError("past-self artifact has conflicting actors")
            result[member.policy_sha256] = actor
        return result

    def close(self) -> None:
        """Release each unique source actor exactly once."""
        actors = {id(actor): actor for actor in self._actors.values()}
        self._actors.clear()
        for actor in actors.values():
            if isinstance(actor, GeneralistSequenceActorPolicy):
                actor.close()


class NativeCollectionWorker:
    """Poll leases, collect locally, and retain parts until exact ACK/receipt."""

    def __init__(
        self,
        config: SimpleStatelessTrainingConfig,
        *,
        worker_id: str,
        coordinator_host: str,
        worker_profile: str,
    ) -> None:
        """Preflight all worker-owned resources before sending READY."""
        if config.collection.backend != "native_distributed":
            raise ValueError(
                "collection-worker role requires native_distributed backend"
            )
        self.config = config
        self.worker_id = worker_id
        self.worker_profile_name = worker_profile
        self.profile = config.native_distributed.worker_profiles[worker_profile]
        torch.cuda.set_device(self.profile.cuda_device_index)
        if self.profile.maximum_cuda_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(
                self.profile.maximum_cuda_memory_fraction,
                device=self.profile.cuda_device_index,
            )
        self.session_id = uuid.uuid4().hex
        self.manifest = build_worker_manifest(
            config,
            worker_id=worker_id,
            session_id=self.session_id,
            worker_profile=worker_profile,
        )
        self.resources = resolve_simple_stateless_training_resources(config)
        transport = config.native_distributed.transport
        endpoints = NativeDistributedEndpoints.tcp(
            coordinator_host,
            control=transport.control_port,
            artifact=transport.artifact_port,
            data=transport.data_port,
        )
        self.sockets = NativeWorkerSockets(
            endpoints,
            worker_id=worker_id,
            session_id=self.session_id,
            io_threads=transport.io_threads,
            high_watermark=transport.socket_high_watermark,
        )
        historical_resources = {
            anchor.member_id: (
                anchor,
                self.resources.opponent_decks[anchor.exact_deck_digest].card_ids,
            )
            for anchor in _legacy_resident_anchors(config.curriculum.anchors)
        }
        self._v1_historical_resources = historical_resources
        self.historical = NativeHistoricalPolicyPool(historical_resources)
        self._historical_binding_mode = "v1"
        self.historical.preload_artifacts()
        self.scripted, self.scripted_bindings = self._build_scripted()
        self.policy_stream_pool = NativePolicyCudaStreamPool()
        policy_group_bank_limit = max(
            tier.native_policy_group_bank_limit for tier in self.manifest.capacity_tiers
        )
        self._artifact_cache: OrderedDict[str, _ArtifactCacheState] = OrderedDict()
        # Current-policy frames are consumed directly and never enter the host
        # cache. Only frozen bank members need cross-attempt CPU residency.
        self._artifact_cache_limit = policy_group_bank_limit
        self._model_cache_limit = 1 + policy_group_bank_limit
        self._artifact_cache_bytes = 0
        self._artifact_kinds: dict[str, str] = {}
        self._model_cache: OrderedDict[
            str,
            tuple[
                NativeBfloat16ArtifactManifest,
                NativeArtifactModelConfig,
                SimpleStatelessPolicyValueNet,
            ],
        ] = OrderedDict()
        self._engine_fact_producer: NativeProspectiveEngineFactProducer | None = None
        self._arena_resource_pool: NativeArenaResourcePool | None = None
        self._active_lease_id: str | None = None
        self._active_attempt_id: str | None = None
        self._attempt_stream_context: (
            tuple[
                NativeCollectionShardLease,
                NativeCollectionAttempt,
                queue.Queue[CompactFragmentPart],
            ]
            | None
        ) = None
        self._attempt_drain_event: threading.Event | None = None
        self.__active_window_id: str | None = None
        self._last_throughput = 0.0
        self._last_attempt_failure: dict[str, Any] | None = None
        self._next_cuda_cache_check_at = 0.0
        self._cuda_cache_trim_count = 0
        self._cuda_cache_trim_released_bytes = 0
        self._cuda_cache_last_trim_released_bytes = 0
        self._next_host_memory_check_at = 0.0
        self._last_host_memory_snapshot: HostMemorySnapshot | None = None
        self._host_memory_pressure_logged = False
        self._host_memory_trim_count = 0
        self._host_memory_trim_released_bytes = 0
        self._host_memory_last_trim_released_bytes = 0
        self._process_collector: NativeProcessCollector | None = None
        self._process_collector_key: (
            tuple[str, SequenceRolloutPrecision, int] | None
        ) = None
        self._status_path = (
            _REPO_ROOT
            / "tmp"
            / "native_distributed"
            / config.run.version
            / worker_id
            / "status.json"
        )
        self._closed = False

    def run(self) -> None:
        """Serve formal collection leases until interrupted."""
        try:
            self._expect_wait_response(
                self._request_control(
                    NativeRegisterRequest(
                        manifest=self.manifest,
                        sent_at_unix_ns=time.time_ns(),
                    )
                ),
                operation="registration",
            )
            self._expect_wait_response(
                self._request_control(
                    NativeReadyRequest(
                        worker_id=self.worker_id,
                        session_id=self.session_id,
                        sent_at_unix_ns=time.time_ns(),
                    )
                ),
                operation="READY",
            )
            while True:
                self._maybe_trim_cuda_cache()
                if self._pause_for_host_memory_pressure():
                    continue
                response = self._request_control(
                    NativeWorkRequest(
                        worker_id=self.worker_id,
                        session_id=self.session_id,
                        sent_at_unix_ns=time.time_ns(),
                    )
                )
                model_name = message_model_name(response)
                if model_name == NativeLeaseResponse.__name__:
                    assigned = decode_message(response, NativeLeaseResponse)
                    self._run_attempt(assigned.lease, assigned.attempt)
                elif model_name == NativeWaitResponse.__name__:
                    wait = decode_message(response, NativeWaitResponse)
                    time.sleep(wait.retry_after_seconds)
                elif model_name == NativeDrainResponse.__name__:
                    decode_message(response, NativeDrainResponse)
                    _discard, receipt = self._heartbeat()
                    if receipt is not None:
                        self._settle_window(receipt)
                    time.sleep(
                        self.config.native_distributed.retry.control_poll_interval_seconds
                    )
                elif model_name == NativeWindowReceiptMessage.__name__:
                    settled = decode_message(response, NativeWindowReceiptMessage)
                    self._settle_window(settled.receipt)
                elif model_name == NativeProtocolAbort.__name__:
                    aborted = decode_message(response, NativeProtocolAbort)
                    raise RuntimeError(
                        "native coordinator aborted worker control: "
                        f"{aborted.error_code}: {aborted.detail}"
                    )
                else:
                    raise RuntimeError(
                        f"native worker received unsupported response: {model_name}"
                    )
        finally:
            self.close()

    def close(self) -> None:
        """Release worker CUDA, native, socket, and artifact-cache ownership."""
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None

        def clean(operation: Any) -> None:
            nonlocal first_error
            try:
                operation()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                else:
                    first_error.add_note(
                        "additional native worker teardown failure: "
                        f"{type(error).__name__}: {error}"
                    )

        process_collector = getattr(self, "_process_collector", None)
        self._process_collector = None
        self._process_collector_key = None
        if process_collector is not None:
            clean(process_collector.close)
        arena_resource_pool = getattr(self, "_arena_resource_pool", None)
        self._arena_resource_pool = None
        if arena_resource_pool is not None:
            clean(arena_resource_pool.close)
        engine_fact_producer = getattr(self, "_engine_fact_producer", None)
        self._engine_fact_producer = None
        if engine_fact_producer is not None:
            clean(engine_fact_producer.close)
        clean(self.policy_stream_pool.close)
        clean(self.historical.close)
        for cached_state in self._artifact_cache.values():
            _release_artifact_state(cached_state)
        self._artifact_cache.clear()
        self._artifact_kinds.clear()
        self._model_cache.clear()
        self._artifact_cache_bytes = 0
        with suppress(OSError, RuntimeError):
            trim_cuda_cache(torch.device("cuda", self.profile.cuda_device_index))
        clean(self.sockets.close)
        with suppress(OSError, RuntimeError):
            self._record_host_memory_trim(reason="worker_close")
        if first_error is not None:
            raise first_error

    def _run_attempt(
        self,
        lease: NativeCollectionShardLease,
        attempt: NativeCollectionAttempt,
    ) -> None:
        self._active_lease_id = lease.lease_id
        self._active_attempt_id = attempt.attempt_id
        self._active_window_id = lease.window.identity.window_id
        started_at = time.perf_counter()
        drain_event = threading.Event()
        self._attempt_drain_event = drain_event
        try:
            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"native-shard-{lease.shard_sequence_id}",
            ) as executor:
                part_queue: queue.Queue[CompactFragmentPart] = queue.Queue(
                    maxsize=(
                        self.config.native_distributed.transport.worker_part_queue_capacity
                    )
                )
                discard_parts = threading.Event()

                def publish_part(part: CompactFragmentPart) -> None:
                    while not discard_parts.is_set():
                        try:
                            part_queue.put(part, timeout=0.1)
                            return
                        except queue.Full:
                            continue

                future = executor.submit(
                    self._collect_shard,
                    lease,
                    part_sink=publish_part,
                    drain_signal=drain_event.is_set,
                )
                self._attempt_stream_context = (lease, attempt, part_queue)
                try:
                    (
                        result,
                        discard_attempt,
                        receipt,
                        streamed_parts,
                        fragment_count,
                        decision_count,
                    ) = self._await_collection(
                        future,
                        discard_parts=discard_parts,
                    )
                finally:
                    # A control/data-plane failure must not deadlock executor
                    # teardown behind a full queue that nobody will ACK.
                    if not future.done():
                        discard_parts.set()
            if discard_attempt:
                if receipt is not None:
                    self._settle_window(receipt)
                return
            elapsed = max(time.perf_counter() - started_at, 1.0e-9)
            shard_result = NativeCollectionShardResult(
                lease_id=lease.lease_id,
                attempt_id=attempt.attempt_id,
                shard_sequence_id=lease.shard_sequence_id,
                part_count=streamed_parts,
                fragment_count=fragment_count,
                decision_count=decision_count,
                outcomes=result.outcomes,
                outcomes_fingerprint=native_outcome_fingerprint(result.outcomes),
                report=result.report,
                elapsed_seconds=elapsed,
            )
            self._expect_wait_response(
                self._request_control(
                    NativeShardCompleteRequest(
                        worker_id=self.worker_id,
                        session_id=self.session_id,
                        result=shard_result,
                        sent_at_unix_ns=time.time_ns(),
                    )
                ),
                operation="shard completion",
            )
            self._last_throughput = decision_count / elapsed
        except Exception as exc:
            if isinstance(exc, NativeWorkerProcessRecycleError):
                # Recycling is a process-lifecycle signal, not a failed shard.
                # It can originate from a receipt observed inside an active
                # attempt, so it must cross this retry/reporting boundary.
                raise
            reason = f"{type(exc).__name__}: {exc}"
            self._last_attempt_failure = {
                "recorded_at_unix_ns": time.time_ns(),
                "lease_id": lease.lease_id,
                "attempt_id": attempt.attempt_id,
                "reason": reason,
            }
            _LOGGER.exception(
                "native collection attempt failed: worker=%s lease=%s attempt=%s",
                self.worker_id,
                lease.lease_id,
                attempt.attempt_id,
            )
            with suppress(BaseException):
                self._expect_wait_response(
                    self._request_control(
                        NativeAttemptFailedRequest(
                            worker_id=self.worker_id,
                            session_id=self.session_id,
                            lease_id=lease.lease_id,
                            attempt_id=attempt.attempt_id,
                            reason=reason,
                            sent_at_unix_ns=time.time_ns(),
                        )
                    ),
                    operation="attempt failure",
                )
            if isinstance(exc, torch.OutOfMemoryError):
                # The advertised capacity is no longer safe for this CUDA
                # session. Retiring it prevents one oversubscribed process from
                # immediately consuming every attempt assigned to the same
                # immutable shard.
                raise
        finally:
            self._attempt_stream_context = None
            self._attempt_drain_event = None
            self._active_lease_id = None
            self._active_attempt_id = None
            self._settle_attempt_memory()
            self._write_status()

    def _await_collection(
        self,
        future: Future[StatelessCollectionResult],
        *,
        discard_parts: threading.Event,
    ) -> tuple[
        StatelessCollectionResult,
        bool,
        NativeCollectionWindowReceipt | None,
        int,
        int,
        int,
    ]:
        context = self._attempt_stream_context
        if context is None:
            raise RuntimeError("native worker has no active part stream")
        lease, attempt, part_queue = context
        heartbeat_interval = (
            self.config.native_distributed.quorum.heartbeat_interval_seconds
        )
        discard_attempt = False
        receipt: NativeCollectionWindowReceipt | None = None
        next_heartbeat = time.monotonic() + heartbeat_interval
        part_sequence = 0
        fragment_count = 0
        decision_count = 0
        while True:
            while not discard_attempt:
                try:
                    part = part_queue.get_nowait()
                except queue.Empty:
                    break
                if self._send_part(
                    lease,
                    attempt,
                    part,
                    part_sequence=part_sequence,
                ):
                    discard_attempt = True
                    discard_parts.set()
                    break
                part_sequence += 1
                fragment_count += part.fragment_count
                decision_count += part.decision_count
                now = time.monotonic()
                if now >= next_heartbeat:
                    heartbeat_discard, heartbeat_receipt = self._heartbeat()
                    discard_attempt = discard_attempt or heartbeat_discard
                    if discard_attempt:
                        discard_parts.set()
                    if heartbeat_receipt is not None:
                        receipt = heartbeat_receipt
                    next_heartbeat = now + heartbeat_interval
            if future.done():
                result = future.result()
                if not discard_attempt and not part_queue.empty():
                    continue
                final_discard, final_receipt = self._heartbeat()
                if final_discard:
                    discard_parts.set()
                retained_parts = result.compact_parts
                retained_fragments = sum(
                    part.fragment_count for part in retained_parts
                )
                retained_decisions = sum(
                    part.decision_count for part in retained_parts
                )
                if (
                    retained_decisions
                    != result.report.native_trainable_decisions
                ):
                    raise RuntimeError(
                        "native retained part rows differ from collection report"
                    )
                return (
                    result,
                    discard_attempt or final_discard,
                    final_receipt if final_receipt is not None else receipt,
                    len(retained_parts),
                    retained_fragments,
                    retained_decisions,
                )
            now = time.monotonic()
            if now >= next_heartbeat:
                heartbeat_discard, heartbeat_receipt = self._heartbeat()
                discard_attempt = discard_attempt or heartbeat_discard
                if discard_attempt:
                    discard_parts.set()
                if heartbeat_receipt is not None:
                    receipt = heartbeat_receipt
                next_heartbeat = now + heartbeat_interval
            time.sleep(0.01)

    def _send_part(
        self,
        lease: NativeCollectionShardLease,
        attempt: NativeCollectionAttempt,
        part: CompactFragmentPart,
        *,
        part_sequence: int,
    ) -> bool:
        """Publish and ACK one complete compact part while collection continues."""
        envelope, payload_header, frames = encode_native_collection_part(
            lease,
            attempt,
            self.manifest,
            part,
            part_sequence_id=part_sequence,
        )
        self.sockets.data.send_multipart(
            (envelope, payload_header, *frames),
            copy=False,
        )
        ack_frame = self._recv_one(
            self.sockets.data,
            timeout_seconds=(
                self.config.native_distributed.retry.part_ack_timeout_seconds
            ),
        )
        model_name = message_model_name(ack_frame)
        if model_name == NativeDrainResponse.__name__:
            decode_message(ack_frame, NativeDrainResponse)
            # The attempt is discarded server-side; latch the running
            # collection so it stops within a wave instead of playing its
            # remaining games into a zombie tail nobody will accept.
            self._request_attempt_drain()
            return True
        if model_name == NativeProtocolAbort.__name__:
            aborted = decode_message(ack_frame, NativeProtocolAbort)
            raise RuntimeError(
                "native coordinator rejected a fragment part: "
                f"{aborted.error_code}: {aborted.detail}"
            )
        ack = decode_message(ack_frame, NativePartAck)
        if (
            ack.part.attempt_id != attempt.attempt_id
            or ack.part.part_sequence_id != part_sequence
        ):
            raise RuntimeError("native collection part ACK identity differs")
        if ack.drain_hint:
            self._request_attempt_drain()
        return False

    def _request_attempt_drain(self) -> None:
        """Latch cutoff so the active attempt submits without a long tail."""
        drain_event = self._attempt_drain_event
        if drain_event is not None:
            drain_event.set()

    def _collect_shard(
        self,
        lease: NativeCollectionShardLease,
        *,
        part_sink: Any = None,
        drain_signal: Callable[[], bool] | None = None,
    ) -> StatelessCollectionResult:
        tier = self._tier_for_lease(lease)
        required_artifact_ids = lease.effective_required_artifact_ids
        required_artifact_set = set(required_artifact_ids)
        manifests = {item.artifact_id: item for item in lease.window.active_artifacts}
        wire_artifact_ids = required_artifact_set & set(manifests)
        historical_artifact_ids = required_artifact_set - set(manifests)
        self._configure_historical_pool(lease)
        if lease.window.shard_protocol_version == 1:
            self.historical.preload_artifacts()
        else:
            self.historical.ensure_artifacts(tuple(sorted(historical_artifact_ids)))
        current_manifest = next(
            manifests[item]
            for item in required_artifact_ids
            if item in wire_artifact_ids and manifests[item].kind == "current"
        )
        model_configs = {
            item.artifact_id: item
            for item in lease.window.artifact_model_configs
            if item.artifact_id in wire_artifact_ids
        }
        current_config = model_configs[current_manifest.artifact_id]
        current_model = self._model_for(
            current_manifest,
            model_config=current_config,
        )
        identity = self._fragment_identity(
            current_manifest,
            model_config=current_config,
        )
        assignments = tuple(item.to_assignment() for item in lease.assignments)
        temporal_slots = tier.native_arena_capacity * (
            1 + int(self.config.collection.mirror_bilateral_trajectories)
        )
        (
            frozen_batch_min_rows,
            frozen_batch_max_wait_waves,
            sequence_rollout_precision,
        ) = _distributed_collection_execution(self.config)
        if (
            tier.native_process_workers > 1
            and self.resources.route_input_contracts
        ):
            raise RuntimeError(
                "route-specific public catalogs require direct native workers"
            )
        actor = GeneralistSequenceActorPolicy(
            current_model,
            identity=identity,
            device=f"cuda:{self.profile.cuda_device_index}",
            verify_model_state=False,
            retain_raw_blocks=False,
            temporal_cache_slots=temporal_slots,
            rollout_precision=sequence_rollout_precision,
        )
        members = _lease_pfsp_members(lease)
        past_self = self._past_self_pool(
            lease,
            members=members,
            model_configs=model_configs,
            required_artifact_ids=wire_artifact_ids,
        )
        if tier.native_process_workers > 1:
            process_failure: BaseException | None = None
            try:
                return self._collect_process_shard(
                    lease,
                    tier=tier,
                    assignments=assignments,
                    actor=actor,
                    identity=identity,
                    members=members,
                    past_self=past_self,
                    frozen_batch_min_rows=frozen_batch_min_rows,
                    frozen_batch_max_wait_waves=frozen_batch_max_wait_waves,
                    sequence_rollout_precision=sequence_rollout_precision,
                    part_sink=part_sink,
                )
            except BaseException as exc:
                process_failure = exc
                raise
            finally:
                try:
                    actor.close()
                    past_self.close()
                except BaseException as cleanup_error:
                    if process_failure is None:
                        raise
                    process_failure.add_note(
                        "native process shard cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
        collector = NativeStatelessCollector(
            actor=actor,
            identity=identity,
            catalog=self.resources.catalog,
            active_decks=self.resources.active_decks,
            opponent_decks=self.resources.opponent_decks,
            members=members,
            past_self_pool=past_self,  # type: ignore[arg-type]
            historical_pool=self.historical,
            scripted_policies=self.scripted,
            scripted_bindings=self.scripted_bindings,
            maximum_engine_steps=self.config.collection.maximum_engine_steps,
            seed=lease.shard_seed,
            fragments_per_part=self.config.collection.fragments_per_part,
            mirror_bilateral_trajectories=(
                self.config.collection.mirror_bilateral_trajectories
            ),
            arena_capacity=tier.native_arena_capacity,
            engine_shards=tier.native_engine_shards,
            policy_cohort_slots=tier.native_policy_cohort_slots,
            policy_group_bank_limit=tier.native_policy_group_bank_limit,
            policy_cohort_wait_ms=(self.config.collection.native_policy_cohort_wait_ms),
            # The lease credit is a scheduling estimate, not a safe fragment
            # boundary. A local decision cutoff can erase a large fraction of
            # an otherwise healthy full-game batch once cutoff games are
            # transactionally excluded. The finite assignment lease and game
            # step limit bound this shard; the coordinator's drain signal
            # still cuts surplus in-flight work after terminal rows reach the
            # global window target.
            trainable_decision_budget=None,
            frozen_batch_min_rows=frozen_batch_min_rows,
            frozen_batch_max_wait_waves=frozen_batch_max_wait_waves,
            sequence_rollout_precision=sequence_rollout_precision,
            library_path=self.resources.native_library_path,
            engine_fact_producer=(self._engine_fact_producer_for(tier)),
            owns_engine_fact_producer=False,
            policy_stream_pool=self.policy_stream_pool,
            arena_resource_pool=self._arena_resource_pool_for(),
            route_input_contracts=self.resources.route_input_contracts,
            compact_part_sink=part_sink,
            external_drain_signal=drain_signal,
            immediate_whole_game_cutoff_on_drain=(
                lease.window.immediate_whole_game_cutoff_on_drain
            ),
        )
        failure: BaseException | None = None
        try:
            return collect_native_banked_assigned(collector, assignments)
        except BaseException as exc:
            failure = exc
            arena_resource_pool = self._arena_resource_pool
            self._arena_resource_pool = None
            if arena_resource_pool is not None:
                try:
                    arena_resource_pool.close()
                except BaseException as close_error:
                    exc.add_note(
                        "persistent native arena cleanup also failed: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
            raise
        finally:
            try:
                collector.close()
                actor.close()
                past_self.close()
            except BaseException as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(
                    "native distributed shard cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )

    def _past_self_pool(
        self,
        lease: NativeCollectionShardLease,
        *,
        members: Sequence[PfspMember],
        model_configs: dict[str, NativeArtifactModelConfig],
        required_artifact_ids: set[str],
    ) -> _LoadedPastSelfPool:
        pool = _LoadedPastSelfPool()
        by_policy: dict[str, tuple[str, ...]] = {}
        for member in members:
            if member.source in {"past_self", "fixed_stateless_anchor"}:
                if member.policy_sha256 not in required_artifact_ids:
                    raise RuntimeError("lease omitted its assigned past-self artifact")
                by_policy.setdefault(member.policy_sha256, ())
                by_policy[member.policy_sha256] += (member.member_id,)
        manifests = {item.artifact_id: item for item in lease.window.active_artifacts}
        for artifact_id, member_ids in by_policy.items():
            try:
                manifest = manifests[artifact_id]
            except KeyError as exc:
                raise RuntimeError(
                    "active past-self assignment has no BF16 artifact"
                ) from exc
            artifact_model_config = model_configs[manifest.artifact_id]
            model = self._model_for(
                manifest,
                model_config=artifact_model_config,
            )
            artifact_members = tuple(
                member
                for member in members
                if member.policy_sha256 == artifact_id
            )
            input_contracts = {
                member.input_contract_fingerprint for member in artifact_members
            }
            if len(input_contracts) != 1:
                raise ValueError(
                    "one past-self artifact has conflicting input contracts"
                )
            identity = self._fragment_identity(
                manifest,
                model_config=artifact_model_config,
                input_contract_fingerprint=next(iter(input_contracts)),
            )
            if model.sequence is None:
                source: PastSelfNativeSource = SimpleStatelessActorPolicy(
                    model,
                    identity=identity,
                    device=f"cuda:{self.profile.cuda_device_index}",
                    verify_model_state=False,
                )
            else:
                source = GeneralistSequenceActorPolicy(
                    model,
                    identity=identity,
                    device=f"cuda:{self.profile.cuda_device_index}",
                    verify_model_state=False,
                    retain_raw_blocks=False,
                    temporal_cache_slots=None,
                    rollout_precision="bf16",
                )
            pool.bind(member_ids, source)
        return pool

    def _collect_process_shard(
        self,
        lease: NativeCollectionShardLease,
        *,
        tier: NativeCollectionCapacityTier,
        assignments: tuple[Any, ...],
        actor: GeneralistSequenceActorPolicy,
        identity: StatelessFragmentIdentity,
        members: tuple[PfspMember, ...],
        past_self: _LoadedPastSelfPool,
        frozen_batch_min_rows: int,
        frozen_batch_max_wait_waves: int,
        sequence_rollout_precision: SequenceRolloutPrecision,
        part_sink: Callable[[CompactFragmentPart], None] | None,
    ) -> StatelessCollectionResult:
        """Run stable CPU collectors behind one centralized sequence actor."""
        process_collector = self._process_collector_for(
            tier,
            sequence_rollout_precision=sequence_rollout_precision,
        )
        process_collector.historical_pool = self.historical
        temporary_root = (
            _REPO_ROOT
            / "tmp"
            / "native_distributed_process"
            / self.config.run.version
            / self.worker_id
            / lease.lease_id
        )
        try:
            collected = process_collector.collect(
                actor=actor,
                identity=identity,
                assignments=assignments,
                members=members,
                temporary_root=temporary_root,
                seed=lease.shard_seed,
                arena_capacity=tier.native_arena_capacity,
                # Keep the process backend semantically identical to the
                # in-process path: estimated credit schedules capacity but
                # never truncates a shard needed by the global window.
                trainable_decision_budget=None,
                frozen_batch_min_rows=frozen_batch_min_rows,
                frozen_batch_max_wait_waves=frozen_batch_max_wait_waves,
                past_executors=cast(
                    Any,
                    past_self.executors_by_artifact(members),
                ),
                memory_only_parts=True,
                compact_part_sink=part_sink,
            )
            # Process workers persist NPZ parts only as bounded local IPC.
            # Loading already detaches every array; distributed transport must
            # not retain the path that this method deletes in its finalizer.
            collected.compact_parts = _memory_only_compact_parts(
                collected.compact_parts
            )
            collected.compact_part_paths = ()
            return collected
        except BaseException as error:
            # A failed window can leave a broker response or native arena in an
            # unknown phase. Retire only this optional worker's local process
            # group; the next coordinator retry creates a clean stable group.
            try:
                self._discard_process_collector(process_collector)
            except BaseException as close_error:
                error.add_note(
                    "native process collector cleanup also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
            raise
        finally:
            if temporary_root.exists():
                shutil.rmtree(temporary_root)

    def _process_collector_for(
        self,
        tier: NativeCollectionCapacityTier,
        *,
        sequence_rollout_precision: SequenceRolloutPrecision,
    ) -> NativeProcessCollector:
        """Return the stable process group for the current capacity tier."""
        key = ("persistent", sequence_rollout_precision, tier.native_process_workers)
        existing = self._process_collector
        if existing is not None and self._process_collector_key == key:
            return existing
        if existing is not None:
            self._discard_process_collector(existing)

        worker_count = tier.native_process_workers
        compatible_tiers = tuple(
            item
            for item in getattr(
                getattr(self, "manifest", None),
                "capacity_tiers",
                (tier,),
            )
            if item.native_process_workers == worker_count
        )
        context = multiprocessing.get_context("spawn")
        request_queue = context.Queue(maxsize=worker_count * 2)
        response_queues = {
            index: context.Queue(maxsize=2) for index in range(worker_count)
        }
        feature_path = Path(self.resources.model_config.card_encoder.feature_table_path)
        if not feature_path.is_absolute():
            feature_path = _REPO_ROOT / feature_path
        sequence = self.resources.model_config.sequence
        engine_fact_config = (
            sequence.engine_facts
            if sequence is not None and sequence.engine_facts.enabled
            else None
        )
        maximum_engine_shards = max(
            item.native_engine_shards for item in compatible_tiers
        )
        feeder_engine_shards = max(
            1,
            (maximum_engine_shards + worker_count - 1) // worker_count,
        )
        created = NativeProcessCollector(
            context=context,
            worker_specs=tuple(
                NativeProcessWorkerSpec(
                    worker_index=index,
                    catalog=self.resources.catalog,
                    active_decks=self.resources.active_decks,
                    opponent_decks=self.resources.opponent_decks,
                    scripted=self.config.curriculum.scripted,
                    static_feature_path=feature_path,
                    maximum_engine_steps=(self.config.collection.maximum_engine_steps),
                    fragments_per_part=self.config.collection.fragments_per_part,
                    mirror_bilateral_trajectories=(
                        self.config.collection.mirror_bilateral_trajectories
                    ),
                    options_per_lane=256,
                    library_path=self.resources.native_library_path,
                    inference_timeout_seconds=(
                        self.config.collection.inference_timeout_seconds
                    ),
                    sequence_rollout_precision=sequence_rollout_precision,
                    engine_shards=feeder_engine_shards,
                    policy_cohort_wait_ms=(
                        self.config.collection.native_policy_cohort_wait_ms
                    ),
                    engine_fact_config=engine_fact_config,
                    engine_fact_workers=max(
                        1,
                        max(
                            item.native_engine_fact_workers for item in compatible_tiers
                        )
                        // worker_count,
                    ),
                    maximum_arena_capacity=max(
                        1,
                        max(item.native_arena_capacity for item in compatible_tiers)
                        // worker_count,
                    ),
                    scripted_artifacts={
                        opponent_id: resolved.artifact
                        for opponent_id, resolved in (
                            self.resources.scripted_opponents.items()
                        )
                    },
                )
                for index in range(worker_count)
            ),
            request_queue=request_queue,
            response_queues=response_queues,
            past_self_pool=cast(Any, None),
            historical_pool=self.historical,
            max_batch_rows=(
                max(
                    item.native_policy_cohort_slots or item.native_arena_capacity
                    for item in compatible_tiers
                )
            ),
            batch_wait_seconds=(
                self.config.collection.inference_batch_wait_ms / 1000.0
            ),
            result_timeout_seconds=(self.config.collection.collection_timeout_seconds),
        )
        self._process_collector = created
        self._process_collector_key = key
        return created

    def _engine_fact_producer_for(
        self,
        _tier: NativeCollectionCapacityTier,
    ) -> NativeProspectiveEngineFactProducer | None:
        """Return the worker-lifetime fact pool shared by every direct shard."""
        sequence = self.resources.model_config.sequence
        if sequence is None or not sequence.engine_facts.enabled:
            return None
        existing = self._engine_fact_producer
        if existing is not None:
            return existing
        maximum_workers = max(
            item.native_engine_fact_workers for item in self.manifest.capacity_tiers
        )
        created = NativeProspectiveEngineFactProducer(
            ProspectiveEngineFactProducer(
                sampler=BeliefSampler(config=sequence.engine_facts.sampler),
                config=sequence.engine_facts,
            ),
            maximum_workers=maximum_workers,
        )
        self._engine_fact_producer = created
        return created

    def _arena_resource_pool_for(self) -> NativeArenaResourcePool:
        """Return native lanes sized once for every advertised direct tier."""
        existing = self._arena_resource_pool
        if existing is not None:
            return existing
        resource_count = max(
            item.native_engine_shards for item in self.manifest.capacity_tiers
        )
        lane_capacity = max(
            (item.native_arena_capacity + item.native_engine_shards - 1)
            // item.native_engine_shards
            for item in self.manifest.capacity_tiers
        )
        created = NativeArenaResourcePool(
            resource_count=resource_count,
            lane_capacity=lane_capacity,
            options_per_lane=256,
            library_path=self.resources.native_library_path,
            catalog=self.resources.catalog,
            input_contract_fingerprint=(
                self.resources.policy_identity.input_contract_fingerprint
            ),
            route_input_contracts=self.resources.route_input_contracts,
        )
        self._arena_resource_pool = created
        return created

    def _discard_process_collector(
        self,
        collector: NativeProcessCollector,
    ) -> None:
        """Retire one matching process group without touching CUDA service state."""
        if self._process_collector is collector:
            self._process_collector = None
            self._process_collector_key = None
        collector.close()

    def _model_for(
        self,
        manifest: NativeBfloat16ArtifactManifest,
        *,
        model_config: NativeArtifactModelConfig,
    ) -> SimpleStatelessPolicyValueNet:
        if model_config.artifact_id != manifest.artifact_id:
            raise ValueError("rollout artifact model configuration differs")
        config_value = model_config.to_model_config()
        if manifest.kind == "current" and config_value != self.resources.model_config:
            raise ValueError("current rollout model configuration differs")
        cached_model = self._model_cache.get(manifest.artifact_id)
        if cached_model is not None:
            cached_manifest, cached_config, model = cached_model
            if cached_manifest != manifest or cached_config != model_config:
                raise ValueError("cached rollout model identity differs")
            self._model_cache.move_to_end(manifest.artifact_id)
            return model
        cached_state = self._artifact_cache.get(manifest.artifact_id)
        transient_state: DecodedBfloat16RolloutArtifact | None = None
        if cached_state is None:
            request = NativeArtifactRequest(
                request_id=uuid.uuid4().hex,
                worker_id=self.worker_id,
                session_id=self.session_id,
                window_id=self._window_id(manifest),
                artifact_id=manifest.artifact_id,
                expected_manifest=manifest,
            )
            self.sockets.artifact.send(encode_message(request), copy=True)
            message = self._recv_multipart(
                self.sockets.artifact,
                timeout_seconds=self.config.collection.inference_timeout_seconds,
            )
            if len(message) == 1 and (
                message_model_name(message[0]) == NativeProtocolAbort.__name__
            ):
                aborted = decode_message(message[0], NativeProtocolAbort)
                raise RuntimeError(
                    "native artifact request was rejected: "
                    f"{aborted.error_code}: {aborted.detail}"
                )
            decoded = decode_bfloat16_rollout_artifact(
                message[0],
                message[1:],
                expected_manifest=manifest,
            )
            message.clear()
            if manifest.kind == "current":
                # Loading directly from verified borrowed frames avoids one
                # model-sized CPU state clone per policy version.
                transient_state = decoded
            else:
                self._cache_artifact_state(
                    manifest.artifact_id,
                    decoded,
                    kind=manifest.kind,
                )
            state = _artifact_tensors(decoded)
        else:
            self._artifact_cache.move_to_end(manifest.artifact_id)
            state = _artifact_tensors(cached_state)
        self._make_model_cache_room()
        try:
            # A normal constructor allocates and initializes a complete FP32
            # CPU parameter set even with ``initialize=False``. Building on
            # meta and assigning the verified BF16 frame views prevents that
            # throwaway model-sized host allocation before the CUDA transfer.
            model = materialize_simple_stateless_checkpoint_model(
                config_value,
                state,
            )
            model.to(
                device=f"cuda:{self.profile.cuda_device_index}",
                dtype=torch.bfloat16,
            )
            model.eval().requires_grad_(False)
        finally:
            if transient_state is not None:
                transient_state.release()
        self._model_cache[manifest.artifact_id] = (
            manifest,
            model_config,
            model,
        )
        self._model_cache.move_to_end(manifest.artifact_id)
        self._artifact_kinds[manifest.artifact_id] = manifest.kind
        return model

    def _make_model_cache_room(self) -> None:
        """Evict GPU weights before allocating a replacement model."""
        model_cache_limit = getattr(
            self,
            "_model_cache_limit",
            self._artifact_cache_limit,
        )
        while len(self._model_cache) >= model_cache_limit:
            artifact_id, evicted = self._model_cache.popitem(last=False)
            del evicted
            self._forget_artifact_kind_if_uncached(artifact_id)

    def _cache_artifact_state(
        self,
        artifact_id: str,
        state: _ArtifactCacheState,
        *,
        kind: str | None = None,
    ) -> None:
        """Insert one CPU state into the bounded cross-lease LRU cache."""
        previous = self._artifact_cache.pop(artifact_id, None)
        if previous is not None:
            self._artifact_cache_bytes -= _state_nbytes(previous)
            _release_artifact_state(previous)
        while len(self._artifact_cache) >= self._artifact_cache_limit:
            evicted_id, evicted = self._artifact_cache.popitem(last=False)
            self._artifact_cache_bytes -= _state_nbytes(evicted)
            _release_artifact_state(evicted)
            self._forget_artifact_kind_if_uncached(evicted_id)
        self._artifact_cache[artifact_id] = state
        artifact_kinds = getattr(self, "_artifact_kinds", None)
        if artifact_kinds is not None and kind is not None:
            artifact_kinds[artifact_id] = kind
        self._artifact_cache.move_to_end(artifact_id)
        self._artifact_cache_bytes += _state_nbytes(state)

    def _forget_artifact_kind_if_uncached(self, artifact_id: str) -> None:
        """Forget identity metadata only after both cache tiers release it."""
        if artifact_id in self._artifact_cache or artifact_id in getattr(
            self, "_model_cache", ()
        ):
            return
        artifact_kinds = getattr(self, "_artifact_kinds", None)
        if artifact_kinds is not None:
            artifact_kinds.pop(artifact_id, None)

    def _configure_historical_pool(
        self,
        lease: NativeCollectionShardLease,
    ) -> None:
        """Bind legacy runtime resources to the active V1 or V2 window."""
        if lease.window.shard_protocol_version == 1:
            desired_mode = "v1"
            if self._historical_binding_mode == desired_mode:
                return
            resources = self._v1_historical_resources
        else:
            desired_mode = "v2"
            resources = _v2_historical_resources(
                lease.window.historical_artifact_bindings,
                repo_root=_REPO_ROOT,
            )
            if self._historical_binding_mode == desired_mode:
                # Windows with one opponent-pool revision may carry different
                # active member subsets. Merge their immutable bindings into
                # the long-lived pool so already-loaded checkpoint runtimes
                # remain warm while newly referenced fingerprints are valid.
                self.historical.ensure_resources(resources)
                return
        self.historical.close()
        self.historical = NativeHistoricalPolicyPool(resources)
        self._historical_binding_mode = desired_mode

    def _window_id(self, manifest: NativeBfloat16ArtifactManifest) -> str:
        # The manifest is supplied only by the active lease currently owned by
        # this worker. Keep the exact window ID in the active lease status.
        if self._active_window_id is None:
            raise RuntimeError("native worker has no active artifact window")
        return self._active_window_id

    @property
    def _active_window_id(self) -> str | None:
        lease_id = self._active_lease_id
        if lease_id is None:
            return None
        return self.__active_window_id

    @_active_window_id.setter
    def _active_window_id(self, value: str | None) -> None:
        self.__active_window_id = value

    def _fragment_identity(
        self,
        manifest: NativeBfloat16ArtifactManifest,
        *,
        model_config: NativeArtifactModelConfig,
        input_contract_fingerprint: str | None = None,
    ) -> StatelessFragmentIdentity:
        identity = self.resources.policy_identity
        config_value = model_config.to_model_config()
        sequence = config_value.sequence is not None
        public_catalog_fingerprint = (
            config_value.public_deck_catalog_fingerprint
        )
        if public_catalog_fingerprint is None:
            raise ValueError("rollout artifact model omitted its public catalog")
        return StatelessFragmentIdentity(
            schema_version=2 if sequence else 1,
            horizon=self.config.collection.fragment_horizon,
            behavior_policy_version=manifest.source_policy_version,
            behavior_policy_fingerprint=manifest.source_fp32_fingerprint,
            model_config_fingerprint=model_config.model_config_fingerprint,
            action_schema_fingerprint=identity.action_schema_fingerprint,
            public_context_fingerprint=identity.public_context_fingerprint,
            card_catalog_fingerprint=identity.card_catalog_fingerprint,
            public_deck_catalog_fingerprint=public_catalog_fingerprint,
            exact_registry_fingerprint=model_config.exact_registry_fingerprint,
            belief_target_semantics_fingerprint=(
                identity.belief_target_semantics_fingerprint
            ),
            input_contract_fingerprint=(
                identity.input_contract_fingerprint
                if input_contract_fingerprint is None
                else input_contract_fingerprint
            ),
            resolved_config_fingerprint=identity.resolved_config_fingerprint,
            sequence_contract_fingerprint=(
                identity.sequence_contract_fingerprint if sequence else None
            ),
        )

    def _tier(self, tier_id: str) -> NativeCollectionCapacityTier:
        try:
            return next(
                item for item in self.manifest.capacity_tiers if item.tier_id == tier_id
            )
        except StopIteration as exc:
            raise RuntimeError(
                f"native collection lease requested unknown tier: {tier_id}"
            ) from exc

    def _tier_for_lease(
        self,
        lease: NativeCollectionShardLease,
    ) -> NativeCollectionCapacityTier:
        """Resolve a safe worker-local tier for an original or retried lease."""
        tier = native_retry_capacity_tier(self.manifest.capacity_tiers, lease)
        if tier is None:
            raise RuntimeError(
                "native collection worker cannot execute the lease geometry"
            )
        return tier

    def _build_scripted(
        self,
    ) -> tuple[
        dict[str, NativeScriptedPolicy],
        dict[str, tuple[str, str]],
    ]:
        static_features = np.load(
            self._repo_path(
                self.resources.model_config.card_encoder.feature_table_path
            ),
            mmap_mode="r",
            allow_pickle=False,
        )
        policies: dict[str, NativeScriptedPolicy] = {}
        bindings: dict[str, tuple[str, str]] = {}
        for item in self.config.curriculum.scripted:
            if item.opponent_name == "mixed75":
                policies[item.opponent_id] = NativeMixed75Policy(
                    scripted_deck=self.resources.opponent_decks[
                        item.exact_deck_digest
                    ].card_ids,
                    static_features=static_features,
                )
            else:
                policies[item.opponent_id] = NativePublicScriptedPolicy(
                    self.resources.scripted_opponents[item.opponent_id]
                )
            bindings[item.opponent_id] = (
                item.artifact_fingerprint,
                item.exact_deck_digest,
            )
        return policies, bindings

    def _heartbeat(
        self,
    ) -> tuple[bool, NativeCollectionWindowReceipt | None]:
        response = self._request_control(
            NativeHeartbeatRequest(
                worker_id=self.worker_id,
                session_id=self.session_id,
                metrics=self._metrics(),
                sent_at_unix_ns=time.time_ns(),
            )
        )
        model_name = message_model_name(response)
        directive: tuple[bool, NativeCollectionWindowReceipt | None]
        if model_name == NativeWaitResponse.__name__:
            wait = decode_message(response, NativeWaitResponse)
            if wait.drain_hint:
                self._request_attempt_drain()
            directive = (False, None)
        elif model_name == NativeDrainResponse.__name__:
            decode_message(response, NativeDrainResponse)
            self._request_attempt_drain()
            directive = (True, None)
        elif model_name == NativeWindowReceiptMessage.__name__:
            settled = decode_message(response, NativeWindowReceiptMessage)
            self._request_attempt_drain()
            directive = (True, settled.receipt)
        elif model_name == NativeProtocolAbort.__name__:
            aborted = decode_message(response, NativeProtocolAbort)
            raise RuntimeError(
                "native coordinator aborted worker heartbeat: "
                f"{aborted.error_code}: {aborted.detail}"
            )
        else:
            raise RuntimeError(
                f"native heartbeat received unsupported response: {model_name}"
            )
        self._write_status()
        return directive

    def _settle_window(self, receipt: NativeCollectionWindowReceipt) -> None:
        current_ids = {
            artifact_id
            for artifact_id, kind in self._artifact_kinds.items()
            if kind == "current"
        }
        for artifact_id in current_ids:
            state = self._artifact_cache.pop(artifact_id, None)
            if state is not None:
                self._artifact_cache_bytes -= _state_nbytes(state)
                _release_artifact_state(state)
            self._artifact_kinds.pop(artifact_id, None)
            evicted = self._model_cache.pop(artifact_id, None)
            del evicted
        self._active_window_id = None
        self._prune_settled_model_cache()
        self._settle_window_host_memory()
        self._maybe_trim_cuda_cache(force_check=True, window_settlement=True)
        self._write_status(receipt=receipt)
        self._maybe_recycle_after_window(window_sequence=receipt.sequence_id)

    def _settle_attempt_memory(self) -> None:
        """Release attempt-local native and allocator residency at quiescence."""
        host_policy = self.config.native_distributed.host_memory
        if host_policy.release_arena_resources_after_attempt:
            self._release_arena_resources(reason="attempt")
        self._prune_settled_model_cache()
        self._maybe_trim_cuda_cache(force_check=True, attempt_settlement=True)
        if host_policy.enabled and host_policy.trim_after_attempt:
            try:
                self._record_host_memory_trim(reason="attempt_settlement")
            except (OSError, RuntimeError):
                _LOGGER.exception(
                    "native worker host heap trim failed at attempt boundary: "
                    "worker=%s",
                    self.worker_id,
                )

    def _settle_window_host_memory(self) -> None:
        """Release window-scoped native residency before accepting more work."""
        config = getattr(self, "config", None)
        native = getattr(config, "native_distributed", None)
        host_policy = getattr(native, "host_memory", None)
        if host_policy is None:
            return
        if getattr(host_policy, "release_arena_resources_after_window", False):
            self._release_arena_resources(reason="window")
        if host_policy.enabled and getattr(host_policy, "trim_after_window", False):
            try:
                self._record_host_memory_trim(reason="window_settlement")
            except (OSError, RuntimeError):
                _LOGGER.exception(
                    "native worker host heap trim failed at window boundary: worker=%s",
                    self.worker_id,
                )

    def _release_arena_resources(self, *, reason: str) -> None:
        """Close the persistent native arena pool at a quiescent boundary."""
        arena_resource_pool = self._arena_resource_pool
        self._arena_resource_pool = None
        if arena_resource_pool is None:
            return
        try:
            arena_resource_pool.close()
        except BaseException:
            _LOGGER.exception(
                "native worker arena release failed at %s boundary: worker=%s",
                reason,
                self.worker_id,
            )

    def _maybe_recycle_after_window(
        self,
        *,
        window_sequence: int | None = None,
    ) -> None:
        """Recycle the process if quiescent RSS remains above its safety guard."""
        config = getattr(self, "config", None)
        native = getattr(config, "native_distributed", None)
        policy = getattr(native, "host_memory", None)
        if policy is None or not policy.enabled:
            return
        limit = getattr(policy, "process_rss_recycle_limit_bytes", None)
        if limit is None:
            return
        snapshot = self._last_host_memory_snapshot
        if snapshot is None:
            snapshot = self._observe_host_memory(force_check=True)
        if snapshot is None or snapshot.process_rss_bytes < limit:
            return
        quorum = getattr(native, "quorum", None)
        required_worker_ids = tuple(
            getattr(quorum, "required_worker_ids", ())
        )
        if self.worker_id in required_worker_ids and window_sequence is not None:
            recycle_worker_id = required_worker_ids[
                window_sequence % len(required_worker_ids)
            ]
            if self.worker_id != recycle_worker_id:
                _LOGGER.warning(
                    "native worker deferred a clean process recycle to avoid "
                    "quorum-wide turnover: worker=%s rss_bytes=%d "
                    "recycle_limit_bytes=%d window_sequence=%d "
                    "scheduled_worker=%s",
                    self.worker_id,
                    snapshot.process_rss_bytes,
                    limit,
                    window_sequence,
                    recycle_worker_id,
                )
                return
        _LOGGER.warning(
            "native worker requested a clean process recycle after window "
            "settlement: worker=%s rss_bytes=%d recycle_limit_bytes=%d "
            "window_sequence=%s",
            self.worker_id,
            snapshot.process_rss_bytes,
            limit,
            window_sequence,
        )
        raise NativeWorkerProcessRecycleError(
            "native worker quiescent RSS exceeded its process recycle limit"
        )

    def _prune_settled_model_cache(self) -> None:
        """Keep only the configured warm GPU models at a safe boundary."""
        config = getattr(self, "config", None)
        native = getattr(config, "native_distributed", None)
        cuda_policy = getattr(native, "cuda_cache", None)
        limit = getattr(cuda_policy, "settled_model_cache_limit", None)
        if limit is None:
            return
        evicted_count = 0
        while len(self._model_cache) > limit:
            artifact_id = next(
                (
                    candidate
                    for candidate in self._model_cache
                    if self._artifact_kinds.get(candidate) != "current"
                ),
                next(iter(self._model_cache)),
            )
            evicted = self._model_cache.pop(artifact_id)
            del evicted
            self._forget_artifact_kind_if_uncached(artifact_id)
            evicted_count += 1
        if evicted_count:
            _LOGGER.info(
                "native worker pruned settled CUDA model cache: worker=%s "
                "evicted=%d remaining=%d limit=%d",
                self.worker_id,
                evicted_count,
                len(self._model_cache),
                limit,
            )

    def _maybe_trim_cuda_cache(
        self,
        *,
        force_check: bool = False,
        attempt_settlement: bool = False,
        window_settlement: bool = False,
    ) -> None:
        """Return inactive CUDA pages at safe boundaries under device pressure."""
        policy = self.config.native_distributed.cuda_cache
        if not policy.enabled:
            return
        now = time.monotonic()
        if not force_check and now < self._next_cuda_cache_check_at:
            return
        self._next_cuda_cache_check_at = now + policy.check_interval_seconds
        device = torch.device("cuda", self.profile.cuda_device_index)
        force_trim = (
            attempt_settlement
            and getattr(policy, "force_attempt_settlement_trim", False)
        ) or (window_settlement and policy.force_window_settlement_trim)
        try:
            result = (
                trim_cuda_cache(
                    device,
                    clear_cublas_workspaces=(
                        window_settlement
                        and getattr(
                            policy,
                            "clear_cublas_workspaces_at_window_settlement",
                            False,
                        )
                    ),
                )
                if force_trim
                else trim_cuda_cache_if_needed(
                    device,
                    device_free_floor_fraction=policy.device_free_floor_fraction,
                    minimum_reclaimable_device_fraction=(
                        policy.minimum_reclaimable_device_fraction
                    ),
                )
            )
        except (OSError, RuntimeError) as exc:
            _LOGGER.warning(
                "native worker CUDA cache pressure check failed: worker=%s error=%s",
                self.worker_id,
                exc,
            )
            return
        if result is None:
            return
        released_bytes = result.allocator_released_bytes
        self._cuda_cache_trim_count += 1
        self._cuda_cache_trim_released_bytes += released_bytes
        self._cuda_cache_last_trim_released_bytes = released_bytes
        _LOGGER.info(
            "native worker released inactive CUDA cache: worker=%s "
            "allocator_bytes=%d device_bytes=%d reserved_after=%d "
            "device_free_after=%d collected_objects=%d "
            "cublas_workspaces_cleared=%s reason=%s",
            self.worker_id,
            released_bytes,
            result.device_released_bytes,
            result.after.reserved_bytes,
            result.after.device_free_bytes,
            result.collected_objects,
            result.cublas_workspaces_cleared,
            (
                "attempt_settlement"
                if attempt_settlement and force_trim
                else "window_settlement"
                if window_settlement and force_trim
                else "device_pressure"
            ),
        )

    def _observe_host_memory(
        self,
        *,
        force_check: bool = False,
    ) -> HostMemorySnapshot | None:
        """Sample host memory at a bounded cadence and drain under pressure."""
        native = self.config.native_distributed
        policy = getattr(native, "host_memory", None)
        now = time.monotonic()
        cached = cast(
            HostMemorySnapshot | None,
            getattr(self, "_last_host_memory_snapshot", None),
        )
        next_check = float(getattr(self, "_next_host_memory_check_at", 0.0))
        if not force_check and cached is not None and now < next_check:
            return cached
        try:
            snapshot = host_memory_snapshot()
        except (OSError, RuntimeError) as exc:
            _LOGGER.warning(
                "native worker host memory check failed: worker=%s error=%s",
                self.worker_id,
                exc,
            )
            return cached
        self._last_host_memory_snapshot = snapshot
        interval = 15.0 if policy is None else policy.check_interval_seconds
        self._next_host_memory_check_at = now + interval
        if policy is None or not policy.enabled:
            return snapshot

        reasons: list[str] = []
        if (
            policy.process_rss_soft_limit_bytes is not None
            and snapshot.process_rss_bytes >= policy.process_rss_soft_limit_bytes
        ):
            reasons.append("process_rss_soft_limit")
        if (
            policy.process_rss_hard_limit_bytes is not None
            and snapshot.process_rss_bytes >= policy.process_rss_hard_limit_bytes
        ):
            reasons.append("process_rss_hard_limit")
        if reasons:
            self._request_attempt_drain()
            if not getattr(self, "_host_memory_pressure_logged", False):
                _LOGGER.warning(
                    "native worker requested an attempt drain for host memory "
                    "pressure: worker=%s reasons=%s rss_bytes=%d "
                    "system_available_bytes=%d",
                    self.worker_id,
                    ",".join(reasons),
                    snapshot.process_rss_bytes,
                    snapshot.system_available_bytes,
                )
                self._host_memory_pressure_logged = True
        else:
            self._host_memory_pressure_logged = False
        return snapshot

    def _pause_for_host_memory_pressure(self) -> bool:
        """Avoid accepting a new lease until post-trim host headroom returns."""
        policy = self.config.native_distributed.host_memory
        if not policy.enabled:
            return False
        snapshot = self._observe_host_memory(force_check=True)
        if snapshot is None:
            return False
        if (
            policy.process_rss_hard_limit_bytes is not None
            and snapshot.process_rss_bytes >= policy.process_rss_hard_limit_bytes
        ):
            self._record_host_memory_trim(reason="pre_lease_hard_limit")
            snapshot = self._observe_host_memory(force_check=True) or snapshot
            if snapshot.process_rss_bytes >= policy.process_rss_hard_limit_bytes:
                raise MemoryError(
                    "native worker RSS remains above its hard limit after trim"
                )
        if (
            policy.system_available_memory_floor_bytes is None
            or snapshot.system_available_bytes
            >= policy.system_available_memory_floor_bytes
        ):
            return False

        _discard, receipt = self._heartbeat()
        if receipt is not None:
            self._settle_window(receipt)
        time.sleep(self.config.native_distributed.retry.control_poll_interval_seconds)
        return True

    def _record_host_memory_trim(self, *, reason: str) -> None:
        """Trim the process heap and publish its observed RSS release."""
        result = trim_process_heap()
        released_bytes = result.released_bytes
        self._last_host_memory_snapshot = result.after
        self._host_memory_trim_count = getattr(self, "_host_memory_trim_count", 0) + 1
        self._host_memory_trim_released_bytes = (
            getattr(self, "_host_memory_trim_released_bytes", 0) + released_bytes
        )
        self._host_memory_last_trim_released_bytes = released_bytes
        _LOGGER.info(
            "native worker trimmed host heap: worker=%s reason=%s "
            "rss_before=%d rss_after=%d released_bytes=%d "
            "collected_objects=%d allocator_trimmed=%s",
            self.worker_id,
            reason,
            result.before.process_rss_bytes,
            result.after.process_rss_bytes,
            released_bytes,
            result.collected_objects,
            result.allocator_trimmed,
        )

    def _metrics(self) -> NativeWorkerMetrics:
        device = self.profile.cuda_device_index
        include_gpu_metrics = (
            self.config.native_distributed.status.include_worker_gpu_metrics
        )
        if include_gpu_metrics:
            memory = cuda_memory_snapshot(torch.device("cuda", device))
            try:
                gpu_utilization = float(torch.cuda.utilization(device))
            except (AttributeError, ModuleNotFoundError, OSError, RuntimeError):
                gpu_utilization = 0.0
        else:
            memory = None
            gpu_utilization = 0.0
        host_memory = self._observe_host_memory()
        return NativeWorkerMetrics(
            gpu_memory_allocated_bytes=(
                0 if memory is None else memory.allocated_bytes
            ),
            gpu_memory_reserved_bytes=(0 if memory is None else memory.reserved_bytes),
            gpu_memory_reclaimable_bytes=(
                0 if memory is None else memory.reclaimable_bytes
            ),
            gpu_device_free_bytes=(0 if memory is None else memory.device_free_bytes),
            gpu_utilization_percent=gpu_utilization,
            cuda_cache_trim_count=self._cuda_cache_trim_count,
            cuda_cache_trim_released_bytes=self._cuda_cache_trim_released_bytes,
            cuda_cache_last_trim_released_bytes=(
                self._cuda_cache_last_trim_released_bytes
            ),
            process_rss_bytes=(
                0 if host_memory is None else host_memory.process_rss_bytes
            ),
            system_available_memory_bytes=(
                0 if host_memory is None else host_memory.system_available_bytes
            ),
            cgroup_memory_current_bytes=(
                0
                if host_memory is None
                or host_memory.cgroup_memory_current_bytes is None
                else host_memory.cgroup_memory_current_bytes
            ),
            cgroup_memory_limit_bytes=(
                0
                if host_memory is None or host_memory.cgroup_memory_limit_bytes is None
                else host_memory.cgroup_memory_limit_bytes
            ),
            host_memory_trim_count=getattr(self, "_host_memory_trim_count", 0),
            host_memory_trim_released_bytes=getattr(
                self, "_host_memory_trim_released_bytes", 0
            ),
            host_memory_last_trim_released_bytes=getattr(
                self, "_host_memory_last_trim_released_bytes", 0
            ),
            artifact_cache_entries=len(self._artifact_cache),
            artifact_cache_bytes=self._artifact_cache_bytes,
            model_cache_entries=len(getattr(self, "_model_cache", ())),
            collection_decisions_per_second=self._last_throughput,
            active_lease_id=self._active_lease_id,
            active_attempt_id=self._active_attempt_id,
        )

    def _write_status(
        self,
        *,
        receipt: NativeCollectionWindowReceipt | None = None,
    ) -> None:
        try:
            atomic_write_bytes(
                self._status_path,
                json_payload(
                    {
                        "format": "native_distributed_worker_status_v1",
                        "recorded_at_unix_ns": time.time_ns(),
                        "worker_id": self.worker_id,
                        "session_id": self.session_id,
                        "worker_profile": self.worker_profile_name,
                        "metrics": self._metrics().model_dump(mode="json"),
                        "last_attempt_failure": self._last_attempt_failure,
                        "last_receipt": (
                            None if receipt is None else receipt.model_dump(mode="json")
                        ),
                    }
                ),
                overwrite=True,
            )
        except Exception:
            _LOGGER.exception(
                "native worker status publication failed; collection will continue: "
                "worker=%s",
                self.worker_id,
            )

    def _request_control(self, request: Any) -> Any:
        self.sockets.control.send(encode_message(request), copy=True)
        return self._recv_one(
            self.sockets.control,
            timeout_seconds=min(
                self.config.collection.inference_timeout_seconds,
                self.config.native_distributed.quorum.heartbeat_timeout_seconds,
            ),
        )

    @staticmethod
    def _expect_wait_response(frame: Any, *, operation: str) -> None:
        model_name = message_model_name(frame)
        if model_name == NativeProtocolAbort.__name__:
            aborted = decode_message(frame, NativeProtocolAbort)
            raise RuntimeError(
                f"native coordinator rejected {operation}: "
                f"{aborted.error_code}: {aborted.detail}"
            )
        if model_name != NativeWaitResponse.__name__:
            raise RuntimeError(
                f"native coordinator returned unexpected {operation} response: "
                f"{model_name}"
            )
        decode_message(frame, NativeWaitResponse)

    @staticmethod
    def _recv_one(socket: Any, *, timeout_seconds: float) -> Any:
        message = NativeCollectionWorker._recv_multipart(
            socket,
            timeout_seconds=timeout_seconds,
        )
        if len(message) != 1:
            raise RuntimeError("native worker expected one response frame")
        return message[0]

    @staticmethod
    def _recv_multipart(socket: Any, *, timeout_seconds: float) -> list[Any]:
        timeout_ms = max(int(timeout_seconds * 1000), 1)
        if not socket.poll(timeout=timeout_ms, flags=zmq.POLLIN) & zmq.POLLIN:
            raise TimeoutError("native worker channel response timed out")
        return list(socket.recv_multipart(copy=False))

    @staticmethod
    def _repo_path(path: Path) -> Path:
        return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _lease_pfsp_members(
    lease: NativeCollectionShardLease,
) -> tuple[PfspMember, ...]:
    """Return only opponent-pool members referenced by this exact lease."""
    member_ids = {
        item.curriculum.member_id
        for item in lease.assignments
        if item.curriculum.member_id
    }
    members = {member.member_id: member for member in lease.window.pfsp_members}
    missing = member_ids - set(members)
    if missing:
        raise RuntimeError(
            "native lease references absent opponent-pool members: "
            + ", ".join(sorted(missing))
        )
    return tuple(
        member for member in lease.window.pfsp_members if member.member_id in member_ids
    )


def _legacy_resident_anchors(
    anchors: Sequence[StatelessHistoricalAnchorConfig],
) -> tuple[StatelessHistoricalAnchorConfig, ...]:
    """Keep only checkpoints owned by the legacy resident runtime.

    Fixed stateless anchors arrive through the native BF16 artifact channel and
    must not be opened by ``CheckpointPolicy`` during worker startup.
    """
    return tuple(
        anchor for anchor in anchors if anchor.runtime_kind == "legacy_resident"
    )


def _v2_historical_resources(
    bindings: Sequence[StatelessHistoricalAnchorConfig],
    *,
    repo_root: Path,
) -> dict[
    str,
    tuple[StatelessHistoricalAnchorConfig, tuple[int, ...]],
]:
    """Resolve and verify archive bindings independently of the active roster.

    V2 workers require the learner-specified files to be synchronized at the
    same repository-relative locations. Immutable content identities are
    verified locally; content-addressed transfer and caching remain future work.
    """
    resolved_root = repo_root.resolve()
    resources: dict[
        str,
        tuple[StatelessHistoricalAnchorConfig, tuple[int, ...]],
    ] = {}
    for binding in bindings:
        checkpoint_path = _resolve_v2_repo_file(
            binding.checkpoint_path,
            repo_root=resolved_root,
            label="checkpoint",
        )
        checkpoint_size = checkpoint_path.stat().st_size
        if checkpoint_size != binding.checkpoint_size_bytes:
            raise ValueError(
                "native historical checkpoint size mismatch: "
                f"{binding.member_id} expected={binding.checkpoint_size_bytes} "
                f"actual={checkpoint_size}"
            )
        if file_sha256(checkpoint_path) != binding.checkpoint_sha256:
            raise ValueError(
                f"native historical checkpoint SHA-256 mismatch: {binding.member_id}"
            )

        deck_path = _resolve_v2_repo_file(
            binding.exact_deck_path,
            repo_root=resolved_root,
            label="exact deck",
        )
        raw_deck = tuple(deck_records.read_deck(deck_path))
        if canonicalize_deck(raw_deck).deck_digest != binding.exact_deck_digest:
            raise ValueError(
                f"native historical exact-deck digest mismatch: {binding.member_id}"
            )

        belief_path = binding.belief_summary_path
        resolved_belief_path: Path | None = None
        if belief_path is not None:
            resolved_belief_path = _resolve_v2_repo_file(
                belief_path,
                repo_root=resolved_root,
                label="belief summary",
            )
            if file_sha256(resolved_belief_path) != binding.belief_summary_sha256:
                raise ValueError(
                    "native historical belief-summary SHA-256 mismatch: "
                    f"{binding.member_id}"
                )

        if binding.member_id in resources:
            raise ValueError("native historical binding member IDs must be unique")
        resolved_binding = binding.model_copy(
            update={
                "checkpoint_path": checkpoint_path,
                "exact_deck_path": deck_path,
                "belief_summary_path": resolved_belief_path,
            }
        )
        resources[binding.member_id] = (resolved_binding, raw_deck)
    return resources


def _resolve_v2_repo_file(
    path: Path,
    *,
    repo_root: Path,
    label: str,
) -> Path:
    """Resolve one synchronized V2 resource without allowing path escape."""
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"native historical {label} path must be repo-relative")
    resolved = (repo_root / path).resolve()
    if not resolved.is_relative_to(repo_root):
        runtime_root = repo_root / path.parts[0]
        allowed_runtime_root = (
            path.parts[0] in {"outputs", "tmp"}
            and runtime_root.is_symlink()
            and resolved.is_relative_to(runtime_root.resolve())
        )
        if not allowed_runtime_root:
            raise ValueError(f"native historical {label} path escapes the repository")
    if not resolved.is_file():
        raise ValueError(f"native historical {label} file is missing: {path}")
    return resolved


def _artifact_tensors(
    state: _ArtifactCacheState,
) -> Mapping[str, torch.Tensor]:
    """Return verified tensor views from a cached borrowed artifact."""
    if isinstance(state, Mapping):
        return state
    tensors = state.tensors
    if not isinstance(tensors, Mapping):
        raise TypeError("cached rollout artifact tensors are not a mapping")
    return tensors


def _release_artifact_state(state: _ArtifactCacheState) -> None:
    """Release borrowed ZeroMQ frames while preserving synthetic mappings."""
    release = getattr(state, "release", None)
    if callable(release):
        release()


def _state_nbytes(state: _ArtifactCacheState) -> int:
    """Return exact resident CPU tensor bytes for artifact-cache accounting."""
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in _artifact_tensors(state).values()
    )


def _memory_only_compact_parts(
    parts: Sequence[CompactFragmentPart],
) -> tuple[CompactFragmentPart, ...]:
    """Detach local process-IPC paths before ZeroMQ multipart transport."""
    return tuple(CompactFragmentPart(path=None, arrays=part.arrays) for part in parts)


def _distributed_collection_execution(
    config: SimpleStatelessTrainingConfig,
) -> tuple[int, int, SequenceRolloutPrecision]:
    """Project shared execution knobs into every distributed collection shard."""
    collection = config.collection
    return (
        collection.native_frozen_batch_min_rows,
        collection.native_frozen_batch_max_wait_waves,
        collection.native_sequence_rollout_precision,
    )


def run_native_collection_worker(
    config: SimpleStatelessTrainingConfig,
    *,
    worker_id: str,
    coordinator_host: str,
    worker_profile: str,
) -> None:
    """Run exactly one formal worker attempt and surface every failure.

    The launcher may replace the process only for the explicit
    ``NativeWorkerProcessRecycleError`` committed-boundary lifecycle signal.
    All other exits remain terminal so an operator can preserve evidence,
    diagnose the failure, and explicitly decide whether recovery is safe.
    """
    NativeCollectionWorker(
        config,
        worker_id=worker_id,
        coordinator_host=coordinator_host,
        worker_profile=worker_profile,
    ).run()
    raise RuntimeError("native collection worker exited unexpectedly")


__all__ = ["NativeCollectionWorker", "run_native_collection_worker"]
