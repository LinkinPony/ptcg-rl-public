"""Coordinator-side event loop for the three native ZMQ channels."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import zmq

from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload
from ptcg_rl.rl.native_distributed.artifact import (
    EncodedBfloat16RolloutArtifact,
)
from ptcg_rl.rl.native_distributed.contracts import (
    NativeAssignedGame,
    NativeCollectionAttempt,
    NativeCollectionCapacityTier,
    NativeCollectionShardLease,
    NativeCollectionWindow,
    NativeCollectionWindowReceipt,
    NativeCollectionWorkerManifest,
    native_assignment_plan_revision,
    native_required_artifact_ids,
)
from ptcg_rl.rl.native_distributed.coordinator import (
    CommitCallback,
    NativeAttemptCompletionReservation,
    NativeCollectionCoordinator,
    NativeCoordinatorError,
    NativeCoordinatorProtocolError,
    NativePreparedAttemptCompletion,
)
from ptcg_rl.rl.native_distributed.data_plane import (
    decode_native_collection_part,
    peek_native_collection_part_identity,
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
from ptcg_rl.rl.native_distributed.quota_scheduler import NativeQuotaScheduler
from ptcg_rl.rl.native_distributed.transport import (
    NativeCoordinatorSockets,
    recv_router_multipart,
    try_send_router_multipart,
)
from ptcg_rl.rl.stateless_collection import (
    StatelessAssignedGame,
    StatelessCollectionResult,
)
from ptcg_rl.rl.stateless_hybrid import merge_hybrid_collection_results
from ptcg_rl.rl.stateless_quota_assignments import StatelessQuotaAssignmentPlan

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _V2ShardTemplate:
    """One final precomputed assignment shard with immutable membership."""

    sequence_id: int
    assignment_indices: tuple[int, ...]
    required_artifact_ids: tuple[str, ...]


class NativeCoordinatorService:
    """Pump worker requests while one core owns all mutable window state."""

    def __init__(
        self,
        coordinator: NativeCollectionCoordinator,
        sockets: NativeCoordinatorSockets,
        *,
        control_poll_interval_seconds: float,
        status_path: Path | None = None,
        status_interval_seconds: float = 5.0,
    ) -> None:
        """Bind the state machine to sockets without opening a window."""
        if control_poll_interval_seconds <= 0.0 or status_interval_seconds <= 0.0:
            raise ValueError("native coordinator service intervals must be positive")
        self.coordinator = coordinator
        self.sockets = sockets
        self.control_poll_interval_seconds = control_poll_interval_seconds
        self.status_path = status_path
        self.status_interval_seconds = status_interval_seconds
        self._artifacts: dict[str, EncodedBfloat16RolloutArtifact] = {}
        self._assignments: tuple[StatelessAssignedGame, ...] = ()
        self._assignment_cursor = 0
        self._issued_assignment_indices: set[int] = set()
        self._v2_shard_templates: tuple[_V2ShardTemplate, ...] = ()
        self._issued_v2_template_ids: set[int] = set()
        self._v2_requirements: tuple[tuple[str, ...], ...] = ()
        self._available_v2_assignment_indices: set[int] = set()
        self._next_v2_shard_sequence = 0
        self._v3_initial_wave_workers: set[str] = set()
        self._quota_plan: StatelessQuotaAssignmentPlan | None = None
        self._quota_scheduler: NativeQuotaScheduler | None = None
        self.last_assignment_planning_seconds = 0.0
        self._worker_control_routes: dict[str, bytes] = {}
        self._worker_metrics: dict[str, NativeWorkerMetrics] = {}
        self._receipt_delivered: set[str] = set()
        self._router_delivery_failures = {
            "control": 0,
            "artifact": 0,
            "data": 0,
        }
        self._last_status_at = 0.0
        self._window_started_at = 0.0
        self._completion_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="native-terminal-projection",
        )
        self._completion_futures: dict[
            Future[NativePreparedAttemptCompletion],
            tuple[NativeAttemptCompletionReservation, float],
        ] = {}
        self._poller = zmq.Poller()
        self._poller.register(self.sockets.control, zmq.POLLIN)
        self._poller.register(self.sockets.artifact, zmq.POLLIN)
        self._poller.register(self.sockets.data, zmq.POLLIN)

    def begin_window(
        self,
        window: NativeCollectionWindow,
        *,
        artifacts: Sequence[EncodedBfloat16RolloutArtifact],
        assignment_pool: Sequence[StatelessAssignedGame] | StatelessQuotaAssignmentPlan,
        now_unix_ns: int | None = None,
        learner_clocked_primary_only: bool = False,
    ) -> NativeCollectionWindow:
        """Freeze topology, artifacts, and a controller-issued assignment pool."""
        by_id = {artifact.manifest.artifact_id: artifact for artifact in artifacts}
        if len(by_id) != len(artifacts) or tuple(by_id) != tuple(
            artifact.manifest.artifact_id for artifact in artifacts
        ):
            raise ValueError("native window artifacts must be ordered and unique")
        expected = {item.artifact_id for item in window.active_artifacts}
        if set(by_id) != expected:
            raise ValueError("native window artifact bytes differ from manifests")
        if isinstance(assignment_pool, StatelessQuotaAssignmentPlan):
            quota_plan: StatelessQuotaAssignmentPlan | None = assignment_pool
            assignments: tuple[StatelessAssignedGame, ...] = ()
        else:
            quota_plan = None
            assignments = tuple(assignment_pool)
        if window.shard_protocol_version == 3:
            if quota_plan is None:
                raise ValueError("native V3 window requires an aggregate quota plan")
        elif quota_plan is not None:
            raise ValueError("aggregate quota plans require native V3")
        elif not assignments:
            raise ValueError("native distributed assignment pool is empty")
        wire_assignments = tuple(
            NativeAssignedGame.from_assignment(item) for item in assignments
        )
        planning_started_at = time.perf_counter()
        requirements: tuple[tuple[str, ...], ...] = ()
        if window.shard_protocol_version == 2:
            if window.assignment_plan_revision != native_assignment_plan_revision(
                wire_assignments
            ):
                raise ValueError(
                    "native assignment pool differs from its plan revision"
                )
            requirements = tuple(
                native_required_artifact_ids(window, (item,))
                for item in wire_assignments
            )
        elif window.shard_protocol_version == 3:
            if (
                quota_plan is None
                or window.assignment_plan_revision != quota_plan.revision
            ):
                raise ValueError("native quota plan differs from its window revision")
        now = time.time_ns() if now_unix_ns is None else now_unix_ns
        self.coordinator.begin_window(
            window,
            now_unix_ns=now,
            learner_clocked_primary_only=learner_clocked_primary_only,
        )
        active = self.coordinator.window
        if active is None:
            raise RuntimeError("native coordinator lost its opened window")
        quota_scheduler: NativeQuotaScheduler | None = None
        if active.shard_protocol_version in {2, 3}:
            try:
                manifests = self.coordinator.active_worker_manifests()
                template_games, frozen_limit = _common_v2_template_geometry(manifests)
                if active.shard_protocol_version == 2:
                    available = set(range(len(requirements)))
                    required_artifacts = {
                        artifact_id
                        for required in requirements
                        for artifact_id in required[1:]
                    }
                    probes: set[str | None] = (
                        set(required_artifacts) if required_artifacts else {None}
                    )
                    for artifact_id in probes:
                        if not _artifact_coherent_indices(
                            requirements,
                            available,
                            concurrent_games=template_games,
                            frozen_artifact_limit=frozen_limit,
                            required_artifact_id=artifact_id,
                        ):
                            raise ValueError(
                                "native V2 assignment pool cannot fill an "
                                "artifact-coherent shard for "
                                f"{artifact_id or 'current'}"
                            )
                else:
                    if quota_plan is None:
                        raise AssertionError("V3 quota plan disappeared")
                    quota_frozen_limit = _v3_frozen_artifact_limit(frozen_limit)
                    quota_scheduler = NativeQuotaScheduler(
                        tuple(row.game_count for row in quota_plan.rows),
                        tuple(
                            row.frozen_artifact_id or None for row in quota_plan.rows
                        ),
                        cohort_artifact_ids=tuple(
                            row.scheduler_cohort_id or None for row in quota_plan.rows
                        ),
                    )
                    probes = (
                        set(quota_plan.artifact_ids)
                        if quota_plan.artifact_ids
                        else {None}
                    )
                    exposure_games = _v3_exposure_games(
                        manifests,
                        planned_games=quota_plan.exposure_cohort_games,
                    )
                    for artifact_id in probes:
                        if not quota_scheduler.can_take(
                            exposure_games,
                            frozen_artifact_limit=quota_frozen_limit,
                            required_artifact_id=artifact_id,
                        ):
                            raise ValueError(
                                "native V3 quota plan cannot fill an "
                                "artifact-coherent shard for "
                                f"{artifact_id or 'current'}"
                            )
            except BaseException as exc:
                if quota_scheduler is not None:
                    quota_scheduler.close()
                self.coordinator.abort(
                    reason=(
                        f"native shard planning failed: {type(exc).__name__}: {exc}"
                    ),
                    now_unix_ns=now,
                )
                raise
        if self._quota_scheduler is not None:
            self._quota_scheduler.close()
        self._artifacts = by_id
        self._assignments = assignments
        self._assignment_cursor = 0
        self._issued_assignment_indices.clear()
        self._v2_shard_templates = ()
        self._issued_v2_template_ids.clear()
        self._v2_requirements = requirements
        self._available_v2_assignment_indices = set(range(len(requirements)))
        self._next_v2_shard_sequence = 0
        self._v3_initial_wave_workers.clear()
        self._quota_plan = quota_plan
        self._quota_scheduler = quota_scheduler
        self.last_assignment_planning_seconds = (
            time.perf_counter() - planning_started_at
        )
        self._window_started_at = time.perf_counter()
        self._receipt_delivered.clear()
        self._publish_status(now_unix_ns=now, force=True)
        return active

    def wait_for_workers(
        self,
        worker_ids: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> None:
        """Pump registration/READY messages until the requested quorum exists."""
        required = set(worker_ids)
        if not required or timeout_seconds <= 0.0:
            raise ValueError("native worker wait requires IDs and a timeout")
        deadline = time.monotonic() + timeout_seconds
        while True:
            connected = set(
                self.coordinator.status(now_unix_ns=time.time_ns()).connected_workers
            )
            if required <= connected:
                return
            if time.monotonic() >= deadline:
                missing = ", ".join(sorted(required - connected))
                raise TimeoutError(
                    f"native distributed worker READY timeout: {missing}"
                )
            self.serve_once()

    def wait_for_worker_quorum(
        self,
        worker_ids: Sequence[str],
        *,
        minimum_workers: int,
        timeout_seconds: float,
    ) -> None:
        """Pump messages until any configured minimum is READY."""
        eligible = set(worker_ids)
        if (
            not eligible
            or minimum_workers <= 0
            or minimum_workers > len(eligible)
            or timeout_seconds <= 0.0
        ):
            raise ValueError("native worker quorum wait is invalid")
        deadline = time.monotonic() + timeout_seconds
        while True:
            connected = set(
                self.coordinator.status(now_unix_ns=time.time_ns()).connected_workers
            )
            if len(eligible & connected) >= minimum_workers:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "native distributed minimum worker READY quorum timed out: "
                    f"ready={len(eligible & connected)} required={minimum_workers}"
                )
            self.serve_once()

    def serve_until_complete(
        self,
        *,
        timeout_seconds: float,
    ) -> None:
        """Serve channels until accepted evidence reaches the global target."""
        if timeout_seconds <= 0.0:
            raise ValueError("native collection service timeout must be positive")
        deadline = time.monotonic() + timeout_seconds
        while not self.coordinator.ready_to_commit():
            if self.coordinator.exhausted_lease_ids():
                raise NativeCoordinatorError(
                    "native collection shard exhausted its retry budget: "
                    + ", ".join(self.coordinator.exhausted_lease_diagnostics())
                )
            missing_exposure = self.coordinator.missing_v2_artifact_exposure_ids()
            assignments_available = (
                self._quota_scheduler is not None
                and self._quota_scheduler.remaining > 0
            ) or bool(self._available_v2_assignment_indices)
            if (
                missing_exposure
                and not assignments_available
                and not self.coordinator.has_unsettled_shards()
            ):
                raise NativeCoordinatorError(
                    "native V2 templates exhausted before required artifact "
                    "exposure: " + ", ".join(missing_exposure)
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("native distributed collection window timed out")
            self.serve_once()
        self._publish_status(now_unix_ns=time.time_ns(), force=True)

    def serve_once(self) -> None:
        """Process at most one ready message from each independent channel."""
        self._drain_completion_futures()
        timeout_ms = max(int(self.control_poll_interval_seconds * 1000), 1)
        events = dict(self._poller.poll(timeout=timeout_ms))
        if events.get(self.sockets.control, 0) & zmq.POLLIN:
            self._serve_control()
        if events.get(self.sockets.artifact, 0) & zmq.POLLIN:
            self._serve_artifact()
        if events.get(self.sockets.data, 0) & zmq.POLLIN:
            self._serve_data()
        self._drain_completion_futures()
        self._publish_status(now_unix_ns=time.time_ns())

    def commit(
        self,
        callback: CommitCallback,
        *,
        now_unix_ns: int | None = None,
    ) -> NativeCollectionWindowReceipt:
        """Commit controller outcomes once and publish the final receipt."""
        now = time.time_ns() if now_unix_ns is None else now_unix_ns
        receipt = self.coordinator.commit(callback, now_unix_ns=now)
        self._publish_status(now_unix_ns=now, force=True)
        return receipt

    def deliver_receipt(
        self,
        *,
        timeout_seconds: float,
    ) -> None:
        """Pump control until every topology worker has observed settlement."""
        receipt = self.coordinator.receipt
        window = self.coordinator.window
        if receipt is None or window is None:
            raise NativeCoordinatorError(
                "native window receipt is unavailable for delivery"
            )
        deadline = time.monotonic() + timeout_seconds
        while True:
            required = set(
                self.coordinator.receipt_recipient_worker_ids(
                    now_unix_ns=time.time_ns()
                )
            )
            if required <= self._receipt_delivered:
                return
            if time.monotonic() >= deadline:
                missing = ", ".join(sorted(required - self._receipt_delivered))
                raise TimeoutError(
                    f"native window receipt delivery timed out: {missing}"
                )
            self.serve_once()

    def receipt_delivery_complete(self) -> bool:
        """Return whether every live frozen-topology session saw settlement."""
        receipt = self.coordinator.receipt
        window = self.coordinator.window
        return (
            receipt is not None
            and window is not None
            and set(
                self.coordinator.receipt_recipient_worker_ids(
                    now_unix_ns=time.time_ns()
                )
            )
            <= self._receipt_delivered
        )

    def abort(
        self,
        *,
        reason: str,
        now_unix_ns: int | None = None,
    ) -> NativeCollectionWindowReceipt:
        """Abort and release one half-window after any coordinator failure."""
        now = time.time_ns() if now_unix_ns is None else now_unix_ns
        receipt = self.coordinator.abort(reason=reason, now_unix_ns=now)
        self._publish_status(now_unix_ns=now, force=True)
        return receipt

    def close(self) -> None:
        """Finish terminal projections and release socket-thread resources."""
        self._completion_executor.shutdown(wait=True, cancel_futures=False)
        self._drain_completion_futures()
        if self._quota_scheduler is not None:
            self._quota_scheduler.close()
            self._quota_scheduler = None

    def unused_assignments(self) -> tuple[StatelessAssignedGame, ...]:
        """Return adopted controller leases absent from accepted execution."""
        quota_plan = getattr(self, "_quota_plan", None)
        if quota_plan is not None:
            # V3 reservations attach only after their accepted started subset
            # is known, so an unstarted reservation needs no controller cancel.
            return ()
        accepted = self.coordinator.accepted_assignment_ids()
        assignments = (
            quota_plan.issued_assignments
            if quota_plan is not None
            else self._assignments
        )
        return tuple(
            assignment
            for assignment in assignments
            if assignment.curriculum.assignment_id not in accepted
        )

    def accepted_assignment_ids(self) -> frozenset[str]:
        """Return accepted reservation identities that reached engine start."""
        return self.coordinator.accepted_assignment_ids()

    def collection_result(self) -> StatelessCollectionResult:
        """Merge complete remote shards in central sequence/part order."""
        completed = self.coordinator.completed_shards()
        assignments, _outcomes = self.coordinator.ordered_assignments_and_outcomes()
        shards_list: list[StatelessCollectionResult] = []
        for lease, result, parts in completed:
            started_ids = {
                outcome.curriculum_assignment_id for outcome in result.outcomes
            }
            shards_list.append(
                StatelessCollectionResult(
                    fragments=(),
                    report=result.report.model_copy(
                        update={
                            "native_trainable_decision_budget": None,
                            "native_trainable_decision_budget_reached": False,
                            "native_trainable_decision_budget_overshoot": 0,
                        }
                    ),
                    assignments=tuple(
                        item.to_assignment()
                        for item in lease.assignments
                        if item.curriculum.assignment_id in started_ids
                    ),
                    outcomes=result.outcomes,
                    compact_parts=tuple(item.part for item in parts),
                )
            )
        shards = tuple(shards_list)
        elapsed = max(time.perf_counter() - self._window_started_at, 1.0e-9)
        merged = merge_hybrid_collection_results(
            shards,
            assignments=assignments,
            elapsed_seconds=elapsed,
            native_phase_seconds=elapsed,
        )
        window = self.coordinator.window
        if window is None:
            raise NativeCoordinatorError("native window identity disappeared")
        accepted = sum(part.decision_count for part in merged.compact_parts)
        return replace(
            merged,
            report=merged.report.model_copy(
                update={
                    "native_trainable_decision_budget": (
                        window.target_trainable_decisions
                    ),
                    "native_trainable_decision_budget_reached": (
                        accepted >= window.target_trainable_decisions
                    ),
                    "native_trainable_decision_budget_overshoot": max(
                        accepted - window.target_trainable_decisions,
                        0,
                    ),
                }
            ),
        )

    def _serve_control(self) -> None:
        routing_id, frames = recv_router_multipart(
            self.sockets.control,
            copy=False,
        )
        if len(frames) != 1:
            self._send_abort(
                self.sockets.control,
                routing_id,
                error_code="DECODE_ERROR",
                detail="control messages require exactly one frame",
            )
            return
        frame = frames[0]
        try:
            received_at_unix_ns = time.time_ns()
            self.coordinator.expire_workers(now_unix_ns=received_at_unix_ns)
            model_name = message_model_name(frame)
            if model_name == NativeRegisterRequest.__name__:
                register = decode_message(frame, NativeRegisterRequest)
                self.coordinator.register_worker(
                    register.manifest,
                    now_unix_ns=received_at_unix_ns,
                )
                self._worker_control_routes[register.manifest.identity.worker_id] = (
                    routing_id
                )
                response: Any = NativeWaitResponse(
                    reason="registered",
                    retry_after_seconds=self.control_poll_interval_seconds,
                )
            elif model_name == NativeReadyRequest.__name__:
                ready = decode_message(frame, NativeReadyRequest)
                self._require_control_route(ready.worker_id, routing_id)
                self.coordinator.mark_ready(
                    ready.worker_id,
                    session_id=ready.session_id,
                    now_unix_ns=received_at_unix_ns,
                )
                response = NativeWaitResponse(
                    reason="ready",
                    retry_after_seconds=self.control_poll_interval_seconds,
                )
            elif model_name == NativeHeartbeatRequest.__name__:
                heartbeat = decode_message(frame, NativeHeartbeatRequest)
                self._require_control_route(heartbeat.worker_id, routing_id)
                self.coordinator.heartbeat(
                    heartbeat.worker_id,
                    session_id=heartbeat.session_id,
                    now_unix_ns=received_at_unix_ns,
                    active_attempt_id=heartbeat.metrics.active_attempt_id,
                )
                self._worker_metrics[heartbeat.worker_id] = heartbeat.metrics
                receipt = self.coordinator.receipt
                discarded_window_id = self.coordinator.discarded_attempt_window_id(
                    heartbeat.metrics.active_attempt_id,
                    worker_id=heartbeat.worker_id,
                    session_id=heartbeat.session_id,
                )
                if receipt is not None and (
                    heartbeat.worker_id not in self._receipt_delivered
                    or discarded_window_id is not None
                ):
                    self._receipt_delivered.add(heartbeat.worker_id)
                    response = NativeWindowReceiptMessage(receipt=receipt)
                elif discarded_window_id is not None:
                    response = NativeDrainResponse(window_id=discarded_window_id)
                else:
                    response = NativeWaitResponse(
                        reason="heartbeat accepted",
                        retry_after_seconds=self.control_poll_interval_seconds,
                        drain_hint=(
                            heartbeat.metrics.active_attempt_id is not None
                            and self.coordinator.window_drain_hint()
                        ),
                    )
            elif model_name == NativeWorkRequest.__name__:
                work = decode_message(frame, NativeWorkRequest)
                self._require_control_route(work.worker_id, routing_id)
                self.coordinator.heartbeat(
                    work.worker_id,
                    session_id=work.session_id,
                    now_unix_ns=received_at_unix_ns,
                )
                self.coordinator.acknowledge_worker_discards(
                    work.worker_id,
                    session_id=work.session_id,
                )
                response = self._work_response_or_settled(
                    work.worker_id,
                    session_id=work.session_id,
                    now_unix_ns=received_at_unix_ns,
                )
            elif model_name == NativeAttemptFailedRequest.__name__:
                failure = decode_message(frame, NativeAttemptFailedRequest)
                self._require_control_route(failure.worker_id, routing_id)
                discarded = self.coordinator.discarded_attempt_context(
                    failure.lease_id,
                    failure.attempt_id,
                )
                if discarded is not None:
                    self._require_discarded_attempt_owner(
                        discarded,
                        worker_id=failure.worker_id,
                        session_id=failure.session_id,
                    )
                    response = NativeWaitResponse(
                        reason="discarded speculative tail acknowledged",
                        retry_after_seconds=self.control_poll_interval_seconds,
                    )
                else:
                    self.coordinator.fail_attempt(
                        failure.attempt_id,
                        reason=failure.reason,
                    )
                    response = NativeWaitResponse(
                        reason="attempt released for retry",
                        retry_after_seconds=self.control_poll_interval_seconds,
                    )
            elif model_name == NativeShardCompleteRequest.__name__:
                completion = decode_message(frame, NativeShardCompleteRequest)
                self._require_control_route(completion.worker_id, routing_id)
                discarded = self.coordinator.discarded_attempt_context(
                    completion.result.lease_id,
                    completion.result.attempt_id,
                )
                if discarded is not None:
                    self._require_discarded_attempt_owner(
                        discarded,
                        worker_id=completion.worker_id,
                        session_id=completion.session_id,
                    )
                    response = NativeWaitResponse(
                        reason="discarded speculative tail acknowledged",
                        retry_after_seconds=self.control_poll_interval_seconds,
                    )
                else:
                    self.coordinator.heartbeat(
                        completion.worker_id,
                        session_id=completion.session_id,
                        now_unix_ns=received_at_unix_ns,
                        active_attempt_id=completion.result.attempt_id,
                    )
                    reservation = self.coordinator.reserve_attempt_completion(
                        completion.result
                    )
                    future = self._completion_executor.submit(
                        self.coordinator.project_attempt_completion,
                        reservation,
                    )
                    self._completion_futures[future] = (
                        reservation,
                        time.perf_counter(),
                    )
                    response = NativeWaitResponse(
                        reason="shard completion queued",
                        retry_after_seconds=self.control_poll_interval_seconds,
                    )
            else:
                raise NativeCoordinatorProtocolError(
                    f"unsupported control message model: {model_name}"
                )
        except Exception as exc:
            self._send_abort(
                self.sockets.control,
                routing_id,
                error_code="IDENTITY_MISMATCH",
                detail=str(exc),
            )
            return
        self._send_router_response(
            self.sockets.control,
            routing_id,
            (encode_message(response),),
            channel="control",
            copy=True,
        )

    def _drain_completion_futures(self) -> None:
        """Finalize ready terminal projections without blocking socket polling."""
        for future, (reservation, started_at) in tuple(
            self._completion_futures.items()
        ):
            if not future.done():
                continue
            del self._completion_futures[future]
            elapsed = time.perf_counter() - started_at
            try:
                prepared = future.result()
                accepted = self.coordinator.finalize_attempt_completion(prepared)
            except Exception as exc:
                reason = f"completion projection failed: {type(exc).__name__}: {exc}"
                try:
                    self.coordinator.reject_attempt_completion(
                        reservation,
                        reason=reason,
                    )
                except Exception:
                    _LOGGER.exception(
                        "native coordinator could not reject failed completion: "
                        "attempt=%s",
                        reservation.result.attempt_id,
                    )
                _LOGGER.exception(
                    "native coordinator terminal projection failed: attempt=%s "
                    "seconds=%.3f",
                    reservation.result.attempt_id,
                    elapsed,
                )
                continue
            if not accepted:
                _LOGGER.info(
                    "native coordinator discarded deferred speculative-tail "
                    "projection: attempt=%s seconds=%.3f",
                    reservation.result.attempt_id,
                    elapsed,
                )
                continue
            _LOGGER.info(
                "native coordinator completed deferred terminal projection: "
                "attempt=%s parts=%d decisions=%d seconds=%.3f",
                reservation.result.attempt_id,
                len(prepared.retained_parts),
                reservation.result.decision_count,
                elapsed,
            )

    def _work_response_or_settled(
        self,
        worker_id: str,
        *,
        session_id: str,
        now_unix_ns: int,
    ) -> Any:
        """Resolve work while tolerating a concurrent learner commit."""
        try:
            return self._work_response(
                worker_id,
                session_id=session_id,
                now_unix_ns=now_unix_ns,
            )
        except NativeCoordinatorError:
            # The learner commits from a different thread. It can settle the
            # window between _work_response's receipt check and a
            # collecting-only lease accessor. Deliver the new receipt instead
            # of turning this benign boundary race into an identity-mismatch
            # abort that kills the worker.
            receipt = self.coordinator.receipt
            if receipt is None:
                raise
            return self._settled_work_response(worker_id, receipt)

    def _work_response(
        self,
        worker_id: str,
        *,
        session_id: str,
        now_unix_ns: int,
    ) -> Any:
        receipt = self.coordinator.receipt
        if receipt is not None:
            return self._settled_work_response(worker_id, receipt)
        window = self.coordinator.window
        if window is None:
            return NativeWaitResponse(
                reason="no open collection window",
                retry_after_seconds=self.control_poll_interval_seconds,
            )
        if not self.coordinator.session_is_in_active_topology(
            worker_id,
            session_id=session_id,
        ):
            return NativeWaitResponse(
                reason="worker session is waiting for the next topology boundary",
                retry_after_seconds=self.control_poll_interval_seconds,
            )
        retry = self.coordinator.retryable_lease(worker_id)
        if retry is not None:
            attempt = self.coordinator.retry_shard(
                retry.lease_id,
                worker_id,
                now_unix_ns=now_unix_ns,
            )
            return NativeLeaseResponse(lease=retry, attempt=attempt)
        if window.shard_protocol_version == 3:
            return self._v3_work_response(
                worker_id,
                window=window,
                now_unix_ns=now_unix_ns,
            )
        if window.shard_protocol_version == 2:
            return self._v2_work_response(
                worker_id,
                window=window,
                now_unix_ns=now_unix_ns,
            )
        tier = self.coordinator.select_capacity_tier(
            worker_id,
            maximum_games=len(self._assignments) - self._assignment_cursor,
        )
        if tier is None:
            return NativeDrainResponse(window_id=window.identity.window_id)
        stop = self._assignment_cursor + tier.concurrent_games
        if stop > len(self._assignments):
            raise NativeCoordinatorError(
                "native distributed assignment pool cannot fill selected tier"
            )
        assignments = self._assignments[self._assignment_cursor : stop]
        self._assignment_cursor = stop
        first_cursor = assignments[0].balance.assignment_cursor
        lease, attempt = self.coordinator.issue_shard(
            worker_id,
            assignments,
            capacity_tier_id=tier.tier_id,
            shard_seed=window.identity.sequence_id + first_cursor,
            now_unix_ns=now_unix_ns,
            estimated_decision_credit=tier.estimated_trainable_decisions,
        )
        return NativeLeaseResponse(lease=lease, attempt=attempt)

    def _settled_work_response(
        self,
        worker_id: str,
        receipt: NativeCollectionWindowReceipt,
    ) -> NativeWaitResponse | NativeWindowReceiptMessage:
        """Deliver one terminal receipt, then keep the worker boundary-idle."""
        if worker_id in self._receipt_delivered:
            return NativeWaitResponse(
                reason="window settled; waiting for the next window",
                retry_after_seconds=self.control_poll_interval_seconds,
            )
        self._receipt_delivered.add(worker_id)
        return NativeWindowReceiptMessage(receipt=receipt)

    def _v3_work_response(
        self,
        worker_id: str,
        *,
        window: NativeCollectionWindow,
        now_unix_ns: int,
    ) -> Any:
        """Materialize one artifact-coherent shard from aggregate quotas."""
        plan = self._quota_plan
        scheduler = self._quota_scheduler
        if plan is None or scheduler is None:
            raise NativeCoordinatorError("native V3 quota scheduler is absent")
        tier = self.coordinator.select_capacity_tier(
            worker_id,
            maximum_games=scheduler.remaining,
        )
        if tier is None:
            return NativeDrainResponse(window_id=window.identity.window_id)
        missing = self.coordinator.missing_v2_artifact_exposure_ids(
            include_unsettled=True
        )
        manifest = next(
            item
            for item in self.coordinator.active_worker_manifests()
            if item.identity.worker_id == worker_id
        )
        required_artifact_id = None if not missing else missing[0]
        initial_wave_workers: set[str] = getattr(
            self,
            "_v3_initial_wave_workers",
            set(),
        )
        first_worker_shard = worker_id not in initial_wave_workers
        exposure_wave = bool(window.active_pfsp_artifact_ids) and (
            required_artifact_id is not None or first_worker_shard
        )
        candidate_tiers = _v3_candidate_capacity_tiers(
            manifest,
            tier,
            remaining_games=scheduler.remaining,
            exposure_games=(plan.exposure_cohort_games if exposure_wave else None),
            primary_only=self.coordinator.primary_only_during_window(),
        )
        selected_tier = None
        for candidate in candidate_tiers:
            if scheduler.can_take(
                candidate.concurrent_games,
                frozen_artifact_limit=_v3_frozen_artifact_limit(
                    candidate.native_policy_group_bank_limit
                ),
                required_artifact_id=required_artifact_id,
            ):
                selected_tier = candidate
                break
        if selected_tier is None and required_artifact_id is not None:
            for candidate in candidate_tiers:
                if scheduler.can_take(
                    candidate.concurrent_games,
                    frozen_artifact_limit=_v3_frozen_artifact_limit(
                        candidate.native_policy_group_bank_limit
                    ),
                ):
                    selected_tier = candidate
                    required_artifact_id = None
                    break
        if selected_tier is None:
            return NativeDrainResponse(window_id=window.identity.window_id)
        row_indices = scheduler.take(
            selected_tier.concurrent_games,
            frozen_artifact_limit=_v3_frozen_artifact_limit(
                selected_tier.native_policy_group_bank_limit
            ),
            required_artifact_id=required_artifact_id,
        )
        assignments = plan.materialize(row_indices)
        wire_assignments = tuple(
            NativeAssignedGame.from_assignment(item) for item in assignments
        )
        required_artifact_ids = native_required_artifact_ids(
            window,
            wire_assignments,
        )
        first_cursor = assignments[0].balance.assignment_cursor
        lease, attempt = self.coordinator.issue_shard(
            worker_id,
            assignments,
            capacity_tier_id=selected_tier.tier_id,
            shard_seed=window.identity.sequence_id + first_cursor,
            now_unix_ns=now_unix_ns,
            required_artifact_ids=required_artifact_ids,
            shard_sequence_id=self._next_v2_shard_sequence,
            estimated_decision_credit=(selected_tier.estimated_trainable_decisions),
        )
        self._next_v2_shard_sequence += 1
        self._assignment_cursor += len(assignments)
        if exposure_wave and first_worker_shard:
            initial_wave_workers.add(worker_id)
            self._v3_initial_wave_workers = initial_wave_workers
        return NativeLeaseResponse(lease=lease, attempt=attempt)

    def _v2_work_response(
        self,
        worker_id: str,
        *,
        window: NativeCollectionWindow,
        now_unix_ns: int,
    ) -> Any:
        """Issue one deterministic artifact-coherent V2 assignment shard."""
        # Compatibility for focused protocol tests that inject an immutable
        # template directly. Production windows populate the lazy requirement
        # table instead and reserve one shard only when a worker asks for it.
        if self._v2_shard_templates:
            return self._legacy_v2_work_response(
                worker_id,
                window=window,
                now_unix_ns=now_unix_ns,
            )
        available = self._available_v2_assignment_indices
        tier = self.coordinator.select_capacity_tier(
            worker_id,
            maximum_games=len(available),
        )
        if tier is None:
            return NativeDrainResponse(window_id=window.identity.window_id)
        missing = self.coordinator.missing_v2_artifact_exposure_ids(
            include_unsettled=True
        )
        manifest = next(
            item
            for item in self.coordinator.active_worker_manifests()
            if item.identity.worker_id == worker_id
        )
        candidate_tiers: tuple[NativeCollectionCapacityTier, ...] = (tier,)
        if not self.coordinator.primary_only_during_window():
            candidate_tiers = (
                tier,
                *(
                    item
                    for item in sorted(
                        manifest.capacity_tiers,
                        key=lambda value: (
                            -value.estimated_trainable_decisions,
                            value.tier_id,
                        ),
                    )
                    if item.tier_id != tier.tier_id
                    and item.concurrent_games <= tier.concurrent_games
                    and item.concurrent_games <= len(available)
                ),
            )
        indices: tuple[int, ...] = ()
        selected_tier = tier
        for candidate in candidate_tiers:
            indices = _artifact_coherent_indices(
                self._v2_requirements,
                available,
                concurrent_games=candidate.concurrent_games,
                frozen_artifact_limit=(candidate.native_policy_group_bank_limit),
                required_artifact_id=(None if not missing else missing[0]),
            )
            if not indices and missing:
                indices = _artifact_coherent_indices(
                    self._v2_requirements,
                    available,
                    concurrent_games=candidate.concurrent_games,
                    frozen_artifact_limit=(candidate.native_policy_group_bank_limit),
                )
            if indices:
                selected_tier = candidate
                break
        if not indices:
            return NativeDrainResponse(window_id=window.identity.window_id)
        required_artifact_ids = _assignment_artifact_union(
            self._v2_requirements,
            indices,
        )
        assignments = tuple(self._assignments[index] for index in indices)
        first_cursor = assignments[0].balance.assignment_cursor
        lease, attempt = self.coordinator.issue_shard(
            worker_id,
            assignments,
            capacity_tier_id=selected_tier.tier_id,
            shard_seed=window.identity.sequence_id + first_cursor,
            now_unix_ns=now_unix_ns,
            required_artifact_ids=required_artifact_ids,
            shard_sequence_id=self._next_v2_shard_sequence,
            estimated_decision_credit=(selected_tier.estimated_trainable_decisions),
        )
        self._issued_assignment_indices.update(indices)
        available.difference_update(indices)
        self._next_v2_shard_sequence += 1
        self._assignment_cursor = len(self._issued_assignment_indices)
        return NativeLeaseResponse(lease=lease, attempt=attempt)

    def _legacy_v2_work_response(
        self,
        worker_id: str,
        *,
        window: NativeCollectionWindow,
        now_unix_ns: int,
    ) -> Any:
        """Execute an explicitly injected immutable V2 template."""
        available_template_ids = set(range(len(self._v2_shard_templates))) - (
            self._issued_v2_template_ids
        )
        if not available_template_ids:
            return NativeDrainResponse(window_id=window.identity.window_id)
        template_id = min(available_template_ids)
        template = self._v2_shard_templates[template_id]
        tier = self.coordinator.select_capacity_tier(
            worker_id,
            exact_games=len(template.assignment_indices),
        )
        if tier is None:
            return NativeDrainResponse(window_id=window.identity.window_id)
        assignments = tuple(
            self._assignments[index] for index in template.assignment_indices
        )
        first_cursor = assignments[0].balance.assignment_cursor
        lease, attempt = self.coordinator.issue_shard(
            worker_id,
            assignments,
            capacity_tier_id=tier.tier_id,
            shard_seed=window.identity.sequence_id + first_cursor,
            now_unix_ns=now_unix_ns,
            required_artifact_ids=template.required_artifact_ids,
            shard_sequence_id=template.sequence_id,
            estimated_decision_credit=tier.estimated_trainable_decisions,
        )
        self._issued_assignment_indices.update(template.assignment_indices)
        self._issued_v2_template_ids.add(template_id)
        self._assignment_cursor = len(self._issued_assignment_indices)
        return NativeLeaseResponse(lease=lease, attempt=attempt)

    def _serve_artifact(self) -> None:
        routing_id, frames = recv_router_multipart(
            self.sockets.artifact,
            copy=False,
        )
        if len(frames) != 1:
            self._send_abort(
                self.sockets.artifact,
                routing_id,
                error_code="DECODE_ERROR",
                detail="artifact requests require exactly one frame",
            )
            return
        try:
            request = decode_message(frames[0], NativeArtifactRequest)
            self._require_channel_route(
                request.worker_id,
                request.session_id,
                "artifact",
                routing_id,
            )
            window = self.coordinator.window
            if (
                window is None
                or request.window_id != window.identity.window_id
                or request.expected_manifest not in window.active_artifacts
            ):
                raise NativeCoordinatorProtocolError(
                    "artifact request crossed the active window"
                )
            artifact = self._artifacts[request.artifact_id]
            if artifact.manifest != request.expected_manifest:
                raise NativeCoordinatorProtocolError(
                    "artifact request manifest differs"
                )
            self._send_router_response(
                self.sockets.artifact,
                routing_id,
                (artifact.header, *artifact.frames),
                channel="artifact",
                copy=False,
            )
        except Exception as exc:
            self._send_abort(
                self.sockets.artifact,
                routing_id,
                error_code="UNKNOWN_ARTIFACT",
                detail=str(exc),
            )

    def _serve_data(self) -> None:
        routing_id, frames = recv_router_multipart(
            self.sockets.data,
            copy=False,
        )
        if len(frames) < 3:
            self._send_abort(
                self.sockets.data,
                routing_id,
                error_code="DECODE_ERROR",
                detail="native collection part is truncated",
            )
            return
        attempt_id: str | None = None
        try:
            self.coordinator.expire_workers(now_unix_ns=time.time_ns())
            claimed = peek_native_collection_part_identity(frames[0])
            attempt_id = claimed.attempt_id
            discarded = self.coordinator.discarded_attempt_context(
                claimed.lease_id,
                claimed.attempt_id,
            )
            if discarded is not None:
                lease, _attempt, worker = discarded
                self._require_channel_route(
                    worker.identity.worker_id,
                    worker.identity.session_id,
                    "data",
                    routing_id,
                )
                response: Any = NativeDrainResponse(
                    window_id=lease.window.identity.window_id
                )
            else:
                lease, attempt, worker = self.coordinator.attempt_context(
                    claimed.lease_id,
                    claimed.attempt_id,
                )
                self._require_channel_route(
                    worker.identity.worker_id,
                    worker.identity.session_id,
                    "data",
                    routing_id,
                )
                decoded = decode_native_collection_part(
                    frames[0],
                    frames[1],
                    frames[2:],
                    expected_lease=lease,
                    expected_attempt=attempt,
                    expected_worker=worker,
                )
                self.coordinator.accept_part(decoded)
                response = NativePartAck(
                    part=decoded.identity,
                    drain_hint=self.coordinator.window_drain_hint(),
                )
        except Exception as exc:
            if attempt_id is not None:
                with suppress(Exception):
                    self.coordinator.fail_attempt(
                        attempt_id,
                        reason=f"data-plane rejection: {exc}",
                    )
            response = NativeProtocolAbort(
                error_code="DECODE_ERROR",
                detail=str(exc),
            )
        self._send_router_response(
            self.sockets.data,
            routing_id,
            (encode_message(response),),
            channel="data",
            copy=True,
        )

    @staticmethod
    def _require_discarded_attempt_owner(
        context: tuple[
            NativeCollectionShardLease,
            NativeCollectionAttempt,
            NativeCollectionWorkerManifest,
        ],
        *,
        worker_id: str,
        session_id: str,
    ) -> None:
        """Reject a late tail message routed from a different worker session."""
        _lease, attempt, _worker = context
        if attempt.worker_id != worker_id or attempt.worker_session_id != session_id:
            raise NativeCoordinatorProtocolError(
                "discarded attempt belongs to another worker session"
            )

    def _require_control_route(self, worker_id: str, routing_id: bytes) -> None:
        if self._worker_control_routes.get(worker_id) != routing_id:
            raise NativeCoordinatorProtocolError(
                "worker control routing identity differs"
            )

    @staticmethod
    def _require_channel_route(
        worker_id: str,
        session_id: str,
        channel: Literal["artifact", "data"],
        routing_id: bytes,
    ) -> None:
        expected = f"{worker_id}/{session_id}/{channel}".encode()
        if routing_id != expected:
            raise NativeCoordinatorProtocolError(
                f"worker {channel} routing identity differs"
            )

    def _send_abort(
        self,
        socket: Any,
        routing_id: bytes,
        *,
        error_code: Literal[
            "DECODE_ERROR",
            "IDENTITY_MISMATCH",
            "OLD_ATTEMPT",
            "DUPLICATE_PART",
            "SEQUENCE_MISMATCH",
            "UNKNOWN_ARTIFACT",
            "WINDOW_ABORTED",
        ],
        detail: str,
    ) -> None:
        self._send_router_response(
            socket,
            routing_id,
            (
                encode_message(
                    NativeProtocolAbort(
                        error_code=error_code,
                        detail=detail or "native protocol error",
                    )
                ),
            ),
            channel=self._socket_channel(socket),
            copy=True,
        )

    def _send_router_response(
        self,
        socket: Any,
        routing_id: bytes,
        frames: tuple[Any, ...] | list[Any],
        *,
        channel: Literal["control", "artifact", "data"],
        copy: bool,
    ) -> None:
        """Drop only an unroutable peer response and keep serving other peers."""
        if try_send_router_multipart(
            socket,
            routing_id,
            frames,
            copy=copy,
        ):
            return
        self._router_delivery_failures[channel] += 1
        _LOGGER.warning(
            "native coordinator dropped an unroutable %s response: route=%r "
            "delivery_failures=%d",
            channel,
            routing_id,
            self._router_delivery_failures[channel],
        )

    def _socket_channel(
        self,
        socket: Any,
    ) -> Literal["control", "artifact", "data"]:
        """Name one coordinator socket for delivery diagnostics."""
        if socket is self.sockets.control:
            return "control"
        if socket is self.sockets.artifact:
            return "artifact"
        if socket is self.sockets.data:
            return "data"
        raise ValueError("unknown native coordinator socket")

    def _publish_status(self, *, now_unix_ns: int, force: bool = False) -> None:
        if self.status_path is None:
            return
        now = time.monotonic()
        if not force and now - self._last_status_at < self.status_interval_seconds:
            return
        self._last_status_at = now
        coordinator_status = self.coordinator.status(now_unix_ns=now_unix_ns)
        payload = {
            "format": "native_distributed_collection_status_v1",
            "recorded_at_unix_ns": now_unix_ns,
            "coordinator": coordinator_status.model_dump(mode="json"),
            "workers": {
                worker_id: metrics.model_dump(mode="json")
                for worker_id, metrics in sorted(self._worker_metrics.items())
            },
            "router_delivery_failures": dict(self._router_delivery_failures),
            "assignment_pool_total": (
                self._quota_plan.total_games
                if self._quota_plan is not None
                else len(self._assignments)
            ),
            "assignment_pool_issued": self._assignment_cursor,
            "quota_execution": (
                None
                if self._quota_plan is None
                else {
                    "exposure_cohort_games": (self._quota_plan.exposure_cohort_games),
                    "initial_wave_workers_issued": len(self._v3_initial_wave_workers),
                    "initial_wave_workers_total": len(
                        coordinator_status.topology_workers
                    ),
                }
            ),
        }
        atomic_write_bytes(
            self.status_path,
            json_payload(payload),
            overwrite=True,
        )


def _artifact_coherent_indices(
    requirements: Sequence[tuple[str, ...]],
    available_indices: set[int],
    *,
    concurrent_games: int,
    frozen_artifact_limit: int,
    required_artifact_id: str | None = None,
) -> tuple[int, ...]:
    """Select a stable full shard without crossing its frozen-artifact limit."""
    if concurrent_games <= 0 or frozen_artifact_limit <= 0:
        raise ValueError("native V2 shard capacity must be positive")
    if any(index < 0 or index >= len(requirements) for index in available_indices):
        raise ValueError("native V2 scheduler received an invalid assignment index")
    grouped: dict[tuple[str, ...], list[int]] = {}
    current_ids: set[str] = set()
    for index in sorted(available_indices):
        required = requirements[index]
        if not required:
            raise ValueError("native V2 assignment has no required artifact")
        current_ids.add(required[0])
        grouped.setdefault(required, []).append(index)
    if not grouped or len(current_ids) != 1:
        return ()
    (current_id,) = tuple(current_ids)
    ordered_groups = sorted(
        grouped,
        key=lambda item: (-len(grouped[item]), item),
    )
    if any(
        required[0] != current_id or current_id not in required
        for required in ordered_groups
    ):
        raise ValueError("native V2 artifact requirements crossed current policy")

    selected: set[tuple[str, ...]] = set()
    if required_artifact_id is None:
        selected_frozen: set[str] = set()
        for required in ordered_groups:
            frozen = set(required) - {current_id}
            combined = selected_frozen | frozen
            if len(combined) > frozen_artifact_limit:
                continue
            selected.add(required)
            selected_frozen = combined
    else:
        candidates: list[tuple[int, int, tuple[str, ...], set[tuple[str, ...]]]] = []
        for seed in ordered_groups:
            if required_artifact_id not in seed:
                continue
            candidate = {seed}
            candidate_frozen = set(seed) - {current_id}
            if len(candidate_frozen) > frozen_artifact_limit:
                continue
            for required in ordered_groups:
                combined = candidate_frozen | (set(required) - {current_id})
                if len(combined) > frozen_artifact_limit:
                    continue
                candidate.add(required)
                candidate_frozen = combined
            candidate_count = sum(len(grouped[item]) for item in candidate)
            candidates.append((candidate_count, len(grouped[seed]), seed, candidate))
        if candidates:
            _count, _seed_count, _seed, selected = min(
                candidates,
                key=lambda item: (-item[0], -item[1], item[2]),
            )

    selected_count = sum(len(grouped[required]) for required in selected)
    if selected_count < concurrent_games:
        return ()

    # Draw proportionally from every compatible requirement group. Taking a
    # prefix of the largest group creates pure current-only or pure-PFSP shards;
    # the coordinator may then reach its decision budget before the accepted
    # prefix reflects the controller's lane and artifact quotas.
    ordered_selected = tuple(sorted(selected))
    total_count = sum(len(indices) for indices in grouped.values())
    total_frozen = sum(
        len(grouped[required]) for required in grouped if set(required) - {current_id}
    )
    total_current_only = total_count - total_frozen
    selected_frozen_count = sum(
        len(grouped[required])
        for required in ordered_selected
        if set(required) - {current_id}
    )
    selected_current_only_count = sum(
        len(grouped[required])
        for required in ordered_selected
        if not set(required) - {current_id}
    )
    group_masses = {
        required: (
            Fraction(
                total_frozen * len(grouped[required]),
                total_count * selected_frozen_count,
            )
            if set(required) - {current_id}
            else Fraction(
                total_current_only * len(grouped[required]),
                total_count * selected_current_only_count,
            )
        )
        for required in ordered_selected
    }
    selected_per_group = dict.fromkeys(ordered_selected, 0)
    indices: list[int] = []
    start_position = 0
    if required_artifact_id is not None:
        forced_group = min(
            (
                required
                for required in ordered_selected
                if required_artifact_id in required
            ),
            key=lambda item: (-len(grouped[item]), item),
        )
        indices.append(grouped[forced_group][0])
        selected_per_group[forced_group] = 1
        start_position = 1
    for position in range(start_position, concurrent_games):
        available = tuple(
            required
            for required in ordered_selected
            if selected_per_group[required] < len(grouped[required])
        )
        required = max(
            available,
            key=lambda item: (
                group_masses[item] * (position + 1) - selected_per_group[item],
                item,
            ),
        )
        offset = selected_per_group[required]
        indices.append(grouped[required][offset])
        selected_per_group[required] = offset + 1
    return tuple(sorted(indices))


def _v3_frozen_artifact_limit(advertised_limit: int) -> int:
    """Use one frozen artifact so every persistent arena remains balanced.

    Direct workers divide their persistent lane capacity evenly across the
    execution-bank ring.  A quota shard containing several skewed frozen
    artifacts can satisfy the bank-count limit while still overflowing one
    physical lane.  Keeping one frozen artifact per lease lets the existing
    bank planner stripe that artifact and current-policy games across every
    bank, which is capacity-safe and also produces larger homogeneous CUDA
    batches.  The advertised value remains a maximum, not a utilization goal.
    """
    if advertised_limit <= 0:
        raise ValueError("native V3 artifact bank limit must be positive")
    return 1


def _v3_exposure_games(
    manifests: Sequence[NativeCollectionWorkerManifest],
    *,
    planned_games: int,
) -> int:
    """Validate the plan-owned exposure geometry against active workers."""
    if planned_games <= 0 or not manifests:
        raise ValueError("native V3 exposure geometry is absent")
    if any(
        not any(
            tier.concurrent_games == planned_games for tier in manifest.capacity_tiers
        )
        for manifest in manifests
    ):
        raise ValueError("native V3 exposure geometry is not shared by every worker")
    return planned_games


def _v3_candidate_capacity_tiers(
    manifest: NativeCollectionWorkerManifest,
    selected: NativeCollectionCapacityTier,
    *,
    remaining_games: int,
    exposure_games: int | None,
    primary_only: bool,
) -> tuple[NativeCollectionCapacityTier, ...]:
    """Order executable V3 tiers without shrinking ordinary primary batches."""
    smaller = tuple(
        tier
        for tier in manifest.capacity_tiers
        if tier.tier_id != selected.tier_id
        and tier.concurrent_games <= selected.concurrent_games
        and tier.concurrent_games <= remaining_games
    )
    if exposure_games is not None:
        # The capacity-aware first wave is a deliberate exception to
        # learner-clocked primary-only scheduling. Every worker uses the same
        # full-arena, single-wave geometry so coverage does not create a GPU
        # warm-up bubble.
        return tuple(
            sorted(
                (
                    tier
                    for tier in (selected, *smaller)
                    if tier.concurrent_games == exposure_games
                ),
                key=lambda tier: (
                    -tier.native_arena_capacity,
                    tier.estimated_trainable_decisions,
                    tier.tier_id,
                ),
            )
        )
    if primary_only:
        return (selected,)
    return (
        selected,
        *sorted(
            smaller,
            key=lambda tier: (
                -tier.estimated_trainable_decisions,
                tier.tier_id,
            ),
        ),
    )


def _assignment_artifact_union(
    requirements: Sequence[tuple[str, ...]],
    indices: Sequence[int],
) -> tuple[str, ...]:
    """Return current-first canonical artifact identity for one selected shard."""
    if not indices:
        raise ValueError("native V2 artifact union requires assignments")
    current = requirements[indices[0]][0]
    required = {artifact_id for index in indices for artifact_id in requirements[index]}
    if any(requirements[index][0] != current for index in indices):
        raise ValueError("native V2 artifact union crossed current policy")
    return (current, *sorted(required - {current}))


def _plan_artifact_coherent_templates(
    requirements: Sequence[tuple[str, ...]],
    *,
    template_games: int,
    frozen_artifact_limit: int,
) -> tuple[_V2ShardTemplate, ...]:
    """Freeze and fairly interleave final shards before workers can claim them."""
    if template_games <= 0:
        raise ValueError("native V2 shard template size must be positive")

    def plan_remaining(
        available: set[int],
    ) -> list[tuple[tuple[int, ...], tuple[str, ...]]]:
        planned: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
        while len(available) >= template_games:
            indices = _artifact_coherent_indices(
                requirements,
                available,
                concurrent_games=template_games,
                frozen_artifact_limit=frozen_artifact_limit,
            )
            if not indices:
                break
            required_artifact_ids = _assignment_artifact_union(
                requirements,
                indices,
            )
            planned.append((indices, required_artifact_ids))
            available.difference_update(indices)
        return planned

    available = set(range(len(requirements)))
    unsequenced = plan_remaining(available)

    expected_artifacts = {
        artifact_id for required in requirements for artifact_id in required
    }
    planned_artifacts = {
        artifact_id for _indices, required in unsequenced for artifact_id in required
    }
    missing_artifacts = expected_artifacts - planned_artifacts
    if missing_artifacts:
        # The proportional greedy pass can strand a valid low-mass artifact in
        # a sub-shard tail. Replan from the full pool, forcing the rarest
        # artifacts into an executable shard before applying the unchanged
        # proportional planner to the remainder.
        available = set(range(len(requirements)))
        unsequenced = []
        current_ids = {required[0] for required in requirements if required}
        if len(current_ids) == 1:
            (current_id,) = tuple(current_ids)
            artifact_order = sorted(
                expected_artifacts - {current_id},
                key=lambda artifact_id: (
                    sum(artifact_id in required for required in requirements),
                    artifact_id,
                ),
            )
            planned_artifacts = set()
            for artifact_id in artifact_order:
                if artifact_id in planned_artifacts:
                    continue
                indices = _artifact_coherent_indices(
                    requirements,
                    available,
                    concurrent_games=template_games,
                    frozen_artifact_limit=frozen_artifact_limit,
                    required_artifact_id=artifact_id,
                )
                if not indices:
                    break
                required_artifact_ids = _assignment_artifact_union(
                    requirements,
                    indices,
                )
                unsequenced.append((indices, required_artifact_ids))
                planned_artifacts.update(required_artifact_ids)
                available.difference_update(indices)
            unsequenced.extend(plan_remaining(available))

        planned_artifacts = {
            artifact_id
            for _indices, required in unsequenced
            for artifact_id in required
        }
        missing_artifacts = expected_artifacts - planned_artifacts
        if missing_artifacts:
            raise ValueError(
                "native V2 plan cannot fill a shard for artifacts: "
                + ", ".join(sorted(missing_artifacts))
            )

    ordered = _interleave_artifact_cohorts(unsequenced)
    return tuple(
        _V2ShardTemplate(
            sequence_id=sequence_id,
            assignment_indices=indices,
            required_artifact_ids=required_artifact_ids,
        )
        for sequence_id, (indices, required_artifact_ids) in enumerate(ordered)
    )


def _common_v2_template_geometry(
    manifests: Sequence[NativeCollectionWorkerManifest],
) -> tuple[int, int]:
    """Choose one shard tier executable by every active V2 worker."""
    if not manifests:
        raise ValueError("native V2 has no active worker manifests")
    common_games = {tier.concurrent_games for tier in manifests[0].capacity_tiers}
    for manifest in manifests[1:]:
        common_games &= {tier.concurrent_games for tier in manifest.capacity_tiers}
    if not common_games:
        raise ValueError("native V2 workers have no common shard tier")
    # Use the largest geometry that every frozen topology member can execute.
    # Choosing the smallest common tail tier forces otherwise capable workers
    # to run the entire V2 window as tiny shards and destroys policy batching.
    template_games = max(common_games)
    frozen_limit = min(
        max(
            tier.native_policy_group_bank_limit
            for tier in manifest.capacity_tiers
            if tier.concurrent_games == template_games
        )
        for manifest in manifests
    )
    if frozen_limit <= 0:
        raise ValueError("native V2 workers cannot host a frozen artifact")
    return template_games, frozen_limit


def _interleave_artifact_cohorts(
    templates: Sequence[tuple[tuple[int, ...], tuple[str, ...]]],
) -> tuple[tuple[tuple[int, ...], tuple[str, ...]], ...]:
    """Advance every artifact cohort at equal normalized progress.

    This preserves every cohort's exact final quota while ensuring that a
    majority cohort cannot consume the decision budget before each fillable
    minority cohort has supplied its first immutable template.
    """
    cohorts: dict[
        tuple[str, ...],
        list[tuple[tuple[int, ...], tuple[str, ...]]],
    ] = {}
    for template in templates:
        cohorts.setdefault(template[1], []).append(template)
    offsets = dict.fromkeys(cohorts, 0)
    ordered: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
    while len(ordered) < len(templates):
        available = tuple(
            artifact_ids
            for artifact_ids, cohort in cohorts.items()
            if offsets[artifact_ids] < len(cohort)
        )
        artifact_ids = min(
            available,
            key=lambda item: (
                Fraction(offsets[item], len(cohorts[item])),
                item,
            ),
        )
        offset = offsets[artifact_ids]
        ordered.append(cohorts[artifact_ids][offset])
        offsets[artifact_ids] = offset + 1
    return tuple(ordered)


__all__ = ["NativeCoordinatorService"]
