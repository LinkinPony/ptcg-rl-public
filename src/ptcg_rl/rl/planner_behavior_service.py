"""Shared production planner behavior service for rollout and serving."""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Any, Literal, Protocol, Self, cast

import torch

from ptcg_rl.agent.search.fixed_select_v5_probe import (
    execute_reusable_v5_fixed_select_probe,
)
from ptcg_rl.agent.search.planner_fallback import PlannerFallbackReason
from ptcg_rl.agent.search.planner_scoring import SharedRootInformationLeafScorer
from ptcg_rl.agent.search.planning_session_contract import ContinuationPrompt
from ptcg_rl.agent.search.planning_session_tree import (
    HierarchicalPlanningSessionExecutor,
)
from ptcg_rl.agent.search.prompt_actions import describe_prompt_action_space
from ptcg_rl.agent.search.root_information_context import PublicBeliefFeatureProducer
from ptcg_rl.agent.search.root_information_tensorizer import (
    ProductionRootInformationTensorizer,
    RootInformationModelInputBatch,
)
from ptcg_rl.belief.runtime_identity import belief_runtime_fingerprint
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.engine.consequence_request_identity import (
    root_observation_fingerprint,
)
from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.engine.native_planning_session_pool import (
    NativePlanningSessionLanePool,
)
from ptcg_rl.rl.planner_behavior import (
    PlannerCollectionMetadata,
    base_fallback_evidence,
)
from ptcg_rl.rl.planner_behavior_policy import PlannerBehaviorPolicy
from ptcg_rl.rl.planner_behavior_policy_contract import (
    PlannerCandidateEvaluator,
    PlannerDecisionRequest,
    PlannerPolicyDecision,
)
from ptcg_rl.rl.planner_evidence import (
    PLANNER_SOURCE_NAMES,
    ScenarioSupportMode,
)
from ptcg_rl.rl.planner_inference_session import (
    PlannerInferenceLease,
    PlannerInferenceSession,
)
from ptcg_rl.rl.planner_runtime_identity import (
    ResolvedPlannerRuntimeConfig,
    ResolvedPlannerRuntimeIdentity,
)
from ptcg_rl.rl.planner_service_inputs import (
    PlannerDecisionRequestFactory,
    PlannerRootRow,
)
from ptcg_rl.runtime.planner_resources import (
    PlannerBufferLedger,
    PlannerBufferLedgerStats,
    PlannerBufferReservation,
)
from ptcg_rl.runtime.planner_telemetry import (
    PlannerDecisionRuntimeStats,
    PlannerRequestTelemetry,
    PlannerStage,
    PlannerStageEvent,
)
from ptcg_rl.runtime.work_ledger import PlannerRequestLedger

_FALLBACK_SUPPORT_DOMAIN = b"ptcg-rl/planner-fallback-support/v1\x00"


class _PlannerQueueFullError(RuntimeError):
    """Internal signal converted to explicit schema-9 queue fallback."""


class PlannerLeaseIdentityResolver(Protocol):
    """Resolve mutable model publication data against static planner semantics."""

    def __call__(
        self,
        policy: object,
        runtime: ResolvedPlannerRuntimeConfig,
    ) -> ResolvedPlannerRuntimeIdentity:
        """Return one exact per-lease identity for the supplied policy."""


@dataclass(frozen=True, slots=True)
class PlannerBehaviorServiceStats:
    """Thread-safe bounded queue and lane-occupancy telemetry."""

    submitted_rows: int
    completed_rows: int
    failed_rows: int
    queue_rejected_rows: int
    inflight_rows: int
    active_rows: int
    peak_active_rows: int
    queue_capacity: int
    lane_count: int
    native_active_sessions: int
    native_active_calls: int
    buffers: PlannerBufferLedgerStats


class PlannerBehaviorService:
    """Use one implementation for candidate rollout and packaged act-time rows."""

    def __init__(
        self,
        *,
        runtime_config: ResolvedPlannerRuntimeConfig,
        session_pool: NativePlanningSessionLanePool,
        belief_sampler: BeliefSampler,
        belief_feature_producer: PublicBeliefFeatureProducer | None,
        lease_identity_resolver: PlannerLeaseIdentityResolver | None = None,
        stochastic_seed: int = 0,
    ) -> None:
        self.runtime_config = runtime_config
        self._session_pool = session_pool
        self._belief_sampler = belief_sampler
        self._belief_feature_producer = belief_feature_producer
        actual_belief_fingerprint = belief_runtime_fingerprint(
            belief_sampler,
            belief_feature_producer,
        )
        if (
            runtime_config.scenario.belief_sampler_fingerprint
            != actual_belief_fingerprint
        ):
            raise ValueError(
                "planner belief runtime differs from its static semantic identity"
            )
        expected_engine = runtime_config.engine
        actual_engine = (
            str(getattr(session_pool, "engine_library_fingerprint", "")),
            str(getattr(session_pool, "native_abi_fingerprint", "")),
            str(getattr(session_pool, "native_schema_fingerprint", "")),
        )
        if actual_engine != (
            expected_engine.library_fingerprint,
            expected_engine.native_abi_fingerprint,
            expected_engine.native_schema_fingerprint,
        ):
            raise ValueError(
                "planner native session pool differs from its static engine identity"
            )
        self._custom_lease_identity_resolver = lease_identity_resolver
        self._resolve_lease = lease_identity_resolver or _default_lease_identity
        self._stochastic_seed = int(stochastic_seed)
        self._lane_count = runtime_config.session_pool.lane_count
        self._queue_capacity = runtime_config.batching.planner_queue_capacity
        self._buffer_ledger = PlannerBufferLedger(
            runtime_config.buffers,
            cells_per_request=(
                runtime_config.planner_behavior.constructor.k_total
                * runtime_config.scenario.belief_world_count
            ),
            unique_leaves_per_request=(runtime_config.tensorizer.max_unique_leaves),
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self._lane_count,
            thread_name_prefix="planner-behavior",
        )
        self._admission = threading.BoundedSemaphore(
            self._lane_count + self._queue_capacity
        )
        self._stats_lock = threading.Lock()
        self._closed = False
        self._submitted_rows = 0
        self._completed_rows = 0
        self._failed_rows = 0
        self._queue_rejected_rows = 0
        self._active_rows = 0
        self._peak_active_rows = 0

    def __enter__(self) -> Self:
        return self

    @property
    def belief_summary_width(self) -> int:
        """Return the fixed root-value belief width used by every leaf."""
        return self.runtime_config.tensorizer.belief_summary_dim

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        """Drain the persistent bounded row executor exactly once."""
        with self._stats_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)

    def stats(self) -> PlannerBehaviorServiceStats:
        """Return queue/lane occupancy plus the native pool snapshot."""
        native = self._session_pool.stats()
        with self._stats_lock:
            return PlannerBehaviorServiceStats(
                submitted_rows=self._submitted_rows,
                completed_rows=self._completed_rows,
                failed_rows=self._failed_rows,
                queue_rejected_rows=self._queue_rejected_rows,
                inflight_rows=self._submitted_rows - self._completed_rows,
                active_rows=self._active_rows,
                peak_active_rows=self._peak_active_rows,
                queue_capacity=self._queue_capacity,
                lane_count=self._lane_count,
                native_active_sessions=native.active_sessions,
                native_active_calls=native.active_calls,
                buffers=self._buffer_ledger.stats(),
            )

    def _try_reserve_row(self) -> PlannerBufferReservation | None:
        """Reserve queue and fixed staging capacity before proposal work."""
        with self._stats_lock:
            if self._closed:
                raise RuntimeError("planner behavior service is closed")
        if not self._admission.acquire(blocking=False):
            with self._stats_lock:
                self._queue_rejected_rows += 1
            return None
        buffer_reservation = self._buffer_ledger.try_acquire()
        if buffer_reservation is not None:
            return buffer_reservation
        self._admission.release()
        with self._stats_lock:
            self._queue_rejected_rows += 1
        return None

    def _release_reserved_row(
        self,
        buffer_reservation: PlannerBufferReservation,
    ) -> None:
        """Release a pre-proposal reservation that was never submitted."""
        try:
            buffer_reservation.release()
        finally:
            self._admission.release()

    def _submit_reserved_row(
        self,
        operation: Any,
        buffer_reservation: PlannerBufferReservation,
        **kwargs: Any,
    ) -> Future[PlannerPolicyDecision]:
        """Submit one row whose queue and byte capacity is already reserved."""
        with self._stats_lock:
            if self._closed:
                self._release_reserved_row(buffer_reservation)
                raise RuntimeError("planner behavior service is closed")
            self._submitted_rows += 1
        try:
            submitted_at = time.perf_counter()
            return self._executor.submit(
                self._run_submitted_row,
                operation,
                kwargs,
                buffer_reservation,
                submitted_at,
            )
        except Exception:
            buffer_reservation.release()
            self._admission.release()
            with self._stats_lock:
                self._completed_rows += 1
                self._failed_rows += 1
            raise

    def _run_submitted_row(
        self,
        operation: Any,
        kwargs: Mapping[str, Any],
        buffer_reservation: PlannerBufferReservation,
        submitted_at: float,
    ) -> PlannerPolicyDecision:
        with self._stats_lock:
            self._active_rows += 1
            self._peak_active_rows = max(
                self._peak_active_rows,
                self._active_rows,
            )
        success = False
        try:
            telemetry = kwargs.get("telemetry")
            if isinstance(telemetry, PlannerRequestTelemetry):
                telemetry.record(
                    PlannerStageEvent(
                        stage=PlannerStage.QUEUE_WAIT,
                        seconds=max(0.0, time.perf_counter() - submitted_at),
                        rows=1,
                    )
                )
            result = cast(PlannerPolicyDecision, operation(**kwargs))
            success = True
            return result
        finally:
            with self._stats_lock:
                self._active_rows -= 1
                self._completed_rows += 1
                self._failed_rows += int(not success)
            try:
                buffer_reservation.release()
            finally:
                self._admission.release()

    def plan_batch(self, batch: Any) -> Sequence[PlannerPolicyDecision | None]:
        """Plan candidate-policy rows and explicitly leave frozen rows untouched."""
        rows = tuple(batch.rows)
        candidate_indices = tuple(
            index
            for index, row in enumerate(rows)
            if str(row.policy_role) == "candidate"
        )
        results: list[PlannerPolicyDecision | None] = [None] * len(rows)
        if not candidate_indices:
            _release_unplanned_contexts(batch, tuple(range(len(rows))))
            return tuple(results)
        root_fallback_reason = getattr(batch, "root_fallback_reason", None)
        if root_fallback_reason is not None:
            if tuple(batch.planner_context_handles):
                raise RuntimeError("planner root fallback cannot own contexts")
            reason = PlannerFallbackReason(root_fallback_reason)
            if reason is not PlannerFallbackReason.MODEL_LEASE_CAPACITY:
                raise ValueError("unsupported pre-planner root fallback reason")
            resolved_fallback = self.runtime_config.resolve_for_lease(
                model_fingerprint=str(batch.model_fingerprint),
                policy_version=int(batch.policy_version),
                proposal_version=int(batch.proposal_version),
            )
            for row_index in candidate_indices:
                results[row_index] = replace(
                    self._fallback(
                        row=_rollout_root_row(
                            rows[row_index],
                            base_action=batch.base_actions[row_index],
                            base_old_logprob=batch.base_logprobs[row_index],
                        ),
                        resolved=resolved_fallback,
                        reason=reason,
                    ),
                    runtime_stats=PlannerDecisionRuntimeStats(
                        batch_row_position=row_index,
                    ),
                )
            return tuple(results)

        resolved: ResolvedPlannerRuntimeIdentity | None = None
        session: PlannerInferenceSession | None = None
        reservations: dict[int, PlannerBufferReservation] = {}
        requests: dict[int, PlannerDecisionRequest] = {}
        ledgers: dict[int, PlannerRequestLedger] = {}
        telemetry_by_row = {
            row_index: PlannerRequestTelemetry() for row_index in candidate_indices
        }
        action_spaces = {
            row_index: describe_prompt_action_space(
                rows[row_index].observation.get("select")
            )
            for row_index in candidate_indices
        }
        fast_path_indices = tuple(
            row_index
            for row_index in candidate_indices
            if action_spaces[row_index].legal_action_count == 1
        )
        fast_path_set = frozenset(fast_path_indices)
        proposal_unsupported_indices = tuple(
            row_index
            for row_index in candidate_indices
            if row_index not in fast_path_set
            and (
                action_spaces[row_index].max_count
                > self.runtime_config.proposal_search.depth_limit
                or action_spaces[row_index].max_count
                > self.runtime_config.proposal_search.prefix_node_limit
            )
        )
        proposal_unsupported_set = frozenset(proposal_unsupported_indices)
        planner_indices = tuple(
            row_index
            for row_index in candidate_indices
            if row_index not in fast_path_set
            and row_index not in proposal_unsupported_set
        )
        try:
            started = time.monotonic()
            if self._custom_lease_identity_resolver is None:
                resolved = self.runtime_config.resolve_for_lease(
                    model_fingerprint=str(batch.model_fingerprint),
                    policy_version=int(batch.policy_version),
                    proposal_version=int(batch.proposal_version),
                )
            else:
                resolved = self._resolve_lease(batch.policy, self.runtime_config)
            identity = resolved.runtime_identity
            if (
                identity.policy_version != int(batch.policy_version)
                or identity.model_fingerprint != str(batch.model_fingerprint)
                or identity.proposal_version != int(batch.proposal_version)
            ):
                raise RuntimeError("rollout batch differs from planner model lease")
            for row_index in fast_path_indices:
                results[row_index] = replace(
                    self._fallback(
                        row=_rollout_root_row(
                            rows[row_index],
                            base_action=batch.base_actions[row_index],
                            base_old_logprob=batch.base_logprobs[row_index],
                        ),
                        resolved=resolved,
                        reason=PlannerFallbackReason.INELIGIBLE,
                    ),
                    runtime_stats=PlannerDecisionRuntimeStats(
                        batch_row_position=row_index,
                    ),
                )
            for row_index in proposal_unsupported_indices:
                results[row_index] = replace(
                    self._fallback(
                        row=_rollout_root_row(
                            rows[row_index],
                            base_action=batch.base_actions[row_index],
                            base_old_logprob=batch.base_logprobs[row_index],
                        ),
                        resolved=resolved,
                        reason=PlannerFallbackReason.BUDGET_TRUNCATED,
                    ),
                    runtime_stats=PlannerDecisionRuntimeStats(
                        batch_row_position=row_index,
                    ),
                )
            if not planner_indices:
                if (
                    tuple(batch.planner_context_handles)
                    and not proposal_unsupported_indices
                ):
                    raise RuntimeError(
                        "planner base-fast-path batch retained root contexts"
                    )
                return tuple(results)
            if len(tuple(batch.planner_context_handles)) != len(rows):
                raise RuntimeError("planner work has incomplete root contexts")
            deadline = _effective_planner_deadline(
                self.runtime_config,
                started=started,
            )
            _bind_retained_contexts(
                batch.policy,
                batch.planner_context_handles,
                policy_version=identity.policy_version,
                tensor_schema_fingerprint=resolved.tensor_schema_fingerprint,
                deadline_monotonic=deadline,
            )
            session = PlannerInferenceSession(
                policy=batch.policy,
                states=batch.states,
                options=batch.options,
                decks=batch.decks,
                context_handles=batch.planner_context_handles,
                lease=PlannerInferenceLease(
                    model_fingerprint=identity.model_fingerprint,
                    policy_version=identity.policy_version,
                    tensor_schema_fingerprint=resolved.tensor_schema_fingerprint,
                    deadline_monotonic=deadline,
                    inference_device_type=_planner_inference_device_type(batch.policy),
                    inference_timeout_seconds=(
                        self.runtime_config.deadlines.inference_timeout_seconds
                    ),
                ),
            )

            rejected_indices: list[int] = []
            for row_index in planner_indices:
                reservation = self._try_reserve_row()
                if reservation is None:
                    rejected_indices.append(row_index)
                    results[row_index] = replace(
                        self._fallback(
                            row=_rollout_root_row(
                                rows[row_index],
                                base_action=batch.base_actions[row_index],
                                base_old_logprob=batch.base_logprobs[row_index],
                            ),
                            resolved=resolved,
                            reason=PlannerFallbackReason.QUEUE_FULL,
                        ),
                        runtime_stats=PlannerDecisionRuntimeStats(
                            batch_row_position=row_index,
                        ),
                    )
                else:
                    reservations[row_index] = reservation
            release_before_proposal = (
                fast_path_indices
                + proposal_unsupported_indices
                + tuple(rejected_indices)
            )
            if release_before_proposal:
                session.release_rows(
                    release_before_proposal,
                    deadline_monotonic=(
                        time.monotonic()
                        + self.runtime_config.deadlines.cleanup_timeout_seconds
                    ),
                )
            admitted_indices = tuple(reservations)
            if not admitted_indices:
                return tuple(results)
            ordered = torch.tensor(
                [
                    describe_prompt_action_space(
                        rows[index].observation.get("select")
                    ).ordered
                    for index in admitted_indices
                ],
                dtype=torch.bool,
            )
            proposal_started = time.perf_counter()
            proposals = session.generate_proposals(
                admitted_indices,
                ordered_rows=ordered,
                limits=self.runtime_config.proposal_search,
            )
            proposal_event = PlannerStageEvent(
                stage=PlannerStage.BASE_PROPOSAL,
                seconds=time.perf_counter() - proposal_started,
                rows=len(admitted_indices),
                batch_capacity=(self.runtime_config.batching.proposal_microbatch_rows),
            )
            for row_index in admitted_indices:
                telemetry_by_row[row_index].record(proposal_event)
            factory = self._request_factory(resolved)
            for proposal_index, row_index in enumerate(admitted_indices):
                request = factory.build(
                    _rollout_root_row(
                        rows[row_index],
                        base_action=batch.base_actions[row_index],
                        base_old_logprob=batch.base_logprobs[row_index],
                    ),
                    greedy_action=proposals.base_greedy_actions[proposal_index],
                    proposal_actions=tuple(
                        candidate.action
                        for candidate in proposals.decisions[proposal_index].candidates
                    ),
                    identity=identity,
                )
                requests[row_index] = request
                ledgers[row_index] = PlannerRequestLedger(
                    self.runtime_config.work_limits,
                    deadline_monotonic=deadline,
                    started_monotonic=started,
                )
            futures: dict[int, Future[PlannerPolicyDecision]] = {}
            for row_index in admitted_indices:
                reservation = reservations.pop(row_index)
                try:
                    futures[row_index] = self._submit_reserved_row(
                        self._plan_row,
                        reservation,
                        session=session,
                        resolved=resolved,
                        row_index=row_index,
                        request=requests[row_index],
                        ledger=ledgers[row_index],
                        own_deck=(
                            getattr(rows[row_index], "model_deck", None)
                            or rows[row_index].deck_pair[rows[row_index].seat]
                        ),
                        telemetry=telemetry_by_row[row_index],
                        batch_row_position=row_index,
                    )
                except Exception as exc:
                    results[row_index] = replace(
                        self._fallback(
                            row=_rollout_root_row(
                                rows[row_index],
                                base_action=batch.base_actions[row_index],
                                base_old_logprob=batch.base_logprobs[row_index],
                            ),
                            resolved=resolved,
                            reason=_fallback_reason(exc),
                        ),
                        runtime_stats=PlannerDecisionRuntimeStats(
                            batch_row_position=row_index,
                        ),
                    )
            for row_index, future in futures.items():
                try:
                    results[row_index] = future.result()
                except Exception as exc:
                    fallback = self._fallback(
                        row=_rollout_root_row(
                            rows[row_index],
                            base_action=batch.base_actions[row_index],
                            base_old_logprob=batch.base_logprobs[row_index],
                        ),
                        resolved=resolved,
                        reason=_fallback_reason(exc),
                    )
                    results[row_index] = replace(
                        fallback,
                        telemetry_events=telemetry_by_row[row_index].events,
                        runtime_stats=PlannerDecisionRuntimeStats(
                            batch_row_position=row_index,
                        ),
                    )
        except Exception as exc:
            reason = _fallback_reason(exc)
            if resolved is None:
                resolved = _best_effort_resolved_identity(
                    batch.policy,
                    self.runtime_config,
                    resolver=self._resolve_lease,
                )
            if resolved is None:
                raise RuntimeError(
                    "planner cannot emit schema-9 fallback without lease identity"
                ) from exc
            for row_index in candidate_indices:
                if results[row_index] is not None:
                    continue
                fallback = self._fallback(
                    row=_rollout_root_row(
                        rows[row_index],
                        base_action=batch.base_actions[row_index],
                        base_old_logprob=batch.base_logprobs[row_index],
                    ),
                    resolved=resolved,
                    reason=reason,
                )
                row_telemetry = telemetry_by_row.get(row_index)
                results[row_index] = replace(
                    fallback,
                    telemetry_events=(
                        () if row_telemetry is None else row_telemetry.events
                    ),
                    runtime_stats=PlannerDecisionRuntimeStats(
                        batch_row_position=row_index,
                    ),
                )
        finally:
            for reservation in reservations.values():
                self._release_reserved_row(reservation)
            if session is not None:
                with suppress(Exception):
                    session.release_rows(
                        tuple(range(len(rows))),
                        deadline_monotonic=(
                            time.monotonic()
                            + self.runtime_config.deadlines.cleanup_timeout_seconds
                        ),
                    )
            else:
                _release_unplanned_contexts(batch, tuple(range(len(rows))))
        return tuple(results)

    def plan_runtime_row(
        self,
        *,
        policy: object,
        row: PlannerRootRow,
        inference_session: PlannerInferenceSession,
        row_index: int = 0,
    ) -> PlannerPolicyDecision:
        """Plan one packaged row through the same proposal/engine/behavior core."""
        started = time.monotonic()
        effective_deadline = _effective_planner_deadline(
            self.runtime_config,
            started=started,
            outer_deadline=inference_session.lease.deadline_monotonic,
        )
        active_session = inference_session.narrow_deadline(effective_deadline)
        resolved: ResolvedPlannerRuntimeIdentity | None = None
        buffer_reservation: PlannerBufferReservation | None = None
        telemetry = PlannerRequestTelemetry()
        try:
            resolved = self._resolve_lease(policy, self.runtime_config)
            _bind_retained_contexts(
                policy,
                inference_session.context_handles,
                policy_version=inference_session.lease.policy_version,
                tensor_schema_fingerprint=(
                    inference_session.lease.tensor_schema_fingerprint
                ),
                deadline_monotonic=effective_deadline,
            )
            buffer_reservation = self._buffer_ledger.try_acquire()
            if buffer_reservation is None:
                raise _PlannerQueueFullError("planner buffer ledger is full")
            identity = resolved.runtime_identity
            ordered = describe_prompt_action_space(
                row.observation.get("select")
            ).ordered
            proposal_started = time.perf_counter()
            proposals = active_session.generate_proposals(
                (row_index,),
                ordered_rows=torch.tensor((ordered,), dtype=torch.bool),
                limits=self.runtime_config.proposal_search,
            )
            telemetry.record(
                PlannerStageEvent(
                    stage=PlannerStage.BASE_PROPOSAL,
                    seconds=time.perf_counter() - proposal_started,
                    rows=1,
                    batch_capacity=(
                        self.runtime_config.batching.proposal_microbatch_rows
                    ),
                )
            )
            request = self._request_factory(resolved).build(
                row,
                greedy_action=proposals.base_greedy_actions[0],
                proposal_actions=tuple(
                    candidate.action for candidate in proposals.decisions[0].candidates
                ),
                identity=identity,
            )
            ledger = PlannerRequestLedger(
                self.runtime_config.work_limits,
                deadline_monotonic=effective_deadline,
                started_monotonic=started,
            )
            return self._plan_row(
                session=active_session,
                resolved=resolved,
                row_index=row_index,
                request=request,
                ledger=ledger,
                own_deck=row.own_deck,
                telemetry=telemetry,
                batch_row_position=row_index,
            )
        except Exception as exc:
            if resolved is None:
                raise RuntimeError(
                    "packaged planner cannot resolve its immutable lease"
                ) from exc
            fallback = self._fallback(
                row=row,
                resolved=resolved,
                reason=_fallback_reason(exc),
            )
            return replace(
                fallback,
                telemetry_events=telemetry.events,
                runtime_stats=PlannerDecisionRuntimeStats(
                    batch_row_position=row_index,
                ),
            )
        finally:
            if buffer_reservation is not None:
                buffer_reservation.release()
            with suppress(Exception):
                active_session.release_rows(
                    (row_index,),
                    deadline_monotonic=(
                        time.monotonic()
                        + self.runtime_config.deadlines.cleanup_timeout_seconds
                    ),
                )

    def _plan_row(
        self,
        *,
        session: PlannerInferenceSession,
        resolved: ResolvedPlannerRuntimeIdentity,
        row_index: int,
        request: PlannerDecisionRequest,
        ledger: PlannerRequestLedger,
        own_deck: Sequence[int],
        telemetry: PlannerRequestTelemetry,
        batch_row_position: int,
    ) -> PlannerPolicyDecision:
        controller = _SessionContinuationProvider(
            session=session,
            own_deck=tuple(int(value) for value in own_deck),
        )
        executor = HierarchicalPlanningSessionExecutor(
            config=self.runtime_config.hierarchical_search,
            pool=self._session_pool,
            controller=controller,
            controller_identity=resolved.controller,
            telemetry=telemetry,
        )
        tensorizer = ProductionRootInformationTensorizer(self.runtime_config.tensorizer)
        scorer = SharedRootInformationLeafScorer[RootInformationModelInputBatch](
            config=self.runtime_config.scoring,
            tensorizer=tensorizer,
            value_provider=session.root_value_provider(
                root_deck=own_deck,
                max_rows=(self.runtime_config.batching.root_value_microbatch_rows),
                telemetry=telemetry,
            ),
            telemetry=telemetry,
        )
        behavior = PlannerBehaviorPolicy[RootInformationModelInputBatch](
            config=self.runtime_config.planner_behavior,
            executor=executor,
            scorer=scorer,
            controller_identity=resolved.controller,
            generator=_row_generator(request.stochastic_seed),
            telemetry=telemetry,
        )
        evaluator = _SessionCandidateEvaluator(
            session=session,
            row_index=row_index,
            ordered=request.budget_request.ordered,
            telemetry=telemetry,
            batch_capacity=(self.runtime_config.batching.candidate_microbatch_rows),
        )
        if _requires_fixed_select_probe(request, self.runtime_config):
            probe = execute_reusable_v5_fixed_select_probe(
                request,
                executor=executor,
                scorer=scorer,
                ledger=ledger,
            )
            request = replace(request, equivalence_probe=probe)
        decision = behavior.decide(request, evaluator=evaluator, ledger=ledger)
        (
            engine_reuse_hits,
            native_rows,
            prefix_reuse_count,
            leaf_lookups,
            unique_leaves,
            consequence_cells,
        ) = behavior.runtime_stats
        runtime_stats = PlannerDecisionRuntimeStats(
            root_context_hits=1 + evaluator.evaluation_calls,
            root_context_misses=1,
            engine_reuse_hits=engine_reuse_hits,
            engine_reuse_misses=native_rows,
            leaf_reuse_hits=max(0, leaf_lookups - unique_leaves),
            leaf_reuse_misses=unique_leaves,
            prefix_reuse_count=prefix_reuse_count,
            unique_leaf_count=unique_leaves,
            consequence_cell_count=consequence_cells,
            batch_row_position=batch_row_position,
        )
        return replace(
            decision,
            telemetry_events=telemetry.events,
            runtime_stats=runtime_stats,
        )

    def _request_factory(
        self,
        resolved: ResolvedPlannerRuntimeIdentity,
    ) -> PlannerDecisionRequestFactory:
        return PlannerDecisionRequestFactory(
            belief_sampler=self._belief_sampler,
            scenario_count=self.runtime_config.scenario.belief_world_count,
            belief_summary_dim=self.runtime_config.tensorizer.belief_summary_dim,
            costs=self.runtime_config.request_costs,
            producer_contract_fingerprint=bytes.fromhex(
                resolved.continuation_semantics_fingerprint
            ),
            belief_feature_producer=self._belief_feature_producer,
            stochastic_seed=self._stochastic_seed,
        )

    def _fallback(
        self,
        *,
        row: PlannerRootRow,
        resolved: ResolvedPlannerRuntimeIdentity,
        reason: PlannerFallbackReason,
    ) -> PlannerPolicyDecision:
        identity = resolved.runtime_identity
        budget = self.runtime_config.planner_behavior.constructor.work_budget
        try:
            legal_action_count = describe_prompt_action_space(
                row.observation.get("select")
            ).legal_action_count
        except (TypeError, ValueError):
            legal_action_count = 1
        root_fingerprint = root_observation_fingerprint(row.observation)
        support_fingerprint = hashlib.sha256(
            _FALLBACK_SUPPORT_DOMAIN
            + bytes.fromhex(root_fingerprint)
            + bytes.fromhex(resolved.continuation_semantics_fingerprint)
        ).hexdigest()
        configured = _configured_source_quotas(self.runtime_config)
        metadata = PlannerCollectionMetadata(
            root_information_fingerprint=root_fingerprint,
            scenario_support_fingerprint=support_fingerprint,
            model_fingerprint=identity.model_fingerprint,
            constructor_fingerprint=identity.constructor_fingerprint,
            scorer_fingerprint=identity.scorer_fingerprint,
            controller_fingerprint=identity.controller_fingerprint,
            planner_fingerprint=identity.planner_fingerprint,
            policy_version=identity.policy_version,
            proposal_version=identity.proposal_version,
            constructor_version=identity.constructor_version,
            planner_version=identity.planner_version,
            scenario_support_mode=(
                ScenarioSupportMode.BELIEF_SAMPLED_CHANCE_ENUMERATED
            ),
            legal_action_count=max(1, legal_action_count),
            scenario_count=0,
            support_exhaustive=False,
            scenario_grid_complete=False,
            leaf_bootstrapped=False,
            configured_source_quotas=configured,
            used_source_quotas=(0,) * len(PLANNER_SOURCE_NAMES),
            engine_transition_limit=budget.engine_transition_limit,
            engine_transitions_used=0,
            prefix_node_limit=budget.prefix_node_limit,
            prefix_nodes_used=0,
            wall_clock_limit_ms=budget.wall_clock_limit_ms,
            wall_clock_used_ms=0,
            planner_temperature=self.runtime_config.scoring.planner_temperature,
        )
        return PlannerPolicyDecision(
            action=row.base_action,
            old_logprob=row.base_old_logprob,
            planner_behavior=base_fallback_evidence(
                metadata=metadata,
                reason=reason,
            ),
            used_base_trace=True,
        )


@dataclass(slots=True)
class _SessionCandidateEvaluator(PlannerCandidateEvaluator):
    session: PlannerInferenceSession
    row_index: int
    ordered: bool
    telemetry: PlannerRequestTelemetry
    batch_capacity: int
    evaluation_calls: int = 0

    def evaluate_planner_candidates(
        self,
        *,
        actions: tuple[tuple[int, ...], ...],
        aggregate_features: torch.Tensor,
    ) -> Any:
        started = time.perf_counter()
        try:
            return self.session.evaluate_candidates(
                self.row_index,
                actions=actions,
                aggregate_features=aggregate_features,
                ordered=self.ordered,
            )
        finally:
            self.evaluation_calls += 1
            self.telemetry.record(
                PlannerStageEvent(
                    stage=PlannerStage.CANDIDATE_MODEL,
                    seconds=time.perf_counter() - started,
                    rows=len(actions),
                    batch_capacity=self.batch_capacity,
                )
            )


@dataclass(slots=True)
class _SessionContinuationProvider:
    session: PlannerInferenceSession
    own_deck: tuple[int, ...]

    def select_actions(
        self,
        prompts: tuple[ContinuationPrompt, ...],
    ) -> Sequence[Sequence[int]]:
        return self.session.continuation_actions(
            prompts,
            root_deck=self.own_deck,
        )


def _rollout_root_row(
    row: Any,
    *,
    base_action: Sequence[int],
    base_old_logprob: float,
) -> PlannerRootRow:
    return PlannerRootRow(
        row_id=str(row.game_id),
        seat=int(row.seat),
        observation=cast(Any, row.observation),
        context_features=cast(Any, row.context_features),
        context_snapshot=cast(Any, row.context_snapshot),
        own_deck=tuple(int(value) for value in row.deck_pair[row.seat]),
        base_action=tuple(int(value) for value in base_action),
        base_old_logprob=float(base_old_logprob),
    )


def _default_lease_identity(
    policy: object,
    runtime: ResolvedPlannerRuntimeConfig,
) -> ResolvedPlannerRuntimeIdentity:
    model_fingerprint = getattr(policy, "model_fingerprint", None)
    policy_version = int(getattr(policy, "policy_version", -1))
    proposal_version = int(getattr(policy, "proposal_version", -1))
    if not isinstance(model_fingerprint, str):
        raise RuntimeError("planner policy has no immutable model fingerprint")
    return runtime.resolve_for_lease(
        model_fingerprint=model_fingerprint,
        policy_version=policy_version,
        proposal_version=proposal_version,
    )


def _planner_inference_device_type(
    policy: object,
) -> Literal["cpu", "cuda"]:
    """Return the policy-bound execution device, never the actor input device."""
    device_type = getattr(policy, "planner_inference_device_type", None)
    if device_type not in ("cpu", "cuda"):
        raise RuntimeError(
            "planner policy must bind its inference execution device type"
        )
    return cast(Literal["cpu", "cuda"], device_type)


def _best_effort_resolved_identity(
    policy: object,
    runtime: ResolvedPlannerRuntimeConfig,
    *,
    resolver: PlannerLeaseIdentityResolver,
) -> ResolvedPlannerRuntimeIdentity | None:
    try:
        return resolver(policy, runtime)
    except (RuntimeError, TypeError, ValueError):
        return None


def _bind_retained_contexts(
    policy: object,
    handles: Sequence[str],
    *,
    policy_version: int,
    tensor_schema_fingerprint: str,
    deadline_monotonic: float,
) -> None:
    """Bind direct-policy root handles before their first planner lookup.

    Remote policies have no binding surface because the inference server binds
    their handles before returning the root decode. Direct local policies retain
    an unbound context during decode, so the shared service must attach the exact
    runtime schema before proposal, reranker, or leaf inference can consume it.
    """
    binder = getattr(policy, "bind_planner_context_handles", None)
    if not callable(binder):
        return
    kwargs: dict[str, Any] = {
        "policy_version": int(policy_version),
        "tensor_schema_fingerprint": tensor_schema_fingerprint,
    }
    if callable(getattr(policy, "expire_planner_context_handles", None)):
        kwargs["deadline_monotonic"] = float(deadline_monotonic)
    cast(Any, binder)(handles, **kwargs)


def _release_unplanned_contexts(batch: Any, row_indices: Sequence[int]) -> None:
    if not tuple(batch.planner_context_handles):
        return
    if len(tuple(batch.planner_context_handles)) != len(tuple(batch.rows)):
        raise RuntimeError("planner context handles are row-misaligned")
    handles = tuple(batch.planner_context_handles[index] for index in row_indices)
    if not handles:
        return
    direct = getattr(batch.policy, "release_planner_context_handles", None)
    if callable(direct):
        released = int(direct(handles))
        if released != len(handles):
            raise RuntimeError("unplanned context release count is incomplete")
        return
    scheduled = getattr(batch.policy, "release_planner_contexts_until", None)
    if not callable(scheduled):
        raise RuntimeError("planner policy cannot release unplanned contexts")
    deadline = time.monotonic() + 0.1
    indices = torch.tensor(tuple(row_indices), dtype=torch.long)
    from ptcg_rl.rl.planner_inference_session import select_state_rows

    submit_only = getattr(
        batch.policy,
        "submit_planner_context_release_until",
        None,
    )
    if callable(submit_only):
        submit_only(
            select_state_rows(batch.states, indices),
            batch.decks.select(indices.to(device=batch.decks.card_ids.device)),
            planner_context_handles=handles,
            deadline_monotonic=deadline,
            model_version_lease=int(batch.policy_version),
            tensor_schema_fingerprint=(
                getattr(batch.policy, "planner_tensor_schema_fingerprint", "")
            ),
        )
        return

    released = int(
        scheduled(
            select_state_rows(batch.states, indices),
            batch.decks.select(indices.to(device=batch.decks.card_ids.device)),
            planner_context_handles=handles,
            deadline_monotonic=deadline,
            model_version_lease=int(batch.policy_version),
            tensor_schema_fingerprint=(
                getattr(batch.policy, "planner_tensor_schema_fingerprint", "")
            ),
        )
    )
    if released != len(handles):
        raise RuntimeError("unplanned remote context release count is incomplete")


def release_rollout_planner_batch_contexts(batch: Any) -> None:
    """Release every retained root when post-behavior work is not accepted."""
    _release_unplanned_contexts(batch, tuple(range(len(tuple(batch.rows)))))


def _fallback_reason(exc: Exception) -> PlannerFallbackReason:
    if isinstance(exc, TimeoutError):
        return PlannerFallbackReason.DEADLINE
    if isinstance(exc, (RuntimeError, ValueError, TypeError)):
        message = str(exc).lower()
        if "lease" in message or "version" in message or "fingerprint" in message:
            return PlannerFallbackReason.MODEL_VERSION_MISMATCH
        if "queue" in message:
            return PlannerFallbackReason.QUEUE_FULL
        if "belief" in message or "scenario" in message:
            return PlannerFallbackReason.EVIDENCE_ABSENT
    return PlannerFallbackReason.ENGINE_ERROR


def _effective_planner_deadline(
    runtime: ResolvedPlannerRuntimeConfig,
    *,
    started: float,
    outer_deadline: float | None = None,
) -> float:
    """Resolve one action-critical deadline including the return reserve."""
    deadlines = runtime.deadlines
    budget_seconds = (
        runtime.planner_behavior.constructor.work_budget.wall_clock_limit_ms / 1_000.0
    )
    candidates: list[float] = [
        started + deadlines.request_timeout_seconds - deadlines.return_guard_seconds,
        started + budget_seconds,
    ]
    if outer_deadline is not None:
        candidates.append(float(outer_deadline))
    deadline = float(min(candidates))
    if deadline <= started:
        raise TimeoutError("planner has no foreground deadline budget")
    return deadline


def _requires_fixed_select_probe(
    request: PlannerDecisionRequest,
    runtime: ResolvedPlannerRuntimeConfig,
) -> bool:
    select = request.root_observation.get("select")
    space = describe_prompt_action_space(select)
    context_raw = (
        select.get("context", -1)
        if isinstance(select, Mapping)
        else getattr(select, "context", -1)
    )
    context = -1 if context_raw is None else int(context_raw)
    return bool(
        context != int(SelectContext.MAIN)
        and not space.ordered
        and space.min_count == 1
        and space.max_count == 1
        and 1
        < space.legal_action_count
        <= runtime.planner_behavior.eligibility.single_select_probe_cap
    )


def _configured_source_quotas(
    runtime: ResolvedPlannerRuntimeConfig,
) -> tuple[int, ...]:
    constructor = runtime.planner_behavior.constructor
    seed = constructor.seed_quotas.as_dict()
    expansion = constructor.expansion_quotas.as_dict()
    return tuple(
        int(seed.get(cast(Any, name), expansion.get(cast(Any, name), 0)))
        if name != "exhaustive"
        else 0
        for name in PLANNER_SOURCE_NAMES
    )


def _row_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed & ((1 << 63) - 1))
    return generator


__all__ = [
    "PlannerBehaviorService",
    "PlannerBehaviorServiceStats",
    "PlannerLeaseIdentityResolver",
    "release_rollout_planner_batch_contexts",
]
