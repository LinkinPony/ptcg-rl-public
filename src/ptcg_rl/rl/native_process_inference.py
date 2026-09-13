"""Shared-tensor inference boundary for native rollout worker processes."""

from __future__ import annotations

import cProfile
import os
import queue
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from ptcg_rl.context.public_event_arrays import (
    PublicEventBatch,
    concatenate_public_event_batches,
)
from ptcg_rl.decks.identity import CanonicalDeck, canonicalize_deck
from ptcg_rl.engine.native_public_context import NativeKnownOpponentBatch
from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.engine.native_training_view import (
    concatenate_native_training_views,
)
from ptcg_rl.rl.native_historical_inference import NativeHistoricalPolicyPool
from ptcg_rl.rl.native_policy_bank import NativePolicyInferenceBank
from ptcg_rl.rl.native_policy_batch import (
    NativeSimpleStatelessPolicyBatch,
    concatenate_native_simple_stateless_batches,
    move_native_simple_stateless_batch,
)
from ptcg_rl.rl.native_policy_inference import NativePolicyInferenceExecutor
from ptcg_rl.rl.native_policy_selection import (
    select_native_policy_action_rows,
    select_native_policy_trace_rows,
)
from ptcg_rl.rl.native_policy_trace import (
    NativePolicyNumpyActionBatch,
    NativePolicyNumpyTrace,
)
from ptcg_rl.rl.native_process_batch_scheduler import (
    NativeProcessBatchScheduler,
    NativeProcessBrokerRequest,
)
from ptcg_rl.rl.native_process_client import (
    NativeProcessHistoricalRequest,
    NativeProcessInferenceFailure,
    NativeProcessInferenceRequest,
    NativeProcessInferenceResponse,
    NativeProcessRouteKind,
    NativeProcessSequenceControl,
    NativeProcessSequenceRequest,
    NativeProcessSequenceRows,
    native_sampling_uniforms,
)
from ptcg_rl.rl.native_process_shared_batch import (
    NativeSharedBatchRegistry,
)
from ptcg_rl.rl.native_route_scheduler import NativeRouteKey
from ptcg_rl.rl.native_sequence_runtime import native_trace_from_actor_batch
from ptcg_rl.rl.policy_inputs import (
    SimpleStatelessActorRow,
    collate_simple_stateless_actor_rows,
)
from ptcg_rl.rl.sequence_actor import GeneralistSequenceActorPolicy
from ptcg_rl.rl.sequence_actor_transfer import SequenceActorHostTransfer
from ptcg_rl.rl.sequence_types import SequenceDecisionIdentity
from ptcg_rl.rl.stateless_actor import (
    StatelessActorBatchTrace,
    StatelessActorDecisionTrace,
)
from ptcg_rl.rl.stateless_inference import (
    CURRENT_POLICY_ROUTE,
    StatelessInferenceFailure,
    StatelessInferenceRequest,
    StatelessInferenceResponse,
)

_INTEGRATED_BATCH_WAIT_MULTIPLIER = 4.0
_NATIVE_WORKER_GATHER_MAX_SECONDS = 0.050
_QUEUE_POLL_SECONDS = 0.0002
_BROKER_PROFILE_ENV = "PTCG_RL_NATIVE_BROKER_PROFILE_PATH"
_REPO_TEMP_ROOT = Path(__file__).resolve().parents[3] / "tmp"


def _optional_broker_profiler() -> cProfile.Profile | None:
    """Create an opt-in broker-thread profile under the project temp root."""
    raw_path = os.environ.get(_BROKER_PROFILE_ENV, "").strip()
    if not raw_path:
        return None
    profile_path = Path(raw_path).resolve()
    if not profile_path.is_relative_to(_REPO_TEMP_ROOT.resolve()):
        raise ValueError("native broker profile must be stored under repository tmp")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    return cProfile.Profile()


@dataclass(frozen=True, slots=True)
class _PreparedSequenceBrokerGroup:
    """One exact recurrent route ready for concurrent CUDA submission."""

    kind: NativeProcessRouteKind
    artifact_sha256: str
    requests: tuple[NativeProcessSequenceRequest, ...]
    actor: GeneralistSequenceActorPolicy
    host_batch: NativeSimpleStatelessPolicyBatch
    host_events: PublicEventBatch
    uniforms: Tensor
    row_segments: tuple[tuple[SimpleStatelessActorRow, ...], ...]
    rows: tuple[SimpleStatelessActorRow, ...]
    route: NativeRouteKey


class NativeProcessInferenceBroker:
    """Merge shared worker batches while keeping CUDA in one process."""

    def __init__(
        self,
        *,
        current: NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
        past: Mapping[
            str,
            NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
        ],
        historical: NativeHistoricalPolicyPool,
        request_queue: Any,
        scripted_request_queue: Any | None = None,
        response_queues: Mapping[int, Any],
        scripted_response_queues: Mapping[int, Any] | None = None,
        max_batch_rows: int,
        batch_wait_seconds: float,
        scripted_sampling_seed: int = 0,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        if max_batch_rows <= 0 or batch_wait_seconds < 0.0:
            raise ValueError("native process batch controls are invalid")
        self.current = current
        self.past = dict(past)
        self.historical = historical
        self.request_queue = request_queue
        self.scripted_request_queue = scripted_request_queue
        self.response_queues = dict(response_queues)
        self.scripted_response_queues = dict(scripted_response_queues or {})
        self.max_batch_rows = int(max_batch_rows)
        self.batch_wait_seconds = float(batch_wait_seconds)
        self.cuda_stream = cuda_stream
        self._scripted_generator = torch.Generator(device="cpu")
        self._scripted_generator.manual_seed(int(scripted_sampling_seed))
        self._thread = threading.Thread(
            target=self._serve,
            name="native-process-inference",
            daemon=True,
        )
        self._closed = False
        self._fatal_error: BaseException | None = None
        self.route_batches: Counter[NativeRouteKey] = Counter()
        self.route_rows: Counter[NativeRouteKey] = Counter()
        self.route_seconds: defaultdict[NativeRouteKey, float] = defaultdict(float)
        self.requests = 0
        self.threshold_batches = 0
        self.deadline_batches = 0
        self.unblock_batches = 0
        self.shared_rows = 0
        self.integrated_scripted_requests = 0
        self.integrated_scripted_batches = 0
        self.integrated_scripted_rows = 0
        self.integrated_scripted_mixed_batches = 0
        self.integrated_scripted_mixed_rows = 0
        self._shared_batches = NativeSharedBatchRegistry()
        self._policy_bank = NativePolicyInferenceBank()
        sequence_actors = tuple(
            actor
            for actor in (self.current, *self.past.values())
            if isinstance(actor, GeneralistSequenceActorPolicy)
        )
        self._sequence_decks_by_actor = {
            id(actor): (
                actor,
                {
                    route.signature: canonicalize_deck(route.canonical_card_ids)
                    for route in actor.model.config.exact_routes
                },
            )
            for actor in sequence_actors
        }
        self._sequence_pending: dict[
            tuple[int, str, str, str, int, str],
            tuple[
                GeneralistSequenceActorPolicy,
                SimpleStatelessActorRow,
                StatelessActorDecisionTrace,
            ],
        ] = {}

    @property
    def fatal_error(self) -> BaseException | None:
        """Return a fatal broker exception, if one occurred."""
        return self._fatal_error

    def start(self) -> None:
        """Start the single-owner service thread."""
        self._thread.start()

    def close(self) -> None:
        """Drain already-enqueued requests and stop the service."""
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + 30.0
        while self._thread.is_alive():
            try:
                self.request_queue.put_nowait(None)
                break
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "native process inference stop signal remained blocked"
                    ) from None
                self._thread.join(timeout=0.05)
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            raise TimeoutError("native process inference broker did not stop")
        self._policy_bank.close()

    def _serve(self) -> None:
        profiler: cProfile.Profile | None = None
        try:
            profiler = _optional_broker_profiler()
            if profiler is not None:
                profiler.enable()
            if self.cuda_stream is None:
                self._serve_requests()
            else:
                with torch.cuda.stream(self.cuda_stream):
                    self._serve_requests()
        except BaseException as error:
            self._fatal_error = error
            message = f"{type(error).__name__}: {error}"
            for response_queue in self.response_queues.values():
                with suppress(queue.Full):
                    response_queue.put_nowait(
                        NativeProcessInferenceFailure(error=message)
                    )
            for response_queue in self.scripted_response_queues.values():
                with suppress(queue.Full):
                    response_queue.put_nowait(
                        StatelessInferenceFailure(error=message)
                    )
        finally:
            if profiler is not None:
                with suppress(BaseException):
                    profiler.disable()
                    profiler.dump_stats(os.environ[_BROKER_PROFILE_ENV])

    def _serve_requests(self) -> None:
        """Serve route-aware batches without blocking closed-loop feeders."""
        scheduler = NativeProcessBatchScheduler(
            worker_ids=self.response_queues,
            max_batch_rows=self.max_batch_rows,
            batch_wait_seconds=self.batch_wait_seconds,
            current_wait_multiplier=_INTEGRATED_BATCH_WAIT_MULTIPLIER,
            current_gather_max_seconds=_NATIVE_WORKER_GATHER_MAX_SECONDS,
        )
        stop_requested = False
        while scheduler.has_pending or not stop_requested:
            if not scheduler.has_pending:
                item = self._get_request(timeout=None)
                stop_requested = self._accept_item(
                    item,
                    scheduler=scheduler,
                )
                continue

            while not stop_requested:
                try:
                    item = self._get_request(timeout=0.0)
                except queue.Empty:
                    break
                stop_requested = self._accept_item(
                    item,
                    scheduler=scheduler,
                )

            now = time.monotonic()
            selection = scheduler.select(
                now=now,
                stop_requested=stop_requested,
            )
            if selection is not None:
                if selection.reason == "threshold":
                    self.threshold_batches += 1
                elif selection.reason == "unblock":
                    self.unblock_batches += 1
                else:
                    self.deadline_batches += 1
                self._serve_batch(selection.requests)
                continue

            timeout = scheduler.wait_seconds(now=now)
            try:
                item = self._get_request(timeout=timeout)
            except queue.Empty:
                continue
            stop_requested = self._accept_item(
                item,
                scheduler=scheduler,
            )

    def _accept_item(
        self,
        item: object,
        *,
        scheduler: NativeProcessBatchScheduler,
    ) -> bool:
        """Process one transport item and return whether shutdown was seen."""
        if item is None:
            return True
        if isinstance(item, NativeProcessSequenceControl):
            control_started_at = time.monotonic()
            self._serve_sequence_control(item)
            control_seconds = time.monotonic() - control_started_at
            scheduler.delay(control_seconds)
            return False
        request = self._validated_request(item)
        scheduler.add(request, arrived_at=time.monotonic())
        return False

    def _get_request(self, *, timeout: float | None) -> object:
        """Prefer descriptor traffic without starving scripted actor rows."""
        deadline = None if timeout is None else time.monotonic() + timeout
        queues = (
            (self.request_queue, self.scripted_request_queue)
            if self.scripted_request_queue is not None
            else (self.request_queue,)
        )
        while True:
            for request_queue in queues:
                if request_queue is None:
                    continue
                try:
                    return request_queue.get_nowait()
                except queue.Empty:
                    continue
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise queue.Empty
                time.sleep(min(_QUEUE_POLL_SECONDS, remaining))
            else:
                time.sleep(_QUEUE_POLL_SECONDS)

    def _validated_request(
        self,
        value: object,
    ) -> (
        NativeProcessInferenceRequest
        | NativeProcessHistoricalRequest
        | NativeProcessSequenceRequest
        | StatelessInferenceRequest
    ):
        if not isinstance(
            value,
            (
                NativeProcessInferenceRequest,
                NativeProcessHistoricalRequest,
                NativeProcessSequenceRequest,
                StatelessInferenceRequest,
            ),
        ):
            raise TypeError("native process inference request is invalid")
        if isinstance(value, StatelessInferenceRequest):
            if value.actor_index not in self.scripted_response_queues:
                raise ValueError("integrated scripted inference actor is unknown")
            if value.policy_route != CURRENT_POLICY_ROUTE:
                raise ValueError(
                    "integrated scripted inference accepts only current policy"
                )
            self.integrated_scripted_requests += 1
            return value
        if value.worker_index not in self.response_queues:
            raise ValueError("native process inference worker is unknown")
        self.requests += 1
        return value

    def _serve_batch(
        self,
        requests: Sequence[NativeProcessBrokerRequest],
    ) -> None:
        routed: defaultdict[
            tuple[str, str],
            list[NativeProcessInferenceRequest],
        ] = defaultdict(list)
        historical: list[NativeProcessHistoricalRequest] = []
        sequence: list[NativeProcessSequenceRequest] = []
        scripted: list[StatelessInferenceRequest] = []
        for request in requests:
            if isinstance(request, NativeProcessHistoricalRequest):
                historical.append(request)
            elif isinstance(request, NativeProcessSequenceRequest):
                sequence.append(request)
            elif isinstance(request, StatelessInferenceRequest):
                scripted.append(request)
            else:
                routed[(request.route_kind, request.artifact_sha256)].append(request)
        current = [
            request
            for (kind, _artifact), selected in routed.items()
            if kind == "current"
            for request in selected
        ]
        temperatures = {request.temperature for request in current}
        temperatures.update(request.temperature for request in scripted)
        for temperature in sorted(temperatures):
            self._serve_current_group(
                tuple(
                    request
                    for request in current
                    if request.temperature == temperature
                ),
                tuple(
                    request
                    for request in scripted
                    if request.temperature == temperature
                ),
            )
        for (kind, artifact), selected in sorted(routed.items()):
            if kind == "current":
                continue
            self._serve_policy_group(
                cast(NativeProcessRouteKind, kind),
                artifact,
                selected,
            )
        if historical:
            self._serve_historical(historical)
        sequence_groups: defaultdict[
            tuple[str, str, float],
            list[NativeProcessSequenceRequest],
        ] = defaultdict(list)
        for request in sequence:
            sequence_groups[
                (request.route_kind, request.artifact_sha256, request.temperature)
            ].append(request)
        if sequence_groups:
            self._serve_sequence_groups(
                tuple(
                    self._prepare_sequence_group(
                        cast(NativeProcessRouteKind, kind),
                        artifact,
                        grouped,
                    )
                    for (kind, artifact, _temperature), grouped in sorted(
                        sequence_groups.items()
                    )
                )
            )

    def _serve_current_group(
        self,
        native_requests: Sequence[NativeProcessInferenceRequest],
        scripted_requests: Sequence[StatelessInferenceRequest],
    ) -> None:
        """Run native and Python-engine current rows in one model forward."""
        if isinstance(self.current, GeneralistSequenceActorPolicy):
            raise TypeError("stateless requests require a stateless current actor")
        expected_artifact = self.current.identity.behavior_policy_fingerprint
        if any(
            request.artifact_sha256 != expected_artifact
            for request in native_requests
        ):
            raise ValueError("native current request crossed behavior identity")
        native_resolved = tuple(
            self._shared_batches.resolve(
                request.worker_index,
                request.shared_batch,
            )
            for request in native_requests
        )
        batches = [batch for batch, _uniforms, _events in native_resolved]
        uniforms = [
            cast(Tensor, sampling_uniforms)
            for _batch, sampling_uniforms, _events in native_resolved
        ]
        scripted_batch: NativeSimpleStatelessPolicyBatch | None = None
        if scripted_requests:
            scripted_batch = _collate_scripted_requests(scripted_requests)
            batches.append(scripted_batch)
            uniforms.append(
                native_sampling_uniforms(
                    scripted_batch,
                    generator=self._scripted_generator,
                )
            )
        if not batches:
            raise ValueError("current inference group requires rows")
        host_batch = concatenate_native_simple_stateless_batches(tuple(batches))
        merged_uniforms = _concatenate_uniforms(tuple(uniforms))
        started_at = time.perf_counter()
        device_batch = move_native_simple_stateless_batch(
            host_batch,
            device=self.current.device,
            non_blocking=self.current.device.type == "cuda",
        )
        result = self.current.sample(
            device_batch,
            sampling_uniforms=merged_uniforms.to(
                device=self.current.device,
                non_blocking=self.current.device.type == "cuda",
            ),
            temperature=(
                native_requests[0].temperature
                if native_requests
                else scripted_requests[0].temperature
            ),
        )
        route = NativeRouteKey(kind="current", artifact_sha256=expected_artifact)
        self.route_seconds[route] += time.perf_counter() - started_at
        self.route_batches[route] += 1
        self.route_rows[route] += host_batch.batch_size
        self.shared_rows += sum(
            batch.batch_size for batch, _uniforms, _events in native_resolved
        )
        offset = 0
        for request, (batch, _uniforms, _events) in zip(
            native_requests,
            native_resolved,
            strict=True,
        ):
            stop = offset + batch.batch_size
            trace = select_native_policy_trace_rows(
                result,
                np.arange(offset, stop, dtype=np.int64),
            )
            self.response_queues[request.worker_index].put(
                NativeProcessInferenceResponse(
                    request_id=request.request_id,
                    trace=trace,
                )
            )
            offset = stop
        if scripted_batch is not None:
            self.integrated_scripted_batches += 1
            self.integrated_scripted_rows += scripted_batch.batch_size
            if native_requests:
                self.integrated_scripted_mixed_batches += 1
                self.integrated_scripted_mixed_rows += scripted_batch.batch_size
            scripted_trace = select_native_policy_trace_rows(
                result,
                np.arange(
                    offset,
                    offset + scripted_batch.batch_size,
                    dtype=np.int64,
                ),
            )
            decisions = _stateless_decisions(scripted_trace)
            decision_offset = 0
            for scripted_request in scripted_requests:
                stop = decision_offset + len(scripted_request.rows)
                self.scripted_response_queues[scripted_request.actor_index].put(
                    StatelessInferenceResponse(
                        request_id=scripted_request.request_id,
                        trace=StatelessActorBatchTrace(
                            behavior_policy_version=(
                                scripted_trace.behavior_policy_version
                            ),
                            behavior_policy_fingerprint=(
                                scripted_trace.behavior_policy_fingerprint
                            ),
                            input_contract_fingerprint=(
                                scripted_trace.input_contract_fingerprint
                            ),
                            decisions=decisions[decision_offset:stop],
                        ),
                    )
                )
                decision_offset = stop

    def _prepare_sequence_group(
        self,
        kind: NativeProcessRouteKind,
        artifact_sha256: str,
        requests: Sequence[NativeProcessSequenceRequest],
    ) -> _PreparedSequenceBrokerGroup:
        """Merge recurrent worker rows without serializing their CUDA routes."""
        actor = self.current if kind == "current" else self.past[artifact_sha256]
        if not isinstance(actor, GeneralistSequenceActorPolicy):
            raise TypeError("sequence requests require a generalist sequence actor")
        if any(request.temperature != requests[0].temperature for request in requests):
            raise ValueError("native sequence batch mixes temperatures")
        if kind == "current" and artifact_sha256 != (
            actor.identity.behavior_policy_fingerprint
        ):
            raise ValueError("native sequence current request crossed behavior identity")
        resolved = tuple(
            self._shared_batches.resolve(
                request.worker_index,
                request.shared_batch,
            )
            for request in requests
        )
        batches = tuple(batch for batch, _uniforms, _events in resolved)
        if any(uniforms is None for _batch, uniforms, _events in resolved):
            raise ValueError("native sequence request omitted sampling uniforms")
        if any(events is None for _batch, _uniforms, events in resolved):
            raise ValueError("native sequence request omitted public events")
        host_batch = concatenate_native_simple_stateless_batches(batches)
        host_events = concatenate_public_event_batches(
            tuple(
                cast(PublicEventBatch, events)
                for _batch, _uniforms, events in resolved
            )
        )
        uniforms = _concatenate_uniforms(
            tuple(
                cast(Tensor, sampling_uniforms)
                for _batch, sampling_uniforms, _events in resolved
            )
        )
        deck_owner, decks_by_signature = self._sequence_decks_by_actor[id(actor)]
        if deck_owner is not actor:
            raise RuntimeError("native sequence deck cache changed actor owners")
        row_segments = tuple(
            _sequence_actor_rows(decks_by_signature, request.rows, batch)
            for request, batch in zip(requests, batches, strict=True)
        )
        rows = tuple(row for segment in row_segments for row in segment)
        return _PreparedSequenceBrokerGroup(
            kind=kind,
            artifact_sha256=artifact_sha256,
            requests=tuple(requests),
            actor=actor,
            host_batch=host_batch,
            host_events=host_events,
            uniforms=uniforms,
            row_segments=row_segments,
            rows=rows,
            route=NativeRouteKey(kind=kind, artifact_sha256=artifact_sha256),
        )

    def _serve_sequence_groups(
        self,
        groups: Sequence[_PreparedSequenceBrokerGroup],
    ) -> None:
        """Overlap exact recurrent routes on persistent current/frozen streams."""
        with self._policy_bank.wave():
            for group in groups:
                started_at = time.perf_counter()
                actor = group.actor
                with self._policy_bank.route_stream(
                    group.route,
                    device=actor.device,
                ):
                    device_batch = move_native_simple_stateless_batch(
                        group.host_batch,
                        device=actor.device,
                        non_blocking=actor.device.type == "cuda",
                        move_exact_deck_tensors=False,
                    )
                    continuation = actor.begin_preencoded_deferred(
                        group.rows,
                        device_batch,
                        temperature=group.requests[0].temperature,
                        sampling_uniforms=group.uniforms.to(
                            device=actor.device,
                            non_blocking=actor.device.type == "cuda",
                        ),
                        copy_stream=self._policy_bank.copy_stream(actor.device),
                        public_event_batch=group.host_events,
                        host_semantic_batch=group.host_batch,
                        materialize_host_actions=False,
                    )
                    transfer = continuation.resume(await_option_features=None)
                self._policy_bank.defer(
                    partial(
                        self._complete_sequence_transfer,
                        group,
                        transfer,
                        started_at=started_at,
                    ),
                    on_abort=transfer.cancel,
                )

    def _complete_sequence_transfer(
        self,
        group: _PreparedSequenceBrokerGroup,
        transfer: SequenceActorHostTransfer,
        *,
        started_at: float,
    ) -> None:
        """Finish one recurrent host transfer and publish its response rows."""
        trace, native_trace = transfer.finish_native_ready()
        self._complete_sequence_group(
            group,
            trace,
            native_trace=(
                native_trace_from_actor_batch(group.actor.identity, trace)
                if native_trace is None
                else native_trace
            ),
            started_at=started_at,
        )

    def _complete_sequence_group(
        self,
        group: _PreparedSequenceBrokerGroup,
        trace: StatelessActorBatchTrace,
        *,
        native_trace: NativePolicyNumpyTrace,
        started_at: float,
    ) -> None:
        """Publish one synchronized recurrent route to its worker FIFOs."""
        actor = group.actor
        requests = group.requests
        row_segments = group.row_segments
        kind = group.kind
        artifact_sha256 = group.artifact_sha256
        self.route_seconds[group.route] += time.perf_counter() - started_at
        self.route_batches[group.route] += 1
        self.route_rows[group.route] += group.host_batch.batch_size
        self.shared_rows += group.host_batch.batch_size
        offset = 0
        for request, segment in zip(requests, row_segments, strict=True):
            stop = offset + len(segment)
            decisions = trace.decisions[offset:stop]
            compact_trace = select_native_policy_trace_rows(
                native_trace,
                np.arange(offset, stop, dtype=np.int64),
            )
            for row, decision in zip(segment, decisions, strict=True):
                identity = _required_sequence_identity(row)
                key = _sequence_pending_key(
                    request.worker_index,
                    kind,
                    artifact_sha256,
                    identity,
                )
                if key in self._sequence_pending:
                    raise RuntimeError("native sequence proposal key was duplicated")
                self._sequence_pending[key] = (actor, row, decision)
            self.response_queues[request.worker_index].put(
                NativeProcessInferenceResponse(
                    request_id=request.request_id,
                    trace=compact_trace,
                    sequence_trace=StatelessActorBatchTrace(
                        behavior_policy_version=trace.behavior_policy_version,
                        behavior_policy_fingerprint=(
                            trace.behavior_policy_fingerprint
                        ),
                        input_contract_fingerprint=(
                            trace.input_contract_fingerprint
                        ),
                        decisions=decisions,
                    ),
                )
            )
            offset = stop

    def _serve_sequence_control(
        self,
        control: NativeProcessSequenceControl,
    ) -> None:
        """Publish engine outcomes before the worker's next FIFO request."""
        if control.worker_index not in self.response_queues:
            raise ValueError("native sequence control worker is unknown")
        actor = (
            self.current
            if control.route_kind == "current"
            else self.past[control.artifact_sha256]
        )
        if not isinstance(actor, GeneralistSequenceActorPolicy):
            raise TypeError("native sequence control resolved a stateless actor")
        for resolution in control.resolutions:
            key = _sequence_pending_key(
                control.worker_index,
                control.route_kind,
                control.artifact_sha256,
                resolution.identity,
            )
            try:
                pending_actor, row, trace = self._sequence_pending.pop(key)
            except KeyError as error:
                raise RuntimeError(
                    "native sequence control has no pending proposal"
                ) from error
            if pending_actor is not actor:
                raise RuntimeError("native sequence proposal changed actor owners")
            if resolution.commit:
                actor.commit_decision(row, trace)
            else:
                actor.abort_decision(row, trace)
        for game_id, seat in control.releases:
            actor.release_game(game_id=game_id, seat=seat)
        if control.barrier_request_id is not None:
            self.response_queues[control.worker_index].put(
                NativeProcessInferenceResponse(
                    request_id=control.barrier_request_id,
                )
            )

    def _serve_policy_group(
        self,
        kind: NativeProcessRouteKind,
        artifact_sha256: str,
        requests: Sequence[NativeProcessInferenceRequest],
    ) -> None:
        if any(request.temperature != requests[0].temperature for request in requests):
            raise ValueError("native process batch mixes temperatures")
        if kind == "current":
            raise RuntimeError("current routes use the integrated inference path")
        executor = self.past[artifact_sha256]
        if isinstance(executor, GeneralistSequenceActorPolicy):
            raise TypeError("stateless request resolved a sequence past actor")
        resolved = tuple(
            self._shared_batches.resolve(
                request.worker_index,
                request.shared_batch,
            )
            for request in requests
        )
        host_batch = concatenate_native_simple_stateless_batches(
            tuple(batch for batch, _uniforms, _events in resolved)
        )
        uniforms = _concatenate_uniforms(
            tuple(
                cast(Tensor, sampling_uniforms)
                for _batch, sampling_uniforms, _events in resolved
            )
        )
        started_at = time.perf_counter()
        device_batch = move_native_simple_stateless_batch(
            host_batch,
            device=executor.device,
            non_blocking=executor.device.type == "cuda",
        )
        device_uniforms = uniforms.to(
            device=executor.device,
            non_blocking=executor.device.type == "cuda",
        )
        result: NativePolicyNumpyActionBatch = executor.sample_actions(
            device_batch,
            sampling_uniforms=device_uniforms,
            temperature=requests[0].temperature,
        )
        route = NativeRouteKey(kind=kind, artifact_sha256=artifact_sha256)
        self.route_seconds[route] += time.perf_counter() - started_at
        self.route_batches[route] += 1
        self.route_rows[route] += host_batch.batch_size
        self.shared_rows += host_batch.batch_size
        offset = 0
        for request, (batch, _uniforms, _events) in zip(
            requests,
            resolved,
            strict=True,
        ):
            stop = offset + batch.batch_size
            rows = np.arange(offset, stop, dtype=np.int64)
            response = NativeProcessInferenceResponse(
                request_id=request.request_id,
                action_batch=select_native_policy_action_rows(
                    result,
                    rows,
                ),
            )
            self.response_queues[request.worker_index].put(response)
            offset = stop

    def _serve_historical(
        self,
        requests: Sequence[NativeProcessHistoricalRequest],
    ) -> None:
        for artifact_sha256, grouped in _group_historical(requests).items():
            batches = tuple(
                self._shared_batches.resolve(
                    request.worker_index,
                    request.shared_batch,
                )[0]
                for request in grouped
            )
            host_batch = concatenate_native_simple_stateless_batches(batches)
            views = _rebased_views(tuple(request.view for request in grouped))
            view = concatenate_native_training_views(views)
            known = _concatenate_known(tuple(request.known for request in grouped))
            member_ids = tuple(
                member_id for request in grouped for member_id in request.member_ids
            )
            started_at = time.perf_counter()
            actions = self.historical.act_many_preencoded(
                view,
                host_batch.states,
                host_batch.options,
                known,
                member_ids=member_ids,
                deck_signatures=host_batch.deck_signatures,
                model_encoding_fingerprint=grouped[0].model_encoding_fingerprint,
            )
            route = NativeRouteKey("historical", artifact_sha256)
            self.route_seconds[route] += time.perf_counter() - started_at
            self.route_batches[route] += 1
            self.route_rows[route] += view.batch_size
            self.shared_rows += view.batch_size
            offset = 0
            for request, batch in zip(grouped, batches, strict=True):
                stop = offset + batch.batch_size
                self.response_queues[request.worker_index].put(
                    NativeProcessInferenceResponse(
                        request_id=request.request_id,
                        historical_actions=actions[offset:stop],
                    )
                )
                offset = stop


def _concatenate_uniforms(uniforms: Sequence[Tensor]) -> Tensor:
    width = max(int(value.shape[1]) for value in uniforms)
    padded = [
        torch.nn.functional.pad(
            value,
            (0, width - int(value.shape[1])),
            value=0.5,
        )
        for value in uniforms
    ]
    return torch.cat(tuple(padded), dim=0)


def _sequence_actor_rows(
    decks_by_signature: Mapping[str, CanonicalDeck],
    metadata: NativeProcessSequenceRows,
    batch: NativeSimpleStatelessPolicyBatch,
) -> tuple[SimpleStatelessActorRow, ...]:
    """Rebuild only metadata used by the central transactional actor."""
    if not (
        metadata.batch_size
        == batch.batch_size
        == len(metadata.exact_deck_digests)
        == len(metadata.engine_fact_producer_fingerprints)
    ):
        raise ValueError("native sequence metadata is misaligned")
    rows: list[SimpleStatelessActorRow] = []
    for index, identity in enumerate(metadata.identities):
        signature = batch.deck_signatures[index]
        try:
            deck = decks_by_signature[signature]
        except KeyError as error:
            raise ValueError("native sequence deck route is absent") from error
        if deck.deck_digest != metadata.exact_deck_digests[index]:
            raise ValueError("native sequence exact deck identity changed")
        rows.append(
            SimpleStatelessActorRow(
                state=cast(Any, None),
                options=cast(Any, None),
                min_count=batch.min_counts[index],
                max_count=batch.max_counts[index],
                own_deck=deck,
                belief_summary=cast(Any, None),
                catalog_fingerprint=batch.public_deck_catalog_fingerprint,
                input_contract_fingerprint=batch.input_contract_fingerprint,
                engine_fact_producer_fingerprint=(
                    metadata.engine_fact_producer_fingerprints[index]
                ),
                sequence_identity=identity,
            )
        )
    return tuple(rows)


def _required_sequence_identity(
    row: SimpleStatelessActorRow,
) -> SequenceDecisionIdentity:
    identity = row.sequence_identity
    if identity is None:
        raise ValueError("native sequence row has no absolute coordinates")
    return identity


def _sequence_pending_key(
    worker_index: int,
    route_kind: NativeProcessRouteKind,
    artifact_sha256: str,
    identity: SequenceDecisionIdentity,
) -> tuple[int, str, str, str, int, str]:
    return (
        int(worker_index),
        route_kind,
        artifact_sha256,
        identity.game_id,
        int(identity.seat),
        identity.request_id,
    )


def _collate_scripted_requests(
    requests: Sequence[StatelessInferenceRequest],
) -> NativeSimpleStatelessPolicyBatch:
    rows = tuple(row for request in requests for row in request.rows)
    batch = collate_simple_stateless_actor_rows(rows, device="cpu")
    return NativeSimpleStatelessPolicyBatch(
        states=batch.states,
        options=batch.options,
        unique_deck_card_ids=batch.unique_deck_card_ids,
        deck_counts=batch.deck_counts,
        deck_valid_mask=batch.deck_valid_mask,
        deck_signatures=batch.deck_signatures,
        belief_summary=batch.belief_summary,
        min_counts=batch.min_counts,
        max_counts=batch.max_counts,
        public_deck_catalog_fingerprint=rows[0].catalog_fingerprint,
        input_contract_fingerprint=batch.input_contract_fingerprint,
    )


def _stateless_decisions(
    trace: NativePolicyNumpyTrace,
) -> tuple[StatelessActorDecisionTrace, ...]:
    decisions: list[StatelessActorDecisionTrace] = []
    for row in range(trace.batch_size):
        action_start = int(trace.action_offsets[row])
        action_stop = int(trace.action_offsets[row + 1])
        token_start = int(trace.token_offsets[row])
        token_stop = int(trace.token_offsets[row + 1])
        decisions.append(
            StatelessActorDecisionTrace(
                action=tuple(
                    int(value)
                    for value in trace.action_choices[
                        action_start:action_stop
                    ].tolist()
                ),
                action_logprob=float(trace.action_logprobs[row]),
                token_logprobs=tuple(
                    float(value)
                    for value in trace.token_logprobs[
                        token_start:token_stop
                    ].tolist()
                ),
                prefix_values=tuple(
                    float(value)
                    for value in trace.prefix_values[
                        token_start:token_stop
                    ].tolist()
                ),
                root_value=float(trace.root_values[row]),
                stop_sampled=bool(trace.stop_sampled[row]),
            )
        )
    return tuple(decisions)


def _group_historical(
    requests: Sequence[NativeProcessHistoricalRequest],
) -> dict[str, list[NativeProcessHistoricalRequest]]:
    grouped: defaultdict[str, list[NativeProcessHistoricalRequest]] = defaultdict(list)
    for request in requests:
        grouped[request.artifact_sha256].append(request)
    return dict(grouped)


def _rebased_views(
    views: Sequence[NativeTrainingBatchView],
) -> tuple[NativeTrainingBatchView, ...]:
    result: list[NativeTrainingBatchView] = []
    next_slot = 0
    for view in views:
        stop = next_slot + view.batch_size
        result.append(
            replace(
                view,
                slots=np.arange(next_slot, stop, dtype=np.uint32),
            )
        )
        next_slot = stop
    return tuple(result)


def _concatenate_known(
    batches: Sequence[NativeKnownOpponentBatch],
) -> NativeKnownOpponentBatch:
    lengths = np.concatenate(
        tuple(np.diff(batch.offsets.astype(np.int64, copy=False)) for batch in batches)
    )
    offsets = np.zeros(lengths.size + 1, dtype=np.uint32)
    np.cumsum(lengths, dtype=np.uint32, out=offsets[1:])
    return NativeKnownOpponentBatch(
        offsets=offsets,
        card_ids=np.concatenate(tuple(batch.card_ids for batch in batches)),
        counts=np.concatenate(tuple(batch.counts for batch in batches)),
    )


__all__ = [
    "NativeProcessInferenceBroker",
]
