"""Worker-side recurrent proxy for one process-owned CUDA sequence actor."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from ptcg_rl.context.public_event_arrays import PublicEventBatch
from ptcg_rl.model.sequence.host_action import (
    build_tensor_host_accepted_action_records,
)
from ptcg_rl.rl.native_policy_batch import NativeSimpleStatelessPolicyBatch
from ptcg_rl.rl.native_policy_inference import PolicyInputBatch
from ptcg_rl.rl.native_process_client import (
    NativeProcessRouteKind,
    NativeProcessSequenceControl,
    NativeProcessSequenceRequest,
    NativeProcessSequenceResolution,
    NativeProcessSequenceRows,
    native_sampling_uniforms,
    receive_native_process_response,
)
from ptcg_rl.rl.native_process_shared_batch import NativeSharedBatchWriter
from ptcg_rl.rl.policy_inputs import SimpleStatelessActorRow
from ptcg_rl.rl.sequence_actor import (
    GeneralistSequenceActorPolicy,
    SequenceActorSampleContinuation,
    SequenceRolloutPrecision,
)
from ptcg_rl.rl.sequence_actor_transfer import SequenceActorHostTransfer
from ptcg_rl.rl.sequence_types import SequenceDecisionIdentity
from ptcg_rl.rl.stateless_actor import (
    StatelessActorBatchTrace,
    StatelessActorDecisionTrace,
)
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity


class RemoteGeneralistSequenceActorPolicy(GeneralistSequenceActorPolicy):
    """Preserve native sequence semantics while CUDA lives in the parent."""

    def __init__(
        self,
        *,
        worker_index: int,
        identity: StatelessFragmentIdentity,
        route_kind: NativeProcessRouteKind,
        artifact_sha256: str,
        request_queue: Any,
        response_queue: Any,
        timeout_seconds: float,
        shared_batch_writer: NativeSharedBatchWriter,
        rollout_precision: SequenceRolloutPrecision,
        temporal_cache_slots: int | None,
    ) -> None:
        """Bind a CPU transport façade without constructing a local model."""
        if timeout_seconds <= 0.0:
            raise ValueError("native process inference timeout must be positive")
        self.worker_index = int(worker_index)
        self.identity = identity
        self.route_kind = route_kind
        self.artifact_sha256 = artifact_sha256
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout_seconds = float(timeout_seconds)
        self.shared_batch_writer = shared_batch_writer
        self.device = torch.device("cpu")
        self.rollout_precision = rollout_precision
        self.temporal_cache_slot_capacity = temporal_cache_slots
        self.retain_raw_blocks = False
        self._request_id = 0
        self._resolutions: list[NativeProcessSequenceResolution] = []
        self._releases: list[tuple[str, int]] = []
        self._closed = False

    def begin_preencoded_deferred(
        self,
        rows: Sequence[SimpleStatelessActorRow],
        batch: PolicyInputBatch,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        sampling_uniforms: Tensor | None = None,
        copy_stream: Any | None = None,
        public_event_batch: PublicEventBatch | None = None,
        host_semantic_batch: NativeSimpleStatelessPolicyBatch | None = None,
        materialize_host_actions: bool = True,
        evaluation_action_only: bool = False,
    ) -> SequenceActorSampleContinuation:
        """Defer the shared request until local option facts are complete."""
        del copy_stream, materialize_host_actions
        if evaluation_action_only:
            raise ValueError("remote training actor cannot use evaluation action-only")
        if self._closed:
            raise RuntimeError("remote sequence actor is closed")
        if host_semantic_batch is None:
            raise ValueError("remote sequence actor requires its host semantic batch")
        if batch.batch_size != host_semantic_batch.batch_size:
            raise ValueError("remote sequence device and semantic batches differ")
        if public_event_batch is None:
            raise ValueError("remote sequence actor requires public event tensors")
        actor_rows = tuple(rows)
        if len(actor_rows) != host_semantic_batch.batch_size:
            raise ValueError("remote sequence actor rows are misaligned")
        if sampling_uniforms is None:
            if generator is None:
                raise ValueError("remote sequence actor requires a worker RNG")
            sampling_uniforms = native_sampling_uniforms(
                host_semantic_batch,
                generator=generator,
            )
        if sampling_uniforms.device.type != "cpu":
            raise ValueError("remote sequence sampling uniforms must be on CPU")

        def resume(
            await_option_features: Callable[[], None] | None,
        ) -> SequenceActorHostTransfer:
            if await_option_features is not None:
                await_option_features()
            trace = self._request(
                actor_rows,
                host_semantic_batch,
                public_event_batch=public_event_batch,
                sampling_uniforms=sampling_uniforms,
                temperature=temperature,
            )
            return SequenceActorHostTransfer.completed(trace)

        return SequenceActorSampleContinuation(_resume_callback=resume)

    def commit_decision(
        self,
        row: SimpleStatelessActorRow,
        trace: StatelessActorDecisionTrace,
    ) -> None:
        """Buffer an engine-accepted proposal for one FIFO control flush."""
        self._buffer_resolution(row, trace, commit=True)

    def abort_decision(
        self,
        row: SimpleStatelessActorRow,
        trace: StatelessActorDecisionTrace,
    ) -> None:
        """Buffer an engine-rejected proposal for one FIFO control flush."""
        self._buffer_resolution(row, trace, commit=False)

    def release_game(self, *, game_id: str, seat: int) -> None:
        """Queue terminal cache release behind preceding resolutions."""
        self._releases.append((game_id, int(seat)))

    def close(self) -> None:
        """Publish all final transactions without owning parent CUDA state."""
        if self._closed:
            return
        self._flush_control(barrier=True)
        self._closed = True

    def _request(
        self,
        rows: tuple[SimpleStatelessActorRow, ...],
        batch: NativeSimpleStatelessPolicyBatch,
        *,
        public_event_batch: PublicEventBatch,
        sampling_uniforms: Tensor,
        temperature: float,
    ) -> StatelessActorBatchTrace:
        self._flush_control()
        request_id = self._request_id
        self._request_id += 1
        descriptor = self.shared_batch_writer.write(
            batch,
            sampling_uniforms=sampling_uniforms,
            public_events=public_event_batch,
        )
        self.request_queue.put(
            NativeProcessSequenceRequest(
                worker_index=self.worker_index,
                request_id=request_id,
                route_kind=self.route_kind,
                artifact_sha256=self.artifact_sha256,
                shared_batch=descriptor,
                rows=NativeProcessSequenceRows(
                    identities=tuple(
                        _required_sequence_identity(row) for row in rows
                    ),
                    exact_deck_digests=tuple(
                        row.own_deck.deck_digest for row in rows
                    ),
                    engine_fact_producer_fingerprints=tuple(
                        row.engine_fact_producer_fingerprint for row in rows
                    ),
                ),
                temperature=float(temperature),
            ),
            timeout=self.timeout_seconds,
        )
        response = receive_native_process_response(
            self.response_queue,
            request_id=request_id,
            timeout_seconds=self.timeout_seconds,
        )
        if response.sequence_trace is None:
            raise RuntimeError("native process inference omitted its sequence trace")
        if response.trace is None:
            raise RuntimeError("native process inference omitted its compact trace")
        if response.sequence_trace.behavior_policy_fingerprint != (
            self.identity.behavior_policy_fingerprint
        ):
            raise RuntimeError("native process sequence identity changed")
        if response.trace.identity != self.identity:
            raise RuntimeError("native process compact trace identity changed")
        if response.trace.batch_size != len(rows):
            raise RuntimeError("native process compact trace rows are misaligned")
        if len(response.sequence_trace.decisions) != len(rows):
            raise RuntimeError("native process sequence trace rows are misaligned")
        if self.route_kind != "current":
            return response.sequence_trace
        accepted_actions = build_tensor_host_accepted_action_records(
            states=batch.states,
            options=batch.options,
            action_offsets=response.trace.action_offsets,
            action_choices=response.trace.action_choices,
            min_counts=batch.min_counts,
            max_counts=batch.max_counts,
            stop_sampled=response.trace.stop_sampled,
        )
        return replace(
            response.sequence_trace,
            decisions=tuple(
                replace(decision, accepted_action=accepted_action)
                for decision, accepted_action in zip(
                    response.sequence_trace.decisions,
                    accepted_actions,
                    strict=True,
                )
            ),
        )

    def _buffer_resolution(
        self,
        row: SimpleStatelessActorRow,
        trace: StatelessActorDecisionTrace,
        *,
        commit: bool,
    ) -> None:
        identity = _required_sequence_identity(row)
        if trace.sequence_request_id != identity.request_id:
            raise ValueError("remote sequence resolution crossed request identities")
        self._resolutions.append(
            NativeProcessSequenceResolution(identity=identity, commit=commit)
        )

    def _flush_control(self, *, barrier: bool = False) -> None:
        if not self._resolutions and not self._releases and not barrier:
            return
        barrier_request_id = None
        if barrier:
            barrier_request_id = self._request_id
            self._request_id += 1
        message = NativeProcessSequenceControl(
            worker_index=self.worker_index,
            route_kind=self.route_kind,
            artifact_sha256=self.artifact_sha256,
            resolutions=tuple(self._resolutions),
            releases=tuple(self._releases),
            barrier_request_id=barrier_request_id,
        )
        self.request_queue.put(message, timeout=self.timeout_seconds)
        self._resolutions.clear()
        self._releases.clear()
        if barrier_request_id is not None:
            receive_native_process_response(
                self.response_queue,
                request_id=barrier_request_id,
                timeout_seconds=self.timeout_seconds,
            )


def _required_sequence_identity(
    row: SimpleStatelessActorRow,
) -> SequenceDecisionIdentity:
    identity = row.sequence_identity
    if identity is None:
        raise ValueError("remote sequence row has no absolute coordinates")
    return identity


__all__ = ["RemoteGeneralistSequenceActorPolicy"]
