"""Worker-side shared-tensor clients for parent-owned native inference."""

from __future__ import annotations

import queue
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import torch

from ptcg_rl.engine.native_public_context import NativeKnownOpponentBatch
from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.model.simple_stateless.belief import PublicBeliefSummaryBatch
from ptcg_rl.rl.native_policy_batch import NativeSimpleStatelessPolicyBatch
from ptcg_rl.rl.native_policy_trace import (
    NativePolicyNumpyActionBatch,
    NativePolicyNumpyTrace,
)
from ptcg_rl.rl.native_process_shared_batch import (
    NativeSharedBatchDescriptor,
    NativeSharedBatchWriter,
)
from ptcg_rl.rl.sequence_types import SequenceDecisionIdentity
from ptcg_rl.rl.stateless_actor import StatelessActorBatchTrace
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity

NativeProcessRouteKind = Literal["current", "past_self"]


@dataclass(frozen=True)
class NativeProcessInferenceRequest:
    """One worker-local descriptor backed by a persistent shared slab."""

    worker_index: int
    request_id: int
    route_kind: NativeProcessRouteKind
    artifact_sha256: str
    shared_batch: NativeSharedBatchDescriptor
    temperature: float


@dataclass(frozen=True)
class NativeProcessHistoricalRequest:
    """Small historical control view plus a persistent shared slab descriptor."""

    worker_index: int
    request_id: int
    artifact_sha256: str
    shared_batch: NativeSharedBatchDescriptor
    view: NativeTrainingBatchView
    known: NativeKnownOpponentBatch
    member_ids: tuple[str, ...]
    model_encoding_fingerprint: str


@dataclass(frozen=True)
class NativeProcessSequenceRows:
    """Compact recurrent coordinates accompanying shared model tensors."""

    identities: tuple[SequenceDecisionIdentity, ...]
    exact_deck_digests: tuple[str, ...]
    engine_fact_producer_fingerprints: tuple[str | None, ...]

    @property
    def batch_size(self) -> int:
        """Return the number of aligned recurrent rows."""
        return len(self.identities)


@dataclass(frozen=True)
class NativeProcessSequenceRequest:
    """One sequence-policy request backed by a persistent shared slab."""

    worker_index: int
    request_id: int
    route_kind: NativeProcessRouteKind
    artifact_sha256: str
    shared_batch: NativeSharedBatchDescriptor
    rows: NativeProcessSequenceRows
    temperature: float


@dataclass(frozen=True)
class NativeProcessSequenceResolution:
    """One provisional sequence decision accepted or rejected by the engine."""

    identity: SequenceDecisionIdentity
    commit: bool


@dataclass(frozen=True)
class NativeProcessSequenceControl:
    """Batched worker-local sequence transactions and terminal releases."""

    worker_index: int
    route_kind: NativeProcessRouteKind
    artifact_sha256: str
    resolutions: tuple[NativeProcessSequenceResolution, ...] = ()
    releases: tuple[tuple[str, int], ...] = ()
    barrier_request_id: int | None = None


@dataclass(frozen=True)
class NativeProcessInferenceResponse:
    """One exact trace/action reply or a fail-closed service error."""

    request_id: int
    trace: NativePolicyNumpyTrace | None = None
    action_batch: NativePolicyNumpyActionBatch | None = None
    sequence_trace: StatelessActorBatchTrace | None = None
    historical_actions: tuple[tuple[int, ...], ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class NativeProcessInferenceFailure:
    """Fatal broker failure broadcast to every worker response lane."""

    error: str


class RemoteNativePolicyExecutor:
    """Worker-side proxy preserving an independent per-route RNG stream."""

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
    ) -> None:
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
        self._request_id = 0

    def sample(
        self,
        batch: NativeSimpleStatelessPolicyBatch,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> NativePolicyNumpyTrace:
        """Request a full behavior trace from the single CUDA owner."""
        if self.route_kind != "current":
            raise RuntimeError("only the current route can request full traces")
        response = self._request(
            batch,
            temperature=temperature,
            generator=generator,
        )
        if response.trace is None:
            raise RuntimeError("native process inference omitted its trace")
        if response.trace.identity != self.identity:
            raise RuntimeError("native process trace identity changed")
        return response.trace

    def sample_actions(
        self,
        batch: NativeSimpleStatelessPolicyBatch,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> NativePolicyNumpyActionBatch:
        """Request action-only inference for one frozen exact artifact."""
        response = self._request(
            batch,
            temperature=temperature,
            generator=generator,
        )
        if response.action_batch is None:
            raise RuntimeError("native process inference omitted its actions")
        if response.action_batch.identity != self.identity:
            raise RuntimeError("native process action identity changed")
        return response.action_batch

    def _request(
        self,
        batch: NativeSimpleStatelessPolicyBatch,
        *,
        temperature: float,
        generator: torch.Generator | None,
    ) -> NativeProcessInferenceResponse:
        if generator is None:
            raise ValueError("native process inference requires a worker RNG")
        request_id = self._request_id
        self._request_id += 1
        shared_batch = self.shared_batch_writer.write(
            batch,
            sampling_uniforms=native_sampling_uniforms(
                batch,
                generator=generator,
            ),
        )
        self.request_queue.put(
            NativeProcessInferenceRequest(
                worker_index=self.worker_index,
                request_id=request_id,
                route_kind=self.route_kind,
                artifact_sha256=self.artifact_sha256,
                shared_batch=shared_batch,
                temperature=float(temperature),
            ),
            timeout=self.timeout_seconds,
        )
        return receive_native_process_response(
            self.response_queue,
            request_id=request_id,
            timeout_seconds=self.timeout_seconds,
        )


class RemoteNativeHistoricalPolicyPool:
    """Worker-side proxy for legacy historical GPU artifacts."""

    def __init__(
        self,
        *,
        worker_index: int,
        request_queue: Any,
        response_queue: Any,
        timeout_seconds: float,
        member_artifacts: Mapping[str, str],
        shared_batch_writer: NativeSharedBatchWriter,
    ) -> None:
        self.worker_index = int(worker_index)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout_seconds = float(timeout_seconds)
        self.member_artifacts = dict(member_artifacts)
        self.shared_batch_writer = shared_batch_writer
        self._request_id = 1 << 60

    def act_many_preencoded(
        self,
        view: NativeTrainingBatchView,
        base_states: Any,
        base_options: Any,
        known_opponent: NativeKnownOpponentBatch,
        *,
        member_ids: Sequence[str],
        deck_signatures: Sequence[str],
        model_encoding_fingerprint: str,
        forced_actions: Sequence[tuple[int, ...] | None] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Request legacy actions without loading checkpoint weights in workers."""
        if forced_actions is not None:
            raise ValueError("native process historical requests are never forced")
        request_id = self._request_id
        self._request_id += 1
        batch = self.shared_batch_writer.write(
            NativeSimpleStatelessPolicyBatch(
                states=base_states,
                options=base_options,
                unique_deck_card_ids=torch.zeros(
                    (view.batch_size, 1),
                    dtype=torch.long,
                ),
                deck_counts=torch.zeros(
                    (view.batch_size, 1),
                    dtype=torch.float32,
                ),
                deck_valid_mask=torch.zeros(
                    (view.batch_size, 1),
                    dtype=torch.bool,
                ),
                deck_signatures=tuple(deck_signatures),
                belief_summary=_empty_belief(
                    view.batch_size,
                    catalog_fingerprint="historical",
                ),
                min_counts=tuple(int(value) for value in view.select_min),
                max_counts=tuple(int(value) for value in view.select_max),
                public_deck_catalog_fingerprint="historical",
                input_contract_fingerprint=model_encoding_fingerprint,
            )
        )
        artifacts = {
            self.member_artifacts[member_id] for member_id in member_ids
        }
        if len(artifacts) != 1:
            raise ValueError("one historical request crossed checkpoint artifacts")
        artifact_sha256 = next(iter(artifacts))
        self.request_queue.put(
            NativeProcessHistoricalRequest(
                worker_index=self.worker_index,
                request_id=request_id,
                artifact_sha256=artifact_sha256,
                shared_batch=batch,
                view=replace(view, _owner=cast(Any, None)),
                known=known_opponent,
                member_ids=tuple(member_ids),
                model_encoding_fingerprint=model_encoding_fingerprint,
            ),
            timeout=self.timeout_seconds,
        )
        response = receive_native_process_response(
            self.response_queue,
            request_id=request_id,
            timeout_seconds=self.timeout_seconds,
        )
        if len(response.historical_actions) != view.batch_size:
            raise RuntimeError("native process historical actions are misaligned")
        return response.historical_actions


def native_sampling_uniforms(
    batch: NativeSimpleStatelessPolicyBatch,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate the explicit worker-independent decode random table."""
    maximum_steps = (
        max(batch.options.maximum_counts, default=0)
        if batch.options.maximum_counts
        else int(batch.options.valid_options.shape[1]) + 1
    )
    return torch.rand(
        (batch.batch_size, maximum_steps + 1),
        dtype=torch.float32,
        generator=generator,
    )


def receive_native_process_response(
    response_queue: Any,
    *,
    request_id: int,
    timeout_seconds: float,
) -> NativeProcessInferenceResponse:
    try:
        response = response_queue.get(timeout=timeout_seconds)
    except queue.Empty as error:
        raise TimeoutError("native process inference timed out") from error
    if isinstance(response, NativeProcessInferenceFailure):
        raise RuntimeError(f"native process inference failed: {response.error}")
    if not isinstance(response, NativeProcessInferenceResponse):
        raise TypeError("native process inference response is invalid")
    if response.request_id != request_id:
        raise RuntimeError("native process inference response crossed requests")
    if response.error is not None:
        raise RuntimeError(f"native process inference failed: {response.error}")
    return response


def _empty_belief(
    batch_size: int,
    *,
    catalog_fingerprint: str,
) -> PublicBeliefSummaryBatch:
    return PublicBeliefSummaryBatch(
        card_ids=torch.zeros((batch_size, 1), dtype=torch.long),
        expected_counts=torch.zeros((batch_size, 1), dtype=torch.float32),
        valid_mask=torch.zeros((batch_size, 1), dtype=torch.bool),
        scalars=torch.zeros((batch_size, 4), dtype=torch.float32),
        catalog_fingerprint=catalog_fingerprint,
    )


__all__ = [
    "NativeProcessHistoricalRequest",
    "NativeProcessInferenceFailure",
    "NativeProcessInferenceRequest",
    "NativeProcessInferenceResponse",
    "NativeProcessRouteKind",
    "NativeProcessSequenceControl",
    "NativeProcessSequenceRequest",
    "NativeProcessSequenceResolution",
    "NativeProcessSequenceRows",
    "RemoteNativeHistoricalPolicyPool",
    "RemoteNativePolicyExecutor",
    "native_sampling_uniforms",
    "receive_native_process_response",
]
