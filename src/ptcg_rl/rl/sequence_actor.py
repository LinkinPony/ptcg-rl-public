"""Transactional actor for the generalist sequence policy."""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from itertools import groupby
from typing import Any, Literal, TypeAlias

import torch
from torch import Tensor

from ptcg_rl.actions.selection import ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
from ptcg_rl.context.public_event_arrays import (
    PublicEventBatch,
    collate_public_event_deltas,
    move_public_event_batch,
)
from ptcg_rl.model.sequence.action import (
    AcceptedActionBatch,
    build_accepted_action_records,
)
from ptcg_rl.model.sequence.config import (
    GENERALIST_SEQUENCE_CONTRACT_FINGERPRINT,
)
from ptcg_rl.model.sequence.core import TemporalKvCache, TemporalKvSlotPool
from ptcg_rl.model.sequence.device_action import (
    build_device_accepted_action_batch,
)
from ptcg_rl.model.sequence.host_action import (
    build_tensor_host_accepted_action_records,
)
from ptcg_rl.model.sequence.network import TemporalPreparedDecision
from ptcg_rl.model.simple_stateless import (
    SimpleExactRoutePlan,
    SimpleStatelessPolicyValueNet,
    resolve_simple_exact_routes,
)
from ptcg_rl.model.simple_stateless.rollout_cache import (
    mark_immutable_rollout_cache_generation,
)
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.native_policy_batch import NativeSimpleStatelessPolicyBatch
from ptcg_rl.rl.native_policy_inference import PolicyInputBatch
from ptcg_rl.rl.native_policy_trace import (
    NativePolicyNumpyTrace,
    NativePolicyTensorTrace,
    evaluation_action_only_tensor_trace,
)
from ptcg_rl.rl.policy_inputs import (
    SimpleStatelessActorRow,
    collate_simple_stateless_actor_rows,
)
from ptcg_rl.rl.sequence_actor_transfer import (
    SequenceActorHostTransfer,
    decisions_from_native_trace,
)
from ptcg_rl.rl.sequence_runtime import (
    SequenceCacheFork,
    SequenceCacheIdentity,
    SequenceRawBlock,
    StagedSequenceProposal,
    TransactionalSequenceCache,
)
from ptcg_rl.rl.sequence_types import SequenceDecisionIdentity
from ptcg_rl.rl.stateless_actor import (
    StatelessActorBatchTrace,
    StatelessActorDecisionTrace,
)
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity

SequenceRolloutPrecision: TypeAlias = Literal["fp32", "bf16"]


def _canonical_runtime_device(device: torch.device | str) -> torch.device:
    """Resolve an unindexed CUDA device to the process's current device."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


@dataclass(frozen=True)
class _InflightSequenceRow:
    """One row reserved by a deferred host transfer."""

    transfer: SequenceActorHostTransfer
    row_index: int
    request_id: str
    block_index: int


@dataclass(frozen=True)
class _ResolvedSequenceDeckBatch:
    """Reusable route layout and exact-deck token rows for one batch layout."""

    route_plan: SimpleExactRoutePlan
    token_rows: Tensor


@dataclass(slots=True)
class SequenceActorSampleContinuation:
    """Resume one prefixed sequence forward exactly once at the fact barrier."""

    _resume_callback: (
        Callable[[Callable[[], None] | None], SequenceActorHostTransfer] | None
    )
    _cancel_callback: Callable[[], None] | None = None
    _result: SequenceActorHostTransfer | None = None
    _error: BaseException | None = None
    _resuming: bool = False
    _cancelled: bool = False

    def resume(
        self,
        *,
        await_option_features: Callable[[], None] | None = None,
    ) -> SequenceActorHostTransfer:
        """Join option facts, enqueue decode, and return the deferred host copy."""
        if self._result is not None:
            return self._result
        if self._error is not None:
            raise self._error
        if self._cancelled:
            raise RuntimeError("sequence actor continuation was cancelled")
        if self._resuming:
            raise RuntimeError("sequence actor continuation is already resuming")
        callback = self._resume_callback
        if callback is None:
            raise RuntimeError("sequence actor continuation has no resume callback")
        self._resuming = True
        try:
            self._result = callback(await_option_features)
        except BaseException as error:
            self._error = error
            raise
        finally:
            self._resuming = False
            self._resume_callback = None
            self._cancel_callback = None
        return self._result

    def cancel(self) -> None:
        """Drop an unresumed prefix without sampling or staging actor state."""
        if self._result is not None or self._error is not None or self._cancelled:
            return
        if self._resuming:
            raise RuntimeError("sequence actor continuation is already resuming")
        callback = self._cancel_callback
        self._cancelled = True
        try:
            if callback is not None:
                callback()
        finally:
            self._resume_callback = None
            self._cancel_callback = None


class GeneralistSequenceActorPolicy:
    """Run snapshot rows in batch and temporal KV per game-seat transaction."""

    def __init__(
        self,
        model: SimpleStatelessPolicyValueNet,
        *,
        identity: StatelessFragmentIdentity,
        device: torch.device | str,
        verify_model_state: bool = True,
        retain_raw_blocks: bool = True,
        temporal_cache_slots: int | None = None,
        rollout_precision: SequenceRolloutPrecision = "fp32",
    ) -> None:
        """Bind one immutable artifact and an isolated cache registry."""
        if model.sequence is None or model.config.sequence is None:
            raise ValueError("generalist sequence actor requires temporal model")
        if identity.schema_version != 2:
            raise ValueError("generalist sequence actor requires fragment schema V2")
        if (
            identity.sequence_contract_fingerprint
            != GENERALIST_SEQUENCE_CONTRACT_FINGERPRINT
        ):
            raise ValueError("actor sequence contract differs from the model")
        if rollout_precision not in {"fp32", "bf16"}:
            raise ValueError("sequence rollout precision must be fp32 or bf16")
        self.device = _canonical_runtime_device(device)
        self.identity = identity
        self.model = model.to(self.device).eval()
        if verify_model_state:
            actual = canonical_model_state_fingerprint(self.model)
            if actual != identity.behavior_policy_fingerprint:
                raise ValueError("sequence actor model differs from behavior identity")
        self.rollout_precision = rollout_precision
        self._rollout_model: SimpleStatelessPolicyValueNet | None = (
            _resident_bfloat16_shadow(self.model, device=self.device)
            if rollout_precision == "bf16"
            else self.model
        )
        self.retain_raw_blocks = bool(retain_raw_blocks)
        self.cache = TransactionalSequenceCache()
        self._cache_identities: dict[tuple[str, int], SequenceCacheIdentity] = {}
        self._pending: dict[
            tuple[SequenceCacheIdentity, str],
            tuple[
                StagedSequenceProposal,
                StatelessActorDecisionTrace,
                TemporalKvCache,
            ],
        ] = {}
        self._inflight: dict[SequenceCacheIdentity, _InflightSequenceRow] = {}
        self._closed = False
        if temporal_cache_slots is not None and temporal_cache_slots <= 0:
            raise ValueError("temporal cache slot capacity must be positive")
        self.temporal_cache_slot_capacity = temporal_cache_slots
        self._temporal_slot_pool = (
            None
            if temporal_cache_slots is None
            else TemporalKvSlotPool.allocate(
                model.config.sequence,
                slots=temporal_cache_slots,
                device=self.device,
                dtype=(
                    torch.bfloat16
                    if self.device.type == "cuda"
                    else next(self._require_rollout_model().parameters()).dtype
                ),
            )
        )
        self._free_temporal_slots = (
            []
            if temporal_cache_slots is None
            else list(range(temporal_cache_slots - 1, -1, -1))
        )
        self._temporal_slots: dict[SequenceCacheIdentity, int] = {}
        (
            self._deck_token_rows,
            self._exact_deck_tokens,
        ) = self._build_exact_deck_token_cache()
        self._resolved_deck_batches: OrderedDict[
            tuple[str, ...],
            _ResolvedSequenceDeckBatch,
        ] = OrderedDict()

    @property
    def uses_bfloat16_autocast(self) -> bool:
        """Return whether an FP32 rollout model needs CUDA BF16 autocast."""
        return self.device.type == "cuda" and self.rollout_precision == "fp32"

    @property
    def uses_pure_bfloat16(self) -> bool:
        """Return whether model weights and the main residual stream are BF16."""
        return self.rollout_precision == "bf16"

    @property
    def uses_generalist_sequence(self) -> bool:
        """Expose transactional sequence ownership to collection transports."""
        return True

    @property
    def rollout_model(self) -> SimpleStatelessPolicyValueNet:
        """Return the collection-local inference model while it is resident."""
        return self._require_rollout_model()

    def sample(
        self,
        rows: Sequence[SimpleStatelessActorRow],
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        sampling_uniforms: Tensor | None = None,
    ) -> StatelessActorBatchTrace:
        """Collate raw actor rows and sample one transactional sequence wave."""
        return self._sample_deferred(
            rows,
            batch=None,
            temperature=temperature,
            generator=generator,
            sampling_uniforms=sampling_uniforms,
            copy_stream=None,
        ).finish()

    def sample_preencoded(
        self,
        rows: Sequence[SimpleStatelessActorRow],
        batch: PolicyInputBatch,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        sampling_uniforms: Tensor | None = None,
        await_option_features: Callable[[], None] | None = None,
        public_event_batch: PublicEventBatch | None = None,
        host_semantic_batch: NativeSimpleStatelessPolicyBatch | None = None,
        evaluation_action_only: bool = False,
    ) -> StatelessActorBatchTrace:
        """Sample from an aligned model-ready batch without recollating rows."""
        return self.sample_preencoded_deferred(
            rows,
            batch,
            temperature=temperature,
            generator=generator,
            sampling_uniforms=sampling_uniforms,
            copy_stream=None,
            await_option_features=await_option_features,
            public_event_batch=public_event_batch,
            host_semantic_batch=host_semantic_batch,
            evaluation_action_only=evaluation_action_only,
        ).finish()

    def sample_preencoded_deferred(
        self,
        rows: Sequence[SimpleStatelessActorRow],
        batch: PolicyInputBatch,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        sampling_uniforms: Tensor | None = None,
        copy_stream: Any | None = None,
        await_option_features: Callable[[], None] | None = None,
        public_event_batch: PublicEventBatch | None = None,
        host_semantic_batch: NativeSimpleStatelessPolicyBatch | None = None,
        evaluation_action_only: bool = False,
    ) -> SequenceActorHostTransfer:
        """Queue one packed host copy and defer object/transaction finalization."""
        return self.begin_preencoded_deferred(
            rows,
            batch,
            temperature=temperature,
            generator=generator,
            sampling_uniforms=sampling_uniforms,
            copy_stream=copy_stream,
            public_event_batch=public_event_batch,
            host_semantic_batch=host_semantic_batch,
            evaluation_action_only=evaluation_action_only,
        ).resume(
            await_option_features=await_option_features,
        )

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
        """Enqueue snapshot/temporal prefix without waiting for option facts."""
        return self._begin_sample_deferred(
            rows,
            batch=batch,
            temperature=temperature,
            generator=generator,
            sampling_uniforms=sampling_uniforms,
            copy_stream=copy_stream,
            public_event_batch=public_event_batch,
            host_semantic_batch=host_semantic_batch,
            materialize_host_actions=materialize_host_actions,
            evaluation_action_only=evaluation_action_only,
        )

    def _sample_deferred(
        self,
        rows: Sequence[SimpleStatelessActorRow],
        *,
        batch: PolicyInputBatch | None,
        temperature: float,
        generator: torch.Generator | None,
        sampling_uniforms: Tensor | None,
        copy_stream: Any | None,
        await_option_features: Callable[[], None] | None = None,
        public_event_batch: PublicEventBatch | None = None,
        host_semantic_batch: NativeSimpleStatelessPolicyBatch | None = None,
    ) -> SequenceActorHostTransfer:
        return self._begin_sample_deferred(
            rows,
            batch=batch,
            temperature=temperature,
            generator=generator,
            sampling_uniforms=sampling_uniforms,
            copy_stream=copy_stream,
            public_event_batch=public_event_batch,
            host_semantic_batch=host_semantic_batch,
            materialize_host_actions=True,
            evaluation_action_only=False,
        ).resume(
            await_option_features=await_option_features,
        )

    def _begin_sample_deferred(
        self,
        rows: Sequence[SimpleStatelessActorRow],
        *,
        batch: PolicyInputBatch | None,
        temperature: float,
        generator: torch.Generator | None,
        sampling_uniforms: Tensor | None,
        copy_stream: Any | None,
        public_event_batch: PublicEventBatch | None = None,
        host_semantic_batch: NativeSimpleStatelessPolicyBatch | None = None,
        materialize_host_actions: bool = True,
        evaluation_action_only: bool = False,
    ) -> SequenceActorSampleContinuation:
        """Enqueue work through temporal conditioning without consuming RNG."""
        if self._closed:
            raise RuntimeError("sequence actor is closed")
        rows = tuple(rows)
        if not rows:
            raise ValueError("sequence actor requires decision rows")
        if any(row.sequence_identity is None for row in rows):
            raise ValueError("sequence actor row is missing absolute coordinates")
        if any(row.max_count == 0 for row in rows):
            raise ValueError("forced prompts cannot advance the sequence clock")
        self._validate_generator(generator)
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("sequence temperature must be finite and non-negative")
        if evaluation_action_only != (temperature == 0.0):
            raise ValueError(
                "T=0 sequence inference requires evaluation_action_only and "
                "evaluation_action_only requires T=0"
            )
        if evaluation_action_only and sampling_uniforms is not None:
            raise ValueError("greedy sequence evaluation cannot consume uniforms")
        if sampling_uniforms is not None and sampling_uniforms.device != self.device:
            raise ValueError("sequence sampling uniforms must share the actor device")
        if self.device.type != "cuda" and copy_stream is not None:
            raise ValueError("CPU policy transfer cannot use a CUDA copy stream")
        inflight_duplicate = self._inflight_duplicate_transfer(rows)
        if inflight_duplicate is not None:
            return SequenceActorSampleContinuation(
                _resume_callback=lambda await_option_features: _await_then_return(
                    await_option_features,
                    inflight_duplicate,
                ),
            )
        prepared_or_duplicates = tuple(
            self.cache.prepare(
                self._cache_identity(row),
                request_id=self._coordinates(row).request_id,
                block_index=self._coordinates(row).decision_index,
            )
            for row in rows
        )
        duplicates = tuple(
            isinstance(value, StagedSequenceProposal)
            for value in prepared_or_duplicates
        )
        if any(duplicates):
            if not all(duplicates):
                raise RuntimeError(
                    "mixed fresh and duplicate sequence rows must be partitioned"
                )
            duplicate_decisions = tuple(
                self._pending[(proposal.fork.identity, proposal.fork.request_id)][1]
                for proposal in prepared_or_duplicates
                if isinstance(proposal, StagedSequenceProposal)
            )
            duplicate_transfer = SequenceActorHostTransfer.completed(
                self._batch_trace(duplicate_decisions),
            )
            return SequenceActorSampleContinuation(
                _resume_callback=lambda await_option_features: _await_then_return(
                    await_option_features,
                    duplicate_transfer,
                ),
            )

        event_batch = (
            collate_public_event_deltas(
                tuple(row.public_event_delta for row in rows),
                device=self.device,
            )
            if public_event_batch is None
            else move_public_event_batch(
                public_event_batch,
                device=self.device,
                non_blocking=self.device.type == "cuda",
            )
        )
        if event_batch.batch_size != len(rows):
            raise ValueError(
                "pre-collated public events and sequence rows are misaligned"
            )
        if batch is None:
            batch = collate_simple_stateless_actor_rows(rows, device=self.device)
        if host_semantic_batch is not None and (
            host_semantic_batch.batch_size != len(rows)
            or host_semantic_batch.input_contract_fingerprint
            != batch.input_contract_fingerprint
            or host_semantic_batch.deck_signatures != batch.deck_signatures
            or host_semantic_batch.min_counts != batch.min_counts
            or host_semantic_batch.max_counts != batch.max_counts
        ):
            raise ValueError("native host semantic batch is misaligned")
        self._validate_preencoded_batch(rows, batch)
        rollout_model = self._require_rollout_model()
        rollout_dtype = next(rollout_model.parameters()).dtype
        resolved_decks = self._resolve_exact_deck_batch(
            batch.deck_signatures,
        )
        routes = resolved_decks.route_plan
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.uses_bfloat16_autocast,
            ),
        ):
            snapshots = rollout_model.encode_observation_state_with_deck_tokens(
                state=batch.states,
                deck_tokens=self._exact_deck_tokens.index_select(
                    0,
                    resolved_decks.token_rows,
                ),
                belief_summary=batch.belief_summary,
                route_plan=routes,
            )
            forks = tuple(
                self._fork_with_preallocated_slot(fork)
                for fork in prepared_or_duplicates
                if not isinstance(fork, StagedSequenceProposal)
            )
            if len(forks) != len(rows):
                raise RuntimeError("fresh sequence branch unexpectedly duplicated")
            temporal = rollout_model.prepare_sequence_incremental_many(
                snapshots,
                event_batch,
                block_indices=tuple(
                    self._coordinates(row).decision_index for row in rows
                ),
                caches=tuple(fork.committed_cache for fork in forks),
            )
            conditioned = rollout_model.condition_sequence(
                snapshots,
                torch.cat(tuple(item.context for item in temporal), dim=0),
            )

        def resume(
            await_option_features: Callable[[], None] | None,
        ) -> SequenceActorHostTransfer:
            if await_option_features is not None:
                await_option_features()
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.uses_bfloat16_autocast,
                ),
            ):
                option_embeddings = rollout_model.encode_legal_options(
                    conditioned,
                    batch.options,
                    route_plan=routes,
                )
                if evaluation_action_only:
                    greedy = rollout_model.heads.greedy_decode_trace(
                        conditioned.policy,
                        conditioned.opponent_belief,
                        option_embeddings,
                        batch.options,
                        route_plan=routes,
                    )
                    action_choices = greedy.actions.choice_indices
                    action_lengths = greedy.actions.lengths
                    stop_sampled = greedy.stop_sampled
                    tensor_trace = evaluation_action_only_tensor_trace(
                        identity=self.identity,
                        action_choices=action_choices,
                        action_lengths=action_lengths,
                        stop_sampled=stop_sampled,
                    )
                else:
                    sampled = rollout_model.heads.sample_decode_with_trace(
                        conditioned.policy,
                        conditioned.opponent_belief,
                        option_embeddings,
                        batch.options,
                        route_plan=routes,
                        temperature=temperature,
                        generator=generator,
                        sampling_uniforms=sampling_uniforms,
                    )
                    root_values = rollout_model.heads.root_value(
                        conditioned.value,
                        conditioned.opponent_belief,
                        route_plan=routes,
                    )
                    action_choices = sampled.actions.choice_indices
                    action_lengths = sampled.actions.lengths
                    stop_sampled = sampled.stop_sampled
                    tensor_trace = NativePolicyTensorTrace(
                        identity=self.identity,
                        action_choices=action_choices,
                        action_lengths=action_lengths,
                        action_logprobs=sampled.action_logprobs,
                        token_logprobs=sampled.token_logprobs,
                        token_mask=sampled.token_mask,
                        prefix_values=sampled.prefix_values.float(),
                        root_values=root_values.float(),
                        stop_sampled=stop_sampled,
                    )
                accepted_actions = build_device_accepted_action_batch(
                    states=batch.states,
                    options=batch.options,
                    choice_indices=action_choices,
                    lengths=action_lengths,
                    stop_sampled=stop_sampled,
                    unordered_rows=_unordered_action_rows(
                        rows,
                        host_semantic_batch=host_semantic_batch,
                    ),
                )
                committed_caches = rollout_model.commit_sequence_incremental_many(
                    temporal,
                    _accepted_action_batch_float_dtype(
                        accepted_actions,
                        dtype=rollout_dtype,
                    ),
                )
            native_transfer = tensor_trace.defer_to_host(copy_stream=copy_stream)

            transfer: SequenceActorHostTransfer

            def finalize(
                host_trace: NativePolicyNumpyTrace,
            ) -> StatelessActorBatchTrace:
                try:
                    return self._finalize_host_trace(
                        rows=rows,
                        forks=forks,
                        temporal=temporal,
                        committed_caches=committed_caches,
                        host_trace=host_trace,
                        host_semantic_batch=host_semantic_batch,
                        materialize_host_actions=materialize_host_actions,
                    )
                finally:
                    self._release_inflight(transfer, forks)

            transfer = SequenceActorHostTransfer.from_native_trace(
                native_transfer,
                finalize=finalize,
                cancel=lambda: self._release_inflight(transfer, forks),
            )
            try:
                for row_index, fork in enumerate(forks):
                    if fork.identity in self._inflight:
                        raise RuntimeError(
                            "sequence cache already has a deferred proposal"
                        )
                    self._inflight[fork.identity] = _InflightSequenceRow(
                        transfer=transfer,
                        row_index=row_index,
                        request_id=fork.request_id,
                        block_index=fork.block_index,
                    )
            except BaseException:
                self._release_inflight(transfer, forks)
                raise
            return transfer

        return SequenceActorSampleContinuation(_resume_callback=resume)

    def _finalize_host_trace(
        self,
        *,
        rows: tuple[SimpleStatelessActorRow, ...],
        forks: tuple[SequenceCacheFork, ...],
        temporal: Sequence[TemporalPreparedDecision],
        committed_caches: Sequence[TemporalKvCache],
        host_trace: NativePolicyNumpyTrace,
        host_semantic_batch: NativeSimpleStatelessPolicyBatch | None = None,
        materialize_host_actions: bool = True,
    ) -> StatelessActorBatchTrace:
        """Build stable records and stage KV after the packed trace is ready."""
        if host_trace.identity != self.identity:
            raise ValueError("deferred sequence trace changed behavior identity")
        detached = decisions_from_native_trace(host_trace)
        if not (
            len(detached)
            == len(rows)
            == len(forks)
            == len(temporal)
            == len(committed_caches)
        ):
            raise ValueError("deferred sequence trace rows are misaligned")
        accepted_actions = (
            (
                build_accepted_action_records(
                    states=tuple(row.state for row in rows),
                    options=tuple(row.options for row in rows),
                    actions=tuple(trace.action for trace in detached),
                    min_counts=tuple(row.min_count for row in rows),
                    max_counts=tuple(row.max_count for row in rows),
                    stop_sampled=tuple(trace.stop_sampled for trace in detached),
                )
                if host_semantic_batch is None
                else build_tensor_host_accepted_action_records(
                    states=host_semantic_batch.states,
                    options=host_semantic_batch.options,
                    action_offsets=host_trace.action_offsets,
                    action_choices=host_trace.action_choices,
                    min_counts=host_semantic_batch.min_counts,
                    max_counts=host_semantic_batch.max_counts,
                    stop_sampled=host_trace.stop_sampled,
                )
            )
            if materialize_host_actions
            else (None,) * len(rows)
        )
        decisions = tuple(
            replace(
                trace,
                accepted_action=accepted,
                sequence_request_id=fork.request_id,
            )
            for trace, fork, accepted in zip(
                detached,
                forks,
                accepted_actions,
                strict=True,
            )
        )
        keys = tuple((fork.identity, fork.request_id) for fork in forks)
        if len(set(keys)) != len(keys) or any(key in self._pending for key in keys):
            raise RuntimeError("deferred sequence proposal identity is not unique")
        proposals: list[StagedSequenceProposal] = []
        try:
            for fork, prepared, accepted in zip(
                forks,
                temporal,
                accepted_actions,
                strict=True,
            ):
                proposals.append(
                    self.cache.stage(
                        fork,
                        prepared=prepared,
                        accepted_action=accepted,
                    )
                )
        except BaseException:
            for proposal in reversed(proposals):
                self.cache.abort(proposal)
            raise
        for key, proposal, decision, committed in zip(
            keys,
            proposals,
            decisions,
            committed_caches,
            strict=True,
        ):
            self._pending[key] = (
                proposal,
                decision,
                committed,
            )
        return self._batch_trace(decisions)

    def _inflight_duplicate_transfer(
        self,
        rows: tuple[SimpleStatelessActorRow, ...],
    ) -> SequenceActorHostTransfer | None:
        """Return an ordered duplicate view over existing deferred rows."""
        entries = tuple(self._inflight.get(self._cache_identity(row)) for row in rows)
        if not any(entry is not None for entry in entries):
            return None
        if not all(entry is not None for entry in entries):
            raise RuntimeError(
                "mixed fresh and duplicate sequence rows must be partitioned"
            )
        resolved: list[_InflightSequenceRow] = []
        for row, entry in zip(rows, entries, strict=True):
            assert entry is not None
            coordinates = self._coordinates(row)
            if (
                entry.request_id != coordinates.request_id
                or entry.block_index != coordinates.decision_index
            ):
                raise RuntimeError("sequence cache already has a provisional proposal")
            resolved.append(entry)

        def materialize(*, ready: bool) -> StatelessActorBatchTrace:
            decisions = tuple(
                (
                    entry.transfer.finish_ready() if ready else entry.transfer.finish()
                ).decisions[entry.row_index]
                for entry in resolved
            )
            return self._batch_trace(decisions)

        return SequenceActorHostTransfer.from_callbacks(
            finish=lambda: materialize(ready=False),
            finish_ready=lambda: materialize(ready=True),
        )

    def _release_inflight(
        self,
        transfer: SequenceActorHostTransfer,
        forks: Sequence[SequenceCacheFork],
    ) -> None:
        """Release only reservations still owned by this transfer."""
        for fork in forks:
            entry = self._inflight.get(fork.identity)
            if entry is not None and entry.transfer is transfer:
                del self._inflight[fork.identity]

    def commit_decision(
        self,
        row: SimpleStatelessActorRow,
        trace: StatelessActorDecisionTrace,
    ) -> None:
        """Append the actually accepted ACTION and atomically publish its block."""
        request_id = trace.sequence_request_id
        if request_id is None:
            raise ValueError("sequence trace has no request identity")
        if trace.accepted_action is None and self.retain_raw_blocks:
            raise ValueError("retained sequence trace has no accepted action")
        identity = self._cache_identity(row)
        key = (identity, request_id)
        pending = self._pending.get(key)
        if pending is None:
            raise RuntimeError("sequence proposal is not pending")
        proposal, expected_trace, committed = pending
        if expected_trace is not trace:
            raise ValueError("served trace differs from staged sequence proposal")
        raw_block = (
            None
            if trace.accepted_action is None
            else SequenceRawBlock(
                block_index=proposal.fork.block_index,
                public_events=row.public_event_delta,
                snapshot=row,
                accepted_action=trace.accepted_action,
                engine_fact_producer_fingerprint=(
                    row.engine_fact_producer_fingerprint
                ),
            )
        )
        self.cache.commit(
            proposal,
            committed_cache=committed,
            raw_block=raw_block,
            retain_raw_block=self.retain_raw_blocks,
        )
        del self._pending[key]

    def abort_decision(
        self,
        row: SimpleStatelessActorRow,
        trace: StatelessActorDecisionTrace,
    ) -> None:
        """Discard a rejected proposal without consuming public events."""
        request_id = trace.sequence_request_id
        if request_id is None:
            raise ValueError("sequence trace has no request identity")
        identity = self._cache_identity(row)
        key = (identity, request_id)
        pending = self._pending.get(key)
        if pending is None:
            raise RuntimeError("sequence proposal is not pending")
        self.cache.abort(pending[0])
        del self._pending[key]

    def release_game(self, *, game_id: str, seat: int) -> None:
        """Release terminal cache residency and reject pending proposals."""
        identities = self.cache.identities_for_game_seat(
            game_id=game_id,
            seat=seat,
        )
        for identity in identities:
            if identity in self._inflight:
                raise RuntimeError("cannot release sequence with deferred proposal")
            if self.cache.has_staged(identity):
                raise RuntimeError("cannot release sequence with pending proposal")
            self.cache.release(identity)
            slot = self._temporal_slots.pop(identity, None)
            if slot is not None:
                self._free_temporal_slots.append(slot)
        cache_identities = getattr(self, "_cache_identities", None)
        if cache_identities is not None:
            cache_identities.pop((game_id, seat), None)

    def close(self) -> None:
        """Release transactional state and any collection-local model shadow."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        transfers = {
            id(entry.transfer): entry.transfer for entry in self._inflight.values()
        }
        for transfer in transfers.values():
            # A failed CUDA wave may already have poisoned its completion event.
            # The reservation callback still runs from cancel's finally block.
            with suppress(BaseException):
                transfer.cancel()
        self._inflight.clear()
        for proposal, _trace, _committed in tuple(self._pending.values()):
            with suppress(BaseException):
                self.cache.abort(proposal)
        self._pending.clear()
        for identity in self.cache.identities():
            self.cache.release(identity)
        self._temporal_slots.clear()
        cache_identities = getattr(self, "_cache_identities", None)
        if cache_identities is not None:
            cache_identities.clear()
        self._free_temporal_slots.clear()
        self._temporal_slot_pool = None
        self._deck_token_rows.clear()
        self._exact_deck_tokens = torch.empty(0, device="cpu")
        resolved_deck_batches = getattr(self, "_resolved_deck_batches", None)
        if resolved_deck_batches is not None:
            resolved_deck_batches.clear()
        self._rollout_model = None

    def _cache_identity(
        self,
        row: SimpleStatelessActorRow,
    ) -> SequenceCacheIdentity:
        coordinates = self._coordinates(row)
        config = self.model.config.sequence
        if config is None:
            raise RuntimeError("sequence actor lost temporal config")
        game_seat = (coordinates.game_id, coordinates.seat)
        identities: dict[tuple[str, int], SequenceCacheIdentity] | None = getattr(
            self,
            "_cache_identities",
            None,
        )
        if identities is None:
            identities = {}
            self._cache_identities = identities
        existing = identities.get(game_seat)
        if existing is not None:
            if (
                existing.exact_deck_digest != row.own_deck.deck_digest
                or existing.input_contract_fingerprint
                != row.input_contract_fingerprint
            ):
                raise ValueError("sequence cache identity changed within one game seat")
            return existing
        identity = SequenceCacheIdentity(
            game_id=coordinates.game_id,
            seat=coordinates.seat,
            exact_deck_digest=row.own_deck.deck_digest,
            policy_artifact_fingerprint=(self.identity.behavior_policy_fingerprint),
            model_config_fingerprint=self.identity.model_config_fingerprint,
            input_contract_fingerprint=row.input_contract_fingerprint,
            sequence_contract_fingerprint=config.contract_fingerprint,
            max_context_blocks=config.max_context_blocks,
        )
        identities[game_seat] = identity
        return identity

    @staticmethod
    def _coordinates(row: SimpleStatelessActorRow) -> SequenceDecisionIdentity:
        identity = row.sequence_identity
        if identity is None:
            raise ValueError("sequence actor row has no coordinates")
        return identity

    def _batch_trace(
        self,
        decisions: tuple[StatelessActorDecisionTrace, ...],
    ) -> StatelessActorBatchTrace:
        return StatelessActorBatchTrace(
            behavior_policy_version=self.identity.behavior_policy_version,
            behavior_policy_fingerprint=self.identity.behavior_policy_fingerprint,
            input_contract_fingerprint=self.identity.input_contract_fingerprint,
            decisions=decisions,
        )

    def _validate_generator(self, generator: torch.Generator | None) -> None:
        if generator is None:
            return
        if torch.device(generator.device).type != self.device.type:
            raise ValueError("sampling generator must use actor device type")

    def _validate_preencoded_batch(
        self,
        rows: Sequence[SimpleStatelessActorRow],
        batch: PolicyInputBatch,
    ) -> None:
        """Require model tensors and raw transactional evidence to stay aligned."""
        if batch.batch_size != len(rows):
            raise ValueError("sequence policy batch and raw rows are misaligned")
        if batch.input_contract_fingerprint != self.identity.input_contract_fingerprint:
            raise ValueError("sequence policy batch differs from behavior contract")
        if any(
            row.input_contract_fingerprint != batch.input_contract_fingerprint
            for row in rows
        ):
            raise ValueError("sequence raw rows cross input contracts")
        if batch.deck_signatures != tuple(row.own_deck.signature for row in rows):
            raise ValueError("sequence policy batch and exact decks are misaligned")
        if batch.min_counts != tuple(row.min_count for row in rows) or (
            batch.max_counts != tuple(row.max_count for row in rows)
        ):
            raise ValueError("sequence policy batch select counts are misaligned")
        batch_device = batch.states.card_ids.device
        if batch_device.type != self.device.type or (
            self.device.index is not None and batch_device.index != self.device.index
        ):
            raise ValueError("sequence policy batch differs from actor device")
        catalog = batch.belief_summary.catalog_fingerprint
        if catalog != self.identity.public_deck_catalog_fingerprint or any(
            row.catalog_fingerprint != catalog for row in rows
        ):
            raise ValueError("sequence policy batch crosses public deck catalogs")

    def _fork_with_preallocated_slot(
        self,
        fork: SequenceCacheFork,
    ) -> SequenceCacheFork:
        """Bind a fresh cache fork to its persistent game-seat KV slot."""
        pool = self._temporal_slot_pool
        if pool is None or fork.committed_cache is not None:
            return fork
        slot = self._temporal_slots.get(fork.identity)
        if slot is None:
            if not self._free_temporal_slots:
                raise RuntimeError("temporal KV slot capacity was exhausted")
            slot = self._free_temporal_slots.pop()
            self._temporal_slots[fork.identity] = slot
        return replace(fork, committed_cache=pool.empty_cache(slot))

    def _build_exact_deck_token_cache(self) -> tuple[dict[str, int], Tensor]:
        """Encode every immutable exact deck once for this policy artifact."""
        rollout_model = self._require_rollout_model()
        routes = rollout_model.config.exact_routes
        if not routes:
            raise ValueError("sequence actor requires resolved exact deck routes")
        encoded = tuple(
            tuple(
                (int(card_id), sum(1 for _unused in repeated))
                for card_id, repeated in groupby(route.canonical_card_ids)
            )
            for route in routes
        )
        width = max(len(row) for row in encoded)
        card_ids = torch.zeros(
            (len(encoded), width),
            dtype=torch.long,
            device=self.device,
        )
        counts = torch.zeros(
            (len(encoded), width),
            dtype=torch.float32,
            device=self.device,
        )
        valid = torch.zeros(
            (len(encoded), width),
            dtype=torch.bool,
            device=self.device,
        )
        for row_index, row in enumerate(encoded):
            row_width = len(row)
            card_ids[row_index, :row_width] = torch.tensor(
                tuple(card_id for card_id, _count in row),
                dtype=torch.long,
                device=self.device,
            )
            counts[row_index, :row_width] = torch.tensor(
                tuple(count for _card_id, count in row),
                dtype=torch.float32,
                device=self.device,
            )
            valid[row_index, :row_width] = True
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.uses_bfloat16_autocast,
            ),
        ):
            tokens = rollout_model.encode_exact_deck_tokens(
                unique_deck_card_ids=card_ids,
                deck_counts=counts,
                deck_valid_mask=valid,
            )
        return (
            {route.signature: row for row, route in enumerate(routes)},
            tokens,
        )

    def _resolve_exact_deck_batch(
        self,
        deck_signatures: Sequence[str],
    ) -> _ResolvedSequenceDeckBatch:
        """Reuse device route and token-row layouts across repeated cohorts."""
        key = tuple(deck_signatures)
        cached = self._resolved_deck_batches.get(key)
        if cached is not None:
            self._resolved_deck_batches.move_to_end(key)
            return cached
        rollout_model = self._require_rollout_model()
        resolved = _ResolvedSequenceDeckBatch(
            route_plan=resolve_simple_exact_routes(
                key,
                rollout_model.config,
                device=self.device,
            ),
            token_rows=torch.tensor(
                tuple(self._deck_token_rows[signature] for signature in key),
                dtype=torch.long,
                device=self.device,
            ),
        )
        self._resolved_deck_batches[key] = resolved
        if len(self._resolved_deck_batches) > 256:
            self._resolved_deck_batches.popitem(last=False)
        return resolved

    def _require_rollout_model(self) -> SimpleStatelessPolicyValueNet:
        """Return the live inference model or reject post-close use."""
        model = self._rollout_model
        if model is None:
            raise RuntimeError("sequence actor rollout model is released")
        return model


def _accepted_action_batch_float_dtype(
    batch: AcceptedActionBatch,
    *,
    dtype: torch.dtype,
) -> AcceptedActionBatch:
    """Cast canonical actions only after exact FP32 semantic-byte sorting."""
    if batch.option_scalars.dtype == dtype and batch.entity_scalars.dtype == dtype:
        return batch
    return replace(
        batch,
        option_scalars=batch.option_scalars.to(dtype=dtype),
        entity_scalars=batch.entity_scalars.to(dtype=dtype),
    )


def _await_then_return(
    await_callback: Callable[[], None] | None,
    result: SequenceActorHostTransfer,
) -> SequenceActorHostTransfer:
    """Run one fact barrier before returning an already prepared transfer."""
    if await_callback is not None:
        await_callback()
    return result


def _resident_bfloat16_shadow(
    model: SimpleStatelessPolicyValueNet,
    *,
    device: torch.device,
) -> SimpleStatelessPolicyValueNet:
    """Clone one FP32 master into a gradient-free resident BF16 rollout model."""
    parameter_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    if parameter_dtypes == {torch.bfloat16}:
        # Distributed workers receive an already authenticated BF16 wire
        # artifact and intentionally own no FP32 learner master. Reuse that
        # sole resident model instead of doubling worker VRAM.
        shadow = model.to(device=device).eval()
    elif parameter_dtypes == {torch.float32}:
        shadow = copy.deepcopy(model)
        shadow.to(device=device, dtype=torch.bfloat16).eval()
    else:
        raise ValueError(
            "BF16 rollout model requires a uniform FP32 source or BF16 artifact"
        )
    shadow.requires_grad_(False)
    for parameter in shadow.parameters():
        parameter.grad = None
    mark_immutable_rollout_cache_generation(shadow)
    if (
        device.type == "cuda"
        and isinstance(shadow, SimpleStatelessPolicyValueNet)
        and shadow.supports_bfloat16_rollout_inductor
    ):
        shadow.enable_bfloat16_rollout_inductor()
    return shadow


def _unordered_action_rows(
    rows: Sequence[SimpleStatelessActorRow],
    *,
    host_semantic_batch: NativeSimpleStatelessPolicyBatch | None,
) -> tuple[int, ...]:
    """Return host-known rows that require stable semantic canonicalization."""
    if host_semantic_batch is None:
        return tuple(
            row_index
            for row_index, row in enumerate(rows)
            if (
                len(row.options)
                and int(row.options.contexts[0]) in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
            )
        )
    contexts = host_semantic_batch.options.contexts[:, 0].tolist()
    return tuple(
        row_index
        for row_index, context in enumerate(contexts)
        if int(context) in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
    )


__all__ = [
    "GeneralistSequenceActorPolicy",
    "SequenceActorSampleContinuation",
    "SequenceRolloutPrecision",
]
