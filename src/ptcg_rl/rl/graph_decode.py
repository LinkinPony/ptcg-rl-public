"""CUDA graph backed static-shape rollout decoding."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from ptcg_rl.decks import (
    DeckBatch,
    copy_deck_batch_,
    empty_deck_batch_like,
)
from ptcg_rl.model import AgentPolicyValueNet, OptionBatch, StateBatch
from ptcg_rl.model.deck_conditioning import DeckRoutePlan, resolve_deck_route_plan
from ptcg_rl.model.network import SampleDecodeTensorTrace, SampleDecodeTrace
from ptcg_rl.model.policy import actions_from_decode_tensors


@dataclass(frozen=True)
class CudaGraphDecodeKey:
    """Static tensor shape for one captured decode bucket."""

    batch_size: int
    tokens: int
    options: int
    state_scalar_width: int
    attachments: int
    entity_slots: int
    option_scalar_width: int
    effect_width: int
    max_select_steps: int
    device: str
    deck_signature: str
    private_route_key: str
    resolved_registry_sha256: str


@dataclass(frozen=True)
class CudaGraphDecodeStats:
    """Lifetime CUDA graph cache activity for serving diagnostics."""

    captures: int
    replay_hits: int
    replay_calls: int
    eager_fallbacks: int
    capture_deferrals: int
    replay_deferrals: int
    capture_rejections: int
    evictions: int
    resident_captures: int

    def as_dict(self) -> dict[str, int]:
        """Return stable counter names for JSON/runtime aggregation."""
        return {
            "captures": self.captures,
            "replay_hits": self.replay_hits,
            "replay_calls": self.replay_calls,
            "eager_fallbacks": self.eager_fallbacks,
            "capture_deferrals": self.capture_deferrals,
            "replay_deferrals": self.replay_deferrals,
            "capture_rejections": self.capture_rejections,
            "evictions": self.evictions,
            "resident_captures": self.resident_captures,
        }


@dataclass
class _CapturedDecode:
    """One captured CUDA graph and its static buffers."""

    graph: Any
    states: StateBatch
    options: OptionBatch
    decks: DeckBatch
    route_plan: DeckRoutePlan | None
    temperature: Tensor
    gumbel_noise: Tensor
    choice_indices: Tensor
    append_masks: Tensor
    action_logprobs: Tensor
    values: Tensor
    token_logprobs: Tensor
    prefix_values: Tensor
    token_mask: Tensor
    stop_sampled: Tensor


class CudaGraphDecodeRunner:
    """Lazy per-bucket CUDA graph runner for static rollout decoding."""

    def __init__(
        self,
        model: AgentPolicyValueNet,
        *,
        warmup_steps: int = 2,
        max_captures: int = 0,
        capture_idle_replays: int = 64,
        capture_allowed: Callable[[], bool] | None = None,
        replay_allowed: Callable[[], bool] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        """Initialize an empty capture cache."""
        if model.config.recurrent is not None:
            raise ValueError(
                "CUDA graph decode does not support recurrent models: public "
                "events and recurrent state are not part of CudaGraphDecodeKey "
                "or captured buffers"
            )
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if max_captures < 0:
            raise ValueError("max_captures must be non-negative")
        if capture_idle_replays <= 0:
            raise ValueError("capture_idle_replays must be positive")
        self._model = model
        self._warmup_steps = int(warmup_steps)
        self._max_captures = int(max_captures)
        self._capture_idle_replays = int(capture_idle_replays)
        self._capture_allowed = capture_allowed
        # Preserve the original one-gate behavior for callers that do not opt
        # into concurrent learner-owned replay.
        self._replay_allowed = (
            capture_allowed if replay_allowed is None else replay_allowed
        )
        self._generator = generator
        self._captures: dict[CudaGraphDecodeKey, _CapturedDecode] = {}
        self._capture_last_used: dict[CudaGraphDecodeKey, int] = {}
        self._replay_index = 0
        self._capture_count = 0
        self._replay_hit_count = 0
        self._replay_call_count = 0
        self._eager_fallback_count = 0
        self._capture_deferral_count = 0
        self._replay_deferral_count = 0
        self._capture_rejection_count = 0
        self._eviction_count = 0

    @property
    def captured_buckets(self) -> int:
        """Return the number of lazily captured decode buckets."""
        return len(self._captures)

    def stats(self) -> CudaGraphDecodeStats:
        """Return a side-effect-free snapshot of graph cache counters."""
        return CudaGraphDecodeStats(
            captures=self._capture_count,
            replay_hits=self._replay_hit_count,
            replay_calls=self._replay_call_count,
            eager_fallbacks=self._eager_fallback_count,
            capture_deferrals=self._capture_deferral_count,
            replay_deferrals=self._replay_deferral_count,
            capture_rejections=self._capture_rejection_count,
            evictions=self._eviction_count,
            resident_captures=len(self._captures),
        )

    def sample_decode_static(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
        max_select_steps: int,
        gumbel_noise: Tensor | None = None,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        """Replay a static graph and materialize sampled action tuples."""
        trace = self.sample_decode_static_with_trace(
            states,
            options,
            decks,
            temperature=temperature,
            max_select_steps=max_select_steps,
            gumbel_noise=gumbel_noise,
        )
        return (trace.actions, trace.action_logprobs, trace.values)

    def sample_decode_static_with_trace(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
        max_select_steps: int,
        gumbel_noise: Tensor | None = None,
    ) -> SampleDecodeTrace:
        """Replay a static graph and materialize its token-level trace."""
        trace = self.sample_decode_tensors_with_trace(
            states,
            options,
            decks,
            temperature=temperature,
            max_select_steps=max_select_steps,
            gumbel_noise=gumbel_noise,
        )
        return SampleDecodeTrace(
            actions=actions_from_decode_tensors(
                trace.choice_indices,
                trace.append_masks,
            ),
            action_logprobs=trace.action_logprobs,
            values=trace.values,
            token_logprobs=trace.token_logprobs,
            prefix_values=trace.prefix_values,
            token_mask=trace.token_mask,
            stop_sampled=trace.stop_sampled,
        )

    def sample_decode_tensors(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
        max_select_steps: int,
        gumbel_noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Replay a static graph and return tensor decode traces."""
        trace = self.sample_decode_tensors_with_trace(
            states,
            options,
            decks,
            temperature=temperature,
            max_select_steps=max_select_steps,
            gumbel_noise=gumbel_noise,
        )
        return (
            trace.choice_indices,
            trace.append_masks,
            trace.action_logprobs,
            trace.values,
        )

    def sample_decode_tensors_with_trace(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
        max_select_steps: int,
        gumbel_noise: Tensor | None = None,
    ) -> SampleDecodeTensorTrace:
        """Replay a static graph and return all token-level tensors."""
        required_steps = int(options.max_counts.max().item())
        if max_select_steps < required_steps:
            raise ValueError("max_select_steps must cover every normalized max_count")
        if temperature == 0.0:
            self._eager_fallback_count += 1
            return self._model.sample_decode_tensors_with_trace(
                states,
                options,
                decks,
                temperature=temperature,
                max_select_steps=max_select_steps,
                generator=self._generator,
            )
        _validate_cuda_inputs(states, options, decks)
        _validate_positive_temperature(temperature)
        key = _decode_key(
            self._model,
            states,
            options,
            decks,
            max_select_steps=max_select_steps,
        )
        self._replay_index += 1
        capture = self._captures.get(key)
        if (
            capture is not None
            and self._replay_allowed is not None
            and not self._replay_allowed()
        ):
            # A deferred resident route remains part of the active working set.
            self._replay_deferral_count += 1
            self._capture_last_used[key] = self._replay_index
            self._eager_fallback_count += 1
            return self._model.sample_decode_tensors_with_trace(
                states,
                options,
                decks,
                temperature=temperature,
                max_select_steps=max_select_steps,
                gumbel_noise=gumbel_noise,
                generator=self._generator,
            )
        if capture is None:
            if self._capture_allowed is not None and not self._capture_allowed():
                self._capture_deferral_count += 1
                self._eager_fallback_count += 1
                return self._model.sample_decode_tensors_with_trace(
                    states,
                    options,
                    decks,
                    temperature=temperature,
                    max_select_steps=max_select_steps,
                    gumbel_noise=gumbel_noise,
                    generator=self._generator,
                )
            if not self._admit_capture():
                self._capture_rejection_count += 1
                self._eager_fallback_count += 1
                return self._model.sample_decode_tensors_with_trace(
                    states,
                    options,
                    decks,
                    temperature=temperature,
                    max_select_steps=max_select_steps,
                    gumbel_noise=gumbel_noise,
                    generator=self._generator,
                )
            capture = self._capture_bucket(
                states,
                options,
                decks,
                temperature=temperature,
                max_select_steps=max_select_steps,
                gumbel_noise=gumbel_noise,
            )
            self._captures[key] = capture
            self._capture_count += 1
        else:
            self._replay_hit_count += 1
        self._capture_last_used[key] = self._replay_index
        _copy_inputs(
            capture.states,
            capture.options,
            capture.decks,
            states,
            options,
            decks,
        )
        _copy_temperature(capture.temperature, temperature)
        if gumbel_noise is None:
            _fill_gumbel_noise(
                capture.gumbel_noise,
                generator=self._generator,
            )
        else:
            _copy_gumbel_noise(capture.gumbel_noise, gumbel_noise)
        try:
            cast(Any, capture.graph).replay()
            self._replay_call_count += 1
        except torch.AcceleratorError as exc:
            raise RuntimeError(f"CUDA graph decode replay failed for {key}") from exc
        return SampleDecodeTensorTrace(
            # A single static graph may serve several chunks in one inference
            # step. Preserve each replay before the next one overwrites the
            # graph-owned output buffers.
            choice_indices=capture.choice_indices.clone(),
            append_masks=capture.append_masks.clone(),
            action_logprobs=capture.action_logprobs.clone(),
            values=capture.values.clone(),
            token_logprobs=capture.token_logprobs.clone(),
            prefix_values=capture.prefix_values.clone(),
            token_mask=capture.token_mask.clone(),
            stop_sampled=capture.stop_sampled.clone(),
        )

    def _admit_capture(self) -> bool:
        """Bound graph pools without thrashing during deck-cohort transitions."""
        if self._max_captures == 0 or len(self._captures) < self._max_captures:
            return True
        oldest_key = min(
            self._capture_last_used,
            key=self._capture_last_used.__getitem__,
        )
        idle_replays = self._replay_index - self._capture_last_used[oldest_key]
        if idle_replays < self._capture_idle_replays:
            return False
        retired = self._captures.pop(oldest_key)
        self._capture_last_used.pop(oldest_key)
        self._eviction_count += 1
        del retired
        torch.cuda.empty_cache()
        return True

    def _capture_bucket(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float,
        max_select_steps: int,
        gumbel_noise: Tensor | None,
    ) -> _CapturedDecode:
        static_states = _empty_state_like(states)
        static_options = _empty_option_like(options)
        static_decks = empty_deck_batch_like(decks)
        _copy_inputs(
            static_states,
            static_options,
            static_decks,
            states,
            options,
            decks,
        )
        static_temperature = torch.empty((), device=states.card_ids.device)
        _copy_temperature(static_temperature, temperature)
        static_noise = torch.empty(
            _gumbel_noise_shape(options, max_select_steps=max_select_steps),
            dtype=torch.float32,
            device=states.card_ids.device,
        )
        if gumbel_noise is None:
            _fill_gumbel_noise(static_noise, generator=self._generator)
        else:
            _copy_gumbel_noise(static_noise, gumbel_noise)

        conditioning = self._model.config.deck_conditioning
        route_plan: DeckRoutePlan | None = None
        if conditioning is not None and conditioning.enabled:
            # Materialize immutable route indices before CUDA capture. The
            # model resolves the same cached plan inside the captured call.
            # Retain it with the graph: the global bounded route-plan cache is
            # shared with shuffled learner layouts, so cache eviction must not
            # free CUDA index tensors whose addresses the graph captured.
            route_plan = resolve_deck_route_plan(static_decks, conditioning)

        with _mha_fastpath_disabled():
            _warmup_decode(
                self._model,
                states=static_states,
                options=static_options,
                decks=static_decks,
                temperature=static_temperature,
                max_select_steps=max_select_steps,
                gumbel_noise=static_noise,
                warmup_steps=self._warmup_steps,
            )

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                trace = self._model.sample_decode_tensors_with_trace(
                    static_states,
                    static_options,
                    static_decks,
                    temperature=static_temperature,
                    max_select_steps=max_select_steps,
                    gumbel_noise=static_noise,
                    validate_decode_cap=False,
                )
        return _CapturedDecode(
            graph=graph,
            states=static_states,
            options=static_options,
            decks=static_decks,
            route_plan=route_plan,
            temperature=static_temperature,
            gumbel_noise=static_noise,
            choice_indices=trace.choice_indices,
            append_masks=trace.append_masks,
            action_logprobs=trace.action_logprobs,
            values=trace.values,
            token_logprobs=trace.token_logprobs,
            prefix_values=trace.prefix_values,
            token_mask=trace.token_mask,
            stop_sampled=trace.stop_sampled,
        )


def _validate_cuda_inputs(
    states: StateBatch,
    options: OptionBatch,
    decks: DeckBatch,
) -> None:
    if states.card_ids.device.type != "cuda":
        raise RuntimeError("CudaGraphDecodeRunner requires CUDA state tensors")
    if options.valid_options.device != states.card_ids.device:
        raise RuntimeError("state and option tensors must be on the same CUDA device")
    if decks.card_ids.device != states.card_ids.device:
        raise RuntimeError("state and deck tensors must be on the same CUDA device")
    if len(decks) != int(states.card_ids.shape[0]):
        raise ValueError("deck batch must align with state batch rows")


def _validate_positive_temperature(temperature: float) -> None:
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0.0:
        return


def _decode_key(
    model: AgentPolicyValueNet,
    states: StateBatch,
    options: OptionBatch,
    decks: DeckBatch,
    *,
    max_select_steps: int,
) -> CudaGraphDecodeKey:
    if max_select_steps < 0:
        raise ValueError("max_select_steps must be non-negative")
    signatures = tuple(decks.signatures)
    unique_signatures = set(signatures)
    deck_signature = (
        signatures[0]
        if len(unique_signatures) == 1
        else f"layout:{_signature_layout_fingerprint(signatures)}"
    )
    conditioning = model.config.deck_conditioning
    if conditioning is None or not conditioning.enabled:
        private_route_key = "legacy"
        resolved_registry_sha256 = ""
    else:
        if conditioning.resolved_registry_sha256 is None:
            raise RuntimeError("enabled deck conditioning registry is unresolved")
        profile_by_signature = conditioning.profile_by_signature
        route_keys = tuple(
            (
                "generic"
                if profile_by_signature.get(signature) is None
                else cast(Any, profile_by_signature[signature]).module_key
            )
            for signature in signatures
        )
        private_route_key = (
            route_keys[0]
            if len(set(route_keys)) == 1
            else f"layout:{_signature_layout_fingerprint(route_keys)}"
        )
        resolved_registry_sha256 = conditioning.resolved_registry_sha256
    return CudaGraphDecodeKey(
        batch_size=int(states.card_ids.shape[0]),
        tokens=int(states.card_ids.shape[1]),
        options=int(options.valid_options.shape[1]),
        state_scalar_width=int(states.scalars.shape[2]),
        attachments=(
            0
            if states.attachment_card_ids is None
            else int(states.attachment_card_ids.shape[1])
        ),
        entity_slots=int(options.entity_slots.shape[2]),
        option_scalar_width=int(options.scalars.shape[2]),
        effect_width=int(options.dynamic_effect_features.shape[2]),
        max_select_steps=int(max_select_steps),
        device=str(states.card_ids.device),
        deck_signature=deck_signature,
        private_route_key=private_route_key,
        resolved_registry_sha256=resolved_registry_sha256,
    )


def _signature_layout_fingerprint(values: tuple[str, ...]) -> str:
    """Hash one exact CPU routing layout for a CUDA graph cache key."""
    digest = hashlib.sha256()
    digest.update(b"ptcg-rl/cuda-graph-route-layout/v1\0")
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _gumbel_noise_shape(
    options: OptionBatch,
    *,
    max_select_steps: int,
) -> tuple[int, int, int]:
    batch_size, max_options = options.valid_options.shape
    decode_steps = min(int(max_options), int(max_select_steps)) + 1
    return (int(batch_size), decode_steps, int(max_options) + 1)


def _empty_state_like(states: StateBatch) -> StateBatch:
    return StateBatch(
        card_ids=torch.empty_like(states.card_ids),
        areas=torch.empty_like(states.areas),
        owner_roles=torch.empty_like(states.owner_roles),
        token_kinds=torch.empty_like(states.token_kinds),
        scalars=torch.empty_like(states.scalars),
        last_attack_ids=torch.empty_like(states.last_attack_ids),
        padding_mask=torch.empty_like(states.padding_mask),
        attachment_card_ids=_empty_optional_tensor(states.attachment_card_ids),
        attachment_parent_indices=_empty_optional_tensor(
            states.attachment_parent_indices
        ),
        attachment_kinds=_empty_optional_tensor(states.attachment_kinds),
        entity_slots=_empty_optional_tensor(states.entity_slots),
    )


def _empty_option_like(options: OptionBatch) -> OptionBatch:
    return OptionBatch(
        option_types=torch.empty_like(options.option_types),
        contexts=torch.empty_like(options.contexts),
        entity_slots=torch.empty_like(options.entity_slots),
        entity_slot_mask=torch.empty_like(options.entity_slot_mask),
        attack_ids=torch.empty_like(options.attack_ids),
        card_ids=torch.empty_like(options.card_ids),
        scalars=torch.empty_like(options.scalars),
        dynamic_effect_features=torch.empty_like(options.dynamic_effect_features),
        dynamic_effect_masks=torch.empty_like(options.dynamic_effect_masks),
        valid_options=torch.empty_like(options.valid_options),
        min_counts=torch.empty_like(options.min_counts),
        max_counts=torch.empty_like(options.max_counts),
    )


def _copy_inputs(
    target_states: StateBatch,
    target_options: OptionBatch,
    target_decks: DeckBatch,
    source_states: StateBatch,
    source_options: OptionBatch,
    source_decks: DeckBatch,
) -> None:
    target_states.card_ids.copy_(source_states.card_ids)
    target_states.areas.copy_(source_states.areas)
    target_states.owner_roles.copy_(source_states.owner_roles)
    target_states.token_kinds.copy_(source_states.token_kinds)
    target_states.scalars.copy_(source_states.scalars)
    target_states.last_attack_ids.copy_(source_states.last_attack_ids)
    target_states.padding_mask.copy_(source_states.padding_mask)
    _copy_optional_tensor(
        target_states.attachment_card_ids,
        source_states.attachment_card_ids,
        name="attachment_card_ids",
    )
    _copy_optional_tensor(
        target_states.attachment_parent_indices,
        source_states.attachment_parent_indices,
        name="attachment_parent_indices",
    )
    _copy_optional_tensor(
        target_states.attachment_kinds,
        source_states.attachment_kinds,
        name="attachment_kinds",
    )
    _copy_optional_tensor(
        target_states.entity_slots,
        source_states.entity_slots,
        name="entity_slots",
    )
    target_options.option_types.copy_(source_options.option_types)
    target_options.contexts.copy_(source_options.contexts)
    target_options.entity_slots.copy_(source_options.entity_slots)
    target_options.entity_slot_mask.copy_(source_options.entity_slot_mask)
    target_options.attack_ids.copy_(source_options.attack_ids)
    target_options.card_ids.copy_(source_options.card_ids)
    target_options.scalars.copy_(source_options.scalars)
    target_options.dynamic_effect_features.copy_(source_options.dynamic_effect_features)
    target_options.dynamic_effect_masks.copy_(source_options.dynamic_effect_masks)
    target_options.valid_options.copy_(source_options.valid_options)
    target_options.min_counts.copy_(source_options.min_counts)
    target_options.max_counts.copy_(source_options.max_counts)
    copy_deck_batch_(target_decks, source_decks)


def _empty_optional_tensor(tensor: Tensor | None) -> Tensor | None:
    if tensor is None:
        return None
    return torch.empty_like(tensor)


def _copy_optional_tensor(
    target: Tensor | None,
    source: Tensor | None,
    *,
    name: str,
) -> None:
    if target is None and source is None:
        return
    if target is None or source is None:
        raise ValueError(f"state tensor presence mismatch: {name}")
    target.copy_(source)


def _copy_temperature(target: Tensor, temperature: float) -> None:
    target.fill_(float(temperature))


def _fill_gumbel_noise(
    target: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> None:
    target.exponential_(generator=generator)
    target.log_()
    target.neg_()


def _copy_gumbel_noise(target: Tensor, source: Tensor) -> None:
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(
            f"gumbel_noise must have shape {tuple(target.shape)}, "
            f"got {tuple(source.shape)}"
        )
    if not source.is_floating_point():
        raise ValueError("gumbel_noise must be floating point")
    target.copy_(source)


def _warmup_decode(
    model: AgentPolicyValueNet,
    *,
    states: StateBatch,
    options: OptionBatch,
    decks: DeckBatch,
    temperature: Tensor,
    max_select_steps: int,
    gumbel_noise: Tensor,
    warmup_steps: int,
) -> None:
    if warmup_steps <= 0:
        return
    device = states.card_ids.device
    stream_factory = cast(Any, torch.cuda.Stream)
    warmup_stream = stream_factory(device=device)
    current_stream = torch.cuda.current_stream(device=device)
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream):
        for _ in range(warmup_steps):
            model.sample_decode_tensors_with_trace(
                states,
                options,
                decks,
                temperature=temperature,
                max_select_steps=max_select_steps,
                gumbel_noise=gumbel_noise,
            )
    current_stream.wait_stream(warmup_stream)


@contextmanager
def _mha_fastpath_disabled() -> Iterator[None]:
    previous = bool(torch.backends.mha.get_fastpath_enabled())
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
