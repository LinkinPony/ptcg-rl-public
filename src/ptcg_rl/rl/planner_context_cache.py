"""Bounded GPU policy-context handles shared across planner inference stages."""

from __future__ import annotations

import hashlib
import secrets
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.model.deck_conditioning import (
    DeckConditioningConfig,
    resolve_deck_route_plan,
)
from ptcg_rl.model.network import PolicyEvaluationContext
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.root_input_fingerprint import (
    canonical_planner_root_input_fingerprint,
)
from ptcg_rl.model.state_encoder import StateBatch
from ptcg_rl.runtime.bounded_store import (
    BoundedAdmissionStore,
    BoundedStoreCapacityError,
    BoundedStoreStats,
)


class PlannerContextCapacityError(RuntimeError):
    """Raised when a complete root batch cannot be retained without eviction."""


@dataclass(frozen=True)
class CachedPlannerPolicyRow:
    """One actual-width root context bound to immutable serving identities."""

    policy_global: Tensor
    option_embeddings: Tensor
    projected_options: Tensor
    option_count: int
    deck_signature: str
    policy_version: int
    tensor_schema_fingerprint: str
    deck_conditioned: bool
    root_input_fingerprint: str


class PlannerPolicyContextCache:
    """Issue opaque bounded handles for root contexts retained on device."""

    def __init__(self, capacity: int) -> None:
        self._store = BoundedAdmissionStore[str, CachedPlannerPolicyRow](capacity)
        self._nonce = secrets.token_hex(8)
        self._counter = 0
        self._lock = threading.Lock()

    def store_batch(
        self,
        context: PolicyEvaluationContext,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        policy_version: int,
        tensor_schema_fingerprint: str = "",
    ) -> tuple[str, ...]:
        """Store actual-width rows and return decision-aligned opaque handles."""
        batch_size = int(context.policy_global.shape[0])
        if (
            int(states.card_ids.shape[0]) != batch_size
            or int(options.valid_options.shape[0]) != batch_size
            or len(decks) != batch_size
        ):
            raise ValueError("planner context inputs must align")
        rows: dict[str, CachedPlannerPolicyRow] = {}
        option_counts = options.valid_options.sum(dim=1).detach().cpu().tolist()
        for row_index, raw_count in enumerate(option_counts):
            option_count = int(raw_count)
            handle = self._next_handle(policy_version=policy_version)
            row = slice(row_index, row_index + 1)
            rows[handle] = CachedPlannerPolicyRow(
                # A slice is a view into the full padded decode batch.  One
                # surviving handle must not retain every other row (or the
                # batch-wide padded option storage) on the accelerator.
                policy_global=context.policy_global[row]
                .detach()
                .clone(memory_format=torch.contiguous_format),
                option_embeddings=(
                    context.option_embeddings[row, :option_count]
                    .detach()
                    .clone(memory_format=torch.contiguous_format)
                ),
                projected_options=(
                    context.projected_options[row, :option_count]
                    .detach()
                    .clone(memory_format=torch.contiguous_format)
                ),
                option_count=option_count,
                deck_signature=decks.signatures[row_index],
                policy_version=int(policy_version),
                tensor_schema_fingerprint=tensor_schema_fingerprint,
                deck_conditioned=context.route_plan is not None,
                root_input_fingerprint=_row_root_input_fingerprint(
                    states,
                    options,
                    row_index=row_index,
                    option_count=option_count,
                ),
            )
        try:
            self._store.put_many(rows)
        except BoundedStoreCapacityError as exc:
            raise PlannerContextCapacityError(
                "planner root context store is at capacity"
            ) from exc
        return tuple(rows)

    def bind_schema(
        self,
        handles: Sequence[str],
        *,
        policy_version: int,
        tensor_schema_fingerprint: str,
    ) -> None:
        """Bind handles emitted during decode to the request's wire schema."""
        for handle in handles:
            cached = self._required(handle)
            if cached.policy_version != policy_version:
                raise RuntimeError("planner context model version changed")
            if (
                cached.tensor_schema_fingerprint
                and cached.tensor_schema_fingerprint != tensor_schema_fingerprint
            ):
                raise RuntimeError("planner context schema was already bound")
            self._store.put(
                handle,
                replace(
                    cached,
                    tensor_schema_fingerprint=tensor_schema_fingerprint,
                ),
            )

    def acquire_batch(
        self,
        handles: Sequence[str],
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        policy_version: int,
        tensor_schema_fingerprint: str,
        deck_conditioning: DeckConditioningConfig | None,
    ) -> PolicyEvaluationContext:
        """Reassemble padded rows only after every immutable identity matches."""
        frozen = tuple(handles)
        batch_size = int(options.valid_options.shape[0])
        if (
            len(frozen) != batch_size
            or int(states.card_ids.shape[0]) != batch_size
            or len(decks) != batch_size
        ):
            raise ValueError("planner context handles must align with request rows")
        cached_rows = tuple(self._required(handle) for handle in frozen)
        option_counts = tuple(
            int(value)
            for value in options.valid_options.sum(dim=1).detach().cpu().tolist()
        )
        for row_index, (cached, option_count) in enumerate(
            zip(cached_rows, option_counts, strict=True)
        ):
            if cached.policy_version != policy_version:
                raise RuntimeError("planner context belongs to another model version")
            if cached.tensor_schema_fingerprint != tensor_schema_fingerprint:
                raise RuntimeError("planner context belongs to another tensor schema")
            if cached.option_count != option_count:
                raise RuntimeError("planner context option shape changed")
            if cached.deck_signature != decks.signatures[row_index]:
                raise RuntimeError("planner context deck route changed")
            if cached.root_input_fingerprint != _row_root_input_fingerprint(
                states,
                options,
                row_index=row_index,
                option_count=option_count,
            ):
                raise RuntimeError("planner context root input changed")
        width = int(options.valid_options.shape[1])
        option_embeddings = _pad_and_concatenate(
            tuple(cached.option_embeddings for cached in cached_rows),
            width=width,
        )
        projected_options = _pad_and_concatenate(
            tuple(cached.projected_options for cached in cached_rows),
            width=width,
        )
        conditioned = {cached.deck_conditioned for cached in cached_rows}
        if len(conditioned) != 1:
            raise RuntimeError("planner context mixed deck-conditioning modes")
        route_plan = None
        if conditioned == {True}:
            if deck_conditioning is None:
                raise RuntimeError("planner context lost deck conditioning config")
            route_plan = resolve_deck_route_plan(decks, deck_conditioning)
        return PolicyEvaluationContext(
            policy_global=torch.cat(
                tuple(cached.policy_global for cached in cached_rows),
                dim=0,
            ),
            option_embeddings=option_embeddings,
            projected_options=projected_options,
            route_plan=route_plan,
        )

    def clear(self) -> None:
        """Invalidate all retained GPU contexts after a model publication."""
        self._store.clear()

    def release(self, handles: Sequence[str]) -> int:
        """Explicitly release completed or base-fast-path root contexts."""
        return sum(int(self._store.remove(handle)) for handle in handles)

    def stats(self) -> BoundedStoreStats:
        """Return bounded lifecycle-store access and admission telemetry."""
        return self._store.stats()

    def _required(self, handle: str) -> CachedPlannerPolicyRow:
        cached = self._store.get(handle)
        if cached is None:
            raise RuntimeError("planner policy context handle is unavailable")
        return cached

    def _next_handle(self, *, policy_version: int) -> str:
        with self._lock:
            self._counter += 1
            return f"ctx-{policy_version}-{self._nonce}-{self._counter}"


def _pad_and_concatenate(rows: Sequence[Tensor], *, width: int) -> Tensor:
    padded: list[Tensor] = []
    for row in rows:
        if int(row.shape[1]) > width:
            raise RuntimeError("cached planner option width exceeds request width")
        if int(row.shape[1]) == width:
            padded.append(row)
            continue
        padding = row.new_zeros((1, width - int(row.shape[1]), row.shape[2]))
        padded.append(torch.cat((row, padding), dim=1))
    return torch.cat(tuple(padded), dim=0)


def _row_root_input_fingerprint(
    states: StateBatch,
    options: OptionBatch,
    *,
    row_index: int,
    option_count: int,
) -> str:
    fingerprints = states.root_input_fingerprints
    if fingerprints:
        if len(fingerprints) != int(states.card_ids.shape[0]):
            raise ValueError("planner CPU root fingerprints are misaligned")
        value = fingerprints[row_index]
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("planner CPU root fingerprint must be SHA-256")
        return value
    # Compatibility-only fallback for synthetic/direct callers. Production
    # collation attaches CPU fingerprints before any device transfer.
    digest = hashlib.sha256(b"ptcg-rl/planner-context-root/v1\x00")
    token_count = int((~states.padding_mask[row_index]).sum().item())
    state_tensors = (
        states.card_ids,
        states.areas,
        states.owner_roles,
        states.token_kinds,
        states.scalars,
        states.last_attack_ids,
        states.padding_mask,
    )
    option_tensors = (
        options.option_types,
        options.contexts,
        options.entity_slots,
        options.entity_slot_mask,
        options.attack_ids,
        options.card_ids,
        options.scalars,
        options.dynamic_effect_features,
        options.dynamic_effect_masks,
        options.valid_options,
    )
    for tensor in state_tensors:
        _update_tensor_digest(
            digest,
            tensor[row_index, :token_count],
        )
    for optional_tensor in (
        states.attachment_card_ids,
        states.attachment_parent_indices,
        states.attachment_kinds,
        states.entity_slots,
    ):
        if optional_tensor is None:
            digest.update(b"none\x00")
        else:
            _update_tensor_digest(digest, optional_tensor[row_index])
    for tensor in option_tensors:
        _update_tensor_digest(
            digest,
            tensor[row_index, :option_count],
        )
    _update_tensor_digest(digest, options.min_counts[row_index : row_index + 1])
    _update_tensor_digest(digest, options.max_counts[row_index : row_index + 1])
    return digest.hexdigest()


def _update_tensor_digest(digest: Any, tensor: Tensor) -> None:
    values = tensor.detach().cpu().contiguous()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(tuple(values.shape)).encode("ascii"))
    digest.update(values.numpy().tobytes())


__all__ = [
    "CachedPlannerPolicyRow",
    "PlannerContextCapacityError",
    "PlannerPolicyContextCache",
    "canonical_planner_root_input_fingerprint",
]
