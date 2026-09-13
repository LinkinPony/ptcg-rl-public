"""Persistent grow-only shared tensor slabs for native rollout workers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from ptcg_rl.context.public_event_arrays import PublicEventBatch
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.simple_stateless.belief import PublicBeliefSummaryBatch
from ptcg_rl.model.state_encoder import StateBatch
from ptcg_rl.rl.native_policy_batch import NativeSimpleStatelessPolicyBatch

_SAMPLING_UNIFORMS = "sampling_uniforms"
_PUBLIC_EVENT_PREFIX = "public_event."


@dataclass(frozen=True, slots=True)
class NativeSharedBatchMetadata:
    """Non-tensor leaves required to reconstruct one exact policy batch."""

    state_root_input_fingerprints: tuple[str, ...]
    state_sequence_lengths: tuple[int, ...]
    option_lengths: tuple[int, ...]
    option_maximum_counts: tuple[int, ...]
    deck_signatures: tuple[str, ...]
    min_counts: tuple[int, ...]
    max_counts: tuple[int, ...]
    belief_catalog_fingerprint: str
    public_deck_catalog_fingerprint: str
    input_contract_fingerprint: str


@dataclass(frozen=True, slots=True)
class NativeSharedBatchDescriptor:
    """Small per-request descriptor plus rare slab registration payload."""

    generation: int
    tensor_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    metadata: NativeSharedBatchMetadata
    registered_tensors: Mapping[str, Tensor] | None = None

    @property
    def batch_size(self) -> int:
        """Return decision rows without resolving shared tensor views."""
        return len(self.metadata.deck_signatures)


class NativeSharedBatchWriter:
    """Copy worker batches into persistent shared storage before requesting CUDA."""

    def __init__(self) -> None:
        self._generation = 0
        self._buffers: dict[str, Tensor] = {}

    def write(
        self,
        batch: NativeSimpleStatelessPolicyBatch,
        *,
        sampling_uniforms: Tensor | None = None,
        public_events: PublicEventBatch | None = None,
    ) -> NativeSharedBatchDescriptor:
        """Publish one batch; registrations occur only after a slab grows."""
        sources = _batch_tensor_mapping(batch)
        if sampling_uniforms is not None:
            sources[_SAMPLING_UNIFORMS] = sampling_uniforms
        if public_events is not None:
            if public_events.batch_size != batch.batch_size:
                raise ValueError("native shared public events are misaligned")
            sources.update(_public_event_tensor_mapping(public_events))
        if any(value.device.type != "cpu" for value in sources.values()):
            raise ValueError("native shared slabs accept only CPU tensors")

        grew = False
        for name, source in sources.items():
            buffer = self._buffers.get(name)
            if (
                buffer is None
                or buffer.dtype != source.dtype
                or buffer.ndim != source.ndim
                or any(
                    int(capacity) < int(required)
                    for capacity, required in zip(
                        buffer.shape,
                        source.shape,
                        strict=True,
                    )
                )
            ):
                buffer = torch.empty(
                    _capacity_shape(source.shape),
                    dtype=source.dtype,
                    device="cpu",
                )
                cast(Any, buffer).share_memory_()
                self._buffers[name] = buffer
                grew = True
            _tensor_view(buffer, tuple(int(value) for value in source.shape)).copy_(
                source
            )

        if grew:
            self._generation += 1
        if self._generation <= 0:
            raise RuntimeError("native shared slab was not initialized")
        return NativeSharedBatchDescriptor(
            generation=self._generation,
            tensor_shapes=tuple(
                (name, tuple(int(value) for value in tensor.shape))
                for name, tensor in sorted(sources.items())
            ),
            metadata=_batch_metadata(batch),
            registered_tensors=(dict(self._buffers) if grew else None),
        )


class NativeSharedBatchRegistry:
    """Resolve worker descriptors against parent-retained shared mappings."""

    def __init__(self) -> None:
        self._workers: dict[int, tuple[int, dict[str, Tensor]]] = {}

    def resolve(
        self,
        worker_index: int,
        descriptor: NativeSharedBatchDescriptor,
    ) -> tuple[
        NativeSimpleStatelessPolicyBatch,
        Tensor | None,
        PublicEventBatch | None,
    ]:
        """Return zero-copy tensor views and the optional sampling table."""
        if descriptor.registered_tensors is not None:
            previous = self._workers.get(worker_index)
            if previous is not None and descriptor.generation <= previous[0]:
                raise RuntimeError("native shared slab generation did not advance")
            registered = dict(descriptor.registered_tensors)
            if not registered or any(
                tensor.device.type != "cpu"
                or not cast(Any, tensor).is_shared()
                for tensor in registered.values()
            ):
                raise ValueError("native shared slab registration is invalid")
            self._workers[worker_index] = (descriptor.generation, registered)
        try:
            generation, buffers = self._workers[worker_index]
        except KeyError as error:
            raise RuntimeError("native shared slab was not registered") from error
        if generation != descriptor.generation:
            raise RuntimeError("native shared slab generation is stale")

        tensors: dict[str, Tensor] = {}
        for name, shape in descriptor.tensor_shapes:
            if name in tensors:
                raise ValueError("native shared descriptor duplicated a tensor")
            try:
                buffer = buffers[name]
            except KeyError as error:
                raise RuntimeError("native shared descriptor tensor is absent") from error
            if buffer.ndim != len(shape) or any(
                required < 0 or required > int(capacity)
                for required, capacity in zip(shape, buffer.shape, strict=True)
            ):
                raise ValueError("native shared descriptor shape is invalid")
            tensors[name] = _tensor_view(buffer, shape)
        batch = _batch_from_tensors(tensors, descriptor.metadata)
        return (
            batch,
            tensors.get(_SAMPLING_UNIFORMS),
            _public_events_from_tensors(tensors),
        )


def _public_event_tensor_mapping(batch: PublicEventBatch) -> dict[str, Tensor]:
    return {
        f"{_PUBLIC_EVENT_PREFIX}{name}": getattr(batch, name)
        for name in (
            "event_types",
            "actor_roles",
            "from_areas",
            "to_areas",
            "card_ids",
            "serials",
            "entity_mask",
            "attack_ids",
            "attack_id_mask",
            "values",
            "value_mask",
            "categorical_values",
            "padding_mask",
            "overflow_type_actor_counts",
        )
    }


def _public_events_from_tensors(
    tensors: Mapping[str, Tensor],
) -> PublicEventBatch | None:
    names = (
        "event_types",
        "actor_roles",
        "from_areas",
        "to_areas",
        "card_ids",
        "serials",
        "entity_mask",
        "attack_ids",
        "attack_id_mask",
        "values",
        "value_mask",
        "categorical_values",
        "padding_mask",
        "overflow_type_actor_counts",
    )
    present = tuple(
        f"{_PUBLIC_EVENT_PREFIX}{name}" in tensors for name in names
    )
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError("native shared public-event tensors are incomplete")
    return PublicEventBatch(
        **{
            name: tensors[f"{_PUBLIC_EVENT_PREFIX}{name}"]
            for name in names
        }
    )


def _batch_tensor_mapping(
    batch: NativeSimpleStatelessPolicyBatch,
) -> dict[str, Tensor]:
    states = batch.states
    options = batch.options
    belief = batch.belief_summary
    tensors = {
        "state.card_ids": states.card_ids,
        "state.areas": states.areas,
        "state.owner_roles": states.owner_roles,
        "state.token_kinds": states.token_kinds,
        "state.scalars": states.scalars,
        "state.last_attack_ids": states.last_attack_ids,
        "state.padding_mask": states.padding_mask,
        "option.option_types": options.option_types,
        "option.contexts": options.contexts,
        "option.entity_slots": options.entity_slots,
        "option.entity_slot_mask": options.entity_slot_mask,
        "option.attack_ids": options.attack_ids,
        "option.card_ids": options.card_ids,
        "option.scalars": options.scalars,
        "option.dynamic_effect_features": options.dynamic_effect_features,
        "option.dynamic_effect_masks": options.dynamic_effect_masks,
        "option.valid_options": options.valid_options,
        "option.min_counts": options.min_counts,
        "option.max_counts": options.max_counts,
        "deck.card_ids": batch.unique_deck_card_ids,
        "deck.counts": batch.deck_counts,
        "deck.valid_mask": batch.deck_valid_mask,
        "belief.card_ids": belief.card_ids,
        "belief.expected_counts": belief.expected_counts,
        "belief.valid_mask": belief.valid_mask,
        "belief.scalars": belief.scalars,
    }
    optional = {
        "state.attachment_card_ids": states.attachment_card_ids,
        "state.attachment_parent_indices": states.attachment_parent_indices,
        "state.attachment_kinds": states.attachment_kinds,
        "state.entity_slots": states.entity_slots,
        "belief.row_indices": belief.row_indices,
    }
    tensors.update(
        {name: value for name, value in optional.items() if value is not None}
    )
    return tensors


def _batch_metadata(
    batch: NativeSimpleStatelessPolicyBatch,
) -> NativeSharedBatchMetadata:
    return NativeSharedBatchMetadata(
        state_root_input_fingerprints=batch.states.root_input_fingerprints,
        state_sequence_lengths=batch.states.sequence_lengths,
        option_lengths=batch.options.option_lengths,
        option_maximum_counts=batch.options.maximum_counts,
        deck_signatures=batch.deck_signatures,
        min_counts=batch.min_counts,
        max_counts=batch.max_counts,
        belief_catalog_fingerprint=batch.belief_summary.catalog_fingerprint,
        public_deck_catalog_fingerprint=batch.public_deck_catalog_fingerprint,
        input_contract_fingerprint=batch.input_contract_fingerprint,
    )


def _batch_from_tensors(
    tensors: Mapping[str, Tensor],
    metadata: NativeSharedBatchMetadata,
) -> NativeSimpleStatelessPolicyBatch:
    def required(name: str) -> Tensor:
        try:
            return tensors[name]
        except KeyError as error:
            raise RuntimeError(f"native shared tensor {name!r} is absent") from error

    states = StateBatch(
        card_ids=required("state.card_ids"),
        areas=required("state.areas"),
        owner_roles=required("state.owner_roles"),
        token_kinds=required("state.token_kinds"),
        scalars=required("state.scalars"),
        last_attack_ids=required("state.last_attack_ids"),
        padding_mask=required("state.padding_mask"),
        attachment_card_ids=tensors.get("state.attachment_card_ids"),
        attachment_parent_indices=tensors.get(
            "state.attachment_parent_indices"
        ),
        attachment_kinds=tensors.get("state.attachment_kinds"),
        entity_slots=tensors.get("state.entity_slots"),
        root_input_fingerprints=metadata.state_root_input_fingerprints,
        sequence_lengths=metadata.state_sequence_lengths,
    )
    options = OptionBatch(
        option_types=required("option.option_types"),
        contexts=required("option.contexts"),
        entity_slots=required("option.entity_slots"),
        entity_slot_mask=required("option.entity_slot_mask"),
        attack_ids=required("option.attack_ids"),
        card_ids=required("option.card_ids"),
        scalars=required("option.scalars"),
        dynamic_effect_features=required("option.dynamic_effect_features"),
        dynamic_effect_masks=required("option.dynamic_effect_masks"),
        valid_options=required("option.valid_options"),
        min_counts=required("option.min_counts"),
        max_counts=required("option.max_counts"),
        option_lengths=metadata.option_lengths,
        maximum_counts=metadata.option_maximum_counts,
    )
    belief = PublicBeliefSummaryBatch(
        card_ids=required("belief.card_ids"),
        expected_counts=required("belief.expected_counts"),
        valid_mask=required("belief.valid_mask"),
        scalars=required("belief.scalars"),
        catalog_fingerprint=metadata.belief_catalog_fingerprint,
        row_indices=tensors.get("belief.row_indices"),
    )
    return NativeSimpleStatelessPolicyBatch(
        states=states,
        options=options,
        unique_deck_card_ids=required("deck.card_ids"),
        deck_counts=required("deck.counts"),
        deck_valid_mask=required("deck.valid_mask"),
        deck_signatures=metadata.deck_signatures,
        belief_summary=belief,
        min_counts=metadata.min_counts,
        max_counts=metadata.max_counts,
        public_deck_catalog_fingerprint=(
            metadata.public_deck_catalog_fingerprint
        ),
        input_contract_fingerprint=metadata.input_contract_fingerprint,
    )


def _capacity_shape(shape: torch.Size) -> tuple[int, ...]:
    return tuple(_next_power_of_two(max(int(value), 1)) for value in shape)


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _tensor_view(buffer: Tensor, shape: tuple[int, ...]) -> Tensor:
    return buffer[tuple(slice(0, size) for size in shape)]


__all__ = [
    "NativeSharedBatchDescriptor",
    "NativeSharedBatchRegistry",
    "NativeSharedBatchWriter",
]
