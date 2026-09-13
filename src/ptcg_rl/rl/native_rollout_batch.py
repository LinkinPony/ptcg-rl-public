"""Model-ready pinned tensors written directly by the C++ rollout encoder."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from ptcg_rl.engine.native_rollout import (
    NativeRolloutEncoder,
    NativeRolloutShape,
    _CgTrainRolloutOutput,
)
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.simple_stateless.belief import PublicBeliefSummaryBatch
from ptcg_rl.model.state_encoder import StateBatch
from ptcg_rl.rl.native_policy_batch import NativeSimpleStatelessPolicyBatch

_TOKEN_SCALAR_SIZE = 59
_OPTION_SCALAR_SIZE = 9
_DYNAMIC_EFFECT_SIZE = 33
_MAXIMUM_ENTITY_SLOTS = 2
_BELIEF_SCALAR_SIZE = 4


@dataclass(frozen=True, slots=True)
class NativeRolloutSource:
    """One ordered selected-slot segment from a stateful native arena."""

    encoder: NativeRolloutEncoder
    slots: npt.NDArray[np.uint32]
    perspectives: npt.NDArray[np.int32]

    def __post_init__(self) -> None:
        """Require stable contiguous vectors before crossing the C ABI."""
        if (
            self.slots.ndim != 1
            or self.slots.size <= 0
            or self.slots.dtype != np.uint32
            or not self.slots.flags.c_contiguous
            or self.perspectives.shape != self.slots.shape
            or self.perspectives.dtype != np.int32
            or not self.perspectives.flags.c_contiguous
        ):
            raise ValueError("native rollout source vectors are invalid")

    @classmethod
    def from_arrays(
        cls,
        encoder: NativeRolloutEncoder,
        slots: npt.ArrayLike,
        perspectives: npt.ArrayLike,
    ) -> NativeRolloutSource:
        """Normalize one source without selecting or copying CSR columns."""
        return cls(
            encoder=encoder,
            slots=np.ascontiguousarray(slots, dtype=np.uint32),
            perspectives=np.ascontiguousarray(
                perspectives,
                dtype=np.int32,
            ),
        )


@dataclass(frozen=True, slots=True)
class _CombinedShape:
    row_count: int
    state_token_width: int
    state_attachment_width: int
    option_width: int
    deck_width: int
    belief_row_count: int
    belief_width: int


def encode_native_rollout_sources(
    sources: tuple[NativeRolloutSource, ...],
    *,
    pin_memory: bool,
) -> NativeSimpleStatelessPolicyBatch:
    """Fuse direct selections into one model-ready CPU tensor batch."""
    if not sources:
        raise ValueError("native rollout encoding requires at least one source")
    plans = tuple(
        source.encoder.plan_rows(source.slots, source.perspectives)
        for source in sources
    )
    shape = _combined_shape(plans)
    catalog_fingerprint = sources[0].encoder.catalog_fingerprint
    input_contract_fingerprint = sources[0].encoder.input_contract_fingerprint
    if any(
        source.encoder.catalog_fingerprint != catalog_fingerprint
        or source.encoder.input_contract_fingerprint != input_contract_fingerprint
        for source in sources[1:]
    ):
        raise ValueError("native rollout source identities differ")

    state_card_ids = _empty(
        (shape.row_count, shape.state_token_width),
        dtype=torch.int64,
        pin_memory=pin_memory,
    )
    state_areas = _empty_like(state_card_ids, pin_memory=pin_memory)
    state_owner_roles = _empty_like(state_card_ids, pin_memory=pin_memory)
    state_token_kinds = _empty_like(state_card_ids, pin_memory=pin_memory)
    state_scalars = _empty(
        (
            shape.row_count,
            shape.state_token_width,
            _TOKEN_SCALAR_SIZE,
        ),
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    state_last_attack_ids = _empty_like(
        state_card_ids,
        pin_memory=pin_memory,
    )
    state_padding_mask = _empty(
        (shape.row_count, shape.state_token_width),
        dtype=torch.bool,
        pin_memory=pin_memory,
    )
    state_attachment_card_ids = _empty(
        (shape.row_count, shape.state_attachment_width),
        dtype=torch.uint16,
        pin_memory=pin_memory,
    )
    state_attachment_parent_indices = _empty_like(
        state_attachment_card_ids,
        pin_memory=pin_memory,
    )
    state_attachment_kinds = _empty(
        (shape.row_count, shape.state_attachment_width),
        dtype=torch.uint8,
        pin_memory=pin_memory,
    )
    state_entity_slots = _empty(
        (shape.row_count, shape.state_token_width),
        dtype=torch.uint8,
        pin_memory=pin_memory,
    )
    state_sequence_lengths = _empty(
        (shape.row_count,),
        dtype=torch.int32,
        pin_memory=pin_memory,
    )

    option_types = _empty(
        (shape.row_count, shape.option_width),
        dtype=torch.int64,
        pin_memory=pin_memory,
    )
    option_contexts = _empty_like(option_types, pin_memory=pin_memory)
    option_entity_slots = _empty(
        (
            shape.row_count,
            shape.option_width,
            _MAXIMUM_ENTITY_SLOTS,
        ),
        dtype=torch.int64,
        pin_memory=pin_memory,
    )
    option_entity_slot_mask = _empty(
        (
            shape.row_count,
            shape.option_width,
            _MAXIMUM_ENTITY_SLOTS,
        ),
        dtype=torch.bool,
        pin_memory=pin_memory,
    )
    option_attack_ids = _empty_like(option_types, pin_memory=pin_memory)
    option_card_ids = _empty_like(option_types, pin_memory=pin_memory)
    option_scalars = _empty(
        (
            shape.row_count,
            shape.option_width,
            _OPTION_SCALAR_SIZE,
        ),
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    option_dynamic_effect_features = _empty(
        (
            shape.row_count,
            shape.option_width,
            _DYNAMIC_EFFECT_SIZE,
        ),
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    option_dynamic_effect_masks = _empty(
        (shape.row_count, shape.option_width),
        dtype=torch.bool,
        pin_memory=pin_memory,
    )
    option_valid = _empty_like(
        option_dynamic_effect_masks,
        pin_memory=pin_memory,
    )
    option_min_counts = _empty(
        (shape.row_count,),
        dtype=torch.int64,
        pin_memory=pin_memory,
    )
    option_max_counts = _empty_like(
        option_min_counts,
        pin_memory=pin_memory,
    )
    option_lengths = _empty(
        (shape.row_count,),
        dtype=torch.int32,
        pin_memory=pin_memory,
    )
    option_maximum_counts = _empty_like(
        option_lengths,
        pin_memory=pin_memory,
    )

    deck_card_ids = _empty(
        (shape.row_count, shape.deck_width),
        dtype=torch.int64,
        pin_memory=pin_memory,
    )
    deck_counts = _empty(
        (shape.row_count, shape.deck_width),
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    deck_valid = _empty(
        (shape.row_count, shape.deck_width),
        dtype=torch.bool,
        pin_memory=pin_memory,
    )
    belief_card_ids = _empty(
        (shape.belief_row_count, shape.belief_width),
        dtype=torch.int64,
        pin_memory=pin_memory,
    )
    belief_expected_counts = _empty(
        (shape.belief_row_count, shape.belief_width),
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    belief_valid = _empty(
        (shape.belief_row_count, shape.belief_width),
        dtype=torch.bool,
        pin_memory=pin_memory,
    )
    belief_scalars = _empty(
        (shape.belief_row_count, _BELIEF_SCALAR_SIZE),
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    belief_row_indices = _empty(
        (shape.row_count,),
        dtype=torch.int64,
        pin_memory=pin_memory,
    )

    native_output = _CgTrainRolloutOutput(
        struct_size=ctypes.sizeof(_CgTrainRolloutOutput),
        row_capacity=shape.row_count,
        state_token_width=shape.state_token_width,
        state_attachment_width=shape.state_attachment_width,
        option_width=shape.option_width,
        deck_width=shape.deck_width,
        belief_row_capacity=shape.belief_row_count,
        belief_width=shape.belief_width,
        state_card_ids=_tensor_pointer(state_card_ids, ctypes.c_int64),
        state_areas=_tensor_pointer(state_areas, ctypes.c_int64),
        state_owner_roles=_tensor_pointer(
            state_owner_roles,
            ctypes.c_int64,
        ),
        state_token_kinds=_tensor_pointer(
            state_token_kinds,
            ctypes.c_int64,
        ),
        state_scalars=_tensor_pointer(state_scalars, ctypes.c_float),
        state_last_attack_ids=_tensor_pointer(
            state_last_attack_ids,
            ctypes.c_int64,
        ),
        state_padding_mask=_tensor_pointer(
            state_padding_mask,
            ctypes.c_uint8,
        ),
        state_attachment_card_ids=_tensor_pointer(
            state_attachment_card_ids,
            ctypes.c_uint16,
        ),
        state_attachment_parent_indices=_tensor_pointer(
            state_attachment_parent_indices,
            ctypes.c_uint16,
        ),
        state_attachment_kinds=_tensor_pointer(
            state_attachment_kinds,
            ctypes.c_uint8,
        ),
        state_entity_slots=_tensor_pointer(
            state_entity_slots,
            ctypes.c_uint8,
        ),
        state_sequence_lengths=_tensor_pointer(
            state_sequence_lengths,
            ctypes.c_uint32,
        ),
        option_types=_tensor_pointer(option_types, ctypes.c_int64),
        option_contexts=_tensor_pointer(option_contexts, ctypes.c_int64),
        option_entity_slots=_tensor_pointer(
            option_entity_slots,
            ctypes.c_int64,
        ),
        option_entity_slot_mask=_tensor_pointer(
            option_entity_slot_mask,
            ctypes.c_uint8,
        ),
        option_attack_ids=_tensor_pointer(
            option_attack_ids,
            ctypes.c_int64,
        ),
        option_card_ids=_tensor_pointer(
            option_card_ids,
            ctypes.c_int64,
        ),
        option_scalars=_tensor_pointer(option_scalars, ctypes.c_float),
        option_dynamic_effect_features=_tensor_pointer(
            option_dynamic_effect_features,
            ctypes.c_float,
        ),
        option_dynamic_effect_masks=_tensor_pointer(
            option_dynamic_effect_masks,
            ctypes.c_uint8,
        ),
        option_valid=_tensor_pointer(option_valid, ctypes.c_uint8),
        option_min_counts=_tensor_pointer(
            option_min_counts,
            ctypes.c_int64,
        ),
        option_max_counts=_tensor_pointer(
            option_max_counts,
            ctypes.c_int64,
        ),
        option_lengths=_tensor_pointer(option_lengths, ctypes.c_uint32),
        option_maximum_counts=_tensor_pointer(
            option_maximum_counts,
            ctypes.c_uint32,
        ),
        deck_card_ids=_tensor_pointer(deck_card_ids, ctypes.c_int64),
        deck_counts=_tensor_pointer(deck_counts, ctypes.c_float),
        deck_valid=_tensor_pointer(deck_valid, ctypes.c_uint8),
        belief_card_ids=_tensor_pointer(
            belief_card_ids,
            ctypes.c_int64,
        ),
        belief_expected_counts=_tensor_pointer(
            belief_expected_counts,
            ctypes.c_float,
        ),
        belief_valid=_tensor_pointer(belief_valid, ctypes.c_uint8),
        belief_scalars=_tensor_pointer(
            belief_scalars,
            ctypes.c_float,
        ),
        belief_row_indices=_tensor_pointer(
            belief_row_indices,
            ctypes.c_int64,
        ),
    )
    row_offset = 0
    belief_offset = 0
    signatures: list[str] = []
    for source, plan in zip(sources, plans, strict=True):
        source.encoder.encode_rows(
            source.slots,
            source.perspectives,
            row_offset=row_offset,
            belief_row_offset=belief_offset,
            output=native_output,
        )
        signatures.extend(
            source.encoder.deck_signatures(
                source.slots,
                source.perspectives,
            )
        )
        row_offset += plan.row_count
        belief_offset += plan.belief_row_count

    sequence_lengths = tuple(int(value) for value in state_sequence_lengths.tolist())
    option_length_values = tuple(int(value) for value in option_lengths.tolist())
    maximum_count_values = tuple(int(value) for value in option_maximum_counts.tolist())
    minimum_count_values = tuple(int(value) for value in option_min_counts.tolist())
    belief_inverse: Tensor | None = belief_row_indices
    if shape.belief_row_count == shape.row_count and torch.equal(
        belief_row_indices,
        torch.arange(shape.row_count, dtype=torch.int64),
    ):
        belief_inverse = None
    states = StateBatch(
        card_ids=state_card_ids,
        areas=state_areas,
        owner_roles=state_owner_roles,
        token_kinds=state_token_kinds,
        scalars=state_scalars,
        last_attack_ids=state_last_attack_ids,
        padding_mask=state_padding_mask,
        attachment_card_ids=state_attachment_card_ids,
        attachment_parent_indices=state_attachment_parent_indices,
        attachment_kinds=state_attachment_kinds,
        entity_slots=state_entity_slots,
        sequence_lengths=sequence_lengths,
    )
    options = OptionBatch(
        option_types=option_types,
        contexts=option_contexts,
        entity_slots=option_entity_slots,
        entity_slot_mask=option_entity_slot_mask,
        attack_ids=option_attack_ids,
        card_ids=option_card_ids,
        scalars=option_scalars,
        dynamic_effect_features=option_dynamic_effect_features,
        dynamic_effect_masks=option_dynamic_effect_masks,
        valid_options=option_valid,
        min_counts=option_min_counts,
        max_counts=option_max_counts,
        option_lengths=option_length_values,
        maximum_counts=maximum_count_values,
    )
    belief = PublicBeliefSummaryBatch(
        card_ids=belief_card_ids,
        expected_counts=belief_expected_counts,
        valid_mask=belief_valid,
        scalars=belief_scalars,
        catalog_fingerprint=catalog_fingerprint,
        row_indices=belief_inverse,
    )
    return NativeSimpleStatelessPolicyBatch(
        states=states,
        options=options,
        unique_deck_card_ids=deck_card_ids,
        deck_counts=deck_counts,
        deck_valid_mask=deck_valid,
        deck_signatures=tuple(signatures),
        belief_summary=belief,
        min_counts=minimum_count_values,
        max_counts=maximum_count_values,
        public_deck_catalog_fingerprint=catalog_fingerprint,
        input_contract_fingerprint=input_contract_fingerprint,
    )


def _combined_shape(
    plans: tuple[NativeRolloutShape, ...],
) -> _CombinedShape:
    return _CombinedShape(
        row_count=sum(plan.row_count for plan in plans),
        state_token_width=max(plan.state_token_width for plan in plans),
        state_attachment_width=max(plan.state_attachment_width for plan in plans),
        option_width=max(plan.option_width for plan in plans),
        deck_width=max(plan.deck_width for plan in plans),
        belief_row_count=sum(plan.belief_row_count for plan in plans),
        belief_width=max(plan.belief_width for plan in plans),
    )


def _empty(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    pin_memory: bool,
) -> Tensor:
    return torch.empty(
        shape,
        dtype=dtype,
        device="cpu",
        pin_memory=pin_memory,
    )


def _empty_like(tensor: Tensor, *, pin_memory: bool) -> Tensor:
    return _empty(
        tuple(tensor.shape),
        dtype=tensor.dtype,
        pin_memory=pin_memory,
    )


def _tensor_pointer(
    tensor: Tensor,
    ctype: Any,
) -> Any:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("native rollout output tensors must be contiguous CPU")
    return ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctype))


__all__ = [
    "NativeRolloutSource",
    "encode_native_rollout_sources",
]
