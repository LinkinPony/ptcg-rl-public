"""Aligned row selection for native policy tensors and behavior traces."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.simple_stateless.belief import PublicBeliefSummaryBatch
from ptcg_rl.model.state_encoder import StateBatch
from ptcg_rl.rl.native_policy_batch import NativeSimpleStatelessPolicyBatch
from ptcg_rl.rl.native_policy_trace import (
    NativePolicyNumpyActionBatch,
    NativePolicyNumpyTrace,
)


def select_native_policy_batch_rows(
    batch: NativeSimpleStatelessPolicyBatch,
    rows: npt.ArrayLike,
) -> NativeSimpleStatelessPolicyBatch:
    """Select unique policy rows without reconstructing feature objects."""
    selected = _validated_rows(rows, size=batch.batch_size)
    index = torch.as_tensor(
        selected,
        dtype=torch.long,
        device=batch.states.card_ids.device,
    )

    def take(value: Tensor | None) -> Tensor | None:
        if value is None:
            return None
        if value.device != index.device:
            raise ValueError("native policy batch tensors do not share one device")
        return value.index_select(0, index)

    fingerprints = batch.states.root_input_fingerprints
    if fingerprints and len(fingerprints) != batch.batch_size:
        raise ValueError("native root-input fingerprints do not align")
    states = StateBatch(
        card_ids=batch.states.card_ids.index_select(0, index),
        areas=batch.states.areas.index_select(0, index),
        owner_roles=batch.states.owner_roles.index_select(0, index),
        token_kinds=batch.states.token_kinds.index_select(0, index),
        scalars=batch.states.scalars.index_select(0, index),
        last_attack_ids=batch.states.last_attack_ids.index_select(0, index),
        padding_mask=batch.states.padding_mask.index_select(0, index),
        attachment_card_ids=take(batch.states.attachment_card_ids),
        attachment_parent_indices=take(
            batch.states.attachment_parent_indices
        ),
        attachment_kinds=take(batch.states.attachment_kinds),
        entity_slots=take(batch.states.entity_slots),
        root_input_fingerprints=(
            tuple(fingerprints[int(row)] for row in selected)
            if fingerprints
            else ()
        ),
        sequence_lengths=(
            tuple(batch.states.sequence_lengths[int(row)] for row in selected)
            if batch.states.sequence_lengths
            else ()
        ),
    )
    options = OptionBatch(
        option_types=batch.options.option_types.index_select(0, index),
        contexts=batch.options.contexts.index_select(0, index),
        entity_slots=batch.options.entity_slots.index_select(0, index),
        entity_slot_mask=batch.options.entity_slot_mask.index_select(0, index),
        attack_ids=batch.options.attack_ids.index_select(0, index),
        card_ids=batch.options.card_ids.index_select(0, index),
        scalars=batch.options.scalars.index_select(0, index),
        dynamic_effect_features=(
            batch.options.dynamic_effect_features.index_select(0, index)
        ),
        dynamic_effect_masks=(
            batch.options.dynamic_effect_masks.index_select(0, index)
        ),
        valid_options=batch.options.valid_options.index_select(0, index),
        min_counts=batch.options.min_counts.index_select(0, index),
        max_counts=batch.options.max_counts.index_select(0, index),
        option_lengths=(
            tuple(batch.options.option_lengths[int(row)] for row in selected)
            if batch.options.option_lengths
            else ()
        ),
        maximum_counts=(
            tuple(batch.options.maximum_counts[int(row)] for row in selected)
            if batch.options.maximum_counts
            else ()
        ),
    )
    source_belief_rows = index
    if batch.belief_summary.row_indices is not None:
        source_belief_rows = batch.belief_summary.row_indices.index_select(
            0,
            index,
        )
    belief = PublicBeliefSummaryBatch(
        card_ids=batch.belief_summary.card_ids.index_select(
            0,
            source_belief_rows,
        ),
        expected_counts=batch.belief_summary.expected_counts.index_select(
            0,
            source_belief_rows,
        ),
        valid_mask=batch.belief_summary.valid_mask.index_select(
            0,
            source_belief_rows,
        ),
        scalars=batch.belief_summary.scalars.index_select(
            0,
            source_belief_rows,
        ),
        catalog_fingerprint=batch.belief_summary.catalog_fingerprint,
    )
    return NativeSimpleStatelessPolicyBatch(
        states=states,
        options=options,
        unique_deck_card_ids=batch.unique_deck_card_ids.index_select(0, index),
        deck_counts=batch.deck_counts.index_select(0, index),
        deck_valid_mask=batch.deck_valid_mask.index_select(0, index),
        deck_signatures=tuple(
            batch.deck_signatures[int(row)] for row in selected
        ),
        belief_summary=belief,
        min_counts=tuple(batch.min_counts[int(row)] for row in selected),
        max_counts=tuple(batch.max_counts[int(row)] for row in selected),
        public_deck_catalog_fingerprint=(
            batch.public_deck_catalog_fingerprint
        ),
        input_contract_fingerprint=batch.input_contract_fingerprint,
    )


def select_native_policy_trace_rows(
    trace: NativePolicyNumpyTrace,
    rows: npt.ArrayLike,
) -> NativePolicyNumpyTrace:
    """Select host trace rows and rebase both action and token CSR arrays."""
    selected = _validated_rows(rows, size=trace.batch_size)
    action_offsets, action_choices = _select_csr(
        trace.action_offsets,
        trace.action_choices,
        selected,
    )
    token_offsets, token_logprobs = _select_csr(
        trace.token_offsets,
        trace.token_logprobs,
        selected,
    )
    _prefix_offsets, prefix_values = _select_csr(
        trace.token_offsets,
        trace.prefix_values,
        selected,
    )
    return NativePolicyNumpyTrace(
        identity=trace.identity,
        action_offsets=action_offsets,
        action_choices=cast(npt.NDArray[np.int32], action_choices),
        action_logprobs=trace.action_logprobs[selected].copy(),
        token_offsets=token_offsets,
        token_logprobs=cast(npt.NDArray[np.float32], token_logprobs),
        prefix_values=cast(npt.NDArray[np.float32], prefix_values),
        root_values=trace.root_values[selected].copy(),
        stop_sampled=trace.stop_sampled[selected].copy(),
    )


def select_native_policy_action_rows(
    actions: NativePolicyNumpyActionBatch,
    rows: npt.ArrayLike,
) -> NativePolicyNumpyActionBatch:
    """Select host action rows and rebase their compact CSR offsets."""
    selected = _validated_rows(rows, size=actions.batch_size)
    offsets, choices = _select_csr(
        actions.action_offsets,
        actions.action_choices,
        selected,
    )
    return NativePolicyNumpyActionBatch(
        identity=actions.identity,
        action_offsets=offsets,
        action_choices=cast(npt.NDArray[np.int32], choices),
    )


def _select_csr(
    offsets: npt.NDArray[np.int64],
    values: npt.NDArray[np.generic],
    rows: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.generic]]:
    starts = offsets[rows]
    stops = offsets[rows + 1]
    lengths = stops - starts
    destination_offsets = np.zeros(rows.size + 1, dtype=np.int64)
    destination_offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    destination = np.empty(int(destination_offsets[-1]), dtype=values.dtype)
    for row, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        destination[
            int(destination_offsets[row]) : int(destination_offsets[row + 1])
        ] = values[int(start) : int(stop)]
    return destination_offsets, destination


def _validated_rows(
    values: npt.ArrayLike,
    *,
    size: int,
) -> npt.NDArray[np.int64]:
    rows = np.asarray(values)
    if rows.ndim != 1 or rows.size <= 0:
        raise ValueError("native policy row selection must be a non-empty vector")
    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError("native policy row selection must use an integer dtype")
    normalized = rows.astype(np.int64, copy=False)
    if (
        np.any(normalized < 0)
        or np.any(normalized >= size)
        or np.unique(normalized).size != normalized.size
    ):
        raise ValueError("native policy row selection must be unique and in range")
    return normalized


__all__ = [
    "select_native_policy_action_rows",
    "select_native_policy_batch_rows",
    "select_native_policy_trace_rows",
]
