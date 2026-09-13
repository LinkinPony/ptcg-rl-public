"""Small batching helpers shared by policy-iteration learner lanes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ptcg_rl.actions.encoding import EncodedOptionInput
from ptcg_rl.actions.selection import ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.decks.identity import CanonicalDeck
from ptcg_rl.model import (
    OptionBatch,
    StateBatch,
    collate_encoded_options,
    collate_state_tokens,
)
from ptcg_rl.model.state_encoder import StateTokenInput


@dataclass(frozen=True, slots=True)
class InformationSetBatch:
    """Actor-visible model inputs aligned by information set."""

    states: StateBatch
    options: OptionBatch
    decks: DeckBatch

    def __len__(self) -> int:
        return len(self.decks)


def collate_information_sets(
    *,
    states: Sequence[StateTokenInput],
    options: Sequence[EncodedOptionInput],
    min_counts: Sequence[int],
    max_counts: Sequence[int],
    decks: Sequence[CanonicalDeck],
    device: torch.device | str,
) -> InformationSetBatch:
    """Collate information-set-safe fields without protected engine material."""
    row_count = len(states)
    if not row_count or any(
        len(values) != row_count
        for values in (options, min_counts, max_counts, decks)
    ):
        raise ValueError("information-set batch fields must be non-empty and aligned")
    return InformationSetBatch(
        states=collate_state_tokens(states, device=device),
        options=collate_encoded_options(
            options,
            min_counts=min_counts,
            max_counts=max_counts,
            device=device,
        ),
        decks=DeckBatch.from_decks(decks, device=device),
    )


def ordered_rows(options: OptionBatch) -> Tensor:
    """Return action-order semantics without synchronizing each device row.

    This helper runs on full H200 learner batches.  Pulling scalar fields through
    ``Tensor.item()`` once per row serializes the CPU with the CUDA stream tens
    of thousands of times in each Retrace update, so keep the decision entirely
    tensorized on the option-batch device.
    """
    has_options = options.valid_options.any(dim=1)
    if int(options.contexts.shape[1]) == 0:
        first_context = torch.full_like(options.max_counts, -1)
    else:
        first_context = options.contexts[:, 0]
    unordered_context = torch.zeros_like(has_options)
    for context in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS:
        unordered_context |= first_context.eq(int(context))
    valid_bounds = (
        options.min_counts.ge(0)
        & options.min_counts.le(options.max_counts)
        & options.max_counts.gt(1)
    )
    unordered = has_options & unordered_context & valid_bounds
    return options.max_counts.gt(1) & ~unordered


def select_information_set_batch(
    batch: InformationSetBatch,
    indices: Tensor,
    *,
    deck_indices: Sequence[int] | None = None,
) -> InformationSetBatch:
    """Select aligned rows while preserving exact deck routing."""
    if indices.ndim != 1 or indices.dtype != torch.long:
        raise TypeError("information-set indices must be a one-dimensional long tensor")
    if deck_indices is not None and len(deck_indices) != int(indices.shape[0]):
        raise ValueError("deck indices must align with tensor indices")
    return InformationSetBatch(
        states=_select_state_batch(batch.states, indices),
        options=_select_option_batch(batch.options, indices),
        decks=batch.decks.select(indices if deck_indices is None else deck_indices),
    )


def _select_state_batch(states: StateBatch, indices: Tensor) -> StateBatch:
    def select(value: Tensor) -> Tensor:
        return value.index_select(0, indices.to(device=value.device))

    def select_optional(value: Tensor | None) -> Tensor | None:
        if value is None:
            return None
        return value.index_select(0, indices.to(device=value.device))

    return StateBatch(
        card_ids=select(states.card_ids),
        areas=select(states.areas),
        owner_roles=select(states.owner_roles),
        token_kinds=select(states.token_kinds),
        scalars=select(states.scalars),
        last_attack_ids=select(states.last_attack_ids),
        padding_mask=select(states.padding_mask),
        attachment_card_ids=select_optional(states.attachment_card_ids),
        attachment_parent_indices=select_optional(states.attachment_parent_indices),
        attachment_kinds=select_optional(states.attachment_kinds),
        entity_slots=select_optional(states.entity_slots),
    )


def _select_option_batch(options: OptionBatch, indices: Tensor) -> OptionBatch:
    def select(value: Tensor) -> Tensor:
        return value.index_select(0, indices.to(device=value.device))

    return OptionBatch(
        option_types=select(options.option_types),
        contexts=select(options.contexts),
        entity_slots=select(options.entity_slots),
        entity_slot_mask=select(options.entity_slot_mask),
        attack_ids=select(options.attack_ids),
        card_ids=select(options.card_ids),
        scalars=select(options.scalars),
        dynamic_effect_features=select(options.dynamic_effect_features),
        dynamic_effect_masks=select(options.dynamic_effect_masks),
        valid_options=select(options.valid_options),
        min_counts=select(options.min_counts),
        max_counts=select(options.max_counts),
    )


__all__ = [
    "InformationSetBatch",
    "collate_information_sets",
    "ordered_rows",
    "select_information_set_batch",
]
