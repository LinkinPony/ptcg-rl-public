"""Direct native legal-option to simple-stateless ``OptionBatch`` encoding."""

from __future__ import annotations

import numpy as np
import torch

from ptcg_rl.actions.encoding import SCALAR_FEATURE_SIZE
from ptcg_rl.engine.constants import AreaType, OptionType
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.model.policy import MAX_ENTITY_SLOTS, OptionBatch
from ptcg_rl.rl.native_policy_keys import (
    area_pointer_keys,
    attachment_pointer_keys,
    exact_lookup,
    serial_pointer_keys,
)
from ptcg_rl.rl.native_policy_state import NativeTokenLookup

_ATTACHMENT_ENERGY_KIND = 1
_ATTACHMENT_TOOL_KIND = 2


def encode_native_option_batch(
    view: NativeTrainingBatchView,
    lookup: NativeTokenLookup,
    *,
    device: torch.device | str | None = None,
) -> OptionBatch:
    """Encode all engine-legal options with array pointer joins."""
    if lookup.batch_size != view.batch_size:
        raise ValueError("native token lookup does not align with option rows")
    _validate_option_columns(view)
    offsets = view.option_offsets.astype(np.int64, copy=False)
    lengths = np.diff(offsets)
    rows = np.repeat(np.arange(view.batch_size, dtype=np.int64), lengths)
    starts = np.repeat(offsets[:-1], lengths)
    local = np.arange(view.option_count, dtype=np.int64) - starts
    maximum = int(lengths.max())
    shape = (view.batch_size, maximum)

    option_types = np.zeros(shape, dtype=np.int64)
    contexts = np.zeros(shape, dtype=np.int64)
    entity_slots = np.zeros((*shape, MAX_ENTITY_SLOTS), dtype=np.int64)
    entity_masks = np.zeros((*shape, MAX_ENTITY_SLOTS), dtype=np.bool_)
    attack_ids = np.zeros(shape, dtype=np.int64)
    card_ids = np.zeros(shape, dtype=np.int64)
    scalars = np.zeros((*shape, SCALAR_FEATURE_SIZE), dtype=np.float32)
    dynamic_features = np.zeros(
        (*shape, DYNAMIC_EFFECT_FEATURE_SIZE),
        dtype=np.float32,
    )
    dynamic_masks = np.zeros(shape, dtype=np.bool_)
    valid = np.zeros(shape, dtype=np.bool_)

    types = view.option_type.astype(np.int64, copy=False)
    params = tuple(column.astype(np.int64, copy=False) for column in view.option_params)
    option_types[rows, local] = types
    contexts[rows, local] = np.maximum(
        0,
        view.select_context[rows].astype(np.int64) - 1,
    )
    valid[rows, local] = True

    flat_attack_ids = np.zeros(view.option_count, dtype=np.int64)
    flat_card_ids = np.zeros(view.option_count, dtype=np.int64)
    flat_scalars = np.zeros(
        (view.option_count, SCALAR_FEATURE_SIZE),
        dtype=np.float32,
    )
    number = types == int(OptionType.NUMBER)
    energy = types == int(OptionType.ENERGY)
    energy_card = types == int(OptionType.ENERGY_CARD)
    tool_card = types == int(OptionType.TOOL_CARD)
    special = types == int(OptionType.SPECIAL_CONDITION)
    attack = types == int(OptionType.ATTACK)
    skill = types == int(OptionType.SKILL)
    flat_scalars[number, 0] = params[0][number] / 10.0
    flat_scalars[energy, 1] = params[4][energy] / 10.0
    energy_pointer = energy | energy_card
    flat_scalars[energy_pointer, 2] = params[3][energy_pointer] / 16.0
    flat_scalars[energy_pointer, 5] = 1.0
    flat_scalars[tool_card, 3] = params[3][tool_card] / 8.0
    flat_scalars[tool_card, 6] = 1.0
    flat_scalars[special, 4] = params[0][special] / 4.0
    flat_attack_ids[attack] = np.maximum(0, params[0][attack])
    flat_card_ids[skill] = np.maximum(0, params[0][skill])

    primary = np.full(view.option_count, -1, dtype=np.int64)
    secondary = np.full(view.option_count, -1, dtype=np.int64)
    card_family = (types == int(OptionType.CARD)) | tool_card | energy_card | energy
    _resolve_into(
        primary,
        card_family,
        rows,
        params[0],
        params[2],
        params[1],
        lookup,
    )
    _resolve_attachment_identities(
        flat_card_ids,
        flat_scalars,
        types=types,
        rows=rows,
        parent_tokens=primary,
        attachment_indices=params[3],
        lookup=lookup,
    )

    play = types == int(OptionType.PLAY)
    _resolve_into(
        primary,
        play,
        rows,
        np.full(view.option_count, int(AreaType.HAND), dtype=np.int64),
        lookup.perspectives[rows],
        params[0],
        lookup,
    )
    attach_or_evolve = (types == int(OptionType.ATTACH)) | (
        types == int(OptionType.EVOLVE)
    )
    _resolve_into(
        primary,
        attach_or_evolve,
        rows,
        params[0],
        lookup.perspectives[rows],
        params[1],
        lookup,
    )
    _resolve_into(
        secondary,
        attach_or_evolve,
        rows,
        params[2],
        lookup.perspectives[rows],
        params[3],
        lookup,
    )
    ability_or_discard = (types == int(OptionType.ABILITY)) | (
        types == int(OptionType.DISCARD)
    )
    _resolve_into(
        primary,
        ability_or_discard,
        rows,
        params[0],
        lookup.perspectives[rows],
        params[1],
        lookup,
    )
    _resolve_into(
        primary,
        attack,
        rows,
        np.full(view.option_count, int(AreaType.ACTIVE), dtype=np.int64),
        lookup.perspectives[rows],
        np.zeros(view.option_count, dtype=np.int64),
        lookup,
    )
    _resolve_skill_tokens(
        primary,
        skill,
        rows=rows,
        card_ids=params[0],
        serials=params[1],
        lookup=lookup,
    )

    flat_entities = np.zeros(
        (view.option_count, MAX_ENTITY_SLOTS),
        dtype=np.int64,
    )
    flat_entity_masks = np.zeros(
        (view.option_count, MAX_ENTITY_SLOTS),
        dtype=np.bool_,
    )
    has_primary = primary >= 0
    has_secondary = secondary >= 0
    flat_entities[has_primary, 0] = primary[has_primary]
    flat_entity_masks[has_primary, 0] = True
    secondary_slot = has_primary & has_secondary
    flat_entities[secondary_slot, 1] = secondary[secondary_slot]
    flat_entity_masks[secondary_slot, 1] = True
    secondary_only = ~has_primary & has_secondary
    flat_entities[secondary_only, 0] = secondary[secondary_only]
    flat_entity_masks[secondary_only, 0] = True

    entity_slots[rows, local] = flat_entities
    entity_masks[rows, local] = flat_entity_masks
    attack_ids[rows, local] = flat_attack_ids
    card_ids[rows, local] = flat_card_ids
    scalars[rows, local] = flat_scalars
    minimums = np.minimum(lengths, np.maximum(0, view.select_min))
    maximums = np.minimum(
        lengths,
        np.maximum(minimums, view.select_max),
    )
    return OptionBatch(
        option_types=torch.as_tensor(option_types, device=device),
        contexts=torch.as_tensor(contexts, device=device),
        entity_slots=torch.as_tensor(entity_slots, device=device),
        entity_slot_mask=torch.as_tensor(entity_masks, device=device),
        attack_ids=torch.as_tensor(attack_ids, device=device),
        card_ids=torch.as_tensor(card_ids, device=device),
        scalars=torch.as_tensor(scalars, device=device),
        dynamic_effect_features=torch.as_tensor(
            dynamic_features,
            device=device,
        ),
        dynamic_effect_masks=torch.as_tensor(
            dynamic_masks,
            device=device,
        ),
        valid_options=torch.as_tensor(valid, device=device),
        min_counts=torch.as_tensor(minimums, device=device),
        max_counts=torch.as_tensor(maximums, device=device),
        option_lengths=tuple(int(length) for length in lengths),
        maximum_counts=tuple(int(value) for value in maximums),
    )


def _validate_option_columns(view: NativeTrainingBatchView) -> None:
    if view.option_count <= 0:
        raise ValueError("native policy batch has no legal options")
    columns = (view.option_type, *view.option_params)
    if any(column.shape != (view.option_count,) for column in columns):
        raise ValueError("native option columns do not align")


def _resolve_into(
    destination: np.ndarray,
    mask: np.ndarray,
    rows: np.ndarray,
    areas: np.ndarray,
    owners: np.ndarray,
    indices: np.ndarray,
    lookup: NativeTokenLookup,
) -> None:
    selected = np.flatnonzero(mask)
    if not selected.size:
        return
    selected_rows = rows[selected]
    selected_areas = areas[selected].astype(np.int64, copy=False)
    selected_owners = owners[selected].astype(np.int64, copy=False)
    selected_indices = indices[selected].astype(np.int64, copy=False)
    selected_owners = np.where(
        selected_owners < 0,
        lookup.perspectives[selected_rows],
        selected_owners,
    )
    valid = (
        (selected_areas >= 0)
        & (selected_areas < 16)
        & (selected_indices >= 0)
        & (selected_indices < 256)
        & (
            ((selected_owners >= 0) & (selected_owners <= 1))
            | (selected_areas == int(AreaType.STADIUM))
        )
    )
    if not np.any(valid):
        return
    query = area_pointer_keys(
        selected_rows[valid],
        selected_areas[valid],
        selected_owners[valid],
        selected_indices[valid],
    )
    destination[selected[valid]] = exact_lookup(
        lookup.area_keys,
        lookup.area_tokens,
        query,
    )


def _resolve_attachment_identities(
    card_ids: np.ndarray,
    scalars: np.ndarray,
    *,
    types: np.ndarray,
    rows: np.ndarray,
    parent_tokens: np.ndarray,
    attachment_indices: np.ndarray,
    lookup: NativeTokenLookup,
) -> None:
    for option_type, native_kind in (
        (int(OptionType.TOOL_CARD), _ATTACHMENT_TOOL_KIND),
        (int(OptionType.ENERGY_CARD), _ATTACHMENT_ENERGY_KIND),
        (int(OptionType.ENERGY), _ATTACHMENT_ENERGY_KIND),
    ):
        selected = np.flatnonzero(types == option_type)
        if not selected.size:
            continue
        indices = attachment_indices[selected]
        valid = (parent_tokens[selected] >= 0) & (indices >= 0) & (indices < 256)
        if not np.any(valid):
            continue
        targets = selected[valid]
        keys = attachment_pointer_keys(
            rows[targets],
            parent_tokens[targets],
            np.full(targets.shape, native_kind, dtype=np.int64),
            attachment_indices[targets],
        )
        identities = exact_lookup(
            lookup.attachment_keys,
            lookup.attachment_card_ids,
            keys,
        )
        serials = exact_lookup(
            lookup.attachment_keys,
            lookup.attachment_serials,
            keys,
        )
        present = identities >= 0
        if np.any(present):
            resolved = targets[present]
            card_ids[resolved] = identities[present]
            scalars[resolved, 7] = serials[present] / 128.0
            scalars[resolved, 8] = 1.0


def _resolve_skill_tokens(
    destination: np.ndarray,
    mask: np.ndarray,
    *,
    rows: np.ndarray,
    card_ids: np.ndarray,
    serials: np.ndarray,
    lookup: NativeTokenLookup,
) -> None:
    selected = np.flatnonzero(mask)
    if not selected.size:
        return
    special = selected[card_ids[selected] == 0]
    destination[special] = 1
    card_selected = selected[(card_ids[selected] > 0) & (serials[selected] > 0)]
    if not card_selected.size:
        return
    keys = serial_pointer_keys(
        rows[card_selected],
        serials[card_selected],
    )
    own_tokens = exact_lookup(
        lookup.serial_own_keys,
        lookup.serial_own_tokens,
        keys,
    )
    own_cards = exact_lookup(
        lookup.serial_own_keys,
        lookup.serial_own_card_ids,
        keys,
    )
    own_present = own_tokens >= 0
    own_valid = own_present & (
        (own_cards == 0) | (own_cards == card_ids[card_selected])
    )
    if np.any(own_valid):
        destination[card_selected[own_valid]] = own_tokens[own_valid]

    fallback = ~own_present
    if not np.any(fallback):
        return
    any_tokens = exact_lookup(
        lookup.serial_any_keys,
        lookup.serial_any_tokens,
        keys[fallback],
    )
    any_cards = exact_lookup(
        lookup.serial_any_keys,
        lookup.serial_any_card_ids,
        keys[fallback],
    )
    any_valid = (any_tokens >= 0) & (
        (any_cards == 0) | (any_cards == card_ids[card_selected[fallback]])
    )
    if np.any(any_valid):
        fallback_rows = card_selected[fallback]
        destination[fallback_rows[any_valid]] = any_tokens[any_valid]


__all__ = ["encode_native_option_batch"]
