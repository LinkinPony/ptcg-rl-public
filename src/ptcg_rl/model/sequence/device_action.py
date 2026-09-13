"""Device-resident accepted-action collation for temporal commits."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from ptcg_rl.actions.encoding import SCALAR_FEATURE_SIZE
from ptcg_rl.actions.selection import ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
from ptcg_rl.model.policy import CONTEXT_OOV_INDEX, MAX_ENTITY_SLOTS, OptionBatch
from ptcg_rl.model.sequence.action import AcceptedActionBatch
from ptcg_rl.model.state_encoder import StateBatch


def build_device_accepted_action_batch(
    *,
    states: StateBatch,
    options: OptionBatch,
    choice_indices: Tensor,
    lengths: Tensor,
    stop_sampled: Tensor,
    fallback: Tensor | None = None,
    unordered_rows: Sequence[int] | None = None,
) -> AcceptedActionBatch:
    """Gather sampled complete actions without crossing the device boundary.

    ``stable_identities`` is deliberately empty: SHA-256 identities remain a
    host artifact and are materialized from the detached trace after the
    provisional temporal ACTION commit. ``unordered_rows`` is an optional host
    hint that limits semantic-byte canonicalization to known unordered prompts.
    Omitting it preserves correctness by considering every row.
    """
    _validate_device_action_inputs(
        states=states,
        options=options,
        choice_indices=choice_indices,
        lengths=lengths,
        stop_sampled=stop_sampled,
        fallback=fallback,
    )
    batch_size, choice_width = choice_indices.shape
    maximum_length = max(options.maximum_counts, default=choice_width)
    if maximum_length < 0:
        raise ValueError("accepted-action maximum length must be non-negative")
    selected_width = min(choice_width, maximum_length)
    width = max(1, selected_width)
    device = choice_indices.device
    if selected_width:
        choices = choice_indices[:, :selected_width].to(dtype=torch.long)
        if selected_width < width:
            choices = torch.nn.functional.pad(
                choices,
                (0, width - selected_width),
                value=0,
            )
    else:
        choices = torch.zeros(
            (batch_size, width),
            dtype=torch.long,
            device=device,
        )
    normalized_lengths = lengths.to(dtype=torch.long)
    valid_mask = torch.arange(device=device, end=width).unsqueeze(0) < (
        normalized_lengths.unsqueeze(1)
    )

    option_count = int(options.option_types.shape[1])
    if option_count:
        safe_choices = choices.clamp(min=0, max=option_count - 1)
        option_types = _gather_option_values(options.option_types, safe_choices)
        option_contexts = _gather_option_values(options.contexts, safe_choices)
        card_ids = _gather_option_values(options.card_ids, safe_choices)
        attack_ids = _gather_option_values(options.attack_ids, safe_choices)
        option_scalars = _gather_option_values(options.scalars, safe_choices)
        entity_indices = _gather_option_values(
            options.entity_slots,
            safe_choices,
        )
        entity_present = _gather_option_values(
            options.entity_slot_mask,
            safe_choices,
        )
        prompt_contexts = torch.where(
            options.valid_options[:, 0],
            options.contexts[:, 0],
            CONTEXT_OOV_INDEX,
        )
    else:
        option_types = torch.zeros(
            (batch_size, width),
            dtype=torch.long,
            device=device,
        )
        option_contexts = torch.zeros_like(option_types)
        card_ids = torch.zeros_like(option_types)
        attack_ids = torch.zeros_like(option_types)
        option_scalars = torch.zeros(
            (batch_size, width, SCALAR_FEATURE_SIZE),
            dtype=options.scalars.dtype,
            device=device,
        )
        entity_indices = torch.zeros(
            (batch_size, width, MAX_ENTITY_SLOTS),
            dtype=torch.long,
            device=device,
        )
        entity_present = torch.zeros_like(entity_indices, dtype=torch.bool)
        prompt_contexts = torch.full(
            (batch_size,),
            CONTEXT_OOV_INDEX,
            dtype=torch.long,
            device=device,
        )

    state_width = int(states.card_ids.shape[1])
    safe_entities = entity_indices.to(dtype=torch.long).clamp(
        min=0,
        max=state_width - 1,
    )
    entity_card_ids = _gather_state_values(states.card_ids, safe_entities)
    entity_areas = _gather_state_values(states.areas, safe_entities)
    entity_owner_roles = _gather_state_values(
        states.owner_roles,
        safe_entities,
    )
    entity_token_kinds = _gather_state_values(
        states.token_kinds,
        safe_entities,
    )
    entity_scalars = _gather_state_values(states.scalars, safe_entities)
    entity_card_ids = _zero_where_invalid(entity_card_ids, entity_present)
    entity_areas = _zero_where_invalid(entity_areas, entity_present)
    entity_owner_roles = _zero_where_invalid(
        entity_owner_roles,
        entity_present,
    )
    entity_token_kinds = _zero_where_invalid(
        entity_token_kinds,
        entity_present,
    )
    entity_scalars = _zero_where_invalid(entity_scalars, entity_present)

    option_types = _zero_where_invalid(option_types, valid_mask)
    option_contexts = _zero_where_invalid(option_contexts, valid_mask)
    card_ids = _zero_where_invalid(card_ids, valid_mask)
    attack_ids = _zero_where_invalid(attack_ids, valid_mask)
    option_scalars = _zero_where_invalid(option_scalars, valid_mask)
    entity_card_ids = _zero_where_invalid(entity_card_ids, valid_mask)
    entity_areas = _zero_where_invalid(entity_areas, valid_mask)
    entity_owner_roles = _zero_where_invalid(entity_owner_roles, valid_mask)
    entity_token_kinds = _zero_where_invalid(entity_token_kinds, valid_mask)
    entity_scalars = _zero_where_invalid(entity_scalars, valid_mask)

    ordered = torch.ones_like(prompt_contexts, dtype=torch.bool)
    for context in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS:
        ordered = ordered & prompt_contexts.ne(int(context))
    needs_canonical_order = width > 1 and (
        unordered_rows is None or len(unordered_rows) > 0
    )
    if needs_canonical_order:
        permutation = _device_action_permutation(
            option_types=option_types,
            option_contexts=option_contexts,
            card_ids=card_ids,
            attack_ids=attack_ids,
            option_scalars=option_scalars,
            entity_card_ids=entity_card_ids,
            entity_areas=entity_areas,
            entity_owner_roles=entity_owner_roles,
            entity_token_kinds=entity_token_kinds,
            entity_scalars=entity_scalars,
            valid_mask=valid_mask,
            ordered=ordered,
            unordered_rows=unordered_rows,
        )

        def canonical(value: Tensor) -> Tensor:
            return _gather_option_values(value, permutation)

    else:

        def canonical(value: Tensor) -> Tensor:
            return value

    fallback_rows = (
        torch.zeros(batch_size, dtype=torch.bool, device=device)
        if fallback is None
        else fallback.to(dtype=torch.bool)
    )
    return AcceptedActionBatch(
        option_types=canonical(option_types),
        option_contexts=canonical(option_contexts),
        card_ids=canonical(card_ids),
        attack_ids=canonical(attack_ids),
        option_scalars=canonical(option_scalars),
        entity_card_ids=canonical(entity_card_ids),
        entity_areas=canonical(entity_areas),
        entity_owner_roles=canonical(entity_owner_roles),
        entity_token_kinds=canonical(entity_token_kinds),
        entity_scalars=canonical(entity_scalars),
        valid_mask=canonical(valid_mask),
        prompt_contexts=prompt_contexts,
        lengths=normalized_lengths,
        ordered=ordered,
        stop_sampled=stop_sampled.to(dtype=torch.bool),
        fallback=fallback_rows,
        stable_identities=(),
    )


def _validate_device_action_inputs(
    *,
    states: StateBatch,
    options: OptionBatch,
    choice_indices: Tensor,
    lengths: Tensor,
    stop_sampled: Tensor,
    fallback: Tensor | None,
) -> None:
    """Reject structurally unsafe tensor-native action inputs."""
    if choice_indices.ndim != 2:
        raise ValueError("sampled choices must have shape [batch, choices]")
    batch_size = int(choice_indices.shape[0])
    if lengths.shape != (batch_size,) or stop_sampled.shape != (batch_size,):
        raise ValueError("sampled action row tensors are misaligned")
    if fallback is not None and fallback.shape != (batch_size,):
        raise ValueError("sampled fallback rows are misaligned")
    if options.option_types.ndim != 2 or int(options.option_types.shape[0]) != (
        batch_size
    ):
        raise ValueError("sampled choices and options are misaligned")
    if states.card_ids.ndim != 2 or int(states.card_ids.shape[0]) != batch_size:
        raise ValueError("sampled choices and states are misaligned")
    if int(states.card_ids.shape[1]) <= 0:
        raise ValueError("accepted actions require non-empty state rows")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if (
        choice_indices.dtype not in integer_dtypes
        or lengths.dtype not in integer_dtypes
    ):
        raise TypeError("sampled choices and lengths must be integer tensors")
    device = choice_indices.device
    required = [
        lengths,
        stop_sampled,
        options.option_types,
        options.contexts,
        options.entity_slots,
        options.entity_slot_mask,
        options.card_ids,
        options.attack_ids,
        options.scalars,
        options.valid_options,
        states.card_ids,
        states.areas,
        states.owner_roles,
        states.token_kinds,
        states.scalars,
    ]
    if fallback is not None:
        required.append(fallback)
    if any(value.device != device for value in required):
        raise ValueError("accepted-action inputs must share one device")
    if options.maximum_counts and len(options.maximum_counts) != batch_size:
        raise ValueError("accepted-action maximum counts are misaligned")


def _gather_option_values(values: Tensor, indices: Tensor) -> Tensor:
    """Gather the option-width dimension while preserving trailing fields."""
    gather_indices = indices
    for _unused in range(values.ndim - 2):
        gather_indices = gather_indices.unsqueeze(-1)
    gather_indices = gather_indices.expand(
        *indices.shape,
        *values.shape[2:],
    )
    return torch.gather(values, dim=1, index=gather_indices)


def _gather_state_values(values: Tensor, indices: Tensor) -> Tensor:
    """Gather per-option entity references from padded state rows."""
    batch_size, width, entity_slots = indices.shape
    flat_indices = indices.flatten(start_dim=1)
    gather_indices = flat_indices
    for _unused in range(values.ndim - 2):
        gather_indices = gather_indices.unsqueeze(-1)
    gather_indices = gather_indices.expand(
        batch_size,
        width * entity_slots,
        *values.shape[2:],
    )
    gathered = torch.gather(values, dim=1, index=gather_indices)
    expected = (batch_size, width, entity_slots, *values.shape[2:])
    return gathered.reshape(expected)


def _zero_where_invalid(values: Tensor, valid: Tensor) -> Tensor:
    """Replace invalid padded payloads with exact zeros."""
    expanded = valid
    for _unused in range(values.ndim - valid.ndim):
        expanded = expanded.unsqueeze(-1)
    return torch.where(expanded, values, 0)


def _action_semantic_bytes(
    *,
    option_types: Tensor,
    option_contexts: Tensor,
    card_ids: Tensor,
    attack_ids: Tensor,
    option_scalars: Tensor,
    entity_card_ids: Tensor,
    entity_areas: Tensor,
    entity_owner_roles: Tensor,
    entity_token_kinds: Tensor,
    entity_scalars: Tensor,
) -> Tensor:
    """Pack the exact little-endian payload used by host identity sorting."""
    integer_fields = torch.cat(
        (
            option_types.unsqueeze(-1),
            option_contexts.unsqueeze(-1),
            card_ids.unsqueeze(-1),
            attack_ids.unsqueeze(-1),
            entity_card_ids,
            entity_areas,
            entity_owner_roles,
            entity_token_kinds,
        ),
        dim=-1,
    ).to(dtype=torch.int64)
    numeric_fields = torch.cat(
        (
            option_scalars,
            entity_scalars.flatten(start_dim=2),
        ),
        dim=-1,
    ).to(dtype=torch.float32)
    integer_bytes = integer_fields.contiguous().view(torch.uint8)
    numeric_bytes = numeric_fields.contiguous().view(torch.uint8)
    return torch.cat((integer_bytes, numeric_bytes), dim=-1)


def _device_action_permutation(
    *,
    option_types: Tensor,
    option_contexts: Tensor,
    card_ids: Tensor,
    attack_ids: Tensor,
    option_scalars: Tensor,
    entity_card_ids: Tensor,
    entity_areas: Tensor,
    entity_owner_roles: Tensor,
    entity_token_kinds: Tensor,
    entity_scalars: Tensor,
    valid_mask: Tensor,
    ordered: Tensor,
    unordered_rows: Sequence[int] | None,
) -> Tensor:
    """Canonicalize only the host-known unordered subset on the device."""
    batch_size, width = valid_mask.shape
    identity = torch.arange(width, device=valid_mask.device).expand(
        batch_size,
        -1,
    )
    if width <= 1:
        return identity
    if unordered_rows is None:
        row_indices = torch.arange(batch_size, device=valid_mask.device)
    else:
        normalized = tuple(int(row) for row in unordered_rows)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("unordered action rows must be sorted and unique")
        if any(row < 0 or row >= batch_size for row in normalized):
            raise ValueError("unordered action row is outside the batch")
        if not normalized:
            return identity
        row_indices = torch.tensor(
            normalized,
            dtype=torch.long,
            device=valid_mask.device,
        )

    def selected(value: Tensor) -> Tensor:
        return value.index_select(0, row_indices)

    semantic_bytes = _action_semantic_bytes(
        option_types=selected(option_types),
        option_contexts=selected(option_contexts),
        card_ids=selected(card_ids),
        attack_ids=selected(attack_ids),
        option_scalars=selected(option_scalars),
        entity_card_ids=selected(entity_card_ids),
        entity_areas=selected(entity_areas),
        entity_owner_roles=selected(entity_owner_roles),
        entity_token_kinds=selected(entity_token_kinds),
        entity_scalars=selected(entity_scalars),
    )
    selected_permutation = _action_canonical_permutation(
        semantic_bytes,
        valid_mask=selected(valid_mask),
    )
    canonical = identity.index_copy(0, row_indices, selected_permutation)
    return torch.where(ordered.unsqueeze(1), identity, canonical)


def _action_canonical_permutation(
    semantic_bytes: Tensor,
    *,
    valid_mask: Tensor,
) -> Tensor:
    """Return stable semantic-byte order for unordered selected prefixes."""
    batch_size, width, _payload_width = semantic_bytes.shape
    identity = torch.arange(width, device=semantic_bytes.device).expand(
        batch_size,
        -1,
    )
    if width <= 1:
        return identity
    different = semantic_bytes.unsqueeze(2).ne(semantic_bytes.unsqueeze(1))
    has_difference, first_difference = different.max(dim=-1)
    pair_width = int(semantic_bytes.shape[1])
    left = semantic_bytes.unsqueeze(2).expand(-1, -1, pair_width, -1)
    right = semantic_bytes.unsqueeze(1).expand(-1, pair_width, -1, -1)
    first = first_difference.unsqueeze(-1)
    left_first = torch.gather(left, dim=-1, index=first).squeeze(-1)
    right_first = torch.gather(right, dim=-1, index=first).squeeze(-1)
    left_less = has_difference & left_first.lt(right_first)
    equal = ~has_difference
    positions = torch.arange(pair_width, device=semantic_bytes.device)
    candidate_valid = valid_mask.unsqueeze(1)
    smaller_count = (left_less.transpose(1, 2) & candidate_valid).sum(dim=-1)
    earlier = positions.view(1, 1, pair_width) < positions.view(
        1,
        pair_width,
        1,
    )
    equal_earlier_count = (equal & earlier & candidate_valid).sum(dim=-1)
    ranks = smaller_count + equal_earlier_count
    ranks = torch.where(
        valid_mask,
        ranks,
        pair_width + positions.unsqueeze(0),
    )
    return torch.argsort(ranks, dim=1)

__all__ = ["build_device_accepted_action_batch"]
