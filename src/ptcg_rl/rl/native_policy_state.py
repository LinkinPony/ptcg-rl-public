"""Direct native public-state to simple-stateless ``StateBatch`` encoding."""

from __future__ import annotations

import numpy as np
import torch

from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.model.state_encoder import (
    OWNER_OPPONENT,
    OWNER_SELF,
    OWNER_SHARED,
    OWNER_UNKNOWN,
    TOKEN_KIND_OOV_INDEX,
    TOKEN_KIND_TO_INDEX,
    TOKEN_SCALAR_SIZE,
    StateBatch,
)
from ptcg_rl.rl.native_policy_context import NativePublicContextBatch
from ptcg_rl.rl.native_policy_state_features import (
    fill_context_tokens,
    fill_global_scalars,
    fill_visible_scalars,
    visible_last_attacks,
)
from ptcg_rl.rl.native_policy_state_layout import (
    NativeTokenLookup,
    attachment_columns,
    csr_positions,
    owner_roles,
    safe_areas,
    token_lookup,
    validate_native_view,
    visible_token_kinds,
)


def encode_native_state_batch(
    view: NativeTrainingBatchView,
    context: NativePublicContextBatch,
    *,
    device: torch.device | str | None = None,
) -> tuple[StateBatch, NativeTokenLookup]:
    """Encode a native arena view without observations or token-layout objects."""
    batch_size = view.batch_size
    context.validate(batch_size=batch_size)
    validate_native_view(view)

    visible_offsets = view.visible_card_offsets.astype(np.int64, copy=False)
    visible_lengths = np.diff(visible_offsets)
    own_lengths = np.diff(context.own_unseen_offsets.astype(np.int64, copy=False))
    revealed_lengths = np.diff(
        context.opponent_revealed_offsets.astype(np.int64, copy=False)
    )
    token_lengths = 2 + visible_lengths + own_lengths + revealed_lengths
    maximum_tokens = int(token_lengths.max())
    if maximum_tokens > np.iinfo(np.uint16).max:
        raise ValueError("native state token count exceeds uint16 pointer capacity")

    shape = (batch_size, maximum_tokens)
    card_ids = np.zeros(shape, dtype=np.int64)
    areas = np.zeros(shape, dtype=np.int64)
    relative_owners = np.full(shape, OWNER_UNKNOWN, dtype=np.int64)
    token_kinds = np.full(shape, TOKEN_KIND_OOV_INDEX, dtype=np.int64)
    scalars = np.zeros((*shape, TOKEN_SCALAR_SIZE), dtype=np.float32)
    last_attack_ids = np.zeros(shape, dtype=np.int64)
    entity_slots = np.zeros(shape, dtype=np.uint8)
    padding_mask = np.ones(shape, dtype=np.bool_)

    row_axis = np.arange(batch_size, dtype=np.int64)
    padding_mask[row_axis, 0] = False
    padding_mask[row_axis, 1] = False
    relative_owners[:, :2] = OWNER_SHARED
    token_kinds[:, 0] = TOKEN_KIND_TO_INDEX["global"]
    token_kinds[:, 1] = TOKEN_KIND_TO_INDEX["special_condition"]
    fill_global_scalars(scalars[:, 0, :], view, context)

    visible_rows, _visible_local, visible_tokens = csr_positions(visible_offsets)
    visible_absolute = np.arange(view.visible_card_count, dtype=np.int64)
    if visible_absolute.size:
        padding_mask[visible_rows, visible_tokens] = False
        card_ids[visible_rows, visible_tokens] = np.maximum(
            0,
            view.visible_card_id.astype(np.int64, copy=False),
        )
        areas[visible_rows, visible_tokens] = safe_areas(view.visible_card_area)
        relative_owners[visible_rows, visible_tokens] = owner_roles(
            view.visible_card_owner,
            view.select_player[visible_rows],
        )
        token_kinds[visible_rows, visible_tokens] = visible_token_kinds(
            view,
            visible_rows=visible_rows,
            visible_absolute=visible_absolute,
        )
        fill_visible_scalars(
            scalars,
            entity_slots,
            view,
            visible_rows=visible_rows,
            visible_tokens=visible_tokens,
            visible_offsets=visible_offsets,
        )
        last_attack_ids[visible_rows, visible_tokens] = visible_last_attacks(
            view,
            context,
            visible_rows=visible_rows,
        )

    own_rows, own_local, _unused = csr_positions(
        context.own_unseen_offsets.astype(np.int64, copy=False)
    )
    own_tokens = (
        2 + visible_lengths[own_rows] + own_local if own_rows.size else own_local
    )
    fill_context_tokens(
        card_ids,
        areas,
        relative_owners,
        token_kinds,
        scalars,
        padding_mask,
        rows=own_rows,
        tokens=own_tokens,
        source_card_ids=context.own_unseen_card_ids,
        source_counts=context.own_unseen_counts,
        owner_role=OWNER_SELF,
        owner_scalar=1.0,
        token_kind=TOKEN_KIND_TO_INDEX["own_unseen"],
    )

    revealed_rows, revealed_local, _unused = csr_positions(
        context.opponent_revealed_offsets.astype(np.int64, copy=False)
    )
    revealed_tokens = (
        2 + visible_lengths[revealed_rows] + own_lengths[revealed_rows] + revealed_local
        if revealed_rows.size
        else revealed_local
    )
    fill_context_tokens(
        card_ids,
        areas,
        relative_owners,
        token_kinds,
        scalars,
        padding_mask,
        rows=revealed_rows,
        tokens=revealed_tokens,
        source_card_ids=context.opponent_revealed_card_ids,
        source_counts=context.opponent_revealed_counts,
        owner_role=OWNER_OPPONENT,
        owner_scalar=-1.0,
        token_kind=TOKEN_KIND_TO_INDEX["opponent_revealed"],
    )

    (
        attachment_cards,
        attachment_parents,
        attachment_kinds,
        attachment_keys,
        attachment_identity_ids,
        attachment_identity_serials,
    ) = attachment_columns(view, visible_offsets=visible_offsets)
    lookup = token_lookup(
        view,
        visible_rows=visible_rows,
        visible_tokens=visible_tokens,
        attachment_keys=attachment_keys,
        attachment_card_ids=attachment_identity_ids,
        attachment_serials=attachment_identity_serials,
    )
    return (
        StateBatch(
            card_ids=torch.as_tensor(card_ids, device=device),
            areas=torch.as_tensor(areas, device=device),
            owner_roles=torch.as_tensor(relative_owners, device=device),
            token_kinds=torch.as_tensor(token_kinds, device=device),
            scalars=torch.as_tensor(scalars, device=device),
            last_attack_ids=torch.as_tensor(last_attack_ids, device=device),
            padding_mask=torch.as_tensor(padding_mask, device=device),
            attachment_card_ids=torch.as_tensor(
                attachment_cards,
                device=device,
            ),
            attachment_parent_indices=torch.as_tensor(
                attachment_parents,
                device=device,
            ),
            attachment_kinds=torch.as_tensor(
                attachment_kinds,
                device=device,
            ),
            entity_slots=torch.as_tensor(entity_slots, device=device),
            sequence_lengths=tuple(int(length) for length in token_lengths),
        ),
        lookup,
    )


__all__ = [
    "NativeTokenLookup",
    "encode_native_state_batch",
]
