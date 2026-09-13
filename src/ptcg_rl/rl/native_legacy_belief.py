"""Exact legacy belief features on native public columns."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from ptcg_rl.context import ExpectedCardCount
from ptcg_rl.engine.constants import AreaType
from ptcg_rl.engine.native_public_context import NativeKnownOpponentBatch
from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.model.state_encoder import (
    OWNER_OPPONENT,
    TOKEN_KIND_OOV_INDEX,
    TOKEN_KIND_TO_INDEX,
    StateBatch,
)

LEGACY_BELIEF_COUNT_SCALAR = 52
LEGACY_BELIEF_ENTROPY_SCALAR = 53
LEGACY_BELIEF_EMPTY_SCALAR = 54
LegacyBelief = tuple[tuple[ExpectedCardCount, ...], float, bool]


def legacy_known_counts(
    view: NativeTrainingBatchView,
    known: NativeKnownOpponentBatch,
    *,
    row: int,
) -> Counter[int]:
    """Restore the legacy observation traversal order for floating-point parity."""
    known_start = int(known.offsets[row])
    known_stop = int(known.offsets[row + 1])
    totals = Counter(
        {
            int(card_id): int(count)
            for card_id, count in zip(
                known.card_ids[known_start:known_stop],
                known.counts[known_start:known_stop],
                strict=True,
            )
        }
    )
    perspective = int(view.select_player[row])
    opponent = 1 - perspective
    card_start = int(view.visible_card_offsets[row])
    card_stop = int(view.visible_card_offsets[row + 1])
    attachment_start = int(view.attachment_offsets[row])
    attachment_stop = int(view.attachment_offsets[row + 1])
    attachments: defaultdict[int, list[int]] = defaultdict(list)
    for attachment in range(attachment_start, attachment_stop):
        card_id = int(view.attachment_card_id[attachment])
        if card_id > 0:
            attachments[int(view.attachment_parent[attachment])].append(card_id)

    visible_areas = {
        int(AreaType.ACTIVE),
        int(AreaType.BENCH),
        int(AreaType.HAND),
        int(AreaType.DISCARD),
        int(AreaType.PRIZE),
        int(AreaType.STADIUM),
        int(AreaType.LOOKING),
    }
    in_play_areas = {int(AreaType.ACTIVE), int(AreaType.BENCH)}
    ordered: Counter[int] = Counter()
    for card_row in range(card_start, card_stop):
        area = int(view.visible_card_area[card_row])
        if (
            int(view.visible_card_owner[card_row]) != opponent
            or area not in visible_areas
        ):
            continue
        card_id = int(view.visible_card_id[card_row])
        if card_id <= 0:
            continue
        ordered[card_id] += 1
        if area in in_play_areas:
            ordered.update(attachments.get(card_row, ()))

    for card_id, count in totals.items():
        if ordered[card_id] > count:
            raise ValueError(
                "native current opponent evidence exceeds tracked known counts"
            )
        ordered[card_id] = count
    if ordered != totals:
        raise ValueError("native legacy known-card ordering lost public evidence")
    return ordered


def append_legacy_belief_tokens(
    states: StateBatch,
    beliefs: Sequence[LegacyBelief],
) -> StateBatch:
    """Append exact legacy top-k tokens and overwrite their global scalars."""
    rows = int(states.card_ids.shape[0])
    if len(beliefs) != rows:
        raise ValueError("legacy belief rows must align with state rows")
    feature_counts = tuple(len(features) for features, _entropy, _empty in beliefs)
    features = tuple(
        feature
        for row_features, _entropy, _empty in beliefs
        for feature in row_features
    )
    if states.padding_mask.device.type == "cpu" and not states.scalars.requires_grad:
        return _append_legacy_belief_tokens_cpu(
            states,
            beliefs,
            feature_counts=feature_counts,
            features=features,
        )

    lengths = (~states.padding_mask).sum(dim=1, dtype=torch.long)
    belief_lengths = torch.tensor(
        feature_counts,
        dtype=torch.long,
        device=lengths.device,
    )
    old_width = int(states.card_ids.shape[1])
    width = max(old_width, int((lengths + belief_lengths).max().item()))
    output = _widen_state_batch(states, width=width)
    output.scalars[:, 0, LEGACY_BELIEF_ENTROPY_SCALAR] = (
        output.scalars.new_tensor(
            [float(entropy) for _features, entropy, _empty in beliefs]
        )
    )
    output.scalars[:, 0, LEGACY_BELIEF_EMPTY_SCALAR] = (
        output.scalars.new_tensor(
            [float(is_empty) for _features, _entropy, is_empty in beliefs]
        )
    )

    if not features:
        return output

    row_indices = torch.repeat_interleave(
        torch.arange(rows, dtype=torch.long, device=lengths.device),
        belief_lengths,
        output_size=len(features),
    )
    flat_offsets = torch.arange(
        len(features),
        dtype=torch.long,
        device=lengths.device,
    )
    row_starts = torch.repeat_interleave(
        torch.cumsum(belief_lengths, dim=0) - belief_lengths,
        belief_lengths,
        output_size=len(features),
    )
    token_indices = lengths.index_select(0, row_indices) + (
        flat_offsets - row_starts
    )
    output.card_ids[row_indices, token_indices] = output.card_ids.new_tensor(
        [int(feature.card_id) for feature in features]
    )
    output.owner_roles[row_indices, token_indices] = OWNER_OPPONENT
    output.token_kinds[row_indices, token_indices] = TOKEN_KIND_TO_INDEX[
        "opponent_belief"
    ]
    output.scalars[row_indices, token_indices, 3] = -1.0
    output.scalars[
        row_indices,
        token_indices,
        LEGACY_BELIEF_COUNT_SCALAR,
    ] = output.scalars.new_tensor(
        [
            min(1.0, max(0.0, float(feature.expected_count) / 60.0))
            for feature in features
        ]
    )
    output.padding_mask[row_indices, token_indices] = False
    return output


def _append_legacy_belief_tokens_cpu(
    states: StateBatch,
    beliefs: Sequence[LegacyBelief],
    *,
    feature_counts: tuple[int, ...],
    features: tuple[ExpectedCardCount, ...],
) -> StateBatch:
    """Append one CPU batch without entering PyTorch's small-op thread pool."""
    padding_mask = states.padding_mask.numpy()
    lengths = np.count_nonzero(~padding_mask, axis=1)
    belief_lengths = np.asarray(feature_counts, dtype=np.int64)
    old_width = int(states.card_ids.shape[1])
    width = max(old_width, int(np.max(lengths + belief_lengths)))
    output = _widen_state_batch(states, width=width)
    scalars = output.scalars.numpy()
    scalars[:, 0, LEGACY_BELIEF_ENTROPY_SCALAR] = np.asarray(
        [float(entropy) for _features, entropy, _empty in beliefs],
        dtype=scalars.dtype,
    )
    scalars[:, 0, LEGACY_BELIEF_EMPTY_SCALAR] = np.asarray(
        [float(is_empty) for _features, _entropy, is_empty in beliefs],
        dtype=scalars.dtype,
    )
    if not features:
        return output

    row_indices = np.repeat(
        np.arange(len(beliefs), dtype=np.int64),
        belief_lengths,
    )
    flat_offsets = np.arange(len(features), dtype=np.int64)
    row_starts = np.repeat(
        np.cumsum(belief_lengths) - belief_lengths,
        belief_lengths,
    )
    token_indices = lengths[row_indices] + flat_offsets - row_starts
    card_ids = output.card_ids.numpy()
    card_ids[row_indices, token_indices] = np.asarray(
        [int(feature.card_id) for feature in features],
        dtype=card_ids.dtype,
    )
    output.owner_roles.numpy()[row_indices, token_indices] = OWNER_OPPONENT
    output.token_kinds.numpy()[row_indices, token_indices] = TOKEN_KIND_TO_INDEX[
        "opponent_belief"
    ]
    scalars[row_indices, token_indices, 3] = -1.0
    scalars[
        row_indices,
        token_indices,
        LEGACY_BELIEF_COUNT_SCALAR,
    ] = np.asarray(
        [
            min(1.0, max(0.0, float(feature.expected_count) / 60.0))
            for feature in features
        ],
        dtype=scalars.dtype,
    )
    output.padding_mask.numpy()[row_indices, token_indices] = False
    return output


def _widen_state_batch(states: StateBatch, *, width: int) -> StateBatch:
    old_width = int(states.card_ids.shape[1])
    if width < old_width:
        raise ValueError("state token widening cannot shrink the input")

    def widen(values: Tensor, fill: int | float | bool = 0) -> Tensor:
        shape = (int(values.shape[0]), width, *values.shape[2:])
        if values.device.type == "cpu" and not values.requires_grad:
            source = values.numpy()
            array = np.full(shape, fill, dtype=source.dtype)
            array[:, :old_width] = source
            return torch.from_numpy(array)
        output = values.new_full(shape, fill)
        output[:, :old_width] = values
        return output

    return StateBatch(
        card_ids=widen(states.card_ids),
        areas=widen(states.areas),
        owner_roles=widen(states.owner_roles),
        token_kinds=widen(states.token_kinds, TOKEN_KIND_OOV_INDEX),
        scalars=widen(states.scalars),
        last_attack_ids=widen(states.last_attack_ids),
        padding_mask=widen(states.padding_mask, True),
        attachment_card_ids=states.attachment_card_ids,
        attachment_parent_indices=states.attachment_parent_indices,
        attachment_kinds=states.attachment_kinds,
        entity_slots=(
            None if states.entity_slots is None else widen(states.entity_slots)
        ),
        root_input_fingerprints=states.root_input_fingerprints,
    )


__all__ = [
    "LegacyBelief",
    "append_legacy_belief_tokens",
    "legacy_known_counts",
]
