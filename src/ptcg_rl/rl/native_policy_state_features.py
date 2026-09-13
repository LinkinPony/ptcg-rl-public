"""Scalar feature fills for native simple-stateless state encoding."""

from __future__ import annotations

import numpy as np

from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.model.state_encoder import (
    ATTACHMENT_KIND_ENERGY,
    ATTACHMENT_KIND_PRE_EVOLUTION,
    ATTACHMENT_KIND_TOOL,
    ENERGY_TYPE_COUNT,
    ENTITY_SLOT_OOV_INDEX,
    GLOBAL_LONG_TURN_SCALAR,
    GLOBAL_UNCLIPPED_HISTORY_SCALAR_START,
    PUBLIC_STATE_SCALAR_START,
)
from ptcg_rl.rl.native_policy_context import NativePublicContextBatch
from ptcg_rl.rl.native_policy_keys import (
    exact_lookup,
    serial_pointer_keys,
    sorted_key_values,
)
from ptcg_rl.rl.native_policy_state_layout import csr_positions

_ACTIVE_AREA = 4
_BENCH_AREA = 5


def fill_global_scalars(
    destination: np.ndarray,
    view: NativeTrainingBatchView,
    context: NativePublicContextBatch,
) -> None:
    """Fill exact global/current/context scalars in the legacy schema."""
    perspective = view.select_player.astype(np.int64, copy=False)
    opponent = 1 - perspective
    rows = np.arange(view.batch_size, dtype=np.int64)
    player_decks = np.stack(view.player_deck_counts, axis=1)
    player_hands = np.stack(view.player_hand_counts, axis=1)
    player_prizes = np.stack(view.player_prize_counts, axis=1)
    player_benches = np.stack(view.player_bench_max, axis=1)

    deck_flow = context.deck_flow_counts.astype(np.float64, copy=False)
    for player_offset in (0, 7):
        destination[:, player_offset] = _saturating_ratio(
            deck_flow[:, player_offset],
            40.0,
        )
        destination[:, player_offset + 1] = _saturating_ratio(
            deck_flow[:, player_offset + 1],
            20.0,
        )
        destination[:, player_offset + 2] = _saturating_ratio(
            deck_flow[:, player_offset + 2],
            4.0,
        )
        destination[:, player_offset + 3] = _saturating_ratio(
            deck_flow[:, player_offset + 3],
            4.0,
        )
        destination[:, player_offset + 4] = _signed_saturating_ratio(
            deck_flow[:, player_offset + 4],
            4.0,
        )
        destination[:, player_offset + 5] = _saturating_ratio(
            deck_flow[:, player_offset + 5],
            10.0,
        )
        destination[:, player_offset + 6] = _saturating_ratio(
            deck_flow[:, player_offset + 6],
            10.0,
        )
    destination[:, GLOBAL_LONG_TURN_SCALAR] = _saturating_ratio(
        view.turn,
        100.0,
    )
    history = context.history_counts.astype(np.float64, copy=False)
    destination[
        :,
        GLOBAL_UNCLIPPED_HISTORY_SCALAR_START : (
            GLOBAL_UNCLIPPED_HISTORY_SCALAR_START + history.shape[1]
        ),
    ] = _saturating_ratio(history, 20.0)

    destination[:, 24] = np.clip(view.turn / 100.0, 0.0, 1.0)
    destination[:, 25] = np.clip(view.turn_action_count / 20.0, 0.0, 1.0)
    destination[:, 26] = (view.first_player == perspective).astype(np.float32)
    flags = view.turn_flags.astype(np.uint32, copy=False)
    for bit, scalar_index in enumerate(range(27, 31)):
        destination[:, scalar_index] = ((flags >> bit) & 1).astype(np.float32)
    destination[:, 31] = view.result / 2.0
    destination[:, 32] = np.clip(
        player_prizes[rows, perspective] / 6.0,
        0.0,
        1.0,
    )
    destination[:, 33] = np.clip(
        player_prizes[rows, opponent] / 6.0,
        0.0,
        1.0,
    )
    destination[:, 34] = np.clip(
        player_decks[rows, perspective] / 60.0,
        0.0,
        1.0,
    )
    destination[:, 35] = np.clip(
        player_decks[rows, opponent] / 60.0,
        0.0,
        1.0,
    )
    destination[:, 36] = np.clip(
        player_hands[rows, perspective] / 20.0,
        0.0,
        1.0,
    )
    destination[:, 37] = np.clip(
        player_hands[rows, opponent] / 20.0,
        0.0,
        1.0,
    )
    destination[:, 38] = np.clip(view.select_min / 8.0, 0.0, 1.0)
    destination[:, 39] = np.clip(view.select_max / 8.0, 0.0, 1.0)
    destination[:, 40] = np.clip(
        view.remain_damage_counter / 50.0,
        0.0,
        1.0,
    )
    destination[:, 41] = np.clip(
        view.remain_energy_cost / 12.0,
        0.0,
        1.0,
    )
    destination[:, 42] = np.clip(
        np.maximum(0, view.select_type - 1) / 64.0,
        0.0,
        1.0,
    )
    destination[:, 43] = np.clip(
        np.maximum(0, view.select_context - 1) / 64.0,
        0.0,
        1.0,
    )
    destination[:, 44:52] = np.clip(history / 40.0, 0.0, 1.0)
    # The simple public-catalog path keeps legacy state belief tokens empty.
    destination[:, 53] = 0.0
    destination[:, 54] = 1.0
    destination[:, PUBLIC_STATE_SCALAR_START] = player_benches[rows, perspective] / 8.0
    destination[:, PUBLIC_STATE_SCALAR_START + 1] = 1.0
    destination[:, PUBLIC_STATE_SCALAR_START + 2] = player_benches[rows, opponent] / 8.0
    destination[:, PUBLIC_STATE_SCALAR_START + 3] = 1.0


def fill_visible_scalars(
    scalars: np.ndarray,
    entity_slots: np.ndarray,
    view: NativeTrainingBatchView,
    *,
    visible_rows: np.ndarray,
    visible_tokens: np.ndarray,
    visible_offsets: np.ndarray,
) -> None:
    """Fill owner, Pokémon, public status, and attachment scalar features."""
    owners = view.visible_card_owner.astype(np.int64, copy=False)
    perspectives = view.select_player[visible_rows]
    owner_values = np.where(
        owners == perspectives,
        1.0,
        np.where((owners == 0) | (owners == 1), -1.0, 0.0),
    )
    scalars[visible_rows, visible_tokens, 3] = owner_values

    areas = view.visible_card_area.astype(np.int64, copy=False)
    card_ids = view.visible_card_id.astype(np.int64, copy=False)
    pokemon = ((areas == _ACTIVE_AREA) | (areas == _BENCH_AREA)) & (card_ids > 0)
    if np.any(pokemon):
        rows = visible_rows[pokemon]
        tokens = visible_tokens[pokemon]
        hp = np.maximum(0.0, view.visible_card_hp[pokemon].astype(np.float64))
        maximum_hp = np.maximum(
            1.0,
            view.visible_card_max_hp[pokemon].astype(np.float64),
        )
        scalars[rows, tokens, 0] = np.clip(hp / maximum_hp, 0.0, 1.0)
        scalars[rows, tokens, 1] = np.clip(
            (maximum_hp - hp) / maximum_hp,
            0.0,
            1.0,
        )
        scalars[rows, tokens, 2] = np.clip(maximum_hp / 400.0, 0.0, 1.0)
        scalars[rows, tokens, 22] = (
            view.visible_card_appear_this_turn[pokemon] != 0
        ).astype(np.float32)
        entity_slots[rows, tokens] = np.minimum(
            ENTITY_SLOT_OOV_INDEX,
            np.maximum(
                0,
                view.visible_card_area_index[pokemon].astype(np.int64) + 1,
            ),
        ).astype(np.uint8)

        active = pokemon & (areas == _ACTIVE_AREA)
        if np.any(active):
            active_rows = visible_rows[active]
            active_tokens = visible_tokens[active]
            status_by_player = np.stack(view.player_status_flags, axis=1)
            status = status_by_player[active_rows, owners[active]]
            for bit, scalar_index in enumerate(range(17, 22)):
                scalars[active_rows, active_tokens, scalar_index] = (
                    (status >> bit) & 1
                ).astype(np.float32)

    _fill_attachment_scalars(
        scalars,
        view,
        visible_offsets=visible_offsets,
    )


def visible_last_attacks(
    view: NativeTrainingBatchView,
    context: NativePublicContextBatch,
    *,
    visible_rows: np.ndarray,
) -> np.ndarray:
    """Join last-attack IDs to visible in-play serials."""
    output = np.zeros(view.visible_card_count, dtype=np.int64)
    areas = view.visible_card_area.astype(np.int64, copy=False)
    serials = view.visible_card_serial.astype(np.int64, copy=False)
    candidates = ((areas == _ACTIVE_AREA) | (areas == _BENCH_AREA)) & (serials > 0)
    if not np.any(candidates):
        return output
    offsets = context.last_attack_offsets.astype(np.int64, copy=False)
    lengths = np.diff(offsets)
    attack_rows = np.repeat(
        np.arange(view.batch_size, dtype=np.int64),
        lengths,
    )
    keys = serial_pointer_keys(attack_rows, context.last_attack_serials)
    sorted_keys, sorted_ids = sorted_key_values(
        keys,
        context.last_attack_ids.astype(np.int64, copy=False),
    )
    query = serial_pointer_keys(
        visible_rows[candidates],
        serials[candidates],
    )
    output[candidates] = np.maximum(
        0,
        exact_lookup(sorted_keys, sorted_ids, query, missing=0),
    )
    return output


def fill_context_tokens(
    card_ids: np.ndarray,
    areas: np.ndarray,
    owner_roles: np.ndarray,
    token_kinds: np.ndarray,
    scalars: np.ndarray,
    padding_mask: np.ndarray,
    *,
    rows: np.ndarray,
    tokens: np.ndarray,
    source_card_ids: np.ndarray,
    source_counts: np.ndarray,
    owner_role: int,
    owner_scalar: float,
    token_kind: int,
) -> None:
    """Scatter one public context CSR into the padded state batch."""
    if not rows.size:
        return
    padding_mask[rows, tokens] = False
    card_ids[rows, tokens] = source_card_ids.astype(np.int64, copy=False)
    areas[rows, tokens] = 0
    owner_roles[rows, tokens] = owner_role
    token_kinds[rows, tokens] = token_kind
    scalars[rows, tokens, 3] = owner_scalar
    scalars[rows, tokens, 52] = np.clip(
        source_counts.astype(np.float64, copy=False) / 60.0,
        0.0,
        1.0,
    )


def _fill_attachment_scalars(
    scalars: np.ndarray,
    view: NativeTrainingBatchView,
    *,
    visible_offsets: np.ndarray,
) -> None:
    attachment_offsets = view.attachment_offsets.astype(np.int64, copy=False)
    attachment_rows, _local, _tokens = csr_positions(attachment_offsets)
    if not attachment_rows.size:
        return
    parents = view.attachment_parent.astype(np.int64, copy=False)
    starts = visible_offsets[attachment_rows]
    ends = visible_offsets[attachment_rows + 1]
    if np.any((parents < starts) | (parents >= ends)):
        raise ValueError("native attachment parent crosses a public-state row")
    parent_tokens = parents - starts + 2
    kinds = view.attachment_kind.astype(np.int64, copy=False)

    energy = kinds == ATTACHMENT_KIND_ENERGY
    if np.any(energy):
        energy_types = view.attachment_energy_type[energy].astype(
            np.int64,
            copy=False,
        )
        if np.any((energy_types < 0) | (energy_types >= ENERGY_TYPE_COUNT)):
            raise ValueError("native effective energy type is out of range")
        units = view.attachment_energy_units[energy].astype(np.float64)
        if np.any(units < 0):
            raise ValueError("native effective energy units cannot be negative")
        np.add.at(
            scalars,
            (
                attachment_rows[energy],
                parent_tokens[energy],
                4 + energy_types,
            ),
            units / 8.0,
        )
        energy_rows = attachment_rows[energy]
        energy_parents = parent_tokens[energy]
        scalars[energy_rows, energy_parents, 4:16] = np.clip(
            scalars[energy_rows, energy_parents, 4:16],
            0.0,
            1.0,
        )
    for kind, scalar_index, scale in (
        (ATTACHMENT_KIND_TOOL, 16, 4.0),
        (ATTACHMENT_KIND_PRE_EVOLUTION, 23, 3.0),
    ):
        selected = kinds == kind
        if np.any(selected):
            np.add.at(
                scalars,
                (
                    attachment_rows[selected],
                    parent_tokens[selected],
                    np.full(np.count_nonzero(selected), scalar_index),
                ),
                1.0 / scale,
            )
            selected_rows = attachment_rows[selected]
            selected_parents = parent_tokens[selected]
            scalars[selected_rows, selected_parents, scalar_index] = np.clip(
                scalars[selected_rows, selected_parents, scalar_index],
                0.0,
                1.0,
            )


def _saturating_ratio(values: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.maximum(0.0, values.astype(np.float64, copy=False))
    return normalized / (normalized + scale)


def _signed_saturating_ratio(values: np.ndarray, scale: float) -> np.ndarray:
    normalized = values.astype(np.float64, copy=False)
    magnitude = np.abs(normalized)
    return np.sign(normalized) * magnitude / (magnitude + scale)


__all__ = [
    "fill_context_tokens",
    "fill_global_scalars",
    "fill_visible_scalars",
    "visible_last_attacks",
]
