"""Validated numeric token layout for native public-state columns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.model.state_encoder import (
    AREA_OOV_INDEX,
    ATTACHMENT_KIND_COUNT,
    OWNER_OPPONENT,
    OWNER_SELF,
    OWNER_SHARED,
    OWNER_UNKNOWN,
    TOKEN_KIND_OOV_INDEX,
    TOKEN_KIND_TO_INDEX,
)
from ptcg_rl.rl.native_policy_keys import (
    area_pointer_keys,
    attachment_pointer_keys,
    serial_pointer_keys,
    sorted_key_values,
)

_READY_STATUS = 1
_NO_ERROR = 0
_VIRTUAL_AREA = 0
_DECK_AREA = 1
_HAND_AREA = 2
_DISCARD_AREA = 3
_ACTIVE_AREA = 4
_BENCH_AREA = 5
_PRIZE_AREA = 6
_STADIUM_AREA = 7
_LOOKING_AREA = 12
_UINT32_MAX = np.iinfo(np.uint32).max


@dataclass(frozen=True, slots=True)
class NativeTokenLookup:
    """Sorted numeric joins shared by native state and option encoders."""

    batch_size: int
    perspectives: np.ndarray
    area_keys: np.ndarray
    area_tokens: np.ndarray
    serial_any_keys: np.ndarray
    serial_any_tokens: np.ndarray
    serial_any_card_ids: np.ndarray
    serial_own_keys: np.ndarray
    serial_own_tokens: np.ndarray
    serial_own_card_ids: np.ndarray
    attachment_keys: np.ndarray
    attachment_card_ids: np.ndarray
    attachment_serials: np.ndarray


def validate_native_view(view: NativeTrainingBatchView) -> None:
    """Validate the native columns consumed by simple-stateless encoding."""
    rows = view.batch_size
    if rows <= 0:
        raise ValueError("native state batch must be non-empty")
    scalar_columns = (
        view.status,
        view.error,
        view.select_player,
        view.select_type,
        view.select_context,
        view.select_min,
        view.select_max,
        view.result,
        view.turn,
        view.turn_action_count,
        view.first_player,
        view.turn_flags,
        view.remain_damage_counter,
        view.remain_energy_cost,
        *view.player_deck_counts,
        *view.player_hand_counts,
        *view.player_prize_counts,
        *view.player_bench_max,
        *view.player_status_flags,
        view.context_card_row,
        view.effect_card_row,
    )
    if any(column.shape != (rows,) for column in scalar_columns):
        raise ValueError("native scalar columns do not align")
    if np.any(view.status != _READY_STATUS) or np.any(view.error != _NO_ERROR):
        raise ValueError("native policy encoding accepts ready error-free rows only")
    if np.any((view.select_player < 0) | (view.select_player > 1)):
        raise ValueError("native select_player must be an acting seat")
    _validate_csr_offsets(
        view.option_offsets,
        rows=rows,
        value_count=view.option_count,
        name="option",
    )
    if np.any(np.diff(view.option_offsets.astype(np.int64, copy=False)) <= 0):
        raise ValueError("native ready policy rows must have legal options")
    _validate_csr_offsets(
        view.visible_card_offsets,
        rows=rows,
        value_count=view.visible_card_count,
        name="visible-card",
    )
    _validate_csr_offsets(
        view.attachment_offsets,
        rows=rows,
        value_count=view.attachment_count,
        name="attachment",
    )
    visible_columns = (
        view.visible_card_owner,
        view.visible_card_area,
        view.visible_card_area_index,
        view.visible_card_id,
        view.visible_card_serial,
        view.visible_card_hp,
        view.visible_card_max_hp,
        view.visible_card_appear_this_turn,
    )
    if any(column.shape != (view.visible_card_count,) for column in visible_columns):
        raise ValueError("native visible-card columns do not align")
    attachment_columns = (
        view.attachment_parent,
        view.attachment_kind,
        view.attachment_card_id,
        view.attachment_card_serial,
        view.attachment_energy_type,
        view.attachment_energy_units,
    )
    if any(column.shape != (view.attachment_count,) for column in attachment_columns):
        raise ValueError("native attachment columns do not align")


def csr_positions(
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened row, local, and state-token positions for CSR."""
    lengths = np.diff(offsets)
    rows = np.repeat(np.arange(lengths.shape[0], dtype=np.int64), lengths)
    starts = np.repeat(offsets[:-1], lengths)
    local = np.arange(int(offsets[-1]), dtype=np.int64) - starts
    return rows, local, local + 2


def attachment_columns(
    view: NativeTrainingBatchView,
    *,
    visible_offsets: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build padded attachment tensors and sorted identity pointer columns."""
    offsets = view.attachment_offsets.astype(np.int64, copy=False)
    lengths = np.diff(offsets)
    maximum = max(1, int(lengths.max(initial=0)))
    cards = np.zeros((view.batch_size, maximum), dtype=np.uint16)
    parents = np.zeros((view.batch_size, maximum), dtype=np.uint16)
    kinds_output = np.zeros((view.batch_size, maximum), dtype=np.uint8)
    if view.attachment_count == 0:
        empty = np.zeros(0, dtype=np.int64)
        return cards, parents, kinds_output, empty, empty, empty

    rows, local, _unused = csr_positions(offsets)
    absolute_parents = view.attachment_parent.astype(np.int64, copy=False)
    starts = visible_offsets[rows]
    ends = visible_offsets[rows + 1]
    if np.any((absolute_parents < starts) | (absolute_parents >= ends)):
        raise ValueError("native attachment parent crosses a public-state row")
    parent_tokens = absolute_parents - starts + 2
    identity_ids = view.attachment_card_id.astype(np.int64, copy=False)
    identity_serials = view.attachment_card_serial.astype(np.int64, copy=False)
    kinds = view.attachment_kind.astype(np.int64, copy=False)
    if (
        np.any(identity_ids <= 0)
        or np.any(identity_ids > np.iinfo(np.uint16).max)
        or np.any(parent_tokens > np.iinfo(np.uint16).max)
        or np.any((kinds < 1) | (kinds > ATTACHMENT_KIND_COUNT))
    ):
        raise ValueError("native attachment identity exceeds model schema")
    cards[rows, local] = identity_ids.astype(np.uint16)
    parents[rows, local] = parent_tokens.astype(np.uint16)
    kinds_output[rows, local] = kinds.astype(np.uint8)

    group_keys = (rows * (1 << 32) + absolute_parents) * 4 + kinds
    positions = np.arange(view.attachment_count, dtype=np.int64)
    starts_mask = np.empty(view.attachment_count, dtype=np.bool_)
    starts_mask[0] = True
    starts_mask[1:] = group_keys[1:] != group_keys[:-1]
    group_starts = np.maximum.accumulate(np.where(starts_mask, positions, 0))
    attachment_indices = positions - group_starts
    if np.any(attachment_indices >= 256):
        raise ValueError("native attachment index exceeds pointer-key capacity")
    keys = attachment_pointer_keys(
        rows,
        parent_tokens,
        kinds,
        attachment_indices,
    )
    sorted_keys, sorted_ids, sorted_serials = sorted_key_values(
        keys,
        identity_ids,
        identity_serials,
    )
    return (
        cards,
        parents,
        kinds_output,
        sorted_keys,
        sorted_ids,
        sorted_serials,
    )


def token_lookup(
    view: NativeTrainingBatchView,
    *,
    visible_rows: np.ndarray,
    visible_tokens: np.ndarray,
    attachment_keys: np.ndarray,
    attachment_card_ids: np.ndarray,
    attachment_serials: np.ndarray,
) -> NativeTokenLookup:
    """Build the sorted public area, serial, and attachment joins."""
    areas = view.visible_card_area.astype(np.int64, copy=False)
    owners = view.visible_card_owner.astype(np.int64, copy=False)
    indices = view.visible_card_area_index.astype(np.int64, copy=False)
    if np.any((areas < 0) | (areas >= 16) | (indices < 0) | (indices >= 256)):
        raise ValueError("native public card pointer exceeds lookup-key capacity")
    area_keys = area_pointer_keys(visible_rows, areas, owners, indices)
    sorted_area_keys, sorted_area_tokens = sorted_key_values(
        area_keys,
        visible_tokens,
    )

    serials = view.visible_card_serial.astype(np.int64, copy=False)
    card_ids = view.visible_card_id.astype(np.int64, copy=False)
    serialized = serials > 0
    any_keys = serial_pointer_keys(
        visible_rows[serialized],
        serials[serialized],
    )
    sorted_any_keys, sorted_any_tokens, sorted_any_cards = sorted_key_values(
        any_keys,
        visible_tokens[serialized],
        card_ids[serialized],
    )
    own = serialized & (owners == view.select_player[visible_rows])
    own_keys = serial_pointer_keys(
        visible_rows[own],
        serials[own],
    )
    sorted_own_keys, sorted_own_tokens, sorted_own_cards = sorted_key_values(
        own_keys,
        visible_tokens[own],
        card_ids[own],
    )
    return NativeTokenLookup(
        batch_size=view.batch_size,
        perspectives=view.select_player.astype(np.int64, copy=False),
        area_keys=sorted_area_keys,
        area_tokens=sorted_area_tokens,
        serial_any_keys=sorted_any_keys,
        serial_any_tokens=sorted_any_tokens,
        serial_any_card_ids=sorted_any_cards,
        serial_own_keys=sorted_own_keys,
        serial_own_tokens=sorted_own_tokens,
        serial_own_card_ids=sorted_own_cards,
        attachment_keys=attachment_keys,
        attachment_card_ids=attachment_card_ids,
        attachment_serials=attachment_serials,
    )


def visible_token_kinds(
    view: NativeTrainingBatchView,
    *,
    visible_rows: np.ndarray,
    visible_absolute: np.ndarray,
) -> np.ndarray:
    """Map native public areas and virtual-row references to token kinds."""
    areas = view.visible_card_area.astype(np.int64, copy=False)
    kind_by_area = np.full(13, TOKEN_KIND_OOV_INDEX, dtype=np.int64)
    for area, name in (
        (_DECK_AREA, "deck"),
        (_HAND_AREA, "hand"),
        (_DISCARD_AREA, "discard"),
        (_ACTIVE_AREA, "active"),
        (_BENCH_AREA, "bench"),
        (_PRIZE_AREA, "prize"),
        (_STADIUM_AREA, "stadium"),
        (_LOOKING_AREA, "looking"),
    ):
        kind_by_area[area] = TOKEN_KIND_TO_INDEX[name]
    safe = (areas >= 0) & (areas < kind_by_area.shape[0])
    output = np.full(areas.shape, TOKEN_KIND_OOV_INDEX, dtype=np.int64)
    output[safe] = kind_by_area[areas[safe]]
    for references, expected_index, name in (
        (view.context_card_row, 0, "contextCard"),
        (view.effect_card_row, 1, "effect"),
    ):
        present_rows = np.flatnonzero(references != _UINT32_MAX)
        if not present_rows.size:
            continue
        absolute = references[present_rows].astype(np.int64, copy=False)
        starts = view.visible_card_offsets[present_rows].astype(
            np.int64,
            copy=False,
        )
        ends = view.visible_card_offsets[present_rows + 1].astype(
            np.int64,
            copy=False,
        )
        if np.any((absolute < starts) | (absolute >= ends)):
            raise ValueError(f"native {name} row crosses a public-state row")
        if np.any(
            (view.visible_card_area[absolute] != _VIRTUAL_AREA)
            | (view.visible_card_area_index[absolute] != expected_index)
            | (visible_rows[absolute] != present_rows)
        ):
            raise ValueError(f"native {name} row has inconsistent metadata")
        output[absolute] = TOKEN_KIND_TO_INDEX[name]
    virtual = areas == _VIRTUAL_AREA
    if np.any(virtual & (output == TOKEN_KIND_OOV_INDEX)):
        raise ValueError("native virtual card row is not contextCard or effect")
    if output.shape != visible_absolute.shape:
        raise RuntimeError("native visible token-kind encoding lost row alignment")
    return output


def safe_areas(values: np.ndarray) -> np.ndarray:
    """Map raw public areas to the existing model OOV bucket."""
    normalized = values.astype(np.int64, copy=False)
    return np.where(
        (normalized >= _VIRTUAL_AREA) & (normalized < AREA_OOV_INDEX),
        normalized,
        AREA_OOV_INDEX,
    )


def owner_roles(owners: np.ndarray, perspectives: np.ndarray) -> np.ndarray:
    """Map absolute public owners to perspective-relative model roles."""
    normalized = owners.astype(np.int64, copy=False)
    return np.where(
        normalized == perspectives,
        OWNER_SELF,
        np.where(
            (normalized == 0) | (normalized == 1),
            OWNER_OPPONENT,
            np.where(normalized < 0, OWNER_SHARED, OWNER_UNKNOWN),
        ),
    )


def _validate_csr_offsets(
    offsets: np.ndarray,
    *,
    rows: int,
    value_count: int,
    name: str,
) -> None:
    if offsets.shape != (rows + 1,) or not np.issubdtype(
        offsets.dtype,
        np.integer,
    ):
        raise ValueError(f"native {name} offsets have an invalid shape or dtype")
    normalized = offsets.astype(np.int64, copy=False)
    if (
        int(normalized[0]) != 0
        or int(normalized[-1]) != value_count
        or np.any(normalized[1:] < normalized[:-1])
    ):
        raise ValueError(f"native {name} offsets are not canonical")


__all__ = [
    "NativeTokenLookup",
    "attachment_columns",
    "csr_positions",
    "owner_roles",
    "safe_areas",
    "token_lookup",
    "validate_native_view",
    "visible_token_kinds",
]
