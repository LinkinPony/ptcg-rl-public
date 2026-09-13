"""Sequence actor rows reconstructed from model-ready native rollout batches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from torch import Tensor

from ptcg_rl.actions.encoding import SCALAR_FEATURE_SIZE, EncodedOptionArrayFeatures
from ptcg_rl.belief.public_catalog import PublicDeckPosteriorArrays
from ptcg_rl.context.public_events import PublicEventDelta
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.native_public_events import native_public_event_deltas
from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.model.policy import MAX_ENTITY_SLOTS, OptionBatch
from ptcg_rl.model.simple_stateless.belief import PublicBeliefSummaryBatch
from ptcg_rl.model.state_encoder import (
    TOKEN_SCALAR_SIZE,
    StateBatch,
    StateTokenArrayFeatures,
)
from ptcg_rl.rl.native_collection_games import NativeLiveGame
from ptcg_rl.rl.native_policy_batch import NativeSimpleStatelessPolicyBatch
from ptcg_rl.rl.policy_inputs import SimpleStatelessActorRow
from ptcg_rl.rl.sequence_types import SequenceDecisionIdentity

Int64Array = npt.NDArray[np.int64]
Array = npt.NDArray[np.generic]

_EMPTY_VECTOR = np.empty(0, dtype=np.int64)
_EMPTY_STATE = StateTokenArrayFeatures(
    card_ids=_EMPTY_VECTOR,
    areas=_EMPTY_VECTOR,
    owner_roles=_EMPTY_VECTOR,
    token_kinds=_EMPTY_VECTOR,
    scalars=np.empty((0, TOKEN_SCALAR_SIZE), dtype=np.float32),
    last_attack_ids=_EMPTY_VECTOR,
    attachment_card_ids=np.empty(0, dtype=np.uint16),
    attachment_parent_indices=np.empty(0, dtype=np.uint16),
    attachment_kinds=np.empty(0, dtype=np.uint8),
    entity_slots=_EMPTY_VECTOR,
)
_EMPTY_OPTIONS = EncodedOptionArrayFeatures(
    option_types=_EMPTY_VECTOR,
    contexts=_EMPTY_VECTOR,
    entity_slots=np.empty((0, MAX_ENTITY_SLOTS), dtype=np.int64),
    entity_slot_mask=np.empty((0, MAX_ENTITY_SLOTS), dtype=np.bool_),
    attack_ids=_EMPTY_VECTOR,
    card_ids=_EMPTY_VECTOR,
    scalars=np.empty((0, SCALAR_FEATURE_SIZE), dtype=np.float32),
    dynamic_effect_features=np.empty(
        (0, DYNAMIC_EFFECT_FEATURE_SIZE),
        dtype=np.float32,
    ),
    dynamic_effect_masks=np.empty(0, dtype=np.bool_),
)
_EMPTY_BELIEF = PublicDeckPosteriorArrays(
    card_ids=np.empty(0, dtype=np.int32),
    expected_counts=np.empty(0, dtype=np.float32),
    unknown_probability=0.0,
    entropy=0.0,
    compatible_deck_count=0,
    public_evidence_count=0,
)


@dataclass(frozen=True, slots=True)
class _StateArrays:
    """Zero-copy NumPy views over one encoded native state batch."""

    card_ids: Array
    areas: Array
    owner_roles: Array
    token_kinds: Array
    scalars: Array
    last_attack_ids: Array
    padding_mask: Array
    attachment_card_ids: Array
    attachment_parent_indices: Array
    attachment_kinds: Array
    entity_slots: Array
    sequence_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _OptionArrays:
    """Zero-copy NumPy views over one encoded native option batch."""

    option_types: Array
    contexts: Array
    entity_slots: Array
    entity_slot_mask: Array
    attack_ids: Array
    card_ids: Array
    scalars: Array
    dynamic_effect_features: Array
    dynamic_effect_masks: Array
    valid_options: Array
    option_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _BeliefArrays:
    """Zero-copy NumPy views over one encoded native belief batch."""

    card_ids: Array
    expected_counts: Array
    valid_mask: Array
    scalars: Array
    row_indices: Array | None


@dataclass(frozen=True, slots=True)
class _BatchArrays:
    states: _StateArrays
    options: _OptionArrays
    belief: _BeliefArrays


def build_native_sequence_actor_rows(
    batch: NativeSimpleStatelessPolicyBatch,
    *,
    batch_rows: npt.ArrayLike,
    view: NativeTrainingBatchView,
    native_rows: npt.ArrayLike,
    live: Mapping[int, NativeLiveGame],
    engine_fact_producer_fingerprint: str | None,
    materialize_public_events: bool = True,
    materialize_raw_payloads: bool = True,
) -> tuple[SimpleStatelessActorRow, ...]:
    """Restore transaction rows without reconstructing engine observations.

    Native formal collection does not retain raw sequence blocks.  Its rows
    therefore carry only transaction/deck identity; accepted-action semantics
    are materialized later from the tensor-native host batch.
    """
    selected = _rows(batch_rows, size=batch.batch_size, name="policy batch")
    source_rows = _rows(native_rows, size=view.batch_size, name="native view")
    if selected.shape != source_rows.shape:
        raise ValueError("native sequence policy and engine rows are misaligned")
    event_deltas = (
        native_public_event_deltas(view, rows=source_rows)
        if materialize_public_events
        else (PublicEventDelta(),) * int(source_rows.size)
    )
    arrays = _batch_arrays(batch) if materialize_raw_payloads else None
    return tuple(
        _actor_row(
            batch,
            arrays=arrays,
            batch_row=int(batch_row),
            view=view,
            view_row=int(source_rows[local_row]),
            live=live,
            public_events=event_deltas[local_row],
            engine_fact_producer_fingerprint=(engine_fact_producer_fingerprint),
        )
        for local_row, batch_row in enumerate(selected)
    )


def _actor_row(
    batch: NativeSimpleStatelessPolicyBatch,
    *,
    arrays: _BatchArrays | None,
    batch_row: int,
    view: NativeTrainingBatchView,
    view_row: int,
    live: Mapping[int, NativeLiveGame],
    public_events: PublicEventDelta,
    engine_fact_producer_fingerprint: str | None,
) -> SimpleStatelessActorRow:
    slot = int(view.slots[view_row])
    try:
        game = live[slot]
    except KeyError as error:
        raise KeyError("native sequence row references an unknown game") from error
    seat = int(view.select_player[view_row])
    if seat not in (0, 1):
        raise ValueError("native sequence row has no acting seat")
    normalized_seat: Literal[0, 1] = 0 if seat == 0 else 1
    own_deck = game.candidate if seat == game.candidate_seat else game.opponent
    decision_index = game.sequence_decisions_by_seat[seat]
    return SimpleStatelessActorRow(
        state=_EMPTY_STATE if arrays is None else _state_row(arrays.states, batch_row),
        options=(
            _EMPTY_OPTIONS if arrays is None else _option_row(arrays.options, batch_row)
        ),
        min_count=int(batch.min_counts[batch_row]),
        max_count=int(batch.max_counts[batch_row]),
        own_deck=own_deck,
        belief_summary=(
            _EMPTY_BELIEF if arrays is None else _belief_row(arrays.belief, batch_row)
        ),
        catalog_fingerprint=batch.public_deck_catalog_fingerprint,
        input_contract_fingerprint=batch.input_contract_fingerprint,
        public_event_delta=public_events,
        engine_fact_producer_fingerprint=engine_fact_producer_fingerprint,
        sequence_identity=SequenceDecisionIdentity(
            game_id=game.game_id,
            seat=normalized_seat,
            decision_index=decision_index,
            request_id=f"native-{game.game_id}-{seat}-{decision_index}",
        ),
    )


def _state_row(
    state: _StateArrays | NativeSimpleStatelessPolicyBatch,
    row: int,
) -> StateTokenArrayFeatures:
    if not isinstance(state, _StateArrays):
        state = _state_arrays(state.states)
    padding = state.padding_mask[row].astype(np.bool_, copy=False)
    width = (
        state.sequence_lengths[row] if state.sequence_lengths else int((~padding).sum())
    )
    if (
        width <= 0
        or width > padding.size
        or np.any(padding[:width])
        or np.any(~padding[width:])
    ):
        raise ValueError("native sequence state row is not prefix packed")
    attachments = state.attachment_kinds[row] != 0
    _prefix_mask(attachments, name="attachment", allow_empty=True)
    attachment_width = int(attachments.sum())
    return StateTokenArrayFeatures(
        card_ids=state.card_ids[row, :width],
        areas=state.areas[row, :width],
        owner_roles=state.owner_roles[row, :width],
        token_kinds=state.token_kinds[row, :width],
        scalars=state.scalars[row, :width],
        last_attack_ids=state.last_attack_ids[row, :width],
        attachment_card_ids=state.attachment_card_ids[row, :attachment_width],
        attachment_parent_indices=(
            state.attachment_parent_indices[row, :attachment_width]
        ),
        attachment_kinds=state.attachment_kinds[row, :attachment_width],
        entity_slots=state.entity_slots[row, :width],
    )


def _option_row(
    options: _OptionArrays | NativeSimpleStatelessPolicyBatch,
    row: int,
) -> EncodedOptionArrayFeatures:
    if not isinstance(options, _OptionArrays):
        options = _option_arrays(options.options)
    valid = options.valid_options[row].astype(np.bool_, copy=False)
    width = options.option_lengths[row] if options.option_lengths else int(valid.sum())
    if (
        width <= 0
        or width > valid.size
        or not np.all(valid[:width])
        or np.any(valid[width:])
    ):
        raise ValueError("native sequence option row is not prefix packed")
    return EncodedOptionArrayFeatures(
        option_types=options.option_types[row, :width],
        contexts=options.contexts[row, :width],
        entity_slots=options.entity_slots[row, :width],
        entity_slot_mask=options.entity_slot_mask[row, :width],
        attack_ids=options.attack_ids[row, :width],
        card_ids=options.card_ids[row, :width],
        scalars=options.scalars[row, :width],
        dynamic_effect_features=options.dynamic_effect_features[row, :width],
        dynamic_effect_masks=options.dynamic_effect_masks[row, :width],
    )


def _belief_row(
    belief: _BeliefArrays | NativeSimpleStatelessPolicyBatch,
    row: int,
) -> PublicDeckPosteriorArrays:
    if not isinstance(belief, _BeliefArrays):
        belief = _belief_arrays(belief.belief_summary)
    source_row = row if belief.row_indices is None else int(belief.row_indices[row])
    valid = belief.valid_mask[source_row].astype(np.bool_, copy=False)
    _prefix_mask(valid, name="belief", allow_empty=True)
    scalars = belief.scalars[source_row]
    if scalars.shape != (4,):
        raise ValueError("native sequence belief scalars are malformed")
    return PublicDeckPosteriorArrays(
        card_ids=np.asarray(
            belief.card_ids[source_row][valid],
            dtype=np.int32,
        ),
        expected_counts=np.asarray(
            belief.expected_counts[source_row][valid],
            dtype=np.float32,
        ),
        entropy=float(scalars[0]),
        compatible_deck_count=int(round(float(scalars[1]))),
        public_evidence_count=int(round(float(scalars[2]))),
        unknown_probability=float(scalars[3]),
    )


def _batch_arrays(batch: NativeSimpleStatelessPolicyBatch) -> _BatchArrays:
    """Create one zero-copy tensor-to-NumPy view per encoded batch column."""
    return _BatchArrays(
        states=_state_arrays(batch.states),
        options=_option_arrays(batch.options),
        belief=_belief_arrays(batch.belief_summary),
    )


def _state_arrays(state: StateBatch) -> _StateArrays:
    return _StateArrays(
        card_ids=_numpy(state.card_ids),
        areas=_numpy(state.areas),
        owner_roles=_numpy(state.owner_roles),
        token_kinds=_numpy(state.token_kinds),
        scalars=_numpy(state.scalars),
        last_attack_ids=_numpy(state.last_attack_ids),
        padding_mask=_numpy(state.padding_mask),
        attachment_card_ids=_numpy_required(
            state.attachment_card_ids,
            name="attachment card IDs",
        ),
        attachment_parent_indices=_numpy_required(
            state.attachment_parent_indices,
            name="attachment parent indices",
        ),
        attachment_kinds=_numpy_required(
            state.attachment_kinds,
            name="attachment kinds",
        ),
        entity_slots=_numpy_required(
            state.entity_slots,
            name="state entity slots",
        ),
        sequence_lengths=getattr(state, "sequence_lengths", ()),
    )


def _option_arrays(options: OptionBatch) -> _OptionArrays:
    return _OptionArrays(
        option_types=_numpy(options.option_types),
        contexts=_numpy(options.contexts),
        entity_slots=_numpy(options.entity_slots),
        entity_slot_mask=_numpy(options.entity_slot_mask),
        attack_ids=_numpy(options.attack_ids),
        card_ids=_numpy(options.card_ids),
        scalars=_numpy(options.scalars),
        dynamic_effect_features=_numpy(options.dynamic_effect_features),
        dynamic_effect_masks=_numpy(options.dynamic_effect_masks),
        valid_options=_numpy(options.valid_options),
        option_lengths=getattr(options, "option_lengths", ()),
    )


def _belief_arrays(belief: PublicBeliefSummaryBatch) -> _BeliefArrays:
    return _BeliefArrays(
        card_ids=_numpy(belief.card_ids),
        expected_counts=_numpy(belief.expected_counts),
        valid_mask=_numpy(belief.valid_mask),
        scalars=_numpy(belief.scalars),
        row_indices=(
            None if belief.row_indices is None else _numpy(belief.row_indices)
        ),
    )


def _rows(values: npt.ArrayLike, *, size: int, name: str) -> Int64Array:
    rows = np.asarray(values)
    if rows.ndim != 1 or rows.size <= 0:
        raise ValueError(f"{name} rows must be a non-empty vector")
    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError(f"{name} rows must use integers")
    normalized = rows.astype(np.int64, copy=False)
    if (
        np.any(normalized < 0)
        or np.any(normalized >= size)
        or np.unique(normalized).size != normalized.size
    ):
        raise ValueError(f"{name} rows are invalid")
    return normalized


def _numpy(value: Tensor) -> Array:
    if value.device.type != "cpu":
        raise ValueError("native sequence row reconstruction requires CPU tensors")
    return value.detach().numpy()


def _numpy_required(value: Tensor | None, *, name: str) -> Array:
    if value is None:
        raise ValueError(f"native sequence batch has no {name}")
    return _numpy(value)


def _prefix_mask(
    mask: npt.NDArray[np.bool_],
    *,
    name: str,
    allow_empty: bool = False,
) -> None:
    if mask.ndim != 1 or (not allow_empty and not np.any(mask)):
        raise ValueError(f"native sequence {name} row is empty or malformed")
    if mask.size > 1 and np.any(mask[1:] & ~mask[:-1]):
        raise ValueError(f"native sequence {name} row is not prefix packed")


__all__ = ["build_native_sequence_actor_rows"]
