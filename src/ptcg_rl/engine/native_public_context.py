"""Columnar public-history tracking for the native training arena.

The native arena already projects both current state and incremental logs for
the acting seat.  This module keeps the small amount of policy state that is
not game-rule state: public card evidence, history counters, deck circulation,
and a cached public-catalog posterior.  It never reconstructs an observation
mapping or a :class:`GameContext` on the rollout hot path.

Tracker mutation is transactional with respect to each accepted batch.  Native
capacity failures do not return a batch view and therefore never call this
tracker.  For a returned view, malformed columns fail before state is committed;
slot-error rows must have an empty log delta and retain their previous context.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ptcg_rl.belief.public_catalog import (
    PublicDeckCatalog,
    PublicDeckPosteriorArrays,
)
from ptcg_rl.context.game import default_supporter_card_ids
from ptcg_rl.decks.identity import CanonicalDeck, canonicalize_deck
from ptcg_rl.engine.constants import AreaType, LogType
from ptcg_rl.engine.native_public_history import (
    EXPECTED_LOG_PARAM_COUNTS,
    LOG_PARAM_WIDTH,
    SUCCESS_STATUSES,
    NativePublicHistoryState,
)
from ptcg_rl.rl.native_policy_context import NativePublicContextBatch

if TYPE_CHECKING:
    from ptcg_rl.engine.native_training import NativeTrainingBatchView

Int32Array = npt.NDArray[np.int32]
Uint32Array = npt.NDArray[np.uint32]

_VISIBLE_CARD_VIRTUAL_AREA = 0
_VISIBLE_CARD_DECK_AREA = int(AreaType.DECK)
_VISIBLE_CARD_HAND_AREA = int(AreaType.HAND)
_DEFAULT_POSTERIOR_CACHE_CAPACITY = 4096
_KnownIdentity = tuple[tuple[int, int], ...]


class NativePublicContextError(RuntimeError):
    """Raised when native public columns cannot advance tracker state safely."""


@dataclass(frozen=True, slots=True)
class NativeKnownOpponentBatch:
    """Sparse exact public evidence used only for learner belief targets."""

    offsets: Uint32Array
    card_ids: Int32Array
    counts: Int32Array

    @property
    def batch_size(self) -> int:
        """Return the number of aligned decision rows."""
        return int(self.offsets.shape[0] - 1)


@dataclass(frozen=True, slots=True)
class _VisibleEvidence:
    own_counts: Counter[int]
    opponent_counts: Counter[int]
    opponent_by_serial: dict[int, int]


class NativePublicContextTracker:
    """Track public model context for fixed native arena slots and both seats."""

    def __init__(
        self,
        *,
        slot_capacity: int,
        catalog: PublicDeckCatalog,
        input_contract_fingerprint: str,
        supporter_card_ids: Sequence[int] | None = None,
        posterior_cache_capacity: int = _DEFAULT_POSTERIOR_CACHE_CAPACITY,
    ) -> None:
        """Bind immutable catalog/runtime identity to reusable slot state."""
        if slot_capacity <= 0:
            raise ValueError("native public tracker capacity must be positive")
        if posterior_cache_capacity <= 0:
            raise ValueError("native posterior cache capacity must be positive")
        self.slot_capacity = int(slot_capacity)
        self.catalog = catalog
        self.input_contract_fingerprint = input_contract_fingerprint
        self.supporter_card_ids = frozenset(
            int(card_id)
            for card_id in (
                default_supporter_card_ids()
                if supporter_card_ids is None
                else supporter_card_ids
            )
        )
        self._states: dict[tuple[int, int], NativePublicHistoryState] = {}
        self._posterior_cache_capacity = int(posterior_cache_capacity)
        self._posterior_cache: OrderedDict[
            _KnownIdentity,
            PublicDeckPosteriorArrays,
        ] = OrderedDict()

    def consume_reset(
        self,
        batch: NativeTrainingBatchView,
        decks: npt.ArrayLike,
    ) -> NativePublicContextBatch:
        """Reset successful slots, consume their first log delta, and emit context.

        A failed reset row leaves both seat contexts untouched.  If that slot
        has never had a successful reset, no aligned model input exists and the
        method fails closed.
        """
        deck_rows = np.ascontiguousarray(decks, dtype=np.int32)
        expected_shape = (batch.batch_size, 2, 60)
        if deck_rows.shape != expected_shape:
            raise ValueError(
                f"native reset decks must have shape {expected_shape}, "
                f"got {deck_rows.shape}"
            )
        canonical_rows: dict[tuple[int, int], CanonicalDeck] = {}
        for row in range(batch.batch_size):
            if int(batch.status[row]) not in SUCCESS_STATUSES:
                continue
            for perspective in range(2):
                canonical_rows[(row, perspective)] = canonicalize_deck(
                    deck_rows[row, perspective]
                )
        return self._consume(
            batch,
            reset_decks=canonical_rows,
        )

    def consume_step(
        self,
        batch: NativeTrainingBatchView,
    ) -> NativePublicContextBatch:
        """Consume one native step output without reconstructing observations."""
        return self._consume(batch, reset_decks=None)

    def clear_slots(self, slots: npt.ArrayLike) -> None:
        """Forget explicitly retired slots so later reuse requires a reset."""
        slot_rows = np.asarray(slots)
        if slot_rows.ndim != 1:
            raise ValueError("native public tracker slots must be one-dimensional")
        normalized = tuple(int(slot) for slot in slot_rows)
        if any(slot < 0 or slot >= self.slot_capacity for slot in normalized):
            raise ValueError("native public tracker slot is out of range")
        for slot in normalized:
            self._states.pop((slot, 0), None)
            self._states.pop((slot, 1), None)

    def known_opponent_batch(
        self,
        slots: npt.ArrayLike,
        perspectives: npt.ArrayLike,
    ) -> NativeKnownOpponentBatch:
        """Return learner-only known-card CSR after a successful consume call."""
        slot_rows = np.asarray(slots)
        perspective_rows = np.asarray(perspectives)
        if (
            slot_rows.ndim != 1
            or perspective_rows.shape != slot_rows.shape
            or slot_rows.shape[0] <= 0
        ):
            raise ValueError("known-opponent rows must be aligned vectors")
        rows: list[_KnownIdentity] = []
        for raw_slot, raw_perspective in zip(
            slot_rows,
            perspective_rows,
            strict=True,
        ):
            key = (int(raw_slot), int(raw_perspective))
            state = self._states.get(key)
            if state is None or state.cached_known is None:
                raise NativePublicContextError(
                    "known-opponent evidence requested before context consumption"
                )
            rows.append(state.cached_known)
        offsets, card_ids, counts = _count_csr(rows)
        return NativeKnownOpponentBatch(
            offsets=offsets,
            card_ids=card_ids,
            counts=counts,
        )

    def _consume(
        self,
        batch: NativeTrainingBatchView,
        *,
        reset_decks: dict[tuple[int, int], CanonicalDeck] | None,
    ) -> NativePublicContextBatch:
        _validate_native_batch(batch, slot_capacity=self.slot_capacity)
        candidates: dict[tuple[int, int], NativePublicHistoryState] = {}
        if reset_decks is not None:
            for row in range(batch.batch_size):
                if int(batch.status[row]) not in SUCCESS_STATUSES:
                    continue
                slot = int(batch.slots[row])
                for perspective in range(2):
                    candidates[(slot, perspective)] = NativePublicHistoryState(
                        perspective=perspective,
                        own_deck=reset_decks[(row, perspective)],
                        supporter_card_ids=self.supporter_card_ids,
                    )

        row_states: list[NativePublicHistoryState] = []
        visible_rows: list[_VisibleEvidence] = []
        log_players, deck_deltas = _project_log_columns(batch)
        for row in range(batch.batch_size):
            slot = int(batch.slots[row])
            perspective = int(batch.select_player[row])
            key = (slot, perspective)
            current = candidates.get(key, self._states.get(key))
            if current is None:
                raise NativePublicContextError(
                    "native output references a slot/perspective without a "
                    "successful tracked reset"
                )
            if int(batch.status[row]) in SUCCESS_STATUSES:
                if reset_decks is None:
                    current = current.clone()
                    candidates[key] = current
                visible = _visible_evidence(batch, row, perspective=perspective)
                current.record_current_opponent(visible.opponent_by_serial)
                current.apply_log_delta(
                    batch,
                    row,
                    log_players=log_players,
                    deck_deltas=deck_deltas,
                )
            else:
                visible = _visible_evidence(batch, row, perspective=perspective)
            row_states.append(current)
            visible_rows.append(visible)

        output = _build_context_batch(
            row_states,
            visible_rows,
            posterior_for_known=self._posterior_for_known,
            catalog_fingerprint=self.catalog.fingerprint,
            input_contract_fingerprint=self.input_contract_fingerprint,
        )
        output.validate(batch_size=batch.batch_size)
        self._states.update(candidates)
        return output

    def _posterior_for_known(
        self,
        identity: _KnownIdentity,
    ) -> PublicDeckPosteriorArrays:
        cached = self._posterior_cache.get(identity)
        if cached is not None:
            self._posterior_cache.move_to_end(identity)
            return cached
        posterior = self.catalog.posterior_arrays(Counter(dict(identity)))
        self._posterior_cache[identity] = posterior
        if len(self._posterior_cache) > self._posterior_cache_capacity:
            self._posterior_cache.popitem(last=False)
        return posterior


def _validate_native_batch(
    batch: NativeTrainingBatchView,
    *,
    slot_capacity: int,
) -> None:
    rows = batch.batch_size
    if rows <= 0:
        raise NativePublicContextError("native public batch must be non-empty")
    if (
        batch.slots.shape != (rows,)
        or batch.status.shape != (rows,)
        or batch.select_player.shape != (rows,)
    ):
        raise NativePublicContextError("native public scalar columns do not align")
    slots = batch.slots.astype(np.int64, copy=False)
    if (
        np.any(slots < 0)
        or np.any(slots >= slot_capacity)
        or np.unique(slots).shape[0] != rows
    ):
        raise NativePublicContextError("native public slots are invalid")
    if np.any((batch.select_player < 0) | (batch.select_player > 1)):
        raise NativePublicContextError(
            "native public output has no valid acting perspective"
        )
    _validate_csr(
        batch.visible_card_offsets,
        rows=rows,
        value_count=batch.visible_card_count,
        name="visible card",
    )
    _validate_csr(
        batch.attachment_offsets,
        rows=rows,
        value_count=batch.attachment_count,
        name="attachment",
    )
    _validate_csr(
        batch.log_offsets,
        rows=rows,
        value_count=batch.log_count,
        name="public log",
    )
    visible_columns = (
        batch.visible_card_owner,
        batch.visible_card_area,
        batch.visible_card_id,
        batch.visible_card_serial,
    )
    if any(column.shape != (batch.visible_card_count,) for column in visible_columns):
        raise NativePublicContextError("native visible-card columns do not align")
    attachment_columns = (
        batch.attachment_parent,
        batch.attachment_card_id,
        batch.attachment_card_serial,
    )
    if any(column.shape != (batch.attachment_count,) for column in attachment_columns):
        raise NativePublicContextError("native attachment columns do not align")
    if batch.log_type.shape != (batch.log_count,) or batch.log_param_count.shape != (
        batch.log_count,
    ):
        raise NativePublicContextError("native public-log columns do not align")
    if len(batch.log_params) != LOG_PARAM_WIDTH or any(
        column.shape != (batch.log_count,) for column in batch.log_params
    ):
        raise NativePublicContextError("native public-log parameter columns do not align")
    log_lengths = np.diff(batch.log_offsets.astype(np.int64, copy=False))
    success = np.isin(batch.status, tuple(SUCCESS_STATUSES))
    if np.any((~success) & (log_lengths != 0)):
        raise NativePublicContextError("native slot-error row exposed log state")
    log_types = batch.log_type.astype(np.int64, copy=False)
    if np.any((log_types < 0) | (log_types >= len(EXPECTED_LOG_PARAM_COUNTS))):
        raise NativePublicContextError("native public log type is unsupported")
    expected_counts = np.asarray(
        EXPECTED_LOG_PARAM_COUNTS,
        dtype=np.int64,
    )[log_types]
    if np.any(
        batch.log_param_count.astype(np.int64, copy=False) != expected_counts
    ):
        raise NativePublicContextError(
            "native public log parameter count is not canonical"
        )
    for index, column in enumerate(batch.log_params):
        if np.any(
            (expected_counts <= index)
            & (column.astype(np.int64, copy=False) != 0)
        ):
            raise NativePublicContextError(
                "native public log has nonzero private/unused parameters"
            )


def _project_log_columns(
    batch: NativeTrainingBatchView,
) -> tuple[Int32Array, npt.NDArray[np.int8]]:
    """Project repeated log player and deck-delta lookups once per batch."""
    log_types = batch.log_type.astype(np.int64, copy=False)
    players = batch.log_params[0].astype(np.int32, copy=True)
    players[log_types == int(LogType.RESULT)] = -1
    deltas = np.zeros(batch.log_count, dtype=np.int8)
    draw = (log_types == int(LogType.DRAW)) | (
        log_types == int(LogType.DRAW_REVERSE)
    )
    deltas[draw] = -1
    move = log_types == int(LogType.MOVE_CARD)
    deltas[move] = (
        (batch.log_params[4][move] == int(AreaType.DECK)).astype(np.int8)
        - (batch.log_params[3][move] == int(AreaType.DECK)).astype(np.int8)
    )
    reverse = log_types == int(LogType.MOVE_CARD_REVERSE)
    deltas[reverse] = (
        (batch.log_params[2][reverse] == int(AreaType.DECK)).astype(np.int8)
        - (batch.log_params[1][reverse] == int(AreaType.DECK)).astype(np.int8)
    )
    return players, deltas


def _validate_csr(
    offsets: np.ndarray,
    *,
    rows: int,
    value_count: int,
    name: str,
) -> None:
    values = offsets.astype(np.int64, copy=False)
    if (
        offsets.shape != (rows + 1,)
        or int(values[0]) != 0
        or int(values[-1]) != value_count
        or np.any(values[1:] < values[:-1])
    ):
        raise NativePublicContextError(f"native {name} offsets are invalid")


def _visible_evidence(
    batch: NativeTrainingBatchView,
    row: int,
    *,
    perspective: int,
) -> _VisibleEvidence:
    own_counts: Counter[int] = Counter()
    opponent_counts: Counter[int] = Counter()
    opponent_by_serial: dict[int, int] = {}
    opponent = 1 - perspective
    start = int(batch.visible_card_offsets[row])
    stop = int(batch.visible_card_offsets[row + 1])
    for card_row in range(start, stop):
        card_id = int(batch.visible_card_id[card_row])
        if card_id <= 0:
            continue
        owner = int(batch.visible_card_owner[card_row])
        area = int(batch.visible_card_area[card_row])
        if area == _VISIBLE_CARD_VIRTUAL_AREA:
            continue
        if owner == perspective:
            own_counts[card_id] += 1
        elif owner == opponent:
            if area == _VISIBLE_CARD_HAND_AREA:
                raise NativePublicContextError(
                    "native public state exposed opponent hand identity"
                )
            if area == _VISIBLE_CARD_DECK_AREA:
                raise NativePublicContextError(
                    "native public state exposed opponent deck identity"
                )
            opponent_counts[card_id] += 1
            serial = int(batch.visible_card_serial[card_row])
            if serial > 0:
                opponent_by_serial.setdefault(serial, card_id)

    attachment_start = int(batch.attachment_offsets[row])
    attachment_stop = int(batch.attachment_offsets[row + 1])
    for attachment_row in range(attachment_start, attachment_stop):
        parent = int(batch.attachment_parent[attachment_row])
        if parent < start or parent >= stop:
            raise NativePublicContextError(
                "native attachment parent points outside its public row"
            )
        card_id = int(batch.attachment_card_id[attachment_row])
        if card_id <= 0:
            continue
        owner = int(batch.visible_card_owner[parent])
        if owner == perspective:
            own_counts[card_id] += 1
        elif owner == opponent:
            opponent_counts[card_id] += 1
            serial = int(batch.attachment_card_serial[attachment_row])
            if serial > 0:
                opponent_by_serial.setdefault(serial, card_id)
    return _VisibleEvidence(
        own_counts=own_counts,
        opponent_counts=opponent_counts,
        opponent_by_serial=opponent_by_serial,
    )


def _build_context_batch(
    states: Sequence[NativePublicHistoryState],
    visible_rows: Sequence[_VisibleEvidence],
    *,
    posterior_for_known: Callable[[_KnownIdentity], PublicDeckPosteriorArrays],
    catalog_fingerprint: str,
    input_contract_fingerprint: str,
) -> NativePublicContextBatch:
    own_unseen_rows: list[tuple[tuple[int, int], ...]] = []
    revealed_rows: list[tuple[tuple[int, int], ...]] = []
    attack_rows: list[tuple[tuple[int, int], ...]] = []
    posterior_rows: list[PublicDeckPosteriorArrays] = []
    posterior_identities: list[_KnownIdentity] = []
    history_rows: list[tuple[int, ...]] = []
    flow_rows: list[tuple[int, ...]] = []
    for state, visible in zip(states, visible_rows, strict=True):
        own_unseen_rows.append(
            _subtract_sorted_counts(
                state.own_deck_count_items,
                visible.own_counts,
            )
        )
        revealed, known_identity = _revealed_and_known_counts(
            state.opponent_revealed_counts,
            visible.opponent_counts,
        )
        revealed_rows.append(revealed)
        if state.cached_known != known_identity or state.cached_posterior is None:
            state.cached_known = known_identity
            state.cached_posterior = posterior_for_known(known_identity)
        posterior_rows.append(state.cached_posterior)
        posterior_identities.append(known_identity)
        history_rows.append(state.history_counts())
        flow_rows.append(state.deck_flow_counts())
        attack_rows.append(tuple(sorted(state.last_attack_by_serial.items())))

    own_offsets, own_ids, own_unseen_counts = _count_csr(own_unseen_rows)
    revealed_offsets, revealed_ids, revealed_counts = _count_csr(revealed_rows)
    attack_offsets, attack_serials, attack_ids = _count_csr(attack_rows)
    (
        belief_offsets,
        belief_ids,
        belief_counts,
        belief_scalars,
        belief_row_indices,
    ) = _unique_posterior_csr(
        posterior_identities,
        posterior_rows,
    )
    return NativePublicContextBatch(
        own_unseen_offsets=own_offsets,
        own_unseen_card_ids=own_ids,
        own_unseen_counts=own_unseen_counts,
        opponent_revealed_offsets=revealed_offsets,
        opponent_revealed_card_ids=revealed_ids,
        opponent_revealed_counts=revealed_counts,
        history_counts=np.asarray(history_rows, dtype=np.int32),
        deck_flow_counts=np.asarray(flow_rows, dtype=np.int32),
        last_attack_offsets=attack_offsets,
        last_attack_serials=attack_serials,
        last_attack_ids=attack_ids,
        belief_summary_offsets=belief_offsets,
        belief_summary_card_ids=belief_ids,
        belief_summary_expected_counts=belief_counts,
        belief_summary_scalars=belief_scalars,
        belief_summary_row_indices=belief_row_indices,
        own_decks=np.asarray(
            [state.own_deck.card_ids for state in states],
            dtype=np.int32,
        ),
        deck_signatures=tuple(state.own_deck.signature for state in states),
        catalog_fingerprint=catalog_fingerprint,
        input_contract_fingerprint=input_contract_fingerprint,
    )


def _subtract_sorted_counts(
    source: Sequence[tuple[int, int]],
    removed: Mapping[int, int],
) -> tuple[tuple[int, int], ...]:
    """Subtract sparse visible counts from an already sorted exact deck."""
    return tuple(
        (card_id, remaining)
        for card_id, count in source
        if (remaining := int(count) - int(removed.get(card_id, 0))) > 0
    )


def _revealed_and_known_counts(
    tracked: Mapping[int, int],
    visible: Mapping[int, int],
) -> tuple[_KnownIdentity, _KnownIdentity]:
    """Return off-board revealed counts and total public evidence together."""
    card_ids = sorted(set(tracked) | set(visible))
    revealed = tuple(
        (card_id, count)
        for card_id in card_ids
        if (count := int(tracked.get(card_id, 0)) - int(visible.get(card_id, 0)))
        > 0
    )
    known = tuple(
        (
            card_id,
            max(int(tracked.get(card_id, 0)), int(visible.get(card_id, 0))),
        )
        for card_id in card_ids
    )
    return revealed, known


def _count_csr(
    rows: Sequence[Sequence[tuple[int, int]]],
) -> tuple[Uint32Array, Int32Array, Int32Array]:
    lengths = np.fromiter(
        (len(values) for values in rows),
        dtype=np.uint64,
        count=len(rows),
    )
    wide_offsets = np.empty(len(rows) + 1, dtype=np.uint64)
    wide_offsets[0] = 0
    np.cumsum(lengths, out=wide_offsets[1:])
    total = int(wide_offsets[-1])
    if total > int(np.iinfo(np.uint32).max):
        raise OverflowError("native public context CSR exceeds uint32 capacity")
    return (
        wide_offsets.astype(np.uint32),
        np.fromiter(
            (item[0] for values in rows for item in values),
            dtype=np.int32,
            count=total,
        ),
        np.fromiter(
            (item[1] for values in rows for item in values),
            dtype=np.int32,
            count=total,
        ),
    )


def _unique_posterior_csr(
    identities: Sequence[_KnownIdentity],
    rows: Sequence[PublicDeckPosteriorArrays],
) -> tuple[
    Uint32Array,
    Int32Array,
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    Uint32Array,
]:
    """Pack each distinct posterior once and map policy rows onto it."""
    if len(identities) != len(rows) or not rows:
        raise NativePublicContextError(
            "native posterior rows and identities must be non-empty and aligned"
        )
    unique_rows: list[PublicDeckPosteriorArrays] = []
    unique_scalars: list[npt.NDArray[np.float32]] = []
    unique_by_identity: dict[_KnownIdentity, int] = {}
    inverse = np.empty(len(rows), dtype=np.uint32)
    for row, (identity, posterior) in enumerate(
        zip(identities, rows, strict=True)
    ):
        unique_index = unique_by_identity.get(identity)
        if unique_index is None:
            scalars = np.asarray(
                (
                    posterior.entropy,
                    posterior.compatible_deck_count,
                    posterior.public_evidence_count,
                    posterior.unknown_probability,
                ),
                dtype=np.float32,
            )
            unique_index = len(unique_rows)
            unique_rows.append(posterior)
            unique_scalars.append(scalars)
            unique_by_identity[identity] = unique_index
        inverse[row] = unique_index

    lengths = np.fromiter(
        (posterior.card_ids.shape[0] for posterior in unique_rows),
        dtype=np.uint32,
        count=len(unique_rows),
    )
    offsets = np.empty(len(unique_rows) + 1, dtype=np.uint32)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    if int(offsets[-1]) == 0:
        return (
            offsets,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.asarray(unique_scalars, dtype=np.float32),
            inverse,
        )
    return (
        offsets,
        np.concatenate(
            tuple(
                posterior.card_ids.astype(np.int32, copy=False)
                for posterior in unique_rows
            )
        ),
        np.concatenate(
            tuple(
                posterior.expected_counts.astype(np.float32, copy=False)
                for posterior in unique_rows
            )
        ),
        np.asarray(unique_scalars, dtype=np.float32),
        inverse,
    )


__all__ = [
    "NativeKnownOpponentBatch",
    "NativePublicContextError",
    "NativePublicContextTracker",
]
