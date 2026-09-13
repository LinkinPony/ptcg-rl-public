"""Fixed-horizon native rollout pages without Python decision objects."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ptcg_rl.decks.identity import CanonicalDeck
from ptcg_rl.engine.native_public_context import NativeKnownOpponentBatch
from ptcg_rl.model.sequence.action import AcceptedActionRecord
from ptcg_rl.rl.native_policy_batch import NativeSimpleStatelessPolicyBatch
from ptcg_rl.rl.native_policy_trace import NativePolicyNumpyTrace
from ptcg_rl.rl.native_trajectory_columns import (
    NativeDecisionChunk,
    build_native_decision_chunk,
    gather_native_decision_columns,
    native_row_references,
    referenced_chunk_ids,
)
from ptcg_rl.rl.policy_inputs import SimpleStatelessActorRow
from ptcg_rl.rl.stateless_array_replay import _validate_source_semantics
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity
from ptcg_rl.rl.stateless_fragment_io import CompactFragmentPart

Array = npt.NDArray[np.generic]
Int64Array = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FRAGMENT_ID_DOMAIN = b"ptcg-rl/stateless-fragment-id/v1\x00"
_DEFAULT_FRAGMENTS_PER_PART = 64
_IDENTITY_STRING_FIELDS = (
    ("behavior_policy_fingerprints", "behavior_policy_fingerprint"),
    ("model_config_fingerprints", "model_config_fingerprint"),
    ("action_schema_fingerprints", "action_schema_fingerprint"),
    ("public_context_fingerprints", "public_context_fingerprint"),
    ("card_catalog_fingerprints", "card_catalog_fingerprint"),
    ("public_deck_catalog_fingerprints", "public_deck_catalog_fingerprint"),
    ("exact_registry_fingerprints", "exact_registry_fingerprint"),
    (
        "belief_target_semantics_fingerprints",
        "belief_target_semantics_fingerprint",
    ),
    ("input_contract_fingerprints", "input_contract_fingerprint"),
    ("resolved_config_fingerprints", "resolved_config_fingerprint"),
)


@dataclass(frozen=True, slots=True)
class NativeTrajectoryGame:
    """Immutable learner metadata for one policy-controlled seat perspective."""

    slot: int
    game_id: str
    candidate_seat: int
    own_deck: CanonicalDeck
    opponent_deck: CanonicalDeck
    curriculum_generation: int
    assignment_id: str
    opponent_artifact_fingerprint: str
    initial_decision_index: int = 0

    def __post_init__(self) -> None:
        """Reject ambiguous game identities before rollout state is created."""
        if self.slot < 0:
            raise ValueError("native trajectory slot must be non-negative")
        if self.candidate_seat not in (0, 1):
            raise ValueError("native trajectory candidate seat must be zero or one")
        if self.curriculum_generation < 0 or self.initial_decision_index < 0:
            raise ValueError("native trajectory counters must be non-negative")
        if (
            not self.game_id
            or self.game_id.strip() != self.game_id
            or not self.assignment_id
            or self.assignment_id.strip() != self.assignment_id
        ):
            raise ValueError("native trajectory game and assignment IDs are invalid")
        if _SHA256_PATTERN.fullmatch(self.opponent_artifact_fingerprint) is None:
            raise ValueError("native trajectory opponent artifact is not SHA-256")


@dataclass(slots=True)
class _OpenUnit:
    game: NativeTrajectoryGame
    start_decision_index: int
    references: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _CompletedUnit:
    game: NativeTrajectoryGame
    start_decision_index: int
    references: Int64Array
    fragment_id: str
    terminal: bool
    truncated: bool
    bootstrap_value: float
    terminal_reward: float


@dataclass(slots=True)
class _SlotState:
    game: NativeTrajectoryGame
    next_decision_index: int
    active: _OpenUnit | None = None


class NativeTrajectoryPage:
    """Accumulate native decisions into recoverable compact fragment parts.

    Each inference batch becomes one immutable NumPy column block. Open
    fragments retain only packed ``(chunk, row)`` integers, so no
    ``StatelessFragmentDecision``, actor row, or fragment object is created.
    A full-horizon unit becomes publishable as soon as the next behavior value
    supplies its bootstrap. The collection delivery remains transactional so a
    later game cancellation can revoke already published units by game ID.
    """

    def __init__(
        self,
        identity: StatelessFragmentIdentity,
        *,
        fragments_per_part: int = _DEFAULT_FRAGMENTS_PER_PART,
    ) -> None:
        """Bind one behavior publication and an in-memory part size."""
        if fragments_per_part <= 0:
            raise ValueError("native trajectory part size must be positive")
        self.identity = identity
        self.fragments_per_part = int(fragments_per_part)
        self._slots: dict[tuple[int, int], _SlotState] = {}
        self._chunks: dict[int, NativeDecisionChunk] = {}
        self._completed: list[_CompletedUnit] = []
        self._fragment_ids: set[str] = set()
        self._next_chunk_id = 0

    @property
    def live_game_count(self) -> int:
        """Return registered games that have not reached engine terminal."""
        return len(self._slots)

    @property
    def completed_fragment_count(self) -> int:
        """Return completed fragments waiting for part publication."""
        return len(self._completed)

    @property
    def decision_chunk_count(self) -> int:
        """Return retained immutable inference blocks."""
        return len(self._chunks)

    def register_games(self, games: Sequence[NativeTrajectoryGame]) -> None:
        """Register freshly reset seat perspectives before their first decision."""
        rows = tuple(games)
        keys = tuple((game.slot, game.candidate_seat) for game in rows)
        if len(keys) != len(set(keys)):
            raise ValueError(
                "native trajectory game registration repeats a slot perspective"
            )
        occupied = sorted(set(keys).intersection(self._slots))
        if occupied:
            raise ValueError(
                f"native trajectory slot perspectives are already live: {occupied}"
            )
        for game in rows:
            self._slots[(game.slot, game.candidate_seat)] = _SlotState(
                game=game,
                next_decision_index=game.initial_decision_index,
            )

    def append_batch(
        self,
        batch: NativeSimpleStatelessPolicyBatch,
        slot_ids: npt.ArrayLike,
        known_opponents: NativeKnownOpponentBatch,
        trace: NativePolicyNumpyTrace,
        *,
        seat_ids: npt.ArrayLike,
        batch_rows: npt.ArrayLike | None = None,
        sequence_rows: Sequence[SimpleStatelessActorRow] | None = None,
        accepted_actions: Sequence[AcceptedActionRecord] | None = None,
    ) -> tuple[CompactFragmentPart, ...]:
        """Append aligned policy-perspective decisions and publish full parts.

        If an existing slot fragment already has ``identity.horizon`` rows,
        its bootstrap is this batch row's root value. The current decision then
        starts the continuation fragment.
        """
        parts, sealed = self._append_batch(
            batch,
            slot_ids,
            known_opponents,
            trace,
            seat_ids=seat_ids,
            batch_rows=batch_rows,
            sequence_rows=sequence_rows,
            accepted_actions=accepted_actions,
            seal_if_full=None,
        )
        if np.any(sealed):
            raise RuntimeError(
                "normal native trajectory append unexpectedly sealed rows"
            )
        return parts

    def append_or_seal_batch(
        self,
        batch: NativeSimpleStatelessPolicyBatch,
        slot_ids: npt.ArrayLike,
        known_opponents: NativeKnownOpponentBatch,
        trace: NativePolicyNumpyTrace,
        *,
        seat_ids: npt.ArrayLike,
        seal_if_full: npt.ArrayLike,
        batch_rows: npt.ArrayLike | None = None,
        sequence_rows: Sequence[SimpleStatelessActorRow] | None = None,
        accepted_actions: Sequence[AcceptedActionRecord] | None = None,
    ) -> tuple[tuple[CompactFragmentPart, ...], BoolArray]:
        """Append decisions or seal drain rows at their next decision.

        A row requested for sealing closes and retires its slot perspective
        immediately: its root value bootstrap-closes the open unit even when
        the unit has not reached the fixed horizon, while the sampled current
        action is deliberately excluded from trajectory storage. Rows without
        a seal request retain normal append behavior. The returned immutable
        mask is aligned to the input rows and identifies retired perspectives.
        """
        return self._append_batch(
            batch,
            slot_ids,
            known_opponents,
            trace,
            seat_ids=seat_ids,
            batch_rows=batch_rows,
            sequence_rows=sequence_rows,
            accepted_actions=accepted_actions,
            seal_if_full=seal_if_full,
        )

    def _append_batch(
        self,
        batch: NativeSimpleStatelessPolicyBatch,
        slot_ids: npt.ArrayLike,
        known_opponents: NativeKnownOpponentBatch,
        trace: NativePolicyNumpyTrace,
        *,
        seat_ids: npt.ArrayLike,
        batch_rows: npt.ArrayLike | None,
        sequence_rows: Sequence[SimpleStatelessActorRow] | None,
        accepted_actions: Sequence[AcceptedActionRecord] | None,
        seal_if_full: npt.ArrayLike | None,
    ) -> tuple[tuple[CompactFragmentPart, ...], BoolArray]:
        """Apply one validated normal or drain-aware trajectory batch."""
        slots = _slot_vector(slot_ids, rows=trace.batch_size)
        seats = _seat_vector(seat_ids, rows=trace.batch_size)
        seal_requested = (
            np.zeros(trace.batch_size, dtype=np.bool_)
            if seal_if_full is None
            else _bool_vector(
                seal_if_full,
                rows=trace.batch_size,
                name="native trajectory seal requests",
            )
        )
        selected = self._validate_append_inputs(
            batch,
            slots,
            seats,
            known_opponents,
            trace,
            batch_rows=batch_rows,
        )
        chunk = build_native_decision_chunk(
            batch,
            known_opponents,
            trace,
            batch_rows=selected,
            sequence_rows=sequence_rows,
            accepted_actions=accepted_actions,
        )
        chunk_id = self._next_chunk_id
        references = native_row_references(chunk_id, chunk.row_count)
        self._chunks[chunk_id] = chunk
        self._next_chunk_id += 1

        sealed = np.zeros(trace.batch_size, dtype=np.bool_)
        for row, (raw_slot, raw_seat, reference) in enumerate(
            zip(slots, seats, references, strict=True)
        ):
            key = (int(raw_slot), int(raw_seat))
            state = self._slots[key]
            if state.active is not None and (
                bool(seal_requested[row])
                or len(state.active.references) == self.identity.horizon
            ):
                self._complete_unit(
                    state.active,
                    terminal=False,
                    bootstrap_value=float(trace.root_values[row]),
                    terminal_reward=0.0,
                )
                state.active = None
            if bool(seal_requested[row]):
                self._slots.pop(key)
                sealed[row] = True
                continue
            if state.active is None:
                state.active = _OpenUnit(
                    game=state.game,
                    start_decision_index=state.next_decision_index,
                )
            state.active.references.append(int(reference))
            state.next_decision_index += 1
        parts = self.publish_ready()
        if np.any(sealed):
            self._release_unused_chunks()
        sealed.setflags(write=False)
        return parts, sealed

    def finish_terminals(
        self,
        slot_ids: npt.ArrayLike,
        engine_rewards: npt.ArrayLike,
        *,
        seat_ids: npt.ArrayLike,
    ) -> tuple[CompactFragmentPart, ...]:
        """Close scored endings for explicit seat perspectives."""
        slots = np.asarray(slot_ids)
        seats = _seat_vector(seat_ids, rows=int(slots.size))
        rewards = np.asarray(engine_rewards, dtype=np.float64)
        if slots.ndim != 1 or rewards.shape != slots.shape or slots.size <= 0:
            raise ValueError("native terminal slots and rewards must be aligned")
        normalized_slots = tuple(int(value) for value in slots)
        keys = tuple(
            (slot, int(seat))
            for slot, seat in zip(normalized_slots, seats, strict=True)
        )
        if len(keys) != len(set(keys)):
            raise ValueError("native terminal batch repeats a slot perspective")
        for key, reward in zip(keys, rewards, strict=True):
            state = self._slots.get(key)
            if state is None:
                raise ValueError(
                    "native terminal references an unknown slot perspective"
                )
            if float(reward) not in (-1.0, 0.0, 1.0):
                raise ValueError("native engine reward must be win/loss/draw")
        for key, reward in zip(keys, rewards, strict=True):
            state = self._slots.pop(key)
            if state.active is None:
                continue
            self._complete_unit(
                state.active,
                terminal=True,
                bootstrap_value=0.0,
                terminal_reward=float(reward),
            )
        return self.publish_ready()

    def discard_games(self, games: Sequence[NativeTrajectoryGame]) -> None:
        """Discard unpublished completed and open units for failed games.

        Complete units already materialized into parts are revoked separately
        by ``NativePartDelivery``. Keeping that rollback outside the page lets
        the normal path release referenced inference chunks immediately.
        """
        rows, keys = self._validated_registered_games(games, operation="discard")
        self._discard_completed_units(rows)
        for key in keys:
            self._slots.pop(key)
        self._release_unused_chunks()

    def discard_sealed_games(self, games: Sequence[NativeTrajectoryGame]) -> None:
        """Discard completed units after drain sealing retired their slots."""
        rows = tuple(games)
        keys = tuple((game.slot, game.candidate_seat) for game in rows)
        if not rows or len(keys) != len(set(keys)):
            raise ValueError(
                "native trajectory sealed discard requires unique slot perspectives"
            )
        if any(key in self._slots for key in keys):
            raise ValueError(
                "native trajectory sealed discard references an active perspective"
            )
        self._discard_completed_units(rows)
        self._release_unused_chunks()

    def _discard_completed_units(
        self,
        games: Sequence[NativeTrajectoryGame],
    ) -> None:
        """Remove unpublished units matching exact whole-game identities."""
        discarded_identities = {
            (
                game.slot,
                game.candidate_seat,
                game.game_id,
                game.assignment_id,
            )
            for game in games
        }
        discarded = [
            record
            for record in self._completed
            if (
                record.game.slot,
                record.game.candidate_seat,
                record.game.game_id,
                record.game.assignment_id,
            )
            in discarded_identities
        ]
        if discarded:
            discarded_ids = {record.fragment_id for record in discarded}
            self._completed[:] = [
                record
                for record in self._completed
                if record.fragment_id not in discarded_ids
            ]
            self._fragment_ids.difference_update(discarded_ids)

    def discard_open_tails(
        self,
        games: Sequence[NativeTrajectoryGame],
    ) -> tuple[int, ...]:
        """Retire games while preserving every completed horizon fragment.

        Return the number of unclosed decisions discarded for each input
        perspective, in input order. A full active unit is still unclosed until
        a later behavior value supplies its bootstrap, so it is also a tail.
        """
        _rows, keys = self._validated_registered_games(
            games,
            operation="open-tail discard",
        )
        discarded = tuple(
            len(state.active.references) if state.active is not None else 0
            for state in (self._slots[key] for key in keys)
        )
        for key in keys:
            self._slots.pop(key)
        self._release_unused_chunks()
        return discarded

    def publish_ready(
        self,
        *,
        include_partial: bool = False,
    ) -> tuple[CompactFragmentPart, ...]:
        """Materialize complete bounded in-memory fragment parts.

        ``include_partial`` emits a final shorter part made only from
        already completed fragments. It never invents a bootstrap for an open
        fragment.
        """
        parts: list[CompactFragmentPart] = []
        while len(self._completed) >= self.fragments_per_part:
            parts.append(self._publish(self.fragments_per_part))
        if include_partial and self._completed:
            parts.append(self._publish(len(self._completed)))
        return tuple(parts)

    def _validate_append_inputs(
        self,
        batch: NativeSimpleStatelessPolicyBatch,
        slots: Int64Array,
        seats: Int64Array,
        known: NativeKnownOpponentBatch,
        trace: NativePolicyNumpyTrace,
        *,
        batch_rows: npt.ArrayLike | None,
    ) -> Int64Array:
        selected = _selected_batch_rows(
            batch_rows,
            source_rows=batch.batch_size,
            output_rows=int(slots.size),
        )
        rows = int(selected.size)
        if (
            known.batch_size != rows
            or trace.batch_size != rows
            or trace.identity != self.identity
        ):
            raise ValueError("native trajectory behavior inputs are misaligned")
        if (
            batch.input_contract_fingerprint != self.identity.input_contract_fingerprint
            or batch.public_deck_catalog_fingerprint
            != self.identity.public_deck_catalog_fingerprint
            or batch.belief_summary.catalog_fingerprint
            != self.identity.public_deck_catalog_fingerprint
        ):
            raise ValueError("native trajectory policy contract differs from behavior")
        keys = tuple(
            (int(slot), int(seat)) for slot, seat in zip(slots, seats, strict=True)
        )
        if len(set(keys)) != rows:
            raise ValueError(
                "native trajectory inference batch repeats a slot perspective"
            )
        for key, batch_row in zip(keys, selected, strict=True):
            state = self._slots.get(key)
            if state is None:
                raise ValueError(
                    "native trajectory decision references an unknown slot perspective"
                )
            if batch.deck_signatures[int(batch_row)] != state.game.own_deck.signature:
                raise ValueError("native trajectory policy deck differs from game")
        return selected

    def _complete_unit(
        self,
        active: _OpenUnit,
        *,
        terminal: bool,
        bootstrap_value: float,
        terminal_reward: float,
    ) -> None:
        rows = np.asarray(active.references, dtype=np.int64)
        if rows.size <= 0 or rows.size > self.identity.horizon:
            raise RuntimeError("native trajectory fragment length is invalid")
        if not math.isfinite(bootstrap_value) or abs(bootstrap_value) > 1.0:
            raise ValueError("native trajectory bootstrap is outside [-1, 1]")
        fragment_id = _fragment_id(
            self.identity,
            active.game,
            start_decision_index=active.start_decision_index,
            decision_count=int(rows.size),
            terminal=terminal,
        )
        if fragment_id in self._fragment_ids:
            raise ValueError("native trajectory produced a duplicate fragment ID")
        self._fragment_ids.add(fragment_id)
        rows.setflags(write=False)
        record = _CompletedUnit(
            game=active.game,
            start_decision_index=active.start_decision_index,
            references=rows,
            fragment_id=fragment_id,
            terminal=terminal,
            truncated=not terminal,
            bootstrap_value=float(bootstrap_value),
            terminal_reward=float(terminal_reward),
        )
        self._completed.append(record)

    def _publish(self, count: int) -> CompactFragmentPart:
        records = tuple(self._completed[:count])
        arrays = self._part_arrays(records)
        static_fingerprints = _validate_source_semantics(arrays)
        if set(static_fingerprints) != {self.identity.static_contract_fingerprint}:
            raise RuntimeError("native trajectory part changed its static contract")
        part = CompactFragmentPart(path=None, arrays=arrays)
        del self._completed[:count]
        self._release_unused_chunks()
        return part

    def _part_arrays(
        self,
        records: Sequence[_CompletedUnit],
    ) -> dict[str, Array]:
        fragments = len(records)
        lengths = np.asarray(
            [record.references.size for record in records],
            dtype=np.int64,
        )
        fragment_offsets = np.zeros(fragments + 1, dtype=np.int64)
        fragment_offsets[1:] = np.cumsum(lengths, dtype=np.int64)
        references = np.concatenate(
            [record.references for record in records],
        )
        decisions = int(references.size)
        columns = gather_native_decision_columns(self._chunks, references)
        terminal = np.asarray(
            [record.terminal for record in records],
            dtype=np.bool_,
        )
        terminal_rewards = np.asarray(
            [record.terminal_reward for record in records],
            dtype=np.float32,
        )
        rewards = np.zeros(decisions, dtype=np.float32)
        final_rows = fragment_offsets[1:] - 1
        rewards[final_rows[terminal]] = terminal_rewards[terminal]
        starts = np.asarray(
            [record.start_decision_index for record in records],
            dtype=np.int64,
        )
        fragment_rows = np.repeat(np.arange(fragments, dtype=np.int32), lengths)
        decision_positions = np.arange(decisions, dtype=np.int64)
        decision_positions -= np.repeat(fragment_offsets[:-1], lengths)
        identity_strings = {
            output: _strings([str(getattr(self.identity, source))] * fragments)
            for output, source in _IDENTITY_STRING_FIELDS
        }
        arrays: dict[str, Array] = {
            "schema_version": np.asarray(
                [self.identity.schema_version],
                dtype=np.int16,
            ),
            "fragment_ids": _strings([record.fragment_id for record in records]),
            "fragment_decision_offsets": fragment_offsets,
            "game_ids": _strings([record.game.game_id for record in records]),
            "seats": np.asarray(
                [record.game.candidate_seat for record in records],
                dtype=np.int8,
            ),
            "start_decision_indices": starts,
            "own_decks": np.asarray(
                [record.game.own_deck.card_ids for record in records],
                dtype=np.int32,
            ),
            "opponent_decks": np.asarray(
                [record.game.opponent_deck.card_ids for record in records],
                dtype=np.int32,
            ),
            "own_deck_digests": _strings(
                [record.game.own_deck.deck_digest for record in records]
            ),
            "opponent_deck_digests": _strings(
                [record.game.opponent_deck.deck_digest for record in records]
            ),
            "curriculum_generations": np.asarray(
                [record.game.curriculum_generation for record in records],
                dtype=np.int64,
            ),
            "assignment_ids": _strings(
                [record.game.assignment_id for record in records]
            ),
            "opponent_artifact_fingerprints": _strings(
                [record.game.opponent_artifact_fingerprint for record in records]
            ),
            "terminal": terminal,
            "truncated": ~terminal,
            "bootstrap_values": np.asarray(
                [record.bootstrap_value for record in records],
                dtype=np.float32,
            ),
            "terminal_rewards": terminal_rewards,
            "horizons": np.full(
                fragments,
                self.identity.horizon,
                dtype=np.int32,
            ),
            "behavior_policy_versions": np.full(
                fragments,
                self.identity.behavior_policy_version,
                dtype=np.int64,
            ),
            **identity_strings,
            "decision_fragment_indices": fragment_rows,
            "decision_indices": starts[fragment_rows] + decision_positions,
            "rewards": rewards,
            **columns,
        }
        if self.identity.schema_version == 2:
            sequence_fingerprint = self.identity.sequence_contract_fingerprint
            if sequence_fingerprint is None:
                raise RuntimeError("native sequence identity lost its contract")
            arrays.update(
                {
                    "fragment_schema_versions": np.full(
                        fragments,
                        2,
                        dtype=np.int16,
                    ),
                    "sequence_contract_fingerprints": _strings(
                        [sequence_fingerprint] * fragments
                    ),
                }
            )
        return arrays

    def _release_unused_chunks(self) -> None:
        references = [record.references for record in self._completed] + [
            np.asarray(state.active.references, dtype=np.int64)
            for state in self._slots.values()
            if state.active is not None and state.active.references
        ]
        retained = referenced_chunk_ids(references)
        for chunk_id in tuple(self._chunks):
            if chunk_id not in retained:
                del self._chunks[chunk_id]

    def _validated_registered_games(
        self,
        games: Sequence[NativeTrajectoryGame],
        *,
        operation: str,
    ) -> tuple[
        tuple[NativeTrajectoryGame, ...],
        tuple[tuple[int, int], ...],
    ]:
        """Validate unique registered perspectives without mutating the page."""
        rows = tuple(games)
        keys = tuple((game.slot, game.candidate_seat) for game in rows)
        if not rows or len(keys) != len(set(keys)):
            raise ValueError(
                f"native trajectory {operation} requires unique slot perspectives"
            )
        for key, game in zip(keys, rows, strict=True):
            state = self._slots.get(key)
            if state is None:
                raise ValueError(
                    "native trajectory "
                    f"{operation} references an unknown slot perspective"
                )
            if (
                state.game.game_id != game.game_id
                or state.game.assignment_id != game.assignment_id
            ):
                raise ValueError(f"native trajectory {operation} changed game identity")
        return rows, keys


def _slot_vector(values: npt.ArrayLike, *, rows: int) -> Int64Array:
    slots = np.asarray(values)
    if slots.ndim != 1 or slots.shape != (rows,):
        raise ValueError("native trajectory slots must align with policy rows")
    if not np.issubdtype(slots.dtype, np.integer):
        raise TypeError("native trajectory slots must use an integer dtype")
    normalized = slots.astype(np.int64, copy=False)
    if np.any(normalized < 0):
        raise ValueError("native trajectory slots must be non-negative")
    return normalized


def _seat_vector(values: npt.ArrayLike, *, rows: int) -> Int64Array:
    seats = np.asarray(values)
    if seats.ndim != 1 or seats.shape != (rows,):
        raise ValueError("native trajectory seats must align with policy rows")
    if not np.issubdtype(seats.dtype, np.integer):
        raise TypeError("native trajectory seats must use an integer dtype")
    normalized = seats.astype(np.int64, copy=False)
    if np.any((normalized != 0) & (normalized != 1)):
        raise ValueError("native trajectory seats must be zero or one")
    return normalized


def _bool_vector(
    values: npt.ArrayLike,
    *,
    rows: int,
    name: str,
) -> BoolArray:
    result = np.asarray(values)
    if result.ndim != 1 or result.shape != (rows,):
        raise ValueError(f"{name} must align with policy rows")
    if result.dtype != np.bool_:
        raise TypeError(f"{name} must use bool")
    return result


def _selected_batch_rows(
    values: npt.ArrayLike | None,
    *,
    source_rows: int,
    output_rows: int,
) -> Int64Array:
    selected = (
        np.arange(source_rows, dtype=np.int64) if values is None else np.asarray(values)
    )
    if selected.ndim != 1 or selected.shape != (output_rows,):
        raise ValueError("native trajectory batch selection is misaligned")
    if not np.issubdtype(selected.dtype, np.integer):
        raise TypeError("native trajectory batch selection must use integers")
    normalized = selected.astype(np.int64, copy=False)
    if (
        np.any(normalized < 0)
        or np.any(normalized >= source_rows)
        or np.unique(normalized).size != normalized.size
    ):
        raise ValueError("native trajectory batch selection is invalid")
    return normalized


def _fragment_id(
    identity: StatelessFragmentIdentity,
    game: NativeTrajectoryGame,
    *,
    start_decision_index: int,
    decision_count: int,
    terminal: bool,
) -> str:
    payload = {
        "identity": identity.fingerprint,
        "game_id": game.game_id,
        "seat": game.candidate_seat,
        "start_decision_index": start_decision_index,
        "decisions": decision_count,
        "terminal": terminal,
        "truncated": not terminal,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(_FRAGMENT_ID_DOMAIN + encoded).hexdigest()


def _strings(values: Sequence[str]) -> npt.NDArray[np.str_]:
    width = max(1, *(len(value) for value in values))
    return np.asarray(values, dtype=f"U{width}")


__all__ = [
    "NativeTrajectoryGame",
    "NativeTrajectoryPage",
]
