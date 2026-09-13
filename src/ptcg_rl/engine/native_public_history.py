"""Incremental history state over canonical native public-log columns."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ptcg_rl.belief.public_catalog import PublicDeckPosteriorArrays
from ptcg_rl.decks.identity import CanonicalDeck
from ptcg_rl.engine.constants import AreaType, LogType

if TYPE_CHECKING:
    from ptcg_rl.engine.native_training import NativeTrainingBatchView

SUCCESS_STATUSES = frozenset((1, 2))
LOG_PARAM_WIDTH = 7
EXPECTED_LOG_PARAM_COUNTS: tuple[int, ...] = (
    1,  # Shuffle
    2,  # HasBasicPokemon
    1,  # TurnStart
    1,  # TurnEnd
    3,  # Draw
    1,  # DrawReverse
    5,  # MoveCard
    3,  # MoveCardReverse
    5,  # Switch
    5,  # Change
    3,  # Play
    5,  # Attach
    5,  # Evolve
    5,  # Devolve
    7,  # MoveAttached
    4,  # Attack
    5,  # HpChange
    4,  # Poisoned
    4,  # Burned
    4,  # Asleep
    4,  # Paralyzed
    4,  # Confused
    2,  # Coin
    2,  # Result
)

_HISTORY_KIND_COUNT = 4
_ATTACK_COUNTER = 0
_SUPPORTER_COUNTER = 1
_ENERGY_ATTACH_COUNTER = 2
_RETREAT_COUNTER = 3
_LOW_DECK_REBOUND_MAX_COUNT = 5
_HIDDEN_LOG_TYPES = frozenset(
    (int(LogType.DRAW_REVERSE), int(LogType.MOVE_CARD_REVERSE))
)


@dataclass
class NativePublicHistoryState:
    """Mutable public policy state for one ``(slot, perspective)``."""

    perspective: int
    own_deck: CanonicalDeck
    supporter_card_ids: frozenset[int]
    opponent_revealed_by_serial: dict[int, int] = field(default_factory=dict)
    opponent_revealed_no_serial_counts: Counter[int] = field(
        default_factory=Counter
    )
    own_deck_count_items: tuple[tuple[int, int], ...] = ()
    opponent_revealed_counts: Counter[int] = field(default_factory=Counter)
    history_by_player: list[list[int]] = field(
        default_factory=lambda: [[0] * _HISTORY_KIND_COUNT for _ in range(2)]
    )
    draw_counts_by_player: list[int] = field(default_factory=lambda: [0, 0])
    return_to_deck_counts_by_player: list[int] = field(
        default_factory=lambda: [0, 0]
    )
    recent_turn_draw_counts_by_player: list[int] = field(
        default_factory=lambda: [0, 0]
    )
    recent_turn_return_counts_by_player: list[int] = field(
        default_factory=lambda: [0, 0]
    )
    recent_turn_deck_deltas_by_player: list[int] = field(
        default_factory=lambda: [0, 0]
    )
    deck_rebound_counts_by_player: list[int] = field(default_factory=lambda: [0, 0])
    no_attack_turns_by_player: list[int] = field(default_factory=lambda: [0, 0])
    current_turn_draw_counts_by_player: list[int] = field(
        default_factory=lambda: [0, 0]
    )
    current_turn_return_counts_by_player: list[int] = field(
        default_factory=lambda: [0, 0]
    )
    current_turn_deck_deltas_by_player: list[int] = field(
        default_factory=lambda: [0, 0]
    )
    turn_open_by_player: list[bool] = field(default_factory=lambda: [False, False])
    turn_attacked_by_player: list[bool] = field(
        default_factory=lambda: [False, False]
    )
    turn_rebounded_by_player: list[bool] = field(
        default_factory=lambda: [False, False]
    )
    observed_deck_counts_by_player: list[int | None] = field(
        default_factory=lambda: [None, None]
    )
    last_attack_by_serial: dict[int, int] = field(default_factory=dict)
    cached_known: tuple[tuple[int, int], ...] | None = None
    cached_posterior: PublicDeckPosteriorArrays | None = None

    def __post_init__(self) -> None:
        """Precompute immutable and incrementally maintained sparse counts."""
        if not self.own_deck_count_items:
            self.own_deck_count_items = tuple(
                sorted(Counter(self.own_deck.card_ids).items())
            )
        if (
            not self.opponent_revealed_counts
            and (
                self.opponent_revealed_by_serial
                or self.opponent_revealed_no_serial_counts
            )
        ):
            self.opponent_revealed_counts.update(
                self.opponent_revealed_by_serial.values()
            )
            self.opponent_revealed_counts.update(
                self.opponent_revealed_no_serial_counts
            )

    def clone(self) -> NativePublicHistoryState:
        """Copy one seat before applying an all-or-nothing batch update."""
        return NativePublicHistoryState(
            perspective=self.perspective,
            own_deck=self.own_deck,
            supporter_card_ids=self.supporter_card_ids,
            opponent_revealed_by_serial=dict(self.opponent_revealed_by_serial),
            opponent_revealed_no_serial_counts=Counter(
                self.opponent_revealed_no_serial_counts
            ),
            own_deck_count_items=self.own_deck_count_items,
            opponent_revealed_counts=Counter(self.opponent_revealed_counts),
            history_by_player=[
                list(self.history_by_player[0]),
                list(self.history_by_player[1]),
            ],
            draw_counts_by_player=list(self.draw_counts_by_player),
            return_to_deck_counts_by_player=list(
                self.return_to_deck_counts_by_player
            ),
            recent_turn_draw_counts_by_player=list(
                self.recent_turn_draw_counts_by_player
            ),
            recent_turn_return_counts_by_player=list(
                self.recent_turn_return_counts_by_player
            ),
            recent_turn_deck_deltas_by_player=list(
                self.recent_turn_deck_deltas_by_player
            ),
            deck_rebound_counts_by_player=list(self.deck_rebound_counts_by_player),
            no_attack_turns_by_player=list(self.no_attack_turns_by_player),
            current_turn_draw_counts_by_player=list(
                self.current_turn_draw_counts_by_player
            ),
            current_turn_return_counts_by_player=list(
                self.current_turn_return_counts_by_player
            ),
            current_turn_deck_deltas_by_player=list(
                self.current_turn_deck_deltas_by_player
            ),
            turn_open_by_player=list(self.turn_open_by_player),
            turn_attacked_by_player=list(self.turn_attacked_by_player),
            turn_rebounded_by_player=list(self.turn_rebounded_by_player),
            observed_deck_counts_by_player=list(
                self.observed_deck_counts_by_player
            ),
            last_attack_by_serial=dict(self.last_attack_by_serial),
            cached_known=self.cached_known,
            cached_posterior=self.cached_posterior,
        )

    def record_current_opponent(self, visible_by_serial: dict[int, int]) -> None:
        """Accumulate newly visible physical opponent identities."""
        for serial, card_id in visible_by_serial.items():
            if serial not in self.opponent_revealed_by_serial:
                self.opponent_revealed_by_serial[serial] = card_id
                self.opponent_revealed_counts[card_id] += 1

    def apply_log_delta(
        self,
        batch: NativeTrainingBatchView,
        row: int,
        *,
        log_players: npt.NDArray[np.int32],
        deck_deltas: npt.NDArray[np.int8],
    ) -> None:
        """Apply one already privacy-projected native log CSR row."""
        start = int(batch.log_offsets[row])
        stop = int(batch.log_offsets[row + 1])
        observed_counts = (
            int(batch.player_deck_counts[0][row]),
            int(batch.player_deck_counts[1][row]),
        )
        running_counts = list(self.observed_deck_counts_by_player)
        row_players = log_players[start:stop]
        row_deltas = deck_deltas[start:stop]
        for candidate_player in range(2):
            net_delta = int(
                row_deltas[row_players == candidate_player].sum(dtype=np.int64)
            )
            running_counts[candidate_player] = max(
                0,
                observed_counts[candidate_player] - net_delta,
            )

        opponent = 1 - self.perspective
        for log_index in range(start, stop):
            raw_player = int(log_players[log_index])
            player = raw_player if raw_player in (0, 1) else None
            if player in (0, 1):
                deck_before = running_counts[player]
                deck_delta = int(deck_deltas[log_index])
                _update_deck_flow(
                    batch,
                    log_index,
                    self,
                    player=player,
                    deck_count_before=deck_before,
                    deck_delta=deck_delta,
                )
                if deck_before is not None:
                    running_counts[player] = max(
                        0,
                        deck_before + deck_delta,
                    )
                _update_history(batch, log_index, self, player=player)
            if player == opponent:
                _record_revealed_log(batch, log_index, self)
        self.observed_deck_counts_by_player = [
            max(0, observed_counts[0]),
            max(0, observed_counts[1]),
        ]

    def history_counts(self) -> tuple[int, ...]:
        """Return self/opponent history in the canonical model order."""
        own = self.perspective
        opponent = 1 - own
        return tuple(
            int(value)
            for value in (
                *self.history_by_player[own],
                *self.history_by_player[opponent],
            )
        )

    def deck_flow_counts(self) -> tuple[int, ...]:
        """Return self/opponent deck circulation in the canonical model order."""
        values: list[int] = []
        for player in (self.perspective, 1 - self.perspective):
            if self.turn_open_by_player[player]:
                recent_draws = self.current_turn_draw_counts_by_player[player]
                recent_returns = self.current_turn_return_counts_by_player[player]
                recent_delta = self.current_turn_deck_deltas_by_player[player]
            else:
                recent_draws = self.recent_turn_draw_counts_by_player[player]
                recent_returns = self.recent_turn_return_counts_by_player[player]
                recent_delta = self.recent_turn_deck_deltas_by_player[player]
            values.extend(
                (
                    self.draw_counts_by_player[player],
                    self.return_to_deck_counts_by_player[player],
                    recent_draws,
                    recent_returns,
                    recent_delta,
                    self.deck_rebound_counts_by_player[player],
                    self.no_attack_turns_by_player[player],
                )
            )
        return tuple(int(value) for value in values)


def _update_history(
    batch: NativeTrainingBatchView,
    log_index: int,
    state: NativePublicHistoryState,
    *,
    player: int,
) -> None:
    log_type = int(batch.log_type[log_index])
    if log_type == int(LogType.ATTACK):
        state.history_by_player[player][_ATTACK_COUNTER] += 1
        state.turn_attacked_by_player[player] = True
        serial = int(batch.log_params[2][log_index])
        attack_id = int(batch.log_params[3][log_index])
        if serial > 0 and attack_id > 0:
            state.last_attack_by_serial[serial] = attack_id
    elif log_type == int(LogType.PLAY):
        card_id = int(batch.log_params[1][log_index])
        if card_id in state.supporter_card_ids:
            state.history_by_player[player][_SUPPORTER_COUNTER] += 1
    elif log_type == int(LogType.ATTACH):
        state.history_by_player[player][_ENERGY_ATTACH_COUNTER] += 1
    elif log_type == int(LogType.SWITCH):
        # LogJson carries no explicit cause. This deliberately matches
        # GameContext's canonical fallback, where every Switch is a retreat.
        state.history_by_player[player][_RETREAT_COUNTER] += 1


def _update_deck_flow(
    batch: NativeTrainingBatchView,
    log_index: int,
    state: NativePublicHistoryState,
    *,
    player: int,
    deck_count_before: int | None,
    deck_delta: int,
) -> None:
    log_type = int(batch.log_type[log_index])
    if log_type == int(LogType.TURN_START):
        state.current_turn_draw_counts_by_player[player] = 0
        state.current_turn_return_counts_by_player[player] = 0
        state.current_turn_deck_deltas_by_player[player] = 0
        state.turn_open_by_player[player] = True
        state.turn_attacked_by_player[player] = False
        state.turn_rebounded_by_player[player] = False
        return
    if log_type == int(LogType.TURN_END):
        if state.turn_open_by_player[player]:
            state.recent_turn_draw_counts_by_player[player] = (
                state.current_turn_draw_counts_by_player[player]
            )
            state.recent_turn_return_counts_by_player[player] = (
                state.current_turn_return_counts_by_player[player]
            )
            state.recent_turn_deck_deltas_by_player[player] = (
                state.current_turn_deck_deltas_by_player[player]
            )
            if state.turn_attacked_by_player[player]:
                state.no_attack_turns_by_player[player] = 0
            else:
                state.no_attack_turns_by_player[player] += 1
            state.turn_open_by_player[player] = False
        return
    if log_type in (int(LogType.DRAW), int(LogType.DRAW_REVERSE)):
        state.draw_counts_by_player[player] += 1
        if state.turn_open_by_player[player]:
            state.current_turn_draw_counts_by_player[player] += 1
            state.current_turn_deck_deltas_by_player[player] -= 1
        return
    if log_type not in (int(LogType.MOVE_CARD), int(LogType.MOVE_CARD_REVERSE)):
        return
    if log_type == int(LogType.MOVE_CARD):
        from_area = int(batch.log_params[3][log_index])
        to_area = int(batch.log_params[4][log_index])
    else:
        from_area = int(batch.log_params[1][log_index])
        to_area = int(batch.log_params[2][log_index])
    if state.turn_open_by_player[player]:
        if from_area == int(AreaType.DECK):
            state.current_turn_deck_deltas_by_player[player] -= 1
        if to_area == int(AreaType.DECK):
            state.current_turn_deck_deltas_by_player[player] += 1
    if to_area != int(AreaType.DECK):
        return
    state.return_to_deck_counts_by_player[player] += 1
    if state.turn_open_by_player[player]:
        state.current_turn_return_counts_by_player[player] += 1
        if (
            deck_count_before is not None
            and deck_count_before <= _LOW_DECK_REBOUND_MAX_COUNT
            and deck_delta > 0
            and not state.turn_rebounded_by_player[player]
        ):
            state.deck_rebound_counts_by_player[player] += 1
            state.turn_rebounded_by_player[player] = True


def _record_revealed_log(
    batch: NativeTrainingBatchView,
    log_index: int,
    state: NativePublicHistoryState,
) -> None:
    log_type = int(batch.log_type[log_index])
    if log_type in _HIDDEN_LOG_TYPES:
        return
    for card_id, serial in _revealed_identity_pairs(
        batch,
        log_index,
        log_type=log_type,
    ):
        if card_id <= 0:
            continue
        if serial is None:
            state.opponent_revealed_no_serial_counts[card_id] += 1
            state.opponent_revealed_counts[card_id] += 1
        else:
            if serial not in state.opponent_revealed_by_serial:
                state.opponent_revealed_by_serial[serial] = card_id
                state.opponent_revealed_counts[card_id] += 1


def _revealed_identity_pairs(
    batch: NativeTrainingBatchView,
    log_index: int,
    *,
    log_type: int,
) -> tuple[tuple[int, int | None], ...]:
    params = batch.log_params
    if log_type in (
        int(LogType.DRAW),
        int(LogType.MOVE_CARD),
        int(LogType.PLAY),
        int(LogType.ATTACK),
        int(LogType.HP_CHANGE),
    ):
        return ((int(params[1][log_index]), int(params[2][log_index])),)
    if log_type in (int(LogType.SWITCH), int(LogType.CHANGE)):
        return (
            (int(params[1][log_index]), int(params[2][log_index])),
            (int(params[3][log_index]), int(params[4][log_index])),
        )
    if log_type in (
        int(LogType.ATTACH),
        int(LogType.EVOLVE),
        int(LogType.DEVOLVE),
    ):
        return (
            (int(params[1][log_index]), int(params[2][log_index])),
            (int(params[3][log_index]), int(params[4][log_index])),
        )
    if log_type == int(LogType.MOVE_ATTACHED):
        return (
            (int(params[1][log_index]), int(params[2][log_index])),
            (int(params[3][log_index]), int(params[4][log_index])),
            (int(params[5][log_index]), int(params[6][log_index])),
        )
    if log_type in (
        int(LogType.POISONED),
        int(LogType.BURNED),
        int(LogType.ASLEEP),
        int(LogType.PARALYZED),
        int(LogType.CONFUSED),
    ):
        return ((int(params[2][log_index]), int(params[3][log_index])),)
    return ()


__all__ = [
    "EXPECTED_LOG_PARAM_COUNTS",
    "LOG_PARAM_WIDTH",
    "SUCCESS_STATUSES",
    "NativePublicHistoryState",
]
