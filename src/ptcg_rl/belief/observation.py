"""Extract hidden-information evidence from engine observations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ptcg_rl.engine.constants import LogType
from ptcg_rl.engine.protocols import ObservationInput
from ptcg_rl.engine.runtime import to_engine_observation


@dataclass
class ObservationEvidence:
    """Public constraints relevant to one hidden-state determinization."""

    your_index: int
    opponent_index: int
    your_deck_count: int
    your_prize_count: int
    opponent_deck_count: int
    opponent_prize_count: int
    opponent_hand_count: int
    opponent_active_facedown: bool
    setup_requires_basic_in_opponent_deck: bool
    search_ignores_your_deck: bool
    your_non_deck_visible_counts: Counter[int] = field(default_factory=Counter)
    your_visible_deck_counts: Counter[int] = field(default_factory=Counter)
    opponent_current_visible_counts: Counter[int] = field(default_factory=Counter)
    opponent_revealed_by_serial: dict[int, int] = field(default_factory=dict)
    opponent_revealed_no_serial_counts: Counter[int] = field(default_factory=Counter)

    @property
    def your_visible_counts(self) -> Counter[int]:
        """Return visible own-card counts across non-deck and visible deck zones."""
        counts = Counter(self.your_non_deck_visible_counts)
        counts.update(self.your_visible_deck_counts)
        return counts

    @property
    def opponent_revealed_counts(self) -> Counter[int]:
        """Return lower-bound counts for opponent cards revealed so far."""
        counts: Counter[int] = Counter(self.opponent_revealed_by_serial.values())
        counts.update(self.opponent_revealed_no_serial_counts)
        return counts


def extract_observation_evidence(observation: ObservationInput) -> ObservationEvidence:
    """Extract public hidden-information constraints from an observation."""
    obs = (
        observation
        if isinstance(observation, Mapping)
        else to_engine_observation(observation)
    )
    state = _field(obs, "current")
    if state is None:
        raise ValueError("cannot build belief evidence without current state")
    your_index = _int_field(state, "yourIndex", 0)
    opponent_index = 1 - your_index
    players = _sequence(_field(state, "players", ()))
    your_state = players[your_index]
    opponent_state = players[opponent_index]
    opponent_active = _sequence(_field(opponent_state, "active", ()))
    opponent_active_facedown = bool(opponent_active and opponent_active[0] is None)

    your_non_deck_visible, your_serials = _current_player_visible_cards(
        state,
        your_index,
        include_hand=True,
    )
    opponent_visible, opponent_serials = _current_player_visible_cards(
        state,
        opponent_index,
        include_hand=False,
    )
    select = _field(obs, "select")
    your_visible_deck_counts = Counter(
        _int_field(card, "id", 0)
        for card in _sequence(_field(select, "deck", ()))
        if _int_field(card, "playerIndex", -1) == your_index
    )

    revealed_by_serial = dict(opponent_serials)
    revealed_no_serial: Counter[int] = Counter()
    _add_revealed_logs(
        _sequence(_field(obs, "logs", ())),
        opponent_index=opponent_index,
        revealed_by_serial=revealed_by_serial,
        revealed_no_serial=revealed_no_serial,
    )

    del your_serials
    return ObservationEvidence(
        your_index=your_index,
        opponent_index=opponent_index,
        your_deck_count=_int_field(your_state, "deckCount", 0),
        your_prize_count=len(_sequence(_field(your_state, "prize", ()))),
        opponent_deck_count=_int_field(opponent_state, "deckCount", 0),
        opponent_prize_count=len(_sequence(_field(opponent_state, "prize", ()))),
        opponent_hand_count=_int_field(opponent_state, "handCount", 0),
        opponent_active_facedown=opponent_active_facedown,
        setup_requires_basic_in_opponent_deck=(
            _int_field(state, "turn", 0) == 0 or opponent_active_facedown
        ),
        search_ignores_your_deck=bool(
            select is not None and _field(select, "deck") is not None
        ),
        your_non_deck_visible_counts=your_non_deck_visible,
        your_visible_deck_counts=your_visible_deck_counts,
        opponent_current_visible_counts=opponent_visible,
        opponent_revealed_by_serial=revealed_by_serial,
        opponent_revealed_no_serial_counts=revealed_no_serial,
    )


def _current_player_visible_cards(
    state: Any,
    player_index: int,
    *,
    include_hand: bool,
) -> tuple[Counter[int], dict[int, int]]:
    counts: Counter[int] = Counter()
    serials: dict[int, int] = {}
    players = _sequence(_field(state, "players", ()))
    player = players[player_index]
    for pokemon in _sequence(_field(player, "active", ())):
        if pokemon is not None:
            _add_pokemon(pokemon, counts, serials)
    for pokemon in _sequence(_field(player, "bench", ())):
        _add_pokemon(pokemon, counts, serials)
    for discard_card in _sequence(_field(player, "discard", ())):
        _add_card(discard_card, counts, serials)
    for prize_card in _sequence(_field(player, "prize", ())):
        if prize_card is not None:
            _add_card(prize_card, counts, serials)
    hand = _field(player, "hand")
    if include_hand and hand is not None:
        for hand_card in _sequence(hand):
            _add_card(hand_card, counts, serials)
    for stadium_card in _sequence(_field(state, "stadium", ())):
        if _int_field(stadium_card, "playerIndex", -1) == player_index:
            _add_card(stadium_card, counts, serials)
    looking = _field(state, "looking")
    if looking is not None:
        for looking_card in _sequence(looking):
            if (
                looking_card is not None
                and _int_field(looking_card, "playerIndex", -1) == player_index
            ):
                _add_card(looking_card, counts, serials)
    return counts, serials


def _add_pokemon(
    pokemon: Any,
    counts: Counter[int],
    serials: dict[int, int],
) -> None:
    _add_card(pokemon, counts, serials)
    for field_name in ("energyCards", "tools", "preEvolution"):
        for card in _sequence(_field(pokemon, field_name, ())):
            if card is not None:
                _add_card(card, counts, serials)


def _add_card(
    card: Any,
    counts: Counter[int],
    serials: dict[int, int],
) -> None:
    card_id = _int_field(card, "id", 0)
    counts[card_id] += 1
    serials.setdefault(_int_field(card, "serial", 0), card_id)


_HIDDEN_LOG_TYPES = {int(LogType.DRAW_REVERSE), int(LogType.MOVE_CARD_REVERSE)}
_CARD_ID_SERIAL_FIELDS = (
    ("cardId", "serial"),
    ("cardIdActive", "serialActive"),
    ("cardIdBench", "serialBench"),
    ("cardIdBefore", "serialBefore"),
    ("cardIdAfter", "serialAfter"),
    ("cardIdTarget", "serialTarget"),
)


def _add_revealed_logs(
    logs: Sequence[Any],
    *,
    opponent_index: int,
    revealed_by_serial: dict[int, int],
    revealed_no_serial: Counter[int],
) -> None:
    for log in logs:
        if not _reveals_player_card(log, opponent_index):
            continue
        for card_field, serial_field in _CARD_ID_SERIAL_FIELDS:
            card_id = _field(log, card_field)
            if card_id is None:
                continue
            serial = _field(log, serial_field)
            if serial is None:
                revealed_no_serial[int(card_id)] += 1
            else:
                revealed_by_serial.setdefault(int(serial), int(card_id))


def _reveals_player_card(log: Any, opponent_index: int) -> bool:
    if _int_field(log, "type", -1) in _HIDDEN_LOG_TYPES:
        return False
    player_index = _field(log, "playerIndex")
    return player_index is not None and int(player_index) == opponent_index


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_field(value: Any, name: str, default: int) -> int:
    field_value = _field(value, name, default)
    return int(field_value) if field_value is not None else default
