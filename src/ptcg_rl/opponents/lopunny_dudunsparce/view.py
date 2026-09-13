"""Public-observation helpers for the Lopunny/Dudunsparce pilot."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

from ptcg_rl.engine.constants import AreaType, OptionType


def as_sequence(value: Any) -> Sequence[Any]:
    """Return a non-string sequence, or an empty tuple."""
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def player(observation: Any, player_index: int) -> Any | None:
    """Return one visible player state."""
    state = getattr(observation, "current", None)
    players = as_sequence(getattr(state, "players", ()))
    if 0 <= player_index < len(players):
        return players[player_index]
    return None


def opponent_index(observation: Any, player_index: int) -> int:
    """Return the other seat in the two-player engine."""
    players = as_sequence(getattr(getattr(observation, "current", None), "players", ()))
    for index in range(len(players)):
        if index != player_index:
            return index
    return 1 - player_index


def active(observation: Any, player_index: int) -> Any | None:
    """Return one player's Active Pokemon."""
    active_cards = as_sequence(getattr(player(observation, player_index), "active", ()))
    return active_cards[0] if active_cards and active_cards[0] is not None else None


def bench(observation: Any, player_index: int) -> tuple[Any, ...]:
    """Return one player's visible Bench Pokemon."""
    return tuple(
        card
        for card in as_sequence(getattr(player(observation, player_index), "bench", ()))
        if card is not None
    )


def hand(observation: Any, player_index: int) -> tuple[Any, ...]:
    """Return the acting player's visible hand."""
    return tuple(
        card
        for card in as_sequence(getattr(player(observation, player_index), "hand", ()))
        if card is not None
    )


def discard(observation: Any, player_index: int) -> tuple[Any, ...]:
    """Return one player's visible discard pile."""
    return tuple(
        card
        for card in as_sequence(
            getattr(player(observation, player_index), "discard", ())
        )
        if card is not None
    )


def field_pokemon(observation: Any, player_index: int) -> tuple[Any, ...]:
    """Return Active followed by Bench Pokemon."""
    active_card = active(observation, player_index)
    return ((active_card,) if active_card is not None else ()) + bench(
        observation, player_index
    )


def card_id(card: Any) -> int | None:
    """Return a visible card ID."""
    value = getattr(card, "id", None)
    return integer(value) if value is not None else None


def card_ids(values: Sequence[Any]) -> tuple[int, ...]:
    """Return visible card IDs, omitting unknown entries."""
    return tuple(value for card in values if (value := card_id(card)) is not None)


def serial(pokemon: Any) -> int | None:
    """Return the engine instance identity for an in-play Pokemon."""
    value = getattr(pokemon, "serial", None)
    return integer(value) if value is not None else None


def energy_cards(pokemon: Any) -> tuple[Any, ...]:
    """Return physical Energy cards attached to a Pokemon."""
    return tuple(as_sequence(getattr(pokemon, "energyCards", ())))


def energy_count(pokemon: Any) -> int:
    """Return engine-resolved attached Colorless energy units."""
    return len(as_sequence(getattr(pokemon, "energies", ())))


def tools(pokemon: Any) -> tuple[Any, ...]:
    """Return Tools attached to a Pokemon."""
    return tuple(as_sequence(getattr(pokemon, "tools", ())))


def damage(pokemon: Any) -> int:
    """Return visible accumulated damage."""
    hp = integer(getattr(pokemon, "hp", 0), 0)
    maximum = integer(getattr(pokemon, "maxHp", hp), hp)
    return max(0, maximum - hp)


def remaining_hp(pokemon: Any) -> int:
    """Return visible remaining HP."""
    return max(0, integer(getattr(pokemon, "hp", 0), 0))


def appeared_this_turn(pokemon: Any) -> bool:
    """Return the evolution-lock flag exposed by the engine."""
    return bool(getattr(pokemon, "appearThisTurn", False))


def field_count(observation: Any, player_index: int, wanted: int) -> int:
    """Count one Pokemon identity in play."""
    return sum(
        card_id(card) == wanted for card in field_pokemon(observation, player_index)
    )


def hand_count(observation: Any, player_index: int, wanted: int) -> int:
    """Count one card identity in the visible hand."""
    return sum(card_id(card) == wanted for card in hand(observation, player_index))


def bench_space(observation: Any, player_index: int) -> int:
    """Return available Bench slots."""
    owner = player(observation, player_index)
    maximum = integer(getattr(owner, "benchMax", 5), 5)
    return max(0, maximum - len(bench(observation, player_index)))


def prize_count(observation: Any, player_index: int) -> int:
    """Return the number of Prize cards remaining."""
    return len(as_sequence(getattr(player(observation, player_index), "prize", ())))


def deck_count(observation: Any, player_index: int) -> int:
    """Return the public deck count."""
    return integer(getattr(player(observation, player_index), "deckCount", 0), 0)


def effect_card_id(observation: Any) -> int | None:
    """Return the card that owns the current follow-up prompt."""
    select = getattr(observation, "select", None)
    for attribute in ("effect", "contextCard"):
        value = card_id(getattr(select, attribute, None))
        if value is not None:
            return value
    return None


def option_card_id(observation: Any, player_index: int, option: Any) -> int | None:
    """Resolve the visible card referenced by an engine option."""
    raw_card_id = getattr(option, "cardId", None)
    if raw_card_id:
        return integer(raw_card_id)

    option_type = integer(getattr(option, "type", None))
    index = integer(getattr(option, "index", None))
    owner = integer(getattr(option, "playerIndex", None), player_index)
    if option_type == int(OptionType.PLAY) and index is not None:
        hand_cards = hand(observation, player_index)
        return card_id(hand_cards[index]) if 0 <= index < len(hand_cards) else None

    area = integer(getattr(option, "area", None))
    if area is None or index is None:
        return None
    resolved_cards: Sequence[Any]
    if area == int(AreaType.DECK):
        select = getattr(observation, "select", None)
        resolved_cards = as_sequence(getattr(select, "deck", ()))
    elif area == int(AreaType.HAND):
        resolved_cards = hand(observation, owner)
    elif area == int(AreaType.DISCARD):
        resolved_cards = discard(observation, owner)
    elif area == int(AreaType.LOOKING):
        state = getattr(observation, "current", None)
        resolved_cards = as_sequence(getattr(state, "looking", ()))
    elif area == int(AreaType.STADIUM):
        state = getattr(observation, "current", None)
        resolved_cards = as_sequence(getattr(state, "stadium", ()))
    elif area == int(AreaType.ACTIVE):
        return card_id(active(observation, owner))
    elif area == int(AreaType.BENCH):
        resolved_cards = bench(observation, owner)
    else:
        return None
    return card_id(resolved_cards[index]) if 0 <= index < len(resolved_cards) else None


def option_pokemon(observation: Any, player_index: int, option: Any) -> Any | None:
    """Resolve the in-play Pokemon targeted by an engine option."""
    area = integer(getattr(option, "inPlayArea", None))
    index = integer(getattr(option, "inPlayIndex", None))
    if area is None:
        area = integer(getattr(option, "area", None))
        index = integer(getattr(option, "index", None))
    owner = integer(getattr(option, "playerIndex", None), player_index)
    if area == int(AreaType.ACTIVE):
        return active(observation, owner)
    if area == int(AreaType.BENCH) and index is not None:
        cards = bench(observation, owner)
        return cards[index] if 0 <= index < len(cards) else None
    return None


@overload
def integer(value: Any, default: int) -> int: ...


@overload
def integer(value: Any, default: None = None) -> int | None: ...


def integer(value: Any, default: int | None = None) -> int | None:
    """Coerce an engine scalar to int without raising."""
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
