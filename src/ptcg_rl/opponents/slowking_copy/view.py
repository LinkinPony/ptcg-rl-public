"""Public-observation helpers for the Slowking Copy Engine opponent."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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


def active(observation: Any, player_index: int) -> Any | None:
    """Return one player's Active Pokémon."""
    active_cards = as_sequence(getattr(player(observation, player_index), "active", ()))
    return active_cards[0] if active_cards and active_cards[0] is not None else None


def bench(observation: Any, player_index: int) -> tuple[Any, ...]:
    """Return one player's visible Bench Pokémon."""
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


def card_id(card: Any) -> int | None:
    """Return a visible card ID."""
    raw = getattr(card, "id", None)
    return int(raw) if raw is not None else None


def card_ids(cards: Sequence[Any]) -> tuple[int, ...]:
    """Return visible card IDs, omitting unknown entries."""
    return tuple(card_id_ for card in cards if (card_id_ := card_id(card)) is not None)


def option_card_id(observation: Any, player_index: int, option: Any) -> int | None:
    """Resolve the visible card referenced by a legal engine option."""
    raw_card_id = getattr(option, "cardId", None)
    if raw_card_id:
        return int(raw_card_id)

    option_type = _int(getattr(option, "type", None))
    index = _int(getattr(option, "index", None))
    if option_type == int(OptionType.PLAY) and index is not None:
        cards = hand(observation, player_index)
        return card_id(cards[index]) if 0 <= index < len(cards) else None

    area = _int(getattr(option, "area", None))
    owner = _int(getattr(option, "playerIndex", None))
    owner = player_index if owner is None else owner
    if area is None or index is None:
        return None
    if area == int(AreaType.DECK):
        select = getattr(observation, "select", None)
        deck = as_sequence(getattr(select, "deck", ()))
        return card_id(deck[index]) if 0 <= index < len(deck) else None
    if area == int(AreaType.HAND):
        cards = hand(observation, owner)
    elif area == int(AreaType.DISCARD):
        cards = discard(observation, owner)
    elif area == int(AreaType.LOOKING):
        state = getattr(observation, "current", None)
        cards = tuple(as_sequence(getattr(state, "looking", ())))
    elif area == int(AreaType.STADIUM):
        state = getattr(observation, "current", None)
        cards = tuple(as_sequence(getattr(state, "stadium", ())))
    elif area == int(AreaType.ACTIVE):
        target = active(observation, owner)
        return card_id(target)
    elif area == int(AreaType.BENCH):
        cards = bench(observation, owner)
    else:
        return None
    return card_id(cards[index]) if 0 <= index < len(cards) else None


def option_pokemon(observation: Any, player_index: int, option: Any) -> Any | None:
    """Resolve the in-play Pokémon referenced by a legal option."""
    area = _int(getattr(option, "inPlayArea", None))
    index = _int(getattr(option, "inPlayIndex", None))
    if area is None:
        area = _int(getattr(option, "area", None))
        index = _int(getattr(option, "index", None))
    owner = _int(getattr(option, "playerIndex", None))
    owner = player_index if owner is None else owner
    if area == int(AreaType.ACTIVE):
        return active(observation, owner)
    if area == int(AreaType.BENCH) and index is not None:
        cards = bench(observation, owner)
        return cards[index] if 0 <= index < len(cards) else None
    return None


def field_pokemon(observation: Any, player_index: int) -> tuple[Any, ...]:
    """Return Active plus Bench Pokémon."""
    active_card = active(observation, player_index)
    return ((active_card,) if active_card is not None else ()) + bench(
        observation, player_index
    )


def energy_count(pokemon: Any) -> int:
    """Return the engine-resolved attached energy-unit count."""
    return len(as_sequence(getattr(pokemon, "energies", ())))


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
