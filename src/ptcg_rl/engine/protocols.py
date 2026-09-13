"""Structural types for the public ``cg.api`` dataclasses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class CardLike(Protocol):
    """Visible card instance from an engine observation."""

    id: int
    serial: int
    playerIndex: int  # noqa: N815


class PokemonLike(Protocol):
    """Visible Pokemon instance from an engine observation."""

    id: int
    serial: int
    hp: int
    maxHp: int  # noqa: N815
    appearThisTurn: bool  # noqa: N815
    energies: Sequence[int]
    energyCards: Sequence[CardLike]  # noqa: N815
    tools: Sequence[CardLike]
    preEvolution: Sequence[CardLike]  # noqa: N815


class PlayerStateLike(Protocol):
    """Per-player public state in an engine observation."""

    active: Sequence[PokemonLike | None]
    bench: Sequence[PokemonLike]
    benchMax: int  # noqa: N815
    deckCount: int  # noqa: N815
    discard: Sequence[CardLike]
    prize: Sequence[CardLike | None]
    handCount: int  # noqa: N815
    hand: Sequence[CardLike] | None
    poisoned: bool
    burned: bool
    asleep: bool
    paralyzed: bool
    confused: bool


class StateLike(Protocol):
    """Full public game state in an engine observation."""

    turn: int
    turnActionCount: int  # noqa: N815
    yourIndex: int  # noqa: N815
    firstPlayer: int  # noqa: N815
    supporterPlayed: bool  # noqa: N815
    stadiumPlayed: bool  # noqa: N815
    energyAttached: bool  # noqa: N815
    retreated: bool
    result: int
    stadium: Sequence[CardLike]
    looking: Sequence[CardLike | None] | None
    players: Sequence[PlayerStateLike]


class OptionLike(Protocol):
    """Selectable option from an engine observation."""

    type: int
    number: int | None
    area: int | None
    index: int | None
    playerIndex: int | None  # noqa: N815
    toolIndex: int | None  # noqa: N815
    energyIndex: int | None  # noqa: N815
    count: int | None
    inPlayArea: int | None  # noqa: N815
    inPlayIndex: int | None  # noqa: N815
    attackId: int | None  # noqa: N815
    cardId: int | None  # noqa: N815
    serial: int | None
    specialConditionType: int | None  # noqa: N815


class SelectDataLike(Protocol):
    """Selection prompt from an engine observation."""

    type: int
    context: int
    minCount: int  # noqa: N815
    maxCount: int  # noqa: N815
    remainDamageCounter: int  # noqa: N815
    remainEnergyCost: int  # noqa: N815
    option: Sequence[OptionLike]
    deck: Sequence[CardLike] | None
    contextCard: CardLike | None  # noqa: N815
    effect: CardLike | None


class LogLike(Protocol):
    """Event log emitted by the engine since the previous selection."""

    type: int
    playerIndex: int | None  # noqa: N815
    hasBasicPokemon: bool | None  # noqa: N815
    cardId: int | None  # noqa: N815
    serial: int | None
    fromArea: int | None  # noqa: N815
    toArea: int | None  # noqa: N815
    cardIdActive: int | None  # noqa: N815
    serialActive: int | None  # noqa: N815
    cardIdBench: int | None  # noqa: N815
    serialBench: int | None  # noqa: N815
    cardIdBefore: int | None  # noqa: N815
    serialBefore: int | None  # noqa: N815
    cardIdAfter: int | None  # noqa: N815
    serialAfter: int | None  # noqa: N815
    cardIdTarget: int | None  # noqa: N815
    serialTarget: int | None  # noqa: N815
    attackId: int | None  # noqa: N815
    value: int | None
    putDamageCounter: bool | None  # noqa: N815
    isRecover: bool | None  # noqa: N815
    head: bool | None
    result: int | None
    reason: int | None


class ObservationLike(Protocol):
    """Search or battle observation from the engine."""

    select: SelectDataLike | None
    logs: Sequence[LogLike]
    current: StateLike | None
    search_begin_input: str | None


class SearchStateLike(Protocol):
    """Search tree state returned by the engine."""

    observation: ObservationLike
    searchId: int  # noqa: N815


ObservationInput = ObservationLike | Mapping[str, Any]
