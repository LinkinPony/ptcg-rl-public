"""Feature-complete same-seat context reconstruction for search states."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ptcg_rl.context import (
    COUNTERPARTY_LOG_TYPE_MAP,
    KNOWN_LOG_TYPES,
    PRIVATE_MOVE_LOG_TYPES,
    PRIVATE_REVERSE_LOG_FIELDS,
    PUBLIC_LOG_FIELDS,
    GameContext,
    GameContextFeatures,
    GameContextSnapshot,
    OpponentBeliefFeatureProducer,
)
from ptcg_rl.engine.constants import LogType

_OPTION_FIELDS = (
    "type",
    "number",
    "area",
    "index",
    "playerIndex",
    "toolIndex",
    "energyIndex",
    "count",
    "inPlayArea",
    "inPlayIndex",
    "attackId",
    "cardId",
    "serial",
    "specialConditionType",
)


@dataclass
class SimulatedContext:
    """Isolated public-context trackers for both seats in one search branch."""

    tracker: GameContext
    belief: OpponentBeliefFeatureProducer | None = None
    counterparty_tracker: GameContext | None = None

    @classmethod
    def from_snapshot(
        cls,
        snapshot: GameContextSnapshot,
        *,
        belief: OpponentBeliefFeatureProducer | None = None,
        counterparty_snapshot: GameContextSnapshot | None = None,
    ) -> SimulatedContext:
        """Create a branch without copying engine or model objects."""
        counterparty_tracker = None
        if counterparty_snapshot is not None:
            if counterparty_snapshot.own_deck_counts:
                raise ValueError(
                    "counterparty context snapshot must not contain a private deck"
                )
            counterparty_tracker = GameContext.from_snapshot(counterparty_snapshot)
        return cls(
            GameContext.from_snapshot(snapshot),
            belief=belief,
            counterparty_tracker=counterparty_tracker,
        )

    def enrich(self, observation: Any, *, update: bool = True) -> Any:
        """Attach history, unseen, revealed, and belief features to a state."""
        public_observation = public_search_observation(
            observation,
            perspective_player_index=self.tracker.player_index,
        )
        features = (
            self.tracker.update(public_observation)
            if update
            else self.tracker.features(public_observation)
        )
        if update and self.counterparty_tracker is not None:
            counterparty_observation = public_search_observation(
                observation,
                perspective_player_index=self.counterparty_tracker.player_index,
            )
            self.counterparty_tracker.update(counterparty_observation)
        if self.belief is not None:
            features = self.belief.augment(public_observation, features)
        return observation_with_context(public_observation, features)

    def enrich_for_actor(
        self,
        observation: Any,
        *,
        actor_player_index: int,
        actor_deck: Sequence[int],
    ) -> Any:
        """Build one legal actor-perspective critic observation.

        The matching seat tracker supplies that actor's public history and
        revealed-card bookkeeping. The caller-provided determinization deck is
        bound only after selecting the tracker, so the rollout opponent's real
        private deck cannot enter a handoff value request.
        """
        if actor_player_index not in (0, 1):
            raise ValueError("actor_player_index must be 0 or 1")
        public_observation = public_search_observation(
            observation,
            perspective_player_index=actor_player_index,
        )
        actor_tracker = self._tracker_for_actor(actor_player_index)
        actor_tracker.set_own_deck(actor_deck)
        features = actor_tracker.features(public_observation)
        if self.belief is not None:
            features = self.belief.augment(public_observation, features)
        return observation_with_context(public_observation, features)

    def fork(self) -> SimulatedContext:
        """Return an independent branch with identical public evidence."""
        return SimulatedContext(
            self.tracker.fork(),
            belief=self.belief,
            counterparty_tracker=(
                None
                if self.counterparty_tracker is None
                else self.counterparty_tracker.fork()
            ),
        )

    def _tracker_for_actor(self, actor_player_index: int) -> GameContext:
        """Fork the actor's own public tracker, with a legacy-safe fallback."""
        if self.tracker.player_index == actor_player_index:
            return self.tracker.fork()
        if (
            self.counterparty_tracker is not None
            and self.counterparty_tracker.player_index == actor_player_index
        ):
            return self.counterparty_tracker.fork()

        actor_tracker = self.tracker.fork()
        actor_tracker.player_index = actor_player_index
        actor_tracker.opponent_revealed_by_serial.clear()
        actor_tracker.opponent_revealed_no_serial_counts.clear()
        return actor_tracker


def public_search_observation(
    observation: Any,
    *,
    root_reference: Any | None = None,
    perspective_player_index: int | None = None,
) -> dict[str, Any]:
    """Project a Search state into one actor-visible public observation.

    Search receives complete hidden zones. This projection is therefore the
    privacy boundary for every network-visible child: only the acting hand and
    private look are retained, prizes and facedown opposing setup Pokemon are
    masked, and reverse-log identities are removed.
    """
    reference = root_reference if root_reference is not None else observation
    raw_current = _field(reference, "current")
    raw_actor_index = _int_field(raw_current, "yourIndex", -1)
    perspective = (
        int(perspective_player_index)
        if perspective_player_index in (0, 1)
        else raw_actor_index
    )
    current = _public_current(
        raw_current,
        perspective_player_index=perspective,
    )
    select = (
        _public_select(_field(reference, "select"))
        if perspective == raw_actor_index
        else None
    )
    return {
        "remainingOverageTime": _field(
            reference,
            "remainingOverageTime",
        ),
        "current": current,
        "logs": _public_logs(
            _field(observation, "logs", ()),
            projected_to_other_player=(perspective != raw_actor_index),
        ),
        "search_begin_input": _field(reference, "search_begin_input"),
        "select": select,
    }


def actor_visible_prompt_fingerprint(
    observation: Any,
    *,
    actor_player_index: int,
) -> str:
    """Return a canonical identity for the exact prompt visible to an actor."""
    if actor_player_index not in (0, 1):
        raise ValueError("actor_player_index must be 0 or 1")
    public_observation = public_search_observation(
        observation,
        perspective_player_index=actor_player_index,
    )
    select = public_observation["select"]
    if select is None:
        raise ValueError("actor_player_index is not the selecting actor")
    return json.dumps(
        select,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def observation_with_context(
    observation: Any,
    context_features: GameContextFeatures,
) -> dict[str, Any]:
    """Return one canonical mapping wrapper around mapping or engine objects."""
    context_dict = context_features.as_observation_dict()
    if isinstance(observation, Mapping):
        copied = dict(observation)
        copied["gameContext"] = context_dict
        return copied
    return {
        "remainingOverageTime": _field(observation, "remainingOverageTime"),
        "current": _field(observation, "current"),
        "logs": _field(observation, "logs", ()),
        "search_begin_input": _field(observation, "search_begin_input"),
        "select": _field(observation, "select"),
        "gameContext": context_dict,
    }


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _has_field(value: Any, name: str) -> bool:
    if isinstance(value, Mapping):
        return name in value
    return hasattr(value, name)


def _int_field(value: Any, name: str, default: int) -> int:
    item = _field(value, name, default)
    return int(item) if item is not None else default


def _public_current(
    current: Any,
    *,
    perspective_player_index: int | None = None,
) -> Any:
    if current is None:
        return None
    actor_index = _int_field(current, "yourIndex", 0)
    your_index = (
        int(perspective_player_index)
        if perspective_player_index in (0, 1)
        else actor_index
    )
    players = tuple(_sequence(_field(current, "players", ())))
    turn = _int_field(current, "turn", 0)
    raw_looking = _field(current, "looking")
    looking = (
        _public_cards(raw_looking)
        if your_index == actor_index
        else _masked_cards(raw_looking)
    )
    return {
        "turn": turn,
        "turnActionCount": _field(current, "turnActionCount", 0),
        "yourIndex": your_index,
        "firstPlayer": _field(current, "firstPlayer", -1),
        "supporterPlayed": _field(current, "supporterPlayed", False),
        "stadiumPlayed": _field(current, "stadiumPlayed", False),
        "energyAttached": _field(current, "energyAttached", False),
        "retreated": _field(current, "retreated", False),
        "result": _field(current, "result", -1),
        "stadium": _public_cards(_field(current, "stadium", ())),
        "looking": looking,
        "players": [
            _public_player(
                player,
                player_index=index,
                your_index=your_index,
                turn=turn,
            )
            for index, player in enumerate(players)
        ],
    }


def _public_player(
    player: Any,
    *,
    player_index: int,
    your_index: int,
    turn: int,
) -> dict[str, Any]:
    prizes = tuple(_sequence(_field(player, "prize", ())))
    hide_setup_pokemon = turn <= 0 and player_index != your_index
    return {
        "active": _public_in_play_cards(
            _field(player, "active", ()),
            hide=hide_setup_pokemon,
        ),
        "bench": _public_in_play_cards(
            _field(player, "bench", ()),
            hide=hide_setup_pokemon,
        ),
        "benchMax": _field(player, "benchMax", 0),
        "deckCount": _field(player, "deckCount", 0),
        "discard": _public_cards(_field(player, "discard", ())),
        "prize": [None] * len(prizes),
        "handCount": _field(player, "handCount", 0),
        "hand": (
            _public_cards(_field(player, "hand"))
            if player_index == your_index
            else None
        ),
        "poisoned": _field(player, "poisoned", False),
        "burned": _field(player, "burned", False),
        "asleep": _field(player, "asleep", False),
        "paralyzed": _field(player, "paralyzed", False),
        "confused": _field(player, "confused", False),
    }


def _public_select(select: Any) -> dict[str, Any] | None:
    if select is None:
        return None
    return {
        "type": _field(select, "type"),
        "context": _field(select, "context"),
        "minCount": _field(select, "minCount", 0),
        "maxCount": _field(select, "maxCount", 0),
        "remainDamageCounter": _field(select, "remainDamageCounter", 0),
        "remainEnergyCost": _field(select, "remainEnergyCost", 0),
        "option": [
            {name: _field(option, name) for name in _OPTION_FIELDS}
            for option in _sequence(_field(select, "option", ()))
        ],
        "deck": _public_cards(_field(select, "deck")),
        "contextCard": _public_card(_field(select, "contextCard")),
        "effect": _public_card(_field(select, "effect")),
    }


def _public_logs(
    logs: Any,
    *,
    projected_to_other_player: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for log in _sequence(logs):
        log_type = _int_field(log, "type", -1)
        unknown_counterparty_log = (
            projected_to_other_player and log_type not in KNOWN_LOG_TYPES
        )
        projected_type = (
            COUNTERPARTY_LOG_TYPE_MAP.get(log_type, log_type)
            if projected_to_other_player
            else log_type
        )
        hides_identity = log_type in {
            int(LogType.DRAW_REVERSE),
            int(LogType.MOVE_CARD_REVERSE),
        } or (
            projected_to_other_player
            and (log_type in PRIVATE_MOVE_LOG_TYPES or log_type not in KNOWN_LOG_TYPES)
        )
        result.append(
            {
                name: (
                    None
                    if (
                        unknown_counterparty_log and name not in {"type", "playerIndex"}
                    )
                    or (hides_identity and name in PRIVATE_REVERSE_LOG_FIELDS)
                    else projected_type
                    if name == "type"
                    else _field(log, name)
                )
                for name in PUBLIC_LOG_FIELDS
            }
        )
    return result


def _public_in_play_cards(cards: Any, *, hide: bool) -> list[Any]:
    visible: list[Any] = []
    for card in _sequence(cards):
        if card is None or hide or _is_facedown(card):
            visible.append(None)
        else:
            visible.append(_public_card(card))
    return visible


def _is_facedown(card: Any) -> bool:
    return any(
        bool(_field(card, field_name, False))
        for field_name in ("faceDown", "facedown", "isFaceDown")
    )


def _masked_cards(cards: Any) -> list[None] | None:
    if cards is None:
        return None
    return [None] * len(_sequence(cards))


def _public_cards(cards: Any) -> list[Any] | None:
    if cards is None:
        return None
    return [_public_card(card) for card in _sequence(cards)]


def _public_card(card: Any) -> dict[str, Any] | None:
    if card is None:
        return None
    result = {
        name: _field(card, name)
        for name in (
            "id",
            "serial",
            "playerIndex",
            "hp",
            "maxHp",
            "appearThisTurn",
        )
        if _has_field(card, name)
    }
    if _has_field(card, "energies"):
        result["energies"] = list(_sequence(_field(card, "energies", ())))
    for name in ("energyCards", "tools", "preEvolution"):
        if _has_field(card, name):
            result[name] = _public_cards(_field(card, name, ()))
    return result


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
