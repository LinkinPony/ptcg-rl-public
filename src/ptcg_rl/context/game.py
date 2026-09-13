"""Cross-step game-context evidence accumulation.

The context tracks public evidence only. It does not infer game-rule effects;
card effects and action legality remain delegated to the simulator engine.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ptcg_rl.context.public_events import (
    PUBLIC_EVENT_CATEGORICAL_SIZE,
    PUBLIC_EVENT_ENTITY_COUNT,
    PublicEvent,
    PublicEventActorRole,
    PublicEventDecisionToken,
    PublicEventDelta,
    PublicEventOverflowCount,
    append_public_events,
)
from ptcg_rl.engine.constants import AreaType, LogType

HISTORY_COUNTER_NAMES: tuple[str, ...] = (
    "self_attack_count",
    "self_supporter_count",
    "self_energy_attach_count",
    "self_retreat_count",
    "opponent_attack_count",
    "opponent_supporter_count",
    "opponent_energy_attach_count",
    "opponent_retreat_count",
)
HISTORY_COUNTER_SIZE = len(HISTORY_COUNTER_NAMES)
DECK_FLOW_FEATURE_NAMES: tuple[str, ...] = (
    "self_draw_count",
    "self_return_to_deck_count",
    "self_recent_turn_draw_count",
    "self_recent_turn_return_to_deck_count",
    "self_recent_turn_deck_delta",
    "self_deck_rebound_count",
    "self_no_attack_turns",
    "opponent_draw_count",
    "opponent_return_to_deck_count",
    "opponent_recent_turn_draw_count",
    "opponent_recent_turn_return_to_deck_count",
    "opponent_recent_turn_deck_delta",
    "opponent_deck_rebound_count",
    "opponent_no_attack_turns",
)
DECK_FLOW_FEATURE_SIZE = len(DECK_FLOW_FEATURE_NAMES)
_HISTORY_KIND_COUNT = 4
_ATTACK_COUNTER = 0
_SUPPORTER_COUNTER = 1
_ENERGY_ATTACH_COUNTER = 2
_RETREAT_COUNTER = 3
_LOW_DECK_REBOUND_MAX_COUNT = 5
_CARD_ID_SERIAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("cardId", "serial"),
    ("cardIdActive", "serialActive"),
    ("cardIdBench", "serialBench"),
    ("cardIdBefore", "serialBefore"),
    ("cardIdAfter", "serialAfter"),
    ("cardIdTarget", "serialTarget"),
)
_HIDDEN_LOG_TYPES = {int(LogType.DRAW_REVERSE), int(LogType.MOVE_CARD_REVERSE)}
_LogBatchSignature = tuple[int, int]


@dataclass(frozen=True, order=True)
class CardCount:
    """One card-id/count pair in a compact multiset."""

    card_id: int
    count: int


@dataclass(frozen=True, order=True)
class ExpectedCardCount:
    """One card-id/expected-count pair from an opponent belief posterior."""

    card_id: int
    expected_count: float


@dataclass(frozen=True, order=True)
class LastAttackRecord:
    """Last declared attack for a visible Pokemon serial."""

    serial: int
    attack_id: int


@dataclass(frozen=True)
class GameContextFeatures:
    """Derived context features consumed by encoders and row writers."""

    own_unseen: tuple[CardCount, ...] = ()
    opponent_revealed: tuple[CardCount, ...] = ()
    opponent_belief: tuple[ExpectedCardCount, ...] = ()
    opponent_belief_entropy: float = 0.0
    opponent_belief_empty: bool = True
    history_counts: tuple[int, ...] = (0,) * HISTORY_COUNTER_SIZE
    deck_flow_counts: tuple[int, ...] = (0,) * DECK_FLOW_FEATURE_SIZE
    last_attacks: tuple[LastAttackRecord, ...] = ()
    public_event_delta: PublicEventDelta = PublicEventDelta()

    def as_observation_dict(self) -> dict[str, Any]:
        """Return a JSON-like shape embedded into reconstructed observations."""
        return {
            "ownUnseen": [
                {"cardId": item.card_id, "count": item.count}
                for item in self.own_unseen
            ],
            "opponentRevealed": [
                {"cardId": item.card_id, "count": item.count}
                for item in self.opponent_revealed
            ],
            "opponentBelief": [
                {"cardId": item.card_id, "expectedCount": item.expected_count}
                for item in self.opponent_belief
            ],
            "opponentBeliefEntropy": float(self.opponent_belief_entropy),
            "opponentBeliefEmpty": bool(self.opponent_belief_empty),
            "historyCounts": list(self.history_counts),
            "deckFlowCounts": list(self.deck_flow_counts),
            "lastAttacks": [
                {"serial": item.serial, "attackId": item.attack_id}
                for item in self.last_attacks
            ],
            "publicEventDelta": {
                "droppedCount": self.public_event_delta.dropped_count,
                "overflow": [
                    {
                        "eventType": item.event_type,
                        "actorRole": int(item.actor_role),
                        "count": item.count,
                    }
                    for item in self.public_event_delta.overflow
                ],
                "events": [
                    {
                        "eventType": event.event_type,
                        "actorRole": int(event.actor_role),
                        "fromArea": event.from_area,
                        "toArea": event.to_area,
                        "cardIds": list(event.card_ids),
                        "serials": list(event.serials),
                        "entityMask": list(event.entity_mask),
                        "attackId": event.attack_id,
                        "attackIdPresent": event.attack_id_present,
                        "value": event.value,
                        "valuePresent": event.value_present,
                        "categoricalValues": list(event.categorical_values),
                    }
                    for event in self.public_event_delta.events
                ],
            },
        }

    def row_fields(self) -> dict[str, Any]:
        """Return compact Parquet row fields for this context."""
        return {
            "own_unseen_ids": [item.card_id for item in self.own_unseen],
            "own_unseen_counts": [item.count for item in self.own_unseen],
            "opp_revealed_ids": [item.card_id for item in self.opponent_revealed],
            "opp_revealed_counts": [item.count for item in self.opponent_revealed],
            "history_counts": list(self.history_counts),
            "deck_flow_counts": list(self.deck_flow_counts),
            "last_attack_serials": [item.serial for item in self.last_attacks],
            "last_attack_ids": [item.attack_id for item in self.last_attacks],
        }

    def last_attack_by_serial(self) -> dict[int, int]:
        """Return serial -> attack id lookup."""
        return {item.serial: item.attack_id for item in self.last_attacks}


def _default_supporter_card_id_set() -> frozenset[int]:
    return frozenset(default_supporter_card_ids())


@dataclass(frozen=True)
class GameContextSnapshot:
    """Explicit immutable state used to fork simulated same-seat branches."""

    player_index: int | None
    own_deck_counts: tuple[tuple[int, int], ...]
    supporter_card_ids: frozenset[int]
    opponent_revealed_by_serial: tuple[tuple[int, int], ...]
    opponent_revealed_no_serial_counts: tuple[tuple[int, int], ...]
    history_by_player: tuple[tuple[int, ...], tuple[int, ...]]
    draw_counts_by_player: tuple[int, int]
    return_to_deck_counts_by_player: tuple[int, int]
    recent_turn_draw_counts_by_player: tuple[int, int]
    recent_turn_return_counts_by_player: tuple[int, int]
    recent_turn_deck_deltas_by_player: tuple[int, int]
    deck_rebound_counts_by_player: tuple[int, int]
    no_attack_turns_by_player: tuple[int, int]
    current_turn_draw_counts_by_player: tuple[int, int]
    current_turn_return_counts_by_player: tuple[int, int]
    current_turn_deck_deltas_by_player: tuple[int, int]
    turn_open_by_player: tuple[bool, bool]
    turn_attacked_by_player: tuple[bool, bool]
    turn_rebounded_by_player: tuple[bool, bool]
    observed_deck_counts_by_player: tuple[int | None, int | None]
    last_log_batch_signature: _LogBatchSignature | None
    last_attack_by_serial: tuple[tuple[int, int], ...]
    pending_public_events: tuple[PublicEvent, ...] = ()
    public_event_overflow: tuple[PublicEventOverflowCount, ...] = ()
    public_event_generation: int = 0


@dataclass
class GameContext:
    """Online public-evidence accumulator for one player's perspective."""

    player_index: int | None = None
    own_deck_counts: Counter[int] = field(default_factory=Counter)
    supporter_card_ids: frozenset[int] = field(
        default_factory=_default_supporter_card_id_set
    )
    opponent_revealed_by_serial: dict[int, int] = field(default_factory=dict)
    opponent_revealed_no_serial_counts: Counter[int] = field(default_factory=Counter)
    history_by_player: list[list[int]] = field(
        default_factory=lambda: [[0] * _HISTORY_KIND_COUNT for _ in range(2)]
    )
    draw_counts_by_player: list[int] = field(default_factory=lambda: [0, 0])
    return_to_deck_counts_by_player: list[int] = field(default_factory=lambda: [0, 0])
    recent_turn_draw_counts_by_player: list[int] = field(default_factory=lambda: [0, 0])
    recent_turn_return_counts_by_player: list[int] = field(
        default_factory=lambda: [0, 0]
    )
    recent_turn_deck_deltas_by_player: list[int] = field(default_factory=lambda: [0, 0])
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
    turn_attacked_by_player: list[bool] = field(default_factory=lambda: [False, False])
    turn_rebounded_by_player: list[bool] = field(default_factory=lambda: [False, False])
    observed_deck_counts_by_player: list[int | None] = field(
        default_factory=lambda: [None, None]
    )
    last_log_batch_signature: _LogBatchSignature | None = None
    last_attack_by_serial: dict[int, int] = field(default_factory=dict)
    pending_public_events: list[PublicEvent] = field(default_factory=list)
    public_event_overflow: tuple[PublicEventOverflowCount, ...] = ()
    public_event_generation: int = 0

    def reset(
        self,
        *,
        player_index: int | None = None,
        own_deck: Sequence[int] | None = None,
    ) -> None:
        """Clear episode state and optionally set player/deck metadata."""
        self.player_index = player_index
        self.own_deck_counts = Counter(int(card_id) for card_id in own_deck or ())
        self.opponent_revealed_by_serial.clear()
        self.opponent_revealed_no_serial_counts.clear()
        self.history_by_player = [[0] * _HISTORY_KIND_COUNT for _ in range(2)]
        self.draw_counts_by_player = [0, 0]
        self.return_to_deck_counts_by_player = [0, 0]
        self.recent_turn_draw_counts_by_player = [0, 0]
        self.recent_turn_return_counts_by_player = [0, 0]
        self.recent_turn_deck_deltas_by_player = [0, 0]
        self.deck_rebound_counts_by_player = [0, 0]
        self.no_attack_turns_by_player = [0, 0]
        self.current_turn_draw_counts_by_player = [0, 0]
        self.current_turn_return_counts_by_player = [0, 0]
        self.current_turn_deck_deltas_by_player = [0, 0]
        self.turn_open_by_player = [False, False]
        self.turn_attacked_by_player = [False, False]
        self.turn_rebounded_by_player = [False, False]
        self.observed_deck_counts_by_player = [None, None]
        self.last_log_batch_signature = None
        self.last_attack_by_serial.clear()
        self.pending_public_events.clear()
        self.public_event_overflow = ()
        self.public_event_generation += 1

    def set_own_deck(self, card_ids: Sequence[int]) -> None:
        """Set the known 60-card deck list for this player."""
        self.own_deck_counts = Counter(int(card_id) for card_id in card_ids)

    def snapshot(self) -> GameContextSnapshot:
        """Return a stable public-evidence snapshot without arbitrary deepcopy."""
        return GameContextSnapshot(
            player_index=self.player_index,
            own_deck_counts=tuple(sorted(self.own_deck_counts.items())),
            supporter_card_ids=self.supporter_card_ids,
            opponent_revealed_by_serial=tuple(
                sorted(self.opponent_revealed_by_serial.items())
            ),
            opponent_revealed_no_serial_counts=tuple(
                sorted(self.opponent_revealed_no_serial_counts.items())
            ),
            history_by_player=(
                tuple(self.history_by_player[0]),
                tuple(self.history_by_player[1]),
            ),
            draw_counts_by_player=_pair(self.draw_counts_by_player),
            return_to_deck_counts_by_player=_pair(self.return_to_deck_counts_by_player),
            recent_turn_draw_counts_by_player=_pair(
                self.recent_turn_draw_counts_by_player
            ),
            recent_turn_return_counts_by_player=_pair(
                self.recent_turn_return_counts_by_player
            ),
            recent_turn_deck_deltas_by_player=_pair(
                self.recent_turn_deck_deltas_by_player
            ),
            deck_rebound_counts_by_player=_pair(self.deck_rebound_counts_by_player),
            no_attack_turns_by_player=_pair(self.no_attack_turns_by_player),
            current_turn_draw_counts_by_player=_pair(
                self.current_turn_draw_counts_by_player
            ),
            current_turn_return_counts_by_player=_pair(
                self.current_turn_return_counts_by_player
            ),
            current_turn_deck_deltas_by_player=_pair(
                self.current_turn_deck_deltas_by_player
            ),
            turn_open_by_player=_bool_pair(self.turn_open_by_player),
            turn_attacked_by_player=_bool_pair(self.turn_attacked_by_player),
            turn_rebounded_by_player=_bool_pair(self.turn_rebounded_by_player),
            observed_deck_counts_by_player=_optional_int_pair(
                self.observed_deck_counts_by_player
            ),
            last_log_batch_signature=self.last_log_batch_signature,
            last_attack_by_serial=tuple(sorted(self.last_attack_by_serial.items())),
            pending_public_events=tuple(self.pending_public_events),
            public_event_overflow=self.public_event_overflow,
            public_event_generation=self.public_event_generation,
        )

    @classmethod
    def from_snapshot(cls, snapshot: GameContextSnapshot) -> GameContext:
        """Construct an independent context branch from an explicit snapshot."""
        return cls(
            player_index=snapshot.player_index,
            own_deck_counts=Counter(dict(snapshot.own_deck_counts)),
            supporter_card_ids=snapshot.supporter_card_ids,
            opponent_revealed_by_serial=dict(snapshot.opponent_revealed_by_serial),
            opponent_revealed_no_serial_counts=Counter(
                dict(snapshot.opponent_revealed_no_serial_counts)
            ),
            history_by_player=[
                list(snapshot.history_by_player[0]),
                list(snapshot.history_by_player[1]),
            ],
            draw_counts_by_player=list(snapshot.draw_counts_by_player),
            return_to_deck_counts_by_player=list(
                snapshot.return_to_deck_counts_by_player
            ),
            recent_turn_draw_counts_by_player=list(
                snapshot.recent_turn_draw_counts_by_player
            ),
            recent_turn_return_counts_by_player=list(
                snapshot.recent_turn_return_counts_by_player
            ),
            recent_turn_deck_deltas_by_player=list(
                snapshot.recent_turn_deck_deltas_by_player
            ),
            deck_rebound_counts_by_player=list(snapshot.deck_rebound_counts_by_player),
            no_attack_turns_by_player=list(snapshot.no_attack_turns_by_player),
            current_turn_draw_counts_by_player=list(
                snapshot.current_turn_draw_counts_by_player
            ),
            current_turn_return_counts_by_player=list(
                snapshot.current_turn_return_counts_by_player
            ),
            current_turn_deck_deltas_by_player=list(
                snapshot.current_turn_deck_deltas_by_player
            ),
            turn_open_by_player=list(snapshot.turn_open_by_player),
            turn_attacked_by_player=list(snapshot.turn_attacked_by_player),
            turn_rebounded_by_player=list(snapshot.turn_rebounded_by_player),
            observed_deck_counts_by_player=list(
                snapshot.observed_deck_counts_by_player
            ),
            last_log_batch_signature=snapshot.last_log_batch_signature,
            last_attack_by_serial=dict(snapshot.last_attack_by_serial),
            pending_public_events=list(snapshot.pending_public_events),
            public_event_overflow=snapshot.public_event_overflow,
            public_event_generation=snapshot.public_event_generation,
        )

    def fork(self) -> GameContext:
        """Return an isolated simulated branch with identical public evidence."""
        return self.from_snapshot(self.snapshot())

    def prepare_decision(self) -> PublicEventDecisionToken:
        """Freeze the current event input without advancing the decision clock."""
        return PublicEventDecisionToken(
            generation=self.public_event_generation,
            delta=self._public_event_delta(),
        )

    def commit_decision(
        self,
        token: PublicEventDecisionToken,
    ) -> PublicEventDelta:
        """Consume one prepared delta after the real policy action is accepted."""
        self._validate_decision_token(token)
        self.pending_public_events.clear()
        self.public_event_overflow = ()
        self.public_event_generation += 1
        return token.delta

    def abort_decision(self, token: PublicEventDecisionToken) -> None:
        """Validate and retain an unaccepted prepared decision."""
        self._validate_decision_token(token)

    def _validate_decision_token(self, token: PublicEventDecisionToken) -> None:
        if token.generation != self.public_event_generation:
            raise RuntimeError("public event decision token is stale or committed")
        if token.delta != self._public_event_delta():
            raise RuntimeError("public events changed after decision preparation")

    def update(
        self,
        observation: Any,
        *,
        accumulate_public_events: bool = True,
        deduplicate_log_batch: bool = True,
    ) -> GameContextFeatures:
        """Accumulate evidence from one ACTIVE observation.

        Synthetic counterparty projections may update explicit public context
        for immediate search handoffs, but must leave event consumption to that
        seat's next authoritative engine callback.
        """
        current = _mapping_or_object(_field(observation, "current"))
        if current is None:
            return self.features(observation)
        your_index = _int_field(current, "yourIndex", self.player_index or 0)
        if your_index not in (0, 1):
            return self.features(observation)
        if self.player_index is None:
            self.player_index = your_index
        opponent_index = 1 - int(self.player_index)

        visible_by_serial = _visible_cards_by_serial(
            current,
            opponent_index,
            include_hand=False,
        )
        for serial, card_id in visible_by_serial.items():
            self.opponent_revealed_by_serial.setdefault(serial, card_id)

        logs = _sequence(_field(observation, "logs", ()))
        observed_deck_counts = _observed_deck_counts(current)
        batch_signature = _callback_cursor(current)
        is_duplicate_batch = (
            deduplicate_log_batch
            and batch_signature is not None
            and batch_signature == self.last_log_batch_signature
        )
        if not is_duplicate_batch:
            if accumulate_public_events:
                event_delta = append_public_events(
                    self._public_event_delta(),
                    logs,
                    player_index=self.player_index,
                )
                self.pending_public_events = list(event_delta.events)
                self.public_event_overflow = event_delta.overflow
            self._update_from_logs(
                logs,
                opponent_index=opponent_index,
                observed_deck_counts=observed_deck_counts,
            )
        else:
            self._sync_observed_deck_counts(observed_deck_counts)
        self.last_log_batch_signature = batch_signature
        return self.features(observation)

    def features(self, observation: Any | None = None) -> GameContextFeatures:
        """Return derived compact context features for the current observation."""
        current = (
            _mapping_or_object(_field(observation, "current")) if observation else None
        )
        own_visible: Counter[int] = Counter()
        opponent_visible: Counter[int] = Counter()
        if current is not None and self.player_index in (0, 1):
            own_visible = _visible_card_counts(
                current,
                int(self.player_index),
                include_hand=True,
            )
            own_visible.update(
                _visible_select_deck_counts(observation, int(self.player_index))
            )
            opponent_visible = _visible_card_counts(
                current,
                1 - int(self.player_index),
                include_hand=False,
            )

        own_unseen = _positive_counts(self.own_deck_counts - own_visible)
        opponent_known = self._opponent_known_counts()
        opponent_revealed = _positive_counts(opponent_known - opponent_visible)
        return GameContextFeatures(
            own_unseen=_counter_to_card_counts(own_unseen),
            opponent_revealed=_counter_to_card_counts(opponent_revealed),
            history_counts=self._history_counts_for_perspective(),
            deck_flow_counts=self._deck_flow_counts_for_perspective(),
            last_attacks=tuple(
                LastAttackRecord(serial=serial, attack_id=attack_id)
                for serial, attack_id in sorted(self.last_attack_by_serial.items())
            ),
            public_event_delta=self._public_event_delta(),
        )

    def _public_event_delta(self) -> PublicEventDelta:
        """Return the currently unconsumed bounded event history."""
        return PublicEventDelta(
            events=tuple(self.pending_public_events),
            overflow=self.public_event_overflow,
        )

    def _update_from_logs(
        self,
        logs: Sequence[Any],
        *,
        opponent_index: int,
        observed_deck_counts: tuple[int | None, int | None],
    ) -> None:
        running_deck_counts = list(self.observed_deck_counts_by_player)
        for candidate_player_index, observed_count in enumerate(observed_deck_counts):
            if observed_count is None:
                continue
            net_delta = sum(
                _deck_count_delta(log)
                for log in logs
                if _optional_int(_field(log, "playerIndex")) == candidate_player_index
            )
            running_deck_counts[candidate_player_index] = max(
                0,
                observed_count - net_delta,
            )

        for log in logs:
            player_index = _optional_int(_field(log, "playerIndex"))
            if player_index in (0, 1):
                deck_count_before = running_deck_counts[player_index]
                self._update_deck_flow_from_log(
                    log,
                    player_index,
                    deck_count_before=deck_count_before,
                )
                if deck_count_before is not None:
                    running_deck_counts[player_index] = max(
                        0,
                        deck_count_before + _deck_count_delta(log),
                    )
                self._update_history_from_log(log, player_index)
            if player_index == opponent_index:
                self._update_revealed_from_log(log)

        self.observed_deck_counts_by_player = [
            observed_count
            if observed_count is not None
            else running_deck_counts[player_index]
            for player_index, observed_count in enumerate(observed_deck_counts)
        ]

    def _sync_observed_deck_counts(
        self,
        observed_deck_counts: tuple[int | None, int | None],
    ) -> None:
        """Refresh public deck sizes when a repeated log batch is suppressed."""
        self.observed_deck_counts_by_player = [
            observed_count
            if observed_count is not None
            else self.observed_deck_counts_by_player[player_index]
            for player_index, observed_count in enumerate(observed_deck_counts)
        ]

    def _update_history_from_log(self, log: Any, player_index: int) -> None:
        log_type = _int_field(log, "type", -1)
        if log_type == int(LogType.ATTACK):
            self.history_by_player[player_index][_ATTACK_COUNTER] += 1
            self.turn_attacked_by_player[player_index] = True
            serial = _optional_int(_field(log, "serial"))
            attack_id = _optional_int(_field(log, "attackId"))
            if serial is not None and attack_id is not None:
                self.last_attack_by_serial[serial] = attack_id
            return
        if log_type == int(LogType.PLAY):
            card_id = _optional_int(_field(log, "cardId"))
            if card_id in self.supporter_card_ids:
                self.history_by_player[player_index][_SUPPORTER_COUNTER] += 1
            return
        if log_type == int(LogType.ATTACH):
            self.history_by_player[player_index][_ENERGY_ATTACH_COUNTER] += 1
            return
        if log_type == int(LogType.SWITCH) and _is_retreat_log(log):
            self.history_by_player[player_index][_RETREAT_COUNTER] += 1

    def _update_deck_flow_from_log(
        self,
        log: Any,
        player_index: int,
        *,
        deck_count_before: int | None,
    ) -> None:
        """Accumulate public deck movement from engine-emitted event logs."""
        log_type = _int_field(log, "type", -1)
        if log_type == int(LogType.TURN_START):
            self._start_turn(player_index)
            return
        if log_type == int(LogType.TURN_END):
            self._finish_turn(player_index)
            return
        if log_type in {int(LogType.DRAW), int(LogType.DRAW_REVERSE)}:
            self.draw_counts_by_player[player_index] += 1
            if self.turn_open_by_player[player_index]:
                self.current_turn_draw_counts_by_player[player_index] += 1
                self.current_turn_deck_deltas_by_player[player_index] -= 1
            return
        if log_type not in {int(LogType.MOVE_CARD), int(LogType.MOVE_CARD_REVERSE)}:
            return

        from_area = _optional_int(_field(log, "fromArea"))
        to_area = _optional_int(_field(log, "toArea"))
        if self.turn_open_by_player[player_index]:
            if from_area == int(AreaType.DECK):
                self.current_turn_deck_deltas_by_player[player_index] -= 1
            if to_area == int(AreaType.DECK):
                self.current_turn_deck_deltas_by_player[player_index] += 1
        if to_area != int(AreaType.DECK):
            return
        self.return_to_deck_counts_by_player[player_index] += 1
        if self.turn_open_by_player[player_index]:
            self.current_turn_return_counts_by_player[player_index] += 1
            is_low_deck_increase = (
                deck_count_before is not None
                and deck_count_before <= _LOW_DECK_REBOUND_MAX_COUNT
                and _deck_count_delta(log) > 0
            )
            if is_low_deck_increase and not self.turn_rebounded_by_player[player_index]:
                self.deck_rebound_counts_by_player[player_index] += 1
                self.turn_rebounded_by_player[player_index] = True

    def _start_turn(self, player_index: int) -> None:
        self.current_turn_draw_counts_by_player[player_index] = 0
        self.current_turn_return_counts_by_player[player_index] = 0
        self.current_turn_deck_deltas_by_player[player_index] = 0
        self.turn_open_by_player[player_index] = True
        self.turn_attacked_by_player[player_index] = False
        self.turn_rebounded_by_player[player_index] = False

    def _finish_turn(self, player_index: int) -> None:
        if not self.turn_open_by_player[player_index]:
            return
        self.recent_turn_draw_counts_by_player[player_index] = (
            self.current_turn_draw_counts_by_player[player_index]
        )
        self.recent_turn_return_counts_by_player[player_index] = (
            self.current_turn_return_counts_by_player[player_index]
        )
        self.recent_turn_deck_deltas_by_player[player_index] = (
            self.current_turn_deck_deltas_by_player[player_index]
        )
        if self.turn_attacked_by_player[player_index]:
            self.no_attack_turns_by_player[player_index] = 0
        else:
            self.no_attack_turns_by_player[player_index] += 1
        self.turn_open_by_player[player_index] = False

    def _update_revealed_from_log(self, log: Any) -> None:
        log_type = _int_field(log, "type", -1)
        if log_type in _HIDDEN_LOG_TYPES:
            return
        for card_field, serial_field in _CARD_ID_SERIAL_FIELDS:
            card_id = _optional_int(_field(log, card_field))
            if card_id is None:
                continue
            serial = _optional_int(_field(log, serial_field))
            if serial is None:
                self.opponent_revealed_no_serial_counts[card_id] += 1
            else:
                self.opponent_revealed_by_serial.setdefault(serial, card_id)

    def _opponent_known_counts(self) -> Counter[int]:
        counts: Counter[int] = Counter(self.opponent_revealed_by_serial.values())
        counts.update(self.opponent_revealed_no_serial_counts)
        return counts

    def _history_counts_for_perspective(self) -> tuple[int, ...]:
        if self.player_index not in (0, 1):
            return (0,) * HISTORY_COUNTER_SIZE
        own = self.history_by_player[int(self.player_index)]
        opponent = self.history_by_player[1 - int(self.player_index)]
        return tuple(int(value) for value in (*own, *opponent))

    def _deck_flow_counts_for_perspective(self) -> tuple[int, ...]:
        if self.player_index not in (0, 1):
            return (0,) * DECK_FLOW_FEATURE_SIZE
        own = int(self.player_index)
        opponent = 1 - own
        values: list[int] = []
        for player_index in (own, opponent):
            if self.turn_open_by_player[player_index]:
                recent_draws = self.current_turn_draw_counts_by_player[player_index]
                recent_returns = self.current_turn_return_counts_by_player[player_index]
                recent_delta = self.current_turn_deck_deltas_by_player[player_index]
            else:
                recent_draws = self.recent_turn_draw_counts_by_player[player_index]
                recent_returns = self.recent_turn_return_counts_by_player[player_index]
                recent_delta = self.recent_turn_deck_deltas_by_player[player_index]
            values.extend(
                (
                    self.draw_counts_by_player[player_index],
                    self.return_to_deck_counts_by_player[player_index],
                    recent_draws,
                    recent_returns,
                    recent_delta,
                    self.deck_rebound_counts_by_player[player_index],
                    self.no_attack_turns_by_player[player_index],
                )
            )
        return tuple(int(value) for value in values)


def context_features_from_row(row: Mapping[str, Any]) -> GameContextFeatures:
    """Reconstruct context features from the new step-row schema."""
    history = tuple(int(value) for value in _sequence(row["history_counts"]))
    if len(history) != HISTORY_COUNTER_SIZE:
        raise ValueError("history_counts has invalid width")
    deck_flow = _fixed_width_ints(
        row.get("deck_flow_counts"),
        width=DECK_FLOW_FEATURE_SIZE,
        field_name="deck_flow_counts",
    )
    return GameContextFeatures(
        own_unseen=_card_counts_from_columns(
            row["own_unseen_ids"],
            row["own_unseen_counts"],
        ),
        opponent_revealed=_card_counts_from_columns(
            row["opp_revealed_ids"],
            row["opp_revealed_counts"],
        ),
        opponent_belief=(),
        opponent_belief_entropy=0.0,
        opponent_belief_empty=True,
        history_counts=history,
        deck_flow_counts=deck_flow,
        last_attacks=tuple(
            LastAttackRecord(serial=serial, attack_id=attack_id)
            for serial, attack_id in zip(
                (int(value) for value in _sequence(row["last_attack_serials"])),
                (int(value) for value in _sequence(row["last_attack_ids"])),
                strict=True,
            )
        ),
    )


def context_features_from_observation(observation: Any) -> GameContextFeatures:
    """Read embedded context features from an observation mapping/object."""
    raw_context = _field(observation, "gameContext")
    if raw_context is None:
        return GameContextFeatures()
    history = tuple(
        int(value) for value in _sequence(_field(raw_context, "historyCounts"))
    )
    if len(history) != HISTORY_COUNTER_SIZE:
        raise ValueError("gameContext.historyCounts has invalid width")
    deck_flow = _fixed_width_ints(
        _field(raw_context, "deckFlowCounts"),
        width=DECK_FLOW_FEATURE_SIZE,
        field_name="gameContext.deckFlowCounts",
    )
    public_event_delta = _public_event_delta_from_observation(raw_context)
    return GameContextFeatures(
        own_unseen=_card_counts_from_items(_sequence(_field(raw_context, "ownUnseen"))),
        opponent_revealed=_card_counts_from_items(
            _sequence(_field(raw_context, "opponentRevealed"))
        ),
        opponent_belief=_expected_card_counts_from_items(
            _sequence(_field(raw_context, "opponentBelief"))
        ),
        opponent_belief_entropy=_float_field(
            raw_context,
            "opponentBeliefEntropy",
            0.0,
        ),
        opponent_belief_empty=bool(_field(raw_context, "opponentBeliefEmpty", True)),
        history_counts=history,
        deck_flow_counts=deck_flow,
        last_attacks=tuple(
            LastAttackRecord(
                serial=_int_field(item, "serial", 0),
                attack_id=_int_field(item, "attackId", 0),
            )
            for item in _sequence(_field(raw_context, "lastAttacks"))
        ),
        public_event_delta=public_event_delta,
    )


def _public_event_delta_from_observation(raw_context: Any) -> PublicEventDelta:
    """Rebuild the canonical event delta embedded by ``GameContextFeatures``."""
    raw_delta = _field(raw_context, "publicEventDelta")
    if raw_delta is None:
        return PublicEventDelta()
    events = tuple(
        PublicEvent(
            event_type=_int_field(item, "eventType", 0),
            actor_role=_public_event_actor_role(item),
            from_area=_int_field(item, "fromArea", 0),
            to_area=_int_field(item, "toArea", 0),
            card_ids=_fixed_width_ints(
                _field(item, "cardIds"),
                width=PUBLIC_EVENT_ENTITY_COUNT,
                field_name="gameContext.publicEventDelta.cardIds",
            ),
            serials=_fixed_width_ints(
                _field(item, "serials"),
                width=PUBLIC_EVENT_ENTITY_COUNT,
                field_name="gameContext.publicEventDelta.serials",
            ),
            entity_mask=tuple(
                bool(value) for value in _sequence(_field(item, "entityMask"))
            ),
            attack_id=_int_field(item, "attackId", 0),
            attack_id_present=bool(_field(item, "attackIdPresent", False)),
            value=_float_field(item, "value", 0.0),
            value_present=bool(_field(item, "valuePresent", False)),
            categorical_values=_fixed_width_ints(
                _field(item, "categoricalValues"),
                width=PUBLIC_EVENT_CATEGORICAL_SIZE,
                field_name="gameContext.publicEventDelta.categoricalValues",
            ),
        )
        for item in _sequence(_field(raw_delta, "events"))
    )
    delta = PublicEventDelta(
        events=events,
        overflow=tuple(
            PublicEventOverflowCount(
                event_type=_int_field(item, "eventType", 0),
                actor_role=_public_event_actor_role(item),
                count=_int_field(item, "count", 0),
            )
            for item in _sequence(_field(raw_delta, "overflow"))
        ),
    )
    declared_dropped = _field(raw_delta, "droppedCount")
    if declared_dropped is not None and int(declared_dropped) != delta.dropped_count:
        raise ValueError("gameContext public event overflow count is inconsistent")
    return delta


def _public_event_actor_role(item: Any) -> PublicEventActorRole:
    return PublicEventActorRole(_int_field(item, "actorRole", 0))


@lru_cache(maxsize=1)
def default_supporter_card_ids() -> tuple[int, ...]:
    """Load Supporter card ids from the competition card metadata CSV."""
    csv_path = Path(__file__).resolve().parents[3] / "data" / "EN_Card_Data.csv"
    if not csv_path.exists():
        return ()
    supporter_ids: list[int] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            if _stage_or_type(row).strip().lower() == "supporter":
                supporter_ids.append(int(row["Card ID"]))
    return tuple(sorted(set(supporter_ids)))


def _card_counts_from_columns(ids: Any, counts: Any) -> tuple[CardCount, ...]:
    card_ids = [int(value) for value in _sequence(ids)]
    card_counts = [int(value) for value in _sequence(counts)]
    if len(card_ids) != len(card_counts):
        raise ValueError("card id/count columns must align")
    return tuple(
        CardCount(card_id=card_id, count=count)
        for card_id, count in zip(card_ids, card_counts, strict=True)
        if card_id > 0 and count > 0
    )


def _card_counts_from_items(items: Sequence[Any]) -> tuple[CardCount, ...]:
    return tuple(
        CardCount(
            card_id=_int_field(item, "cardId", 0),
            count=_int_field(item, "count", 0),
        )
        for item in items
        if _int_field(item, "cardId", 0) > 0 and _int_field(item, "count", 0) > 0
    )


def _expected_card_counts_from_items(
    items: Sequence[Any],
) -> tuple[ExpectedCardCount, ...]:
    return tuple(
        ExpectedCardCount(
            card_id=_int_field(item, "cardId", 0),
            expected_count=_float_field(item, "expectedCount", 0.0),
        )
        for item in items
        if _int_field(item, "cardId", 0) > 0
        and _float_field(item, "expectedCount", 0.0) > 0.0
    )


def _counter_to_card_counts(counter: Counter[int]) -> tuple[CardCount, ...]:
    return tuple(
        CardCount(card_id=card_id, count=count)
        for card_id, count in sorted(counter.items())
        if card_id > 0 and count > 0
    )


def _positive_counts(counter: Counter[int]) -> Counter[int]:
    return Counter({card_id: count for card_id, count in counter.items() if count > 0})


def _fixed_width_ints(
    values: Any,
    *,
    width: int,
    field_name: str,
) -> tuple[int, ...]:
    raw_values = tuple(int(value) for value in _sequence(values))
    if not raw_values and values is None:
        return (0,) * width
    if len(raw_values) != width:
        raise ValueError(f"{field_name} has invalid width")
    return raw_values


def _pair(values: Sequence[int]) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError("player counter must have width 2")
    return (int(values[0]), int(values[1]))


def _bool_pair(values: Sequence[bool]) -> tuple[bool, bool]:
    if len(values) != 2:
        raise ValueError("player flag must have width 2")
    return (bool(values[0]), bool(values[1]))


def _optional_int_pair(
    values: Sequence[int | None],
) -> tuple[int | None, int | None]:
    if len(values) != 2:
        raise ValueError("player counter must have width 2")
    return (
        int(values[0]) if values[0] is not None else None,
        int(values[1]) if values[1] is not None else None,
    )


def _observed_deck_counts(state: Any) -> tuple[int | None, int | None]:
    players = _sequence(_field(state, "players", ()))
    values: list[int | None] = []
    for player_index in range(2):
        if player_index >= len(players):
            values.append(None)
            continue
        values.append(_optional_int(_field(players[player_index], "deckCount")))
    return _optional_int_pair(values)


def _callback_cursor(current: Any) -> _LogBatchSignature | None:
    """Identify one engine callback independently of its event contents.

    The bundled engine increments ``turnActionCount`` before every select
    callback (``ptcgProgram 22/State.h``) and changes ``turn`` at turn reset.
    Re-observing the same cursor is therefore a retry; identical logs at the
    next cursor are a distinct event batch and must not be collapsed.
    """
    turn = _optional_int(_field(current, "turn"))
    turn_action_count = _optional_int(_field(current, "turnActionCount"))
    if turn is None or turn_action_count is None:
        return None
    return (turn, turn_action_count)


def _deck_count_delta(log: Any) -> int:
    """Return the public deck-size delta emitted by one engine log."""
    log_type = _int_field(log, "type", -1)
    if log_type in {int(LogType.DRAW), int(LogType.DRAW_REVERSE)}:
        return -1
    if log_type not in {int(LogType.MOVE_CARD), int(LogType.MOVE_CARD_REVERSE)}:
        return 0
    return int(_optional_int(_field(log, "toArea")) == int(AreaType.DECK)) - int(
        _optional_int(_field(log, "fromArea")) == int(AreaType.DECK)
    )


def _visible_select_deck_counts(observation: Any, player_index: int) -> Counter[int]:
    select = _field(observation, "select")
    counts: Counter[int] = Counter()
    for card in _sequence(_field(select, "deck", ())):
        if _int_field(card, "playerIndex", player_index) == player_index:
            card_id = _int_field(card, "id", 0)
            if card_id > 0:
                counts[card_id] += 1
    return counts


def _visible_card_counts(
    state: Any,
    player_index: int,
    *,
    include_hand: bool,
) -> Counter[int]:
    counts: Counter[int] = Counter()
    _add_visible_cards(state, player_index, include_hand=include_hand, counts=counts)
    return counts


def _visible_cards_by_serial(
    state: Any,
    player_index: int,
    *,
    include_hand: bool,
) -> dict[int, int]:
    serials: dict[int, int] = {}
    _add_visible_cards(state, player_index, include_hand=include_hand, serials=serials)
    return serials


def _add_visible_cards(
    state: Any,
    player_index: int,
    *,
    include_hand: bool,
    counts: Counter[int] | None = None,
    serials: dict[int, int] | None = None,
) -> None:
    players = _sequence(_field(state, "players", ()))
    if player_index < 0 or player_index >= len(players):
        return
    player = players[player_index]
    for pokemon in _sequence(_field(player, "active", ())):
        if pokemon is not None:
            _add_pokemon(pokemon, counts, serials)
    for pokemon in _sequence(_field(player, "bench", ())):
        _add_pokemon(pokemon, counts, serials)
    for card in _sequence(_field(player, "discard", ())):
        _add_card(card, counts, serials)
    for card in _sequence(_field(player, "prize", ())):
        if card is not None:
            _add_card(card, counts, serials)
    if include_hand:
        for card in _sequence(_field(player, "hand", ())):
            _add_card(card, counts, serials)
    for card in _sequence(_field(state, "stadium", ())):
        if _int_field(card, "playerIndex", -1) == player_index:
            _add_card(card, counts, serials)
    for card in _sequence(_field(state, "looking", ())):
        if card is not None and _int_field(card, "playerIndex", -1) == player_index:
            _add_card(card, counts, serials)


def _add_pokemon(
    pokemon: Any,
    counts: Counter[int] | None,
    serials: dict[int, int] | None,
) -> None:
    _add_card(pokemon, counts, serials)
    for field_name in ("energyCards", "tools", "preEvolution"):
        for card in _sequence(_field(pokemon, field_name, ())):
            _add_card(card, counts, serials)


def _add_card(
    card: Any,
    counts: Counter[int] | None,
    serials: dict[int, int] | None,
) -> None:
    card_id = _int_field(card, "id", 0)
    if card_id <= 0:
        return
    if counts is not None:
        counts[card_id] += 1
    serial = _int_field(card, "serial", 0)
    if serials is not None and serial > 0:
        serials.setdefault(serial, card_id)


def _is_retreat_log(log: Any) -> bool:
    return _int_field(log, "fromArea", int(AreaType.ACTIVE)) == int(
        AreaType.ACTIVE
    ) and _int_field(log, "toArea", int(AreaType.BENCH)) == int(AreaType.BENCH)


def _stage_or_type(row: Mapping[str, str]) -> str:
    for key, value in row.items():
        if key.startswith("Stage ("):
            return value
    return ""


def _mapping_or_object(value: Any) -> Any | None:
    return None if value is None else value


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_field(value: Any, name: str, default: int) -> int:
    field_value = _field(value, name, default)
    return int(field_value) if field_value is not None else default


def _float_field(value: Any, name: str, default: float) -> float:
    field_value = _field(value, name, default)
    return float(field_value) if field_value is not None else default


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None
