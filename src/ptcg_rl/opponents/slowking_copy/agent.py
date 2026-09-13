"""Deterministic public-observation pilot for Slowking Copy Engine."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ptcg_rl.actions.selection import forced_action, is_legal_action
from ptcg_rl.engine.constants import LogType, OptionType, SelectContext
from ptcg_rl.engine.protocols import ObservationInput
from ptcg_rl.engine.runtime import to_engine_observation
from ptcg_rl.opponents.slowking_copy import cards, tactics, view


@dataclass
class _IntentMemory:
    """Short-lived state needed to finish multi-prompt engine effects."""

    turn: int = -1
    player_index: int = -1
    source_card: int | None = None
    source_attack: int | None = None
    copied_attack: int | None = None
    copied_payload: int | None = None
    known_top: int | None = None
    pending_top: int | None = None
    counters: Counter[str] = field(default_factory=Counter)

    def reset(self) -> None:
        """Reset all per-game state."""
        self.turn = -1
        self.player_index = -1
        self.source_card = None
        self.source_attack = None
        self.copied_attack = None
        self.copied_payload = None
        self.known_top = None
        self.pending_top = None
        self.counters.clear()


class SlowkingCopyAgent:
    """Scripted Slowking pilot with explicit stack/copy intent memory."""

    def __init__(self, name: str = cards.OPPONENT_NAME) -> None:
        self.name = name
        self._memory = _IntentMemory()

    def reset(self) -> None:
        """Reset state before a fresh battle."""
        self._memory.reset()

    def begin_game(
        self,
        *,
        player_index: int | None = None,
        own_deck: Sequence[int] | None = None,
    ) -> None:
        """Bind the expected seat and reject accidental deck substitution."""
        self.reset()
        if player_index is not None:
            self._memory.player_index = int(player_index)
        if own_deck is not None and Counter(own_deck) != Counter(cards.DECK):
            raise ValueError("slowking_copy_v1 requires its immutable exact deck")

    def act(self, observation: ObservationInput) -> Sequence[int]:
        """Choose a legal option using the Slowking combo state machine."""
        if isinstance(observation, Mapping) and observation.get("select") is None:
            self.reset()
            return cards.DECK
        engine_observation = to_engine_observation(observation)
        select = getattr(engine_observation, "select", None)
        if select is None:
            self.reset()
            return cards.DECK

        self._prepare(engine_observation)
        action = self._special_action(engine_observation)
        if action is None:
            action = forced_action(select)
        if action is None:
            action = self._fallback_action(engine_observation)
        if not is_legal_action(select, action):
            self._memory.counters["INVALID_SPECIAL_FALLBACK"] += 1
            action = self._minimum_legal_action(select)
        self._remember_choice(engine_observation, action)
        return tuple(action)

    def diagnostics(self) -> dict[str, int]:
        """Return deterministic per-game decision counters."""
        return dict(sorted(self._memory.counters.items()))

    def _prepare(self, observation: Any) -> None:
        state = observation.current
        player_index = int(state.yourIndex)
        turn = int(state.turn)
        if self._memory.turn != turn:
            self._memory.turn = turn
            self._memory.source_card = None
            self._memory.source_attack = None
            self._memory.copied_attack = None
            self._memory.copied_payload = None
            self._memory.known_top = None
            self._memory.pending_top = None
        self._memory.player_index = player_index

        for log in getattr(observation, "logs", ()) or ():
            if _integer(getattr(log, "playerIndex", None), -1) != player_index:
                continue
            log_type = _integer(getattr(log, "type", None), -1)
            if log_type in {int(LogType.DRAW), int(LogType.SHUFFLE)}:
                self._memory.known_top = None

        if self._memory.pending_top is not None:
            self._memory.known_top = self._memory.pending_top
            self._memory.pending_top = None
            self._memory.counters["TOP_STACK_COMMITTED"] += 1

        if int(observation.select.context) == int(SelectContext.MAIN):
            self._memory.source_card = None
            self._memory.source_attack = None

    def _special_action(self, observation: Any) -> tuple[int, ...] | None:
        select = observation.select
        context = int(select.context)
        if context == int(SelectContext.SETUP_ACTIVE_POKEMON):
            return self._ranked_choice(
                observation,
                lambda option: tactics.setup_score(
                    observation,
                    self._memory.player_index,
                    option,
                    active_slot=True,
                ),
            )
        if context == int(SelectContext.SETUP_BENCH_POKEMON):
            return self._ranked_choice(
                observation,
                lambda option: tactics.setup_score(
                    observation,
                    self._memory.player_index,
                    option,
                    active_slot=False,
                ),
                positive_only=True,
            )
        if context == int(SelectContext.MAIN):
            return self._main_action(observation)
        if context == int(SelectContext.TO_DECK):
            if self._memory.source_card == cards.ACADEMY_AT_NIGHT:
                return self._academy_stack_action(observation)
            if self._memory.source_card == cards.CIPHERMANIAC:
                return self._cipher_stack_action(observation)
        if (
            context == int(SelectContext.ATTACK)
            and self._memory.source_attack == cards.SEEK_INSPIRATION
        ):
            return self._copied_attack_action(observation)
        if self._memory.copied_attack == cards.TRIFROST and context in {
            int(SelectContext.DAMAGE),
            int(SelectContext.EFFECT_TARGET),
        }:
            return self._trifrost_targets(observation)
        if context in {
            int(SelectContext.SWITCH),
            int(SelectContext.TO_ACTIVE),
        }:
            return self._ranked_choice(
                observation,
                lambda option: tactics.switch_score(
                    observation, self._memory.player_index, option
                ),
            )
        if context in {
            int(SelectContext.ATTACH_FROM),
            int(SelectContext.ATTACH_TO),
        }:
            return self._ranked_choice(
                observation,
                lambda option: tactics.attach_target_score(
                    observation, self._memory.player_index, option
                ),
            )
        if (
            context
            in {
                int(SelectContext.TO_HAND),
                int(SelectContext.LOOK),
            }
            and self._memory.source_card is not None
        ):
            return self._ranked_choice(
                observation,
                lambda option: tactics.search_score(
                    observation,
                    self._memory.player_index,
                    self._memory.source_card or 0,
                    option,
                ),
                positive_only=True,
            )
        if context in {
            int(SelectContext.DISCARD),
            int(SelectContext.TO_DECK_BOTTOM),
        }:
            return self._ranked_choice(
                observation,
                lambda option: (
                    -tactics.keep_score(observation, self._memory.player_index, option)
                ),
            )
        return None

    def _main_action(self, observation: Any) -> tuple[int, ...] | None:
        select = observation.select
        options = tuple(select.option)
        seek = _first_attack(options, cards.SEEK_INSPIRATION)
        if seek is not None and self._memory.known_top in cards.PAYLOAD_POKEMON:
            self._memory.counters["STACKED_SEEK"] += 1
            return (seek,)

        scored: list[tuple[float, int]] = []
        for index, option in enumerate(options):
            score = tactics.main_combo_score(
                observation,
                self._memory.player_index,
                option,
                seek_ready=seek is not None,
            )
            if score is not None:
                scored.append((score, index))
        if not scored:
            return None
        score, index = max(scored, key=lambda item: (item[0], -item[1]))
        if score <= 0:
            return None
        return (index,)

    def _academy_stack_action(self, observation: Any) -> tuple[int, ...] | None:
        payloads = self._payload_options(observation)
        if not payloads:
            return None
        _, index, _ = max(payloads, key=lambda item: (item[0], -item[1]))
        return (index,)

    def _cipher_stack_action(self, observation: Any) -> tuple[int, ...] | None:
        select = observation.select
        option_count = len(select.option)
        minimum = min(option_count, max(0, int(select.minCount)))
        maximum = min(option_count, max(minimum, int(select.maxCount)))
        if maximum <= 0:
            return ()

        payloads = sorted(
            self._payload_options(observation),
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        if not payloads:
            return None
        immediate = payloads[0]
        remaining = [
            (
                tactics.next_draw_score(observation, self._memory.player_index, option),
                index,
                view.option_card_id(observation, self._memory.player_index, option),
            )
            for index, option in enumerate(select.option)
            if index != immediate[1]
        ]
        remaining.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        count = max(minimum, min(maximum, 2))
        chosen = [item[1] for item in remaining[: max(0, count - 1)]]
        chosen.append(immediate[1])
        if len(chosen) < minimum:
            return None
        # Engine ToDeckReverse inserts selected cards in order; the last selected
        # card becomes the top card. Keep the immediate copy payload last.
        return tuple(chosen)

    def _copied_attack_action(self, observation: Any) -> tuple[int, ...] | None:
        payload = self._memory.copied_payload
        desired = {
            cards.CONKELDURR: cards.GUTSY_SWING,
            cards.KYUREM: cards.TRIFROST,
            cards.ANNIHILAPE: cards.DESTINED_FIGHT,
        }.get(payload or 0)
        if desired is None:
            return None
        index = _first_attack(tuple(observation.select.option), desired)
        return (index,) if index is not None else None

    def _trifrost_targets(self, observation: Any) -> tuple[int, ...]:
        return self._ranked_choice(
            observation,
            lambda option: tactics.trifrost_target_score(
                observation, self._memory.player_index, option
            ),
        )

    def _payload_options(
        self,
        observation: Any,
    ) -> list[tuple[float, int, int]]:
        output: list[tuple[float, int, int]] = []
        for index, option in enumerate(observation.select.option):
            card_id = view.option_card_id(
                observation, self._memory.player_index, option
            )
            if card_id in cards.PAYLOAD_POKEMON:
                output.append(
                    (
                        tactics.payload_score(
                            observation, self._memory.player_index, card_id
                        ),
                        index,
                        card_id,
                    )
                )
        return output

    def _fallback_action(self, observation: Any) -> tuple[int, ...]:
        select = observation.select
        context = int(select.context)
        if context == int(SelectContext.IS_FIRST):
            return _first_option_of_type(select.option, OptionType.NO)
        if context in {
            int(SelectContext.MULLIGAN),
            int(SelectContext.ACTIVATE),
            int(SelectContext.FIRST_EFFECT),
            int(SelectContext.COIN_HEAD),
        }:
            return _first_option_of_type(select.option, OptionType.YES)
        if context == int(SelectContext.MORE_DEVOLVE):
            return _first_option_of_type(select.option, OptionType.NO)
        if context == int(SelectContext.MAIN):
            return self._fallback_main(observation)
        return self._ranked_choice(
            observation,
            lambda option: tactics.generic_option_score(
                observation, self._memory.player_index, option
            ),
        )

    def _fallback_main(self, observation: Any) -> tuple[int, ...]:
        scored: list[tuple[float, int]] = []
        for index, option in enumerate(observation.select.option):
            score = tactics.generic_main_score(
                observation, self._memory.player_index, option
            )
            if (
                _integer(getattr(option, "attackId", None), -1)
                == cards.SEEK_INSPIRATION
                and self._memory.known_top is not None
                and self._memory.known_top not in cards.PAYLOAD_POKEMON
            ):
                score = -10.0
            scored.append((score, index))
        _, index = max(scored, key=lambda item: (item[0], -item[1]))
        return (index,)

    def _ranked_choice(
        self,
        observation: Any,
        scorer: Callable[[Any], float],
        *,
        positive_only: bool = False,
    ) -> tuple[int, ...]:
        select = observation.select
        options = tuple(select.option)
        minimum = min(len(options), max(0, int(select.minCount)))
        maximum = min(len(options), max(minimum, int(select.maxCount)))
        ranked = sorted(
            ((scorer(option), index) for index, option in enumerate(options)),
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        if positive_only and minimum == 0:
            count = min(maximum, sum(score > 0 for score, _ in ranked))
        else:
            count = maximum
        count = max(minimum, count)
        return tuple(index for _, index in ranked[:count])

    def _remember_choice(
        self,
        observation: Any,
        action: Sequence[int],
    ) -> None:
        select = observation.select
        if not action:
            return
        context = int(select.context)
        options = tuple(select.option)
        selected = [options[index] for index in action]
        if context == int(SelectContext.MAIN):
            option = selected[0]
            option_type = int(option.type)
            card_id = view.option_card_id(
                observation, self._memory.player_index, option
            )
            if option_type in {int(OptionType.PLAY), int(OptionType.ABILITY)}:
                self._memory.source_card = card_id
            if option_type == int(OptionType.ATTACK):
                attack_id = _integer(getattr(option, "attackId", None), -1)
                self._memory.source_attack = attack_id
                if attack_id == cards.SEEK_INSPIRATION:
                    self._memory.copied_payload = self._memory.known_top
                    if self._memory.copied_payload == cards.KYUREM:
                        self._memory.copied_attack = cards.TRIFROST
                    self._memory.known_top = None
            return
        if context == int(SelectContext.TO_DECK):
            selected_ids = [
                view.option_card_id(observation, self._memory.player_index, option)
                for option in selected
            ]
            if self._memory.source_card == cards.ACADEMY_AT_NIGHT:
                self._memory.pending_top = selected_ids[0]
                if selected_ids[0] in cards.PAYLOAD_POKEMON:
                    self._memory.counters["ACADEMY_STACK"] += 1
            elif self._memory.source_card == cards.CIPHERMANIAC:
                self._memory.pending_top = selected_ids[-1]
                if selected_ids[-1] in cards.PAYLOAD_POKEMON:
                    self._memory.counters["CIPHER_STACK"] += 1
            return
        if (
            context == int(SelectContext.ATTACK)
            and self._memory.source_attack == cards.SEEK_INSPIRATION
        ):
            self._memory.copied_attack = _integer(
                getattr(selected[0], "attackId", None), -1
            )

    @staticmethod
    def _minimum_legal_action(select: Any) -> tuple[int, ...]:
        count = min(
            len(select.option),
            max(0, int(getattr(select, "minCount", 0) or 0)),
        )
        return tuple(range(count))


def build_slowking_copy_agent(
    name: str = cards.OPPONENT_NAME,
) -> SlowkingCopyAgent:
    """Build one game-local deterministic Slowking opponent."""
    return SlowkingCopyAgent(name=name)


def _first_attack(options: Sequence[Any], attack_id: int) -> int | None:
    for index, option in enumerate(options):
        if (
            int(option.type) == int(OptionType.ATTACK)
            and _integer(getattr(option, "attackId", None), -1) == attack_id
        ):
            return index
    return None


def _first_option_of_type(
    options: Sequence[Any],
    option_type: OptionType,
) -> tuple[int, ...]:
    for index, option in enumerate(options):
        if int(option.type) == int(option_type):
            return (index,)
    return (0,) if options else ()


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
