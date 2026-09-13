"""Deterministic component-aware pilot for the immutable Slowking deck."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ptcg_rl.engine.constants import OptionType, SelectContext
from ptcg_rl.engine.protocols import ObservationInput
from ptcg_rl.opponents.slowking_copy import view
from ptcg_rl.opponents.slowking_copy.agent import (
    SlowkingCopyAgent as _SlowkingCopyV1Agent,
)
from ptcg_rl.opponents.slowking_copy_v2 import cards, tactics


class SlowkingCopyV2Agent(_SlowkingCopyV1Agent):
    """Slowking pilot that acts only on currently reachable combo components."""

    def __init__(self, name: str = cards.OPPONENT_NAME) -> None:
        super().__init__(name=name)
        self._preserve_hand_turn = -1
        self._preserved_line_card: int | None = None

    def reset(self) -> None:
        """Reset inherited intent plus v2's same-turn hand guard."""
        super().reset()
        self._preserve_hand_turn = -1
        self._preserved_line_card = None

    def begin_game(
        self,
        *,
        player_index: int | None = None,
        own_deck: Sequence[int] | None = None,
    ) -> None:
        """Bind the seat and reject substitution of the immutable v1 deck."""
        self.reset()
        if player_index is not None:
            self._memory.player_index = int(player_index)
        if own_deck is not None and Counter(own_deck) != Counter(cards.DECK):
            raise ValueError("slowking_copy_v2 requires its immutable exact deck")

    def act(self, observation: ObservationInput) -> Sequence[int]:
        """Return the v2 component-aware action for one engine prompt."""
        if isinstance(observation, Mapping) and observation.get("select") is None:
            self.reset()
            return cards.DECK
        return super().act(observation)

    def _special_action(self, observation: Any) -> tuple[int, ...] | None:
        context = int(observation.select.context)
        if context == int(SelectContext.SETUP_ACTIVE_POKEMON):
            return self._ranked_choice(
                observation,
                lambda option: tactics.setup_active_score(
                    observation,
                    self._memory.player_index,
                    option,
                ),
            )
        if context == int(SelectContext.SETUP_BENCH_POKEMON):
            return self._setup_bench_action(observation)
        if (
            context == int(SelectContext.TO_BENCH)
            and tactics.effect_card_id(observation)
            == cards.TELEPATH_PSYCHIC_ENERGY
        ):
            return self._telepath_bench_action(observation)
        if context in {
            int(SelectContext.SWITCH),
            int(SelectContext.TO_ACTIVE),
        }:
            return self._ranked_choice(
                observation,
                lambda option: tactics.switch_score(
                    observation,
                    self._memory.player_index,
                    option,
                ),
            )
        if context in {
            int(SelectContext.ATTACH_FROM),
            int(SelectContext.ATTACH_TO),
        }:
            return self._ranked_choice(
                observation,
                lambda option: tactics.attachment_prompt_score(
                    observation,
                    self._memory.player_index,
                    option,
                    source_card=self._memory.source_card,
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
        return super()._special_action(observation)

    def _main_action(self, observation: Any) -> tuple[int, ...] | None:
        options = tuple(observation.select.option)
        if not options:
            return ()
        seek_index = _attack_index(options, cards.SEEK_INSPIRATION)
        if (
            seek_index is not None
            and self._memory.known_top in cards.PAYLOAD_POKEMON
        ):
            self._memory.counters["STACKED_SEEK"] += 1
            return (seek_index,)

        scored: list[tuple[float, int]] = []
        for index, option in enumerate(options):
            score = tactics.main_action_score(
                observation,
                self._memory.player_index,
                option,
                seek_ready=seek_index is not None,
                known_top=self._memory.known_top,
            )
            if (
                self._preserve_hand_turn == self._memory.turn
                and self._preserved_line_card is not None
                and int(option.type) == int(OptionType.PLAY)
                and view.option_card_id(
                    observation,
                    self._memory.player_index,
                    option,
                )
                == cards.LILLIES_DETERMINATION
            ):
                score = -10000.0
            scored.append((score, index))
        _, index = max(scored, key=lambda item: (item[0], -item[1]))
        return (index,)

    def _fallback_main(self, observation: Any) -> tuple[int, ...]:
        action = self._main_action(observation)
        return action or ()

    def _copied_attack_action(self, observation: Any) -> tuple[int, ...] | None:
        payload = self._memory.copied_payload
        desired = {
            cards.CONKELDURR: cards.GUTSY_SWING,
            cards.KYUREM: cards.TRIFROST,
            cards.ANNIHILAPE: cards.DESTINED_FIGHT,
        }.get(payload or 0)
        options = tuple(observation.select.option)
        if desired is not None:
            desired_index = _attack_index(options, desired)
            if desired_index is not None:
                return (desired_index,)
        if not options:
            return ()
        _, index = max(
            (
                (tactics.copied_attack_score(option), index)
                for index, option in enumerate(options)
            ),
            key=lambda item: (item[0], -item[1]),
        )
        self._memory.counters["BLIND_COPY_ATTACK"] += 1
        return (index,)

    def _setup_bench_action(self, observation: Any) -> tuple[int, ...]:
        select = observation.select
        options = tuple(select.option)
        minimum, maximum = _selection_bounds(select, len(options))
        if maximum <= 0:
            return ()

        remaining_slots = tactics.bench_space(
            observation,
            self._memory.player_index,
        )
        optional_limit = min(maximum, max(0, remaining_slots - 1), 3)
        limit = max(minimum, optional_limit)
        chosen: list[int] = []

        if tactics.latias_needed(observation, self._memory.player_index):
            latias = _first_card_option(
                observation,
                self._memory.player_index,
                options,
                cards.LATIAS_EX,
            )
            if latias is not None and len(chosen) < limit:
                chosen.append(latias)

        line_room = max(
            0,
            2
            - tactics.slow_line_count(
                observation,
                self._memory.player_index,
            ),
        )
        for index in _card_option_indices(
            observation,
            self._memory.player_index,
            options,
            cards.SLOWPOKE,
        )[:line_room]:
            if len(chosen) >= limit:
                break
            chosen.append(index)

        # A lone 30-HP Smoochum is easily donked before Telepath or Kiss can
        # establish the line. Keep up to three Bench bodies when the opening
        # hand offers them, while still reserving two slots for later setup.
        survival_minimum = min(limit, 3)
        if len(chosen) < survival_minimum:
            ranked = sorted(
                (
                    (
                        _setup_survival_score(
                            observation,
                            self._memory.player_index,
                            option,
                        ),
                        index,
                    )
                    for index, option in enumerate(options)
                    if index not in chosen
                ),
                key=lambda item: (item[0], -item[1]),
                reverse=True,
            )
            chosen.extend(
                index
                for _, index in ranked[: survival_minimum - len(chosen)]
            )

        if len(chosen) < minimum:
            ranked = sorted(
                (
                    (
                        tactics.setup_bench_score(
                            observation,
                            self._memory.player_index,
                            option,
                        ),
                        index,
                    )
                    for index, option in enumerate(options)
                    if index not in chosen
                ),
                key=lambda item: (item[0], -item[1]),
                reverse=True,
            )
            chosen.extend(index for _, index in ranked[: minimum - len(chosen)])
        return tuple(chosen[:maximum])

    def _telepath_bench_action(self, observation: Any) -> tuple[int, ...]:
        select = observation.select
        options = tuple(select.option)
        minimum, maximum = _selection_bounds(select, len(options))
        line_room = max(
            0,
            2
            - tactics.slow_line_count(
                observation,
                self._memory.player_index,
            ),
        )
        chosen = list(
            _card_option_indices(
                observation,
                self._memory.player_index,
                options,
                cards.SLOWPOKE,
            )[: min(2, line_room, maximum)]
        )
        if len(chosen) < minimum:
            # The engine currently exposes this as an optional search. Preserve
            # legality if a future engine revision makes part of it mandatory.
            chosen.extend(
                index
                for index in range(len(options))
                if index not in chosen
            )
        self._memory.counters["TELEPATH_SLOWPOKE"] += min(2, len(chosen))
        return tuple(chosen[:maximum])

    def _remember_choice(
        self,
        observation: Any,
        action: Sequence[int],
    ) -> None:
        context = int(observation.select.context)
        if action and context in {
            int(SelectContext.TO_HAND),
            int(SelectContext.LOOK),
        }:
            selected_ids = {
                card_id
                for index in action
                if (
                    card_id := view.option_card_id(
                        observation,
                        self._memory.player_index,
                        observation.select.option[index],
                    )
                )
                is not None
            }
            selected_line = selected_ids & {cards.SLOWPOKE, cards.SLOWKING}
            if selected_line:
                self._preserve_hand_turn = self._memory.turn
                self._preserved_line_card = min(selected_line)
                self._memory.counters["SEARCHED_LINE_PRESERVED"] += 1

        super()._remember_choice(observation, action)
        if not action or context != int(SelectContext.MAIN):
            return
        option = observation.select.option[action[0]]
        option_type = int(option.type)
        card_id = view.option_card_id(
            observation,
            self._memory.player_index,
            option,
        )
        if (
            option_type in {int(OptionType.PLAY), int(OptionType.EVOLVE)}
            and card_id == self._preserved_line_card
        ):
            self._preserved_line_card = None
        if int(option.type) == int(OptionType.ATTACH):
            self._memory.source_card = card_id


# Keep the common class name for callers that treat script versions uniformly.
SlowkingCopyAgent = SlowkingCopyV2Agent


def build_slowking_copy_agent(
    name: str = cards.OPPONENT_NAME,
) -> SlowkingCopyV2Agent:
    """Build one game-local deterministic Slowking v2 opponent."""
    return SlowkingCopyV2Agent(name=name)


def _selection_bounds(select: Any, option_count: int) -> tuple[int, int]:
    minimum = min(
        option_count,
        max(0, _integer(getattr(select, "minCount", 0), 0)),
    )
    maximum = min(
        option_count,
        max(minimum, _integer(getattr(select, "maxCount", option_count), option_count)),
    )
    return minimum, maximum


def _attack_index(options: Sequence[Any], attack_id: int) -> int | None:
    for index, option in enumerate(options):
        if (
            int(option.type) == int(OptionType.ATTACK)
            and _integer(getattr(option, "attackId", None), -1) == attack_id
        ):
            return index
    return None


def _first_card_option(
    observation: Any,
    player_index: int,
    options: Sequence[Any],
    card_id: int,
) -> int | None:
    indices = _card_option_indices(
        observation,
        player_index,
        options,
        card_id,
    )
    return indices[0] if indices else None


def _card_option_indices(
    observation: Any,
    player_index: int,
    options: Sequence[Any],
    card_id: int,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, option in enumerate(options)
        if view.option_card_id(observation, player_index, option) == card_id
    )


def _setup_survival_score(
    observation: Any,
    player_index: int,
    option: Any,
) -> float:
    """Prefer a utility body over a third Slowpoke after the line quota."""
    score = tactics.setup_bench_score(observation, player_index, option)
    if view.option_card_id(observation, player_index, option) == cards.SLOWPOKE:
        return min(score, 600.0)
    return score


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
