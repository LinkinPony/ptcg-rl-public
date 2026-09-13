"""Deterministic engine-facing pilot for the Majkel Lopunny deck."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ptcg_rl.actions.selection import forced_action, is_legal_action
from ptcg_rl.engine.constants import OptionType, SelectContext
from ptcg_rl.engine.protocols import ObservationInput
from ptcg_rl.engine.runtime import to_engine_observation
from ptcg_rl.opponents.lopunny_dudunsparce import cards, oracle, tactics, view


@dataclass
class _IntentMemory:
    """Small causal state that is absent from the public observation."""

    turn: int = -1
    player_index: int = -1
    source_card: int | None = None
    source_attack: int | None = None
    gale_ready_serial: int | None = None
    blocked_target_serial: int | None = None
    run_away_count: int = 0
    counters: Counter[str] = field(default_factory=Counter)

    def reset(self) -> None:
        """Reset all per-game state."""
        self.turn = -1
        self.player_index = -1
        self.source_card = None
        self.source_attack = None
        self.gale_ready_serial = None
        self.blocked_target_serial = None
        self.run_away_count = 0
        self.counters.clear()


class LopunnyDudunsparceAgent:
    """Scripted exact-deck pilot centered on same-turn Gale Thrust loops."""

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
        """Bind the seat and reject substitution of the immutable deck."""
        self.reset()
        if player_index is not None:
            self._memory.player_index = int(player_index)
        if own_deck is not None and Counter(own_deck) != Counter(cards.DECK):
            raise ValueError("lopunny_dudunsparce_v1 requires its exact deck")

    def act(self, observation: ObservationInput) -> Sequence[int]:
        """Return one legal action from the engine's current option list."""
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
            self._memory.counters["INVALID_POLICY_FALLBACK"] += 1
            action = self._minimum_legal_action(select)
        self._remember_choice(engine_observation, action)
        return tuple(action)

    def diagnostics(self) -> dict[str, int]:
        """Return deterministic decision counters for debugging/evaluation."""
        return dict(sorted(self._memory.counters.items()))

    def _prepare(self, observation: Any) -> None:
        state = observation.current
        turn = int(state.turn)
        if self._memory.turn != turn:
            self._memory.turn = turn
            self._memory.source_card = None
            self._memory.source_attack = None
            self._memory.gale_ready_serial = None
            self._memory.run_away_count = 0
        self._memory.player_index = int(state.yourIndex)
        opponent = view.opponent_index(observation, self._memory.player_index)
        if self._memory.blocked_target_serial != view.serial(
            view.active(observation, opponent)
        ):
            self._memory.blocked_target_serial = None
        if int(observation.select.context) == int(SelectContext.MAIN):
            self._memory.source_card = None
            self._memory.source_attack = None

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
        if context == int(SelectContext.DRAW_COUNT):
            return self._ranked_choice(
                observation,
                lambda option: (
                    -abs(float(view.integer(getattr(option, "number", None), 0)) - 1.0)
                ),
            )
        if context == int(SelectContext.MAIN):
            return self._main_action(observation)
        if context in {int(SelectContext.SWITCH), int(SelectContext.TO_ACTIVE)}:
            return self._switch_action(observation)
        if context in {
            int(SelectContext.TO_BENCH),
            int(SelectContext.TO_HAND),
            int(SelectContext.LOOK),
        }:
            source = self._effect_source(observation)
            if source is not None:
                return self._search_action(observation, source)
        if context in {
            int(SelectContext.DISCARD),
            int(SelectContext.TO_DECK_BOTTOM),
        }:
            return self._ranked_choice(
                observation,
                lambda option: (
                    -tactics.keep_score(
                        observation,
                        self._memory.player_index,
                        option,
                    )
                ),
            )
        if context == int(SelectContext.DISCARD_ENERGY):
            return self._ranked_choice(
                observation,
                lambda option: tactics.retreat_energy_score(
                    observation,
                    self._memory.player_index,
                    option,
                ),
            )
        if context == int(SelectContext.HEAL):
            return self._ranked_choice(
                observation,
                lambda option: tactics.heal_score(
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
                ),
            )
        return None

    def _main_action(self, observation: Any) -> tuple[int, ...]:
        options = tuple(observation.select.option)
        if not options:
            return ()
        hopper_needed = self._hopper_needed(observation)
        attack_indices = tuple(
            index
            for index, option in enumerate(options)
            if int(option.type) == int(OptionType.ATTACK)
        )
        boss_indices = tuple(
            index
            for index, option in enumerate(options)
            if int(option.type) == int(OptionType.PLAY)
            and view.option_card_id(
                observation,
                self._memory.player_index,
                option,
            )
            == cards.BOSSES_ORDERS
        )
        base_scores = tuple(
            tactics.main_action_score(
                observation,
                self._memory.player_index,
                option,
                gale_ready_serial=self._memory.gale_ready_serial,
                run_away_count=self._memory.run_away_count,
                hopper_needed=hopper_needed,
            )
            for option in options
        )
        macro_index: int | None = None
        try:
            macro = oracle.best_same_turn_lopunny_attack(
                observation,
                self._memory.player_index,
            )
        except (RuntimeError, ValueError):
            self._memory.counters["TURN_MACRO_FALLBACK"] += 1
        else:
            if macro is not None and int(options[macro.root_option_index].type) != int(
                OptionType.ATTACK
            ):
                macro_index = macro.root_option_index
        base_index = max(
            range(len(options)),
            key=lambda index: (base_scores[index], -index),
        )
        best_attack: int | None = None
        outcomes: tuple[oracle.AttackOutcome, ...] = ()
        boss_competing = base_index in boss_indices or int(
            options[base_index].type
        ) == int(OptionType.ATTACK)
        if attack_indices and boss_competing:
            try:
                outcomes = oracle.attack_outcomes(
                    observation,
                    self._memory.player_index,
                    attack_indices,
                )
            except (RuntimeError, ValueError):
                self._memory.counters["ATTACK_ORACLE_FALLBACK"] += 1
            else:
                if outcomes:
                    best_attack = max(
                        outcomes,
                        key=lambda value: (value.utility, -value.option_index),
                    ).option_index
                    self._memory.counters["ATTACK_ORACLE"] += 1
        best_outcome = next(
            (outcome for outcome in outcomes if outcome.option_index == best_attack),
            None,
        )
        best_boss: oracle.BossOutcome | None = None
        if boss_indices and boss_competing:
            try:
                exact_boss = oracle.boss_outcomes(
                    observation,
                    self._memory.player_index,
                    boss_indices[:1],
                )
            except (RuntimeError, ValueError):
                self._memory.counters["BOSS_ORACLE_FALLBACK"] += 1
            else:
                if exact_boss:
                    best_boss = max(
                        exact_boss,
                        key=lambda value: (value.utility, -value.option_index),
                    )
                    self._memory.counters["BOSS_ORACLE"] += 1
        keep_active_target = bool(
            best_outcome is not None
            and best_boss is not None
            and best_outcome.utility >= best_boss.utility
        )
        winning_boss_index = (
            best_boss.option_index
            if best_boss is not None and best_boss.attack.terminal_win
            else None
        )
        scored = (
            (
                base_scores[index]
                + (15000.0 if index == macro_index else 0.0)
                + (250.0 if index == best_attack else 0.0)
                + (12000.0 if keep_active_target and index == best_attack else 0.0)
                + (12000.0 if index == winning_boss_index else 0.0),
                index,
            )
            for index in range(len(options))
        )
        score, index = max(scored, key=lambda item: (item[0], -item[1]))
        if macro_index is not None and index == macro_index:
            self._memory.counters["TURN_MACRO_SELECTED"] += 1
        if score < 0:
            end = _first_option_of_type(options, OptionType.END)
            if end:
                return end
        if int(options[index].type) == int(OptionType.ATTACK):
            outcome = next(
                (value for value in outcomes if value.option_index == index),
                None,
            )
            if outcome is not None:
                self._remember_attack_outcome(observation, options[index], outcome)
        return (index,)

    def _hopper_needed(self, observation: Any) -> bool:
        """Return whether native evidence says Gale cannot hit this target."""
        opponent = view.opponent_index(observation, self._memory.player_index)
        target = view.active(observation, opponent)
        active = view.active(observation, self._memory.player_index)
        return (
            self._memory.blocked_target_serial is not None
            and view.serial(target) == self._memory.blocked_target_serial
            and view.card_id(active) == cards.MEGA_LOPUNNY_EX
            and view.energy_count(active) == 1
        )

    def _remember_attack_outcome(
        self,
        observation: Any,
        option: Any,
        outcome: oracle.AttackOutcome,
    ) -> None:
        """Retain exact zero-damage evidence until the target leaves Active."""
        attack_id = view.integer(getattr(option, "attackId", None), -1)
        opponent = view.opponent_index(observation, self._memory.player_index)
        target_serial = view.serial(view.active(observation, opponent))
        if (
            attack_id == cards.LOPUNNY_GALE_THRUST
            and outcome.damage == 0
            and outcome.knockouts == 0
            and target_serial is not None
        ):
            self._memory.blocked_target_serial = target_serial
            self._memory.counters["GALE_BLOCKED"] += 1
        elif outcome.damage > 0 or outcome.knockouts > 0:
            self._memory.blocked_target_serial = None

    def _setup_bench_action(self, observation: Any) -> tuple[int, ...]:
        select = observation.select
        options = tuple(select.option)
        minimum, maximum = _selection_bounds(select, len(options))
        if maximum <= 0:
            return ()

        # Every Basic in this exact deck advances a live evolution or setup
        # role, and public replays consistently place all legal opening bodies.
        limit = maximum
        chosen: list[int] = []
        active_id = view.card_id(view.active(observation, self._memory.player_index))

        def take(card_id: int, count: int) -> None:
            for index in self._card_option_indices(observation, options, card_id):
                if len(chosen) >= limit or count <= 0:
                    break
                if index not in chosen:
                    chosen.append(index)
                    count -= 1

        if active_id != cards.FAN_ROTOM:
            take(cards.FAN_ROTOM, 1)
        take(cards.BUNEARY, 2 if active_id != cards.BUNEARY else 1)
        take(cards.DUNSPARCE, 3 if active_id != cards.DUNSPARCE else 2)
        self._fill_ranked_setup(observation, options, chosen, limit)
        return tuple(chosen[:limit])

    def _switch_action(self, observation: Any) -> tuple[int, ...]:
        if self._memory.source_card == cards.BOSSES_ORDERS:
            option_indices = tuple(range(len(observation.select.option)))
            try:
                best_target = oracle.best_boss_target_index(
                    observation,
                    self._memory.player_index,
                    option_indices,
                )
            except (RuntimeError, ValueError):
                self._memory.counters["BOSS_TARGET_ORACLE_FALLBACK"] += 1
            else:
                if best_target is not None:
                    self._memory.counters["BOSS_TARGET_ORACLE"] += 1
                    return (best_target,)
        chain_run_away = self._memory.source_card == cards.DUDUNSPARCE
        forced_promotion = int(observation.select.context) == int(
            SelectContext.TO_ACTIVE
        )
        return self._ranked_choice(
            observation,
            lambda option: tactics.switch_score(
                observation,
                self._memory.player_index,
                option,
                chain_run_away=chain_run_away,
                forced_promotion=forced_promotion,
            ),
        )

    def _search_action(self, observation: Any, source: int) -> tuple[int, ...]:
        if source in {cards.FAN_ROTOM, cards.BUDDY_BUDDY_POFFIN}:
            return self._balanced_basic_search(observation, source)
        return self._ranked_choice(
            observation,
            lambda option: tactics.search_score(
                observation,
                self._memory.player_index,
                source,
                option,
            ),
            positive_only=True,
        )

    def _balanced_basic_search(
        self,
        observation: Any,
        source: int,
    ) -> tuple[int, ...]:
        select = observation.select
        options = tuple(select.option)
        minimum, maximum = _selection_bounds(select, len(options))
        chosen: list[int] = []

        def take(card_id: int) -> None:
            for index in self._card_option_indices(observation, options, card_id):
                if index not in chosen:
                    chosen.append(index)
                    return

        if source == cards.BUDDY_BUDDY_POFFIN:
            state = observation.current
            if (
                int(state.turn) <= 2
                and view.field_count(
                    observation, self._memory.player_index, cards.FAN_ROTOM
                )
                == 0
            ):
                take(cards.FAN_ROTOM)
            if tactics.lopunny_line_count(observation, self._memory.player_index) < 1:
                take(cards.BUNEARY)
            if tactics.dunsparce_line_count(observation, self._memory.player_index) < 2:
                take(cards.DUNSPARCE)
            if tactics.lopunny_line_count(observation, self._memory.player_index) < 2:
                take(cards.BUNEARY)
            take(cards.DUNSPARCE)
        else:
            take(cards.BUNEARY)
            take(cards.DUNSPARCE)
            take(cards.BUNEARY)
            take(cards.DUNSPARCE)

        ranked = sorted(
            (
                (
                    tactics.search_score(
                        observation,
                        self._memory.player_index,
                        source,
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
        for score, index in ranked:
            if len(chosen) >= maximum or (score <= 0 and len(chosen) >= minimum):
                break
            chosen.append(index)
        return tuple(chosen[:maximum])

    def _fallback_action(self, observation: Any) -> tuple[int, ...]:
        context = int(observation.select.context)
        if context == int(SelectContext.IS_FIRST):
            return _first_option_of_type(observation.select.option, OptionType.YES)
        if context in {
            int(SelectContext.MULLIGAN),
            int(SelectContext.ACTIVATE),
            int(SelectContext.FIRST_EFFECT),
            int(SelectContext.COIN_HEAD),
        }:
            return _first_option_of_type(observation.select.option, OptionType.YES)
        if context == int(SelectContext.MORE_DEVOLVE):
            return _first_option_of_type(observation.select.option, OptionType.NO)
        if context == int(SelectContext.MAIN):
            return self._main_action(observation)
        return self._ranked_choice(observation, self._generic_option_score)

    def _generic_option_score(self, option: Any) -> float:
        option_type = view.integer(getattr(option, "type", None), -1)
        if option_type == int(OptionType.YES):
            return 2000.0
        if option_type == int(OptionType.NO):
            return 1000.0
        number = view.integer(getattr(option, "number", None))
        if number is not None:
            return float(number)
        return 0.0

    def _ranked_choice(
        self,
        observation: Any,
        scorer: Callable[[Any], float],
        *,
        positive_only: bool = False,
    ) -> tuple[int, ...]:
        select = observation.select
        options = tuple(select.option)
        minimum, maximum = _selection_bounds(select, len(options))
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

    def _effect_source(self, observation: Any) -> int | None:
        return self._memory.source_card or view.effect_card_id(observation)

    def _remember_choice(self, observation: Any, action: Sequence[int]) -> None:
        if not action:
            return
        select = observation.select
        context = int(select.context)
        options = tuple(select.option)
        selected = [options[index] for index in action]
        if context == int(SelectContext.MAIN):
            option = selected[0]
            option_type = int(option.type)
            card_id = view.option_card_id(
                observation,
                self._memory.player_index,
                option,
            )
            if option_type == int(OptionType.ATTACK):
                action_id = view.integer(getattr(option, "attackId", None), 0)
            else:
                action_id = card_id or 0
            try:
                option_name = OptionType(option_type).name
            except ValueError:
                option_name = f"TYPE_{option_type}"
            self._memory.counters[f"MAIN_{option_name}_{action_id}"] += 1
            if option_type in {int(OptionType.PLAY), int(OptionType.ABILITY)}:
                self._memory.source_card = card_id
            if option_type == int(OptionType.ABILITY) and card_id == cards.DUDUNSPARCE:
                self._memory.run_away_count += 1
            if option_type == int(OptionType.ATTACK):
                self._memory.source_attack = view.integer(
                    getattr(option, "attackId", None),
                    -1,
                )
            return
        if context not in {int(SelectContext.SWITCH), int(SelectContext.TO_ACTIVE)}:
            return
        target_option = selected[0]
        owner = view.integer(
            getattr(target_option, "playerIndex", None),
            self._memory.player_index,
        )
        if owner != self._memory.player_index:
            return
        target = view.option_pokemon(
            observation,
            self._memory.player_index,
            target_option,
        )
        if view.card_id(target) == cards.MEGA_LOPUNNY_EX:
            self._memory.gale_ready_serial = view.serial(target)
            self._memory.counters["GALE_PROMOTION"] += 1

    def _card_option_indices(
        self,
        observation: Any,
        options: Sequence[Any],
        card_id: int,
    ) -> tuple[int, ...]:
        return tuple(
            index
            for index, option in enumerate(options)
            if view.option_card_id(
                observation,
                self._memory.player_index,
                option,
            )
            == card_id
        )

    def _fill_ranked_setup(
        self,
        observation: Any,
        options: Sequence[Any],
        chosen: list[int],
        limit: int,
    ) -> None:
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
        chosen.extend(index for _, index in ranked[: max(0, limit - len(chosen))])

    @staticmethod
    def _minimum_legal_action(select: Any) -> tuple[int, ...]:
        count = min(
            len(select.option),
            max(0, int(getattr(select, "minCount", 0) or 0)),
        )
        return tuple(range(count))


def build_lopunny_dudunsparce_agent(
    name: str = cards.OPPONENT_NAME,
) -> LopunnyDudunsparceAgent:
    """Build one fresh deterministic exact-deck opponent."""
    return LopunnyDudunsparceAgent(name=name)


def _selection_bounds(select: Any, option_count: int) -> tuple[int, int]:
    minimum = min(option_count, max(0, int(getattr(select, "minCount", 0) or 0)))
    maximum = min(
        option_count,
        max(minimum, int(getattr(select, "maxCount", option_count) or 0)),
    )
    return minimum, maximum


def _first_option_of_type(
    options: Sequence[Any],
    option_type: OptionType,
) -> tuple[int, ...]:
    for index, option in enumerate(options):
        if int(option.type) == int(option_type):
            return (index,)
    return (0,) if options else ()
