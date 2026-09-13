"""Bounded native forward checks for terminal Lopunny attack choices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ptcg_rl.actions.selection import forced_action
from ptcg_rl.engine.constants import AreaType, LogType, OptionType, SelectContext
from ptcg_rl.engine.session import HiddenInformation, SearchSession
from ptcg_rl.opponents.lopunny_dudunsparce import cards, view


@dataclass(frozen=True)
class AttackOutcome:
    """Immediate engine consequence of one legal attack selection."""

    option_index: int
    damage: int
    knockouts: int
    prizes_taken: int
    terminal_win: bool

    @property
    def utility(self) -> int:
        """Rank exact immediate consequences without a printed-damage model."""
        return (
            self.damage
            + 1000 * self.knockouts
            + 10000 * self.prizes_taken
            + 100000 * self.terminal_win
        )


@dataclass(frozen=True)
class BossOutcome:
    """Best exact attack continuation after one Boss play option."""

    option_index: int
    attack: AttackOutcome

    @property
    def utility(self) -> int:
        """Expose the continuation utility for a root action comparison."""
        return self.attack.utility


@dataclass(frozen=True)
class SameTurnAttackPlan:
    """Best native Lopunny attack reachable from one root MAIN action."""

    root_option_index: int
    attack: AttackOutcome
    attack_id: int
    decision_steps: int

    @property
    def key(self) -> tuple[int, int, int]:
        """Prefer exact consequence, damage, then the shortest sequence."""
        return self.attack.utility, self.attack.damage, -self.decision_steps


_MACRO_NODE_CAP = 96
_MACRO_DEPTH_CAP = 8


def best_attack_index(
    observation: Any,
    player_index: int,
    option_indices: tuple[int, ...],
) -> int | None:
    """Use the native forward model to distinguish legal terminal attacks."""
    if len(option_indices) < 2:
        return None
    outcomes = attack_outcomes(observation, player_index, option_indices)
    if not outcomes:
        return None
    return max(
        outcomes, key=lambda value: (value.utility, -value.option_index)
    ).option_index


def attack_outcomes(
    observation: Any,
    player_index: int,
    option_indices: tuple[int, ...],
) -> tuple[AttackOutcome, ...]:
    """Evaluate exact immediate outcomes for the supplied legal attacks."""
    if not option_indices or getattr(observation, "search_begin_input", None) is None:
        return ()
    hidden = _placeholder_hidden(observation, player_index)
    before_prizes = view.prize_count(observation, player_index)
    outcomes: list[AttackOutcome] = []
    with SearchSession.begin(observation, hidden) as search:
        root_id = int(search.root.searchId)
        for option_index in option_indices:
            successor = search.step(root_id, (option_index,))
            try:
                outcomes.append(
                    _outcome(
                        successor.observation,
                        player_index=player_index,
                        option_index=option_index,
                        before_prizes=before_prizes,
                    )
                )
            finally:
                search.release(int(successor.searchId))
    return tuple(outcomes)


def best_same_turn_lopunny_attack(
    observation: Any,
    player_index: int,
) -> SameTurnAttackPlan | None:
    """Find a bounded native sequence ending in a Lopunny attack this turn."""
    if getattr(observation, "search_begin_input", None) is None:
        return None
    hidden = _placeholder_hidden(observation, player_index)
    before_prizes = view.prize_count(observation, player_index)
    start_turn = view.integer(
        getattr(getattr(observation, "current", None), "turn", None),
        -1,
    )
    budget = [0]
    plans: list[SameTurnAttackPlan] = []
    with SearchSession.begin(observation, hidden) as search:
        root_id = int(search.root.searchId)
        options = view.as_sequence(observation.select.option)
        for root_index in _macro_main_indices(observation, player_index):
            if budget[0] >= _MACRO_NODE_CAP:
                break
            budget[0] += 1
            successor = search.step(root_id, (root_index,))
            try:
                plans.extend(
                    _walk_same_turn(
                        search,
                        successor,
                        player_index=player_index,
                        start_turn=start_turn,
                        before_prizes=before_prizes,
                        root_option_index=root_index,
                        decision_steps=1,
                        budget=budget,
                        selected_attack_id=_option_attack_id(options[root_index]),
                    )
                )
            finally:
                search.release(int(successor.searchId))
    if not plans:
        return None
    return max(plans, key=lambda plan: (plan.key, -plan.root_option_index))


def _walk_same_turn(
    search: SearchSession,
    node: Any,
    *,
    player_index: int,
    start_turn: int,
    before_prizes: int,
    root_option_index: int,
    decision_steps: int,
    budget: list[int],
    selected_attack_id: int,
) -> list[SameTurnAttackPlan]:
    """Depth-first native expansion under strict depth and node caps."""
    observation = node.observation
    if selected_attack_id in {
        cards.LOPUNNY_GALE_THRUST,
        cards.LOPUNNY_SPIKY_HOPPER,
    }:
        return [
            SameTurnAttackPlan(
                root_option_index=root_option_index,
                attack=_outcome(
                    observation,
                    player_index=player_index,
                    option_index=-1,
                    before_prizes=before_prizes,
                ),
                attack_id=selected_attack_id,
                decision_steps=decision_steps,
            )
        ]
    if decision_steps >= _MACRO_DEPTH_CAP or budget[0] >= _MACRO_NODE_CAP:
        return []
    state = getattr(observation, "current", None)
    if (
        view.integer(getattr(state, "turn", None), -1) != start_turn
        or view.integer(getattr(state, "yourIndex", None), -1) != player_index
        or view.integer(getattr(state, "result", None), -1) >= 0
    ):
        return []
    select = getattr(observation, "select", None)
    if select is None:
        return []
    context = view.integer(getattr(select, "context", None), -1)
    actions: tuple[tuple[int, ...], ...]
    if context == int(SelectContext.MAIN):
        actions = tuple(
            (index,) for index in _macro_main_indices(observation, player_index)
        )
    else:
        actions = _macro_prompt_actions(select)
    plans: list[SameTurnAttackPlan] = []
    options = view.as_sequence(getattr(select, "option", ()))
    for action in actions:
        if budget[0] >= _MACRO_NODE_CAP:
            break
        budget[0] += 1
        child = search.step(int(node.searchId), action)
        try:
            attack_id = (
                _option_attack_id(options[action[0]])
                if context == int(SelectContext.MAIN) and action
                else -1
            )
            plans.extend(
                _walk_same_turn(
                    search,
                    child,
                    player_index=player_index,
                    start_turn=start_turn,
                    before_prizes=before_prizes,
                    root_option_index=root_option_index,
                    decision_steps=decision_steps + 1,
                    budget=budget,
                    selected_attack_id=attack_id,
                )
            )
        finally:
            search.release(int(child.searchId))
    return plans


def _macro_main_indices(observation: Any, player_index: int) -> tuple[int, ...]:
    """Keep only engine actions that can form a bounded Lopunny attack macro."""
    select = observation.select
    ranked: list[tuple[int, int]] = []
    for index, option in enumerate(view.as_sequence(select.option)):
        option_type = view.integer(getattr(option, "type", None), -1)
        card_id = view.option_card_id(observation, player_index, option)
        target = view.option_pokemon(observation, player_index, option)
        if option_type == int(OptionType.ATTACK):
            if _option_attack_id(option) in {
                cards.LOPUNNY_GALE_THRUST,
                cards.LOPUNNY_SPIKY_HOPPER,
            }:
                ranked.append((0, index))
        elif option_type == int(OptionType.EVOLVE) and card_id in {
            cards.MEGA_LOPUNNY_EX,
            cards.DUDUNSPARCE,
        }:
            priority = 1 if card_id == cards.MEGA_LOPUNNY_EX else 3
            ranked.append((priority, index))
        elif option_type == int(OptionType.ABILITY) and card_id == cards.DUDUNSPARCE:
            ranked.append((4, index))
        elif option_type == int(OptionType.ATTACH):
            target_id = view.card_id(target)
            if target_id in {cards.BUNEARY, cards.MEGA_LOPUNNY_EX}:
                ranked.append((2, index))
            elif card_id == cards.AIR_BALLOON:
                ranked.append((5, index))
        elif option_type == int(OptionType.RETREAT):
            ranked.append((6, index))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return tuple(index for _, index in ranked[:16])


def _macro_prompt_actions(select: Any) -> tuple[tuple[int, ...], ...]:
    """Enumerate bounded legal continuations for a non-MAIN prompt."""
    forced = forced_action(select)
    if forced is not None:
        return (forced,)
    options = view.as_sequence(getattr(select, "option", ()))
    option_count = len(options)
    minimum = min(
        option_count,
        max(0, view.integer(getattr(select, "minCount", None), 0) or 0),
    )
    maximum = min(
        option_count,
        max(minimum, view.integer(getattr(select, "maxCount", None), option_count)),
    )
    if maximum <= 1:
        actions = tuple((index,) for index in range(option_count))
        return ((),) + actions if minimum == 0 else actions
    return (tuple(range(minimum)),)


def _option_attack_id(option: Any) -> int:
    return view.integer(getattr(option, "attackId", None), -1)


def best_boss_target_index(
    observation: Any,
    player_index: int,
    option_indices: tuple[int, ...],
) -> int | None:
    """Select the Boss target with the best exact attack continuation."""
    if not option_indices or getattr(observation, "search_begin_input", None) is None:
        return None
    hidden = _placeholder_hidden(observation, player_index)
    before_prizes = view.prize_count(observation, player_index)
    scored_targets: list[tuple[int, int]] = []
    with SearchSession.begin(observation, hidden) as search:
        root_id = int(search.root.searchId)
        for target_index in option_indices:
            switched = search.step(root_id, (target_index,))
            try:
                select = getattr(switched.observation, "select", None)
                attack_indices = tuple(
                    index
                    for index, option in enumerate(
                        view.as_sequence(getattr(select, "option", ()))
                    )
                    if view.integer(getattr(option, "type", None), -1)
                    == int(OptionType.ATTACK)
                )
                target_outcomes: list[AttackOutcome] = []
                for attack_index in attack_indices:
                    attacked = search.step(int(switched.searchId), (attack_index,))
                    try:
                        target_outcomes.append(
                            _outcome(
                                attacked.observation,
                                player_index=player_index,
                                option_index=attack_index,
                                before_prizes=before_prizes,
                            )
                        )
                    finally:
                        search.release(int(attacked.searchId))
                if target_outcomes:
                    best = max(
                        target_outcomes,
                        key=lambda value: (value.utility, -value.option_index),
                    )
                    scored_targets.append((best.utility, target_index))
            finally:
                search.release(int(switched.searchId))
    if not scored_targets:
        return None
    return max(scored_targets, key=lambda item: (item[0], -item[1]))[1]


def boss_outcomes(
    observation: Any,
    player_index: int,
    option_indices: tuple[int, ...],
) -> tuple[BossOutcome, ...]:
    """Evaluate each legal Boss play through its best target and attack."""
    if not option_indices or getattr(observation, "search_begin_input", None) is None:
        return ()
    hidden = _placeholder_hidden(observation, player_index)
    before_prizes = view.prize_count(observation, player_index)
    outcomes: list[BossOutcome] = []
    with SearchSession.begin(observation, hidden) as search:
        root_id = int(search.root.searchId)
        for boss_index in option_indices:
            played = search.step(root_id, (boss_index,))
            try:
                switch_options = view.as_sequence(
                    getattr(getattr(played.observation, "select", None), "option", ())
                )
                continuations: list[AttackOutcome] = []
                for switch_index in range(len(switch_options)):
                    switched = search.step(int(played.searchId), (switch_index,))
                    try:
                        attack_options = view.as_sequence(
                            getattr(
                                getattr(switched.observation, "select", None),
                                "option",
                                (),
                            )
                        )
                        for attack_index, option in enumerate(attack_options):
                            if view.integer(getattr(option, "type", None), -1) != int(
                                OptionType.ATTACK
                            ):
                                continue
                            attacked = search.step(
                                int(switched.searchId),
                                (attack_index,),
                            )
                            try:
                                continuations.append(
                                    _outcome(
                                        attacked.observation,
                                        player_index=player_index,
                                        option_index=attack_index,
                                        before_prizes=before_prizes,
                                    )
                                )
                            finally:
                                search.release(int(attacked.searchId))
                    finally:
                        search.release(int(switched.searchId))
                if continuations:
                    outcomes.append(
                        BossOutcome(
                            option_index=boss_index,
                            attack=max(
                                continuations,
                                key=lambda value: (
                                    value.utility,
                                    -value.option_index,
                                ),
                            ),
                        )
                    )
            finally:
                search.release(int(played.searchId))
    return tuple(outcomes)


def _placeholder_hidden(observation: Any, player_index: int) -> HiddenInformation:
    """Build deterministic size-correct zones for bounded forward checks.

    Immediate attack and Boss continuations do not consume these zones. The
    same-turn macro can cross a Run Away Draw transition, so neutral Dunsparce
    placeholders deliberately avoid assuming a helpful unseen card. Each real
    post-draw decision is recomputed from the actual public observation.
    """
    own = view.player(observation, player_index)
    opponent_index = view.opponent_index(observation, player_index)
    opponent = view.player(observation, opponent_index)
    placeholder = cards.DUNSPARCE
    opponent_active = view.active(observation, opponent_index)
    hidden_active = () if opponent_active is not None else (placeholder,)
    return HiddenInformation.from_sequences(
        your_deck=(placeholder,) * _count(own, "deckCount"),
        your_prize=(placeholder,) * len(view.as_sequence(getattr(own, "prize", ()))),
        opponent_deck=(placeholder,) * _count(opponent, "deckCount"),
        opponent_prize=(placeholder,)
        * len(view.as_sequence(getattr(opponent, "prize", ()))),
        opponent_hand=(placeholder,) * _count(opponent, "handCount"),
        opponent_active=hidden_active,
    )


def _outcome(
    observation: Any,
    *,
    player_index: int,
    option_index: int,
    before_prizes: int,
) -> AttackOutcome:
    """Read one immediate native attack transition."""
    opponent_index = view.opponent_index(observation, player_index)
    damage = 0
    knockouts = 0
    for log in getattr(observation, "logs", ()) or ():
        if view.integer(getattr(log, "playerIndex", None), -1) != opponent_index:
            continue
        log_type = view.integer(getattr(log, "type", None), -1)
        if log_type == int(LogType.HP_CHANGE) and not bool(
            getattr(log, "putDamageCounter", False)
        ):
            value = view.integer(getattr(log, "value", None), 0)
            damage += max(0, -value)
        if (
            log_type == int(LogType.MOVE_CARD)
            and view.integer(getattr(log, "fromArea", None), -1) == int(AreaType.ACTIVE)
            and view.integer(getattr(log, "toArea", None), -1) == int(AreaType.DISCARD)
        ):
            knockouts += 1
    state = getattr(observation, "current", None)
    result = view.integer(getattr(state, "result", None), -1)
    after_prizes = view.prize_count(observation, player_index)
    return AttackOutcome(
        option_index=option_index,
        damage=damage,
        knockouts=knockouts,
        prizes_taken=max(0, before_prizes - after_prizes),
        terminal_win=result == player_index,
    )


def _count(owner: Any, attribute: str) -> int:
    return max(0, view.integer(getattr(owner, attribute, 0), 0))
