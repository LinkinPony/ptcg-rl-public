"""Selection-prompt and board-state scoring for the Lopunny pilot."""

from __future__ import annotations

from typing import Any

from ptcg_rl.engine.constants import OptionType
from ptcg_rl.opponents.lopunny_dudunsparce import cards, view


def search_score(
    observation: Any,
    player_index: int,
    source_card: int,
    option: Any,
) -> float:
    """Rank cards for one known Trainer or Ability search effect."""
    card_id = view.option_card_id(observation, player_index, option)
    if card_id is None:
        return -1000.0
    if source_card == cards.FAN_ROTOM:
        return fan_search_score(observation, player_index, card_id)
    if source_card == cards.BUDDY_BUDDY_POFFIN:
        return poffin_search_score(observation, player_index, card_id)
    if source_card == cards.ULTRA_BALL:
        return pokemon_target_score(observation, player_index, card_id, any_rule=True)
    if source_card == cards.POKE_PAD:
        return pokemon_target_score(observation, player_index, card_id, any_rule=False)
    if source_card == cards.POKEGEAR_30:
        return supporter_search_score(observation, player_index, card_id)
    if source_card == cards.HILDA:
        if card_id in cards.ENERGY_CARDS:
            enriching_score = (
                4800.0 if view.deck_count(observation, player_index) >= 5 else 3500.0
            )
            return {
                cards.ENRICHING_ENERGY: enriching_score,
                cards.SPIKY_ENERGY: 4200.0,
                cards.MIST_ENERGY: 4000.0,
            }[card_id]
        return pokemon_target_score(observation, player_index, card_id, any_rule=True)
    return keep_score(observation, player_index, option)


def fan_search_score(observation: Any, player_index: int, card_id: int) -> float:
    """Use Fan Call to stock both kinds of Basic evolution line."""
    field_and_hand = view.card_ids(
        view.field_pokemon(observation, player_index)
    ) + view.card_ids(view.hand(observation, player_index))
    if card_id == cards.DUNSPARCE:
        count = sum(
            value in {cards.DUNSPARCE, cards.DUDUNSPARCE} for value in field_and_hand
        )
        return 4500.0 if count < 3 else 2600.0
    if card_id == cards.BUNEARY:
        count = sum(
            value in {cards.BUNEARY, cards.MEGA_LOPUNNY_EX} for value in field_and_hand
        )
        return 4400.0 if count < 2 else 2500.0
    return -1000.0


def poffin_search_score(observation: Any, player_index: int, card_id: int) -> float:
    """Place a first-turn Fan, then balance draw and attacker Basics."""
    if (
        card_id == cards.FAN_ROTOM
        and view.field_count(observation, player_index, cards.FAN_ROTOM) == 0
    ):
        return 4900.0
    if card_id == cards.BUNEARY:
        return 4550.0 if lopunny_line_count(observation, player_index) < 2 else 2800.0
    if card_id == cards.DUNSPARCE:
        return 4600.0 if dunsparce_line_count(observation, player_index) < 3 else 3000.0
    return -1000.0


def pokemon_target_score(
    observation: Any,
    player_index: int,
    card_id: int,
    *,
    any_rule: bool,
) -> float:
    """Rank a Pokemon search by an immediately reachable evolution line."""
    lone_pokemon = (
        len(view.field_pokemon(observation, player_index)) == 1
        and view.bench_space(observation, player_index) > 0
    )
    if lone_pokemon:
        if card_id == cards.DUNSPARCE:
            return 6200.0
        if card_id == cards.BUNEARY:
            return 6100.0
        if card_id == cards.FAN_ROTOM:
            return 6000.0
        if card_id in cards.EVOLUTION_POKEMON:
            return 1200.0
    if any_rule and card_id == cards.MEGA_LOPUNNY_EX:
        return 5000.0 if unevolved_buneary_count(observation, player_index) else 2500.0
    if card_id == cards.DUDUNSPARCE:
        return (
            4800.0 if unevolved_dunsparce_count(observation, player_index) else 3000.0
        )
    if card_id == cards.DUNSPARCE:
        return 4200.0 if dunsparce_line_count(observation, player_index) < 3 else 2500.0
    if card_id == cards.BUNEARY:
        return 4100.0 if lopunny_line_count(observation, player_index) < 2 else 2300.0
    if card_id == cards.FAN_ROTOM:
        return (
            4000.0
            if view.field_count(observation, player_index, card_id) == 0
            else 1800.0
        )
    return -800.0


def supporter_search_score(
    observation: Any,
    player_index: int,
    card_id: int,
) -> float:
    """Choose the Supporter most useful on the current board."""
    if card_id == cards.WALLYS_COMPASSION:
        if damaged_mega_count(observation, player_index):
            return 5400.0
        if view.field_count(observation, player_index, cards.MEGA_LOPUNNY_EX):
            return 5100.0
        return 3500.0
    if card_id == cards.HILDA:
        return 4900.0 if hilda_useful(observation, player_index) else 3100.0
    if card_id == cards.BOSSES_ORDERS:
        return 4550.0 if boss_useful(observation, player_index) else 2800.0
    if card_id == cards.LILLIES_DETERMINATION:
        return 4700.0 if lillie_score(observation, player_index) > 0 else 3000.0
    if card_id == cards.XEROSICS_MACHINATIONS:
        return 2600.0
    return -500.0


def switch_score(
    observation: Any,
    player_index: int,
    option: Any,
    *,
    chain_run_away: bool,
    forced_promotion: bool = False,
) -> float:
    """Rank own pivots or an opponent target selected by Boss."""
    target = view.option_pokemon(observation, player_index, option)
    owner = view.integer(getattr(option, "playerIndex", None), player_index)
    if owner != player_index:
        return opponent_target_score(target)
    if target is None:
        return -1000.0

    card_id = view.card_id(target)
    current_active = view.active(observation, player_index)
    leaving_mega = view.card_id(current_active) == cards.MEGA_LOPUNNY_EX
    if (
        forced_promotion
        and not chain_run_away
        and future_gale_pivot(observation, player_index, target)
    ):
        return 6000.0
    if card_id == cards.DUDUNSPARCE:
        return 5600.0 if leaving_mega or chain_run_away else 4800.0
    if card_id == cards.DUNSPARCE:
        can_chain = view.hand_count(
            observation, player_index, cards.DUDUNSPARCE
        ) > 0 and not view.appeared_this_turn(target)
        if can_chain and (leaving_mega or chain_run_away):
            return 5500.0
        return 4000.0 if chain_run_away else 2600.0
    if card_id == cards.MEGA_LOPUNNY_EX:
        return 5800.0 if view.energy_count(target) >= 1 else 3100.0
    if card_id == cards.FAN_ROTOM:
        return 2300.0
    if card_id == cards.BUNEARY:
        return 2100.0
    return 1000.0


def future_gale_pivot(observation: Any, player_index: int, pokemon: Any) -> bool:
    """Return whether a forced promotion can enable next-turn Gale natively."""
    if pokemon is None:
        return False
    has_bench_mega = any(
        other is not pokemon and view.card_id(other) == cards.MEGA_LOPUNNY_EX
        for other in view.field_pokemon(observation, player_index)
    )
    if not has_bench_mega:
        return False
    card_id = view.card_id(pokemon)
    if card_id == cards.DUDUNSPARCE:
        return True
    if card_id == cards.DUNSPARCE:
        return (
            view.hand_count(observation, player_index, cards.DUDUNSPARCE) > 0
            and not view.appeared_this_turn(pokemon)
        ) or cards.AIR_BALLOON in view.card_ids(view.tools(pokemon))
    return card_id in {
        cards.FAN_ROTOM,
        cards.BUNEARY,
    } and cards.AIR_BALLOON in view.card_ids(view.tools(pokemon))


def opponent_target_score(pokemon: Any) -> float:
    """Prefer damaged low-HP targets without freezing a prize lookup table."""
    if pokemon is None:
        return -1000.0
    remaining = view.remaining_hp(pokemon)
    maximum = view.integer(getattr(pokemon, "maxHp", remaining), remaining) or remaining
    return 5000.0 - float(remaining) + 0.15 * float(maximum)


def heal_score(observation: Any, player_index: int, option: Any) -> float:
    """Target the most damaged Mega for Wally's full heal."""
    pokemon = view.option_pokemon(observation, player_index, option)
    if view.card_id(pokemon) != cards.MEGA_LOPUNNY_EX:
        return -1000.0
    state = getattr(observation, "current", None)
    if bool(getattr(state, "energyAttached", False)) and pokemon is view.active(
        observation, player_index
    ):
        # Wally would return every Energy after this turn's attachment was
        # spent, potentially turning a live attack into an avoidable pass.
        return -1000.0
    return 3000.0 + float(view.damage(pokemon))


def keep_score(observation: Any, player_index: int, option: Any) -> float:
    """Return the value of retaining a card when paying a discard cost."""
    card_id = view.option_card_id(observation, player_index, option)
    if card_id is None:
        return 1000.0
    if card_id == cards.ENRICHING_ENERGY:
        return 9000.0
    if card_id in {cards.SPIKY_ENERGY, cards.MIST_ENERGY}:
        return 6500.0
    if card_id == cards.MEGA_LOPUNNY_EX:
        return 8200.0 if unevolved_buneary_count(observation, player_index) else 4200.0
    if card_id == cards.DUDUNSPARCE:
        return (
            8000.0 if unevolved_dunsparce_count(observation, player_index) else 4300.0
        )
    if card_id == cards.WALLYS_COMPASSION:
        if damaged_mega_count(observation, player_index):
            return 8000.0
        if view.field_count(observation, player_index, cards.MEGA_LOPUNNY_EX):
            return 7000.0
        return 4300.0
    if card_id == cards.AIR_BALLOON:
        return 7000.0 if balloon_needed(observation, player_index) else 2800.0
    if card_id == cards.BUNEARY:
        return 6100.0 if lopunny_line_count(observation, player_index) < 2 else 2400.0
    if card_id == cards.DUNSPARCE:
        return 6200.0 if dunsparce_line_count(observation, player_index) < 3 else 2500.0
    if card_id == cards.FAN_ROTOM:
        return (
            5600.0
            if view.field_count(observation, player_index, card_id) == 0
            else 1800.0
        )
    if card_id == cards.BOSSES_ORDERS:
        return 6800.0
    if card_id == cards.XEROSICS_MACHINATIONS:
        return 4300.0
    if card_id == cards.LILLIES_DETERMINATION:
        return (
            2600.0
            if view.hand_count(observation, player_index, card_id) > 1
            else 4700.0
        )
    if card_id == cards.HILDA:
        return (
            3000.0
            if view.hand_count(observation, player_index, card_id) > 1
            else 4400.0
        )
    if card_id == cards.BUDDY_BUDDY_POFFIN:
        if view.bench_space(observation, player_index) == 0:
            return 1200.0
        return 3600.0 if missing_basic_setup(observation, player_index) else 1600.0
    if card_id == cards.POKE_PAD:
        return 3600.0 if pokemon_search_useful(observation, player_index) else 1600.0
    if card_id == cards.POKEGEAR_30:
        return 3200.0
    if card_id == cards.ULTRA_BALL:
        return 2600.0
    return 3000.0


def retreat_energy_score(
    observation: Any,
    player_index: int,
    option: Any,
) -> float:
    """Discard the least strategically valuable attached Energy first."""
    pokemon = view.option_pokemon(observation, player_index, option)
    energy_index = view.integer(getattr(option, "energyIndex", None), -1)
    attached = view.energy_cards(pokemon)
    card_id = (
        view.card_id(attached[energy_index])
        if 0 <= energy_index < len(attached)
        else None
    )
    return {
        cards.SPIKY_ENERGY: 3000.0,
        cards.MIST_ENERGY: 2200.0,
        cards.ENRICHING_ENERGY: 1000.0,
    }.get(card_id or 0, 2000.0)


def generic_target_score(pokemon: Any) -> float:
    """Provide a stable fallback ranking for visible Pokemon targets."""
    card_id = view.card_id(pokemon)
    return {
        cards.MEGA_LOPUNNY_EX: 3000.0,
        cards.DUDUNSPARCE: 2400.0,
        cards.DUNSPARCE: 2200.0,
        cards.BUNEARY: 2100.0,
        cards.FAN_ROTOM: 2000.0,
    }.get(card_id or 0, 1000.0)


def damaged_mega_count(observation: Any, player_index: int) -> int:
    """Count damaged Mega Lopunny in play."""
    return sum(
        view.card_id(pokemon) == cards.MEGA_LOPUNNY_EX and view.damage(pokemon) > 0
        for pokemon in view.field_pokemon(observation, player_index)
    )


def wally_score(observation: Any, player_index: int) -> float:
    """Heal before attachment, or heal a safe Bench Mega afterwards."""
    damaged = tuple(
        pokemon
        for pokemon in view.field_pokemon(observation, player_index)
        if view.card_id(pokemon) == cards.MEGA_LOPUNNY_EX and view.damage(pokemon) > 0
    )
    if not damaged:
        return -900.0
    state = getattr(observation, "current", None)
    if not bool(getattr(state, "energyAttached", False)):
        active = view.active(observation, player_index)
        if active in damaged:
            return 9900.0 if view.damage(active) >= 100 else 8400.0
        maximum_damage = max(view.damage(pokemon) for pokemon in damaged)
        return 9550.0 if maximum_damage >= 160 else 8200.0
    active = view.active(observation, player_index)
    safe_bench_damage = max(
        (view.damage(pokemon) for pokemon in damaged if pokemon is not active),
        default=0,
    )
    if (
        safe_bench_damage >= 160
        and view.card_id(active) == cards.MEGA_LOPUNNY_EX
        and view.energy_count(active) >= 1
    ):
        return 8700.0
    return -900.0


def unevolved_buneary_count(observation: Any, player_index: int) -> int:
    """Count Buneary currently able to receive a future Mega evolution."""
    return view.field_count(observation, player_index, cards.BUNEARY)


def unevolved_dunsparce_count(observation: Any, player_index: int) -> int:
    """Count Dunsparce currently able to receive a future evolution."""
    return view.field_count(observation, player_index, cards.DUNSPARCE)


def lopunny_line_count(observation: Any, player_index: int) -> int:
    """Count Buneary and Mega Lopunny lines in play."""
    return sum(
        view.card_id(pokemon) in {cards.BUNEARY, cards.MEGA_LOPUNNY_EX}
        for pokemon in view.field_pokemon(observation, player_index)
    )


def dunsparce_line_count(observation: Any, player_index: int) -> int:
    """Count Dunsparce and Dudunsparce lines in play."""
    return sum(
        view.card_id(pokemon) in {cards.DUNSPARCE, cards.DUDUNSPARCE}
        for pokemon in view.field_pokemon(observation, player_index)
    )


def missing_basic_setup(observation: Any, player_index: int) -> bool:
    """Return whether Poffin can improve the current field."""
    return (
        lopunny_line_count(observation, player_index) < 2
        or dunsparce_line_count(observation, player_index) < 3
    )


def pokemon_search_useful(observation: Any, player_index: int) -> bool:
    """Return whether a Pokemon search has a reachable missing component."""
    hand_ids = view.card_ids(view.hand(observation, player_index))
    return (
        (
            unevolved_buneary_count(observation, player_index) > 0
            and cards.MEGA_LOPUNNY_EX not in hand_ids
        )
        or (
            unevolved_dunsparce_count(observation, player_index) > 0
            and cards.DUDUNSPARCE not in hand_ids
        )
        or missing_basic_setup(observation, player_index)
    )


def hilda_useful(
    observation: Any,
    player_index: int,
    *,
    hopper_needed: bool = False,
) -> bool:
    """Return whether Hilda can supply an evolution or attachment."""
    hand_ids = view.card_ids(view.hand(observation, player_index))
    has_energy = any(card_id in cards.ENERGY_CARDS for card_id in hand_ids)
    needs_evolution = (
        unevolved_buneary_count(observation, player_index) > 0
        and cards.MEGA_LOPUNNY_EX not in hand_ids
    ) or (
        unevolved_dunsparce_count(observation, player_index) > 0
        and cards.DUDUNSPARCE not in hand_ids
    )
    needs_energy = any(
        view.card_id(pokemon) in {cards.BUNEARY, cards.MEGA_LOPUNNY_EX}
        and view.energy_count(pokemon)
        < (
            2
            if hopper_needed
            and pokemon is view.active(observation, player_index)
            and view.card_id(pokemon) == cards.MEGA_LOPUNNY_EX
            else 1
        )
        for pokemon in view.field_pokemon(observation, player_index)
    )
    return needs_evolution or (needs_energy and not has_energy)


def hand_has_energy(observation: Any, player_index: int) -> bool:
    """Return whether a manual Energy attachment is already available."""
    return any(
        card_id in cards.ENERGY_CARDS
        for card_id in view.card_ids(view.hand(observation, player_index))
    )


def boss_useful(observation: Any, player_index: int) -> bool:
    """Return whether Boss has at least one legal Bench target."""
    opponent = view.opponent_index(observation, player_index)
    return bool(view.bench(observation, opponent))


def attack_available(observation: Any) -> bool:
    """Return whether MAIN currently exposes a terminal attack option."""
    select = getattr(observation, "select", None)
    return any(
        view.integer(getattr(option, "type", None), -1) == int(OptionType.ATTACK)
        for option in view.as_sequence(getattr(select, "option", ()))
    )


def lillie_score(observation: Any, player_index: int) -> float:
    """Use Lillie for meaningful net draw without shuffling a fresh line."""
    hand = view.hand(observation, player_index)
    opening_draw = view.prize_count(observation, player_index) == 6
    if any(
        view.card_id(pokemon) in {cards.BUNEARY, cards.DUNSPARCE}
        and view.appeared_this_turn(pokemon)
        for pokemon in view.field_pokemon(observation, player_index)
    ) and any(
        card_id in {cards.MEGA_LOPUNNY_EX, cards.DUDUNSPARCE}
        for card_id in view.card_ids(hand)
    ):
        return -900.0
    if view.deck_count(observation, player_index) <= 2 and len(hand) > 1:
        return 9300.0
    useful_hand_limit = 6 if opening_draw else 5
    return 8800.0 if len(hand) <= useful_hand_limit else -500.0


def balloon_needed(observation: Any, player_index: int) -> bool:
    """Return whether the field still lacks a useful retreat pivot."""
    if field_tool_count(observation, player_index, cards.AIR_BALLOON) >= 2:
        return False
    active = view.active(observation, player_index)
    return (
        active is not None
        and view.card_id(active) != cards.DUDUNSPARCE
        and cards.AIR_BALLOON not in view.card_ids(view.tools(active))
    )


def field_tool_count(observation: Any, player_index: int, card_id: int) -> int:
    """Count one Tool identity currently attached to field Pokemon."""
    return sum(
        card_id in view.card_ids(view.tools(pokemon))
        for pokemon in view.field_pokemon(observation, player_index)
    )
