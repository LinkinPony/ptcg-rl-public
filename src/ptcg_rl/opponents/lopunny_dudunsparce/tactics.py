"""Tactical scoring for the Majkel Lopunny/Dudunsparce pilot."""

from __future__ import annotations

from typing import Any

from ptcg_rl.engine.constants import OptionType
from ptcg_rl.opponents.lopunny_dudunsparce import (
    cards,
    view,
)
from ptcg_rl.opponents.lopunny_dudunsparce import (
    selection_tactics as _selection_tactics,
)

search_score = _selection_tactics.search_score
fan_search_score = _selection_tactics.fan_search_score
poffin_search_score = _selection_tactics.poffin_search_score
pokemon_target_score = _selection_tactics.pokemon_target_score
supporter_search_score = _selection_tactics.supporter_search_score
switch_score = _selection_tactics.switch_score
future_gale_pivot = _selection_tactics.future_gale_pivot
opponent_target_score = _selection_tactics.opponent_target_score
heal_score = _selection_tactics.heal_score
keep_score = _selection_tactics.keep_score
retreat_energy_score = _selection_tactics.retreat_energy_score
generic_target_score = _selection_tactics.generic_target_score
damaged_mega_count = _selection_tactics.damaged_mega_count
wally_score = _selection_tactics.wally_score
unevolved_buneary_count = _selection_tactics.unevolved_buneary_count
unevolved_dunsparce_count = _selection_tactics.unevolved_dunsparce_count
lopunny_line_count = _selection_tactics.lopunny_line_count
dunsparce_line_count = _selection_tactics.dunsparce_line_count
missing_basic_setup = _selection_tactics.missing_basic_setup
pokemon_search_useful = _selection_tactics.pokemon_search_useful
hilda_useful = _selection_tactics.hilda_useful
hand_has_energy = _selection_tactics.hand_has_energy
boss_useful = _selection_tactics.boss_useful
attack_available = _selection_tactics.attack_available
lillie_score = _selection_tactics.lillie_score
balloon_needed = _selection_tactics.balloon_needed
field_tool_count = _selection_tactics.field_tool_count

_SETUP_ACTIVE_SCORE = {
    cards.FAN_ROTOM: 3000.0,
    cards.DUNSPARCE: 2000.0,
    cards.BUNEARY: 1000.0,
}


def setup_active_score(observation: Any, player_index: int, option: Any) -> float:
    """Prefer the replay-supported low-risk opening pivots."""
    return _SETUP_ACTIVE_SCORE.get(
        view.option_card_id(observation, player_index, option) or 0,
        -1000.0,
    )


def setup_bench_score(observation: Any, player_index: int, option: Any) -> float:
    """Rank opening Basics while preserving both engine and attacker lines."""
    card_id = view.option_card_id(observation, player_index, option)
    return {
        cards.BUNEARY: 3000.0,
        cards.DUNSPARCE: 2900.0,
        cards.FAN_ROTOM: 2800.0,
    }.get(card_id or 0, -1000.0)


def main_action_score(
    observation: Any,
    player_index: int,
    option: Any,
    *,
    gale_ready_serial: int | None,
    run_away_count: int = 0,
    hopper_needed: bool = False,
) -> float:
    """Score a MAIN option by progress toward a boosted Lopunny attack."""
    option_type = view.integer(getattr(option, "type", None), -1)
    card_id = view.option_card_id(observation, player_index, option)

    if option_type == int(OptionType.ABILITY):
        if card_id == cards.FAN_ROTOM:
            return 9950.0
        if card_id == cards.DUDUNSPARCE:
            active = view.active(observation, player_index)
            if (
                active is not None
                and view.card_id(active) == cards.DUDUNSPARCE
                and not view.bench(observation, player_index)
            ):
                # Run Away Draw would remove the last Pokemon and lose
                # immediately before its self-recycling value can matter.
                return -1800.0
            if (
                view.card_id(active) == cards.MEGA_LOPUNNY_EX
                and not gale_ready(observation, player_index, gale_ready_serial)
                and loop_retreat_available(observation, player_index)
                and view.field_count(observation, player_index, cards.DUDUNSPARCE) == 1
                and ready_bench_mega(observation, player_index) is None
            ):
                # Make a Bench Dudunsparce the pivot before consuming it.
                return 9000.0
            if run_away_count >= 2 and view.deck_count(observation, player_index) > 2:
                return 7200.0
            # Run Away Draw also restores its own line to an empty late-game
            # deck, so ordinary draw-before-deckout guards do not apply.
            return 9820.0
        # The public Spikemuth ability has no matching Marnie Pokemon in this
        # exact deck. Unknown shared abilities must not outrank useful cards.
        return 500.0

    if option_type == int(OptionType.EVOLVE):
        target = view.option_pokemon(observation, player_index, option)
        active_target = target is view.active(observation, player_index)
        if card_id == cards.DUDUNSPARCE:
            if run_away_count >= 2 and attack_available(observation):
                return 7400.0
            return 9740.0 if active_target else 9700.0
        if card_id == cards.MEGA_LOPUNNY_EX:
            active = view.active(observation, player_index)
            if (
                not active_target
                and view.energy_count(target) >= 1
                and view.card_id(active) in {cards.DUNSPARCE, cards.DUDUNSPARCE}
            ):
                # Build the attacker before consuming the active Run Away
                # pivot, so the ensuing promotion activates Gale Thrust.
                return 9940.0
            if view.field_count(
                observation, player_index, cards.MEGA_LOPUNNY_EX
            ) >= 2 and attack_available(observation):
                return 8200.0
            # Evolving on the Bench preserves a route to boosted Gale Thrust.
            return 9650.0 if not active_target else 9500.0
        return 300.0

    if option_type == int(OptionType.ATTACH):
        return attachment_main_score(
            observation,
            player_index,
            option,
            gale_ready_serial=gale_ready_serial,
            hopper_needed=hopper_needed,
        )

    if option_type == int(OptionType.PLAY):
        return play_score(
            observation,
            player_index,
            card_id,
            hopper_needed=hopper_needed,
        )

    if option_type == int(OptionType.RETREAT):
        if should_retreat(
            observation,
            player_index,
            gale_ready_serial=gale_ready_serial,
        ):
            return 9250.0
        return -1600.0

    if option_type == int(OptionType.ATTACK):
        return attack_score(
            observation,
            player_index,
            option,
            gale_ready_serial=gale_ready_serial,
        )
    if option_type == int(OptionType.END):
        return 0.0
    return -300.0


def play_score(
    observation: Any,
    player_index: int,
    card_id: int | None,
    *,
    hopper_needed: bool = False,
) -> float:
    """Score a playable card only when its effect advances this board."""
    if card_id is None:
        return -1000.0
    ready_attack = attack_available(observation)
    if card_id == cards.WALLYS_COMPASSION:
        return wally_score(observation, player_index)
    if card_id == cards.BUDDY_BUDDY_POFFIN:
        return (
            8700.0
            if view.bench_space(observation, player_index) > 0
            and missing_basic_setup(observation, player_index)
            else -700.0
        )
    if card_id == cards.POKEGEAR_30:
        return 8650.0 if view.deck_count(observation, player_index) > 0 else -800.0
    if card_id == cards.POKE_PAD:
        if not pokemon_search_useful(observation, player_index):
            return -700.0
        return 7800.0 if ready_attack else 8600.0
    if card_id == cards.ULTRA_BALL:
        if len(view.hand(observation, player_index)) < 3 or not pokemon_search_useful(
            observation, player_index
        ):
            return -800.0
        return 7600.0 if ready_attack else 8550.0
    if card_id in cards.BASIC_POKEMON:
        return basic_play_score(observation, player_index, card_id)
    if card_id == cards.HILDA:
        if hopper_needed and not hand_has_energy(observation, player_index):
            return 9100.0
        return (
            7600.0
            if hilda_useful(
                observation,
                player_index,
                hopper_needed=hopper_needed,
            )
            else -650.0
        )
    if card_id == cards.BOSSES_ORDERS:
        if not boss_useful(observation, player_index):
            return -650.0
        return 9000.0 if ready_attack else -650.0
    if card_id == cards.LILLIES_DETERMINATION:
        return lillie_score(observation, player_index)
    if card_id == cards.XEROSICS_MACHINATIONS:
        opponent = view.player(
            observation,
            view.opponent_index(observation, player_index),
        )
        opponent_hand = view.integer(getattr(opponent, "handCount", 0), 0) or 0
        return 8000.0 if opponent_hand >= 8 else -650.0
    return -700.0


def basic_play_score(observation: Any, player_index: int, card_id: int) -> float:
    """Fill the Bench with two attack lines and recyclable draw bodies."""
    if view.bench_space(observation, player_index) <= 0:
        return -1000.0
    field = view.field_pokemon(observation, player_index)
    if card_id == cards.FAN_ROTOM:
        return (
            8800.0
            if view.field_count(observation, player_index, card_id) == 0
            else -800.0
        )
    if card_id == cards.DUNSPARCE:
        duns_lines = sum(
            view.card_id(pokemon) in {cards.DUNSPARCE, cards.DUDUNSPARCE}
            for pokemon in field
        )
        return 8750.0 if duns_lines < 3 else 6500.0
    if card_id == cards.BUNEARY:
        lopunny_lines = sum(
            view.card_id(pokemon) in {cards.BUNEARY, cards.MEGA_LOPUNNY_EX}
            for pokemon in field
        )
        return 8700.0 if lopunny_lines < 2 else 6400.0
    return -900.0


def attachment_main_score(
    observation: Any,
    player_index: int,
    option: Any,
    *,
    gale_ready_serial: int | None,
    hopper_needed: bool = False,
) -> float:
    """Rank Energy and Balloon attachment targets without feeding Run Away."""
    card_id = view.option_card_id(observation, player_index, option)
    target = view.option_pokemon(observation, player_index, option)
    if card_id == cards.AIR_BALLOON:
        return balloon_main_score(
            observation,
            player_index,
            target,
            gale_ready_serial=gale_ready_serial,
        )
    if card_id in cards.ENERGY_CARDS:
        if (
            card_id == cards.ENRICHING_ENERGY
            and view.deck_count(observation, player_index) < 5
        ):
            return -900.0
        if (
            hopper_needed
            and target is view.active(observation, player_index)
            and view.card_id(target) == cards.MEGA_LOPUNNY_EX
            and view.energy_count(target) == 1
        ):
            return 9900.0
        active = view.active(observation, player_index)
        if (
            target is not active
            and view.card_id(target) == cards.MEGA_LOPUNNY_EX
            and view.energy_count(target) == 0
            and future_gale_pivot(observation, player_index, active)
        ):
            # On a forced-promotion pivot turn, attach before evolving or
            # retreating the pivot; the same turn can then end in boosted Gale.
            return 9900.0
        score = energy_target_score(observation, player_index, target, card_id)
        bonus = 8850.0 if card_id == cards.ENRICHING_ENERGY else 8350.0
        return bonus + score if score > 0 else -700.0
    return -800.0


def balloon_target_score(
    observation: Any,
    player_index: int,
    pokemon: Any,
    *,
    gale_ready_serial: int | None,
) -> float:
    """Prefer free-retreat participants that do not self-shuffle Tools."""
    if pokemon is None or cards.AIR_BALLOON in view.card_ids(view.tools(pokemon)):
        return -1000.0
    card_id = view.card_id(pokemon)
    active = pokemon is view.active(observation, player_index)
    if card_id == cards.MEGA_LOPUNNY_EX:
        if active and not gale_ready(observation, player_index, gale_ready_serial):
            return 700.0
        return 600.0
    if card_id == cards.DUNSPARCE:
        return 520.0 if active else 350.0
    if card_id == cards.FAN_ROTOM:
        return 500.0 if active else 300.0
    if card_id == cards.BUNEARY:
        return 460.0 if active else 320.0
    # Dudunsparce would shuffle the Tool away and still costs one to retreat.
    return -500.0


def balloon_main_score(
    observation: Any,
    player_index: int,
    pokemon: Any,
    *,
    gale_ready_serial: int | None,
) -> float:
    """Attach Balloon only to a concrete pivot instead of every field body."""
    target_score = balloon_target_score(
        observation,
        player_index,
        pokemon,
        gale_ready_serial=gale_ready_serial,
    )
    if target_score <= 0 or pokemon is None:
        return -750.0
    active = pokemon is view.active(observation, player_index)
    attached_count = field_tool_count(
        observation,
        player_index,
        cards.AIR_BALLOON,
    )
    if active and retreat_destination_available(observation, player_index):
        return 9350.0
    if active:
        return 8800.0 if attached_count == 0 else 8250.0
    if view.card_id(pokemon) == cards.MEGA_LOPUNNY_EX and attached_count < 2:
        return 8350.0
    if attached_count < 2:
        return 6200.0
    return -750.0


def energy_target_score(
    observation: Any,
    player_index: int,
    pokemon: Any,
    energy_card: int,
) -> float:
    """Build one- and two-Energy Mega attackers, using Buneary as staging."""
    if pokemon is None:
        return -1000.0
    card_id = view.card_id(pokemon)
    energy = view.energy_count(pokemon)
    active = pokemon is view.active(observation, player_index)
    if card_id == cards.MEGA_LOPUNNY_EX:
        if energy == 0:
            return 900.0 if not active else 850.0
        if energy == 1:
            return 720.0 if not active else 650.0
        return 180.0 if energy_card == cards.ENRICHING_ENERGY else -200.0
    if card_id == cards.BUNEARY:
        return 560.0 if energy == 0 else 260.0
    if card_id == cards.DUNSPARCE:
        return 220.0 if energy == 0 else -250.0
    if card_id == cards.FAN_ROTOM:
        return 180.0 if energy == 0 else -300.0
    return -900.0


def attack_score(
    observation: Any,
    player_index: int,
    option: Any,
    *,
    gale_ready_serial: int | None,
) -> float:
    """Rank terminal attacks using engine-verified identities."""
    attack_id = view.integer(getattr(option, "attackId", None), -1)
    if attack_id == cards.LOPUNNY_GALE_THRUST:
        return (
            8500.0
            if gale_ready(observation, player_index, gale_ready_serial)
            else 6200.0
        )
    if attack_id == cards.LOPUNNY_SPIKY_HOPPER:
        return (
            8450.0
            if not gale_ready(observation, player_index, gale_ready_serial)
            else 8400.0
        )
    if attack_id == cards.DUDUNSPARCE_LAND_CRUSH:
        return 6500.0
    if attack_id in {cards.DUNSPARCE_RAM, cards.BUNEARY_KICK}:
        return 5400.0
    if attack_id == cards.FAN_ASSAULT_LANDING:
        stadium = view.as_sequence(
            getattr(getattr(observation, "current", None), "stadium", ())
        )
        return 6000.0 if stadium else -100.0
    if attack_id in {
        cards.DUNSPARCE_TRADING_PLACES,
        cards.BUNEARY_RUN_AROUND,
    }:
        return 200.0
    return 5000.0


def should_retreat(
    observation: Any,
    player_index: int,
    *,
    gale_ready_serial: int | None,
) -> bool:
    """Return whether a legal retreat advances the same-turn Gale loop."""
    active = view.active(observation, player_index)
    active_id = view.card_id(active)
    if active_id == cards.MEGA_LOPUNNY_EX:
        return not gale_ready(
            observation, player_index, gale_ready_serial
        ) and retreat_destination_available(observation, player_index)
    return ready_bench_mega(observation, player_index) is not None


def loop_retreat_available(observation: Any, player_index: int) -> bool:
    """Return whether the Bench has a usable Run Away pivot."""
    hand_has_evolution = (
        view.hand_count(observation, player_index, cards.DUDUNSPARCE) > 0
    )
    for pokemon in view.bench(observation, player_index):
        card_id = view.card_id(pokemon)
        if card_id == cards.DUDUNSPARCE:
            return True
        if (
            card_id == cards.DUNSPARCE
            and hand_has_evolution
            and not view.appeared_this_turn(pokemon)
        ):
            return True
    return False


def retreat_destination_available(observation: Any, player_index: int) -> bool:
    """Return whether retreat can promote a ready Mega or use a draw pivot."""
    return ready_bench_mega(
        observation, player_index
    ) is not None or loop_retreat_available(observation, player_index)


def gale_ready(
    observation: Any,
    player_index: int,
    gale_ready_serial: int | None,
) -> bool:
    """Match the remembered Bench-to-Active event to the current Mega."""
    active = view.active(observation, player_index)
    return (
        view.card_id(active) == cards.MEGA_LOPUNNY_EX
        and gale_ready_serial is not None
        and view.serial(active) == gale_ready_serial
    )


def ready_bench_mega(observation: Any, player_index: int) -> Any | None:
    """Return a Bench Mega that can pay for Gale Thrust."""
    return next(
        (
            pokemon
            for pokemon in view.bench(observation, player_index)
            if view.card_id(pokemon) == cards.MEGA_LOPUNNY_EX
            and view.energy_count(pokemon) >= 1
        ),
        None,
    )


def attachment_prompt_score(
    observation: Any,
    player_index: int,
    option: Any,
) -> float:
    """Rank a target in a split attachment prompt."""
    source = view.effect_card_id(observation)
    target = view.option_pokemon(observation, player_index, option)
    if source == cards.AIR_BALLOON:
        return balloon_target_score(
            observation,
            player_index,
            target,
            gale_ready_serial=None,
        )
    if source in cards.ENERGY_CARDS:
        return energy_target_score(observation, player_index, target, source)
    return generic_target_score(target)
