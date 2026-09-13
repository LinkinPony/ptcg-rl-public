"""Prompt-local tactical scoring for the Slowking scripted opponent."""

from __future__ import annotations

from typing import Any

from ptcg_rl.engine.constants import OptionType
from ptcg_rl.opponents.slowking_copy import cards, view


def main_combo_score(
    observation: Any,
    player_index: int,
    option: Any,
    *,
    seek_ready: bool,
) -> float | None:
    """Score only actions that advance the Slowking copy route."""
    option_type = int(option.type)
    card_id = view.option_card_id(observation, player_index, option)
    target = view.option_pokemon(observation, player_index, option)
    target_id = view.card_id(target)
    active_id = view.card_id(view.active(observation, player_index))
    hand_ids = view.card_ids(view.hand(observation, player_index))

    if option_type == int(OptionType.EVOLVE) and card_id == cards.SLOWKING:
        return 960.0 + (30.0 if is_active_target(option) else 0.0)
    if option_type == int(OptionType.ATTACH):
        if target_id == cards.SLOWKING:
            if view.energy_count(target) >= 2:
                return 180.0
            return 940.0 if is_active_target(option) else 610.0
        if target_id == cards.SLOWPOKE:
            if view.energy_count(target) >= 2:
                return 160.0
            return 590.0
    if (
        option_type == int(OptionType.ABILITY)
        and card_id == cards.ACADEMY_AT_NIGHT
        and seek_ready
        and any(card in cards.PAYLOAD_POKEMON for card in hand_ids)
    ):
        return 920.0
    if (
        option_type == int(OptionType.PLAY)
        and card_id == cards.CIPHERMANIAC
        and seek_ready
    ):
        return 910.0
    if (
        option_type == int(OptionType.PLAY)
        and card_id == cards.ACADEMY_AT_NIGHT
        and seek_ready
        and any(card in cards.PAYLOAD_POKEMON for card in hand_ids)
    ):
        return 900.0
    if option_type == int(OptionType.PLAY) and card_id == cards.SLOWPOKE:
        slowpoke_count = field_count(observation, player_index, cards.SLOWPOKE)
        return 760.0 - slowpoke_count * 80.0
    if (
        option_type == int(OptionType.PLAY)
        and card_id in {cards.POKE_PAD, cards.ULTRA_BALL}
        and needs_line_piece(observation, player_index)
    ):
        return 700.0 if card_id == cards.POKE_PAD else 680.0
    if (
        option_type == int(OptionType.PLAY)
        and card_id == cards.WONDROUS_PATCH
        and field_count(observation, player_index, cards.SLOWKING) > 0
    ):
        return 640.0
    if (
        option_type == int(OptionType.PLAY)
        and card_id == cards.NIGHT_STRETCHER
        and discard_has_line_piece(observation, player_index)
    ):
        return 630.0
    if (
        option_type == int(OptionType.RETREAT)
        and active_id != cards.SLOWKING
        and ready_bench_slowking(observation, player_index)
    ):
        return 620.0
    if (
        option_type == int(OptionType.ATTACK)
        and integer(getattr(option, "attackId", None), -1) == cards.DELIGHTFUL_KISS
        and field_count(observation, player_index, cards.SLOWPOKE) > 0
    ):
        return 500.0
    return None


def payload_score(observation: Any, player_index: int, card_id: int) -> float:
    """Score the three non-Rule-Box copy payloads from public state."""
    opponent = 1 - player_index
    opponent_active = view.active(observation, opponent)
    active_hp = integer(getattr(opponent_active, "hp", None), 0)
    spread_kos = sum(
        integer(getattr(pokemon, "hp", None), 0) <= 110
        for pokemon in view.field_pokemon(observation, opponent)
    )
    if card_id == cards.KYUREM:
        return 1050.0 if spread_kos >= 2 else 650.0 + spread_kos * 80.0
    if card_id == cards.CONKELDURR:
        return 1000.0 if 0 < active_hp <= 250 else 780.0
    if card_id == cards.ANNIHILAPE:
        return 980.0 if active_hp > 250 else 620.0
    return 0.0


def next_draw_score(
    observation: Any,
    player_index: int,
    option: Any,
) -> float:
    """Score the card placed immediately below Cipher's copy payload."""
    card_id = view.option_card_id(observation, player_index, option)
    if card_id in cards.PAYLOAD_POKEMON and academy_in_play(observation):
        return 900.0
    priorities = {
        cards.CIPHERMANIAC: 800.0,
        cards.ACADEMY_AT_NIGHT: 760.0,
        cards.SLOWKING: 720.0,
        cards.SLOWPOKE: 680.0,
        cards.BASIC_PSYCHIC: 640.0,
        cards.TELEPATH_PSYCHIC_ENERGY: 620.0,
    }
    return priorities.get(card_id or 0, 100.0)


def search_score(
    observation: Any,
    player_index: int,
    source_card: int,
    option: Any,
) -> float:
    """Score a legal search/recovery target."""
    card_id = view.option_card_id(observation, player_index, option)
    if card_id is None:
        return 0.0
    if source_card == cards.MEOWTH_EX:
        return 1000.0 if card_id == cards.CIPHERMANIAC else 100.0
    if source_card == cards.SECRET_BOX:
        return {
            cards.CIPHERMANIAC: 1000.0,
            cards.ACADEMY_AT_NIGHT: 950.0,
            cards.COUNTER_GAIN: 900.0,
            cards.POKE_PAD: 850.0,
            cards.WONDROUS_PATCH: 800.0,
        }.get(card_id, 100.0)
    if (
        source_card in {cards.POKE_PAD, cards.ULTRA_BALL}
        and ready_active_slowking(observation, player_index)
        and academy_in_play(observation)
        and card_id in cards.PAYLOAD_POKEMON
    ):
        return payload_score(observation, player_index, card_id) + 500.0
    if (
        field_count(observation, player_index, cards.SLOWPOKE) == 0
        and card_id == cards.SLOWPOKE
    ):
        return 1100.0
    if (
        field_count(observation, player_index, cards.SLOWKING) == 0
        and card_id == cards.SLOWKING
    ):
        return 1100.0
    if source_card == cards.NIGHT_STRETCHER:
        return {
            cards.SLOWKING: 1000.0,
            cards.SLOWPOKE: 900.0,
            cards.BASIC_PSYCHIC: 700.0,
        }.get(card_id, 200.0)
    return {
        cards.SLOWKING: 850.0,
        cards.SLOWPOKE: 800.0,
        cards.CIPHERMANIAC: 750.0,
        cards.ACADEMY_AT_NIGHT: 700.0,
    }.get(card_id, 100.0)


def keep_score(observation: Any, player_index: int, option: Any) -> float:
    """Score the opportunity cost of discarding a legal card option."""
    card_id = view.option_card_id(observation, player_index, option)
    if card_id is None:
        return 0.0
    score = {
        cards.SECRET_BOX: 1000.0,
        cards.COUNTER_GAIN: 900.0,
        cards.SLOWKING: 850.0,
        cards.SLOWPOKE: 800.0,
        cards.CIPHERMANIAC: 780.0,
        cards.ACADEMY_AT_NIGHT: 760.0,
        cards.BASIC_PSYCHIC: 700.0,
        cards.TELEPATH_PSYCHIC_ENERGY: 680.0,
        cards.BOOMERANG_ENERGY: 620.0,
    }.get(card_id, 300.0)
    if card_id in cards.PAYLOAD_POKEMON and academy_in_play(observation):
        score = 880.0
    if card_id == cards.ACADEMY_AT_NIGHT and academy_in_play(observation):
        score = 120.0
    return score


def setup_score(
    observation: Any,
    player_index: int,
    option: Any,
    *,
    active_slot: bool,
) -> float:
    """Score opening Active/Bench placement."""
    card_id = view.option_card_id(observation, player_index, option)
    if active_slot:
        return {
            cards.SMOOCHUM: 1000.0,
            cards.SLOWPOKE: 900.0,
            cards.MEGA_KANGASKHAN_EX: 500.0,
            cards.FEZANDIPITI_EX: 400.0,
            cards.LATIAS_EX: 300.0,
            cards.MEOWTH_EX: 100.0,
        }.get(card_id or 0, 0.0)
    return {
        cards.SLOWPOKE: 1000.0,
        cards.LATIAS_EX: 800.0,
        cards.SMOOCHUM: 600.0,
        cards.FEZANDIPITI_EX: 400.0,
        cards.MEGA_KANGASKHAN_EX: 300.0,
        # Preserve Last-Ditch Catch by playing Meowth from hand later.
        cards.MEOWTH_EX: -100.0,
    }.get(card_id or 0, 0.0)


def switch_score(observation: Any, player_index: int, option: Any) -> float:
    """Score a legal switch target."""
    pokemon = view.option_pokemon(observation, player_index, option)
    card_id = view.card_id(pokemon)
    if card_id == cards.SLOWKING:
        return 1000.0 + view.energy_count(pokemon) * 20.0
    return {
        cards.SMOOCHUM: 600.0,
        cards.SLOWPOKE: 500.0,
        cards.MEGA_KANGASKHAN_EX: 400.0,
    }.get(card_id or 0, 100.0)


def attach_target_score(
    observation: Any,
    player_index: int,
    option: Any,
) -> float:
    """Score an engine-legal attachment target."""
    pokemon = view.option_pokemon(observation, player_index, option)
    card_id = view.card_id(pokemon)
    active_bonus = 100.0 if is_active_target(option) else 0.0
    if card_id == cards.SLOWKING:
        return 1000.0 + active_bonus - view.energy_count(pokemon) * 100.0
    if card_id == cards.SLOWPOKE:
        return 850.0 + active_bonus - view.energy_count(pokemon) * 80.0
    return 200.0 + active_bonus


def trifrost_target_score(
    observation: Any,
    player_index: int,
    option: Any,
) -> float:
    """Prioritize 110-damage KOs for a copied Trifrost."""
    pokemon = view.option_pokemon(observation, player_index, option)
    hp = integer(getattr(pokemon, "hp", None), 0)
    max_hp = integer(getattr(pokemon, "maxHp", None), hp)
    score = 1000.0 if 0 < hp <= 110 else 200.0
    score += max(0, max_hp - hp) * 0.5
    if is_active_target(option):
        score += 20.0
    return score


def generic_main_score(
    observation: Any,
    player_index: int,
    option: Any,
) -> float:
    """Order ordinary legal main actions without deriving card effects."""
    option_type = int(option.type)
    attack_id = integer(getattr(option, "attackId", None), -1)
    card_id = view.option_card_id(observation, player_index, option)
    if option_type == int(OptionType.ATTACK):
        return {
            cards.SUPER_PSY_BOLT: 98.0,
            cards.DELIGHTFUL_KISS: 92.0,
            cards.SEEK_INSPIRATION: 15.0,
        }.get(attack_id, 85.0)
    if option_type == int(OptionType.EVOLVE):
        return 80.0
    if option_type == int(OptionType.ATTACH):
        return 75.0
    if option_type == int(OptionType.ABILITY):
        if card_id == cards.ACADEMY_AT_NIGHT:
            return -5.0
        return 70.0
    if option_type == int(OptionType.PLAY):
        if card_id == cards.CIPHERMANIAC:
            return 20.0
        return 65.0 if card_id in cards.POKEMON else 55.0
    if option_type == int(OptionType.RETREAT):
        return 35.0
    if option_type == int(OptionType.END):
        return 0.0
    return 10.0


def generic_option_score(
    observation: Any,
    player_index: int,
    option: Any,
) -> float:
    """Score an otherwise-unhandled legal prompt option."""
    card_id = view.option_card_id(observation, player_index, option)
    pokemon = view.option_pokemon(observation, player_index, option)
    if pokemon is not None:
        return float(integer(getattr(pokemon, "hp", None), 0))
    if card_id == cards.SLOWKING:
        return 100.0
    if card_id == cards.SLOWPOKE:
        return 90.0
    if card_id in cards.PAYLOAD_POKEMON:
        return payload_score(observation, player_index, card_id)
    number = getattr(option, "number", None)
    return float(number) if number is not None else 1.0


def field_count(observation: Any, player_index: int, card_id: int) -> int:
    """Count a card ID in the acting player's field."""
    return sum(
        view.card_id(pokemon) == card_id
        for pokemon in view.field_pokemon(observation, player_index)
    )


def needs_line_piece(observation: Any, player_index: int) -> bool:
    """Return whether no Slowpoke or no Slowking is currently in play."""
    return (
        field_count(observation, player_index, cards.SLOWPOKE) == 0
        or field_count(observation, player_index, cards.SLOWKING) == 0
    )


def discard_has_line_piece(observation: Any, player_index: int) -> bool:
    """Return whether Night Stretcher can recover a core line resource."""
    discard_ids = view.card_ids(view.discard(observation, player_index))
    return any(
        card_id in {cards.SLOWPOKE, cards.SLOWKING, cards.BASIC_PSYCHIC}
        for card_id in discard_ids
    )


def academy_in_play(observation: Any) -> bool:
    """Return whether Academy at Night is the current Stadium."""
    stadium = view.as_sequence(getattr(observation.current, "stadium", ()))
    return any(view.card_id(card) == cards.ACADEMY_AT_NIGHT for card in stadium)


def ready_active_slowking(observation: Any, player_index: int) -> bool:
    """Return whether the Active is an energy-ready Slowking."""
    active = view.active(observation, player_index)
    return view.card_id(active) == cards.SLOWKING and view.energy_count(active) >= 2


def ready_bench_slowking(observation: Any, player_index: int) -> bool:
    """Return whether an energy-ready Slowking is on the Bench."""
    return any(
        view.card_id(pokemon) == cards.SLOWKING and view.energy_count(pokemon) >= 2
        for pokemon in view.bench(observation, player_index)
    )


def is_active_target(option: Any) -> bool:
    """Return whether an option's in-play target is Active."""
    return integer(getattr(option, "inPlayArea", None), -1) == 4 or (
        getattr(option, "inPlayArea", None) is None
        and integer(getattr(option, "area", None), -1) == 4
    )


def integer(value: Any, default: int) -> int:
    """Best-effort integer conversion."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
