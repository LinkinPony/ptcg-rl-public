"""Component-aware tactical scoring for the Slowking Copy v2 pilot."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from ptcg_rl.engine.constants import OptionType
from ptcg_rl.opponents.slowking_copy import tactics as v1_tactics
from ptcg_rl.opponents.slowking_copy import view
from ptcg_rl.opponents.slowking_copy_v2 import cards

_PSYCHIC_ENERGY_TYPE = 5
_FREE_DRAW_POKEMON = frozenset(
    {
        cards.FEZANDIPITI_EX,
        cards.MEGA_KANGASKHAN_EX,
    }
)
_SETUP_ACTIVE_SCORE = {
    cards.SMOOCHUM: 1000.0,
    cards.SLOWPOKE: 900.0,
    cards.MEGA_KANGASKHAN_EX: 500.0,
    cards.FEZANDIPITI_EX: 400.0,
    cards.LATIAS_EX: 300.0,
    cards.MEOWTH_EX: 100.0,
}
_BLIND_COPY_ATTACK_SCORE = {
    cards.GUTSY_SWING: 3000.0,
    cards.TRIFROST: 2000.0,
    cards.DESTINED_FIGHT: 1000.0,
    cards.ANNIHILAPE_TANTRUM: 40.0,
    cards.CONKELDURR_TANTRUM: 30.0,
}


def main_action_score(
    observation: Any,
    player_index: int,
    option: Any,
    *,
    seek_ready: bool,
    known_top: int | None,
) -> float:
    """Score a MAIN option by the combo component that it advances."""
    option_type = integer(getattr(option, "type", None), -1)
    card_id = view.option_card_id(observation, player_index, option)

    if option_type == int(OptionType.ABILITY):
        if card_id in _FREE_DRAW_POKEMON:
            return 8000.0
        if card_id == cards.ACADEMY_AT_NIGHT:
            if seek_ready and payload_in_hand(observation, player_index):
                return 4300.0
            return -1000.0
        return -500.0

    if option_type == int(OptionType.EVOLVE):
        if card_id == cards.SLOWKING:
            target = view.option_pokemon(observation, player_index, option)
            return 5600.0 + attachment_progress_score(
                observation,
                player_index,
                target,
                cards.BASIC_PSYCHIC,
            )
        return -500.0

    if option_type == int(OptionType.ATTACH):
        return attachment_main_score(observation, player_index, option)

    if option_type == int(OptionType.PLAY):
        return play_score(
            observation,
            player_index,
            card_id,
            seek_ready=seek_ready,
        )

    if option_type == int(OptionType.RETREAT):
        if (
            not ready_active_slowking(observation, player_index)
            and ready_bench_slowking(observation, player_index)
        ):
            return 4700.0
        return -1200.0

    if option_type == int(OptionType.ATTACK):
        return main_attack_score(
            observation,
            player_index,
            option,
            known_top=known_top,
        )
    if option_type == int(OptionType.END):
        return 0.0
    return -250.0


def play_score(
    observation: Any,
    player_index: int,
    card_id: int | None,
    *,
    seek_ready: bool,
) -> float:
    """Score a playable hand card only when it advances a missing component."""
    if card_id is None:
        return -900.0
    if card_id in cards.PAYLOAD_POKEMON:
        # Payloads belong in hand/deck for Seek, never on the ordinary Bench.
        return -10000.0
    if card_id == cards.CIPHERMANIAC:
        return 4200.0 if seek_ready else -10000.0
    if card_id == cards.ACADEMY_AT_NIGHT:
        if (
            seek_ready
            and not academy_in_play(observation)
            and payload_in_hand(observation, player_index)
        ):
            return 4250.0
        return -900.0
    if card_id == cards.SLOWPOKE:
        return 5100.0 if should_play_slowpoke(observation, player_index) else -700.0
    if card_id == cards.LATIAS_EX:
        return 4900.0 if latias_needed(observation, player_index) else -650.0
    if card_id == cards.FEZANDIPITI_EX:
        return (
            4750.0
            if field_count(observation, player_index, cards.FEZANDIPITI_EX) == 0
            and bench_space(observation, player_index) > 1
            and len(view.bench(observation, player_index)) < 2
            else -700.0
        )
    if card_id == cards.SMOOCHUM:
        return (
            4650.0
            if field_count(observation, player_index, cards.SMOOCHUM) == 0
            and bench_space(observation, player_index) > 1
            and not view.bench(observation, player_index)
            else -700.0
        )
    if card_id == cards.MEGA_KANGASKHAN_EX:
        return (
            4500.0
            if bench_space(observation, player_index) > 1
            and not view.bench(observation, player_index)
            else -750.0
        )
    if card_id in {cards.POKE_PAD, cards.ULTRA_BALL}:
        return (
            4000.0
            if pokemon_search_useful(
                observation,
                player_index,
                seek_ready=seek_ready,
            )
            else -800.0
        )
    if card_id == cards.NIGHT_STRETCHER:
        return (
            4050.0
            if night_stretcher_useful(
                observation,
                player_index,
                seek_ready=seek_ready,
            )
            else -800.0
        )
    if card_id == cards.WONDROUS_PATCH:
        return (
            5000.0
            if wondrous_patch_useful(observation, player_index)
            else -800.0
        )
    if card_id == cards.MEOWTH_EX:
        if (
            bench_space(observation, player_index) > 1
            and (
                len(view.bench(observation, player_index)) < 2
                or seek_ready
            )
            and (
                not seek_ready
                or not stack_route_in_hand(observation, player_index)
            )
        ):
            return 3900.0
        return -750.0
    if card_id == cards.SECRET_BOX:
        return (
            3800.0
            if secret_box_useful(
                observation,
                player_index,
                seek_ready=seek_ready,
            )
            else -950.0
        )
    if card_id == cards.LILLIES_DETERMINATION:
        if pending_slowpoke_evolution(observation, player_index):
            return -1000.0
        hand_count = len(view.hand(observation, player_index))
        draw_count = 8 if own_prize_count(observation, player_index) == 6 else 6
        return 3200.0 if hand_count <= draw_count - 2 else -600.0
    # Do not burn arbitrary Trainers or fill the Bench merely because PLAY is
    # legal. A later turn may make the same resource actionable.
    return -700.0


def main_attack_score(
    observation: Any,
    player_index: int,
    option: Any,
    *,
    known_top: int | None,
) -> float:
    """Rank terminal attacks after all useful non-terminal components."""
    attack_id = integer(getattr(option, "attackId", None), -1)
    if attack_id == cards.SEEK_INSPIRATION:
        if known_top in cards.PAYLOAD_POKEMON:
            return 2900.0
        if known_top is not None:
            return -1000.0
        return 1100.0
    if attack_id == cards.SUPER_PSY_BOLT:
        return 1250.0
    if attack_id == cards.DELIGHTFUL_KISS:
        return (
            1500.0
            if underpowered_bench_line(observation, player_index)
            else 900.0
        )
    return 1000.0


def attachment_main_score(
    observation: Any,
    player_index: int,
    option: Any,
) -> float:
    """Distinguish an Energy attachment from the Counter Gain Tool."""
    card_id = view.option_card_id(observation, player_index, option)
    target = view.option_pokemon(observation, player_index, option)
    if card_id in cards.ENERGY_CARDS:
        progress = attachment_progress_score(
            observation,
            player_index,
            target,
            card_id,
        )
        return 5300.0 + progress if progress > 0 else -500.0
    if card_id == cards.COUNTER_GAIN:
        progress = counter_gain_target_score(observation, player_index, target)
        return 5200.0 + progress if progress > 0 else -600.0
    return -700.0


def attachment_prompt_score(
    observation: Any,
    player_index: int,
    option: Any,
    *,
    source_card: int | None,
) -> float:
    """Score a target or card in a follow-up attachment prompt."""
    effect_id = effect_card_id(observation)
    source = effect_id if effect_id is not None else source_card
    target = view.option_pokemon(observation, player_index, option)
    if target is not None:
        if source == cards.COUNTER_GAIN:
            return counter_gain_target_score(observation, player_index, target)
        energy_id = (
            source
            if source in cards.ENERGY_CARDS
            else cards.BASIC_PSYCHIC
        )
        return attachment_progress_score(
            observation,
            player_index,
            target,
            energy_id,
        )

    card_id = view.option_card_id(observation, player_index, option)
    return {
        cards.BASIC_PSYCHIC: 1000.0,
        cards.TELEPATH_PSYCHIC_ENERGY: 900.0,
        cards.BOOMERANG_ENERGY: 600.0,
        cards.COUNTER_GAIN: 300.0,
    }.get(card_id or 0, -100.0)


def attachment_progress_score(
    observation: Any,
    player_index: int,
    pokemon: Any,
    energy_card_id: int,
) -> float:
    """Return positive value only for Energy that advances a Slowking line."""
    card_id = view.card_id(pokemon)
    if card_id not in {cards.SLOWPOKE, cards.SLOWKING}:
        if (
            energy_card_id == cards.TELEPATH_PSYCHIC_ENERGY
            and slow_line_count(observation, player_index) < 2
            and bench_space(observation, player_index) > 0
            and card_id in {cards.SMOOCHUM, cards.LATIAS_EX}
        ):
            # Telepath also permits non-Psychic attachment, but only these
            # Psychic pivots trigger its engine-owned Slowpoke search.
            return 500.0
        return -100.0
    before = seek_cost_ready(observation, player_index, pokemon)
    if (
        before
        and energy_card_id == cards.TELEPATH_PSYCHIC_ENERGY
        and slow_line_count(observation, player_index) < 2
        and bench_space(observation, player_index) > 0
    ):
        # The attachment is redundant as Energy but still invokes Telepath's
        # engine-owned setup effect for the missing second line.
        return 450.0
    after = seek_cost_ready(
        observation,
        player_index,
        pokemon,
        extra_energy_card=energy_card_id,
    )
    if before:
        return -50.0
    active_bonus = 250.0 if pokemon is view.active(observation, player_index) else 0.0
    ready_bonus = 500.0 if after else 0.0
    evolution_bonus = (
        120.0
        if card_id == cards.SLOWPOKE
        and cards.SLOWKING in view.card_ids(view.hand(observation, player_index))
        else 0.0
    )
    telepath_bonus = (
        80.0
        if energy_card_id == cards.TELEPATH_PSYCHIC_ENERGY
        and slow_line_count(observation, player_index) < 2
        else 0.0
    )
    return 200.0 + active_bonus + ready_bonus + evolution_bonus + telepath_bonus


def counter_gain_target_score(
    observation: Any,
    player_index: int,
    pokemon: Any,
) -> float:
    """Prefer Counter Gain only on a line where its cost reduction matters."""
    if view.card_id(pokemon) != cards.SLOWKING or has_counter_gain(pokemon):
        return -100.0
    before = seek_cost_ready(observation, player_index, pokemon)
    after = seek_cost_ready(
        observation,
        player_index,
        pokemon,
        extra_counter_gain=True,
    )
    active_bonus = 250.0 if pokemon is view.active(observation, player_index) else 0.0
    ready_bonus = 600.0 if after and not before else 0.0
    future_bonus = 100.0 if behind_on_prizes(observation, player_index) else 20.0
    return 200.0 + active_bonus + ready_bonus + future_bonus


def copied_attack_score(option: Any) -> float:
    """Rank a blind Seek ATTACK prompt by engine attack identity."""
    attack_id = integer(getattr(option, "attackId", None), -1)
    # Unknown IDs remain deterministic without accidentally selecting option 0
    # merely because no payload identity was known before the prompt.
    return _BLIND_COPY_ATTACK_SCORE.get(attack_id, float(attack_id) / 10000.0)


def setup_active_score(
    observation: Any,
    player_index: int,
    option: Any,
) -> float:
    """Prefer a low-cost pivot or a Slowpoke in the opening Active Spot."""
    card_id = view.option_card_id(observation, player_index, option)
    return _SETUP_ACTIVE_SCORE.get(card_id or 0, 0.0)


def setup_bench_score(
    observation: Any,
    player_index: int,
    option: Any,
) -> float:
    """Rank only the two useful opening Bench components."""
    card_id = view.option_card_id(observation, player_index, option)
    if card_id == cards.LATIAS_EX and latias_needed(observation, player_index):
        return 2000.0
    if card_id == cards.SLOWPOKE:
        return 1000.0
    return {
        cards.FEZANDIPITI_EX: 900.0,
        cards.LATIAS_EX: 850.0,
        cards.SMOOCHUM: 800.0,
        cards.MEOWTH_EX: 700.0,
        cards.MEGA_KANGASKHAN_EX: 650.0,
        cards.CONKELDURR: 300.0,
        cards.KYUREM: 280.0,
        cards.ANNIHILAPE: 260.0,
    }.get(card_id or 0, -100.0)


def switch_score(observation: Any, player_index: int, option: Any) -> float:
    """Give the Slowking preference only to an actually ready copy attacker."""
    pokemon = view.option_pokemon(observation, player_index, option)
    card_id = view.card_id(pokemon)
    if card_id == cards.SLOWKING and seek_cost_ready(
        observation,
        player_index,
        pokemon,
    ):
        return 5000.0 + float(view.energy_count(pokemon))
    return {
        cards.SMOOCHUM: 1000.0,
        cards.LATIAS_EX: 800.0,
        cards.MEGA_KANGASKHAN_EX: 650.0,
        cards.FEZANDIPITI_EX: 600.0,
        cards.MEOWTH_EX: 550.0,
        cards.SLOWPOKE: 400.0,
        cards.SLOWKING: 300.0,
    }.get(card_id or 0, 200.0)


def search_score(
    observation: Any,
    player_index: int,
    source_card: int,
    option: Any,
) -> float:
    """Choose a search result for the component that justified the card."""
    card_id = view.option_card_id(observation, player_index, option)
    if card_id is None:
        return -100.0

    needs_payload = (
        ready_active_slowking(observation, player_index)
        and academy_in_play(observation)
        and not payload_in_hand(observation, player_index)
    )
    if (
        needs_payload
        and source_card in {
            cards.POKE_PAD,
            cards.ULTRA_BALL,
            cards.NIGHT_STRETCHER,
        }
        and card_id in cards.PAYLOAD_POKEMON
    ):
        return 3000.0 + v1_tactics.payload_score(
            observation,
            player_index,
            card_id,
        )

    missing = missing_line_card_ids(observation, player_index)
    if card_id in missing:
        return 2600.0 - float(missing.index(card_id) * 50)

    if source_card == cards.MEOWTH_EX:
        if ready_active_slowking(observation, player_index):
            return 2500.0 if card_id == cards.CIPHERMANIAC else 50.0
        return 2500.0 if card_id == cards.LILLIES_DETERMINATION else 50.0
    if source_card == cards.SECRET_BOX:
        return {
            cards.CIPHERMANIAC: 2400.0,
            cards.ACADEMY_AT_NIGHT: 2300.0,
            cards.COUNTER_GAIN: 1800.0,
            cards.POKE_PAD: 1700.0,
            cards.WONDROUS_PATCH: 1600.0,
        }.get(card_id, 50.0)
    if source_card == cards.NIGHT_STRETCHER:
        if card_id == cards.BASIC_PSYCHIC and underpowered_bench_line(
            observation,
            player_index,
        ):
            return 1500.0
        return 50.0
    return 50.0


def payload_in_hand(observation: Any, player_index: int) -> bool:
    """Return whether a copy payload is currently visible in hand."""
    return any(
        card_id in cards.PAYLOAD_POKEMON
        for card_id in view.card_ids(view.hand(observation, player_index))
    )


def stack_route_in_hand(observation: Any, player_index: int) -> bool:
    """Return whether hand already contains a direct top-deck route."""
    hand_ids = view.card_ids(view.hand(observation, player_index))
    return cards.CIPHERMANIAC in hand_ids or (
        academy_in_play(observation)
        and any(card_id in cards.PAYLOAD_POKEMON for card_id in hand_ids)
    )


def missing_line_card_ids(
    observation: Any,
    player_index: int,
) -> tuple[int, ...]:
    """Identify components needed to maintain two independent Slowking lines."""
    hand_ids = view.card_ids(view.hand(observation, player_index))
    field_ids = tuple(
        card_id
        for pokemon in view.field_pokemon(observation, player_index)
        if (card_id := view.card_id(pokemon)) is not None
    )
    slowpoke_field = field_ids.count(cards.SLOWPOKE)
    slowking_field = field_ids.count(cards.SLOWKING)
    line_count = slowpoke_field + slowking_field
    missing: list[int] = []
    if line_count < 2 and cards.SLOWPOKE not in hand_ids:
        missing.append(cards.SLOWPOKE)
    if (
        slowpoke_field > 0 or cards.SLOWPOKE in hand_ids
    ) and cards.SLOWKING not in hand_ids:
        missing.append(cards.SLOWKING)
    return tuple(missing)


def should_play_slowpoke(observation: Any, player_index: int) -> bool:
    """Play a base only while fewer than two Slowking lines occupy the field."""
    return (
        bench_space(observation, player_index) > 1
        and slow_line_count(observation, player_index) < 2
    )


def pending_slowpoke_evolution(observation: Any, player_index: int) -> bool:
    """Keep Slowking while a newly played Slowpoke waits to evolve."""
    hand_ids = view.card_ids(view.hand(observation, player_index))
    if cards.SLOWKING not in hand_ids:
        return False
    return any(
        view.card_id(pokemon) == cards.SLOWPOKE
        and bool(getattr(pokemon, "appearThisTurn", False))
        for pokemon in view.field_pokemon(observation, player_index)
    )


def slow_line_count(observation: Any, player_index: int) -> int:
    """Count Slowpoke/Slowking bodies already committed to the field."""
    return sum(
        view.card_id(pokemon) in {cards.SLOWPOKE, cards.SLOWKING}
        for pokemon in view.field_pokemon(observation, player_index)
    )


def field_count(observation: Any, player_index: int, card_id: int) -> int:
    """Count one card identity among the acting player's in-play Pokémon."""
    return v1_tactics.field_count(observation, player_index, card_id)


def own_prize_count(observation: Any, player_index: int) -> int:
    """Return the acting player's visible number of remaining Prize cards."""
    player = view.player(observation, player_index)
    return len(view.as_sequence(getattr(player, "prize", ())))


def academy_in_play(observation: Any) -> bool:
    """Return whether Academy at Night is the current Stadium."""
    return v1_tactics.academy_in_play(observation)


def ready_active_slowking(observation: Any, player_index: int) -> bool:
    """Return whether the Active Slowking can pay Seek's effective cost."""
    active = view.active(observation, player_index)
    return view.card_id(active) == cards.SLOWKING and seek_cost_ready(
        observation,
        player_index,
        active,
    )


def ready_bench_slowking(observation: Any, player_index: int) -> bool:
    """Return whether a Bench Slowking can pay Seek's effective cost."""
    return any(
        view.card_id(pokemon) == cards.SLOWKING
        and seek_cost_ready(observation, player_index, pokemon)
        for pokemon in view.bench(observation, player_index)
    )


def seek_cost_ready(
    observation: Any,
    player_index: int,
    pokemon: Any,
    *,
    extra_energy_card: int | None = None,
    extra_counter_gain: bool = False,
) -> bool:
    """Approximate Seek readiness from resolved Energy units and Tool state."""
    energies = tuple(view.as_sequence(getattr(pokemon, "energies", ())))
    total_energy = len(energies)
    psychic_energy = sum(
        integer(energy, -1) == _PSYCHIC_ENERGY_TYPE for energy in energies
    )
    if extra_energy_card is not None:
        total_energy += 1
        if extra_energy_card in {
            cards.BASIC_PSYCHIC,
            cards.TELEPATH_PSYCHIC_ENERGY,
        }:
            psychic_energy += 1
    discounted = behind_on_prizes(observation, player_index) and (
        extra_counter_gain or has_counter_gain(pokemon)
    )
    return psychic_energy >= 1 and total_energy >= (1 if discounted else 2)


def has_counter_gain(pokemon: Any) -> bool:
    """Return whether Counter Gain is attached as a Tool."""
    return cards.COUNTER_GAIN in view.card_ids(
        view.as_sequence(getattr(pokemon, "tools", ()))
    )


def behind_on_prizes(observation: Any, player_index: int) -> bool:
    """Return whether Counter Gain's prize-count condition is active."""
    own = view.player(observation, player_index)
    opponent = view.player(observation, 1 - player_index)
    own_prizes = len(view.as_sequence(getattr(own, "prize", ())))
    opponent_prizes = len(view.as_sequence(getattr(opponent, "prize", ())))
    return own_prizes > opponent_prizes


def pokemon_search_useful(
    observation: Any,
    player_index: int,
    *,
    seek_ready: bool,
) -> bool:
    """Return whether Ball/Pad has a currently actionable target."""
    payload_route = (
        seek_ready
        and academy_in_play(observation)
        and not payload_in_hand(observation, player_index)
    )
    return payload_route or bool(missing_line_card_ids(observation, player_index))


def night_stretcher_useful(
    observation: Any,
    player_index: int,
    *,
    seek_ready: bool,
) -> bool:
    """Return whether Night Stretcher can recover an actionable component."""
    discard_ids = set(view.card_ids(view.discard(observation, player_index)))
    if (
        seek_ready
        and academy_in_play(observation)
        and not payload_in_hand(observation, player_index)
        and bool(discard_ids & cards.PAYLOAD_POKEMON)
    ):
        return True
    if discard_ids & set(missing_line_card_ids(observation, player_index)):
        return True
    return (
        cards.BASIC_PSYCHIC in discard_ids
        and underpowered_bench_line(observation, player_index)
    )


def wondrous_patch_useful(observation: Any, player_index: int) -> bool:
    """Return whether Patch can accelerate a visible Bench Slowking line."""
    discard_ids = view.card_ids(view.discard(observation, player_index))
    return cards.BASIC_PSYCHIC in discard_ids and underpowered_bench_line(
        observation,
        player_index,
    )


def secret_box_useful(
    observation: Any,
    player_index: int,
    *,
    seek_ready: bool,
) -> bool:
    """Use the expensive ACE SPEC only for a missing line or stack route."""
    return bool(missing_line_card_ids(observation, player_index)) or (
        seek_ready and not stack_route_in_hand(observation, player_index)
    )


def underpowered_bench_line(observation: Any, player_index: int) -> bool:
    """Return whether a Bench Slowpoke/Slowking still needs Energy."""
    return any(
        view.card_id(pokemon) in {cards.SLOWPOKE, cards.SLOWKING}
        and not seek_cost_ready(observation, player_index, pokemon)
        for pokemon in view.bench(observation, player_index)
    )


def latias_needed(observation: Any, player_index: int) -> bool:
    """Return whether Skyliner unlocks retreat for the current Basic Active."""
    field_ids = {
        view.card_id(pokemon)
        for pokemon in view.field_pokemon(observation, player_index)
    }
    if cards.LATIAS_EX in field_ids:
        return False
    active_id = view.card_id(view.active(observation, player_index))
    return active_id in {
        cards.SLOWPOKE,
        cards.FEZANDIPITI_EX,
        cards.MEGA_KANGASKHAN_EX,
        cards.MEOWTH_EX,
    }


def bench_space(observation: Any, player_index: int) -> int:
    """Return the number of visible free Bench slots."""
    player = view.player(observation, player_index)
    maximum = integer(getattr(player, "benchMax", None), 5)
    return max(0, maximum - len(view.bench(observation, player_index)))


def effect_card_id(observation: Any) -> int | None:
    """Return the card identity attached to the current effect prompt."""
    effect = getattr(getattr(observation, "select", None), "effect", None)
    raw_id = getattr(effect, "id", None)
    if raw_id is None:
        raw_id = getattr(effect, "cardId", None)
    try:
        return int(raw_id) if raw_id is not None else None
    except (TypeError, ValueError):
        return None


def contains_card(cards_or_ids: Collection[Any], card_id: int) -> bool:
    """Return whether a card collection contains the requested identity."""
    return card_id in {
        value if isinstance(value, int) else view.card_id(value)
        for value in cards_or_ids
    }


def integer(value: Any, default: int) -> int:
    """Best-effort integer conversion."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
