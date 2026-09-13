"""Fast feature math for native probe compact transitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from ptcg_rl.engine.constants import AreaType, LogType, SpecialCondition
from ptcg_rl.engine.feature_vectors import (
    DYNAMIC_EFFECT_FEATURE_SIZE,
    MAX_BENCH_COUNT_FEATURE,
    MAX_BENCH_DAMAGE_FEATURE,
    MAX_COIN_FEATURE,
    MAX_DAMAGE_FEATURE,
    MAX_DISCARD_INFLOW_FEATURE,
    MAX_DRAW_FEATURE,
    MAX_ENERGY_DELTA_FEATURE,
    MAX_KO_COUNT_FEATURE,
    MAX_PRIZE_FEATURE,
)

_STATUS_LOG_TYPES: Mapping[int, int] = {
    int(LogType.POISONED): int(SpecialCondition.POISONED),
    int(LogType.BURNED): int(SpecialCondition.BURNED),
    int(LogType.ASLEEP): int(SpecialCondition.ASLEEP),
    int(LogType.PARALYZED): int(SpecialCondition.PARALYZED),
    int(LogType.CONFUSED): int(SpecialCondition.CONFUSED),
}
CardRef = tuple[int, int, int]
PokemonSnapshot = tuple[int, int, int, int]
_REF_NONE: CardRef | None = None


def native_transition_feature_vector(
    *,
    before_state: Mapping[str, Any],
    after_state: Mapping[str, Any],
    logs: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    """Build the 33-wide feature vector from a native compact transition."""
    perspective = _int_field(before_state, "yourIndex", 0)
    opponent = 1 - perspective
    before_index = _pokemon_index(before_state)
    after_index = _pokemon_index(after_state)
    opponent_active = _active_ref(before_state, opponent)
    self_active = _active_ref(before_state, perspective)
    opponent_bench = _bench_refs(before_state, opponent)
    self_bench = _bench_refs(before_state, perspective)

    damage_by_ref: dict[CardRef, int] = {}
    healing_by_ref: dict[CardRef, int] = {}
    status_by_ref: dict[CardRef, set[int]] = {}
    coins: list[bool] = []
    draws_by_player = [0, 0]
    self_discard_inflow = 0
    opponent_discard_inflow = 0
    moves_from_play_to_discard: set[CardRef] = set()

    for log in logs:
        log_type = _int_field(log, "type", -1)
        if log_type == int(LogType.HP_CHANGE):
            _add_hp_change(log, before_index, after_index, damage_by_ref, healing_by_ref)
            continue
        if log_type in _STATUS_LOG_TYPES:
            if not bool(log.get("isRecover")):
                target = _log_ref(log)
                if target is not None:
                    status_by_ref.setdefault(target, set()).add(
                        _STATUS_LOG_TYPES[log_type]
                    )
            continue
        if log_type == int(LogType.COIN):
            coins.append(bool(log.get("head")))
            continue
        if log_type in {int(LogType.DRAW), int(LogType.DRAW_REVERSE)}:
            # Reverse draws hide identity, but their count is public.
            _increment(draws_by_player, _optional_int(log.get("playerIndex")))
            continue
        if log_type in {int(LogType.MOVE_CARD), int(LogType.MOVE_CARD_REVERSE)}:
            player_index = _optional_int(log.get("playerIndex"))
            if log.get("toArea") == int(AreaType.DISCARD):
                if player_index == perspective:
                    self_discard_inflow += 1
                elif player_index == opponent:
                    opponent_discard_inflow += 1
            if (
                log_type == int(LogType.MOVE_CARD)
                and log.get("fromArea") in {int(AreaType.ACTIVE), int(AreaType.BENCH)}
                and log.get("toArea") == int(AreaType.DISCARD)
            ):
                target = _log_ref(log)
                if target is not None:
                    moves_from_play_to_discard.add(target)

    prizes_taken_by_player = (
        max(0, _prize_count(before_state, 0) - _prize_count(after_state, 0)),
        max(0, _prize_count(before_state, 1) - _prize_count(after_state, 1)),
    )
    knockouts = _conservative_knockouts(
        before_index,
        after_index,
        damage_by_ref=damage_by_ref,
        moves_from_play_to_discard=moves_from_play_to_discard,
        prizes_taken_by_player=prizes_taken_by_player,
    )
    opponent_bench_damage = [
        damage_by_ref.get(ref, 0) for ref in opponent_bench if ref is not None
    ]
    self_bench_damage = [
        damage_by_ref.get(ref, 0) for ref in self_bench if ref is not None
    ]
    terminal_win, terminal_loss, terminal_draw = _terminal_features(
        after_state,
        perspective,
    )
    self_ko_count = sum(1 for ref in knockouts if ref[0] == perspective)
    opponent_ko_count = sum(1 for ref in knockouts if ref[0] == opponent)
    vector = (
        _norm(_amount(damage_by_ref, opponent_active), MAX_DAMAGE_FEATURE),
        float(opponent_active is not None and opponent_active in knockouts),
        _norm(sum(opponent_bench_damage), MAX_BENCH_DAMAGE_FEATURE),
        _norm(max(opponent_bench_damage, default=0), MAX_DAMAGE_FEATURE),
        _norm(
            sum(1 for amount in opponent_bench_damage if amount > 0),
            MAX_BENCH_COUNT_FEATURE,
        ),
        _norm(_amount(damage_by_ref, self_active), MAX_DAMAGE_FEATURE),
        float(self_active is not None and self_active in knockouts),
        _norm(sum(self_bench_damage), MAX_BENCH_DAMAGE_FEATURE),
        *_status_flags(status_by_ref, opponent_active),
        _norm(draws_by_player[perspective], MAX_DRAW_FEATURE),
        _signed_norm(
            _total_energy(after_state, perspective)
            - _total_energy(before_state, perspective),
            MAX_ENERGY_DELTA_FEATURE,
        ),
        _norm(
            _prize_count(before_state, perspective)
            - _prize_count(after_state, perspective),
            MAX_PRIZE_FEATURE,
        ),
        _norm(len(coins), MAX_COIN_FEATURE),
        _norm(sum(1 for coin in coins if coin), MAX_COIN_FEATURE),
        terminal_win,
        terminal_loss,
        terminal_draw,
        *_status_flags(status_by_ref, self_active),
        _norm(_amount(healing_by_ref, self_active), MAX_DAMAGE_FEATURE),
        _signed_norm(
            _total_energy(after_state, opponent)
            - _total_energy(before_state, opponent),
            MAX_ENERGY_DELTA_FEATURE,
        ),
        _norm(draws_by_player[opponent], MAX_DRAW_FEATURE),
        _norm(self_discard_inflow, MAX_DISCARD_INFLOW_FEATURE),
        _norm(opponent_discard_inflow, MAX_DISCARD_INFLOW_FEATURE),
        _norm(self_ko_count, MAX_KO_COUNT_FEATURE),
        _norm(opponent_ko_count, MAX_KO_COUNT_FEATURE),
    )
    if len(vector) != DYNAMIC_EFFECT_FEATURE_SIZE:
        raise RuntimeError("native dynamic feature width mismatch")
    return tuple(float(value) for value in vector)


def _add_hp_change(
    log: Mapping[str, Any],
    before_index: Mapping[tuple[int, int], PokemonSnapshot],
    after_index: Mapping[tuple[int, int], PokemonSnapshot],
    damage_by_ref: dict[CardRef, int],
    healing_by_ref: dict[CardRef, int],
) -> None:
    target = _log_ref(log)
    if target is None:
        return
    key = (target[0], target[2])
    before = before_index.get(key)
    after = after_index.get(key)
    raw_value = _int_field(log, "value", 0)
    damage = 0
    healing = 0
    # Native v2 exposes only root/final states for a forced chain. The signed
    # engine log is exact per event and avoids repeating the full transition
    # delta for every hit; state delta is only a fallback for a zero-valued log.
    if raw_value < 0:
        damage = -raw_value
    elif raw_value > 0:
        healing = raw_value
    elif before is not None and after is not None:
        hp_delta = before[3] - after[3]
        if hp_delta > 0:
            damage = hp_delta
        elif hp_delta < 0:
            healing = -hp_delta
    if damage > 0:
        damage_by_ref[target] = damage_by_ref.get(target, 0) + damage
    if healing > 0:
        healing_by_ref[target] = healing_by_ref.get(target, 0) + healing


def _conservative_knockouts(
    before_index: Mapping[tuple[int, int], PokemonSnapshot],
    after_index: Mapping[tuple[int, int], PokemonSnapshot],
    *,
    damage_by_ref: Mapping[CardRef, int],
    moves_from_play_to_discard: set[CardRef],
    prizes_taken_by_player: Sequence[int],
) -> set[CardRef]:
    """Infer only KOs supported by lethal damage or prize evidence."""
    knockouts: set[CardRef] = set()
    for key, snapshot in before_index.items():
        successor = after_index.get(key)
        disappeared = successor is None
        zero_hp = successor is not None and successor[3] <= 0
        if not disappeared and not zero_hp:
            continue
        target = (snapshot[0], snapshot[1], snapshot[2])
        lethal_damage = damage_by_ref.get(target, 0) >= max(1, snapshot[3])
        prize_taker = 1 - snapshot[0]
        prize_evidence = (
            target in moves_from_play_to_discard
            and 0 <= prize_taker < len(prizes_taken_by_player)
            and prizes_taken_by_player[prize_taker] > 0
        )
        if lethal_damage or prize_evidence:
            knockouts.add(target)
    return knockouts


def _pokemon_index(
    state: Mapping[str, Any],
) -> dict[tuple[int, int], PokemonSnapshot]:
    indexed: dict[tuple[int, int], PokemonSnapshot] = {}
    for player_index, player in enumerate(_players(state)):
        for pokemon in _sequence(player.get("active", ())):
            _index_pokemon(indexed, pokemon, player_index)
        for pokemon in _sequence(player.get("bench", ())):
            _index_pokemon(indexed, pokemon, player_index)
    return indexed


def _index_pokemon(
    indexed: dict[tuple[int, int], PokemonSnapshot],
    pokemon: Any,
    player_index: int,
) -> None:
    if not isinstance(pokemon, Mapping):
        return
    serial = _int_field(pokemon, "serial", 0)
    indexed[(player_index, serial)] = (
        player_index,
        _int_field(pokemon, "id", 0),
        serial,
        _int_field(pokemon, "hp", 0),
    )


def _active_ref(state: Mapping[str, Any], player_index: int) -> CardRef | None:
    players = _players(state)
    if player_index < 0 or player_index >= len(players):
        return _REF_NONE
    active = _sequence(players[player_index].get("active", ()))
    if not active or not isinstance(active[0], Mapping):
        return _REF_NONE
    return _pokemon_ref(active[0], player_index)


def _bench_refs(state: Mapping[str, Any], player_index: int) -> tuple[CardRef, ...]:
    players = _players(state)
    if player_index < 0 or player_index >= len(players):
        return ()
    return tuple(
        ref
        for pokemon in _sequence(players[player_index].get("bench", ()))
        if isinstance(pokemon, Mapping)
        for ref in (_pokemon_ref(pokemon, player_index),)
        if ref is not None
    )


def _pokemon_ref(pokemon: Mapping[str, Any], player_index: int) -> CardRef:
    return (
        player_index,
        _int_field(pokemon, "id", 0),
        _int_field(pokemon, "serial", 0),
    )


def _log_ref(log: Mapping[str, Any]) -> CardRef | None:
    player_index = _optional_int(log.get("playerIndex"))
    card_id = _optional_int(log.get("cardId"))
    serial = _optional_int(log.get("serial"))
    if player_index is None or card_id is None or serial is None:
        return None
    return (player_index, card_id, serial)


def _players(state: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    players = state.get("players", ())
    return cast(Sequence[Mapping[str, Any]], players) if isinstance(players, Sequence) else ()


def _prize_count(state: Mapping[str, Any], player_index: int) -> int:
    players = _players(state)
    if player_index < 0 or player_index >= len(players):
        return 0
    return len(_sequence(players[player_index].get("prize", ())))


def _total_energy(state: Mapping[str, Any], player_index: int) -> int:
    players = _players(state)
    if player_index < 0 or player_index >= len(players):
        return 0
    total = 0
    for zone in ("active", "bench"):
        for pokemon in _sequence(players[player_index].get(zone, ())):
            if isinstance(pokemon, Mapping):
                total += len(_sequence(pokemon.get("energyCards", ())))
    return total


def _terminal_features(
    state: Mapping[str, Any],
    perspective_player: int,
) -> tuple[float, float, float]:
    result = _int_field(state, "result", -1)
    if result < 0:
        return 0.0, 0.0, 0.0
    if result == 2:
        return 0.0, 0.0, 1.0
    if result == perspective_player:
        return 1.0, 0.0, 0.0
    return 0.0, 1.0, 0.0


def _status_flags(
    status_by_ref: Mapping[CardRef, set[int]],
    ref: CardRef | None,
) -> tuple[float, float, float, float, float]:
    flags = [0.0] * len(SpecialCondition)
    if ref is None:
        return cast(tuple[float, float, float, float, float], tuple(flags))
    for status in status_by_ref.get(ref, ()):
        flags[int(status)] = 1.0
    return cast(tuple[float, float, float, float, float], tuple(flags))


def _amount(
    amounts: Mapping[CardRef, int],
    ref: CardRef | None,
) -> int:
    return amounts.get(ref, 0) if ref is not None else 0


def _increment(counter: list[int], player_index: int | None) -> None:
    if player_index is not None and 0 <= player_index < len(counter):
        counter[player_index] += 1


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _int_field(value: Mapping[str, Any], name: str, default: int) -> int:
    raw = value.get(name, default)
    return int(raw) if raw is not None else default


def _norm(value: int | float, denominator: float) -> float:
    return min(max(float(value), 0.0), denominator) / denominator


def _signed_norm(value: int | float, denominator: float) -> float:
    clipped = min(max(float(value), -denominator), denominator)
    return clipped / denominator
