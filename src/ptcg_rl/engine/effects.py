"""Parse engine logs into structured effect consequences."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from ptcg_rl.engine.constants import AreaType, LogType, SpecialCondition
from ptcg_rl.engine.effect_types import (
    AttachmentEvent,
    AttackEvent,
    BoardPosition,
    CardMove,
    CardRef,
    CoinFlip,
    DrawEvent,
    EffectSummary,
    EvolutionEvent,
    HpChange,
    Knockout,
    MatchResult,
    PokemonSnapshot,
    StatusChange,
    SwitchEvent,
    TargetAmount,
)
from ptcg_rl.engine.protocols import LogLike, PokemonLike

PLAYER_COUNT = 2
EffectLogStep = tuple[Sequence[Any], Any | None, Any | None]


def parse_effect_logs(
    logs: Sequence[Any],
    *,
    before_state: Any | None = None,
    after_state: Any | None = None,
) -> EffectSummary:
    """Convert engine logs plus optional state diff into an ``EffectSummary``."""
    before_pokemon = index_visible_pokemon(before_state)
    after_pokemon = index_visible_pokemon(after_state)

    hp_changes: list[HpChange] = []
    status_changes: list[StatusChange] = []
    coins: list[CoinFlip] = []
    card_moves: list[CardMove] = []
    draws: list[DrawEvent] = []
    attacks: list[AttackEvent] = []
    attachments: list[AttachmentEvent] = []
    evolutions: list[EvolutionEvent] = []
    switches: list[SwitchEvent] = []
    knockouts: list[Knockout] = []
    result: MatchResult | None = None
    draws_by_player = [0, 0]
    hidden_draws_by_player = [0, 0]
    prizes_taken_by_player = [0, 0]
    energy_delta_by_player = [0, 0]
    unknown_log_types: list[int] = []

    for log in logs:
        log_type = int(_field(log, "type", -1))
        if log_type == LogType.HP_CHANGE:
            change = _parse_hp_change(log, before_pokemon, after_pokemon)
            if change is not None:
                hp_changes.append(change)
            continue
        if log_type in _STATUS_LOG_TYPES:
            status_change = _parse_status_change(log, _STATUS_LOG_TYPES[log_type])
            if status_change is not None:
                status_changes.append(status_change)
            continue
        if log_type == LogType.COIN:
            coins.append(
                CoinFlip(
                    player_index=_field(log, "playerIndex"),
                    head=bool(_field(log, "head")),
                )
            )
            continue
        if log_type in {LogType.MOVE_CARD, LogType.MOVE_CARD_REVERSE}:
            move = _parse_card_move(log, hidden=log_type == LogType.MOVE_CARD_REVERSE)
            card_moves.append(move)
            _update_move_derived_counts(move, prizes_taken_by_player, energy_delta_by_player)
            continue
        if log_type in {LogType.DRAW, LogType.DRAW_REVERSE}:
            draw = _parse_draw(log, hidden=log_type == LogType.DRAW_REVERSE)
            draws.append(draw)
            # DRAW_REVERSE hides card identity, not the publicly visible count.
            _increment_counter(draws_by_player, draw.player_index)
            if draw.hidden:
                _increment_counter(hidden_draws_by_player, draw.player_index)
            continue
        if log_type == LogType.ATTACK:
            attack = _parse_attack(log)
            if attack is not None:
                attacks.append(attack)
            continue
        if log_type == LogType.ATTACH:
            attachment = _parse_attachment(log)
            if attachment is not None:
                attachments.append(attachment)
            continue
        if log_type in {LogType.EVOLVE, LogType.DEVOLVE}:
            evolution = _parse_evolution(log, devolve=log_type == LogType.DEVOLVE)
            if evolution is not None:
                evolutions.append(evolution)
            continue
        if log_type == LogType.SWITCH:
            switch = _parse_switch(log)
            if switch is not None:
                switches.append(switch)
            continue
        if log_type == LogType.RESULT:
            log_result = _field(log, "result")
            if log_result is not None:
                result = MatchResult(
                    result=int(log_result),
                    reason=_field(log, "reason"),
                )
            continue
        if log_type not in _IGNORED_LOG_TYPES:
            unknown_log_types.append(log_type)

    if before_state is not None and after_state is not None:
        prizes_taken_by_player = list(_prize_delta_by_player(before_state, after_state))
        energy_delta_by_player = list(_energy_delta_by_player(before_state, after_state))
        knockouts.extend(
            _conservative_knockouts(
                before_state,
                after_state,
                card_moves=card_moves,
                hp_changes=hp_changes,
                prizes_taken_by_player=prizes_taken_by_player,
            )
        )

    return EffectSummary(
        hp_changes=tuple(hp_changes),
        damage_by_target=_aggregate_hp_amounts(hp_changes, damage=True),
        healing_by_target=_aggregate_hp_amounts(hp_changes, damage=False),
        status_changes=tuple(status_changes),
        coins=tuple(coins),
        card_moves=tuple(card_moves),
        draws=tuple(draws),
        attacks=tuple(attacks),
        attachments=tuple(attachments),
        evolutions=tuple(evolutions),
        switches=tuple(switches),
        knockouts=tuple(knockouts),
        result=result,
        draws_by_player=_as_player_tuple(draws_by_player),
        hidden_draws_by_player=_as_player_tuple(hidden_draws_by_player),
        prizes_taken_by_player=_as_player_tuple(prizes_taken_by_player),
        energy_delta_by_player=_as_player_tuple(energy_delta_by_player),
        unknown_log_types=tuple(unknown_log_types),
    )


def parse_effect_log_steps(steps: Sequence[EffectLogStep]) -> EffectSummary:
    """Parse and combine engine logs grounded against each Search API step.

    Forced prompt chains can contain several state transitions. Grounding every
    log against only the root and final states repeats the full HP delta for
    every log on the same target. Each step therefore retains its own before
    and after state; the resulting summaries are combined only afterwards.
    """
    summaries = tuple(
        parse_effect_logs(logs, before_state=before, after_state=after)
        for logs, before, after in steps
    )
    combined = _merge_effect_summaries(summaries)
    if not steps or steps[0][1] is None or steps[-1][2] is None:
        return combined
    chain_knockouts = _conservative_knockouts(
        steps[0][1],
        steps[-1][2],
        card_moves=combined.card_moves,
        hp_changes=combined.hp_changes,
        prizes_taken_by_player=combined.prizes_taken_by_player,
    )
    knockouts_by_target = {
        knockout.target: knockout
        for knockout in (*combined.knockouts, *chain_knockouts)
    }
    return replace(combined, knockouts=tuple(knockouts_by_target.values()))


def _merge_effect_summaries(summaries: Sequence[EffectSummary]) -> EffectSummary:
    hp_changes = tuple(change for summary in summaries for change in summary.hp_changes)
    knockouts: list[Knockout] = []
    seen_knockouts: set[CardRef] = set()
    for summary in summaries:
        for knockout in summary.knockouts:
            if knockout.target not in seen_knockouts:
                knockouts.append(knockout)
                seen_knockouts.add(knockout.target)
    result = next(
        (summary.result for summary in reversed(summaries) if summary.result is not None),
        None,
    )
    return EffectSummary(
        hp_changes=hp_changes,
        damage_by_target=_aggregate_hp_amounts(hp_changes, damage=True),
        healing_by_target=_aggregate_hp_amounts(hp_changes, damage=False),
        status_changes=tuple(
            event for summary in summaries for event in summary.status_changes
        ),
        coins=tuple(event for summary in summaries for event in summary.coins),
        card_moves=tuple(event for summary in summaries for event in summary.card_moves),
        draws=tuple(event for summary in summaries for event in summary.draws),
        attacks=tuple(event for summary in summaries for event in summary.attacks),
        attachments=tuple(
            event for summary in summaries for event in summary.attachments
        ),
        evolutions=tuple(
            event for summary in summaries for event in summary.evolutions
        ),
        switches=tuple(event for summary in summaries for event in summary.switches),
        knockouts=tuple(knockouts),
        result=result,
        draws_by_player=_sum_player_counters(
            summary.draws_by_player for summary in summaries
        ),
        hidden_draws_by_player=_sum_player_counters(
            summary.hidden_draws_by_player for summary in summaries
        ),
        prizes_taken_by_player=_sum_player_counters(
            summary.prizes_taken_by_player for summary in summaries
        ),
        energy_delta_by_player=_sum_player_counters(
            summary.energy_delta_by_player for summary in summaries
        ),
        unknown_log_types=tuple(
            log_type for summary in summaries for log_type in summary.unknown_log_types
        ),
    )


def index_visible_pokemon(
    state: Any | None,
) -> Mapping[tuple[int, int], PokemonSnapshot]:
    """Index visible in-play Pokemon by ``(player_index, serial)``."""
    if state is None:
        return {}
    indexed: dict[tuple[int, int], PokemonSnapshot] = {}
    for player_index, player_state in enumerate(
        _sequence(_field(state, "players", ()))
    ):
        for active_index, pokemon in enumerate(
            _sequence(_field(player_state, "active", ()))
        ):
            if pokemon is not None:
                snapshot = _pokemon_snapshot(
                    pokemon,
                    player_index=player_index,
                    position=BoardPosition(int(AreaType.ACTIVE), active_index),
                )
                indexed[(player_index, snapshot.ref.serial)] = snapshot
        for bench_index, pokemon in enumerate(
            _sequence(_field(player_state, "bench", ()))
        ):
            snapshot = _pokemon_snapshot(
                pokemon,
                player_index=player_index,
                position=BoardPosition(int(AreaType.BENCH), bench_index),
            )
            indexed[(player_index, snapshot.ref.serial)] = snapshot
    return indexed


def active_pokemon_ref(state: Any | None, player_index: int) -> CardRef | None:
    """Return the visible active Pokemon ref for ``player_index``."""
    players = _sequence(_field(state, "players", ())) if state is not None else ()
    if player_index < 0 or player_index >= len(players):
        return None
    active = _sequence(_field(players[player_index], "active", ()))
    if not active or active[0] is None:
        return None
    pokemon = active[0]
    return CardRef(
        player_index,
        _int_field(pokemon, "id", 0),
        _int_field(pokemon, "serial", 0),
    )


def bench_pokemon_refs(state: Any | None, player_index: int) -> tuple[CardRef, ...]:
    """Return visible bench Pokemon refs for ``player_index``."""
    players = _sequence(_field(state, "players", ())) if state is not None else ()
    if player_index < 0 or player_index >= len(players):
        return ()
    return tuple(
        CardRef(
            player_index,
            _int_field(pokemon, "id", 0),
            _int_field(pokemon, "serial", 0),
        )
        for pokemon in _sequence(_field(players[player_index], "bench", ()))
    )


def total_attached_energy_cards(state: Any, player_index: int) -> int:
    """Count visible attached energy cards for one player."""
    players = _sequence(_field(state, "players", ()))
    player_state = players[player_index]
    total = 0
    for pokemon in _sequence(_field(player_state, "active", ())):
        if pokemon is not None:
            total += len(_sequence(_field(pokemon, "energyCards", ())))
    for pokemon in _sequence(_field(player_state, "bench", ())):
        total += len(_sequence(_field(pokemon, "energyCards", ())))
    return total


def _pokemon_snapshot(
    pokemon: PokemonLike,
    *,
    player_index: int,
    position: BoardPosition,
) -> PokemonSnapshot:
    return PokemonSnapshot(
        ref=CardRef(
            player_index,
            _int_field(pokemon, "id", 0),
            _int_field(pokemon, "serial", 0),
        ),
        hp=_int_field(pokemon, "hp", 0),
        max_hp=_int_field(pokemon, "maxHp", 0),
        position=position,
    )


def _parse_hp_change(
    log: LogLike,
    before_pokemon: Mapping[tuple[int, int], PokemonSnapshot],
    after_pokemon: Mapping[tuple[int, int], PokemonSnapshot],
) -> HpChange | None:
    player_index = _field(log, "playerIndex")
    card_id = _field(log, "cardId")
    serial = _field(log, "serial")
    if player_index is None or card_id is None or serial is None:
        return None
    target = CardRef(int(player_index), int(card_id), int(serial))
    before = before_pokemon.get((target.player_index, target.serial))
    after = after_pokemon.get((target.player_index, target.serial))
    raw_value = int(_field(log, "value", 0) or 0)
    damage, healing = _ground_hp_amounts(raw_value, before, after)
    return HpChange(
        target=target,
        raw_value=raw_value,
        damage=damage,
        healing=healing,
        put_damage_counter=bool(_field(log, "putDamageCounter")),
        before=before,
        after=after,
    )


def _ground_hp_amounts(
    raw_value: int,
    before: PokemonSnapshot | None,
    after: PokemonSnapshot | None,
) -> tuple[int, int]:
    # The engine wire contract uses negative values for damage and positive
    # values for healing. The log remains available when a lethal target has
    # already disappeared from the successor state, and it also preserves
    # distinct multi-hit events. State delta is only a fallback for a zero log.
    if raw_value < 0:
        return -raw_value, 0
    if raw_value > 0:
        return 0, raw_value
    if before is not None and after is not None:
        hp_delta = before.hp - after.hp
        if hp_delta > 0:
            return hp_delta, 0
        if hp_delta < 0:
            return 0, -hp_delta
    return 0, 0


_STATUS_LOG_TYPES: Mapping[int, SpecialCondition] = {
    int(LogType.POISONED): SpecialCondition.POISONED,
    int(LogType.BURNED): SpecialCondition.BURNED,
    int(LogType.ASLEEP): SpecialCondition.ASLEEP,
    int(LogType.PARALYZED): SpecialCondition.PARALYZED,
    int(LogType.CONFUSED): SpecialCondition.CONFUSED,
}


_IGNORED_LOG_TYPES = {
    int(LogType.SHUFFLE),
    int(LogType.HAS_BASIC_POKEMON),
    int(LogType.TURN_START),
    int(LogType.TURN_END),
    int(LogType.PLAY),
    int(LogType.CHANGE),
    int(LogType.MOVE_ATTACHED),
}


def _parse_status_change(
    log: LogLike,
    condition: SpecialCondition,
) -> StatusChange | None:
    player_index = _field(log, "playerIndex")
    card_id = _field(log, "cardId")
    serial = _field(log, "serial")
    if player_index is None or card_id is None or serial is None:
        return None
    return StatusChange(
        target=CardRef(int(player_index), int(card_id), int(serial)),
        condition=condition,
        recovered=bool(_field(log, "isRecover")),
    )


def _parse_card_move(log: LogLike, *, hidden: bool) -> CardMove:
    card: CardRef | None = None
    player_index = _field(log, "playerIndex")
    card_id = _field(log, "cardId")
    serial = _field(log, "serial")
    if not hidden and player_index is not None and card_id is not None and serial is not None:
        card = CardRef(int(player_index), int(card_id), int(serial))
    return CardMove(
        player_index=player_index,
        card=card,
        from_area=_field(log, "fromArea"),
        to_area=_field(log, "toArea"),
        hidden=hidden,
    )


def _parse_draw(log: LogLike, *, hidden: bool) -> DrawEvent:
    card: CardRef | None = None
    player_index = _field(log, "playerIndex")
    card_id = _field(log, "cardId")
    serial = _field(log, "serial")
    if not hidden and player_index is not None and card_id is not None and serial is not None:
        card = CardRef(int(player_index), int(card_id), int(serial))
    return DrawEvent(player_index=player_index, card=card, hidden=hidden)


def _parse_attack(log: LogLike) -> AttackEvent | None:
    if (
        _field(log, "playerIndex") is None
        or _field(log, "cardId") is None
        or _field(log, "serial") is None
        or _field(log, "attackId") is None
    ):
        return None
    return AttackEvent(
        attacker=CardRef(
            _int_field(log, "playerIndex", 0),
            _int_field(log, "cardId", 0),
            _int_field(log, "serial", 0),
        ),
        attack_id=_int_field(log, "attackId", 0),
    )


def _parse_attachment(log: LogLike) -> AttachmentEvent | None:
    if (
        _field(log, "playerIndex") is None
        or _field(log, "cardId") is None
        or _field(log, "serial") is None
        or _field(log, "cardIdTarget") is None
        or _field(log, "serialTarget") is None
    ):
        return None
    return AttachmentEvent(
        player_index=_int_field(log, "playerIndex", 0),
        attached=CardRef(
            _int_field(log, "playerIndex", 0),
            _int_field(log, "cardId", 0),
            _int_field(log, "serial", 0),
        ),
        target=CardRef(
            _int_field(log, "playerIndex", 0),
            _int_field(log, "cardIdTarget", 0),
            _int_field(log, "serialTarget", 0),
        ),
    )


def _parse_evolution(log: LogLike, *, devolve: bool) -> EvolutionEvent | None:
    if (
        _field(log, "playerIndex") is None
        or _field(log, "cardId") is None
        or _field(log, "serial") is None
        or _field(log, "cardIdTarget") is None
        or _field(log, "serialTarget") is None
    ):
        return None
    return EvolutionEvent(
        player_index=_int_field(log, "playerIndex", 0),
        card=CardRef(
            _int_field(log, "playerIndex", 0),
            _int_field(log, "cardId", 0),
            _int_field(log, "serial", 0),
        ),
        target=CardRef(
            _int_field(log, "playerIndex", 0),
            _int_field(log, "cardIdTarget", 0),
            _int_field(log, "serialTarget", 0),
        ),
        devolve=devolve,
    )


def _parse_switch(log: LogLike) -> SwitchEvent | None:
    if (
        _field(log, "playerIndex") is None
        or _field(log, "cardIdActive") is None
        or _field(log, "serialActive") is None
        or _field(log, "cardIdBench") is None
        or _field(log, "serialBench") is None
    ):
        return None
    player_index = _int_field(log, "playerIndex", 0)
    return SwitchEvent(
        player_index=player_index,
        active_to_bench=CardRef(
            player_index,
            _int_field(log, "cardIdActive", 0),
            _int_field(log, "serialActive", 0),
        ),
        bench_to_active=CardRef(
            player_index,
            _int_field(log, "cardIdBench", 0),
            _int_field(log, "serialBench", 0),
        ),
    )


def _update_move_derived_counts(
    move: CardMove,
    prizes_taken_by_player: list[int],
    energy_delta_by_player: list[int],
) -> None:
    if move.player_index is None:
        return
    if move.from_area == int(AreaType.PRIZE) and move.to_area == int(AreaType.HAND):
        _increment_counter(prizes_taken_by_player, move.player_index)
    if move.to_area == int(AreaType.ENERGY) and move.from_area != int(AreaType.ENERGY):
        _increment_counter(energy_delta_by_player, move.player_index)
    if move.from_area == int(AreaType.ENERGY) and move.to_area != int(AreaType.ENERGY):
        _increment_counter(energy_delta_by_player, move.player_index, delta=-1)


def _conservative_knockouts(
    before_state: Any,
    after_state: Any,
    *,
    card_moves: Sequence[CardMove],
    hp_changes: Sequence[HpChange],
    prizes_taken_by_player: Sequence[int],
) -> tuple[Knockout, ...]:
    """Return KOs supported by lethal HP or prize-taking evidence.

    A Pokemon disappearing from active/bench is not sufficient: switch,
    evolution, devolution, and direct zone effects can all remove that serial.
    """
    before = index_visible_pokemon(before_state)
    after = index_visible_pokemon(after_state)
    moves_from_play = {
        move.card
        for move in card_moves
        if move.card is not None
        and move.from_area in {int(AreaType.ACTIVE), int(AreaType.BENCH)}
        and move.to_area == int(AreaType.DISCARD)
    }
    damage_by_ref = {
        amount.target: amount.amount
        for amount in _aggregate_hp_amounts(hp_changes, damage=True)
    }
    inferred: list[Knockout] = []
    for key, snapshot in before.items():
        successor = after.get(key)
        disappeared = successor is None
        zero_hp = successor is not None and successor.hp <= 0
        if not disappeared and not zero_hp:
            continue
        lethal_damage = damage_by_ref.get(snapshot.ref, 0) >= max(1, snapshot.hp)
        prize_taker = 1 - snapshot.ref.player_index
        prize_evidence = (
            snapshot.ref in moves_from_play
            and 0 <= prize_taker < len(prizes_taken_by_player)
            and prizes_taken_by_player[prize_taker] > 0
        )
        if lethal_damage or prize_evidence:
            inferred.append(
                Knockout(target=snapshot.ref, from_area=int(snapshot.position.area))
            )
    return tuple(inferred)


def _prize_delta_by_player(
    before_state: Any,
    after_state: Any,
) -> tuple[int, int]:
    before_players = _sequence(_field(before_state, "players", ()))
    after_players = _sequence(_field(after_state, "players", ()))
    return (
        max(
            0,
            _prize_count(before_players, 0) - _prize_count(after_players, 0),
        ),
        max(
            0,
            _prize_count(before_players, 1) - _prize_count(after_players, 1),
        ),
    )


def _prize_count(players: Sequence[Any], player_index: int) -> int:
    if player_index < 0 or player_index >= len(players):
        return 0
    return len(_sequence(_field(players[player_index], "prize", ())))


def _energy_delta_by_player(
    before_state: Any,
    after_state: Any,
) -> tuple[int, int]:
    return (
        total_attached_energy_cards(after_state, 0)
        - total_attached_energy_cards(before_state, 0),
        total_attached_energy_cards(after_state, 1)
        - total_attached_energy_cards(before_state, 1),
    )


def _aggregate_hp_amounts(
    hp_changes: Sequence[HpChange],
    *,
    damage: bool,
) -> tuple[TargetAmount, ...]:
    totals: dict[CardRef, int] = {}
    for change in hp_changes:
        amount = change.damage if damage else change.healing
        if amount == 0:
            continue
        totals[change.target] = totals.get(change.target, 0) + amount
    return tuple(TargetAmount(target=target, amount=amount) for target, amount in totals.items())


def _increment_counter(
    counter: list[int],
    player_index: int | None,
    *,
    delta: int = 1,
) -> None:
    if player_index is None or player_index < 0 or player_index >= len(counter):
        return
    counter[int(player_index)] += delta


def _as_player_tuple(counter: Sequence[int]) -> tuple[int, int]:
    if len(counter) != PLAYER_COUNT:
        raise ValueError(f"expected {PLAYER_COUNT} player counters")
    return int(counter[0]), int(counter[1])


def _sum_player_counters(counters: Iterable[Sequence[int]]) -> tuple[int, int]:
    totals = [0, 0]
    for counter in counters:
        if len(counter) != PLAYER_COUNT:
            raise ValueError(f"expected {PLAYER_COUNT} player counters")
        totals[0] += int(counter[0])
        totals[1] += int(counter[1])
    return totals[0], totals[1]


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_field(value: Any, name: str, default: int) -> int:
    field_value = _field(value, name, default)
    return int(field_value) if field_value is not None else default
