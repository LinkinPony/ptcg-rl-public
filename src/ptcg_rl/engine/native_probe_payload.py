"""Binary payload parsing for the native probe backend."""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ptcg_rl.engine.constants import AreaType, LogType

_MAGIC = 0x31504743
_VERSION = 2
_INT = struct.Struct("<i")


@dataclass(frozen=True)
class NativeProbeTransition:
    """One native probe transition decoded from ``libcg_probe.so``."""

    error: int
    resolved: bool
    forced_steps: int
    before_state: Mapping[str, Any]
    after_state: Mapping[str, Any]
    logs: tuple[Mapping[str, Any], ...]


def parse_native_probe_payload(
    payload: bytes,
    *,
    expected_worlds: int,
    expected_candidates: int,
) -> tuple[NativeProbeTransition, ...]:
    """Parse one native probe payload into compact transitions."""
    offset = 0
    magic, offset = _read_int(payload, offset)
    version, offset = _read_int(payload, offset)
    worlds, offset = _read_int(payload, offset)
    candidates, offset = _read_int(payload, offset)
    if magic != _MAGIC or version != _VERSION:
        raise RuntimeError("native probe payload has an unsupported format")
    if worlds != expected_worlds or candidates != expected_candidates:
        raise RuntimeError("native probe payload dimensions do not match request")
    transitions: list[NativeProbeTransition] = []
    for _ in range(worlds * candidates):
        error, offset = _read_int(payload, offset)
        resolved, offset = _read_int(payload, offset)
        forced_steps, offset = _read_int(payload, offset)
        if resolved not in (0, 1):
            raise RuntimeError("native probe payload has an invalid resolved flag")
        if forced_steps < 0:
            raise RuntimeError("native probe payload has a negative forced-step count")
        before_state, offset = _read_state(payload, offset)
        after_state, offset = _read_state(payload, offset)
        logs, offset = _read_logs(payload, offset)
        transitions.append(
            NativeProbeTransition(
                error=error,
                resolved=bool(resolved),
                forced_steps=forced_steps,
                before_state=before_state,
                after_state=after_state,
                logs=logs,
            )
        )
    if offset != len(payload):
        raise RuntimeError("native probe payload has trailing bytes")
    return tuple(transitions)


def _read_int(payload: bytes, offset: int) -> tuple[int, int]:
    if offset + _INT.size > len(payload):
        raise RuntimeError("native probe payload ended unexpectedly")
    return int(_INT.unpack_from(payload, offset)[0]), offset + _INT.size


def _read_state(payload: bytes, offset: int) -> tuple[Mapping[str, Any], int]:
    your_index, offset = _read_int(payload, offset)
    result, offset = _read_int(payload, offset)
    prize_0, offset = _read_int(payload, offset)
    prize_1, offset = _read_int(payload, offset)
    pokemon_count, offset = _read_int(payload, offset)
    players: list[dict[str, Any]] = [
        {
            "active": [],
            "bench": [],
            "prize": [None] * max(0, prize_0),
        },
        {
            "active": [],
            "bench": [],
            "prize": [None] * max(0, prize_1),
        },
    ]
    for _ in range(pokemon_count):
        player_index, offset = _read_int(payload, offset)
        area, offset = _read_int(payload, offset)
        area_index, offset = _read_int(payload, offset)
        visible, offset = _read_int(payload, offset)
        card_id, offset = _read_int(payload, offset)
        serial, offset = _read_int(payload, offset)
        hp, offset = _read_int(payload, offset)
        max_hp, offset = _read_int(payload, offset)
        energy_count, offset = _read_int(payload, offset)
        if player_index not in (0, 1) or area_index < 0:
            continue
        pokemon: Mapping[str, Any] | None
        if visible:
            pokemon = {
                "id": card_id,
                "serial": serial,
                "playerIndex": player_index,
                "hp": hp,
                "maxHp": max_hp,
                "energyCards": [None] * max(0, energy_count),
            }
        else:
            pokemon = None
        if area == int(AreaType.ACTIVE):
            _assign_index(players[player_index]["active"], area_index, pokemon)
        elif area == int(AreaType.BENCH):
            _assign_index(players[player_index]["bench"], area_index, pokemon)
    return {
        "yourIndex": your_index,
        "result": result,
        "players": players,
    }, offset


def _assign_index(items: list[Any], index: int, value: Any) -> None:
    while len(items) <= index:
        items.append(None)
    items[index] = value


def _read_logs(
    payload: bytes,
    offset: int,
) -> tuple[tuple[Mapping[str, Any], ...], int]:
    log_count, offset = _read_int(payload, offset)
    logs: list[Mapping[str, Any]] = []
    for _ in range(log_count):
        log_type, offset = _read_int(payload, offset)
        params: list[int] = []
        for _ in range(7):
            value, offset = _read_int(payload, offset)
            params.append(value)
        logs.append(_log_from_wire(log_type, params))
    return tuple(logs), offset


def _log_from_wire(log_type: int, params: Sequence[int]) -> Mapping[str, Any]:
    log: dict[str, Any] = {"type": log_type}
    if log_type in {
        int(LogType.SHUFFLE),
        int(LogType.HAS_BASIC_POKEMON),
        int(LogType.TURN_START),
        int(LogType.TURN_END),
        int(LogType.DRAW_REVERSE),
        int(LogType.MOVE_CARD_REVERSE),
        int(LogType.PLAY),
        int(LogType.MOVE_ATTACHED),
    }:
        log["playerIndex"] = params[0]
    if log_type == int(LogType.HAS_BASIC_POKEMON):
        log["hasBasicPokemon"] = bool(params[1])
    elif log_type == int(LogType.DRAW):
        log.update(
            {"playerIndex": params[0], "cardId": params[1], "serial": params[2]}
        )
    elif log_type == int(LogType.MOVE_CARD):
        log.update(
            {
                "playerIndex": params[0],
                "cardId": params[1],
                "serial": params[2],
                "fromArea": params[3],
                "toArea": params[4],
            }
        )
    elif log_type == int(LogType.MOVE_CARD_REVERSE):
        log.update({"fromArea": params[1], "toArea": params[2]})
    elif log_type == int(LogType.SWITCH):
        log.update(
            {
                "playerIndex": params[0],
                "cardIdActive": params[1],
                "serialActive": params[2],
                "cardIdBench": params[3],
                "serialBench": params[4],
            }
        )
    elif log_type == int(LogType.CHANGE):
        log.update(
            {
                "playerIndex": params[0],
                "cardIdBefore": params[1],
                "serialBefore": params[2],
                "cardIdAfter": params[3],
                "serialAfter": params[4],
            }
        )
    elif log_type in {int(LogType.ATTACH), int(LogType.EVOLVE), int(LogType.DEVOLVE)}:
        log.update(
            {
                "playerIndex": params[0],
                "cardId": params[1],
                "serial": params[2],
                "cardIdTarget": params[3],
                "serialTarget": params[4],
            }
        )
    elif log_type == int(LogType.MOVE_ATTACHED):
        log.update(
            {
                "cardId": params[1],
                "serial": params[2],
                "cardIdBefore": params[3],
                "serialBefore": params[4],
                "cardIdAfter": params[5],
                "serialAfter": params[6],
            }
        )
    elif log_type == int(LogType.ATTACK):
        log.update(
            {
                "playerIndex": params[0],
                "cardId": params[1],
                "serial": params[2],
                "attackId": params[3],
            }
        )
    elif log_type == int(LogType.HP_CHANGE):
        log.update(
            {
                "playerIndex": params[0],
                "cardId": params[1],
                "serial": params[2],
                "value": params[3],
                "putDamageCounter": bool(params[4]),
            }
        )
    elif log_type in {
        int(LogType.POISONED),
        int(LogType.BURNED),
        int(LogType.ASLEEP),
        int(LogType.PARALYZED),
        int(LogType.CONFUSED),
    }:
        log.update(
            {
                "playerIndex": params[0],
                "isRecover": bool(params[1]),
                "cardId": params[2],
                "serial": params[3],
            }
        )
    elif log_type == int(LogType.COIN):
        log.update({"playerIndex": params[0], "head": bool(params[1])})
    elif log_type == int(LogType.RESULT):
        log.update({"result": params[0], "reason": params[1]})
    return log
