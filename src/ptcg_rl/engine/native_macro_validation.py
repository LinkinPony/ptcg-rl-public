"""Correctness checks for exact native action macros."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

from ptcg_rl.engine.constants import AreaType, LogType
from ptcg_rl.engine.forward_model import dynamic_effect_feature_from_dict_resolution
from ptcg_rl.engine.native_macro import NativeMacro, NativeMacroBackend
from ptcg_rl.engine.native_probe_features import native_transition_feature_vector
from ptcg_rl.engine.native_probe_payload import NativeProbeTransition
from ptcg_rl.engine.search_api_probe import SearchApiProbeBackend
from ptcg_rl.engine.session import HiddenInformation


class NativeMacroValidationCase(Protocol):
    """Case fields required by native/Search macro validation."""

    @property
    def episode_id(self) -> int: ...

    @property
    def step_index(self) -> int: ...

    @property
    def player_index(self) -> int: ...

    @property
    def state_token(self) -> str: ...

    @property
    def hidden(self) -> HiddenInformation: ...

    @property
    def observed_macro(self) -> NativeMacro: ...

    @property
    def root_action(self) -> tuple[int, ...]: ...

    @property
    def discard_action(self) -> tuple[int, ...]: ...


def search_api_macro_parity(
    native_backend: NativeMacroBackend,
    cases: Sequence[NativeMacroValidationCase],
    *,
    attack_id: int,
) -> dict[str, Any]:
    """Compare native endpoints, logs, and features with the public Search API."""
    search_backend = SearchApiProbeBackend()
    failures: list[dict[str, Any]] = []
    max_feature_abs_diff = 0.0
    for case in cases:
        native = native_backend.run(
            case.state_token,
            hidden_worlds=[case.hidden],
            macros=[case.observed_macro],
        ).transitions[0]
        search_state = search_backend._begin(
            {"search_begin_input": case.state_token}, case.hidden
        )
        try:
            root_observation = cast(Mapping[str, Any], search_state["observation"])
            search_logs: tuple[Mapping[str, Any], ...] = ()
            for action in case.observed_macro:
                search_state = search_backend._step(
                    int(search_state["searchId"]), action
                )
                step_observation = cast(
                    Mapping[str, Any], search_state["observation"]
                )
                search_logs = merge_search_logs(
                    search_logs,
                    _mapping_logs(step_observation.get("logs")),
                )
            observation = cast(Mapping[str, Any], search_state["observation"])
            current = cast(Mapping[str, Any], observation["current"])
            native_summary = _state_summary(native.after_state)
            search_summary = _state_summary(current)
            native_signature = effect_log_signature(native.logs)
            search_signature = effect_log_signature(search_logs)
            native_feature = native_transition_feature_vector(
                before_state=native.before_state,
                after_state=native.after_state,
                logs=native.logs,
            )
            search_feature = dynamic_effect_feature_from_dict_resolution(
                select=case.root_action,
                before_observation=root_observation,
                after_observation=observation,
                logs=search_logs,
                perspective_player=case.player_index,
            ).vector
            feature_abs_diff = max(
                abs(float(native_value) - float(search_value))
                for native_value, search_value in zip(
                    native_feature,
                    search_feature,
                    strict=True,
                )
            )
            max_feature_abs_diff = max(max_feature_abs_diff, feature_abs_diff)
            native_facts_match = target_attack_fact_matches(
                case,
                native,
                attack_id=attack_id,
            )
            search_facts_match = target_attack_logs_match(
                case,
                search_logs,
                attack_id=attack_id,
                expected_discard_count=len(case.discard_action),
            )
            if (
                native.error != 0
                or not native.resolved
                or native_summary != search_summary
                or native_signature != search_signature
                or feature_abs_diff > 1.0e-6
                or not native_facts_match
                or not search_facts_match
            ):
                failures.append(
                    {
                        "episode_id": case.episode_id,
                        "step_index": case.step_index,
                        "native_error": native.error,
                        "native_resolved": native.resolved,
                        "native": native_summary,
                        "search_api": search_summary,
                        "native_log_signature": native_signature,
                        "search_api_log_signature": search_signature,
                        "feature_max_abs_diff": feature_abs_diff,
                        "native_facts_match": native_facts_match,
                        "search_api_facts_match": search_facts_match,
                    }
                )
        finally:
            search_backend._end()
    return {
        "comparison_count": len(cases),
        "failure_count": len(failures),
        "max_feature_abs_diff": max_feature_abs_diff,
        "failures": failures,
    }


def target_attack_fact_matches(
    case: NativeMacroValidationCase,
    transition: NativeProbeTransition,
    *,
    attack_id: int,
    expected_discard_count: int | None = None,
) -> bool:
    """Check an exact transition contains the expected target-attack facts."""
    if transition.error != 0 or not transition.resolved:
        return False
    return target_attack_logs_match(
        case,
        transition.logs,
        attack_id=attack_id,
        expected_discard_count=(
            len(case.discard_action)
            if expected_discard_count is None
            else expected_discard_count
        ),
    )


def target_attack_logs_match(
    case: NativeMacroValidationCase,
    logs: Sequence[Mapping[str, Any]],
    *,
    attack_id: int,
    expected_discard_count: int,
) -> bool:
    """Check attack identity, selected resource loss, and an effect event."""
    attacks = sum(
        int(log.get("type", -1)) == int(LogType.ATTACK)
        and int(log.get("playerIndex", -1)) == case.player_index
        and int(log.get("attackId", -1)) == attack_id
        for log in logs
    )
    hand_discards = sum(
        int(log.get("type", -1)) == int(LogType.MOVE_CARD)
        and int(log.get("playerIndex", -1)) == case.player_index
        and int(log.get("fromArea", -1)) == int(AreaType.HAND)
        and int(log.get("toArea", -1)) == int(AreaType.DISCARD)
        for log in logs
    )
    damage_events = sum(
        1
        for log in logs
        if int(log.get("type", -1)) == int(LogType.HP_CHANGE)
        and int(log.get("playerIndex", -1)) == 1 - case.player_index
    )
    return (
        attacks == 1
        and hand_discards == expected_discard_count
        and (expected_discard_count == 0 or damage_events > 0)
    )


def merge_search_logs(
    accumulated: Sequence[Mapping[str, Any]],
    step_logs: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Merge Search logs when a perspective switch repeats prior events."""
    existing = tuple(accumulated)
    incoming = tuple(step_logs)
    existing_keys = effect_log_signature(existing)
    incoming_keys = effect_log_signature(incoming)
    max_overlap = min(len(existing_keys), len(incoming_keys))
    for overlap in range(max_overlap, 0, -1):
        if existing_keys[-overlap:] == incoming_keys[:overlap]:
            return (*existing, *incoming[overlap:])
    return (*existing, *incoming)


def effect_log_signature(
    logs: Sequence[Mapping[str, Any]],
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Return a privacy-safe public signature for an effect log sequence."""
    return tuple(_canonical_effect_log(log) for log in logs)


def _mapping_logs(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(log for log in value if isinstance(log, Mapping))


def _canonical_effect_log(log: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    log_type = int(log.get("type", -1))
    if log_type in {int(LogType.DRAW), int(LogType.DRAW_REVERSE)}:
        return (("type", int(LogType.DRAW)), ("playerIndex", _log_int(log, "playerIndex")))
    if log_type in {int(LogType.MOVE_CARD), int(LogType.MOVE_CARD_REVERSE)}:
        return (
            ("type", int(LogType.MOVE_CARD)),
            ("playerIndex", _log_int(log, "playerIndex")),
            ("fromArea", _log_int(log, "fromArea")),
            ("toArea", _log_int(log, "toArea")),
        )
    fields = (
        "playerIndex",
        "attackId",
        "cardId",
        "serial",
        "cardIdActive",
        "serialActive",
        "cardIdBench",
        "serialBench",
        "cardIdTarget",
        "serialTarget",
        "cardIdBefore",
        "serialBefore",
        "cardIdAfter",
        "serialAfter",
        "value",
        "putDamageCounter",
        "isRecover",
        "head",
        "result",
        "reason",
    )
    return (("type", log_type),) + tuple(
        (field, _log_int(log, field)) for field in fields if field in log
    )


def _log_int(log: Mapping[str, Any], field: str) -> int:
    value = log.get(field, -1)
    return int(value) if value is not None else -1


def _state_summary(state: Mapping[str, Any]) -> Mapping[str, Any]:
    players = cast(Sequence[Any], state.get("players", ()))
    player_summaries: list[Mapping[str, Any]] = []
    for player in players[:2]:
        player_map = cast(Mapping[str, Any], player)
        player_summaries.append(
            {
                "prizes": len(cast(Sequence[Any], player_map.get("prize", ()))),
                "active": _pokemon_summary(player_map.get("active", ())),
                "bench": _pokemon_summary(player_map.get("bench", ())),
            }
        )
    return {
        "your_index": int(state.get("yourIndex", -1)),
        "result": int(state.get("result", -1)),
        "players": player_summaries,
    }


def _pokemon_summary(raw_pokemon: Any) -> list[tuple[int, int, int, int, int]]:
    output: list[tuple[int, int, int, int, int]] = []
    for value in cast(Sequence[Any], raw_pokemon or ()):
        if not isinstance(value, Mapping):
            continue
        output.append(
            (
                int(value.get("id", 0)),
                int(value.get("serial", 0)),
                int(value.get("hp", 0)),
                int(value.get("maxHp", 0)),
                len(cast(Sequence[Any], value.get("energyCards", ()))),
            )
        )
    return output


__all__ = [
    "NativeMacroValidationCase",
    "effect_log_signature",
    "merge_search_logs",
    "search_api_macro_parity",
    "target_attack_fact_matches",
    "target_attack_logs_match",
]
