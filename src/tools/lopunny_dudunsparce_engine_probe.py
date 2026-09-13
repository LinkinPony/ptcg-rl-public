"""Verify Lopunny/Dudunsparce scripted tactics against the bundled engine.

The probe runs bounded exact-deck mirror games. Attack damage is read from
native ``HP_CHANGE`` logs, while Dudunsparce's ability is checked through a
legal engine transition that removes the selected in-play instance.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.engine.constants import LogType, OptionType, SelectContext
from ptcg_rl.engine.runtime import to_engine_observation
from ptcg_rl.engine.session import BattleSession
from ptcg_rl.opponents.lopunny_dudunsparce import (
    build_lopunny_dudunsparce_agent,
    cards,
    view,
)


class LopunnyDudunsparceEngineProbeConfig(BaseModel):
    """Hydra-backed limits and output path for the engine probe."""

    model_config = ConfigDict(extra="forbid")

    max_games: int = 64
    max_steps_per_game: int = 10_000
    output_path: Path = Path("tmp/lopunny_dudunsparce_v1/engine_probe.json")

    @field_validator("max_games", "max_steps_per_game")
    @classmethod
    def valid_positive(cls, value: int) -> int:
        """Reject non-positive probe limits."""
        if value <= 0:
            raise ValueError("probe limits must be positive")
        return value


def run_probe(config: LopunnyDudunsparceEngineProbeConfig) -> dict[str, Any]:
    """Run bounded mirror games and return engine-derived semantics evidence."""
    attack_counts: Counter[int] = Counter()
    attack_damage_counts: Counter[tuple[int, int]] = Counter()
    diagnostic_counts: Counter[str] = Counter()
    ability_actions = 0
    ability_instances_removed = 0
    illegal_actions = 0
    steps = 0
    completed_games = 0

    for _game_index in range(config.max_games):
        agents = (
            build_lopunny_dudunsparce_agent(),
            build_lopunny_dudunsparce_agent(),
        )
        for player_index, agent in enumerate(agents):
            agent.begin_game(player_index=player_index, own_deck=cards.DECK)

        with BattleSession(cards.DECK, cards.DECK) as battle:
            _raise_deck_error_if_any(battle.start_data)
            observation: Mapping[str, Any] = battle.observation_dict
            for _ in range(config.max_steps_per_game):
                if _result(observation) >= 0:
                    completed_games += 1
                    break
                engine_observation = to_engine_observation(observation)
                state = engine_observation.current
                select = engine_observation.select
                if state is None or select is None:
                    raise AssertionError("active battle omitted current/select state")
                player_index = int(state.yourIndex)
                if player_index not in (0, 1):
                    raise AssertionError(f"invalid acting player: {player_index}")

                action = tuple(
                    int(index) for index in agents[player_index].act(observation)
                )
                if not is_legal_action(select, action):
                    illegal_actions += 1
                    raise AssertionError(
                        "script returned an illegal action: "
                        f"player={player_index}, context={int(select.context)}, "
                        f"action={action}"
                    )

                selected_attack = _selected_attack_id(engine_observation, action)
                ability_serial = _selected_dudunsparce_ability_serial(
                    engine_observation,
                    player_index,
                    action,
                )
                observation = battle.select(action)
                steps += 1

                if selected_attack is not None:
                    damage = _opponent_direct_damage(
                        observation.get("logs", ()),
                        opponent_index=1 - player_index,
                    )
                    attack_counts[selected_attack] += 1
                    attack_damage_counts[(selected_attack, damage)] += 1
                if ability_serial is not None:
                    ability_actions += 1
                    after = to_engine_observation(observation)
                    remaining_serials = {
                        view.serial(pokemon)
                        for pokemon in view.field_pokemon(after, player_index)
                    }
                    if ability_serial in remaining_serials:
                        raise AssertionError(
                            "Dudunsparce ability transition left its source in play: "
                            f"serial={ability_serial}"
                        )
                    ability_instances_removed += 1
            else:
                raise AssertionError(
                    "mirror game exceeded the bounded step limit: "
                    f"max_steps_per_game={config.max_steps_per_game}"
                )

        for agent in agents:
            diagnostic_counts.update(agent.diagnostics())
        fallback_counts = {
            key: count
            for key, count in sorted(diagnostic_counts.items())
            if "FALLBACK" in key and count
        }
        if fallback_counts:
            raise AssertionError(
                f"native/script fallback observed during probe: {fallback_counts}"
            )
        if _required_evidence_present(
            attack_damage_counts,
            ability_instances_removed=ability_instances_removed,
            diagnostic_counts=diagnostic_counts,
        ):
            break

    _assert_required_evidence(
        attack_damage_counts,
        ability_instances_removed=ability_instances_removed,
        diagnostic_counts=diagnostic_counts,
        games_run=completed_games,
    )
    report = {
        "engine_source_of_truth": True,
        "deck_digest": cards.DECK_DIGEST,
        "script_name": cards.SCRIPT_NAME,
        "games_run": completed_games,
        "steps": steps,
        "illegal_actions": illegal_actions,
        "native_fallbacks": 0,
        "dudunsparce_ability_actions": ability_actions,
        "dudunsparce_instances_removed": ability_instances_removed,
        "turn_macro_roots_selected": diagnostic_counts.get("TURN_MACRO_SELECTED", 0),
        "attack_counts": {
            str(attack_id): attack_counts[attack_id]
            for attack_id in sorted(attack_counts)
        },
        "attack_damage_counts": {
            f"{attack_id}:{damage}": count
            for (attack_id, damage), count in sorted(attack_damage_counts.items())
        },
        "verified": {
            "gale_thrust_230": True,
            "spiky_hopper_160": True,
            "dudunsparce_ability_legal_and_removed": True,
            "same_turn_native_macro_root_selected": True,
        },
        "limits": {
            "max_games": config.max_games,
            "max_steps_per_game": config.max_steps_per_game,
        },
    }
    output_path = records.repo_path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _selected_attack_id(observation: Any, action: Sequence[int]) -> int | None:
    """Return the selected MAIN attack identity, if any."""
    select = observation.select
    if (
        int(select.context) != int(SelectContext.MAIN)
        or len(action) != 1
        or not 0 <= action[0] < len(select.option)
    ):
        return None
    option = select.option[action[0]]
    if int(option.type) != int(OptionType.ATTACK):
        return None
    return view.integer(getattr(option, "attackId", None))


def _selected_dudunsparce_ability_serial(
    observation: Any,
    player_index: int,
    action: Sequence[int],
) -> int | None:
    """Return the source serial for a selected Dudunsparce ability."""
    select = observation.select
    if (
        int(select.context) != int(SelectContext.MAIN)
        or len(action) != 1
        or not 0 <= action[0] < len(select.option)
    ):
        return None
    option = select.option[action[0]]
    if int(option.type) != int(OptionType.ABILITY):
        return None
    if view.option_card_id(observation, player_index, option) != cards.DUDUNSPARCE:
        return None
    source = view.option_pokemon(observation, player_index, option)
    serial = view.serial(source)
    if serial is None:
        raise AssertionError("Dudunsparce ability option omitted its source serial")
    return serial


def _opponent_direct_damage(
    logs: Sequence[Mapping[str, Any]],
    *,
    opponent_index: int,
) -> int:
    """Sum non-counter damage to the opposing side from native logs."""
    return sum(
        max(0, -int(log.get("value", 0)))
        for log in logs
        if int(log.get("type", -1)) == int(LogType.HP_CHANGE)
        and int(log.get("playerIndex", -1)) == opponent_index
        and not bool(log.get("putDamageCounter", False))
    )


def _required_evidence_present(
    attack_damage_counts: Counter[tuple[int, int]],
    *,
    ability_instances_removed: int,
    diagnostic_counts: Counter[str],
) -> bool:
    """Return whether every required engine path has been observed."""
    return (
        attack_damage_counts[(cards.LOPUNNY_GALE_THRUST, 230)] > 0
        and attack_damage_counts[(cards.LOPUNNY_SPIKY_HOPPER, 160)] > 0
        and ability_instances_removed > 0
        and diagnostic_counts["TURN_MACRO_SELECTED"] > 0
    )


def _assert_required_evidence(
    attack_damage_counts: Counter[tuple[int, int]],
    *,
    ability_instances_removed: int,
    diagnostic_counts: Counter[str],
    games_run: int,
) -> None:
    """Fail clearly when the bounded engine evidence is incomplete."""
    if _required_evidence_present(
        attack_damage_counts,
        ability_instances_removed=ability_instances_removed,
        diagnostic_counts=diagnostic_counts,
    ):
        return
    raise AssertionError(
        "Lopunny/Dudunsparce engine evidence was incomplete after "
        f"{games_run} games; attack_damage_counts="
        f"{dict(sorted(attack_damage_counts.items()))}, "
        f"ability_instances_removed={ability_instances_removed}, "
        "turn_macro_roots_selected="
        f"{diagnostic_counts['TURN_MACRO_SELECTED']}"
    )


def _result(observation: Mapping[str, Any]) -> int:
    current = observation.get("current")
    if isinstance(current, Mapping):
        return int(current.get("result", -1))
    return int(getattr(current, "result", -1))


def _raise_deck_error_if_any(start_data: object) -> None:
    """Raise when the bundled engine rejects the exact deck."""
    error_player = int(getattr(start_data, "errorPlayer", -1))
    if error_player < 0:
        return
    raise ValueError(
        f"deck error: player={error_player} "
        f"type={getattr(start_data, 'errorType', None)}"
    )


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="opponents/lopunny_dudunsparce_engine_probe",
)
def main(hydra_config: DictConfig) -> None:
    """Run the Hydra-configured engine semantics probe."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = LopunnyDudunsparceEngineProbeConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    print(json.dumps(run_probe(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
