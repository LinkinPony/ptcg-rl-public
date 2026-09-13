"""Offline parity probe for forward-model log parsing.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/engine_parity_probe.py
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.belief.sampling import BeliefSampler, BeliefSamplerConfig
from ptcg_rl.engine.constants import OptionType
from ptcg_rl.engine.effect_types import EffectSummary
from ptcg_rl.engine.forward_model import (
    enumerate_select_actions,
    resolve_action_from_session,
)
from ptcg_rl.engine.protocols import ObservationLike
from ptcg_rl.engine.runtime import to_engine_observation
from ptcg_rl.engine.session import BattleSession, SearchSession

SAMPLE_DECK: tuple[int, ...] = (
    721,
    721,
    722,
    722,
    722,
    722,
    723,
    723,
    723,
    723,
    1092,
    1121,
    1121,
    1145,
    1145,
    1163,
    1163,
    1219,
    1219,
    1219,
    1219,
    1227,
    1227,
    1227,
    1227,
    1262,
    1262,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
)


class EngineParityProbeConfig(BaseModel):
    """Hydra-backed config for the forward-model parity probe."""

    model_config = ConfigDict(extra="forbid")

    max_battle_steps: int = 80
    max_candidate_actions: int = 64
    require_hp_change: bool = True
    deck0: tuple[int, ...] = SAMPLE_DECK
    deck1: tuple[int, ...] = SAMPLE_DECK
    belief: BeliefSamplerConfig = BeliefSamplerConfig(mode="placeholder")
    belief_seed: int = 0

    @field_validator("max_battle_steps", "max_candidate_actions")
    @classmethod
    def valid_positive(cls, value: int) -> int:
        """Reject non-positive limits."""
        if value <= 0:
            raise ValueError("limits must be positive")
        return value

    @field_validator("deck0", "deck1")
    @classmethod
    def valid_deck_length(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """The engine Battle API requires exactly 60 cards."""
        if len(value) != 60:
            raise ValueError("battle decks must contain exactly 60 cards")
        return value


def run_probe(config: EngineParityProbeConfig) -> dict[str, Any]:
    """Run the parity probe and return a JSON-serializable report."""
    sampler = BeliefSampler(config=config.belief)
    rng = random.Random(config.belief_seed)
    decks = (config.deck0, config.deck1)
    with BattleSession(config.deck0, config.deck1) as battle:
        _raise_deck_error_if_any(battle.start_data)
        for battle_step in range(config.max_battle_steps):
            observation = to_engine_observation(battle.observation_dict)
            if observation.current is None or observation.current.result >= 0:
                break
            report = _try_probe_observation(
                observation,
                max_candidate_actions=config.max_candidate_actions,
                require_hp_change=config.require_hp_change,
                sampler=sampler,
                decks=decks,
                rng=rng,
            )
            if report is not None:
                report["battle_step"] = battle_step
                print(json.dumps(report, indent=2, sort_keys=True))
                return report
            battle.select(_deterministic_battle_action(observation))
    raise RuntimeError("could not find a representative attack transition to probe")


def _try_probe_observation(
    observation: ObservationLike,
    *,
    max_candidate_actions: int,
    require_hp_change: bool,
    sampler: BeliefSampler,
    decks: tuple[Sequence[int], Sequence[int]],
    rng: random.Random,
) -> dict[str, Any] | None:
    state = observation.current
    if state is None:
        return None
    your_index = int(state.yourIndex)
    if your_index not in (0, 1):
        return None
    determinization = sampler.sample(
        observation,
        your_deck=decks[your_index],
        rng=rng,
    )
    with SearchSession.begin(observation, determinization.hidden) as session:
        actions = enumerate_select_actions(
            session.root.observation.select,
            max_actions=max_candidate_actions,
        )
        for action in actions:
            option_types = _option_types(session.root.observation, action)
            if int(OptionType.ATTACK) not in option_types:
                continue
            resolution = resolve_action_from_session(session, session.root, action)
            try:
                if require_hp_change and not _has_nonzero_hp_change(resolution.summary):
                    continue
                _assert_hp_parity(resolution.summary)
                return {
                    "action": list(action),
                    "attack_ids": list(resolution.attack_ids),
                    "hp_change_count": len(resolution.summary.hp_changes),
                    "damage_total": sum(
                        change.damage for change in resolution.summary.hp_changes
                    ),
                    "healing_total": sum(
                        change.healing for change in resolution.summary.hp_changes
                    ),
                    "knockout_count": len(resolution.summary.knockouts),
                    "prizes_taken_by_player": list(
                        resolution.summary.prizes_taken_by_player
                    ),
                    "result": (
                        resolution.successor.observation.current.result
                        if resolution.successor.observation.current is not None
                        else None
                    ),
                    "belief_mode": sampler.config.mode,
                    "hidden_source": determinization.source,
                }
            finally:
                session.release(resolution.search_id)
    return None


def _has_nonzero_hp_change(summary: EffectSummary) -> bool:
    return any(change.damage > 0 or change.healing > 0 for change in summary.hp_changes)


def _assert_hp_parity(summary: EffectSummary) -> None:
    """Assert parsed HP changes agree with engine State snapshots."""
    for change in summary.hp_changes:
        if change.before is None or change.after is None:
            continue
        hp_delta = change.before.hp - change.after.hp
        expected_damage = max(hp_delta, 0)
        expected_healing = max(-hp_delta, 0)
        if change.damage != expected_damage or change.healing != expected_healing:
            raise AssertionError(
                "HP_CHANGE parse mismatch: "
                f"target={change.target} raw={change.raw_value} "
                f"parsed=({change.damage}, {change.healing}) "
                f"expected=({expected_damage}, {expected_healing})"
            )


def _deterministic_battle_action(observation: ObservationLike) -> list[int]:
    select = observation.select
    if select is None:
        return []
    if int(select.maxCount) <= 0:
        return []
    for option_type in (
        OptionType.ATTACH,
        OptionType.PLAY,
        OptionType.EVOLVE,
        OptionType.ABILITY,
        OptionType.ATTACK,
        OptionType.END,
    ):
        for index, option in enumerate(select.option):
            if int(option.type) == int(option_type):
                return [index]
    return list(range(min(int(select.maxCount), len(select.option))))


def _option_types(
    observation: ObservationLike,
    action: Sequence[int],
) -> set[int]:
    if observation.select is None:
        return set()
    return {int(observation.select.option[index].type) for index in action}


def _raise_deck_error_if_any(start_data: object) -> None:
    error_player = getattr(start_data, "errorPlayer", -1)
    if error_player < 0:
        return
    raise ValueError(
        f"deck error: player={error_player} type={getattr(start_data, 'errorType', None)}"
    )


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="engine/parity_probe",
)
def main(hydra_config: DictConfig) -> None:
    """Hydra entry point."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = EngineParityProbeConfig.model_validate(cast(dict[str, Any], raw_config))
    run_probe(config)


if __name__ == "__main__":
    main()
