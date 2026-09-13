"""Micro-benchmark for Search API single-step forward-model calls.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/forward_model_benchmark.py
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.belief.sampling import BeliefSampler, BeliefSamplerConfig
from ptcg_rl.engine.forward_model import extract_dynamic_effect_features
from ptcg_rl.engine.protocols import ObservationInput, ObservationLike
from ptcg_rl.engine.runtime import to_engine_observation
from ptcg_rl.engine.session import BattleSession, HiddenInformation
from tools.engine_parity_probe import (
    SAMPLE_DECK,
    _deterministic_battle_action,
    _raise_deck_error_if_any,
)


class ForwardModelBenchmarkConfig(BaseModel):
    """Hydra-backed benchmark config."""

    model_config = ConfigDict(extra="forbid")

    iterations: int = 20
    max_battle_steps: int = 80
    max_candidate_actions: int = 32
    manual_coin: bool = False
    deck0: tuple[int, ...] = SAMPLE_DECK
    deck1: tuple[int, ...] = SAMPLE_DECK
    belief: BeliefSamplerConfig = BeliefSamplerConfig(mode="placeholder")
    belief_seed: int = 0

    @field_validator("iterations", "max_battle_steps", "max_candidate_actions")
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


def run_benchmark(config: ForwardModelBenchmarkConfig) -> dict[str, Any]:
    """Benchmark dynamic effect feature extraction at one representative state."""
    sampler = BeliefSampler(config=config.belief)
    rng = random.Random(config.belief_seed)
    decks = (config.deck0, config.deck1)
    with BattleSession(config.deck0, config.deck1) as battle:
        _raise_deck_error_if_any(battle.start_data)
        observation = to_engine_observation(battle.observation_dict)
        for _ in range(config.max_battle_steps):
            if observation.current is None or observation.current.result >= 0:
                break
            hidden = _sample_hidden(observation, sampler=sampler, decks=decks, rng=rng)
            rows = extract_dynamic_effect_features(
                observation,
                hidden,
                max_actions=config.max_candidate_actions,
                manual_coin=config.manual_coin,
            )
            if rows:
                elapsed = _time_feature_extraction(config, observation, hidden)
                report = {
                    "iterations": config.iterations,
                    "candidate_rows": len(rows),
                    "total_seconds": elapsed,
                    "seconds_per_iteration": elapsed / config.iterations,
                    "seconds_per_candidate": elapsed / (config.iterations * len(rows)),
                    "manual_coin": config.manual_coin,
                    "belief_mode": sampler.config.mode,
                }
                print(json.dumps(report, indent=2, sort_keys=True))
                return report
            battle.select(_deterministic_battle_action(observation))
            observation = to_engine_observation(battle.observation_dict)
    raise RuntimeError("could not find a state with attack/ability feature candidates")


def _time_feature_extraction(
    config: ForwardModelBenchmarkConfig,
    observation: ObservationInput,
    hidden: HiddenInformation,
) -> float:
    start = time.perf_counter()
    for _ in range(config.iterations):
        extract_dynamic_effect_features(
            observation,
            hidden,
            max_actions=config.max_candidate_actions,
            manual_coin=config.manual_coin,
        )
    return time.perf_counter() - start


def _sample_hidden(
    observation: ObservationLike,
    *,
    sampler: BeliefSampler,
    decks: tuple[Sequence[int], Sequence[int]],
    rng: random.Random,
) -> HiddenInformation:
    if observation.current is None:
        raise ValueError("observation must have current state")
    your_index = int(observation.current.yourIndex)
    if your_index not in (0, 1):
        raise ValueError(f"invalid yourIndex: {your_index}")
    return sampler.sample(
        observation,
        your_deck=decks[your_index],
        rng=rng,
    ).hidden


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="engine/forward_model_benchmark",
)
def main(hydra_config: DictConfig) -> None:
    """Hydra entry point."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = ForwardModelBenchmarkConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    run_benchmark(config)


if __name__ == "__main__":
    main()
