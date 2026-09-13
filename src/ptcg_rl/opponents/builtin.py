"""Built-in opponent implementations."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ptcg_rl.actions.selection import (
    forced_action,
    is_legal_action,
    normalize_action_order,
    random_legal_action,
)
from ptcg_rl.context import ContextBeliefTracker, OpponentBeliefFeatureConfig
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.engine.constants import OptionType
from ptcg_rl.engine.protocols import ObservationInput
from ptcg_rl.opponents.spec import BattleAgent, OpponentSpec
from ptcg_rl.training.arena_utils import field_value, int_field

_MIXED_HEURISTIC_PROBABILITY = {
    "mixed25": 0.25,
    "mixed50": 0.50,
    "mixed75": 0.75,
}


@dataclass
class RandomOpponent:
    """Legal random baseline with forced-prompt shortcuts."""

    name: str
    rng: random.Random

    def act(self, observation: ObservationInput) -> Sequence[int]:
        """Return a legal random action for the current select prompt."""
        select = field_value(observation, "select")
        action = forced_action(select)
        if action is not None:
            return action
        return random_legal_action(select, rng=self.rng)

    def reset(self) -> None:
        """Reset per-game state."""


@dataclass
class EndOpponent:
    """Baseline that ends the turn whenever the engine exposes that option."""

    name: str

    def act(self, observation: ObservationInput) -> Sequence[int]:
        """Return an END action when legal, otherwise the shortest legal action."""
        select = field_value(observation, "select")
        action = _end_action(select)
        if action is not None:
            return action
        forced = forced_action(select)
        if forced is not None:
            return forced
        return _shortest_legal_action(select)

    def reset(self) -> None:
        """Reset per-game state."""


@dataclass
class MixedOpponent:
    """Random/heuristic strength interpolation baseline."""

    name: str
    rng: random.Random
    heuristic_probability: float
    heuristic_agent: BattleAgent

    def act(self, observation: ObservationInput) -> Sequence[int]:
        """Choose a heuristic action with probability p, otherwise random."""
        select = field_value(observation, "select")
        action = forced_action(select)
        if action is not None:
            return action
        if self.rng.random() < self.heuristic_probability:
            return self.heuristic_agent.act(observation)
        return random_legal_action(select, rng=self.rng)

    def reset(self) -> None:
        """Reset the wrapped heuristic agent before a fresh battle."""
        self.heuristic_agent.reset()


class CheckpointPolicyOpponent:
    """Greedy frozen-checkpoint policy opponent with context tracking."""

    def __init__(
        self,
        *,
        name: str,
        checkpoint_path: Path,
        device: str = "cpu",
        belief: OpponentBeliefFeatureConfig | None = None,
    ) -> None:
        """Load a policy/value checkpoint for inference."""
        from ptcg_rl.agent.runtime import CheckpointPolicy

        self.name = name
        self._policy = CheckpointPolicy(checkpoint_path, device=device)
        self._tracker = ContextBeliefTracker(belief=belief)

    def act(self, observation: ObservationInput) -> Sequence[int]:
        """Greedily decode a legal index sequence from the policy head."""
        select = field_value(observation, "select")
        action = forced_action(select)
        context_observation = self._tracker.observation_with_context(observation)
        if action is not None:
            return action
        return self._policy.select_action(context_observation)

    def reset(self) -> None:
        """Reset per-game state."""
        self._tracker.begin_game()

    def begin_game(
        self,
        *,
        player_index: int | None = None,
        own_deck: Sequence[int] | None = None,
    ) -> None:
        """Reset per-game context with seat and deck metadata."""
        self._tracker.begin_game(player_index=player_index, own_deck=own_deck)
        if own_deck is not None:
            self._policy.bind_own_deck(own_deck)


def build_builtin_opponent(spec: OpponentSpec, *, seed: int) -> BattleAgent:
    """Build a built-in or frozen-checkpoint opponent."""
    if spec.source == "policy":
        if spec.checkpoint_path is None:
            raise ValueError(f"policy opponent requires checkpoint_path: {spec.name}")
        belief = None
        if spec.belief_summary_path is not None:
            belief = OpponentBeliefFeatureConfig(
                deck_signature_summary_path=records.repo_path(spec.belief_summary_path),
            )
        return CheckpointPolicyOpponent(
            name=spec.name,
            checkpoint_path=records.repo_path(spec.checkpoint_path),
            device=spec.device,
            belief=belief,
        )
    if spec.name == "random":
        return RandomOpponent(name=spec.name, rng=random.Random(seed))
    if spec.name == "end":
        return EndOpponent(name=spec.name)
    if spec.name in _MIXED_HEURISTIC_PROBABILITY:
        from ptcg_rl.opponents.third_party import build_heuristic_agent

        return MixedOpponent(
            name=spec.name,
            rng=random.Random(seed),
            heuristic_probability=_MIXED_HEURISTIC_PROBABILITY[spec.name],
            heuristic_agent=build_heuristic_agent("heuristic"),
        )
    raise KeyError(f"unknown built-in opponent: {spec.name}")


def _end_action(select: Any) -> tuple[int, ...] | None:
    for index, option in enumerate(_options(select)):
        if int_field(option, "type", -1) == int(OptionType.END):
            action = (index,)
            if is_legal_action(select, action):
                return action
    return None


def _shortest_legal_action(select: Any) -> tuple[int, ...]:
    option_count = len(_options(select))
    if option_count <= 0:
        return ()
    count = min(option_count, max(0, int_field(select, "minCount", 0)))
    if count <= 0:
        return ()
    return normalize_action_order(select, tuple(range(count)))


def _options(select: Any) -> Sequence[Any]:
    value = field_value(select, "option", ())
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
