"""Built-in agents for local arena evaluation."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.actions.selection import (
    forced_action,
    random_legal_action,
)
from ptcg_rl.context import ContextBeliefTracker, OpponentBeliefFeatureConfig
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.engine.protocols import ObservationInput
from ptcg_rl.training.arena_utils import field_value


class ArenaAgent(Protocol):
    """Agent interface consumed by the local battle runner."""

    name: str

    def act(self, observation: ObservationInput) -> Sequence[int]:
        """Return selected option indices for the current prompt."""


class ArenaAgentConfig(BaseModel):
    """Hydra-backed agent selector for the arena."""

    model_config = ConfigDict(extra="forbid")

    kind: str = "random"
    label: str | None = None
    checkpoint_path: Path | None = None
    device: str = "cpu"

    @field_validator("kind")
    @classmethod
    def valid_kind(cls, value: str) -> str:
        """Reject unknown built-in agent kinds."""
        normalized = value.lower()
        if normalized not in {"random", "policy_greedy"}:
            raise ValueError("kind must be 'random' or 'policy_greedy'")
        return normalized

    @field_validator("device")
    @classmethod
    def valid_device(cls, value: str) -> str:
        """Reject empty device strings."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("device must be non-empty")
        return normalized


@dataclass
class RandomSelectAgent:
    """Legal random baseline with forced-prompt shortcuts."""

    name: str
    rng: random.Random

    def act(self, observation: ObservationInput) -> Sequence[int]:
        """Return a legal random action for the current select prompt."""
        select = field_value(observation, "select")
        action = forced_action(select)
        if action is None:
            action = random_legal_action(select, rng=self.rng)
        return action


class PolicyGreedyAgent:
    """Greedy pointer-policy agent loaded from a training checkpoint.

    The agent tracks per-game context (opponent revealed cards, history
    counters) and optionally belief tokens so local evaluation matches the
    observation shape used by training rollouts and the Kaggle runtime.
    """

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

    def reset(self) -> None:
        """Reset per-game context before a fresh battle."""
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

    def act(self, observation: ObservationInput) -> Sequence[int]:
        """Greedily decode a legal index sequence from the policy head."""
        select = field_value(observation, "select")
        action = forced_action(select)
        context_observation = self._tracker.observation_with_context(observation)
        if action is not None:
            return action
        return self._policy.select_action(context_observation)


def build_arena_agent(config: ArenaAgentConfig, *, seed: int) -> ArenaAgent:
    """Build one built-in arena agent from config."""
    label = config.label or config.kind
    if config.kind == "random":
        return RandomSelectAgent(name=label, rng=random.Random(seed))
    if config.checkpoint_path is None:
        raise ValueError("policy_greedy requires checkpoint_path")
    return PolicyGreedyAgent(
        name=label,
        checkpoint_path=records.repo_path(config.checkpoint_path),
        device=config.device,
    )
