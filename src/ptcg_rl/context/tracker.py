"""Per-game context + belief tracking shared by local evaluation agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ptcg_rl.context.belief import (
    OpponentBeliefFeatureConfig,
    OpponentBeliefFeatureProducer,
)
from ptcg_rl.context.game import GameContext, GameContextFeatures


class ContextBeliefTracker:
    """Accumulate one player's game context and optional belief posterior.

    Local evaluation agents (gauntlet, head-to-head, deck ladder) historically
    ran the bare checkpoint policy on raw observations, while both training
    rollouts and the Kaggle runtime feed context features (opponent revealed
    cards, history counters, optional belief tokens). This tracker gives local
    agents the same observation shape as training and serving.
    """

    def __init__(
        self,
        *,
        belief: OpponentBeliefFeatureConfig | None = None,
    ) -> None:
        """Build the context accumulator and optional belief producer."""
        self._context = GameContext()
        self._belief_producer = (
            OpponentBeliefFeatureProducer.from_config(belief)
            if belief is not None
            else None
        )

    def begin_game(
        self,
        *,
        player_index: int | None = None,
        own_deck: Sequence[int] | None = None,
    ) -> None:
        """Reset per-game evidence before a fresh battle."""
        self._context.reset(player_index=player_index, own_deck=own_deck)

    def observation_with_context(self, observation: Any) -> Any:
        """Update evidence and return the observation with embedded context."""
        features = self._context.update(observation)
        if self._belief_producer is not None:
            features = self._belief_producer.augment(observation, features)
        return _embed_context(observation, features)


def _embed_context(observation: Any, features: GameContextFeatures) -> Any:
    context_dict = features.as_observation_dict()
    if isinstance(observation, Mapping):
        copied = dict(observation)
        copied["gameContext"] = context_dict
        return copied
    return {
        "remainingOverageTime": _field(observation, "remainingOverageTime"),
        "current": _field(observation, "current"),
        "logs": _field(observation, "logs", ()),
        "search_begin_input": _field(observation, "search_begin_input"),
        "select": _field(observation, "select"),
        "gameContext": context_dict,
    }


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)
