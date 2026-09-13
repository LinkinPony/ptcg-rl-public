"""Opponent pool adapters for local evaluation and rollout."""

from ptcg_rl.opponents.spec import (
    BattleAgent,
    OpponentPoolConfig,
    OpponentSpec,
    PolicyOpponentConfig,
    build_opponent,
    opponent_registry,
    opponents_by_tier,
    select_opponents,
)

__all__ = [
    "BattleAgent",
    "OpponentPoolConfig",
    "OpponentSpec",
    "PolicyOpponentConfig",
    "build_opponent",
    "opponent_registry",
    "opponents_by_tier",
    "select_opponents",
]
