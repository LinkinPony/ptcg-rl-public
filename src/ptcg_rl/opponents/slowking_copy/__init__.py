"""Slowking Copy Engine scripted opponent."""

from ptcg_rl.opponents.slowking_copy.agent import (
    SlowkingCopyAgent,
    build_slowking_copy_agent,
)
from ptcg_rl.opponents.slowking_copy.cards import (
    DECK,
    DECK_DIGEST,
    IMPLEMENTATION_FILES,
    OPPONENT_NAME,
    SCRIPT_NAME,
)

__all__ = [
    "DECK",
    "DECK_DIGEST",
    "IMPLEMENTATION_FILES",
    "OPPONENT_NAME",
    "SCRIPT_NAME",
    "SlowkingCopyAgent",
    "build_slowking_copy_agent",
]
