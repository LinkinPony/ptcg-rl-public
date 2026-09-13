"""Versioned deterministic opponent for the Majkel Lopunny exact deck."""

from ptcg_rl.opponents.lopunny_dudunsparce.agent import (
    LopunnyDudunsparceAgent,
    build_lopunny_dudunsparce_agent,
)
from ptcg_rl.opponents.lopunny_dudunsparce.cards import (
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
    "LopunnyDudunsparceAgent",
    "OPPONENT_NAME",
    "SCRIPT_NAME",
    "build_lopunny_dudunsparce_agent",
]
