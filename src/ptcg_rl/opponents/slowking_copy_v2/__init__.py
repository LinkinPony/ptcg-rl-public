"""Second-generation deterministic Slowking Copy opponent."""

from ptcg_rl.opponents.slowking_copy_v2.agent import (
    SlowkingCopyAgent,
    SlowkingCopyV2Agent,
    build_slowking_copy_agent,
)
from ptcg_rl.opponents.slowking_copy_v2.cards import (
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
    "SlowkingCopyV2Agent",
    "build_slowking_copy_agent",
]
