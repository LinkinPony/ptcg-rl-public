"""Shared output schema for decision-local factual prediction heads."""

from __future__ import annotations

from enum import IntEnum

FACTUAL_KNOWN_CONTEXT_COUNT = 49
FACTUAL_NEXT_CONTEXT_OOV = FACTUAL_KNOWN_CONTEXT_COUNT
FACTUAL_NEXT_CONTEXT_TERMINAL = FACTUAL_KNOWN_CONTEXT_COUNT + 1
FACTUAL_NEXT_CONTEXT_COUNT = FACTUAL_KNOWN_CONTEXT_COUNT + 2


class FactualActorRelation(IntEnum):
    """Actor at the next non-forced decision, relative to the root actor."""

    SAME_SEAT = 0
    OTHER_SEAT = 1
    TERMINAL = 2


FACTUAL_ACTOR_RELATION_COUNT = len(FactualActorRelation)


__all__ = [
    "FACTUAL_ACTOR_RELATION_COUNT",
    "FACTUAL_KNOWN_CONTEXT_COUNT",
    "FACTUAL_NEXT_CONTEXT_COUNT",
    "FACTUAL_NEXT_CONTEXT_OOV",
    "FACTUAL_NEXT_CONTEXT_TERMINAL",
    "FactualActorRelation",
]
