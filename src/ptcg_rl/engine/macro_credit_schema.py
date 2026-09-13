"""Wire-stable dimensions for root-only executed-macro credit."""

from __future__ import annotations

from enum import IntEnum

from ptcg_rl.engine.constants import OptionType
from ptcg_rl.engine.factual_schema import FACTUAL_NEXT_CONTEXT_COUNT

MACRO_ENDPOINT_COUNT = 3
MACRO_IDENTITY_BUCKET_COUNT = 32
MACRO_ACTION_COUNT_BUCKETS = 8
MACRO_OPTION_TYPE_COUNT = len(OptionType)
MACRO_CONTINUATION_SUMMARY_SIZE = (
    FACTUAL_NEXT_CONTEXT_COUNT
    + MACRO_OPTION_TYPE_COUNT
    + 1
    + MACRO_IDENTITY_BUCKET_COUNT
    + MACRO_ACTION_COUNT_BUCKETS
    + 4
)


class MacroEndpoint(IntEnum):
    """Wire-stable semantic endpoint for one executed macro."""

    TERMINAL = 0
    TURN_HANDOFF = 1
    SAME_SEAT_MAIN = 2


__all__ = [
    "MACRO_ACTION_COUNT_BUCKETS",
    "MACRO_CONTINUATION_SUMMARY_SIZE",
    "MACRO_ENDPOINT_COUNT",
    "MACRO_IDENTITY_BUCKET_COUNT",
    "MACRO_OPTION_TYPE_COUNT",
    "MacroEndpoint",
]
