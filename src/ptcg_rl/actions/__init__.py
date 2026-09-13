"""Action-space helpers for Kaggle select prompts."""

from ptcg_rl.actions.encoding import (
    EncodedOption,
    EncodedOptionArrayFeatures,
    EncodedOptionInput,
    StateToken,
    StateTokenKey,
    StateTokenLayout,
    build_state_token_layout,
    encode_option_arrays,
    encode_options,
)
from ptcg_rl.actions.selection import (
    ENGINE_PROVEN_UNORDERED_SET_CONTEXTS,
    forced_action,
    is_forced,
    is_legal_action,
    is_unordered_set_selection,
    normalize_action_order,
    random_legal_action,
)

__all__ = [
    "EncodedOption",
    "EncodedOptionArrayFeatures",
    "EncodedOptionInput",
    "StateToken",
    "StateTokenKey",
    "StateTokenLayout",
    "build_state_token_layout",
    "encode_option_arrays",
    "encode_options",
    "ENGINE_PROVEN_UNORDERED_SET_CONTEXTS",
    "forced_action",
    "is_forced",
    "is_legal_action",
    "is_unordered_set_selection",
    "normalize_action_order",
    "random_legal_action",
]
