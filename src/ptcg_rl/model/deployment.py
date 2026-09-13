"""Removal of training-only model components from direct-policy artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ptcg_rl.model.network import AgentNetworkConfig

_ACTION_VALUE_PREFIX = "action_value_head."


@dataclass(frozen=True, slots=True)
class DirectActorModelState:
    """A loadable direct-actor state/config pair with no training Q head."""

    state_dict: dict[str, Any]
    model_config: AgentNetworkConfig
    stripped_keys: tuple[str, ...]


def strip_training_only_action_value(
    state_dict: Mapping[str, Any],
    model_config: AgentNetworkConfig,
) -> DirectActorModelState:
    """Strip the persistent Q head while preserving the direct policy exactly."""
    stripped_keys = tuple(
        sorted(str(key) for key in state_dict if str(key).startswith(_ACTION_VALUE_PREFIX))
    )
    if model_config.action_value.enabled and not stripped_keys:
        raise ValueError("enabled action-value config has no checkpoint tensors")
    if not model_config.action_value.enabled and stripped_keys:
        raise ValueError("disabled action-value config unexpectedly has Q tensors")
    runtime_state = {
        str(key): value
        for key, value in state_dict.items()
        if not str(key).startswith(_ACTION_VALUE_PREFIX)
    }
    runtime_config = model_config.model_copy(
        update={
            "action_value": model_config.action_value.model_copy(
                update={"enabled": False}
            )
        }
    )
    return DirectActorModelState(
        state_dict=runtime_state,
        model_config=runtime_config,
        stripped_keys=stripped_keys,
    )


__all__ = ["DirectActorModelState", "strip_training_only_action_value"]
