"""Canonical policy tensorization shared by serving and search adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ptcg_rl.actions.encoding import EncodedOption, StateTokenLayout, encode_options
from ptcg_rl.agent.probe import encoded_options_with_probe_features
from ptcg_rl.model.state_encoder import StateTokenFeatures, encode_observation_tokens


@dataclass(frozen=True)
class CanonicalPolicyInput:
    """Numeric state/option features before device-specific collation."""

    state: StateTokenFeatures
    options: tuple[EncodedOption, ...]
    min_count: int
    max_count: int


def build_canonical_policy_input(
    observation: Any,
    *,
    require_options: bool = True,
) -> CanonicalPolicyInput | None:
    """Encode one observation through the sole serving/search feature path."""
    select = _field(observation, "select")
    if select is None and require_options:
        return None
    layout = StateTokenLayout.from_observation(observation)
    state = encode_observation_tokens(observation, layout=layout)
    options = (
        encoded_options_with_probe_features(
            encode_options(select, layout),
            observation,
        )
        if select is not None
        else ()
    )
    if require_options and not options:
        return None
    return CanonicalPolicyInput(
        state=state,
        options=options,
        min_count=_int_field(select, "minCount", 0),
        max_count=_int_field(select, "maxCount", len(options)),
    )


def canonical_inputs_bitwise_equal(
    left: CanonicalPolicyInput,
    right: CanonicalPolicyInput,
) -> bool:
    """Compare all network-visible numeric fields exactly."""
    return (
        left.state.without_layout() == right.state.without_layout()
        and left.options == right.options
        and left.min_count == right.min_count
        and left.max_count == right.max_count
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    item = _field(value, name, default)
    return int(item) if item is not None else default
