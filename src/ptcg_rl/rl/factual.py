"""Decision-local factual targets derived from live engine transitions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from ptcg_rl.actions.selection import forced_action
from ptcg_rl.engine.factual_schema import (
    FACTUAL_KNOWN_CONTEXT_COUNT,
    FACTUAL_NEXT_CONTEXT_COUNT,
    FACTUAL_NEXT_CONTEXT_OOV,
    FACTUAL_NEXT_CONTEXT_TERMINAL,
    FactualActorRelation,
)
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.forward_model import dynamic_effect_feature_from_dict_resolution
from ptcg_rl.engine.probe_resolution import ProbeTransition


@dataclass(frozen=True)
class FactualSuccessor:
    """Public decision boundary reached after forced continuations."""

    actor_relation: FactualActorRelation
    next_context: int

    def __post_init__(self) -> None:
        """Canonicalize and validate the factored successor label."""
        try:
            relation = FactualActorRelation(int(self.actor_relation))
        except (TypeError, ValueError) as exc:
            raise ValueError("factual successor has an invalid actor relation") from exc
        context = int(self.next_context)
        if context < 0 or context >= FACTUAL_NEXT_CONTEXT_COUNT:
            raise ValueError("factual successor has an invalid next context")
        if relation is FactualActorRelation.TERMINAL:
            if context != FACTUAL_NEXT_CONTEXT_TERMINAL:
                raise ValueError("terminal factual successor requires terminal context")
        elif context == FACTUAL_NEXT_CONTEXT_TERMINAL:
            raise ValueError(
                "non-terminal factual successor cannot use terminal context"
            )
        object.__setattr__(self, "actor_relation", relation)
        object.__setattr__(self, "next_context", context)


@dataclass(frozen=True)
class FactualTransitionTarget:
    """Privacy-safe consequence target for one executed policy decision."""

    effect_features: tuple[float, ...]
    actor_relation: FactualActorRelation
    next_context: int

    def __post_init__(self) -> None:
        """Canonicalize and validate fixed-width finite target data."""
        features = tuple(float(value) for value in self.effect_features)
        if len(features) != DYNAMIC_EFFECT_FEATURE_SIZE:
            raise ValueError(
                "factual effect target has invalid width: "
                f"{len(features)} != {DYNAMIC_EFFECT_FEATURE_SIZE}"
            )
        if not all(math.isfinite(value) for value in features):
            raise ValueError("factual effect target must contain finite values")
        successor = FactualSuccessor(
            actor_relation=self.actor_relation,
            next_context=self.next_context,
        )
        object.__setattr__(self, "effect_features", features)
        object.__setattr__(self, "actor_relation", successor.actor_relation)
        object.__setattr__(self, "next_context", successor.next_context)


class FactualLaneConfig(BaseModel):
    """Hydra-backed switch for dense executed-transition supervision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False


def factual_successor(
    observation: Mapping[str, Any],
    *,
    root_player_index: int,
) -> FactualSuccessor | None:
    """Return the first terminal or non-forced successor of a policy decision.

    A successor with exactly one useful legal response is still part of the
    root decision's engine transition chain. It therefore returns ``None`` and
    is accumulated until the next genuinely selectable prompt or terminal.
    """
    if root_player_index not in (0, 1):
        raise ValueError("root_player_index must be 0 or 1")
    current = _field(observation, "current")
    if current is None:
        raise ValueError("factual successor observation has no current state")
    if _int_field(current, "result", -1) >= 0:
        return FactualSuccessor(
            actor_relation=FactualActorRelation.TERMINAL,
            next_context=FACTUAL_NEXT_CONTEXT_TERMINAL,
        )
    select = _field(observation, "select")
    if select is None:
        raise ValueError("non-terminal factual successor has no select prompt")
    if forced_action(select) is not None:
        return None
    actor = _int_field(current, "yourIndex", -1)
    if actor not in (0, 1):
        raise ValueError("factual successor observation has an invalid actor seat")
    raw_context = _int_field(select, "context", -1)
    next_context = (
        raw_context
        if 0 <= raw_context < FACTUAL_KNOWN_CONTEXT_COUNT
        else FACTUAL_NEXT_CONTEXT_OOV
    )
    return FactualSuccessor(
        actor_relation=(
            FactualActorRelation.SAME_SEAT
            if actor == root_player_index
            else FactualActorRelation.OTHER_SEAT
        ),
        next_context=next_context,
    )


def build_factual_transition_target(
    *,
    root_action: Sequence[int],
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    transitions: Sequence[ProbeTransition],
    perspective_player: int,
    successor: FactualSuccessor | None = None,
) -> FactualTransitionTarget:
    """Build one target from a policy action through forced continuations."""
    if perspective_player not in (0, 1):
        raise ValueError("perspective_player must be 0 or 1")
    if not transitions:
        raise ValueError("factual target requires at least one engine transition")
    if _field(before_observation, "current") is None:
        raise ValueError("factual root observation has no current state")
    inferred_successor = factual_successor(
        after_observation,
        root_player_index=perspective_player,
    )
    if inferred_successor is None:
        raise ValueError("factual target has not reached a decision-local boundary")
    if successor is not None and successor != inferred_successor:
        raise ValueError("declared factual successor disagrees with the observation")
    row = dynamic_effect_feature_from_dict_resolution(
        select=tuple(int(index) for index in root_action),
        before_observation=before_observation,
        after_observation=after_observation,
        probe_transitions=transitions,
        perspective_player=perspective_player,
    )
    return FactualTransitionTarget(
        effect_features=row.vector,
        actor_relation=inferred_successor.actor_relation,
        next_context=inferred_successor.next_context,
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    item = _field(value, name, default)
    return int(item) if item is not None else default


__all__ = [
    "FACTUAL_KNOWN_CONTEXT_COUNT",
    "FACTUAL_NEXT_CONTEXT_COUNT",
    "FACTUAL_NEXT_CONTEXT_OOV",
    "FACTUAL_NEXT_CONTEXT_TERMINAL",
    "FactualActorRelation",
    "FactualLaneConfig",
    "FactualSuccessor",
    "FactualTransitionTarget",
    "build_factual_transition_target",
    "factual_successor",
]
