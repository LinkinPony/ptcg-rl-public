"""Production actual-endpoint leaves for root-information value learning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import orjson

from ptcg_rl.agent.search.context import public_search_observation
from ptcg_rl.agent.search.root_information import (
    RootActorRelation,
    RootInformationLeaf,
)
from ptcg_rl.agent.search.root_information_context import (
    encode_root_information_producer_context,
)
from ptcg_rl.context import GameContextFeatures
from ptcg_rl.engine.compact_consequence import SemanticEndpoint
from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.engine.factual_schema import (
    FACTUAL_KNOWN_CONTEXT_COUNT,
    FACTUAL_NEXT_CONTEXT_COUNT,
    FACTUAL_NEXT_CONTEXT_OOV,
    FACTUAL_NEXT_CONTEXT_TERMINAL,
    FactualActorRelation,
)


def build_executed_endpoint_leaf(
    observation: Mapping[str, Any],
    *,
    root_player: int,
    context_features: GameContextFeatures,
    actor_relation: FactualActorRelation,
    next_context: int,
    exact_effect: tuple[float, ...],
    belief_summary: tuple[float, ...] = (),
) -> RootInformationLeaf | None:
    """Build the same public leaf contract used by native planner successors.

    Terminal transitions have a factual W/D/L target but no bootstrapped value
    input, so they deliberately return ``None``. Handoff states are projected
    back to the immutable root seat; their active opponent prompt is therefore
    hidden while the endpoint relation remains explicit.
    """
    if root_player not in (0, 1):
        raise ValueError("root_player must be 0 or 1")
    context = int(next_context)
    if context < 0 or context >= FACTUAL_NEXT_CONTEXT_COUNT:
        raise ValueError("actual endpoint has an invalid next context")
    if actor_relation is FactualActorRelation.TERMINAL:
        if context != FACTUAL_NEXT_CONTEXT_TERMINAL:
            raise ValueError("terminal actual endpoint requires terminal context")
        return None
    if context == FACTUAL_NEXT_CONTEXT_TERMINAL:
        raise ValueError("non-terminal actual endpoint cannot use terminal context")
    if context != _observation_next_context(observation):
        raise ValueError("actual endpoint context differs from its observation")
    if actor_relation is FactualActorRelation.SAME_SEAT:
        if context != int(SelectContext.MAIN):
            return None
        relation = RootActorRelation.SAME_SEAT
        endpoint = SemanticEndpoint.SAME_SEAT_MAIN
    elif actor_relation is FactualActorRelation.OTHER_SEAT:
        relation = RootActorRelation.OTHER_SEAT
        endpoint = SemanticEndpoint.TURN_HANDOFF
    else:
        raise ValueError("actual endpoint has an unsupported actor relation")
    projected = public_search_observation(
        observation,
        perspective_player_index=root_player,
    )
    root_visible = {
        "select": projected["select"],
        "logs": projected["logs"],
        "current": projected["current"],
    }
    return RootInformationLeaf(
        root_observable_state=orjson.dumps(root_visible),
        producer_context=encode_root_information_producer_context(
            context_features,
            root_player=root_player,
        ),
        belief_summary=belief_summary,
        exact_effect=exact_effect,
        actor_relation=relation,
        endpoint=endpoint,
    )


def _observation_next_context(observation: Mapping[str, Any]) -> int:
    select = observation.get("select")
    if not isinstance(select, Mapping):
        raise ValueError("non-terminal actual endpoint has no select prompt")
    raw_context = select.get("context", -1)
    context = -1 if raw_context is None else int(raw_context)
    return (
        context
        if 0 <= context < FACTUAL_KNOWN_CONTEXT_COUNT
        else FACTUAL_NEXT_CONTEXT_OOV
    )


__all__ = ["build_executed_endpoint_leaf"]
