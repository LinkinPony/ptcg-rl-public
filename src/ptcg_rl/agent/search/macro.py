"""Lifecycle-safe forced closure and same-turn engine macro transitions."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ptcg_rl.actions.selection import forced_action, is_legal_action
from ptcg_rl.agent.probe import (
    RuntimeProbeResult,
    core_option_candidates,
    observation_with_probe_features,
)
from ptcg_rl.agent.search.context import SimulatedContext
from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.engine.effect_types import EffectSummary
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.forward_model import (
    extract_dynamic_effect_features_from_session,
    resolve_action_from_session,
)
from ptcg_rl.engine.session import SearchSession

Clock = Callable[[], float]
ContinuationPolicy = Callable[[Any], Sequence[int]]


class MacroEndpoint(StrEnum):
    """Exhaustive endpoint classes for a P0 same-turn macro."""

    TERMINAL = "terminal"
    SAME_SEAT_MAIN = "same_seat_main"
    TURN_HANDOFF = "turn_handoff"
    DEADLINE = "deadline"
    STEP_CAP = "step_cap"
    ENGINE_ERROR = "engine_error"


@dataclass(frozen=True)
class MacroTransition:
    """One root action advanced to a semantic leaf boundary."""

    root_action: tuple[int, ...]
    endpoint: MacroEndpoint
    leaf_observation: Any | None
    summaries: tuple[EffectSummary, ...]
    steps: int
    forced_steps: int
    continuation_steps: int
    stop_detail: str
    state_pool_peak: int
    state_leaks: int
    error: str | None = None


def advance_same_turn_macro(
    session: SearchSession,
    *,
    root_action: Sequence[int],
    context: SimulatedContext,
    root_player_index: int,
    deadline: float,
    continuation_policy: ContinuationPolicy | None,
    forced_step_cap: int,
    continuation_step_cap: int,
    node_cap: int,
    clock: Clock = time.perf_counter,
) -> MacroTransition:
    """Resolve an action, forced prompts, and optional same-turn continuations."""
    action = tuple(int(index) for index in root_action)
    parent = session.root
    summaries: list[EffectSummary] = []
    steps = 0
    forced_steps = 0
    continuation_steps = 0
    leaf_observation: Any | None = None
    current_child_id: int | None = None
    endpoint = MacroEndpoint.ENGINE_ERROR
    stop_detail = "engine_error"
    error: str | None = None

    try:
        while True:
            if clock() >= deadline:
                endpoint = MacroEndpoint.DEADLINE
                stop_detail = "soft_deadline"
                break
            if steps >= node_cap:
                endpoint = MacroEndpoint.STEP_CAP
                stop_detail = "node_cap"
                break
            parent_select = parent.observation.select
            if not is_legal_action(parent_select, action):
                endpoint = MacroEndpoint.ENGINE_ERROR
                stop_detail = "illegal_continuation"
                break

            resolution = resolve_action_from_session(session, parent, action)
            if current_child_id is not None:
                session.release(current_child_id)
            current_child_id = resolution.search_id
            parent = resolution.successor
            summaries.append(resolution.summary)
            steps += 1
            leaf_observation = context.enrich(parent.observation)

            semantic_endpoint = semantic_macro_endpoint(
                parent.observation,
                root_player_index=root_player_index,
            )
            if semantic_endpoint is not None:
                endpoint = semantic_endpoint
                stop_detail = endpoint.value
                break

            select = parent.observation.select
            forced = forced_action(select)
            if forced is not None:
                if forced_steps >= forced_step_cap:
                    endpoint = MacroEndpoint.STEP_CAP
                    stop_detail = "forced_step_cap"
                    break
                forced_steps += 1
                action = forced
                continue

            if continuation_policy is None:
                endpoint = MacroEndpoint.STEP_CAP
                stop_detail = "non_forced_boundary"
                break
            if continuation_steps >= continuation_step_cap:
                endpoint = MacroEndpoint.STEP_CAP
                stop_detail = "continuation_step_cap"
                break
            continuation_steps += 1
            continuation_observation = _with_child_probe_features(
                session,
                parent,
                leaf_observation,
            )
            action = tuple(
                int(index) for index in continuation_policy(continuation_observation)
            )
    except Exception as exc:  # Search failures become a classified greedy fallback.
        endpoint = MacroEndpoint.ENGINE_ERROR
        stop_detail = "engine_exception"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if current_child_id is not None:
            session.release(current_child_id)

    return MacroTransition(
        root_action=tuple(int(index) for index in root_action),
        endpoint=endpoint,
        leaf_observation=leaf_observation,
        summaries=tuple(summaries),
        steps=steps,
        forced_steps=forced_steps,
        continuation_steps=continuation_steps,
        stop_detail=stop_detail,
        state_pool_peak=session.peak_live_state_count,
        state_leaks=session.live_state_count,
        error=error,
    )


def semantic_macro_endpoint(
    observation: Any,
    *,
    root_player_index: int,
) -> MacroEndpoint | None:
    current = _field(observation, "current")
    result = _int_field(current, "result", -1)
    if result >= 0:
        return MacroEndpoint.TERMINAL
    your_index = _int_field(current, "yourIndex", root_player_index)
    if your_index != root_player_index:
        return MacroEndpoint.TURN_HANDOFF
    select = _field(observation, "select")
    if _int_field(select, "context", -1) == int(SelectContext.MAIN):
        return MacroEndpoint.SAME_SEAT_MAIN
    return None


def _with_child_probe_features(
    session: SearchSession,
    parent: Any,
    observation: Any,
) -> Any:
    """Probe child policy candidates in the current session, never recursively."""
    candidates = core_option_candidates(parent.observation.select)
    if not candidates:
        return observation
    rows = extract_dynamic_effect_features_from_session(
        session,
        parent,
        candidates=candidates,
        release_successors=True,
    )
    options = _sequence(_field(parent.observation.select, "option", ()))
    features = [(0.0,) * DYNAMIC_EFFECT_FEATURE_SIZE for _ in options]
    masks = [False] * len(options)
    vectors: dict[tuple[int, ...], tuple[tuple[float, ...], ...]] = {}
    for row in rows:
        vectors[row.select] = (row.vector,)
        if len(row.select) != 1:
            continue
        option_index = row.select[0]
        if 0 <= option_index < len(features):
            features[option_index] = row.vector
            masks[option_index] = True
    return observation_with_probe_features(
        observation,
        RuntimeProbeResult(
            features=tuple(features),
            masks=tuple(masks),
            world_vectors=vectors,
            worlds_requested=1,
        ),
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    item = _field(value, name, default)
    return int(item) if item is not None else default


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
