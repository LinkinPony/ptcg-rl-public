"""Engine-backed branching planner for complete strategic continuations."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from ptcg_rl.actions.selection import (
    forced_action,
    is_legal_action,
    normalize_action_order,
)
from ptcg_rl.agent.search.context import SimulatedContext
from ptcg_rl.agent.search.continuation_types import (
    CompleteContinuation,
    CompleteContinuationPlan,
    ContinuationPlannerLimits,
    ContinuationProposalPolicy,
    ContinuationValueDecks,
    ContinuationValueRequest,
    PromptExpansion,
)
from ptcg_rl.agent.search.macro import MacroEndpoint, semantic_macro_endpoint
from ptcg_rl.agent.search.prompt_actions import (
    PromptActionCandidates,
    build_prompt_action_candidates,
    describe_prompt_action_space,
)
from ptcg_rl.engine.effect_types import EffectSummary
from ptcg_rl.engine.forward_model import resolve_action_from_session
from ptcg_rl.engine.session import SearchSession

Clock = Callable[[], float]


def plan_complete_continuations(
    session: SearchSession,
    *,
    root_action: Sequence[int],
    context: SimulatedContext,
    root_player_index: int,
    policy: ContinuationProposalPolicy,
    value_decks: ContinuationValueDecks,
    deadline: float,
    limits: ContinuationPlannerLimits | None = None,
    clock: Clock = time.perf_counter,
) -> CompleteContinuationPlan:
    """Branch strategic prompts until MAIN, turn handoff, or terminal.

    Search API children are retained only while their descendants are expanded
    and are always released. Small prompt spaces are exhaustive. Larger spaces
    use a policy-led local beam, are explicitly marked non-exhaustive, and can
    therefore be inspected but not silently consumed as exact supervision.
    """
    if root_player_index not in (0, 1):
        raise ValueError("root_player_index must be 0 or 1")
    active_limits = limits or ContinuationPlannerLimits()
    normalized_root = normalize_action_order(
        session.root.observation.select,
        root_action,
    )
    if not is_legal_action(session.root.observation.select, normalized_root):
        raise ValueError("root_action is not legal for the engine prompt")
    planner = _ContinuationPlanner(
        session=session,
        root_action=normalized_root,
        root_player_index=int(root_player_index),
        policy=policy,
        value_decks=value_decks,
        deadline=float(deadline),
        limits=active_limits,
        clock=clock,
    )
    planner.run(context)
    return planner.result()


class _ContinuationPlanner:
    def __init__(
        self,
        *,
        session: SearchSession,
        root_action: tuple[int, ...],
        root_player_index: int,
        policy: ContinuationProposalPolicy,
        value_decks: ContinuationValueDecks,
        deadline: float,
        limits: ContinuationPlannerLimits,
        clock: Clock,
    ) -> None:
        self._session = session
        self._root_action = root_action
        self._root_player_index = root_player_index
        self._policy = policy
        self._value_decks = value_decks
        self._deadline = deadline
        self._limits = limits
        self._clock = clock
        self._leaves: list[CompleteContinuation] = []
        self._prompt_expansions: list[PromptExpansion] = []
        self._nodes_expanded = 0
        self._proposal_errors = 0
        self._halt_reason: str | None = None

    def run(self, context: SimulatedContext) -> None:
        """Advance the requested root action and recursively release children."""
        root_observation = context.enrich(
            self._session.root.observation,
            update=False,
        )
        self._step_and_visit(
            self._session.root,
            self._root_action,
            parent_context=context,
            parent_observation=root_observation,
            action_path=(),
            summaries=(),
            forced_steps=0,
            strategic_steps=0,
        )

    def result(self) -> CompleteContinuationPlan:
        """Freeze accumulated evidence after all child states were released."""
        incomplete = [leaf for leaf in self._leaves if not leaf.complete]
        complete_coverage = (
            self._halt_reason is None
            and not incomplete
            and bool(self._leaves)
            and all(leaf.error is None for leaf in self._leaves)
        )
        exhaustive = all(item.exhaustive for item in self._prompt_expansions)
        stop_reason = self._stop_reason(complete_coverage, exhaustive)
        return CompleteContinuationPlan(
            root_action=self._root_action,
            leaves=tuple(self._leaves),
            prompt_expansions=tuple(self._prompt_expansions),
            nodes_expanded=self._nodes_expanded,
            proposal_errors=self._proposal_errors,
            complete_coverage=complete_coverage,
            exhaustive=exhaustive,
            stop_reason=stop_reason,
            state_pool_peak=self._session.peak_live_state_count,
            state_leaks=self._session.live_state_count,
        )

    def _step_and_visit(
        self,
        parent: Any,
        action: tuple[int, ...],
        *,
        parent_context: SimulatedContext,
        parent_observation: Any,
        action_path: tuple[tuple[int, ...], ...],
        summaries: tuple[EffectSummary, ...],
        forced_steps: int,
        strategic_steps: int,
    ) -> None:
        if self._halt_reason is not None:
            return
        if self._clock() >= self._deadline:
            self._halt(
                MacroEndpoint.DEADLINE,
                "soft_deadline",
                parent_observation,
                action_path,
                summaries,
                forced_steps,
                strategic_steps,
            )
            return
        if self._nodes_expanded >= self._limits.node_cap:
            self._halt(
                MacroEndpoint.STEP_CAP,
                "node_cap",
                parent_observation,
                action_path,
                summaries,
                forced_steps,
                strategic_steps,
            )
            return
        if not is_legal_action(parent.observation.select, action):
            self._record_leaf(
                MacroEndpoint.ENGINE_ERROR,
                "illegal_continuation",
                parent_observation,
                action_path,
                summaries,
                forced_steps,
                strategic_steps,
                error="planner proposed an illegal continuation",
            )
            return

        child_id: int | None = None
        try:
            resolution = resolve_action_from_session(self._session, parent, action)
            child_id = resolution.search_id
            self._nodes_expanded += 1
            child_context = parent_context.fork()
            child_observation = child_context.enrich(resolution.successor.observation)
            self._visit_reached_state(
                resolution.successor,
                context=child_context,
                observation=child_observation,
                action_path=(*action_path, action),
                summaries=(*summaries, resolution.summary),
                forced_steps=forced_steps,
                strategic_steps=strategic_steps,
            )
        except Exception as exc:
            self._halt_reason = "engine_error"
            self._record_leaf(
                MacroEndpoint.ENGINE_ERROR,
                "engine_exception",
                parent_observation,
                action_path,
                summaries,
                forced_steps,
                strategic_steps,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if child_id is not None:
                self._session.release(child_id)

    def _visit_reached_state(
        self,
        state: Any,
        *,
        context: SimulatedContext,
        observation: Any,
        action_path: tuple[tuple[int, ...], ...],
        summaries: tuple[EffectSummary, ...],
        forced_steps: int,
        strategic_steps: int,
    ) -> None:
        endpoint = semantic_macro_endpoint(
            state.observation,
            root_player_index=self._root_player_index,
        )
        if endpoint is not None:
            leaf_observation = observation
            value_request: ContinuationValueRequest | None = None
            if endpoint == MacroEndpoint.SAME_SEAT_MAIN:
                leaf_observation = context.enrich_for_actor(
                    state.observation,
                    actor_player_index=self._root_player_index,
                    actor_deck=self._value_decks.root_full_deck,
                )
                value_request = ContinuationValueRequest(
                    observation=leaf_observation,
                    deck=self._value_decks.root_full_deck,
                    perspective_player_index=self._root_player_index,
                    root_value_sign=1,
                )
            elif endpoint == MacroEndpoint.TURN_HANDOFF:
                actor_index = _int_field(
                    _field(state.observation, "current"),
                    "yourIndex",
                    -1,
                )
                leaf_observation = context.enrich_for_actor(
                    state.observation,
                    actor_player_index=actor_index,
                    actor_deck=self._value_decks.sampled_opponent_full_deck,
                )
                value_request = ContinuationValueRequest(
                    observation=leaf_observation,
                    deck=self._value_decks.sampled_opponent_full_deck,
                    perspective_player_index=actor_index,
                    root_value_sign=-1,
                )
            self._record_leaf(
                endpoint,
                endpoint.value,
                leaf_observation,
                action_path,
                summaries,
                forced_steps,
                strategic_steps,
                value_request=value_request,
            )
            return

        select = state.observation.select
        if select is None:
            self._record_leaf(
                MacroEndpoint.ENGINE_ERROR,
                "missing_select",
                observation,
                action_path,
                summaries,
                forced_steps,
                strategic_steps,
                error="nonterminal Search state has no select prompt",
            )
            return
        forced = forced_action(select)
        if forced is not None:
            if forced_steps >= self._limits.forced_step_cap:
                self._record_leaf(
                    MacroEndpoint.STEP_CAP,
                    "forced_step_cap",
                    observation,
                    action_path,
                    summaries,
                    forced_steps,
                    strategic_steps,
                )
                return
            self._step_and_visit(
                state,
                forced,
                parent_context=context,
                parent_observation=observation,
                action_path=action_path,
                summaries=summaries,
                forced_steps=forced_steps + 1,
                strategic_steps=strategic_steps,
            )
            return
        if strategic_steps >= self._limits.strategic_step_cap:
            self._record_leaf(
                MacroEndpoint.STEP_CAP,
                "strategic_step_cap",
                observation,
                action_path,
                summaries,
                forced_steps,
                strategic_steps,
            )
            return

        candidates = self._continuation_candidates(select, observation)
        self._prompt_expansions.append(
            _prompt_expansion(action_path, observation, candidates)
        )
        for candidate in candidates.actions:
            self._step_and_visit(
                state,
                candidate,
                parent_context=context,
                parent_observation=observation,
                action_path=action_path,
                summaries=summaries,
                forced_steps=forced_steps,
                strategic_steps=strategic_steps + 1,
            )
            if self._halt_reason is not None:
                break

    def _continuation_candidates(
        self,
        select: Any,
        observation: Any,
    ) -> PromptActionCandidates:
        space = describe_prompt_action_space(select)
        if space.legal_action_count <= self._limits.exhaustive_action_cap:
            return build_prompt_action_candidates(
                select,
                greedy_action=None,
                exhaustive_action_cap=self._limits.exhaustive_action_cap,
                beam_width=self._limits.beam_width,
            )
        try:
            ranked = tuple(
                tuple(int(index) for index in action)
                for action in self._policy.rank_actions(
                    observation,
                    top_k=self._limits.beam_width,
                )
            )
        except Exception:
            self._proposal_errors += 1
            ranked = ()
        greedy = ranked[0] if ranked else None
        if greedy is not None and not is_legal_action(
            select,
            normalize_action_order(select, greedy),
        ):
            self._proposal_errors += 1
            greedy = None
        return build_prompt_action_candidates(
            select,
            greedy_action=greedy,
            ranked_actions=ranked,
            exhaustive_action_cap=self._limits.exhaustive_action_cap,
            beam_width=self._limits.beam_width,
        )

    def _halt(
        self,
        endpoint: MacroEndpoint,
        detail: str,
        observation: Any,
        action_path: tuple[tuple[int, ...], ...],
        summaries: tuple[EffectSummary, ...],
        forced_steps: int,
        strategic_steps: int,
    ) -> None:
        self._halt_reason = detail
        self._record_leaf(
            endpoint,
            detail,
            observation,
            action_path,
            summaries,
            forced_steps,
            strategic_steps,
        )

    def _record_leaf(
        self,
        endpoint: MacroEndpoint,
        detail: str,
        observation: Any,
        action_path: tuple[tuple[int, ...], ...],
        summaries: tuple[EffectSummary, ...],
        forced_steps: int,
        strategic_steps: int,
        *,
        error: str | None = None,
        value_request: ContinuationValueRequest | None = None,
    ) -> None:
        self._leaves.append(
            CompleteContinuation(
                root_action=self._root_action,
                action_path=action_path,
                endpoint=endpoint,
                leaf_observation=observation,
                value_request=value_request,
                summaries=summaries,
                forced_steps=forced_steps,
                strategic_steps=strategic_steps,
                stop_detail=detail,
                error=error,
            )
        )

    def _stop_reason(self, complete_coverage: bool, exhaustive: bool) -> str:
        if self._halt_reason is not None:
            return self._halt_reason
        if any(leaf.error is not None for leaf in self._leaves):
            return "engine_error"
        if not complete_coverage:
            return "incomplete"
        return "complete_exact" if exhaustive else "complete_beam"


def _prompt_expansion(
    action_path: tuple[tuple[int, ...], ...],
    observation: Any,
    candidates: PromptActionCandidates,
) -> PromptExpansion:
    select = _field(observation, "select")
    return PromptExpansion(
        action_path=action_path,
        prompt_key=_public_prompt_key(observation),
        context=_int_field(select, "context", -1),
        legal_action_count=candidates.legal_action_count,
        candidates=candidates.actions,
        sources=candidates.sources,
        exhaustive=candidates.exhaustive,
        ordered=candidates.ordered,
    )


def _public_prompt_key(observation: Any) -> tuple[Any, ...]:
    """Freeze only the sanitized actor-visible decision information."""
    return (
        _freeze_public_value(_field(observation, "current")),
        _freeze_public_value(_field(observation, "logs", ())),
        _freeze_public_value(_field(observation, "select")),
        _freeze_public_value(_field(observation, "gameContext")),
    )


def _freeze_public_value(value: Any) -> Any:
    """Convert public observation values into stable hashable structures."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_public_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_public_value(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            (field.name, _freeze_public_value(getattr(value, field.name)))
            for field in fields(value)
        )
    return (type(value).__qualname__, str(value))


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    item = _field(value, name, default)
    return int(item) if item is not None else default


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


__all__ = [
    "CompleteContinuation",
    "CompleteContinuationPlan",
    "ContinuationPlannerLimits",
    "ContinuationProposalPolicy",
    "ContinuationValueDecks",
    "ContinuationValueRequest",
    "PromptExpansion",
    "plan_complete_continuations",
]
