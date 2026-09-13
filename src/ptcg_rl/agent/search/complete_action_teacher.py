"""Paired-world complete-action targets for online RL supervision."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from ptcg_rl.actions.selection import is_legal_action, normalize_action_order
from ptcg_rl.agent.search.complete_action_scoring import score_complete_action_plans
from ptcg_rl.agent.search.complete_action_types import (
    ActorPerspectiveLeafValues,
    CompleteActionTeacherConfig,
    CompleteActionTeacherTarget,
    ContinuationLeafScore,
    RootActionTeacherScore,
    WorldRootPlan,
    WorldSessionFactory,
)
from ptcg_rl.agent.search.config import EngineTacticalScoreConfig
from ptcg_rl.agent.search.context import (
    SimulatedContext,
    actor_visible_prompt_fingerprint,
)
from ptcg_rl.agent.search.continuation_planner import (
    ContinuationPlannerLimits,
    ContinuationProposalPolicy,
    plan_complete_continuations,
)
from ptcg_rl.agent.search.continuation_types import (
    ContinuationValueDecks,
    ContinuationValueRequest,
)
from ptcg_rl.agent.search.prompt_actions import (
    build_prompt_action_candidates,
    describe_prompt_action_space,
)
from ptcg_rl.engine.session import HiddenInformation, SearchSession

Clock = Callable[[], float]


def produce_complete_action_teacher_target(
    root_observation: Any,
    *,
    world_count: int,
    open_session: WorldSessionFactory,
    behavior_action: Sequence[int],
    context: SimulatedContext,
    root_player_index: int,
    policy: ContinuationProposalPolicy,
    leaf_values: ActorPerspectiveLeafValues,
    world_value_decks: Sequence[ContinuationValueDecks],
    deadline: float,
    planner_limits: ContinuationPlannerLimits | None = None,
    config: CompleteActionTeacherConfig | None = None,
    tactical_config: EngineTacticalScoreConfig | None = None,
    clock: Clock = time.perf_counter,
) -> CompleteActionTeacherTarget:
    """Compare actions using one process-global Search session at a time.

    The factory is entered and closed once per world. Detached continuations are
    value-scored only after every opened Search lifecycle has ended. Beam search
    remains usable with reduced structural coverage; incomplete lifecycles,
    retained branches, paired grids, chance evidence, or values are invalid.
    """
    if world_count <= 0:
        raise ValueError("world_count must be positive")
    if len(world_value_decks) != world_count:
        raise ValueError("world_value_decks must align with world_count")
    active_config = config or CompleteActionTeacherConfig()
    active_limits = planner_limits or ContinuationPlannerLimits()
    root_select = _field(root_observation, "select")
    behavior = normalize_action_order(
        root_select,
        behavior_action,
    )
    if not is_legal_action(root_select, behavior):
        raise ValueError("behavior_action must be legal in the root observation")
    root_prompt_fingerprint = actor_visible_prompt_fingerprint(
        root_observation,
        actor_player_index=root_player_index,
    )

    public_root = context.enrich(root_observation, update=False)
    root_space = describe_prompt_action_space(root_select)
    ranked = (
        ()
        if root_space.legal_action_count
        <= active_config.root_exhaustive_action_cap
        else _rank_root_actions(
            policy,
            public_root,
            top_k=active_config.root_beam_width,
        )
    )
    root_candidates = build_prompt_action_candidates(
        root_select,
        greedy_action=behavior,
        ranked_actions=ranked,
        exhaustive_action_cap=active_config.root_exhaustive_action_cap,
        beam_width=active_config.root_beam_width,
    )

    plans: list[WorldRootPlan] = []
    nodes_remaining = active_config.total_node_cap
    world_error: str | None = None
    for world_index in range(world_count):
        if nodes_remaining <= 0 or clock() >= deadline:
            break
        try:
            with open_session(world_index) as session:
                if actor_visible_prompt_fingerprint(
                    session.root.observation,
                    actor_player_index=root_player_index,
                ) != root_prompt_fingerprint:
                    raise ValueError(
                        "paired world root prompt fingerprint differs"
                    )
                session_select = session.root.observation.select
                if any(
                    not is_legal_action(session_select, action)
                    for action in root_candidates.actions
                ):
                    raise ValueError("paired world root action grid differs")
                for root_action in _world_action_order(
                    root_candidates.actions,
                    world_index,
                ):
                    if nodes_remaining <= 0 or clock() >= deadline:
                        break
                    per_plan_limits = replace(
                        active_limits,
                        node_cap=min(active_limits.node_cap, nodes_remaining),
                    )
                    plan = plan_complete_continuations(
                        session,
                        root_action=root_action,
                        context=context,
                        root_player_index=root_player_index,
                        policy=policy,
                        value_decks=world_value_decks[world_index],
                        deadline=deadline,
                        limits=per_plan_limits,
                        clock=clock,
                    )
                    nodes_remaining -= plan.nodes_expanded
                    plans.append(
                        WorldRootPlan(
                            world_index=world_index,
                            root_action=root_action,
                            plan=plan,
                        )
                    )
        except Exception as exc:
            world_error = f"world_session_error:{type(exc).__name__}: {exc}"
            break

    evidence = score_complete_action_plans(
        plans,
        root_candidates.actions,
        worlds=world_count,
        root_player_index=root_player_index,
        leaf_values=leaf_values,
        deadline=deadline,
        config=active_config,
        tactical_config=tactical_config,
        clock=clock,
    )
    provisional, margin = _best_action(
        evidence.action_scores,
        fallback=behavior,
        tie_preference=behavior,
    )

    expected_plans = world_count * len(root_candidates.actions)
    completed_plans = sum(plan.plan.valid for plan in plans)
    paired_coverage = (
        float(completed_plans) / float(expected_plans) if expected_plans else 0.0
    )
    root_coverage = (
        float(len(root_candidates.actions))
        / float(max(1, root_candidates.legal_action_count))
    )
    plan_coverage = _geometric_mean(
        tuple(item.plan.search_coverage for item in plans)
    )
    structural_coverage = paired_coverage * root_coverage * plan_coverage
    coverage = structural_coverage * evidence.leaf_score_coverage
    valid, reason = _target_validity(
        plans=plans,
        expected_plans=expected_plans,
        action_scores=evidence.action_scores,
        expected_actions=len(root_candidates.actions),
        leaf_score_coverage=evidence.leaf_score_coverage,
        value_error=evidence.value_error,
        backup_error=evidence.backup_error,
        world_error=world_error,
        deadline_expired=clock() >= deadline,
    )
    target = provisional if valid else behavior
    confidence = _margin_confidence(
        margin,
        coverage=coverage,
        action_count=len(root_candidates.actions),
        temperature=active_config.confidence_temperature,
    ) if valid else 0.0
    return CompleteActionTeacherTarget(
        behavior_action=behavior,
        target_action=target,
        provisional_action=provisional,
        valid=valid,
        reason=(
            "complete_exact"
            if valid
            and root_candidates.exhaustive
            and all(item.plan.exact for item in plans)
            else reason
        ),
        confidence=confidence,
        coverage=coverage,
        score_margin=margin,
        root_candidates=root_candidates,
        action_scores=evidence.action_scores,
        leaf_scores=evidence.leaf_scores,
        plans=tuple(plans),
        worlds=world_count,
        nodes_expanded=sum(item.plan.nodes_expanded for item in plans),
    )


def _best_action(
    scores: Sequence[RootActionTeacherScore],
    *,
    fallback: tuple[int, ...],
    tie_preference: tuple[int, ...],
) -> tuple[tuple[int, ...], float]:
    if not scores:
        return fallback, 0.0
    ordered = sorted(
        scores,
        key=lambda item: (
            item.robust_score,
            item.root_action == tie_preference,
            item.root_action,
        ),
        reverse=True,
    )
    margin = (
        ordered[0].robust_score - ordered[1].robust_score
        if len(ordered) > 1
        else 2.0
    )
    return ordered[0].root_action, max(0.0, float(margin))


def _target_validity(
    *,
    plans: Sequence[WorldRootPlan],
    expected_plans: int,
    action_scores: Sequence[RootActionTeacherScore],
    expected_actions: int,
    leaf_score_coverage: float,
    value_error: str | None,
    backup_error: str | None,
    world_error: str | None,
    deadline_expired: bool,
) -> tuple[bool, str]:
    if deadline_expired:
        return False, "deadline_expired"
    if world_error is not None:
        return False, world_error
    if len(plans) != expected_plans:
        return False, "paired_grid_incomplete"
    if any(not item.plan.valid for item in plans):
        return False, "continuation_incomplete"
    if value_error is not None:
        return False, value_error
    if backup_error is not None:
        return False, backup_error
    if leaf_score_coverage < 1.0:
        return False, "leaf_scores_incomplete"
    if len(action_scores) != expected_actions:
        return False, "paired_scores_incomplete"
    if any(not score.strategy_consistent for score in action_scores):
        return False, "strategy_fusion"
    return True, "complete_approximate"


def _margin_confidence(
    margin: float,
    *,
    coverage: float,
    action_count: int,
    temperature: float,
) -> float:
    if action_count <= 1:
        return min(max(coverage, 0.0), 1.0)
    smooth_margin = 1.0 - math.exp(-max(0.0, margin) / temperature)
    return min(max(coverage * smooth_margin, 0.0), 1.0)


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    mean_log = sum(
        math.log(max(min(float(value), 1.0), 1.0e-12))
        for value in values
    ) / len(values)
    return math.exp(mean_log)


def _rank_root_actions(
    policy: ContinuationProposalPolicy,
    observation: Any,
    *,
    top_k: int,
) -> tuple[tuple[int, ...], ...]:
    try:
        return tuple(policy.rank_actions(observation, top_k=top_k))
    except Exception:
        return ()


def _world_action_order(
    actions: Sequence[tuple[int, ...]],
    world_index: int,
) -> tuple[tuple[int, ...], ...]:
    """Rotate root order across sequential worlds to reduce deadline bias."""
    if not actions:
        return ()
    offset = world_index % len(actions)
    return (*actions[offset:], *actions[:offset])


def make_hidden_world_session_factory(
    root_observation: Any,
    hidden_worlds: Sequence[HiddenInformation],
    *,
    manual_coin: bool = True,
) -> WorldSessionFactory:
    """Build a lazy factory; no two Search lifecycles are opened together."""
    worlds = tuple(hidden_worlds)

    def open_session(world_index: int) -> SearchSession:
        return SearchSession.begin(
            root_observation,
            worlds[world_index],
            manual_coin=manual_coin,
        )

    return open_session


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = [
    "CompleteActionTeacherConfig",
    "CompleteActionTeacherTarget",
    "ContinuationLeafScore",
    "RootActionTeacherScore",
    "ActorPerspectiveLeafValues",
    "ContinuationValueDecks",
    "ContinuationValueRequest",
    "WorldRootPlan",
    "make_hidden_world_session_factory",
    "produce_complete_action_teacher_target",
]
