"""Engine/value scoring and information-set backup for complete actions."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ptcg_rl.agent.search.complete_action_types import (
    ActorPerspectiveLeafValues,
    CompleteActionTeacherConfig,
    ContinuationLeafScore,
    RootActionTeacherScore,
    WorldRootPlan,
)
from ptcg_rl.agent.search.config import EngineTacticalScoreConfig
from ptcg_rl.agent.search.continuation_types import (
    CompleteContinuation,
    CompleteContinuationPlan,
)
from ptcg_rl.agent.search.macro import MacroEndpoint
from ptcg_rl.agent.search.scoring import engine_transition_score
from ptcg_rl.engine.constants import SelectContext

Clock = Callable[[], float]
Expired = Callable[[], bool]


@dataclass(frozen=True)
class CompleteActionScoreEvidence:
    """Leaf and paired-root scores plus any structural scoring error."""

    leaf_scores: tuple[ContinuationLeafScore, ...]
    action_scores: tuple[RootActionTeacherScore, ...]
    leaf_score_coverage: float
    value_error: str | None
    backup_error: str | None


def score_complete_action_plans(
    plans: Sequence[WorldRootPlan],
    actions: Sequence[tuple[int, ...]],
    *,
    worlds: int,
    root_player_index: int,
    leaf_values: ActorPerspectiveLeafValues,
    deadline: float,
    config: CompleteActionTeacherConfig,
    tactical_config: EngineTacticalScoreConfig | None,
    clock: Clock,
) -> CompleteActionScoreEvidence:
    """Score all leaves, back up chance nodes, and aggregate paired worlds."""
    leaf_scores, coverage, value_error = _score_leaves(
        plans,
        root_player_index=root_player_index,
        leaf_values=leaf_values,
        deadline=deadline,
        config=config,
        tactical_config=tactical_config,
        clock=clock,
    )
    action_scores, backup_error = _aggregate_root_scores(
        leaf_scores,
        plans,
        actions,
        worlds=worlds,
        risk_std_weight=config.risk_std_weight,
        joint_strategy_cap=config.joint_strategy_cap,
        expired=lambda: clock() >= deadline,
    )
    return CompleteActionScoreEvidence(
        leaf_scores=leaf_scores,
        action_scores=action_scores,
        leaf_score_coverage=coverage,
        value_error=value_error,
        backup_error=backup_error,
    )


def _score_leaves(
    plans: Sequence[WorldRootPlan],
    *,
    root_player_index: int,
    leaf_values: ActorPerspectiveLeafValues,
    deadline: float,
    config: CompleteActionTeacherConfig,
    tactical_config: EngineTacticalScoreConfig | None,
    clock: Clock,
) -> tuple[tuple[ContinuationLeafScore, ...], float, str | None]:
    pending: list[tuple[WorldRootPlan, CompleteContinuation, float]] = []
    scores: list[ContinuationLeafScore] = []
    complete_leaf_count = 0
    for world_plan in plans:
        for leaf in world_plan.plan.complete_leaves:
            complete_leaf_count += 1
            transition = leaf.as_macro_transition(
                state_pool_peak=world_plan.plan.state_pool_peak,
                state_leaks=world_plan.plan.state_leaks,
            )
            engine_score = engine_transition_score(
                transition,
                root_player_index=root_player_index,
                config=tactical_config,
            )
            if engine_score is None or not math.isfinite(engine_score):
                continue
            if leaf.endpoint == MacroEndpoint.TERMINAL:
                scores.append(
                    _leaf_score(
                        world_plan,
                        leaf,
                        engine_score,
                        actor_value=None,
                        config=config,
                    )
                )
            elif leaf.value_request is not None:
                pending.append((world_plan, leaf, engine_score))

    value_error: str | None = None
    for offset in range(0, len(pending), config.leaf_value_batch_size):
        if clock() >= deadline:
            value_error = "value_deadline"
            break
        chunk = pending[offset : offset + config.leaf_value_batch_size]
        requests = tuple(item[1].value_request for item in chunk)
        if any(request is None for request in requests):
            value_error = "missing_value_request"
            break
        required = tuple(request for request in requests if request is not None)
        try:
            values = tuple(float(value) for value in leaf_values(required))
            if len(values) != len(chunk):
                raise ValueError("leaf value callback returned the wrong value count")
            if any(not math.isfinite(value) for value in values):
                raise ValueError("leaf value callback returned a non-finite value")
        except Exception as exc:
            value_error = f"value_error:{type(exc).__name__}: {exc}"
            break
        if clock() >= deadline:
            value_error = "value_deadline"
            break
        for (world_plan, leaf, engine_score), actor_value in zip(
            chunk,
            values,
            strict=True,
        ):
            request = leaf.value_request
            if request is None:
                raise RuntimeError("validated value request disappeared")
            scores.append(
                _leaf_score(
                    world_plan,
                    leaf,
                    engine_score,
                    actor_value=actor_value,
                    root_value_sign=request.root_value_sign,
                    config=config,
                )
            )
    coverage = (
        float(len(scores)) / float(complete_leaf_count)
        if complete_leaf_count
        else 0.0
    )
    return tuple(scores), coverage, value_error


def _leaf_score(
    world_plan: WorldRootPlan,
    leaf: CompleteContinuation,
    engine_score: float,
    *,
    actor_value: float | None,
    root_value_sign: int | None = None,
    config: CompleteActionTeacherConfig,
) -> ContinuationLeafScore:
    leaf_value = (
        None
        if actor_value is None or root_value_sign is None
        else actor_value * root_value_sign
    )
    score = (
        engine_score
        if leaf_value is None
        else leaf_value + config.engine_tiebreak_weight * engine_score
    )
    score = min(max(score, -config.score_clip), config.score_clip)
    return ContinuationLeafScore(
        world_index=world_plan.world_index,
        root_action=world_plan.root_action,
        action_path=leaf.action_path,
        endpoint=leaf.endpoint,
        engine_score=float(engine_score),
        actor_value=actor_value,
        root_value_sign=root_value_sign,
        leaf_value=leaf_value,
        score=float(score),
    )


def _aggregate_root_scores(
    leaf_scores: Sequence[ContinuationLeafScore],
    plans: Sequence[WorldRootPlan],
    actions: Sequence[tuple[int, ...]],
    *,
    worlds: int,
    risk_std_weight: float,
    joint_strategy_cap: int,
    expired: Expired,
) -> tuple[tuple[RootActionTeacherScore, ...], str | None]:
    indexed: dict[tuple[tuple[int, ...], int], list[ContinuationLeafScore]] = {}
    for score in leaf_scores:
        indexed.setdefault((score.root_action, score.world_index), []).append(score)
    plan_index = {
        (item.root_action, item.world_index): item.plan for item in plans
    }

    result: list[RootActionTeacherScore] = []
    for action in actions:
        if expired():
            return tuple(result), "backup_deadline"
        world_strategies: list[tuple[_WorldTreeBackup, ...]] = []
        for world_index in range(worlds):
            if expired():
                return tuple(result), "backup_deadline"
            plan = plan_index.get((action, world_index))
            candidates = indexed.get((action, world_index), ())
            if plan is None or not candidates:
                break
            if _has_uncontrolled_coin(plan):
                return tuple(result), "uncontrolled_coin"
            strategies, error = _enumerate_world_tree_strategies(
                plan,
                candidates,
                strategy_cap=joint_strategy_cap,
                expired=expired,
            )
            if error is not None or not strategies:
                return tuple(result), error
            world_strategies.append(strategies)
        if len(world_strategies) != worlds:
            continue
        paired, error = _select_nonanticipative_strategy(
            world_strategies,
            risk_std_weight=risk_std_weight,
            strategy_cap=joint_strategy_cap,
            expired=expired,
        )
        if error is not None:
            return tuple(result), error
        if paired is None:
            return tuple(result), "nonanticipative_strategy_unavailable"
        world_scores = paired.world_scores
        mean_score = statistics.fmean(world_scores)
        score_std = statistics.pstdev(world_scores)
        result.append(
            RootActionTeacherScore(
                root_action=action,
                world_scores=world_scores,
                continuation_paths=paired.representative_paths,
                mean_score=mean_score,
                score_std=score_std,
                robust_score=mean_score - risk_std_weight * score_std,
                strategy_consistent=True,
            )
        )
    return tuple(result), None


_DecisionKey = tuple[Any, ...]


@dataclass(frozen=True)
class _WorldTreeBackup:
    score: float
    representative_path: tuple[tuple[int, ...], ...]
    decisions: tuple[tuple[_DecisionKey, tuple[int, ...]], ...]


@dataclass(frozen=True)
class _PairedTreeBackup:
    world_scores: tuple[float, ...]
    representative_paths: tuple[tuple[tuple[int, ...], ...], ...]
    decisions: tuple[tuple[_DecisionKey, tuple[int, ...]], ...]


def _enumerate_world_tree_strategies(
    plan: CompleteContinuationPlan,
    leaf_scores: Sequence[ContinuationLeafScore],
    *,
    strategy_cap: int,
    expired: Expired,
) -> tuple[tuple[_WorldTreeBackup, ...], str | None]:
    leaf_index = {item.action_path: item for item in leaf_scores}
    if len(leaf_index) != len(leaf_scores):
        return (), "duplicate_leaf_path"
    expansion_index = {
        item.action_path: item for item in plan.prompt_expansions
    }
    if len(expansion_index) != len(plan.prompt_expansions):
        return (), "duplicate_prompt_path"
    children: dict[
        tuple[tuple[int, ...], ...],
        set[tuple[int, ...]],
    ] = {}
    for path in leaf_index:
        for offset, action in enumerate(path):
            children.setdefault(path[:offset], set()).add(action)

    memo: dict[
        tuple[tuple[int, ...], ...],
        tuple[_WorldTreeBackup, ...],
    ] = {}

    def visit(
        prefix: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[_WorldTreeBackup, ...], str | None]:
        if expired():
            return (), "backup_deadline"
        if prefix in memo:
            return memo[prefix], None
        leaf = leaf_index.get(prefix)
        if leaf is not None:
            leaf_result = (
                _WorldTreeBackup(
                    score=leaf.score,
                    representative_path=leaf.action_path,
                    decisions=(),
                ),
            )
            memo[prefix] = leaf_result
            return leaf_result, None

        expansion = expansion_index.get(prefix)
        next_actions = children.get(prefix, set())
        if expansion is None:
            if len(next_actions) != 1:
                return (), "malformed_forced_tree"
            action = next(iter(next_actions))
            forced_results, error = visit((*prefix, action))
            if not error:
                memo[prefix] = forced_results
            return forced_results, error

        branches: list[
            tuple[tuple[int, ...], tuple[_WorldTreeBackup, ...]]
        ] = []
        for action in expansion.candidates:
            if expired():
                return (), "backup_deadline"
            if action not in next_actions:
                return (), "continuation_branch_missing"
            child_strategies, error = visit((*prefix, action))
            if error is not None or not child_strategies:
                return (), error or "continuation_branch_missing"
            branches.append((action, child_strategies))

        strategies_result: tuple[_WorldTreeBackup, ...]
        if expansion.context == int(SelectContext.COIN_HEAD):
            if not expansion.exhaustive or len(branches) != 2:
                return (), "coin_chance_incomplete"
            strategies_result, error = _chance_strategies(
                branches,
                strategy_cap=strategy_cap,
                expired=expired,
            )
        else:
            strategies_result, error = _controlled_strategies(
                expansion.prompt_key,
                branches,
                strategy_cap=strategy_cap,
                expired=expired,
            )
        if error is not None:
            return (), error
        memo[prefix] = strategies_result
        return strategies_result, None

    return visit((plan.root_action,))


def _chance_strategies(
    branches: Sequence[
        tuple[tuple[int, ...], tuple[_WorldTreeBackup, ...]]
    ],
    *,
    strategy_cap: int,
    expired: Expired,
) -> tuple[tuple[_WorldTreeBackup, ...], str | None]:
    partials: list[tuple[_WorldTreeBackup, ...]] = [()]
    for _action, branch_strategies in branches:
        if expired():
            return (), "backup_deadline"
        expanded: list[tuple[_WorldTreeBackup, ...]] = []
        for partial in partials:
            for strategy in branch_strategies:
                if expired():
                    return (), "backup_deadline"
                if _merge_decisions(
                    *(item.decisions for item in partial),
                    strategy.decisions,
                ) is None:
                    continue
                expanded.append((*partial, strategy))
                if len(expanded) > strategy_cap:
                    return (), "joint_strategy_cap"
        partials = expanded
    strategies = tuple(
        _WorldTreeBackup(
            score=statistics.fmean(item.score for item in partial),
            representative_path=max(
                partial,
                key=lambda item: (item.score, item.representative_path),
            ).representative_path,
            decisions=_required_merged_decisions(
                *(item.decisions for item in partial)
            ),
        )
        for partial in partials
    )
    return _deduplicate_world_strategies(strategies), None


def _controlled_strategies(
    prompt_key: tuple[Any, ...],
    branches: Sequence[
        tuple[tuple[int, ...], tuple[_WorldTreeBackup, ...]]
    ],
    *,
    strategy_cap: int,
    expired: Expired,
) -> tuple[tuple[_WorldTreeBackup, ...], str | None]:
    strategies: list[_WorldTreeBackup] = []
    for action, child_strategies in branches:
        for child in child_strategies:
            if expired():
                return (), "backup_deadline"
            decisions = _merge_decisions(child.decisions, ((prompt_key, action),))
            if decisions is None:
                continue
            strategies.append(
                _WorldTreeBackup(
                    score=child.score,
                    representative_path=child.representative_path,
                    decisions=decisions,
                )
            )
            if len(strategies) > strategy_cap:
                return (), "joint_strategy_cap"
    return _deduplicate_world_strategies(strategies), None


def _select_nonanticipative_strategy(
    worlds: Sequence[Sequence[_WorldTreeBackup]],
    *,
    risk_std_weight: float,
    strategy_cap: int,
    expired: Expired,
) -> tuple[_PairedTreeBackup | None, str | None]:
    partials: tuple[_PairedTreeBackup, ...] = (
        _PairedTreeBackup(
            world_scores=(),
            representative_paths=(),
            decisions=(),
        ),
    )
    for world in worlds:
        if expired():
            return None, "backup_deadline"
        expanded: list[_PairedTreeBackup] = []
        for partial in partials:
            for strategy in world:
                if expired():
                    return None, "backup_deadline"
                decisions = _merge_decisions(
                    partial.decisions,
                    strategy.decisions,
                )
                if decisions is None:
                    continue
                expanded.append(
                    _PairedTreeBackup(
                        world_scores=(*partial.world_scores, strategy.score),
                        representative_paths=(
                            *partial.representative_paths,
                            strategy.representative_path,
                        ),
                        decisions=decisions,
                    )
                )
                if len(expanded) > strategy_cap:
                    return None, "joint_strategy_cap"
        partials = tuple(expanded)
        if not partials:
            return None, "nonanticipative_strategy_unavailable"

    def robust_score(item: _PairedTreeBackup) -> float:
        return statistics.fmean(item.world_scores) - risk_std_weight * (
            statistics.pstdev(item.world_scores)
        )

    return (
        max(
            partials,
            key=lambda item: (
                robust_score(item),
                statistics.fmean(item.world_scores),
                item.world_scores,
                item.representative_paths,
                repr(item.decisions),
            ),
        ),
        None,
    )


def _merge_decisions(
    *groups: Sequence[tuple[_DecisionKey, tuple[int, ...]]],
) -> tuple[tuple[_DecisionKey, tuple[int, ...]], ...] | None:
    merged: dict[_DecisionKey, tuple[int, ...]] = {}
    for group in groups:
        for key, action in group:
            previous = merged.setdefault(key, action)
            if previous != action:
                return None
    return tuple(sorted(merged.items(), key=lambda item: repr(item[0])))


def _required_merged_decisions(
    *groups: Sequence[tuple[_DecisionKey, tuple[int, ...]]],
) -> tuple[tuple[_DecisionKey, tuple[int, ...]], ...]:
    merged = _merge_decisions(*groups)
    if merged is None:
        raise RuntimeError("validated strategy decisions became inconsistent")
    return merged


def _deduplicate_world_strategies(
    strategies: Sequence[_WorldTreeBackup],
) -> tuple[_WorldTreeBackup, ...]:
    retained: dict[
        tuple[tuple[_DecisionKey, tuple[int, ...]], ...],
        _WorldTreeBackup,
    ] = {}
    for strategy in strategies:
        previous = retained.get(strategy.decisions)
        if previous is None or (
            strategy.score,
            strategy.representative_path,
        ) > (
            previous.score,
            previous.representative_path,
        ):
            retained[strategy.decisions] = strategy
    return tuple(retained.values())


def _has_uncontrolled_coin(plan: CompleteContinuationPlan) -> bool:
    coin_prompts = tuple(
        prompt
        for prompt in plan.prompt_expansions
        if prompt.context == int(SelectContext.COIN_HEAD)
    )
    for leaf in plan.complete_leaves:
        explicit_events = sum(
            leaf.action_path[: len(prompt.action_path)] == prompt.action_path
            for prompt in coin_prompts
        )
        observed_events = sum(len(summary.coins) for summary in leaf.summaries)
        if observed_events > explicit_events:
            return True
    return False


__all__ = ["CompleteActionScoreEvidence", "score_complete_action_plans"]
