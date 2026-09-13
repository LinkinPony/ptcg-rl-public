"""Versioned multi-source complete-action candidate construction."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from ptcg_rl.actions.selection import is_legal_action, random_legal_action
from ptcg_rl.agent.search.candidate_budget import (
    EXPANSION_SOURCE_ORDER,
    SEED_SOURCE_ORDER,
    CandidateBudgetPlan,
    CandidateConstructorConfig,
    ExpansionSource,
    SeedSource,
)
from ptcg_rl.agent.search.prompt_actions import (
    PromptActionSpace,
    describe_prompt_action_space,
)

_SourceT = TypeVar("_SourceT", bound=str)


@dataclass(frozen=True)
class CandidateSet:
    """Deduplicated complete actions and merged inclusion provenance."""

    actions: tuple[tuple[int, ...], ...]
    sources: tuple[tuple[str, ...], ...]
    architecture_version: int = 0
    seed_count: int = 0
    exhaustive: bool = False
    stochastic_seed: int | None = None

    def contains(self, action: Sequence[int]) -> bool:
        """Return whether an exact complete action is retained."""
        normalized = tuple(int(index) for index in action)
        return normalized in self.actions


@dataclass(frozen=True)
class CandidateSourceInputs:
    """Independently ranked complete actions from all seed sources."""

    base: tuple[tuple[int, ...], ...] = ()
    proposal: tuple[tuple[int, ...], ...] = ()
    cardinality: tuple[tuple[int, ...], ...] = ()
    stochastic: tuple[tuple[int, ...], ...] = ()
    structural: tuple[tuple[int, ...], ...] = ()
    random: tuple[tuple[int, ...], ...] = ()

    def as_dict(self) -> dict[SeedSource, tuple[tuple[int, ...], ...]]:
        """Return typed source streams in their fixed versioned order."""
        return {
            source: cast(tuple[tuple[int, ...], ...], getattr(self, source))
            for source in SEED_SOURCE_ORDER
        }


@dataclass(frozen=True)
class CandidateExpansionInputs:
    """Ranked novel complete actions after initial exact evaluation."""

    mutation: tuple[tuple[int, ...], ...] = ()
    novelty: tuple[tuple[int, ...], ...] = ()

    def as_dict(self) -> dict[ExpansionSource, tuple[tuple[int, ...], ...]]:
        """Return typed expansion streams without mixing seed quotas."""
        return {
            source: cast(tuple[tuple[int, ...], ...], getattr(self, source))
            for source in EXPANSION_SOURCE_ORDER
        }


@dataclass(frozen=True)
class CandidateConstructionResult:
    """Auditable output or an explicit base-policy fallback request."""

    candidates: CandidateSet
    valid: bool
    fallback_reason: str | None
    seed_source_usage: tuple[tuple[str, int], ...]
    expansion_source_usage: tuple[tuple[str, int], ...]


class MultiSourceCandidateConstructor:
    """Build a bounded support with anchors, quotas, dedup, and refill."""

    def __init__(self, config: CandidateConstructorConfig) -> None:
        self.config = config

    def construct(
        self,
        select: Any,
        *,
        greedy_action: Sequence[int],
        budget: CandidateBudgetPlan,
        seed_inputs: CandidateSourceInputs | None = None,
        expansion_inputs: CandidateExpansionInputs | None = None,
        ordered: bool | None = None,
        stochastic_seed: int = 0,
    ) -> CandidateConstructionResult:
        """Construct seed support and a separately budgeted expansion union."""
        if budget.architecture_version != self.config.architecture_version:
            raise ValueError("candidate budget and constructor versions differ")
        if not budget.feasible:
            return _fallback_result(
                version=self.config.architecture_version,
                reason=budget.fallback_reason or "budget_infeasible",
                stochastic_seed=stochastic_seed,
            )
        space = describe_prompt_action_space(select)
        order_sensitive = space.ordered if ordered is None else bool(ordered)
        legal_action_count = _legal_action_count(space, ordered=order_sensitive)
        if budget.k_total > legal_action_count:
            raise ValueError("candidate budget exceeds the legal action space")

        if budget.exhaustive:
            actions = _enumerate_actions(space, ordered=order_sensitive)
            if len(actions) != budget.k_seed:
                raise ValueError("exhaustive budget does not match the prompt")
            candidates = CandidateSet(
                actions=actions,
                sources=tuple(("exhaustive",) for _ in actions),
                architecture_version=self.config.architecture_version,
                seed_count=len(actions),
                exhaustive=True,
                stochastic_seed=stochastic_seed,
            )
            return CandidateConstructionResult(
                candidates=candidates,
                valid=True,
                fallback_reason=None,
                seed_source_usage=(),
                expansion_source_usage=(),
            )

        greedy = _normalize(greedy_action, ordered=order_sensitive)
        if not is_legal_action(select, greedy):
            return _fallback_result(
                version=self.config.architecture_version,
                reason="mandatory_base_greedy_illegal",
                stochastic_seed=stochastic_seed,
            )

        seed_streams = (seed_inputs or CandidateSourceInputs()).as_dict()
        seed_streams["random"] = (
            *seed_streams["random"],
            *_random_actions(
                select,
                ordered=order_sensitive,
                seed=stochastic_seed,
                attempts=max(16, budget.k_seed * 16),
            ),
        )
        normalized_seed = {
            source: _legal_unique_stream(
                select,
                actions,
                ordered=order_sensitive,
            )
            for source, actions in seed_streams.items()
        }
        by_action: dict[tuple[int, ...], list[str]] = {greedy: ["base_greedy"]}
        seed_usage = dict.fromkeys(SEED_SOURCE_ORDER, 0)
        seed_usage["base"] = 1
        _admit_stage(
            streams=normalized_seed,
            quotas=budget.seed_quotas.as_dict(),
            refill_priority=self.config.seed_refill_priority,
            limit=budget.k_seed,
            by_action=by_action,
            usage=seed_usage,
        )
        _merge_offered_provenance(by_action, normalized_seed)
        seed_count = len(by_action)

        expansion_streams = {
            source: _legal_unique_stream(
                select,
                actions,
                ordered=order_sensitive,
            )
            for source, actions in (expansion_inputs or CandidateExpansionInputs())
            .as_dict()
            .items()
        }
        expansion_usage = dict.fromkeys(EXPANSION_SOURCE_ORDER, 0)
        expansion_capacity = budget.k_total - budget.k_seed
        _admit_stage(
            streams=expansion_streams,
            quotas=budget.expansion_quotas.as_dict(),
            refill_priority=self.config.expansion_refill_priority,
            limit=min(budget.k_total, len(by_action) + expansion_capacity),
            by_action=by_action,
            usage=expansion_usage,
        )
        _merge_offered_provenance(by_action, expansion_streams)

        actions = tuple(by_action)
        candidates = CandidateSet(
            actions=actions,
            sources=tuple(tuple(by_action[action]) for action in actions),
            architecture_version=self.config.architecture_version,
            seed_count=seed_count,
            exhaustive=False,
            stochastic_seed=stochastic_seed,
        )
        return CandidateConstructionResult(
            candidates=candidates,
            valid=True,
            fallback_reason=None,
            seed_source_usage=tuple(
                (source, seed_usage[source]) for source in SEED_SOURCE_ORDER
            ),
            expansion_source_usage=tuple(
                (source, expansion_usage[source]) for source in EXPANSION_SOURCE_ORDER
            ),
        )


def build_candidate_set(
    select: Any,
    *,
    greedy_action: Sequence[int],
    policy_actions: Sequence[Sequence[int]] = (),
    structural_actions: Sequence[Sequence[int]] = (),
    exploration_actions: int = 0,
    rng: random.Random | None = None,
) -> CandidateSet:
    """Build the focused legacy runtime union without tactical-exact claims.

    Structural anchors are never labelled as exact evidence. New planner code
    must use :class:`MultiSourceCandidateConstructor`.
    """
    by_action: dict[tuple[int, ...], list[str]] = {}

    def add(action: Sequence[int], source: str) -> None:
        normalized = _normalize(
            action, ordered=describe_prompt_action_space(select).ordered
        )
        if not is_legal_action(select, normalized):
            return
        labels = by_action.setdefault(normalized, [])
        if source not in labels:
            labels.append(source)

    add(greedy_action, "greedy")
    for action in policy_actions:
        add(action, "policy_top_k")
    for action in structural_actions:
        add(action, "structural")

    active_rng = rng or random.Random()
    attempts = 0
    target_count = len(by_action) + max(0, int(exploration_actions))
    max_attempts = max(16, exploration_actions * 16)
    while len(by_action) < target_count and attempts < max_attempts:
        add(random_legal_action(select, rng=active_rng), "exploration")
        attempts += 1

    actions = tuple(by_action)
    return CandidateSet(
        actions=actions,
        sources=tuple(tuple(by_action[action]) for action in actions),
    )


def _admit_stage(
    *,
    streams: Mapping[_SourceT, tuple[tuple[int, ...], ...]],
    quotas: Mapping[_SourceT, int],
    refill_priority: Sequence[_SourceT],
    limit: int,
    by_action: dict[tuple[int, ...], list[str]],
    usage: dict[_SourceT, int],
) -> None:
    cursors = dict.fromkeys(streams, 0)

    def add_next(source: _SourceT) -> bool:
        stream = streams[source]
        cursor = cursors[source]
        while cursor < len(stream):
            action = stream[cursor]
            cursor += 1
            cursors[source] = cursor
            if action in by_action:
                _append_label(by_action[action], source)
                continue
            by_action[action] = [source]
            usage[source] += 1
            return True
        return False

    for source in streams:
        while len(by_action) < limit and usage[source] < int(quotas[source]):
            if not add_next(source):
                break
    for source in refill_priority:
        while len(by_action) < limit and add_next(source):
            pass


def _merge_offered_provenance(
    by_action: dict[tuple[int, ...], list[str]],
    streams: Mapping[_SourceT, tuple[tuple[int, ...], ...]],
) -> None:
    for source, actions in streams.items():
        for action in actions:
            labels = by_action.get(action)
            if labels is not None:
                _append_label(labels, source)


def _append_label(labels: list[str], source: str) -> None:
    if source not in labels:
        labels.append(source)


def _legal_unique_stream(
    select: Any,
    actions: Sequence[Sequence[int]],
    *,
    ordered: bool,
) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for action in actions:
        normalized = _normalize(action, ordered=ordered)
        if normalized in seen or not is_legal_action(select, normalized):
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _random_actions(
    select: Any,
    *,
    ordered: bool,
    seed: int,
    attempts: int,
) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    actions: list[tuple[int, ...]] = []
    for _ in range(attempts):
        action = list(random_legal_action(select, rng=rng))
        if ordered and len(action) > 1:
            rng.shuffle(action)
        actions.append(_normalize(action, ordered=ordered))
    return tuple(actions)


def _enumerate_actions(
    space: PromptActionSpace,
    *,
    ordered: bool,
) -> tuple[tuple[int, ...], ...]:
    from itertools import combinations, permutations

    factory = permutations if ordered else combinations
    return tuple(
        tuple(action)
        for count in range(space.min_count, space.max_count + 1)
        for action in factory(range(space.option_count), count)
    )


def _legal_action_count(space: PromptActionSpace, *, ordered: bool) -> int:
    import math

    counter = math.perm if ordered else math.comb
    return sum(
        counter(space.option_count, count)
        for count in range(space.min_count, space.max_count + 1)
    )


def _normalize(action: Sequence[int], *, ordered: bool) -> tuple[int, ...]:
    normalized = tuple(int(index) for index in action)
    return normalized if ordered else tuple(sorted(normalized))


def _fallback_result(
    *,
    version: int,
    reason: str,
    stochastic_seed: int,
) -> CandidateConstructionResult:
    return CandidateConstructionResult(
        candidates=CandidateSet(
            actions=(),
            sources=(),
            architecture_version=version,
            stochastic_seed=stochastic_seed,
        ),
        valid=False,
        fallback_reason=reason,
        seed_source_usage=(),
        expansion_source_usage=(),
    )


__all__ = [
    "CandidateConstructionResult",
    "CandidateExpansionInputs",
    "CandidateSet",
    "CandidateSourceInputs",
    "MultiSourceCandidateConstructor",
    "build_candidate_set",
]
