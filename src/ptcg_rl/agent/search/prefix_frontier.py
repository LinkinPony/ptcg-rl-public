"""Deterministic globally bounded best-first selection-prefix search."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ptcg_rl.agent.search.selection_prefix import (
    SelectionPrefix,
    initial_selection_prefix,
)

STOP_CHOICE = -1
PrefixScoreFunction = Callable[[SelectionPrefix], Mapping[int, float]]


@dataclass(frozen=True)
class PrefixSearchProblem:
    """One prompt sharing a request-global frontier with other prompts."""

    root_key: str
    option_count: int
    min_count: int
    max_count: int
    ordered: bool
    score_next: PrefixScoreFunction


@dataclass(frozen=True)
class PrefixSearchLimits:
    """Global limits that are never multiplied by roots or depth."""

    node_budget: int
    frontier_limit: int
    per_depth_width: int
    candidate_limit: int
    depth_limit: int

    def __post_init__(self) -> None:
        for name in (
            "node_budget",
            "frontier_limit",
            "per_depth_width",
            "candidate_limit",
            "depth_limit",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class CompletedPrefixAction:
    """One complete action emitted by bounded prefix search."""

    root_key: str
    action: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class PrefixSearchResult:
    """Auditable completed support and globally consumed work."""

    actions: tuple[CompletedPrefixAction, ...]
    nodes_expanded: int
    frontier_peak: int
    baseline_completions: int
    stop_reason: str


@dataclass(frozen=True)
class _FrontierNode:
    root_ordinal: int
    root_key: str
    prefix: SelectionPrefix
    score: float

    @property
    def depth(self) -> int:
        return self.prefix.selected_count

    @property
    def identity(self) -> tuple[int, tuple[int, ...]]:
        return (self.root_ordinal, self.prefix.selected)

    @property
    def rank(self) -> tuple[float, int, tuple[int, ...]]:
        return (-self.score, self.root_ordinal, self.prefix.selected)


class GlobalPrefixFrontier:
    """Best-first frontier with request-global and per-depth beam ceilings."""

    def __init__(self, *, frontier_limit: int, per_depth_width: int) -> None:
        if frontier_limit <= 0 or per_depth_width <= 0:
            raise ValueError("frontier limits must be positive")
        self._frontier_limit = frontier_limit
        self._per_depth_width = per_depth_width
        self._nodes: dict[tuple[int, tuple[int, ...]], _FrontierNode] = {}
        self.peak_size = 0

    def offer(self, node: _FrontierNode) -> bool:
        """Retain a node only if it fits the global deterministic beam."""
        existing = self._nodes.get(node.identity)
        if existing is not None:
            if existing.rank <= node.rank:
                return False
            self._nodes[node.identity] = node
            return True

        same_depth = [item for item in self._nodes.values() if item.depth == node.depth]
        if len(same_depth) >= self._per_depth_width:
            worst = max(same_depth, key=lambda item: item.rank)
            if worst.rank <= node.rank:
                return False
            del self._nodes[worst.identity]

        if len(self._nodes) >= self._frontier_limit:
            worst = max(self._nodes.values(), key=lambda item: item.rank)
            if worst.rank <= node.rank:
                return False
            del self._nodes[worst.identity]
        self._nodes[node.identity] = node
        self.peak_size = max(self.peak_size, len(self._nodes))
        return True

    def pop_best(self) -> _FrontierNode | None:
        """Pop the globally best node with stable root/prefix tie breaks."""
        if not self._nodes:
            return None
        best = min(self._nodes.values(), key=lambda item: item.rank)
        del self._nodes[best.identity]
        return best

    def __bool__(self) -> bool:
        return bool(self._nodes)


def bounded_best_first_prefix_search(
    problems: Sequence[PrefixSearchProblem],
    limits: PrefixSearchLimits,
) -> PrefixSearchResult:
    """Search all problems with one frontier, node budget, and candidate cap."""
    if not problems:
        return PrefixSearchResult((), 0, 0, 0, "frontier_empty")
    if len({problem.root_key for problem in problems}) != len(problems):
        raise ValueError("prefix-search root keys must be unique")
    root_count = len(problems)
    if (
        limits.frontier_limit < root_count
        or limits.per_depth_width < root_count
        or limits.candidate_limit < root_count
        or limits.node_budget < root_count
    ):
        raise ValueError("global prefix limits must reserve every root anchor")
    if limits.node_budget < sum(problem.max_count for problem in problems):
        raise ValueError("node budget cannot cover one completion per root")

    by_root = {problem.root_key: problem for problem in problems}
    completed: dict[tuple[str, tuple[int, ...]], CompletedPrefixAction] = {}
    anchors: list[CompletedPrefixAction] = []
    nodes_expanded = 0
    for problem in problems:
        anchor, nodes_used = _greedy_completion(problem)
        nodes_expanded += nodes_used
        anchors.append(anchor)
        completed[(anchor.root_key, anchor.action)] = anchor
    frontier = GlobalPrefixFrontier(
        frontier_limit=limits.frontier_limit,
        per_depth_width=limits.per_depth_width,
    )
    for root_ordinal, problem in enumerate(problems):
        frontier.offer(
            _FrontierNode(
                root_ordinal=root_ordinal,
                root_key=problem.root_key,
                prefix=initial_selection_prefix(
                    option_count=problem.option_count,
                    min_count=problem.min_count,
                    max_count=problem.max_count,
                    ordered=problem.ordered,
                ),
                score=0.0,
            )
        )

    while frontier and nodes_expanded < limits.node_budget:
        if len(completed) >= limits.candidate_limit:
            break
        node = frontier.pop_best()
        if node is None:
            break
        nodes_expanded += 1
        problem = by_root[node.root_key]
        prefix = node.prefix
        if prefix.complete_at_max:
            _retain_completion(completed, node, node.score)
            continue
        next_scores = problem.score_next(prefix)
        _validate_next_scores(prefix, next_scores)
        stop_score = next_scores.get(STOP_CHOICE)
        if prefix.stop_allowed and stop_score is not None:
            _retain_completion(completed, node, node.score + stop_score)
        if prefix.selected_count >= limits.depth_limit:
            continue
        for option_index in range(prefix.option_count):
            option_score = next_scores.get(option_index)
            if option_score is None or not prefix.remaining_mask[option_index]:
                continue
            child = prefix.extend(option_index)
            frontier.offer(
                _FrontierNode(
                    root_ordinal=node.root_ordinal,
                    root_key=node.root_key,
                    prefix=child,
                    score=node.score + option_score,
                )
            )

    anchor_keys = {(anchor.root_key, anchor.action) for anchor in anchors}
    extras = sorted(
        (action for key, action in completed.items() if key not in anchor_keys),
        key=lambda item: (-item.score, item.root_key, item.action),
    )[: limits.candidate_limit - len(anchors)]
    ranked = tuple(
        sorted(
            (*anchors, *extras),
            key=lambda item: (-item.score, item.root_key, item.action),
        )
    )
    if len(ranked) >= limits.candidate_limit:
        reason = "candidate_limit"
    elif nodes_expanded >= limits.node_budget and frontier:
        reason = "node_budget"
    else:
        reason = "frontier_empty"
    return PrefixSearchResult(
        actions=ranked,
        nodes_expanded=nodes_expanded,
        frontier_peak=frontier.peak_size,
        baseline_completions=len(anchors),
        stop_reason=reason,
    )


def _greedy_completion(
    problem: PrefixSearchProblem,
) -> tuple[CompletedPrefixAction, int]:
    """Materialize one mandatory controller completion for a root."""
    prefix = initial_selection_prefix(
        option_count=problem.option_count,
        min_count=problem.min_count,
        max_count=problem.max_count,
        ordered=problem.ordered,
    )
    score = 0.0
    nodes_used = 0
    while not prefix.complete_at_max:
        next_scores = problem.score_next(prefix)
        _validate_next_scores(prefix, next_scores)
        nodes_used += 1
        choices = [
            (option_index, next_scores[option_index])
            for option_index in range(prefix.option_count)
            if prefix.remaining_mask[option_index] and option_index in next_scores
        ]
        if prefix.stop_allowed and STOP_CHOICE in next_scores:
            choices.append((STOP_CHOICE, next_scores[STOP_CHOICE]))
        if not choices:
            raise ValueError("prefix scorer cannot complete a legal root action")
        choice, choice_score = min(
            choices,
            key=lambda item: (
                -item[1],
                problem.option_count if item[0] == STOP_CHOICE else item[0],
            ),
        )
        score += choice_score
        if choice == STOP_CHOICE:
            break
        prefix = prefix.extend(choice)
    return (
        CompletedPrefixAction(
            root_key=problem.root_key,
            action=prefix.selected,
            score=score,
        ),
        nodes_used,
    )


def _retain_completion(
    completed: dict[tuple[str, tuple[int, ...]], CompletedPrefixAction],
    node: _FrontierNode,
    score: float,
) -> None:
    key = (node.root_key, node.prefix.selected)
    candidate = CompletedPrefixAction(
        root_key=node.root_key,
        action=node.prefix.selected,
        score=score,
    )
    existing = completed.get(key)
    if existing is None or candidate.score > existing.score:
        completed[key] = candidate


def _validate_next_scores(
    prefix: SelectionPrefix,
    scores: Mapping[int, float],
) -> None:
    for choice, score in scores.items():
        if choice != STOP_CHOICE and (choice < 0 or choice >= prefix.option_count):
            raise ValueError("prefix scorer returned an invalid option")
        if not math.isfinite(score):
            raise ValueError("prefix scores must be finite")


__all__ = [
    "CompletedPrefixAction",
    "GlobalPrefixFrontier",
    "PrefixSearchLimits",
    "PrefixSearchProblem",
    "PrefixSearchResult",
    "STOP_CHOICE",
    "bounded_best_first_prefix_search",
]
