"""Deterministic batched top-K search under learned proposal prefix logits."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.agent.search.selection_prefix import (
    SelectionPrefix,
    initial_selection_prefix,
)

_FINGERPRINT_DOMAIN = b"ptcg-rl/planner-proposal-search/v1\x00"
PLANNER_PROPOSAL_ARCHITECTURE_VERSION = 1


class PlannerProposalDeadlineError(TimeoutError):
    """Raised before a proposal model launch that missed its deadline."""


def ensure_planner_proposal_deadline(
    deadline_monotonic: float | None,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Reject stale proposal work immediately before a model invocation."""
    if deadline_monotonic is None:
        return
    deadline = float(deadline_monotonic)
    if not math.isfinite(deadline) or deadline <= 0.0:
        raise ValueError("proposal deadline must be finite and positive")
    if clock() >= deadline:
        raise PlannerProposalDeadlineError(
            "planner proposal deadline expired before model launch"
        )


class PlannerProposalSearchLimits(BaseModel):
    """Per-decision search limits with a cross-decision scoring batch cap."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_version: Literal[1] = 1
    candidate_limit: int
    prefix_node_limit: int
    frontier_limit: int
    per_depth_width: int
    depth_limit: int
    max_scoring_rows: int

    @field_validator(
        "candidate_limit",
        "prefix_node_limit",
        "frontier_limit",
        "per_depth_width",
        "depth_limit",
        "max_scoring_rows",
    )
    @classmethod
    def positive_limit(cls, value: int) -> int:
        """Require every proposal-search capacity to be positive."""
        if value <= 0:
            raise ValueError("proposal search limits must be positive")
        return value

    @model_validator(mode="after")
    def coherent_frontier(self) -> Self:
        """Ensure the declared beam can retain at least one candidate path."""
        if self.frontier_limit < self.per_depth_width:
            raise ValueError("frontier_limit cannot be below per_depth_width")
        return self

    @property
    def fingerprint(self) -> str:
        """Return the immutable proposal-generation semantics identity."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(_FINGERPRINT_DOMAIN + payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PlannerProposalProblem:
    """One root-observable prompt in a shared GPU scoring campaign."""

    root_index: int
    option_count: int
    min_count: int
    max_count: int
    ordered: bool

    def __post_init__(self) -> None:
        if self.root_index < 0:
            raise ValueError("proposal root_index must be non-negative")
        initial_selection_prefix(
            option_count=self.option_count,
            min_count=self.min_count,
            max_count=self.max_count,
            ordered=self.ordered,
        )


@dataclass(frozen=True, slots=True)
class PlannerProposalPrefixQuery:
    """One prefix row sent through the shared pointer-decoder trunk."""

    root_index: int
    prefix: SelectionPrefix


class PlannerProposalPrefixScorer(Protocol):
    """Batched learned proposal scorer used by deterministic search."""

    def score_prefixes(
        self,
        queries: tuple[PlannerProposalPrefixQuery, ...],
    ) -> Sequence[Sequence[float]]:
        """Return option-plus-STOP log-probabilities for every prefix."""


@dataclass(frozen=True, slots=True)
class PlannerProposalCandidate:
    """One complete action ranked by learned proposal log-probability."""

    action: tuple[int, ...]
    proposal_logprob: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.proposal_logprob):
            raise ValueError("proposal candidate log-probability must be finite")


@dataclass(frozen=True, slots=True)
class PlannerProposalDecisionResult:
    """Bounded proposal support and per-decision work telemetry."""

    candidates: tuple[PlannerProposalCandidate, ...]
    nodes_expanded: int
    frontier_peak: int
    pruned_prefixes: int
    discarded_score_upper_bound: float
    greedy_anchor: tuple[int, ...]
    stop_reason: Literal[
        "top_k_certified",
        "node_budget",
        "beam_pruned",
        "frontier_empty",
    ]


@dataclass(frozen=True, slots=True)
class PlannerProposalBatchResult:
    """Decision-aligned proposal supports produced by shared score batches."""

    decisions: tuple[PlannerProposalDecisionResult, ...]
    scoring_batches: int
    scoring_rows: int
    search_fingerprint: str
    # The generic prefix-search helper does not know the base policy.  Model
    # serving fills this field from the same retained policy context so the
    # mandatory constructor anchor is the actual base-policy greedy action,
    # never the proposal-head greedy path.
    base_greedy_actions: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.base_greedy_actions and len(self.base_greedy_actions) != len(
            self.decisions
        ):
            raise ValueError("base greedy actions must align with proposal roots")

    @property
    def actions(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Return decision-aligned top-K complete actions."""
        return tuple(
            tuple(candidate.action for candidate in decision.candidates)
            for decision in self.decisions
        )


@dataclass(frozen=True, slots=True)
class _FrontierNode:
    prefix: SelectionPrefix
    score: float

    @property
    def rank(self) -> tuple[float, tuple[int, ...]]:
        return (-self.score, self.prefix.selected)


@dataclass
class _DecisionSearchState:
    problem: PlannerProposalProblem
    limits: PlannerProposalSearchLimits
    frontier: dict[tuple[int, ...], _FrontierNode] = field(default_factory=dict)
    completed: dict[tuple[int, ...], float] = field(default_factory=dict)
    nodes_expanded: int = 0
    frontier_peak: int = 0
    pruned_prefixes: int = 0
    discarded_score_upper_bound: float = -math.inf
    greedy_anchor: tuple[int, ...] | None = None

    def record_pruned(self, node: _FrontierNode) -> None:
        """Retain a sound upper bound for every discarded prefix subtree."""
        self.pruned_prefixes += 1
        self.discarded_score_upper_bound = max(
            self.discarded_score_upper_bound,
            node.score,
        )

    def offer(self, node: _FrontierNode) -> None:
        identity = node.prefix.selected
        existing = self.frontier.get(identity)
        if existing is not None:
            if existing.rank <= node.rank:
                return
            self.frontier[identity] = node
            return
        same_depth = tuple(
            item
            for item in self.frontier.values()
            if item.prefix.selected_count == node.prefix.selected_count
        )
        if len(same_depth) >= self.limits.per_depth_width:
            worst = max(same_depth, key=lambda item: item.rank)
            if worst.rank <= node.rank:
                self.record_pruned(node)
                return
            del self.frontier[worst.prefix.selected]
            self.record_pruned(worst)
        if len(self.frontier) >= self.limits.frontier_limit:
            worst = max(self.frontier.values(), key=lambda item: item.rank)
            if worst.rank <= node.rank:
                self.record_pruned(node)
                return
            del self.frontier[worst.prefix.selected]
            self.record_pruned(worst)
        self.frontier[identity] = node
        self.frontier_peak = max(self.frontier_peak, len(self.frontier))

    def pop_best(self) -> _FrontierNode | None:
        if not self.frontier:
            return None
        best = min(self.frontier.values(), key=lambda item: item.rank)
        del self.frontier[best.prefix.selected]
        return best

    def retain(self, action: tuple[int, ...], score: float) -> None:
        existing = self.completed.get(action)
        if existing is None or score > existing:
            self.completed[action] = score
        if len(self.completed) > self.limits.candidate_limit:
            worst_action, _worst_score = max(
                self.completed.items(),
                key=lambda item: (-item[1], item[0]),
            )
            del self.completed[worst_action]

    def top_k_certified(self) -> bool:
        if len(self.completed) < self.limits.candidate_limit:
            return False
        ranked = sorted(
            self.completed.items(),
            key=lambda item: (-item[1], item[0]),
        )
        threshold = ranked[self.limits.candidate_limit - 1][1]
        return (
            self.discarded_score_upper_bound < threshold
            and all(node.score < threshold for node in self.frontier.values())
        )


def generate_batched_planner_proposals(
    problems: Sequence[PlannerProposalProblem],
    *,
    limits: PlannerProposalSearchLimits,
    scorer: PlannerProposalPrefixScorer,
    deadline_monotonic: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> PlannerProposalBatchResult:
    """Generate deterministic top-K supports with per-root global budgets."""
    ensure_planner_proposal_deadline(deadline_monotonic, clock=clock)
    frozen = tuple(problems)
    if not frozen:
        return PlannerProposalBatchResult((), 0, 0, limits.fingerprint)
    root_indices = tuple(problem.root_index for problem in frozen)
    if root_indices != tuple(range(len(frozen))):
        raise ValueError("proposal problems must use contiguous root indices")
    for problem in frozen:
        if problem.max_count > limits.depth_limit:
            raise ValueError("proposal depth_limit cannot complete one root")
        if problem.max_count > limits.prefix_node_limit:
            raise ValueError("proposal node budget cannot complete one root")
    states = tuple(
        _DecisionSearchState(problem=problem, limits=limits) for problem in frozen
    )
    scoring_batches = 0
    scoring_rows = 0

    active_anchors: dict[int, tuple[SelectionPrefix, float]] = {
        state.problem.root_index: (
            initial_selection_prefix(
                option_count=state.problem.option_count,
                min_count=state.problem.min_count,
                max_count=state.problem.max_count,
                ordered=state.problem.ordered,
            ),
            0.0,
        )
        for state in states
    }
    while active_anchors:
        next_active: dict[int, tuple[SelectionPrefix, float]] = {}
        active_items = tuple(active_anchors.items())
        for offset in range(0, len(active_items), limits.max_scoring_rows):
            chunk = active_items[offset : offset + limits.max_scoring_rows]
            queries = tuple(
                PlannerProposalPrefixQuery(root_index=root, prefix=prefix)
                for root, (prefix, _score) in chunk
            )
            ensure_planner_proposal_deadline(deadline_monotonic, clock=clock)
            score_rows = _score_queries(scorer, queries)
            scoring_batches += 1
            scoring_rows += len(queries)
            for query, row in zip(queries, score_rows, strict=True):
                state = states[query.root_index]
                state.nodes_expanded += 1
                prefix_score = active_anchors[query.root_index][1]
                choice, logprob = _greedy_choice(query.prefix, row)
                completed_score = prefix_score + logprob
                if choice == query.prefix.option_count:
                    state.greedy_anchor = query.prefix.selected
                    state.retain(query.prefix.selected, completed_score)
                    continue
                child = query.prefix.extend(choice)
                if child.complete_at_max:
                    state.greedy_anchor = child.selected
                    state.retain(child.selected, completed_score)
                else:
                    next_active[query.root_index] = (child, completed_score)
        active_anchors = next_active

    for state in states:
        if state.greedy_anchor is None:
            if state.problem.max_count != 0:
                raise RuntimeError("proposal greedy anchor did not complete")
            state.greedy_anchor = ()
            state.retain((), 0.0)
        if state.nodes_expanded < limits.prefix_node_limit:
            state.offer(
                _FrontierNode(
                    prefix=initial_selection_prefix(
                        option_count=state.problem.option_count,
                        min_count=state.problem.min_count,
                        max_count=state.problem.max_count,
                        ordered=state.problem.ordered,
                    ),
                    score=0.0,
                )
            )

    while True:
        scheduled: list[tuple[int, _FrontierNode]] = []
        for state in states:
            if (
                state.nodes_expanded >= limits.prefix_node_limit
                or state.top_k_certified()
            ):
                continue
            node = state.pop_best()
            if node is None:
                continue
            scheduled.append((state.problem.root_index, node))
            state.nodes_expanded += 1
        if not scheduled:
            break
        for offset in range(0, len(scheduled), limits.max_scoring_rows):
            scheduled_chunk = scheduled[
                offset : offset + limits.max_scoring_rows
            ]
            queries = tuple(
                PlannerProposalPrefixQuery(root_index=root, prefix=node.prefix)
                for root, node in scheduled_chunk
            )
            ensure_planner_proposal_deadline(deadline_monotonic, clock=clock)
            score_rows = _score_queries(scorer, queries)
            scoring_batches += 1
            scoring_rows += len(queries)
            for (root, node), row in zip(
                scheduled_chunk,
                score_rows,
                strict=True,
            ):
                state = states[root]
                prefix = node.prefix
                stop_logprob = row[prefix.option_count]
                if prefix.stop_allowed and math.isfinite(stop_logprob):
                    state.retain(prefix.selected, node.score + stop_logprob)
                if prefix.selected_count >= limits.depth_limit:
                    continue
                for option_index in range(prefix.option_count):
                    logprob = row[option_index]
                    if not math.isfinite(logprob):
                        continue
                    child = prefix.extend(option_index)
                    child_node = _FrontierNode(
                        prefix=child,
                        score=node.score + logprob,
                    )
                    if child.complete_at_max:
                        state.retain(child.selected, child_node.score)
                    else:
                        state.offer(child_node)

    decisions: list[PlannerProposalDecisionResult] = []
    for state in states:
        ranked = sorted(
            state.completed.items(),
            key=lambda item: (-item[1], item[0]),
        )[: limits.candidate_limit]
        if not ranked:
            raise RuntimeError("proposal search produced no complete candidate")
        greedy_anchor = state.greedy_anchor
        if greedy_anchor is None:
            raise RuntimeError("proposal greedy anchor was not materialized")
        if state.top_k_certified():
            stop_reason: Literal[
                "top_k_certified",
                "node_budget",
                "beam_pruned",
                "frontier_empty",
            ] = "top_k_certified"
        elif state.nodes_expanded >= limits.prefix_node_limit and state.frontier:
            stop_reason = "node_budget"
        elif state.pruned_prefixes:
            stop_reason = "beam_pruned"
        else:
            stop_reason = "frontier_empty"
        decisions.append(
            PlannerProposalDecisionResult(
                candidates=tuple(
                    PlannerProposalCandidate(action=action, proposal_logprob=score)
                    for action, score in ranked
                ),
                nodes_expanded=state.nodes_expanded,
                frontier_peak=state.frontier_peak,
                pruned_prefixes=state.pruned_prefixes,
                discarded_score_upper_bound=(
                    state.discarded_score_upper_bound
                ),
                greedy_anchor=greedy_anchor,
                stop_reason=stop_reason,
            )
        )
    return PlannerProposalBatchResult(
        decisions=tuple(decisions),
        scoring_batches=scoring_batches,
        scoring_rows=scoring_rows,
        search_fingerprint=limits.fingerprint,
    )


def _score_queries(
    scorer: PlannerProposalPrefixScorer,
    queries: tuple[PlannerProposalPrefixQuery, ...],
) -> tuple[tuple[float, ...], ...]:
    raw = tuple(
        tuple(float(value) for value in row)
        for row in scorer.score_prefixes(queries)
    )
    if len(raw) != len(queries):
        raise ValueError("proposal scorer returned another batch size")
    for query, row in zip(queries, raw, strict=True):
        if len(row) != query.prefix.option_count + 1:
            raise ValueError("proposal score row has the wrong width")
        if any(math.isnan(value) or value == math.inf for value in row):
            raise ValueError("proposal scores contain NaN or positive infinity")
        if any(value > 1.0e-6 for value in row):
            raise ValueError("proposal scorer must return log-probabilities")
        prefix = query.prefix
        for option_index, available in enumerate(prefix.remaining_mask):
            if not available and math.isfinite(row[option_index]):
                raise ValueError("proposal scorer assigned mass to a used option")
        stop_logprob = row[prefix.option_count]
        if not prefix.stop_allowed and math.isfinite(stop_logprob):
            raise ValueError("proposal scorer assigned mass to an illegal STOP")
        if prefix.complete_at_max and any(
            math.isfinite(row[index]) for index in range(prefix.option_count)
        ):
            raise ValueError("proposal scorer extended a complete prefix")
        if not any(math.isfinite(value) for value in row):
            raise ValueError("proposal prefix has no finite continuation")
    return raw


def _greedy_choice(
    prefix: SelectionPrefix,
    scores: tuple[float, ...],
) -> tuple[int, float]:
    choices = tuple(
        (index, score)
        for index, score in enumerate(scores)
        if math.isfinite(score)
        and (
            (index < prefix.option_count and prefix.remaining_mask[index])
            or (index == prefix.option_count and prefix.stop_allowed)
        )
    )
    if not choices:
        raise ValueError("proposal prefix has no legal greedy choice")
    return min(choices, key=lambda item: (-item[1], item[0]))


__all__ = [
    "PlannerProposalDeadlineError",
    "PlannerProposalBatchResult",
    "PlannerProposalCandidate",
    "PlannerProposalDecisionResult",
    "PlannerProposalPrefixQuery",
    "PlannerProposalPrefixScorer",
    "PlannerProposalProblem",
    "PlannerProposalSearchLimits",
    "ensure_planner_proposal_deadline",
    "generate_batched_planner_proposals",
]
