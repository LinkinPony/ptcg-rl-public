"""Determinized root search over belief samples."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.belief.state import Determinization, OpponentBeliefState
from ptcg_rl.engine.forward_model import enumerate_select_actions
from ptcg_rl.engine.protocols import ObservationInput, ObservationLike, SearchStateLike
from ptcg_rl.engine.session import SearchSession

RolloutScoreFn = Callable[[ObservationLike, int], float]


class PolicyValueEvaluator(Protocol):
    """Policy priors and value estimates used by guided root search."""

    def action_priors(
        self,
        observation: ObservationLike,
        actions: Sequence[tuple[int, ...]],
    ) -> Mapping[tuple[int, ...], float]:
        """Return policy prior scores for complete candidate actions."""

    def value(self, observation: ObservationLike, root_player_index: int) -> float:
        """Return a value estimate from the root player's perspective."""


class DeterminizedSearchConfig(BaseModel):
    """Budget knobs for information-set root search."""

    model_config = ConfigDict(extra="forbid")

    determinizations: int = 4
    rollouts_per_action: int = 1
    rollout_depth: int = 3
    max_root_actions: int = 64
    max_rollout_actions: int = 32
    puct_simulations: int = 64
    puct_exploration: float = 1.25
    max_tree_nodes: int = 96
    wall_time_seconds: float | None = None
    manual_coin: bool = False

    @field_validator(
        "determinizations",
        "rollouts_per_action",
        "max_root_actions",
        "max_rollout_actions",
        "puct_simulations",
    )
    @classmethod
    def valid_positive(cls, value: int) -> int:
        """Reject non-positive budget values."""
        if value <= 0:
            raise ValueError("search budget values must be positive")
        return value

    @field_validator("rollout_depth")
    @classmethod
    def valid_rollout_depth(cls, value: int) -> int:
        """Reject negative rollout depths."""
        if value < 0:
            raise ValueError("rollout_depth must be non-negative")
        return value

    @field_validator("max_tree_nodes")
    @classmethod
    def valid_tree_nodes(cls, value: int) -> int:
        """Keep stored Search API states inside the engine's 128-slot pool."""
        if value <= 0 or value > 128:
            raise ValueError("max_tree_nodes must be in [1, 128]")
        return value

    @field_validator("puct_exploration")
    @classmethod
    def valid_exploration(cls, value: float) -> float:
        """Reject non-positive PUCT exploration constants."""
        if value <= 0.0:
            raise ValueError("puct_exploration must be positive")
        return value

    @field_validator("wall_time_seconds")
    @classmethod
    def valid_wall_time(cls, value: float | None) -> float | None:
        """Reject non-positive wall-clock budgets."""
        if value is not None and value <= 0.0:
            raise ValueError("wall_time_seconds must be positive when set")
        return value


@dataclass
class RootActionStats:
    """Aggregated root-action statistics across determinizations."""

    action: tuple[int, ...]
    visits: int = 0
    total_score: float = 0.0
    total_prior: float = 0.0
    sources: dict[str, int] = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        """Return average rollout score for this action."""
        if self.visits == 0:
            return 0.0
        return self.total_score / float(self.visits)

    @property
    def mean_prior(self) -> float:
        """Return average policy prior for this action."""
        if self.visits == 0:
            return 0.0
        return self.total_prior / float(self.visits)

    def add(self, score: float, source: str, *, prior: float = 0.0) -> None:
        """Add one rollout score."""
        self.visits += 1
        self.total_score += score
        self.total_prior += prior
        self.sources[source] = self.sources.get(source, 0) + 1


@dataclass(frozen=True)
class DeterminizedSearchResult:
    """Root-level information-set search result."""

    stats: tuple[RootActionStats, ...]
    determinizations: tuple[Determinization, ...]

    @property
    def best_action(self) -> tuple[int, ...]:
        """Return the highest-mean action, or empty selection if none exists."""
        if not self.stats:
            return ()
        return max(
            self.stats,
            key=lambda item: (item.mean_score, item.mean_prior, item.visits),
        ).action


def run_determinized_root_search(
    observation: ObservationInput,
    *,
    your_deck: Sequence[int],
    sampler: BeliefSampler,
    opponent_state: OpponentBeliefState | None = None,
    config: DeterminizedSearchConfig | None = None,
    rng: random.Random | None = None,
    score_fn: RolloutScoreFn | None = None,
    evaluator: PolicyValueEvaluator | None = None,
) -> DeterminizedSearchResult:
    """Run flat IS-MCTS-style root aggregation over sampled determinizations."""
    active_config = config or DeterminizedSearchConfig()
    active_rng = rng or random.Random()
    scorer = score_fn or default_rollout_score
    stats: dict[tuple[int, ...], RootActionStats] = {}
    samples: list[Determinization] = []

    for _ in range(active_config.determinizations):
        sample = sampler.sample(
            observation,
            your_deck=your_deck,
            opponent_state=opponent_state,
            rng=active_rng,
        )
        samples.append(sample)
        with SearchSession.begin(
            observation,
            sample.hidden,
            manual_coin=active_config.manual_coin,
        ) as session:
            root_actions = enumerate_select_actions(
                session.root.observation.select,
                max_actions=active_config.max_root_actions,
                rng=active_rng,
            )
            root_priors = _action_priors(evaluator, session.root.observation, root_actions)
            for action in root_actions:
                action_stats = stats.setdefault(action, RootActionStats(action=action))
                for _ in range(active_config.rollouts_per_action):
                    score = _rollout_root_action(
                        session,
                        action,
                        root_player_index=session.root.observation.current.yourIndex
                        if session.root.observation.current is not None
                        else 0,
                        depth=active_config.rollout_depth,
                        rng=active_rng,
                        score_fn=scorer,
                        evaluator=evaluator,
                        rollout_action_limit=active_config.max_rollout_actions,
                    )
                    action_stats.add(
                        score,
                        sample.source,
                        prior=root_priors.get(action, 0.0),
                    )

    ordered_stats = tuple(
        sorted(
            stats.values(),
            key=lambda item: (item.mean_score, item.mean_prior, item.visits),
            reverse=True,
        )
    )
    return DeterminizedSearchResult(
        stats=ordered_stats,
        determinizations=tuple(samples),
    )


def run_determinized_puct_search(
    observation: ObservationInput,
    *,
    your_deck: Sequence[int],
    sampler: BeliefSampler,
    opponent_state: OpponentBeliefState | None = None,
    config: DeterminizedSearchConfig | None = None,
    rng: random.Random | None = None,
    score_fn: RolloutScoreFn | None = None,
    evaluator: PolicyValueEvaluator | None = None,
) -> DeterminizedSearchResult:
    """Run bounded PUCT trees over sampled determinizations."""
    active_config = config or DeterminizedSearchConfig()
    active_rng = rng or random.Random()
    scorer = score_fn or default_rollout_score
    deadline = _deadline(active_config.wall_time_seconds)
    stats: dict[tuple[int, ...], RootActionStats] = {}
    samples: list[Determinization] = []

    for _ in range(active_config.determinizations):
        if _deadline_expired(deadline):
            break
        sample = sampler.sample(
            observation,
            your_deck=your_deck,
            opponent_state=opponent_state,
            rng=active_rng,
        )
        samples.append(sample)
        with SearchSession.begin(
            observation,
            sample.hidden,
            manual_coin=active_config.manual_coin,
        ) as session:
            root_node = _PuctNode.from_search_state(session.root)
            root_player_index = root_node.player_index
            tree = _PuctTree(
                session=session,
                root=root_node,
                max_nodes=active_config.max_tree_nodes,
            )
            try:
                _expand_puct_node(
                    root_node,
                    evaluator=evaluator,
                    max_actions=active_config.max_root_actions,
                    rng=active_rng,
                )
                for _simulation in range(active_config.puct_simulations):
                    if _deadline_expired(deadline) or not root_node.edges:
                        break
                    _run_puct_simulation(
                        tree,
                        root_player_index=root_player_index,
                        evaluator=evaluator,
                        score_fn=scorer,
                        max_actions=active_config.max_rollout_actions,
                        exploration=active_config.puct_exploration,
                        rng=active_rng,
                    )
                _merge_puct_root_stats(stats, root_node, source=sample.source)
            finally:
                tree.release()

    ordered_stats = tuple(
        sorted(
            stats.values(),
            key=lambda item: (item.mean_score, item.mean_prior, item.visits),
            reverse=True,
        )
    )
    return DeterminizedSearchResult(
        stats=ordered_stats,
        determinizations=tuple(samples),
    )


def default_rollout_score(observation: ObservationLike, root_player_index: int) -> float:
    """Score a rollout leaf from the original agent player's perspective."""
    state = observation.current
    if state is None:
        return 0.0
    if int(state.result) == root_player_index:
        return 1.0
    if int(state.result) == 1 - root_player_index:
        return -1.0
    if int(state.result) == 2:
        return 0.0
    your_prizes_left = len(state.players[root_player_index].prize)
    opponent_prizes_left = len(state.players[1 - root_player_index].prize)
    return float(opponent_prizes_left - your_prizes_left) / 6.0


def _rollout_root_action(
    session: SearchSession,
    action: Sequence[int],
    *,
    root_player_index: int,
    depth: int,
    rng: random.Random,
    score_fn: RolloutScoreFn,
    evaluator: PolicyValueEvaluator | None,
    rollout_action_limit: int,
) -> float:
    visited: list[int] = []
    try:
        state = session.step(session.root.searchId, action)
        visited.append(int(state.searchId))
        state = _random_rollout(
            session,
            state,
            depth=max(0, depth - 1),
            rng=rng,
            visited=visited,
            evaluator=evaluator,
            rollout_action_limit=rollout_action_limit,
        )
        return _leaf_score(
            state.observation,
            root_player_index,
            evaluator=evaluator,
            score_fn=score_fn,
        )
    finally:
        for search_id in reversed(visited):
            session.release(search_id)


def _random_rollout(
    session: SearchSession,
    state: SearchStateLike,
    *,
    depth: int,
    rng: random.Random,
    visited: list[int],
    evaluator: PolicyValueEvaluator | None,
    rollout_action_limit: int,
) -> SearchStateLike:
    current = state
    for _ in range(depth):
        if current.observation.current is not None and int(current.observation.current.result) != -1:
            break
        actions = enumerate_select_actions(
            current.observation.select,
            max_actions=rollout_action_limit,
            rng=rng,
        )
        if not actions:
            break
        current = session.step(
            current.searchId,
            _rollout_action(current.observation, actions, evaluator=evaluator, rng=rng),
        )
        visited.append(int(current.searchId))
    return current


def _leaf_score(
    observation: ObservationLike,
    root_player_index: int,
    *,
    evaluator: PolicyValueEvaluator | None,
    score_fn: RolloutScoreFn,
) -> float:
    if evaluator is not None:
        return evaluator.value(observation, root_player_index)
    return score_fn(observation, root_player_index)


@dataclass
class _PuctEdge:
    """One stored complete-action edge in a PUCT tree."""

    action: tuple[int, ...]
    prior: float
    visits: int = 0
    total_value: float = 0.0
    child: _PuctNode | None = None

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / float(self.visits)


@dataclass
class _PuctNode:
    """Stored Search API state plus lazily expanded action edges."""

    search_id: int
    observation: ObservationLike
    player_index: int
    edges: list[_PuctEdge] = field(default_factory=list)
    expanded: bool = False

    @classmethod
    def from_search_state(cls, state: SearchStateLike) -> _PuctNode:
        """Create a tree node from one engine search state."""
        return cls(
            search_id=int(state.searchId),
            observation=state.observation,
            player_index=_player_index(state.observation),
        )


@dataclass
class _PuctTree:
    """Bounded set of live Search API states for one determinization."""

    session: SearchSession
    root: _PuctNode
    max_nodes: int
    node_count: int = 1
    released: bool = False

    def can_store_child(self) -> bool:
        """Return whether another child state can be kept live."""
        return self.node_count < self.max_nodes

    def register_child(self) -> None:
        """Record one newly stored child state."""
        self.node_count += 1

    def release(self) -> None:
        """Release stored non-root states before ending the Search session."""
        if self.released:
            return
        _release_puct_children(self.session, self.root)
        self.released = True


def _run_puct_simulation(
    tree: _PuctTree,
    *,
    root_player_index: int,
    evaluator: PolicyValueEvaluator | None,
    score_fn: RolloutScoreFn,
    max_actions: int,
    exploration: float,
    rng: random.Random,
) -> float:
    node = tree.root
    path: list[_PuctEdge] = []

    while True:
        if _terminal_observation(node.observation) or not node.edges:
            value = _leaf_score(
                node.observation,
                root_player_index,
                evaluator=evaluator,
                score_fn=score_fn,
            )
            break

        edge = _select_puct_edge(
            node,
            root_player_index=root_player_index,
            exploration=exploration,
        )
        path.append(edge)
        if edge.child is None:
            value = _evaluate_new_puct_child(
                tree,
                node,
                edge,
                root_player_index=root_player_index,
                evaluator=evaluator,
                score_fn=score_fn,
                max_actions=max_actions,
                rng=rng,
            )
            break
        node = edge.child

    for edge in path:
        edge.visits += 1
        edge.total_value += value
    return value


def _evaluate_new_puct_child(
    tree: _PuctTree,
    parent: _PuctNode,
    edge: _PuctEdge,
    *,
    root_player_index: int,
    evaluator: PolicyValueEvaluator | None,
    score_fn: RolloutScoreFn,
    max_actions: int,
    rng: random.Random,
) -> float:
    state = tree.session.step(parent.search_id, edge.action)
    if not tree.can_store_child():
        try:
            return _leaf_score(
                state.observation,
                root_player_index,
                evaluator=evaluator,
                score_fn=score_fn,
            )
        finally:
            tree.session.release(int(state.searchId))

    child = _PuctNode.from_search_state(state)
    tree.register_child()
    edge.child = child
    _expand_puct_node(
        child,
        evaluator=evaluator,
        max_actions=max_actions,
        rng=rng,
    )
    return _leaf_score(
        child.observation,
        root_player_index,
        evaluator=evaluator,
        score_fn=score_fn,
    )


def _expand_puct_node(
    node: _PuctNode,
    *,
    evaluator: PolicyValueEvaluator | None,
    max_actions: int,
    rng: random.Random,
) -> None:
    if node.expanded:
        return
    node.expanded = True
    if _terminal_observation(node.observation):
        return
    actions = enumerate_select_actions(
        node.observation.select,
        max_actions=max_actions,
        rng=rng,
    )
    priors = _normalized_priors(
        _action_priors(evaluator, node.observation, actions),
        actions,
    )
    node.edges = [
        _PuctEdge(action=action, prior=priors.get(action, 0.0))
        for action in actions
    ]


def _select_puct_edge(
    node: _PuctNode,
    *,
    root_player_index: int,
    exploration: float,
) -> _PuctEdge:
    parent_visits = max(1, sum(edge.visits for edge in node.edges))
    parent_sqrt = float(parent_visits) ** 0.5
    direction = 1.0 if node.player_index == root_player_index else -1.0
    return max(
        node.edges,
        key=lambda edge: (
            direction * edge.mean_value
            + exploration * edge.prior * parent_sqrt / float(edge.visits + 1),
            edge.prior,
            -len(edge.action),
        ),
    )


def _merge_puct_root_stats(
    stats: dict[tuple[int, ...], RootActionStats],
    root: _PuctNode,
    *,
    source: str,
) -> None:
    for edge in root.edges:
        if edge.visits <= 0:
            continue
        action_stats = stats.setdefault(edge.action, RootActionStats(action=edge.action))
        action_stats.visits += edge.visits
        action_stats.total_score += edge.total_value
        action_stats.total_prior += edge.prior * float(edge.visits)
        action_stats.sources[source] = action_stats.sources.get(source, 0) + edge.visits


def _normalized_priors(
    priors: Mapping[tuple[int, ...], float],
    actions: Sequence[tuple[int, ...]],
) -> Mapping[tuple[int, ...], float]:
    if not actions:
        return {}
    nonnegative = {
        action: max(0.0, float(priors.get(action, 0.0)))
        for action in actions
    }
    total = sum(nonnegative.values())
    if total <= 0.0:
        uniform = 1.0 / float(len(actions))
        return dict.fromkeys(actions, uniform)
    return {action: value / total for action, value in nonnegative.items()}


def _release_puct_children(session: SearchSession, node: _PuctNode) -> None:
    for edge in node.edges:
        if edge.child is None:
            continue
        _release_puct_children(session, edge.child)
        session.release(edge.child.search_id)


def _terminal_observation(observation: ObservationLike) -> bool:
    state = observation.current
    return state is not None and int(state.result) != -1


def _player_index(observation: ObservationLike) -> int:
    state = observation.current
    if state is None:
        return 0
    player_index = int(state.yourIndex)
    return player_index if player_index in (0, 1) else 0


def _deadline(wall_time_seconds: float | None) -> float | None:
    if wall_time_seconds is None:
        return None
    return time.perf_counter() + wall_time_seconds


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _rollout_action(
    observation: ObservationLike,
    actions: Sequence[tuple[int, ...]],
    *,
    evaluator: PolicyValueEvaluator | None,
    rng: random.Random,
) -> tuple[int, ...]:
    if evaluator is None:
        return rng.choice(tuple(actions))
    priors = _action_priors(evaluator, observation, actions)
    if not priors:
        return rng.choice(tuple(actions))
    return max(actions, key=lambda action: (priors.get(action, 0.0), -len(action)))


def _action_priors(
    evaluator: PolicyValueEvaluator | None,
    observation: ObservationLike,
    actions: Sequence[tuple[int, ...]],
) -> Mapping[tuple[int, ...], float]:
    if evaluator is None or not actions:
        return {}
    return evaluator.action_priors(observation, actions)
