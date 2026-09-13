"""Paired-world same-turn macro search for P0 shadow and P1 reranking."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

from ptcg_rl.agent.probe import core_option_candidates
from ptcg_rl.agent.search.candidates import CandidateSet, build_candidate_set
from ptcg_rl.agent.search.config import MacroSearchConfig
from ptcg_rl.agent.search.context import SimulatedContext
from ptcg_rl.agent.search.macro import (
    MacroEndpoint,
    MacroTransition,
    advance_same_turn_macro,
)
from ptcg_rl.agent.search.scoring import (
    PairedRerankDecision,
    PairedWorldScore,
    engine_transition_score,
    select_paired_action,
)
from ptcg_rl.belief.observation import extract_observation_evidence
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import (
    GameContextFeatures,
    GameContextSnapshot,
    OpponentBeliefFeatureProducer,
    opponent_belief_state_from_evidence,
)
from ptcg_rl.engine.session import HiddenInformation, SearchSession


class MacroSearchPolicy(Protocol):
    """Policy/value surface used by the same-turn search adapter."""

    def select_action(self, observation: Any) -> tuple[int, ...]:
        """Return one complete greedy action."""

    def rank_actions(
        self,
        observation: Any,
        *,
        top_k: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Return policy-led complete actions."""

    def value(self, observation: Any, root_player_index: int) -> float:
        """Return a context-complete root-perspective value."""


class BatchedMacroSearchPolicy(MacroSearchPolicy, Protocol):
    """Optional policy surface for one-forward leaf value batches."""

    def values(
        self,
        observations: Sequence[Any],
        root_player_index: int,
    ) -> tuple[float, ...]:
        """Return root-perspective values aligned with observations."""


class RootInformationMacroSearchPolicy(MacroSearchPolicy, Protocol):
    """Optional root-adapter surface for semantic leaf batches."""

    def root_information_values(
        self,
        observations: Sequence[Any],
        root_player_index: int,
        endpoints: Sequence[MacroEndpoint],
    ) -> tuple[float, ...]:
        """Return root-adapted values aligned with semantic leaves."""


@dataclass(frozen=True)
class MacroWorldEvaluation:
    """One action in one fixed determinized world."""

    action: tuple[int, ...]
    world_index: int
    endpoint: MacroEndpoint
    steps: int
    engine_score: float | None
    critic_value: float | None
    stop_detail: str
    transition: MacroTransition | None
    error: str | None = None


@dataclass(frozen=True)
class MacroSearchResult:
    """Paired search evidence plus a separately gated action recommendation."""

    candidates: CandidateSet
    evaluations: tuple[MacroWorldEvaluation, ...]
    decision: PairedRerankDecision
    worlds_requested: int
    worlds_sampled: int
    complete_coverage: bool
    stop_reason: str
    transitions: int
    engine_sessions: int
    state_pool_peak: int
    state_leaks: int
    same_seat_value_rows: int = 0
    handoff_value_rows: int = 0

    @property
    def worlds_completed(self) -> int:
        """Return worlds having a non-error evaluation for every action."""
        if not self.candidates.actions:
            return 0
        successful = {
            (row.world_index, row.action)
            for row in self.evaluations
            if row.endpoint not in {MacroEndpoint.DEADLINE, MacroEndpoint.ENGINE_ERROR}
        }
        return sum(
            all((world, action) in successful for action in self.candidates.actions)
            for world in range(self.worlds_sampled)
        )


class PairedMacroSearcher:
    """Sample worlds first, then evaluate the action/world grid fairly."""

    def __init__(
        self,
        *,
        policy: MacroSearchPolicy,
        sampler: BeliefSampler,
        your_deck: Sequence[int],
        context_snapshot: GameContextSnapshot,
        root_context_features: GameContextFeatures,
        belief_producer: OpponentBeliefFeatureProducer | None,
        config: MacroSearchConfig,
        rng: random.Random,
        opponent_card_probs: Sequence[float] | None = None,
        opponent_hand_weights: Sequence[float] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._policy = policy
        self._sampler = sampler
        self._your_deck = tuple(int(card_id) for card_id in your_deck)
        self._context_snapshot = context_snapshot
        self._root_context_features = root_context_features
        self._belief_producer = belief_producer
        self._config = config
        self._rng = rng
        self._opponent_card_probs = opponent_card_probs
        self._opponent_hand_weights = opponent_hand_weights
        self._clock = clock

    def run(
        self,
        observation: Any,
        *,
        greedy_action: Sequence[int],
        deadline: float,
    ) -> MacroSearchResult:
        """Run lifecycle-safe paired search and return an auditable recommendation."""
        normalized_greedy = tuple(int(index) for index in greedy_action)
        select = _field(observation, "select")
        ranked = self._policy.rank_actions(observation, top_k=self._config.top_k)
        candidates = build_candidate_set(
            select,
            greedy_action=greedy_action,
            policy_actions=ranked,
            structural_actions=core_option_candidates(select),
            exploration_actions=self._config.exploration_actions,
            rng=self._rng,
        )
        if not candidates.actions:
            return _empty_result(
                candidates,
                normalized_greedy,
                self._config.worlds,
                "no_candidates",
            )

        try:
            worlds = self._sample_worlds(observation)
        except Exception as exc:
            return _empty_result(
                candidates,
                normalized_greedy,
                self._config.worlds,
                f"belief_error:{type(exc).__name__}",
            )

        root_player_index = _int_field(
            _field(observation, "current"),
            "yourIndex",
            0,
        )
        evaluations: list[MacroWorldEvaluation] = []
        engine_sessions = 0
        state_pool_peak = 0
        state_leaks = 0
        transitions = 0
        work_deadline = max(
            self._clock(),
            deadline - self._config.budget.uninterruptible_guard_seconds,
        )
        jobs = _round_robin_jobs(
            len(worlds),
            candidates.actions,
        )
        value_batch_size = self._value_batch_size()
        for offset in range(0, len(jobs), value_batch_size):
            chunk_rows: list[MacroWorldEvaluation] = []
            for world_index, action in jobs[offset : offset + value_batch_size]:
                if self._clock() >= work_deadline:
                    break
                row, session_peak, session_leaks = self._evaluate_one(
                    observation,
                    hidden=worlds[world_index],
                    world_index=world_index,
                    action=action,
                    root_player_index=root_player_index,
                    deadline=work_deadline,
                    evaluate_critic=value_batch_size == 1,
                )
                chunk_rows.append(row)
                engine_sessions += 1
                state_pool_peak = max(state_pool_peak, session_peak)
                state_leaks += session_leaks
                if row.transition is not None:
                    transitions += row.transition.steps
            if value_batch_size > 1 and chunk_rows:
                chunk_rows = self._attach_batched_critic_values(
                    chunk_rows,
                    root_player_index=root_player_index,
                    deadline=work_deadline,
                )
            evaluations.extend(chunk_rows)
            if len(chunk_rows) < value_batch_size:
                break

        expected = len(worlds) * len(candidates.actions)
        complete = len(evaluations) == expected and all(
            row.endpoint
            not in {
                MacroEndpoint.DEADLINE,
                MacroEndpoint.STEP_CAP,
                MacroEndpoint.ENGINE_ERROR,
            }
            for row in evaluations
        )
        if complete:
            stop_reason = "complete"
        elif self._clock() >= work_deadline:
            stop_reason = "deadline"
        elif any(row.error is not None for row in evaluations):
            stop_reason = "engine_error"
        else:
            stop_reason = "incomplete_coverage"
        decision = select_paired_action(
            tuple(
                PairedWorldScore(
                    action=row.action,
                    world_index=row.world_index,
                    endpoint=row.endpoint,
                    engine_score=row.engine_score,
                    critic_value=row.critic_value,
                    error=row.error,
                )
                for row in evaluations
            ),
            actions=candidates.actions,
            greedy_action=normalized_greedy,
            worlds_requested=self._config.worlds,
            config=self._config.rerank,
        )
        return MacroSearchResult(
            candidates=candidates,
            evaluations=tuple(evaluations),
            decision=decision,
            worlds_requested=self._config.worlds,
            worlds_sampled=len(worlds),
            complete_coverage=complete,
            stop_reason=stop_reason,
            transitions=transitions,
            engine_sessions=engine_sessions,
            state_pool_peak=state_pool_peak,
            state_leaks=state_leaks,
            same_seat_value_rows=sum(
                row.endpoint is MacroEndpoint.SAME_SEAT_MAIN
                and row.critic_value is not None
                for row in evaluations
            ),
            handoff_value_rows=sum(
                row.endpoint is MacroEndpoint.TURN_HANDOFF
                and row.critic_value is not None
                for row in evaluations
            ),
        )

    def _sample_worlds(self, observation: Any) -> tuple[HiddenInformation, ...]:
        evidence = extract_observation_evidence(observation)
        opponent_state = opponent_belief_state_from_evidence(
            evidence,
            self._root_context_features,
        )
        return tuple(
            self._sampler.sample_from_evidence(
                evidence,
                your_deck=self._your_deck,
                opponent_state=opponent_state,
                opponent_card_probs=self._opponent_card_probs,
                opponent_hand_weights=self._opponent_hand_weights,
                rng=self._rng,
            ).hidden
            for _ in range(self._config.worlds)
        )

    def _evaluate_one(
        self,
        observation: Any,
        *,
        hidden: HiddenInformation,
        world_index: int,
        action: tuple[int, ...],
        root_player_index: int,
        deadline: float,
        evaluate_critic: bool = True,
    ) -> tuple[MacroWorldEvaluation, int, int]:
        session: SearchSession | None = None
        try:
            session = SearchSession.begin(
                observation,
                hidden,
                manual_coin=self._config.manual_coin,
            )
            with session:
                context = SimulatedContext.from_snapshot(
                    self._context_snapshot,
                    belief=self._belief_producer,
                )
                transition = advance_same_turn_macro(
                    session,
                    root_action=action,
                    context=context,
                    root_player_index=root_player_index,
                    deadline=deadline,
                    continuation_policy=self._policy.select_action,
                    forced_step_cap=self._config.forced_step_cap,
                    continuation_step_cap=self._config.continuation_step_cap,
                    node_cap=self._config.node_cap,
                    clock=self._clock,
                )
                engine_score = engine_transition_score(
                    transition,
                    root_player_index=root_player_index,
                    config=self._config.rerank.tactical,
                )
                critic_value = (
                    None
                    if not evaluate_critic
                    or self._config.rerank.score_mode == "engine_only"
                    else self._critic_value(
                        transition,
                        root_player_index,
                        deadline=deadline,
                    )
                )
            return (
                MacroWorldEvaluation(
                    action=action,
                    world_index=world_index,
                    endpoint=transition.endpoint,
                    steps=transition.steps,
                    engine_score=engine_score,
                    critic_value=critic_value,
                    stop_detail=transition.stop_detail,
                    transition=transition,
                    error=transition.error,
                ),
                session.peak_live_state_count,
                session.live_state_count,
            )
        except Exception as exc:
            peak = session.peak_live_state_count if session is not None else 0
            leaks = session.live_state_count if session is not None else 0
            return (
                MacroWorldEvaluation(
                    action=action,
                    world_index=world_index,
                    endpoint=MacroEndpoint.ENGINE_ERROR,
                    steps=0,
                    engine_score=None,
                    critic_value=None,
                    stop_detail="session_exception",
                    transition=None,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                peak,
                leaks,
            )

    def _value_batch_size(self) -> int:
        values = getattr(self._policy, "values", None)
        adapter_values = getattr(self._policy, "root_information_values", None)
        if self._config.rerank.score_mode == "engine_only":
            return 1
        if self._config.rerank.handoff_score_mode == "root_value_adapter":
            return self._config.leaf_value_batch_size if callable(adapter_values) else 1
        if not callable(values):
            return 1
        return self._config.leaf_value_batch_size

    def _attach_batched_critic_values(
        self,
        rows: list[MacroWorldEvaluation],
        *,
        root_player_index: int,
        deadline: float,
    ) -> list[MacroWorldEvaluation]:
        eligible_endpoints = {MacroEndpoint.SAME_SEAT_MAIN}
        if self._config.rerank.handoff_score_mode == "root_value_adapter":
            eligible_endpoints.add(MacroEndpoint.TURN_HANDOFF)
        eligible_indices = [
            index
            for index, row in enumerate(rows)
            if row.transition is not None
            and row.transition.leaf_observation is not None
            and row.transition.endpoint in eligible_endpoints
        ]
        if not eligible_indices or (
            self._clock() + self._config.budget.uninterruptible_guard_seconds
            >= deadline
        ):
            return rows
        observations = tuple(
            rows[index].transition.leaf_observation  # type: ignore[union-attr]
            for index in eligible_indices
        )
        try:
            if self._config.rerank.handoff_score_mode == "root_value_adapter":
                endpoints = tuple(rows[index].endpoint for index in eligible_indices)
                root_values = cast(
                    RootInformationMacroSearchPolicy,
                    self._policy,
                ).root_information_values
                critic_values = tuple(
                    root_values(observations, root_player_index, endpoints)
                )
            else:
                values = cast(BatchedMacroSearchPolicy, self._policy).values
                critic_values = tuple(values(observations, root_player_index))
            if len(critic_values) != len(eligible_indices):
                raise ValueError("batched policy returned the wrong value count")
        except Exception as exc:
            error = f"batch_value_error:{type(exc).__name__}: {exc}"
            for index in eligible_indices:
                rows[index] = replace(
                    rows[index],
                    endpoint=MacroEndpoint.ENGINE_ERROR,
                    error=error,
                )
            return rows
        for index, critic_value in zip(
            eligible_indices,
            critic_values,
            strict=True,
        ):
            rows[index] = replace(rows[index], critic_value=float(critic_value))
        return rows

    def _critic_value(
        self,
        transition: MacroTransition,
        root_player_index: int,
        *,
        deadline: float,
    ) -> float | None:
        if transition.leaf_observation is None:
            return None
        eligible_endpoints = {MacroEndpoint.SAME_SEAT_MAIN}
        if self._config.rerank.handoff_score_mode == "root_value_adapter":
            eligible_endpoints.add(MacroEndpoint.TURN_HANDOFF)
        if transition.endpoint not in eligible_endpoints:
            return None
        if (
            self._clock() + self._config.budget.uninterruptible_guard_seconds
            >= deadline
        ):
            return None
        if self._config.rerank.handoff_score_mode == "root_value_adapter":
            root_values = cast(
                RootInformationMacroSearchPolicy,
                self._policy,
            ).root_information_values
            return float(
                root_values(
                    (transition.leaf_observation,),
                    root_player_index,
                    (transition.endpoint,),
                )[0]
            )
        return float(self._policy.value(transition.leaf_observation, root_player_index))


def _round_robin_jobs(
    world_count: int,
    actions: Sequence[tuple[int, ...]],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return a Latin-square order so neither a world nor action monopolizes time."""
    if world_count <= 0 or not actions:
        return ()
    return tuple(
        (world_index, actions[(world_index + round_index) % len(actions)])
        for round_index in range(len(actions))
        for world_index in range(world_count)
    )


def _empty_result(
    candidates: CandidateSet,
    greedy_action: tuple[int, ...],
    worlds_requested: int,
    stop_reason: str,
) -> MacroSearchResult:
    return MacroSearchResult(
        candidates=candidates,
        evaluations=(),
        decision=PairedRerankDecision(
            greedy_action=greedy_action,
            selected_action=greedy_action,
            reason=stop_reason,
        ),
        worlds_requested=worlds_requested,
        worlds_sampled=0,
        complete_coverage=False,
        stop_reason=stop_reason,
        transitions=0,
        engine_sessions=0,
        state_pool_peak=0,
        state_leaks=0,
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    item = _field(value, name, default)
    return int(item) if item is not None else default
