"""Isolated engine-teacher computation and parent inference protocol."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import torch

from ptcg_rl.agent.search.complete_action_teacher import (
    CompleteActionTeacherTarget,
    make_hidden_world_session_factory,
    produce_complete_action_teacher_target,
)
from ptcg_rl.agent.search.context import SimulatedContext
from ptcg_rl.agent.search.continuation_types import ContinuationValueDecks
from ptcg_rl.agent.search.macro import MacroEndpoint
from ptcg_rl.belief.observation import extract_observation_evidence
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.belief.state import Determinization
from ptcg_rl.context import (
    OpponentBeliefFeatureProducer,
    opponent_belief_state_from_evidence,
)
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.engine.runtime import load_cg_api
from ptcg_rl.engine.search_evidence import (
    SearchCandidateEvidence,
    SearchEvidence,
    search_candidate_features,
)
from ptcg_rl.model.state_encoder import StateBatch
from ptcg_rl.rl.bounded_worker import WorkerBroker
from ptcg_rl.rl.engine_teacher import EngineTeacherRequest
from ptcg_rl.rl.engine_teacher_policy import DecodePolicy, EngineTeacherPolicyAdapter
from ptcg_rl.rl.online_engine_teacher_config import OnlineEngineTeacherConfig


class TeacherDeadlineExpiredError(RuntimeError):
    """Cooperative worker deadline expiration before native hard timeout."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class CompactTeacherResult:
    """Only evidence consumed by the parent scheduler and trajectory target."""

    behavior_action: tuple[int, ...]
    target_action: tuple[int, ...]
    valid: bool
    reason: str
    confidence: float
    coverage: float
    worlds: int
    nodes_expanded: int
    hit_deadline: bool
    search_evidence: SearchEvidence | None

    @classmethod
    def from_target(
        cls,
        target: CompleteActionTeacherTarget,
    ) -> CompactTeacherResult:
        """Drop raw plans and leaves while retaining detached aggregates."""
        return cls(
            behavior_action=target.behavior_action,
            target_action=target.target_action,
            valid=target.valid,
            reason=target.reason,
            confidence=target.confidence,
            coverage=target.coverage,
            worlds=target.worlds,
            nodes_expanded=target.nodes_expanded,
            hit_deadline=any(
                "deadline" in world_plan.plan.stop_reason for world_plan in target.plans
            ),
            search_evidence=search_evidence_from_complete_action_target(target),
        )


def search_evidence_from_complete_action_target(
    target: CompleteActionTeacherTarget,
) -> SearchEvidence | None:
    """Detach a complete paired grid into public candidate-level aggregates.

    Any incomplete or internally inconsistent action/world grid falls back to
    no evidence.  This prevents partial deadline-biased searches from becoming
    model input while keeping the producer's raw tree and hidden worlds out of
    IPC and trajectories. Endpoint and path-length fields summarize the
    backed-up representative continuation in each world; they are structural
    summaries, not chance-weighted branch occupancy statistics.
    """
    try:
        return _search_evidence_from_complete_action_target(target)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _search_evidence_from_complete_action_target(
    target: CompleteActionTeacherTarget,
) -> SearchEvidence | None:
    if not target.valid or target.worlds <= 0:
        return None
    candidates = tuple(
        tuple(int(index) for index in action)
        for action in target.root_candidates.actions
    )
    if not candidates or target.target_action not in candidates:
        return None
    if len(set(candidates)) != len(candidates):
        return None
    if len(target.action_scores) != len(candidates):
        return None
    legal_action_count = int(target.root_candidates.legal_action_count)
    if legal_action_count < len(candidates):
        return None
    root_coverage = float(len(candidates)) / float(legal_action_count)

    scores_by_action = {score.root_action: score for score in target.action_scores}
    if len(scores_by_action) != len(candidates) or set(scores_by_action) != set(
        candidates
    ):
        return None

    plan_index = {
        (item.root_action, item.world_index): item.plan for item in target.plans
    }
    expected_plan_keys = {
        (action, world_index)
        for action in candidates
        for world_index in range(target.worlds)
    }
    if len(target.plans) != len(expected_plan_keys):
        return None
    if set(plan_index) != expected_plan_keys:
        return None
    if any(not plan.valid for plan in plan_index.values()):
        return None

    leaf_index = {
        (leaf.root_action, leaf.world_index, leaf.action_path): leaf
        for leaf in target.leaf_scores
    }
    if len(leaf_index) != len(target.leaf_scores):
        return None

    exact = bool(
        target.root_candidates.exhaustive
        and all(plan.exact for plan in plan_index.values())
    )
    rows: list[SearchCandidateEvidence] = []
    for action in candidates:
        score = scores_by_action[action]
        if not score.strategy_consistent:
            return None
        if len(score.world_scores) != target.worlds:
            return None
        if len(score.continuation_paths) != target.worlds:
            return None
        if any(not math.isfinite(float(value)) for value in score.world_scores):
            return None
        if not math.isclose(
            float(score.mean_score),
            float(statistics.fmean(score.world_scores)),
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            return None
        if not math.isclose(
            float(score.score_std),
            float(statistics.pstdev(score.world_scores)),
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            return None

        selected_leaves = []
        for world_index, path in enumerate(score.continuation_paths):
            leaf = leaf_index.get((action, world_index, path))
            if leaf is None:
                return None
            selected_leaves.append(leaf)
        endpoints = tuple(leaf.endpoint for leaf in selected_leaves)
        allowed_endpoints = {
            MacroEndpoint.TERMINAL,
            MacroEndpoint.SAME_SEAT_MAIN,
            MacroEndpoint.TURN_HANDOFF,
        }
        if any(endpoint not in allowed_endpoints for endpoint in endpoints):
            return None
        world_count = float(target.worlds)
        features = search_candidate_features(
            world_scores=score.world_scores,
            robust_score=score.robust_score,
            # This field has one producer-independent meaning: retained root
            # candidates divided by the full legal root action count. The
            # target confidence separately retains total root/continuation/
            # leaf coverage and must not be copied into model evidence.
            coverage=root_coverage,
            terminal_fraction=endpoints.count(MacroEndpoint.TERMINAL) / world_count,
            same_seat_main_fraction=(
                endpoints.count(MacroEndpoint.SAME_SEAT_MAIN) / world_count
            ),
            turn_handoff_fraction=(
                endpoints.count(MacroEndpoint.TURN_HANDOFF) / world_count
            ),
            mean_path_steps=statistics.fmean(
                len(path) for path in score.continuation_paths
            ),
            exact=exact,
        )
        rows.append(SearchCandidateEvidence(action=action, features=features))

    return SearchEvidence(
        candidates=tuple(rows),
        legal_action_count=legal_action_count,
        world_count=target.worlds,
        exhaustive=bool(target.root_candidates.exhaustive),
        exact=exact,
    )


@dataclass(frozen=True)
class WorkerInit:
    config: OnlineEngineTeacherConfig
    belief_producer: OpponentBeliefFeatureProducer | None


@dataclass(frozen=True)
class WorkerState:
    config: OnlineEngineTeacherConfig
    belief_producer: OpponentBeliefFeatureProducer | None
    sampler: BeliefSampler


@dataclass(frozen=True)
class WorkerTask:
    request: EngineTeacherRequest
    decision_seed: int
    planner_deadline: float


@dataclass(frozen=True)
class DecodeCall:
    states: Any
    options: Any
    decks: Any
    temperature: float
    deadline: float


@dataclass(frozen=True)
class ValueCall:
    states: Any
    decks: Any
    deadline: float


@dataclass(frozen=True)
class DecodeReply:
    actions: tuple[tuple[int, ...], ...]
    logprobs: torch.Tensor
    values: torch.Tensor
    policy_version: int | None


@dataclass(frozen=True)
class ValueReply:
    values: torch.Tensor
    policy_version: int | None


class BrokerDecodePolicy:
    """Child-side DecodePolicy proxy backed by the parent inference broker."""

    def __init__(self, broker: WorkerBroker, *, deadline: float) -> None:
        self._broker = broker
        self._deadline = float(deadline)
        self.last_response_policy_version: int | None = None

    def sample_decode(
        self,
        states: Any,
        options: Any,
        decks: Any,
        *,
        temperature: float = 1.0,
    ) -> tuple[tuple[tuple[int, ...], ...], torch.Tensor, torch.Tensor]:
        reply = self._broker.call(
            DecodeCall(
                states=_wire_safe_states(states),
                options=options,
                decks=decks,
                temperature=float(temperature),
                deadline=self._deadline,
            )
        )
        if not isinstance(reply, DecodeReply):
            raise RuntimeError("isolated decode broker returned an invalid reply")
        self.last_response_policy_version = reply.policy_version
        return reply.actions, reply.logprobs, reply.values

    def predict_values(self, states: Any, decks: Any) -> torch.Tensor:
        reply = self._broker.call(
            ValueCall(
                states=_wire_safe_states(states),
                decks=decks,
                deadline=self._deadline,
            )
        )
        if not isinstance(reply, ValueReply):
            raise RuntimeError("isolated value broker returned an invalid reply")
        self.last_response_policy_version = reply.policy_version
        return reply.values


def _wire_safe_states(states: Any) -> Any:
    """Cast PyTorch uint16 fields that stdlib pickle cannot reconstruct."""
    if not isinstance(states, StateBatch):
        return states
    attachment_card_ids = states.attachment_card_ids
    attachment_parent_indices = states.attachment_parent_indices
    if attachment_card_ids is not None:
        attachment_card_ids = attachment_card_ids.to(dtype=torch.long)
    if attachment_parent_indices is not None:
        attachment_parent_indices = attachment_parent_indices.to(dtype=torch.long)
    return replace(
        states,
        attachment_card_ids=attachment_card_ids,
        attachment_parent_indices=attachment_parent_indices,
    )


def initialize_worker(payload: Any) -> WorkerState:
    """Load native code and immutable priors before accepting timed work."""
    if not isinstance(payload, WorkerInit):
        raise TypeError("isolated teacher received invalid initializer data")
    load_cg_api()
    return WorkerState(
        config=payload.config,
        belief_producer=payload.belief_producer,
        sampler=BeliefSampler(config=payload.config.sampler),
    )


def worker_produce_target(
    state: Any,
    payload: Any,
    broker: WorkerBroker,
    deadline: float,
) -> CompactTeacherResult:
    """Produce one target using child-native state and parent-owned inference."""
    if not isinstance(state, WorkerState) or not isinstance(payload, WorkerTask):
        raise TypeError("isolated teacher received invalid work data")
    planner_deadline = min(deadline, payload.planner_deadline)
    return CompactTeacherResult.from_target(
        produce_target_impl(
            payload.request,
            decision_seed=payload.decision_seed,
            deadline=planner_deadline,
            config=state.config,
            sampler=state.sampler,
            belief_producer=state.belief_producer,
            policy=cast(
                DecodePolicy,
                BrokerDecodePolicy(broker, deadline=planner_deadline),
            ),
            device=None,
        )
    )


def produce_target_impl(
    request: EngineTeacherRequest,
    *,
    decision_seed: int,
    deadline: float,
    config: OnlineEngineTeacherConfig,
    sampler: BeliefSampler,
    belief_producer: OpponentBeliefFeatureProducer | None,
    policy: DecodePolicy,
    device: torch.device | str | None,
) -> CompleteActionTeacherTarget:
    """Sample worlds and run complete engine action search."""
    if request.seat not in (0, 1):
        raise ValueError("engine teacher request has an invalid seat")
    if _field(request.observation, "search_begin_input") is None:
        raise ValueError("engine teacher requires search_begin_input")
    if time.perf_counter() >= deadline:
        raise TeacherDeadlineExpiredError("deadline_expired")

    evidence = extract_observation_evidence(request.observation)
    opponent_state = opponent_belief_state_from_evidence(
        evidence,
        request.context_features,
        context_snapshot=request.context_snapshot,
    )
    own_deck = canonicalize_deck(request.own_deck).card_ids
    sampled_world_seed = world_seed(decision_seed, request.behavior_action)
    determinizations = sample_worlds_impl(
        evidence=evidence,
        opponent_state=opponent_state,
        own_deck=own_deck,
        seed=sampled_world_seed,
        deadline=deadline,
        worlds=config.worlds,
        sampler=sampler,
    )
    hidden_worlds = tuple(item.hidden for item in determinizations)
    world_value_decks = tuple(
        ContinuationValueDecks.from_sequences(
            root_full_deck=own_deck,
            sampled_opponent_full_deck=_sampled_opponent_deck(item),
        )
        for item in determinizations
    )
    adapter = EngineTeacherPolicyAdapter(
        policy,
        own_deck=own_deck,
        device=device,
        value_batch_size=config.teacher.leaf_value_batch_size,
    )
    simulated_context = SimulatedContext.from_snapshot(
        request.context_snapshot,
        belief=belief_producer,
        counterparty_snapshot=request.counterparty_context_snapshot,
    )
    return produce_complete_action_teacher_target(
        request.observation,
        world_count=len(determinizations),
        open_session=make_hidden_world_session_factory(
            request.observation,
            hidden_worlds,
            manual_coin=config.manual_coin,
        ),
        behavior_action=request.behavior_action,
        context=simulated_context,
        root_player_index=request.seat,
        policy=adapter,
        leaf_values=adapter.values,
        world_value_decks=world_value_decks,
        deadline=deadline,
        planner_limits=config.planner,
        config=config.teacher,
    )


def sample_worlds_impl(
    *,
    evidence: Any,
    opponent_state: Any,
    own_deck: Sequence[int],
    seed: int,
    deadline: float,
    worlds: int,
    sampler: BeliefSampler,
) -> tuple[Determinization, ...]:
    """Draw IID determinizations while retaining duplicate posterior mass."""
    rng = random.Random(seed)
    samples: list[Determinization] = []
    for _world_index in range(worlds):
        if time.perf_counter() >= deadline:
            raise TeacherDeadlineExpiredError("deadline_expired")
        sample = sampler.sample_from_evidence(
            evidence,
            your_deck=own_deck,
            opponent_state=opponent_state,
            rng=rng,
        )
        samples.append(sample)
    return tuple(samples)


def policy_response_version(policy: DecodePolicy) -> int | None:
    """Read the version attached to the most recent parent inference reply."""
    version = getattr(policy, "last_response_policy_version", None)
    if version is None:
        version = getattr(policy, "policy_version", None)
    return None if version is None else int(version)


def world_seed(decision_seed: int, behavior_action: Sequence[int]) -> int:
    payload = ",".join(str(int(index)) for index in behavior_action)
    encoded = f"worlds|{payload}|{decision_seed}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _sampled_opponent_deck(sample: Determinization) -> tuple[int, ...]:
    return canonicalize_deck(sample.opponent_deck_counts.elements()).card_ids


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = [
    "CompactTeacherResult",
    "DecodeCall",
    "DecodeReply",
    "TeacherDeadlineExpiredError",
    "ValueCall",
    "ValueReply",
    "WorkerInit",
    "WorkerTask",
    "initialize_worker",
    "policy_response_version",
    "produce_target_impl",
    "sample_worlds_impl",
    "search_evidence_from_complete_action_target",
    "worker_produce_target",
    "world_seed",
]
