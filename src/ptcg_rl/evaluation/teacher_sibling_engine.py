"""Engine execution for one public-teacher sibling root group."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.agent.search.config import EngineTacticalScoreConfig
from ptcg_rl.agent.search.context import SimulatedContext
from ptcg_rl.agent.search.macro import (
    MacroEndpoint,
    MacroTransition,
    advance_same_turn_macro,
)
from ptcg_rl.agent.search.scoring import engine_transition_score
from ptcg_rl.belief.observation import extract_observation_evidence
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.belief.state import Determinization
from ptcg_rl.context import (
    GameContextSnapshot,
    OpponentBeliefFeatureProducer,
    context_features_from_observation,
    opponent_belief_state_from_evidence,
)
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.data.kaggle_steps.probe_sidecar import your_deck_from_step_row
from ptcg_rl.engine.session import HiddenInformation, SearchSession
from ptcg_rl.evaluation.teacher_sibling_config import TeacherSiblingConfig
from ptcg_rl.evaluation.teacher_sibling_metrics import (
    SiblingEvaluation,
    TeacherSiblingMetrics,
)
from ptcg_rl.evaluation.teacher_sibling_support import (
    FrozenCandidates,
    SampledTeacherRoot,
    build_frozen_candidates,
    deterministic_root_seed,
    game_context_from_step_row,
)
from ptcg_rl.training.bc_dataset import observation_from_step_row

RootGroup: TypeAlias = tuple[
    dict[str, Any],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]


@dataclass(frozen=True)
class _PendingEvaluation:
    """One engine transition awaiting paired checkpoint value batches."""

    world_index: int
    candidate_index: int
    action: tuple[int, ...]
    sources: tuple[str, ...]
    transition: MacroTransition | None
    error: str | None = None


def evaluate_teacher_root(
    sampled: SampledTeacherRoot,
    *,
    config: TeacherSiblingConfig,
    campaign_fp: str,
    initial_policy: CheckpointPolicy,
    trained_policy: CheckpointPolicy,
    belief_producer: OpponentBeliefFeatureProducer,
    sampler: BeliefSampler,
    metrics: TeacherSiblingMetrics,
) -> RootGroup | None:
    """Resolve every frozen candidate across paired belief worlds."""
    started_at = time.perf_counter()
    row = sampled.row
    initial_policy.clear_inference_cache()
    trained_policy.clear_inference_cache()
    observation = observation_from_step_row(row, belief_producer=belief_producer)
    root_player_index = int(
        _field(_field(observation, "current"), "yourIndex", 0)
    )
    teacher_action = step_action(row.get("action"))
    select = observation.get("select")
    if select is None or not is_legal_action(select, teacher_action):
        return None
    candidates = build_frozen_candidates(
        observation,
        teacher_action=teacher_action,
        current_policy=trained_policy,
        top_k=config.policy_top_k,
    )
    if teacher_action not in candidates.actions or len(candidates.actions) < 2:
        return None
    illegal_candidates = sum(
        not is_legal_action(select, action) for action in candidates.actions
    )
    your_deck = your_deck_from_step_row(row)
    context = game_context_from_step_row(row, observation, your_deck=your_deck)
    context_features = context_features_from_observation(observation)
    evidence = extract_observation_evidence(observation)
    opponent_state = opponent_belief_state_from_evidence(
        evidence,
        context_features,
    )
    distributions = trained_policy.belief_distributions(observation)
    opponent_card_probs = distributions[0] if distributions is not None else None
    opponent_hand_weights = distributions[1] if distributions is not None else None
    rng = random.Random(deterministic_root_seed(config.seed, sampled.root_id))
    determinizations = tuple(
        sampler.sample_from_evidence(
            evidence,
            your_deck=your_deck,
            opponent_state=opponent_state,
            opponent_card_probs=opponent_card_probs,
            opponent_hand_weights=opponent_hand_weights,
            rng=rng,
        )
        for _ in range(config.worlds)
    )
    world_rows = tuple(
        _world_row(campaign_fp, sampled.root_id, index, determinization)
        for index, determinization in enumerate(determinizations)
    )
    pending: list[_PendingEvaluation] = []
    state_pool_peak = 0
    state_leaks = 0
    for world_index, determinization in enumerate(determinizations):
        world_pending, world_peak, world_leaks = _evaluate_world(
            observation,
            candidates=candidates,
            hidden=determinization.hidden,
            world_index=world_index,
            context_snapshot=context.snapshot(),
            belief_producer=belief_producer,
            root_player_index=root_player_index,
            config=config,
        )
        pending.extend(world_pending)
        state_pool_peak = max(state_pool_peak, world_peak)
        state_leaks += world_leaks
    evaluation_rows, metric_rows = _attach_values(
        pending,
        campaign_fp=campaign_fp,
        root_id=sampled.root_id,
        behavior_kind=config.behavior_kind,
        root_player_index=root_player_index,
        initial_policy=initial_policy,
        trained_policy=trained_policy,
        tactical=config.tactical,
    )
    pair_rows = metrics.update(
        root_id=sampled.root_id,
        teacher_action=teacher_action,
        candidate_actions=candidates.actions,
        evaluations=metric_rows,
        worlds_requested=config.worlds,
        illegal_candidates=illegal_candidates,
        state_leaks=state_leaks,
        state_pool_peak=state_pool_peak,
        behavior_kind=config.behavior_kind,
    )
    root_row = {
        "campaign_fp": campaign_fp,
        "root_id": sampled.root_id,
        "source_shard": records.display_path(sampled.source_shard),
        "source_row_index": sampled.source_row_index,
        "episode_id": int(row.get("episode_id") or 0),
        "step_index": int(row.get("step_index") or 0),
        "player_index": int(row.get("player_index") or 0),
        "teacher_action": list(teacher_action),
        "current_greedy": list(candidates.current_greedy),
        "candidate_actions": [list(action) for action in candidates.actions],
        "candidate_sources": [list(sources) for sources in candidates.sources],
        "current_priors": list(candidates.priors),
        "worlds_requested": config.worlds,
        "candidate_count": len(candidates.actions),
        "illegal_candidates": illegal_candidates,
        "complete_coverage": (
            len(metric_rows) == config.worlds * len(candidates.actions)
        ),
        "state_pool_peak": state_pool_peak,
        "state_leaks": state_leaks,
        "elapsed_seconds": time.perf_counter() - started_at,
        "behavior_kind": config.behavior_kind,
        "ppo_ratio_eligible": False,
    }
    persisted_pairs = tuple(
        {"campaign_fp": campaign_fp, **pair_row} for pair_row in pair_rows
    )
    return root_row, world_rows, evaluation_rows, persisted_pairs


def step_action(value: Any) -> tuple[int, ...]:
    """Normalize a compact Parquet action list."""
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(int(index) for index in value)


def _evaluate_world(
    observation: Any,
    *,
    candidates: FrozenCandidates,
    hidden: HiddenInformation,
    world_index: int,
    context_snapshot: GameContextSnapshot,
    belief_producer: OpponentBeliefFeatureProducer,
    root_player_index: int,
    config: TeacherSiblingConfig,
) -> tuple[tuple[_PendingEvaluation, ...], int, int]:
    session: SearchSession | None = None
    pending: list[_PendingEvaluation] = []
    try:
        session = SearchSession.begin(
            observation,
            hidden,
            manual_coin=config.manual_coin,
        )
        with session:
            for candidate_index, (action, sources) in enumerate(
                zip(candidates.actions, candidates.sources, strict=True)
            ):
                transition = advance_same_turn_macro(
                    session,
                    root_action=action,
                    context=SimulatedContext.from_snapshot(
                        context_snapshot,
                        belief=belief_producer,
                    ),
                    root_player_index=root_player_index,
                    deadline=math.inf,
                    continuation_policy=None,
                    forced_step_cap=config.forced_step_cap,
                    continuation_step_cap=1,
                    node_cap=config.node_cap,
                )
                pending.append(
                    _PendingEvaluation(
                        world_index=world_index,
                        candidate_index=candidate_index,
                        action=action,
                        sources=sources,
                        transition=transition,
                        error=transition.error,
                    )
                )
        return tuple(pending), session.peak_live_state_count, session.live_state_count
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        existing = {row.action for row in pending}
        for candidate_index, (action, sources) in enumerate(
            zip(candidates.actions, candidates.sources, strict=True)
        ):
            if action not in existing:
                pending.append(
                    _PendingEvaluation(
                        world_index=world_index,
                        candidate_index=candidate_index,
                        action=action,
                        sources=sources,
                        transition=None,
                        error=error,
                    )
                )
        peak = session.peak_live_state_count if session is not None else 0
        leaks = session.live_state_count if session is not None else 0
        return tuple(pending), peak, leaks


def _attach_values(
    pending: Sequence[_PendingEvaluation],
    *,
    campaign_fp: str,
    root_id: str,
    behavior_kind: str,
    root_player_index: int,
    initial_policy: CheckpointPolicy,
    trained_policy: CheckpointPolicy,
    tactical: EngineTacticalScoreConfig,
) -> tuple[tuple[dict[str, Any], ...], tuple[SiblingEvaluation, ...]]:
    leaf_indices = [
        index
        for index, row in enumerate(pending)
        if row.error is None
        and row.transition is not None
        and row.transition.leaf_observation is not None
    ]
    leaf_observations = tuple(
        pending[index].transition.leaf_observation  # type: ignore[union-attr]
        for index in leaf_indices
    )
    initial_values = (
        initial_policy.values(leaf_observations, root_player_index)
        if leaf_observations
        else ()
    )
    trained_values = (
        trained_policy.values(leaf_observations, root_player_index)
        if leaf_observations
        else ()
    )
    initial_by_index = dict(zip(leaf_indices, initial_values, strict=True))
    trained_by_index = dict(zip(leaf_indices, trained_values, strict=True))
    persisted: list[dict[str, Any]] = []
    metric_rows: list[SiblingEvaluation] = []
    for index, row in enumerate(pending):
        transition = row.transition
        endpoint = (
            transition.endpoint if transition is not None else MacroEndpoint.ENGINE_ERROR
        )
        engine_score = (
            engine_transition_score(
                transition,
                root_player_index=root_player_index,
                config=tactical,
            )
            if transition is not None
            else None
        )
        leaf_available = transition is not None and transition.leaf_observation is not None
        error = row.error or (transition.error if transition is not None else None)
        initial_value = initial_by_index.get(index)
        trained_value = trained_by_index.get(index)
        metric_rows.append(
            SiblingEvaluation(
                action=row.action,
                world_index=row.world_index,
                endpoint=endpoint,
                engine_score=engine_score,
                initial_value=initial_value,
                trained_value=trained_value,
                leaf_available=leaf_available,
                error=error,
            )
        )
        persisted.append(
            {
                "campaign_fp": campaign_fp,
                "root_id": root_id,
                "world_index": row.world_index,
                "candidate_index": row.candidate_index,
                "action": list(row.action),
                "sources": list(row.sources),
                "endpoint": endpoint.value,
                "engine_score": engine_score,
                "initial_value": initial_value,
                "trained_value": trained_value,
                "steps": transition.steps if transition is not None else 0,
                "forced_steps": transition.forced_steps if transition is not None else 0,
                "stop_detail": (
                    transition.stop_detail
                    if transition is not None
                    else "session_exception"
                ),
                "leaf_available": leaf_available,
                "error": error,
                "behavior_kind": behavior_kind,
                "ppo_ratio_eligible": False,
            }
        )
    return tuple(persisted), tuple(metric_rows)


def _world_row(
    campaign_fp: str,
    root_id: str,
    world_index: int,
    determinization: Determinization,
) -> dict[str, Any]:
    hidden = determinization.hidden
    return {
        "campaign_fp": campaign_fp,
        "root_id": root_id,
        "world_index": world_index,
        "determinization_source": determinization.source,
        "archetype_signature": determinization.archetype_signature,
        "archetype_label": determinization.archetype_label,
        "your_deck": list(hidden.your_deck),
        "your_prize": list(hidden.your_prize),
        "opponent_deck": list(hidden.opponent_deck),
        "opponent_prize": list(hidden.opponent_prize),
        "opponent_hand": list(hidden.opponent_hand),
        "opponent_active": list(hidden.opponent_active),
    }


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = ["RootGroup", "evaluate_teacher_root", "step_action"]
