"""Paired-world scoring and conservative P1 override decisions."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ptcg_rl.agent.search.config import (
    EngineTacticalScoreConfig,
    PairedRerankConfig,
)
from ptcg_rl.agent.search.macro import MacroEndpoint, MacroTransition


@dataclass(frozen=True)
class PairedWorldScore:
    """Comparable engine/value evidence for one action in one fixed world."""

    action: tuple[int, ...]
    world_index: int
    endpoint: MacroEndpoint
    engine_score: float | None
    critic_value: float | None
    error: str | None = None


@dataclass(frozen=True)
class PairedActionScore:
    """Paired deltas for one candidate relative to the saved greedy action."""

    action: tuple[int, ...]
    paired_worlds: int
    mean_delta: float
    std_delta: float
    robust_delta: float
    downside_cvar: float
    minimum_delta: float


@dataclass(frozen=True)
class PairedRerankDecision:
    """Auditable recommendation; callers still own legality and budget checks."""

    greedy_action: tuple[int, ...]
    selected_action: tuple[int, ...]
    reason: str
    action_scores: tuple[PairedActionScore, ...] = ()

    @property
    def action_changed(self) -> bool:
        """Return whether the recommendation differs from greedy."""
        return self.selected_action != self.greedy_action

    @property
    def selected_score(self) -> PairedActionScore | None:
        """Return diagnostics for the selected non-greedy action, if any."""
        return next(
            (
                score
                for score in self.action_scores
                if score.action == self.selected_action
            ),
            None,
        )


def select_paired_action(
    evaluations: Sequence[PairedWorldScore],
    *,
    actions: Sequence[Sequence[int]],
    greedy_action: Sequence[int],
    worlds_requested: int,
    config: PairedRerankConfig,
) -> PairedRerankDecision:
    """Select a risk-gated action using only complete paired-world evidence."""
    greedy = tuple(int(index) for index in greedy_action)
    normalized_actions = tuple(
        tuple(int(index) for index in action) for action in actions
    )
    if worlds_requested <= 0:
        return _keep_greedy(greedy, "no_worlds")
    if greedy not in normalized_actions:
        return _keep_greedy(greedy, "greedy_missing")

    indexed: dict[tuple[int, tuple[int, ...]], PairedWorldScore] = {}
    for world_score in evaluations:
        key = (int(world_score.world_index), tuple(world_score.action))
        if key in indexed:
            return _keep_greedy(greedy, "duplicate_evaluation")
        indexed[key] = world_score

    expected_worlds = range(worlds_requested)
    scores_by_action: dict[tuple[int, ...], tuple[float, ...]] = {}
    evaluations_by_action: dict[tuple[int, ...], tuple[PairedWorldScore, ...]] = {}
    for action in normalized_actions:
        action_scores: list[float] = []
        action_evaluations: list[PairedWorldScore] = []
        for world_index in expected_worlds:
            evaluation = indexed.get((world_index, action))
            if evaluation is None:
                return _keep_greedy(greedy, "incomplete_or_incomparable_coverage")
            score = combined_world_score(evaluation, config)
            if score is None:
                return _keep_greedy(greedy, "incomplete_or_incomparable_coverage")
            action_scores.append(score)
            action_evaluations.append(evaluation)
        scores_by_action[action] = tuple(action_scores)
        evaluations_by_action[action] = tuple(action_evaluations)

    greedy_scores = scores_by_action[greedy]
    greedy_evaluations = evaluations_by_action[greedy]
    candidate_scores: list[PairedActionScore] = []
    for action in normalized_actions:
        if action == greedy:
            continue
        if any(
            not _endpoints_comparable(candidate.endpoint, base.endpoint)
            for candidate, base in zip(
                evaluations_by_action[action],
                greedy_evaluations,
                strict=True,
            )
        ):
            return _keep_greedy(greedy, "incomplete_or_incomparable_coverage")
        deltas = tuple(
            candidate - base
            for candidate, base in zip(
                scores_by_action[action],
                greedy_scores,
                strict=True,
            )
        )
        mean_delta = statistics.fmean(deltas)
        std_delta = statistics.pstdev(deltas)
        downside_count = max(1, math.ceil(config.downside_fraction * len(deltas)))
        downside_cvar = statistics.fmean(sorted(deltas)[:downside_count])
        candidate_scores.append(
            PairedActionScore(
                action=action,
                paired_worlds=len(deltas),
                mean_delta=mean_delta,
                std_delta=std_delta,
                robust_delta=mean_delta - config.risk_std_weight * std_delta,
                downside_cvar=downside_cvar,
                minimum_delta=min(deltas),
            )
        )

    eligible = [
        score
        for score in candidate_scores
        if score.robust_delta > config.switch_margin
        and score.downside_cvar >= config.minimum_downside_delta
    ]
    if not eligible:
        return PairedRerankDecision(
            greedy_action=greedy,
            selected_action=greedy,
            reason="margin_or_downside_gate",
            action_scores=tuple(candidate_scores),
        )
    selected = max(
        eligible,
        key=lambda score: (
            score.robust_delta,
            score.downside_cvar,
            score.mean_delta,
        ),
    )
    return PairedRerankDecision(
        greedy_action=greedy,
        selected_action=selected.action,
        reason="paired_margin",
        action_scores=tuple(candidate_scores),
    )


def engine_transition_score(
    transition: MacroTransition,
    *,
    root_player_index: int,
    config: EngineTacticalScoreConfig | None = None,
) -> float | None:
    """Return a bounded score derived only from engine state and effect logs."""
    scoring = config or EngineTacticalScoreConfig()
    if transition.endpoint == MacroEndpoint.TERMINAL:
        return _terminal_score(transition.leaf_observation, root_player_index)
    if transition.endpoint not in {
        MacroEndpoint.SAME_SEAT_MAIN,
        MacroEndpoint.TURN_HANDOFF,
    }:
        return None

    opponent_index = 1 - root_player_index
    damage_delta = 0
    ko_delta = 0
    prize_delta = 0
    draw_delta = 0
    energy_delta = 0
    status_delta = 0
    for summary in transition.summaries:
        for amount in summary.damage_by_target:
            if amount.target.player_index == opponent_index:
                damage_delta += amount.amount
            elif amount.target.player_index == root_player_index:
                damage_delta -= amount.amount
        for knockout in summary.knockouts:
            if knockout.target.player_index == opponent_index:
                ko_delta += 1
            elif knockout.target.player_index == root_player_index:
                ko_delta -= 1
        prize_delta += (
            summary.prizes_taken_by_player[root_player_index]
            - summary.prizes_taken_by_player[opponent_index]
        )
        draw_delta += (
            summary.draws_by_player[root_player_index]
            - summary.draws_by_player[opponent_index]
        )
        energy_delta += (
            summary.energy_delta_by_player[root_player_index]
            - summary.energy_delta_by_player[opponent_index]
        )
        for status in summary.status_changes:
            direction = 1 if status.target.player_index == opponent_index else -1
            status_delta += -direction if status.recovered else direction

    score = (
        scoring.damage_weight * _signed_unit(damage_delta, scoring.damage_scale)
        + scoring.knockout_weight * ko_delta
        + scoring.prize_weight * prize_delta
        + scoring.draw_weight * _signed_unit(draw_delta, scoring.draw_scale)
        + scoring.energy_weight * _signed_unit(energy_delta, scoring.energy_scale)
        + scoring.status_weight * status_delta
    )
    return min(max(score, -scoring.score_clip), scoring.score_clip)


def combined_world_score(
    evaluation: PairedWorldScore | None,
    config: PairedRerankConfig,
) -> float | None:
    """Return the public aggregate score used by paired-action comparison."""
    if (
        evaluation is None
        or evaluation.error is not None
        or evaluation.endpoint
        not in {
            MacroEndpoint.TERMINAL,
            MacroEndpoint.SAME_SEAT_MAIN,
            MacroEndpoint.TURN_HANDOFF,
        }
        or evaluation.engine_score is None
        or not math.isfinite(evaluation.engine_score)
    ):
        return None
    engine_only_handoff = (
        evaluation.endpoint is MacroEndpoint.TURN_HANDOFF
        and config.handoff_score_mode == "engine_only"
    )
    if (
        evaluation.endpoint is MacroEndpoint.TERMINAL
        or engine_only_handoff
        or config.score_mode == "engine_only"
    ):
        combined = evaluation.engine_score
    else:
        if evaluation.critic_value is None or not math.isfinite(
            evaluation.critic_value
        ):
            return None
        if config.score_mode == "engine_value_tiebreak":
            combined = evaluation.engine_score + (
                config.value_tiebreak_weight * evaluation.critic_value
            )
        else:
            combined = evaluation.critic_value + (
                config.engine_tiebreak_weight * evaluation.engine_score
            )
    clip = config.combined_score_clip
    if clip is None:
        return combined
    return min(max(combined, -clip), clip)


def _endpoints_comparable(left: MacroEndpoint, right: MacroEndpoint) -> bool:
    if MacroEndpoint.TERMINAL in {left, right}:
        return True
    return left == right


def _terminal_score(observation: Any, root_player_index: int) -> float | None:
    result = _int_field(_field(observation, "current"), "result", -1)
    if result < 0:
        return None
    if result == root_player_index:
        return 1.0
    if result == 2:
        return 0.0
    return -1.0


def _signed_unit(value: int | float, denominator: float) -> float:
    return min(max(float(value) / denominator, -1.0), 1.0)


def _keep_greedy(greedy: tuple[int, ...], reason: str) -> PairedRerankDecision:
    return PairedRerankDecision(
        greedy_action=greedy,
        selected_action=greedy,
        reason=reason,
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    item = _field(value, name, default)
    return int(item) if item is not None else default
