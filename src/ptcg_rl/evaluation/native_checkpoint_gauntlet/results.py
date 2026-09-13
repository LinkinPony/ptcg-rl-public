"""Compact result conversion for cross-checkpoint native matches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.continuous_league.native_match import NativeMatchResponse
from ptcg_rl.evaluation.native_checkpoint_gauntlet.models import (
    RESULTS_FORMAT,
    ScheduledCrossCheckpointGame,
)
from ptcg_rl.evaluation.native_deck_elo.storage import safe_rate


def result_row(
    game: ScheduledCrossCheckpointGame,
    response: NativeMatchResponse,
    *,
    campaign_fingerprint: str,
    candidate_label: str,
    baseline_label: str,
) -> dict[str, Any]:
    """Convert one response without retaining its large action trace."""
    candidate_result = _candidate_result(game, response)
    baseline_result = _inverse_result(candidate_result)
    telemetry = response.telemetry
    decision_counts = _int_pair(telemetry.get("decision_counts"))
    action_seconds = _float_pair(telemetry.get("action_seconds"))
    agent_seeds = _int_pair(telemetry.get("agent_seeds"), default=-1)
    engine_seed = _optional_int(telemetry.get("engine_seed"))
    first_player = _optional_player(telemetry.get("first_player"))
    candidate_decisions = decision_counts[game.candidate_seat]
    baseline_decisions = decision_counts[1 - game.candidate_seat]
    candidate_seconds = action_seconds[game.candidate_seat]
    baseline_seconds = action_seconds[1 - game.candidate_seat]
    batch = _batch_telemetry(
        telemetry.get("actions"), candidate_seat=game.candidate_seat
    )
    return {
        "format": RESULTS_FORMAT,
        "campaign_fingerprint": campaign_fingerprint,
        "game_index": game.game_index,
        "match_id": game.match_id,
        "engine_seed": engine_seed,
        "candidate_agent_seed": agent_seeds[game.candidate_seat],
        "baseline_agent_seed": agent_seeds[1 - game.candidate_seat],
        "first_player": first_player,
        "candidate_went_first": (
            None if first_player is None else first_player == game.candidate_seat
        ),
        "candidate_checkpoint": candidate_label,
        "baseline_checkpoint": baseline_label,
        "candidate_deck_id": game.candidate_deck.deck_digest,
        "candidate_deck_hash": game.candidate_deck.deck_hash,
        "candidate_deck_label": game.candidate_deck.label,
        "candidate_deck_signature": game.candidate_deck.deck_signature,
        "candidate_deck_source": records.display_path(game.candidate_deck.path),
        "baseline_deck_id": game.baseline_deck.deck_digest,
        "baseline_deck_hash": game.baseline_deck.deck_hash,
        "baseline_deck_label": game.baseline_deck.label,
        "baseline_deck_signature": game.baseline_deck.deck_signature,
        "baseline_deck_source": records.display_path(game.baseline_deck.path),
        "candidate_seat": game.candidate_seat,
        "baseline_seat": 1 - game.candidate_seat,
        "candidate_result": candidate_result,
        "baseline_result": baseline_result,
        "candidate_score": _score(candidate_result),
        "baseline_score": _score(baseline_result),
        "outcome": response.outcome,
        "terminal_reason": response.terminal_reason,
        "started_at": response.started_at,
        "finished_at": response.finished_at,
        "steps": response.steps,
        "duration_seconds": response.duration_seconds,
        "candidate_decisions": candidate_decisions,
        "baseline_decisions": baseline_decisions,
        "candidate_action_seconds": candidate_seconds,
        "baseline_action_seconds": baseline_seconds,
        "candidate_mean_action_seconds": safe_rate(
            candidate_seconds, candidate_decisions
        ),
        "baseline_mean_action_seconds": safe_rate(baseline_seconds, baseline_decisions),
        **batch,
    }


def _candidate_result(
    game: ScheduledCrossCheckpointGame,
    response: NativeMatchResponse,
) -> str:
    if response.outcome == "unresolved":
        return "unresolved"
    if response.outcome == "draw":
        return "draw"
    seat_zero_won = response.outcome == "side_a_win"
    candidate_won = seat_zero_won == (game.candidate_seat == 0)
    return "win" if candidate_won else "loss"


def _batch_telemetry(
    raw_actions: Any,
    *,
    candidate_seat: int,
) -> dict[str, int | float]:
    actions = raw_actions if isinstance(raw_actions, Sequence) else ()
    values: dict[str, int | float] = {}
    for role, seat in (
        ("candidate", candidate_seat),
        ("baseline", 1 - candidate_seat),
    ):
        batch_sizes: list[int] = []
        queue_seconds = 0.0
        service_seconds = 0.0
        for raw_action in actions:
            if not isinstance(raw_action, Mapping) or raw_action.get("seat") != seat:
                continue
            runtime = raw_action.get("runtime")
            if not isinstance(runtime, Mapping):
                continue
            batch_size = int(runtime.get("policy_batch_size", 0) or 0)
            if batch_size <= 0:
                continue
            batch_sizes.append(batch_size)
            queue_seconds += float(
                runtime.get("policy_batch_queue_seconds", 0.0) or 0.0
            )
            service_seconds += float(
                runtime.get("policy_batch_service_seconds", 0.0) or 0.0
            )
        values[f"{role}_policy_decisions"] = len(batch_sizes)
        values[f"{role}_policy_batch_size_sum"] = sum(batch_sizes)
        values[f"{role}_policy_batch_size_max"] = max(batch_sizes, default=0)
        values[f"{role}_policy_batch_queue_seconds"] = queue_seconds
        values[f"{role}_policy_batch_service_seconds"] = service_seconds
    return values


def _int_pair(value: Any, *, default: int = 0) -> tuple[int, int]:
    if isinstance(value, Sequence) and len(value) == 2:
        return int(value[0]), int(value[1])
    return default, default


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_player(value: Any) -> int | None:
    parsed = _optional_int(value)
    return parsed if parsed in (0, 1) else None


def _float_pair(value: Any) -> tuple[float, float]:
    if isinstance(value, Sequence) and len(value) == 2:
        return float(value[0]), float(value[1])
    return 0.0, 0.0


def _inverse_result(result: str) -> str:
    if result == "win":
        return "loss"
    if result == "loss":
        return "win"
    return result


def _score(result: str) -> float:
    if result == "win":
        return 1.0
    if result == "loss":
        return 0.0
    if result == "draw":
        return 0.5
    return float("nan")


__all__ = ["result_row"]
