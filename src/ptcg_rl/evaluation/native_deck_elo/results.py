"""Compact native match result conversion for deck Elo campaigns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.continuous_league.native_match import NativeMatchResponse
from ptcg_rl.evaluation.native_deck_elo.models import RESULTS_FORMAT, ScheduledGame
from ptcg_rl.evaluation.native_deck_elo.storage import safe_rate


def result_row(
    game: ScheduledGame,
    response: NativeMatchResponse,
    *,
    campaign_fingerprint: str,
) -> dict[str, Any]:
    """Convert one native response without retaining its large action trace."""
    if response.outcome == "unresolved":
        deck_a_result = "unresolved"
    elif response.outcome == "draw":
        deck_a_result = "draw"
    else:
        seat_zero_won = response.outcome == "side_a_win"
        deck_a_won = seat_zero_won == (game.deck_a_seat == 0)
        deck_a_result = "win" if deck_a_won else "loss"
    telemetry = response.telemetry
    decision_counts = _int_pair(telemetry.get("decision_counts"))
    action_seconds = _float_pair(telemetry.get("action_seconds"))
    deck_a_decisions = decision_counts[game.deck_a_seat]
    deck_b_decisions = decision_counts[1 - game.deck_a_seat]
    deck_a_seconds = action_seconds[game.deck_a_seat]
    deck_b_seconds = action_seconds[1 - game.deck_a_seat]
    batch = _batch_telemetry(telemetry.get("actions"))
    inverse = _inverse_result(deck_a_result)
    return {
        "format": RESULTS_FORMAT,
        "campaign_fingerprint": campaign_fingerprint,
        "game_index": game.game_index,
        "match_id": game.match_id,
        "deck_a_id": game.deck_a.deck_digest,
        "deck_a_hash": game.deck_a.deck_hash,
        "deck_a_label": game.deck_a.label,
        "deck_a_signature": game.deck_a.deck_signature,
        "deck_a_source": records.display_path(game.deck_a.path),
        "deck_b_id": game.deck_b.deck_digest,
        "deck_b_hash": game.deck_b.deck_hash,
        "deck_b_label": game.deck_b.label,
        "deck_b_signature": game.deck_b.deck_signature,
        "deck_b_source": records.display_path(game.deck_b.path),
        "deck_a_seat": game.deck_a_seat,
        "deck_b_seat": 1 - game.deck_a_seat,
        "deck_a_result": deck_a_result,
        "deck_b_result": inverse,
        "deck_a_score": _score(deck_a_result),
        "deck_b_score": _score(inverse),
        "outcome": response.outcome,
        "terminal_reason": response.terminal_reason,
        "started_at": response.started_at,
        "finished_at": response.finished_at,
        "steps": response.steps,
        "duration_seconds": response.duration_seconds,
        "deck_a_decisions": deck_a_decisions,
        "deck_b_decisions": deck_b_decisions,
        "deck_a_action_seconds": deck_a_seconds,
        "deck_b_action_seconds": deck_b_seconds,
        "deck_a_mean_action_seconds": safe_rate(deck_a_seconds, deck_a_decisions),
        "deck_b_mean_action_seconds": safe_rate(deck_b_seconds, deck_b_decisions),
        **batch,
    }


def _batch_telemetry(raw_actions: Any) -> dict[str, int | float]:
    actions = raw_actions if isinstance(raw_actions, Sequence) else ()
    batch_sizes: list[int] = []
    queue_seconds = 0.0
    service_seconds = 0.0
    for raw_action in actions:
        if not isinstance(raw_action, Mapping):
            continue
        runtime = raw_action.get("runtime")
        if not isinstance(runtime, Mapping):
            continue
        batch_size = int(runtime.get("policy_batch_size", 0) or 0)
        if batch_size <= 0:
            continue
        batch_sizes.append(batch_size)
        queue_seconds += float(runtime.get("policy_batch_queue_seconds", 0.0) or 0.0)
        service_seconds += float(
            runtime.get("policy_batch_service_seconds", 0.0) or 0.0
        )
    return {
        "policy_decisions": len(batch_sizes),
        "policy_batch_size_sum": sum(batch_sizes),
        "policy_batch_size_max": max(batch_sizes, default=0),
        "policy_batch_queue_seconds": queue_seconds,
        "policy_batch_service_seconds": service_seconds,
    }


def _int_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, Sequence) and len(value) == 2:
        return int(value[0]), int(value[1])
    return 0, 0


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
