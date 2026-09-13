"""Aggregation and Markdown reporting for gauntlet runs."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.training.run_config import resolved_training_config_dump

if TYPE_CHECKING:
    from ptcg_rl.training.gauntlet import GauntletConfig


def gauntlet_matchup_rows(
    game_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Aggregate gauntlet game rows by candidate/opponent/deck matchup."""
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in game_rows:
        key = (
            str(row["candidate_agent"]),
            str(row["opponent_agent"]),
            str(row["candidate_deck_id"]),
            str(row["opponent_deck_id"]),
        )
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for rows in grouped.values():
        output.append(_summary_row(rows[0], rows, include_decks=True))
    output.sort(key=lambda row: (row["opponent_tier"], row["opponent_agent"]))
    return output


def gauntlet_tier_rows(game_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate gauntlet game rows by opponent tier."""
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in game_rows:
        grouped.setdefault(int(row["opponent_tier"]), []).append(row)
    output = [
        _summary_row(rows[0], rows, include_decks=False, tier=tier)
        for tier, rows in grouped.items()
    ]
    output.sort(key=lambda row: int(row["opponent_tier"]))
    return output


def gauntlet_summary(
    config: GauntletConfig,
    *,
    game_rows: Sequence[Mapping[str, Any]],
    matchup_rows: Sequence[Mapping[str, Any]],
    tier_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    games_path: Path,
    matchups_path: Path,
) -> dict[str, Any]:
    """Build the JSON summary for one gauntlet run."""
    overall = _summary_row(game_rows[0], game_rows, include_decks=False)
    overall["opponent_agent"] = "all"
    overall["opponent_tier"] = -1
    return {
        **overall,
        "matchups": len(matchup_rows),
        "candidate_mode": config.candidate.mode,
        "num_workers": config.num_workers,
        "run": config.run.model_dump(mode="json"),
        "output_dir": records.display_path(output_dir),
        "games_path": records.display_path(games_path),
        "matchups_path": records.display_path(matchups_path),
        "report_path": records.display_path(output_dir / "report.md"),
        "tiers": [dict(row) for row in tier_rows],
        "matchup_rows": [dict(row) for row in matchup_rows],
        "config": resolved_training_config_dump(
            config,
            task_name="arena",
            run=config.run,
            output_dir=config.output_dir,
        ),
    }


def render_gauntlet_report(summary: Mapping[str, Any]) -> str:
    """Render a compact Markdown report from a gauntlet summary."""
    run = cast(Mapping[str, Any], summary["run"])
    lines = [
        f"# Gauntlet Report: {run['version']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Games | {summary['games']} |",
        f"| Matchups | {summary['matchups']} |",
        f"| Candidate win rate | {_pct(float(summary['win_rate']))} |",
        f"| Candidate score rate | {_pct(float(summary['score_rate']))} |",
        f"| Score CI95 | {_pct(float(summary['score_ci95_low']))} - {_pct(float(summary['score_ci95_high']))} |",
        f"| Agent error games | {summary['agent_error_games']} |",
        f"| Candidate illegal actions | {summary['candidate_illegal_actions']} |",
        f"| Opponent illegal actions | {summary['opponent_illegal_actions']} |",
        "",
        "## By Tier",
        "",
        "| Tier | Games | Win Rate | Score Rate | CI95 | Errors |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cast(Sequence[Mapping[str, Any]], summary["tiers"]):
        lines.append(
            "| {tier} | {games} | {win_rate} | {score_rate} | {ci} | {errors} |".format(
                tier=row["opponent_tier"],
                games=row["games"],
                win_rate=_pct(float(row["win_rate"])),
                score_rate=_pct(float(row["score_rate"])),
                ci=(
                    f"{_pct(float(row['score_ci95_low']))} - "
                    f"{_pct(float(row['score_ci95_high']))}"
                ),
                errors=row["agent_error_games"],
            )
        )
    lines.extend(
        [
            "",
            "## By Matchup",
            "",
            "| Opponent | Tier | Candidate Deck | Opponent Deck | Games | Win Rate | Score Rate | Errors |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in cast(Sequence[Mapping[str, Any]], summary["matchup_rows"]):
        lines.append(
            "| {opponent} | {tier} | {candidate_deck} | {opponent_deck} | {games} | {win_rate} | {score_rate} | {errors} |".format(
                opponent=row["opponent_agent"],
                tier=row["opponent_tier"],
                candidate_deck=row["candidate_deck_label"],
                opponent_deck=row["opponent_deck_label"],
                games=row["games"],
                win_rate=_pct(float(row["win_rate"])),
                score_rate=_pct(float(row["score_rate"])),
                errors=row["agent_error_games"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _summary_row(
    first: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    include_decks: bool,
    tier: int | None = None,
) -> dict[str, Any]:
    games = len(rows)
    result_counts = Counter(str(row["candidate_result"]) for row in rows)
    score_rate = sum(_score(row["candidate_result"]) for row in rows) / float(games)
    ci_low, ci_high = _normal_ci(score_rate, games)
    candidate_decisions = sum(int(row["candidate_decisions"]) for row in rows)
    opponent_decisions = sum(int(row["opponent_decisions"]) for row in rows)
    output: dict[str, Any] = {
        "candidate_agent": str(first["candidate_agent"]),
        "opponent_agent": str(first.get("opponent_agent", "")),
        "opponent_tier": int(tier if tier is not None else first["opponent_tier"]),
        "games": games,
        "wins": result_counts.get("win", 0),
        "losses": result_counts.get("loss", 0),
        "draws": result_counts.get("draw", 0),
        "truncated": result_counts.get("truncated", 0),
        "agent_error_games": sum(
            1 for row in rows if row["terminal_reason"] == "agent_error"
        ),
        "win_rate": result_counts.get("win", 0) / float(games),
        "score_rate": score_rate,
        "score_ci95_low": ci_low,
        "score_ci95_high": ci_high,
        "candidate_mean_action_seconds": _safe_rate(
            sum(float(row["candidate_action_seconds"]) for row in rows),
            candidate_decisions,
        ),
        "opponent_mean_action_seconds": _safe_rate(
            sum(float(row["opponent_action_seconds"]) for row in rows),
            opponent_decisions,
        ),
        "candidate_illegal_actions": sum(
            int(row["candidate_illegal_actions"]) for row in rows
        ),
        "opponent_illegal_actions": sum(
            int(row["opponent_illegal_actions"]) for row in rows
        ),
        "candidate_overage_exceeded_games": sum(
            1 for row in rows if bool(row["candidate_overage_exceeded"])
        ),
    }
    if include_decks:
        output.update(
            {
                "candidate_deck_id": first["candidate_deck_id"],
                "candidate_deck_hash": first["candidate_deck_hash"],
                "candidate_deck_label": first["candidate_deck_label"],
                "opponent_deck_id": first["opponent_deck_id"],
                "opponent_deck_hash": first["opponent_deck_hash"],
                "opponent_deck_label": first["opponent_deck_label"],
            }
        )
    return output


def _score(result: Any) -> float:
    if result == "win":
        return 1.0
    if result == "draw":
        return 0.5
    return 0.0


def _normal_ci(rate: float, count: int) -> tuple[float, float]:
    if count <= 0:
        return (0.0, 0.0)
    radius = 1.96 * math.sqrt(max(0.0, rate * (1.0 - rate)) / float(count))
    return (max(0.0, rate - radius), min(1.0, rate + radius))


def _safe_rate(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / float(denominator)


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"
