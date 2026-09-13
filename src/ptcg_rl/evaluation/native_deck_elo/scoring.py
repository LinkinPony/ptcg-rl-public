"""Online Elo scoring and final artifact generation."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.native_deck_elo.models import FORMAT, NativeDeckEloConfig
from ptcg_rl.evaluation.native_deck_elo.storage import (
    resolve_path,
    safe_rate,
    write_json_atomic,
    write_part,
    write_text_atomic,
)


def write_final_artifacts(
    config: NativeDeckEloConfig,
    *,
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    campaign_fingerprint: str,
    runtime_fingerprint: str,
    belief_fingerprint: str,
    elapsed_seconds: float,
    resumed_games: int,
) -> dict[str, Any]:
    """Compact committed shards and publish Elo standings and summary."""
    ordered = sorted(rows, key=lambda row: int(row["game_index"]))
    standings = standings_rows(
        ordered,
        elo_initial=config.elo_initial,
        elo_k=config.elo_k,
    )
    matchups = matchup_rows(ordered)
    games_path = output_dir / "games.parquet"
    standings_path = output_dir / "standings.parquet"
    matchups_path = output_dir / "matchups.parquet"
    write_part(games_path, ordered, compression=config.compression)
    write_part(standings_path, standings, compression=config.compression)
    write_part(matchups_path, matchups, compression=config.compression)
    resolved = sum(1 for row in ordered if row["deck_a_result"] != "unresolved")
    policy_decisions = sum(int(row["policy_decisions"]) for row in ordered)
    batch_size_sum = sum(int(row["policy_batch_size_sum"]) for row in ordered)
    total_decisions = sum(
        int(row["deck_a_decisions"]) + int(row["deck_b_decisions"]) for row in ordered
    )
    total_action_seconds = sum(
        float(row["deck_a_action_seconds"]) + float(row["deck_b_action_seconds"])
        for row in ordered
    )
    reasons = Counter(str(row["terminal_reason"]) for row in ordered)
    root = records.repo_path(Path("."))
    summary: dict[str, Any] = {
        "format": FORMAT,
        "campaign_fingerprint": campaign_fingerprint,
        "checkpoint_path": records.display_path(
            resolve_path(config.checkpoint_path, root=root)
        ),
        "checkpoint_sha256": config.expected_checkpoint_sha256,
        "checkpoint_source_commit": config.checkpoint_source_commit,
        "runner_source_commit": config.runner_source_commit,
        "runtime_fingerprint": runtime_fingerprint,
        "belief_fingerprint": belief_fingerprint,
        "games": len(ordered),
        "resolved_games": resolved,
        "unresolved_games": len(ordered) - resolved,
        "decks": len(standings),
        "matchups": len(matchups),
        "terminal_reasons": dict(sorted(reasons.items())),
        "elo_initial": config.elo_initial,
        "elo_k": config.elo_k,
        "elo_semantics": "online_k32_game_index_order_unresolved_excluded",
        "concurrency": config.concurrency,
        "policy_batch_max_rows": config.policy_batch_max_rows,
        "policy_batch_wait_ms": config.policy_batch_wait_ms,
        "policy_decisions": policy_decisions,
        "effective_policy_batch_size": safe_rate(batch_size_sum, policy_decisions),
        "mean_action_seconds": safe_rate(total_action_seconds, total_decisions),
        "elapsed_seconds": elapsed_seconds,
        "session_games_per_second": safe_rate(
            len(ordered) - resumed_games, elapsed_seconds
        ),
        "resumed_games": resumed_games,
        "standings": standings,
        "games_path": records.display_path(games_path),
        "standings_path": records.display_path(standings_path),
        "matchups_path": records.display_path(matchups_path),
        "summary_path": records.display_path(output_dir / "summary.json"),
        "report_path": records.display_path(output_dir / "report.md"),
    }
    write_json_atomic(output_dir / "summary.json", summary)
    write_text_atomic(output_dir / "report.md", render_report(summary))
    return summary


def standings_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    elo_initial: float,
    elo_k: float,
) -> list[dict[str, Any]]:
    """Apply online Elo updates in caller-provided deterministic order."""
    stats: dict[str, dict[str, Any]] = {}
    ratings: dict[str, float] = {}
    for row in rows:
        for prefix in ("deck_a", "deck_b"):
            deck_id = str(row[f"{prefix}_id"])
            if deck_id not in stats:
                stats[deck_id] = _initial_stats(row, prefix)
                ratings[deck_id] = elo_initial
            stats[deck_id]["scheduled_games"] += 1
        if row["deck_a_result"] == "unresolved":
            stats[str(row["deck_a_id"])]["unresolved"] += 1
            stats[str(row["deck_b_id"])]["unresolved"] += 1
            continue
        deck_a_id = str(row["deck_a_id"])
        deck_b_id = str(row["deck_b_id"])
        score_a = float(row["deck_a_score"])
        expected_a = 1.0 / (
            1.0 + math.pow(10.0, (ratings[deck_b_id] - ratings[deck_a_id]) / 400.0)
        )
        ratings[deck_a_id] += elo_k * (score_a - expected_a)
        ratings[deck_b_id] += elo_k * ((1.0 - score_a) - (1.0 - expected_a))
        _observe(stats[deck_a_id], str(row["deck_a_result"]), score_a)
        _observe(stats[deck_b_id], str(row["deck_b_result"]), 1.0 - score_a)
    standings: list[dict[str, Any]] = []
    for deck_id, item in stats.items():
        games = int(item["games"])
        standings.append(
            {
                **{key: value for key, value in item.items() if key != "score_total"},
                "score_rate": safe_rate(float(item["score_total"]), games),
                "elo": ratings[deck_id],
                "elo_delta": ratings[deck_id] - elo_initial,
            }
        )
    standings.sort(
        key=lambda row: (
            -float(row["elo"]),
            -float(row["score_rate"]),
            str(row["deck_label"]),
        )
    )
    for rank, row in enumerate(standings, start=1):
        row["rank"] = rank
    return standings


def matchup_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate exact pair cells while retaining seat-balance evidence."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["deck_a_id"]), str(row["deck_b_id"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for values in grouped.values():
        first = values[0]
        resolved = [row for row in values if row["deck_a_result"] != "unresolved"]
        score = sum(float(row["deck_a_score"]) for row in resolved)
        counts = Counter(str(row["deck_a_result"]) for row in values)
        output.append(
            {
                "deck_a_id": first["deck_a_id"],
                "deck_a_hash": first["deck_a_hash"],
                "deck_a_label": first["deck_a_label"],
                "deck_b_id": first["deck_b_id"],
                "deck_b_hash": first["deck_b_hash"],
                "deck_b_label": first["deck_b_label"],
                "scheduled_games": len(values),
                "resolved_games": len(resolved),
                "deck_a_wins": counts.get("win", 0),
                "deck_b_wins": counts.get("loss", 0),
                "draws": counts.get("draw", 0),
                "unresolved": counts.get("unresolved", 0),
                "deck_a_score_rate": safe_rate(score, len(resolved)),
                "deck_b_score_rate": safe_rate(len(resolved) - score, len(resolved)),
                "deck_a_seat0_games": sum(
                    1 for row in values if int(row["deck_a_seat"]) == 0
                ),
                "deck_a_seat1_games": sum(
                    1 for row in values if int(row["deck_a_seat"]) == 1
                ),
            }
        )
    output.sort(key=lambda row: (str(row["deck_a_label"]), str(row["deck_b_label"])))
    return output


def render_report(summary: Mapping[str, Any]) -> str:
    """Render a compact human-readable campaign report."""
    lines = [
        "# Native Deck Elo Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Games | {summary['games']} |",
        f"| Resolved | {summary['resolved_games']} |",
        f"| Unresolved | {summary['unresolved_games']} |",
        f"| Throughput | {float(summary['session_games_per_second']):.3f} games/s |",
        f"| Effective policy batch | {float(summary['effective_policy_batch_size']):.2f} |",
        "",
        "## Standings",
        "",
        "| Rank | Deck | Deck hash | Elo | W-L-D | Score | Unresolved |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for raw_row in cast(Sequence[Mapping[str, Any]], summary["standings"]):
        lines.append(
            "| {rank} | {label} | {deck_hash} | {elo:.1f} | "
            "{wins}-{losses}-{draws} | {score:.2%} | {unresolved} |".format(
                rank=raw_row["rank"],
                label=raw_row["deck_label"],
                deck_hash=raw_row["deck_hash"],
                elo=float(raw_row["elo"]),
                wins=raw_row["wins"],
                losses=raw_row["losses"],
                draws=raw_row["draws"],
                score=float(raw_row["score_rate"]),
                unresolved=raw_row["unresolved"],
            )
        )
    lines.extend(
        [
            "",
            "Elo is the order-dependent online K-factor diagnostic in game-index "
            "order. Unresolved games are excluded from rating updates.",
            "",
        ]
    )
    return "\n".join(lines)


def _initial_stats(row: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "deck_id": str(row[f"{prefix}_id"]),
        "deck_hash": str(row[f"{prefix}_hash"]),
        "deck_label": str(row[f"{prefix}_label"]),
        "deck_signature": str(row[f"{prefix}_signature"]),
        "deck_source": str(row[f"{prefix}_source"]),
        "scheduled_games": 0,
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "unresolved": 0,
        "score_total": 0.0,
    }


def _observe(stats: dict[str, Any], result: str, score: float) -> None:
    stats["games"] += 1
    stats["score_total"] += score
    if result == "win":
        stats["wins"] += 1
    elif result == "loss":
        stats["losses"] += 1
    else:
        stats["draws"] += 1


__all__ = ["matchup_rows", "standings_rows", "write_final_artifacts"]
