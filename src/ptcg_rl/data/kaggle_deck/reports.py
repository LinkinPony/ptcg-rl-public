"""Report builders for Kaggle deck environment analysis."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ptcg_rl.data.kaggle_deck.records as records


@dataclass(frozen=True)
class PublicWebRanking:
    """One public-web EV ranking row."""

    name: str
    expected_win_rate: str
    overall_win_rate: str
    meta_share: str
    public_rank: str


def deck_summaries(
    side_rows: list[dict[str, Any]],
    card_meta: dict[int, records.CardMeta],
    *,
    min_games_for_win_rate: int,
) -> list[dict[str, Any]]:
    """Aggregate exact deck-signature summaries."""
    groups: dict[str, dict[str, Any]] = {}
    for row in side_rows:
        signature = row["deck_signature"]
        group = groups.setdefault(
            signature,
            _new_group(row["deck_label"], row["known_deck"], signature),
        )
        _add_result(group, row)
        group["team_names"].add(row["team_name"])
        group["dates"].add(row["date"])

    summaries = [
        _finalize_deck_group(group, card_meta, min_games_for_win_rate)
        for group in groups.values()
    ]
    summaries.sort(key=lambda row: (row["games"], row["win_rate"]), reverse=True)
    for rank, row in enumerate(summaries, start=1):
        row["rank"] = rank
    return summaries


def daily_deck_summaries(
    side_rows: list[dict[str, Any]],
    card_meta: dict[int, records.CardMeta],
    *,
    min_games_for_win_rate: int,
) -> list[dict[str, Any]]:
    """Aggregate exact deck signatures by date."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in side_rows:
        key = (row["date"], row["deck_signature"])
        group = groups.setdefault(
            key,
            _new_group(row["deck_label"], row["known_deck"], row["deck_signature"]),
        )
        group["date"] = row["date"]
        _add_result(group, row)
        group["team_names"].add(row["team_name"])
        group["dates"].add(row["date"])

    summaries = [
        _finalize_deck_group(group, card_meta, min_games_for_win_rate)
        for group in groups.values()
    ]
    summaries.sort(
        key=lambda row: (row["date"], row["games"], row["win_rate"]), reverse=True
    )
    return summaries


def team_deck_summaries(
    side_rows: list[dict[str, Any]],
    *,
    min_games_for_win_rate: int,
) -> list[dict[str, Any]]:
    """Aggregate team-by-deck observed results."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in side_rows:
        key = (row["team_name"], row["deck_signature"])
        group = groups.setdefault(
            key,
            _new_group(row["deck_label"], row["known_deck"], row["deck_signature"]),
        )
        group["team_name"] = row["team_name"]
        _add_result(group, row)
        group["dates"].add(row["date"])

    summaries: list[dict[str, Any]] = []
    for group in groups.values():
        result_games = group["wins"] + group["losses"] + group["draws"]
        summaries.append(
            {
                "team_name": group["team_name"],
                "deck_hash": records.signature_hash(group["deck_signature"]),
                "deck_label": group["deck_label"],
                "known_deck": group["known_deck"],
                "games": group["games"],
                "result_games": result_games,
                "wins": group["wins"],
                "losses": group["losses"],
                "draws": group["draws"],
                "other": group["other"],
                "win_rate": _rate(group["wins"], result_games),
                "min_sample_met": result_games >= min_games_for_win_rate,
                "first_date": min(group["dates"]) if group["dates"] else "",
                "last_date": max(group["dates"]) if group["dates"] else "",
                "deck_signature": group["deck_signature"],
            }
        )
    summaries.sort(key=lambda row: (row["games"], row["win_rate"]), reverse=True)
    return summaries


def matchup_matrix(
    side_rows: list[dict[str, Any]],
    *,
    min_games_for_win_rate: int,
) -> list[dict[str, Any]]:
    """Build a directional deck-vs-deck observed matchup matrix."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in side_rows:
        opponent_signature = row["opponent_deck_signature"]
        if not opponent_signature:
            continue
        key = (row["deck_signature"], opponent_signature)
        group = groups.setdefault(
            key,
            {
                "candidate_deck_hash": row["deck_hash"],
                "candidate_deck_label": row["deck_label"],
                "candidate_known_deck": row["known_deck"],
                "candidate_deck_signature": row["deck_signature"],
                "opponent_deck_hash": row["opponent_deck_hash"],
                "opponent_deck_label": row["opponent_deck_label"],
                "opponent_deck_signature": opponent_signature,
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "other": 0,
            },
        )
        group["games"] += 1
        if row["result"] == "win":
            group["wins"] += 1
        elif row["result"] == "loss":
            group["losses"] += 1
        elif row["result"] == "draw":
            group["draws"] += 1
        else:
            group["other"] += 1

    rows: list[dict[str, Any]] = []
    for group in groups.values():
        result_games = group["wins"] + group["losses"] + group["draws"]
        group["result_games"] = result_games
        group["win_rate"] = _rate(group["wins"], result_games)
        group["min_sample_met"] = result_games >= min_games_for_win_rate
        rows.append(group)
    rows.sort(key=lambda row: (row["games"], row["win_rate"]), reverse=True)
    return rows


def card_usage(
    side_rows: list[dict[str, Any]],
    card_meta: dict[int, records.CardMeta],
) -> list[dict[str, Any]]:
    """Aggregate card usage across observed player-games."""
    usage: dict[int, dict[str, Any]] = {}
    signatures_by_card: dict[int, set[str]] = defaultdict(set)
    for row in side_rows:
        counts = records.signature_counts(row["deck_signature"])
        for card_id, count in counts.items():
            meta = card_meta.get(card_id, records.missing_card(card_id))
            card = usage.setdefault(
                card_id,
                {
                    "card_id": card_id,
                    "card_name": meta.name,
                    "stage_or_type": meta.stage_or_type,
                    "rule": meta.rule,
                    "player_games": 0,
                    "total_copies": 0,
                },
            )
            card["player_games"] += 1
            card["total_copies"] += count
            signatures_by_card[card_id].add(row["deck_signature"])

    rows: list[dict[str, Any]] = []
    for card_id, row in usage.items():
        player_games = row["player_games"]
        row["unique_deck_signatures"] = len(signatures_by_card[card_id])
        row["avg_copies_when_present"] = _rate(row["total_copies"], player_games)
        rows.append(row)
    rows.sort(key=lambda row: (row["player_games"], row["total_copies"]), reverse=True)
    return rows


def public_web_comparison(
    deck_rows: list[dict[str, Any]],
    *,
    public_web_rankings_csv: Path | None,
    min_public_match_score: float,
    top_n_decks: int,
) -> list[dict[str, Any]]:
    """Compare Kaggle deck labels to existing public-web EV rows."""
    public_rows = (
        _load_public_web_rankings(records.repo_path(public_web_rankings_csv))
        if public_web_rankings_csv
        else []
    )
    comparisons: list[dict[str, Any]] = []
    for kaggle_rank, deck in enumerate(deck_rows[:top_n_decks], start=1):
        public_match, match_score = _best_public_match(deck["deck_label"], public_rows)
        if public_match is None or match_score < min_public_match_score:
            public_match = None
            match_score = 0.0
        comparisons.append(
            {
                "kaggle_rank": kaggle_rank,
                "kaggle_deck_hash": deck["deck_hash"],
                "kaggle_deck_label": deck["deck_label"],
                "kaggle_known_deck": deck["known_deck"],
                "kaggle_games": deck["games"],
                "kaggle_result_games": deck["result_games"],
                "kaggle_win_rate": deck["win_rate"],
                "kaggle_min_sample_met": deck["min_sample_met"],
                "public_match_name": public_match.name if public_match else "",
                "public_match_score": match_score,
                "public_expected_win_rate": (
                    public_match.expected_win_rate if public_match else ""
                ),
                "public_overall_win_rate": (
                    public_match.overall_win_rate if public_match else ""
                ),
                "public_meta_share": public_match.meta_share if public_match else "",
                "public_rank": public_match.public_rank if public_match else "",
            }
        )
    return comparisons


def write_outputs(
    output_dir: Path,
    *,
    side_rows: list[dict[str, Any]],
    deck_rows: list[dict[str, Any]],
    daily_rows: list[dict[str, Any]],
    team_rows: list[dict[str, Any]],
    matchup_rows: list[dict[str, Any]],
    card_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    write_side_observations: bool,
) -> None:
    """Write all report artifacts."""
    if write_side_observations:
        write_parquet(output_dir / "side_observations.parquet", side_rows)
    write_csv(output_dir / "deck_signature_summary.csv", deck_rows, deck_fields())
    write_csv(output_dir / "daily_deck_summary.csv", daily_rows, daily_fields())
    write_csv(output_dir / "team_deck_summary.csv", team_rows, team_fields())
    write_csv(output_dir / "matchup_matrix.csv", matchup_rows, matchup_fields())
    write_csv(output_dir / "card_usage.csv", card_rows, card_usage_fields())
    write_csv(
        output_dir / "public_web_comparison.csv",
        comparison_rows,
        public_comparison_fields(),
    )


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as Parquet using pyarrow."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write rows as a small human-readable CSV summary."""
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write stable pretty JSON."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def output_paths(output_dir: Path) -> dict[str, str]:
    """Return standard output paths for summary JSON."""
    return {
        "card_usage": records.display_path(output_dir / "card_usage.csv"),
        "daily_deck_summary": records.display_path(
            output_dir / "daily_deck_summary.csv"
        ),
        "deck_signature_summary": records.display_path(
            output_dir / "deck_signature_summary.csv",
        ),
        "matchup_matrix": records.display_path(output_dir / "matchup_matrix.csv"),
        "public_web_comparison": records.display_path(
            output_dir / "public_web_comparison.csv",
        ),
        "side_observations": records.display_path(
            output_dir / "side_observations.parquet"
        ),
        "team_deck_summary": records.display_path(output_dir / "team_deck_summary.csv"),
    }


def deck_fields() -> list[str]:
    """Return deck summary CSV fields."""
    return [
        "rank",
        "deck_hash",
        "deck_label",
        "known_deck",
        "games",
        "result_games",
        "wins",
        "losses",
        "draws",
        "other",
        "win_rate",
        "loss_rate",
        "draw_rate",
        "min_sample_met",
        "unique_team_names",
        "first_date",
        "last_date",
        "unique_card_ids",
        "total_cards",
        "pokemon_summary",
        "top_cards",
        "deck_signature",
    ]


def daily_fields() -> list[str]:
    """Return daily deck summary CSV fields."""
    return ["date", *deck_fields()]


def team_fields() -> list[str]:
    """Return team deck summary CSV fields."""
    return [
        "team_name",
        "deck_hash",
        "deck_label",
        "known_deck",
        "games",
        "result_games",
        "wins",
        "losses",
        "draws",
        "other",
        "win_rate",
        "min_sample_met",
        "first_date",
        "last_date",
        "deck_signature",
    ]


def matchup_fields() -> list[str]:
    """Return matchup matrix CSV fields."""
    return [
        "candidate_deck_hash",
        "candidate_deck_label",
        "candidate_known_deck",
        "opponent_deck_hash",
        "opponent_deck_label",
        "games",
        "result_games",
        "wins",
        "losses",
        "draws",
        "other",
        "win_rate",
        "min_sample_met",
        "candidate_deck_signature",
        "opponent_deck_signature",
    ]


def card_usage_fields() -> list[str]:
    """Return card usage CSV fields."""
    return [
        "card_id",
        "card_name",
        "stage_or_type",
        "rule",
        "player_games",
        "total_copies",
        "unique_deck_signatures",
        "avg_copies_when_present",
    ]


def public_comparison_fields() -> list[str]:
    """Return public-web comparison CSV fields."""
    return [
        "kaggle_rank",
        "kaggle_deck_hash",
        "kaggle_deck_label",
        "kaggle_known_deck",
        "kaggle_games",
        "kaggle_result_games",
        "kaggle_win_rate",
        "kaggle_min_sample_met",
        "public_match_name",
        "public_match_score",
        "public_expected_win_rate",
        "public_overall_win_rate",
        "public_meta_share",
        "public_rank",
    ]


def _load_public_web_rankings(path: Path) -> list[PublicWebRanking]:
    if not path.exists():
        return []
    rankings: list[PublicWebRanking] = []
    with path.open(encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            rankings.append(
                PublicWebRanking(
                    name=row.get("name", ""),
                    expected_win_rate=row.get("expected_win_rate", ""),
                    overall_win_rate=row.get("overall_win_rate", ""),
                    meta_share=row.get("meta_share", ""),
                    public_rank=row.get("rank_order", row.get("rank", "")),
                )
            )
    return rankings


def _best_public_match(
    deck_label: str,
    public_rows: list[PublicWebRanking],
) -> tuple[PublicWebRanking | None, float]:
    deck_tokens = _tokens(deck_label)
    best: PublicWebRanking | None = None
    best_score = 0.0
    for row in public_rows:
        row_tokens = _tokens(row.name)
        if not deck_tokens or not row_tokens:
            score = 0.0
        else:
            score = len(deck_tokens & row_tokens) / len(deck_tokens | row_tokens)
        if score > best_score:
            best = row
            best_score = score
    return best, best_score


def _tokens(value: str) -> set[str]:
    stopwords = {
        "ai",
        "deck",
        "ex",
        "kaggle",
        "public",
        "score",
        "the",
        "v0",
        "v1",
        "v2",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in stopwords and not token.isdigit()
    }


def _new_group(deck_label: str, known_deck: str, signature: str) -> dict[str, Any]:
    return {
        "deck_label": deck_label,
        "known_deck": known_deck,
        "deck_signature": signature,
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "other": 0,
        "team_names": set(),
        "dates": set(),
    }


def _add_result(group: dict[str, Any], row: dict[str, Any]) -> None:
    group["games"] += 1
    if row["result"] == "win":
        group["wins"] += 1
    elif row["result"] == "loss":
        group["losses"] += 1
    elif row["result"] == "draw":
        group["draws"] += 1
    else:
        group["other"] += 1


def _finalize_deck_group(
    group: dict[str, Any],
    card_meta: dict[int, records.CardMeta],
    min_games_for_win_rate: int,
) -> dict[str, Any]:
    result_games = group["wins"] + group["losses"] + group["draws"]
    counts = records.signature_counts(group["deck_signature"])
    row = {
        "rank": 0,
        "deck_hash": records.signature_hash(group["deck_signature"]),
        "deck_label": group["deck_label"],
        "known_deck": group["known_deck"],
        "games": group["games"],
        "result_games": result_games,
        "wins": group["wins"],
        "losses": group["losses"],
        "draws": group["draws"],
        "other": group["other"],
        "win_rate": _rate(group["wins"], result_games),
        "loss_rate": _rate(group["losses"], result_games),
        "draw_rate": _rate(group["draws"], result_games),
        "min_sample_met": result_games >= min_games_for_win_rate,
        "unique_team_names": len(group["team_names"]),
        "first_date": min(group["dates"]) if group["dates"] else "",
        "last_date": max(group["dates"]) if group["dates"] else "",
        "unique_card_ids": len(counts),
        "total_cards": sum(counts.values()),
        "pokemon_summary": records.pokemon_summary(counts, card_meta),
        "top_cards": records.top_cards(counts, card_meta, limit=12),
        "deck_signature": group["deck_signature"],
    }
    if "date" in group:
        row["date"] = group["date"]
    return row


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
