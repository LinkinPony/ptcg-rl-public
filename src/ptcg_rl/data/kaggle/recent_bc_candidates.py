"""Current-roster coverage and missing exact-deck evidence analysis."""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ptcg_rl.data.kaggle.recent_bc_artifacts import read_json, repo_path
from ptcg_rl.data.kaggle.recent_bc_models import RecentBCSelectionConfig
from ptcg_rl.decks.identity import parse_canonical_signature


def load_active_routes(
    resolved: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Load and cross-check active route metadata and exact signatures."""
    raw_routes = resolved.get("active_deck_routes")
    raw_model = resolved.get("resolved_model_config")
    if not isinstance(raw_routes, list) or not isinstance(raw_model, Mapping):
        raise ValueError("resolved run has no active exact roster")
    raw_exact = raw_model.get("exact_routes")
    if not isinstance(raw_exact, list):
        raise ValueError("resolved run has no exact registry")
    signatures: dict[str, str] = {}
    for raw in raw_exact:
        if not isinstance(raw, Mapping):
            continue
        digest = str(raw.get("deck_digest", ""))
        signature = str(raw.get("signature", ""))
        if parse_canonical_signature(signature).deck_digest != digest:
            raise ValueError(f"active exact signature changed: {digest}")
        signatures[digest] = signature
    routes: dict[str, dict[str, Any]] = {}
    for raw in raw_routes:
        if not isinstance(raw, Mapping):
            continue
        digest = str(raw.get("deck_digest", ""))
        if digest not in signatures:
            raise ValueError(f"active route is absent from exact registry: {digest}")
        routes[digest] = dict(raw)
    if not routes or set(routes) != set(signatures):
        raise ValueError("active route and exact registry coverage differ")
    return routes, signatures


def load_rank10_evidence(
    config: RecentBCSelectionConfig,
) -> dict[str, dict[str, Any]]:
    """Bind each scoring submission to its exact deck and rank evidence."""
    manifest = read_json(repo_path(config.rank10_collection_manifest_path))
    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("rank10 collection manifest has no sources")
    sources: dict[int, dict[str, Any]] = {}
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            continue
        submission_id = int(raw["submission_id"])
        sources[submission_id] = {
            "rank": int(raw["rank"]),
            "team_name": str(raw["team_name"]),
            "submission_id": submission_id,
        }
    evidence: dict[str, dict[str, Any]] = {}
    matched_submissions: set[int] = set()
    with repo_path(config.rank10_team_deck_summary_path).open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            submission_ids = {
                int(value)
                for value in str(row.get("submission_ids", "")).split("|")
                if value
            }
            matched = submission_ids.intersection(sources)
            if not matched:
                continue
            if len(matched) != 1:
                raise ValueError("rank10 deck row matches multiple scoring submissions")
            submission_id = next(iter(matched))
            matched_submissions.add(submission_id)
            signature = str(row["deck_signature"])
            digest = parse_canonical_signature(signature).deck_digest
            candidate = {
                **sources[submission_id],
                "deck_hash": str(row["deck_hash"]),
                "games": int(row["games"]),
                "score_rate": float(row["score_rate"]),
            }
            previous = evidence.get(digest)
            if previous is None or int(candidate["rank"]) < int(previous["rank"]):
                evidence[digest] = candidate
    if matched_submissions != set(sources):
        raise ValueError("rank10 scoring submissions lack exact-deck evidence")
    return evidence


def candidate_deck_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_rows: Sequence[Mapping[str, Any]],
    active_routes: Mapping[str, Mapping[str, Any]],
    active_signatures: Mapping[str, str],
    active_deck_hashes: Mapping[str, str],
    rank10: Mapping[str, Mapping[str, Any]],
    card_data_csv: Path,
    dates: tuple[str, ...],
    wilson_z: float,
) -> list[dict[str, Any]]:
    """Build evidence rows for every exact deck absent from the active roster."""
    card_meta = _load_card_meta(card_data_csv)
    pokemon_ids = {
        card_id
        for card_id, (_name, category) in card_meta.items()
        if category.endswith("Pokémon")
    }
    by_deck: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_deck.setdefault(str(row["deck_digest"]), []).append(row)
    qualified_by_deck: dict[str, list[Mapping[str, Any]]] = {}
    for row in group_rows:
        if bool(row["qualified"]):
            qualified_by_deck.setdefault(str(row["deck_digest"]), []).append(row)
    output: list[dict[str, Any]] = []
    for digest, deck_rows in sorted(by_deck.items()):
        if digest in active_signatures:
            continue
        output.append(
            _candidate_row(
                digest,
                deck_rows,
                qualified=qualified_by_deck.get(digest, []),
                active_routes=active_routes,
                active_signatures=active_signatures,
                active_deck_hashes=active_deck_hashes,
                rank10=rank10.get(digest),
                card_meta=card_meta,
                pokemon_ids=pokemon_ids,
                dates=dates,
                wilson_z=wilson_z,
            )
        )
    return output


def _candidate_row(
    digest: str,
    deck_rows: Sequence[Mapping[str, Any]],
    *,
    qualified: Sequence[Mapping[str, Any]],
    active_routes: Mapping[str, Mapping[str, Any]],
    active_signatures: Mapping[str, str],
    active_deck_hashes: Mapping[str, str],
    rank10: Mapping[str, Any] | None,
    card_meta: Mapping[int, tuple[str, str]],
    pokemon_ids: set[int],
    dates: tuple[str, ...],
    wilson_z: float,
) -> dict[str, Any]:
    signature = single_value(str(row["deck_signature"]) for row in deck_rows)
    deck_counts = Counter(parse_canonical_signature(signature).card_ids)
    nearest_digest = min(
        active_signatures,
        key=lambda item: (replacement_distance(signature, active_signatures[item]), item),
    )
    nearest_pokemon_digest = min(
        active_signatures,
        key=lambda item: (
            candidate_unmatched_cards(
                signature, active_signatures[item], allowed=pokemon_ids
            ),
            item,
        ),
    )
    wins = sum(str(row["result"]) == "win" for row in deck_rows)
    draws = sum(str(row["result"]) == "draw" for row in deck_rows)
    losses = sum(str(row["result"]) == "loss" for row in deck_rows)
    first_rows = [row for row in deck_rows if bool(row["went_first"])]
    second_rows = [row for row in deck_rows if not bool(row["went_first"])]
    daily: dict[str, Any] = {}
    for selected_date in dates:
        date_rows = [row for row in deck_rows if str(row["date"]) == selected_date]
        daily[f"sides_{selected_date}"] = len(date_rows)
        daily[f"score_rate_{selected_date}"] = score_rate(date_rows)
    pokemon_summary = "; ".join(
        f"{card_meta[card_id][0]} x{count}"
        for card_id, count in sorted(
            (
                (card_id, count)
                for card_id, count in deck_counts.items()
                if card_id in pokemon_ids
            ),
            key=lambda item: (-item[1], card_meta[item[0]][0], item[0]),
        )[:8]
    )
    nearest_route = active_routes[nearest_digest]
    nearest_pokemon_route = active_routes[nearest_pokemon_digest]
    return {
        "deck_digest": digest,
        "deck_hash": single_deck_hash(deck_rows),
        "deck_label": mode(str(row["deck_label"]) for row in deck_rows),
        "deck_signature": signature,
        "pokemon_summary": pokemon_summary,
        "sides": len(deck_rows),
        "pilots": len({str(row["pilot_key"]) for row in deck_rows}),
        "distinct_dates": len({str(row["date"]) for row in deck_rows}),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_rate": (wins + 0.5 * draws) / len(deck_rows),
        "wilson_lcb": wilson_lcb(
            wins + 0.5 * draws, len(deck_rows), z=wilson_z
        ),
        "first_sides": len(first_rows),
        "first_score_rate": score_rate(first_rows),
        "second_sides": len(second_rows),
        "second_score_rate": score_rate(second_rows),
        "qualified_groups": len(qualified),
        "qualified_sides": sum(int(row["sides"]) for row in qualified),
        "best_group_wilson_lcb": max(
            (float(row["wilson_lcb"]) for row in qualified), default=None
        ),
        "nearest_active_deck_digest": nearest_digest,
        "nearest_active_deck_hash": active_deck_hashes.get(nearest_digest),
        "nearest_active_deck_hash_available": nearest_digest in active_deck_hashes,
        "nearest_active_route_label": str(nearest_route.get("label", "")),
        "nearest_active_family_id": nearest_route.get("family_id"),
        "replacement_distance": replacement_distance(
            signature, active_signatures[nearest_digest]
        ),
        "nearest_pokemon_active_deck_digest": nearest_pokemon_digest,
        "nearest_pokemon_active_deck_hash": active_deck_hashes.get(
            nearest_pokemon_digest
        ),
        "nearest_pokemon_route_label": str(
            nearest_pokemon_route.get("label", "")
        ),
        "candidate_pokemon_cards_unmatched": candidate_unmatched_cards(
            signature,
            active_signatures[nearest_pokemon_digest],
            allowed=pokemon_ids,
        ),
        "rank10_present": rank10 is not None,
        "rank10_rank": None if rank10 is None else int(rank10["rank"]),
        "rank10_team_name": None if rank10 is None else str(rank10["team_name"]),
        "rank10_submission_id": (
            None if rank10 is None else int(rank10["submission_id"])
        ),
        "rank10_games": None if rank10 is None else int(rank10["games"]),
        "rank10_score_rate": (
            None if rank10 is None else float(rank10["score_rate"])
        ),
        **daily,
    }


def deck_hashes_by_digest(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Resolve authoritative observed compact identifiers by exact digest."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["deck_digest"]), []).append(row)
    return {digest: single_deck_hash(group) for digest, group in grouped.items()}


def single_deck_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    """Require one stable compact deck identifier across evidence rows."""
    return single_value(str(row["deck_hash"]) for row in rows)


def single_value(values: Iterable[str]) -> str:
    """Require and return one stable string value."""
    materialized = set(values)
    if len(materialized) != 1:
        raise ValueError(f"expected one stable value, observed {sorted(materialized)}")
    return next(iter(materialized))


def mode(values: Iterable[str]) -> str:
    """Return the deterministic modal label."""
    counts = Counter(values)
    if not counts:
        return ""
    return min(counts, key=lambda value: (-counts[value], value))


def score_rate(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Score wins as one and draws as one half."""
    if not rows:
        return None
    return (
        sum(str(row["result"]) == "win" for row in rows)
        + 0.5 * sum(str(row["result"]) == "draw" for row in rows)
    ) / len(rows)


def wilson_lcb(successes: float, trials: int, *, z: float) -> float:
    """Return a one-sided Wilson lower confidence bound."""
    if trials <= 0:
        return 0.0
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    return (
        rate
        + z * z / (2.0 * trials)
        - z
        * math.sqrt(
            rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)
        )
    ) / denominator


def replacement_distance(left: str, right: str) -> int:
    """Count card copies that must be replaced between exact 60-card lists."""
    left_counts = Counter(parse_canonical_signature(left).card_ids)
    right_counts = Counter(parse_canonical_signature(right).card_ids)
    return sum(
        max(0, left_counts[card_id] - right_counts[card_id])
        for card_id in left_counts
    )


def candidate_unmatched_cards(
    left: str,
    right: str,
    *,
    allowed: set[int],
) -> int:
    """Count candidate card copies not covered by one active deck subset."""
    left_counts = Counter(parse_canonical_signature(left).card_ids)
    right_counts = Counter(parse_canonical_signature(right).card_ids)
    return sum(
        max(0, left_counts[card_id] - right_counts[card_id])
        for card_id in allowed
    )


def candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Put leaderboard and qualified-data evidence first."""
    return (
        -int(bool(row["rank10_present"])),
        int(row["rank10_rank"] or 999),
        -int(row["qualified_sides"]),
        -int(row["sides"]),
        str(row["deck_hash"]),
    )


def _load_card_meta(path: Path) -> dict[int, tuple[str, str]]:
    result: dict[int, tuple[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            card_id = int(row["Card ID"])
            name = str(row["Card Name"])
            category = str(row["Stage (Pokémon)/Type (Energy and Trainer)"])
            previous = result.setdefault(card_id, (name, category))
            if previous != (name, category):
                raise ValueError(f"card metadata conflicts for ID {card_id}")
    return result


__all__ = [
    "candidate_deck_rows",
    "candidate_sort_key",
    "deck_hashes_by_digest",
    "load_active_routes",
    "load_rank10_evidence",
    "mode",
    "score_rate",
    "single_deck_hash",
    "wilson_lcb",
]
