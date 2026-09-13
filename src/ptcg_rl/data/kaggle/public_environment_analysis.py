"""Build public Daily meta and roster posterior snapshots from compact partitions."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.dashboard.deck_routes import label_hash
from ptcg_rl.data.kaggle.public_environment_ingest import (
    partition_directory,
    repo_path,
)
from ptcg_rl.data.kaggle.public_environment_models import PublicEnvironmentConfig
from ptcg_rl.decks.identity import parse_canonical_signature
from ptcg_rl.evaluation.posterior import MatchupOutcome, evaluate_bundle_posteriors
from ptcg_rl.rl.performance_state import atomic_write_json

ANALYSIS_REVISION = 1
SNAPSHOT_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class RosterDeck:
    """One active exact route with its authoritative human compact identity."""

    label: str
    deck_hash: str
    deck_digest: str
    family_id: str | None


@dataclass(slots=True)
class _Counts:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float | None:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else None

    def add(self, result: str) -> None:
        if result == "win":
            self.wins += 1
        elif result == "draw":
            self.draws += 1
        elif result == "loss":
            self.losses += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "score": self.score,
        }


def load_roster(
    config: PublicEnvironmentConfig,
    *,
    run_id: str,
) -> tuple[RosterDeck, ...]:
    """Load exact routes without changing their historical compact identifiers."""
    run_dir = repo_path(config.run_root) / run_id
    resolved = _read_json(run_dir / "resolved_config.json")
    raw_routes = resolved.get("active_deck_routes")
    model = resolved.get("resolved_model_config")
    if not isinstance(raw_routes, list) or not isinstance(model, dict):
        raise ValueError(f"run has no authoritative active exact routes: {run_id}")
    exact_routes = model.get("exact_routes")
    if not isinstance(exact_routes, list):
        raise ValueError(f"run has no resolved exact-route registry: {run_id}")
    signatures: dict[str, str] = {}
    for raw in exact_routes:
        if not isinstance(raw, dict):
            continue
        digest = raw.get("deck_digest")
        signature = raw.get("signature")
        if isinstance(digest, str) and isinstance(signature, str):
            canonical = parse_canonical_signature(signature)
            if canonical.deck_digest != digest:
                raise ValueError(f"exact route signature/digest mismatch: {digest}")
            signatures[digest] = signature

    checkpoint_digests = _checkpoint_digests(
        run_dir,
        checkpoint_version=config.checkpoint_version,
    )
    roster: list[RosterDeck] = []
    for raw in raw_routes:
        if not isinstance(raw, dict):
            continue
        label = raw.get("label")
        digest = raw.get("deck_digest")
        family = raw.get("family_id")
        if not isinstance(label, str) or not isinstance(digest, str):
            continue
        if checkpoint_digests is not None and digest not in checkpoint_digests:
            continue
        if digest not in signatures:
            raise ValueError(f"active route is absent from exact registry: {label}")
        compact = label_hash(label)
        if compact is None:
            raise ValueError(
                f"authoritative active label has no compact deck_hash: {label}"
            )
        roster.append(
            RosterDeck(
                label=label,
                deck_hash=compact,
                deck_digest=digest,
                family_id=family if isinstance(family, str) else None,
            )
        )
    if not roster:
        raise ValueError(f"resolved active roster is empty: {run_id}")
    if len({item.deck_digest for item in roster}) != len(roster):
        raise ValueError("active roster deck digests must be unique")
    return tuple(roster)


def roster_fingerprint(roster: tuple[RosterDeck, ...]) -> str:
    """Return the run-independent identity of one authoritative exact roster."""
    return _fingerprint(
        [
            {
                "label": item.label,
                "deck_hash": item.deck_hash,
                "deck_digest": item.deck_digest,
                "family_id": item.family_id,
            }
            for item in roster
        ]
    )


def build_snapshots(
    config: PublicEnvironmentConfig,
    *,
    run_id: str,
    latest_date: str,
) -> dict[str, Any]:
    """Build every configured exact-date window and atomically publish latest."""
    roster = load_roster(config, run_id=run_id)
    latest = date.fromisoformat(latest_date)
    partition_manifests = _partition_manifests(config)
    roster_identity = roster_fingerprint(roster)
    required_dates = {
        (latest - timedelta(days=offset)).isoformat()
        for offset in range(max(config.windows))
    }
    source_fingerprint = _fingerprint(
        {
            key: value.get("parquet_sha256")
            for key, value in sorted(partition_manifests.items())
            if key in required_dates
        }
    )
    snapshot_fingerprint = _fingerprint(
        {
            "latest_date": latest_date,
            "roster_fingerprint": roster_identity,
            "source_fingerprint": source_fingerprint,
            "scoring": _scoring_identity(config),
        }
    )
    target_root = _target_root(config, roster_identity)
    snapshot_dir = target_root / "snapshots" / snapshot_fingerprint
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = _read_json(snapshot_dir / "manifest.json")
    if _snapshot_is_reusable(
        config,
        snapshot_dir=snapshot_dir,
        snapshot_fingerprint=snapshot_fingerprint,
        manifest=existing_manifest,
    ):
        atomic_write_json(target_root / "latest.json", existing_manifest)
        return existing_manifest
    windows: dict[str, Any] = {}
    for window_days in config.windows:
        dates = tuple(
            (latest - timedelta(days=offset)).isoformat()
            for offset in reversed(range(window_days))
        )
        missing = [item for item in dates if item not in partition_manifests]
        if missing:
            payload = _unavailable_payload(
                run_id=run_id,
                checkpoint_version=config.checkpoint_version,
                window_days=window_days,
                latest_date=latest_date,
                required_dates=dates,
                missing_dates=tuple(missing),
                snapshot_fingerprint=snapshot_fingerprint,
            )
        else:
            payload = _analyze_window(
                config,
                run_id=run_id,
                roster=roster,
                dates=dates,
                snapshot_fingerprint=snapshot_fingerprint,
            )
        path = snapshot_dir / f"window_{window_days}.json"
        atomic_write_json(path, payload)
        if payload["available"]:
            _write_tables(snapshot_dir, window_days, payload)
        windows[str(window_days)] = {
            "available": payload["available"],
            "missing_dates": payload["missing_dates"],
            "path": str(path.relative_to(repo_path(config.output_root))),
            "sha256": _sha256_file(path),
        }
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "run_id": run_id,
        "checkpoint_version": config.checkpoint_version,
        "latest_date": latest_date,
        "snapshot_fingerprint": snapshot_fingerprint,
        "roster_fingerprint": roster_identity,
        "source_fingerprint": source_fingerprint,
        "windows": windows,
    }
    atomic_write_json(snapshot_dir / "manifest.json", manifest)
    _write_report(snapshot_dir / "report.md", manifest)
    atomic_write_json(target_root / "latest.json", manifest)
    return manifest


def _analyze_window(
    config: PublicEnvironmentConfig,
    *,
    run_id: str,
    roster: tuple[RosterDeck, ...],
    dates: tuple[str, ...],
    snapshot_fingerprint: str,
) -> dict[str, Any]:
    roster_by_digest = {item.deck_digest: item for item in roster}
    meta_counts: Counter[str] = Counter()
    daily_counts: dict[str, Counter[str]] = defaultdict(Counter)
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    compact_hashes: dict[str, Counter[str]] = defaultdict(Counter)
    pilots: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_sides: Counter[str] = Counter()
    valid_sides: Counter[str] = Counter()
    seat_sides: dict[str, Counter[bool]] = defaultdict(Counter)
    observed: dict[str, _Counts] = defaultdict(_Counts)
    exact_cells: dict[tuple[str, str, bool], _Counts] = defaultdict(_Counts)
    quality: Counter[str] = Counter()

    for day in dates:
        manifest = _read_json(partition_directory(config, day) / "manifest.json")
        quality["source_missing_episodes"] += int(
            manifest.get("source_missing_episodes", 0)
        )
        quality["source_missing_bytes"] += int(manifest.get("source_missing_bytes", 0))

    columns = [
        field.name
        for field in pq.read_schema(
            partition_directory(config, dates[0]) / "sides.parquet"
        )
    ]
    for day in dates:
        parquet_path = partition_directory(config, day) / "sides.parquet"
        for batch in pq.ParquetFile(parquet_path).iter_batches(
            batch_size=65_536,
            columns=columns,
        ):
            for row in batch.to_pylist():
                digest = str(row["deck_digest"])
                meta_counts[digest] += 1
                daily_counts[day][digest] += 1
                labels[digest][str(row["deck_label"])] += 1
                compact_hashes[digest][str(row["deck_hash"])] += 1
                pilots[digest][str(row["pilot_key"])] += 1
                quality["sides"] += 1
                if not bool(row["terminal_valid"]):
                    quality["unresolved_sides"] += 1
                    continue
                quality["valid_sides"] += 1
                roster_deck = roster_by_digest.get(digest)
                if roster_deck is None:
                    continue
                went_first = row["went_first"]
                if not isinstance(went_first, bool):
                    continue
                opponent = str(row["opponent_deck_digest"])
                result = str(row["result"])
                candidate_sides[digest] += 1
                valid_sides[digest] += 1
                seat_sides[digest][went_first] += 1
                observed[digest].add(result)
                exact_cells[(digest, opponent, went_first)].add(result)

    total_sides = sum(meta_counts.values())
    if total_sides <= 0:
        raise ValueError("public environment window contains no exact deck sides")
    explicit = {
        digest
        for digest, count in meta_counts.items()
        if count >= config.min_explicit_opponent_sides
    }
    weights: dict[str, float] = {}
    unknown_sides = 0
    for digest, count in meta_counts.items():
        if digest in explicit:
            share = count / total_sides
            weights[_opponent_id(digest, True)] = 0.5 * share
            weights[_opponent_id(digest, False)] = 0.5 * share
        else:
            unknown_sides += count
    if unknown_sides:
        share = unknown_sides / total_sides
        weights[_opponent_id("__rare_tail__", True)] = 0.5 * share
        weights[_opponent_id("__rare_tail__", False)] = 0.5 * share

    outcomes: list[MatchupOutcome] = []
    for (candidate_digest, opponent, went_first), counts in exact_cells.items():
        bucket = opponent if opponent in explicit else "__rare_tail__"
        outcomes.append(
            MatchupOutcome(
                candidate_id=candidate_digest,
                opponent_id=_opponent_id(bucket, went_first),
                wins=counts.wins,
                draws=counts.draws,
                losses=counts.losses,
            )
        )
    posterior = evaluate_bundle_posteriors(
        candidate_ids=[item.deck_digest for item in roster],
        outcomes=outcomes,
        meta_weights=weights,
        unknown_mass=0.0,
        config=config.posterior,
        self_opponents={},
    )
    posterior_by_digest = {item.candidate_id: item for item in posterior.candidates}
    eligible = {
        item.deck_digest
        for item in roster
        if valid_sides[item.deck_digest] >= config.min_rank_sides
        and seat_sides[item.deck_digest][True] >= config.min_rank_sides_per_seat
        and seat_sides[item.deck_digest][False] >= config.min_rank_sides_per_seat
    }
    ranked = sorted(
        eligible,
        key=lambda digest: (-posterior_by_digest[digest].deploy_lcb, digest),
    )
    ranks = {digest: index + 1 for index, digest in enumerate(ranked)}

    roster_rows: list[dict[str, Any]] = []
    for item in roster:
        summary = posterior_by_digest[item.deck_digest]
        roster_rows.append(
            {
                "deck_label": item.label,
                "deck_hash": item.deck_hash,
                "deck_digest": item.deck_digest,
                "family_id": item.family_id,
                "rank": ranks.get(item.deck_digest),
                "evidence_status": (
                    "eligible" if item.deck_digest in eligible else "sparse"
                ),
                "environment_sides": meta_counts[item.deck_digest],
                "valid_games": valid_sides[item.deck_digest],
                "first_games": seat_sides[item.deck_digest][True],
                "second_games": seat_sides[item.deck_digest][False],
                "observed_score": observed[item.deck_digest].score,
                "deploy_mean": summary.deploy_mean,
                "deploy_credible_low": summary.deploy_credible_low,
                "deploy_credible_high": summary.deploy_credible_high,
                "deploy_lcb": summary.deploy_lcb,
                "matchup_cvar_mean": summary.matchup_cvar_mean,
                "probability_above_even": summary.probability_above_even,
                "observed_meta_mass": summary.observed_meta_mass,
                "prior_only_meta_mass": summary.prior_only_meta_mass,
            }
        )
    roster_rows.sort(
        key=lambda row: (
            row["rank"] is None,
            row["rank"] if row["rank"] is not None else 1_000_000,
            str(row["deck_hash"]),
        )
    )

    meta_rows = _meta_rows(
        config,
        dates=dates,
        meta_counts=meta_counts,
        daily_counts=daily_counts,
        labels=labels,
        compact_hashes=compact_hashes,
        pilots=pilots,
        roster_by_digest=roster_by_digest,
    )
    top_opponents = [
        str(row["deck_digest"])
        for row in meta_rows
        if str(row["deck_digest"]) in explicit
    ][: config.top_meta_decks]
    matrix = _matchup_matrix(
        config,
        roster=roster,
        opponent_digests=top_opponents,
        exact_cells=exact_cells,
        meta_rows=meta_rows,
    )
    coverage = _coverage_rows(matrix, meta_rows=meta_rows)
    shares = np.asarray([count / total_sides for count in meta_counts.values()])
    return {
        "schema_version": 1,
        "available": True,
        "run_id": run_id,
        "checkpoint_version": config.checkpoint_version,
        "window_days": len(dates),
        "as_of_date": dates[-1],
        "required_dates": list(dates),
        "missing_dates": [],
        "generated_at_utc": _utc_now(),
        "snapshot_fingerprint": snapshot_fingerprint,
        "source_scope": "kaggle_daily_all_episodes",
        "score_semantics": {
            "primary": "posterior_5pct_lcb",
            "seat_weights": {"first": 0.5, "second": 0.5},
            "meta_weighting": "episode_side_equal",
            "draw_score": 0.5,
            "non_done": "unresolved_excluded",
            "submission_data_used": False,
        },
        "quality": {
            "episodes": quality["sides"] // 2,
            "sides": quality["sides"],
            "valid_episodes": quality["valid_sides"] // 2,
            "unresolved_episodes": quality["unresolved_sides"] // 2,
            "source_missing_episodes": quality["source_missing_episodes"],
            "source_missing_bytes": quality["source_missing_bytes"],
            "explicit_meta_mass": 1.0 - unknown_sides / total_sides,
            "unknown_tail_mass": unknown_sides / total_sides,
            "eligible_roster_decks": len(eligible),
            "roster_decks": len(roster),
        },
        "overview": {
            "exact_decks": len(meta_counts),
            "pilot_keys": len({key for counts in pilots.values() for key in counts}),
            "meta_hhi": float(np.sum(shares * shares)),
            "top_10_share": sum(row["share"] for row in meta_rows[:10]),
        },
        "meta_decks": meta_rows[: config.top_meta_decks],
        "roster_standings": roster_rows,
        "coverage": coverage,
        "matrix": matrix,
    }


def _meta_rows(
    config: PublicEnvironmentConfig,
    *,
    dates: tuple[str, ...],
    meta_counts: Counter[str],
    daily_counts: dict[str, Counter[str]],
    labels: dict[str, Counter[str]],
    compact_hashes: dict[str, Counter[str]],
    pilots: dict[str, Counter[str]],
    roster_by_digest: dict[str, RosterDeck],
) -> list[dict[str, Any]]:
    total = sum(meta_counts.values())
    split = max(1, len(dates) // 2)
    early_dates = dates[:split]
    late_dates = dates[split:] or dates[-1:]
    early_total = sum(sum(daily_counts[day].values()) for day in early_dates)
    late_total = sum(sum(daily_counts[day].values()) for day in late_dates)
    output: list[dict[str, Any]] = []
    for digest, count in meta_counts.most_common():
        active = roster_by_digest.get(digest)
        pilot_counts = pilots[digest]
        pilot_total = sum(pilot_counts.values())
        pilot_hhi = sum((value / pilot_total) ** 2 for value in pilot_counts.values())
        early = sum(daily_counts[day][digest] for day in early_dates)
        late = sum(daily_counts[day][digest] for day in late_dates)
        early_share = early / early_total if early_total else 0.0
        late_share = late / late_total if late_total else 0.0
        output.append(
            {
                "deck_hash": (
                    active.deck_hash
                    if active is not None
                    else _most_common(compact_hashes[digest])
                ),
                "deck_digest": digest,
                "deck_label": (
                    active.label if active is not None else _most_common(labels[digest])
                ),
                "active_roster": active is not None,
                "sides": count,
                "share": count / total,
                "early_share": early_share,
                "late_share": late_share,
                "share_delta": late_share - early_share,
                "new_in_late_half": early == 0 and late >= config.min_rank_sides,
                "unique_pilots": len(pilot_counts),
                "pilot_hhi": pilot_hhi,
                "effective_pilots": 1.0 / pilot_hhi if pilot_hhi else 0.0,
            }
        )
    return output


def _matchup_matrix(
    config: PublicEnvironmentConfig,
    *,
    roster: tuple[RosterDeck, ...],
    opponent_digests: list[str],
    exact_cells: dict[tuple[str, str, bool], _Counts],
    meta_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    meta_by_digest = {str(row["deck_digest"]): row for row in meta_rows}
    rng = np.random.default_rng(config.posterior.seed + 1)
    output: list[dict[str, Any]] = []
    lower = (1.0 - config.posterior.credible_mass) / 2.0
    upper = 1.0 - lower
    for candidate in roster:
        for opponent_digest in opponent_digests:
            first = exact_cells[(candidate.deck_digest, opponent_digest, True)]
            second = exact_cells[(candidate.deck_digest, opponent_digest, False)]
            first_draws = rng.beta(
                config.posterior.beta_prior_alpha + first.wins + 0.5 * first.draws,
                config.posterior.beta_prior_beta + first.losses + 0.5 * first.draws,
                size=config.matchup_sample_count,
            )
            second_draws = rng.beta(
                config.posterior.beta_prior_alpha + second.wins + 0.5 * second.draws,
                config.posterior.beta_prior_beta + second.losses + 0.5 * second.draws,
                size=config.matchup_sample_count,
            )
            samples = 0.5 * (first_draws + second_draws)
            games = first.games + second.games
            evidence_eligible = bool(
                games >= config.min_matchup_sides
                and first.games >= config.min_matchup_sides_per_seat
                and second.games >= config.min_matchup_sides_per_seat
            )
            opponent = meta_by_digest[opponent_digest]
            output.append(
                {
                    "candidate_deck_hash": candidate.deck_hash,
                    "candidate_deck_digest": candidate.deck_digest,
                    "opponent_deck_hash": opponent["deck_hash"],
                    "opponent_deck_digest": opponent_digest,
                    "opponent_deck_label": opponent["deck_label"],
                    "games": games,
                    "first_games": first.games,
                    "second_games": second.games,
                    "evidence_eligible": evidence_eligible,
                    "posterior_mean": float(np.mean(samples)),
                    "credible_low": float(np.quantile(samples, lower)),
                    "credible_high": float(np.quantile(samples, upper)),
                    "lcb": float(np.quantile(samples, config.posterior.lcb_quantile)),
                }
            )
    return output


def _coverage_rows(
    matrix: list[dict[str, Any]],
    *,
    meta_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    meta_by_digest = {str(row["deck_digest"]): row for row in meta_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in matrix:
        grouped[str(cell["opponent_deck_digest"])].append(cell)
    output: list[dict[str, Any]] = []
    for digest, cells in grouped.items():
        eligible = [cell for cell in cells if cell["evidence_eligible"]]
        best = max(eligible, key=lambda cell: float(cell["lcb"])) if eligible else None
        state = (
            "evidence_blind_spot"
            if best is None
            else "verified_responder"
            if float(best["lcb"]) > 0.5
            else "unresolved"
        )
        meta = meta_by_digest[digest]
        output.append(
            {
                "opponent_deck_hash": meta["deck_hash"],
                "opponent_deck_digest": digest,
                "opponent_deck_label": meta["deck_label"],
                "meta_share": meta["share"],
                "state": state,
                "best_candidate_deck_hash": (
                    None if best is None else best["candidate_deck_hash"]
                ),
                "best_lcb": None if best is None else best["lcb"],
                "eligible_candidates": len(eligible),
            }
        )
    output.sort(key=lambda row: -float(row["meta_share"]))
    return output


def _unavailable_payload(
    *,
    run_id: str,
    checkpoint_version: int | None,
    window_days: int,
    latest_date: str,
    required_dates: tuple[str, ...],
    missing_dates: tuple[str, ...],
    snapshot_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "available": False,
        "run_id": run_id,
        "checkpoint_version": checkpoint_version,
        "window_days": window_days,
        "as_of_date": latest_date,
        "required_dates": list(required_dates),
        "missing_dates": list(missing_dates),
        "generated_at_utc": _utc_now(),
        "snapshot_fingerprint": snapshot_fingerprint,
        "source_scope": "kaggle_daily_all_episodes",
        "unavailable_reason": "requested calendar-day window is incomplete",
        "score_semantics": {"submission_data_used": False},
        "quality": {},
        "overview": {},
        "meta_decks": [],
        "roster_standings": [],
        "coverage": [],
        "matrix": [],
    }


def _partition_manifests(config: PublicEnvironmentConfig) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    daily_root = repo_path(config.processed_root) / "daily"
    for path in daily_root.glob("date=*/manifest.json"):
        payload = _read_json(path)
        day = payload.get("date")
        if isinstance(day, str) and (path.parent / "sides.parquet").is_file():
            output[day] = payload
    return output


def _checkpoint_digests(
    run_dir: Path,
    *,
    checkpoint_version: int | None,
) -> set[str] | None:
    if checkpoint_version is None:
        return None
    pair = _read_json(
        run_dir / "weights" / f"checkpoint_pair_v{checkpoint_version}.json"
    )
    metadata = pair.get("metadata")
    identity = (
        metadata.get("simple_stateless_identity")
        if isinstance(metadata, dict)
        else None
    )
    values = (
        identity.get("active_exact_deck_digests")
        if isinstance(identity, dict)
        else None
    )
    if not isinstance(values, list):
        return None
    return {str(value) for value in values}


def _target_root(config: PublicEnvironmentConfig, roster_identity: str) -> Path:
    return repo_path(config.output_root) / "rosters" / roster_identity


def _opponent_id(digest: str, went_first: bool) -> str:
    return f"{digest}|candidate_{'first' if went_first else 'second'}"


def _scoring_identity(config: PublicEnvironmentConfig) -> dict[str, Any]:
    return {
        "analysis_revision": ANALYSIS_REVISION,
        "windows": config.windows,
        "min_explicit_opponent_sides": config.min_explicit_opponent_sides,
        "min_rank_sides": config.min_rank_sides,
        "min_rank_sides_per_seat": config.min_rank_sides_per_seat,
        "min_matchup_sides": config.min_matchup_sides,
        "min_matchup_sides_per_seat": config.min_matchup_sides_per_seat,
        "top_meta_decks": config.top_meta_decks,
        "matchup_sample_count": config.matchup_sample_count,
        "posterior": config.posterior.model_dump(mode="json"),
    }


def _snapshot_is_reusable(
    config: PublicEnvironmentConfig,
    *,
    snapshot_dir: Path,
    snapshot_fingerprint: str,
    manifest: dict[str, Any],
) -> bool:
    """Accept an existing fingerprint only when every window is intact."""
    if (
        manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        or manifest.get("snapshot_fingerprint") != snapshot_fingerprint
    ):
        return False
    windows = manifest.get("windows")
    if not isinstance(windows, dict) or set(windows) != {
        str(value) for value in config.windows
    }:
        return False
    output_root = repo_path(config.output_root)
    for window_days in config.windows:
        detail = windows.get(str(window_days))
        if not isinstance(detail, dict):
            return False
        relative = detail.get("path")
        expected_sha256 = detail.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha256, str):
            return False
        path = (output_root / relative).resolve()
        expected_path = (snapshot_dir / f"window_{window_days}.json").resolve()
        if path != expected_path or not path.is_file():
            return False
        if _sha256_file(path) != expected_sha256:
            return False
    return True


def _write_tables(
    snapshot_dir: Path, window_days: int, payload: dict[str, Any]
) -> None:
    for name, key in (("meta", "meta_decks"), ("matchups", "matrix")):
        rows = payload[key]
        if not rows:
            continue
        output = snapshot_dir / f"window_{window_days}_{name}.parquet"
        temp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        pq.write_table(pa.Table.from_pylist(rows), temp, compression="zstd")
        os.replace(temp, output)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Kaggle Daily public environment snapshot",
        "",
        f"- Generated: `{manifest['generated_at_utc']}`",
        f"- Latest complete Daily: `{manifest['latest_date']}`",
        f"- Run roster: `{manifest['run_id']}`",
        f"- Snapshot fingerprint: `{manifest['snapshot_fingerprint']}`",
        "- Submission or leaderboard inputs: `none`",
        "",
        "## Windows",
        "",
    ]
    for window, details in manifest["windows"].items():
        status = "available" if details["available"] else "incomplete"
        lines.append(f"- {window} days: **{status}**")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _most_common(values: Counter[str]) -> str | None:
    return values.most_common(1)[0][0] if values else None


def _fingerprint(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "RosterDeck",
    "build_snapshots",
    "load_roster",
    "roster_fingerprint",
]
