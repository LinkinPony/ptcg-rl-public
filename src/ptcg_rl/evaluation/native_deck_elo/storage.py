"""Artifact identity, persistence, and recovery helpers for native deck Elo."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.native_deck_elo.models import (
    FORMAT,
    RESULTS_FORMAT,
    DeckAsset,
    NativeDeckEloConfig,
    ScheduledGame,
)


def campaign_payload(
    config: NativeDeckEloConfig,
    *,
    decks: Sequence[DeckAsset],
    runtime_fingerprint: str,
    belief_fingerprint: str,
) -> dict[str, Any]:
    """Build the immutable campaign identity payload."""
    return {
        "format": FORMAT,
        "checkpoint": {
            "path": records.display_path(config.checkpoint_path),
            "sha256": config.expected_checkpoint_sha256,
            "source_commit": config.checkpoint_source_commit,
        },
        "runner_source_commit": config.runner_source_commit,
        "public_catalog": {
            "path": records.display_path(config.public_catalog_manifest_path),
            "sha256": config.expected_public_catalog_manifest_sha256,
        },
        "native_library": {
            "path": records.display_path(config.native_library_path),
            "sha256": config.expected_native_library_sha256,
        },
        "runtime_fingerprint": runtime_fingerprint,
        "belief_fingerprint": belief_fingerprint,
        "decks": [
            {
                "label": deck.label,
                "path": records.display_path(deck.path),
                "deck_digest": deck.deck_digest,
                "deck_hash": deck.deck_hash,
                "deck_signature": deck.deck_signature,
            }
            for deck in decks
        ],
        "schedule": {
            "total_games": config.total_games,
            "seed": config.seed,
            "mirrored_seats": True,
            "full_pair_coverage": True,
        },
        "execution": {
            "concurrency": config.concurrency,
            "policy_batch_max_rows": config.policy_batch_max_rows,
            "policy_batch_wait_ms": config.policy_batch_wait_ms,
            "native_lane_worker_count": config.native_lane_worker_count,
            "native_option_capacity": config.native_option_capacity,
            "maximum_engine_steps": config.maximum_engine_steps,
            "checkpoint_cache_entries": config.checkpoint_cache_entries,
            "device": config.device,
            "act_time": config.act_time.model_dump(mode="json"),
        },
        "elo": {"initial": config.elo_initial, "k": config.elo_k},
    }


def publish_or_validate_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a campaign manifest or require an exact match."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing native deck Elo manifest differs from config")
        return
    write_json_atomic(path, payload)


def load_parts(
    parts_dir: Path,
    *,
    games: Sequence[ScheduledGame],
    campaign_fingerprint: str,
) -> list[dict[str, Any]]:
    """Load and validate all atomically committed result shards."""
    expected = {game.game_index: game for game in games}
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in sorted(parts_dir.glob("part-*.parquet")):
        for raw_row in pq.read_table(path).to_pylist():
            row = cast(dict[str, Any], raw_row)
            game_index = int(row["game_index"])
            game = expected.get(game_index)
            if game is None or game_index in seen:
                raise ValueError(f"invalid or duplicate game index in {path}")
            if (
                row.get("format") != RESULTS_FORMAT
                or row.get("campaign_fingerprint") != campaign_fingerprint
                or row.get("match_id") != game.match_id
            ):
                raise ValueError(f"game result identity differs in {path}")
            seen.add(game_index)
            rows.append(row)
    rows.sort(key=lambda row: int(row["game_index"]))
    return rows


def next_part_index(parts_dir: Path) -> int:
    """Return the next monotonically increasing shard index."""
    indices = [
        int(path.stem.removeprefix("part-"))
        for path in parts_dir.glob("part-*.parquet")
    ]
    return max(indices, default=-1) + 1


def write_part(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    compression: str,
) -> None:
    """Write a Parquet artifact and atomically publish it."""
    if not rows:
        return
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(
        pa.Table.from_pylist([dict(row) for row in rows]),
        temporary,
        compression=compression,
    )
    temporary.replace(path)


def write_progress(
    path: Path,
    *,
    total_games: int,
    completed_games: int,
    resumed_games: int,
    started_clock: float,
    complete: bool,
) -> None:
    """Publish recoverable committed-game progress."""
    elapsed = max(0.0, time.perf_counter() - started_clock)
    new_games = max(0, completed_games - resumed_games)
    write_json_atomic(
        path,
        {
            "complete": complete,
            "games_total": total_games,
            "games_committed": completed_games,
            "games_remaining": max(0, total_games - completed_games),
            "resumed_games": resumed_games,
            "session_games": new_games,
            "elapsed_seconds": elapsed,
            "session_games_per_second": safe_rate(new_games, elapsed),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def verified_file(path: Path, *, root: Path, expected_sha256: str) -> Path:
    """Resolve a required file and validate its complete SHA-256."""
    resolved = resolve_file(path, root=root)
    actual = file_sha256(resolved)
    if actual != expected_sha256:
        raise ValueError(f"artifact SHA-256 mismatch: {resolved}")
    return resolved


def resolve_file(path: Path, *, root: Path) -> Path:
    """Resolve a required repository-relative or absolute file."""
    resolved = resolve_path(path, root=root)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_path(path: Path, *, root: Path) -> Path:
    """Resolve a repository-relative or absolute path."""
    return (path if path.is_absolute() else root / path).resolve()


def file_sha256(path: Path) -> str:
    """Stream one file into SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a domain-separated canonical campaign fingerprint."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"ptcg-rl/native-deck-elo/v1\0" + encoded).hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write stable pretty JSON."""
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text_atomic(path: Path, value: str) -> None:
    """Atomically publish UTF-8 text."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def safe_rate(numerator: float, denominator: float | int) -> float:
    """Compute a rate while keeping empty aggregates finite."""
    return numerator / float(denominator) if denominator else 0.0


__all__ = [
    "campaign_payload",
    "file_sha256",
    "fingerprint",
    "load_parts",
    "next_part_index",
    "publish_or_validate_manifest",
    "resolve_file",
    "resolve_path",
    "safe_rate",
    "verified_file",
    "write_json_atomic",
    "write_part",
    "write_progress",
    "write_text_atomic",
]
