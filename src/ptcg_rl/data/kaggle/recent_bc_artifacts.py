"""Verified source I/O and compact artifacts for recent BC selection."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.data.kaggle.recent_bc_models import RecentBCSelectionConfig

REPO_ROOT = Path(__file__).resolve().parents[4]


def load_daily_rows(
    config: RecentBCSelectionConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify and load every configured compact Daily partition."""
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    episode_dates: dict[int, str] = {}
    for selected_date in config.dates:
        partition_dir = (
            repo_path(config.processed_root) / "daily" / f"date={selected_date}"
        )
        manifest_path = partition_dir / "manifest.json"
        manifest = read_json(manifest_path)
        parquet_path = partition_dir / "sides.parquet"
        archive_path = repo_path(config.package_root) / selected_date / "episodes.zip"
        package_manifest_path = archive_path.with_name("package_manifest.json")
        package_manifest = read_json(package_manifest_path)
        parquet_sha256 = sha256_file(parquet_path)
        archive_sha256 = sha256_file(archive_path)
        if parquet_sha256 != str(manifest.get("parquet_sha256")):
            raise ValueError(f"daily parquet identity changed: {selected_date}")
        if archive_sha256 != str(manifest.get("source_sha256")):
            raise ValueError(f"daily source identity changed: {selected_date}")
        if archive_sha256 != str(package_manifest.get("sha256")):
            raise ValueError(f"package source identity changed: {selected_date}")
        table = pq.ParquetFile(parquet_path).read()
        date_rows = cast(list[dict[str, Any]], table.to_pylist())
        if table.num_rows != int(manifest["sides"]):
            raise ValueError(f"daily side count changed: {selected_date}")
        by_episode = Counter(int(row["episode_id"]) for row in date_rows)
        if len(by_episode) != int(manifest["episodes"]) or set(by_episode.values()) != {2}:
            raise ValueError(f"daily episodes do not contain exactly two sides: {selected_date}")
        for episode_id in by_episode:
            previous = episode_dates.setdefault(episode_id, selected_date)
            if previous != selected_date:
                raise ValueError(f"episode appears in multiple dates: {episode_id}")
        rows.extend(date_rows)
        sources.append(
            {
                "date": selected_date,
                "partition_manifest_path": display_path(manifest_path),
                "partition_manifest_sha256": sha256_file(manifest_path),
                "parquet_path": display_path(parquet_path),
                "parquet_sha256": parquet_sha256,
                "archive_path": display_path(archive_path),
                "archive_sha256": archive_sha256,
                "archive_bytes": archive_path.stat().st_size,
                "episodes": len(by_episode),
                "sides": len(date_rows),
            }
        )
    return rows, sources


def validate_selected_episode_members(
    config: RecentBCSelectionConfig,
    *,
    selected_by_episode: Mapping[int, list[dict[str, Any]]],
    source_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], int]]:
    """Stream selected ZIP members and validate their selected action chains."""
    by_date: dict[str, list[int]] = {}
    for episode_id, rows in selected_by_episode.items():
        dates = {str(row["date"]) for row in rows}
        if len(dates) != 1:
            raise ValueError(f"selected episode spans dates: {episode_id}")
        by_date.setdefault(next(iter(dates)), []).append(episode_id)
    source_by_date = {str(row["date"]): row for row in source_records}
    replay_rows: list[dict[str, Any]] = []
    decisions: dict[tuple[int, int], int] = {}
    for selected_date, episode_ids in sorted(by_date.items()):
        source = source_by_date[selected_date]
        archive_path = repo_path(Path(str(source["archive_path"])))
        with zipfile.ZipFile(archive_path) as archive:
            for episode_id in sorted(episode_ids):
                replay_row, replay_decisions = _validate_member(
                    config,
                    archive=archive,
                    episode_id=episode_id,
                    selected_rows=selected_by_episode[episode_id],
                    source=source,
                    selected_date=selected_date,
                )
                replay_rows.append(replay_row)
                decisions.update(replay_decisions)
    return replay_rows, decisions


def _validate_member(
    config: RecentBCSelectionConfig,
    *,
    archive: zipfile.ZipFile,
    episode_id: int,
    selected_rows: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    selected_date: str,
) -> tuple[dict[str, Any], dict[tuple[int, int], int]]:
    member = f"{episode_id}.json"
    info = archive.getinfo(member)
    replay_sha256: str | None = None
    materialized_path: str | None = None
    decisions: dict[tuple[int, int], int] = {}
    if config.validate_selected_replays:
        payload_bytes = archive.read(info)
        replay_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        payload = orjson.loads(payload_bytes)
        if not isinstance(payload, dict):
            raise ValueError(f"replay is not an object: {episode_id}")
        replay_info = payload.get("info")
        if not isinstance(replay_info, Mapping):
            raise ValueError(f"replay has no info: {episode_id}")
        if int(replay_info.get("EpisodeId", -1)) != episode_id:
            raise ValueError(f"replay episode identity changed: {episode_id}")
        team_names = replay_info.get("TeamNames")
        if not isinstance(team_names, list) or len(team_names) != 2:
            raise ValueError(f"replay team names are incomplete: {episode_id}")
        selected_indices = {
            int(row["player_index"]): str(row["pilot_key"]) for row in selected_rows
        }
        for player_index, pilot_key in selected_indices.items():
            if _pilot_key(str(team_names[player_index])) != pilot_key:
                raise ValueError(f"selected replay pilot identity changed: {episode_id}")
        decisions = _validate_replay_actions(
            payload,
            episode_id=episode_id,
            player_indices=tuple(sorted(selected_indices)),
        )
        if config.write_pretraining_source:
            target = (
                repo_path(config.output_dir)
                / "replays"
                / selected_date
                / member
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if (
                    target.stat().st_size != len(payload_bytes)
                    or sha256_file(target) != replay_sha256
                ):
                    raise ValueError(
                        f"materialized replay identity changed: {episode_id}"
                    )
            else:
                atomic_write_bytes(target, payload_bytes)
            materialized_path = target.relative_to(
                repo_path(config.output_dir)
            ).as_posix()
    replay_row = {
        "date": selected_date,
        "episode_id": episode_id,
        "archive_path": str(source["archive_path"]),
        "archive_sha256": str(source["archive_sha256"]),
        "member": member,
        "member_uncompressed_bytes": info.file_size,
        "member_compressed_bytes": info.compress_size,
        "member_crc32": f"{info.CRC:08x}",
        "replay_sha256": replay_sha256,
        "materialized_path": materialized_path,
        "split": validation_split(
            episode_id,
            seed=config.split_seed,
            fraction=config.validation_fraction,
        ),
        "selected_player_indices": sorted(
            int(row["player_index"]) for row in selected_rows
        ),
        "selected_sides": len(selected_rows),
        "roster_covered_selected_sides": sum(
            bool(row.get("roster_covered")) for row in selected_rows
        ),
    }
    return replay_row, decisions


def _validate_replay_actions(
    payload: Mapping[str, Any],
    *,
    episode_id: int,
    player_indices: tuple[int, ...],
) -> dict[tuple[int, int], int]:
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"replay has no steps: {episode_id}")
    pending: dict[int, Mapping[str, Any]] = {}
    counts = Counter(dict.fromkeys(player_indices, 0))
    for step_index, raw_sides in enumerate(steps):
        if not isinstance(raw_sides, list):
            raise ValueError(f"replay step is not a side list: {episode_id}:{step_index}")
        for player_index in player_indices:
            if player_index >= len(raw_sides) or not isinstance(
                raw_sides[player_index], Mapping
            ):
                raise ValueError(
                    f"replay step is missing a selected side: {episode_id}:{step_index}"
                )
            side = cast(Mapping[str, Any], raw_sides[player_index])
            waiting = pending.pop(player_index, None)
            if waiting is not None:
                action = _integer_action(side.get("action"))
                if action is None or not is_legal_action(waiting, action):
                    raise ValueError(
                        "replay action is illegal for its pending prompt: "
                        f"{episode_id}:{step_index}:{player_index}"
                    )
                counts[player_index] += 1
            if str(side.get("status", "")) != "ACTIVE":
                continue
            observation = side.get("observation")
            if not isinstance(observation, Mapping):
                continue
            select = observation.get("select")
            if isinstance(select, Mapping) and select:
                pending[player_index] = cast(Mapping[str, Any], select)
    if pending:
        raise ValueError(f"replay ended with pending prompts: {episode_id}")
    return {
        (episode_id, player_index): counts[player_index]
        for player_index in player_indices
    }


def write_binding_csvs(
    output_dir: Path,
    *,
    selected_sides: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Write side bindings and archive-member inventory without raw JSON copies."""
    bindings = [
        {
            "episode_id": int(row["episode_id"]),
            "player_index": int(row["player_index"]),
            "pilot_key": str(row["pilot_key"]),
            "deck_digest": str(row["deck_digest"]),
            "deck_hash": str(row["deck_hash"]),
            "roster_covered": str(bool(row["roster_covered"])).lower(),
            "split": str(row["split"]),
        }
        for row in selected_sides
    ]
    replay_inventory = [
        {
            "date": str(row["date"]),
            "episode_id": int(row["episode_id"]),
            "archive_path": str(row["archive_path"]),
            "archive_sha256": str(row["archive_sha256"]),
            "member": str(row["member"]),
            "member_uncompressed_bytes": int(row["member_uncompressed_bytes"]),
            "member_crc32": str(row["member_crc32"]),
            "replay_sha256": str(row["replay_sha256"] or ""),
            "split": str(row["split"]),
        }
        for row in replay_rows
    ]
    return {
        "episode_side_bindings": _write_csv(
            output_dir / "episode_sides.csv", bindings
        ),
        "archive_member_inventory": _write_csv(
            output_dir / "replay_members.csv", replay_inventory
        ),
    }


def write_pretraining_source_csvs(
    output_dir: Path,
    *,
    selected_sides: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Publish the small manifests consumed by temporal replay pretraining."""
    replay_manifest: list[dict[str, Any]] = [
        {
            "date": str(row["date"]),
            "episode_id": int(row["episode_id"]),
            "relative_path": str(row["materialized_path"]),
            "size_bytes": int(row["member_uncompressed_bytes"]),
            "sha256": str(row["replay_sha256"]),
            "split": str(row["split"]),
        }
        for row in replay_rows
    ]
    if any(
        not row["relative_path"] or len(str(row["sha256"])) != 64
        for row in replay_manifest
    ):
        raise ValueError("pretraining replay materialization is incomplete")
    teams_by_name: dict[str, int] = {}
    episode_teams: set[tuple[int, str]] = set()
    for row in selected_sides:
        team_name = str(row.get("team_name", "")).strip()
        raw_rank = row.get("leaderboard_rank")
        if type(raw_rank) is not int:
            raise ValueError("pretraining side has no integer leaderboard rank")
        rank = raw_rank
        if not team_name or rank <= 0:
            raise ValueError("pretraining side lacks leaderboard provenance")
        previous = teams_by_name.setdefault(team_name, rank)
        if previous != rank:
            raise ValueError("leaderboard team rank changed within the selection")
        episode_teams.add((int(row["episode_id"]), team_name))
    team_rows = [
        {"Rank": rank, "TeamName": team_name}
        for team_name, rank in sorted(
            teams_by_name.items(), key=lambda item: (item[1], item[0].casefold())
        )
    ]
    binding_rows = [
        {
            "episode_id": episode_id,
            "submission_id": "",
            "team_name": team_name,
        }
        for episode_id, team_name in sorted(
            episode_teams, key=lambda item: (item[0], item[1].casefold())
        )
    ]
    outputs = {
        "replays_manifest": _write_csv(output_dir / "replays.csv", replay_manifest),
        "top_teams": _write_csv(output_dir / "teams.csv", team_rows),
        "episode_teams": _write_csv(
            output_dir / "episode_teams.csv", binding_rows
        ),
    }
    per_date: list[dict[str, Any]] = []
    for selected_date in sorted({str(row["date"]) for row in replay_manifest}):
        rows = [row for row in replay_manifest if row["date"] == selected_date]
        per_date.append(
            {
                "date": selected_date,
                "replay_count": len(rows),
                "replay_bytes": sum(int(row["size_bytes"]) for row in rows),
            }
        )
    return outputs, per_date


def write_parquet(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Atomically publish one compact Parquet table and its identity."""
    if not rows:
        raise ValueError(f"cannot write an empty parquet artifact: {path.name}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    table = pa.Table.from_pylist([dict(row) for row in rows])
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _output_identity(path, rows=len(rows))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"cannot write an empty CSV artifact: {path.name}")
    fieldnames = tuple(rows[0])
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _output_identity(path, rows=len(rows))


def _output_identity(path: Path, *, rows: int) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "bytes": path.stat().st_size,
    }


def validation_split(episode_id: int, *, seed: int, fraction: float) -> str:
    """Assign a complete episode to a stable train/validation split."""
    payload = (
        f"ptcg-rl/recent-bc-selection-split/v1\0{seed}\0{episode_id}"
    ).encode()
    sample = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return "validation" if sample < int(fraction * (1 << 64)) else "train"


def _integer_action(value: Any) -> tuple[int, ...] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or not all(type(item) is int for item in value)
    ):
        return None
    return tuple(int(item) for item in value)


def _pilot_key(value: str) -> str:
    normalized = " ".join(value.strip().casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON object."""
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], value)


def fingerprint(value: object) -> str:
    """Hash one canonical small identity object."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming file SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically publish bytes in the target directory."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def repo_path(path: Path) -> Path:
    """Resolve one repository-relative path."""
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    """Return a stable repository-relative path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def utc_now() -> str:
    """Return current UTC in a JSON-friendly representation."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "atomic_write_bytes",
    "display_path",
    "fingerprint",
    "load_daily_rows",
    "read_json",
    "repo_path",
    "sha256_file",
    "utc_now",
    "validate_selected_episode_members",
    "validation_split",
    "write_binding_csvs",
    "write_parquet",
    "write_pretraining_source_csvs",
]
