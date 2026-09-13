"""Build one immutable BC source inventory from submission replay manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import uuid
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[3]
REPLAY_COLUMNS = (
    "date",
    "episode_id",
    "relative_path",
    "size_bytes",
    "sha256",
    "split",
)
EPISODE_TEAM_COLUMNS = ("episode_id", "submission_id", "team_name")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_split(episode_id: int, *, seed: int, fraction: float) -> str:
    payload = f"ptcg-rl/public-pilot-bc-split/v1\0{seed}\0{episode_id}".encode()
    sample = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    threshold = int(fraction * (1 << 64))
    return "validation" if sample < threshold else "train"


def _balanced_kfold_assignments(
    episode_groups: Mapping[int, str],
    *,
    seed: int,
    fold_count: int,
) -> dict[int, int]:
    """Assign every episode once while balancing each source group by fold."""
    if fold_count < 2:
        raise ValueError("k-fold count must be at least two")
    grouped: dict[str, list[int]] = defaultdict(list)
    for episode_id, group in episode_groups.items():
        grouped[group].append(episode_id)
    assignments: dict[int, int] = {}
    domain = b"ptcg-rl/public-pilot-bc-kfold/v1\0"
    for group, episode_ids in sorted(grouped.items()):
        group_bytes = group.encode("utf-8")
        offset = (
            int.from_bytes(
                hashlib.sha256(
                    domain + str(seed).encode() + b"\0" + group_bytes
                ).digest()[:8],
                "big",
            )
            % fold_count
        )
        ordered = sorted(
            episode_ids,
            key=lambda episode_id: (
                hashlib.sha256(
                    domain
                    + str(seed).encode()
                    + b"\0"
                    + group_bytes
                    + b"\0"
                    + str(episode_id).encode()
                ).digest(),
                episode_id,
            ),
        )
        for rank, episode_id in enumerate(ordered):
            assignments[episode_id] = (offset + rank) % fold_count
    if assignments.keys() != episode_groups.keys():
        raise RuntimeError("k-fold assignment lost an episode")
    return assignments


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as file_obj:
            file_obj.write(data)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError(f"submission manifest is not a JSON object: {path}")
    return value


def build_corpus(
    manifest_paths: tuple[Path, ...],
    *,
    output_dir: Path,
    validation_fraction: float,
    split_seed: int,
    fold_count: int | None = None,
    validation_fold: int | None = None,
    selection_mode: Literal[
        "team_allowlist", "episode_team_bindings"
    ] = "team_allowlist",
) -> dict[str, Any]:
    """Verify, deduplicate, split, and publish a compact source inventory."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")
    if (fold_count is None) != (validation_fold is None):
        raise ValueError("k-fold count and validation fold must be set together")
    if fold_count is not None and (
        fold_count < 2
        or validation_fold is None
        or validation_fold < 0
        or validation_fold >= fold_count
    ):
        raise ValueError("validation fold must be within the configured k-fold range")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"BC source inventory already exists: {output_dir}")
    rows_by_episode: dict[int, dict[str, object]] = {}
    groups_by_episode: dict[int, set[str]] = defaultdict(set)
    teams: list[str] = []
    episode_team_bindings: set[tuple[int, int, str]] = set()
    source_records: list[dict[str, object]] = []
    for manifest_path in manifest_paths:
        resolved_manifest = manifest_path.resolve()
        manifest = _read_manifest(resolved_manifest)
        config = manifest.get("config")
        episodes = manifest.get("episodes")
        if not isinstance(config, dict) or not isinstance(episodes, list):
            raise ValueError(f"submission manifest is incomplete: {manifest_path}")
        team_name = str(config.get("team_name", "")).strip()
        if not team_name:
            raise ValueError(f"submission manifest has no team name: {manifest_path}")
        raw_aliases = config.get("team_aliases", [])
        if not isinstance(raw_aliases, list):
            raise ValueError(f"submission manifest has invalid aliases: {manifest_path}")
        source_teams = (team_name, *(str(alias).strip() for alias in raw_aliases))
        if any(not team for team in source_teams):
            raise ValueError(f"submission manifest has an empty alias: {manifest_path}")
        known_teams = {team.casefold() for team in teams}
        for source_team in source_teams:
            if source_team.casefold() not in known_teams:
                teams.append(source_team)
                known_teams.add(source_team.casefold())
        accepted = 0
        submission_id = int(config["submission_id"])
        for episode in episodes:
            if not isinstance(episode, dict):
                raise ValueError("submission manifest has a malformed episode row")
            episode_id = int(episode["episode_id"])
            episode_team_bindings.update(
                (episode_id, submission_id, source_team)
                for source_team in source_teams
            )
            replay_path = Path(str(episode["replay_path"]))
            absolute_path = (
                replay_path if replay_path.is_absolute() else REPO_ROOT / replay_path
            )
            absolute_path = absolute_path.resolve()
            try:
                relative_path = absolute_path.relative_to(REPO_ROOT)
            except ValueError as error:
                raise ValueError(
                    f"replay path escapes the repository: {absolute_path}"
                ) from error
            size_bytes = int(episode["bytes"])
            sha256 = str(episode["sha256"]).lower()
            if (
                not absolute_path.is_file()
                or absolute_path.stat().st_size != size_bytes
            ):
                raise ValueError(f"replay file identity changed: {relative_path}")
            if _sha256_file(absolute_path) != sha256:
                raise ValueError(f"replay SHA-256 changed: {relative_path}")
            create_time = str(episode["create_time_utc"])
            date = create_time[:10]
            row: dict[str, object] = {
                "date": date,
                "episode_id": episode_id,
                "relative_path": relative_path.as_posix(),
                "size_bytes": size_bytes,
                "sha256": sha256,
                "split": _validation_split(
                    episode_id,
                    seed=split_seed,
                    fraction=validation_fraction,
                ),
            }
            previous = rows_by_episode.get(episode_id)
            if previous is not None:
                identity_fields = ("date", "size_bytes", "sha256", "split")
                if any(previous[field] != row[field] for field in identity_fields):
                    raise ValueError(
                        f"duplicate episode identity differs: {episode_id}"
                    )
            if previous is None:
                rows_by_episode[episode_id] = row
                accepted += 1
            groups_by_episode[episode_id].update(
                source_team.casefold() for source_team in source_teams
            )
        source_records.append(
            {
                "manifest_path": resolved_manifest.relative_to(REPO_ROOT).as_posix(),
                "manifest_sha256": _sha256_file(resolved_manifest),
                "submission_id": submission_id,
                "team_name": team_name,
                "team_aliases": list(source_teams[1:]),
                "episodes_declared": len(episodes),
                "episodes_added_after_deduplication": accepted,
            }
        )
    if not rows_by_episode:
        raise ValueError("submission manifests selected no replay episodes")

    if fold_count is not None:
        fold_assignments = _balanced_kfold_assignments(
            {
                episode_id: "\0".join(sorted(groups_by_episode[episode_id]))
                for episode_id in rows_by_episode
            },
            seed=split_seed,
            fold_count=fold_count,
        )
        for episode_id, row in rows_by_episode.items():
            row["split"] = (
                "validation"
                if fold_assignments[episode_id] == validation_fold
                else "train"
            )

    ordered_rows = tuple(
        rows_by_episode[episode_id] for episode_id in sorted(rows_by_episode)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / "replays.csv"
    replay_temporary = output_dir / f".replays.{uuid.uuid4().hex}.csv"
    try:
        with replay_temporary.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=REPLAY_COLUMNS)
            writer.writeheader()
            writer.writerows(ordered_rows)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        replay_bytes = replay_temporary.read_bytes()
    finally:
        replay_temporary.unlink(missing_ok=True)
    _atomic_write(replay_path, replay_bytes)

    teams_path = output_dir / "teams.csv"
    teams_lines = [
        "Rank,TeamName",
        *(f"{rank},{team}" for rank, team in enumerate(teams, 1)),
    ]
    _atomic_write(teams_path, ("\n".join(teams_lines) + "\n").encode())

    episode_teams_path: Path | None = None
    episode_teams_sha256: str | None = None
    if selection_mode == "episode_team_bindings":
        episode_teams_path = output_dir / "episode_teams.csv"
        episode_teams_temporary = (
            output_dir / f".episode_teams.{uuid.uuid4().hex}.csv"
        )
        try:
            with episode_teams_temporary.open(
                "w", encoding="utf-8", newline=""
            ) as file_obj:
                writer = csv.DictWriter(
                    file_obj,
                    fieldnames=EPISODE_TEAM_COLUMNS,
                )
                writer.writeheader()
                writer.writerows(
                    {
                        "episode_id": episode_id,
                        "submission_id": submission_id,
                        "team_name": team_name,
                    }
                    for episode_id, submission_id, team_name in sorted(
                        episode_team_bindings,
                        key=lambda row: (row[0], row[1], row[2].casefold()),
                    )
                )
                file_obj.flush()
                os.fsync(file_obj.fileno())
            episode_teams_bytes = episode_teams_temporary.read_bytes()
        finally:
            episode_teams_temporary.unlink(missing_ok=True)
        _atomic_write(episode_teams_path, episode_teams_bytes)
        episode_teams_sha256 = hashlib.sha256(episode_teams_bytes).hexdigest()

    per_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in ordered_rows:
        per_date[str(row["date"])].append(row)
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_root": ".",
        "selection": {
            "mode": selection_mode,
            "leaderboard_cohort_filter": True,
        },
        "split_assignment": (
            {
                "domain": "ptcg-rl/public-pilot-bc-split/v1",
                "seed": split_seed,
                "validation_fraction": validation_fraction,
                "unit": "complete_episode",
            }
            if fold_count is None
            else {
                "domain": "ptcg-rl/public-pilot-bc-kfold/v1",
                "seed": split_seed,
                "fold_count": fold_count,
                "validation_fold": validation_fold,
                "balance_unit": "source_team_group",
                "unit": "complete_episode",
            }
        ),
        "sources": source_records,
        "outputs": {
            "replays_manifest_path": replay_path.relative_to(REPO_ROOT).as_posix(),
            "replays_manifest_sha256": hashlib.sha256(replay_bytes).hexdigest(),
            "top30_path": teams_path.relative_to(REPO_ROOT).as_posix(),
            "top30_sha256": _sha256_file(teams_path),
            "episode_teams_path": (
                None
                if episode_teams_path is None
                else episode_teams_path.relative_to(REPO_ROOT).as_posix()
            ),
            "episode_teams_sha256": episode_teams_sha256,
        },
        "per_date": [
            {
                "date": date,
                "replay_count": len(rows),
                "replay_bytes": sum(int(str(row["size_bytes"])) for row in rows),
            }
            for date, rows in sorted(per_date.items())
        ],
        "summary": {
            "source_replays": len(ordered_rows),
            "source_bytes": sum(int(str(row["size_bytes"])) for row in ordered_rows),
            "train_replays": sum(row["split"] == "train" for row in ordered_rows),
            "validation_replays": sum(
                row["split"] == "validation" for row in ordered_rows
            ),
            "episode_team_bindings": (
                len(episode_team_bindings)
                if selection_mode == "episode_team_bindings"
                else 0
            ),
            "teams": teams,
        },
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return {
        "manifest_path": manifest_path.relative_to(REPO_ROOT).as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
        **manifest["summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-manifest", action="append", required=True, type=Path
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260802)
    parser.add_argument("--fold-count", type=int)
    parser.add_argument("--validation-fold", type=int)
    parser.add_argument(
        "--selection-mode",
        choices=("team_allowlist", "episode_team_bindings"),
        default="team_allowlist",
    )
    args = parser.parse_args()
    result = build_corpus(
        tuple(args.submission_manifest),
        output_dir=(
            args.output_dir
            if args.output_dir.is_absolute()
            else REPO_ROOT / args.output_dir
        ),
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        fold_count=args.fold_count,
        validation_fold=args.validation_fold,
        selection_mode=args.selection_mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
