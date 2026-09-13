"""Inventory, reuse, and atomic download support for top-team replays."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from concurrent import futures
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from ptcg_rl.data.kaggle.top_team_live_models import (
    EpisodeAssociation,
    InventoryEpisode,
    ManifestEpisode,
    ReplayManifest,
    ReplaySource,
    TopTeamLiveConfig,
)
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps import records as step_records


@dataclass(frozen=True)
class DailyReplay:
    """One replay in an overlapping immutable daily dataset."""

    episode_id: int
    path: Path
    create_time_utc: datetime | None


@dataclass
class _InventoryAccumulator:
    episode_id: int
    create_time_utc: datetime
    end_time_utc: datetime
    state: str
    episode_type: str
    associations_by_team: dict[str, EpisodeAssociation]


def fetch_inventory(
    config: TopTeamLiveConfig,
) -> tuple[list[InventoryEpisode], dict[int, int]]:
    """Fetch and deduplicate completed public inventories for all submissions."""
    _ensure_kaggle_available(config.kaggle_binary)
    requests = [
        (team, submission_id)
        for team in config.teams
        for submission_id in team.submission_ids
    ]
    target_teams = {team.team_id: team for team in config.teams}
    rows_by_submission: dict[int, list[dict[str, Any]]] = {}
    worker_count = min(config.download_workers, len(requests))
    with futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending = {
            executor.submit(
                _fetch_submission,
                config,
                submission_id,
                target_teams,
            ): (
                team,
                submission_id,
            )
            for team, submission_id in requests
        }
        for future in futures.as_completed(pending):
            _, submission_id = pending[future]
            rows_by_submission[submission_id] = future.result()

    accumulators: dict[int, _InventoryAccumulator] = {}
    for _team, submission_id in requests:
        for row in rows_by_submission[submission_id]:
            episode_id = int(row["episode_id"])
            existing = accumulators.get(episode_id)
            if existing is None:
                existing = _InventoryAccumulator(
                    episode_id=episode_id,
                    create_time_utc=_datetime(row["create_time_utc"]),
                    end_time_utc=_datetime(row["end_time_utc"]),
                    state=str(row["state"]),
                    episode_type=str(row["episode_type"]),
                    associations_by_team={},
                )
                accumulators[episode_id] = existing
            else:
                _validate_same_inventory_metadata(existing, row)
            for association in row["associations"]:
                if not isinstance(association, EpisodeAssociation):
                    raise TypeError("inventory association must be validated")
                previous = existing.associations_by_team.setdefault(
                    association.team_id,
                    association,
                )
                if previous != association:
                    raise ValueError(
                        "conflicting association metadata for episode "
                        f"{episode_id}, team {association.team_id}"
                    )

    episodes = [
        InventoryEpisode(
            episode_id=row.episode_id,
            create_time_utc=row.create_time_utc,
            end_time_utc=row.end_time_utc,
            state=row.state,
            episode_type=row.episode_type,
            associations=tuple(
                association
                for _, association in sorted(row.associations_by_team.items())
            ),
        )
        for row in accumulators.values()
    ]
    episodes.sort(key=lambda row: (row.create_time_utc, row.episode_id))
    return episodes, {
        submission_id: len(rows_by_submission[submission_id])
        for _, submission_id in requests
    }


def discover_daily_replays(config: TopTeamLiveConfig) -> dict[int, DailyReplay]:
    """Index replay files in daily directories overlapping the UTC window."""
    root = deck_records.repo_path(config.daily_replay_root)
    if not root.exists():
        raise FileNotFoundError(f"daily_replay_root does not exist: {root}")
    replays: dict[int, DailyReplay] = {}
    for day in _window_dates(config.start_time_utc, config.end_time_utc):
        day_dir = root / day.isoformat()
        if not day_dir.is_dir():
            continue
        create_times = _read_daily_manifest(day_dir / "manifest.csv")
        for path in sorted(day_dir.glob("*.json")):
            if not path.stem.isdigit():
                continue
            episode_id = int(path.stem)
            create_time = create_times.get(episode_id)
            if not _daily_file_in_window(path, create_time, config):
                continue
            replay = DailyReplay(
                episode_id=episode_id,
                path=path,
                create_time_utc=create_time,
            )
            previous = replays.setdefault(episode_id, replay)
            if previous.path != path:
                raise ValueError(
                    f"duplicate daily replay episode {episode_id}: "
                    f"{previous.path} and {path}"
                )
    return replays


def materialize_inventory(
    config: TopTeamLiveConfig,
    inventory: list[InventoryEpisode],
    daily_replays: Mapping[int, DailyReplay],
) -> ReplayManifest:
    """Reuse available replays and download one deck anchor per submission."""
    output_root = deck_records.repo_path(config.replay_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    temp_root = output_root / ".downloads"
    temp_root.mkdir(parents=True, exist_ok=True)
    selected = _select_replay_evidence(config, inventory, daily_replays, output_root)
    episodes: list[ManifestEpisode] = []
    if selected:
        worker_count = min(config.download_workers, len(selected))
        with futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            pending = [
                executor.submit(
                    _materialize_episode,
                    config,
                    row,
                    daily_replays.get(row.episode_id),
                    output_root,
                    temp_root,
                )
                for row in selected
            ]
            for future in futures.as_completed(pending):
                episodes.append(future.result())
    episodes.sort(key=lambda row: (row.create_time_utc, row.episode_id))
    with contextlib.suppress(OSError):
        temp_root.rmdir()
    manifest = ReplayManifest(
        created_at_utc=datetime.now(UTC),
        competition=config.competition,
        start_time_utc=config.start_time_utc,
        end_time_utc=config.end_time_utc,
        submission_ids=tuple(
            submission_id
            for team in config.teams
            for submission_id in team.submission_ids
        ),
        episode_count=len(episodes),
        association_count=sum(len(row.associations) for row in episodes),
        total_bytes=sum(row.bytes for row in episodes),
        episodes=tuple(episodes),
    )
    _atomic_write_json(
        output_root / "manifest.json",
        manifest.model_dump(mode="json"),
    )
    return manifest


def _fetch_submission(
    config: TopTeamLiveConfig,
    submission_id: int,
    target_teams: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]

    api = KaggleApi()
    api.authenticate()
    episodes = api.competition_list_episodes(submission_id)
    rows: list[dict[str, Any]] = []
    for raw in episodes:
        create_time = _parse_datetime(raw.create_time)
        if create_time is None:
            continue
        if not config.start_time_utc <= create_time < config.end_time_utc:
            continue
        state = str(raw.state or "")
        episode_type = str(raw.type or "")
        if _enum_tail(state) != "COMPLETED" or _enum_tail(episode_type) != "PUBLIC":
            continue
        end_time = _parse_datetime(raw.end_time)
        if end_time is None:
            raise ValueError(f"completed episode {raw.id} has no endTime")
        raw_agents = tuple(raw.agents or ())
        associations: list[EpisodeAssociation] = []
        for agent in raw_agents:
            team_id = str(agent.team_id)
            target_team = target_teams.get(team_id)
            if target_team is None:
                continue
            player_index = int(agent.index)
            opponent_name = next(
                (
                    str(opponent.team_name)
                    for opponent in raw_agents
                    if int(opponent.index) != player_index
                ),
                "",
            )
            associations.append(
                EpisodeAssociation(
                    team_id=team_id,
                    team_name=str(target_team.team_name),
                    submission_ids=(int(agent.submission_id),),
                    player_index=player_index,
                    opponent_team_name=opponent_name,
                    reward=(
                        float(agent.reward) if agent.reward is not None else None
                    ),
                    agent_state=str(agent.state or ""),
                )
            )
        if not associations:
            raise ValueError(
                f"queried submission {submission_id} returned episode {raw.id} "
                "without a configured target team"
            )
        rows.append(
            {
                "episode_id": int(raw.id),
                "create_time_utc": create_time,
                "end_time_utc": end_time,
                "state": state,
                "episode_type": episode_type,
                "associations": tuple(associations),
            }
        )
    rows.sort(key=lambda row: (_datetime(row["create_time_utc"]), row["episode_id"]))
    return rows


def _select_replay_evidence(
    config: TopTeamLiveConfig,
    inventory: list[InventoryEpisode],
    daily_replays: Mapping[int, DailyReplay],
    output_root: Path,
) -> list[InventoryEpisode]:
    """Select all local evidence plus a latest anchor for unseen submissions."""
    selected: dict[int, InventoryEpisode] = {}
    for episode in inventory:
        cache_path = (
            output_root
            / episode.create_time_utc.date().isoformat()
            / f"{episode.episode_id}.json"
        )
        if episode.episode_id in daily_replays or cache_path.exists():
            selected[episode.episode_id] = episode

    covered = {
        submission_id
        for episode in selected.values()
        for association in episode.associations
        for submission_id in association.submission_ids
    }
    for team in config.teams:
        for submission_id in team.submission_ids:
            if submission_id in covered:
                continue
            candidates = [
                episode
                for episode in inventory
                if any(
                    submission_id in association.submission_ids
                    for association in episode.associations
                )
            ]
            if not candidates:
                continue
            anchor = max(
                candidates,
                key=lambda row: (row.create_time_utc, row.episode_id),
            )
            selected[anchor.episode_id] = anchor
            covered.update(
                value
                for association in anchor.associations
                for value in association.submission_ids
            )
    return sorted(
        selected.values(),
        key=lambda row: (row.create_time_utc, row.episode_id),
    )


def _materialize_episode(
    config: TopTeamLiveConfig,
    episode: InventoryEpisode,
    daily_replay: DailyReplay | None,
    output_root: Path,
    temp_root: Path,
) -> ManifestEpisode:
    source: ReplaySource
    if daily_replay is not None:
        path = daily_replay.path
        source = "daily_reuse"
    else:
        target_dir = output_root / episode.create_time_utc.date().isoformat()
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{episode.episode_id}.json"
        if path.exists():
            source = "live_cache"
        else:
            _download_episode(config, episode, path, temp_root)
            source = "live_download"
    _validate_replay(path, episode)
    return ManifestEpisode(
        **episode.model_dump(),
        source=source,
        replay_path=Path(deck_records.display_path(path)),
        bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _download_episode(
    config: TopTeamLiveConfig,
    episode: InventoryEpisode,
    target_path: Path,
    temp_root: Path,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=f"episode-{episode.episode_id}-",
        dir=temp_root,
    ) as temporary_dir:
        subprocess.run(
            [
                config.kaggle_binary,
                "competitions",
                "replay",
                str(episode.episode_id),
                "--path",
                temporary_dir,
                "--quiet",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        candidates = tuple(Path(temporary_dir).glob("*.json"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one replay JSON for episode {episode.episode_id}, "
                f"found {len(candidates)}"
            )
        _validate_replay(candidates[0], episode)
        os.replace(candidates[0], target_path)


def _validate_replay(path: Path, episode: InventoryEpisode) -> None:
    replay = step_records.replay_stub(path, chunk_size=65_536)
    actual_id = deck_records.episode_id_from_replay(replay, path)
    if actual_id != episode.episode_id:
        raise ValueError(
            f"replay id mismatch: expected={episode.episode_id} actual={actual_id}"
        )
    names = deck_records.team_names_from_replay(replay)
    folded = [name.casefold() for name in names]
    for association in episode.associations:
        count = folded.count(association.team_name.casefold())
        if count != 1:
            raise ValueError(
                f"expected target team {association.team_name!r} exactly once in "
                f"episode {episode.episode_id}, found {count}: {names}"
            )


def _validate_same_inventory_metadata(
    existing: _InventoryAccumulator,
    row: Mapping[str, Any],
) -> None:
    expected = (
        existing.create_time_utc,
        existing.end_time_utc,
        existing.state,
        existing.episode_type,
    )
    actual = (
        _datetime(row["create_time_utc"]),
        _datetime(row["end_time_utc"]),
        str(row["state"]),
        str(row["episode_type"]),
    )
    if actual != expected:
        raise ValueError(
            f"conflicting inventory metadata for episode {existing.episode_id}"
        )


def _read_daily_manifest(path: Path) -> dict[int, datetime]:
    if not path.exists():
        return {}
    result: dict[int, datetime] = {}
    with path.open(encoding="utf-8-sig", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            parsed = _parse_datetime(row.get("create_time"))
            if parsed is not None:
                result[int(str(row["episode_id"]))] = parsed
    return result


def _daily_file_in_window(
    path: Path,
    create_time: datetime | None,
    config: TopTeamLiveConfig,
) -> bool:
    if create_time is not None:
        return config.start_time_utc <= create_time < config.end_time_utc
    day = date.fromisoformat(path.parent.name)
    day_start = datetime.combine(day, time.min, tzinfo=UTC)
    day_end = day_start + timedelta(days=1)
    return config.start_time_utc <= day_start and day_end <= config.end_time_utc


def _window_dates(start: datetime, end: datetime) -> list[date]:
    last = (end - timedelta(microseconds=1)).date()
    day = start.date()
    result: list[date] = []
    while day <= last:
        result.append(day)
        day += timedelta(days=1)
    return result


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _enum_tail(value: str) -> str:
    tail = value.rsplit(".", 1)[-1]
    for prefix in ("EPISODE_STATE_", "EPISODE_TYPE_"):
        if tail.startswith(prefix):
            return tail.removeprefix(prefix)
    return tail


def _datetime(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("expected datetime inventory metadata")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _ensure_kaggle_available(kaggle_binary: str) -> None:
    if shutil.which(kaggle_binary) is None:
        raise RuntimeError(f"Kaggle CLI binary not found: {kaggle_binary}")
