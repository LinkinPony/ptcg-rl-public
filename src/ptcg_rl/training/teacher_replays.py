"""Immutable download inventory for one public-pilot Kaggle submission."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent import futures
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps import records as step_records

_VERIFIED_ARCHIVES: set[tuple[Path, str]] = set()


class PublicPilotReplaySyncConfig(BaseModel):
    """Config for freezing and downloading one submission replay inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    submission_id: int
    team_name: str
    team_aliases: tuple[str, ...] = ()
    output_dir: Path
    created_at_or_after: datetime | None = None
    created_before: datetime
    episode_types: tuple[str, ...] = ("EpisodeType.EPISODE_TYPE_PUBLIC",)
    completed_states: tuple[str, ...] = ("EpisodeState.COMPLETED",)
    max_episodes: int | None = None
    download_workers: int = 4
    kaggle_binary: str = "kaggle"
    fast_prefix_bytes: int = 65_536
    reuse_replay_roots: tuple[Path, ...] = ()
    reuse_archive_roots: tuple[Path, ...] = ()
    archive_replay_cache: Path | None = None
    zstd_binary: str = "zstd"
    tar_binary: str = "tar"
    download_attempts: int = 5
    download_retry_seconds: float = 5.0

    @field_validator(
        "submission_id", "download_workers", "fast_prefix_bytes", "download_attempts"
    )
    @classmethod
    def positive_int(cls, value: int) -> int:
        """Reject non-positive identifiers and limits."""
        if value <= 0:
            raise ValueError("identifiers and limits must be positive")
        return value

    @field_validator("download_retry_seconds")
    @classmethod
    def positive_seconds(cls, value: float) -> float:
        """Reject non-positive retry delays."""
        if value <= 0.0:
            raise ValueError("retry delay must be positive")
        return value

    @field_validator("max_episodes")
    @classmethod
    def optional_positive_int(cls, value: int | None) -> int | None:
        """Reject a non-positive optional episode cap."""
        if value is not None and value <= 0:
            raise ValueError("max_episodes must be positive when set")
        return value

    @field_validator("team_name", "kaggle_binary")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        """Reject empty command and identity strings."""
        if not value.strip():
            raise ValueError("strings must be non-empty")
        return value.strip()

    @field_validator("team_aliases")
    @classmethod
    def normalized_team_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require unique, non-empty historical display names."""
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("team aliases must be non-empty")
        if len({value.casefold() for value in normalized}) != len(normalized):
            raise ValueError("team aliases must be unique")
        return normalized

    @field_validator("created_at_or_after", "created_before")
    @classmethod
    def timezone_aware_datetime(cls, value: datetime | None) -> datetime | None:
        """Require an explicit timezone for the immutable inventory boundary."""
        if value is not None and value.tzinfo is None:
            raise ValueError("inventory timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def valid_time_window(self) -> PublicPilotReplaySyncConfig:
        """Reject an empty or inverted inventory interval."""
        if (
            self.created_at_or_after is not None
            and self.created_at_or_after >= self.created_before
        ):
            raise ValueError("created_at_or_after must precede created_before")
        if self.reuse_archive_roots and self.archive_replay_cache is None:
            raise ValueError("archive replay cache is required for archive reuse")
        return self


class PublicPilotEpisode(BaseModel):
    """One immutable replay entry and its downloaded content fingerprint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: int
    create_time_utc: datetime
    end_time_utc: datetime | None = None
    state: str
    episode_type: str
    replay_path: Path
    bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    materialization: Literal["download", "existing", "reuse"] = "download"


def sync_public_pilot_replays(
    config: PublicPilotReplaySyncConfig,
) -> dict[str, Any]:
    """Freeze the inventory, download missing replays, and verify all files."""
    output_dir = deck_records.repo_path(config.output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _read_existing_manifest(manifest_path, config)
        _verify_manifest_replays(manifest)
        return manifest

    inventory = _fetch_inventory(config)
    if not inventory:
        raise ValueError("public-pilot inventory is empty after applying boundaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_root = output_dir / "replays"
    temp_root = output_dir / ".downloads"
    replay_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    reusable_replays = _reusable_replay_paths(config, inventory)
    reusable_replays.update(
        _extract_replays_from_archives(
            config,
            inventory,
            already_available=set(reusable_replays),
        )
    )

    episodes: list[PublicPilotEpisode] = []
    worker_count = min(config.download_workers, len(inventory))
    with futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending = [
            executor.submit(
                _download_episode,
                config,
                row,
                replay_root=replay_root,
                temp_root=temp_root,
                reusable_path=reusable_replays.get(int(row["episode_id"])),
            )
            for row in inventory
        ]
        for future in futures.as_completed(pending):
            episodes.append(future.result())
    episodes.sort(key=lambda row: (row.create_time_utc, row.episode_id))
    with suppress(OSError):
        temp_root.rmdir()

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "inventory_frozen": True,
        "episode_count": len(episodes),
        "total_bytes": sum(row.bytes for row in episodes),
        "materialization_counts": {
            source: sum(row.materialization == source for row in episodes)
            for source in ("download", "existing", "reuse")
        },
        "replay_root": deck_records.display_path(replay_root),
        "episodes": [row.model_dump(mode="json") for row in episodes],
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _fetch_inventory(config: PublicPilotReplaySyncConfig) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            config.kaggle_binary,
            "competitions",
            "episodes",
            str(config.submission_id),
            "--csv",
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for raw_row in csv.DictReader(io.StringIO(completed.stdout)):
        create_time = _parse_kaggle_datetime(raw_row.get("createTime"))
        if create_time is None or create_time >= config.created_before:
            continue
        if (
            config.created_at_or_after is not None
            and create_time < config.created_at_or_after
        ):
            continue
        state = str(raw_row.get("state") or "")
        episode_type = str(raw_row.get("type") or "")
        if config.completed_states and state not in config.completed_states:
            continue
        if config.episode_types and episode_type not in config.episode_types:
            continue
        rows.append(
            {
                "episode_id": int(str(raw_row["id"])),
                "create_time_utc": create_time,
                "end_time_utc": _parse_kaggle_datetime(raw_row.get("endTime")),
                "state": state,
                "episode_type": episode_type,
            }
        )
    rows.sort(key=lambda row: (row["create_time_utc"], row["episode_id"]))
    if config.max_episodes is not None:
        rows = rows[-config.max_episodes :]
    return rows


def _download_episode(
    config: PublicPilotReplaySyncConfig,
    inventory_row: Mapping[str, Any],
    *,
    replay_root: Path,
    temp_root: Path,
    reusable_path: Path | None,
) -> PublicPilotEpisode:
    episode_id = int(inventory_row["episode_id"])
    create_time = inventory_row["create_time_utc"]
    if not isinstance(create_time, datetime):
        raise TypeError("inventory create_time_utc must be a datetime")
    target_dir = replay_root / create_time.date().isoformat()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{episode_id}.json"
    materialization: Literal["download", "existing", "reuse"] = "existing"
    if not target_path.exists():
        if reusable_path is not None:
            _validate_replay(
                reusable_path,
                episode_id=episode_id,
                team_names=(config.team_name, *config.team_aliases),
                fast_prefix_bytes=config.fast_prefix_bytes,
            )
            _link_or_copy(reusable_path, target_path)
            materialization = "reuse"
        else:
            with tempfile.TemporaryDirectory(
                prefix=f"episode-{episode_id}-",
                dir=temp_root,
            ) as temporary_dir:
                _run_replay_download(
                    config,
                    episode_id=episode_id,
                    output_dir=Path(temporary_dir),
                )
                candidates = tuple(Path(temporary_dir).glob("*.json"))
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"expected one replay JSON for episode {episode_id}, "
                        f"found {len(candidates)}"
                    )
                _validate_replay(
                    candidates[0],
                    episode_id=episode_id,
                    team_names=(config.team_name, *config.team_aliases),
                    fast_prefix_bytes=config.fast_prefix_bytes,
                )
                os.replace(candidates[0], target_path)
                materialization = "download"
    _validate_replay(
        target_path,
        episode_id=episode_id,
        team_names=(config.team_name, *config.team_aliases),
        fast_prefix_bytes=config.fast_prefix_bytes,
    )
    return PublicPilotEpisode(
        **inventory_row,
        replay_path=Path(deck_records.display_path(target_path)),
        bytes=target_path.stat().st_size,
        sha256=_sha256(target_path),
        materialization=materialization,
    )


def _validate_replay(
    path: Path,
    *,
    episode_id: int,
    team_names: Sequence[str],
    fast_prefix_bytes: int,
) -> None:
    replay = step_records.replay_stub(path, chunk_size=fast_prefix_bytes)
    actual_id = deck_records.episode_id_from_replay(replay, path)
    if actual_id != episode_id:
        raise ValueError(
            f"downloaded replay id mismatch: expected={episode_id} actual={actual_id}"
        )
    replay_team_names = deck_records.team_names_from_replay(replay)
    accepted_names = {name.casefold() for name in team_names}
    if not accepted_names & {name.casefold() for name in replay_team_names}:
        raise ValueError(
            f"target teams {tuple(team_names)!r} are absent from episode "
            f"{episode_id}: {replay_team_names}"
        )


def _read_existing_manifest(
    path: Path,
    config: PublicPilotReplaySyncConfig,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid replay manifest: {path}")
    stored_config = PublicPilotReplaySyncConfig.model_validate(manifest.get("config"))
    if stored_config.model_dump(mode="json") != config.model_dump(mode="json"):
        raise ValueError(
            "existing public-pilot inventory config differs; choose a new output_dir"
        )
    if not bool(manifest.get("inventory_frozen", False)):
        raise ValueError("existing public-pilot inventory is not frozen")
    return manifest


def _verify_manifest_replays(manifest: Mapping[str, Any]) -> None:
    raw_episodes = manifest.get("episodes")
    if not isinstance(raw_episodes, Sequence) or isinstance(raw_episodes, str):
        raise ValueError("replay manifest episodes must be a list")
    for raw_episode in raw_episodes:
        episode = PublicPilotEpisode.model_validate(raw_episode)
        path = deck_records.repo_path(episode.replay_path)
        if not path.exists() or path.stat().st_size != episode.bytes:
            raise FileNotFoundError(f"frozen replay is missing or truncated: {path}")
        if _sha256(path) != episode.sha256:
            raise ValueError(f"frozen replay SHA256 mismatch: {path}")


def _parse_kaggle_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _reusable_replay_paths(
    config: PublicPilotReplaySyncConfig,
    inventory: Sequence[Mapping[str, Any]],
) -> dict[int, Path]:
    """Locate immutable replay files in explicitly configured source roots."""
    reusable: dict[int, Path] = {}
    roots = tuple(deck_records.repo_path(root) for root in config.reuse_replay_roots)
    for row in inventory:
        episode_id = int(row["episode_id"])
        create_time = row["create_time_utc"]
        if not isinstance(create_time, datetime):
            raise TypeError("inventory create_time_utc must be a datetime")
        date = create_time.date().isoformat()
        for root in roots:
            candidates = (
                root / date / f"{episode_id}.json",
                root / "replays" / date / f"{episode_id}.json",
                root / date / f"episode-{episode_id}-replay.json",
            )
            source = next((path for path in candidates if path.is_file()), None)
            if source is not None:
                reusable[episode_id] = source
                break
    return reusable


def _extract_replays_from_archives(
    config: PublicPilotReplaySyncConfig,
    inventory: Sequence[Mapping[str, Any]],
    *,
    already_available: set[int],
) -> dict[int, Path]:
    """Extract requested daily members once before using the replay endpoint."""
    if not config.reuse_archive_roots or config.archive_replay_cache is None:
        return {}
    archive_roots = tuple(
        deck_records.repo_path(root) for root in config.reuse_archive_roots
    )
    cache_root = deck_records.repo_path(config.archive_replay_cache)
    cache_root.mkdir(parents=True, exist_ok=True)
    requested_by_date: dict[str, dict[int, Path]] = defaultdict(dict)
    reusable: dict[int, Path] = {}
    for row in inventory:
        episode_id = int(row["episode_id"])
        if episode_id in already_available:
            continue
        create_time = row["create_time_utc"]
        if not isinstance(create_time, datetime):
            raise TypeError("inventory create_time_utc must be a datetime")
        date = create_time.date().isoformat()
        cache_path = cache_root / date / f"{episode_id}.json"
        if cache_path.is_file():
            reusable[episode_id] = cache_path
        else:
            requested_by_date[date][episode_id] = cache_path

    for date, requested in sorted(requested_by_date.items()):
        archive_pair = _daily_archive_pair(archive_roots, date)
        if archive_pair is None:
            continue
        archive_path, archive_manifest_path = archive_pair
        _verify_daily_archive(archive_path, archive_manifest_path, date=date)
        cache_date = cache_root / date
        cache_date.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{date}-",
            dir=cache_root,
        ) as temporary_dir:
            temporary_root = Path(temporary_dir)
            members = [f"{episode_id}.json" for episode_id in sorted(requested)]
            _extract_archive_members(
                archive_path,
                members=members,
                output_dir=temporary_root,
                zstd_binary=config.zstd_binary,
                tar_binary=config.tar_binary,
            )
            for episode_id, cache_path in requested.items():
                extracted = temporary_root / f"{episode_id}.json"
                if not extracted.is_file():
                    continue
                os.replace(extracted, cache_path)
                reusable[episode_id] = cache_path
    return reusable


def _daily_archive_pair(
    archive_roots: Sequence[Path],
    date: str,
) -> tuple[Path, Path] | None:
    for root in archive_roots:
        archive_path = root / f"{date}.tar.zst"
        manifest_path = root / f"{date}.manifest.json"
        if archive_path.is_file() and manifest_path.is_file():
            return archive_path, manifest_path
    return None


def _verify_daily_archive(
    archive_path: Path,
    manifest_path: Path,
    *,
    date: str,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"daily archive manifest is invalid: {manifest_path}")
    expected_sha256 = str(manifest.get("archive_sha256", ""))
    identity = (archive_path, expected_sha256)
    if (
        str(manifest.get("date", "")) != date
        or archive_path.stat().st_size != int(manifest.get("archive_bytes", -1))
        or len(expected_sha256) != 64
    ):
        raise ValueError(f"daily archive identity differs: {archive_path}")
    if identity not in _VERIFIED_ARCHIVES:
        if _sha256(archive_path) != expected_sha256:
            raise ValueError(f"daily archive SHA256 differs: {archive_path}")
        _VERIFIED_ARCHIVES.add(identity)


def _extract_archive_members(
    archive_path: Path,
    *,
    members: Sequence[str],
    output_dir: Path,
    zstd_binary: str,
    tar_binary: str,
) -> None:
    """Stream one trusted tar.zst and retain only requested numeric members."""
    decompressor = subprocess.Popen(
        [zstd_binary, "-dc", str(archive_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if decompressor.stdout is None:
        raise RuntimeError("failed to open daily archive decompressor output")
    extraction = subprocess.run(
        [tar_binary, "-x", "-C", str(output_dir), "--", *members],
        stdin=decompressor.stdout,
        capture_output=True,
        check=False,
    )
    decompressor.stdout.close()
    decompressor_stderr = (
        b"" if decompressor.stderr is None else decompressor.stderr.read()
    )
    decompressor_returncode = decompressor.wait()
    if decompressor_returncode != 0:
        raise RuntimeError(
            f"daily archive decompression failed: {archive_path}: "
            f"{decompressor_stderr.decode(errors='replace')}"
        )
    if extraction.returncode not in (0, 2):
        raise RuntimeError(
            f"daily archive extraction failed: {archive_path}: "
            f"{extraction.stderr.decode(errors='replace')}"
        )


def _link_or_copy(source: Path, target: Path) -> None:
    """Materialize one immutable replay cheaply, falling back across filesystems."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _run_replay_download(
    config: PublicPilotReplaySyncConfig,
    *,
    episode_id: int,
    output_dir: Path,
) -> None:
    """Download one replay with bounded backoff for transient Kaggle failures."""
    command = [
        config.kaggle_binary,
        "competitions",
        "replay",
        str(episode_id),
        "--path",
        str(output_dir),
        "--quiet",
    ]
    last_error = ""
    for attempt in range(config.download_attempts):
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            return
        last_error = (completed.stderr or completed.stdout).strip()
        if attempt + 1 < config.download_attempts:
            time.sleep(min(60.0, config.download_retry_seconds * (2**attempt)))
    raise RuntimeError(
        f"Kaggle replay download failed for episode {episode_id} after "
        f"{config.download_attempts} attempts: {last_error}"
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
