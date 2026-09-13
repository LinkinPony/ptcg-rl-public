"""Immutable collection of replay windows for qualified public submissions."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps import records as step_records
from ptcg_rl.training.teacher_replays import (
    PublicPilotReplaySyncConfig,
    sync_public_pilot_replays,
)


class QualificationEvidence(BaseModel):
    """One immutable observation proving that a submission reached the gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[
        "daily_min_score", "episode_rating", "leaderboard_snapshot"
    ]
    observed_at_utc: datetime
    score: float = Field(ge=1100.0)
    source_path: Path
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode_id: int | None = None

    @field_validator("observed_at_utc")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        """Require an explicit UTC-comparable evidence timestamp."""
        if value.tzinfo is None:
            raise ValueError("qualification evidence timestamp must include timezone")
        return value

    @model_validator(mode="after")
    def daily_evidence_has_episode(self) -> QualificationEvidence:
        """Bind daily score evidence to the exact episode row."""
        if (
            self.kind in {"daily_min_score", "episode_rating"}
            and self.episode_id is None
        ):
            raise ValueError("episode score evidence requires episode_id")
        return self


class QualifiedSubmission(BaseModel):
    """One exact immutable submission selected for full-history collection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    submission_id: int = Field(gt=0)
    team_name: str
    team_aliases: tuple[str, ...] = ()
    deck_hash: str = Field(pattern=r"^[0-9a-f]{12}$")
    deck_path: Path
    qualification: tuple[QualificationEvidence, ...] = Field(min_length=1)

    @field_validator("team_name")
    @classmethod
    def non_empty_team_name(cls, value: str) -> str:
        """Reject an empty primary display name."""
        if not value.strip():
            raise ValueError("team name must be non-empty")
        return value.strip()


class QualifiedReplayCollectionConfig(BaseModel):
    """Validated batch configuration for all qualified submissions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qualification_manifest_path: Path
    qualification_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_window_start: datetime
    qualification_window_end: datetime
    created_at_or_after: datetime | None = None
    created_before: datetime
    output_root: Path
    reuse_replay_roots: tuple[Path, ...] = ()
    reuse_archive_roots: tuple[Path, ...] = ()
    archive_replay_cache: Path | None = None
    zstd_binary: str = "zstd"
    tar_binary: str = "tar"
    download_workers: int = Field(default=4, gt=0)
    fast_prefix_bytes: int = Field(default=65_536, gt=0)
    audit_prefix_limit_bytes: int = Field(default=4_194_304, gt=0)
    kaggle_binary: str = "kaggle"
    download_attempts: int = Field(default=5, gt=0)
    download_retry_seconds: float = Field(default=5.0, gt=0.0)
    submissions: tuple[QualifiedSubmission, ...] = Field(min_length=1)

    @field_validator(
        "qualification_window_start", "qualification_window_end", "created_before"
    )
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        """Require explicit immutable time boundaries."""
        if value.tzinfo is None:
            raise ValueError("collection timestamps must include timezone")
        return value

    @field_validator("created_at_or_after")
    @classmethod
    def optional_timezone_aware(cls, value: datetime | None) -> datetime | None:
        """Require an explicit timezone for an optional replay lower bound."""
        if value is not None and value.tzinfo is None:
            raise ValueError("replay start must include timezone")
        return value

    @model_validator(mode="after")
    def coherent_identity(self) -> QualifiedReplayCollectionConfig:
        """Reject duplicate submissions and evidence outside the frozen window."""
        if self.qualification_window_start >= self.qualification_window_end:
            raise ValueError("qualification window is empty or inverted")
        if self.created_before != self.qualification_window_end:
            raise ValueError("replay cutoff must equal qualification window end")
        if (
            self.created_at_or_after is not None
            and self.created_at_or_after >= self.created_before
        ):
            raise ValueError("replay start must precede replay cutoff")
        if self.fast_prefix_bytes > self.audit_prefix_limit_bytes:
            raise ValueError("audit prefix limit must cover the initial prefix")
        submission_ids = tuple(row.submission_id for row in self.submissions)
        if len(set(submission_ids)) != len(submission_ids):
            raise ValueError("qualified submission IDs must be unique")
        for submission in self.submissions:
            for evidence in submission.qualification:
                if not (
                    self.qualification_window_start
                    <= evidence.observed_at_utc
                    <= self.qualification_window_end
                ):
                    raise ValueError("qualification evidence is outside the window")
        return self


class QualifiedReplayCollectionRequestConfig(BaseModel):
    """Small Hydra request resolved from one immutable selection manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qualification_manifest_path: Path
    qualification_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deck_dir: Path
    output_root: Path
    created_at_or_after: datetime | None = None
    reuse_replay_roots: tuple[Path, ...] = ()
    reuse_archive_roots: tuple[Path, ...] = ()
    archive_replay_cache: Path | None = None
    zstd_binary: str = "zstd"
    tar_binary: str = "tar"
    download_workers: int = Field(default=4, gt=0)
    fast_prefix_bytes: int = Field(default=65_536, gt=0)
    audit_prefix_limit_bytes: int = Field(default=4_194_304, gt=0)
    kaggle_binary: str = "kaggle"
    download_attempts: int = Field(default=5, gt=0)
    download_retry_seconds: float = Field(default=5.0, gt=0.0)


def resolve_qualified_replay_collection(
    request: QualifiedReplayCollectionRequestConfig,
) -> QualifiedReplayCollectionConfig:
    """Resolve the frozen evidence snapshot into executable source rows."""
    manifest_path = deck_records.repo_path(request.qualification_manifest_path)
    if _sha256(manifest_path) != request.qualification_manifest_sha256:
        raise ValueError("qualification manifest SHA256 differs")
    selection = _read_json(manifest_path)
    window = _required_mapping(selection.get("qualification_window"), "window")
    raw_submissions = selection.get("submissions")
    if not isinstance(raw_submissions, list):
        raise ValueError("qualification manifest submissions must be a list")
    expected_count = int(selection.get("expected_submission_count", -1))
    if len(raw_submissions) != expected_count:
        raise ValueError("qualification manifest submission count differs")
    deck_dir = deck_records.repo_path(request.deck_dir)
    submissions: list[QualifiedSubmission] = []
    for raw_submission in raw_submissions:
        row = _required_mapping(raw_submission, "submission")
        raw_aliases = row.get("team_aliases", [])
        if not isinstance(raw_aliases, list):
            raise ValueError("qualified team aliases must be a list")
        evidence_row = _required_mapping(row.get("qualification"), "qualification")
        evidence = QualificationEvidence.model_validate(
            {
                "kind": str(evidence_row["kind"]),
                "observed_at_utc": str(evidence_row["observed_at_utc"]),
                "score": float(evidence_row["score"]),
                "source_path": request.qualification_manifest_path,
                "source_sha256": request.qualification_manifest_sha256,
                "episode_id": (
                    None
                    if evidence_row.get("episode_id") is None
                    else int(evidence_row["episode_id"])
                ),
            }
        )
        deck_hash = str(row["deck_hash"])
        submissions.append(
            QualifiedSubmission(
                submission_id=int(row["submission_id"]),
                team_name=str(row["team_name"]),
                team_aliases=tuple(str(alias) for alias in raw_aliases),
                deck_hash=deck_hash,
                deck_path=Path(deck_records.display_path(deck_dir / f"{deck_hash}.csv")),
                qualification=(evidence,),
            )
        )
    return QualifiedReplayCollectionConfig(
        qualification_manifest_path=request.qualification_manifest_path,
        qualification_manifest_sha256=request.qualification_manifest_sha256,
        qualification_window_start=datetime.fromisoformat(str(window["start_utc"])),
        qualification_window_end=datetime.fromisoformat(str(window["end_utc"])),
        created_at_or_after=request.created_at_or_after,
        created_before=datetime.fromisoformat(str(selection["full_history_cutoff_utc"])),
        output_root=request.output_root,
        reuse_replay_roots=request.reuse_replay_roots,
        reuse_archive_roots=request.reuse_archive_roots,
        archive_replay_cache=request.archive_replay_cache,
        zstd_binary=request.zstd_binary,
        tar_binary=request.tar_binary,
        download_workers=request.download_workers,
        fast_prefix_bytes=request.fast_prefix_bytes,
        audit_prefix_limit_bytes=request.audit_prefix_limit_bytes,
        kaggle_binary=request.kaggle_binary,
        download_attempts=request.download_attempts,
        download_retry_seconds=request.download_retry_seconds,
        submissions=tuple(submissions),
    )


def collect_qualified_replays(
    config: QualifiedReplayCollectionConfig,
) -> dict[str, Any]:
    """Collect, exact-deck audit, and publish all qualified source manifests."""
    _validate_evidence_sources(config.submissions)
    output_root = deck_records.repo_path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reusable_roots = [
        deck_records.repo_path(path) for path in config.reuse_replay_roots
    ]
    source_records: list[dict[str, Any]] = []
    unique_episodes: set[int] = set()
    for submission in config.submissions:
        output_dir = output_root / _submission_directory_name(
            submission,
            cutoff=config.created_before,
        )
        sync_config = PublicPilotReplaySyncConfig(
            submission_id=submission.submission_id,
            team_name=submission.team_name,
            team_aliases=submission.team_aliases,
            output_dir=Path(deck_records.display_path(output_dir)),
            created_at_or_after=config.created_at_or_after,
            created_before=config.created_before,
            max_episodes=None,
            download_workers=config.download_workers,
            kaggle_binary=config.kaggle_binary,
            fast_prefix_bytes=config.fast_prefix_bytes,
            reuse_replay_roots=tuple(
                Path(deck_records.display_path(path)) for path in reusable_roots
            ),
            reuse_archive_roots=config.reuse_archive_roots,
            archive_replay_cache=config.archive_replay_cache,
            zstd_binary=config.zstd_binary,
            tar_binary=config.tar_binary,
            download_attempts=config.download_attempts,
            download_retry_seconds=config.download_retry_seconds,
        )
        replay_manifest = sync_public_pilot_replays(sync_config)
        audit = _audit_exact_deck(
            replay_manifest,
            submission=submission,
            prefix_bytes=config.fast_prefix_bytes,
            prefix_limit_bytes=config.audit_prefix_limit_bytes,
        )
        manifest_path = output_dir / "manifest.json"
        episode_ids = {
            int(row["episode_id"])
            for row in _manifest_episodes(replay_manifest, manifest_path)
        }
        source_records.append(
            {
                "submission_id": submission.submission_id,
                "team_name": submission.team_name,
                "team_aliases": list(submission.team_aliases),
                "deck_hash": submission.deck_hash,
                "manifest_path": deck_records.display_path(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "episode_count": len(episode_ids),
                "episodes_added_after_deduplication": len(
                    episode_ids - unique_episodes
                ),
                "total_bytes": int(replay_manifest["total_bytes"]),
                "materialization_counts": replay_manifest.get(
                    "materialization_counts", {}
                ),
                "exact_deck_audit": audit,
                "qualification": [
                    evidence.model_dump(mode="json")
                    for evidence in submission.qualification
                ],
            }
        )
        unique_episodes.update(episode_ids)
        reusable_roots.append(output_dir / "replays")

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "sources": source_records,
        "summary": {
            "submissions": len(source_records),
            "episode_references": sum(row["episode_count"] for row in source_records),
            "unique_episodes": len(unique_episodes),
            "duplicate_episode_references": sum(
                row["episode_count"] for row in source_records
            )
            - len(unique_episodes),
            "deck_hash_counts": dict(
                sorted(Counter(row["deck_hash"] for row in source_records).items())
            ),
        },
    }
    manifest_path = output_root / "collection_manifest.json"
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        for field in ("schema_version", "config", "sources", "summary"):
            if existing.get(field) != payload[field]:
                raise ValueError(f"existing collection manifest differs at {field}")
        payload = existing
    else:
        _atomic_write_json(manifest_path, payload)
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("collection summary is not a mapping")
    return {
        "collection_manifest_path": deck_records.display_path(manifest_path),
        "collection_manifest_sha256": _sha256(manifest_path),
        **dict(summary),
    }


def _submission_directory_name(
    submission: QualifiedSubmission,
    *,
    cutoff: datetime,
) -> str:
    stamp = cutoff.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{submission.submission_id}_{submission.deck_hash}_through_{stamp}"


def _validate_evidence_sources(
    submissions: tuple[QualifiedSubmission, ...],
) -> None:
    """Verify every frozen qualification artifact before any network work."""
    observed: dict[Path, str] = {}
    for submission in submissions:
        for evidence in submission.qualification:
            source_path = deck_records.repo_path(evidence.source_path)
            actual_sha256 = observed.get(source_path)
            if actual_sha256 is None:
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                actual_sha256 = _sha256(source_path)
                observed[source_path] = actual_sha256
            if actual_sha256 != evidence.source_sha256:
                raise ValueError(
                    f"qualification source SHA256 differs: {source_path}"
                )


def _audit_exact_deck(
    manifest: Mapping[str, Any],
    *,
    submission: QualifiedSubmission,
    prefix_bytes: int,
    prefix_limit_bytes: int,
) -> dict[str, Any]:
    deck_path = deck_records.repo_path(submission.deck_path)
    signature = deck_records.deck_signature(deck_records.read_deck(deck_path))
    if deck_records.signature_hash(signature) != submission.deck_hash:
        raise ValueError(
            f"deck path differs from qualified hash: {submission.submission_id}"
        )
    accepted_names = {
        name.casefold() for name in (submission.team_name, *submission.team_aliases)
    }
    observed_names: set[str] = set()
    observed_hashes: Counter[str] = Counter()
    manifest_path = deck_records.repo_path(Path(str(manifest["replay_root"]))).parent
    for raw_episode in _manifest_episodes(manifest, manifest_path / "manifest.json"):
        replay_path = deck_records.repo_path(Path(str(raw_episode["replay_path"])))
        replay = step_records.replay_stub(replay_path, chunk_size=prefix_bytes)
        replay_names = deck_records.team_names_from_replay(replay)
        target_indices = [
            index
            for index, name in enumerate(replay_names)
            if name.casefold() in accepted_names
        ]
        if len(target_indices) != 1:
            raise ValueError(
                f"qualified team side is ambiguous in episode "
                f"{raw_episode['episode_id']}: {replay_names}"
            )
        decks = _registered_decks_from_prefix(
            replay_path,
            initial_bytes=prefix_bytes,
            limit_bytes=prefix_limit_bytes,
        )
        target_deck = decks.get(target_indices[0])
        if target_deck is None:
            raise ValueError(
                f"qualified side has no registered deck: {raw_episode['episode_id']}"
            )
        observed_hash = deck_records.signature_hash(
            deck_records.deck_signature(target_deck)
        )
        observed_names.add(replay_names[target_indices[0]])
        observed_hashes[observed_hash] += 1
    if set(observed_hashes) != {submission.deck_hash}:
        raise ValueError(
            f"submission {submission.submission_id} registered unexpected decks: "
            f"{dict(observed_hashes)}"
        )
    return {
        "deck_path": deck_records.display_path(deck_path),
        "deck_sha256": _sha256(deck_path),
        "observed_team_names": sorted(observed_names),
        "observed_deck_hash_counts": dict(sorted(observed_hashes.items())),
    }


def _registered_decks_from_prefix(
    path: Path,
    *,
    initial_bytes: int,
    limit_bytes: int,
) -> dict[int, list[int]]:
    read_bytes = initial_bytes
    while read_bytes <= limit_bytes:
        with path.open("rb") as replay_file:
            identity = deck_records.fast_episode_identity_and_decks(
                replay_file.read(read_bytes)
            )
        if identity is not None:
            return identity[1]
        read_bytes *= 2
    raise ValueError(f"registered decks are absent from replay prefix: {path}")


def _manifest_episodes(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> list[Mapping[str, Any]]:
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"submission manifest has invalid episodes: {manifest_path}")
    if not all(isinstance(row, Mapping) for row in episodes):
        raise ValueError(f"submission manifest has malformed episodes: {manifest_path}")
    return episodes


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"qualification manifest {label} must be an object")
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "QualificationEvidence",
    "QualifiedReplayCollectionConfig",
    "QualifiedReplayCollectionRequestConfig",
    "QualifiedSubmission",
    "collect_qualified_replays",
    "resolve_qualified_replay_collection",
]
