"""Durable challenger publication for metric-driven frozen league training."""

from __future__ import annotations

import json
import os
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.rl.curriculum import fingerprint_checkpoint

MAX_FROZEN_LEAGUE_OPPONENTS = 20


class FrozenLeagueConfig(BaseModel):
    """Configure challenger cadence, probe evidence, and promotion decisions."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    snapshot_interval_versions: int = 100
    max_opponents: int = 6
    probe_games_per_member: int = 256
    probe_sampling_fraction: float = 1.0
    promotion_probability: float = 0.95
    minimum_coverage_gain: float = 0.01
    max_versions_without_promotion: int = 400
    posterior_samples: int = 2048
    initial_champion_id: str | None = None

    @field_validator(
        "snapshot_interval_versions",
        "probe_games_per_member",
        "max_versions_without_promotion",
        "posterior_samples",
    )
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Require positive promotion cadence and evidence sizes."""
        if value <= 0:
            raise ValueError("frozen league integer settings must be positive")
        return value

    @field_validator("max_opponents")
    @classmethod
    def valid_max_opponents(cls, value: int) -> int:
        """Bound resident frozen policies to the supported H200 budget."""
        if value <= 1 or value > MAX_FROZEN_LEAGUE_OPPONENTS:
            raise ValueError(
                "max_opponents must be between 2 and "
                f"{MAX_FROZEN_LEAGUE_OPPONENTS}"
            )
        return value

    @field_validator(
        "probe_sampling_fraction",
        "promotion_probability",
        "minimum_coverage_gain",
    )
    @classmethod
    def valid_probability(cls, value: float) -> float:
        """Require finite probability-like promotion settings."""
        if value < 0.0 or value > 1.0:
            raise ValueError("frozen league probabilities must be in [0, 1]")
        return value

    @field_validator("initial_champion_id")
    @classmethod
    def clean_initial_champion_id(cls, value: str | None) -> str | None:
        """Normalize the optional initial champion identity."""
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("initial_champion_id must be non-empty when set")
        return cleaned

    @model_validator(mode="after")
    def valid_promotion_window(self) -> FrozenLeagueConfig:
        """Require timeout promotion to span at least one candidate cadence."""
        if self.max_versions_without_promotion < self.snapshot_interval_versions:
            raise ValueError(
                "max_versions_without_promotion must cover a snapshot interval"
            )
        return self


class FrozenLeagueCandidate(BaseModel):
    """One immutable learner checkpoint waiting for league probing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    opponent_id: str
    policy_version: int
    checkpoint_path: Path
    checkpoint_size_bytes: int
    checkpoint_sha256: str
    recurrent: bool = False
    created_at_utc: str

    @field_validator("schema_version")
    @classmethod
    def valid_schema_version(cls, value: int) -> int:
        """Reject unknown candidate journal schemas."""
        if value != 1:
            raise ValueError("unsupported frozen league candidate schema")
        return value

    @field_validator("policy_version", "checkpoint_size_bytes")
    @classmethod
    def valid_non_negative_int(cls, value: int) -> int:
        """Reject negative checkpoint metadata."""
        if value < 0:
            raise ValueError("frozen league candidate values must be non-negative")
        return value

    @field_validator("opponent_id")
    @classmethod
    def valid_opponent_id(cls, value: str) -> str:
        """Reject empty or path-like opponent identities."""
        cleaned = value.strip()
        if not cleaned or "/" in cleaned or "\\" in cleaned:
            raise ValueError("candidate opponent_id must be one path segment")
        return cleaned

    @field_validator("checkpoint_sha256")
    @classmethod
    def valid_checkpoint_sha256(cls, value: str) -> str:
        """Normalize and validate a checkpoint digest."""
        cleaned = value.strip().lower()
        if len(cleaned) != 64 or any(
            character not in "0123456789abcdef" for character in cleaned
        ):
            raise ValueError("checkpoint_sha256 must be a SHA-256 digest")
        return cleaned


def queue_frozen_league_candidate(
    config: FrozenLeagueConfig,
    *,
    policy_version: int,
    checkpoint_path: Path,
    checkpoint_size_bytes: int | None = None,
    checkpoint_sha256: str | None = None,
    recurrent: bool = False,
    candidates_dir: Path,
) -> Path | None:
    """Atomically publish one complete cadence checkpoint as a challenger."""
    if policy_version < 0:
        raise ValueError("policy_version must be non-negative")
    if (checkpoint_size_bytes is None) != (checkpoint_sha256 is None):
        raise ValueError("checkpoint size and SHA256 must be provided together")
    if not config.enabled or policy_version % config.snapshot_interval_versions != 0:
        return None

    if checkpoint_size_bytes is None or checkpoint_sha256 is None:
        fingerprint = fingerprint_checkpoint(checkpoint_path)
        checkpoint_size_bytes = fingerprint.size_bytes
        checkpoint_sha256 = fingerprint.sha256
    candidate = FrozenLeagueCandidate(
        opponent_id=f"league_v{policy_version}",
        policy_version=policy_version,
        checkpoint_path=checkpoint_path,
        checkpoint_size_bytes=checkpoint_size_bytes,
        checkpoint_sha256=checkpoint_sha256,
        recurrent=recurrent,
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    candidates_dir.mkdir(parents=True, exist_ok=True)
    path = frozen_league_candidate_path(candidates_dir, policy_version)
    if path.exists():
        existing = read_frozen_league_candidate(path)
        if existing != candidate.model_copy(
            update={"created_at_utc": existing.created_at_utc}
        ):
            raise ValueError("frozen league candidate identity changed")
        return None
    pending = candidates_dir / f".{path.name}.{time.time_ns()}.tmp"
    try:
        with pending.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    candidate.model_dump(mode="json"),
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        with suppress(OSError):
            directory_fd = os.open(
                candidates_dir,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        pending.unlink(missing_ok=True)
    return path


def frozen_league_candidate_path(candidates_dir: Path, policy_version: int) -> Path:
    """Return the canonical challenger journal path for one policy version."""
    if policy_version < 0:
        raise ValueError("policy_version must be non-negative")
    return candidates_dir / f"league_v{policy_version}.json"


def read_frozen_league_candidate(path: Path) -> FrozenLeagueCandidate:
    """Read and validate one challenger journal event."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return FrozenLeagueCandidate.model_validate(raw)
