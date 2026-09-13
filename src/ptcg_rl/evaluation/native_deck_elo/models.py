"""Validated models for bounded native deck Elo campaigns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.agent.runtime import ActTimeConfig
from ptcg_rl.evaluation.continuous_league.models import BundleIdentity

FORMAT = "native_deck_elo_v1"
RESULTS_FORMAT = "native_deck_elo_games_v1"


class NativeDeckEloDeckConfig(BaseModel):
    """One exact deck participating in a bounded ladder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    path: Path

    @field_validator("label")
    @classmethod
    def nonempty_label(cls, value: str) -> str:
        """Reject labels that cannot identify a standings row."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("deck label must be non-empty")
        return cleaned


class NativeDeckEloConfig(BaseModel):
    """Immutable inputs and operational settings for one Elo campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_path: Path
    expected_checkpoint_sha256: str
    checkpoint_source_commit: str
    runner_source_commit: str
    public_catalog_manifest_path: Path
    expected_public_catalog_manifest_sha256: str
    native_library_path: Path = Path("src/native/cg_train/libcg_train.so")
    expected_native_library_sha256: str
    decks: tuple[NativeDeckEloDeckConfig, ...]
    output_dir: Path
    total_games: int = 4_000
    concurrency: int = 24
    policy_batch_max_rows: int = 16
    policy_batch_wait_ms: float = 2.0
    native_lane_worker_count: int = 1
    native_option_capacity: int = 2_048
    maximum_engine_steps: int = 10_000
    checkpoint_cache_entries: int = 2
    flush_shard_games: int = 32
    seed: int = 20_260_809
    elo_initial: float = 1_500.0
    elo_k: float = 32.0
    device: str = "cuda"
    compression: str = "zstd"
    act_time: ActTimeConfig = Field(default_factory=ActTimeConfig)

    @field_validator(
        "expected_checkpoint_sha256",
        "expected_public_catalog_manifest_sha256",
        "expected_native_library_sha256",
    )
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require a complete lowercase SHA-256 identity."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            c not in "0123456789abcdef" for c in normalized
        ):
            raise ValueError("artifact identity must be a full SHA-256")
        return normalized

    @field_validator("checkpoint_source_commit", "runner_source_commit")
    @classmethod
    def valid_commit(cls, value: str) -> str:
        """Require a complete hexadecimal source commit."""
        normalized = value.strip().lower()
        if len(normalized) != 40 or any(
            c not in "0123456789abcdef" for c in normalized
        ):
            raise ValueError("source identity must be a full Git commit")
        return normalized

    @field_validator(
        "total_games",
        "concurrency",
        "policy_batch_max_rows",
        "native_lane_worker_count",
        "native_option_capacity",
        "maximum_engine_steps",
        "checkpoint_cache_entries",
        "flush_shard_games",
    )
    @classmethod
    def positive_integer(cls, value: int) -> int:
        """Reject non-positive execution counts."""
        if value <= 0:
            raise ValueError("native deck Elo counts must be positive")
        return value

    @field_validator("policy_batch_wait_ms", "elo_k")
    @classmethod
    def nonnegative_float(cls, value: float) -> float:
        """Reject negative or non-finite operational values."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("native deck Elo float must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def balanced_schedule_is_possible(self) -> NativeDeckEloConfig:
        """Require full pair coverage and mirrored seat blocks."""
        deck_count = len(self.decks)
        if deck_count < 2:
            raise ValueError("native deck Elo requires at least two decks")
        labels = [deck.label for deck in self.decks]
        if len(set(labels)) != deck_count:
            raise ValueError("native deck Elo deck labels must be unique")
        pair_count = deck_count * (deck_count - 1) // 2
        if self.total_games < pair_count * 2 or self.total_games % 2:
            raise ValueError(
                "total_games must cover every pair with a mirrored seat block"
            )
        if self.act_time.planner is not None:
            raise ValueError("native deck Elo does not expose planner lane service")
        return self


@dataclass(frozen=True)
class DeckAsset:
    """Resolved exact deck and its stable human-facing compact identifier."""

    label: str
    path: Path
    deck_digest: str
    deck_hash: str
    deck_signature: str
    bundle: BundleIdentity


@dataclass(frozen=True)
class ScheduledGame:
    """One deterministic game in a mirrored finite schedule."""

    game_index: int
    match_id: str
    deck_a: DeckAsset
    deck_b: DeckAsset
    deck_a_seat: Literal[0, 1]
