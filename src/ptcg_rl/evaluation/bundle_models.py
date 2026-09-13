"""Typed configuration and plans for exact bundle evaluation."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.agent.runtime import ActTimeConfig
from ptcg_rl.agent.search.budget import ActTimeLedgerConfig
from ptcg_rl.training.arena import GamePlan
from ptcg_rl.training.arena_decks import ArenaDeck
from ptcg_rl.training.run_config import TrainingRunConfig

BundleAgentKind = Literal[
    "policy_greedy",
    "runtime",
    "simple_stateless_greedy",
    "registered",
    "release",
]
BundleEvaluationStage = Literal["S2", "S3", "S4", "S5"]


class BundleAgentConfig(BaseModel):
    """Controller configuration for one exact evaluation bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: BundleAgentKind
    checkpoint_path: Path | None = None
    registered_name: str | None = None
    release_manifest_path: Path | None = None
    device: str = "cpu"
    belief_summary_path: Path | None = None
    public_catalog_manifest_path: Path | None = None
    act_time: ActTimeConfig = Field(default_factory=ActTimeConfig)

    @field_validator("device")
    @classmethod
    def nonempty_device(cls, value: str) -> str:
        """Reject empty inference-device strings."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("bundle agent device must be non-empty")
        return cleaned

    @field_validator("registered_name")
    @classmethod
    def nonempty_registered_name(cls, value: str | None) -> str | None:
        """Reject an explicitly blank registry name."""
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("registered_name must be non-empty when set")
        return cleaned

    @model_validator(mode="after")
    def valid_kind_fields(self) -> BundleAgentConfig:
        """Require the identity fields needed by the selected controller."""
        if self.kind in {
            "policy_greedy",
            "runtime",
            "simple_stateless_greedy",
        }:
            if self.checkpoint_path is None:
                raise ValueError(f"{self.kind} requires checkpoint_path")
            if self.registered_name is not None:
                raise ValueError(f"{self.kind} does not accept registered_name")
            if self.release_manifest_path is not None:
                raise ValueError(f"{self.kind} does not accept release_manifest_path")
        elif self.kind == "registered":
            if self.registered_name is None:
                raise ValueError("registered agent requires registered_name")
            if self.release_manifest_path is not None:
                raise ValueError(
                    "registered agent does not accept release_manifest_path"
                )
        else:
            if self.release_manifest_path is None:
                raise ValueError("release agent requires release_manifest_path")
            if self.checkpoint_path is not None or self.registered_name is not None:
                raise ValueError(
                    "release agent does not accept checkpoint_path or registered_name"
                )
            if any(
                "latest" in part.lower() for part in self.release_manifest_path.parts
            ):
                raise ValueError("release agent cannot reference a latest alias")
        if (
            self.kind == "simple_stateless_greedy"
            and self.public_catalog_manifest_path is None
        ):
            raise ValueError(
                "simple_stateless_greedy requires public_catalog_manifest_path"
            )
        if (
            self.kind != "simple_stateless_greedy"
            and self.public_catalog_manifest_path is not None
        ):
            raise ValueError(
                "public_catalog_manifest_path is reserved for simple_stateless_greedy"
            )
        if self.kind == "release" and self.belief_summary_path is not None:
            raise ValueError("release agent owns its packaged belief asset")
        return self


class EvaluationBundleConfig(BaseModel):
    """Hydra-facing definition of one deck-and-pilot bundle.

    ``variant_weight`` is a conditional controller share among bundles with
    the same exact deck signature. It must not be applied a second time to the
    exact-signature prevalence produced by the metagame model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str
    pilot_id: str
    archetype: str
    deck_path: Path
    variant_weight: float = 1.0
    agent: BundleAgentConfig

    @field_validator("bundle_id", "pilot_id", "archetype")
    @classmethod
    def nonempty_identity(cls, value: str) -> str:
        """Require stable non-empty bundle metadata."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("bundle identity fields must be non-empty")
        return cleaned

    @field_validator("variant_weight")
    @classmethod
    def positive_variant_weight(cls, value: float) -> float:
        """Require a finite positive controller-mixture weight."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("variant_weight must be finite and positive")
        return value


class BundleScheduleConfig(BaseModel):
    """Balanced game counts for exact candidate-opponent cells."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    games_per_matchup: int = 20
    games_per_opponent: dict[str, int] = Field(default_factory=dict)
    mirror_sides: bool = True

    @field_validator("games_per_matchup")
    @classmethod
    def positive_games_per_matchup(cls, value: int) -> int:
        """Require a positive default game count."""
        if value <= 0:
            raise ValueError("games_per_matchup must be positive")
        return value

    @field_validator("games_per_opponent")
    @classmethod
    def positive_overrides(cls, value: dict[str, int]) -> dict[str, int]:
        """Require positive opponent-specific game counts."""
        invalid = {key: count for key, count in value.items() if count <= 0}
        if invalid:
            raise ValueError(f"games_per_opponent must be positive: {invalid}")
        return value


class BundleGauntletConfig(BaseModel):
    """Hydra-backed exact-bundle gauntlet configuration."""

    model_config = ConfigDict(extra="forbid")

    candidates: tuple[EvaluationBundleConfig, ...]
    opponents: tuple[EvaluationBundleConfig, ...]
    protocol: str = "bundle-gauntlet-v1"
    experiment_id: str | None = None
    stage: BundleEvaluationStage | None = None
    schedule: BundleScheduleConfig = Field(default_factory=BundleScheduleConfig)
    run: TrainingRunConfig = Field(default_factory=TrainingRunConfig)
    output_dir: Path | None = None
    num_workers: int = 1
    max_steps_per_game: int = 1_000
    result_shard_size: int = 16
    run_timeout_seconds: float | None = None
    seed: int = 0
    compression: str = "zstd"
    fail_on_error: bool = False
    act_time_ledger: ActTimeLedgerConfig = Field(default_factory=ActTimeLedgerConfig)

    @field_validator("num_workers", "max_steps_per_game", "result_shard_size")
    @classmethod
    def positive_runner_limit(cls, value: int) -> int:
        """Require positive worker and step limits."""
        if value <= 0:
            raise ValueError("bundle gauntlet runner limits must be positive")
        return value

    @field_validator("protocol")
    @classmethod
    def immutable_protocol(cls, value: str) -> str:
        """Require a stable non-moving protocol identifier."""
        normalized = value.strip()
        if not normalized or "latest" in normalized.lower():
            raise ValueError("protocol must be non-empty and cannot contain latest")
        return normalized

    @field_validator("experiment_id")
    @classmethod
    def immutable_experiment_id(cls, value: str | None) -> str | None:
        """Reject mutable formal experiment labels."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or "latest" in normalized.lower():
            raise ValueError(
                "experiment_id must be non-empty and cannot contain latest"
            )
        return normalized

    @field_validator("run_timeout_seconds")
    @classmethod
    def positive_run_timeout(cls, value: float | None) -> float | None:
        """Require a finite positive per-game wall timeout when enabled."""
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError("run_timeout_seconds must be finite and positive")
        return value

    @model_validator(mode="after")
    def valid_bundle_suite(self) -> BundleGauntletConfig:
        """Validate exact IDs, supported roles, and balanced seat counts."""
        if not self.candidates:
            raise ValueError("bundle gauntlet requires at least one candidate")
        if not self.opponents:
            raise ValueError("bundle gauntlet requires at least one opponent")
        if (self.experiment_id is None) != (self.stage is None):
            raise ValueError("formal bundle identity requires experiment_id and stage")
        _require_unique_ids(self.candidates, role="candidate")
        _require_unique_ids(self.opponents, role="opponent")
        candidate_by_id = {bundle.bundle_id: bundle for bundle in self.candidates}
        opponent_by_id = {bundle.bundle_id: bundle for bundle in self.opponents}
        conflicting_shared_ids = sorted(
            bundle_id
            for bundle_id in candidate_by_id.keys() & opponent_by_id.keys()
            if candidate_by_id[bundle_id] != opponent_by_id[bundle_id]
        )
        if conflicting_shared_ids:
            raise ValueError(
                "bundle IDs shared across roles must have identical definitions: "
                f"{conflicting_shared_ids}"
            )
        opponent_ids = {bundle.bundle_id for bundle in self.opponents}
        unknown_overrides = set(self.schedule.games_per_opponent) - opponent_ids
        if unknown_overrides:
            raise ValueError(
                "games_per_opponent contains unknown bundle IDs: "
                f"{sorted(unknown_overrides)}"
            )
        if self.schedule.mirror_sides:
            counts = [
                self.schedule.games_per_opponent.get(
                    opponent.bundle_id,
                    self.schedule.games_per_matchup,
                )
                for opponent in self.opponents
            ]
            if any(count % 2 != 0 for count in counts):
                raise ValueError("mirrored bundle matchups require even game counts")
        return self


@dataclass(frozen=True)
class EvaluationBundle:
    """Resolved bundle with its validated 60-card arena deck."""

    config: EvaluationBundleConfig
    deck: ArenaDeck


@dataclass(frozen=True)
class BundleGamePlan:
    """One exact-bundle game in a candidate-independent seat stratum."""

    arena_plan: GamePlan
    candidate: EvaluationBundle
    opponent: EvaluationBundle
    repeat_index: int
    matched_block_index: int
    matched_block_id: str
    matched_block_seed: int

    @property
    def stratum_id(self) -> str:
        """Backward-readable alias for the matched block identifier."""
        return self.matched_block_id

    @property
    def stratum_seed(self) -> int:
        """Backward-readable alias for the matched Python RNG seed."""
        return self.matched_block_seed


def _require_unique_ids(
    bundles: Sequence[EvaluationBundleConfig],
    *,
    role: str,
) -> None:
    counts = Counter(bundle.bundle_id for bundle in bundles)
    duplicates = sorted(bundle_id for bundle_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {role} bundle IDs: {duplicates}")


__all__ = [
    "BundleAgentConfig",
    "BundleAgentKind",
    "BundleGamePlan",
    "BundleGauntletConfig",
    "BundleScheduleConfig",
    "EvaluationBundle",
    "EvaluationBundleConfig",
]
