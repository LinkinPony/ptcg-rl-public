"""Strict configuration for public-teacher engine sibling evaluation."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.agent.search.config import EngineTacticalScoreConfig
from ptcg_rl.belief.sampling import BeliefSamplerConfig
from ptcg_rl.context import OpponentBeliefFeatureConfig

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class TeacherSiblingDataConfig(BaseModel):
    """Immutable public-pilot step source and bounded streaming controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_path: Path
    expected_manifest_sha256: str | None = None
    split_column: str = "teacher_split"
    split_value: str = "validation"
    expected_behavior_kind: str = "public_teacher"
    expected_deck_signature: str
    read_batch_size: int = 256

    @field_validator("expected_manifest_sha256")
    @classmethod
    def valid_optional_sha256(cls, value: str | None) -> str | None:
        """Reject malformed optional manifest identities."""
        if value is not None and not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("expected_manifest_sha256 must be lowercase SHA-256")
        return value

    @field_validator(
        "split_column",
        "split_value",
        "expected_behavior_kind",
        "expected_deck_signature",
    )
    @classmethod
    def nonempty_identity(cls, value: str) -> str:
        """Require explicit data filters rather than moving defaults."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("teacher sibling data identities must be non-empty")
        return normalized

    @field_validator("read_batch_size")
    @classmethod
    def positive_batch_size(cls, value: int) -> int:
        """Require bounded positive Parquet reads."""
        if value <= 0:
            raise ValueError("read_batch_size must be positive")
        return value


class TeacherSiblingCheckpointConfig(BaseModel):
    """One frozen policy/value checkpoint used in the paired comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tag: str
    path: Path
    expected_sha256: str | None = None

    @field_validator("tag")
    @classmethod
    def immutable_tag(cls, value: str) -> str:
        """Reject blank or moving checkpoint labels."""
        normalized = value.strip()
        if not normalized or "latest" in normalized.lower():
            raise ValueError("checkpoint tag must be immutable and non-empty")
        return normalized

    @field_validator("expected_sha256")
    @classmethod
    def valid_optional_sha256(cls, value: str | None) -> str | None:
        """Reject malformed optional checkpoint identities."""
        if value is not None and not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("expected_sha256 must be lowercase SHA-256")
        return value


class TeacherSiblingReferences(BaseModel):
    """Pre-registered integrity and paired-ranking reference bands."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_roots: int = 200
    min_comparable_pairs: int = 200
    min_leaf_coverage_rate: float = 0.98
    min_engine_score_coverage_rate: float = 0.50
    min_teacher_candidate_recall: float = 1.0
    min_pairwise_accuracy_improvement: float = 0.01
    max_top1_regret_increase: float = 0.005
    min_teacher_mrr_improvement: float = 0.01
    min_teacher_pairwise_improvement: float = 0.01

    @field_validator("min_roots", "min_comparable_pairs")
    @classmethod
    def positive_counts(cls, value: int) -> int:
        """Require non-trivial evidence counts."""
        if value <= 0:
            raise ValueError("teacher sibling sample references must be positive")
        return value

    @field_validator(
        "min_leaf_coverage_rate",
        "min_engine_score_coverage_rate",
        "min_teacher_candidate_recall",
    )
    @classmethod
    def probability_reference(cls, value: float) -> float:
        """Restrict coverage references to probabilities."""
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("teacher sibling coverage references must be in [0, 1]")
        return value

    @field_validator(
        "min_pairwise_accuracy_improvement",
        "max_top1_regret_increase",
        "min_teacher_mrr_improvement",
        "min_teacher_pairwise_improvement",
    )
    @classmethod
    def finite_nonnegative_delta(cls, value: float) -> float:
        """Require finite, one-sided benefit margins."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("teacher sibling benefit margins must be non-negative")
        return value


class TeacherSiblingConfig(BaseModel):
    """Hydra-facing engine sibling dataset and comparison campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str
    data: TeacherSiblingDataConfig
    initial_checkpoint: TeacherSiblingCheckpointConfig
    trained_checkpoint: TeacherSiblingCheckpointConfig
    device: str = "cuda"
    worlds: int = 3
    policy_top_k: int = 4
    max_roots: int = 256
    reservoir_size: int = 1024
    seed: int = 20260712
    manual_coin: bool = False
    forced_step_cap: int = 12
    node_cap: int = 13
    belief: OpponentBeliefFeatureConfig = OpponentBeliefFeatureConfig()
    sampler: BeliefSamplerConfig = BeliefSamplerConfig()
    tactical: EngineTacticalScoreConfig = EngineTacticalScoreConfig()
    references: TeacherSiblingReferences = TeacherSiblingReferences()
    behavior_kind: Literal["engine_sibling_supervised"] = (
        "engine_sibling_supervised"
    )
    compression: str = "zstd"
    output_dir: Path

    @field_validator("experiment_id")
    @classmethod
    def immutable_experiment_id(cls, value: str) -> str:
        """Reject blank and moving campaign labels."""
        normalized = value.strip()
        if not normalized or "latest" in normalized.lower():
            raise ValueError("experiment_id must be immutable and non-empty")
        return normalized

    @field_validator("device", "compression")
    @classmethod
    def nonempty_runtime_string(cls, value: str) -> str:
        """Require explicit runtime settings."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("runtime strings must be non-empty")
        return normalized

    @field_validator(
        "worlds",
        "policy_top_k",
        "max_roots",
        "reservoir_size",
        "forced_step_cap",
        "node_cap",
    )
    @classmethod
    def positive_limits(cls, value: int) -> int:
        """Require positive bounded search and sample limits."""
        if value <= 0:
            raise ValueError("teacher sibling limits must be positive")
        return value

    @model_validator(mode="after")
    def coherent_campaign(self) -> TeacherSiblingConfig:
        """Keep sampling, transition, and checkpoint identities coherent."""
        if self.reservoir_size < self.max_roots:
            raise ValueError("reservoir_size must be >= max_roots")
        if self.node_cap < self.forced_step_cap + 1:
            raise ValueError("node_cap must cover root plus forced steps")
        if self.references.min_roots > self.max_roots:
            raise ValueError("reference min_roots cannot exceed max_roots")
        if self.initial_checkpoint.path == self.trained_checkpoint.path:
            raise ValueError("initial and trained checkpoint paths must differ")
        return self


__all__ = [
    "TeacherSiblingCheckpointConfig",
    "TeacherSiblingConfig",
    "TeacherSiblingDataConfig",
    "TeacherSiblingReferences",
]
