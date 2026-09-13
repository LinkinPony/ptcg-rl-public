"""Validated configuration for the offline candidate-regret audit."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.agent.search.candidate_budget import CandidateConstructorConfig
from ptcg_rl.agent.search.planner_scoring import PlannerScoringConfig

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CandidateRegretSamplingConfig(BaseModel):
    """Bounded two-pass sampling over replay Parquet roots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_strata: int
    roots_per_stratum: int
    max_roots: int
    min_legal_actions: int = 2
    exhaustive_reference_cap: int
    batch_size: int = 65_536
    seed: str

    @field_validator(
        "max_strata",
        "roots_per_stratum",
        "max_roots",
        "min_legal_actions",
        "exhaustive_reference_cap",
        "batch_size",
    )
    @classmethod
    def positive_integer(cls, value: int) -> int:
        """Require explicit positive sampling bounds."""
        if value <= 0:
            raise ValueError("candidate audit sampling bounds must be positive")
        return value

    @field_validator("seed")
    @classmethod
    def nonempty_seed(cls, value: str) -> str:
        """Reject an ambiguous empty deterministic seed."""
        canonical = value.strip()
        if not canonical:
            raise ValueError("candidate audit sampling seed must not be empty")
        return canonical

    @model_validator(mode="after")
    def coherent_sample_bound(self) -> Self:
        """Keep the retained locator set statically bounded."""
        if self.max_roots > self.max_strata * self.roots_per_stratum:
            raise ValueError("max_roots exceeds the stratified reservoir capacity")
        if self.min_legal_actions > self.exhaustive_reference_cap:
            raise ValueError("minimum legal actions exceeds the exhaustive cap")
        return self


class CandidateRegretEngineConfig(BaseModel):
    """Native exact-transition and paired-scenario resource limits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    library_path: Path
    scenario_count: int
    max_cells: int
    max_engine_steps: int
    max_forced_steps: int
    max_observation_bytes: int
    fallback_card_id: int = 1
    fallback_basic_pokemon_id: int = 463

    @field_validator(
        "scenario_count",
        "max_cells",
        "max_engine_steps",
        "max_observation_bytes",
        "fallback_card_id",
        "fallback_basic_pokemon_id",
    )
    @classmethod
    def positive_integer(cls, value: int) -> int:
        """Reject unusable native execution limits."""
        if value <= 0:
            raise ValueError("candidate audit engine limits must be positive")
        return value

    @field_validator("max_forced_steps")
    @classmethod
    def nonnegative_forced_steps(cls, value: int) -> int:
        """Permit a root-only audit while rejecting a negative cap."""
        if value < 0:
            raise ValueError("max_forced_steps must be non-negative")
        return value


class CandidateRegretCostModel(BaseModel):
    """Profiled generic costs supplied to the production budget policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    estimated_engine_steps_per_cell: int
    estimated_prefix_nodes_per_candidate: int
    estimated_cell_time_us: int

    @field_validator(
        "estimated_engine_steps_per_cell",
        "estimated_prefix_nodes_per_candidate",
        "estimated_cell_time_us",
    )
    @classmethod
    def positive_integer(cls, value: int) -> int:
        """Require positive profiled costs."""
        if value <= 0:
            raise ValueError("candidate audit cost estimates must be positive")
        return value


class CandidateRegretAuditConfig(BaseModel):
    """Hydra-facing diagnostic campaign contract."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    steps_globs: tuple[str, ...]
    checkpoint_path: Path
    expected_checkpoint_sha256: str
    device: str = "cuda"
    output_dir: Path
    compression: str = "zstd"
    output_shard_roots: int = 8
    epsilon: float
    k_values: tuple[int, ...]
    controller_version: str
    proposal_source_mode: Literal["base_zero_residual_parity"]
    sampling: CandidateRegretSamplingConfig
    engine: CandidateRegretEngineConfig
    constructor: CandidateConstructorConfig
    cost_model: CandidateRegretCostModel
    scorer: PlannerScoringConfig
    hydra: Mapping[str, Any] | None = None

    @field_validator(
        "experiment_id",
        "device",
        "compression",
        "controller_version",
    )
    @classmethod
    def nonempty_string(cls, value: str) -> str:
        """Require immutable, explicit runtime identities."""
        canonical = value.strip()
        if not canonical or "latest" in canonical.lower():
            raise ValueError("audit identities must be non-empty and immutable")
        return canonical

    @field_validator("expected_checkpoint_sha256")
    @classmethod
    def valid_checkpoint_sha256(cls, value: str) -> str:
        """Bind the scorer to one immutable checkpoint."""
        if _SHA256.fullmatch(value) is None:
            raise ValueError("expected checkpoint SHA-256 must be lowercase hex")
        return value

    @field_validator("steps_globs")
    @classmethod
    def nonempty_globs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require at least one replay Parquet input pattern."""
        if not value or any(not pattern for pattern in value):
            raise ValueError("steps_globs must be non-empty")
        return value

    @field_validator("output_shard_roots")
    @classmethod
    def positive_shard_roots(cls, value: int) -> int:
        """Require a bounded positive output shard size."""
        if value <= 0:
            raise ValueError("output_shard_roots must be positive")
        return value

    @field_validator("epsilon")
    @classmethod
    def valid_epsilon(cls, value: float) -> float:
        """Require a finite non-negative near-best tolerance."""
        if not 0.0 <= value < float("inf"):
            raise ValueError("epsilon must be finite and non-negative")
        return value

    @field_validator("k_values")
    @classmethod
    def valid_k_values(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Require a deterministic strictly increasing K curve."""
        if not value or any(item <= 0 for item in value):
            raise ValueError("k_values must contain positive integers")
        if tuple(sorted(set(value))) != value:
            raise ValueError("k_values must be strictly increasing and unique")
        return value

    @model_validator(mode="after")
    def coherent_campaign(self) -> Self:
        """Prove exhaustive grids and diagnostic curves fit declared bounds."""
        maximum_cells = (
            self.sampling.exhaustive_reference_cap * self.engine.scenario_count
        )
        if maximum_cells > self.engine.max_cells:
            raise ValueError("exhaustive reference grid exceeds engine max_cells")
        required_steps = maximum_cells * (self.engine.max_forced_steps + 1)
        if required_steps > self.engine.max_engine_steps:
            raise ValueError("engine step cap cannot cover the exhaustive grid")
        if self.k_values[-1] > self.constructor.k_total:
            raise ValueError("k_values cannot exceed constructor k_total")
        if self.output_shard_roots > self.sampling.max_roots:
            raise ValueError("output shard size exceeds the total root bound")
        return self


__all__ = [
    "CandidateRegretAuditConfig",
    "CandidateRegretCostModel",
    "CandidateRegretEngineConfig",
    "CandidateRegretSamplingConfig",
]
