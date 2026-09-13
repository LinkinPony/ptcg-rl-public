"""Strict configuration for the S1 replay-root counterfactual audit."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.agent.search.config import MacroSearchConfig
from ptcg_rl.belief.sampling import BeliefSamplerConfig
from ptcg_rl.context import OpponentBeliefFeatureConfig
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.data.kaggle_steps.records import DEFAULT_CHUNK_SIZE
from ptcg_rl.evaluation.search_counterfactual_metrics import (
    CounterfactualMetricReferences,
)


class SearchCounterfactualConfig(BaseModel):
    """Hydra configuration for grouped, immutable S1 replay evidence."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = "v12395_paired_k4_w3_s1_20260711"
    replay_paths: tuple[Path, ...] = ()
    replay_glob: str = (
        "outputs/kaggle_submission_replays/54498922_comfey_v12395/*.json"
    )
    team_name: str | None = "Marshall Maximizer"
    seat_index: int | None = None
    deck_path: Path = Path(
        "docs/experiments/rl_dynamic_deck_pool_20260708/decks/"
        "29_comfey_yveltal_shaymin_4f8e151b4dd0.csv"
    )
    checkpoint_path: Path = Path(
        "outputs/inference_time_search/p0/assets/policy_v12395.pt"
    )
    belief: OpponentBeliefFeatureConfig = Field(
        default_factory=OpponentBeliefFeatureConfig
    )
    sampler: BeliefSamplerConfig = Field(
        default_factory=lambda: BeliefSamplerConfig(mode="archetype")
    )
    macro: MacroSearchConfig = Field(
        default_factory=lambda: MacroSearchConfig(mode="shadow")
    )
    device: str = "cpu"
    precision: Literal["float32"] = "float32"
    include_probe_features: bool = True
    root_quota_seconds: float = 8.0
    max_replays: int | None = 112
    min_roots: int = 500
    max_roots: int = 1_000
    roots_per_phase_per_replay: int = 2
    early_turn_max: int = 3
    mid_turn_max: int = 9
    split_seed: int = 17
    holdout_fraction: float = 0.40
    margin_sweep: tuple[float, ...] = (0.10, 0.15, 0.20)
    seed: int = 0
    chunk_size: int = DEFAULT_CHUNK_SIZE
    compression: str = "zstd"
    output_dir: Path = Path(
        "outputs/evaluation/inference_time_search/"
        "v12395_paired_k4_w3_s1_20260711"
    )
    overwrite: bool = False
    references: CounterfactualMetricReferences = Field(
        default_factory=CounterfactualMetricReferences
    )
    runtime_source_paths: tuple[Path, ...] = (
        Path("src/ptcg_rl/agent/runtime.py"),
        Path("src/ptcg_rl/agent/probe.py"),
        Path("src/ptcg_rl/agent/search/budget.py"),
        Path("src/ptcg_rl/agent/search/candidates.py"),
        Path("src/ptcg_rl/agent/search/config.py"),
        Path("src/ptcg_rl/agent/search/context.py"),
        Path("src/ptcg_rl/agent/search/macro.py"),
        Path("src/ptcg_rl/agent/search/policy_inputs.py"),
        Path("src/ptcg_rl/agent/search/reranker.py"),
        Path("src/ptcg_rl/agent/search/scoring.py"),
    )
    engine_asset_paths: tuple[Path, ...] = (
        Path("data/sample_submission/cg/api.py"),
        Path("data/sample_submission/cg/libcg.so"),
        Path("src/ptcg_rl/engine/effect_types.py"),
        Path("src/ptcg_rl/engine/effects.py"),
        Path("src/ptcg_rl/engine/forward_model.py"),
        Path("src/ptcg_rl/engine/session.py"),
    )

    @field_validator("experiment_id")
    @classmethod
    def immutable_experiment_id(cls, value: str) -> str:
        """Reject empty or moving S1 campaign labels before creating output."""
        normalized = value.strip()
        if not normalized or "latest" in normalized.lower():
            raise ValueError("experiment_id must be non-empty and cannot contain latest")
        return normalized

    @field_validator(
        "min_roots",
        "max_roots",
        "roots_per_phase_per_replay",
        "chunk_size",
    )
    @classmethod
    def positive_limits(cls, value: int) -> int:
        """Require positive streaming and root-selection bounds."""
        if value <= 0:
            raise ValueError("counterfactual audit limits must be positive")
        return value

    @field_validator("max_replays")
    @classmethod
    def optional_positive(cls, value: int | None) -> int | None:
        """Reject non-positive optional replay limits."""
        if value is not None and value <= 0:
            raise ValueError("max_replays must be positive")
        return value

    @field_validator("seat_index")
    @classmethod
    def valid_seat(cls, value: int | None) -> int | None:
        """Restrict an explicit replay seat."""
        if value is not None and value not in (0, 1):
            raise ValueError("seat_index must be 0 or 1")
        return value

    @field_validator("root_quota_seconds")
    @classmethod
    def positive_quota(cls, value: float) -> float:
        """Require a finite positive offline root quota."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("root_quota_seconds must be finite and positive")
        return value

    @field_validator("holdout_fraction")
    @classmethod
    def valid_holdout_fraction(cls, value: float) -> float:
        """Require non-empty dev and holdout episode partitions."""
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("holdout_fraction must be in (0, 1)")
        return value

    @field_validator("margin_sweep")
    @classmethod
    def valid_margin_sweep(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Require unique finite non-negative dev-only margins."""
        if not value or len(set(value)) != len(value):
            raise ValueError("margin_sweep must be non-empty and unique")
        if any(not math.isfinite(margin) or margin < 0.0 for margin in value):
            raise ValueError("margin_sweep values must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def coherent_audit(self) -> SearchCounterfactualConfig:
        """Keep the S1 audit shadow-only and sufficiently critic-complete."""
        if self.max_roots < self.min_roots:
            raise ValueError("max_roots must be >= min_roots")
        if self.mid_turn_max < self.early_turn_max:
            raise ValueError("mid_turn_max must be >= early_turn_max")
        if self.macro.mode != "shadow":
            raise ValueError("S1 counterfactual audit requires macro.mode=shadow")
        if self.macro.rerank.score_mode == "engine_only":
            raise ValueError("S1 critic audit requires critic leaf evaluation")
        if self.references.min_roots > self.max_roots:
            raise ValueError("metric min_roots cannot exceed max_roots")
        belief_path = self.belief.deck_signature_summary_path
        sampler_path = self.sampler.prior_deck_signature_summary_path
        if (
            belief_path is not None
            and sampler_path is not None
            and records.repo_path(belief_path) != records.repo_path(sampler_path)
        ):
            raise ValueError("belief and sampler prior assets must match for S1")
        return self
