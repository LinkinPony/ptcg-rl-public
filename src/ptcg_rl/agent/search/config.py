"""Strict configuration for inference-time search runtime components."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.belief.sampling import BeliefSamplerConfig

SearchMode = Literal["disabled", "shadow", "override", "conditioned"]
ScoreMode = Literal["engine_only", "engine_value_tiebreak", "value_primary"]
HandoffScoreMode = Literal["engine_only", "root_value_adapter"]


class EngineTacticalScoreConfig(BaseModel):
    """Bounded utility weights for structured engine consequence logs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    damage_scale: float = 400.0
    damage_weight: float = 0.60
    knockout_weight: float = 0.50
    prize_weight: float = 0.35
    draw_scale: float = 10.0
    draw_weight: float = 0.10
    energy_scale: float = 10.0
    energy_weight: float = 0.10
    status_weight: float = 0.05
    score_clip: float = 0.95

    @field_validator(
        "damage_scale",
        "draw_scale",
        "energy_scale",
        "score_clip",
    )
    @classmethod
    def finite_positive(cls, value: float) -> float:
        """Require positive normalization and clipping constants."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "tactical score scales and clip must be finite and positive"
            )
        return value

    @field_validator(
        "damage_weight",
        "knockout_weight",
        "prize_weight",
        "draw_weight",
        "energy_weight",
        "status_weight",
    )
    @classmethod
    def finite_weight(cls, value: float) -> float:
        """Require finite tactical weights while permitting signed ablations."""
        if not math.isfinite(value):
            raise ValueError("tactical score weights must be finite")
        return value


class PairedRerankConfig(BaseModel):
    """Risk and calibration limits for a paired-world action switch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    score_mode: ScoreMode = "engine_value_tiebreak"
    handoff_score_mode: HandoffScoreMode = "engine_only"
    switch_margin: float = 0.15
    risk_std_weight: float = 0.5
    downside_fraction: float = 0.25
    minimum_downside_delta: float = -0.20
    value_tiebreak_weight: float = 0.10
    engine_tiebreak_weight: float = 0.10
    combined_score_clip: float | None = None
    tactical: EngineTacticalScoreConfig = Field(
        default_factory=EngineTacticalScoreConfig
    )

    @field_validator(
        "switch_margin",
        "risk_std_weight",
        "value_tiebreak_weight",
        "engine_tiebreak_weight",
    )
    @classmethod
    def finite_non_negative(cls, value: float) -> float:
        """Reject negative or non-finite scoring parameters."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "rerank weights and margin must be finite and non-negative"
            )
        return value

    @field_validator("minimum_downside_delta")
    @classmethod
    def finite_downside(cls, value: float) -> float:
        """Require a finite downside threshold."""
        if not math.isfinite(value):
            raise ValueError("minimum_downside_delta must be finite")
        return value

    @field_validator("combined_score_clip")
    @classmethod
    def positive_combined_score_clip(cls, value: float | None) -> float | None:
        """Validate an optional final score clip for conditioned evidence."""
        if value is None:
            return None
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("combined_score_clip must be finite and positive")
        return value

    @field_validator("downside_fraction")
    @classmethod
    def valid_downside_fraction(cls, value: float) -> float:
        """Restrict CVaR mass to a non-empty fraction of paired worlds."""
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("downside_fraction must be in (0, 1]")
        return value

    @model_validator(mode="after")
    def mode_has_required_weight(self) -> PairedRerankConfig:
        """Prevent a value mode from silently ignoring its intended signal."""
        if (
            self.score_mode == "engine_value_tiebreak"
            and self.value_tiebreak_weight <= 0.0
        ):
            raise ValueError("engine_value_tiebreak requires value_tiebreak_weight > 0")
        return self


class SearchBudgetConfig(BaseModel):
    """Whole-act deadline and cumulative search-bank limits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bank_limit_seconds: float = 180.0
    hard_reserve_seconds: float = 180.0
    taper_window_seconds: float = 120.0
    uninterruptible_guard_seconds: float = 1.5
    return_guard_seconds: float = 0.1
    whole_act_limit_seconds: float = 12.0
    ordinary_main_quota_seconds: float = 2.5
    first_high_value_main_quota_seconds: float = 8.0
    later_high_value_main_quota_seconds: float = 2.0

    @field_validator(
        "bank_limit_seconds",
        "hard_reserve_seconds",
        "taper_window_seconds",
        "uninterruptible_guard_seconds",
        "return_guard_seconds",
        "whole_act_limit_seconds",
        "ordinary_main_quota_seconds",
        "first_high_value_main_quota_seconds",
        "later_high_value_main_quota_seconds",
    )
    @classmethod
    def non_negative_seconds(cls, value: float) -> float:
        """Reject negative or non-finite time limits."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("search time limits must be non-negative")
        return value

    @model_validator(mode="after")
    def valid_deadline_guards(self) -> SearchBudgetConfig:
        """Require a usable taper and whole-act interval."""
        if self.taper_window_seconds <= 0.0:
            raise ValueError("taper_window_seconds must be positive")
        minimum_limit = self.return_guard_seconds + self.uninterruptible_guard_seconds
        if self.whole_act_limit_seconds <= minimum_limit:
            raise ValueError(
                "whole_act_limit_seconds must exceed the return and call guards"
            )
        return self


class MacroSearchConfig(BaseModel):
    """Paired-world same-turn macro search and rerank limits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: SearchMode = "disabled"
    worlds: int = 3
    top_k: int = 4
    exploration_actions: int = 1
    forced_step_cap: int = 12
    continuation_step_cap: int = 8
    node_cap: int = 64
    manual_coin: bool = False
    root_inference_cache_enabled: bool = False
    leaf_value_batch_size: int = 1
    rerank: PairedRerankConfig = Field(default_factory=PairedRerankConfig)
    budget: SearchBudgetConfig = Field(default_factory=SearchBudgetConfig)

    @field_validator(
        "worlds",
        "top_k",
        "forced_step_cap",
        "continuation_step_cap",
        "node_cap",
        "leaf_value_batch_size",
    )
    @classmethod
    def positive_limits(cls, value: int) -> int:
        """Reject non-positive search bounds."""
        if value <= 0:
            raise ValueError("macro search limits must be positive")
        return value

    @field_validator("exploration_actions")
    @classmethod
    def non_negative_exploration(cls, value: int) -> int:
        """Allow disabling exploration while rejecting negative counts."""
        if value < 0:
            raise ValueError("exploration_actions must be non-negative")
        return value


class SearchRuntimeConfig(BaseModel):
    """Act-time probe plus optional P0 macro-shadow configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    conservative_override_enabled: bool = True
    worlds: int = 3
    top_k: int = 4
    manual_coin: bool = False
    sampler: BeliefSamplerConfig = Field(
        default_factory=lambda: BeliefSamplerConfig(mode="archetype")
    )
    macro: MacroSearchConfig = Field(default_factory=MacroSearchConfig)

    @field_validator("worlds", "top_k")
    @classmethod
    def positive_probe_limits(cls, value: int) -> int:
        """Reject non-positive probe limits."""
        if value <= 0:
            raise ValueError("probe limits must be positive")
        return value


# Keep the historical public name used by runtime and evaluation configs.
ActTimeSearchConfig = SearchRuntimeConfig
