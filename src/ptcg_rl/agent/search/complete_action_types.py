"""Configuration and evidence records for complete-action teachers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.agent.search.continuation_types import (
    CompleteContinuationPlan,
    ContinuationValueRequest,
)
from ptcg_rl.agent.search.macro import MacroEndpoint
from ptcg_rl.agent.search.prompt_actions import PromptActionCandidates
from ptcg_rl.engine.session import SearchSession


class ActorPerspectiveLeafValues(Protocol):
    """Batched legal-perspective value callback, independent of model class."""

    def __call__(
        self,
        requests: Sequence[ContinuationValueRequest],
    ) -> Sequence[float]:
        """Return actor-perspective values aligned with explicit deck routes."""


class WorldSessionFactory(Protocol):
    """Open exactly one process-global Search lifecycle for a world."""

    def __call__(self, world_index: int) -> AbstractContextManager[SearchSession]:
        """Return a fresh, unopened-by-others context manager."""


class CompleteActionTeacherConfig(BaseModel):
    """Bounded scoring and paired-world aggregation for auxiliary targets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root_exhaustive_action_cap: int = 128
    root_beam_width: int = 8
    total_node_cap: int = 8192
    joint_strategy_cap: int = 4096
    leaf_value_batch_size: int = 256
    risk_std_weight: float = 0.25
    engine_tiebreak_weight: float = 0.10
    score_clip: float = 1.0
    confidence_temperature: float = 0.25

    @field_validator(
        "root_exhaustive_action_cap",
        "root_beam_width",
        "total_node_cap",
        "joint_strategy_cap",
        "leaf_value_batch_size",
    )
    @classmethod
    def positive_integer(cls, value: int) -> int:
        """Require finite search work units."""
        if value <= 0:
            raise ValueError("complete-action teacher limits must be positive")
        return value

    @field_validator("risk_std_weight", "engine_tiebreak_weight")
    @classmethod
    def non_negative_weight(cls, value: float) -> float:
        """Require finite non-negative score weights."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("complete-action teacher weights must be finite")
        return value

    @field_validator("score_clip", "confidence_temperature")
    @classmethod
    def positive_float(cls, value: float) -> float:
        """Require positive finite normalization constants."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("teacher normalization constants must be positive")
        return value


@dataclass(frozen=True)
class WorldRootPlan:
    """One root action's continuation tree in one determinized world."""

    world_index: int
    root_action: tuple[int, ...]
    plan: CompleteContinuationPlan


@dataclass(frozen=True)
class ContinuationLeafScore:
    """Engine and value evidence for one completed continuation leaf."""

    world_index: int
    root_action: tuple[int, ...]
    action_path: tuple[tuple[int, ...], ...]
    endpoint: MacroEndpoint
    engine_score: float
    actor_value: float | None
    root_value_sign: int | None
    leaf_value: float | None
    score: float


@dataclass(frozen=True)
class RootActionTeacherScore:
    """Paired-world score after optimizing retained continuations per world."""

    root_action: tuple[int, ...]
    world_scores: tuple[float, ...]
    continuation_paths: tuple[tuple[tuple[int, ...], ...], ...]
    mean_score: float
    score_std: float
    robust_score: float
    strategy_consistent: bool


@dataclass(frozen=True)
class CompleteActionTeacherTarget:
    """Producer-only target; callers must keep the executed behavior unchanged."""

    behavior_action: tuple[int, ...]
    target_action: tuple[int, ...]
    provisional_action: tuple[int, ...]
    valid: bool
    reason: str
    confidence: float
    coverage: float
    score_margin: float
    root_candidates: PromptActionCandidates
    action_scores: tuple[RootActionTeacherScore, ...]
    leaf_scores: tuple[ContinuationLeafScore, ...]
    plans: tuple[WorldRootPlan, ...]
    worlds: int
    nodes_expanded: int

    @property
    def target_differs_from_behavior(self) -> bool:
        """Report a training target delta without serving it as an action."""
        return self.valid and self.target_action != self.behavior_action


__all__ = [
    "CompleteActionTeacherConfig",
    "CompleteActionTeacherTarget",
    "ContinuationLeafScore",
    "ActorPerspectiveLeafValues",
    "RootActionTeacherScore",
    "WorldRootPlan",
    "WorldSessionFactory",
]
