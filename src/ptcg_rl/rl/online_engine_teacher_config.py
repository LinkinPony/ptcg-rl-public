"""Configuration for bounded online engine-grounded supervision."""

from __future__ import annotations

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.agent.search.complete_action_types import CompleteActionTeacherConfig
from ptcg_rl.agent.search.continuation_types import ContinuationPlannerLimits
from ptcg_rl.belief.sampling import BeliefSamplerConfig


class OnlineEngineTeacherConfig(BaseModel):
    """Hydra/Pydantic configuration for bounded actor-side improvement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    main_probability: float = 0.002
    strategic_probability: float = 0.02
    worlds: int = 2
    deadline_seconds: float = 0.65
    max_attempts_per_actor_step: int = 1
    max_teacher_seconds_per_actor_step: float = 0.75
    inference_timeout_seconds: float = 0.10
    minimum_inference_budget_seconds: float = 0.02
    async_actor: bool = False
    async_queue_batches: int = 4
    worker_startup_timeout_seconds: float = 10.0
    worker_kill_reap_timeout_seconds: float = 0.5
    worker_max_startup_attempts: int = 3
    manual_coin: bool = True
    target_weight: float = 1.0
    sampler: BeliefSamplerConfig = Field(
        default_factory=lambda: BeliefSamplerConfig(mode="archetype")
    )
    planner: ContinuationPlannerLimits = Field(
        default_factory=ContinuationPlannerLimits
    )
    teacher: CompleteActionTeacherConfig = Field(
        default_factory=CompleteActionTeacherConfig
    )

    @field_validator("main_probability", "strategic_probability")
    @classmethod
    def valid_probability(cls, value: float) -> float:
        """Require finite sampling probabilities in the closed unit interval."""
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("engine teacher probabilities must be in [0, 1]")
        return value

    @field_validator(
        "worlds",
        "max_attempts_per_actor_step",
        "worker_max_startup_attempts",
        "async_queue_batches",
    )
    @classmethod
    def positive_integer(cls, value: int) -> int:
        """Require positive bounded-work counts."""
        if value <= 0:
            raise ValueError("engine teacher work counts must be positive")
        return value

    @field_validator(
        "deadline_seconds",
        "max_teacher_seconds_per_actor_step",
        "inference_timeout_seconds",
        "minimum_inference_budget_seconds",
        "worker_startup_timeout_seconds",
        "worker_kill_reap_timeout_seconds",
        "target_weight",
    )
    @classmethod
    def positive_float(cls, value: float) -> float:
        """Require finite positive work and loss scales."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("engine teacher scales must be finite and positive")
        return value

    @model_validator(mode="after")
    def valid_work_budget(self) -> Self:
        """Ensure paired worlds can be sampled and RPC waits fit the step cap."""
        if self.worlds < 2:
            raise ValueError("online engine teacher requires at least two worlds")
        if self.sampler.mode == "model":
            raise ValueError(
                "online engine teacher does not support model belief sampling"
            )
        if self.inference_timeout_seconds > min(
            self.deadline_seconds,
            self.max_teacher_seconds_per_actor_step,
        ):
            raise ValueError("teacher inference timeout must fit every work deadline")
        if self.minimum_inference_budget_seconds >= self.inference_timeout_seconds:
            raise ValueError(
                "minimum teacher inference budget must be below its timeout"
            )
        return self


__all__ = ["OnlineEngineTeacherConfig"]
