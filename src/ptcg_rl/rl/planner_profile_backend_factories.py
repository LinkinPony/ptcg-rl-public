"""Explicit workload factories used by the integrated planner profile."""

from __future__ import annotations

from dataclasses import dataclass

from ptcg_rl.belief.runtime_identity import belief_runtime_fingerprint
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import OpponentBeliefFeatureProducer
from ptcg_rl.evaluation.planner_profile_config import (
    IntegratedPlannerProfileConfig,
    PlannerBeliefWorkloadConfig,
    PlannerProfileRuntimeConfig,
)
from ptcg_rl.rl.planner_profile_learner import ConcurrentProfileLearner


@dataclass(frozen=True, slots=True)
class ProfileBeliefRuntime:
    """One exact sampler/public-producer pair for an actor or oracle lane."""

    sampler: BeliefSampler
    producer: OpponentBeliefFeatureProducer
    fingerprint: str


def create_profile_belief_sampler(
    config: PlannerBeliefWorkloadConfig,
) -> ProfileBeliefRuntime:
    """Construct and verify the preregistered deployment belief runtime."""
    sampler = BeliefSampler(config=config.sampler)
    producer = OpponentBeliefFeatureProducer.from_config(config.producer)
    return ProfileBeliefRuntime(
        sampler=sampler,
        producer=producer,
        fingerprint=belief_runtime_fingerprint(sampler, producer),
    )


def create_concurrent_learner_workload(
    *,
    config: IntegratedPlannerProfileConfig,
    runtime: PlannerProfileRuntimeConfig,
) -> ConcurrentProfileLearner:
    """Start the fixed synthetic learner-kernel contention child."""
    kernel_runtime = config.runtime_profiles[config.learner_kernel_runtime_id]
    if runtime.learner != kernel_runtime.learner:
        raise ValueError("profile point changed the fixed learner kernel workload")
    return ConcurrentProfileLearner.start(config=config, runtime=kernel_runtime)


__all__ = [
    "ProfileBeliefRuntime",
    "create_concurrent_learner_workload",
    "create_profile_belief_sampler",
]
