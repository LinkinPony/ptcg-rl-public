"""Lifecycle-safe construction of the shared production planner runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from ptcg_rl.agent.search.root_information_context import (
    PublicBeliefFeatureProducer,
)
from ptcg_rl.belief.runtime_identity import belief_runtime_fingerprint
from ptcg_rl.belief.sampling import BeliefSampler, BeliefSamplerConfig
from ptcg_rl.context import (
    OpponentBeliefFeatureConfig,
    OpponentBeliefFeatureProducer,
)
from ptcg_rl.engine.native_planning_session_pool import (
    NativePlanningSessionLanePool,
)
from ptcg_rl.rl.planner_behavior_service import PlannerBehaviorService
from ptcg_rl.rl.planner_runtime_identity import ResolvedPlannerRuntimeConfig


@dataclass(slots=True)
class PlannerBehaviorRuntime:
    """Own one actor-local native pool and its bounded planner service."""

    session_pool: NativePlanningSessionLanePool
    service: PlannerBehaviorService
    _closed: bool = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        """Drain service work before closing its native lane ownership."""
        if self._closed:
            return
        self._closed = True
        try:
            self.service.close()
        finally:
            self.session_pool.close()


def create_planner_behavior_runtime(
    *,
    runtime_config: ResolvedPlannerRuntimeConfig,
    sampler_config: BeliefSamplerConfig,
    belief_config: OpponentBeliefFeatureConfig,
    stochastic_seed: int,
    native_library_path: Path | str | None = None,
) -> PlannerBehaviorRuntime:
    """Build an exact actor-local runtime or fail before rollout starts."""
    sampler = BeliefSampler(config=sampler_config)
    producer = _matching_belief_producer(
        runtime_config,
        sampler=sampler,
        belief_config=belief_config,
    )
    session_pool = NativePlanningSessionLanePool(
        runtime_config.session_pool,
        library_path=native_library_path,
    )
    try:
        service = PlannerBehaviorService(
            runtime_config=runtime_config,
            session_pool=session_pool,
            belief_sampler=sampler,
            belief_feature_producer=producer,
            stochastic_seed=stochastic_seed,
        )
    except Exception:
        session_pool.close()
        raise
    return PlannerBehaviorRuntime(session_pool=session_pool, service=service)


def _matching_belief_producer(
    runtime_config: ResolvedPlannerRuntimeConfig,
    *,
    sampler: BeliefSampler,
    belief_config: OpponentBeliefFeatureConfig,
) -> PublicBeliefFeatureProducer | None:
    """Resolve the explicit producer form named by the runtime fingerprint."""
    expected = runtime_config.scenario.belief_sampler_fingerprint
    producer = OpponentBeliefFeatureProducer.from_config(belief_config)
    if belief_runtime_fingerprint(sampler, producer) == expected:
        return producer
    if belief_runtime_fingerprint(sampler, None) == expected:
        return None
    raise ValueError(
        "rollout sampler/belief producer differs from planner static identity"
    )


__all__ = [
    "PlannerBehaviorRuntime",
    "create_planner_behavior_runtime",
]
