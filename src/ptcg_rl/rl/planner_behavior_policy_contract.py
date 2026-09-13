"""Typed request and model surfaces for production pre-action planning."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor

from ptcg_rl.agent.search.candidate_budget import (
    CandidateBudgetRequest,
    CandidateConstructorConfig,
)
from ptcg_rl.agent.search.candidates import CandidateSourceInputs
from ptcg_rl.agent.search.eligibility import (
    FixedSingleSelectEquivalenceProbe,
    GenericEligibilityConfig,
)
from ptcg_rl.agent.search.root_information_context import (
    PublicBeliefFeatureProducer,
)
from ptcg_rl.context import GameContextSnapshot
from ptcg_rl.engine.consequence_identity import MaterializedScenario
from ptcg_rl.engine.consequence_request_identity import PreparedRequestIdentity
from ptcg_rl.model.network import PlannerCandidateEvaluation
from ptcg_rl.rl.planner_evidence import (
    PlannerBehaviorBranch,
    PlannerBehaviorEvidence,
)
from ptcg_rl.runtime.planner_telemetry import (
    PlannerDecisionRuntimeStats,
    PlannerStageEvent,
)

if TYPE_CHECKING:
    from ptcg_rl.rl.engine_teacher import EngineTeacherTarget

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PlannerBehaviorPolicyConfig(BaseModel):
    """Resolved eligibility, construction, and expansion semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_version: int = 1
    eligibility: GenericEligibilityConfig
    constructor: CandidateConstructorConfig
    max_mutation_parents: int
    emit_macro_teacher: bool = False

    @field_validator("architecture_version", "max_mutation_parents")
    @classmethod
    def positive_value(cls, value: int) -> int:
        """Require positive versions and bounded parent counts."""
        if value <= 0:
            raise ValueError("planner behavior policy limits must be positive")
        return value


class PlannerRuntimeIdentity(BaseModel):
    """Immutable model and semantic identities held by one request lease."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_fingerprint: str
    constructor_fingerprint: str
    scorer_fingerprint: str
    controller_fingerprint: str
    planner_fingerprint: str
    policy_version: int = Field(ge=0)
    proposal_version: int = Field(ge=0)
    constructor_version: int = Field(ge=0)
    planner_version: int = Field(ge=0)

    @field_validator(
        "model_fingerprint",
        "constructor_fingerprint",
        "scorer_fingerprint",
        "controller_fingerprint",
        "planner_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require canonical content identities."""
        if _SHA256.fullmatch(value) is None:
            raise ValueError("planner runtime identities must be SHA-256 hex")
        return value


@dataclass(frozen=True, slots=True)
class PlannerDecisionRequest:
    """One actor decision with a complete base fallback already sampled."""

    state_token: bytes | str
    root_observation: Mapping[str, Any]
    scenarios: tuple[MaterializedScenario, ...]
    scenario_draw_count: int
    root_player: int
    context_snapshot: GameContextSnapshot
    belief_feature_producer: PublicBeliefFeatureProducer | None
    belief_summary_width: int
    producer_context: bytes
    belief_summary: tuple[float, ...]
    producer_contract_fingerprint: bytes
    base_action: tuple[int, ...]
    base_old_logprob: float
    greedy_action: tuple[int, ...]
    seed_inputs: CandidateSourceInputs
    novelty_actions: tuple[tuple[int, ...], ...]
    budget_request: CandidateBudgetRequest
    identity: PlannerRuntimeIdentity
    stochastic_seed: int
    equivalence_probe: FixedSingleSelectEquivalenceProbe | None = None
    expected_probe_identity: PreparedRequestIdentity | None = None

    @classmethod
    def from_sequences(
        cls,
        state_token: bytes | str,
        *,
        root_observation: Mapping[str, Any],
        scenarios: Sequence[MaterializedScenario],
        scenario_draw_count: int | None = None,
        root_player: int,
        context_snapshot: GameContextSnapshot,
        belief_feature_producer: PublicBeliefFeatureProducer | None,
        belief_summary_width: int,
        producer_context: bytes,
        belief_summary: Sequence[float],
        producer_contract_fingerprint: bytes,
        base_action: Sequence[int],
        base_old_logprob: float,
        greedy_action: Sequence[int],
        seed_inputs: CandidateSourceInputs,
        novelty_actions: Sequence[Sequence[int]],
        budget_request: CandidateBudgetRequest,
        identity: PlannerRuntimeIdentity,
        stochastic_seed: int,
        equivalence_probe: FixedSingleSelectEquivalenceProbe | None = None,
        expected_probe_identity: PreparedRequestIdentity | None = None,
    ) -> Self:
        """Freeze ragged action/scenario inputs before planner admission."""
        return cls(
            state_token=state_token,
            root_observation=root_observation,
            scenarios=tuple(scenarios),
            scenario_draw_count=(
                len(tuple(scenarios))
                if scenario_draw_count is None
                else int(scenario_draw_count)
            ),
            root_player=root_player,
            context_snapshot=context_snapshot,
            belief_feature_producer=belief_feature_producer,
            belief_summary_width=int(belief_summary_width),
            producer_context=bytes(producer_context),
            belief_summary=tuple(float(value) for value in belief_summary),
            producer_contract_fingerprint=bytes(producer_contract_fingerprint),
            base_action=tuple(int(index) for index in base_action),
            base_old_logprob=float(base_old_logprob),
            greedy_action=tuple(int(index) for index in greedy_action),
            seed_inputs=seed_inputs,
            novelty_actions=tuple(
                tuple(int(index) for index in action) for action in novelty_actions
            ),
            budget_request=budget_request,
            identity=identity,
            stochastic_seed=int(stochastic_seed),
            equivalence_probe=equivalence_probe,
            expected_probe_identity=expected_probe_identity,
        )

    def __post_init__(self) -> None:
        """Reject a fallback action that cannot define a behavior row."""
        if not math.isfinite(self.base_old_logprob):
            raise ValueError("base_old_logprob must be finite")
        if self.root_player not in (0, 1):
            raise ValueError("root_player must be 0 or 1")
        if self.belief_summary_width < 0:
            raise ValueError("belief_summary_width must be non-negative")
        if len(self.belief_summary) != self.belief_summary_width:
            raise ValueError("belief_summary width differs from its producer contract")
        if not self.scenarios:
            raise ValueError("planner decision scenarios must not be empty")
        if self.scenario_draw_count < len(self.scenarios):
            raise ValueError("planner scenario draws cannot be below unique support")
        if len(self.producer_contract_fingerprint) != 32:
            raise ValueError("producer contract fingerprint must contain 32 bytes")
        if any(not math.isfinite(value) for value in self.belief_summary):
            raise ValueError("belief_summary must be finite")
        if self.equivalence_probe is None and self.expected_probe_identity is not None:
            raise ValueError("probe request identity requires an equivalence probe")


class PlannerCandidateEvaluator(Protocol):
    """Request-leased dedicated schema-9 planner reranker surface."""

    def evaluate_planner_candidates(
        self,
        *,
        actions: tuple[tuple[int, ...], ...],
        aggregate_features: Tensor,
    ) -> PlannerCandidateEvaluation:
        """Evaluate candidates with the dedicated planner reranker only."""


@dataclass(frozen=True, slots=True)
class PlannerPolicyDecision:
    """Action and exact old probability selected by one behavior branch."""

    action: tuple[int, ...]
    old_logprob: float
    planner_behavior: PlannerBehaviorEvidence
    used_base_trace: bool
    telemetry_events: tuple[PlannerStageEvent, ...] = ()
    runtime_stats: PlannerDecisionRuntimeStats = PlannerDecisionRuntimeStats()
    macro_teacher_target: EngineTeacherTarget | None = None

    def __post_init__(self) -> None:
        """Keep fallback token traces and categorical planner rows disjoint."""
        fallback = self.planner_behavior.branch is PlannerBehaviorBranch.BASE_FALLBACK
        if self.used_base_trace != fallback:
            raise ValueError("used_base_trace must match the planner branch")
        if not math.isfinite(self.old_logprob):
            raise ValueError("planner decision old_logprob must be finite")
        if fallback:
            if self.planner_behavior.selected_action is not None:
                raise ValueError("base fallback cannot retain a selected candidate")
            if self.macro_teacher_target is not None:
                raise ValueError("planner fallback cannot carry macro teacher evidence")
        elif self.action != self.planner_behavior.selected_action:
            raise ValueError("planner action differs from selected evidence")


__all__ = [
    "PlannerBehaviorPolicyConfig",
    "PlannerCandidateEvaluator",
    "PlannerDecisionRequest",
    "PlannerPolicyDecision",
    "PlannerRuntimeIdentity",
]
