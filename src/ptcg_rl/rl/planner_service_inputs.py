"""Root-visible request materialization for the shared planner service."""

from __future__ import annotations

import hashlib
import math
import random
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.agent.probe import core_option_candidates
from ptcg_rl.agent.search.candidate_budget import CandidateBudgetRequest
from ptcg_rl.agent.search.candidates import CandidateSourceInputs
from ptcg_rl.agent.search.prompt_actions import describe_prompt_action_space
from ptcg_rl.agent.search.root_information_context import (
    PublicBeliefFeatureProducer,
    encode_root_information_producer_context,
)
from ptcg_rl.agent.search.root_information_producer import (
    root_information_belief_summary,
)
from ptcg_rl.belief.observation import extract_observation_evidence
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import (
    GameContextFeatures,
    GameContextSnapshot,
)
from ptcg_rl.context.belief import opponent_belief_state_from_evidence
from ptcg_rl.engine.consequence_identity import MaterializedScenario
from ptcg_rl.rl.planner_behavior_policy_contract import (
    PlannerDecisionRequest,
    PlannerRuntimeIdentity,
)

_SEED_DOMAIN = b"ptcg-rl/planner-decision-stochastic-seed/v1\x00"
_ROOT_CHANCE_IDENTITY = b"v5-manual-coin-enumeration/v1"


class PlannerRequestCostConfig(BaseModel):
    """Profile-measured request costs consumed by candidate budgeting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_version: int = 1
    estimated_engine_steps_per_cell: int = Field(gt=0)
    estimated_prefix_nodes_per_candidate: int = Field(gt=0)
    estimated_cell_time_us: int = Field(gt=0)

    @field_validator("architecture_version")
    @classmethod
    def positive_version(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("planner request-cost version must be positive")
        return value


@dataclass(frozen=True, slots=True)
class PlannerRootRow:
    """Serving/training-neutral public row before model proposal generation."""

    row_id: str
    seat: int
    observation: Mapping[str, Any]
    context_features: GameContextFeatures
    context_snapshot: GameContextSnapshot
    own_deck: tuple[int, ...]
    base_action: tuple[int, ...]
    base_old_logprob: float


class PlannerDecisionRequestFactory:
    """Sample paired belief worlds and build one immutable planner request."""

    def __init__(
        self,
        *,
        belief_sampler: BeliefSampler,
        scenario_count: int,
        belief_summary_dim: int,
        costs: PlannerRequestCostConfig,
        producer_contract_fingerprint: bytes,
        belief_feature_producer: PublicBeliefFeatureProducer | None,
        stochastic_seed: int,
    ) -> None:
        if scenario_count <= 0:
            raise ValueError("planner scenario_count must be positive")
        if belief_summary_dim < 0:
            raise ValueError("planner belief summary width must be non-negative")
        if len(producer_contract_fingerprint) != 32:
            raise ValueError("planner producer contract must contain 32 bytes")
        self._belief_sampler = belief_sampler
        self._scenario_count = int(scenario_count)
        self._belief_summary_dim = int(belief_summary_dim)
        self._costs = costs
        self._producer_contract_fingerprint = bytes(
            producer_contract_fingerprint
        )
        self._belief_feature_producer = belief_feature_producer
        self._stochastic_seed = int(stochastic_seed)

    def build(
        self,
        row: PlannerRootRow,
        *,
        greedy_action: Sequence[int],
        proposal_actions: Sequence[Sequence[int]],
        identity: PlannerRuntimeIdentity,
    ) -> PlannerDecisionRequest:
        """Build a complete request or raise before any planner work starts."""
        if row.seat not in (0, 1):
            raise ValueError("planner root seat must be 0 or 1")
        if not math.isfinite(row.base_old_logprob):
            raise ValueError("planner base logprob must be finite")
        select = row.observation.get("select")
        space = describe_prompt_action_space(select)
        state_token = row.observation.get("search_begin_input")
        if not isinstance(state_token, (str, bytes)) or not state_token:
            raise ValueError("planner root observation has no engine state token")
        decision_seed = _decision_seed(
            base_seed=self._stochastic_seed,
            row_id=row.row_id,
            seat=row.seat,
            state_token=state_token,
        )
        scenarios = self._sample_scenarios(row, seed=decision_seed)
        budget = CandidateBudgetRequest(
            legal_action_count=space.legal_action_count,
            option_count=space.option_count,
            min_count=space.min_count,
            max_count=space.max_count,
            ordered=space.ordered,
            scenario_count=len(scenarios),
            estimated_engine_steps_per_cell=(
                self._costs.estimated_engine_steps_per_cell
            ),
            estimated_prefix_nodes_per_candidate=(
                self._costs.estimated_prefix_nodes_per_candidate
            ),
            estimated_cell_time_us=self._costs.estimated_cell_time_us,
            semantic_boundary="terminal_handoff_or_same_main",
        )
        return PlannerDecisionRequest.from_sequences(
            state_token,
            root_observation=row.observation,
            scenarios=scenarios,
            scenario_draw_count=self._scenario_count,
            root_player=row.seat,
            context_snapshot=row.context_snapshot,
            belief_feature_producer=self._belief_feature_producer,
            belief_summary_width=self._belief_summary_dim,
            producer_context=encode_root_information_producer_context(
                row.context_features,
                root_player=row.seat,
            ),
            belief_summary=root_information_belief_summary(
                row.context_features,
                width=self._belief_summary_dim,
            ),
            producer_contract_fingerprint=(
                self._producer_contract_fingerprint
            ),
            base_action=row.base_action,
            base_old_logprob=row.base_old_logprob,
            greedy_action=greedy_action,
            seed_inputs=CandidateSourceInputs(
                base=(row.base_action,),
                proposal=tuple(
                    tuple(int(index) for index in action)
                    for action in proposal_actions
                ),
                structural=core_option_candidates(select),
            ),
            novelty_actions=(),
            budget_request=budget,
            identity=identity,
            stochastic_seed=decision_seed,
        )

    def _sample_scenarios(
        self,
        row: PlannerRootRow,
        *,
        seed: int,
    ) -> tuple[MaterializedScenario, ...]:
        evidence = extract_observation_evidence(row.observation)
        opponent_state = opponent_belief_state_from_evidence(
            evidence,
            row.context_features,
            context_snapshot=row.context_snapshot,
        )
        rng = random.Random(seed)
        draws = []
        for _draw_index in range(self._scenario_count):
            determinization = self._belief_sampler.sample_from_evidence(
                evidence,
                your_deck=row.own_deck,
                opponent_state=opponent_state,
                rng=rng,
            )
            draws.append(determinization.hidden)
        # Preserve the IID posterior mass exactly.  Resampling duplicates until
        # unique would bias concentrated posteriors and can never terminate for
        # a degenerate support.  Consolidation changes only execution cells;
        # each retained row carries its empirical count / requested draw count.
        by_fingerprint: dict[str, tuple[Any, int]] = {}
        order: list[str] = []
        for hidden in draws:
            probe = MaterializedScenario.create(
                hidden_information=hidden,
                belief_world_handle=0,
                chance_support_handle=0,
                chance_support_identity=_ROOT_CHANCE_IDENTITY,
                weight=1.0,
            )
            fingerprint = probe.handle.belief_world_fingerprint
            existing = by_fingerprint.get(fingerprint)
            if existing is None:
                by_fingerprint[fingerprint] = (hidden, 1)
                order.append(fingerprint)
            else:
                by_fingerprint[fingerprint] = (existing[0], existing[1] + 1)
        return tuple(
            MaterializedScenario.create(
                hidden_information=by_fingerprint[fingerprint][0],
                belief_world_handle=index,
                chance_support_handle=0,
                chance_support_identity=_ROOT_CHANCE_IDENTITY,
                weight=(
                    by_fingerprint[fingerprint][1] / self._scenario_count
                ),
            )
            for index, fingerprint in enumerate(order)
        )


def _decision_seed(
    *,
    base_seed: int,
    row_id: str,
    seat: int,
    state_token: bytes | str,
) -> int:
    token = state_token if isinstance(state_token, bytes) else state_token.encode("ascii")
    row = row_id.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_SEED_DOMAIN)
    digest.update(struct.pack(">qii", base_seed, seat, len(row)))
    digest.update(row)
    digest.update(struct.pack(">I", len(token)))
    digest.update(token)
    return int.from_bytes(digest.digest()[:8], "big")


__all__ = [
    "PlannerDecisionRequestFactory",
    "PlannerRequestCostConfig",
    "PlannerRootRow",
]
