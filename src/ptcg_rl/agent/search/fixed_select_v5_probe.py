"""V5 exhaustive fixed-single-select equivalence and direct seed reuse."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

from ptcg_rl.agent.probe import enumerate_select_actions
from ptcg_rl.agent.search.eligibility import (
    EquivalenceProbeCell,
    FixedSingleSelectProbeExecution,
    SuccessorSemanticFingerprint,
    build_fixed_single_select_probe_result,
)
from ptcg_rl.agent.search.planner_fallback import (
    PlannerEvidenceError,
    PlannerFallbackReason,
)
from ptcg_rl.agent.search.planner_scoring import SharedRootInformationLeafScorer
from ptcg_rl.agent.search.planning_seed_reuse import ReusableV5SeedEvidence
from ptcg_rl.agent.search.planning_session_contract import (
    HierarchicalSearchRequest,
)
from ptcg_rl.agent.search.planning_session_scoring import (
    score_hierarchical_outcomes,
)
from ptcg_rl.agent.search.planning_session_tree import (
    HierarchicalPlanningSessionExecutor,
)
from ptcg_rl.agent.search.prompt_actions import describe_prompt_action_space
from ptcg_rl.engine.consequence_identity import candidate_action_fingerprint
from ptcg_rl.rl.planner_behavior_policy_contract import PlannerDecisionRequest
from ptcg_rl.runtime.work_ledger import PlannerRequestLedger

TensorInputs = TypeVar("TensorInputs")
_SUCCESSOR_DOMAIN = b"ptcg-rl/v5-fixed-select-successor/v1\x00"
_CHANCE_DOMAIN = b"ptcg-rl/v5-fixed-select-chance/v1\x00"
_BELIEF_DOMAIN = b"ptcg-rl/v5-fixed-select-belief-unproven/v2\x00"
_ENDPOINT_DOMAIN = b"ptcg-rl/v5-fixed-select-endpoint/v1\x00"


@dataclass(frozen=True, slots=True)
class ReusableV5FixedSingleSelectProbe(Generic[TensorInputs]):
    """Passive probe whose native and GPU work is already ledger-accounted."""

    execution: FixedSingleSelectProbeExecution
    action_count: int
    scenario_count: int
    work_already_accounted: bool = True

    def probe(
        self,
        select: Any,
        *,
        scenario_count: int,
        transition_budget: int,
    ) -> FixedSingleSelectProbeExecution:
        """Return the request-local execution after validating eligibility use."""
        space = describe_prompt_action_space(select)
        required = self.action_count * self.scenario_count
        if space.legal_action_count != self.action_count:
            raise ValueError("fixed-select prompt changed after v5 execution")
        if scenario_count != self.scenario_count:
            raise ValueError("fixed-select scenario support changed after execution")
        if required > transition_budget:
            raise ValueError("fixed-select one-step grid exceeds probe budget")
        return self.execution


def execute_reusable_v5_fixed_select_probe(
    request: PlannerDecisionRequest,
    *,
    executor: HierarchicalPlanningSessionExecutor,
    scorer: SharedRootInformationLeafScorer[TensorInputs],
    ledger: PlannerRequestLedger,
) -> ReusableV5FixedSingleSelectProbe[TensorInputs]:
    """Execute one exhaustive v5 support and retain it as planner seed evidence."""
    select = request.root_observation.get("select")
    space = describe_prompt_action_space(select)
    actions = enumerate_select_actions(
        select,
        max_actions=space.legal_action_count,
    )
    if len(actions) != space.legal_action_count:
        raise PlannerEvidenceError(
            PlannerFallbackReason.CONSTRUCTOR_INVALID,
            "fixed-select exhaustive action enumeration is incomplete",
        )
    search_request = HierarchicalSearchRequest.from_sequences(
        request.state_token,
        root_observation=request.root_observation,
        scenarios=request.scenarios,
        candidate_actions=actions,
        legal_action_count=space.legal_action_count,
        root_player=request.root_player,
        context_snapshot=request.context_snapshot,
        belief_feature_producer=request.belief_feature_producer,
        belief_summary_width=request.belief_summary_width,
        producer_context=request.producer_context,
        belief_summary=request.belief_summary,
        producer_contract_fingerprint=request.producer_contract_fingerprint,
        support_exhaustive=True,
    )
    outcomes = executor.evaluate(search_request, ledger=ledger)
    leaf_count = len(outcomes.leaves.leaves)
    reservation = None if leaf_count == 0 else ledger.reserve(gpu_rows=leaf_count)
    if leaf_count and reservation is None:
        raise PlannerEvidenceError(
            PlannerFallbackReason.BUDGET_TRUNCATED,
            "fixed-select reusable leaves exceed the GPU row ledger",
        )
    success = False
    try:
        scored = score_hierarchical_outcomes(
            outcomes,
            scorer=scorer,
            legal_action_count=space.legal_action_count,
            support_exhaustive=True,
        )
        success = True
    finally:
        if reservation is not None:
            ledger.complete(
                reservation,
                elapsed_seconds=0.0,
                success=success,
            )
    reusable: ReusableV5SeedEvidence[TensorInputs] = ReusableV5SeedEvidence(
        request_fingerprint=outcomes.producer_contract_fingerprint,
        actions=actions,
        base_scenario_count=len(request.scenarios),
        scored=scored,
    )
    cells = _equivalence_cells(
        request=request,
        reusable=reusable,
    )
    result = build_fixed_single_select_probe_result(
        action_count=len(actions),
        scenario_count=len(request.scenarios),
        cells=cells,
        transitions_used=len(actions) * len(request.scenarios),
        request_contract_fingerprint=outcomes.producer_contract_fingerprint,
        stop_reason="complete_v5_direct_seed_reuse",
    )
    return ReusableV5FixedSingleSelectProbe(
        execution=FixedSingleSelectProbeExecution(
            result=result,
            reusable_evidence=None,
            reusable_v5_evidence=reusable,
        ),
        action_count=len(actions),
        scenario_count=len(request.scenarios),
    )


def _equivalence_cells(
    *,
    request: PlannerDecisionRequest,
    reusable: ReusableV5SeedEvidence[TensorInputs],
) -> tuple[EquivalenceProbeCell, ...]:
    scored = reusable.scored
    batch = scored.outcome_batch
    support = batch.outcomes[0].scenario_support
    support_count = len(support.scenarios)
    base_fingerprints = tuple(
        scenario.handle.belief_world_fingerprint for scenario in request.scenarios
    )
    path_indices = tuple(
        tuple(
            index
            for index, scenario in enumerate(support.scenarios)
            if scenario.belief_world_fingerprint == belief_fingerprint
        )
        for belief_fingerprint in base_fingerprints
    )
    if any(not indices for indices in path_indices):
        raise PlannerEvidenceError(
            PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
            "v5 fixed-select evidence lost a base belief world",
        )
    result: list[EquivalenceProbeCell] = []
    for action_index, outcome in enumerate(batch.outcomes):
        for scenario_index, indices in enumerate(path_indices):
            successor = hashlib.sha256(_SUCCESSOR_DOMAIN)
            chance = hashlib.sha256(_CHANCE_DOMAIN)
            endpoints = hashlib.sha256(_ENDPOINT_DOMAIN)
            for support_index in indices:
                cell_index = action_index * support_count + support_index
                cell = outcome.cells[support_index]
                scenario = support.scenarios[support_index]
                successor.update(struct.pack(">i", int(cell.endpoint)))
                successor.update(
                    np.asarray(
                        batch.scoring_cells.exact_effects[cell_index],
                        dtype=">f4",
                    ).tobytes()
                )
                successor.update(struct.pack(">i", int(batch.path_steps[cell_index])))
                if cell.leaf_index is None:
                    if cell.terminal_result is None:
                        raise PlannerEvidenceError(
                            PlannerFallbackReason.CONSTRUCTOR_INVALID,
                            "terminal v5 cell has no engine result",
                        )
                    successor.update(struct.pack(">f", cell.terminal_result))
                else:
                    _update_bytes(
                        successor,
                        batch.leaves.leaves[cell.leaf_index].root_observable_state,
                    )
                chance.update(bytes.fromhex(scenario.chance_support_fingerprint))
                chance.update(struct.pack(">d", float(scenario.weight)))
                endpoints.update(struct.pack(">i", int(cell.endpoint)))
            belief_update = _unproven_belief_update_fingerprint(
                base_belief_world_fingerprint=base_fingerprints[scenario_index],
                action=reusable.actions[action_index],
            )
            result.append(
                EquivalenceProbeCell(
                    action_index=action_index,
                    scenario_index=scenario_index,
                    semantics=SuccessorSemanticFingerprint(
                        successor=successor.hexdigest(),
                        chance_cursor=chance.hexdigest(),
                        belief_update=belief_update,
                        endpoint=endpoints.hexdigest(),
                        unresolved_prompt="fully_resolved_v5",
                    ),
                )
            )
    return tuple(result)


def _unproven_belief_update_fingerprint(
    *,
    base_belief_world_fingerprint: str,
    action: tuple[int, ...],
) -> str:
    """Keep distinct actions non-equivalent without a full successor proof.

    V5 currently exposes root-visible observations and path-local effects, not a
    content digest of the reached private engine state or posterior belief
    update. The original belief-world identity therefore cannot prove that two
    actions have the same hidden successor. Binding the canonical action keeps
    the exact-equivalence fast path disabled until the native ABI supplies that
    missing proof; the already-executed grid remains reusable planner evidence.
    """
    digest = hashlib.sha256(_BELIEF_DOMAIN)
    digest.update(bytes.fromhex(base_belief_world_fingerprint))
    digest.update(bytes.fromhex(candidate_action_fingerprint(action)))
    return digest.hexdigest()


def _update_bytes(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


__all__ = [
    "ReusableV5FixedSingleSelectProbe",
    "execute_reusable_v5_fixed_select_probe",
]
