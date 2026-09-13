"""Consume request-bound v4 probe rows as direct v5 planner seed evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from ptcg_rl.agent.search.consequence_flow import (
    RootInformationCellContext,
    build_direct_option_outcomes,
    build_root_information_transition_batch,
)
from ptcg_rl.agent.search.hierarchical_contract import (
    StableContinuationControllerIdentity,
)
from ptcg_rl.agent.search.planner_fallback import (
    PlannerEvidenceError,
    PlannerFallbackReason,
)
from ptcg_rl.agent.search.planner_scoring import (
    SharedLeafScoreResult,
    SharedRootInformationLeafScorer,
    leaf_scoring_batch_from_compact,
)
from ptcg_rl.agent.search.planning_session_contract import (
    HierarchicalOutcomeBatch,
    HierarchicalSearchRequest,
)
from ptcg_rl.agent.search.planning_session_scoring import (
    ScoredHierarchicalEvidence,
)
from ptcg_rl.agent.search.root_information_producer import (
    RootInformationProducerBridge,
)
from ptcg_rl.engine.compact_consequence import (
    CELL_TRANSITION_STEPS_COLUMN,
    CompactConsequenceBatch,
    CompactConsequenceMetadata,
)
from ptcg_rl.engine.consequence_identity import candidate_action_fingerprint
from ptcg_rl.rl.planner_seed_reuse import ReusedPlannerSeedGrid

TensorInputs = TypeVar("TensorInputs")


@dataclass(frozen=True, slots=True)
class ReusedSeedOutcomeBatch(Generic[TensorInputs]):
    """Direct outcome batch plus its already-computed unique leaf values."""

    outcomes: HierarchicalOutcomeBatch
    leaf_scores: SharedLeafScoreResult


@dataclass(frozen=True, slots=True)
class ReusableV5SeedEvidence(Generic[TensorInputs]):
    """Already executed and scored exhaustive v5 fixed-select support."""

    request_fingerprint: str
    actions: tuple[tuple[int, ...], ...]
    base_scenario_count: int
    scored: ScoredHierarchicalEvidence[TensorInputs]

    def __post_init__(self) -> None:
        if len(self.request_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.request_fingerprint
        ):
            raise ValueError("reusable v5 request fingerprint must be SHA-256")
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("reusable v5 actions must be nonempty and unique")
        if self.base_scenario_count <= 0:
            raise ValueError("reusable v5 evidence requires base scenarios")
        if self.scored.outcome_batch.actions != self.actions:
            raise ValueError("reusable v5 scored actions differ from its support")
        if self.scored.outcome_batch.producer_contract_fingerprint != (
            self.request_fingerprint
        ):
            raise ValueError("reusable v5 producer identity differs from request")


def outcome_batch_from_reused_probe(
    *,
    request: HierarchicalSearchRequest,
    reuse: ReusedPlannerSeedGrid,
    cached: CompactConsequenceBatch,
    root_information_history_fingerprint: str,
    controller: StableContinuationControllerIdentity,
    scorer: SharedRootInformationLeafScorer[TensorInputs],
) -> ReusedSeedOutcomeBatch[TensorInputs]:
    """Gather a complete retained direct grid without repeating engine work."""
    scenario_count = cached.scenario_count
    expected_shape = (len(request.candidate_actions), scenario_count)
    if reuse.retained_cell_to_cached_cell.shape != expected_shape:
        raise PlannerEvidenceError(
            PlannerFallbackReason.FINGERPRINT_MISMATCH,
            "probe reuse gather differs from the retained seed grid",
        )
    gather = reuse.retained_cell_to_cached_cell.reshape(-1)
    if bool(np.any(gather < 0)):
        raise PlannerEvidenceError(
            PlannerFallbackReason.EVIDENCE_ABSENT,
            "probe does not contain every retained seed cell",
        )
    subset = _compact_subset(cached, request=request, gather=gather)
    bridge = RootInformationProducerBridge(
        root_player=request.root_player,
        belief_summary_width=request.belief_summary_width,
        belief_feature_producer=request.belief_feature_producer,
    )
    contexts: list[RootInformationCellContext] = []
    try:
        for candidate_index in range(subset.candidate_count):
            for scenario_index in range(subset.scenario_count):
                observation = subset.row_at(
                    candidate_index,
                    scenario_index,
                ).root_observable_state.payload.tobytes(order="C")
                reached = bridge.advance(
                    parent_context_snapshot=request.context_snapshot,
                    transition_observable_state=observation,
                    model_observable_state=observation,
                ).state
                contexts.append(
                    RootInformationCellContext(
                        producer_context=reached.producer_context,
                        belief_summary=reached.belief_summary,
                    )
                )
    except (TypeError, ValueError) as exc:
        raise PlannerEvidenceError(
            PlannerFallbackReason.FINGERPRINT_MISMATCH,
            "reused probe cannot advance the root-information producer",
        ) from exc
    transitions = build_root_information_transition_batch(
        subset,
        source_information_history_fingerprint=(root_information_history_fingerprint),
        cell_contexts=tuple(contexts),
    )
    scoring_cells = leaf_scoring_batch_from_compact(
        subset,
        root_player=request.root_player,
    )
    scores = scorer.score(leaves=transitions.value_leaves, cells=scoring_cells)
    outcomes = build_direct_option_outcomes(
        subset,
        transitions,
        scores=scores,
        controller=controller,
        root_player=request.root_player,
    )
    path_steps = subset.cell_metadata[:, CELL_TRANSITION_STEPS_COLUMN].copy()
    path_steps.setflags(write=False)
    return ReusedSeedOutcomeBatch(
        outcomes=HierarchicalOutcomeBatch(
            actions=request.candidate_actions,
            outcomes=outcomes,
            leaves=transitions.value_leaves,
            scoring_cells=scoring_cells,
            path_steps=path_steps,
            chance_depth=0,
            continuation_nodes=0,
            producer_contract_fingerprint=reuse.request_fingerprint,
        ),
        leaf_scores=scores,
    )


def _compact_subset(
    cached: CompactConsequenceBatch,
    *,
    request: HierarchicalSearchRequest,
    gather: np.ndarray,
) -> CompactConsequenceBatch:
    chunks: list[bytes] = []
    sizes: list[int] = []
    for raw_index in gather:
        cell_index = int(raw_index)
        start = int(cached.root_observation_offsets[cell_index])
        stop = int(cached.root_observation_offsets[cell_index + 1])
        chunk = cached.root_observation_bytes[start:stop].tobytes()
        chunks.append(chunk)
        sizes.append(len(chunk))
    offsets: npt.NDArray[np.int32] = np.empty(len(gather) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(np.asarray(sizes, dtype=np.int32), out=offsets[1:])
    observation_bytes = np.frombuffer(b"".join(chunks), dtype=np.uint8)
    exact_effects = cached.exact_effects[gather].copy()
    metadata = cached.cell_metadata[gather].copy()
    for values in (offsets, observation_bytes, exact_effects, metadata):
        values.setflags(write=False)
    contract = CompactConsequenceMetadata(
        observation_schema_fingerprint=(cached.contract.observation_schema_fingerprint),
        root_state_fingerprint=cached.contract.root_state_fingerprint,
        candidate_fingerprints=tuple(
            candidate_action_fingerprint(action) for action in request.candidate_actions
        ),
        legal_action_count=request.legal_action_count,
        scenario_support=cached.contract.scenario_support,
        support_exhaustive=request.support_exhaustive,
        scenario_grid_complete=True,
    )
    return CompactConsequenceBatch(
        contract=contract,
        root_observation_offsets=offsets,
        root_observation_bytes=observation_bytes,
        exact_effects=exact_effects,
        cell_metadata=metadata,
    )


__all__ = [
    "ReusableV5SeedEvidence",
    "ReusedSeedOutcomeBatch",
    "outcome_batch_from_reused_probe",
]
