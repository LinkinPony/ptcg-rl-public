"""Shared leaf scoring and public aggregates for v5 option outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from ptcg_rl.agent.search.hierarchical_contract import (
    ContinuationDecision,
    OptionOutcome,
    OptionOutcomeCell,
)
from ptcg_rl.agent.search.paired_scenarios import (
    PairedScenarioAggregation,
    aggregate_paired_option_outcomes,
)
from ptcg_rl.agent.search.planner_scoring import (
    LeafScoringBatch,
    SharedLeafScoreResult,
    SharedRootInformationLeafScorer,
)
from ptcg_rl.agent.search.planning_session_contract import HierarchicalOutcomeBatch
from ptcg_rl.agent.search.root_information import (
    RootInformationLeafCell,
    deduplicate_root_information_leaves,
)
from ptcg_rl.engine.compact_consequence import ScenarioSupport
from ptcg_rl.engine.search_evidence import (
    SEARCH_EVIDENCE_EXACT_INDEX,
    SearchCandidateEvidence,
    search_candidate_features,
)

TensorInputs = TypeVar("TensorInputs")


@dataclass(frozen=True, slots=True)
class PlannerAggregateEvidence:
    """Schema-9 aggregates with rules exactness separate from root coverage."""

    candidates: tuple[SearchCandidateEvidence, ...]
    legal_action_count: int
    scenario_count: int
    exhaustive: bool
    rules_exact: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Validate candidate alignment without schema-8 exactness semantics."""
        if not self.candidates or len(self.candidates) != len(self.rules_exact):
            raise ValueError("planner aggregate candidates and exactness must align")
        if len({candidate.action for candidate in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("planner aggregate actions must be unique")
        if self.legal_action_count < len(self.candidates):
            raise ValueError("planner legal action count is below retained support")
        if self.exhaustive != (self.legal_action_count == len(self.candidates)):
            raise ValueError("planner exhaustive flag differs from retained support")
        if self.scenario_count <= 0:
            raise ValueError("planner aggregate evidence requires scenarios")
        for candidate, exact in zip(
            self.candidates,
            self.rules_exact,
            strict=True,
        ):
            expected = 1.0 if exact else 0.0
            if candidate.features[SEARCH_EVIDENCE_EXACT_INDEX] != expected:
                raise ValueError(
                    "planner aggregate rules exactness differs from features"
                )

    @property
    def actions(self) -> tuple[tuple[int, ...], ...]:
        """Return retained complete actions in immutable planner order."""
        return tuple(candidate.action for candidate in self.candidates)

    @property
    def feature_rows(self) -> tuple[tuple[float, ...], ...]:
        """Return public aggregate features aligned with retained actions."""
        return tuple(candidate.features for candidate in self.candidates)


@dataclass(frozen=True, slots=True)
class ScoredHierarchicalEvidence(Generic[TensorInputs]):
    """One complete immutable support ready for planner behavior sampling."""

    outcome_batch: HierarchicalOutcomeBatch
    leaf_scores: SharedLeafScoreResult
    aggregation: PairedScenarioAggregation
    search_evidence: PlannerAggregateEvidence

    @property
    def robust_scores(self) -> Tensor:
        """Return candidate-aligned detached robust scores on CPU."""
        return torch.from_numpy(self.aggregation.robust_scores.copy())


def score_hierarchical_outcomes(
    batch: HierarchicalOutcomeBatch,
    *,
    scorer: SharedRootInformationLeafScorer[TensorInputs],
    legal_action_count: int,
    support_exhaustive: bool,
    precomputed_leaf_scores: SharedLeafScoreResult | None = None,
) -> ScoredHierarchicalEvidence[TensorInputs]:
    """Value unique leaves once, gather cells, and build fixed-width evidence."""
    if legal_action_count < len(batch.actions):
        raise ValueError("legal_action_count cannot be below retained support")
    if support_exhaustive != (legal_action_count == len(batch.actions)):
        raise ValueError("support_exhaustive differs from retained support")
    scores = precomputed_leaf_scores or scorer.score(
        leaves=batch.leaves,
        cells=batch.scoring_cells,
    )
    if scores.scorer_fingerprint != scorer.scorer_fingerprint:
        raise ValueError("precomputed leaf scores use another scorer contract")
    aggregation = aggregate_paired_option_outcomes(
        batch.outcomes,
        cell_scores=scores.cell_scores,
        scorer_fingerprint=scorer.scorer_fingerprint,
        config=scorer.config,
    )
    scenario_count = len(batch.outcomes[0].cells)
    score_grid = np.asarray(scores.cell_scores, dtype=np.float32).reshape(
        len(batch.actions),
        scenario_count,
    )
    step_grid = batch.path_steps.reshape(len(batch.actions), scenario_count)
    weights = np.asarray(
        tuple(cell.weight for cell in batch.outcomes[0].cells),
        dtype=np.float64,
    )
    coverage = len(batch.actions) / legal_action_count
    rules_exact_grid = batch.scoring_cells.rules_exact_mask.reshape(
        len(batch.actions),
        scenario_count,
    )
    candidate_rules_exact = tuple(bool(values.all()) for values in rules_exact_grid)
    candidates = tuple(
        SearchCandidateEvidence(
            action=action,
            features=search_candidate_features(
                world_scores=tuple(float(value) for value in score_grid[index]),
                robust_score=aggregation.candidates[index].robust_score,
                coverage=coverage,
                terminal_fraction=aggregation.candidates[index].terminal_weight,
                same_seat_main_fraction=(
                    aggregation.candidates[index].same_seat_main_weight
                ),
                turn_handoff_fraction=(
                    aggregation.candidates[index].turn_handoff_weight
                ),
                mean_path_steps=float(np.dot(weights, step_grid[index])),
                exact=candidate_rules_exact[index],
            ),
        )
        for index, action in enumerate(batch.actions)
    )
    evidence = PlannerAggregateEvidence(
        candidates=candidates,
        legal_action_count=legal_action_count,
        scenario_count=scenario_count,
        exhaustive=support_exhaustive,
        rules_exact=candidate_rules_exact,
    )
    return ScoredHierarchicalEvidence(
        outcome_batch=batch,
        leaf_scores=scores,
        aggregation=aggregation,
        search_evidence=evidence,
    )


def merge_scored_hierarchical_evidence(
    parts: tuple[ScoredHierarchicalEvidence[TensorInputs], ...],
    *,
    scorer: SharedRootInformationLeafScorer[TensorInputs],
    legal_action_count: int,
    support_exhaustive: bool,
    producer_contract_fingerprint: str,
) -> ScoredHierarchicalEvidence[TensorInputs]:
    """Merge disjoint v5 candidate shards without re-executing seed cells.

    Each shard may have observed a different manual-coin depth.  Lower-depth
    rows are lifted onto the deepest shard's common paired support by repeating
    their already-complete prefix leaf.  Cell-local effects and scores are
    repeated, while root-information values are deduplicated by public value
    identity across the final union.
    """
    if not parts:
        raise ValueError("hierarchical evidence merge requires at least one part")
    if len(producer_contract_fingerprint) != 64:
        raise ValueError("merged producer contract must be SHA-256 hex")
    actions = tuple(action for part in parts for action in part.outcome_batch.actions)
    if len(set(actions)) != len(actions):
        raise ValueError("merged hierarchical candidate supports must be disjoint")
    target_depth = max(part.outcome_batch.chance_depth for part in parts)
    target_part = next(
        part for part in parts if part.outcome_batch.chance_depth == target_depth
    )
    target_support = target_part.outcome_batch.outcomes[0].scenario_support
    for part in parts:
        batch = part.outcome_batch
        if batch.chance_depth == target_depth and (
            batch.outcomes[0].scenario_support_fingerprint
            != target_support.support_fingerprint
        ):
            raise ValueError("equal-depth planner shards use different supports")

    leaf_cells: list[RootInformationLeafCell] = []
    candidate_specs: list[
        tuple[
            OptionOutcome,
            tuple[OptionOutcomeCell, ...],
            tuple[ContinuationDecision, ...],
        ]
    ] = []
    endpoints: list[np.ndarray] = []
    engine_results: list[np.ndarray] = []
    exact_effects: list[np.ndarray] = []
    rules_exact: list[np.ndarray] = []
    path_steps: list[np.ndarray] = []
    cell_scores: list[np.ndarray] = []
    leaf_value_by_fingerprint: dict[str, np.float32] = {}
    global_cell = 0
    target_count = len(target_support.scenarios)
    for part in parts:
        batch = part.outcome_batch
        scores = part.leaf_scores
        source_support = batch.outcomes[0].scenario_support
        gather = _support_gather(
            source=source_support,
            target=target_support,
            source_depth=batch.chance_depth,
            target_depth=target_depth,
        )
        source_count = len(source_support.scenarios)
        if scores.cell_scores.shape != (len(batch.actions) * source_count,):
            raise ValueError("planner shard cell scores are misaligned")
        if scores.unique_leaf_values.shape != (len(batch.leaves.leaves),):
            raise ValueError("planner shard unique values are misaligned")
        for leaf, value in zip(
            batch.leaves.leaves,
            scores.unique_leaf_values,
            strict=True,
        ):
            fingerprint = leaf.model_input_fingerprint
            previous = leaf_value_by_fingerprint.setdefault(
                fingerprint,
                np.float32(value),
            )
            if previous != np.float32(value):
                raise ValueError("one value identity has different leased values")
        source_cells = batch.scoring_cells
        for candidate_index, outcome in enumerate(batch.outcomes):
            source_start = candidate_index * source_count
            indices: npt.NDArray[np.int64] = source_start + gather
            endpoints.append(source_cells.endpoints[indices])
            engine_results.append(source_cells.engine_results[indices])
            exact_effects.append(source_cells.exact_effects[indices])
            rules_exact.append(source_cells.rules_exact_mask[indices])
            path_steps.append(batch.path_steps[indices])
            cell_scores.append(scores.cell_scores[indices])
            expanded_cells: list[OptionOutcomeCell] = []
            decisions_by_scenario: dict[int, list[ContinuationDecision]] = {}
            for decision in outcome.continuation_decisions:
                decisions_by_scenario.setdefault(decision.scenario_index, []).append(
                    decision
                )
            expanded_decisions: list[ContinuationDecision] = []
            for target_index, source_index_raw in enumerate(gather):
                source_index = int(source_index_raw)
                source_cell = outcome.cells[source_index]
                target_scenario = target_support.scenarios[target_index]
                local_leaf_index = source_cell.leaf_index
                if local_leaf_index is not None:
                    leaf_cells.append(
                        RootInformationLeafCell(
                            cell_index=global_cell,
                            leaf=batch.leaves.leaves[local_leaf_index],
                        )
                    )
                expanded_cells.append(
                    replace(
                        source_cell,
                        scenario_index=target_index,
                        scenario_fingerprint=target_scenario.scenario_fingerprint,
                        weight=target_scenario.weight,
                    )
                )
                expanded_decisions.extend(
                    replace(decision, scenario_index=target_index)
                    for decision in decisions_by_scenario.get(source_index, ())
                )
                global_cell += 1
            candidate_specs.append(
                (
                    outcome,
                    tuple(expanded_cells),
                    tuple(expanded_decisions),
                )
            )
    final_cell_count = len(actions) * target_count
    if global_cell != final_cell_count:
        raise AssertionError("merged planner cell count is inconsistent")
    leaves = deduplicate_root_information_leaves(
        cell_count=final_cell_count,
        cells=tuple(leaf_cells),
    )
    outcomes: list[OptionOutcome] = []
    for candidate_index, (source, cells, decisions) in enumerate(candidate_specs):
        remapped_cells: list[OptionOutcomeCell] = []
        start = candidate_index * target_count
        for offset, cell in enumerate(cells):
            leaf_index = int(leaves.cell_to_leaf[start + offset])
            remapped_cells.append(
                replace(
                    cell,
                    leaf_index=None if leaf_index < 0 else leaf_index,
                )
            )
        outcomes.append(
            replace(
                source,
                scenario_support=target_support,
                cells=tuple(remapped_cells),
                continuation_decisions=decisions,
            )
        )

    merged_cells = LeafScoringBatch(
        endpoints=_immutable_concat(endpoints, dtype=np.int32),
        engine_results=_immutable_concat(engine_results, dtype=np.int32),
        exact_effects=_immutable_concat(exact_effects, dtype=np.float32),
        rules_exact_mask=_immutable_concat(rules_exact, dtype=np.bool_),
        root_player=parts[0].outcome_batch.scoring_cells.root_player,
    )
    merged_steps = _immutable_concat(path_steps, dtype=np.int32)
    merged_cell_scores = _immutable_concat(cell_scores, dtype=np.float32)
    merged_unique_values = np.asarray(
        tuple(
            leaf_value_by_fingerprint[leaf.model_input_fingerprint]
            for leaf in leaves.leaves
        ),
        dtype=np.float32,
    )
    merged_unique_values.setflags(write=False)
    merged_scores = SharedLeafScoreResult(
        cell_scores=merged_cell_scores,
        unique_leaf_values=merged_unique_values,
        scorer_fingerprint=scorer.scorer_fingerprint,
        leaf_bootstrapped=bool(leaves.leaves),
    )
    merged_batch = HierarchicalOutcomeBatch(
        actions=actions,
        outcomes=tuple(outcomes),
        leaves=leaves,
        scoring_cells=merged_cells,
        path_steps=merged_steps,
        chance_depth=target_depth,
        continuation_nodes=sum(part.outcome_batch.continuation_nodes for part in parts),
        producer_contract_fingerprint=producer_contract_fingerprint,
    )
    return score_hierarchical_outcomes(
        merged_batch,
        scorer=scorer,
        legal_action_count=legal_action_count,
        support_exhaustive=support_exhaustive,
        precomputed_leaf_scores=merged_scores,
    )


def _support_gather(
    *,
    source: ScenarioSupport,
    target: ScenarioSupport,
    source_depth: int,
    target_depth: int,
) -> npt.NDArray[np.int64]:
    """Map a deeper common chance grid onto one completed source shard."""
    if source_depth < 0 or target_depth < source_depth:
        raise ValueError("planner chance-depth merge is invalid")
    if source_depth == target_depth:
        if source.support_fingerprint != target.support_fingerprint:
            raise ValueError("planner shards differ at the common chance depth")
        gather: npt.NDArray[np.int64] = np.arange(len(target.scenarios), dtype=np.int64)
        gather.setflags(write=False)
        return gather
    by_key = {
        (scenario.belief_world_fingerprint, scenario.chance_support_handle): index
        for index, scenario in enumerate(source.scenarios)
    }
    by_belief = {
        scenario.belief_world_fingerprint: index
        for index, scenario in enumerate(source.scenarios)
    }
    gather_values: list[int] = []
    target_marker = 1 << target_depth
    source_marker = 1 << source_depth
    for target_scenario in target.scenarios:
        if source_depth == 0:
            source_index = by_belief.get(target_scenario.belief_world_fingerprint)
        else:
            target_path = target_scenario.chance_support_handle - target_marker
            if target_path < 0:
                raise ValueError("target support lacks encoded manual-coin paths")
            source_path = target_path >> (target_depth - source_depth)
            source_index = by_key.get(
                (
                    target_scenario.belief_world_fingerprint,
                    source_marker | source_path,
                )
            )
        if source_index is None:
            raise ValueError("planner shards do not share belief-world support")
        source_scenario = source.scenarios[source_index]
        source_mass = source_scenario.weight * (2**source_depth)
        target_mass = target_scenario.weight * (2**target_depth)
        if not math.isclose(source_mass, target_mass, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("planner shards differ in posterior scenario mass")
        gather_values.append(source_index)
    gather = np.asarray(gather_values, dtype=np.int64)
    gather.setflags(write=False)
    return gather


def _immutable_concat(
    chunks: list[npt.NDArray[Any]],
    *,
    dtype: type[np.generic],
) -> npt.NDArray[Any]:
    if not chunks:
        raise ValueError("planner evidence merge received no aligned chunks")
    values = np.concatenate(chunks, axis=0).astype(dtype, copy=False)
    values.setflags(write=False)
    return cast(npt.NDArray[Any], values)


__all__ = [
    "PlannerAggregateEvidence",
    "ScoredHierarchicalEvidence",
    "merge_scored_hierarchical_evidence",
    "score_hierarchical_outcomes",
]
