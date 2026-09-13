"""Producer flow from compact engine cells to deduplicated information leaves."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import numpy as np

from ptcg_rl.agent.search.hierarchical_contract import (
    DecisionTransition,
    OptionOutcome,
    OptionOutcomeCell,
    StableContinuationControllerIdentity,
)
from ptcg_rl.agent.search.planner_fallback import (
    PlannerEvidenceError,
    PlannerFallbackReason,
)
from ptcg_rl.agent.search.planner_scoring import (
    SharedLeafScoreResult,
    engine_result_to_root_value,
)
from ptcg_rl.agent.search.root_information import (
    RootActorRelation,
    RootInformationLeaf,
    RootInformationLeafBatch,
    RootInformationLeafCell,
    canonical_float32_vector_bytes,
    deduplicate_root_information_leaves,
    validate_information_history_fingerprint,
)
from ptcg_rl.engine.compact_consequence import (
    CompactConsequenceBatch,
    ScenarioSupportMode,
    SemanticEndpoint,
)

_INFORMATION_HISTORY_DOMAIN = b"ptcg-rl/reached-information-history/v2\x00"


@dataclass(frozen=True, slots=True)
class RootInformationCellContext:
    """Root-visible producer inputs aligned with one consequence cell."""

    producer_context: bytes
    belief_summary: tuple[float, ...]

    def __post_init__(self) -> None:
        """Reject missing or non-finite public context features."""
        if not self.producer_context:
            raise ValueError("producer_context must not be empty")
        if any(not math.isfinite(value) for value in self.belief_summary):
            raise ValueError("belief_summary must be finite")


@dataclass(frozen=True, slots=True)
class RootInformationTransitionBatch:
    """Local transitions plus unique value leaves and their cell gather map."""

    transitions: tuple[DecisionTransition, ...]
    value_leaves: RootInformationLeafBatch


def build_direct_option_outcomes(
    compact_batch: CompactConsequenceBatch,
    transition_batch: RootInformationTransitionBatch,
    *,
    scores: SharedLeafScoreResult,
    controller: StableContinuationControllerIdentity,
    root_player: int,
) -> tuple[OptionOutcome, ...]:
    """Build complete outcomes only when every first transition is comparable.

    This producer path is intentionally limited to options that need no later
    strategic choice.  Therefore a complete trace is a validated fact rather
    than a caller-provided assertion.  Hierarchical continuations must use a
    separate trace-producing controller before they can form ``OptionOutcome``.
    """
    if not compact_batch.contract.scenario_grid_complete:
        raise PlannerEvidenceError(
            PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
            "direct outcomes require a complete compact scenario grid",
        )
    if len(transition_batch.transitions) != compact_batch.cell_count:
        raise PlannerEvidenceError(
            PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
            "decision transitions do not cover the compact scenario grid",
        )
    cell_scores = np.asarray(scores.cell_scores, dtype=np.float32)
    if cell_scores.shape != (compact_batch.cell_count,) or not bool(
        np.isfinite(cell_scores).all()
    ):
        raise PlannerEvidenceError(
            PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
            "leaf scores do not cover the compact scenario grid",
        )
    if scores.scorer_fingerprint != controller.scorer_fingerprint:
        raise PlannerEvidenceError(
            PlannerFallbackReason.MODEL_VERSION_MISMATCH,
            "direct outcome scores differ from the continuation controller",
        )

    support = compact_batch.contract.scenario_support
    source_history: str | None = None
    expected_leaf_bootstrapped = False
    outcomes: list[OptionOutcome] = []
    for candidate_index, candidate_fingerprint in enumerate(
        compact_batch.contract.candidate_fingerprints
    ):
        outcome_cells: list[OptionOutcomeCell] = []
        for scenario_index, scenario in enumerate(support.scenarios):
            cell_index = candidate_index * compact_batch.scenario_count + scenario_index
            transition = transition_batch.transitions[cell_index]
            compact_row = compact_batch.row_at(candidate_index, scenario_index)
            if (
                transition.candidate_index != candidate_index
                or transition.scenario_index != scenario_index
                or transition.candidate_fingerprint != candidate_fingerprint
                or transition.scenario_fingerprint != scenario.scenario_fingerprint
                or transition.scenario_support_fingerprint
                != support.support_fingerprint
                or transition.endpoint is not compact_row.endpoint
                or transition.transition_steps != compact_row.transition_steps
                or transition.rules_exact != compact_row.rules_exact
                or transition.error_code != compact_row.error_code
            ):
                raise PlannerEvidenceError(
                    PlannerFallbackReason.FINGERPRINT_MISMATCH,
                    "decision transition differs from its compact grid cell",
                )
            if source_history is None:
                source_history = transition.source_information_history_fingerprint
            elif transition.source_information_history_fingerprint != source_history:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.FINGERPRINT_MISMATCH,
                    "decision transitions use different root information histories",
                )
            if not transition.comparable:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.EVIDENCE_ABSENT,
                    "direct outcome requires a traced strategic continuation",
                )

            leaf_index = int(transition_batch.value_leaves.cell_to_leaf[cell_index])
            terminal_result: float | None = None
            if transition.endpoint is SemanticEndpoint.TERMINAL:
                if leaf_index != -1:
                    raise PlannerEvidenceError(
                        PlannerFallbackReason.FINGERPRINT_MISMATCH,
                        "terminal transition unexpectedly maps to a value leaf",
                    )
                expected = np.float32(
                    engine_result_to_root_value(
                        compact_batch.row_at(
                            candidate_index,
                            scenario_index,
                        ).engine_result,
                        root_player,
                    )
                )
                if np.float32(cell_scores[cell_index]) != expected:
                    raise PlannerEvidenceError(
                        PlannerFallbackReason.FINGERPRINT_MISMATCH,
                        "terminal score contradicts its compact engine result",
                    )
                terminal_result = float(expected)
            else:
                expected_leaf_bootstrapped = True
                if not 0 <= leaf_index < len(transition_batch.value_leaves.leaves):
                    raise PlannerEvidenceError(
                        PlannerFallbackReason.LEAF_VALUE_UNAVAILABLE,
                        "nonterminal transition is missing its value leaf",
                    )
                leaf = transition_batch.value_leaves.leaves[leaf_index]
                if leaf.endpoint is not transition.endpoint:
                    raise PlannerEvidenceError(
                        PlannerFallbackReason.FINGERPRINT_MISMATCH,
                        "transition endpoint differs from its gathered value leaf",
                    )
            outcome_cells.append(
                OptionOutcomeCell(
                    scenario_index=scenario_index,
                    scenario_fingerprint=scenario.scenario_fingerprint,
                    weight=scenario.weight,
                    endpoint=transition.endpoint,
                    information_history_fingerprint=(
                        transition.reached_information_history_fingerprint
                    ),
                    leaf_index=None if leaf_index == -1 else leaf_index,
                    terminal_result=terminal_result,
                    rules_exact=transition.rules_exact,
                )
            )
        if source_history is None:
            raise AssertionError("validated scenario support cannot be empty")
        outcomes.append(
            OptionOutcome(
                root_information_history_fingerprint=source_history,
                root_candidate_fingerprint=candidate_fingerprint,
                scenario_support=support,
                controller=controller,
                cells=tuple(outcome_cells),
                continuation_trace_complete=True,
            )
        )
    if scores.leaf_bootstrapped != expected_leaf_bootstrapped:
        raise PlannerEvidenceError(
            PlannerFallbackReason.FINGERPRINT_MISMATCH,
            "leaf bootstrap exactness differs from the executed endpoints",
        )
    return tuple(outcomes)


def build_root_information_transition_batch(
    batch: CompactConsequenceBatch,
    *,
    source_information_history_fingerprint: str,
    cell_contexts: tuple[RootInformationCellContext, ...],
) -> RootInformationTransitionBatch:
    """Bind exact native cells to public information histories without decode."""
    validate_information_history_fingerprint(
        source_information_history_fingerprint
    )
    if len(cell_contexts) != batch.cell_count:
        raise PlannerEvidenceError(
            PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
            "producer contexts do not cover the consequence grid",
        )
    transitions: list[DecisionTransition] = []
    leaf_cells: list[RootInformationLeafCell] = []
    for candidate_index in range(batch.candidate_count):
        for scenario_index in range(batch.scenario_count):
            cell_index = candidate_index * batch.scenario_count + scenario_index
            row = batch.row_at(candidate_index, scenario_index)
            if not row.valid or row.error_code != 0:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.ENGINE_ERROR,
                    "native decision transition contains an invalid cell",
                )
            if not row.rules_exact:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.RULES_INEXACT,
                    "native decision transition is not rules-exact",
                )
            endpoint = row.endpoint
            if endpoint is SemanticEndpoint.INVALID:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.ENGINE_ERROR,
                    "native decision transition has an invalid endpoint",
                )
            if (
                endpoint is SemanticEndpoint.CHANCE_PROMPT
                and batch.contract.scenario_support.mode
                is ScenarioSupportMode.SAMPLED_BELIEF_CHANCE_UNSUPPORTED
            ):
                raise PlannerEvidenceError(
                    PlannerFallbackReason.UNSUPPORTED_CHANCE,
                    "engine chance surface has no reproducible paired support",
                )
            context = cell_contexts[cell_index]
            observation = row.root_observable_state.payload.tobytes(order="C")
            exact_effect = tuple(float(value) for value in row.exact_effect)
            reached_fingerprint = _reached_information_history_fingerprint(
                source_information_history_fingerprint=(
                    source_information_history_fingerprint
                ),
                candidate_fingerprint=row.identity.candidate_fingerprint,
                observation=observation,
                producer_context=context.producer_context,
                belief_summary=context.belief_summary,
                endpoint=endpoint,
            )
            if endpoint in {
                SemanticEndpoint.SAME_SEAT_MAIN,
                SemanticEndpoint.TURN_HANDOFF,
            }:
                leaf = RootInformationLeaf(
                    root_observable_state=observation,
                    producer_context=context.producer_context,
                    belief_summary=context.belief_summary,
                    exact_effect=exact_effect,
                    actor_relation=(
                        RootActorRelation.SAME_SEAT
                        if endpoint is SemanticEndpoint.SAME_SEAT_MAIN
                        else RootActorRelation.OTHER_SEAT
                    ),
                    endpoint=endpoint,
                )
                leaf_cells.append(
                    RootInformationLeafCell(cell_index=cell_index, leaf=leaf)
                )
            transitions.append(
                DecisionTransition(
                    candidate_index=candidate_index,
                    scenario_index=scenario_index,
                    candidate_fingerprint=row.identity.candidate_fingerprint,
                    scenario_fingerprint=row.identity.scenario_fingerprint,
                    scenario_support_fingerprint=(
                        row.identity.scenario_support_fingerprint
                    ),
                    source_information_history_fingerprint=(
                        source_information_history_fingerprint
                    ),
                    reached_information_history_fingerprint=reached_fingerprint,
                    endpoint=endpoint,
                    transition_steps=row.transition_steps,
                    rules_exact=row.rules_exact,
                    error_code=row.error_code,
                )
            )
    return RootInformationTransitionBatch(
        transitions=tuple(transitions),
        value_leaves=deduplicate_root_information_leaves(
            cell_count=batch.cell_count,
            cells=tuple(leaf_cells),
        ),
    )


def _reached_information_history_fingerprint(
    *,
    source_information_history_fingerprint: str,
    candidate_fingerprint: str,
    observation: bytes,
    producer_context: bytes,
    belief_summary: tuple[float, ...],
    endpoint: SemanticEndpoint,
) -> str:
    digest = hashlib.sha256()
    digest.update(_INFORMATION_HISTORY_DOMAIN)
    digest.update(bytes.fromhex(source_information_history_fingerprint))
    digest.update(bytes.fromhex(candidate_fingerprint))
    for payload in (observation, producer_context):
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(payload)
    belief = canonical_float32_vector_bytes(belief_summary)
    digest.update(struct.pack(">Q", len(belief)))
    digest.update(belief)
    digest.update(struct.pack(">i", int(endpoint)))
    return digest.hexdigest()


__all__ = [
    "RootInformationCellContext",
    "RootInformationTransitionBatch",
    "build_direct_option_outcomes",
    "build_root_information_transition_batch",
]
