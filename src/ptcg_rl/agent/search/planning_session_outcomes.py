"""Convert complete v5 search branches into paired option outcomes."""

from __future__ import annotations

import itertools
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
import orjson

from ptcg_rl.agent.search.hierarchical_contract import (
    ContinuationDecision,
    OptionOutcome,
    OptionOutcomeCell,
    StableContinuationControllerIdentity,
)
from ptcg_rl.agent.search.planner_fallback import (
    PlannerEvidenceError,
    PlannerFallbackReason,
)
from ptcg_rl.agent.search.planner_scoring import (
    LeafScoringBatch,
    engine_result_to_root_value,
)
from ptcg_rl.agent.search.planning_session_contract import (
    HierarchicalOutcomeBatch,
    HierarchicalSearchRequest,
)
from ptcg_rl.agent.search.root_information import (
    RootActorRelation,
    RootInformationLeaf,
    RootInformationLeafCell,
    deduplicate_root_information_leaves,
)
from ptcg_rl.engine.compact_consequence import (
    ScenarioHandle,
    ScenarioSupport,
    ScenarioSupportMode,
    SemanticEndpoint,
    scenario_fingerprint,
    scenario_support_fingerprint,
)
from ptcg_rl.engine.consequence_identity import (
    canonical_chance_support_fingerprint,
)
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.forward_model import (
    dynamic_effect_feature_from_dict_resolution,
)
from ptcg_rl.engine.probe_resolution import ProbeTransition


@dataclass(frozen=True, slots=True)
class CompletedPlanningBranch:
    """One exact comparable leaf before common chance-path expansion."""

    candidate_index: int
    belief_world_index: int
    chance_bits: tuple[int, ...]
    endpoint: SemanticEndpoint
    final_observation_bytes: bytes
    producer_context: bytes
    belief_summary: tuple[float, ...]
    information_history_fingerprint: str
    engine_result: int
    transition_steps: int
    transitions: tuple[ProbeTransition, ...]
    continuation_decisions: tuple[tuple[str, str], ...]


def build_hierarchical_outcome_batch(
    *,
    request: HierarchicalSearchRequest,
    base_support: ScenarioSupport,
    root_information_history_fingerprint: str,
    branches: tuple[CompletedPlanningBranch, ...],
    controller: StableContinuationControllerIdentity,
    continuation_nodes: int,
    producer_contract_fingerprint: str,
) -> HierarchicalOutcomeBatch:
    """Expand prefix-free coin leaves onto one candidate-shared support."""
    candidate_count = len(request.candidate_actions)
    belief_count = len(base_support.scenarios)
    by_root_cell = _index_complete_branches(
        branches,
        candidate_count=candidate_count,
        belief_count=belief_count,
    )
    chance_depth = max((len(branch.chance_bits) for branch in branches), default=0)
    chance_paths = tuple(itertools.product((0, 1), repeat=chance_depth))
    support = _expanded_support(base_support, chance_paths)
    scenario_count = len(support.scenarios)
    cell_count = candidate_count * scenario_count
    resolved: list[CompletedPlanningBranch] = []
    for candidate_index in range(candidate_count):
        for belief_index in range(belief_count):
            cell_branches = by_root_cell[(candidate_index, belief_index)]
            for chance_path in chance_paths:
                resolved.append(_branch_for_path(cell_branches, chance_path))
    if len(resolved) != cell_count:
        raise AssertionError("expanded chance grid has the wrong cell count")

    exact_effects = np.empty(
        (cell_count, DYNAMIC_EFFECT_FEATURE_SIZE),
        dtype=np.float32,
    )
    endpoints = np.empty(cell_count, dtype=np.int32)
    engine_results = np.empty(cell_count, dtype=np.int32)
    rules_exact = np.ones(cell_count, dtype=np.bool_)
    path_steps = np.empty(cell_count, dtype=np.int32)
    leaf_cells: list[RootInformationLeafCell] = []
    terminal_values: list[float | None] = []
    for cell_index, branch in enumerate(resolved):
        candidate_index = cell_index // scenario_count
        after_observation = orjson.loads(branch.final_observation_bytes)
        feature = dynamic_effect_feature_from_dict_resolution(
            select=request.candidate_actions[candidate_index],
            before_observation=request.root_observation,
            after_observation=after_observation,
            probe_transitions=branch.transitions,
            perspective_player=request.root_player,
        )
        exact_effects[cell_index] = feature.to_numpy()
        endpoints[cell_index] = int(branch.endpoint)
        engine_results[cell_index] = branch.engine_result
        path_steps[cell_index] = branch.transition_steps
        if branch.endpoint is SemanticEndpoint.TERMINAL:
            terminal_values.append(
                engine_result_to_root_value(
                    branch.engine_result,
                    request.root_player,
                )
            )
            continue
        terminal_values.append(None)
        relation = (
            RootActorRelation.SAME_SEAT
            if branch.endpoint is SemanticEndpoint.SAME_SEAT_MAIN
            else RootActorRelation.OTHER_SEAT
        )
        leaf_cells.append(
            RootInformationLeafCell(
                cell_index=cell_index,
                leaf=RootInformationLeaf(
                    root_observable_state=branch.final_observation_bytes,
                    producer_context=branch.producer_context,
                    belief_summary=branch.belief_summary,
                    exact_effect=tuple(float(value) for value in feature.to_numpy()),
                    actor_relation=relation,
                    endpoint=branch.endpoint,
                ),
            )
        )
    for values in (exact_effects, endpoints, engine_results, rules_exact, path_steps):
        values.setflags(write=False)
    leaves = deduplicate_root_information_leaves(
        cell_count=cell_count,
        cells=tuple(leaf_cells),
    )
    scoring_cells = LeafScoringBatch(
        endpoints=endpoints,
        engine_results=engine_results,
        exact_effects=exact_effects,
        rules_exact_mask=rules_exact,
        root_player=request.root_player,
    )
    outcomes = _build_option_outcomes(
        request=request,
        support=support,
        controller=controller,
        root_information_history_fingerprint=(
            root_information_history_fingerprint
        ),
        resolved=tuple(resolved),
        terminal_values=tuple(terminal_values),
        leaves=leaves,
    )
    return HierarchicalOutcomeBatch(
        actions=request.candidate_actions,
        outcomes=outcomes,
        leaves=leaves,
        scoring_cells=scoring_cells,
        path_steps=path_steps,
        chance_depth=chance_depth,
        continuation_nodes=continuation_nodes,
        producer_contract_fingerprint=producer_contract_fingerprint,
    )


def _index_complete_branches(
    branches: tuple[CompletedPlanningBranch, ...],
    *,
    candidate_count: int,
    belief_count: int,
) -> dict[tuple[int, int], tuple[CompletedPlanningBranch, ...]]:
    grouped: dict[tuple[int, int], list[CompletedPlanningBranch]] = {
        (candidate, belief): []
        for candidate in range(candidate_count)
        for belief in range(belief_count)
    }
    for branch in branches:
        key = (branch.candidate_index, branch.belief_world_index)
        if key not in grouped:
            raise PlannerEvidenceError(
                PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                "completed branch is outside the root candidate grid",
            )
        grouped[key].append(branch)
    result = {key: tuple(value) for key, value in grouped.items()}
    if any(not value for value in result.values()):
        raise PlannerEvidenceError(
            PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
            "a root candidate/belief cell has no completed chance leaf",
        )
    return result


def _branch_for_path(
    branches: tuple[CompletedPlanningBranch, ...],
    chance_path: tuple[int, ...],
) -> CompletedPlanningBranch:
    matches = tuple(
        branch
        for branch in branches
        if chance_path[: len(branch.chance_bits)] == branch.chance_bits
    )
    if len(matches) != 1:
        raise PlannerEvidenceError(
            PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
            "manual-coin leaves are not one complete prefix-free tree",
        )
    return matches[0]


def _expanded_support(
    base_support: ScenarioSupport,
    chance_paths: tuple[tuple[int, ...], ...],
) -> ScenarioSupport:
    chance_depth = len(chance_paths[0]) if chance_paths else 0
    if chance_depth == 0:
        return base_support
    handles: list[ScenarioHandle] = []
    path_count = len(chance_paths)
    for base in base_support.scenarios:
        for path in chance_paths:
            path_payload = b"manual-coin-bits/v1\x00" + struct.pack(
                ">I", chance_depth
            ) + bytes(path)
            chance_fingerprint = canonical_chance_support_fingerprint(path_payload)
            path_value = 0
            for bit in path:
                path_value = (path_value << 1) | bit
            chance_handle = (1 << chance_depth) | path_value
            handles.append(
                ScenarioHandle(
                    belief_world_handle=base.belief_world_handle,
                    chance_support_handle=chance_handle,
                    belief_world_fingerprint=base.belief_world_fingerprint,
                    chance_support_fingerprint=chance_fingerprint,
                    scenario_fingerprint=scenario_fingerprint(
                        belief_world_fingerprint=base.belief_world_fingerprint,
                        chance_support_fingerprint=chance_fingerprint,
                    ),
                    weight=base.weight / path_count,
                )
            )
    frozen = tuple(handles)
    mode = ScenarioSupportMode.SAMPLED_BELIEF_MANUAL_COIN_ENUMERATED
    return ScenarioSupport(
        mode=mode,
        scenarios=frozen,
        support_fingerprint=scenario_support_fingerprint(mode, frozen),
    )


def _build_option_outcomes(
    *,
    request: HierarchicalSearchRequest,
    support: ScenarioSupport,
    controller: StableContinuationControllerIdentity,
    root_information_history_fingerprint: str,
    resolved: tuple[CompletedPlanningBranch, ...],
    terminal_values: tuple[float | None, ...],
    leaves: Any,
) -> tuple[OptionOutcome, ...]:
    scenario_count = len(support.scenarios)
    outcomes: list[OptionOutcome] = []
    for candidate_index, action in enumerate(request.candidate_actions):
        cells: list[OptionOutcomeCell] = []
        decisions: list[ContinuationDecision] = []
        for scenario_index, scenario in enumerate(support.scenarios):
            cell_index = candidate_index * scenario_count + scenario_index
            branch = resolved[cell_index]
            leaf_index = int(leaves.cell_to_leaf[cell_index])
            cells.append(
                OptionOutcomeCell(
                    scenario_index=scenario_index,
                    scenario_fingerprint=scenario.scenario_fingerprint,
                    weight=scenario.weight,
                    endpoint=branch.endpoint,
                    information_history_fingerprint=(
                        branch.information_history_fingerprint
                    ),
                    leaf_index=None if leaf_index < 0 else leaf_index,
                    terminal_result=terminal_values[cell_index],
                )
            )
            decisions.extend(
                ContinuationDecision(
                    scenario_index=scenario_index,
                    information_history_fingerprint=history,
                    action_fingerprint=action_fingerprint,
                )
                for history, action_fingerprint in branch.continuation_decisions
            )
        from ptcg_rl.engine.consequence_identity import candidate_action_fingerprint

        outcomes.append(
            OptionOutcome(
                root_information_history_fingerprint=(
                    root_information_history_fingerprint
                ),
                root_candidate_fingerprint=candidate_action_fingerprint(action),
                scenario_support=support,
                controller=controller,
                cells=tuple(cells),
                continuation_trace_complete=True,
                continuation_decisions=tuple(decisions),
            )
        )
    return tuple(outcomes)


__all__ = ["CompletedPlanningBranch", "build_hierarchical_outcome_batch"]
