"""Learner-side conversion of native consequences into student-safe targets."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Literal, cast

import torch

from ptcg_rl.engine.consequence_identity import candidate_action_fingerprint
from ptcg_rl.engine.native_consequence_payload import NativeConsequenceEndpoint
from ptcg_rl.model import AgentPolicyValueNet
from ptcg_rl.rl.amortized_policy_iteration.belief_reanalysis import (
    LeafActorInput,
    NativeReanalysisCell,
    NativeReanalysisResult,
    StudentReanalysisRoot,
)
from ptcg_rl.rl.amortized_policy_iteration.target_builder import (
    AggregatedActionTarget,
    Wdl,
    WorldConsequenceTarget,
    aggregate_world_consequences,
)
from ptcg_rl.rl.amortized_policy_iteration.tensor_batch import (
    collate_information_sets,
)

AutocastMode = Literal["bf16", "off"]


@dataclass(frozen=True, slots=True)
class CounterfactualRootTarget:
    """Student-visible retained sibling targets for one information set."""

    root: StudentReanalysisRoot
    actions: tuple[tuple[int, ...], ...]
    target_wdl: tuple[Wdl, ...]
    proposal_probabilities: tuple[float, ...]
    old_policy_probabilities: tuple[float, ...]
    exhaustive: bool
    proposal_policy_version: int
    valid_worlds: int
    omitted_worlds: int

    def __post_init__(self) -> None:
        count = len(self.actions)
        if count <= 0 or any(
            len(values) != count
            for values in (
                self.target_wdl,
                self.proposal_probabilities,
                self.old_policy_probabilities,
            )
        ):
            raise ValueError("counterfactual target fields must be non-empty and aligned")


def build_counterfactual_targets(
    results: Sequence[NativeReanalysisResult],
    *,
    target_model: AgentPolicyValueNet,
    device: torch.device | str,
    autocast: AutocastMode,
) -> tuple[CounterfactualRootTarget, ...]:
    """Bootstrap actual next actors and aggregate shared particles per action."""
    usable = tuple(result for result in results if result.cells and not result.error_message)
    if not usable:
        return ()
    cells = tuple(cell for result in usable for cell in result.cells)
    leaf_cells = tuple(
        cell
        for cell in cells
        if cell.error == 0
        and cell.endpoint
        not in (
            NativeConsequenceEndpoint.TERMINAL,
            NativeConsequenceEndpoint.INVALID,
            NativeConsequenceEndpoint.CHANCE_PROMPT,
        )
        and cell.leaf is not None
    )
    leaf_values: dict[int, Wdl] = {}
    if leaf_cells:
        leaves = tuple(cast(LeafActorInput, cell.leaf) for cell in leaf_cells)
        batch = collate_information_sets(
            states=tuple(leaf.state for leaf in leaves),
            options=tuple(leaf.options for leaf in leaves),
            min_counts=tuple(leaf.min_count for leaf in leaves),
            max_counts=tuple(leaf.max_count for leaf in leaves),
            decks=tuple(leaf.deck for leaf in leaves),
            device=device,
        )
        with torch.inference_mode(), _autocast_context(
            target_model,
            autocast=autocast,
        ):
            conditioned = target_model.encode_conditioned_state(
                batch.states,
                batch.decks,
            )
            probabilities = torch.softmax(
                target_model.action_value_state_logits_from_conditioned(
                    conditioned
                ).float(),
                dim=-1,
            ).cpu()
        for cell, row in zip(leaf_cells, probabilities.tolist(), strict=True):
            leaf_values[id(cell)] = (float(row[0]), float(row[1]), float(row[2]))

    output = []
    for result in usable:
        world_rows = tuple(
            _world_target(result, cell, successor_wdl=leaf_values.get(id(cell)))
            for cell in result.cells
        )
        aggregated = aggregate_world_consequences(world_rows)
        output_row = _student_target(result, aggregated)
        if output_row is not None:
            output.append(output_row)
    return tuple(output)


def _world_target(
    result: NativeReanalysisResult,
    cell: NativeReanalysisCell,
    *,
    successor_wdl: Wdl | None,
) -> WorldConsequenceTarget:
    kind: Literal[
        "terminal", "same_seat", "handoff", "infrastructure_error"
    ]
    if cell.error != 0 or cell.endpoint in (
        NativeConsequenceEndpoint.INVALID,
        NativeConsequenceEndpoint.CHANCE_PROMPT,
    ):
        kind = "infrastructure_error"
    elif cell.endpoint is NativeConsequenceEndpoint.TERMINAL:
        kind = "terminal"
    elif cell.endpoint in (
        NativeConsequenceEndpoint.SAME_SEAT_MAIN,
        NativeConsequenceEndpoint.ROOT_STRATEGIC_PROMPT,
    ):
        kind = "same_seat"
    elif cell.endpoint is NativeConsequenceEndpoint.TURN_HANDOFF:
        kind = "handoff"
    else:
        kind = "infrastructure_error"
    if kind != "terminal" and kind != "infrastructure_error" and successor_wdl is None:
        kind = "infrastructure_error"
    return WorldConsequenceTarget(
        root_id=result.root.student.root_id,
        candidate_id=cell.candidate_id,
        particle_id=cell.particle_id,
        kind=kind,
        root_player=cell.root_player,
        leaf_player=cell.leaf_player if kind in ("same_seat", "handoff") else None,
        engine_result=cell.engine_result if kind == "terminal" else None,
        successor_actor_wdl=(
            successor_wdl if kind in ("same_seat", "handoff") else None
        ),
        sampling_weight=cell.sampling_weight,
    )


def _student_target(
    result: NativeReanalysisResult,
    aggregated: Sequence[AggregatedActionTarget],
) -> CounterfactualRootTarget | None:
    by_candidate = {row.candidate_id: row for row in aggregated}
    retained = []
    for candidate in result.proposal.candidates:
        candidate_id = candidate_action_fingerprint(candidate.action)
        target = by_candidate.get(candidate_id)
        if target is not None:
            retained.append((candidate, target))
    if not retained:
        return None
    return CounterfactualRootTarget(
        root=result.root.student,
        actions=tuple(candidate.action for candidate, _target in retained),
        target_wdl=tuple(target.root_wdl for _candidate, target in retained),
        proposal_probabilities=tuple(
            candidate.q_prop for candidate, _target in retained
        ),
        old_policy_probabilities=tuple(
            candidate.behavior_probability for candidate, _target in retained
        ),
        exhaustive=(
            result.proposal.exhaustive
            and len(retained) == len(result.proposal.candidates)
        ),
        proposal_policy_version=result.proposal_policy_version,
        valid_worlds=sum(target.valid_worlds for _candidate, target in retained),
        omitted_worlds=sum(target.omitted_worlds for _candidate, target in retained),
    )


def _autocast_context(
    model: AgentPolicyValueNet,
    *,
    autocast: AutocastMode,
) -> AbstractContextManager[None]:
    if autocast == "off" or next(model.parameters()).device.type != "cuda":
        return nullcontext()
    return cast(
        AbstractContextManager[None],
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    )


__all__ = ["CounterfactualRootTarget", "build_counterfactual_targets"]
