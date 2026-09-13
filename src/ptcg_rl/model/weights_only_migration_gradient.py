"""One-backward learnability evidence for newly migrated output layers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

import torch
from torch import Tensor
from torch.nn import functional

from ptcg_rl.agent.search.root_information import RootActorRelation
from ptcg_rl.decks import DeckBatch
from ptcg_rl.engine.compact_consequence import SemanticEndpoint
from ptcg_rl.model.network import AgentPolicyValueNet, PlannerCandidateEvaluation
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.state_encoder import StateBatch


class MigrationAuditBatch(Protocol):
    """Structural inputs required by the synthetic backward audit."""

    @property
    def states(self) -> StateBatch:
        """Encoded state batch."""
        ...

    @property
    def options(self) -> OptionBatch:
        """Legal option batch."""
        ...

    @property
    def decks(self) -> DeckBatch | None:
        """Optional deck context."""
        ...

    @property
    def actions(self) -> tuple[tuple[int, ...], ...]:
        """Selected action rows."""
        ...

    @property
    def candidate_actions(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Planner candidate rows."""
        ...

    @property
    def candidate_features(self) -> tuple[Tensor, ...]:
        """Planner candidate features."""
        ...

    @property
    def ordered_rows(self) -> Tensor:
        """Ordered-selection markers."""
        ...


def run_backward_evidence(
    model: AgentPolicyValueNet,
    batch: MigrationAuditBatch,
) -> dict[str, Any]:
    """Prove zero-start planner parity and finite gradients in one backward."""
    model.zero_grad(set_to_none=True)
    conditioned = model.encode_conditioned_state(batch.states, batch.decks)
    evaluation = model.evaluate_actions_from_conditioned(
        conditioned,
        batch.options,
        batch.actions,
    )
    if evaluation.policy_context is None:
        raise RuntimeError("audit backward requires a reusable policy context")
    planner = model.evaluate_planner_candidates_from_context(
        evaluation.policy_context,
        batch.options,
        batch.candidate_actions,
        batch.candidate_features,
        ordered_rows=batch.ordered_rows,
        decks=batch.decks,
    )
    candidate_counts = tuple(len(group) for group in batch.candidate_actions)
    score_prior = torch.cat(
        tuple(
            torch.linspace(
                -0.4,
                0.4,
                count,
                dtype=planner.base_action_logprobs.dtype,
            )
            for count in candidate_counts
        )
    )
    (
        q_planner,
        behavior,
        q_logprobs,
        behavior_logprobs,
        proposal_logprobs,
    ) = _planner_distributions(
        planner,
        score_prior=score_prior,
        candidate_counts=candidate_counts,
    )
    zero_start_behavior_equals_q = torch.equal(q_planner, behavior)

    behavior_loss = planner.base_action_logprobs.new_zeros(())
    proposal_loss = planner.base_action_logprobs.new_zeros(())
    offset = 0
    for group_index, count in enumerate(candidate_counts):
        stop = offset + count
        selected_index = group_index % count
        behavior_loss = behavior_loss - (
            float(group_index + 1) * behavior_logprobs[offset:stop][selected_index]
        )
        target = q_planner[offset:stop]
        proposal_loss = proposal_loss + torch.sum(
            target * (q_logprobs[offset:stop] - proposal_logprobs[offset:stop])
        )
        offset = stop
    behavior_loss = behavior_loss / len(candidate_counts)
    proposal_loss = proposal_loss / len(candidate_counts)

    relations, endpoints = root_relation_endpoint_rows()
    root_predictions = model.root_information_values_from_conditioned(
        conditioned,
        actor_relations=relations,
        endpoints=endpoints,
        belief_summaries=torch.zeros(
            (
                len(batch.actions),
                model.config.root_perspective_value.belief_summary_dim,
            ),
            dtype=torch.float32,
        ),
    )
    root_targets = torch.where(
        evaluation.values.detach().ge(0.0),
        -torch.ones_like(evaluation.values),
        torch.ones_like(evaluation.values),
    )
    root_loss = functional.smooth_l1_loss(root_predictions, root_targets)
    total_loss = behavior_loss + proposal_loss + root_loss
    torch.autograd.backward(total_loss)

    layers = {
        "proposal_projection": _gradient_record(
            model,
            prefixes=("policy_head.proposal_query_projection.",),
        ),
        "schema9_planner_reranker_output": _gradient_record(
            model,
            prefixes=("planner_reranker.residual_head.2.",),
        ),
        "root_perspective_value_output": _gradient_record(
            model,
            prefixes=("root_perspective_value_adapter.residual.2.",),
        ),
    }
    gradients_valid = all(
        bool(record["finite"]) and bool(record["nonzero"]) for record in layers.values()
    )
    loss_finite = math.isfinite(float(total_loss.detach().item()))
    return {
        "q_planner_definition": ("softmax(stop_gradient(base_logp)+fixed_score_prior)"),
        "behavior_definition": (
            "softmax(current_base_logp+fixed_score_prior+planner_residual)"
        ),
        "behavior_equals_q_planner_bit_exact": zero_start_behavior_equals_q,
        "loss": float(total_loss.detach().item()),
        "loss_finite": loss_finite,
        "layers": layers,
        "valid": bool(zero_start_behavior_equals_q and loss_finite and gradients_valid),
    }


def root_relation_endpoint_rows() -> tuple[Tensor, Tensor]:
    """Return aligned same-seat and turn-handoff value-adapter rows."""
    relations = torch.tensor(
        [
            int(RootActorRelation.SAME_SEAT),
            int(RootActorRelation.SAME_SEAT),
            int(RootActorRelation.OTHER_SEAT),
            int(RootActorRelation.OTHER_SEAT),
        ],
        dtype=torch.long,
    )
    endpoints = torch.tensor(
        [
            int(SemanticEndpoint.SAME_SEAT_MAIN),
            int(SemanticEndpoint.SAME_SEAT_MAIN),
            int(SemanticEndpoint.TURN_HANDOFF),
            int(SemanticEndpoint.TURN_HANDOFF),
        ],
        dtype=torch.long,
    )
    return relations, endpoints


def _planner_distributions(
    planner: PlannerCandidateEvaluation,
    *,
    score_prior: Tensor,
    candidate_counts: Sequence[int],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    q_groups: list[Tensor] = []
    behavior_groups: list[Tensor] = []
    q_logprob_groups: list[Tensor] = []
    behavior_logprob_groups: list[Tensor] = []
    proposal_logprob_groups: list[Tensor] = []
    offset = 0
    for count in candidate_counts:
        stop = offset + int(count)
        prior = score_prior[offset:stop]
        base = planner.base_action_logprobs[offset:stop]
        residual = planner.reranker_residuals[offset:stop]
        q_logprobs = torch.log_softmax(base.detach() + prior, dim=0)
        behavior_logprobs = torch.log_softmax(base + prior + residual, dim=0)
        proposal_logprobs = torch.log_softmax(
            planner.proposal_action_logprobs[offset:stop],
            dim=0,
        )
        q_logprob_groups.append(q_logprobs)
        behavior_logprob_groups.append(behavior_logprobs)
        proposal_logprob_groups.append(proposal_logprobs)
        q_groups.append(torch.exp(q_logprobs))
        behavior_groups.append(torch.exp(behavior_logprobs))
        offset = stop
    return (
        torch.cat(q_groups),
        torch.cat(behavior_groups),
        torch.cat(q_logprob_groups),
        torch.cat(behavior_logprob_groups),
        torch.cat(proposal_logprob_groups),
    )


def _gradient_record(
    model: AgentPolicyValueNet,
    *,
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    parameter_records: list[dict[str, Any]] = []
    squared_norm = 0.0
    all_finite = True
    any_nonzero = False
    for name, parameter in model.named_parameters():
        if not name.startswith(prefixes):
            continue
        gradient = parameter.grad
        finite = gradient is not None and bool(torch.isfinite(gradient).all().item())
        nonzero = gradient is not None and bool(torch.count_nonzero(gradient).item())
        if gradient is not None and finite:
            squared_norm += float(torch.sum(gradient.detach().double().square()).item())
        all_finite = all_finite and finite
        any_nonzero = any_nonzero or nonzero
        parameter_records.append(
            {
                "name": name,
                "finite": finite,
                "nonzero": nonzero,
            }
        )
    return {
        "parameters": parameter_records,
        "gradient_l2_norm": math.sqrt(squared_norm),
        "finite": bool(parameter_records) and all_finite,
        "nonzero": any_nonzero,
    }


__all__ = [
    "MigrationAuditBatch",
    "root_relation_endpoint_rows",
    "run_backward_evidence",
]
