"""Functional cold-start and fixed-anchor parity for model migrations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ptcg_rl.actions.encoding import SCALAR_FEATURE_SIZE
from ptcg_rl.decks import DeckBatch, canonicalize_deck
from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.search_evidence import SEARCH_EVIDENCE_FEATURE_SIZE
from ptcg_rl.model.network import AgentNetworkConfig, AgentPolicyValueNet
from ptcg_rl.model.policy import MAX_ENTITY_SLOTS, OptionBatch
from ptcg_rl.model.state_encoder import TOKEN_SCALAR_SIZE, StateBatch
from ptcg_rl.model.weights_only_migration_gradient import (
    root_relation_endpoint_rows,
    run_backward_evidence,
)

_ANCHOR_SURFACES: Mapping[str, tuple[str, ...]] = {
    "proposal_projection": ("policy_head.proposal_query_projection.",),
    "schema8_search_reranker": ("search_reranker.",),
    "schema9_planner_reranker": ("planner_reranker.",),
    "root_perspective_value_adapter": ("root_perspective_value_adapter.",),
}


@dataclass(frozen=True)
class AuditBatch:
    """Minimal action grammars used for functional migration parity."""

    states: StateBatch
    options: OptionBatch
    decks: DeckBatch | None
    actions: tuple[tuple[int, ...], ...]
    candidate_actions: tuple[tuple[tuple[int, ...], ...], ...]
    candidate_features: tuple[Tensor, ...]
    ordered_rows: Tensor
    case_names: tuple[str, ...]


@dataclass(frozen=True)
class _ModelOutputs:
    """Base and schema-9 outputs needed by the audit."""

    action_logprobs: Tensor
    values: Tensor
    step_logits: tuple[Tensor, ...]
    prefix_values: Tensor
    planner_base_logprobs: Tensor
    proposal_logprobs: Tensor
    planner_residuals: Tensor
    root_information_values: Tensor


def run_functional_parity(
    *,
    migrated_model: AgentPolicyValueNet,
    anchor_model: AgentPolicyValueNet,
    config: AgentNetworkConfig,
) -> dict[str, Any]:
    """Prove zero-start behavior, base parity, gradients, and anchor isolation."""
    batch = build_audit_batch(config)
    with torch.inference_mode():
        migrated = _model_outputs(migrated_model, batch)
        anchor = _model_outputs(anchor_model, batch)

    case_results = _case_parity(batch, migrated=migrated, anchor=anchor)
    prefix_values_bit_exact = torch.equal(
        migrated.prefix_values,
        anchor.prefix_values,
    )
    backward = run_backward_evidence(migrated_model, batch)
    anchor_isolation = _anchor_isolation(anchor_model, batch, anchor.step_logits)
    case_valid = all(all(checks.values()) for checks in case_results.values())
    isolation_valid = all(
        bool(record["step_logits_bit_exact"]) and int(record["perturbed_tensors"]) > 0
        for record in anchor_isolation.values()
    )
    return {
        "status": "complete",
        "cases": case_results,
        "prefix_values_base_anchor_bit_exact": prefix_values_bit_exact,
        "zero_start_backward": backward,
        "anchor_isolation": anchor_isolation,
        "valid": bool(
            case_valid
            and prefix_values_bit_exact
            and backward["valid"]
            and isolation_valid
        ),
    }


def build_audit_batch(config: AgentNetworkConfig) -> AuditBatch:
    """Build direct, subset, ordered, and immediate-STOP synthetic rows."""
    batch_size = 4
    token_count = 2
    option_count = 3
    states = StateBatch(
        card_ids=torch.zeros((batch_size, token_count), dtype=torch.long),
        areas=torch.zeros((batch_size, token_count), dtype=torch.long),
        owner_roles=torch.ones((batch_size, token_count), dtype=torch.long),
        token_kinds=torch.zeros((batch_size, token_count), dtype=torch.long),
        scalars=torch.zeros(
            (batch_size, token_count, TOKEN_SCALAR_SIZE),
            dtype=torch.float32,
        ),
        last_attack_ids=torch.zeros((batch_size, token_count), dtype=torch.long),
        padding_mask=torch.zeros((batch_size, token_count), dtype=torch.bool),
    )
    contexts = torch.zeros((batch_size, option_count), dtype=torch.long)
    contexts[2, :] = int(SelectContext.SKILL_ORDER)
    options = OptionBatch(
        option_types=torch.arange(option_count, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, option_count),
        contexts=contexts,
        entity_slots=torch.zeros(
            (batch_size, option_count, MAX_ENTITY_SLOTS),
            dtype=torch.long,
        ),
        entity_slot_mask=torch.zeros(
            (batch_size, option_count, MAX_ENTITY_SLOTS),
            dtype=torch.bool,
        ),
        attack_ids=torch.zeros((batch_size, option_count), dtype=torch.long),
        card_ids=torch.zeros((batch_size, option_count), dtype=torch.long),
        scalars=torch.zeros(
            (batch_size, option_count, SCALAR_FEATURE_SIZE),
            dtype=torch.float32,
        ),
        dynamic_effect_features=torch.zeros(
            (batch_size, option_count, DYNAMIC_EFFECT_FEATURE_SIZE),
            dtype=torch.float32,
        ),
        dynamic_effect_masks=torch.zeros(
            (batch_size, option_count),
            dtype=torch.bool,
        ),
        valid_options=torch.ones(
            (batch_size, option_count),
            dtype=torch.bool,
        ),
        min_counts=torch.tensor([1, 1, 2, 0], dtype=torch.long),
        max_counts=torch.tensor([1, 2, 2, 2], dtype=torch.long),
    )
    candidate_actions = (
        ((1,), (0,)),
        ((0, 2), (1, 2)),
        ((2, 0), (0, 2)),
        ((), (0,)),
    )
    return AuditBatch(
        states=states,
        options=options,
        decks=_audit_decks(config, batch_size=batch_size),
        actions=((1,), (0, 2), (2, 0), ()),
        candidate_actions=candidate_actions,
        candidate_features=tuple(
            torch.zeros(
                (len(candidates), SEARCH_EVIDENCE_FEATURE_SIZE),
                dtype=torch.float32,
            )
            for candidates in candidate_actions
        ),
        ordered_rows=torch.tensor([False, False, True, False], dtype=torch.bool),
        case_names=("direct", "subset", "ordered", "stop"),
    )


def _case_parity(
    batch: AuditBatch,
    *,
    migrated: _ModelOutputs,
    anchor: _ModelOutputs,
) -> dict[str, dict[str, bool]]:
    case_results: dict[str, dict[str, bool]] = {}
    candidate_offset = 0
    for row_index, (case_name, candidates) in enumerate(
        zip(batch.case_names, batch.candidate_actions, strict=True)
    ):
        candidate_stop = candidate_offset + len(candidates)
        case_results[case_name] = {
            "action_logprob_base_anchor_bit_exact": torch.equal(
                migrated.action_logprobs[row_index],
                anchor.action_logprobs[row_index],
            ),
            "value_base_anchor_bit_exact": torch.equal(
                migrated.values[row_index],
                anchor.values[row_index],
            ),
            "step_logits_base_anchor_bit_exact": all(
                torch.equal(
                    migrated_step[row_index],
                    anchor_step[row_index],
                )
                for migrated_step, anchor_step in zip(
                    migrated.step_logits,
                    anchor.step_logits,
                    strict=True,
                )
            ),
            "proposal_equals_base_bit_exact": torch.equal(
                migrated.proposal_logprobs[candidate_offset:candidate_stop],
                migrated.planner_base_logprobs[candidate_offset:candidate_stop],
            ),
            "planner_residual_exact_zero": bool(
                torch.count_nonzero(
                    migrated.planner_residuals[candidate_offset:candidate_stop]
                ).item()
                == 0
            ),
            "root_adapter_equals_base_bit_exact": torch.equal(
                migrated.root_information_values[row_index],
                migrated.values[row_index],
            ),
        }
        candidate_offset = candidate_stop
    return case_results


def _model_outputs(model: AgentPolicyValueNet, batch: AuditBatch) -> _ModelOutputs:
    conditioned = model.encode_conditioned_state(batch.states, batch.decks)
    evaluation = model.evaluate_actions_from_conditioned(
        conditioned,
        batch.options,
        batch.actions,
    )
    if evaluation.policy_context is None or evaluation.prefix_values is None:
        raise RuntimeError("audit requires policy context and prefix values")
    planner = model.evaluate_planner_candidates_from_context(
        evaluation.policy_context,
        batch.options,
        batch.candidate_actions,
        batch.candidate_features,
        ordered_rows=batch.ordered_rows,
        decks=batch.decks,
    )
    relations, endpoints = root_relation_endpoint_rows()
    root_information_values = model.root_information_values_from_conditioned(
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
    return _ModelOutputs(
        action_logprobs=evaluation.action_logprobs.detach().clone(),
        values=evaluation.values.detach().clone(),
        step_logits=tuple(logits.detach().clone() for logits in evaluation.step_logits),
        prefix_values=evaluation.prefix_values.detach().clone(),
        planner_base_logprobs=planner.base_action_logprobs.detach().clone(),
        proposal_logprobs=planner.proposal_action_logprobs.detach().clone(),
        planner_residuals=planner.reranker_residuals.detach().clone(),
        root_information_values=root_information_values.detach().clone(),
    )


def _anchor_isolation(
    anchor_model: AgentPolicyValueNet,
    batch: AuditBatch,
    baseline_logits: tuple[Tensor, ...],
) -> dict[str, dict[str, Any]]:
    isolation: dict[str, dict[str, Any]] = {}
    for surface, prefixes in _ANCHOR_SURFACES.items():
        selected = [
            (name, parameter)
            for name, parameter in anchor_model.named_parameters()
            if name.startswith(prefixes)
        ]
        originals = [parameter.detach().clone() for _name, parameter in selected]
        with torch.no_grad():
            for index, (_name, parameter) in enumerate(selected):
                parameter.add_(float(index + 1))
            perturbed_logits = anchor_model.evaluate_action_step_logits(
                batch.states,
                batch.options,
                batch.actions,
                decks=batch.decks,
            )
            for (_name, parameter), original in zip(
                selected,
                originals,
                strict=True,
            ):
                parameter.copy_(original)
        isolation[surface] = {
            "perturbed_tensors": len(selected),
            "step_logits_bit_exact": all(
                torch.equal(before, after)
                for before, after in zip(
                    baseline_logits,
                    perturbed_logits,
                    strict=True,
                )
            ),
        }
    return isolation


def _audit_decks(
    config: AgentNetworkConfig,
    *,
    batch_size: int,
) -> DeckBatch | None:
    conditioning = config.deck_conditioning
    if conditioning is None or not conditioning.enabled:
        return None
    routes = conditioning.active_routes
    card_ids = routes[0].canonical_card_ids if routes else (1,) * 60
    deck = canonicalize_deck(card_ids)
    return DeckBatch.from_decks((deck,) * batch_size)


__all__ = ["build_audit_batch", "run_functional_parity"]
