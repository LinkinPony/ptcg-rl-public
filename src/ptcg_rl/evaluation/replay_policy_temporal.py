"""Causal sequence execution for matched-state replay policy audits."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeAlias, cast

import torch
from torch import Tensor

from ptcg_rl.context import collate_public_event_deltas
from ptcg_rl.model.sequence.action import (
    AcceptedActionRecord,
    collate_accepted_actions,
)
from ptcg_rl.model.sequence.core import TemporalKvCache
from ptcg_rl.model.simple_stateless import (
    SimpleStatelessPolicyValueNet,
    resolve_simple_exact_routes,
    simple_count_first_rows,
)
from ptcg_rl.rl.policy_inputs import (
    SimpleStatelessActorRow,
    collate_simple_stateless_actor_rows,
)


class TemporalReplayDecision(Protocol):
    """Decision fields required by causal replay execution."""

    episode_id: int
    player_index: int
    decision_index: int | None
    actor_row: SimpleStatelessActorRow
    teacher_action: tuple[int, ...]
    accepted_action: AcceptedActionRecord | None


DecisionRowsBuilder: TypeAlias = Callable[
    [
        tuple[TemporalReplayDecision, ...],
        tuple[tuple[int, ...], ...],
        Any,
        Tensor,
        Tensor,
    ],
    list[dict[str, Any]],
]


def audit_temporal_sequences(
    sequences: tuple[tuple[TemporalReplayDecision, ...], ...],
    *,
    model: SimpleStatelessPolicyValueNet,
    device: torch.device,
    batch_size: int,
    build_rows: DecisionRowsBuilder,
) -> list[dict[str, Any]]:
    """Evaluate causal replay states while committing teacher actions."""
    if model.sequence is None:
        raise ValueError("temporal audit requires a sequence checkpoint")
    if any(not sequence for sequence in sequences):
        raise ValueError("temporal replay source produced no decisions")
    rows: list[dict[str, Any]] = []
    caches: dict[tuple[int, int], TemporalKvCache] = {}
    maximum_decisions = max(len(sequence) for sequence in sequences)
    for decision_index in range(maximum_decisions):
        wave = tuple(
            sequence[decision_index]
            for sequence in sequences
            if decision_index < len(sequence)
        )
        for decision in wave:
            if decision.decision_index != decision_index:
                raise ValueError("temporal replay decision clock is discontinuous")
            if decision.accepted_action is None:
                raise ValueError("temporal replay decision has no accepted action")
        for start in range(0, len(wave), batch_size):
            chunk = wave[start : start + batch_size]
            chunk_rows, committed = _audit_temporal_batch(
                chunk,
                model=model,
                device=device,
                caches=caches,
                build_rows=build_rows,
            )
            rows.extend(chunk_rows)
            caches.update(committed)
    if len(rows) != sum(len(sequence) for sequence in sequences):
        raise RuntimeError("temporal replay audit lost decision rows")
    return rows


def _audit_temporal_batch(
    decisions: tuple[TemporalReplayDecision, ...],
    *,
    model: SimpleStatelessPolicyValueNet,
    device: torch.device,
    caches: Mapping[tuple[int, int], TemporalKvCache],
    build_rows: DecisionRowsBuilder,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], TemporalKvCache]]:
    """Evaluate and teacher-commit one wave of independent sequences."""
    actor_rows = tuple(decision.actor_row for decision in decisions)
    batch = collate_simple_stateless_actor_rows(
        actor_rows,
        device=device,
        deduplicate_belief=True,
    )
    route_plan = resolve_simple_exact_routes(
        batch.deck_signatures,
        model.config,
        device=device,
    )
    keys = tuple((decision.episode_id, decision.player_index) for decision in decisions)
    accepted_actions = tuple(
        cast(AcceptedActionRecord, decision.accepted_action) for decision in decisions
    )
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        snapshots = model.encode_observation_state(
            state=batch.states,
            unique_deck_card_ids=batch.unique_deck_card_ids,
            deck_counts=batch.deck_counts,
            deck_valid_mask=batch.deck_valid_mask,
            belief_summary=batch.belief_summary,
            route_plan=route_plan,
        )
        prepared = model.prepare_sequence_incremental_many(
            snapshots,
            collate_public_event_deltas(
                tuple(row.public_event_delta for row in actor_rows),
                device=device,
            ),
            block_indices=tuple(
                cast(int, decision.decision_index) for decision in decisions
            ),
            caches=tuple(caches.get(key) for key in keys),
        )
        state = model.condition_sequence(
            snapshots,
            torch.cat(tuple(item.context for item in prepared), dim=0),
        )
        option_embeddings = model.encode_legal_options(
            state,
            batch.options,
            route_plan=route_plan,
        )
        actions = tuple(decision.teacher_action for decision in decisions)
        evaluation = model.heads.teacher_forced(
            state.policy,
            state.opponent_belief,
            option_embeddings,
            batch.options,
            actions,
            route_plan=route_plan,
        )
        greedy = model.heads.greedy_decode(
            state.policy,
            state.opponent_belief,
            option_embeddings,
            batch.options,
            route_plan=route_plan,
        )
        root_values = model.heads.root_value(
            state.value,
            state.opponent_belief,
            route_plan=route_plan,
        )
        count_rows = simple_count_first_rows(batch.options)
        committed_caches = model.commit_sequence_incremental_many(
            prepared,
            collate_accepted_actions(accepted_actions, device=device),
        )
    rows = build_rows(
        decisions,
        greedy,
        evaluation,
        root_values,
        count_rows,
    )
    return rows, dict(zip(keys, committed_caches, strict=True))


__all__ = ["DecisionRowsBuilder", "TemporalReplayDecision", "audit_temporal_sequences"]
