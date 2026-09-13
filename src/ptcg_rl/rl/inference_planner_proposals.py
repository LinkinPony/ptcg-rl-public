"""Focused request, response, and policy contracts for planner proposals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor

from ptcg_rl.agent.search.proposal_generation import (
    PlannerProposalBatchResult,
    PlannerProposalDecisionResult,
    PlannerProposalSearchLimits,
)
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.state_encoder import StateBatch


@dataclass(frozen=True)
class PlannerProposalRequestPayload:
    """One actor batch's ordering semantics and immutable search limits."""

    ordered_rows: Tensor
    limits: PlannerProposalSearchLimits
    context_handles: tuple[str, ...]

    def validate(self, *, batch_size: int) -> None:
        """Reject row-misaligned or device-ambiguous ordering semantics."""
        if (
            self.ordered_rows.shape != (batch_size,)
            or self.ordered_rows.dtype != torch.bool
        ):
            raise ValueError(
                "planner proposal ordered_rows must be bool with shape [batch]"
            )
        if (
            len(self.context_handles) != batch_size
            or len(set(self.context_handles)) != batch_size
            or any(not handle for handle in self.context_handles)
        ):
            raise ValueError("planner proposal context handles are misaligned")

    @property
    def batch_identity(self) -> str:
        """Return the limits identity required for cross-actor grouping."""
        return self.limits.fingerprint


@dataclass(frozen=True)
class PlannerProposalResponsePayload:
    """Request-local decisions without duplicated server batch telemetry."""

    decisions: tuple[PlannerProposalDecisionResult, ...]
    base_greedy_actions: tuple[tuple[int, ...], ...]
    search_fingerprint: str
    scoring_rows: int

    def __post_init__(self) -> None:
        if not self.decisions:
            raise ValueError("planner proposal response must contain decisions")
        if len(self.base_greedy_actions) != len(self.decisions):
            raise ValueError("base greedy actions must align with proposal decisions")
        if self.scoring_rows != sum(
            decision.nodes_expanded for decision in self.decisions
        ):
            raise ValueError("planner proposal scoring rows differ from decisions")
        if len(self.search_fingerprint) != 64:
            raise ValueError("planner proposal response fingerprint is invalid")

    def as_batch_result(self) -> PlannerProposalBatchResult:
        """Restore the public result while leaving group-only batches unclaimed."""
        return PlannerProposalBatchResult(
            decisions=self.decisions,
            scoring_batches=0,
            scoring_rows=self.scoring_rows,
            search_fingerprint=self.search_fingerprint,
            base_greedy_actions=self.base_greedy_actions,
        )


class PlannerProposalInferencePolicy(Protocol):
    """Leased policy surface for bounded learned prefix generation."""

    def generate_planner_proposals(
        self,
        states: StateBatch,
        options: OptionBatch,
        *,
        ordered_rows: Tensor,
        limits: PlannerProposalSearchLimits,
        decks: DeckBatch,
        planner_context_handles: Sequence[str],
        model_version_lease: int,
        tensor_schema_fingerprint: str,
        deadline_monotonic: float,
    ) -> PlannerProposalBatchResult:
        """Generate complete learned proposal supports for every root."""


def planner_proposal_predictor(
    policy: object,
) -> PlannerProposalInferencePolicy | None:
    """Return the optional proposal surface without widening base policy types."""
    predictor = getattr(policy, "generate_planner_proposals", None)
    if not callable(predictor):
        return None
    return cast(PlannerProposalInferencePolicy, policy)


def concatenate_planner_proposal_payloads(
    payloads: Sequence[PlannerProposalRequestPayload],
) -> tuple[Tensor, PlannerProposalSearchLimits]:
    """Concatenate ordering rows after proving one immutable limits identity."""
    frozen = tuple(payloads)
    if not frozen:
        raise ValueError("planner proposal payload group must not be empty")
    fingerprints = {payload.batch_identity for payload in frozen}
    if len(fingerprints) != 1:
        raise ValueError("planner proposal group mixed search limits")
    return (
        torch.cat(tuple(payload.ordered_rows for payload in frozen), dim=0),
        frozen[0].limits,
    )


def split_planner_proposal_result(
    result: PlannerProposalBatchResult,
    *,
    request_batch_sizes: Sequence[int],
) -> tuple[PlannerProposalResponsePayload, ...]:
    """Scatter decisions while assigning only additive row telemetry."""
    sizes = tuple(int(size) for size in request_batch_sizes)
    if any(size <= 0 for size in sizes):
        raise ValueError("planner proposal request batch sizes must be positive")
    if sum(sizes) != len(result.decisions):
        raise ValueError("planner proposal result differs from request rows")
    if len(result.base_greedy_actions) != len(result.decisions):
        raise ValueError("served proposal result is missing base greedy anchors")
    payloads: list[PlannerProposalResponsePayload] = []
    offset = 0
    for size in sizes:
        decisions = result.decisions[offset : offset + size]
        payloads.append(
            PlannerProposalResponsePayload(
                decisions=decisions,
                base_greedy_actions=result.base_greedy_actions[
                    offset : offset + size
                ],
                search_fingerprint=result.search_fingerprint,
                scoring_rows=sum(item.nodes_expanded for item in decisions),
            )
        )
        offset += size
    return tuple(payloads)


def evaluate_planner_proposal_group(
    predictor: PlannerProposalInferencePolicy,
    *,
    states: StateBatch,
    options: OptionBatch,
    decks: DeckBatch,
    payloads: Sequence[PlannerProposalRequestPayload],
    request_batch_sizes: Sequence[int],
    model_version_lease: int,
    tensor_schema_fingerprint: str,
    deadline_monotonic: float,
) -> tuple[PlannerProposalBatchResult, tuple[PlannerProposalResponsePayload, ...]]:
    """Execute one limits-homogeneous group and scatter additive provenance."""
    ordered_rows, limits = concatenate_planner_proposal_payloads(payloads)
    ordered_rows = ordered_rows.to(device=options.valid_options.device)
    result = predictor.generate_planner_proposals(
        states,
        options,
        ordered_rows=ordered_rows,
        limits=limits,
        decks=decks,
        planner_context_handles=tuple(
            handle for payload in payloads for handle in payload.context_handles
        ),
        model_version_lease=model_version_lease,
        tensor_schema_fingerprint=tensor_schema_fingerprint,
        deadline_monotonic=deadline_monotonic,
    )
    if result.search_fingerprint != limits.fingerprint:
        raise RuntimeError("policy returned another proposal search identity")
    if result.scoring_rows != sum(
        decision.nodes_expanded for decision in result.decisions
    ):
        raise RuntimeError("policy returned inconsistent proposal row telemetry")
    return (
        result,
        split_planner_proposal_result(
            result,
            request_batch_sizes=request_batch_sizes,
        ),
    )


__all__ = [
    "PlannerProposalInferencePolicy",
    "PlannerProposalRequestPayload",
    "PlannerProposalResponsePayload",
    "concatenate_planner_proposal_payloads",
    "evaluate_planner_proposal_group",
    "planner_proposal_predictor",
    "split_planner_proposal_result",
]
