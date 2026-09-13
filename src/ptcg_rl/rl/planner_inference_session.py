"""One immutable model-lease adapter for all planner inference stages."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from ptcg_rl.agent.search.planner_scoring import RootInformationValueProvider
from ptcg_rl.agent.search.planning_session_contract import (
    ContinuationPrompt,
)
from ptcg_rl.agent.search.policy_inputs import build_canonical_policy_input
from ptcg_rl.agent.search.proposal_generation import (
    PlannerProposalBatchResult,
    PlannerProposalSearchLimits,
)
from ptcg_rl.agent.search.root_information import (
    RootInformationStateTensorBatch,
)
from ptcg_rl.agent.search.root_information_tensorizer import (
    ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT,
    RootInformationModelInputBatch,
)
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.decks.identity import CanonicalDeck, canonicalize_deck
from ptcg_rl.model.network import PlannerCandidateEvaluation
from ptcg_rl.model.policy import OptionBatch, collate_encoded_options
from ptcg_rl.model.state_encoder import StateBatch, collate_state_tokens
from ptcg_rl.runtime.planner_telemetry import (
    PlannerRequestTelemetry,
    PlannerStage,
    PlannerStageEvent,
)


@dataclass(frozen=True, slots=True)
class PlannerInferenceLease:
    """Per-request model identity and absolute action-critical deadline."""

    model_fingerprint: str
    policy_version: int
    tensor_schema_fingerprint: str
    deadline_monotonic: float
    inference_device_type: Literal["cpu", "cuda"]
    inference_timeout_seconds: float

    def __post_init__(self) -> None:
        if len(self.model_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.model_fingerprint
        ):
            raise ValueError("planner lease model fingerprint must be SHA-256")
        if self.policy_version < 0:
            raise ValueError("planner lease policy version must be non-negative")
        if self.tensor_schema_fingerprint != (
            ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT
        ):
            raise ValueError("planner lease tensor schema is incompatible")
        if not math.isfinite(self.deadline_monotonic) or self.deadline_monotonic <= 0:
            raise ValueError("planner lease deadline must be finite and positive")
        if self.inference_device_type not in ("cpu", "cuda"):
            raise ValueError("planner inference device type must be cpu or cuda")
        if (
            not math.isfinite(self.inference_timeout_seconds)
            or self.inference_timeout_seconds <= 0.0
        ):
            raise ValueError("planner inference timeout must be finite and positive")

    def rpc_deadline(self) -> float:
        """Return a per-neural-call deadline inside the global planner lease."""
        return min(
            self.deadline_monotonic,
            time.monotonic() + self.inference_timeout_seconds,
        )


class PlannerInferenceSession:
    """Route proposal, leaf, continuation, and reranker work under one lease."""

    def __init__(
        self,
        *,
        policy: object,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        context_handles: Sequence[str],
        lease: PlannerInferenceLease,
        _released_state: set[int] | None = None,
    ) -> None:
        batch_size = int(states.card_ids.shape[0])
        if (
            int(options.valid_options.shape[0]) != batch_size
            or len(decks) != batch_size
            or len(context_handles) != batch_size
        ):
            raise ValueError("planner inference session rows must align")
        if len(set(context_handles)) != batch_size or any(
            not handle for handle in context_handles
        ):
            raise ValueError("planner inference context handles must be unique")
        served_version = int(getattr(policy, "policy_version", -1))
        if served_version != lease.policy_version:
            raise RuntimeError("planner policy version differs from acquired lease")
        served_fingerprint = getattr(policy, "model_fingerprint", None)
        if served_fingerprint is not None and served_fingerprint != (
            lease.model_fingerprint
        ):
            raise RuntimeError("planner model payload differs from acquired lease")
        self.policy = policy
        self.states = states
        self.options = options
        self.decks = decks
        self.context_handles = tuple(context_handles)
        self.lease = lease
        self._released = set() if _released_state is None else _released_state

    def narrow_deadline(self, deadline_monotonic: float) -> PlannerInferenceSession:
        """Return the same immutable lease with an earlier request deadline."""
        deadline = float(deadline_monotonic)
        if not math.isfinite(deadline) or deadline <= 0.0:
            raise ValueError("planner narrowed deadline must be finite and positive")
        if deadline > self.lease.deadline_monotonic:
            raise ValueError("planner deadline narrowing cannot extend a lease")
        return PlannerInferenceSession(
            policy=self.policy,
            states=self.states,
            options=self.options,
            decks=self.decks,
            context_handles=self.context_handles,
            lease=replace(self.lease, deadline_monotonic=deadline),
            _released_state=self._released,
        )

    def generate_proposals(
        self,
        row_indices: Sequence[int],
        *,
        ordered_rows: Tensor,
        limits: PlannerProposalSearchLimits,
    ) -> PlannerProposalBatchResult:
        """Run one learned proposal campaign for the requested root rows."""
        indices = _index_tensor(row_indices, size=self.batch_size)
        if (
            ordered_rows.shape != (len(row_indices),)
            or ordered_rows.dtype != torch.bool
        ):
            raise ValueError("proposal ordering rows are misaligned")
        states = select_state_rows(self.states, indices)
        options = select_option_rows(self.options, indices)
        decks = self.decks.select(indices.to(device=self.decks.card_ids.device))
        handles = tuple(self.context_handles[index] for index in row_indices)
        scheduled = getattr(self.policy, "generate_planner_proposals_until", None)
        if callable(scheduled):
            result = scheduled(
                states,
                options,
                decks,
                ordered_rows=ordered_rows,
                limits=limits,
                planner_context_handles=handles,
                deadline_monotonic=self.lease.rpc_deadline(),
                model_version_lease=self.lease.policy_version,
                tensor_schema_fingerprint=self.lease.tensor_schema_fingerprint,
            )
        else:
            direct = getattr(self.policy, "generate_planner_proposals", None)
            if not callable(direct):
                raise RuntimeError("planner policy has no proposal inference surface")
            result = direct(
                states,
                options,
                ordered_rows=ordered_rows.to(device=options.valid_options.device),
                limits=limits,
                decks=decks,
                planner_context_handles=handles,
                model_version_lease=self.lease.policy_version,
                tensor_schema_fingerprint=self.lease.tensor_schema_fingerprint,
                deadline_monotonic=self.lease.rpc_deadline(),
            )
        if not isinstance(result, PlannerProposalBatchResult):
            raise TypeError("planner proposal surface returned another result type")
        if len(result.decisions) != len(row_indices):
            raise RuntimeError("planner proposal result rows are misaligned")
        if len(result.base_greedy_actions) != len(row_indices):
            raise RuntimeError("planner proposal result omitted base greedy anchors")
        return result

    def evaluate_candidates(
        self,
        row_index: int,
        *,
        actions: tuple[tuple[int, ...], ...],
        aggregate_features: Tensor,
        ordered: bool,
    ) -> PlannerCandidateEvaluation:
        """Evaluate one categorical candidate support from its retained context."""
        indices = _index_tensor((row_index,), size=self.batch_size)
        states = select_state_rows(self.states, indices)
        options = select_option_rows(self.options, indices)
        decks = self.decks.select(indices.to(device=self.decks.card_ids.device))
        ordered_rows = torch.tensor((ordered,), dtype=torch.bool)
        handles = (self.context_handles[row_index],)
        scheduled = getattr(self.policy, "evaluate_planner_candidates_until", None)
        if callable(scheduled):
            return cast(
                PlannerCandidateEvaluation,
                scheduled(
                    states,
                    options,
                    (actions,),
                    (aggregate_features,),
                    decks,
                    ordered_rows=ordered_rows,
                    planner_context_handles=handles,
                    deadline_monotonic=self.lease.rpc_deadline(),
                    model_version_lease=self.lease.policy_version,
                    tensor_schema_fingerprint=self.lease.tensor_schema_fingerprint,
                ),
            )
        direct = getattr(self.policy, "evaluate_planner_candidates", None)
        if not callable(direct):
            raise RuntimeError("planner policy has no candidate inference surface")
        return cast(
            PlannerCandidateEvaluation,
            direct(
                states,
                options,
                (actions,),
                (aggregate_features,),
                ordered_rows=ordered_rows.to(device=options.valid_options.device),
                decks=decks,
                planner_context_handles=handles,
                model_version_lease=self.lease.policy_version,
                tensor_schema_fingerprint=self.lease.tensor_schema_fingerprint,
            ),
        )

    def root_value_provider(
        self,
        *,
        root_deck: Sequence[int] | CanonicalDeck,
        max_rows: int,
        telemetry: PlannerRequestTelemetry | None = None,
    ) -> RootInformationValueProvider[RootInformationModelInputBatch]:
        """Return a leaf provider pinned to this exact model lease and deck."""
        if max_rows <= 0:
            raise ValueError("root-value microbatch rows must be positive")
        return _PolicyRootInformationValueProvider(
            session=self,
            root_deck=(
                root_deck
                if isinstance(root_deck, CanonicalDeck)
                else canonicalize_deck(root_deck)
            ),
            max_rows=int(max_rows),
            telemetry=telemetry,
        )

    def continuation_actions(
        self,
        prompts: tuple[ContinuationPrompt, ...],
        *,
        root_deck: Sequence[int] | CanonicalDeck,
    ) -> tuple[tuple[int, ...], ...]:
        """Greedily decode continuation prompts under the same model lease."""
        if not prompts:
            return ()
        prepared = tuple(
            build_canonical_policy_input(prompt.observation) for prompt in prompts
        )
        if any(item is None for item in prepared):
            raise ValueError("continuation prompt could not be tensorized")
        canonical = tuple(cast(Any, item) for item in prepared)
        device = self.states.card_ids.device
        states = collate_state_tokens(
            [item.state for item in canonical],
            device=device,
        )
        options = collate_encoded_options(
            [item.options for item in canonical],
            min_counts=[item.min_count for item in canonical],
            max_counts=[item.max_count for item in canonical],
            device=device,
        )
        deck = (
            root_deck
            if isinstance(root_deck, CanonicalDeck)
            else canonicalize_deck(root_deck)
        )
        decks = DeckBatch.from_decks((deck,) * len(prompts), device=device)
        scheduled = getattr(self.policy, "sample_decode_until", None)
        if callable(scheduled):
            actions, _logprobs, _values = scheduled(
                states,
                options,
                decks,
                temperature=0.0,
                deadline_monotonic=self.lease.rpc_deadline(),
                model_version_lease=self.lease.policy_version,
                tensor_schema_fingerprint=self.lease.tensor_schema_fingerprint,
            )
            return tuple(tuple(int(value) for value in action) for action in actions)
        if getattr(self.policy, "request_purpose", None) == "planner_behavior":
            raise RuntimeError(
                "remote planner policy has no lease-bound continuation decode"
            )
        direct = getattr(self.policy, "sample_decode", None)
        if not callable(direct):
            raise RuntimeError("planner policy has no continuation decode surface")
        actions, _logprobs, _values = direct(
            states,
            options,
            decks,
            temperature=0.0,
        )
        return tuple(tuple(int(value) for value in action) for action in actions)

    def release_rows(
        self,
        row_indices: Sequence[int],
        *,
        deadline_monotonic: float | None = None,
    ) -> int:
        """Release each root context at most once, including fallback paths."""
        pending = tuple(
            index
            for index in row_indices
            if 0 <= index < self.batch_size and index not in self._released
        )
        if len(pending) != len(tuple(dict.fromkeys(pending))):
            raise ValueError("planner release rows must be unique")
        if not pending:
            return 0
        handles = tuple(self.context_handles[index] for index in pending)
        scheduled = getattr(self.policy, "release_planner_contexts_until", None)
        if callable(scheduled):
            indices = _index_tensor(pending, size=self.batch_size)
            states = select_state_rows(self.states, indices)
            decks = self.decks.select(indices.to(device=self.decks.card_ids.device))
            released = int(
                scheduled(
                    states,
                    decks,
                    planner_context_handles=handles,
                    deadline_monotonic=(
                        self.lease.deadline_monotonic
                        if deadline_monotonic is None
                        else float(deadline_monotonic)
                    ),
                    model_version_lease=self.lease.policy_version,
                    tensor_schema_fingerprint=self.lease.tensor_schema_fingerprint,
                )
            )
        else:
            direct = getattr(self.policy, "release_planner_context_handles", None)
            if not callable(direct):
                raise RuntimeError("planner policy cannot release retained contexts")
            released = int(direct(handles))
        self._released.update(pending)
        if released != len(pending):
            raise RuntimeError("planner context release count is incomplete")
        return released

    @property
    def batch_size(self) -> int:
        """Return the immutable number of root rows in the session."""
        return int(self.states.card_ids.shape[0])


@dataclass(slots=True)
class _PolicyRootInformationValueProvider(
    RootInformationValueProvider[RootInformationModelInputBatch]
):
    session: PlannerInferenceSession
    root_deck: CanonicalDeck
    max_rows: int
    telemetry: PlannerRequestTelemetry | None = None

    def values(
        self,
        batch: RootInformationStateTensorBatch[RootInformationModelInputBatch],
    ) -> npt.NDArray[np.float32]:
        inputs = batch.model_inputs
        count = len(batch.leaves)
        chunks: list[npt.NDArray[np.float32]] = []
        for start in range(0, count, self.max_rows):
            stop = min(count, start + self.max_rows)
            indices = torch.arange(start, stop, dtype=torch.long)
            chunk = select_root_information_inputs(inputs, indices)
            decks = DeckBatch.from_decks(
                (self.root_deck,) * (stop - start),
                device="cpu",
            )
            started = time.perf_counter()
            values = self._predict_chunk(chunk, decks)
            result = values.detach().to(device="cpu", dtype=torch.float32).numpy()
            if result.shape != (stop - start,):
                raise RuntimeError(
                    "planner root-value response has the wrong shape"
                )
            chunks.append(cast(npt.NDArray[np.float32], result))
            if self.telemetry is not None:
                cuda = self.session.lease.inference_device_type == "cuda"
                self.telemetry.record(
                    PlannerStageEvent(
                        stage=PlannerStage.GPU_LEAF_VALUE,
                        seconds=time.perf_counter() - started,
                        rows=(stop - start) if cuda else 0,
                        batch_capacity=self.max_rows if cuda else 0,
                    )
                )
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    def _predict_chunk(
        self,
        inputs: RootInformationModelInputBatch,
        decks: DeckBatch,
    ) -> Tensor:
        scheduled = getattr(
            self.session.policy,
            "predict_root_information_values_until",
            None,
        )
        if callable(scheduled):
            values = scheduled(
                inputs,
                decks,
                deadline_monotonic=self.session.lease.rpc_deadline(),
                model_version_lease=self.session.lease.policy_version,
                tensor_schema_fingerprint=(inputs.tensor_schema_fingerprint),
            )
        else:
            direct = getattr(
                self.session.policy,
                "predict_root_information_values",
                None,
            )
            if not callable(direct):
                raise RuntimeError("planner policy has no root-value surface")
            device = self.session.states.card_ids.device
            moved = move_root_information_inputs(inputs, device=device)
            values = direct(
                moved.states,
                decks.to(device),
                actor_relations=moved.actor_relations,
                endpoints=moved.endpoints,
                belief_summaries=moved.belief_summaries,
            )
        return cast(Tensor, values)


def select_state_rows(states: StateBatch, indices: Tensor) -> StateBatch:
    """Select state rows while retaining every optional feature tensor."""
    device_indices = indices.to(device=states.card_ids.device)
    selected_fingerprints: tuple[str, ...] = ()
    if states.root_input_fingerprints:
        row_indices = tuple(int(value) for value in indices.detach().cpu().tolist())
        selected_fingerprints = tuple(
            states.root_input_fingerprints[index] for index in row_indices
        )

    def select(values: Tensor | None) -> Tensor | None:
        return None if values is None else values.index_select(0, device_indices)

    return StateBatch(
        card_ids=states.card_ids.index_select(0, device_indices),
        areas=states.areas.index_select(0, device_indices),
        owner_roles=states.owner_roles.index_select(0, device_indices),
        token_kinds=states.token_kinds.index_select(0, device_indices),
        scalars=states.scalars.index_select(0, device_indices),
        last_attack_ids=states.last_attack_ids.index_select(0, device_indices),
        padding_mask=states.padding_mask.index_select(0, device_indices),
        attachment_card_ids=select(states.attachment_card_ids),
        attachment_parent_indices=select(states.attachment_parent_indices),
        attachment_kinds=select(states.attachment_kinds),
        entity_slots=select(states.entity_slots),
        root_input_fingerprints=selected_fingerprints,
    )


def select_option_rows(options: OptionBatch, indices: Tensor) -> OptionBatch:
    """Select option rows without changing their request-local padded width."""
    device_indices = indices.to(device=options.valid_options.device)
    return OptionBatch(
        option_types=options.option_types.index_select(0, device_indices),
        contexts=options.contexts.index_select(0, device_indices),
        entity_slots=options.entity_slots.index_select(0, device_indices),
        entity_slot_mask=options.entity_slot_mask.index_select(0, device_indices),
        attack_ids=options.attack_ids.index_select(0, device_indices),
        card_ids=options.card_ids.index_select(0, device_indices),
        scalars=options.scalars.index_select(0, device_indices),
        dynamic_effect_features=options.dynamic_effect_features.index_select(
            0, device_indices
        ),
        dynamic_effect_masks=options.dynamic_effect_masks.index_select(
            0, device_indices
        ),
        valid_options=options.valid_options.index_select(0, device_indices),
        min_counts=options.min_counts.index_select(0, device_indices),
        max_counts=options.max_counts.index_select(0, device_indices),
    )


def move_root_information_inputs(
    inputs: RootInformationModelInputBatch,
    *,
    device: torch.device | str,
) -> RootInformationModelInputBatch:
    """Move one tensorized unique-leaf batch without changing its schema."""
    target = torch.device(device)

    def move(values: Tensor | None) -> Tensor | None:
        return None if values is None else values.to(device=target, non_blocking=True)

    states = inputs.states
    return RootInformationModelInputBatch(
        states=StateBatch(
            card_ids=states.card_ids.to(device=target, non_blocking=True),
            areas=states.areas.to(device=target, non_blocking=True),
            owner_roles=states.owner_roles.to(device=target, non_blocking=True),
            token_kinds=states.token_kinds.to(device=target, non_blocking=True),
            scalars=states.scalars.to(device=target, non_blocking=True),
            last_attack_ids=states.last_attack_ids.to(device=target, non_blocking=True),
            padding_mask=states.padding_mask.to(device=target, non_blocking=True),
            attachment_card_ids=move(states.attachment_card_ids),
            attachment_parent_indices=move(states.attachment_parent_indices),
            attachment_kinds=move(states.attachment_kinds),
            entity_slots=move(states.entity_slots),
            root_input_fingerprints=states.root_input_fingerprints,
        ),
        actor_relations=inputs.actor_relations.to(device=target, non_blocking=True),
        endpoints=inputs.endpoints.to(device=target, non_blocking=True),
        belief_summaries=inputs.belief_summaries.to(device=target, non_blocking=True),
    )


def select_root_information_inputs(
    inputs: RootInformationModelInputBatch,
    indices: Tensor,
) -> RootInformationModelInputBatch:
    """Select unique-leaf rows without changing their tensor schema."""
    device_indices = indices.to(device=inputs.actor_relations.device)
    return RootInformationModelInputBatch(
        states=select_state_rows(inputs.states, indices),
        actor_relations=inputs.actor_relations.index_select(0, device_indices),
        endpoints=inputs.endpoints.index_select(0, device_indices),
        belief_summaries=inputs.belief_summaries.index_select(0, device_indices),
        tensor_schema_fingerprint=inputs.tensor_schema_fingerprint,
    )


def _index_tensor(indices: Sequence[int], *, size: int) -> Tensor:
    frozen = tuple(int(index) for index in indices)
    if not frozen or len(set(frozen)) != len(frozen):
        raise ValueError("planner row indices must be nonempty and unique")
    if any(index < 0 or index >= size for index in frozen):
        raise IndexError("planner row index is outside the inference batch")
    return torch.tensor(frozen, dtype=torch.long)


__all__ = [
    "PlannerInferenceLease",
    "PlannerInferenceSession",
    "move_root_information_inputs",
    "select_option_rows",
    "select_state_rows",
]
