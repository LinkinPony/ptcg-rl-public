"""Direct temporal model collation from compact sequence fragment columns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ptcg_rl.context.public_event_arrays import (
    PublicEventArrayBlock,
    PublicEventBatch,
    public_event_batch_from_array_block,
)
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.sequence.action import AcceptedActionBatch
from ptcg_rl.model.simple_stateless import (
    PublicBeliefSummaryBatch,
    SimpleTeacherForcedActionBatch,
    SparseBeliefTargets,
)
from ptcg_rl.model.state_encoder import StateBatch
from ptcg_rl.rl.sequence_array_replay import ArraySequenceMicrobatchPlan
from ptcg_rl.rl.stateless_array_collation import (
    SourceArrays,
    _action_batch,
    _belief_batch,
    _deck_batch,
    _gather,
    _gather_decision_values,
    _option_batch,
    _pad,
    _single_fragment_text,
    _source_ragged_plan,
    _source_selection,
    _SourceSelection,
    _sparse_belief_targets,
    _state_batch,
    _tensor,
    _token_targets,
    _validated_indices,
)
from ptcg_rl.rl.stateless_array_replay import StatelessArrayOptimizerWindow


@dataclass(frozen=True)
class StatelessArraySequenceContext:
    """Observation snapshots and accepted temporal payloads for replay."""

    states: StateBatch
    unique_deck_card_ids: Tensor
    deck_counts: Tensor
    deck_valid_mask: Tensor
    deck_signatures: tuple[str, ...]
    belief_summary: PublicBeliefSummaryBatch
    events: PublicEventBatch
    accepted_actions: AcceptedActionBatch
    input_contract_fingerprint: str
    public_deck_catalog_fingerprint: str

    def __post_init__(self) -> None:
        """Require every raw temporal field to share one batch dimension."""
        batch_size = int(self.states.card_ids.shape[0])
        if (
            batch_size <= 0
            or self.unique_deck_card_ids.shape[0] != batch_size
            or self.deck_counts.shape[0] != batch_size
            or self.deck_valid_mask.shape[0] != batch_size
            or len(self.deck_signatures) != batch_size
            or self.events.batch_size != batch_size
            or self.accepted_actions.batch_size != batch_size
            or not self.input_contract_fingerprint
            or not self.public_deck_catalog_fingerprint
            or (
                self.belief_summary.catalog_fingerprint
                != self.public_deck_catalog_fingerprint
            )
        ):
            raise ValueError("array sequence context rows are misaligned")


@dataclass(frozen=True)
class StatelessArraySequenceTargets:
    """Target-only tensors that are absent from the packed context closure."""

    options: OptionBatch
    actions: SimpleTeacherForcedActionBatch
    sparse_belief_targets: SparseBeliefTargets
    old_token_logprobs: Tensor
    old_prefix_values: Tensor
    token_advantages: Tensor
    token_returns: Tensor
    token_mask: Tensor
    old_root_values: Tensor
    root_returns: Tensor
    decision_macro_weights: Tensor
    belief_macro_weights: Tensor
    expected_belief_valid: Tensor
    input_contract_fingerprint: str
    public_deck_catalog_fingerprint: str

    def __post_init__(self) -> None:
        """Require every learner target to share one decision dimension."""
        batch_size = int(self.options.option_types.shape[0])
        token_shape = self.old_token_logprobs.shape
        if (
            batch_size <= 0
            or self.actions.choice_indices.shape[0] != batch_size
            or self.sparse_belief_targets.batch_size != batch_size
            or len(token_shape) != 2
            or token_shape[0] != batch_size
            or self.old_prefix_values.shape != token_shape
            or self.token_advantages.shape != token_shape
            or self.token_returns.shape != token_shape
            or self.token_mask.shape != token_shape
            or self.token_mask.dtype != torch.bool
        ):
            raise ValueError("array sequence target tensors are misaligned")
        for values in (
            self.old_root_values,
            self.root_returns,
            self.decision_macro_weights,
            self.belief_macro_weights,
            self.expected_belief_valid,
        ):
            if values.shape != (batch_size,):
                raise ValueError("array sequence row targets are misaligned")
        if (
            self.expected_belief_valid.dtype != torch.bool
            or not self.input_contract_fingerprint
            or not self.public_deck_catalog_fingerprint
        ):
            raise ValueError("array sequence target contract is incomplete")


def collate_stateless_array_sequence_context(
    window: StatelessArrayOptimizerWindow,
    plan: ArraySequenceMicrobatchPlan,
    *,
    card_vocab_size: int,
    device: torch.device | str | None = None,
) -> StatelessArraySequenceContext:
    """Gather a packed context closure without materializing event objects."""
    if card_vocab_size <= 0:
        raise ValueError("card vocabulary size must be positive")
    sources = window.source_arrays
    decision_selection = _source_selection(
        sources,
        plan.decision_part_indices,
        plan.decision_rows,
        row_field="decision_indices",
    )
    fragment_selection = _source_selection(
        sources,
        plan.fragment_part_indices,
        plan.fragment_rows,
        row_field="fragment_ids",
    )
    input_contract = _single_fragment_text(
        sources,
        fragment_selection,
        "input_contract_fingerprints",
        label="input contract",
    )
    catalog_fingerprint = _single_fragment_text(
        sources,
        fragment_selection,
        "public_deck_catalog_fingerprints",
        label="public deck catalog",
    )
    state_plan = _source_ragged_plan(
        sources,
        decision_selection,
        "state_offsets",
    )
    attachment_plan = _source_ragged_plan(
        sources,
        decision_selection,
        "attachment_offsets",
        minimum_width=1,
    )
    own_decks = np.asarray(
        _gather_decision_values(
            sources,
            fragment_selection,
            "own_decks",
            dtype=np.int64,
        ),
        dtype=np.int64,
    )
    expected_digests = np.asarray(
        _gather_decision_values(
            sources,
            fragment_selection,
            "own_deck_digests",
            dtype=sources[0]["own_deck_digests"].dtype,
        ),
        dtype=np.str_,
    )
    (
        unique_deck_card_ids,
        deck_counts,
        deck_valid_mask,
        deck_signatures,
    ) = _deck_batch(
        own_decks,
        expected_digests=expected_digests,
        card_vocab_size=card_vocab_size,
        device=device,
    )
    return StatelessArraySequenceContext(
        states=_state_batch(
            sources,
            state_plan,
            attachment_plan,
            device=device,
        ),
        unique_deck_card_ids=unique_deck_card_ids,
        deck_counts=deck_counts,
        deck_valid_mask=deck_valid_mask,
        deck_signatures=deck_signatures,
        belief_summary=_belief_batch(
            sources,
            decision_selection,
            catalog_fingerprint=catalog_fingerprint,
            device=device,
        ),
        events=_public_event_batch(
            sources,
            decision_selection,
            device=device,
        ),
        accepted_actions=_accepted_action_batch(
            sources,
            decision_selection,
            device=device,
        ),
        input_contract_fingerprint=input_contract,
        public_deck_catalog_fingerprint=catalog_fingerprint,
    )


def collate_stateless_array_sequence_targets(
    window: StatelessArrayOptimizerWindow,
    decision_indices: tuple[int, ...],
    *,
    card_vocab_size: int,
    device: torch.device | str | None = None,
) -> StatelessArraySequenceTargets:
    """Gather learner-only targets without recollating snapshot inputs."""
    indices = _validated_indices(decision_indices, size=window.decision_count)
    if card_vocab_size <= 0:
        raise ValueError("card vocabulary size must be positive")
    sources = window.source_arrays
    decision_parts = window.decision_part_indices[indices]
    decision_rows = window.decision_rows[indices]
    retained_fragments = window.decision_fragment_indices[indices]
    fragment_parts = window.retained_fragment_part_indices[retained_fragments]
    fragment_rows = window.retained_fragment_rows[retained_fragments]
    decision_selection = _source_selection(
        sources,
        decision_parts,
        decision_rows,
        row_field="decision_indices",
    )
    fragment_selection = _source_selection(
        sources,
        fragment_parts,
        fragment_rows,
        row_field="fragment_ids",
    )
    input_contract = _single_fragment_text(
        sources,
        fragment_selection,
        "input_contract_fingerprints",
        label="input contract",
    )
    catalog_fingerprint = _single_fragment_text(
        sources,
        fragment_selection,
        "public_deck_catalog_fingerprints",
        label="public deck catalog",
    )
    option_plan = _source_ragged_plan(
        sources,
        decision_selection,
        "option_offsets",
    )
    action_plan = _source_ragged_plan(
        sources,
        decision_selection,
        "action_offsets",
    )
    sparse_belief, sparse_belief_valid = _sparse_belief_targets(
        sources,
        decision_selection,
        fragment_selection,
        card_vocab_size=card_vocab_size,
        device=device,
    )
    expected_belief = np.asarray(
        window.belief_target_valid[indices],
        dtype=np.bool_,
    )
    if not np.array_equal(sparse_belief_valid, expected_belief):
        raise RuntimeError("array belief target validity changed during collation")
    (
        old_token_logprobs,
        old_prefix_values,
        token_advantages,
        token_returns,
        token_mask,
    ) = _token_targets(
        window,
        indices,
        action_width=action_plan.width,
        device=device,
    )
    scalar_values = np.stack(
        (
            window.root_values[indices],
            window.return_values[indices],
            window.decision_macro_weights[indices],
            window.belief_macro_weights[indices],
        )
    ).astype(np.float32, copy=False)
    scalar_tensors = _tensor(scalar_values, device=device)
    return StatelessArraySequenceTargets(
        options=_option_batch(
            sources,
            option_plan,
            decision_selection,
            device=device,
        ),
        actions=_action_batch(sources, action_plan, device=device),
        sparse_belief_targets=sparse_belief,
        old_token_logprobs=old_token_logprobs,
        old_prefix_values=old_prefix_values,
        token_advantages=token_advantages,
        token_returns=token_returns,
        token_mask=token_mask,
        old_root_values=scalar_tensors[0],
        root_returns=scalar_tensors[1],
        decision_macro_weights=scalar_tensors[2],
        belief_macro_weights=scalar_tensors[3],
        expected_belief_valid=_tensor(expected_belief, device),
        input_contract_fingerprint=input_contract,
        public_deck_catalog_fingerprint=catalog_fingerprint,
    )


def _public_event_batch(
    sources: tuple[SourceArrays, ...],
    decision_selection: _SourceSelection,
    *,
    device: torch.device | str | None,
) -> PublicEventBatch:
    """Gather event and overflow CSR rows into the existing tensor contract."""
    events = _source_ragged_plan(
        sources,
        decision_selection,
        "event_offsets",
    )
    overflow = _source_ragged_plan(
        sources,
        decision_selection,
        "event_overflow_offsets",
    )
    block = PublicEventArrayBlock(
        event_offsets=np.asarray(events.offsets, dtype=np.int32),
        event_types=_gather(sources, events, "event_types"),
        actor_roles=_gather(sources, events, "event_actor_roles"),
        from_areas=_gather(sources, events, "event_from_areas"),
        to_areas=_gather(sources, events, "event_to_areas"),
        card_ids=_gather(sources, events, "event_card_ids"),
        serials=_gather(sources, events, "event_serials"),
        entity_mask=_gather(sources, events, "event_entity_mask"),
        attack_ids=_gather(sources, events, "event_attack_ids"),
        attack_id_mask=_gather(
            sources,
            events,
            "event_attack_id_mask",
        ),
        values=_gather(sources, events, "event_values"),
        value_mask=_gather(sources, events, "event_value_mask"),
        categorical_values=_gather(
            sources,
            events,
            "event_categorical_values",
        ),
        overflow_offsets=np.asarray(overflow.offsets, dtype=np.int32),
        overflow_event_types=_gather(
            sources,
            overflow,
            "event_overflow_types",
        ),
        overflow_actor_roles=_gather(
            sources,
            overflow,
            "event_overflow_actor_roles",
        ),
        overflow_counts=_gather(
            sources,
            overflow,
            "event_overflow_counts",
        ),
    )
    return public_event_batch_from_array_block(
        block,
        tuple(range(int(events.lengths.shape[0]))),
        device=device,
    )


def _accepted_action_batch(
    sources: tuple[SourceArrays, ...],
    decision_selection: _SourceSelection,
    *,
    device: torch.device | str | None,
) -> AcceptedActionBatch:
    """Gather stable accepted-action semantics directly into padded tensors."""
    actions = _source_ragged_plan(
        sources,
        decision_selection,
        "action_offsets",
        minimum_width=1,
    )
    valid = np.zeros((actions.lengths.shape[0], actions.width), dtype=np.bool_)
    valid[actions.batch_rows, actions.columns] = True
    stable_identities = tuple(
        np.asarray(
            _gather_decision_values(
                sources,
                decision_selection,
                "accepted_action_stable_ids",
                dtype=sources[0]["accepted_action_stable_ids"].dtype,
            ),
            dtype=np.str_,
        ).tolist()
    )
    return AcceptedActionBatch(
        option_types=_tensor(
            _pad(
                sources,
                actions,
                "accepted_action_option_types",
                np.int64,
            ),
            device,
        ),
        option_contexts=_tensor(
            _pad(
                sources,
                actions,
                "accepted_action_option_contexts",
                np.int64,
            ),
            device,
        ),
        card_ids=_tensor(
            _pad(
                sources,
                actions,
                "accepted_action_card_ids",
                np.int64,
            ),
            device,
        ),
        attack_ids=_tensor(
            _pad(
                sources,
                actions,
                "accepted_action_attack_ids",
                np.int64,
            ),
            device,
        ),
        option_scalars=_tensor(
            _pad(
                sources,
                actions,
                "accepted_action_option_scalars",
                np.float32,
            ),
            device,
        ),
        entity_card_ids=_tensor(
            _pad(
                sources,
                actions,
                "accepted_action_entity_card_ids",
                np.int64,
            ),
            device,
        ),
        entity_areas=_tensor(
            _pad(
                sources,
                actions,
                "accepted_action_entity_areas",
                np.int64,
            ),
            device,
        ),
        entity_owner_roles=_tensor(
            _pad(
                sources,
                actions,
                "accepted_action_entity_owner_roles",
                np.int64,
            ),
            device,
        ),
        entity_token_kinds=_tensor(
            _pad(
                sources,
                actions,
                "accepted_action_entity_token_kinds",
                np.int64,
            ),
            device,
        ),
        entity_scalars=_tensor(
            _pad(
                sources,
                actions,
                "accepted_action_entity_scalars",
                np.float32,
            ),
            device,
        ),
        valid_mask=_tensor(valid, device),
        prompt_contexts=_tensor(
            _gather_decision_values(
                sources,
                decision_selection,
                "accepted_action_prompt_contexts",
                dtype=np.int64,
            ),
            device,
        ),
        lengths=_tensor(actions.lengths, device),
        ordered=_tensor(
            _gather_decision_values(
                sources,
                decision_selection,
                "accepted_action_ordered",
                dtype=np.bool_,
            ),
            device,
        ),
        stop_sampled=_tensor(
            _gather_decision_values(
                sources,
                decision_selection,
                "stop_sampled",
                dtype=np.bool_,
            ),
            device,
        ),
        fallback=_tensor(
            _gather_decision_values(
                sources,
                decision_selection,
                "accepted_action_fallback",
                dtype=np.bool_,
            ),
            device,
        ),
        stable_identities=stable_identities,
    )


__all__ = [
    "StatelessArraySequenceContext",
    "StatelessArraySequenceTargets",
    "collate_stateless_array_sequence_context",
    "collate_stateless_array_sequence_targets",
]
