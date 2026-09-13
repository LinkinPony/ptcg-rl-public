"""Direct model tensor collation from array-native stateless replay."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.simple_stateless import (
    PublicBeliefSummaryBatch,
    SimpleTeacherForcedActionBatch,
    SparseBeliefTargets,
)
from ptcg_rl.model.state_encoder import (
    ATTACHMENT_KIND_COUNT,
    OWNER_UNKNOWN,
    TOKEN_KIND_OOV_INDEX,
    StateBatch,
)
from ptcg_rl.rl.stateless_array_replay import StatelessArrayOptimizerWindow

Array = npt.NDArray[np.generic]
SourceArrays = Mapping[str, Array]


@dataclass(frozen=True)
class StatelessArrayMicrobatch:
    """Model inputs, behavior evidence, and learner targets for selected rows."""

    global_decision_indices: npt.NDArray[np.int64]
    states: StateBatch
    options: OptionBatch
    actions: SimpleTeacherForcedActionBatch
    unique_deck_card_ids: Tensor
    deck_counts: Tensor
    deck_valid_mask: Tensor
    deck_signatures: tuple[str, ...]
    belief_summary: PublicBeliefSummaryBatch
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
        """Require every model and target row to share one batch dimension."""
        batch_size = int(self.global_decision_indices.shape[0])
        if batch_size <= 0:
            raise ValueError("array microbatch must contain decisions")
        if (
            self.states.card_ids.shape[0] != batch_size
            or self.options.option_types.shape[0] != batch_size
            or self.actions.choice_indices.shape[0] != batch_size
            or self.unique_deck_card_ids.shape[0] != batch_size
            or len(self.deck_signatures) != batch_size
            or self.sparse_belief_targets.batch_size != batch_size
        ):
            raise ValueError("array microbatch model inputs are misaligned")
        token_shape = self.old_token_logprobs.shape
        if (
            len(token_shape) != 2
            or token_shape[0] != batch_size
            or self.old_prefix_values.shape != token_shape
            or self.token_advantages.shape != token_shape
            or self.token_returns.shape != token_shape
            or self.token_mask.shape != token_shape
            or self.token_mask.dtype != torch.bool
        ):
            raise ValueError("array microbatch token targets are misaligned")
        for values in (
            self.old_root_values,
            self.root_returns,
            self.decision_macro_weights,
            self.belief_macro_weights,
            self.expected_belief_valid,
        ):
            if values.shape != (batch_size,):
                raise ValueError("array microbatch row targets are misaligned")
        if self.expected_belief_valid.dtype != torch.bool:
            raise ValueError("belief validity target must be boolean")
        if (
            not self.input_contract_fingerprint
            or not self.public_deck_catalog_fingerprint
            or self.belief_summary.catalog_fingerprint
            != self.public_deck_catalog_fingerprint
        ):
            raise ValueError("array microbatch input contract is incomplete")

    @property
    def batch_size(self) -> int:
        """Return the number of selected replay decisions."""
        return int(self.global_decision_indices.shape[0])


@dataclass(frozen=True)
class _RaggedPlan:
    """Source and padded-destination coordinates for selected CSR rows."""

    lengths: npt.NDArray[np.int64]
    batch_rows: npt.NDArray[np.int64]
    columns: npt.NDArray[np.int64]
    source_parts: npt.NDArray[np.int32]
    source_rows: npt.NDArray[np.int64]
    source_groups: tuple[_SourceGroup, ...]
    width: int

    @property
    def offsets(self) -> npt.NDArray[np.int64]:
        """Return compact offsets in selected microbatch order."""
        result = np.zeros(self.lengths.shape[0] + 1, dtype=np.int64)
        result[1:] = np.cumsum(self.lengths)
        return result


@dataclass(frozen=True)
class _SourceGroup:
    """Coordinates from one selected compact source part."""

    part_index: int
    output_rows: npt.NDArray[np.int64]
    source_rows: npt.NDArray[np.int64]


@dataclass(frozen=True)
class _SourceSelection:
    """One microbatch coordinate selection grouped once by used source part."""

    part_indices: npt.NDArray[np.int32]
    rows: npt.NDArray[np.int64]
    groups: tuple[_SourceGroup, ...]


def collate_stateless_array_microbatch(
    window: StatelessArrayOptimizerWindow,
    decision_indices: Sequence[int],
    *,
    card_vocab_size: int,
    device: torch.device | str | None = None,
) -> StatelessArrayMicrobatch:
    """Gather arbitrary global replay rows directly into model tensor batches."""
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
    states = _state_batch(sources, state_plan, attachment_plan, device=device)
    options = _option_batch(
        sources,
        option_plan,
        decision_selection,
        device=device,
    )
    actions = _action_batch(sources, action_plan, device=device)

    own_decks = np.asarray(
        _gather_fragment_matrix(
            sources,
            fragment_selection,
            "own_decks",
            dtype=np.int64,
        ),
        dtype=np.int64,
    )
    expected_digests = np.asarray(window.deck_digests[indices], dtype=np.str_)
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
    belief_summary = _belief_batch(
        sources,
        decision_selection,
        catalog_fingerprint=catalog_fingerprint,
        device=device,
    )
    sparse_belief, sparse_belief_valid = _sparse_belief_targets(
        sources,
        decision_selection,
        fragment_selection,
        card_vocab_size=card_vocab_size,
        device=device,
    )
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
    expected_belief = np.asarray(
        window.belief_target_valid[indices],
        dtype=np.bool_,
    )
    if not np.array_equal(sparse_belief_valid, expected_belief):
        raise RuntimeError("array belief target validity changed during collation")
    scalar_values = np.stack(
        (
            window.root_values[indices],
            window.return_values[indices],
            window.decision_macro_weights[indices],
            window.belief_macro_weights[indices],
        )
    ).astype(np.float32, copy=False)
    scalar_tensors = _tensor(scalar_values, device=device)
    return StatelessArrayMicrobatch(
        global_decision_indices=indices,
        states=states,
        options=options,
        actions=actions,
        unique_deck_card_ids=unique_deck_card_ids,
        deck_counts=deck_counts,
        deck_valid_mask=deck_valid_mask,
        deck_signatures=deck_signatures,
        belief_summary=belief_summary,
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
        expected_belief_valid=_tensor(expected_belief, device=device),
        input_contract_fingerprint=input_contract,
        public_deck_catalog_fingerprint=catalog_fingerprint,
    )


def _state_batch(
    sources: tuple[SourceArrays, ...],
    state: _RaggedPlan,
    attachments: _RaggedPlan,
    *,
    device: torch.device | str | None,
) -> StateBatch:
    attachment_parents = np.asarray(
        _gather(sources, attachments, "attachment_parent_indices"),
        dtype=np.int64,
    )
    attachment_kinds = np.asarray(
        _gather(sources, attachments, "attachment_kinds"),
        dtype=np.int64,
    )
    if (
        np.any(attachment_parents < 0)
        or np.any(attachment_parents >= state.lengths[attachments.batch_rows])
        or np.any(attachment_kinds < 1)
        or np.any(attachment_kinds > ATTACHMENT_KIND_COUNT)
    ):
        raise ValueError("array state attachment identity is invalid")
    padding = np.ones((state.lengths.shape[0], state.width), dtype=np.bool_)
    padding[state.batch_rows, state.columns] = False
    return StateBatch(
        card_ids=_tensor(_pad(sources, state, "state_card_ids", np.int64), device),
        areas=_tensor(_pad(sources, state, "state_areas", np.int64), device),
        owner_roles=_tensor(
            _pad(
                sources,
                state,
                "state_owner_roles",
                np.int64,
                fill=OWNER_UNKNOWN,
            ),
            device,
        ),
        token_kinds=_tensor(
            _pad(
                sources,
                state,
                "state_token_kinds",
                np.int64,
                fill=TOKEN_KIND_OOV_INDEX,
            ),
            device,
        ),
        scalars=_tensor(_pad(sources, state, "state_scalars", np.float32), device),
        last_attack_ids=_tensor(
            _pad(sources, state, "state_last_attack_ids", np.int64),
            device,
        ),
        padding_mask=_tensor(padding, device),
        attachment_card_ids=_tensor(
            _pad(
                sources,
                attachments,
                "attachment_card_ids",
                np.uint16,
            ),
            device,
        ),
        attachment_parent_indices=_tensor(
            _pad(
                sources,
                attachments,
                "attachment_parent_indices",
                np.uint16,
            ),
            device,
        ),
        attachment_kinds=_tensor(
            _pad(
                sources,
                attachments,
                "attachment_kinds",
                np.uint8,
            ),
            device,
        ),
        entity_slots=_tensor(
            _pad(sources, state, "state_entity_slots", np.uint8),
            device,
        ),
        sequence_lengths=tuple(int(length) for length in state.lengths),
    )


def _option_batch(
    sources: tuple[SourceArrays, ...],
    options: _RaggedPlan,
    decision_selection: _SourceSelection,
    *,
    device: torch.device | str | None,
) -> OptionBatch:
    valid = np.zeros((options.lengths.shape[0], options.width), dtype=np.bool_)
    valid[options.batch_rows, options.columns] = True
    minimum = _gather_decision_values(
        sources,
        decision_selection,
        "min_counts",
        dtype=np.int64,
    )
    maximum = _gather_decision_values(
        sources,
        decision_selection,
        "max_counts",
        dtype=np.int64,
    )
    minimum = np.minimum(options.lengths, np.maximum(minimum, 0))
    maximum = np.minimum(
        options.lengths,
        np.maximum(minimum, maximum),
    )
    attack_ids = _pad(sources, options, "option_attack_ids", np.int64)
    card_ids = _pad(sources, options, "option_card_ids", np.int64)
    np.maximum(attack_ids, 0, out=attack_ids)
    np.maximum(card_ids, 0, out=card_ids)
    return OptionBatch(
        option_types=_tensor(
            _pad(sources, options, "option_types", np.int64),
            device,
        ),
        contexts=_tensor(
            _pad(sources, options, "option_contexts", np.int64),
            device,
        ),
        entity_slots=_tensor(
            _pad(sources, options, "option_entity_slots", np.int64),
            device,
        ),
        entity_slot_mask=_tensor(
            _pad(sources, options, "option_entity_slot_mask", np.bool_),
            device,
        ),
        attack_ids=_tensor(attack_ids, device),
        card_ids=_tensor(card_ids, device),
        scalars=_tensor(
            _pad(sources, options, "option_scalars", np.float32),
            device,
        ),
        dynamic_effect_features=_tensor(
            _pad(
                sources,
                options,
                "option_dynamic_effect_features",
                np.float32,
            ),
            device,
        ),
        dynamic_effect_masks=_tensor(
            _pad(
                sources,
                options,
                "option_dynamic_effect_masks",
                np.bool_,
            ),
            device,
        ),
        valid_options=_tensor(valid, device),
        min_counts=_tensor(minimum, device),
        max_counts=_tensor(maximum, device),
        option_lengths=tuple(int(length) for length in options.lengths),
        maximum_counts=tuple(int(value) for value in maximum),
    )


def _action_batch(
    sources: tuple[SourceArrays, ...],
    actions: _RaggedPlan,
    *,
    device: torch.device | str | None,
) -> SimpleTeacherForcedActionBatch:
    return SimpleTeacherForcedActionBatch(
        choice_indices=_tensor(
            _pad(sources, actions, "action_choices", np.int64),
            device,
        ),
        lengths=_tensor(actions.lengths, device),
        maximum_length=int(actions.width),
    )


def _token_targets(
    window: StatelessArrayOptimizerWindow,
    indices: npt.NDArray[np.int64],
    *,
    action_width: int,
    device: torch.device | str | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    starts = window.token_offsets[indices]
    lengths = window.token_offsets[indices + 1] - starts
    positions = _expand_ranges(starts, lengths)
    width = action_width + 1
    if np.any(lengths > width):
        raise RuntimeError("array collation omitted behavior decode tokens")
    plan = _coordinate_plan(
        lengths,
        window.token_part_indices[positions],
        window.token_rows[positions],
        minimum_width=width,
    )
    target_rows = plan.batch_rows
    target_columns = plan.columns
    shape = (indices.shape[0], width)
    values = np.zeros((4, *shape), dtype=np.float32)
    values[0, target_rows, target_columns] = _gather(
        window.source_arrays,
        plan,
        "token_logprobs",
    )
    values[1, target_rows, target_columns] = _gather(
        window.source_arrays,
        plan,
        "prefix_values",
    )
    values[2, target_rows, target_columns] = window.token_advantages[positions]
    values[3, target_rows, target_columns] = window.token_returns[positions]
    mask = np.zeros(shape, dtype=np.bool_)
    mask[target_rows, target_columns] = True
    tensors = _tensor(values, device)
    return (
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        _tensor(mask, device),
    )


def _deck_batch(
    decks: npt.NDArray[np.int64],
    *,
    expected_digests: npt.NDArray[np.str_],
    card_vocab_size: int,
    device: torch.device | str | None,
) -> tuple[Tensor, Tensor, Tensor, tuple[str, ...]]:
    if decks.ndim != 2 or decks.shape[1] != 60:
        raise ValueError("array own decks must have shape [batch, 60]")
    if expected_digests.shape != (decks.shape[0],):
        raise ValueError("array own deck digests are misaligned")
    unique_rows, inverse = _stable_unique_rows(decks)
    encoded_rows: list[
        tuple[
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            str,
            str,
        ]
    ] = []
    for deck in decks[unique_rows]:
        raw_card_ids, raw_counts = np.unique(deck, return_counts=True)
        card_ids = np.asarray(raw_card_ids, dtype=np.int64)
        counts = np.asarray(raw_counts, dtype=np.int64)
        signature = ";".join(
            f"{int(card_id)}:{int(count)}"
            for card_id, count in zip(card_ids, counts, strict=True)
        )
        encoded_rows.append(
            (
                card_ids,
                counts,
                signature,
                hashlib.sha256(signature.encode("utf-8")).hexdigest(),
            )
        )
    if any(
        np.any(row[0] <= 0) or np.any(row[0] > card_vocab_size)
        for row in encoded_rows
    ):
        raise ValueError("array own deck card exceeds vocabulary")
    unique_digests = np.asarray(
        [row[3] for row in encoded_rows],
        dtype=np.str_,
    )
    if not np.array_equal(unique_digests[inverse], expected_digests):
        raise ValueError("array own deck route fingerprint changed")
    width = max(int(row[0].shape[0]) for row in encoded_rows)
    unique_card_ids = np.zeros((len(encoded_rows), width), dtype=np.int64)
    unique_counts = np.zeros((len(encoded_rows), width), dtype=np.float32)
    unique_valid = np.zeros((len(encoded_rows), width), dtype=np.bool_)
    for index, (row_ids, row_counts, _signature, _digest) in enumerate(
        encoded_rows
    ):
        length = int(row_ids.shape[0])
        unique_card_ids[index, :length] = row_ids
        unique_counts[index, :length] = row_counts
        unique_valid[index, :length] = True
    unique_signatures = np.asarray(
        [row[2] for row in encoded_rows],
        dtype=np.str_,
    )
    return (
        _tensor(unique_card_ids[inverse], device),
        _tensor(unique_counts[inverse], device),
        _tensor(unique_valid[inverse], device),
        tuple(unique_signatures[inverse].tolist()),
    )


def _belief_batch(
    sources: tuple[SourceArrays, ...],
    decision_selection: _SourceSelection,
    *,
    catalog_fingerprint: str,
    device: torch.device | str | None,
) -> PublicBeliefSummaryBatch:
    plan = _source_ragged_plan(
        sources,
        decision_selection,
        "belief_offsets",
        minimum_width=1,
    )
    card_ids = np.asarray(
        _gather(sources, plan, "belief_card_ids"),
        dtype=np.int32,
    )
    expected = np.asarray(
        _gather(sources, plan, "belief_expected_counts"),
        dtype=np.float32,
    )
    scalars = _gather_decision_values(
        sources,
        decision_selection,
        "belief_scalars",
        dtype=np.float32,
    )
    row_count = int(decision_selection.rows.shape[0])
    padded_card_ids = np.zeros((row_count, plan.width), dtype=np.int32)
    padded_expected = np.zeros((row_count, plan.width), dtype=np.float32)
    padded_card_ids[plan.batch_rows, plan.columns] = card_ids
    padded_expected[plan.batch_rows, plan.columns] = expected

    # The old tuple-of-floats key treated positive and negative zero as equal.
    # Normalize only the scalar key copy so bytewise grouping keeps that behavior.
    scalar_keys = np.ascontiguousarray(scalars)
    if np.any(scalar_keys == 0.0):
        scalar_keys = scalar_keys.copy()
        scalar_keys[scalar_keys == 0.0] = 0.0
    key_dtype = np.dtype(
        [
            ("length", np.int64),
            ("card_ids", np.int32, (plan.width,)),
            ("expected", np.float32, (plan.width,)),
            ("scalars", np.float32, (scalar_keys.shape[1],)),
        ]
    )
    keys = np.empty(row_count, dtype=key_dtype)
    keys["length"] = plan.lengths
    keys["card_ids"] = padded_card_ids
    keys["expected"] = padded_expected
    keys["scalars"] = scalar_keys
    unique_rows, inverse = _stable_unique_rows(keys)
    unique_lengths = plan.lengths[unique_rows]
    unique_valid = (
        np.arange(plan.width, dtype=np.int64)[None, :]
        < unique_lengths[:, None]
    )
    return PublicBeliefSummaryBatch(
        card_ids=_tensor(
            padded_card_ids[unique_rows].astype(np.int64, copy=False),
            device,
        ),
        expected_counts=_tensor(padded_expected[unique_rows], device),
        valid_mask=_tensor(unique_valid, device),
        scalars=_tensor(scalars[unique_rows], device),
        catalog_fingerprint=catalog_fingerprint,
        row_indices=_tensor(inverse, device),
    )


def _sparse_belief_targets(
    sources: tuple[SourceArrays, ...],
    decision_selection: _SourceSelection,
    fragment_selection: _SourceSelection,
    *,
    card_vocab_size: int,
    device: torch.device | str | None,
) -> tuple[SparseBeliefTargets, npt.NDArray[np.bool_]]:
    decks = np.asarray(
        _gather_fragment_matrix(
            sources,
            fragment_selection,
            "opponent_decks",
            dtype=np.int64,
        ),
        dtype=np.int64,
    )
    if (
        decks.shape != (decision_selection.rows.shape[0], 60)
        or np.any(decks <= 0)
        or np.any(decks > card_vocab_size)
    ):
        raise ValueError("array opponent deck exceeds learner vocabulary")
    counts = np.zeros(
        (decision_selection.rows.shape[0], card_vocab_size + 1),
        dtype=np.int16,
    )
    deck_rows = np.repeat(
        np.arange(decision_selection.rows.shape[0], dtype=np.int64),
        60,
    )
    np.add.at(counts, (deck_rows, decks.reshape(-1)), 1)
    known = _source_ragged_plan(
        sources,
        decision_selection,
        "known_offsets",
    )
    known_ids = np.asarray(
        _gather(sources, known, "known_card_ids"),
        dtype=np.int64,
    )
    known_counts = np.asarray(
        _gather(sources, known, "known_counts"),
        dtype=np.int64,
    )
    if (
        np.any(known_ids <= 0)
        or np.any(known_ids > card_vocab_size)
        or np.any(known_counts <= 0)
        or np.any(counts[known.batch_rows, known_ids] < known_counts)
    ):
        raise ValueError(
            "public evidence cannot be subtracted from exact opponent deck"
        )
    counts[known.batch_rows, known_ids] -= known_counts.astype(np.int16)
    remaining = counts[:, 1:] > 0
    row_indices, zero_based_ids = np.nonzero(remaining)
    lengths = np.bincount(
        row_indices,
        minlength=decision_selection.rows.shape[0],
    ).astype(np.int64)
    offsets = np.zeros(decision_selection.rows.shape[0] + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    card_ids = zero_based_ids.astype(np.int64) + 1
    values = counts[row_indices, card_ids].astype(np.float32)
    return (
        SparseBeliefTargets(
            offsets=_tensor(offsets, device),
            card_ids=_tensor(card_ids, device),
            counts=_tensor(values, device),
        ),
        lengths > 0,
    )


def _source_ragged_plan(
    sources: tuple[SourceArrays, ...],
    selection: _SourceSelection,
    offsets_field: str,
    *,
    minimum_width: int = 0,
) -> _RaggedPlan:
    starts = np.empty(selection.rows.shape[0], dtype=np.int64)
    lengths = np.empty(selection.rows.shape[0], dtype=np.int64)
    for group in selection.groups:
        arrays = sources[group.part_index]
        offsets = np.asarray(arrays[offsets_field], dtype=np.int64)
        starts[group.output_rows] = offsets[group.source_rows]
        lengths[group.output_rows] = (
            offsets[group.source_rows + 1] - starts[group.output_rows]
        )
    source_rows = _expand_ranges(starts, lengths)
    return _coordinate_plan(
        lengths,
        np.repeat(selection.part_indices, lengths),
        source_rows,
        minimum_width=minimum_width,
    )


def _coordinate_plan(
    lengths: npt.NDArray[np.int64],
    source_parts: npt.NDArray[np.int32],
    source_rows: npt.NDArray[np.int64],
    *,
    minimum_width: int,
) -> _RaggedPlan:
    output_starts = np.zeros(lengths.shape[0], dtype=np.int64)
    if lengths.shape[0] > 1:
        output_starts[1:] = np.cumsum(lengths[:-1])
    batch_rows = np.repeat(
        np.arange(lengths.shape[0], dtype=np.int64),
        lengths,
    )
    columns = np.arange(source_rows.shape[0], dtype=np.int64) - np.repeat(
        output_starts,
        lengths,
    )
    return _RaggedPlan(
        lengths=lengths,
        batch_rows=batch_rows,
        columns=columns,
        source_parts=np.asarray(source_parts, dtype=np.int32),
        source_rows=np.asarray(source_rows, dtype=np.int64),
        source_groups=_group_source_rows(source_parts, source_rows),
        width=max(minimum_width, int(lengths.max(initial=0))),
    )


def _gather(
    sources: tuple[SourceArrays, ...],
    plan: _RaggedPlan,
    field: str,
) -> Array:
    tail_shape = tuple(sources[0][field].shape[1:])
    output = np.empty(
        (plan.source_rows.shape[0], *tail_shape),
        dtype=sources[0][field].dtype,
    )
    for group in plan.source_groups:
        arrays = sources[group.part_index]
        if tuple(arrays[field].shape[1:]) != tail_shape:
            raise ValueError(f"array source field {field} changed shape")
        output[group.output_rows] = arrays[field][group.source_rows]
    return output


def _pad(
    sources: tuple[SourceArrays, ...],
    plan: _RaggedPlan,
    field: str,
    dtype: npt.DTypeLike,
    *,
    fill: int | float | bool = 0,
) -> Array:
    tail_shape = tuple(sources[0][field].shape[1:])
    output = np.full(
        (plan.lengths.shape[0], plan.width, *tail_shape),
        fill,
        dtype=dtype,
    )
    output[plan.batch_rows, plan.columns] = _gather(sources, plan, field)
    return output


def _gather_decision_values(
    sources: tuple[SourceArrays, ...],
    selection: _SourceSelection,
    field: str,
    *,
    dtype: npt.DTypeLike,
) -> Array:
    tail_shape = tuple(sources[0][field].shape[1:])
    output = np.empty((selection.rows.shape[0], *tail_shape), dtype=dtype)
    for group in selection.groups:
        output[group.output_rows] = sources[group.part_index][field][group.source_rows]
    return output


def _gather_fragment_matrix(
    sources: tuple[SourceArrays, ...],
    selection: _SourceSelection,
    field: str,
    *,
    dtype: npt.DTypeLike,
) -> Array:
    return _gather_decision_values(
        sources,
        selection,
        field,
        dtype=dtype,
    )


def _single_fragment_text(
    sources: tuple[SourceArrays, ...],
    selection: _SourceSelection,
    field: str,
    *,
    label: str,
) -> str:
    values = _gather_decision_values(
        sources,
        selection,
        field,
        dtype=sources[0][field].dtype,
    )
    unique = np.unique(values)
    if unique.shape != (1,) or not str(unique[0]):
        raise ValueError(f"array microbatch mixes {label} identities")
    return str(unique[0])


def _stable_unique_rows(
    rows: npt.NDArray[np.generic],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return first rows and inverse indices in first-occurrence order."""
    if rows.ndim < 1 or rows.shape[0] == 0:
        raise ValueError("cannot group empty array rows")
    contiguous = np.ascontiguousarray(rows)
    byte_rows = contiguous.view(np.uint8).reshape(contiguous.shape[0], -1)
    opaque_rows = byte_rows.view(
        np.dtype((np.void, byte_rows.shape[1]))
    ).reshape(-1)
    _, first_rows, sorted_inverse = np.unique(
        opaque_rows,
        return_index=True,
        return_inverse=True,
    )
    stable_order = np.argsort(first_rows, kind="stable")
    stable_rows = np.asarray(first_rows[stable_order], dtype=np.int64)
    sorted_to_stable = np.empty(stable_order.shape[0], dtype=np.int64)
    sorted_to_stable[stable_order] = np.arange(
        stable_order.shape[0],
        dtype=np.int64,
    )
    inverse = np.asarray(sorted_to_stable[sorted_inverse], dtype=np.int64)
    return stable_rows, inverse


def _source_selection(
    sources: tuple[SourceArrays, ...],
    part_indices: npt.NDArray[np.int32],
    rows: npt.NDArray[np.int64],
    *,
    row_field: str,
) -> _SourceSelection:
    if (
        part_indices.shape != rows.shape
        or np.any(part_indices < 0)
        or np.any(part_indices >= len(sources))
    ):
        raise ValueError("array microbatch source part coordinate is invalid")
    groups = _group_source_rows(part_indices, rows)
    for group in groups:
        limit = int(sources[group.part_index][row_field].shape[0])
        if np.any((group.source_rows < 0) | (group.source_rows >= limit)):
            raise ValueError("array microbatch source row coordinate is invalid")
    return _SourceSelection(
        part_indices=part_indices,
        rows=rows,
        groups=groups,
    )


def _group_source_rows(
    part_indices: npt.NDArray[np.int32],
    rows: npt.NDArray[np.int64],
) -> tuple[_SourceGroup, ...]:
    """Group arbitrary coordinates once while preserving output row order."""
    if part_indices.shape != rows.shape or part_indices.ndim != 1:
        raise ValueError("array source coordinates are misaligned")
    if part_indices.size == 0:
        return ()
    order = np.argsort(part_indices, kind="stable").astype(np.int64, copy=False)
    ordered_parts = part_indices[order]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(ordered_parts[1:] != ordered_parts[:-1]).astype(
                np.int64,
                copy=False,
            )
            + 1,
        )
    )
    stops = np.concatenate(
        (starts[1:], np.asarray([part_indices.shape[0]], dtype=np.int64))
    )
    return tuple(
        _SourceGroup(
            part_index=int(ordered_parts[int(start)]),
            output_rows=order[int(start) : int(stop)],
            source_rows=rows[order[int(start) : int(stop)]],
        )
        for start, stop in zip(starts, stops, strict=True)
    )


def _validated_indices(
    values: Sequence[int],
    *,
    size: int,
) -> npt.NDArray[np.int64]:
    indices = np.asarray(tuple(int(value) for value in values), dtype=np.int64)
    if (
        indices.ndim != 1
        or indices.size == 0
        or np.unique(indices).shape[0] != indices.shape[0]
        or np.any(indices < 0)
        or np.any(indices >= size)
    ):
        raise ValueError("array microbatch indices are empty, duplicate, or invalid")
    return indices


def _expand_ranges(
    starts: npt.NDArray[np.int64],
    lengths: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    output_starts = np.zeros(lengths.shape[0], dtype=np.int64)
    if lengths.shape[0] > 1:
        output_starts[1:] = np.cumsum(lengths[:-1])
    return (
        np.repeat(starts, lengths)
        + np.arange(int(lengths.sum()), dtype=np.int64)
        - np.repeat(output_starts, lengths)
    )


def _tensor(
    values: npt.NDArray[np.generic],
    device: torch.device | str | None,
) -> Tensor:
    return torch.as_tensor(np.ascontiguousarray(values), device=device)


__all__ = [
    "StatelessArrayMicrobatch",
    "collate_stateless_array_microbatch",
]
