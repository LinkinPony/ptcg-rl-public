"""Model-ready simple-stateless batches from native arena columns."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.simple_stateless.belief import PublicBeliefSummaryBatch
from ptcg_rl.model.state_encoder import StateBatch
from ptcg_rl.rl.native_policy_context import NativePublicContextBatch
from ptcg_rl.rl.native_policy_options import encode_native_option_batch
from ptcg_rl.rl.native_policy_state import encode_native_state_batch


@dataclass(frozen=True, slots=True)
class NativeSimpleStatelessPolicyBatch:
    """Object-light model inputs with no reconstructed decision rows."""

    states: StateBatch
    options: OptionBatch
    unique_deck_card_ids: Tensor
    deck_counts: Tensor
    deck_valid_mask: Tensor
    deck_signatures: tuple[str, ...]
    belief_summary: PublicBeliefSummaryBatch
    min_counts: tuple[int, ...]
    max_counts: tuple[int, ...]
    public_deck_catalog_fingerprint: str
    input_contract_fingerprint: str

    @property
    def batch_size(self) -> int:
        """Return the number of aligned native decisions."""
        return len(self.deck_signatures)


@dataclass(frozen=True, slots=True)
class _CachedDeckEncoding:
    canonical_cards: npt.NDArray[np.int64]
    card_ids: npt.NDArray[np.int64]
    counts: npt.NDArray[np.float32]


def _identity_tensor(value: Tensor) -> Tensor:
    """Return host-only sequence metadata without scheduling a device copy."""
    return value


class NativeDeckBatchCache:
    """Cache exact-deck unique-card rows across rollout decisions."""

    def __init__(self, *, maximum_entries: int = 256) -> None:
        if maximum_entries <= 0:
            raise ValueError("native deck cache capacity must be positive")
        self._maximum_entries = int(maximum_entries)
        self._entries: dict[str, _CachedDeckEncoding] = {}

    def collate(
        self,
        decks: npt.NDArray[np.integer],
        signatures: Sequence[str],
        *,
        device: torch.device | str | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Collate cached exact rows while validating signature identity."""
        if decks.ndim != 2 or decks.shape[0] != len(signatures):
            raise ValueError("native deck rows and signatures are misaligned")
        canonical_decks = np.sort(
            decks.astype(np.int64, copy=False),
            axis=1,
        )
        rows_by_signature: dict[str, list[int]] = {}
        for row, signature in enumerate(signatures):
            rows_by_signature.setdefault(signature, []).append(row)
        encoding_by_signature: dict[str, _CachedDeckEncoding] = {}
        for signature, raw_rows in rows_by_signature.items():
            rows = np.asarray(raw_rows, dtype=np.int64)
            canonical_cards = canonical_decks[int(rows[0])]
            existing = self._entries.get(signature)
            if existing is None:
                if len(self._entries) >= self._maximum_entries:
                    raise RuntimeError("native exact-deck cache capacity was exceeded")
                unique_card_ids, unique_counts = np.unique(
                    canonical_cards,
                    return_counts=True,
                )
                existing = _CachedDeckEncoding(
                    canonical_cards=canonical_cards.copy(),
                    card_ids=unique_card_ids.astype(np.int64, copy=False),
                    counts=unique_counts.astype(np.float32, copy=False),
                )
                self._entries[signature] = existing
            if not bool(
                np.all(canonical_decks[rows] == existing.canonical_cards[None, :])
            ):
                raise ValueError(
                    "one native deck signature resolved to different card IDs"
                )
            encoding_by_signature[signature] = existing
        width = max(
            len(encoding.card_ids) for encoding in encoding_by_signature.values()
        )
        card_ids: npt.NDArray[np.int64] = np.zeros(
            (len(signatures), width),
            dtype=np.int64,
        )
        counts: npt.NDArray[np.float32] = np.zeros(
            (len(signatures), width),
            dtype=np.float32,
        )
        valid: npt.NDArray[np.bool_] = np.zeros(
            (len(signatures), width),
            dtype=np.bool_,
        )
        for signature, raw_rows in rows_by_signature.items():
            rows = np.asarray(raw_rows, dtype=np.int64)
            encoding = encoding_by_signature[signature]
            size = len(encoding.card_ids)
            card_ids[rows, :size] = encoding.card_ids
            counts[rows, :size] = encoding.counts
            valid[rows, :size] = True
        return (
            torch.as_tensor(card_ids, device=device),
            torch.as_tensor(counts, device=device),
            torch.as_tensor(valid, device=device),
        )


def move_native_simple_stateless_batch(
    batch: NativeSimpleStatelessPolicyBatch,
    *,
    device: torch.device | str,
    non_blocking: bool = False,
    move_exact_deck_tensors: bool = True,
) -> NativeSimpleStatelessPolicyBatch:
    """Move model tensors while retaining CPU trajectory and identity metadata."""
    target = torch.device(device)

    def transform(value: Tensor) -> Tensor:
        return value.to(
            device=target,
            non_blocking=non_blocking,
        )

    return _map_native_simple_stateless_batch_tensors(
        batch,
        transform,
        exact_deck_transform=(
            transform if move_exact_deck_tensors else _identity_tensor
        ),
    )


def concatenate_native_simple_stateless_batches(
    batches: Sequence[NativeSimpleStatelessPolicyBatch],
) -> NativeSimpleStatelessPolicyBatch:
    """Pad and concatenate worker-local native batches for one GPU forward."""
    if not batches:
        raise ValueError("native policy batch concatenation requires rows")
    if len(batches) == 1:
        return batches[0]
    catalog_fingerprints = {
        batch.public_deck_catalog_fingerprint for batch in batches
    }
    contract_fingerprints = {
        batch.input_contract_fingerprint for batch in batches
    }
    belief_fingerprints = {
        batch.belief_summary.catalog_fingerprint for batch in batches
    }
    if (
        len(catalog_fingerprints) != 1
        or len(contract_fingerprints) != 1
        or belief_fingerprints != catalog_fingerprints
    ):
        raise ValueError("native worker batches cross immutable input contracts")

    states = StateBatch(
        card_ids=_pad_cat(tuple(batch.states.card_ids for batch in batches)),
        areas=_pad_cat(tuple(batch.states.areas for batch in batches)),
        owner_roles=_pad_cat(tuple(batch.states.owner_roles for batch in batches)),
        token_kinds=_pad_cat(tuple(batch.states.token_kinds for batch in batches)),
        scalars=_pad_cat(tuple(batch.states.scalars for batch in batches)),
        last_attack_ids=_pad_cat(
            tuple(batch.states.last_attack_ids for batch in batches)
        ),
        padding_mask=_pad_cat(
            tuple(batch.states.padding_mask for batch in batches),
            fill=True,
        ),
        attachment_card_ids=_pad_cat_optional(
            tuple(batch.states.attachment_card_ids for batch in batches)
        ),
        attachment_parent_indices=_pad_cat_optional(
            tuple(batch.states.attachment_parent_indices for batch in batches)
        ),
        attachment_kinds=_pad_cat_optional(
            tuple(batch.states.attachment_kinds for batch in batches)
        ),
        entity_slots=_pad_cat_optional(
            tuple(batch.states.entity_slots for batch in batches)
        ),
        root_input_fingerprints=tuple(
            value
            for batch in batches
            for value in batch.states.root_input_fingerprints
        ),
        sequence_lengths=tuple(
            value for batch in batches for value in batch.states.sequence_lengths
        ),
    )
    options = OptionBatch(
        option_types=_pad_cat(tuple(batch.options.option_types for batch in batches)),
        contexts=_pad_cat(tuple(batch.options.contexts for batch in batches)),
        entity_slots=_pad_cat(
            tuple(batch.options.entity_slots for batch in batches)
        ),
        entity_slot_mask=_pad_cat(
            tuple(batch.options.entity_slot_mask for batch in batches)
        ),
        attack_ids=_pad_cat(tuple(batch.options.attack_ids for batch in batches)),
        card_ids=_pad_cat(tuple(batch.options.card_ids for batch in batches)),
        scalars=_pad_cat(tuple(batch.options.scalars for batch in batches)),
        dynamic_effect_features=_pad_cat(
            tuple(batch.options.dynamic_effect_features for batch in batches)
        ),
        dynamic_effect_masks=_pad_cat(
            tuple(batch.options.dynamic_effect_masks for batch in batches)
        ),
        valid_options=_pad_cat(
            tuple(batch.options.valid_options for batch in batches),
            fill=False,
        ),
        min_counts=torch.cat(
            tuple(batch.options.min_counts for batch in batches),
            dim=0,
        ),
        max_counts=torch.cat(
            tuple(batch.options.max_counts for batch in batches),
            dim=0,
        ),
        option_lengths=tuple(
            value for batch in batches for value in batch.options.option_lengths
        ),
        maximum_counts=tuple(
            value for batch in batches for value in batch.options.maximum_counts
        ),
    )
    belief_rows = tuple(_expanded_belief_rows(batch) for batch in batches)
    belief = PublicBeliefSummaryBatch(
        card_ids=_pad_cat(tuple(row[0] for row in belief_rows)),
        expected_counts=_pad_cat(tuple(row[1] for row in belief_rows)),
        valid_mask=_pad_cat(
            tuple(row[2] for row in belief_rows),
            fill=False,
        ),
        scalars=torch.cat(tuple(row[3] for row in belief_rows), dim=0),
        catalog_fingerprint=next(iter(catalog_fingerprints)),
        row_indices=None,
    )
    return NativeSimpleStatelessPolicyBatch(
        states=states,
        options=options,
        unique_deck_card_ids=_pad_cat(
            tuple(batch.unique_deck_card_ids for batch in batches)
        ),
        deck_counts=_pad_cat(tuple(batch.deck_counts for batch in batches)),
        deck_valid_mask=_pad_cat(
            tuple(batch.deck_valid_mask for batch in batches),
            fill=False,
        ),
        deck_signatures=tuple(
            signature for batch in batches for signature in batch.deck_signatures
        ),
        belief_summary=belief,
        min_counts=tuple(
            value for batch in batches for value in batch.min_counts
        ),
        max_counts=tuple(
            value for batch in batches for value in batch.max_counts
        ),
        public_deck_catalog_fingerprint=next(iter(catalog_fingerprints)),
        input_contract_fingerprint=next(iter(contract_fingerprints)),
    )


def _map_native_simple_stateless_batch_tensors(
    batch: NativeSimpleStatelessPolicyBatch,
    transform: Callable[[Tensor], Tensor],
    *,
    exact_deck_transform: Callable[[Tensor], Tensor] | None = None,
) -> NativeSimpleStatelessPolicyBatch:
    """Apply one storage transform while preserving immutable host metadata."""
    move_exact_deck = transform if exact_deck_transform is None else exact_deck_transform

    def moved(value: Tensor | None) -> Tensor | None:
        return None if value is None else transform(value)

    states = StateBatch(
        card_ids=transform(batch.states.card_ids),
        areas=transform(batch.states.areas),
        owner_roles=transform(batch.states.owner_roles),
        token_kinds=transform(batch.states.token_kinds),
        scalars=transform(batch.states.scalars),
        last_attack_ids=transform(batch.states.last_attack_ids),
        padding_mask=transform(batch.states.padding_mask),
        attachment_card_ids=moved(batch.states.attachment_card_ids),
        attachment_parent_indices=moved(batch.states.attachment_parent_indices),
        attachment_kinds=moved(batch.states.attachment_kinds),
        entity_slots=moved(batch.states.entity_slots),
        root_input_fingerprints=batch.states.root_input_fingerprints,
        sequence_lengths=batch.states.sequence_lengths,
    )
    options = OptionBatch(
        option_types=transform(batch.options.option_types),
        contexts=transform(batch.options.contexts),
        entity_slots=transform(batch.options.entity_slots),
        entity_slot_mask=transform(batch.options.entity_slot_mask),
        attack_ids=transform(batch.options.attack_ids),
        card_ids=transform(batch.options.card_ids),
        scalars=transform(batch.options.scalars),
        dynamic_effect_features=transform(batch.options.dynamic_effect_features),
        dynamic_effect_masks=transform(batch.options.dynamic_effect_masks),
        valid_options=transform(batch.options.valid_options),
        min_counts=transform(batch.options.min_counts),
        max_counts=transform(batch.options.max_counts),
        option_lengths=batch.options.option_lengths,
        maximum_counts=batch.options.maximum_counts,
    )
    belief = PublicBeliefSummaryBatch(
        card_ids=transform(batch.belief_summary.card_ids),
        expected_counts=transform(batch.belief_summary.expected_counts),
        valid_mask=transform(batch.belief_summary.valid_mask),
        scalars=transform(batch.belief_summary.scalars),
        catalog_fingerprint=batch.belief_summary.catalog_fingerprint,
        row_indices=moved(batch.belief_summary.row_indices),
    )
    return NativeSimpleStatelessPolicyBatch(
        states=states,
        options=options,
        unique_deck_card_ids=move_exact_deck(batch.unique_deck_card_ids),
        deck_counts=move_exact_deck(batch.deck_counts),
        deck_valid_mask=move_exact_deck(batch.deck_valid_mask),
        deck_signatures=batch.deck_signatures,
        belief_summary=belief,
        min_counts=batch.min_counts,
        max_counts=batch.max_counts,
        public_deck_catalog_fingerprint=(batch.public_deck_catalog_fingerprint),
        input_contract_fingerprint=batch.input_contract_fingerprint,
    )


def _expanded_belief_rows(
    batch: NativeSimpleStatelessPolicyBatch,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Expand optional deduplicated belief rows to the decision batch."""
    belief = batch.belief_summary
    if belief.row_indices is None:
        return (
            belief.card_ids,
            belief.expected_counts,
            belief.valid_mask,
            belief.scalars,
        )
    rows = belief.row_indices.to(device=belief.card_ids.device, dtype=torch.long)
    return (
        belief.card_ids.index_select(0, rows),
        belief.expected_counts.index_select(0, rows),
        belief.valid_mask.index_select(0, rows),
        belief.scalars.index_select(0, rows),
    )


def _pad_cat(
    tensors: Sequence[Tensor],
    *,
    fill: int | float | bool = 0,
) -> Tensor:
    """Right-pad non-batch dimensions to their maxima, then concatenate."""
    if not tensors:
        raise ValueError("tensor concatenation requires at least one tensor")
    ranks = {tensor.ndim for tensor in tensors}
    devices = {tensor.device for tensor in tensors}
    dtypes = {tensor.dtype for tensor in tensors}
    if len(ranks) != 1 or len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("padded tensors must share rank, device, and dtype")
    rank = next(iter(ranks))
    if rank < 1:
        raise ValueError("padded tensors require a batch dimension")
    target = tuple(
        max(int(tensor.shape[axis]) for tensor in tensors)
        for axis in range(1, rank)
    )
    padded: list[Tensor] = []
    for tensor in tensors:
        if tuple(int(value) for value in tensor.shape[1:]) == target:
            padded.append(tensor)
            continue
        shape = (int(tensor.shape[0]), *target)
        result = torch.full(
            shape,
            fill,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        slices = (slice(None),) + tuple(
            slice(0, int(tensor.shape[axis])) for axis in range(1, rank)
        )
        result[slices] = tensor
        padded.append(result)
    return torch.cat(tuple(padded), dim=0)


def _pad_cat_optional(tensors: Sequence[Tensor | None]) -> Tensor | None:
    """Concatenate an optional tensor field that must be uniformly present."""
    if all(tensor is None for tensor in tensors):
        return None
    if any(tensor is None for tensor in tensors):
        raise ValueError("native worker batches mix optional tensor schemas")
    return _pad_cat(tuple(cast(Tensor, tensor) for tensor in tensors))


def encode_native_simple_stateless_batch(
    view: NativeTrainingBatchView,
    context: NativePublicContextBatch,
    *,
    device: torch.device | str | None = None,
    deck_cache: NativeDeckBatchCache | None = None,
) -> NativeSimpleStatelessPolicyBatch:
    """Build all simple-stateless model tensors directly from native columns."""
    states, lookup = encode_native_state_batch(
        view,
        context,
        device=device,
    )
    options = encode_native_option_batch(
        view,
        lookup,
        device=device,
    )
    deck_card_ids, deck_counts, deck_valid = (
        _collate_decks(
            context.own_decks,
            device=device,
        )
        if deck_cache is None
        else deck_cache.collate(
            context.own_decks,
            context.deck_signatures,
            device=device,
        )
    )
    belief = _collate_belief(
        context,
        device=device,
    )
    option_lengths = cast(
        npt.NDArray[np.int64],
        np.diff(view.option_offsets.astype(np.int64, copy=False)),
    )
    minimums = np.minimum(option_lengths, np.maximum(0, view.select_min))
    maximums = np.minimum(
        option_lengths,
        np.maximum(minimums, view.select_max),
    )
    return NativeSimpleStatelessPolicyBatch(
        states=states,
        options=options,
        unique_deck_card_ids=deck_card_ids,
        deck_counts=deck_counts,
        deck_valid_mask=deck_valid,
        deck_signatures=context.deck_signatures,
        belief_summary=belief,
        min_counts=tuple(int(value) for value in minimums),
        max_counts=tuple(int(value) for value in maximums),
        public_deck_catalog_fingerprint=context.catalog_fingerprint,
        input_contract_fingerprint=context.input_contract_fingerprint,
    )


def _collate_decks(
    decks: np.ndarray,
    *,
    device: torch.device | str | None,
) -> tuple[Tensor, Tensor, Tensor]:
    sorted_decks = np.sort(decks.astype(np.int64, copy=False), axis=1)
    run_starts = np.ones(sorted_decks.shape, dtype=np.bool_)
    run_starts[:, 1:] = sorted_decks[:, 1:] != sorted_decks[:, :-1]
    slots = np.cumsum(run_starts, axis=1, dtype=np.int64) - 1
    width = int(run_starts.sum(axis=1).max())
    card_ids = np.zeros((decks.shape[0], width), dtype=np.int64)
    counts = np.zeros((decks.shape[0], width), dtype=np.float32)
    valid = np.zeros((decks.shape[0], width), dtype=np.bool_)
    rows = np.broadcast_to(
        np.arange(decks.shape[0], dtype=np.int64)[:, None],
        decks.shape,
    )
    start_rows = rows[run_starts]
    start_slots = slots[run_starts]
    card_ids[start_rows, start_slots] = sorted_decks[run_starts]
    valid[start_rows, start_slots] = True
    np.add.at(counts, (rows.reshape(-1), slots.reshape(-1)), 1.0)
    return (
        torch.as_tensor(card_ids, device=device),
        torch.as_tensor(counts, device=device),
        torch.as_tensor(valid, device=device),
    )


def _collate_belief(
    context: NativePublicContextBatch,
    *,
    device: torch.device | str | None,
) -> PublicBeliefSummaryBatch:
    offsets: npt.NDArray[np.int64] = context.belief_summary_offsets.astype(
        np.int64, copy=False
    )
    lengths = np.diff(offsets)
    width = max(1, int(lengths.max(initial=0)))
    unique_rows = int(lengths.shape[0])
    dense_rows = unique_rows > 0 and bool(np.all(lengths == width))
    if dense_rows:
        card_ids = context.belief_summary_card_ids.reshape(
            unique_rows,
            width,
        ).astype(np.int64, copy=False)
        expected = context.belief_summary_expected_counts.reshape(
            unique_rows,
            width,
        ).astype(np.float32, copy=False)
        valid = np.ones((unique_rows, width), dtype=np.bool_)
    else:
        card_ids = np.zeros(
            (unique_rows, width),
            dtype=np.int64,
        )
        expected = np.zeros(
            (unique_rows, width),
            dtype=np.float32,
        )
        valid = np.zeros(
            (unique_rows, width),
            dtype=np.bool_,
        )
    if int(offsets[-1]) > 0 and not dense_rows:
        value_rows: npt.NDArray[np.int64] = np.repeat(
            np.arange(unique_rows, dtype=np.int64),
            lengths,
        )
        value_starts: npt.NDArray[np.int64] = np.repeat(
            offsets[:-1],
            lengths,
        )
        value_local = np.arange(int(offsets[-1]), dtype=np.int64) - value_starts
        card_ids[value_rows, value_local] = context.belief_summary_card_ids.astype(
            np.int64, copy=False
        )
        expected[value_rows, value_local] = (
            context.belief_summary_expected_counts.astype(
                np.float32,
                copy=False,
            )
        )
        valid[value_rows, value_local] = True

    scalars: npt.NDArray[np.float32] = context.belief_summary_scalars.astype(
        np.float32, copy=False
    )
    inverse: npt.NDArray[np.int64] = context.belief_summary_row_indices.astype(
        np.int64,
        copy=False,
    )
    row_indices: Tensor | None = None
    if unique_rows != context.batch_size or not np.array_equal(
        inverse,
        np.arange(context.batch_size, dtype=np.int64),
    ):
        row_indices = torch.as_tensor(
            inverse,
            dtype=torch.long,
            device=device,
        )
    return PublicBeliefSummaryBatch(
        card_ids=torch.as_tensor(card_ids, device=device),
        expected_counts=torch.as_tensor(expected, device=device),
        valid_mask=torch.as_tensor(valid, device=device),
        scalars=torch.as_tensor(scalars, device=device),
        catalog_fingerprint=context.catalog_fingerprint,
        row_indices=row_indices,
    )


__all__ = [
    "NativeDeckBatchCache",
    "NativeSimpleStatelessPolicyBatch",
    "concatenate_native_simple_stateless_batches",
    "encode_native_simple_stateless_batch",
    "move_native_simple_stateless_batch",
]
