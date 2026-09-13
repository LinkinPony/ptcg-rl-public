"""Columnar public context consumed by native stateless policy encoding.

The native engine exports the current public game state.  Cross-decision
features and immutable policy identities are deliberately supplied separately:
they are policy state, not game-rule state.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ptcg_rl.context import DECK_FLOW_FEATURE_SIZE, HISTORY_COUNTER_SIZE
from ptcg_rl.decks.identity import DECK_SIZE

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
BELIEF_SUMMARY_SCALAR_SIZE = 4


@dataclass(frozen=True, slots=True)
class NativePublicContextBatch:
    """Array-only side input aligned with one native arena output batch.

    CSR value rows preserve the deterministic order emitted by the public
    tracker.  ``belief_summary_*`` is the full catalog posterior consumed by
    ``PublicBeliefSummaryEncoder``; it is intentionally distinct from the
    legacy state-token belief approximation.  The simple public-catalog input
    contract does not add those legacy belief tokens.
    """

    own_unseen_offsets: np.ndarray
    own_unseen_card_ids: np.ndarray
    own_unseen_counts: np.ndarray
    opponent_revealed_offsets: np.ndarray
    opponent_revealed_card_ids: np.ndarray
    opponent_revealed_counts: np.ndarray
    history_counts: np.ndarray
    deck_flow_counts: np.ndarray
    last_attack_offsets: np.ndarray
    last_attack_serials: np.ndarray
    last_attack_ids: np.ndarray
    belief_summary_offsets: np.ndarray
    belief_summary_card_ids: np.ndarray
    belief_summary_expected_counts: np.ndarray
    belief_summary_scalars: np.ndarray
    belief_summary_row_indices: np.ndarray
    own_decks: np.ndarray
    deck_signatures: tuple[str, ...]
    catalog_fingerprint: str
    input_contract_fingerprint: str

    @property
    def batch_size(self) -> int:
        """Return the number of aligned policy rows."""
        if self.history_counts.ndim != 2:
            return 0
        return int(self.history_counts.shape[0])

    def validate(self, *, batch_size: int) -> None:
        """Fail closed before any model tensor is constructed."""
        if batch_size <= 0:
            raise ValueError("native policy batch must be non-empty")
        if self.batch_size != batch_size:
            raise ValueError("native public context batch size does not align")
        _require_shape(
            self.history_counts,
            (batch_size, HISTORY_COUNTER_SIZE),
            "history_counts",
        )
        _require_shape(
            self.deck_flow_counts,
            (batch_size, DECK_FLOW_FEATURE_SIZE),
            "deck_flow_counts",
        )
        _require_integer(self.history_counts, "history_counts")
        _require_integer(self.deck_flow_counts, "deck_flow_counts")
        if np.any(self.history_counts < 0):
            raise ValueError("history_counts must be non-negative")

        _validate_count_csr(
            self.own_unseen_offsets,
            self.own_unseen_card_ids,
            self.own_unseen_counts,
            batch_size=batch_size,
            name="own_unseen",
        )
        _validate_count_csr(
            self.opponent_revealed_offsets,
            self.opponent_revealed_card_ids,
            self.opponent_revealed_counts,
            batch_size=batch_size,
            name="opponent_revealed",
        )
        _validate_pair_csr(
            self.last_attack_offsets,
            self.last_attack_serials,
            self.last_attack_ids,
            batch_size=batch_size,
            name="last_attack",
        )
        if np.any(self.last_attack_serials <= 0) or np.any(self.last_attack_ids <= 0):
            raise ValueError("last-attack serials and attack IDs must be positive")

        if (
            self.belief_summary_scalars.ndim != 2
            or self.belief_summary_scalars.shape[1]
            != BELIEF_SUMMARY_SCALAR_SIZE
        ):
            raise ValueError(
                "belief_summary_scalars must have shape "
                f"[unique_rows, {BELIEF_SUMMARY_SCALAR_SIZE}]"
            )
        unique_belief_rows = int(self.belief_summary_scalars.shape[0])
        if unique_belief_rows <= 0 or unique_belief_rows > batch_size:
            raise ValueError(
                "belief summary unique-row count must be within policy batch"
            )
        _validate_pair_csr(
            self.belief_summary_offsets,
            self.belief_summary_card_ids,
            self.belief_summary_expected_counts,
            batch_size=unique_belief_rows,
            name="belief_summary",
        )
        if np.any(self.belief_summary_card_ids <= 0):
            raise ValueError("belief summary card IDs must be positive")
        if (
            not np.issubdtype(
                self.belief_summary_expected_counts.dtype,
                np.floating,
            )
            or np.any(~np.isfinite(self.belief_summary_expected_counts))
            or np.any(self.belief_summary_expected_counts < 0.0)
        ):
            raise ValueError(
                "belief summary expected counts must be finite and non-negative"
            )
        if not np.issubdtype(self.belief_summary_scalars.dtype, np.floating) or np.any(
            ~np.isfinite(self.belief_summary_scalars)
        ):
            raise ValueError("belief summary scalars must be finite floats")
        _require_shape(
            self.belief_summary_row_indices,
            (batch_size,),
            "belief_summary_row_indices",
        )
        _require_integer(
            self.belief_summary_row_indices,
            "belief_summary_row_indices",
        )
        normalized_belief_rows = self.belief_summary_row_indices.astype(
            np.int64,
            copy=False,
        )
        if np.any(normalized_belief_rows < 0) or np.any(
            normalized_belief_rows >= unique_belief_rows
        ):
            raise ValueError("belief summary row index is out of range")
        referenced_belief_rows = np.zeros(unique_belief_rows, dtype=np.bool_)
        referenced_belief_rows[normalized_belief_rows] = True
        if not bool(np.all(referenced_belief_rows)):
            raise ValueError("belief summary unique rows must all be referenced")

        _require_shape(
            self.own_decks,
            (batch_size, DECK_SIZE),
            "own_decks",
        )
        _require_integer(self.own_decks, "own_decks")
        if np.any(self.own_decks <= 0):
            raise ValueError("own exact decks must contain positive card IDs")
        if len(self.deck_signatures) != batch_size or any(
            not signature for signature in self.deck_signatures
        ):
            raise ValueError("deck signatures must align with native policy rows")
        _require_sha256(self.catalog_fingerprint, "catalog_fingerprint")
        _require_sha256(
            self.input_contract_fingerprint,
            "input_contract_fingerprint",
        )


def concatenate_native_public_context_batches(
    batches: Sequence[NativePublicContextBatch],
) -> NativePublicContextBatch:
    """Concatenate aligned public-context batches and rebase every CSR row."""
    selected = tuple(batches)
    if not selected:
        raise ValueError("native context concatenation requires at least one batch")
    if any(batch.batch_size <= 0 for batch in selected):
        raise ValueError("native context concatenation requires non-empty batches")
    if len(selected) == 1:
        return selected[0]
    catalog_fingerprint = selected[0].catalog_fingerprint
    input_contract_fingerprint = selected[0].input_contract_fingerprint
    if any(
        batch.catalog_fingerprint != catalog_fingerprint
        or batch.input_contract_fingerprint != input_contract_fingerprint
        for batch in selected[1:]
    ):
        raise ValueError("native context concatenation identities differ")

    own_unseen_offsets, own_unseen = _concatenate_pair_csr(
        selected,
        offsets="own_unseen_offsets",
        left="own_unseen_card_ids",
        right="own_unseen_counts",
    )
    opponent_revealed_offsets, opponent_revealed = _concatenate_pair_csr(
        selected,
        offsets="opponent_revealed_offsets",
        left="opponent_revealed_card_ids",
        right="opponent_revealed_counts",
    )
    last_attack_offsets, last_attack = _concatenate_pair_csr(
        selected,
        offsets="last_attack_offsets",
        left="last_attack_serials",
        right="last_attack_ids",
    )
    belief_summary_offsets, belief_summary = _concatenate_pair_csr(
        selected,
        offsets="belief_summary_offsets",
        left="belief_summary_card_ids",
        right="belief_summary_expected_counts",
    )
    belief_bases = np.cumsum(
        np.asarray(
            (0, *(batch.belief_summary_scalars.shape[0] for batch in selected[:-1])),
            dtype=np.uint64,
        )
    )
    if int(belief_bases[-1]) > int(np.iinfo(np.uint32).max):
        raise OverflowError("native belief row concatenation exceeds uint32 capacity")
    result = NativePublicContextBatch(
        own_unseen_offsets=own_unseen_offsets,
        own_unseen_card_ids=own_unseen[0],
        own_unseen_counts=own_unseen[1],
        opponent_revealed_offsets=opponent_revealed_offsets,
        opponent_revealed_card_ids=opponent_revealed[0],
        opponent_revealed_counts=opponent_revealed[1],
        history_counts=np.concatenate(
            tuple(batch.history_counts for batch in selected)
        ),
        deck_flow_counts=np.concatenate(
            tuple(batch.deck_flow_counts for batch in selected)
        ),
        last_attack_offsets=last_attack_offsets,
        last_attack_serials=last_attack[0],
        last_attack_ids=last_attack[1],
        belief_summary_offsets=belief_summary_offsets,
        belief_summary_card_ids=belief_summary[0],
        belief_summary_expected_counts=belief_summary[1],
        belief_summary_scalars=np.concatenate(
            tuple(batch.belief_summary_scalars for batch in selected)
        ),
        belief_summary_row_indices=np.concatenate(
            tuple(
                batch.belief_summary_row_indices + np.uint32(base)
                for batch, base in zip(selected, belief_bases, strict=True)
            )
        ),
        own_decks=np.concatenate(tuple(batch.own_decks for batch in selected)),
        deck_signatures=tuple(
            signature
            for batch in selected
            for signature in batch.deck_signatures
        ),
        catalog_fingerprint=catalog_fingerprint,
        input_contract_fingerprint=input_contract_fingerprint,
    )
    result.validate(batch_size=sum(batch.batch_size for batch in selected))
    return result


def select_native_public_context_rows(
    context: NativePublicContextBatch,
    rows: npt.ArrayLike,
) -> NativePublicContextBatch:
    """Copy arbitrary unique rows and rebase every context CSR column."""
    selected = _validated_rows(rows, size=context.batch_size)
    own_unseen_offsets, own_unseen = _select_pair_csr(
        context.own_unseen_offsets,
        selected,
        context.own_unseen_card_ids,
        context.own_unseen_counts,
    )
    opponent_revealed_offsets, opponent_revealed = _select_pair_csr(
        context.opponent_revealed_offsets,
        selected,
        context.opponent_revealed_card_ids,
        context.opponent_revealed_counts,
    )
    last_attack_offsets, last_attack = _select_pair_csr(
        context.last_attack_offsets,
        selected,
        context.last_attack_serials,
        context.last_attack_ids,
    )
    belief_source_rows = context.belief_summary_row_indices[selected].astype(
        np.int64,
        copy=False,
    )
    belief_unique_rows, belief_row_indices = _compact_referenced_rows(
        belief_source_rows
    )
    belief_summary_offsets, belief_summary = _select_pair_csr(
        context.belief_summary_offsets,
        belief_unique_rows,
        context.belief_summary_card_ids,
        context.belief_summary_expected_counts,
    )
    result = NativePublicContextBatch(
        own_unseen_offsets=own_unseen_offsets,
        own_unseen_card_ids=own_unseen[0],
        own_unseen_counts=own_unseen[1],
        opponent_revealed_offsets=opponent_revealed_offsets,
        opponent_revealed_card_ids=opponent_revealed[0],
        opponent_revealed_counts=opponent_revealed[1],
        history_counts=context.history_counts[selected].copy(),
        deck_flow_counts=context.deck_flow_counts[selected].copy(),
        last_attack_offsets=last_attack_offsets,
        last_attack_serials=last_attack[0],
        last_attack_ids=last_attack[1],
        belief_summary_offsets=belief_summary_offsets,
        belief_summary_card_ids=belief_summary[0],
        belief_summary_expected_counts=belief_summary[1],
        belief_summary_scalars=context.belief_summary_scalars[
            belief_unique_rows
        ].copy(),
        belief_summary_row_indices=belief_row_indices,
        own_decks=context.own_decks[selected].copy(),
        deck_signatures=tuple(context.deck_signatures[int(row)] for row in selected),
        catalog_fingerprint=context.catalog_fingerprint,
        input_contract_fingerprint=context.input_contract_fingerprint,
    )
    result.validate(batch_size=int(selected.size))
    return result


def _compact_referenced_rows(
    source_rows: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.uint32]]:
    """Compact source rows by first appearance and return their inverse."""
    selected: list[int] = []
    destination_by_source: dict[int, int] = {}
    inverse = np.empty(source_rows.shape[0], dtype=np.uint32)
    for row, raw_source in enumerate(source_rows):
        source = int(raw_source)
        destination = destination_by_source.get(source)
        if destination is None:
            destination = len(selected)
            destination_by_source[source] = destination
            selected.append(source)
        inverse[row] = destination
    return np.asarray(selected, dtype=np.int64), inverse


def _validated_rows(values: npt.ArrayLike, *, size: int) -> npt.NDArray[np.int64]:
    rows = np.asarray(values)
    if rows.ndim != 1 or rows.size <= 0:
        raise ValueError("native context row selection must be a non-empty vector")
    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError("native context row selection must use an integer dtype")
    normalized = rows.astype(np.int64, copy=False)
    if (
        np.any(normalized < 0)
        or np.any(normalized >= size)
        or np.unique(normalized).size != normalized.size
    ):
        raise ValueError("native context row selection must be unique and in range")
    return normalized


def _select_pair_csr(
    offsets: np.ndarray,
    rows: npt.NDArray[np.int64],
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    source_starts = offsets[rows].astype(np.int64, copy=False)
    source_stops = offsets[rows + 1].astype(np.int64, copy=False)
    lengths = source_stops - source_starts
    destination_offsets = np.zeros(rows.size + 1, dtype=np.uint32)
    np.cumsum(lengths, dtype=np.uint32, out=destination_offsets[1:])
    destination_left = np.empty(int(destination_offsets[-1]), dtype=left.dtype)
    destination_right = np.empty(int(destination_offsets[-1]), dtype=right.dtype)
    for row, (source_start, source_stop) in enumerate(
        zip(source_starts, source_stops, strict=True)
    ):
        destination_start = int(destination_offsets[row])
        destination_stop = int(destination_offsets[row + 1])
        destination_left[destination_start:destination_stop] = left[
            source_start:source_stop
        ]
        destination_right[destination_start:destination_stop] = right[
            source_start:source_stop
        ]
    return destination_offsets, (destination_left, destination_right)


def _concatenate_pair_csr(
    batches: Sequence[NativePublicContextBatch],
    *,
    offsets: str,
    left: str,
    right: str,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    batch_offsets = tuple(getattr(batch, offsets) for batch in batches)
    lengths = np.concatenate(
        tuple(
            np.diff(values.astype(np.int64, copy=False))
            for values in batch_offsets
        )
    )
    total = int(np.sum(lengths, dtype=np.int64))
    if total > int(np.iinfo(np.uint32).max):
        raise OverflowError("native context CSR exceeds uint32 capacity")
    destination_offsets = np.zeros(lengths.size + 1, dtype=np.uint32)
    np.cumsum(lengths, dtype=np.uint32, out=destination_offsets[1:])
    return destination_offsets, (
        np.concatenate(tuple(getattr(batch, left) for batch in batches)),
        np.concatenate(tuple(getattr(batch, right) for batch in batches)),
    )


def _validate_count_csr(
    offsets: np.ndarray,
    card_ids: np.ndarray,
    counts: np.ndarray,
    *,
    batch_size: int,
    name: str,
) -> None:
    _validate_pair_csr(
        offsets,
        card_ids,
        counts,
        batch_size=batch_size,
        name=name,
    )
    if np.any(card_ids <= 0):
        raise ValueError(f"{name} card IDs must be positive")
    if not np.issubdtype(counts.dtype, np.number) or np.any(~np.isfinite(counts)):
        raise ValueError(f"{name} counts must be finite numbers")
    if np.any(counts <= 0):
        raise ValueError(f"{name} counts must be positive")


def _validate_pair_csr(
    offsets: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    batch_size: int,
    name: str,
) -> None:
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError(f"{name} CSR value arrays must be aligned vectors")
    _require_integer(left, f"{name} left values")
    _validate_offsets(
        offsets,
        value_count=int(left.shape[0]),
        batch_size=batch_size,
        name=name,
    )


def _validate_offsets(
    offsets: np.ndarray,
    *,
    value_count: int,
    batch_size: int,
    name: str,
) -> None:
    _require_shape(offsets, (batch_size + 1,), f"{name}_offsets")
    _require_integer(offsets, f"{name}_offsets")
    normalized = offsets.astype(np.int64, copy=False)
    if (
        int(normalized[0]) != 0
        or int(normalized[-1]) != value_count
        or np.any(normalized[1:] < normalized[:-1])
    ):
        raise ValueError(f"{name} offsets are not canonical CSR offsets")


def _require_shape(
    values: np.ndarray,
    shape: tuple[int, ...],
    name: str,
) -> None:
    if values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {values.shape}")


def _require_integer(values: np.ndarray, name: str) -> None:
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must use an integer dtype")


def _require_sha256(value: str, name: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 fingerprint")


__all__ = [
    "BELIEF_SUMMARY_SCALAR_SIZE",
    "NativePublicContextBatch",
    "concatenate_native_public_context_batches",
    "select_native_public_context_rows",
]
