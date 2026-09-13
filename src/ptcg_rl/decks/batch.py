"""Model-agnostic batches of persistent acting-seat deck identities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from ptcg_rl.decks.identity import DECK_SIZE, CanonicalDeck, canonicalize_deck


@dataclass(frozen=True, slots=True)
class DeckBatch:
    """Canonical deck cards and CPU signatures aligned by batch row."""

    card_ids: Tensor
    signatures: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate tensor shape, dtype, and row/signature correspondence."""
        if self.card_ids.dtype != torch.long:
            raise TypeError("DeckBatch.card_ids must use torch.long dtype")
        if self.card_ids.ndim != 2 or self.card_ids.shape[1] != DECK_SIZE:
            raise ValueError(
                f"DeckBatch.card_ids must have shape [B, {DECK_SIZE}], got "
                f"{tuple(self.card_ids.shape)}"
            )
        if len(self.signatures) != self.card_ids.shape[0]:
            raise ValueError("DeckBatch signatures must match the tensor batch size")
        rows = self.card_ids.detach().cpu().tolist()
        for index, (row, signature) in enumerate(zip(rows, self.signatures, strict=True)):
            deck = canonicalize_deck(row)
            if tuple(row) != deck.card_ids:
                raise ValueError(f"DeckBatch row {index} card IDs must be sorted")
            if signature != deck.signature:
                raise ValueError(
                    f"DeckBatch row {index} signature does not match card IDs"
                )

    def __len__(self) -> int:
        """Return the number of deck rows."""
        return self.card_ids.shape[0]

    @classmethod
    def from_decks(
        cls,
        decks: Sequence[CanonicalDeck],
        *,
        device: torch.device | str | None = None,
    ) -> Self:
        """Collate canonical decks without adding model-specific routing state."""
        if decks:
            card_ids = torch.tensor([deck.card_ids for deck in decks], dtype=torch.long)
        else:
            card_ids = torch.empty((0, DECK_SIZE), dtype=torch.long)
        batch = cls(
            card_ids=card_ids,
            signatures=tuple(deck.signature for deck in decks),
        )
        return batch if device is None else batch.to(device)

    @classmethod
    def from_card_ids(
        cls,
        decks: Iterable[Iterable[object]],
        *,
        device: torch.device | str | None = None,
    ) -> Self:
        """Canonicalize and collate raw card-ID iterables."""
        return cls.from_decks(
            tuple(canonicalize_deck(deck) for deck in decks), device=device
        )

    @classmethod
    def _from_validated(cls, card_ids: Tensor, signatures: tuple[str, ...]) -> Self:
        """Build from operations that preserve an already validated batch."""
        batch = object.__new__(cls)
        object.__setattr__(batch, "card_ids", card_ids)
        object.__setattr__(batch, "signatures", signatures)
        return batch

    def select(self, indices: Sequence[int] | Tensor) -> Self:
        """Select rows while preserving card/signature alignment."""
        if isinstance(indices, Tensor):
            if indices.ndim != 1 or indices.dtype != torch.long:
                raise TypeError("DeckBatch indices tensor must be one-dimensional long")
            cpu_indices = tuple(int(index) for index in indices.detach().cpu().tolist())
            device_indices = indices.to(device=self.card_ids.device)
        else:
            cpu_indices = tuple(indices)
            device_indices = torch.tensor(
                cpu_indices, dtype=torch.long, device=self.card_ids.device
            )
        return type(self)._from_validated(
            self.card_ids.index_select(0, device_indices),
            tuple(self.signatures[index] for index in cpu_indices),
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Self:
        """Move card tensors while retaining CPU-side identity strings."""
        return type(self)._from_validated(
            self.card_ids.to(device=device, non_blocking=non_blocking),
            self.signatures,
        )

    def pin_memory(self) -> Self:
        """Pin the card tensor for asynchronous accelerator transfer."""
        return type(self)._from_validated(
            self.card_ids.pin_memory(), self.signatures
        )


def concatenate_deck_batches(batches: Sequence[DeckBatch]) -> DeckBatch:
    """Concatenate aligned deck batches without introducing route state."""
    if not batches:
        return DeckBatch.from_decks(())
    device = batches[0].card_ids.device
    if any(batch.card_ids.device != device for batch in batches):
        raise ValueError("all DeckBatch tensors must be on the same device")
    return DeckBatch._from_validated(
        torch.cat([batch.card_ids for batch in batches], dim=0),
        tuple(
            signature for batch in batches for signature in batch.signatures
        ),
    )


def pad_deck_batch(
    batch: DeckBatch,
    size: int,
    *,
    source_index: int = 0,
) -> DeckBatch:
    """Pad static serving rows by repeating one real validated deck."""
    if size < len(batch):
        raise ValueError("padded deck batch size cannot shrink existing rows")
    if size == len(batch):
        return batch
    if not 0 <= source_index < len(batch):
        raise ValueError("deck padding requires a valid real source row")
    padding_size = size - len(batch)
    padding = batch.card_ids[source_index].unsqueeze(0).expand(padding_size, -1)
    return DeckBatch._from_validated(
        torch.cat((batch.card_ids, padding), dim=0),
        batch.signatures + (batch.signatures[source_index],) * padding_size,
    )


def empty_deck_batch_like(batch: DeckBatch) -> DeckBatch:
    """Allocate static card storage while retaining a validated route identity."""
    return DeckBatch._from_validated(
        torch.empty_like(batch.card_ids),
        batch.signatures,
    )


def copy_deck_batch_(target: DeckBatch, source: DeckBatch) -> None:
    """Copy cards into a static batch without allowing its route to change."""
    if target.signatures != source.signatures:
        raise ValueError("static deck batch signatures cannot change")
    if target.card_ids.shape != source.card_ids.shape:
        raise ValueError("static deck batch shapes must match")
    target.card_ids.copy_(source.card_ids)
