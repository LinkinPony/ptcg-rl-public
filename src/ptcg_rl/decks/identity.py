"""Canonical exact-deck identity shared by training and runtime code."""

from __future__ import annotations

import hashlib
import operator
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import SupportsIndex, cast

from ptcg_rl.cards.static_features import DEFAULT_NUM_CARD_IDS

DECK_SIZE = 60
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_SIGNATURE_CHUNK_PATTERN = re.compile(r"[1-9][0-9]*:[1-9][0-9]*")


@dataclass(frozen=True, slots=True)
class CanonicalDeck:
    """Validated, order-independent identity for one exact 60-card deck."""

    card_ids: tuple[int, ...]
    signature: str
    deck_digest: str
    module_key: str

    def __post_init__(self) -> None:
        """Reject manually constructed identities with inconsistent fields."""
        normalized = _normalize_card_ids(self.card_ids)
        if self.card_ids != normalized:
            raise ValueError("canonical deck card_ids must be sorted")
        expected_signature = _signature_from_sorted_card_ids(normalized)
        if self.signature != expected_signature:
            raise ValueError("canonical deck signature does not match card_ids")
        expected_digest = deck_digest(expected_signature)
        if self.deck_digest != expected_digest:
            raise ValueError("canonical deck digest does not match signature")
        if self.module_key != deck_module_key(expected_digest):
            raise ValueError("canonical deck module_key does not match digest")


def canonicalize_deck(
    card_ids: Iterable[object],
    *,
    max_card_id: int = DEFAULT_NUM_CARD_IDS,
) -> CanonicalDeck:
    """Validate and canonicalize an exact 60-card deck."""
    normalized = _normalize_card_ids(card_ids, max_card_id=max_card_id)
    signature = _signature_from_sorted_card_ids(normalized)
    digest = deck_digest(signature)
    return CanonicalDeck(
        card_ids=normalized,
        signature=signature,
        deck_digest=digest,
        module_key=deck_module_key(digest),
    )


def canonical_signature(
    card_ids: Iterable[object],
    *,
    max_card_id: int = DEFAULT_NUM_CARD_IDS,
) -> str:
    """Return the canonical sorted card-count signature for one deck."""
    return canonicalize_deck(card_ids, max_card_id=max_card_id).signature


def parse_canonical_signature(
    signature: str,
    *,
    max_card_id: int = DEFAULT_NUM_CARD_IDS,
) -> CanonicalDeck:
    """Parse a strictly canonical signature into its validated deck identity."""
    if not signature:
        raise ValueError("canonical deck signature must be non-empty")
    chunks = signature.split(";")
    if any(_SIGNATURE_CHUNK_PATTERN.fullmatch(chunk) is None for chunk in chunks):
        raise ValueError(f"invalid canonical deck signature: {signature!r}")

    card_ids: list[int] = []
    previous_card_id = 0
    for chunk in chunks:
        card_id_text, count_text = chunk.split(":", 1)
        card_id = int(card_id_text)
        count = int(count_text)
        if card_id <= previous_card_id:
            raise ValueError("canonical signature card IDs must be strictly increasing")
        previous_card_id = card_id
        card_ids.extend([card_id] * count)

    deck = canonicalize_deck(card_ids, max_card_id=max_card_id)
    if deck.signature != signature:
        raise ValueError("deck signature is not in canonical form")
    return deck


def deck_digest(signature: str) -> str:
    """Return the full SHA256 digest of a canonical signature string."""
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def deck_module_key(digest: str) -> str:
    """Return the stable ``ModuleDict`` key for a full deck digest."""
    if _DIGEST_PATTERN.fullmatch(digest) is None:
        raise ValueError("deck digest must be 64 lowercase hexadecimal characters")
    return f"deck_{digest}"


def _normalize_card_ids(
    card_ids: Iterable[object],
    *,
    max_card_id: int = DEFAULT_NUM_CARD_IDS,
) -> tuple[int, ...]:
    if max_card_id <= 0:
        raise ValueError("max_card_id must be positive")
    normalized: list[int] = []
    for value in card_ids:
        if isinstance(value, bool):
            raise TypeError("deck card IDs must be integers, not bool")
        try:
            card_id = operator.index(cast(SupportsIndex, value))
        except TypeError as exc:
            raise TypeError(f"deck card ID must be an integer: {value!r}") from exc
        if not 1 <= card_id <= max_card_id:
            raise ValueError(
                f"deck card ID must be in [1, {max_card_id}], got {card_id}"
            )
        normalized.append(card_id)
    if len(normalized) != DECK_SIZE:
        raise ValueError(
            f"deck must contain exactly {DECK_SIZE} cards, got {len(normalized)}"
        )
    return tuple(sorted(normalized))


def _signature_from_sorted_card_ids(card_ids: tuple[int, ...]) -> str:
    counts = Counter(card_ids)
    return ";".join(f"{card_id}:{counts[card_id]}" for card_id in sorted(counts))
