"""Shared deck-macro weighting for object, array, and logical PPO paths."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def normalized_present_deck_shares(
    deck_digests: Iterable[str],
    target_shares: Mapping[str, float] | None,
) -> dict[str, float]:
    """Normalize configured macro mass over only the decks present in a batch."""
    decks = tuple(sorted(set(deck_digests)))
    if not decks:
        return {}
    if target_shares is None:
        share = 1.0 / float(len(decks))
        return dict.fromkeys(decks, share)
    unknown = set(decks) - set(target_shares)
    if unknown:
        raise ValueError(
            "macro target shares omit batch decks: " + ", ".join(sorted(unknown))
        )
    values = {deck: float(target_shares[deck]) for deck in decks}
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("macro target shares must be positive and finite")
    total = sum(values.values())
    return {deck: value / total for deck, value in values.items()}


__all__ = ["normalized_present_deck_shares"]
