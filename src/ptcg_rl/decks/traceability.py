"""Stable human-facing exact-deck traceability identities."""

from __future__ import annotations

import re

_DECK_HASH_PATTERN = re.compile(r"(?<![0-9a-f])([0-9a-f]{12})(?![0-9a-f])")


def authoritative_deck_hash(label: str, *, explicit: str | None = None) -> str:
    """Resolve only a compact ID explicitly stored in metadata or its label."""
    if explicit is not None:
        normalized = explicit.strip().lower()
        if len(normalized) != 12 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("explicit deck_hash is not a compact 12-character ID")
        return normalized
    matches: tuple[str, ...] = tuple(
        dict.fromkeys(_DECK_HASH_PATTERN.findall(label.lower()))
    )
    if len(matches) != 1:
        raise ValueError(
            "authoritative deck label must contain exactly one compact deck ID"
        )
    return matches[0]


__all__ = ["authoritative_deck_hash"]
