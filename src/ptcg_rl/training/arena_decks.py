"""Deck-pool loading for local arena evaluation."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.data.kaggle_deck import records


class DeckPoolConfig(BaseModel):
    """Deck input source for one side of arena evaluation."""

    model_config = ConfigDict(extra="forbid")

    deck_paths: tuple[Path, ...] = ()
    gauntlet_path: Path | None = None
    gauntlet_top_n: int | None = None
    min_games: int = 1

    @field_validator("gauntlet_top_n")
    @classmethod
    def valid_gauntlet_top_n(cls, value: int | None) -> int | None:
        """Reject non-positive gauntlet limits."""
        if value is not None and value <= 0:
            raise ValueError("gauntlet_top_n must be positive when set")
        return value

    @field_validator("min_games")
    @classmethod
    def valid_min_games(cls, value: int) -> int:
        """Reject negative game filters."""
        if value < 0:
            raise ValueError("min_games must be non-negative")
        return value

    @model_validator(mode="after")
    def has_source(self) -> DeckPoolConfig:
        """Require at least one concrete deck source."""
        if not self.deck_paths and self.gauntlet_path is None:
            raise ValueError("deck_paths or gauntlet_path must be set")
        return self


@dataclass(frozen=True)
class ArenaDeck:
    """One 60-card deck available to the arena."""

    deck_id: str
    label: str
    signature: str
    deck_hash: str
    cards: tuple[int, ...]
    source: str


def load_deck_pool(config: DeckPoolConfig) -> tuple[ArenaDeck, ...]:
    """Load and de-duplicate decks from CSV paths and gauntlet signatures."""
    decks: list[ArenaDeck] = []
    for path in config.deck_paths:
        decks.append(_deck_from_path(path))
    if config.gauntlet_path is not None:
        decks.extend(_decks_from_gauntlet(config))
    return _dedupe_decks(decks)


def _deck_from_path(path: Path) -> ArenaDeck:
    resolved = records.repo_path(path)
    cards = tuple(records.read_deck(resolved))
    signature = records.deck_signature(list(cards))
    label = resolved.parent.name if resolved.name == "deck.csv" else resolved.stem
    deck_hash = records.signature_hash(signature)
    return ArenaDeck(
        deck_id=f"path:{records.display_path(resolved)}",
        label=label,
        signature=signature,
        deck_hash=deck_hash,
        cards=cards,
        source=records.display_path(resolved),
    )


def _decks_from_gauntlet(config: DeckPoolConfig) -> tuple[ArenaDeck, ...]:
    if config.gauntlet_path is None:
        return ()
    path = records.repo_path(config.gauntlet_path)
    decks: list[ArenaDeck] = []
    with path.open(encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            if len(decks) >= (config.gauntlet_top_n or math.inf):
                break
            games = _int_text(row.get("games", "0"))
            if games < config.min_games:
                continue
            signature = row.get("deck_signature", "")
            if not signature:
                continue
            cards = _deck_cards_from_signature(signature)
            deck_hash = row.get("deck_hash", "") or records.signature_hash(signature)
            rank = row.get("rank", str(len(decks) + 1))
            label = row.get("deck_label", "") or f"gauntlet_{rank}"
            decks.append(
                ArenaDeck(
                    deck_id=f"gauntlet:{deck_hash}",
                    label=label,
                    signature=signature,
                    deck_hash=deck_hash,
                    cards=cards,
                    source=records.display_path(path),
                )
            )
    return tuple(decks)


def _deck_cards_from_signature(signature: str) -> tuple[int, ...]:
    counts = records.signature_counts(signature)
    cards = tuple(
        card_id for card_id in sorted(counts) for _ in range(counts[card_id])
    )
    if len(cards) != 60:
        raise ValueError(f"deck signature must expand to 60 cards: {signature}")
    return cards


def _dedupe_decks(decks: Sequence[ArenaDeck]) -> tuple[ArenaDeck, ...]:
    output: list[ArenaDeck] = []
    seen: set[str] = set()
    for deck in decks:
        if deck.signature in seen:
            continue
        seen.add(deck.signature)
        output.append(deck)
    return tuple(output)


def _int_text(value: str | None) -> int:
    return int(value) if value else 0
