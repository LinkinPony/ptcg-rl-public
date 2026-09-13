"""Meta archetype priors built from mined Kaggle deck reports."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.belief.identity import canonical_belief_fingerprint
from ptcg_rl.data.kaggle_deck.records import signature_counts


class ArchetypePriorConfig(BaseModel):
    """Config for loading an archetype prior from deck report CSVs."""

    model_config = ConfigDict(extra="forbid")

    deck_signature_summary_path: Path
    top_n: int | None = 200
    min_games: int = 1
    match_bonus: float = 0.35

    @field_validator("top_n")
    @classmethod
    def valid_top_n(cls, value: int | None) -> int | None:
        """Reject non-positive top-N values."""
        if value is not None and value <= 0:
            raise ValueError("top_n must be positive when set")
        return value

    @field_validator("min_games")
    @classmethod
    def valid_min_games(cls, value: int) -> int:
        """Reject negative game thresholds."""
        if value < 0:
            raise ValueError("min_games must be non-negative")
        return value

    @field_validator("match_bonus")
    @classmethod
    def valid_match_bonus(cls, value: float) -> float:
        """Reject negative match bonuses."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("match_bonus must be finite and non-negative")
        return value


@dataclass
class ArchetypeDeck:
    """One exact deck signature used as a meta prior atom."""

    signature: str
    label: str
    games: int
    win_rate: float
    counts: Counter[int]


@dataclass(frozen=True)
class PosteriorEntry:
    """Posterior probability for one archetype deck."""

    deck: ArchetypeDeck
    probability: float


@dataclass(frozen=True)
class ArchetypePosterior:
    """Normalized archetype distribution after applying seen-card evidence."""

    entries: tuple[PosteriorEntry, ...]

    @property
    def is_empty(self) -> bool:
        """Return whether no archetype can cover the evidence."""
        return not self.entries

    def sample(self, rng: random.Random) -> ArchetypeDeck | None:
        """Sample one archetype deck according to posterior probability."""
        if not self.entries:
            return None
        threshold = rng.random()
        cumulative = 0.0
        for entry in self.entries:
            cumulative += entry.probability
            if threshold <= cumulative:
                return entry.deck
        return self.entries[-1].deck

    def as_dict(self) -> dict[str, float]:
        """Return ``{signature: probability}`` for network features or logging."""
        return {entry.deck.signature: entry.probability for entry in self.entries}


class ArchetypePrior:
    """Collection of exact meta decks with evidence-conditioned sampling."""

    def __init__(self, decks: tuple[ArchetypeDeck, ...], *, match_bonus: float) -> None:
        """Store prior decks sorted by empirical support."""
        if not math.isfinite(match_bonus) or match_bonus < 0.0:
            raise ValueError("match_bonus must be finite and non-negative")
        self.decks = decks
        self.match_bonus = match_bonus

    @property
    def fingerprint(self) -> str:
        """Return the ordered parsed prior that controls posterior sampling."""
        return canonical_belief_fingerprint(
            b"ptcg-rl/archetype-prior-semantics/v1\x00",
            {
                "match_bonus": self.match_bonus,
                "decks": [
                    {
                        "signature": deck.signature,
                        "label": deck.label,
                        "games": deck.games,
                        "win_rate": deck.win_rate,
                        "counts": sorted(
                            (int(card_id), int(count))
                            for card_id, count in deck.counts.items()
                        ),
                    }
                    for deck in self.decks
                ],
            },
        )

    @classmethod
    def from_config(cls, config: ArchetypePriorConfig) -> ArchetypePrior:
        """Load a prior from a validated config."""
        return cls.from_deck_signature_summary(
            config.deck_signature_summary_path,
            top_n=config.top_n,
            min_games=config.min_games,
            match_bonus=config.match_bonus,
        )

    @classmethod
    def from_deck_signature_summary(
        cls,
        path: Path,
        *,
        top_n: int | None = 200,
        min_games: int = 1,
        match_bonus: float = 0.35,
    ) -> ArchetypePrior:
        """Load archetype atoms from ``deck_signature_summary.csv``."""
        decks: list[ArchetypeDeck] = []
        with path.open(encoding="utf-8", newline="") as file_obj:
            for row in csv.DictReader(file_obj):
                games = _int_field(row, "games")
                if games < min_games:
                    continue
                signature = row.get("deck_signature", "")
                if not signature:
                    continue
                decks.append(
                    ArchetypeDeck(
                        signature=signature,
                        label=row.get("deck_label", ""),
                        games=games,
                        win_rate=_float_field(row, "win_rate"),
                        counts=signature_counts(signature),
                    )
                )
        decks.sort(key=lambda deck: (deck.games, deck.win_rate), reverse=True)
        if top_n is not None:
            decks = decks[:top_n]
        return cls(tuple(decks), match_bonus=match_bonus)

    def posterior(self, known_counts: Counter[int]) -> ArchetypePosterior:
        """Return ``P(deck | known_counts)`` over exact meta decks.

        Covering the visible multiset is a hard consistency condition. Among
        covering decks, use the multivariate-hypergeometric numerator as the
        evidence likelihood: seeing a card is more likely when an exact list
        contains more copies of it. The previous matched-card bonus depended
        only on ``known_counts`` and therefore cancelled during normalization.
        """
        scored: list[tuple[ArchetypeDeck, float]] = []
        for deck in self.decks:
            if not _covers(deck.counts, known_counts):
                continue
            log_weight = math.log(max(float(deck.games), 1.0))
            log_weight += self.match_bonus * _known_multiset_log_likelihood(
                deck.counts,
                known_counts,
            )
            scored.append((deck, log_weight))
        if not scored:
            return ArchetypePosterior(())
        max_log_weight = max(log_weight for _, log_weight in scored)
        weighted = [
            (deck, math.exp(log_weight - max_log_weight))
            for deck, log_weight in scored
        ]
        total = sum(weight for _, weight in weighted)
        if total <= 0.0:
            return ArchetypePosterior(())
        return ArchetypePosterior(
            tuple(
                PosteriorEntry(deck=deck, probability=weight / total)
                for deck, weight in weighted
            )
        )

    def model_posterior(
        self,
        known_counts: Counter[int],
        card_probabilities: Sequence[float],
    ) -> ArchetypePosterior:
        """Return a prior over exact decks reweighted by model card probabilities."""
        scored: list[tuple[ArchetypeDeck, float]] = []
        for deck in self.decks:
            if not _covers(deck.counts, known_counts):
                continue
            log_weight = math.log(max(float(deck.games), 1.0))
            for card_id, count in deck.counts.items():
                remaining = int(count) - int(known_counts.get(card_id, 0))
                if remaining <= 0:
                    continue
                log_weight += float(remaining) * math.log(
                    max(_probability_for_card(card_probabilities, card_id), 1.0e-12)
                )
            scored.append((deck, log_weight))
        if not scored:
            return ArchetypePosterior(())
        max_log_weight = max(log_weight for _, log_weight in scored)
        weighted = [
            (deck, math.exp(log_weight - max_log_weight))
            for deck, log_weight in scored
        ]
        total = sum(weight for _, weight in weighted)
        if total <= 0.0:
            return ArchetypePosterior(())
        return ArchetypePosterior(
            tuple(
                PosteriorEntry(deck=deck, probability=weight / total)
                for deck, weight in weighted
            )
        )


def _covers(deck_counts: Counter[int], known_counts: Counter[int]) -> bool:
    return all(deck_counts[card_id] >= count for card_id, count in known_counts.items())


def _known_multiset_log_likelihood(
    deck_counts: Counter[int],
    known_counts: Counter[int],
) -> float:
    """Return the deck-dependent log likelihood of the visible multiset."""
    return sum(
        math.log(math.comb(int(deck_counts[card_id]), int(known_count)))
        for card_id, known_count in known_counts.items()
        if known_count > 0
    )


def _probability_for_card(card_probabilities: Sequence[float], card_id: int) -> float:
    index = int(card_id) - 1
    if index < 0 or index >= len(card_probabilities):
        return 0.0
    return float(card_probabilities[index])


def _int_field(row: dict[str, str], field: str) -> int:
    value = row.get(field, "")
    return int(value) if value else 0


def _float_field(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value else 0.0
