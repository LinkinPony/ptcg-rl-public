"""Card-pool rule helpers for hidden-information sampling."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ptcg_rl.belief.identity import canonical_belief_fingerprint
from ptcg_rl.cards.static_features import (
    BASIC_ENERGY_CARD_TYPE,
    POKEMON_CARD_TYPE,
    load_engine_card_data,
)

DEFAULT_CARD_ID_COUNT = 1_267
DEFAULT_BASIC_POKEMON_ID = 1072
DEFAULT_BASIC_ENERGY_ID = 1
STANDARD_DECK_SIZE = 60
NON_BASIC_ENERGY_COPY_LIMIT = 4
ACE_SPEC_LIMIT = 1


@dataclass(frozen=True)
class CardRule:
    """Rule-relevant metadata for one competition card ID."""

    card_id: int
    name: str
    is_pokemon: bool
    is_basic_pokemon: bool
    is_basic_energy: bool
    is_ace_spec: bool


class CardCatalog:
    """Card-pool predicates and deck legality checks used by belief samplers."""

    def __init__(self, rules: Iterable[CardRule]) -> None:
        """Create a catalog from rule records."""
        by_id = {rule.card_id: rule for rule in rules}
        if not by_id:
            raise ValueError("CardCatalog requires at least one card rule")
        self._by_id = by_id
        self._card_ids = tuple(sorted(by_id))
        self._pokemon_ids = tuple(
            card_id for card_id in self._card_ids if by_id[card_id].is_pokemon
        )
        self._basic_pokemon_ids = tuple(
            card_id for card_id in self._card_ids if by_id[card_id].is_basic_pokemon
        )
        self._basic_energy_ids = tuple(
            card_id for card_id in self._card_ids if by_id[card_id].is_basic_energy
        )

    @classmethod
    def from_engine(cls) -> CardCatalog:
        """Build rules from the bundled engine card metadata."""
        cards, _ = load_engine_card_data()
        return cls(
            CardRule(
                card_id=int(card.cardId),
                name=str(card.name),
                is_pokemon=int(card.cardType) == POKEMON_CARD_TYPE,
                is_basic_pokemon=bool(card.basic),
                is_basic_energy=int(card.cardType) == BASIC_ENERGY_CARD_TYPE,
                is_ace_spec=bool(card.aceSpec),
            )
            for card in cards
        )

    @classmethod
    def from_card_data_csv(cls, path: Path) -> CardCatalog:
        """Build rules from Kaggle's human-readable card metadata CSV."""
        rules: dict[int, CardRule] = {}
        with path.open(encoding="utf-8-sig", newline="") as file_obj:
            for row in csv.DictReader(file_obj):
                card_id = int(row["Card ID"])
                if card_id in rules:
                    continue
                stage = _stage_or_type(row)
                rule_text = row.get("Rule", "")
                rules[card_id] = CardRule(
                    card_id=card_id,
                    name=row["Card Name"],
                    is_pokemon="Pokémon" in stage or "Pokemon" in stage,
                    is_basic_pokemon=stage.startswith("Basic Pokémon")
                    or stage.startswith("Basic Pokemon"),
                    is_basic_energy=stage == "Basic Energy",
                    is_ace_spec="ACE SPEC" in rule_text.upper(),
                )
        return cls(rules.values())

    @classmethod
    def synthetic(cls, *, max_card_id: int = DEFAULT_CARD_ID_COUNT) -> CardCatalog:
        """Return a minimal import-safe catalog for placeholder fallback code."""
        rules = [
            CardRule(
                card_id=card_id,
                name=f"Card {card_id}",
                is_pokemon=card_id == DEFAULT_BASIC_POKEMON_ID,
                is_basic_pokemon=card_id == DEFAULT_BASIC_POKEMON_ID,
                is_basic_energy=card_id == DEFAULT_BASIC_ENERGY_ID,
                is_ace_spec=False,
            )
            for card_id in range(1, max_card_id + 1)
        ]
        return cls(rules)

    @property
    def card_ids(self) -> tuple[int, ...]:
        """Return all known card IDs sorted ascending."""
        return self._card_ids

    @property
    def fingerprint(self) -> str:
        """Return the exact identity of every rule used by determinization."""
        return canonical_belief_fingerprint(
            b"ptcg-rl/card-catalog-semantics/v1\x00",
            [
                {
                    "card_id": rule.card_id,
                    "name": rule.name,
                    "is_pokemon": rule.is_pokemon,
                    "is_basic_pokemon": rule.is_basic_pokemon,
                    "is_basic_energy": rule.is_basic_energy,
                    "is_ace_spec": rule.is_ace_spec,
                }
                for rule in (self._by_id[card_id] for card_id in self._card_ids)
            ],
        )

    @property
    def pokemon_ids(self) -> tuple[int, ...]:
        """Return all known Pokemon card IDs."""
        return self._pokemon_ids

    @property
    def basic_pokemon_ids(self) -> tuple[int, ...]:
        """Return all known Basic Pokemon card IDs."""
        return self._basic_pokemon_ids

    @property
    def basic_energy_ids(self) -> tuple[int, ...]:
        """Return all known Basic Energy card IDs."""
        return self._basic_energy_ids

    @property
    def default_basic_pokemon_id(self) -> int:
        """Return a stable Basic Pokemon fallback."""
        if DEFAULT_BASIC_POKEMON_ID in self._by_id:
            return DEFAULT_BASIC_POKEMON_ID
        if self._basic_pokemon_ids:
            return self._basic_pokemon_ids[0]
        if self._pokemon_ids:
            return self._pokemon_ids[0]
        return self._card_ids[0]

    @property
    def default_basic_energy_id(self) -> int:
        """Return a stable Basic Energy fallback."""
        if DEFAULT_BASIC_ENERGY_ID in self._by_id:
            return DEFAULT_BASIC_ENERGY_ID
        if self._basic_energy_ids:
            return self._basic_energy_ids[0]
        return self._card_ids[0]

    def rule(self, card_id: int) -> CardRule:
        """Return a card rule, or a conservative unknown-card rule."""
        found = self._by_id.get(card_id)
        if found is not None:
            return found
        return CardRule(
            card_id=card_id,
            name=f"Card {card_id}",
            is_pokemon=False,
            is_basic_pokemon=False,
            is_basic_energy=False,
            is_ace_spec=False,
        )

    def is_pokemon(self, card_id: int) -> bool:
        """Return whether ``card_id`` is a Pokemon."""
        return self.rule(card_id).is_pokemon

    def is_basic_pokemon(self, card_id: int) -> bool:
        """Return whether ``card_id`` is a Basic Pokemon."""
        return self.rule(card_id).is_basic_pokemon

    def is_basic_energy(self, card_id: int) -> bool:
        """Return whether ``card_id`` is a Basic Energy."""
        return self.rule(card_id).is_basic_energy

    def is_ace_spec(self, card_id: int) -> bool:
        """Return whether ``card_id`` is an ACE SPEC card."""
        return self.rule(card_id).is_ace_spec

    def has_basic_pokemon(self, card_ids: Sequence[int]) -> bool:
        """Return whether any card in ``card_ids`` is a Basic Pokemon."""
        return any(self.is_basic_pokemon(card_id) for card_id in card_ids)

    def can_add_to_deck(self, counts: Mapping[int, int], card_id: int) -> bool:
        """Return whether one more copy keeps copy-limit rules satisfiable."""
        rule = self.rule(card_id)
        if rule.is_basic_energy:
            return True
        if rule.is_ace_spec and self.ace_spec_count(counts) >= ACE_SPEC_LIMIT:
            return False
        return self.name_count(counts, rule.name) < NON_BASIC_ENERGY_COPY_LIMIT

    def name_count(self, counts: Mapping[int, int], name: str) -> int:
        """Return total copies across card IDs sharing ``name``."""
        total = 0
        for card_id, count in counts.items():
            if count > 0 and self.rule(card_id).name == name:
                total += count
        return total

    def ace_spec_count(self, counts: Mapping[int, int]) -> int:
        """Return total ACE SPEC copies in ``counts``."""
        return sum(
            count
            for card_id, count in counts.items()
            if count > 0 and self.is_ace_spec(card_id)
        )

    def validate_deck_counts(
        self,
        counts: Mapping[int, int],
        *,
        deck_size: int = STANDARD_DECK_SIZE,
    ) -> None:
        """Validate deck-size, copy-limit, ACE SPEC, and Basic Pokemon rules."""
        total = sum(counts.values())
        if total != deck_size:
            raise ValueError(f"deck must contain {deck_size} cards, got {total}")
        if self.ace_spec_count(counts) > ACE_SPEC_LIMIT:
            raise ValueError("deck contains more than one ACE SPEC card")
        if not any(
            count > 0 and self.is_basic_pokemon(card_id)
            for card_id, count in counts.items()
        ):
            raise ValueError("deck must contain at least one Basic Pokemon")

        by_name: Counter[str] = Counter()
        for card_id, count in counts.items():
            if count < 0:
                raise ValueError(f"negative card count for card_id={card_id}")
            rule = self.rule(card_id)
            if rule.is_basic_energy:
                continue
            by_name[rule.name] += count
        over_limit = [
            name
            for name, count in by_name.items()
            if count > NON_BASIC_ENERGY_COPY_LIMIT
        ]
        if over_limit:
            raise ValueError(f"deck exceeds copy limit for {over_limit[0]}")


def _stage_or_type(row: Mapping[str, str]) -> str:
    for key, value in row.items():
        if key.startswith("Stage ("):
            return value
    return ""
