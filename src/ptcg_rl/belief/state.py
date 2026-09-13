"""Belief-state data structures."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ptcg_rl.belief.observation import ObservationEvidence
from ptcg_rl.engine.session import HiddenInformation


@dataclass
class OpponentBeliefState:
    """Online lower-bound state for opponent cards revealed across turns."""

    revealed_by_serial: dict[int, int] = field(default_factory=dict)
    revealed_no_serial_counts: Counter[int] = field(default_factory=Counter)

    @classmethod
    def from_evidence(cls, evidence: ObservationEvidence) -> OpponentBeliefState:
        """Initialize belief state from the current observation evidence."""
        state = cls()
        state.update(evidence)
        return state

    def update(self, evidence: ObservationEvidence) -> None:
        """Add newly visible or logged opponent card revelations."""
        for serial, card_id in evidence.opponent_revealed_by_serial.items():
            self.revealed_by_serial.setdefault(serial, card_id)
        self.revealed_no_serial_counts.update(evidence.opponent_revealed_no_serial_counts)

    def known_counts(self) -> Counter[int]:
        """Return the opponent deck-composition lower-bound multiset."""
        counts: Counter[int] = Counter(self.revealed_by_serial.values())
        counts.update(self.revealed_no_serial_counts)
        return counts


@dataclass
class Determinization:
    """One complete assignment of hidden zones for Search API input."""

    hidden: HiddenInformation
    source: str
    opponent_deck_counts: Counter[int] = field(default_factory=Counter)
    archetype_signature: str | None = None
    archetype_label: str | None = None
