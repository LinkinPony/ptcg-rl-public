"""Belief-state determinization samplers."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.belief.card_rules import STANDARD_DECK_SIZE, CardCatalog
from ptcg_rl.belief.identity import canonical_belief_fingerprint, file_sha256
from ptcg_rl.belief.observation import (
    ObservationEvidence,
    extract_observation_evidence,
)
from ptcg_rl.belief.prior import ArchetypePosterior, ArchetypePrior
from ptcg_rl.belief.state import Determinization, OpponentBeliefState
from ptcg_rl.belief.zones import (
    complete_rule_consistent_deck_counts,
    sample_opponent_hidden_zones,
    sample_your_hidden_zones,
    validate_hidden_information,
)
from ptcg_rl.engine.protocols import ObservationInput
from ptcg_rl.engine.session import HiddenInformation

BeliefMode = Literal["placeholder", "rule", "archetype", "model"]


class BeliefSamplerConfig(BaseModel):
    """Config for hidden-zone determinization sampling."""

    model_config = ConfigDict(extra="forbid")

    mode: BeliefMode = "archetype"
    placeholder_opponent_deck_card_id: int = 1072
    placeholder_opponent_hand_card_id: int = 1
    placeholder_opponent_prize_card_id: int = 1
    strict_own_deck_counts: bool = False
    prior_deck_signature_summary_path: Path | None = None
    prior_deck_signature_summary_sha256: str | None = None
    prior_top_n: int | None = 200
    prior_min_games: int = 1
    prior_match_bonus: float = 0.35

    @field_validator(
        "placeholder_opponent_deck_card_id",
        "placeholder_opponent_hand_card_id",
        "placeholder_opponent_prize_card_id",
    )
    @classmethod
    def valid_card_id(cls, value: int) -> int:
        """Reject non-positive placeholder IDs."""
        if value <= 0:
            raise ValueError("placeholder card IDs must be positive")
        return value

    @field_validator("prior_top_n")
    @classmethod
    def valid_prior_top_n(cls, value: int | None) -> int | None:
        """Reject non-positive prior top-N values."""
        if value is not None and value <= 0:
            raise ValueError("prior_top_n must be positive when set")
        return value

    @field_validator("prior_min_games")
    @classmethod
    def valid_prior_min_games(cls, value: int) -> int:
        """Reject negative prior game thresholds."""
        if value < 0:
            raise ValueError("prior_min_games must be non-negative")
        return value

    @field_validator("prior_match_bonus")
    @classmethod
    def valid_prior_match_bonus(cls, value: float) -> float:
        """Reject negative match bonuses."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("prior_match_bonus must be finite and non-negative")
        return value

    @field_validator("prior_deck_signature_summary_sha256")
    @classmethod
    def valid_prior_sha256(cls, value: str | None) -> str | None:
        """Normalize an optional immutable prior fingerprint."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("prior summary SHA-256 must contain 64 hex digits")
        return normalized


class BeliefSampler:
    """Sample Search API hidden-zone inputs from public evidence."""

    def __init__(
        self,
        *,
        catalog: CardCatalog | None = None,
        prior: ArchetypePrior | None = None,
        config: BeliefSamplerConfig | None = None,
    ) -> None:
        """Create a sampler with optional card catalog and meta prior."""
        self.config = config or BeliefSamplerConfig()
        self.catalog = catalog or CardCatalog.from_engine()
        configured_prior = self._load_prior_from_config()
        if (
            prior is not None
            and configured_prior is not None
            and prior.fingerprint != configured_prior.fingerprint
        ):
            raise ValueError("supplied belief prior differs from configured content")
        self.prior = prior if prior is not None else configured_prior
        prior_path = self.config.prior_deck_signature_summary_path
        self._prior_source_fingerprint = (
            file_sha256(prior_path)
            if prior_path is not None
            else (
                "none"
                if self.prior is None
                else self.prior.fingerprint
            )
        )

    @property
    def semantic_fingerprint(self) -> str:
        """Bind sampler config, source bytes, parsed prior, and all card rules."""
        return canonical_belief_fingerprint(
            b"ptcg-rl/belief-sampler-semantics/v1\x00",
            {
                "config": self.config.model_dump(
                    mode="json",
                    exclude={
                        "prior_deck_signature_summary_path",
                        "prior_deck_signature_summary_sha256",
                    },
                ),
                "prior_source_fingerprint": self._prior_source_fingerprint,
                "prior_fingerprint": (
                    "none" if self.prior is None else self.prior.fingerprint
                ),
                "card_catalog_fingerprint": self.catalog.fingerprint,
            },
        )

    def sample(
        self,
        observation: ObservationInput,
        *,
        your_deck: Sequence[int],
        opponent_state: OpponentBeliefState | None = None,
        opponent_card_probs: Sequence[float] | None = None,
        opponent_hand_weights: Mapping[int, float] | Sequence[float] | None = None,
        rng: random.Random | None = None,
    ) -> Determinization:
        """Sample one determinization for ``observation``."""
        evidence = extract_observation_evidence(observation)
        state = opponent_state or OpponentBeliefState.from_evidence(evidence)
        if opponent_state is not None:
            state.update(evidence)
        return self.sample_from_evidence(
            evidence,
            your_deck=your_deck,
            opponent_state=state,
            opponent_card_probs=opponent_card_probs,
            opponent_hand_weights=opponent_hand_weights,
            rng=rng,
        )

    def sample_from_evidence(
        self,
        evidence: ObservationEvidence,
        *,
        your_deck: Sequence[int],
        opponent_state: OpponentBeliefState,
        opponent_card_probs: Sequence[float] | None = None,
        opponent_hand_weights: Mapping[int, float] | Sequence[float] | None = None,
        rng: random.Random | None = None,
    ) -> Determinization:
        """Sample one determinization from pre-extracted evidence."""
        active_rng = rng or random.Random()
        if self.config.mode == "placeholder":
            determinization = self._sample_placeholder(
                evidence,
                your_deck=your_deck,
                rng=active_rng,
            )
            validate_hidden_information(evidence, determinization.hidden, self.catalog)
            return determinization

        your_hidden_deck, your_prize = sample_your_hidden_zones(
            evidence,
            your_deck=your_deck,
            rng=active_rng,
            strict_counts=self.config.strict_own_deck_counts,
        )
        opponent_counts, source, signature, label = self._sample_opponent_deck_counts(
            opponent_state.known_counts(),
            opponent_card_probs=opponent_card_probs,
            rng=active_rng,
        )
        opponent_deck, opponent_prize, opponent_hand, opponent_active = (
            sample_opponent_hidden_zones(
                evidence,
                opponent_deck_counts=opponent_counts,
                catalog=self.catalog,
                rng=active_rng,
                opponent_hand_weights=opponent_hand_weights,
            )
        )
        hidden = HiddenInformation.from_sequences(
            your_deck=your_hidden_deck,
            your_prize=your_prize,
            opponent_deck=opponent_deck,
            opponent_prize=opponent_prize,
            opponent_hand=opponent_hand,
            opponent_active=opponent_active,
        )
        validate_hidden_information(evidence, hidden, self.catalog)
        return Determinization(
            hidden=hidden,
            source=source,
            opponent_deck_counts=opponent_counts,
            archetype_signature=signature,
            archetype_label=label,
        )

    def posterior(self, opponent_state: OpponentBeliefState) -> ArchetypePosterior | None:
        """Return the current archetype posterior when a prior is configured."""
        if self.prior is None:
            return None
        return self.prior.posterior(opponent_state.known_counts())

    def _sample_placeholder(
        self,
        evidence: ObservationEvidence,
        *,
        your_deck: Sequence[int],
        rng: random.Random,
    ) -> Determinization:
        your_hidden_deck, your_prize = sample_your_hidden_zones(
            evidence,
            your_deck=your_deck,
            rng=rng,
            strict_counts=False,
        )
        active = (
            (self.catalog.default_basic_pokemon_id,)
            if evidence.opponent_active_facedown
            else ()
        )
        hidden = HiddenInformation.from_sequences(
            your_deck=your_hidden_deck,
            your_prize=your_prize,
            opponent_deck=(
                [self.config.placeholder_opponent_deck_card_id]
                * evidence.opponent_deck_count
            ),
            opponent_prize=(
                [self.config.placeholder_opponent_prize_card_id]
                * evidence.opponent_prize_count
            ),
            opponent_hand=(
                [self.config.placeholder_opponent_hand_card_id]
                * evidence.opponent_hand_count
            ),
            opponent_active=active,
        )
        return Determinization(
            hidden=hidden,
            source="placeholder",
            opponent_deck_counts=Counter(hidden.opponent_deck)
            + Counter(hidden.opponent_prize)
            + Counter(hidden.opponent_hand)
            + Counter(hidden.opponent_active),
        )

    def _sample_opponent_deck_counts(
        self,
        known_counts: Counter[int],
        *,
        opponent_card_probs: Sequence[float] | None,
        rng: random.Random,
    ) -> tuple[Counter[int], str, str | None, str | None]:
        bounded_counts = _bounded_rule_counts(known_counts, self.catalog)
        if (
            self.config.mode == "model"
            and self.prior is not None
            and opponent_card_probs is not None
        ):
            posterior = self.prior.model_posterior(known_counts, opponent_card_probs)
            deck = posterior.sample(rng)
            if deck is not None:
                return Counter(deck.counts), "model", deck.signature, deck.label
            if bounded_counts != known_counts:
                posterior = self.prior.model_posterior(
                    bounded_counts,
                    opponent_card_probs,
                )
                deck = posterior.sample(rng)
                if deck is not None:
                    return Counter(deck.counts), "model", deck.signature, deck.label
        if self.config.mode == "archetype" and self.prior is not None:
            posterior = self.prior.posterior(known_counts)
            deck = posterior.sample(rng)
            if deck is not None:
                return Counter(deck.counts), "archetype", deck.signature, deck.label
            if bounded_counts != known_counts:
                posterior = self.prior.posterior(bounded_counts)
                deck = posterior.sample(rng)
                if deck is not None:
                    return Counter(deck.counts), "archetype", deck.signature, deck.label
        if self.config.mode == "model" and self.prior is not None:
            posterior = self.prior.posterior(known_counts)
            deck = posterior.sample(rng)
            if deck is not None:
                return Counter(deck.counts), "archetype", deck.signature, deck.label
            if bounded_counts != known_counts:
                posterior = self.prior.posterior(bounded_counts)
                deck = posterior.sample(rng)
                if deck is not None:
                    return Counter(deck.counts), "archetype", deck.signature, deck.label
        return (
            complete_rule_consistent_deck_counts(
                bounded_counts,
                catalog=self.catalog,
                rng=rng,
            ),
            "rule",
            None,
            None,
        )

    def _load_prior_from_config(self) -> ArchetypePrior | None:
        path = self.config.prior_deck_signature_summary_path
        if path is None:
            if self.config.prior_deck_signature_summary_sha256 is not None:
                raise ValueError("a prior summary fingerprint requires a prior path")
            return None
        expected_sha256 = self.config.prior_deck_signature_summary_sha256
        if expected_sha256 is not None:
            actual_sha256 = file_sha256(path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "belief prior summary SHA-256 mismatch: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
        return ArchetypePrior.from_deck_signature_summary(
            path,
            top_n=self.config.prior_top_n,
            min_games=self.config.prior_min_games,
            match_bonus=self.config.prior_match_bonus,
        )


def _bounded_rule_counts(
    counts: Counter[int],
    catalog: CardCatalog,
    *,
    deck_size: int = STANDARD_DECK_SIZE,
) -> Counter[int]:
    """Trim noisy lower bounds to the nearest deck-rule-consistent multiset."""
    bounded: Counter[int] = Counter()
    total = 0
    for card_id in _bounded_count_order(counts, catalog):
        for _ in range(max(0, int(counts[card_id]))):
            if total >= deck_size:
                return bounded
            if catalog.can_add_to_deck(bounded, card_id):
                bounded[card_id] += 1
                total += 1
    return bounded


def _bounded_count_order(
    counts: Counter[int],
    catalog: CardCatalog,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            (int(card_id) for card_id, count in counts.items() if count > 0),
            key=lambda card_id: (catalog.is_basic_energy(card_id), card_id),
        )
    )


class OpponentBeliefTracker:
    """Stateful P3 online updater for opponent card-composition evidence."""

    def __init__(self) -> None:
        """Create an empty online belief tracker."""
        self.state = OpponentBeliefState()

    def update(self, observation: ObservationInput) -> ObservationEvidence:
        """Update revealed-card lower bounds from a new observation."""
        evidence = extract_observation_evidence(observation)
        self.state.update(evidence)
        return evidence

    def sample(
        self,
        observation: ObservationInput,
        *,
        sampler: BeliefSampler,
        your_deck: Sequence[int],
        rng: random.Random | None = None,
    ) -> Determinization:
        """Update from ``observation`` and sample a determinization."""
        evidence = self.update(observation)
        return sampler.sample_from_evidence(
            evidence,
            your_deck=your_deck,
            opponent_state=self.state,
            rng=rng,
        )
