"""Opponent belief features derived from game-context evidence."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.belief.identity import canonical_belief_fingerprint, file_sha256
from ptcg_rl.belief.observation import (
    ObservationEvidence,
    extract_observation_evidence,
)
from ptcg_rl.belief.prior import ArchetypePrior
from ptcg_rl.belief.state import OpponentBeliefState
from ptcg_rl.context.game import (
    ExpectedCardCount,
    GameContextFeatures,
    GameContextSnapshot,
)

DEFAULT_BELIEF_TOP_K = 16


class OpponentBeliefFeatureConfig(BaseModel):
    """Config for zero-training archetype-posterior belief features."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    top_k: int = DEFAULT_BELIEF_TOP_K
    deck_signature_summary_path: Path | None = None
    deck_signature_summary_sha256: str | None = None
    prior_top_n: int | None = 200
    prior_min_games: int = 1
    prior_match_bonus: float = 0.35

    @field_validator("top_k")
    @classmethod
    def valid_top_k(cls, value: int) -> int:
        """Reject non-positive token limits."""
        if value <= 0:
            raise ValueError("top_k must be positive")
        return value

    @field_validator("prior_top_n")
    @classmethod
    def valid_prior_top_n(cls, value: int | None) -> int | None:
        """Reject non-positive prior limits."""
        if value is not None and value <= 0:
            raise ValueError("prior_top_n must be positive when set")
        return value

    @field_validator("prior_min_games")
    @classmethod
    def valid_prior_min_games(cls, value: int) -> int:
        """Reject negative support thresholds."""
        if value < 0:
            raise ValueError("prior_min_games must be non-negative")
        return value

    @field_validator("prior_match_bonus")
    @classmethod
    def valid_prior_match_bonus(cls, value: float) -> float:
        """Reject negative evidence bonuses."""
        if value < 0.0:
            raise ValueError("prior_match_bonus must be non-negative")
        return value

    @field_validator("deck_signature_summary_sha256")
    @classmethod
    def valid_summary_sha256(cls, value: str | None) -> str | None:
        """Normalize an optional immutable prior fingerprint."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("belief summary SHA-256 must contain 64 hex digits")
        return normalized


class OpponentBeliefFeatureProducer:
    """Build top-k expected opponent remaining-card features from a prior."""

    def __init__(
        self,
        *,
        prior: ArchetypePrior | None,
        top_k: int = DEFAULT_BELIEF_TOP_K,
        enabled: bool = True,
        config_payload: Mapping[str, Any] | None = None,
        prior_source_fingerprint: str | None = None,
    ) -> None:
        """Store a prior and token budget."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.prior = prior
        self.top_k = int(top_k)
        self.enabled = bool(enabled)
        self._config_payload = (
            dict(config_payload)
            if config_payload is not None
            else {"enabled": self.enabled, "top_k": self.top_k}
        )
        self._prior_source_fingerprint = (
            prior_source_fingerprint
            if prior_source_fingerprint is not None
            else ("none" if prior is None else prior.fingerprint)
        )

    @property
    def producer_fingerprint(self) -> str:
        """Return the immutable public-posterior continuation semantics."""
        return canonical_belief_fingerprint(
            b"ptcg-rl/public-belief-feature-producer/v1\x00",
            {
                "config": self._config_payload,
                "prior_source_fingerprint": self._prior_source_fingerprint,
                "prior_fingerprint": (
                    "none" if self.prior is None else self.prior.fingerprint
                ),
            },
        )

    @classmethod
    def from_config(
        cls,
        config: OpponentBeliefFeatureConfig,
    ) -> OpponentBeliefFeatureProducer:
        """Load the configured archetype prior when present."""
        prior = None
        if config.enabled and config.deck_signature_summary_path is not None:
            expected_sha256 = config.deck_signature_summary_sha256
            if expected_sha256 is not None:
                actual_sha256 = file_sha256(config.deck_signature_summary_path)
                if actual_sha256 != expected_sha256:
                    raise ValueError(
                        "belief feature summary SHA-256 mismatch: "
                        f"expected {expected_sha256}, got {actual_sha256}"
                    )
            prior = ArchetypePrior.from_deck_signature_summary(
                config.deck_signature_summary_path,
                top_n=config.prior_top_n,
                min_games=config.prior_min_games,
                match_bonus=config.prior_match_bonus,
            )
        path = config.deck_signature_summary_path
        return cls(
            prior=prior,
            top_k=config.top_k,
            enabled=config.enabled,
            config_payload=config.model_dump(
                mode="json",
                exclude={
                    "deck_signature_summary_path",
                    "deck_signature_summary_sha256",
                },
            ),
            prior_source_fingerprint=(
                "none" if path is None else file_sha256(path)
            ),
        )

    def augment(
        self,
        observation: Any,
        context_features: GameContextFeatures,
    ) -> GameContextFeatures:
        """Return ``context_features`` plus belief posterior summary fields."""
        belief, entropy, is_empty = self.features(observation, context_features)
        return replace(
            context_features,
            opponent_belief=belief,
            opponent_belief_entropy=entropy,
            opponent_belief_empty=is_empty,
        )

    def features(
        self,
        observation: Any,
        context_features: GameContextFeatures,
    ) -> tuple[tuple[ExpectedCardCount, ...], float, bool]:
        """Return top-k expected remaining cards, normalized entropy, empty flag."""
        known_counts = opponent_known_counts_from_context(
            observation,
            context_features,
        )
        return self.features_from_known_counts(known_counts)

    def features_from_known_counts(
        self,
        known_counts: Mapping[int, int],
    ) -> tuple[tuple[ExpectedCardCount, ...], float, bool]:
        """Return legacy belief features from an exact public known multiset.

        This is the observation-free compatibility boundary used by the native
        rollout path.  Keeping the posterior and top-k calculation here makes
        the mapping-based and columnar paths share one implementation.
        """
        if not self.enabled or self.prior is None:
            return (), 0.0, True
        normalized_known = Counter(
            {
                int(card_id): int(count)
                for card_id, count in known_counts.items()
                if int(card_id) > 0 and int(count) > 0
            }
        )
        posterior = self.prior.posterior(normalized_known)
        if posterior.is_empty:
            return (), 0.0, True

        expected: dict[int, float] = {}
        for entry in posterior.entries:
            for card_id, count in entry.deck.counts.items():
                key = int(card_id)
                expected[key] = expected.get(key, 0.0) + (
                    float(entry.probability) * float(count)
                )
        for card_id, count in normalized_known.items():
            key = int(card_id)
            expected[key] = expected.get(key, 0.0) - float(count)

        positive = [
            ExpectedCardCount(card_id=card_id, expected_count=count)
            for card_id, count in expected.items()
            if card_id > 0 and count > 1.0e-6
        ]
        positive.sort(key=lambda item: (-item.expected_count, item.card_id))
        return (
            tuple(positive[: self.top_k]),
            _normalized_entropy([entry.probability for entry in posterior.entries]),
            False,
        )


def opponent_known_counts_from_context(
    observation: Any,
    context_features: GameContextFeatures,
) -> Counter[int]:
    """Return opponent deck-composition evidence visible to the current player."""
    counts: Counter[int] = Counter()
    current = _field(observation, "current")
    your_index = _int_field(current, "yourIndex", 0)
    opponent_index = 1 - your_index if your_index in (0, 1) else -1
    if opponent_index in (0, 1):
        counts.update(_visible_player_counts(current, opponent_index))
    counts.update(
        {
            item.card_id: item.count
            for item in context_features.opponent_revealed
            if item.card_id > 0 and item.count > 0
        }
    )
    return Counter({card_id: count for card_id, count in counts.items() if count > 0})


def opponent_belief_state_from_context(
    observation: Any,
    context_features: GameContextFeatures,
) -> OpponentBeliefState:
    """Build sampler state from current observation plus accumulated context."""
    evidence = extract_observation_evidence(observation)
    return opponent_belief_state_from_evidence(evidence, context_features)


def opponent_belief_state_from_evidence(
    evidence: ObservationEvidence,
    context_features: GameContextFeatures,
    *,
    context_snapshot: GameContextSnapshot | None = None,
) -> OpponentBeliefState:
    """Build one union of current and accumulated public card evidence.

    ``opponent_revealed`` already contains historical evidence represented by
    the current log batch. Adding it to the freshly extracted batch would count
    a returned-to-hidden card twice. Preserve serial identities when a tracker
    snapshot is available, then fill only the per-card residual needed to match
    the strongest observable lower bound.
    """
    serials = dict(evidence.opponent_revealed_by_serial)
    no_serial = Counter(evidence.opponent_revealed_no_serial_counts)
    if context_snapshot is not None:
        for serial, card_id in context_snapshot.opponent_revealed_by_serial:
            serials.setdefault(int(serial), int(card_id))
        for card_id, count in context_snapshot.opponent_revealed_no_serial_counts:
            no_serial[int(card_id)] = max(no_serial[int(card_id)], int(count))

    state = OpponentBeliefState(
        revealed_by_serial=serials,
        revealed_no_serial_counts=no_serial,
    )
    context_known = Counter(evidence.opponent_current_visible_counts)
    context_known.update(
        {
            item.card_id: item.count
            for item in context_features.opponent_revealed
            if item.card_id > 0 and item.count > 0
        }
    )
    evidence_known = evidence.opponent_revealed_counts
    target_known = Counter(
        {
            card_id: max(context_known[card_id], evidence_known[card_id])
            for card_id in context_known.keys() | evidence_known.keys()
        }
    )
    if context_snapshot is not None:
        snapshot_known = Counter(
            int(card_id)
            for _serial, card_id in context_snapshot.opponent_revealed_by_serial
        )
        snapshot_known.update(
            {
                int(card_id): int(count)
                for card_id, count in (
                    context_snapshot.opponent_revealed_no_serial_counts
                )
            }
        )
        for card_id, count in snapshot_known.items():
            target_known[card_id] = max(target_known[card_id], count)

    known = state.known_counts()
    for card_id, count in target_known.items():
        residual = int(count) - int(known[card_id])
        if residual > 0:
            state.revealed_no_serial_counts[int(card_id)] += residual
    return state


def _visible_player_counts(current: Any, player_index: int) -> Counter[int]:
    players = _sequence(_field(current, "players", ()))
    if player_index < 0 or player_index >= len(players):
        return Counter()
    player = players[player_index]
    counts: Counter[int] = Counter()
    for pokemon in _sequence(_field(player, "active", ())):
        if pokemon is not None:
            _add_pokemon_counts(pokemon, counts)
    for pokemon in _sequence(_field(player, "bench", ())):
        _add_pokemon_counts(pokemon, counts)
    for field_name in ("discard", "prize", "hand"):
        for card in _sequence(_field(player, field_name, ())):
            if card is not None:
                _add_card_count(card, counts)
    for card in _sequence(_field(current, "stadium", ())):
        if _int_field(card, "playerIndex", -1) == player_index:
            _add_card_count(card, counts)
    for card in _sequence(_field(current, "looking", ())):
        if card is not None and _int_field(card, "playerIndex", -1) == player_index:
            _add_card_count(card, counts)
    return counts


def _add_pokemon_counts(pokemon: Any, counts: Counter[int]) -> None:
    _add_card_count(pokemon, counts)
    for field_name in ("energyCards", "tools", "preEvolution"):
        for card in _sequence(_field(pokemon, field_name, ())):
            _add_card_count(card, counts)


def _add_card_count(card: Any, counts: Counter[int]) -> None:
    card_id = _int_field(card, "id", 0)
    if card_id > 0:
        counts[card_id] += 1


def _normalized_entropy(probabilities: Sequence[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    return min(1.0, max(0.0, entropy / math.log(float(len(probabilities)))))


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_field(value: Any, name: str, default: int) -> int:
    field_value = _field(value, name, default)
    return int(field_value) if field_value is not None else default
