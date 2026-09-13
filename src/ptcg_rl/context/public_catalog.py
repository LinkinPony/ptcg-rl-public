"""Deterministic public tracker adapter for complete catalog summaries."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ptcg_rl.belief.observation import extract_observation_evidence
from ptcg_rl.belief.public_catalog import PublicDeckCatalog, PublicDeckPosterior
from ptcg_rl.context.belief import opponent_belief_state_from_evidence
from ptcg_rl.context.game import GameContext, GameContextFeatures


@dataclass(frozen=True)
class PublicCatalogContext:
    """Public context features and their raw catalog posterior summary."""

    features: GameContextFeatures
    posterior: PublicDeckPosterior
    catalog_fingerprint: str


class PublicCatalogTracker:
    """Update catalog posterior only when deduplicated public evidence changes."""

    def __init__(
        self,
        catalog: PublicDeckCatalog,
        *,
        context: GameContext | None = None,
    ) -> None:
        """Bind one immutable catalog to one per-game public tracker."""
        self.catalog = catalog
        self.context = context or GameContext()
        self._cached_known: tuple[tuple[int, int], ...] | None = None
        self._cached_posterior: PublicDeckPosterior | None = None

    def begin_game(
        self,
        *,
        player_index: int | None = None,
        own_deck: tuple[int, ...] | None = None,
    ) -> None:
        """Reset public evidence and posterior reuse state."""
        self.context.reset(player_index=player_index, own_deck=own_deck)
        self._cached_known = None
        self._cached_posterior = None

    @property
    def known_opponent_counts(self) -> tuple[tuple[int, int], ...]:
        """Return deduplicated public evidence from the latest update."""
        return self._cached_known or ()

    def update(self, observation: Any) -> PublicCatalogContext:
        """Return deterministic public features and a full raw catalog summary."""
        features = self.context.update(observation)
        evidence = extract_observation_evidence(observation)
        belief_state = opponent_belief_state_from_evidence(
            evidence,
            features,
            context_snapshot=self.context.snapshot(),
        )
        known = belief_state.known_counts()
        identity = tuple(sorted((int(card_id), int(count)) for card_id, count in known.items()))
        if identity != self._cached_known or self._cached_posterior is None:
            self._cached_known = identity
            self._cached_posterior = self.catalog.posterior(Counter(dict(identity)))
        return PublicCatalogContext(
            features=features,
            posterior=self._cached_posterior,
            catalog_fingerprint=self.catalog.fingerprint,
        )
