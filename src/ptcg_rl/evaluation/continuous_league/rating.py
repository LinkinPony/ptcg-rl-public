"""Classic TrueSkill component updates and deterministic promotion estimates."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import trueskill

from ptcg_rl.evaluation.continuous_league.models import (
    ComponentRating,
    GameOutcome,
    PromotionConfig,
    RatingConfig,
)


@dataclass(frozen=True, slots=True)
class BundleRating:
    """Derived two-component bundle presentation values."""

    mu: float
    sigma: float
    conservative: float


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """One deterministic sequential-gate result."""

    probability_top: float
    decided: bool
    accepted: bool
    reason: str | None


class ComponentTrueSkill:
    """Apply one classic TrueSkill update to controller/deck teams.

    A controller or deck can occur on both sides (same checkpoint with different
    decks, or different checkpoints on the same deck). The Python TrueSkill
    package represents each occurrence as a factor-graph player. Shared
    component posteriors are deliberately held fixed, while the non-shared
    components receive their ordinary team-model updates. This preserves the
    configured two-member performance model without assigning contradictory
    winner and loser updates to one shared skill variable.
    """

    def __init__(self, config: RatingConfig) -> None:
        self.config = config
        self.environment = trueskill.TrueSkill(
            mu=config.mu0,
            sigma=config.sigma0,
            beta=config.beta,
            tau=config.tau,
            draw_probability=config.draw_probability,
        )

    def initial(
        self,
        component_id: str,
        component_kind: Literal["controller", "deck"],
    ) -> ComponentRating:
        """Return the fixed season prior for one component."""
        if component_kind not in {"controller", "deck"}:
            raise ValueError("unknown component kind")
        return ComponentRating(
            component_id=component_id,
            component_kind=component_kind,
            mu=self.config.mu0,
            sigma=self.config.sigma0,
        )

    def rate(
        self,
        side_a: Sequence[ComponentRating],
        side_b: Sequence[ComponentRating],
        outcome: GameOutcome,
    ) -> dict[str, ComponentRating]:
        """Rate one resolved game and return every post-game component."""
        if outcome == "unresolved":
            raise ValueError("unresolved games cannot update ratings")
        self._validate_team(side_a, label="side_a")
        self._validate_team(side_b, label="side_b")
        ranks = {
            "side_a_win": [0, 1],
            "side_b_win": [1, 0],
            "draw": [0, 0],
        }[outcome]
        groups = [
            tuple(trueskill.Rating(item.mu, item.sigma) for item in side_a),
            tuple(trueskill.Rating(item.mu, item.sigma) for item in side_b),
        ]
        rated = self.environment.rate(groups, ranks=ranks)
        shared = {item.component_id for item in side_a} & {
            item.component_id for item in side_b
        }
        output: dict[str, ComponentRating] = {}
        for original_team, rated_team in zip((side_a, side_b), rated, strict=True):
            for original, posterior in zip(original_team, rated_team, strict=True):
                if original.component_id in shared:
                    updated = original.model_copy(update={"games": original.games + 1})
                else:
                    updated = original.model_copy(
                        update={
                            "mu": float(posterior.mu),
                            "sigma": float(posterior.sigma),
                            "games": original.games + 1,
                        }
                    )
                output[original.component_id] = updated
        return output

    @staticmethod
    def _validate_team(team: Sequence[ComponentRating], *, label: str) -> None:
        if len(team) != 2:
            raise ValueError(f"{label} must contain controller and deck components")
        kinds = {item.component_kind for item in team}
        if kinds != {"controller", "deck"}:
            raise ValueError(f"{label} must contain one component of each kind")
        ids = [item.component_id for item in team]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{label} component identities must be distinct")


def compose_bundle_rating(
    controller: ComponentRating,
    deck: ComponentRating,
    *,
    conservative_sigma: float = 3.0,
) -> BundleRating:
    """Compose independent controller and exact-deck presentation values."""
    if controller.component_kind != "controller" or deck.component_kind != "deck":
        raise ValueError("bundle composition requires controller then deck")
    mu = controller.mu + deck.mu
    sigma = math.hypot(controller.sigma, deck.sigma)
    return BundleRating(
        mu=mu,
        sigma=sigma,
        conservative=mu - conservative_sigma * sigma,
    )


def probability_top_fraction(
    candidate: ComponentRating,
    incumbents: Sequence[ComponentRating],
    config: PromotionConfig,
    *,
    evidence_games: int,
) -> float:
    """Estimate candidate checkpoint top-fraction probability deterministically."""
    if candidate.component_kind != "controller":
        raise ValueError("promotion probability requires a controller component")
    if any(item.component_kind != "controller" for item in incumbents):
        raise ValueError("incumbent comparison contains a non-controller component")
    population = (candidate, *sorted(incumbents, key=lambda item: item.component_id))
    top_count = max(1, math.ceil(config.top_fraction * len(population)))
    seed = _monte_carlo_seed(
        config.monte_carlo_seed,
        candidate.component_id,
        evidence_games,
        tuple(item.component_id for item in population),
    )
    generator = np.random.default_rng(seed)
    samples = np.column_stack(
        [
            generator.normal(item.mu, item.sigma, config.monte_carlo_samples)
            for item in population
        ]
    )
    candidate_values = samples[:, 0]
    higher = np.sum(samples[:, 1:] > candidate_values[:, None], axis=1)
    return float(np.mean(higher < top_count))


def promotion_decision(
    *,
    probability_top: float,
    decided_games: int,
    anchor_games: int,
    incumbent_count: int,
    candidate_kind: str,
    config: PromotionConfig,
) -> PromotionDecision:
    """Apply bootstrap, manual exemption, and sequential automatic gates."""
    if candidate_kind != "automatic":
        return PromotionDecision(
            probability_top=probability_top,
            decided=False,
            accepted=True,
            reason=None,
        )
    if decided_games < config.minimum_games:
        return PromotionDecision(probability_top, False, False, None)
    if incumbent_count == 0:
        if anchor_games < config.minimum_games:
            return PromotionDecision(probability_top, False, False, None)
        return PromotionDecision(
            probability_top,
            True,
            True,
            "bootstrap_incumbent_after_anchor_calibration",
        )
    if probability_top >= config.early_accept_probability:
        return PromotionDecision(
            probability_top,
            True,
            True,
            "early_top20_probability_accept",
        )
    if probability_top <= config.early_reject_probability:
        return PromotionDecision(
            probability_top,
            True,
            False,
            "early_top20_probability_reject",
        )
    if decided_games < config.maximum_games:
        return PromotionDecision(probability_top, False, False, None)
    accepted = probability_top >= config.final_accept_probability
    return PromotionDecision(
        probability_top,
        True,
        accepted,
        "maximum_games_top20_accept" if accepted else "maximum_games_top20_reject",
    )


def normal_percentiles(ratings: Mapping[str, ComponentRating]) -> dict[str, float]:
    """Return deterministic midpoint empirical percentiles by component mean."""
    ordered = sorted(ratings.values(), key=lambda item: (item.mu, item.component_id))
    count = len(ordered)
    if not count:
        return {}
    return {
        item.component_id: (index + 0.5) / count for index, item in enumerate(ordered)
    }


def _monte_carlo_seed(
    base_seed: int,
    candidate_id: str,
    evidence_games: int,
    population_ids: tuple[str, ...],
) -> int:
    payload = "\0".join(
        (str(base_seed), candidate_id, str(evidence_games), *population_ids)
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


__all__ = [
    "BundleRating",
    "ComponentTrueSkill",
    "PromotionDecision",
    "compose_bundle_rating",
    "normal_percentiles",
    "probability_top_fraction",
    "promotion_decision",
]
