"""Evidence-shrunk adaptive targets for exact historical matchups."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from typing import Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.rl.opponent_pool._identity import (
    Sha256,
    canonical_fingerprint,
    normalize_sha256,
)
from ptcg_rl.rl.opponent_pool.models import CandidateSeat

PortfolioName = Literal[
    "counter",
    "frontier",
    "probe",
    "rehearsal",
    "staleness",
]

_SimplexKey = TypeVar("_SimplexKey", bound=Hashable)

_PORTFOLIOS: tuple[PortfolioName, ...] = (
    "counter",
    "frontier",
    "probe",
    "rehearsal",
    "staleness",
)


class AdaptivePortfolioWeights(BaseModel):
    """Relative learning-value components before coverage constraints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    counter: float = Field(default=0.30, ge=0.0, allow_inf_nan=False)
    frontier: float = Field(default=0.30, ge=0.0, allow_inf_nan=False)
    probe: float = Field(default=0.15, ge=0.0, allow_inf_nan=False)
    rehearsal: float = Field(default=0.15, ge=0.0, allow_inf_nan=False)
    staleness: float = Field(default=0.10, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def positive_simplex(self) -> Self:
        """Require one interpretable normalized portfolio mixture."""
        total = sum(self.as_mapping().values())
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError("adaptive portfolio weights must sum to one")
        return self

    def as_mapping(self) -> dict[PortfolioName, float]:
        """Return the canonical portfolio vector."""
        return {name: float(getattr(self, name)) for name in _PORTFOLIOS}


class AdaptiveEvidenceAllocationConfig(BaseModel):
    """Shared posterior, candidate-balance, and decision-credit controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fast_half_life_windows: float = Field(gt=0.0, allow_inf_nan=False)
    slow_half_life_windows: float = Field(gt=0.0, allow_inf_nan=False)
    hierarchical_prior_games: float = Field(ge=0.0, allow_inf_nan=False)
    fallback_evidence_cap: float = Field(ge=0.0, allow_inf_nan=False)
    confidence_scale: float = Field(gt=0.0, allow_inf_nan=False)
    counter_target_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    frontier_target_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    frontier_width: float = Field(gt=0.0, allow_inf_nan=False)
    staleness_windows: float = Field(gt=0.0, allow_inf_nan=False)
    candidate_base_mix: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    candidate_temperature: float = Field(gt=0.0, allow_inf_nan=False)
    candidate_maximum_share_ratio: float = Field(ge=1.0, allow_inf_nan=False)
    artifact_cvar_fraction: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    recent_artifacts: int = Field(ge=0)
    decision_credit_prior_games: float = Field(gt=0.0, allow_inf_nan=False)
    decision_credit_prior: float = Field(gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def coherent_timescales(self) -> Self:
        """Keep fast evidence more responsive than the reference baseline."""
        if self.fast_half_life_windows >= self.slow_half_life_windows:
            raise ValueError("fast evidence half-life must be shorter than slow")
        return self


class AdaptiveOpponentAllocationConfig(AdaptiveEvidenceAllocationConfig):
    """Legacy V3 weighted-cell target generation and stability controls."""

    uniform_route_mix: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    previous_target_mix: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    maximum_route_share_ratio: float = Field(ge=1.0, allow_inf_nan=False)
    minimum_artifact_games: int = Field(default=1, ge=1)
    portfolios: AdaptivePortfolioWeights


class RoleBudgetOpponentAllocationConfig(AdaptiveEvidenceAllocationConfig):
    """Role-budget evidence and optional sparse-execution controls."""

    recent_artifacts: int = Field(ge=1)
    matchup_game_batch_size: int = Field(default=1, ge=1)
    matchup_coverage_windows: int | None = Field(default=None, ge=1)


class AdaptiveMatchupIdentity(BaseModel):
    """Stable exact cell identity retained outside the active revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_deck_digest: Sha256
    artifact_id: Sha256
    route_id: Sha256
    opponent_deck_digest: Sha256
    candidate_seat: CandidateSeat

    @field_validator(
        "candidate_deck_digest",
        "artifact_id",
        "route_id",
        "opponent_deck_digest",
    )
    @classmethod
    def valid_identity(cls, value: str) -> str:
        """Normalize every internal routing identity."""
        return normalize_sha256(value)

    @property
    def matchup_key(self) -> tuple[str, str, int]:
        """Return the core planner key."""
        return (
            self.candidate_deck_digest,
            self.route_id,
            self.candidate_seat,
        )

    @property
    def evidence_key(self) -> tuple[str, str, str, int]:
        """Return a route-contract-independent grouping key."""
        return (
            self.candidate_deck_digest,
            self.artifact_id,
            self.opponent_deck_digest,
            self.candidate_seat,
        )


class AdaptiveMatchupEvidence(BaseModel):
    """Lazy-decayed sufficient statistics for one exact matchup."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: AdaptiveMatchupIdentity
    fast_score_sum: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    fast_score_weight: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    slow_score_sum: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    slow_score_weight: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    decision_sum: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    exposure_weight: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    total_terminal_games: int = Field(default=0, ge=0)
    total_trainable_decisions: int = Field(default=0, ge=0)
    last_update_window: int = Field(default=0, ge=0)
    last_learning_window: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def sufficient_statistics_are_coherent(self) -> Self:
        """Reject impossible means and clocks."""
        if self.fast_score_sum > self.fast_score_weight + 1e-9:
            raise ValueError("fast score mass exceeds its evidence")
        if self.slow_score_sum > self.slow_score_weight + 1e-9:
            raise ValueError("slow score mass exceeds its evidence")
        if self.last_learning_window is not None and (
            self.last_learning_window > self.last_update_window
        ):
            raise ValueError("learning window exceeds evidence clock")
        return self


class AdaptiveTargetWeight(BaseModel):
    """Compact prior target retained for trust-region smoothing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_deck_digest: Sha256
    route_id: Sha256
    candidate_seat: CandidateSeat
    share: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("candidate_deck_digest", "route_id")
    @classmethod
    def valid_identity(cls, value: str) -> str:
        """Normalize exact routing fields."""
        return normalize_sha256(value)

    @property
    def matchup_key(self) -> tuple[str, str, int]:
        """Return the planner cell key."""
        return (self.candidate_deck_digest, self.route_id, self.candidate_seat)


class AdaptiveCellScore(BaseModel):
    """One posterior score and its auditable demand decomposition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: AdaptiveMatchupIdentity
    posterior_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    slow_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    effective_evidence: float = Field(ge=0.0, allow_inf_nan=False)
    posterior_stddev: float = Field(ge=0.0, allow_inf_nan=False)
    expected_decisions: float = Field(gt=0.0, allow_inf_nan=False)
    base_weight: float = Field(gt=0.0, allow_inf_nan=False)
    components: dict[PortfolioName, float]
    utility: float = Field(gt=0.0, allow_inf_nan=False)
    dominant_portfolio: PortfolioName

    @field_validator("components")
    @classmethod
    def valid_components(
        cls,
        value: dict[PortfolioName, float],
    ) -> dict[PortfolioName, float]:
        """Require one bounded complete component vector."""
        if set(value) != set(_PORTFOLIOS) or any(
            not math.isfinite(item) or item < 0.0 or item > 1.0
            for item in value.values()
        ):
            raise ValueError("adaptive score components are invalid")
        return value


class AdaptiveAllocationSnapshot(BaseModel):
    """Immutable target evidence attached to one pending quota plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    window_sequence: int = Field(ge=0)
    candidate_target_shares: dict[Sha256, float]
    target_weights: tuple[AdaptiveTargetWeight, ...]
    portfolio_target_mass: dict[PortfolioName, float]
    evidence_cells: int = Field(ge=0)
    low_evidence_cells: int = Field(ge=0)

    @field_validator("candidate_target_shares")
    @classmethod
    def valid_candidate_simplex(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:
        """Validate and normalize candidate identities without changing mass."""
        normalized = {normalize_sha256(key): item for key, item in value.items()}
        if len(normalized) != len(value) or any(
            not math.isfinite(item) or item <= 0.0 for item in normalized.values()
        ):
            raise ValueError("candidate target shares must be finite and positive")
        if not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-9):
            raise ValueError("candidate target shares must sum to one")
        return normalized

    @field_validator("portfolio_target_mass")
    @classmethod
    def valid_portfolio_mass(
        cls,
        value: dict[PortfolioName, float],
    ) -> dict[PortfolioName, float]:
        """Require a complete finite diagnostic vector."""
        if set(value) != set(_PORTFOLIOS) or any(
            not math.isfinite(item) or item < 0.0 for item in value.values()
        ):
            raise ValueError("portfolio target mass is invalid")
        total = sum(value.values())
        if total > 0.0 and not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError("portfolio target mass must sum to one")
        return value

    @model_validator(mode="after")
    def targets_are_canonical(self) -> Self:
        """Keep compact targets deterministic and unique."""
        keys = tuple(item.matchup_key for item in self.target_weights)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("adaptive target weights must be canonical")
        return self

    @property
    def fingerprint(self) -> str:
        """Return the target snapshot content identity."""
        return canonical_fingerprint(
            "adaptive-opponent-allocation-snapshot-v1",
            self.model_dump(mode="json"),
        )


def decay_evidence(
    evidence: AdaptiveMatchupEvidence,
    *,
    window_sequence: int,
    config: AdaptiveEvidenceAllocationConfig,
) -> AdaptiveMatchupEvidence:
    """Project one lazily stored row to the requested settled window."""
    if window_sequence < evidence.last_update_window:
        raise ValueError("adaptive evidence cannot move backward")
    elapsed = window_sequence - evidence.last_update_window
    if elapsed == 0:
        return evidence
    fast_decay = math.exp2(-elapsed / config.fast_half_life_windows)
    slow_decay = math.exp2(-elapsed / config.slow_half_life_windows)
    return evidence.model_copy(
        update={
            "fast_score_sum": evidence.fast_score_sum * fast_decay,
            "fast_score_weight": evidence.fast_score_weight * fast_decay,
            "slow_score_sum": evidence.slow_score_sum * slow_decay,
            "slow_score_weight": evidence.slow_score_weight * slow_decay,
            # Coverage debt is an allocation ledger, not a recency estimator.
            # Keeping it cumulative prevents an untrained route from looking
            # fully covered merely because other evidence aged out.
            "decision_sum": evidence.decision_sum,
            "exposure_weight": evidence.exposure_weight,
            "last_update_window": window_sequence,
        }
    )


def observe_evidence(
    evidence: AdaptiveMatchupEvidence | None,
    *,
    identity: AdaptiveMatchupIdentity,
    window_sequence: int,
    candidate_score: float | None,
    candidate_decisions: int,
    config: AdaptiveEvidenceAllocationConfig,
) -> AdaptiveMatchupEvidence:
    """Apply one executed game to the exact-cell sufficient statistics."""
    if candidate_decisions < 0:
        raise ValueError("candidate decisions cannot be negative")
    current = decay_evidence(
        evidence
        or AdaptiveMatchupEvidence(
            identity=identity,
            last_update_window=window_sequence,
        ),
        window_sequence=window_sequence,
        config=config,
    )
    if current.identity != identity:
        raise ValueError("adaptive evidence identity changed")
    score = 0.0 if candidate_score is None else float(candidate_score)
    terminal = candidate_score is not None
    learning = candidate_decisions > 0
    return current.model_copy(
        update={
            "fast_score_sum": current.fast_score_sum + (score if terminal else 0.0),
            "fast_score_weight": current.fast_score_weight + float(terminal),
            "slow_score_sum": current.slow_score_sum + (score if terminal else 0.0),
            "slow_score_weight": current.slow_score_weight + float(terminal),
            "decision_sum": current.decision_sum + float(candidate_decisions),
            "exposure_weight": current.exposure_weight + float(learning),
            "total_terminal_games": current.total_terminal_games + int(terminal),
            "total_trainable_decisions": (
                current.total_trainable_decisions + candidate_decisions
            ),
            "last_learning_window": (
                window_sequence if learning else current.last_learning_window
            ),
        }
    )


class AdaptiveEvidenceIndex:
    """Pre-aggregate hierarchical parents for fast repeated cell scoring."""

    def __init__(
        self,
        evidence: Sequence[AdaptiveMatchupEvidence],
        *,
        window_sequence: int,
        config: AdaptiveEvidenceAllocationConfig,
    ) -> None:
        """Decay evidence once and build crossed parent sufficient statistics."""
        self.config = config
        self.window_sequence = window_sequence
        self.rows = {
            item.identity.evidence_key: decay_evidence(
                item,
                window_sequence=window_sequence,
                config=config,
            )
            for item in evidence
        }
        self._fast: defaultdict[tuple[str, ...], list[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        self._slow: defaultdict[tuple[str, ...], list[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        for row in self.rows.values():
            identity = row.identity
            parents = (
                (
                    "candidate-artifact",
                    identity.candidate_deck_digest,
                    identity.artifact_id,
                ),
                (
                    "candidate-opponent",
                    identity.candidate_deck_digest,
                    identity.opponent_deck_digest,
                ),
                ("candidate", identity.candidate_deck_digest),
                ("artifact", identity.artifact_id),
                ("global",),
            )
            for parent in parents:
                self._fast[parent][0] += row.fast_score_sum
                self._fast[parent][1] += row.fast_score_weight
                self._slow[parent][0] += row.slow_score_sum
                self._slow[parent][1] += row.slow_score_weight

    def score(
        self,
        identity: AdaptiveMatchupIdentity,
        *,
        fallback_score: float,
        fallback_evidence: int,
        base_weight: float,
    ) -> AdaptiveCellScore:
        """Return a hierarchical posterior and learning-value decomposition."""
        if not 0.0 <= fallback_score <= 1.0:
            raise ValueError("adaptive fallback score must be a probability")
        if fallback_evidence < 0 or base_weight <= 0.0:
            raise ValueError("adaptive fallback evidence or base weight is invalid")
        row = self.rows.get(identity.evidence_key)
        fast_sum = 0.0 if row is None else row.fast_score_sum
        fast_weight = 0.0 if row is None else row.fast_score_weight
        slow_sum = 0.0 if row is None else row.slow_score_sum
        slow_weight = 0.0 if row is None else row.slow_score_weight
        fallback_weight = min(
            float(fallback_evidence),
            self.config.fallback_evidence_cap,
        )
        parent_fast = self._parent_mean(
            identity,
            self._fast,
            fallback_score,
            exact_sum=fast_sum,
            exact_weight=fast_weight,
        )
        parent_slow = self._parent_mean(
            identity,
            self._slow,
            fallback_score,
            exact_sum=slow_sum,
            exact_weight=slow_weight,
        )
        prior = self.config.hierarchical_prior_games
        posterior_weight = fast_weight + fallback_weight + prior
        slow_posterior_weight = slow_weight + fallback_weight + prior
        posterior = (
            (fast_sum + fallback_score * fallback_weight + parent_fast * prior)
            / posterior_weight
            if posterior_weight > 0.0
            else fallback_score
        )
        slow = (
            (slow_sum + fallback_score * fallback_weight + parent_slow * prior)
            / slow_posterior_weight
            if slow_posterior_weight > 0.0
            else fallback_score
        )
        variance = posterior * (1.0 - posterior) / max(posterior_weight + 1.0, 1.0)
        stddev = math.sqrt(max(variance, 0.0))
        neutral_stddev = 0.5 / math.sqrt(max(prior + 1.0, 1.0))
        probe = min(1.0, stddev / max(neutral_stddev, 1e-9))
        lower = max(0.0, posterior - self.config.confidence_scale * stddev)
        counter = min(
            1.0,
            max(0.0, self.config.counter_target_score - lower)
            / max(self.config.counter_target_score, 1e-9),
        )
        frontier_distance = (
            posterior - self.config.frontier_target_score
        ) / self.config.frontier_width
        frontier = math.exp(-0.5 * frontier_distance * frontier_distance)
        rehearsal = min(1.0, max(0.0, slow - posterior) * 2.0)
        last_learning = None if row is None else row.last_learning_window
        staleness = (
            1.0
            if last_learning is None
            else min(
                1.0,
                max(0.0, self.window_sequence - last_learning)
                / self.config.staleness_windows,
            )
        )
        components: dict[PortfolioName, float] = {
            "counter": counter,
            "frontier": frontier,
            "probe": probe,
            "rehearsal": rehearsal,
            "staleness": staleness,
        }
        if isinstance(self.config, AdaptiveOpponentAllocationConfig):
            portfolio_weights = self.config.portfolios.as_mapping()
            weighted = {
                name: portfolio_weights[name] * components[name] for name in _PORTFOLIOS
            }
            utility = base_weight * max(sum(weighted.values()), 1e-9)
        else:
            # V4 consumes the individual components under explicit role
            # budgets. These values remain only a backwards-compatible score
            # summary and never determine the V4 allocation.
            weighted = dict(components)
            utility = base_weight * max(max(components.values()), 1e-9)
        dominant = max(
            _PORTFOLIOS, key=lambda name: (weighted[name], -_PORTFOLIOS.index(name))
        )
        expected_decisions = (
            (0.0 if row is None else row.decision_sum)
            + self.config.decision_credit_prior_games
            * self.config.decision_credit_prior
        ) / (
            (0.0 if row is None else row.exposure_weight)
            + self.config.decision_credit_prior_games
        )
        return AdaptiveCellScore(
            identity=identity,
            posterior_score=posterior,
            slow_score=slow,
            effective_evidence=fast_weight + fallback_weight,
            posterior_stddev=stddev,
            expected_decisions=max(expected_decisions, 1e-9),
            base_weight=base_weight,
            components=components,
            utility=utility,
            dominant_portfolio=dominant,
        )

    def _parent_mean(
        self,
        identity: AdaptiveMatchupIdentity,
        table: Mapping[tuple[str, ...], list[float]],
        fallback: float,
        *,
        exact_sum: float,
        exact_weight: float,
    ) -> float:
        """Blend leave-one-out parents without recycling the exact observation."""
        parents = (
            (
                "candidate-artifact",
                identity.candidate_deck_digest,
                identity.artifact_id,
            ),
            (
                "candidate-opponent",
                identity.candidate_deck_digest,
                identity.opponent_deck_digest,
            ),
            ("candidate", identity.candidate_deck_digest),
            ("artifact", identity.artifact_id),
            ("global",),
        )
        weighted_sum = fallback
        total_weight = 1.0
        for parent in parents:
            parent_success, parent_weight = table.get(parent, (0.0, 0.0))
            success = max(0.0, parent_success - exact_sum)
            weight = max(0.0, parent_weight - exact_weight)
            if weight <= 0.0:
                continue
            parent_weight = math.sqrt(weight)
            weighted_sum += parent_weight * success / weight
            total_weight += parent_weight
        return weighted_sum / total_weight


def cvar(values: Sequence[float], *, fraction: float) -> float:
    """Return the upper-tail mean used for weakness and artifact demand."""
    if not values or not 0.0 < fraction <= 1.0:
        raise ValueError("CVaR requires values and a valid fraction")
    ordered = sorted((float(value) for value in values), reverse=True)
    count = max(1, math.ceil(len(ordered) * fraction))
    return sum(ordered[:count]) / float(count)


def bounded_candidate_shares(
    base: Mapping[str, float],
    demands: Mapping[str, float],
    *,
    config: AdaptiveEvidenceAllocationConfig,
) -> dict[str, float]:
    """Blend declared candidate mass with joint exact-matchup demand."""
    if set(base) != set(demands) or not base:
        raise ValueError("candidate bases and demands differ")
    if any(value <= 0.0 or not math.isfinite(value) for value in base.values()):
        raise ValueError("candidate base shares must be positive and finite")
    if not math.isclose(sum(base.values()), 1.0, abs_tol=1e-9):
        raise ValueError("candidate base shares must sum to one")
    highest = max(demands.values())
    priorities = {
        key: math.exp((value - highest) / config.candidate_temperature)
        for key, value in demands.items()
    }
    priority_total = sum(priorities.values())
    raw = {
        key: config.candidate_base_mix * base[key]
        + (1.0 - config.candidate_base_mix) * priorities[key] / priority_total
        for key in base
    }
    maxima = {
        key: min(1.0, config.candidate_maximum_share_ratio * base[key]) for key in base
    }
    return _project_capped_simplex(raw, maxima)


def normalized_matchup_targets(
    scores: Sequence[AdaptiveCellScore],
    *,
    previous: Mapping[tuple[str, str, int], float],
    config: AdaptiveOpponentAllocationConfig,
) -> tuple[AdaptiveTargetWeight, ...]:
    """Create stable bounded route targets within each candidate-seat cell."""
    grouped: defaultdict[tuple[str, int], list[AdaptiveCellScore]] = defaultdict(list)
    for score in scores:
        grouped[
            (score.identity.candidate_deck_digest, score.identity.candidate_seat)
        ].append(score)
    result: list[AdaptiveTargetWeight] = []
    for candidate_cell, cell_scores in sorted(grouped.items()):
        count = len(cell_scores)
        uniform = 1.0 / float(count)
        utility_total = sum(item.utility for item in cell_scores)
        raw = {
            item.identity.matchup_key: (
                config.uniform_route_mix * uniform
                + (1.0 - config.uniform_route_mix) * item.utility / utility_total
            )
            for item in cell_scores
        }
        smoothed = {
            key: (1.0 - config.previous_target_mix) * value
            + config.previous_target_mix * previous.get(key, value)
            for key, value in raw.items()
        }
        normalized_total = sum(smoothed.values())
        normalized = {key: value / normalized_total for key, value in smoothed.items()}
        capped = _project_capped_simplex(
            normalized,
            dict.fromkeys(
                normalized,
                min(1.0, config.maximum_route_share_ratio * uniform),
            ),
        )
        result.extend(
            AdaptiveTargetWeight(
                candidate_deck_digest=candidate_cell[0],
                route_id=key[1],
                candidate_seat=candidate_cell[1],  # type: ignore[arg-type]
                share=share,
            )
            for key, share in sorted(capped.items())
        )
    return tuple(sorted(result, key=lambda item: item.matchup_key))


def portfolio_mass(
    scores: Sequence[AdaptiveCellScore],
    targets: Sequence[AdaptiveTargetWeight],
    candidate_shares: Mapping[str, float],
) -> dict[PortfolioName, float]:
    """Aggregate target mass by each cell's dominant auditable objective."""
    by_key = {item.identity.matchup_key: item for item in scores}
    totals: dict[PortfolioName, float] = dict.fromkeys(_PORTFOLIOS, 0.0)
    for target in targets:
        totals[by_key[target.matchup_key].dominant_portfolio] += (
            target.share * candidate_shares[target.candidate_deck_digest] / 2.0
        )
    total = sum(totals.values())
    return (
        totals
        if total <= 0.0
        else {name: value / total for name, value in totals.items()}
    )


def _project_capped_simplex(
    values: Mapping[_SimplexKey, float],
    maxima: Mapping[_SimplexKey, float],
) -> dict[_SimplexKey, float]:
    """Project positive weights to a simplex with deterministic upper bounds."""
    if set(values) != set(maxima) or not values:
        raise ValueError("capped simplex inputs differ")
    if any(value < 0.0 or not math.isfinite(value) for value in values.values()):
        raise ValueError("capped simplex weights are invalid")
    if any(value <= 0.0 or value > 1.0 for value in maxima.values()):
        raise ValueError("capped simplex maxima are invalid")
    if sum(maxima.values()) < 1.0 - 1e-12:
        raise ValueError("capped simplex maxima cannot carry unit mass")
    remaining = set(values)
    result: dict[_SimplexKey, float] = {}
    residual = 1.0
    weights = dict(values)
    while remaining:
        total = sum(weights[key] for key in remaining)
        if total <= 0.0:
            shares = dict.fromkeys(remaining, residual / float(len(remaining)))
        else:
            shares = {key: residual * weights[key] / total for key in remaining}
        capped = tuple(
            sorted(
                (key for key in remaining if shares[key] > maxima[key] + 1e-15),
                key=str,
            )
        )
        if not capped:
            result.update(shares)
            break
        for key in capped:
            result[key] = maxima[key]
            residual -= maxima[key]
            remaining.remove(key)
    correction = 1.0 - sum(result.values())
    if abs(correction) > 1e-12:
        adjustable = min(
            (key for key in result if result[key] + correction <= maxima[key] + 1e-12),
            key=str,
        )
        result[adjustable] += correction
    return result


__all__ = [
    "AdaptiveAllocationSnapshot",
    "AdaptiveCellScore",
    "AdaptiveEvidenceIndex",
    "AdaptiveEvidenceAllocationConfig",
    "AdaptiveMatchupEvidence",
    "AdaptiveMatchupIdentity",
    "AdaptiveOpponentAllocationConfig",
    "AdaptivePortfolioWeights",
    "AdaptiveTargetWeight",
    "PortfolioName",
    "RoleBudgetOpponentAllocationConfig",
    "bounded_candidate_shares",
    "cvar",
    "decay_evidence",
    "normalized_matchup_targets",
    "observe_evidence",
    "portfolio_mass",
]
