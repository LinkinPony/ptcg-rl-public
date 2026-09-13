"""Bayesian scoring for candidate bundles on a shared target meta.

The evaluator deliberately separates game collection from statistical scoring.
Every candidate is integrated over the same opponent distribution.  Missing
matchups remain explicit Beta-prior cells, exact self matchups are fixed at
one-half, and unobserved meta tail mass is represented by a real opponent
bucket instead of being renormalized away.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "BundleEvaluationResult",
    "BundlePortfolioSummary",
    "BundlePosteriorContrast",
    "BundlePosteriorSummary",
    "MatchupOutcome",
    "PosteriorEvaluationConfig",
    "evaluate_bundle_posteriors",
]


class MatchupOutcome(BaseModel):
    """Aggregated outcomes for one candidate-versus-opponent bundle cell."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    candidate_id: str
    opponent_id: str
    wins: int = Field(default=0, ge=0)
    draws: int = Field(default=0, ge=0)
    losses: int = Field(default=0, ge=0)

    @field_validator("candidate_id", "opponent_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Normalize and reject empty bundle identifiers."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("bundle identifiers must be non-empty")
        return normalized

    @model_validator(mode="after")
    def has_games(self) -> MatchupOutcome:
        """Reject rows which carry no evidence."""
        if self.games <= 0:
            raise ValueError("a matchup outcome row must contain at least one game")
        return self

    @property
    def games(self) -> int:
        """Return the number of games represented by this row."""
        return self.wins + self.draws + self.losses


class PosteriorEvaluationConfig(BaseModel):
    """Monte Carlo and prior configuration for bundle evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    sample_count: int = Field(default=20_000, ge=1)
    sample_batch_size: int = Field(default=2_048, ge=1)
    seed: int = 0
    beta_prior_alpha: float = Field(default=1.0, gt=0.0)
    beta_prior_beta: float = Field(default=1.0, gt=0.0)
    meta_concentration: float = Field(default=200.0, gt=0.0)
    credible_mass: float = Field(default=0.95, gt=0.0, lt=1.0)
    lcb_quantile: float = Field(default=0.05, gt=0.0, lt=1.0)
    cvar_quantile: float = Field(default=0.10, gt=0.0, le=1.0)
    top_k: int = Field(default=3, ge=1)
    unknown_opponent_id: str = "__unknown__"

    @field_validator("unknown_opponent_id")
    @classmethod
    def valid_unknown_id(cls, value: str) -> str:
        """Normalize and reject an empty unknown-opponent identifier."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("unknown_opponent_id must be non-empty")
        return normalized


class BundlePosteriorSummary(BaseModel):
    """Posterior decision metrics for one candidate bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    rank: int = Field(ge=1)
    games: int = Field(ge=0)
    self_games_ignored: int = Field(ge=0)
    observed_matchups: int = Field(ge=0)
    prior_only_matchups: int = Field(ge=0)
    observed_meta_mass: float = Field(ge=0.0, le=1.0)
    self_meta_mass: float = Field(ge=0.0, le=1.0)
    prior_only_meta_mass: float = Field(ge=0.0, le=1.0)
    deploy_mean: float = Field(ge=0.0, le=1.0)
    deploy_standard_deviation: float = Field(ge=0.0)
    deploy_credible_low: float = Field(ge=0.0, le=1.0)
    deploy_credible_high: float = Field(ge=0.0, le=1.0)
    deploy_lcb: float = Field(ge=0.0, le=1.0)
    deploy_uncertainty_cvar: float = Field(ge=0.0, le=1.0)
    matchup_cvar_mean: float = Field(ge=0.0, le=1.0)
    matchup_cvar_credible_low: float = Field(ge=0.0, le=1.0)
    matchup_cvar_credible_high: float = Field(ge=0.0, le=1.0)
    probability_best: float = Field(ge=0.0, le=1.0)
    probability_top_k: float = Field(ge=0.0, le=1.0)
    probability_above_even: float = Field(ge=0.0, le=1.0)
    expected_regret: float = Field(ge=0.0, le=1.0)


class BundlePosteriorContrast(BaseModel):
    """Paired posterior score difference for one candidate and reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    reference_id: str
    mean_delta: float = Field(ge=-1.0, le=1.0)
    standard_deviation: float = Field(ge=0.0)
    credible_low: float = Field(ge=-1.0, le=1.0)
    credible_high: float = Field(ge=-1.0, le=1.0)
    probability_above_zero: float = Field(ge=0.0, le=1.0)
    probability_noninferior_2pp: float = Field(ge=0.0, le=1.0)


class BundlePortfolioSummary(BaseModel):
    """Joint posterior metrics for one unordered two-candidate portfolio."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    candidate_ids: tuple[str, str]
    expected_best_score: float = Field(ge=0.0, le=1.0)
    standard_deviation: float = Field(ge=0.0)
    credible_low: float = Field(ge=0.0, le=1.0)
    credible_high: float = Field(ge=0.0, le=1.0)
    best_score_lcb: float = Field(ge=0.0, le=1.0)
    probability_best: float = Field(ge=0.0, le=1.0)
    probability_at_least_one_above_even: float = Field(ge=0.0, le=1.0)
    joint_downside_probability: float = Field(ge=0.0, le=1.0)
    diversification_gain: float = Field(ge=0.0, le=1.0)
    score_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)
    expected_regret: float = Field(ge=0.0, le=1.0)


class BundleEvaluationResult(BaseModel):
    """Complete posterior report over a common opponent denominator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config: PosteriorEvaluationConfig
    candidate_ids: tuple[str, ...]
    opponent_ids: tuple[str, ...]
    meta_weights: dict[str, float]
    known_meta_mass: float = Field(ge=0.0, le=1.0)
    unknown_mass: float = Field(ge=0.0, le=1.0)
    common_denominator: float = Field(ge=0.0)
    top_k: int = Field(ge=1)
    candidates: tuple[BundlePosteriorSummary, ...]
    contrasts: tuple[BundlePosteriorContrast, ...]
    portfolios: tuple[BundlePortfolioSummary, ...] = ()


def evaluate_bundle_posteriors(
    *,
    candidate_ids: Sequence[str],
    outcomes: Iterable[MatchupOutcome],
    meta_weights: Mapping[str, float],
    unknown_mass: float,
    config: PosteriorEvaluationConfig | None = None,
    self_opponents: Mapping[str, str] | None = None,
    fixed_matchup_scores: Mapping[tuple[str, str], float] | None = None,
    include_portfolios: bool = False,
) -> BundleEvaluationResult:
    """Evaluate candidate bundles against one shared target meta distribution.

    ``meta_weights`` are probabilities for known opponent bundles, not raw
    counts.  Together with ``unknown_mass`` they must sum to one.  Meta-weight
    uncertainty is modeled by a Dirichlet distribution whose effective sample
    size is ``config.meta_concentration``.  The unknown tail, when nonzero, is
    a normal Beta matchup cell and may be updated by outcome rows whose
    ``opponent_id`` equals ``config.unknown_opponent_id``.

    Args:
        candidate_ids: Bundle identifiers to compare. Identifiers are sorted so
            a fixed seed is invariant to caller ordering.
        outcomes: Aggregated matchup evidence. Multiple rows for the same cell
            are combined.
        meta_weights: Known-opponent target probabilities.
        unknown_mass: Explicit probability assigned to unseen opponents.
        config: Prior and Monte Carlo settings.
        self_opponents: Optional candidate-to-opponent identity mapping. Exact
            self cells are fixed at score 0.5. By default matching identifiers
            are treated as self cells.
        fixed_matchup_scores: Optional exact scores for candidate/opponent
            cells that must not be sampled, such as seat-split self matchups.
        include_portfolios: Whether to summarize every unordered candidate pair
            from the shared deployment posterior draws.

    Returns:
        A ranked posterior report. All candidates use the same sampled meta
        weights, including prior-only and unknown cells.

    Raises:
        ValueError: If identifiers, probabilities, or outcome support are
            inconsistent.
    """
    resolved_config = config or PosteriorEvaluationConfig()
    candidates = _validated_candidate_ids(candidate_ids)
    if resolved_config.unknown_opponent_id in candidates:
        raise ValueError("candidate_ids must not use the reserved unknown opponent id")
    weights = _validated_meta_weights(
        meta_weights,
        unknown_mass=unknown_mass,
        unknown_opponent_id=resolved_config.unknown_opponent_id,
    )
    opponents = tuple(sorted(weights))
    self_cells = _resolved_self_cells(
        candidates,
        opponents,
        self_opponents=self_opponents,
    )
    fixed_scores = _validated_fixed_matchup_scores(
        fixed_matchup_scores,
        candidates=candidates,
        opponents=opponents,
    )
    aggregated = _aggregate_outcomes(
        outcomes,
        candidates=candidates,
        opponents=opponents,
    )
    deploy_samples, robust_samples = _draw_score_samples(
        candidates=candidates,
        opponents=opponents,
        weights=weights,
        self_cells=self_cells,
        fixed_matchup_scores=fixed_scores,
        outcomes=aggregated,
        config=resolved_config,
    )
    summaries = _summarize_candidates(
        candidates=candidates,
        opponents=opponents,
        weights=weights,
        self_cells=self_cells,
        fixed_matchup_scores=fixed_scores,
        outcomes=aggregated,
        deploy_samples=deploy_samples,
        robust_samples=robust_samples,
        config=resolved_config,
    )
    contrasts = _summarize_contrasts(
        candidates=candidates,
        deploy_samples=deploy_samples,
        config=resolved_config,
    )
    portfolios = (
        _summarize_portfolios(
            candidates=candidates,
            deploy_samples=deploy_samples,
            candidate_probability_best={
                summary.candidate_id: summary.probability_best for summary in summaries
            },
            config=resolved_config,
        )
        if include_portfolios
        else ()
    )
    normalized_unknown_mass = weights.get(resolved_config.unknown_opponent_id, 0.0)
    return BundleEvaluationResult(
        config=resolved_config,
        candidate_ids=candidates,
        opponent_ids=opponents,
        meta_weights=dict(weights),
        known_meta_mass=1.0 - normalized_unknown_mass,
        unknown_mass=normalized_unknown_mass,
        common_denominator=1.0,
        top_k=min(resolved_config.top_k, len(candidates)),
        candidates=summaries,
        contrasts=contrasts,
        portfolios=portfolios,
    )


def _validated_candidate_ids(candidate_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(candidate_id).strip() for candidate_id in candidate_ids)
    if not normalized or any(not candidate_id for candidate_id in normalized):
        raise ValueError("candidate_ids must contain non-empty identifiers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("candidate_ids must be unique")
    return tuple(sorted(normalized))


def _validated_meta_weights(
    meta_weights: Mapping[str, float],
    *,
    unknown_mass: float,
    unknown_opponent_id: str,
) -> dict[str, float]:
    if not math.isfinite(unknown_mass) or not 0.0 <= unknown_mass <= 1.0:
        raise ValueError("unknown_mass must be finite and between zero and one")
    normalized: dict[str, float] = {}
    for raw_opponent_id, raw_weight in meta_weights.items():
        opponent_id = str(raw_opponent_id).strip()
        weight = float(raw_weight)
        if not opponent_id:
            raise ValueError("meta_weights contains an empty opponent identifier")
        if opponent_id == unknown_opponent_id:
            raise ValueError(
                "meta_weights must not contain the reserved unknown opponent id"
            )
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("meta weights must be finite and non-negative")
        if weight > 0.0:
            if opponent_id in normalized:
                raise ValueError(
                    f"meta_weights contains duplicate normalized id {opponent_id!r}"
                )
            normalized[opponent_id] = weight
    total = sum(normalized.values()) + unknown_mass
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "known meta weights and unknown_mass must sum to one; "
            f"received {total:.12g}"
        )
    if total <= 0.0:
        raise ValueError("target meta distribution must have positive mass")
    normalized = {
        opponent_id: weight / total for opponent_id, weight in normalized.items()
    }
    if unknown_mass > 0.0:
        normalized[unknown_opponent_id] = unknown_mass / total
    return dict(sorted(normalized.items()))


def _resolved_self_cells(
    candidates: tuple[str, ...],
    opponents: tuple[str, ...],
    *,
    self_opponents: Mapping[str, str] | None,
) -> dict[str, str | None]:
    candidate_set = set(candidates)
    opponent_set = set(opponents)
    if self_opponents is not None:
        unknown_candidates = set(self_opponents) - candidate_set
        if unknown_candidates:
            raise ValueError(
                "self_opponents contains unknown candidates: "
                + ", ".join(sorted(unknown_candidates))
            )
    output: dict[str, str | None] = {}
    for candidate_id in candidates:
        opponent_id = (
            str(self_opponents[candidate_id]).strip()
            if self_opponents is not None and candidate_id in self_opponents
            else candidate_id
        )
        if not opponent_id:
            raise ValueError("self opponent identifiers must be non-empty")
        if self_opponents is not None and candidate_id in self_opponents:
            if opponent_id not in opponent_set:
                raise ValueError(
                    f"self opponent {opponent_id!r} is outside the target meta"
                )
            output[candidate_id] = opponent_id
        else:
            output[candidate_id] = opponent_id if opponent_id in opponent_set else None
    return output


def _validated_fixed_matchup_scores(
    fixed_matchup_scores: Mapping[tuple[str, str], float] | None,
    *,
    candidates: tuple[str, ...],
    opponents: tuple[str, ...],
) -> dict[tuple[str, str], float]:
    candidate_set = set(candidates)
    opponent_set = set(opponents)
    output: dict[tuple[str, str], float] = {}
    for raw_key, raw_score in (fixed_matchup_scores or {}).items():
        if len(raw_key) != 2:
            raise ValueError("fixed matchup keys must contain candidate and opponent")
        candidate_id = str(raw_key[0]).strip()
        opponent_id = str(raw_key[1]).strip()
        score = float(raw_score)
        if candidate_id not in candidate_set:
            raise ValueError(f"fixed matchup has unknown candidate {candidate_id!r}")
        if opponent_id not in opponent_set:
            raise ValueError(f"fixed matchup has unknown opponent {opponent_id!r}")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                "fixed matchup scores must be finite and between zero and one"
            )
        key = (candidate_id, opponent_id)
        if key in output:
            raise ValueError(f"duplicate normalized fixed matchup {key!r}")
        output[key] = score
    return dict(sorted(output.items()))


def _aggregate_outcomes(
    outcomes: Iterable[MatchupOutcome],
    *,
    candidates: tuple[str, ...],
    opponents: tuple[str, ...],
) -> dict[tuple[str, str], tuple[int, int, int]]:
    candidate_set = set(candidates)
    opponent_set = set(opponents)
    aggregated: dict[tuple[str, str], tuple[int, int, int]] = {}
    for outcome in outcomes:
        if outcome.candidate_id not in candidate_set:
            raise ValueError(
                f"outcome candidate {outcome.candidate_id!r} was not requested"
            )
        if outcome.opponent_id not in opponent_set:
            raise ValueError(
                f"outcome opponent {outcome.opponent_id!r} has no target meta mass"
            )
        key = (outcome.candidate_id, outcome.opponent_id)
        wins, draws, losses = aggregated.get(key, (0, 0, 0))
        aggregated[key] = (
            wins + outcome.wins,
            draws + outcome.draws,
            losses + outcome.losses,
        )
    return aggregated


def _draw_score_samples(
    *,
    candidates: tuple[str, ...],
    opponents: tuple[str, ...],
    weights: Mapping[str, float],
    self_cells: Mapping[str, str | None],
    fixed_matchup_scores: Mapping[tuple[str, str], float],
    outcomes: Mapping[tuple[str, str], tuple[int, int, int]],
    config: PosteriorEvaluationConfig,
) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(config.seed)
    distribution = np.asarray([weights[opponent_id] for opponent_id in opponents])
    dirichlet_alpha = distribution * config.meta_concentration
    deploy = np.empty((config.sample_count, len(candidates)), dtype=np.float64)
    robust = np.empty_like(deploy)
    for start in range(0, config.sample_count, config.sample_batch_size):
        stop = min(start + config.sample_batch_size, config.sample_count)
        batch_size = stop - start
        meta_draws = rng.dirichlet(dirichlet_alpha, size=batch_size)
        for candidate_index, candidate_id in enumerate(candidates):
            matchup_draws = np.empty(
                (batch_size, len(opponents)),
                dtype=np.float64,
            )
            for opponent_index, opponent_id in enumerate(opponents):
                fixed_score = fixed_matchup_scores.get((candidate_id, opponent_id))
                if fixed_score is not None:
                    matchup_draws[:, opponent_index] = fixed_score
                    continue
                if self_cells[candidate_id] == opponent_id:
                    matchup_draws[:, opponent_index] = 0.5
                    continue
                wins, draws, losses = outcomes.get(
                    (candidate_id, opponent_id),
                    (0, 0, 0),
                )
                alpha = config.beta_prior_alpha + wins + 0.5 * draws
                beta = config.beta_prior_beta + losses + 0.5 * draws
                matchup_draws[:, opponent_index] = rng.beta(
                    alpha,
                    beta,
                    size=batch_size,
                )
            deploy[start:stop, candidate_index] = np.sum(
                meta_draws * matchup_draws,
                axis=1,
            )
            robust[start:stop, candidate_index] = _weighted_lower_tail(
                matchup_draws,
                meta_draws,
                tail_mass=config.cvar_quantile,
            )
    return deploy, robust


def _weighted_lower_tail(
    values: FloatArray,
    weights: FloatArray,
    *,
    tail_mass: float,
) -> FloatArray:
    """Return the weighted mean of the lowest-valued opponent mass per row."""
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = np.take_along_axis(weights, order, axis=1)
    mass_before = np.cumsum(sorted_weights, axis=1) - sorted_weights
    included_mass = np.minimum(
        sorted_weights,
        np.maximum(tail_mass - mass_before, 0.0),
    )
    result: FloatArray = np.asarray(
        np.sum(sorted_values * included_mass, axis=1) / tail_mass,
        dtype=np.float64,
    )
    return result


def _summarize_candidates(
    *,
    candidates: tuple[str, ...],
    opponents: tuple[str, ...],
    weights: Mapping[str, float],
    self_cells: Mapping[str, str | None],
    fixed_matchup_scores: Mapping[tuple[str, str], float],
    outcomes: Mapping[tuple[str, str], tuple[int, int, int]],
    deploy_samples: FloatArray,
    robust_samples: FloatArray,
    config: PosteriorEvaluationConfig,
) -> tuple[BundlePosteriorSummary, ...]:
    lower_quantile = (1.0 - config.credible_mass) / 2.0
    upper_quantile = 1.0 - lower_quantile
    best_scores = np.max(deploy_samples, axis=1)
    probability_best = _rank_membership_probability(deploy_samples, top_k=1)
    effective_top_k = min(config.top_k, len(candidates))
    probability_top_k = _rank_membership_probability(
        deploy_samples,
        top_k=effective_top_k,
    )
    means = np.mean(deploy_samples, axis=0)
    ranking = np.argsort(-means, kind="stable")
    ranks = {
        int(candidate_index): rank for rank, candidate_index in enumerate(ranking, 1)
    }
    summaries: list[BundlePosteriorSummary] = []
    for candidate_index, candidate_id in enumerate(candidates):
        samples = deploy_samples[:, candidate_index]
        matchup_cvar = robust_samples[:, candidate_index]
        self_opponent = self_cells[candidate_id]
        fixed_opponents = {
            opponent_id
            for fixed_candidate, opponent_id in fixed_matchup_scores
            if fixed_candidate == candidate_id
        }
        if self_opponent is not None:
            fixed_opponents.add(self_opponent)
        observed_opponents = {
            opponent_id
            for opponent_id in opponents
            if opponent_id not in fixed_opponents
            and (candidate_id, opponent_id) in outcomes
        }
        observed_meta_mass = _clamp_probability(
            sum(weights[key] for key in observed_opponents)
        )
        self_meta_mass = _clamp_probability(
            sum(weights[opponent_id] for opponent_id in fixed_opponents)
        )
        prior_only_meta_mass = _clamp_probability(
            1.0 - observed_meta_mass - self_meta_mass
        )
        games = 0
        self_games_ignored = 0
        for opponent_id in opponents:
            wins, draws, losses = outcomes.get(
                (candidate_id, opponent_id),
                (0, 0, 0),
            )
            cell_games = wins + draws + losses
            if opponent_id in fixed_opponents:
                self_games_ignored += cell_games
            else:
                games += cell_games
        summaries.append(
            BundlePosteriorSummary(
                candidate_id=candidate_id,
                rank=ranks[candidate_index],
                games=games,
                self_games_ignored=self_games_ignored,
                observed_matchups=len(observed_opponents),
                prior_only_matchups=(
                    len(opponents) - len(observed_opponents) - len(fixed_opponents)
                ),
                observed_meta_mass=observed_meta_mass,
                self_meta_mass=self_meta_mass,
                prior_only_meta_mass=prior_only_meta_mass,
                deploy_mean=float(means[candidate_index]),
                deploy_standard_deviation=float(np.std(samples)),
                deploy_credible_low=float(np.quantile(samples, lower_quantile)),
                deploy_credible_high=float(np.quantile(samples, upper_quantile)),
                deploy_lcb=float(np.quantile(samples, config.lcb_quantile)),
                deploy_uncertainty_cvar=_lower_tail_mean(
                    samples,
                    mass=config.cvar_quantile,
                ),
                matchup_cvar_mean=float(np.mean(matchup_cvar)),
                matchup_cvar_credible_low=float(
                    np.quantile(matchup_cvar, lower_quantile)
                ),
                matchup_cvar_credible_high=float(
                    np.quantile(matchup_cvar, upper_quantile)
                ),
                probability_best=float(probability_best[candidate_index]),
                probability_top_k=float(probability_top_k[candidate_index]),
                probability_above_even=float(np.mean(samples > 0.5)),
                expected_regret=float(np.mean(best_scores - samples)),
            )
        )
    summaries.sort(key=lambda summary: summary.rank)
    return tuple(summaries)


def _lower_tail_mean(samples: FloatArray, *, mass: float) -> float:
    """Calculate empirical lower-tail CVaR with fractional boundary weight."""
    ordered = np.sort(samples)
    target = mass * len(ordered)
    whole_count = int(math.floor(target))
    fractional_count = target - whole_count
    total = float(np.sum(ordered[:whole_count]))
    if fractional_count > 0.0:
        total += fractional_count * float(ordered[whole_count])
    return total / target


def _summarize_portfolios(
    *,
    candidates: tuple[str, ...],
    deploy_samples: FloatArray,
    candidate_probability_best: Mapping[str, float],
    config: PosteriorEvaluationConfig,
) -> tuple[BundlePortfolioSummary, ...]:
    """Summarize every unordered pair from shared deployment draws."""
    lower_quantile = (1.0 - config.credible_mass) / 2.0
    upper_quantile = 1.0 - lower_quantile
    candidate_means = np.mean(deploy_samples, axis=0)
    global_best_scores = np.max(deploy_samples, axis=1)
    rows: list[BundlePortfolioSummary] = []
    for left_index, left_id in enumerate(candidates):
        left_samples = deploy_samples[:, left_index]
        for right_index in range(left_index + 1, len(candidates)):
            right_id = candidates[right_index]
            right_samples = deploy_samples[:, right_index]
            best_samples = np.maximum(left_samples, right_samples)
            expected_best = float(np.mean(best_samples))
            best_single_mean = max(
                float(candidate_means[left_index]),
                float(candidate_means[right_index]),
            )
            rows.append(
                BundlePortfolioSummary(
                    rank=1,
                    candidate_ids=(left_id, right_id),
                    expected_best_score=expected_best,
                    standard_deviation=float(np.std(best_samples)),
                    credible_low=float(np.quantile(best_samples, lower_quantile)),
                    credible_high=float(np.quantile(best_samples, upper_quantile)),
                    best_score_lcb=float(
                        np.quantile(best_samples, config.lcb_quantile)
                    ),
                    probability_best=_clamp_probability(
                        candidate_probability_best[left_id]
                        + candidate_probability_best[right_id]
                    ),
                    probability_at_least_one_above_even=float(
                        np.mean(best_samples > 0.5)
                    ),
                    joint_downside_probability=float(
                        np.mean((left_samples <= 0.5) & (right_samples <= 0.5))
                    ),
                    diversification_gain=_clamp_probability(
                        expected_best - best_single_mean
                    ),
                    score_correlation=_score_correlation(
                        left_samples,
                        right_samples,
                    ),
                    expected_regret=float(np.mean(global_best_scores - best_samples)),
                )
            )
    rows.sort(
        key=lambda row: (
            -row.expected_best_score,
            row.candidate_ids,
        )
    )
    return tuple(
        row.model_copy(update={"rank": rank}) for rank, row in enumerate(rows, start=1)
    )


def _score_correlation(
    left_samples: FloatArray,
    right_samples: FloatArray,
) -> float | None:
    """Return Pearson correlation, or none when either score is constant."""
    left_centered = left_samples - np.mean(left_samples)
    right_centered = right_samples - np.mean(right_samples)
    left_squared = float(np.dot(left_centered, left_centered))
    right_squared = float(np.dot(right_centered, right_centered))
    if left_squared == 0.0 or right_squared == 0.0:
        return None
    correlation = float(
        np.dot(left_centered, right_centered) / math.sqrt(left_squared * right_squared)
    )
    return min(1.0, max(-1.0, correlation))


def _summarize_contrasts(
    *,
    candidates: tuple[str, ...],
    deploy_samples: FloatArray,
    config: PosteriorEvaluationConfig,
) -> tuple[BundlePosteriorContrast, ...]:
    """Summarize correlated differences from the shared posterior draws."""
    lower_quantile = (1.0 - config.credible_mass) / 2.0
    upper_quantile = 1.0 - lower_quantile
    rows: list[BundlePosteriorContrast] = []
    for candidate_index, candidate_id in enumerate(candidates):
        for reference_index, reference_id in enumerate(candidates):
            if candidate_id == reference_id:
                continue
            delta = (
                deploy_samples[:, candidate_index] - deploy_samples[:, reference_index]
            )
            rows.append(
                BundlePosteriorContrast(
                    candidate_id=candidate_id,
                    reference_id=reference_id,
                    mean_delta=float(np.mean(delta)),
                    standard_deviation=float(np.std(delta)),
                    credible_low=float(np.quantile(delta, lower_quantile)),
                    credible_high=float(np.quantile(delta, upper_quantile)),
                    probability_above_zero=float(np.mean(delta > 0.0)),
                    probability_noninferior_2pp=float(np.mean(delta > -0.02)),
                )
            )
    return tuple(rows)


def _rank_membership_probability(
    scores: FloatArray,
    *,
    top_k: int,
) -> FloatArray:
    """Return fair top-k probabilities, splitting exact boundary ties."""
    probabilities = np.empty(scores.shape[1], dtype=np.float64)
    for candidate_index in range(scores.shape[1]):
        candidate_scores = scores[:, candidate_index]
        is_tied = np.isclose(
            scores,
            candidate_scores[:, None],
            rtol=0.0,
            atol=1e-12,
        )
        greater = np.sum(
            scores > candidate_scores[:, None] + 1e-12,
            axis=1,
        )
        tied = np.sum(is_tied, axis=1)
        membership = np.clip((top_k - greater) / tied, 0.0, 1.0)
        probabilities[candidate_index] = np.mean(membership)
    return probabilities


def _clamp_probability(value: float) -> float:
    """Remove harmless floating-point drift from a probability sum."""
    return min(1.0, max(0.0, value))
