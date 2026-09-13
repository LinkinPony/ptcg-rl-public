"""Dependency-light paired estimators for temperature gauntlet contrasts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ptcg_rl.evaluation.native_checkpoint_gauntlet.paired_validation import (
    ValidatedPair,
)

_Z_975 = 1.959963984540054


def build_statistics(
    pair: ValidatedPair,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Compute the prespecified macro effect and bounded secondary analyses."""
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be non-negative")
    main = _effect_summary(
        pair.control_scores,
        pair.treatment_scores,
        pair.baseline_deck_ids,
        pair.candidate_seats,
    )
    incomplete_pairs = int(
        np.count_nonzero(
            np.isnan(pair.planned_control_scores)
            | np.isnan(pair.planned_treatment_scores)
        )
    )
    main["analysis_population"] = (
        "all scheduled pairs"
        if incomplete_pairs == 0
        else "pairs with resolved outcomes in both arms"
    )
    main["strict_complete_schedule_inference"] = incomplete_pairs == 0
    main["excluded_incomplete_pairs"] = incomplete_pairs
    deltas = pair.treatment_scores - pair.control_scores
    main["bootstrap"] = _stratified_bootstrap(
        deltas,
        pair.baseline_deck_ids,
        pair.candidate_seats,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    main["exact_mcnemar"] = _mcnemar(pair.control_scores, pair.treatment_scores)
    sensitivity = _excluding_self(
        pair,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "estimand": {
            "name": "baseline_exact_deck_macro_paired_score_difference",
            "direction": "treatment_minus_control",
            "outcome": "candidate W/D/L score (1/0.5/0)",
            "weighting": "equal baseline exact-deck weight and equal candidate-seat weight",
            "inferential_strata": ["baseline_deck_id", "candidate_seat"],
            "missing_outcome_policy": (
                "complete-pair conditional inference plus finite-schedule sharp "
                "endpoints when a verified max-steps outcome is unresolved"
            ),
        },
        "main_exact_deck_macro": main,
        "full_schedule_outcome_bounds": _full_schedule_outcome_bounds(pair),
        "excluding_self_sensitivity": sensitivity,
        "logical_seat_secondary": _logical_seat_secondary(pair),
        "deck_exploratory": _deck_exploratory(pair),
        "first_player_descriptive": _first_player_descriptive(pair),
    }


def _logical_seat_secondary(pair: ValidatedPair) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seat in (0, 1):
        mask = pair.candidate_seats == seat
        rows.append(
            {
                "candidate_logical_seat": seat,
                **_effect_summary(
                    pair.control_scores[mask],
                    pair.treatment_scores[mask],
                    pair.baseline_deck_ids[mask],
                    pair.candidate_seats[mask],
                ),
            }
        )
    return {
        "status": "secondary; logical seat is preassigned but no interaction was prespecified",
        "seats": rows,
    }


def _full_schedule_outcome_bounds(pair: ValidatedPair) -> dict[str, Any]:
    """Return sharp score bounds without assuming why a scheduled outcome is missing."""
    control = pair.planned_control_scores
    treatment = pair.planned_treatment_scores
    control_missing = np.isnan(control)
    treatment_missing = np.isnan(treatment)
    groups = _strata(pair.planned_baseline_deck_ids, pair.planned_candidate_seats)
    control_low = np.where(control_missing, 0.0, control)
    control_high = np.where(control_missing, 1.0, control)
    treatment_low = np.where(treatment_missing, 0.0, treatment)
    treatment_high = np.where(treatment_missing, 1.0, treatment)
    difference_low = treatment_low - control_high
    difference_high = treatment_high - control_low
    incomplete = control_missing | treatment_missing
    return {
        "method": (
            "sharp W/D/L score bounds over the original balanced schedule; each "
            "unresolved arm score may take any value in [0, 1]"
        ),
        "statistical_uncertainty_included": False,
        "scheduled_pairs": len(control),
        "incomplete_pairs": int(np.count_nonzero(incomplete)),
        "control_unresolved_games": int(np.count_nonzero(control_missing)),
        "treatment_unresolved_games": int(np.count_nonzero(treatment_missing)),
        "control_macro_score_rate_low": _mean_over_strata(control_low, groups),
        "control_macro_score_rate_high": _mean_over_strata(control_high, groups),
        "treatment_macro_score_rate_low": _mean_over_strata(treatment_low, groups),
        "treatment_macro_score_rate_high": _mean_over_strata(treatment_high, groups),
        "paired_difference_low": _mean_over_strata(difference_low, groups),
        "paired_difference_high": _mean_over_strata(difference_high, groups),
    }


def _effect_summary(
    control: np.ndarray,
    treatment: np.ndarray,
    deck_ids: np.ndarray,
    seats: np.ndarray,
) -> dict[str, Any]:
    groups = _strata(deck_ids, seats)
    deltas = treatment - control
    estimate = _mean_over_strata(deltas, groups)
    variance = 0.0
    for indices in groups:
        values = deltas[indices]
        centered = values - float(np.mean(values))
        variance += float(centered @ centered) / (len(values) * (len(values) - 1))
    variance /= len(groups) ** 2
    standard_error = math.sqrt(max(0.0, variance))
    z_score, p_value = _normal_test(estimate, standard_error)
    return {
        "paired_games": len(deltas),
        "baseline_exact_decks": len({str(value) for value in deck_ids}),
        "strata": len(groups),
        "control_macro_score_rate": _mean_over_strata(control, groups),
        "treatment_macro_score_rate": _mean_over_strata(treatment, groups),
        "paired_difference": estimate,
        "stratified_sandwich": {
            "method": "equal-stratum mean with within-stratum empirical sandwich variance",
            "standard_error": standard_error,
            "z_score": z_score,
            "two_sided_p_value": p_value,
            "ci95_low": estimate - _Z_975 * standard_error,
            "ci95_high": estimate + _Z_975 * standard_error,
        },
    }


def _stratified_bootstrap(
    values: np.ndarray,
    deck_ids: np.ndarray,
    seats: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    groups = _strata(deck_ids, seats)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    batch_size = 256
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        batch = np.zeros(size, dtype=np.float64)
        for indices in groups:
            sampled = rng.integers(0, len(indices), size=(size, len(indices)))
            batch += np.mean(values[indices[sampled]], axis=1)
        draws[start : start + size] = batch / len(groups)
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "method": "fixed-seed percentile bootstrap resampled within each inferential stratum",
        "rng": "numpy.default_rng",
        "seed": seed,
        "replicates": replicates,
        "mean": float(np.mean(draws)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _mcnemar(control: np.ndarray, treatment: np.ndarray) -> dict[str, Any]:
    draw_pairs = int(np.count_nonzero((control == 0.5) | (treatment == 0.5)))
    if draw_pairs:
        return {
            "available": False,
            "reason": "exact McNemar is reported only when neither arm contains draws",
            "draw_pairs": draw_pairs,
        }
    loss_to_win = int(np.count_nonzero((control == 0.0) & (treatment == 1.0)))
    win_to_loss = int(np.count_nonzero((control == 1.0) & (treatment == 0.0)))
    discordant = loss_to_win + win_to_loss
    return {
        "available": True,
        "method": "exact two-sided McNemar conditional binomial test",
        "control_loss_treatment_win": loss_to_win,
        "control_win_treatment_loss": win_to_loss,
        "discordant_pairs": discordant,
        "two_sided_p_value": _exact_binomial_two_sided(loss_to_win, win_to_loss),
    }


def _excluding_self(
    pair: ValidatedPair,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    self_mask = pair.baseline_deck_ids == pair.candidate_deck_ids
    excluded_ids = sorted({str(value) for value in pair.baseline_deck_ids[self_mask]})
    if not excluded_ids:
        return {
            "available": False,
            "reason": "candidate has no self matchup in baseline roster",
        }
    keep = ~self_mask
    if not np.any(keep):
        return {
            "available": False,
            "reason": "excluding self removes every paired game",
        }
    result = _effect_summary(
        pair.control_scores[keep],
        pair.treatment_scores[keep],
        pair.baseline_deck_ids[keep],
        pair.candidate_seats[keep],
    )
    result["available"] = True
    result["excluded_deck_ids"] = excluded_ids
    result["excluded_pairs"] = int(np.count_nonzero(self_mask))
    result["bootstrap"] = _stratified_bootstrap(
        pair.treatment_scores[keep] - pair.control_scores[keep],
        pair.baseline_deck_ids[keep],
        pair.candidate_seats[keep],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return result


def _deck_exploratory(pair: ValidatedPair) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for deck_id in sorted(pair.deck_metadata):
        mask = pair.baseline_deck_ids == deck_id
        effect = _effect_summary(
            pair.control_scores[mask],
            pair.treatment_scores[mask],
            pair.baseline_deck_ids[mask],
            pair.candidate_seats[mask],
        )
        sandwich = effect["stratified_sandwich"]
        metadata = pair.deck_metadata[deck_id]
        rows.append(
            {
                **metadata,
                "self_matchup": bool(np.any(pair.candidate_deck_ids[mask] == deck_id)),
                **effect,
                "exact_mcnemar": _mcnemar(
                    pair.control_scores[mask], pair.treatment_scores[mask]
                ),
                "bh_input_p_value": sandwich["two_sided_p_value"],
            }
        )
    adjusted = _benjamini_hochberg([float(row["bh_input_p_value"]) for row in rows])
    for row, q_value in zip(rows, adjusted, strict=True):
        row["bh_q_value"] = q_value
    return {
        "status": "exploratory",
        "multiple_testing": "Benjamini-Hochberg over sandwich two-sided z p-values",
        "decks": rows,
    }


def _first_player_descriptive(pair: ValidatedPair) -> dict[str, Any]:
    control_first = pair.control_candidate_went_first
    treatment_first = pair.treatment_candidate_went_first
    complete = (control_first >= 0) & (treatment_first >= 0)
    return {
        "inferential_role": (
            "none; actual first player may be affected by the temperature treatment"
        ),
        "analysis_population": (
            "first-player rates use the full schedule; conditional score rates exclude "
            "unresolved outcomes"
        ),
        "control": _arm_first_player_summary(
            pair.planned_control_scores, control_first
        ),
        "treatment": _arm_first_player_summary(
            pair.planned_treatment_scores, treatment_first
        ),
        "paired_cross_tab": {
            "both_first": int(
                np.count_nonzero(
                    complete & (control_first == 1) & (treatment_first == 1)
                )
            ),
            "both_second": int(
                np.count_nonzero(
                    complete & (control_first == 0) & (treatment_first == 0)
                )
            ),
            "control_first_treatment_second": int(
                np.count_nonzero(
                    complete & (control_first == 1) & (treatment_first == 0)
                )
            ),
            "control_second_treatment_first": int(
                np.count_nonzero(
                    complete & (control_first == 0) & (treatment_first == 1)
                )
            ),
            "one_or_both_missing": int(np.count_nonzero(~complete)),
        },
    }


def _arm_first_player_summary(
    scores: np.ndarray, went_first: np.ndarray
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for value, label in ((1, "candidate_went_first"), (0, "candidate_went_second")):
        scheduled_mask = went_first == value
        scored_mask = scheduled_mask & np.isfinite(scores)
        if np.any(scheduled_mask):
            groups.append(
                {
                    "group": label,
                    "scheduled_games": int(np.count_nonzero(scheduled_mask)),
                    "resolved_score_games": int(np.count_nonzero(scored_mask)),
                    "score_rate": (
                        None
                        if not np.any(scored_mask)
                        else float(np.mean(scores[scored_mask]))
                    ),
                }
            )
    return {
        "missing_games": int(np.count_nonzero(went_first == -1)),
        "candidate_went_first_rate_among_observed": (
            None
            if not np.any(went_first >= 0)
            else float(np.mean(went_first[went_first >= 0]))
        ),
        "groups": groups,
    }


def _strata(deck_ids: np.ndarray, seats: np.ndarray) -> list[np.ndarray]:
    keys = sorted(
        {(str(deck), int(seat)) for deck, seat in zip(deck_ids, seats, strict=True)}
    )
    return [np.flatnonzero((deck_ids == deck) & (seats == seat)) for deck, seat in keys]


def _mean_over_strata(values: np.ndarray, groups: list[np.ndarray]) -> float:
    return float(np.mean([float(np.mean(values[indices])) for indices in groups]))


def _normal_test(estimate: float, standard_error: float) -> tuple[float | None, float]:
    if standard_error == 0.0:
        return None, 1.0 if estimate == 0.0 else 0.0
    z_score = estimate / standard_error
    return z_score, math.erfc(abs(z_score) / math.sqrt(2.0))


def _exact_binomial_two_sided(first: int, second: int) -> float:
    trials = first + second
    if trials == 0:
        return 1.0
    cutoff = min(first, second)
    logs = [
        math.lgamma(trials + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(trials - successes + 1)
        - trials * math.log(2.0)
        for successes in range(cutoff + 1)
    ]
    maximum = max(logs)
    log_tail = maximum + math.log(sum(math.exp(value - maximum) for value in logs))
    return min(1.0, math.exp(math.log(2.0) + log_tail))


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = len(p_values) - reverse_rank + 1
        running = min(running, p_values[index] * len(p_values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted
