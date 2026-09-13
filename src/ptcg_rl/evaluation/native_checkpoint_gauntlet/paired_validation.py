"""Load and validate common-random-number gauntlet artifacts."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ptcg_rl.evaluation.native_checkpoint_gauntlet.paired_provenance import (
    read_manifest,
    read_summary,
    validate_manifests,
    validate_summaries,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.paired_rows import (
    candidate_went_first,
    games_path,
    index_matches,
    is_resolved,
    read_rows,
    score,
    validate_artifact,
    validate_balance,
    validate_pair_fields,
    validate_rosters,
)


@dataclass(frozen=True)
class ValidatedPair:
    """Aligned observations and immutable metadata for paired inference."""

    control_path: Path
    treatment_path: Path
    control_scores: np.ndarray
    treatment_scores: np.ndarray
    baseline_deck_ids: np.ndarray
    candidate_deck_ids: np.ndarray
    candidate_seats: np.ndarray
    planned_control_scores: np.ndarray
    planned_treatment_scores: np.ndarray
    planned_baseline_deck_ids: np.ndarray
    planned_candidate_seats: np.ndarray
    control_candidate_went_first: np.ndarray
    treatment_candidate_went_first: np.ndarray
    deck_metadata: dict[str, dict[str, str]]
    provenance: dict[str, Any]
    validation: dict[str, Any]


def load_validated_pair(
    control_path: Path,
    treatment_path: Path,
    *,
    control_temperature: float | None = None,
    treatment_temperature: float | None = None,
    baseline_temperature: float = 0.0,
    expected_decks: int | None = None,
    expected_repeats: int | None = None,
    maximum_incomplete_pairs: int = 0,
) -> ValidatedPair:
    """Load, align, and fail closed on any violation of paired provenance."""
    if (
        isinstance(maximum_incomplete_pairs, bool)
        or not isinstance(maximum_incomplete_pairs, int)
        or maximum_incomplete_pairs < 0
    ):
        raise ValueError("maximum_incomplete_pairs must be a non-negative integer")
    control_games = games_path(control_path)
    treatment_games = games_path(treatment_path)
    control_summary = read_summary(control_games)
    treatment_summary = read_summary(treatment_games)
    summary_provenance = validate_summaries(
        control_summary,
        treatment_summary,
        control_temperature=control_temperature,
        treatment_temperature=treatment_temperature,
        baseline_temperature=baseline_temperature,
    )
    control_manifest = read_manifest(control_games)
    treatment_manifest = read_manifest(treatment_games)
    manifest_provenance = validate_manifests(
        control_manifest,
        treatment_manifest,
        control_summary=control_summary,
        treatment_summary=treatment_summary,
    )
    provenance = {**summary_provenance, **manifest_provenance}
    maximum_engine_steps = int(manifest_provenance["maximum_engine_steps"])
    control_rows = read_rows(control_games)
    treatment_rows = read_rows(treatment_games)
    allow_unresolved = maximum_incomplete_pairs > 0
    validate_artifact(
        control_rows,
        control_summary,
        arm="control",
        allow_unresolved=allow_unresolved,
        maximum_engine_steps=maximum_engine_steps,
    )
    validate_artifact(
        treatment_rows,
        treatment_summary,
        arm="treatment",
        allow_unresolved=allow_unresolved,
        maximum_engine_steps=maximum_engine_steps,
    )
    validate_rosters(
        control_rows,
        treatment_rows,
        control_summary,
        treatment_summary,
        control_manifest,
        treatment_manifest,
    )
    control_by_match = index_matches(control_rows, arm="control")
    treatment_by_match = index_matches(treatment_rows, arm="treatment")
    if control_by_match.keys() != treatment_by_match.keys():
        raise ValueError(
            "control and treatment do not have one-to-one match_id support"
        )

    ordered_ids = sorted(control_by_match)
    controls: list[float] = []
    treatments: list[float] = []
    baseline_ids: list[str] = []
    candidate_ids: list[str] = []
    seats: list[int] = []
    control_went_first: list[int] = []
    treatment_went_first: list[int] = []
    planned_controls: list[float] = []
    planned_treatments: list[float] = []
    planned_baseline_ids: list[str] = []
    planned_candidate_ids: list[str] = []
    planned_seats: list[int] = []
    unresolved_pairs: list[dict[str, Any]] = []
    deck_metadata: dict[str, dict[str, str]] = {}
    for match_id in ordered_ids:
        control = control_by_match[match_id]
        treatment = treatment_by_match[match_id]
        validate_pair_fields(control, treatment, match_id=match_id)
        baseline_id = str(control["baseline_deck_id"])
        candidate_id = str(control["candidate_deck_id"])
        seat = int(control["candidate_seat"])
        control_resolved = is_resolved(control)
        treatment_resolved = is_resolved(treatment)
        control_score = score(control, arm="control") if control_resolved else math.nan
        treatment_score = (
            score(treatment, arm="treatment") if treatment_resolved else math.nan
        )
        planned_controls.append(control_score)
        planned_treatments.append(treatment_score)
        planned_baseline_ids.append(baseline_id)
        planned_candidate_ids.append(candidate_id)
        planned_seats.append(seat)
        control_went_first.append(candidate_went_first(control))
        treatment_went_first.append(candidate_went_first(treatment))
        if control_resolved and treatment_resolved:
            controls.append(control_score)
            treatments.append(treatment_score)
            baseline_ids.append(baseline_id)
            candidate_ids.append(candidate_id)
            seats.append(seat)
        else:
            unresolved_pairs.append(
                {
                    "match_id": match_id,
                    "game_index": int(control["game_index"]),
                    "baseline_deck_id": baseline_id,
                    "baseline_deck_hash": str(control["baseline_deck_hash"]),
                    "candidate_seat": seat,
                    "control_resolved": control_resolved,
                    "control_terminal_reason": str(control["terminal_reason"]),
                    "treatment_resolved": treatment_resolved,
                    "treatment_terminal_reason": str(treatment["terminal_reason"]),
                }
            )
        deck_metadata[baseline_id] = {
            "deck_id": baseline_id,
            "deck_hash": str(control["baseline_deck_hash"]),
            "deck_label": str(control["baseline_deck_label"]),
        }

    if len(unresolved_pairs) > maximum_incomplete_pairs:
        raise ValueError(
            "paired unresolved support exceeds maximum_incomplete_pairs: "
            f"{len(unresolved_pairs)} > {maximum_incomplete_pairs}"
        )

    planned_deck_array = np.asarray(planned_baseline_ids, dtype=np.str_)
    planned_seat_array = np.asarray(planned_seats, dtype=np.int8)
    candidate_count = len(set(planned_candidate_ids))
    if candidate_count != 1:
        raise ValueError(
            "paired exact-deck macro analysis requires exactly one candidate deck"
        )
    if not set(planned_candidate_ids).issubset(set(planned_baseline_ids)):
        raise ValueError("candidate exact deck is absent from the baseline full roster")
    balance = validate_balance(
        planned_deck_array,
        planned_seat_array,
        expected_decks=expected_decks,
        expected_repeats=expected_repeats,
    )
    deck_array = np.asarray(baseline_ids, dtype=np.str_)
    seat_array = np.asarray(seats, dtype=np.int8)
    complete_counts = Counter(
        (str(deck_id), int(seat))
        for deck_id, seat in zip(deck_array, seat_array, strict=True)
    )
    planned_keys = {
        (str(deck_id), int(seat))
        for deck_id, seat in zip(
            planned_deck_array,
            planned_seat_array,
            strict=True,
        )
    }
    if set(complete_counts) != planned_keys or min(complete_counts.values()) < 2:
        raise ValueError(
            "resolved paired support must retain every planned stratum with at least "
            "two observations"
        )
    control_unresolved = sum(not is_resolved(row) for row in control_rows)
    treatment_unresolved = sum(not is_resolved(row) for row in treatment_rows)
    validation = {
        "scheduled_paired_games": len(ordered_ids),
        "analyzable_paired_games": len(controls),
        "unresolved_pair_games": len(unresolved_pairs),
        "maximum_incomplete_pairs": maximum_incomplete_pairs,
        "unresolved_pairs": unresolved_pairs,
        "candidate_exact_decks": candidate_count,
        **balance,
        "resolved_min_repeats_per_stratum": min(complete_counts.values()),
        "resolved_max_repeats_per_stratum": max(complete_counts.values()),
        "draw_pairs": sum(
            control == 0.5 or treatment == 0.5
            for control, treatment in zip(controls, treatments, strict=True)
        ),
        "control_actual_first_player_missing_games": control_went_first.count(-1),
        "treatment_actual_first_player_missing_games": treatment_went_first.count(-1),
        "all_static_fields_equal": True,
        "all_role_aligned_seeds_equal": True,
        "all_seeds_valid_uint32": True,
        "control_unresolved_games": control_unresolved,
        "treatment_unresolved_games": treatment_unresolved,
        "unresolved_or_infrastructure_games": (
            control_unresolved + treatment_unresolved
        ),
        "control_terminal_reason_counts": dict(
            sorted(Counter(str(row["terminal_reason"]) for row in control_rows).items())
        ),
        "treatment_terminal_reason_counts": dict(
            sorted(
                Counter(str(row["terminal_reason"]) for row in treatment_rows).items()
            )
        ),
    }
    return ValidatedPair(
        control_path=control_games,
        treatment_path=treatment_games,
        control_scores=np.asarray(controls, dtype=np.float64),
        treatment_scores=np.asarray(treatments, dtype=np.float64),
        baseline_deck_ids=deck_array,
        candidate_deck_ids=np.asarray(candidate_ids, dtype=np.str_),
        candidate_seats=seat_array,
        planned_control_scores=np.asarray(planned_controls, dtype=np.float64),
        planned_treatment_scores=np.asarray(planned_treatments, dtype=np.float64),
        planned_baseline_deck_ids=planned_deck_array,
        planned_candidate_seats=planned_seat_array,
        control_candidate_went_first=np.asarray(control_went_first, dtype=np.int8),
        treatment_candidate_went_first=np.asarray(treatment_went_first, dtype=np.int8),
        deck_metadata=deck_metadata,
        provenance=provenance,
        validation=validation,
    )
