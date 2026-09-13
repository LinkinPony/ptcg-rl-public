"""Public, fixed-width evidence from complete-action engine search.

This module is deliberately independent of agent search implementations.  It
contains only detached aggregates that are safe to persist in trajectories or
pass to a policy.  Hidden-world identities, engine states, continuation trees,
and raw observations must remain on the producer side of this boundary.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

SEARCH_EVIDENCE_MEAN_SCORE_INDEX = 0
SEARCH_EVIDENCE_SCORE_STD_INDEX = 1
SEARCH_EVIDENCE_ROBUST_SCORE_INDEX = 2
SEARCH_EVIDENCE_DOWNSIDE_CVAR_INDEX = 3
SEARCH_EVIDENCE_MIN_SCORE_INDEX = 4
SEARCH_EVIDENCE_MAX_SCORE_INDEX = 5
SEARCH_EVIDENCE_COVERAGE_INDEX = 6
SEARCH_EVIDENCE_TERMINAL_FRACTION_INDEX = 7
SEARCH_EVIDENCE_SAME_SEAT_MAIN_FRACTION_INDEX = 8
SEARCH_EVIDENCE_TURN_HANDOFF_FRACTION_INDEX = 9
SEARCH_EVIDENCE_MEAN_PATH_STEPS_INDEX = 10
SEARCH_EVIDENCE_EXACT_INDEX = 11
SEARCH_EVIDENCE_FEATURE_SIZE = 12

_PATH_STEP_NORMALIZER = 16.0
_FRACTION_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class SearchCandidateEvidence:
    """One complete root action and its public aggregate search features."""

    action: tuple[int, ...]
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        """Freeze and validate the complete action and fixed-width feature row."""
        action = tuple(int(index) for index in self.action)
        features = tuple(float(value) for value in self.features)
        if action != self.action:
            object.__setattr__(self, "action", action)
        if features != self.features:
            object.__setattr__(self, "features", features)
        if any(index < 0 for index in action):
            raise ValueError("search evidence action indices must be non-negative")
        if len(set(action)) != len(action):
            raise ValueError("search evidence action indices must be unique")
        if len(features) != SEARCH_EVIDENCE_FEATURE_SIZE:
            raise ValueError(
                "search evidence feature rows must have "
                f"{SEARCH_EVIDENCE_FEATURE_SIZE} values"
            )
        if any(not math.isfinite(value) for value in features):
            raise ValueError("search evidence features must be finite")
        _validate_unit_interval(
            features[SEARCH_EVIDENCE_COVERAGE_INDEX],
            "search evidence coverage",
        )
        for index, label in (
            (SEARCH_EVIDENCE_TERMINAL_FRACTION_INDEX, "terminal fraction"),
            (
                SEARCH_EVIDENCE_SAME_SEAT_MAIN_FRACTION_INDEX,
                "same-seat MAIN fraction",
            ),
            (
                SEARCH_EVIDENCE_TURN_HANDOFF_FRACTION_INDEX,
                "turn-handoff fraction",
            ),
            (SEARCH_EVIDENCE_MEAN_PATH_STEPS_INDEX, "mean path steps"),
            (SEARCH_EVIDENCE_EXACT_INDEX, "exact flag"),
        ):
            _validate_unit_interval(features[index], f"search evidence {label}")
        endpoint_fraction_sum = sum(
            features[index]
            for index in (
                SEARCH_EVIDENCE_TERMINAL_FRACTION_INDEX,
                SEARCH_EVIDENCE_SAME_SEAT_MAIN_FRACTION_INDEX,
                SEARCH_EVIDENCE_TURN_HANDOFF_FRACTION_INDEX,
            )
        )
        if not math.isclose(
            endpoint_fraction_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=_FRACTION_TOLERANCE,
        ):
            raise ValueError("search evidence endpoint fractions must sum to one")
        exact = features[SEARCH_EVIDENCE_EXACT_INDEX]
        if exact not in (0.0, 1.0):
            raise ValueError("search evidence exact flag must be zero or one")


@dataclass(frozen=True)
class SearchEvidence:
    """Ordered candidate subset and public aggregate complete-search evidence."""

    candidates: tuple[SearchCandidateEvidence, ...]
    legal_action_count: int
    world_count: int
    exhaustive: bool
    exact: bool

    def __post_init__(self) -> None:
        """Reject incomplete metadata and duplicate complete actions."""
        candidates = tuple(self.candidates)
        if candidates != self.candidates:
            object.__setattr__(self, "candidates", candidates)
        if not candidates:
            raise ValueError("search evidence must contain at least one candidate")
        actions = tuple(candidate.action for candidate in candidates)
        if len(set(actions)) != len(actions):
            raise ValueError("search evidence candidate actions must be unique")
        if self.legal_action_count < len(candidates):
            raise ValueError(
                "search evidence legal action count cannot be below candidate count"
            )
        if self.world_count <= 0:
            raise ValueError("search evidence world count must be positive")
        if self.exhaustive != (self.legal_action_count == len(candidates)):
            raise ValueError(
                "search evidence exhaustive flag must match candidate coverage"
            )
        if self.exact and not self.exhaustive:
            raise ValueError("exact search evidence must be root-exhaustive")
        expected_exact = 1.0 if self.exact else 0.0
        if any(
            candidate.features[SEARCH_EVIDENCE_EXACT_INDEX] != expected_exact
            for candidate in candidates
        ):
            raise ValueError(
                "search evidence feature exact flags must match record metadata"
            )

    @property
    def actions(self) -> tuple[tuple[int, ...], ...]:
        """Return complete actions in the producer's retained candidate order."""
        return tuple(candidate.action for candidate in self.candidates)

    @property
    def feature_rows(self) -> tuple[tuple[float, ...], ...]:
        """Return fixed-width feature rows aligned with :attr:`actions`."""
        return tuple(candidate.features for candidate in self.candidates)


def search_candidate_features(
    *,
    world_scores: Sequence[float],
    robust_score: float,
    coverage: float,
    terminal_fraction: float,
    same_seat_main_fraction: float,
    turn_handoff_fraction: float,
    mean_path_steps: float,
    exact: bool,
) -> tuple[float, ...]:
    """Build the canonical 12-value row from detached search aggregates.

    Downside CVaR is the mean of the lower half of equally weighted sampled
    worlds.  Path length is clipped after division by a fixed 16-selection
    horizon so every structural feature remains bounded across producers.
    """
    scores = tuple(float(value) for value in world_scores)
    if not scores or any(not math.isfinite(value) for value in scores):
        raise ValueError("search evidence world scores must be finite and non-empty")
    robust = float(robust_score)
    if not math.isfinite(robust):
        raise ValueError("search evidence robust score must be finite")
    fractions = (
        float(terminal_fraction),
        float(same_seat_main_fraction),
        float(turn_handoff_fraction),
    )
    for value, label in zip(
        fractions,
        ("terminal", "same-seat MAIN", "turn-handoff"),
        strict=True,
    ):
        _validate_unit_interval(value, f"search evidence {label} fraction")
    if not math.isclose(
        sum(fractions),
        1.0,
        rel_tol=0.0,
        abs_tol=_FRACTION_TOLERANCE,
    ):
        raise ValueError("search evidence endpoint fractions must sum to one")
    normalized_coverage = float(coverage)
    _validate_unit_interval(normalized_coverage, "search evidence coverage")
    steps = float(mean_path_steps)
    if not math.isfinite(steps) or steps < 0.0:
        raise ValueError(
            "search evidence mean path steps must be finite and non-negative"
        )

    ordered_scores = sorted(scores)
    downside_count = max(1, (len(ordered_scores) + 1) // 2)
    return (
        float(statistics.fmean(scores)),
        float(statistics.pstdev(scores)),
        robust,
        float(statistics.fmean(ordered_scores[:downside_count])),
        ordered_scores[0],
        ordered_scores[-1],
        normalized_coverage,
        fractions[0],
        fractions[1],
        fractions[2],
        min(steps / _PATH_STEP_NORMALIZER, 1.0),
        1.0 if exact else 0.0,
    )


def _validate_unit_interval(value: float, label: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")


__all__ = [
    "SEARCH_EVIDENCE_COVERAGE_INDEX",
    "SEARCH_EVIDENCE_DOWNSIDE_CVAR_INDEX",
    "SEARCH_EVIDENCE_EXACT_INDEX",
    "SEARCH_EVIDENCE_FEATURE_SIZE",
    "SEARCH_EVIDENCE_MAX_SCORE_INDEX",
    "SEARCH_EVIDENCE_MEAN_PATH_STEPS_INDEX",
    "SEARCH_EVIDENCE_MEAN_SCORE_INDEX",
    "SEARCH_EVIDENCE_MIN_SCORE_INDEX",
    "SEARCH_EVIDENCE_ROBUST_SCORE_INDEX",
    "SEARCH_EVIDENCE_SAME_SEAT_MAIN_FRACTION_INDEX",
    "SEARCH_EVIDENCE_SCORE_STD_INDEX",
    "SEARCH_EVIDENCE_TERMINAL_FRACTION_INDEX",
    "SEARCH_EVIDENCE_TURN_HANDOFF_FRACTION_INDEX",
    "SearchCandidateEvidence",
    "SearchEvidence",
    "search_candidate_features",
]
