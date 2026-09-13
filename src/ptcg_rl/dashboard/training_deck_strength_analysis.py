"""Selection and statistical helpers for observed training evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from statistics import NormalDist
from typing import Any

from ptcg_rl.dashboard.models import OutcomeStats
from ptcg_rl.dashboard.repository import DashboardRepository
from ptcg_rl.dashboard.training_deck_strength_models import (
    TrainingController,
    TrainingEvidenceRange,
    TrainingEvidenceRangeMetadata,
    TrainingPosterior,
)
from ptcg_rl.rl.performance_state import (
    KNOWN_OPPONENT_STRATA,
    OutcomeCounts,
    resolve_opponent_stratum,
)


@dataclass(frozen=True)
class TrainingHistory:
    """Normalized complete-window history used by all strength projections."""

    run_id: str
    payload: Mapping[str, Any]
    windows: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    targets: tuple[str, ...]
    stationary_kinds: frozenset[str]
    exact_available: bool


@dataclass(frozen=True)
class SelectedTrainingWindows:
    """One independently selected evidence range and its provenance."""

    windows: tuple[dict[str, Any], ...]
    metadata: TrainingEvidenceRangeMetadata


def select_training_windows(
    history: TrainingHistory,
    *,
    range_name: TrainingEvidenceRange,
    checkpoint_version: int | None,
) -> SelectedTrainingWindows:
    """Select a range without ever admitting mixed checkpoint windows."""
    considered = history.windows
    mixed = sum(
        1
        for window in considered
        if _known_version(window)
        and window.get("policy_version_min") != window.get("policy_version_max")
    )
    unknown = sum(1 for window in considered if not _known_version(window))
    if range_name == "checkpoint":
        selected = tuple(
            window
            for window in considered
            if checkpoint_version is not None
            and int_or_none(window.get("policy_version_min")) == checkpoint_version
            and int_or_none(window.get("policy_version_max")) == checkpoint_version
        )
    elif range_name == "cumulative":
        selected = considered
    else:
        epochs = tuple(
            epoch
            for window in considered
            if (epoch := _window_epoch(window)) is not None
        )
        latest = max(epochs, default=None)
        seconds = 15 * 60 if range_name == "recent_15m" else 60 * 60
        selected = (
            ()
            if latest is None
            else tuple(
                window
                for window in considered
                if (epoch := _window_epoch(window)) is not None
                and epoch >= latest - seconds
            )
        )
    has_cells = any(
        str(cell.get("candidate_deck_label", "")) in history.targets
        for window in selected
        for cell in window_cells(window)
    )
    available = history.exact_available and bool(selected) and has_cells
    reason = _unavailable_reason(
        history,
        range_name=range_name,
        checkpoint_version=checkpoint_version,
        selected=selected,
        has_cells=has_cells,
    )
    return SelectedTrainingWindows(
        windows=selected,
        metadata=TrainingEvidenceRangeMetadata(
            range=range_name,
            available=available,
            unavailable_reason=reason,
            started_at_utc=(
                None if not selected else str(selected[0].get("started_at_utc", ""))
            ),
            ended_at_utc=(
                None if not selected else str(selected[-1].get("ended_at_utc", ""))
            ),
            windows_considered=len(considered),
            windows_selected=len(selected),
            mixed_version_windows_excluded=mixed if range_name == "checkpoint" else 0,
            unknown_version_windows_excluded=(
                unknown if range_name == "checkpoint" else 0
            ),
        ),
    )


def posterior(counts: OutcomeCounts) -> TrainingPosterior:
    """Approximate the draw-aware Dirichlet posterior from its exact moments."""
    observed = outcome_stats(counts)
    if counts.games == 0:
        return TrainingPosterior(
            observed=observed,
            evidence_state="unavailable",
        )
    alpha = (
        counts.wins + 0.5,
        counts.draws + 0.5,
        counts.losses + 0.5,
    )
    coefficients = (1.0, 0.5, 0.0)
    total = sum(alpha)
    mean = (
        sum(a * coefficient for a, coefficient in zip(alpha, coefficients, strict=True))
        / total
    )
    second_moment = (
        sum(
            a * coefficient * coefficient
            for a, coefficient in zip(alpha, coefficients, strict=True)
        )
        / total
    )
    variance = max(0.0, (second_moment - mean * mean) / (total + 1.0))
    deviation = math.sqrt(variance)
    probability = (
        1.0
        if deviation == 0.0 and mean > 0.5
        else 0.0
        if deviation == 0.0
        else 1.0 - NormalDist(mu=mean, sigma=deviation).cdf(0.5)
    )
    return TrainingPosterior(
        observed=observed,
        posterior_mean=mean,
        credible_low=max(0.0, mean - 1.959963984540054 * deviation),
        credible_high=min(1.0, mean + 1.959963984540054 * deviation),
        probability_above_half=probability,
        evidence_state="ready",
    )


def outcome_stats(counts: OutcomeCounts) -> OutcomeStats:
    """Project mutable counts into the public immutable outcome contract."""
    return OutcomeStats(
        games=counts.games,
        wins=counts.wins,
        draws=counts.draws,
        losses=counts.losses,
        win_rate=None if counts.games == 0 else counts.wins / counts.games,
        score_rate=counts.score,
    )


def merge_cell_counts(cells: Sequence[Mapping[str, Any]]) -> OutcomeCounts:
    """Merge exact cell mappings without materializing games."""
    counts = OutcomeCounts()
    for cell in cells:
        counts.merge(OutcomeCounts.from_mapping(cell))
    return counts


def selected_cells(
    windows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Flatten the already bounded selected windows."""
    return tuple(cell for window in windows for cell in window_cells(window))


def window_cells(window: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return valid exact cell mappings from one complete window."""
    return tuple(
        item for item in sequence(window.get("cells")) if isinstance(item, Mapping)
    )


def cell_matches(
    cell: Mapping[str, Any],
    *,
    controller: TrainingController,
    candidate_seat: int | None,
    stationary_kinds: frozenset[str],
) -> bool:
    """Apply controller and seat filters to one exact cell."""
    if (
        candidate_seat is not None
        and int_or_none(cell.get("candidate_seat")) != candidate_seat
    ):
        return False
    return controller_matches(
        str(cell.get("opponent_kind", "")),
        controller,
        stationary_kinds,
        opponent_stratum=_text_or_none(cell.get("opponent_stratum")),
    )


def controller_matches(
    opponent_kind: str,
    controller: TrainingController,
    stationary_kinds: frozenset[str],
    *,
    opponent_stratum: str | None = None,
) -> bool:
    """Apply canonical strata while retaining deprecated kind aggregates."""
    if controller == "all":
        return True
    stratum = resolve_opponent_stratum(
        opponent_kind=opponent_kind,
        opponent_stratum=opponent_stratum,
    )
    if controller == "stationary":
        return opponent_kind in stationary_kinds
    if controller in KNOWN_OPPONENT_STRATA:
        return stratum == controller
    # Retained for callers and old UI bookmarks; it intentionally combines
    # sentinel and adaptive-history evidence from the legacy frozen kind.
    if controller == "frozen":
        return opponent_kind == "frozen"
    return opponent_kind == controller


def _text_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def active_opponent(
    label: str,
    *,
    targets: Sequence[str],
    target_hashes: set[str | None],
    repository: DashboardRepository,
) -> bool:
    """Match active opponents by exact label or compact exact identity."""
    return label in targets or (
        (opponent_hash := repository.exact_deck_hash(label)) is not None
        and opponent_hash in target_hashes
    )


def opponent_deck_label(cell: Mapping[str, Any]) -> str:
    """Retain a stable opponent axis even when deck metadata is missing."""
    label = str(cell.get("opponent_deck_label", "")).strip()
    return label or str(cell.get("opponent_id", "")).strip() or "unknown"


def sortable_score(value: float | None, *, missing: float = 2.0) -> float:
    """Return a deterministic sentinel for missing ranking evidence."""
    return missing if value is None else value


def int_or_none(value: Any) -> int | None:
    """Parse numeric metadata without accepting booleans or strings."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def mapping(value: Any) -> Mapping[str, Any]:
    """Narrow an arbitrary decoded value to a mapping."""
    return value if isinstance(value, Mapping) else {}


def sequence(value: Any) -> Sequence[Any]:
    """Narrow an arbitrary decoded value to a non-string sequence."""
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _unavailable_reason(
    history: TrainingHistory,
    *,
    range_name: TrainingEvidenceRange,
    checkpoint_version: int | None,
    selected: Sequence[Mapping[str, Any]],
    has_cells: bool,
) -> str | None:
    if not history.exact_available:
        return "performance history has no exact matchup dimensions"
    if range_name == "checkpoint" and checkpoint_version is None:
        return "no checkpoint was selected"
    if not selected and range_name == "checkpoint":
        return (
            f"no complete window contains only policy version {checkpoint_version}; "
            "mixed and unknown-version windows are excluded"
        )
    if not selected:
        return "no complete training window is available in this range"
    if not has_cells:
        return "selected windows contain no exact matchup observations"
    return None


def _known_version(window: Mapping[str, Any]) -> bool:
    return int_or_none(window.get("policy_version_min")) is not None and (
        int_or_none(window.get("policy_version_max")) is not None
    )


def _window_epoch(window: Mapping[str, Any]) -> float | None:
    raw = window.get("ended_at_epoch_seconds")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    ended = window.get("ended_at_utc")
    if not isinstance(ended, str) or not ended:
        return None
    try:
        return datetime.fromisoformat(ended.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


__all__ = [
    "SelectedTrainingWindows",
    "TrainingHistory",
    "active_opponent",
    "cell_matches",
    "controller_matches",
    "int_or_none",
    "mapping",
    "merge_cell_counts",
    "opponent_deck_label",
    "outcome_stats",
    "posterior",
    "select_training_windows",
    "selected_cells",
    "sequence",
    "sortable_score",
    "window_cells",
]
