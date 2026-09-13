"""State and serialization helpers for RL performance diagnostics."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3
KNOWN_OPPONENT_KINDS = ("self_play", "frozen", "scripted")
KNOWN_OPPONENT_STRATA = (
    "self_play",
    "sentinel",
    "adaptive_history",
    "scripted",
    "legacy_unknown",
)


def resolve_opponent_stratum(
    *,
    opponent_kind: str,
    opponent_stratum: str | None = None,
) -> str:
    """Return the canonical opponent stratum without inventing legacy detail."""
    stratum = (opponent_stratum or "").strip()
    if not stratum:
        stratum = {
            "self_play": "self_play",
            "scripted": "scripted",
            "frozen": "legacy_unknown",
        }.get(opponent_kind, "legacy_unknown")
    if stratum not in KNOWN_OPPONENT_STRATA:
        raise ValueError(f"unsupported performance opponent stratum: {stratum}")
    return stratum


def normalize_outcome_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Return one exact cell with the canonical opponent stratum populated."""
    normalized = dict(cell)
    normalized["opponent_stratum"] = resolve_opponent_stratum(
        opponent_kind=str(cell.get("opponent_kind", "")).strip(),
        opponent_stratum=_optional_cell_text(cell.get("opponent_stratum")),
    )
    return normalized


def normalize_performance_window(
    window: Mapping[str, Any],
    *,
    rebuild_stratum_slices: bool,
) -> dict[str, Any]:
    """Return one window with normalized cells and optional legacy strata."""
    normalized = dict(window)
    raw_cells = normalized.get("cells")
    cells = (
        [
            normalize_outcome_cell(cell)
            for cell in raw_cells
            if isinstance(cell, Mapping)
        ]
        if isinstance(raw_cells, list)
        else []
    )
    if isinstance(raw_cells, list):
        normalized["cells"] = cells
    if rebuild_stratum_slices:
        normalized["slices"] = rebuild_legacy_stratum_slices(
            normalized.get("slices"),
            cells=cells,
        )
    return normalized


def normalize_legacy_performance_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose canonical strata for legacy summaries without guessing frozen type."""
    normalized = dict(payload)
    raw_windows = normalized.get("completed_windows")
    if isinstance(raw_windows, list):
        normalized["completed_windows"] = [
            normalize_performance_window(window, rebuild_stratum_slices=True)
            for window in raw_windows
            if isinstance(window, Mapping)
        ]
    raw_partial = normalized.get("last_partial_window")
    if isinstance(raw_partial, Mapping):
        normalized["last_partial_window"] = normalize_performance_window(
            raw_partial,
            rebuild_stratum_slices=True,
        )
    for aggregate_name in ("latest_rolling_window", "cumulative"):
        raw_aggregate = normalized.get(aggregate_name)
        if isinstance(raw_aggregate, Mapping):
            aggregate = dict(raw_aggregate)
            aggregate["slices"] = rebuild_legacy_stratum_slices(
                aggregate.get("slices"),
            )
            normalized[aggregate_name] = aggregate
    return normalized


def rebuild_legacy_stratum_slices(
    raw_slices: Any,
    *,
    cells: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build stratum slices while retaining legacy compatibility aggregates."""
    slices = (
        {
            str(key): dict(value)
            for key, value in raw_slices.items()
            if isinstance(value, Mapping)
        }
        if isinstance(raw_slices, Mapping)
        else {}
    )
    for key in tuple(slices):
        if "/stratum/" in key:
            del slices[key]

    rebuilt: dict[str, OutcomeCounts] = {}
    if cells:
        for raw_cell in cells:
            cell = normalize_outcome_cell(raw_cell)
            counts = OutcomeCounts.from_mapping(cell)
            stratum = str(cell["opponent_stratum"])
            prefixes = ["overall"]
            candidate = str(cell.get("candidate_deck_label", "")).strip()
            if candidate:
                prefixes.append(f"deck/{candidate}")
            for prefix in prefixes:
                _merge_legacy_stratum_slice(
                    rebuilt,
                    prefix=prefix,
                    stratum=stratum,
                    counts=counts,
                )
    else:
        mappings = {
            "self_play": "self_play",
            "frozen": "legacy_unknown",
            "scripted": "scripted",
        }
        for key, raw_counts in tuple(slices.items()):
            prefix, separator, kind = key.rpartition("/")
            mapped_stratum = mappings.get(kind)
            if not separator or mapped_stratum is None:
                continue
            _merge_legacy_stratum_slice(
                rebuilt,
                prefix=prefix,
                stratum=mapped_stratum,
                counts=OutcomeCounts.from_mapping(raw_counts),
            )
    slices.update({key: counts.as_dict() for key, counts in rebuilt.items()})
    return slices


def _merge_legacy_stratum_slice(
    destination: dict[str, OutcomeCounts],
    *,
    prefix: str,
    stratum: str,
    counts: OutcomeCounts,
) -> None:
    """Merge one legacy kind aggregate into its non-invented stratum."""
    destination.setdefault(
        f"{prefix}/stratum/{stratum}", OutcomeCounts()
    ).merge(counts)


def _optional_cell_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


@dataclass(frozen=True)
class PerformanceReporterConfig:
    """Operational settings for one coordinator-side performance reporter."""

    summary_path: Path
    tensorboard_dir: Path
    count_stage: str = "decoded_before_staleness_filter"
    interval_seconds: float = 60.0
    rolling_window_minutes: int = 15
    target_deck_labels: tuple[str, ...] = ()
    stationary_opponent_kinds: tuple[str, ...] = ("frozen", "scripted")
    tensorboard_enabled: bool = True
    tensorboard_flush_seconds: float = 30.0
    recent_window_limit: int = 60
    parquet_shard_windows: int = 15

    def __post_init__(self) -> None:
        """Reject ambiguous time windows and unsupported opponent kinds."""
        if not self.count_stage.strip():
            raise ValueError("performance count_stage must be non-empty")
        if self.interval_seconds <= 0.0:
            raise ValueError("performance interval_seconds must be positive")
        if self.rolling_window_minutes <= 0:
            raise ValueError("rolling_window_minutes must be positive")
        rolling_intervals = self.rolling_window_minutes * 60.0 / self.interval_seconds
        if not math.isclose(rolling_intervals, round(rolling_intervals)):
            raise ValueError(
                "rolling_window_minutes must contain an integer number of intervals"
            )
        if rolling_intervals <= 1.0:
            raise ValueError("rolling window must include more than one interval")
        invalid = set(self.stationary_opponent_kinds) - set(KNOWN_OPPONENT_KINDS)
        if invalid:
            raise ValueError(
                f"unsupported stationary opponent kinds: {sorted(invalid)}"
            )
        if not self.stationary_opponent_kinds:
            raise ValueError("stationary_opponent_kinds must be non-empty")
        if self.recent_window_limit < self.rolling_window_intervals:
            raise ValueError(
                "recent_window_limit must cover the configured rolling window"
            )
        if self.parquet_shard_windows <= 0:
            raise ValueError("parquet_shard_windows must be positive")

    @property
    def rolling_window_intervals(self) -> int:
        """Return the exact number of reporting intervals in the rolling window."""
        return round(self.rolling_window_minutes * 60.0 / self.interval_seconds)

    @property
    def history_dir(self) -> Path:
        """Return the append-only Parquet history directory."""
        return self.summary_path.parent / "history"

    @property
    def history_manifest_path(self) -> Path:
        """Return the immutable-shard manifest path."""
        return self.summary_path.parent / "history_manifest.json"


@dataclass
class OutcomeCounts:
    """Win, loss, and draw totals with the contest's draw-aware score."""

    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float | None:
        if self.games <= 0:
            return None
        return (self.wins + 0.5 * self.draws) / self.games

    def observe(self, reward: float) -> None:
        if reward > 0.0:
            self.wins += 1
        elif reward < 0.0:
            self.losses += 1
        else:
            self.draws += 1

    def merge(self, other: OutcomeCounts) -> None:
        self.wins += other.wins
        self.losses += other.losses
        self.draws += other.draws

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "score": self.score,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> OutcomeCounts:
        return cls(
            wins=int(value.get("wins", 0)),
            losses=int(value.get("losses", 0)),
            draws=int(value.get("draws", 0)),
        )


@dataclass(frozen=True)
class PerformanceOutcome:
    """One scored result with exact dashboard dimensions."""

    candidate_deck_label: str
    opponent_kind: str
    opponent_deck_label: str
    opponent_id: str
    candidate_seat: int
    candidate_reward: float
    policy_version: int
    opponent_stratum: str = ""

    def __post_init__(self) -> None:
        """Reject incomplete or non-scored diagnostic evidence."""
        if not self.candidate_deck_label.strip():
            raise ValueError("performance candidate deck label must be non-empty")
        if self.opponent_kind not in KNOWN_OPPONENT_KINDS:
            raise ValueError("performance opponent kind is unsupported")
        if not self.opponent_deck_label.strip():
            raise ValueError("performance opponent deck label must be non-empty")
        if not self.opponent_id.strip():
            raise ValueError("performance opponent ID must be non-empty")
        if self.candidate_seat not in (0, 1):
            raise ValueError("performance candidate seat must be zero or one")
        if self.candidate_reward not in (-1.0, 0.0, 1.0):
            raise ValueError("performance reward must encode win, draw, or loss")
        if self.policy_version < 0:
            raise ValueError("performance policy version must be non-negative")
        stratum = resolve_opponent_stratum(
            opponent_kind=self.opponent_kind,
            opponent_stratum=self.opponent_stratum,
        )
        object.__setattr__(self, "opponent_stratum", stratum)


@dataclass(frozen=True, order=True)
class OutcomeCellKey:
    """Exact dimensions retained for one performance outcome cell."""

    candidate_deck_label: str
    opponent_kind: str
    opponent_stratum: str
    opponent_deck_label: str
    opponent_id: str
    candidate_seat: int

    def as_dict(self, counts: OutcomeCounts) -> dict[str, Any]:
        """Return one Parquet-ready cell record."""
        return {
            "candidate_deck_label": self.candidate_deck_label,
            "opponent_kind": self.opponent_kind,
            "opponent_stratum": self.opponent_stratum,
            "opponent_deck_label": self.opponent_deck_label,
            "opponent_id": self.opponent_id,
            "candidate_seat": self.candidate_seat,
            "wins": counts.wins,
            "losses": counts.losses,
            "draws": counts.draws,
        }


@dataclass
class MutablePerformanceWindow:
    """In-memory counters for one active wall-clock interval."""

    started_at: float
    counts: dict[str, OutcomeCounts] = field(default_factory=dict)
    cells: dict[OutcomeCellKey, OutcomeCounts] = field(default_factory=dict)
    decoded_games: int = 0
    stale_excluded_games: int = 0
    queued_games: int = 0
    missing_metadata_games: int = 0
    missing_metadata_fields: Counter[str] = field(default_factory=Counter)
    worker_games: Counter[str] = field(default_factory=Counter)
    policy_versions: list[int] = field(default_factory=list)


def merge_window_payloads(
    windows: Sequence[Mapping[str, Any]],
    configured_slice_keys: set[str],
) -> dict[str, Any]:
    """Merge complete minute payloads into one rolling diagnostic window."""
    merged: dict[str, OutcomeCounts] = {
        key: OutcomeCounts() for key in configured_slice_keys
    }
    for window in windows:
        raw_slices = window.get("slices")
        if not isinstance(raw_slices, Mapping):
            continue
        for key, raw_counts in raw_slices.items():
            if isinstance(raw_counts, Mapping):
                merged.setdefault(str(key), OutcomeCounts()).merge(
                    OutcomeCounts.from_mapping(raw_counts)
                )
    return {
        "window_count": len(windows),
        "started_at_utc": windows[0].get("started_at_utc") if windows else None,
        "ended_at_utc": windows[-1].get("ended_at_utc") if windows else None,
        "slices": {key: counts.as_dict() for key, counts in sorted(merged.items())},
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one human-readable diagnostic JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def utc_timestamp(epoch_seconds: float) -> str:
    """Render one epoch timestamp as UTC ISO-8601."""
    return (
        datetime.fromtimestamp(epoch_seconds, tz=UTC).isoformat().replace("+00:00", "Z")
    )
