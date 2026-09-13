"""Reusable aggregation of completed games across RL training segments."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import orjson
from pydantic import BaseModel, ConfigDict, Field

_PERFORMANCE_RELATIVE_PATH = Path("performance/training_performance.json")
_SUPPORTED_COUNT_STAGES = frozenset(
    {
        "engine_terminal_single_writer",
        "scored_terminal_single_writer",
    }
)
_TRAINING_MARKERS = ("learner_status.json", "collection_status.json")

_PathSignature = tuple[int, int]


class TrainingRunGameCount(BaseModel):
    """Validated terminal-game count from one training segment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    games: int = Field(ge=0)
    count_stage: str
    started_at_utc: str | None = None
    updated_at_utc: str


class TrainingGameStatistics(BaseModel):
    """Compact cross-segment count suitable for APIs and operational tools."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    semantics: Literal["terminal_outcome_telemetry_sum_v1"] = (
        "terminal_outcome_telemetry_sum_v1"
    )
    total_games: int = Field(ge=0)
    selected_run_games: int | None = Field(default=None, ge=0)
    other_runs_games: int = Field(ge=0)
    counted_run_count: int = Field(ge=0)
    unreported_run_count: int = Field(ge=0)
    invalid_run_count: int = Field(ge=0)
    is_lower_bound: bool
    count_stages: tuple[str, ...]
    updated_at_utc: str | None = None


@dataclass(frozen=True)
class _CachedRun:
    """One parsed result or parse failure tied to an immutable file signature."""

    signature: _PathSignature
    count: TrainingRunGameCount | None
    error: str | None


class TrainingGameStatisticsCollector:
    """Cache-aware scanner for terminal-game telemetry under one run root."""

    def __init__(
        self,
        run_root: Path,
        *,
        known_run_ids: Sequence[str] = (),
    ) -> None:
        self.run_root = run_root.resolve()
        self.known_run_ids = frozenset(known_run_ids)
        self._cache: dict[str, _CachedRun] = {}
        self._lock = threading.Lock()

    def collect(self, *, selected_run: str | None = None) -> TrainingGameStatistics:
        """Sum each segment once and expose missing historical coverage explicitly."""
        with self._lock:
            return self._collect_locked(selected_run=selected_run)

    def _collect_locked(self, *, selected_run: str | None) -> TrainingGameStatistics:
        paths = tuple(sorted(self.run_root.glob(f"*/{_PERFORMANCE_RELATIVE_PATH}")))
        discovered_run_ids = {path.parent.parent.name for path in paths}
        counts: list[TrainingRunGameCount] = []
        invalid_run_ids: set[str] = set()
        live_cache_keys: set[str] = set()
        for path in paths:
            if not path.is_file():
                continue
            run_id = path.parent.parent.name
            live_cache_keys.add(run_id)
            signature = _path_signature(path)
            cached = self._cache.get(run_id)
            if cached is None or cached.signature != signature:
                cached = _parse_cached_run(path, run_id=run_id, signature=signature)
                self._cache[run_id] = cached
            if cached.count is None:
                invalid_run_ids.add(run_id)
            else:
                counts.append(cached.count)
        self._cache = {
            run_id: cached
            for run_id, cached in self._cache.items()
            if run_id in live_cache_keys
        }

        candidate_run_ids = self.known_run_ids | self._marked_training_runs()
        unreported = candidate_run_ids - discovered_run_ids
        total_games = sum(item.games for item in counts)
        selected_games = next(
            (item.games for item in counts if item.run_id == selected_run),
            None,
        )
        updated_at = max(
            (item.updated_at_utc for item in counts),
            default=None,
        )
        return TrainingGameStatistics(
            total_games=total_games,
            selected_run_games=selected_games,
            other_runs_games=total_games - (selected_games or 0),
            counted_run_count=len(counts),
            unreported_run_count=len(unreported),
            invalid_run_count=len(invalid_run_ids),
            is_lower_bound=bool(unreported or invalid_run_ids),
            count_stages=tuple(sorted({item.count_stage for item in counts})),
            updated_at_utc=updated_at,
        )

    def _marked_training_runs(self) -> frozenset[str]:
        run_ids: set[str] = set()
        for marker in _TRAINING_MARKERS:
            run_ids.update(
                path.parent.name
                for path in self.run_root.glob(f"*/{marker}")
                if path.is_file()
            )
        return frozenset(run_ids)


def _parse_cached_run(
    path: Path,
    *,
    run_id: str,
    signature: _PathSignature,
) -> _CachedRun:
    try:
        return _CachedRun(
            signature=signature,
            count=_read_run_count(path, run_id=run_id),
            error=None,
        )
    except (OSError, TypeError, ValueError) as error:
        return _CachedRun(signature=signature, count=None, error=str(error))


def _read_run_count(path: Path, *, run_id: str) -> TrainingRunGameCount:
    payload = orjson.loads(path.read_bytes())
    root = _mapping(payload, label="performance summary")
    semantics = _mapping(root.get("semantics"), label="performance semantics")
    count_stage = str(semantics.get("count_stage", "")).strip()
    if count_stage not in _SUPPORTED_COUNT_STAGES:
        raise ValueError(f"unsupported terminal-game count stage: {count_stage!r}")
    cumulative = _mapping(root.get("cumulative"), label="cumulative performance")
    slices = _mapping(cumulative.get("slices"), label="cumulative slices")
    raw_overall = slices.get("overall/all")
    if raw_overall is None and not slices:
        wins = draws = losses = 0
        declared_games = 0
    else:
        overall = _mapping(raw_overall, label="overall/all counts")
        wins = _nonnegative_int(overall.get("wins"), label="wins")
        draws = _nonnegative_int(overall.get("draws"), label="draws")
        losses = _nonnegative_int(overall.get("losses"), label="losses")
        declared_games = _nonnegative_int(overall.get("games"), label="games")
    games = wins + draws + losses
    if declared_games != games:
        raise ValueError(
            f"overall game count differs from W/D/L sum: {declared_games} != {games}"
        )
    return TrainingRunGameCount(
        run_id=run_id,
        games=games,
        count_stage=count_stage,
        started_at_utc=_optional_text(root.get("started_at_utc")),
        updated_at_utc=_mtime_utc(path),
    )


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _path_signature(path: Path) -> _PathSignature:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat().replace(
        "+00:00",
        "Z",
    )
