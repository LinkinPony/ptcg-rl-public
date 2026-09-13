"""Characterize replay log increments for context evidence accumulation."""

from __future__ import annotations

import glob
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps.records import DEFAULT_CHUNK_SIZE, iter_replay_steps


class GameContextLogProbeConfig(BaseModel):
    """Hydra-backed config for replay log-increment characterization."""

    model_config = ConfigDict(extra="forbid")

    replay_paths: tuple[Path, ...] = ()
    replay_glob: str = "data/external/kaggle_top_episodes_daily/*/*.json"
    max_replays: int | None = 16
    chunk_size: int = DEFAULT_CHUNK_SIZE
    output_path: Path | None = Path("outputs/context/log_probe/summary.json")

    @field_validator("max_replays")
    @classmethod
    def valid_optional_positive(cls, value: int | None) -> int | None:
        """Reject non-positive optional limits."""
        if value is not None and value <= 0:
            raise ValueError("max_replays must be positive when set")
        return value

    @field_validator("chunk_size")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive parser limits."""
        if value <= 0:
            raise ValueError("chunk_size must be positive")
        return value


@dataclass
class _ProbeStats:
    replays_seen: int = 0
    steps_seen: int = 0
    active_select_observations: int = 0
    active_with_logs: int = 0
    active_without_logs: int = 0
    max_logs_per_active: int = 0
    steps_since_previous_active: Counter[int] = field(default_factory=Counter)
    log_type_counts: Counter[int] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable probe metrics."""
        return {
            "replays_seen": self.replays_seen,
            "steps_seen": self.steps_seen,
            "active_select_observations": self.active_select_observations,
            "active_with_logs": self.active_with_logs,
            "active_without_logs": self.active_without_logs,
            "max_logs_per_active": self.max_logs_per_active,
            "steps_since_previous_active": [
                {"steps": steps, "count": count}
                for steps, count in sorted(self.steps_since_previous_active.items())
            ],
            "log_type_counts": [
                {"log_type": log_type, "count": count}
                for log_type, count in sorted(self.log_type_counts.items())
            ],
        }


def run_game_context_log_probe(
    config: GameContextLogProbeConfig,
) -> dict[str, Any]:
    """Characterize ACTIVE-step replay logs and optionally write a summary."""
    stats = _ProbeStats()
    for replay_path in _resolve_replay_paths(config):
        stats.replays_seen += 1
        _probe_replay(replay_path, config=config, stats=stats)
    summary = stats.as_dict()
    if config.output_path is not None:
        output_path = deck_records.repo_path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _probe_replay(
    replay_path: Path,
    *,
    config: GameContextLogProbeConfig,
    stats: _ProbeStats,
) -> None:
    last_active_step: dict[int, int] = {}
    for step_index, sides in iter_replay_steps(
        replay_path,
        chunk_size=config.chunk_size,
    ):
        stats.steps_seen += 1
        for player_index, side in enumerate(sides):
            if not _has_active_select(side):
                continue
            stats.active_select_observations += 1
            previous = last_active_step.get(player_index)
            if previous is not None:
                stats.steps_since_previous_active[step_index - previous] += 1
            last_active_step[player_index] = step_index
            logs = _sequence(_mapping(side.get("observation")).get("logs"))
            if logs:
                stats.active_with_logs += 1
            else:
                stats.active_without_logs += 1
            stats.max_logs_per_active = max(stats.max_logs_per_active, len(logs))
            for log in logs:
                log_type = _optional_int(_field(log, "type"))
                if log_type is not None:
                    stats.log_type_counts[log_type] += 1


def _resolve_replay_paths(config: GameContextLogProbeConfig) -> tuple[Path, ...]:
    if config.replay_paths:
        paths = tuple(deck_records.repo_path(path) for path in config.replay_paths)
    else:
        pattern = str(deck_records.repo_path(Path(config.replay_glob)))
        paths = tuple(Path(path) for path in sorted(glob.glob(pattern)))
    if config.max_replays is not None:
        paths = paths[: config.max_replays]
    if not paths:
        raise ValueError("no replay paths matched log probe configuration")
    return paths


def _has_active_select(side: Mapping[str, Any]) -> bool:
    if str(side.get("status", "")) != "ACTIVE":
        return False
    observation = _mapping(side.get("observation"))
    return isinstance(observation.get("select"), Mapping)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None
