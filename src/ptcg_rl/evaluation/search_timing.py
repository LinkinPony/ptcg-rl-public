"""Streaming replay ActTime audit for inference-search budget baselines."""

from __future__ import annotations

import glob
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.actions.selection import is_forced
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.data.kaggle_steps.records import (
    DEFAULT_CHUNK_SIZE,
    iter_replay_steps,
    replay_stub,
)
from ptcg_rl.engine.constants import OptionType, SelectContext


class SearchTimingAuditConfig(BaseModel):
    """Hydra-backed raw-replay timing audit configuration."""

    model_config = ConfigDict(extra="forbid")

    replay_paths: tuple[Path, ...] = ()
    replay_glob: str = (
        "outputs/kaggle_submission_replays/54498922_comfey_v12395/*.json"
    )
    team_name: str | None = "Marshall Maximizer"
    seat_index: int | None = None
    max_replays: int | None = 112
    expected_replays: int | None = 112
    chunk_size: int = DEFAULT_CHUNK_SIZE
    output_parquet: Path = Path(
        "outputs/inference_time_search/p0/replay_timing/decisions.parquet"
    )
    output_summary: Path = Path(
        "outputs/inference_time_search/p0/replay_timing/summary.json"
    )
    compression: str = "zstd"

    @field_validator("seat_index")
    @classmethod
    def valid_seat(cls, value: int | None) -> int | None:
        """Restrict explicit replay seats to the two game players."""
        if value is not None and value not in (0, 1):
            raise ValueError("seat_index must be 0 or 1")
        return value

    @field_validator("max_replays", "expected_replays")
    @classmethod
    def optional_positive(cls, value: int | None) -> int | None:
        """Reject non-positive optional replay counts."""
        if value is not None and value <= 0:
            raise ValueError("optional replay counts must be positive")
        return value

    @field_validator("chunk_size")
    @classmethod
    def positive_chunk_size(cls, value: int) -> int:
        """Reject non-positive stream chunks."""
        if value <= 0:
            raise ValueError("chunk_size must be positive")
        return value


@dataclass
class _TimingStats:
    replay_totals: list[float] = field(default_factory=list)
    callback_elapsed: list[float] = field(default_factory=list)
    startup_elapsed: list[float] = field(default_factory=list)
    non_startup_elapsed: list[float] = field(default_factory=list)
    callbacks_per_replay: list[int] = field(default_factory=list)
    replay_steps: list[int] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    source_hash: hashlib._Hash = field(default_factory=hashlib.sha256)


def run_search_timing_audit(config: SearchTimingAuditConfig) -> dict[str, Any]:
    """Stream raw replays into decision rows and a compact timing summary."""
    replay_paths = _resolve_replay_paths(config)
    output_path = records.repo_path(config.output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    stats = _TimingStats()
    writer: pq.ParquetWriter | None = None
    rows_buffer: list[dict[str, Any]] = []
    try:
        writer = pq.ParquetWriter(
            temporary_path,
            _timing_schema(),
            compression=config.compression,
        )
        for replay_path in replay_paths:
            rows = _audit_replay(replay_path, config=config, stats=stats)
            rows_buffer.extend(rows)
            if len(rows_buffer) >= 1_024:
                writer.write_table(
                    pa.Table.from_pylist(rows_buffer, schema=_timing_schema())
                )
                rows_buffer.clear()
        if rows_buffer:
            writer.write_table(pa.Table.from_pylist(rows_buffer, schema=_timing_schema()))
            rows_buffer.clear()
        writer.close()
        writer = None
        temporary_path.replace(output_path)
    finally:
        if writer is not None:
            writer.close()
        if temporary_path.exists():
            temporary_path.unlink()

    summary = _timing_summary(config, replay_paths, stats, output_path)
    _write_json_atomic(records.repo_path(config.output_summary), summary)
    return summary


def _audit_replay(
    replay_path: Path,
    *,
    config: SearchTimingAuditConfig,
    stats: _TimingStats,
) -> list[dict[str, Any]]:
    metadata = replay_stub(replay_path, chunk_size=config.chunk_size)
    seat = _resolve_seat(metadata, config)
    episode_id = int(_mapping(metadata.get("info")).get("EpisodeId", replay_path.stem))
    previous_remaining: float | None = None
    previous_step = -1
    previous_side: Mapping[str, Any] = {}
    rows: list[dict[str, Any]] = []
    final_remaining: float | None = None
    max_step = 0
    stats.source_hash.update(replay_path.name.encode("utf-8"))
    stats.source_hash.update(_file_sha256(replay_path).encode("ascii"))
    for step_index, sides in iter_replay_steps(
        replay_path,
        chunk_size=config.chunk_size,
    ):
        max_step = step_index
        if seat >= len(sides):
            continue
        side = sides[seat]
        observation = _mapping(side.get("observation"))
        remaining = _optional_float(observation.get("remainingOverageTime"))
        if remaining is None:
            continue
        final_remaining = remaining
        if previous_remaining is not None:
            elapsed = previous_remaining - remaining
            if elapsed > 0.0 and math.isfinite(elapsed):
                callback_index = len(rows)
                row = _timing_row(
                    episode_id=episode_id,
                    replay_path=replay_path,
                    seat=seat,
                    callback_index=callback_index,
                    observation_step=previous_step,
                    charged_at_step=step_index,
                    remaining_before=previous_remaining,
                    remaining_after=remaining,
                    elapsed=elapsed,
                    side=previous_side,
                )
                rows.append(row)
                stats.callback_elapsed.append(elapsed)
                if callback_index == 0:
                    stats.startup_elapsed.append(elapsed)
                else:
                    stats.non_startup_elapsed.append(elapsed)
                callback_class = str(row["callback_class"])
                stats.class_counts[callback_class] = (
                    stats.class_counts.get(callback_class, 0) + 1
                )
        previous_remaining = remaining
        previous_step = step_index
        previous_side = side

    if final_remaining is None:
        raise ValueError(f"replay contains no remainingOverageTime for seat {seat}: {replay_path}")
    initial = float(rows[0]["remaining_before"]) if rows else final_remaining
    stats.replay_totals.append(max(0.0, initial - final_remaining))
    stats.callbacks_per_replay.append(len(rows))
    stats.replay_steps.append(max_step + 1)
    return rows


def _timing_row(
    *,
    episode_id: int,
    replay_path: Path,
    seat: int,
    callback_index: int,
    observation_step: int,
    charged_at_step: int,
    remaining_before: float,
    remaining_after: float,
    elapsed: float,
    side: Mapping[str, Any],
) -> dict[str, Any]:
    observation = _mapping(side.get("observation"))
    select = _mapping(observation.get("select"))
    options = _sequence(select.get("option"))
    is_startup = callback_index == 0
    return {
        "episode_id": episode_id,
        "replay_path": records.display_path(replay_path),
        "seat": seat,
        "callback_index": callback_index,
        "observation_step": observation_step,
        "charged_at_step": charged_at_step,
        "remaining_before": remaining_before,
        "remaining_after": remaining_after,
        "elapsed_seconds": elapsed,
        "is_startup": is_startup,
        "status": str(side.get("status", "")),
        "callback_class": _callback_class(select, is_startup=is_startup),
        "select_type": _optional_int(select.get("type")),
        "select_context": _optional_int(select.get("context")),
        "option_count": len(options),
        "is_forced": bool(select) and is_forced(select),
    }


def _callback_class(select: Mapping[str, Any], *, is_startup: bool) -> str:
    if is_startup:
        return "startup"
    if not select:
        return "registration_or_idle"
    forced = is_forced(select)
    is_main = _optional_int(select.get("context")) == int(SelectContext.MAIN)
    option_types = {
        _optional_int(_field(option, "type")) for option in _sequence(select.get("option"))
    }
    if is_main and forced:
        return "forced_main"
    if is_main and int(OptionType.ATTACK) in option_types:
        return "nonforced_main_attack"
    if is_main:
        return "nonforced_main_other"
    return "non_main_forced" if forced else "non_main_choice"


def _timing_summary(
    config: SearchTimingAuditConfig,
    replay_paths: Sequence[Path],
    stats: _TimingStats,
    output_path: Path,
) -> dict[str, Any]:
    replay_count = len(replay_paths)
    expected = config.expected_replays
    return {
        "protocol": "ITS-P0-ACTTIME-v1",
        "replays": replay_count,
        "expected_replays": expected,
        "baseline_complete": expected is None or replay_count == expected,
        "callbacks": len(stats.callback_elapsed),
        "non_startup_callbacks": len(stats.non_startup_elapsed),
        "replay_total_seconds": _distribution(stats.replay_totals),
        "startup_seconds": _distribution(stats.startup_elapsed),
        "non_startup_seconds": _distribution(stats.non_startup_elapsed),
        "callbacks_per_replay": _distribution(stats.callbacks_per_replay),
        "replay_steps": _distribution(stats.replay_steps),
        "callback_class_counts": dict(sorted(stats.class_counts.items())),
        "source_fingerprint": stats.source_hash.hexdigest(),
        "decisions_path": records.display_path(output_path),
        "config": config.model_dump(mode="json"),
    }


def _distribution(values: Sequence[float | int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _timing_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("episode_id", pa.int64()),
            pa.field("replay_path", pa.string()),
            pa.field("seat", pa.int8()),
            pa.field("callback_index", pa.int32()),
            pa.field("observation_step", pa.int32()),
            pa.field("charged_at_step", pa.int32()),
            pa.field("remaining_before", pa.float64()),
            pa.field("remaining_after", pa.float64()),
            pa.field("elapsed_seconds", pa.float64()),
            pa.field("is_startup", pa.bool_()),
            pa.field("status", pa.string()),
            pa.field("callback_class", pa.string()),
            pa.field("select_type", pa.int16()),
            pa.field("select_context", pa.int16()),
            pa.field("option_count", pa.int16()),
            pa.field("is_forced", pa.bool_()),
        ]
    )


def _resolve_replay_paths(config: SearchTimingAuditConfig) -> tuple[Path, ...]:
    if config.replay_paths:
        paths = tuple(records.repo_path(path) for path in config.replay_paths)
    else:
        paths = tuple(
            Path(path)
            for path in sorted(glob.glob(str(records.repo_path(Path(config.replay_glob)))))
        )
    if config.max_replays is not None:
        paths = paths[: config.max_replays]
    if not paths:
        raise ValueError("no replay paths matched search timing audit")
    return paths


def _resolve_seat(metadata: Mapping[str, Any], config: SearchTimingAuditConfig) -> int:
    if config.seat_index is not None:
        return config.seat_index
    team_names = _sequence(_mapping(metadata.get("info")).get("TeamNames"))
    matches = [
        index
        for index, team_name in enumerate(team_names)
        if str(team_name) == str(config.team_name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one team_name={config.team_name!r} seat, found {matches}"
        )
    return matches[0]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
