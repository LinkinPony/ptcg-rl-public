"""Atomic bounded-memory Parquet output for teacher sibling campaigns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.data.kaggle_deck import records

ROOTS_FILE = "roots.parquet"
WORLDS_FILE = "worlds.parquet"
EVALUATIONS_FILE = "evaluations.parquet"
PAIRS_FILE = "pairs.parquet"


class TeacherSiblingWriter:
    """Stream complete root groups to temporary Parquet files then publish."""

    def __init__(
        self,
        output_dir: Path,
        *,
        compression: str = "zstd",
        buffer_size: int = 256,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("teacher sibling writer buffer_size must be positive")
        self.output_dir = records.repo_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._schemas = schemas()
        self._temporary_paths = {
            name: self.output_dir / f".{name}.tmp" for name in self._schemas
        }
        self._final_paths = {
            name: self.output_dir / name for name in self._schemas
        }
        for path in (*self._temporary_paths.values(), *self._final_paths.values()):
            if path.exists():
                raise FileExistsError(f"teacher sibling artifact already exists: {path}")
        self._writers = {
            name: pq.ParquetWriter(
                self._temporary_paths[name],
                schema,
                compression=compression,
            )
            for name, schema in self._schemas.items()
        }
        self._buffers: dict[str, list[Mapping[str, Any]]] = {
            name: [] for name in self._schemas
        }
        self._buffer_size = buffer_size
        self._closed = False

    @property
    def paths(self) -> Mapping[str, Path]:
        """Return final paths keyed by artifact filename."""
        return dict(self._final_paths)

    def write_root_group(
        self,
        *,
        root: Mapping[str, Any],
        worlds: Sequence[Mapping[str, Any]],
        evaluations: Sequence[Mapping[str, Any]],
        pairs: Sequence[Mapping[str, Any]],
    ) -> None:
        """Append one complete root group to bounded buffers."""
        self._append(ROOTS_FILE, (root,))
        self._append(WORLDS_FILE, worlds)
        self._append(EVALUATIONS_FILE, evaluations)
        self._append(PAIRS_FILE, pairs)

    def commit(self) -> None:
        """Close complete temporary files and atomically publish each artifact."""
        if self._closed:
            return
        try:
            for name in self._schemas:
                self._flush(name)
            for writer in self._writers.values():
                writer.close()
            for name in self._schemas:
                self._temporary_paths[name].replace(self._final_paths[name])
        except Exception:
            self.abort()
            raise
        self._closed = True

    def abort(self) -> None:
        """Close writers and remove every unpublished partial artifact."""
        if self._closed:
            return
        for writer in self._writers.values():
            with suppress(Exception):
                writer.close()
        for path in self._temporary_paths.values():
            with suppress(FileNotFoundError):
                path.unlink()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc, traceback
        if exc_type is None:
            self.commit()
        else:
            self.abort()

    def _append(
        self,
        name: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        if self._closed:
            raise RuntimeError("teacher sibling writer is closed")
        self._buffers[name].extend(rows)
        if len(self._buffers[name]) >= self._buffer_size:
            self._flush(name)

    def _flush(self, name: str) -> None:
        buffer = self._buffers[name]
        if not buffer:
            return
        self._writers[name].write_table(
            pa.Table.from_pylist(buffer, schema=self._schemas[name])
        )
        buffer.clear()


def schemas() -> dict[str, pa.Schema]:
    """Return stable schemas for replayable sibling evidence."""
    identity = [
        pa.field("campaign_fp", pa.string(), nullable=False),
        pa.field("root_id", pa.string(), nullable=False),
    ]
    behavior = [
        pa.field("behavior_kind", pa.string(), nullable=False),
        pa.field("ppo_ratio_eligible", pa.bool_(), nullable=False),
    ]
    return {
        ROOTS_FILE: pa.schema(
            [
                *identity,
                pa.field("source_shard", pa.string(), nullable=False),
                pa.field("source_row_index", pa.int64(), nullable=False),
                pa.field("episode_id", pa.int64(), nullable=False),
                pa.field("step_index", pa.int32(), nullable=False),
                pa.field("player_index", pa.int8(), nullable=False),
                pa.field("teacher_action", pa.list_(pa.int16()), nullable=False),
                pa.field("current_greedy", pa.list_(pa.int16()), nullable=False),
                pa.field(
                    "candidate_actions",
                    pa.list_(pa.list_(pa.int16())),
                    nullable=False,
                ),
                pa.field(
                    "candidate_sources",
                    pa.list_(pa.list_(pa.string())),
                    nullable=False,
                ),
                pa.field("current_priors", pa.list_(pa.float32()), nullable=False),
                pa.field("worlds_requested", pa.int16(), nullable=False),
                pa.field("candidate_count", pa.int16(), nullable=False),
                pa.field("illegal_candidates", pa.int16(), nullable=False),
                pa.field("complete_coverage", pa.bool_(), nullable=False),
                pa.field("state_pool_peak", pa.int16(), nullable=False),
                pa.field("state_leaks", pa.int16(), nullable=False),
                pa.field("elapsed_seconds", pa.float64(), nullable=False),
                *behavior,
            ]
        ),
        WORLDS_FILE: pa.schema(
            [
                *identity,
                pa.field("world_index", pa.int16(), nullable=False),
                pa.field("determinization_source", pa.string(), nullable=False),
                pa.field("archetype_signature", pa.string()),
                pa.field("archetype_label", pa.string()),
                pa.field("your_deck", pa.list_(pa.int32()), nullable=False),
                pa.field("your_prize", pa.list_(pa.int32()), nullable=False),
                pa.field("opponent_deck", pa.list_(pa.int32()), nullable=False),
                pa.field("opponent_prize", pa.list_(pa.int32()), nullable=False),
                pa.field("opponent_hand", pa.list_(pa.int32()), nullable=False),
                pa.field("opponent_active", pa.list_(pa.int32()), nullable=False),
            ]
        ),
        EVALUATIONS_FILE: pa.schema(
            [
                *identity,
                pa.field("world_index", pa.int16(), nullable=False),
                pa.field("candidate_index", pa.int16(), nullable=False),
                pa.field("action", pa.list_(pa.int16()), nullable=False),
                pa.field("sources", pa.list_(pa.string()), nullable=False),
                pa.field("endpoint", pa.string(), nullable=False),
                pa.field("engine_score", pa.float32()),
                pa.field("initial_value", pa.float32()),
                pa.field("trained_value", pa.float32()),
                pa.field("steps", pa.int16(), nullable=False),
                pa.field("forced_steps", pa.int16(), nullable=False),
                pa.field("stop_detail", pa.string(), nullable=False),
                pa.field("leaf_available", pa.bool_(), nullable=False),
                pa.field("error", pa.string()),
                *behavior,
            ]
        ),
        PAIRS_FILE: pa.schema(
            [
                *identity,
                pa.field("world_index", pa.int16(), nullable=False),
                pa.field("left_action", pa.list_(pa.int16()), nullable=False),
                pa.field("right_action", pa.list_(pa.int16()), nullable=False),
                pa.field("preferred_action", pa.list_(pa.int16()), nullable=False),
                pa.field("engine_delta", pa.float32(), nullable=False),
                pa.field("initial_value_delta", pa.float32(), nullable=False),
                pa.field("trained_value_delta", pa.float32(), nullable=False),
                pa.field("initial_credit", pa.float32(), nullable=False),
                pa.field("trained_credit", pa.float32(), nullable=False),
                *behavior,
            ]
        ),
    }


__all__ = [
    "EVALUATIONS_FILE",
    "PAIRS_FILE",
    "ROOTS_FILE",
    "WORLDS_FILE",
    "TeacherSiblingWriter",
    "schemas",
]
