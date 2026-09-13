"""Atomic streaming Parquet output for the S1 counterfactual audit."""

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
CANDIDATES_FILE = "candidates.parquet"
EVALUATIONS_FILE = "evaluations.parquet"


class CounterfactualAuditWriter:
    """Write root-grouped audit rows without retaining the dataset in memory."""

    def __init__(
        self,
        output_dir: Path,
        *,
        compression: str = "zstd",
        root_buffer_size: int = 32,
        candidate_buffer_size: int = 128,
        evaluation_buffer_size: int = 256,
    ) -> None:
        if min(root_buffer_size, candidate_buffer_size, evaluation_buffer_size) <= 0:
            raise ValueError("counterfactual writer buffer sizes must be positive")
        self.output_dir = records.repo_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._schemas = {
            ROOTS_FILE: root_schema(),
            CANDIDATES_FILE: candidate_schema(),
            EVALUATIONS_FILE: evaluation_schema(),
        }
        self._limits = {
            ROOTS_FILE: root_buffer_size,
            CANDIDATES_FILE: candidate_buffer_size,
            EVALUATIONS_FILE: evaluation_buffer_size,
        }
        self._buffers: dict[str, list[Mapping[str, Any]]] = {
            name: [] for name in self._schemas
        }
        self._temporary_paths = {
            name: self.output_dir / f"{name}.tmp" for name in self._schemas
        }
        self._final_paths = {
            name: self.output_dir / name for name in self._schemas
        }
        for path in self._temporary_paths.values():
            if path.exists():
                path.unlink()
        self._writers = {
            name: pq.ParquetWriter(
                self._temporary_paths[name],
                schema,
                compression=compression,
            )
            for name, schema in self._schemas.items()
        }
        self._closed = False

    @property
    def paths(self) -> dict[str, Path]:
        """Return final artifact paths keyed by semantic row type."""
        return dict(self._final_paths)

    def write_root_group(
        self,
        root: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        evaluations: Sequence[Mapping[str, Any]],
    ) -> None:
        """Append one root and its normalized child rows to bounded buffers."""
        self._require_open()
        self._append(ROOTS_FILE, (root,))
        self._append(CANDIDATES_FILE, candidates)
        self._append(EVALUATIONS_FILE, evaluations)

    def commit(self) -> None:
        """Close complete temporary files, then publish every Parquet artifact."""
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
        """Close writers and remove unpublished temporary files."""
        if self._closed:
            return
        for writer in self._writers.values():
            with suppress(Exception):
                writer.close()
        for path in self._temporary_paths.values():
            if path.exists():
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
        buffer = self._buffers[name]
        buffer.extend(rows)
        if len(buffer) >= self._limits[name]:
            self._flush(name)

    def _flush(self, name: str) -> None:
        buffer = self._buffers[name]
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=self._schemas[name])
        self._writers[name].write_table(table)
        buffer.clear()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("counterfactual writer is already closed")


def root_schema() -> pa.Schema:
    """Return one-row-per-root S1 audit schema."""
    return pa.schema(
        [
            pa.field("campaign_fp", pa.string(), nullable=False),
            pa.field("stage_fp", pa.string(), nullable=False),
            pa.field("root_id", pa.string(), nullable=False),
            pa.field("episode_id", pa.int64(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("step_index", pa.int32(), nullable=False),
            pa.field("seat", pa.int8(), nullable=False),
            pa.field("phase", pa.string(), nullable=False),
            pa.field("turn", pa.int16()),
            pa.field("select_context", pa.int16()),
            pa.field("option_count", pa.int16()),
            pa.field("terminal_value", pa.float32()),
            pa.field("resolved", pa.bool_(), nullable=False),
            pa.field("root_value", pa.float32()),
            pa.field("policy_entropy", pa.float32()),
            pa.field("policy_top_gap", pa.float32()),
            pa.field("greedy_action", pa.list_(pa.int16()), nullable=False),
            pa.field("recommended_action", pa.list_(pa.int16()), nullable=False),
            pa.field("recommendation_changed", pa.bool_(), nullable=False),
            pa.field("selection_reason", pa.string(), nullable=False),
            pa.field("candidate_count", pa.int16(), nullable=False),
            pa.field("greedy_included", pa.bool_(), nullable=False),
            pa.field("illegal_candidate_count", pa.int16(), nullable=False),
            pa.field("worlds_requested", pa.int16(), nullable=False),
            pa.field("worlds_sampled", pa.int16(), nullable=False),
            pa.field("worlds_completed", pa.int16(), nullable=False),
            pa.field("complete_coverage", pa.bool_(), nullable=False),
            pa.field("stop_reason", pa.string(), nullable=False),
            pa.field("transitions", pa.int32(), nullable=False),
            pa.field("engine_sessions", pa.int16(), nullable=False),
            pa.field("state_pool_peak", pa.int16(), nullable=False),
            pa.field("state_leaks", pa.int16(), nullable=False),
            pa.field("preparation_seconds", pa.float64(), nullable=False),
            pa.field("probe_seconds", pa.float64(), nullable=False),
            pa.field("base_policy_seconds", pa.float64(), nullable=False),
            pa.field("search_seconds", pa.float64(), nullable=False),
            pa.field("whole_act_seconds", pa.float64(), nullable=False),
            pa.field("audit_seconds", pa.float64(), nullable=False),
            pa.field("deadline_overshoot_seconds", pa.float64(), nullable=False),
            pa.field("telemetry_complete", pa.bool_(), nullable=False),
        ]
    )


def candidate_schema() -> pa.Schema:
    """Return one-row-per-root-action candidate schema."""
    return pa.schema(
        [
            pa.field("campaign_fp", pa.string(), nullable=False),
            pa.field("stage_fp", pa.string(), nullable=False),
            pa.field("root_id", pa.string(), nullable=False),
            pa.field("candidate_index", pa.int16(), nullable=False),
            pa.field("action", pa.list_(pa.int16()), nullable=False),
            pa.field("sources", pa.list_(pa.string()), nullable=False),
            pa.field("prior", pa.float64()),
            pa.field("paired_worlds", pa.int16(), nullable=False),
            pa.field("engine_mean", pa.float64()),
            pa.field("engine_std", pa.float64()),
            pa.field("critic_mean", pa.float64()),
            pa.field("critic_std", pa.float64()),
            pa.field("mean_delta", pa.float64()),
            pa.field("std_delta", pa.float64()),
            pa.field("robust_delta", pa.float64()),
            pa.field("downside_cvar", pa.float64()),
            pa.field("minimum_delta", pa.float64()),
        ]
    )


def evaluation_schema() -> pa.Schema:
    """Return one-row-per-root-action-world evidence schema."""
    return pa.schema(
        [
            pa.field("campaign_fp", pa.string(), nullable=False),
            pa.field("stage_fp", pa.string(), nullable=False),
            pa.field("root_id", pa.string(), nullable=False),
            pa.field("candidate_index", pa.int16(), nullable=False),
            pa.field("world_index", pa.int16(), nullable=False),
            pa.field("action", pa.list_(pa.int16()), nullable=False),
            pa.field("endpoint", pa.string(), nullable=False),
            pa.field("steps", pa.int16(), nullable=False),
            pa.field("engine_score", pa.float64()),
            pa.field("critic_value", pa.float64()),
            pa.field("stop_detail", pa.string(), nullable=False),
            pa.field("error", pa.string()),
        ]
    )
