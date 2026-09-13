"""Atomic streaming Parquet output for S2 trace stress."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.data.kaggle_deck import records

ACTIONS_FILE = "actions.parquet"
RUNS_FILE = "stress_runs.parquet"


class SearchStressWriter:
    """Stream callback rows and atomically publish callback/run tables."""

    def __init__(
        self,
        output_dir: Path,
        *,
        compression: str = "zstd",
        action_buffer_size: int = 128,
    ) -> None:
        if action_buffer_size <= 0:
            raise ValueError("stress action_buffer_size must be positive")
        self.output_dir = records.repo_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._action_buffer_size = action_buffer_size
        self._action_buffer: list[Mapping[str, Any]] = []
        self._run_rows: list[Mapping[str, Any]] = []
        self._action_temporary = self.output_dir / f"{ACTIONS_FILE}.tmp"
        self._run_temporary = self.output_dir / f"{RUNS_FILE}.tmp"
        self._action_final = self.output_dir / ACTIONS_FILE
        self._run_final = self.output_dir / RUNS_FILE
        for path in (self._action_temporary, self._run_temporary):
            path.unlink(missing_ok=True)
        self._action_writer = pq.ParquetWriter(
            self._action_temporary,
            action_schema(),
            compression=compression,
        )
        self._compression = compression
        self._closed = False

    @property
    def paths(self) -> dict[str, Path]:
        """Return final artifact paths."""
        return {"actions": self._action_final, "stress_runs": self._run_final}

    def write_action(self, row: Mapping[str, Any]) -> None:
        """Append one callback to a bounded streaming buffer."""
        self._require_open()
        self._action_buffer.append(row)
        if len(self._action_buffer) >= self._action_buffer_size:
            self._flush_actions()

    def write_runs(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Retain the small fixed run matrix until diagnostics are complete."""
        self._require_open()
        self._run_rows.extend(rows)

    def commit(self) -> None:
        """Publish complete callback and run tables."""
        if self._closed:
            return
        try:
            self._flush_actions()
            self._action_writer.close()
            run_table = pa.Table.from_pylist(self._run_rows, schema=run_schema())
            pq.write_table(
                run_table,
                self._run_temporary,
                compression=self._compression,
            )
            self._action_temporary.replace(self._action_final)
            self._run_temporary.replace(self._run_final)
        except Exception:
            self.abort()
            raise
        self._closed = True

    def abort(self) -> None:
        """Remove incomplete temporary outputs."""
        if self._closed:
            return
        try:
            self._action_writer.close()
        finally:
            for path in (self._action_temporary, self._run_temporary):
                path.unlink(missing_ok=True)
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

    def _flush_actions(self) -> None:
        if not self._action_buffer:
            return
        table = pa.Table.from_pylist(
            self._action_buffer,
            schema=action_schema(),
        )
        self._action_writer.write_table(table)
        self._action_buffer.clear()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("stress writer is already closed")


def action_schema() -> pa.Schema:
    """Return one-row-per-runtime-callback trace schema."""
    return pa.schema(
        [
            pa.field("campaign_fp", pa.string(), nullable=False),
            pa.field("stage_fp", pa.string(), nullable=False),
            pa.field("run_kind", pa.string(), nullable=False),
            pa.field("cell_id", pa.string(), nullable=False),
            pa.field("controller", pa.string(), nullable=False),
            pa.field("seat", pa.int8(), nullable=False),
            pa.field("global_steps_requested", pa.int16(), nullable=False),
            pa.field("slowdown_factor", pa.float64(), nullable=False),
            pa.field("global_step", pa.int32()),
            pa.field("callback_index", pa.int32(), nullable=False),
            pa.field("source_replay_index", pa.int16()),
            pa.field("source_episode_id", pa.int64()),
            pa.field("source_step", pa.int32()),
            pa.field("remaining_before", pa.float64(), nullable=False),
            pa.field("remaining_after", pa.float64(), nullable=False),
            pa.field("real_elapsed_seconds", pa.float64(), nullable=False),
            pa.field("virtual_elapsed_seconds", pa.float64(), nullable=False),
            pa.field("action", pa.list_(pa.int16()), nullable=False),
            pa.field("legal", pa.bool_(), nullable=False),
            pa.field("timed_out", pa.bool_(), nullable=False),
            pa.field("error_type", pa.string()),
            pa.field("telemetry_present", pa.bool_(), nullable=False),
            pa.field("telemetry_stop_reason", pa.string()),
            pa.field("planned_quota_seconds", pa.float64()),
            pa.field("actual_search_seconds", pa.float64()),
            pa.field("whole_act_seconds", pa.float64()),
            pa.field("startup_seconds", pa.float64()),
            pa.field("probe_seconds", pa.float64()),
            pa.field("base_policy_seconds", pa.float64()),
            pa.field("bank_spent_seconds", pa.float64()),
            pa.field("deadline_overshoot_seconds", pa.float64()),
            pa.field("search_start_remaining_overage", pa.float64()),
            pa.field("candidates", pa.int16()),
            pa.field("worlds_requested", pa.int16()),
            pa.field("worlds_completed", pa.int16()),
            pa.field("state_pool_peak", pa.int16()),
            pa.field("state_leaks", pa.int16()),
            pa.field("fallback_available", pa.bool_()),
            pa.field("recommendation_changed", pa.bool_()),
            pa.field("action_changed", pa.bool_()),
        ]
    )


def run_schema() -> pa.Schema:
    """Return one-row-per-stress-cell summary schema."""
    return pa.schema(
        [
            pa.field("campaign_fp", pa.string(), nullable=False),
            pa.field("stage_fp", pa.string(), nullable=False),
            pa.field("run_kind", pa.string(), nullable=False),
            pa.field("cell_id", pa.string(), nullable=False),
            pa.field("controller", pa.string(), nullable=False),
            pa.field("seat", pa.int8(), nullable=False),
            pa.field("global_steps_requested", pa.int16(), nullable=False),
            pa.field("global_steps_completed", pa.int16(), nullable=False),
            pa.field("slowdown_factor", pa.float64(), nullable=False),
            pa.field("boundary_remaining_overage", pa.float64()),
            pa.field("callbacks", pa.int32(), nullable=False),
            pa.field("source_replays", pa.int16(), nullable=False),
            pa.field("real_elapsed_seconds", pa.float64(), nullable=False),
            pa.field("virtual_elapsed_seconds", pa.float64(), nullable=False),
            pa.field("final_remaining_overage", pa.float64(), nullable=False),
            pa.field("timed_out", pa.bool_(), nullable=False),
            pa.field("illegal_actions", pa.int32(), nullable=False),
            pa.field("agent_errors", pa.int32(), nullable=False),
            pa.field("telemetry_callbacks", pa.int32(), nullable=False),
            pa.field("telemetry_coverage", pa.float64(), nullable=False),
            pa.field("search_roots", pa.int32(), nullable=False),
            pa.field("search_seconds", pa.float64(), nullable=False),
            pa.field("max_bank_spent_seconds", pa.float64(), nullable=False),
            pa.field("min_search_start_remaining_overage", pa.float64()),
            pa.field("max_deadline_overshoot_seconds", pa.float64(), nullable=False),
            pa.field("max_whole_act_seconds", pa.float64(), nullable=False),
            pa.field("fallback_failures", pa.int32(), nullable=False),
            pa.field("state_leaks", pa.int32(), nullable=False),
            pa.field("max_state_pool_peak", pa.int16(), nullable=False),
            pa.field("recommendation_changes", pa.int32(), nullable=False),
            pa.field("action_changes", pa.int32(), nullable=False),
            pa.field("action_fingerprint", pa.string(), nullable=False),
            pa.field("equivalence_match", pa.bool_()),
            pa.field("safety_checks_observed", pa.bool_(), nullable=False),
            pa.field("diagnostic", pa.string(), nullable=False),
        ]
    )
