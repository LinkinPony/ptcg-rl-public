"""Atomic compact Parquet artifacts for candidate-regret diagnostics."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pyarrow as pa
import pyarrow.parquet as pq


def root_schema() -> pa.Schema:
    """Return the stable per-root diagnostic schema."""
    return pa.schema(
        [
            pa.field("case_id", pa.string(), nullable=False),
            pa.field("root_state_fingerprint", pa.string(), nullable=False),
            pa.field("deck_fingerprint", pa.string(), nullable=False),
            pa.field("stratum_fingerprint", pa.string(), nullable=False),
            pa.field("select_type", pa.int16(), nullable=False),
            pa.field("select_context", pa.int16(), nullable=False),
            pa.field("option_count", pa.int16(), nullable=False),
            pa.field("min_count", pa.int16(), nullable=False),
            pa.field("max_count", pa.int16(), nullable=False),
            pa.field("ordered", pa.bool_(), nullable=False),
            pa.field("legal_action_count", pa.int32(), nullable=False),
            pa.field("reference_exhaustive", pa.bool_(), nullable=False),
            pa.field("rules_exact", pa.bool_(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("scenario_count_requested", pa.int16(), nullable=False),
            pa.field("scenario_count_used", pa.int16(), nullable=False),
            pa.field("scenario_support_mode", pa.string(), nullable=False),
            pa.field("scenario_support_exhaustive", pa.bool_(), nullable=False),
            pa.field("scenario_support_fingerprint", pa.string()),
            pa.field("scenario_grid_complete", pa.bool_(), nullable=False),
            pa.field("paired_support_integrity", pa.bool_(), nullable=False),
            pa.field("nonanticipativity_integrity", pa.bool_(), nullable=False),
            pa.field("constructor_valid", pa.bool_(), nullable=False),
            pa.field("constructor_fallback_reason", pa.string()),
            pa.field("constructor_exhaustive_branch", pa.bool_(), nullable=False),
            pa.field(
                "retained_action_support_complete", pa.bool_(), nullable=False
            ),
            pa.field("seed_count", pa.int16(), nullable=False),
            pa.field("retained_count", pa.int16(), nullable=False),
            pa.field("offered_by_source_json", pa.string(), nullable=False),
            pa.field("configured_seed_quotas_json", pa.string(), nullable=False),
            pa.field("used_seed_quotas_json", pa.string(), nullable=False),
            pa.field("configured_expansion_quotas_json", pa.string(), nullable=False),
            pa.field("used_expansion_quotas_json", pa.string(), nullable=False),
            pa.field("provenance_counts_json", pa.string(), nullable=False),
            pa.field("refill_slots", pa.int16(), nullable=False),
            pa.field("duplicate_offers", pa.int32(), nullable=False),
            pa.field("multi_source_candidates", pa.int16(), nullable=False),
            pa.field("retained_cardinality_count", pa.int16(), nullable=False),
            pa.field("post_state_diversity", pa.int16(), nullable=False),
            pa.field("post_state_alias_groups", pa.int16(), nullable=False),
            pa.field(
                "post_state_alias_score_disagreements", pa.int16(), nullable=False
            ),
            pa.field("leaf_bootstrapped", pa.bool_(), nullable=False),
            pa.field("unique_endpoint_value_rows", pa.int16(), nullable=False),
            pa.field("endpoint_counts_json", pa.string(), nullable=False),
            pa.field("native_error_counts_json", pa.string(), nullable=False),
            pa.field("scorer_fingerprint", pa.string()),
            pa.field("controller_fingerprint", pa.string()),
            pa.field("native_pack_ms", pa.float32(), nullable=False),
            pa.field("native_call_ms", pa.float32(), nullable=False),
            pa.field("native_parse_ms", pa.float32(), nullable=False),
            pa.field("native_payload_bytes", pa.int32(), nullable=False),
        ]
    )


def candidate_schema() -> pa.Schema:
    """Return the privacy-safe per-exhaustive-candidate schema."""
    return pa.schema(
        [
            pa.field("case_id", pa.string(), nullable=False),
            pa.field("candidate_fingerprint", pa.string(), nullable=False),
            pa.field("action_length", pa.int16(), nullable=False),
            pa.field("robust_score", pa.float32(), nullable=False),
            pa.field("weighted_mean", pa.float32(), nullable=False),
            pa.field("weighted_std", pa.float32(), nullable=False),
            pa.field("downside_minimum", pa.float32(), nullable=False),
            pa.field("terminal_weight", pa.float32(), nullable=False),
            pa.field("same_seat_main_weight", pa.float32(), nullable=False),
            pa.field("turn_handoff_weight", pa.float32(), nullable=False),
            pa.field("information_history_forked", pa.bool_(), nullable=False),
            pa.field("retained_rank", pa.int16()),
            pa.field("sources", pa.list_(pa.string()), nullable=False),
            pa.field("near_best", pa.bool_(), nullable=False),
            pa.field(
                "root_observable_successor_fingerprint", pa.string(), nullable=False
            ),
        ]
    )


def regret_schema() -> pa.Schema:
    """Return the per-root best-regret/epsilon-recall curve schema."""
    return pa.schema(
        [
            pa.field("case_id", pa.string(), nullable=False),
            pa.field("k", pa.int16(), nullable=False),
            pa.field("retained_count", pa.int16(), nullable=False),
            pa.field("exhaustive_best_score", pa.float32(), nullable=False),
            pa.field("retained_best_score", pa.float32(), nullable=False),
            pa.field("best_regret", pa.float32(), nullable=False),
            pa.field("epsilon_recall", pa.bool_(), nullable=False),
        ]
    )


class CandidateRegretWriter:
    """Publish grouped root/candidate/K rows as immutable Parquet parts."""

    def __init__(
        self,
        output_dir: Path,
        *,
        shard_roots: int,
        compression: str,
    ) -> None:
        if shard_roots <= 0:
            raise ValueError("shard_roots must be positive")
        self._root_parts = _AtomicParquetParts(
            output_dir / "roots_parts",
            schema=root_schema(),
            compression=compression,
        )
        self._candidate_parts = _AtomicParquetParts(
            output_dir / "candidates_parts",
            schema=candidate_schema(),
            compression=compression,
        )
        self._regret_parts = _AtomicParquetParts(
            output_dir / "regret_parts",
            schema=regret_schema(),
            compression=compression,
        )
        self._shard_roots = shard_roots
        self._root_rows: list[dict[str, Any]] = []
        self._candidate_rows: list[dict[str, Any]] = []
        self._regret_rows: list[dict[str, Any]] = []
        self._closed = False

    @property
    def root_count(self) -> int:
        """Return accepted root rows, including a buffered tail."""
        return self._root_parts.rows_written + len(self._root_rows)

    @property
    def part_counts(self) -> Mapping[str, int]:
        """Return published part counts for the summary manifest."""
        return {
            "roots": self._root_parts.part_count,
            "candidates": self._candidate_parts.part_count,
            "regret": self._regret_parts.part_count,
        }

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del traceback
        if exc_type is None and exc is None:
            self.close()
        else:
            self.abort()

    def append_group(
        self,
        root: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        regrets: Sequence[Mapping[str, Any]],
    ) -> None:
        """Append one complete diagnostic group without partial publication."""
        if self._closed:
            raise RuntimeError("cannot append to a closed candidate audit writer")
        self._root_rows.append(dict(root))
        self._candidate_rows.extend(dict(row) for row in candidates)
        self._regret_rows.extend(dict(row) for row in regrets)
        if len(self._root_rows) >= self._shard_roots:
            self._flush()

    def close(self) -> None:
        """Publish the buffered tail."""
        if self._closed:
            return
        if self._root_rows:
            self._flush()
        self._closed = True

    def abort(self) -> None:
        """Discard only unpublished temporary files and buffers."""
        for parts in (self._root_parts, self._candidate_parts, self._regret_parts):
            parts.abort()
        self._root_rows.clear()
        self._candidate_rows.clear()
        self._regret_rows.clear()
        self._closed = True

    def _flush(self) -> None:
        self._root_parts.write(self._root_rows)
        self._candidate_parts.write(self._candidate_rows)
        self._regret_parts.write(self._regret_rows)
        self._root_rows.clear()
        self._candidate_rows.clear()
        self._regret_rows.clear()


class _AtomicParquetParts:
    """Small reusable atomic part publisher with a fixed Arrow schema."""

    def __init__(
        self,
        directory: Path,
        *,
        schema: pa.Schema,
        compression: str,
    ) -> None:
        self._directory = directory
        self._schema = schema
        self._compression = compression
        self._part_index = 0
        self.rows_written = 0
        directory.mkdir(parents=True, exist_ok=True)
        if tuple(directory.glob("part-*.parquet")):
            raise FileExistsError(f"candidate audit output already exists: {directory}")

    @property
    def part_count(self) -> int:
        """Return atomically published part count."""
        return self._part_index

    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Publish one non-empty part; empty sibling groups need no part."""
        if not rows:
            return
        destination = self._directory / f"part-{self._part_index:05d}.parquet"
        temporary = self._directory / f".{destination.name}.tmp"
        if destination.exists() or temporary.exists():
            raise FileExistsError(
                f"refusing to replace candidate audit part: {destination}"
            )
        try:
            table = pa.Table.from_pylist(
                [dict(row) for row in rows], schema=self._schema
            )
            pq.write_table(
                table,
                temporary,
                compression=self._compression,
                write_statistics=True,
            )
            with temporary.open("rb") as source:
                os.fsync(source.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.rows_written += len(rows)
        self._part_index += 1

    def abort(self) -> None:
        """Remove only unpublished temporary files."""
        for temporary in self._directory.glob(".*.tmp"):
            temporary.unlink()


__all__ = [
    "CandidateRegretWriter",
    "candidate_schema",
    "regret_schema",
    "root_schema",
]
