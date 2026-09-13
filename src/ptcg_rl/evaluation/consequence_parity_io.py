"""Bounded atomic Parquet/NPZ shards for decision-transition parity rows."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

AuditOutputFormat = Literal["parquet", "npz"]


def consequence_parity_schema() -> pa.Schema:
    """Return the stable privacy-safe audit row schema."""
    return pa.schema(
        [
            pa.field("case_id", pa.string(), nullable=False),
            pa.field("world_index", pa.int16(), nullable=False),
            pa.field("candidate_index", pa.int32(), nullable=False),
            pa.field("candidate_action_length", pa.int16(), nullable=False),
            pa.field("candidate_fingerprint", pa.string(), nullable=False),
            pa.field("legal_action_count", pa.int64(), nullable=False),
            pa.field("support_exhaustive", pa.bool_(), nullable=False),
            pa.field("root_direct", pa.bool_(), nullable=False),
            pa.field("root_subset", pa.bool_(), nullable=False),
            pa.field("root_ordered", pa.bool_(), nullable=False),
            pa.field("root_multi_prompt", pa.bool_(), nullable=False),
            pa.field("root_manual_coin", pa.bool_(), nullable=False),
            pa.field("root_handoff", pa.bool_(), nullable=False),
            pa.field("direct", pa.bool_(), nullable=False),
            pa.field("subset", pa.bool_(), nullable=False),
            pa.field("ordered", pa.bool_(), nullable=False),
            pa.field("multi_prompt", pa.bool_(), nullable=False),
            pa.field("manual_coin", pa.bool_(), nullable=False),
            pa.field("handoff", pa.bool_(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("native_error", pa.int32(), nullable=False),
            pa.field("reference_error", pa.int32(), nullable=False),
            pa.field("native_endpoint", pa.int8(), nullable=False),
            pa.field("reference_endpoint", pa.int8(), nullable=False),
            pa.field("native_transition_steps", pa.int16(), nullable=False),
            pa.field("reference_transition_steps", pa.int16(), nullable=False),
            pa.field("native_forced_steps", pa.int16(), nullable=False),
            pa.field("reference_forced_steps", pa.int16(), nullable=False),
            pa.field("native_leaf_player", pa.int8(), nullable=False),
            pa.field("reference_leaf_player", pa.int8(), nullable=False),
            pa.field("endpoint_match", pa.bool_()),
            pa.field("transition_steps_match", pa.bool_()),
            pa.field("leaf_player_match", pa.bool_()),
            pa.field("leaf_actor_state_match", pa.bool_()),
            pa.field("leaf_actor_log_match", pa.bool_()),
            pa.field("log_match", pa.bool_()),
            pa.field("state_match", pa.bool_()),
            pa.field("effect_match", pa.bool_()),
            pa.field("parity_match", pa.bool_(), nullable=False),
            pa.field("native_parity_failure", pa.bool_(), nullable=False),
            pa.field("native_isolated_match", pa.bool_(), nullable=False),
            pa.field("native_isolated_metadata_match", pa.bool_(), nullable=False),
            pa.field(
                "native_isolated_root_observation_match",
                pa.bool_(),
                nullable=False,
            ),
            pa.field(
                "native_isolated_leaf_observation_match",
                pa.bool_(),
                nullable=False,
            ),
            pa.field("effect_max_abs_diff", pa.float32()),
            pa.field("native_state_digest", pa.string()),
            pa.field("reference_state_digest", pa.string()),
            pa.field("native_log_digest", pa.string()),
            pa.field("reference_log_digest", pa.string()),
            pa.field("native_rng_unsupported", pa.bool_(), nullable=False),
            pa.field("reference_prize_defect_exposed", pa.bool_(), nullable=False),
            pa.field("reference_prize_defect_relevant", pa.bool_(), nullable=False),
            pa.field("damage_effect", pa.bool_(), nullable=False),
            pa.field("healing_effect", pa.bool_(), nullable=False),
            pa.field("prize_effect", pa.bool_(), nullable=False),
            pa.field("status_effect", pa.bool_(), nullable=False),
            pa.field("random_effect", pa.bool_(), nullable=False),
            pa.field("native_pack_ms", pa.float32(), nullable=False),
            pa.field("native_call_ms", pa.float32(), nullable=False),
            pa.field("native_parse_ms", pa.float32(), nullable=False),
            pa.field("native_payload_bytes", pa.int32(), nullable=False),
        ]
    )


class AtomicAuditShardWriter:
    """Write bounded immutable parts without publishing partial shards."""

    def __init__(
        self,
        output_dir: Path,
        *,
        output_format: AuditOutputFormat,
        shard_rows: int,
        max_rows: int,
        compression: str,
    ) -> None:
        if output_format not in ("parquet", "npz"):
            raise ValueError("output_format must be parquet or npz")
        if shard_rows <= 0 or max_rows <= 0:
            raise ValueError("writer row bounds must be positive")
        if shard_rows > max_rows:
            raise ValueError("shard_rows cannot exceed max_rows")
        self._output_dir = Path(output_dir)
        self._parts_dir = self._output_dir / "parts"
        self._format: AuditOutputFormat = output_format
        self._shard_rows = int(shard_rows)
        self._max_rows = int(max_rows)
        self._compression = compression
        self._rows: list[dict[str, Any]] = []
        self._rows_written = 0
        self._part_index = 0
        self._closed = False
        self._parts_dir.mkdir(parents=True, exist_ok=True)
        existing = tuple(self._parts_dir.glob("part-*"))
        if existing:
            raise FileExistsError(
                "audit output already contains parts; use a fresh output directory"
            )

    @property
    def rows_written(self) -> int:
        """Return rows already accepted, including the buffered tail."""
        return self._rows_written + len(self._rows)

    @property
    def part_count(self) -> int:
        """Return the number of atomically published parts."""
        return self._part_index

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

    def append(self, row: Mapping[str, Any]) -> None:
        """Append one row, flushing a full atomic shard as needed."""
        if self._closed:
            raise RuntimeError("cannot append to a closed audit writer")
        if self.rows_written >= self._max_rows:
            raise ValueError("audit output row bound would be exceeded")
        self._rows.append(dict(row))
        if len(self._rows) >= self._shard_rows:
            self._flush()

    def close(self) -> None:
        """Publish the buffered tail and close the writer."""
        if self._closed:
            return
        if self._rows:
            self._flush()
        self._closed = True

    def abort(self) -> None:
        """Discard only unpublished temporary files and buffered rows."""
        for temporary in self._parts_dir.glob(".*.tmp"):
            temporary.unlink()
        self._rows.clear()
        self._closed = True

    def _flush(self) -> None:
        suffix = ".parquet" if self._format == "parquet" else ".npz"
        destination = self._parts_dir / f"part-{self._part_index:05d}{suffix}"
        temporary = self._parts_dir / f".{destination.name}.tmp"
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"refusing to replace audit shard: {destination}")
        try:
            if self._format == "parquet":
                table = pa.Table.from_pylist(
                    self._rows,
                    schema=consequence_parity_schema(),
                )
                pq.write_table(
                    table,
                    temporary,
                    compression=self._compression,
                    write_statistics=True,
                )
            else:
                arrays = _npz_arrays(self._rows)
                with temporary.open("wb") as output:
                    np.savez_compressed(output, **cast(Any, arrays))
                    output.flush()
                    os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._rows_written += len(self._rows)
        self._rows.clear()
        self._part_index += 1


def _npz_arrays(rows: list[dict[str, Any]]) -> Mapping[str, np.ndarray]:
    schema = consequence_parity_schema()
    arrays: dict[str, np.ndarray] = {}
    for field in schema:
        values = [row.get(field.name) for row in rows]
        if pa.types.is_boolean(field.type):
            arrays[field.name] = np.asarray(
                [-1 if value is None else int(bool(value)) for value in values],
                dtype=np.int8,
            )
        elif pa.types.is_integer(field.type):
            arrays[field.name] = np.asarray(
                [0 if value is None else int(value) for value in values],
                dtype=_numpy_integer_dtype(field.type),
            )
        elif pa.types.is_floating(field.type):
            arrays[field.name] = np.asarray(
                [np.nan if value is None else float(value) for value in values],
                dtype=np.float32,
            )
        else:
            arrays[field.name] = np.asarray(
                ["" if value is None else str(value) for value in values],
                dtype=np.str_,
            )
    return arrays


def _numpy_integer_dtype(data_type: pa.DataType) -> np.dtype[Any]:
    if pa.types.is_int8(data_type):
        return np.dtype(np.int8)
    if pa.types.is_int16(data_type):
        return np.dtype(np.int16)
    if pa.types.is_int32(data_type):
        return np.dtype(np.int32)
    return np.dtype(np.int64)


__all__ = [
    "AtomicAuditShardWriter",
    "AuditOutputFormat",
    "consequence_parity_schema",
]
