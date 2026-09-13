"""Atomic bounded Parquet writers for integrated planner profiling."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.evaluation.planner_profile_records import (
    PlannerProfileDecisionRecord,
    PlannerProfileRunRecord,
)
from ptcg_rl.evaluation.planner_profile_schemas import (
    decision_schema,
    run_schema,
    stage_schema,
)
from ptcg_rl.runtime.planner_telemetry import PlannerStageEvent


class PlannerProfileWriter:
    """Atomically publish bounded Parquet shards while a campaign runs."""

    def __init__(
        self,
        output_dir: Path,
        *,
        rows_per_shard: int,
        compression: str,
    ) -> None:
        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        self._aligned_parts = _AtomicDecisionStageParts(
            output_dir,
            decision_schema=decision_schema(),
            stage_schema=stage_schema(),
            compression=compression,
        )
        self._run_parts = _AtomicParquetParts(
            output_dir / "runs_parts",
            schema=run_schema(),
            compression=compression,
        )
        self._rows_per_shard = rows_per_shard
        self._decisions: list[dict[str, Any]] = []
        self._stages: list[dict[str, Any]] = []
        self._runs: list[dict[str, Any]] = []
        self._closed = False

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

    def append_decision(
        self,
        record: PlannerProfileDecisionRecord,
        events: Sequence[PlannerStageEvent],
    ) -> None:
        """Append one request and all fixed-shape stage events as one group."""
        self._require_open()
        self._decisions.append(record.model_dump(mode="json"))
        self._stages.extend(
            {
                "campaign_id": record.campaign_id,
                "point_id": record.point_id,
                "environment": record.environment,
                "decision_index": record.decision_index,
                "stage": str(event.stage),
                "seconds": event.seconds,
                "rows": event.rows,
                "bytes_count": event.bytes_count,
                "batch_capacity": event.batch_capacity,
            }
            for event in events
        )
        if len(self._decisions) >= self._rows_per_shard:
            self._flush_decisions()

    def append_run(self, record: PlannerProfileRunRecord) -> None:
        self._require_open()
        self._runs.append(record.model_dump(mode="json"))

    def close(self) -> None:
        if self._closed:
            return
        self._flush_decisions()
        self._run_parts.write(self._runs)
        self._runs.clear()
        self._closed = True

    def abort(self) -> None:
        self._aligned_parts.abort()
        self._run_parts.abort()
        self._decisions.clear()
        self._stages.clear()
        self._runs.clear()
        self._closed = True

    @property
    def part_counts(self) -> Mapping[str, int]:
        return {
            "decisions": self._aligned_parts.part_count,
            "stages": self._aligned_parts.part_count,
            "runs": self._run_parts.part_count,
        }

    def _flush_decisions(self) -> None:
        if not self._decisions:
            return
        self._aligned_parts.write(self._decisions, self._stages)
        self._decisions.clear()
        self._stages.clear()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed planner profile writer")


class _AtomicParquetParts:
    def __init__(self, directory: Path, *, schema: pa.Schema, compression: str) -> None:
        self._directory = directory
        self._schema = schema
        self._compression = compression
        self._part_index = 0
        directory.mkdir(parents=True, exist_ok=True)
        if tuple(directory.glob("part-*.parquet")):
            raise FileExistsError(f"planner profile output exists: {directory}")

    @property
    def part_count(self) -> int:
        return self._part_index

    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        destination = self._directory / f"part-{self._part_index:05d}.parquet"
        temporary = self._directory / f".{destination.name}.tmp"
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"refusing to replace profile part: {destination}")
        try:
            table = pa.Table.from_pylist([dict(row) for row in rows], self._schema)
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
            temporary.unlink(missing_ok=True)
        self._part_index += 1

    def abort(self) -> None:
        for temporary in self._directory.glob(".*.tmp"):
            temporary.unlink(missing_ok=True)


class _AtomicDecisionStageParts:
    """Publish decision/stage pairs only after one verifiable commit marker."""

    def __init__(
        self,
        output_dir: Path,
        *,
        decision_schema: pa.Schema,
        stage_schema: pa.Schema,
        compression: str,
    ) -> None:
        self._decision_dir = output_dir / "decisions_parts"
        self._stage_dir = output_dir / "stages_parts"
        self._commit_dir = output_dir / "aligned_part_commits"
        self._decision_schema = decision_schema
        self._stage_schema = stage_schema
        self._compression = compression
        self._part_index = 0
        self._published_without_commit: set[Path] = set()
        for directory in (
            self._decision_dir,
            self._stage_dir,
            self._commit_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        existing = (
            tuple(self._decision_dir.glob("part-*.parquet"))
            + tuple(self._stage_dir.glob("part-*.parquet"))
            + tuple(self._commit_dir.glob("part-*.json"))
        )
        if existing:
            raise FileExistsError("planner profile aligned output already exists")

    @property
    def part_count(self) -> int:
        return self._part_index

    def write(
        self,
        decisions: Sequence[Mapping[str, Any]],
        stages: Sequence[Mapping[str, Any]],
    ) -> None:
        if not decisions:
            if stages:
                raise ValueError("stage rows cannot be committed without decisions")
            return
        name = f"part-{self._part_index:05d}"
        decision_path = self._decision_dir / f"{name}.parquet"
        stage_path = self._stage_dir / f"{name}.parquet"
        commit_path = self._commit_dir / f"{name}.json"
        decision_temporary = self._decision_dir / f".{name}.parquet.tmp"
        stage_temporary = self._stage_dir / f".{name}.parquet.tmp"
        commit_temporary = self._commit_dir / f".{name}.json.tmp"
        paths = (
            decision_path,
            stage_path,
            commit_path,
            decision_temporary,
            stage_temporary,
            commit_temporary,
        )
        if any(path.exists() for path in paths):
            raise FileExistsError(f"refusing to replace aligned profile part: {name}")
        try:
            _write_parquet_temporary(
                decision_temporary,
                decisions,
                schema=self._decision_schema,
                compression=self._compression,
            )
            _write_parquet_temporary(
                stage_temporary,
                stages,
                schema=self._stage_schema,
                compression=self._compression,
            )
            os.replace(decision_temporary, decision_path)
            self._published_without_commit.add(decision_path)
            os.replace(stage_temporary, stage_path)
            self._published_without_commit.add(stage_path)
            marker = {
                "schema_version": 1,
                "part_index": self._part_index,
                "decision_rows": len(decisions),
                "stage_rows": len(stages),
                "decision_sha256": _file_sha256(decision_path),
                "stage_sha256": _file_sha256(stage_path),
            }
            with commit_temporary.open("w", encoding="utf-8") as output:
                json.dump(marker, output, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(commit_temporary, commit_path)
            self._published_without_commit.difference_update(
                (decision_path, stage_path)
            )
            self._part_index += 1
        finally:
            for temporary in (
                decision_temporary,
                stage_temporary,
                commit_temporary,
            ):
                temporary.unlink(missing_ok=True)

    def abort(self) -> None:
        for directory in (
            self._decision_dir,
            self._stage_dir,
            self._commit_dir,
        ):
            for temporary in directory.glob(".*.tmp"):
                temporary.unlink(missing_ok=True)
        for path in self._published_without_commit:
            path.unlink(missing_ok=True)
        self._published_without_commit.clear()


def _write_parquet_temporary(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    schema: pa.Schema,
    compression: str,
) -> None:
    table = pa.Table.from_pylist([dict(row) for row in rows], schema)
    pq.write_table(
        table,
        path,
        compression=compression,
        write_statistics=True,
    )
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_profile_summary(path: Path, summary: Mapping[str, Any]) -> None:
    """Atomically publish one small campaign summary without replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to replace planner profile summary: {path}")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(dict(summary), output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["PlannerProfileWriter", "write_profile_summary"]
