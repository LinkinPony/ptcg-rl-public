"""Build Stage-0 advantage-weighted rollout manifests."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.data.kaggle_deck import records as deck_records


class AdvantageWeightRewriteConfig(BaseModel):
    """Config for appending an advantage-weight column to rollout shards."""

    model_config = ConfigDict(extra="forbid")

    input_manifest_path: Path
    output_dir: Path = Path("outputs/rl/rollouts/stage0_advfilter")
    weight_column: str = "advantage_weight"
    reward_column: str = "reward"
    value_column: str = "value_pred"
    margin: float = 0.0
    positive_weight: float = 1.0
    non_positive_weight: float = 0.0
    read_batch_size: int = 65_536
    compression: str = "zstd"
    copy_games: bool = True

    @field_validator("weight_column", "reward_column", "value_column", "compression")
    @classmethod
    def valid_non_empty_string(cls, value: str) -> str:
        """Reject empty string settings."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("advantage rewrite string settings must be non-empty")
        return cleaned

    @field_validator("positive_weight", "non_positive_weight")
    @classmethod
    def valid_non_negative_weight(cls, value: float) -> float:
        """Reject negative row weights."""
        if value < 0.0:
            raise ValueError("advantage weights must be non-negative")
        return value

    @field_validator("read_batch_size")
    @classmethod
    def valid_read_batch_size(cls, value: int) -> int:
        """Reject invalid read batch sizes."""
        if value <= 0:
            raise ValueError("read_batch_size must be positive")
        return value


def rewrite_advantage_weight_manifest(
    config: AdvantageWeightRewriteConfig,
) -> Mapping[str, Any]:
    """Append advantage weights to every shard and write a new manifest."""
    input_manifest_path = _manifest_path(config.input_manifest_path)
    manifest = _read_manifest(input_manifest_path)
    output_dir = deck_records.repo_path(config.output_dir)
    shard_dir = output_dir / "shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_reports: list[dict[str, Any]] = []
    positive_rows = 0
    total_rows = 0
    output_schema: pa.Schema | None = None
    for shard_index, shard in enumerate(_manifest_shards(manifest)):
        input_path = deck_records.repo_path(Path(str(shard["path"])))
        output_path = shard_dir / Path(input_path).name
        report = _rewrite_shard(
            input_path,
            output_path,
            config,
            shard_index=shard_index,
        )
        shard_reports.append(report.manifest_entry)
        positive_rows += report.positive_rows
        total_rows += report.rows
        output_schema = report.output_schema

    games_entry, game_bytes = _copy_games(manifest, output_dir) if config.copy_games else ({}, 0)
    manifest_path = output_dir / "manifest.json"
    output_manifest = _output_manifest(
        source_manifest=manifest,
        source_manifest_path=input_manifest_path,
        config=config,
        output_dir=output_dir,
        manifest_path=manifest_path,
        output_schema=output_schema,
        shard_reports=shard_reports,
        games_entry=games_entry,
        game_bytes=game_bytes,
        positive_rows=positive_rows,
        total_rows=total_rows,
    )
    _write_json_atomic(manifest_path, output_manifest)
    return {
        "manifest_path": deck_records.display_path(manifest_path),
        "output_dir": deck_records.display_path(output_dir),
        "rows": total_rows,
        "positive_rows": positive_rows,
        "non_positive_rows": total_rows - positive_rows,
        "positive_fraction": positive_rows / total_rows if total_rows else 0.0,
        "shards": len(shard_reports),
    }


class _ShardRewriteReport(BaseModel):
    """Stats from one rewritten Parquet shard."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    manifest_entry: dict[str, Any]
    rows: int
    positive_rows: int
    output_schema: pa.Schema


def _rewrite_shard(
    input_path: Path,
    output_path: Path,
    config: AdvantageWeightRewriteConfig,
    *,
    shard_index: int,
) -> _ShardRewriteReport:
    if not input_path.is_file():
        raise FileNotFoundError(f"input shard does not exist: {input_path}")
    parquet_file = pq.ParquetFile(input_path)
    writer: pq.ParquetWriter | None = None
    rows = 0
    positive_rows = 0
    schema: pa.Schema | None = None
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        for batch in parquet_file.iter_batches(batch_size=config.read_batch_size):
            table = _with_advantage_weight(pa.Table.from_batches([batch]), config)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(
                    tmp_path,
                    schema,
                    compression=config.compression,
                )
            writer.write_table(table)
            row_count = table.num_rows
            rows += row_count
            positive_rows += _positive_weight_count(table, config.weight_column)
    finally:
        if writer is not None:
            writer.close()

    if schema is None:
        schema = _empty_output_schema(parquet_file.schema_arrow, config.weight_column)
        pq.write_table(pa.Table.from_batches([], schema=schema), tmp_path)
    tmp_path.replace(output_path)
    return _ShardRewriteReport(
        manifest_entry={
            "path": deck_records.display_path(output_path),
            "rows": rows,
            "bytes": output_path.stat().st_size,
            "source_path": deck_records.display_path(input_path),
            "source_shard_index": shard_index,
        },
        rows=rows,
        positive_rows=positive_rows,
        output_schema=schema,
    )


def _with_advantage_weight(
    table: pa.Table,
    config: AdvantageWeightRewriteConfig,
) -> pa.Table:
    _require_columns(table, (config.reward_column, config.value_column))
    pc_any = cast(Any, pc)
    reward = pc.cast(table[config.reward_column], pa.float32())
    value = pc.cast(table[config.value_column], pa.float32())
    advantage = pc_any.subtract(reward, value)
    positive = pc_any.greater(advantage, pa.scalar(config.margin, type=pa.float32()))
    weights = pc_any.if_else(
        positive,
        pa.scalar(config.positive_weight, type=pa.float32()),
        pa.scalar(config.non_positive_weight, type=pa.float32()),
    )
    if config.weight_column in table.column_names:
        table = table.drop([config.weight_column])
    return table.append_column(config.weight_column, weights)


def _positive_weight_count(table: pa.Table, weight_column: str) -> int:
    pc_any = cast(Any, pc)
    weights = table[weight_column]
    positive = pc_any.greater(weights, pa.scalar(0.0, type=pa.float32()))
    total = pc_any.sum(pc.cast(positive, pa.int64())).as_py()
    return int(total or 0)


def _empty_output_schema(schema: pa.Schema, weight_column: str) -> pa.Schema:
    if weight_column in schema.names:
        schema = schema.remove(schema.get_field_index(weight_column))
    return schema.append(pa.field(weight_column, pa.float32()))


def _copy_games(
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], int]:
    games = manifest.get("games")
    if not isinstance(games, Mapping):
        return ({}, 0)
    raw_path = games.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return ({}, 0)
    source = deck_records.repo_path(Path(raw_path))
    if not source.exists():
        return ({}, 0)
    output = output_dir / "games.parquet"
    shutil.copy2(source, output)
    return (
        {
            "path": deck_records.display_path(output),
            "rows": int(games.get("rows", 0)),
            "bytes": output.stat().st_size,
            "source_path": deck_records.display_path(source),
        },
        output.stat().st_size,
    )


def _output_manifest(
    *,
    source_manifest: Mapping[str, Any],
    source_manifest_path: Path,
    config: AdvantageWeightRewriteConfig,
    output_dir: Path,
    manifest_path: Path,
    output_schema: pa.Schema | None,
    shard_reports: Sequence[Mapping[str, Any]],
    games_entry: Mapping[str, Any],
    game_bytes: int,
    positive_rows: int,
    total_rows: int,
) -> dict[str, Any]:
    row_bytes = sum(int(shard["bytes"]) for shard in shard_reports)
    manifest = dict(source_manifest)
    metadata = dict(cast(Mapping[str, Any], source_manifest.get("metadata", {})))
    metadata["advantage_weight_rewrite"] = {
        "source_manifest_path": deck_records.display_path(source_manifest_path),
        "weight_column": config.weight_column,
        "reward_column": config.reward_column,
        "value_column": config.value_column,
        "margin": config.margin,
        "positive_weight": config.positive_weight,
        "non_positive_weight": config.non_positive_weight,
    }
    manifest.update(
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "metadata": metadata,
            "schema": output_schema.to_string() if output_schema is not None else "",
            "shards": [dict(shard) for shard in shard_reports],
            "games": dict(games_entry),
            "output_dir": deck_records.display_path(output_dir),
            "manifest_path": deck_records.display_path(manifest_path),
            "summary": {
                "games": int(
                    cast(Mapping[str, Any], source_manifest.get("summary", {})).get(
                        "games", 0
                    )
                ),
                "rows": total_rows,
                "shards": len(shard_reports),
                "bytes": row_bytes + game_bytes,
                "row_bytes": row_bytes,
                "game_bytes": game_bytes,
                "advantage_positive_rows": positive_rows,
                "advantage_non_positive_rows": total_rows - positive_rows,
            },
        }
    )
    return manifest


def _manifest_path(path: Path) -> Path:
    resolved = deck_records.repo_path(path)
    if resolved.is_dir():
        return resolved / "manifest.json"
    return resolved


def _read_manifest(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return cast(Mapping[str, Any], raw)


def _manifest_shards(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, Sequence) or isinstance(raw_shards, str | bytes):
        raise ValueError("manifest shards must be a sequence")
    shards: list[Mapping[str, Any]] = []
    for shard in raw_shards:
        if not isinstance(shard, Mapping):
            continue
        raw_path = shard.get("path")
        if isinstance(raw_path, str) and raw_path:
            shards.append(cast(Mapping[str, Any], shard))
    if not shards:
        raise ValueError("manifest contains no shard paths")
    return tuple(shards)


def _require_columns(table: pa.Table, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in table.column_names]
    if missing:
        raise ValueError(f"input shard is missing required columns: {missing}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
