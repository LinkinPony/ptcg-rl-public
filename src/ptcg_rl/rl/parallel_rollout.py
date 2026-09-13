"""Multi-process rollout collection with merged manifests."""

from __future__ import annotations

import concurrent.futures
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch.multiprocessing as torch_mp

from ptcg_rl.data.kaggle_deck import records as deck_records


@dataclass(frozen=True)
class _WorkerSpec:
    worker_index: int
    config_data: dict[str, Any]
    output_dir: Path


def run_parallel_rollout(config: Any) -> dict[str, Any]:
    """Run rollout workers concurrently and write a merged manifest."""
    from ptcg_rl.rl.collection import RolloutConfig, resolve_rollout_output_dir

    rollout_config = RolloutConfig.model_validate(config)
    if rollout_config.parallel_workers <= 1:
        raise ValueError("parallel rollout requires parallel_workers > 1")

    output_dir = deck_records.repo_path(resolve_rollout_output_dir(rollout_config))
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_specs = _worker_specs(rollout_config, output_dir=output_dir)

    start = time.perf_counter()
    context = torch_mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=rollout_config.parallel_workers,
        mp_context=context,
    ) as executor:
        worker_summaries = tuple(executor.map(_run_rollout_worker, worker_specs))

    manifest = merge_parallel_rollout_manifests(
        output_dir=output_dir,
        worker_summaries=worker_summaries,
        config=rollout_config.model_dump(mode="json"),
        compression=rollout_config.compression,
    )
    elapsed_seconds = time.perf_counter() - start
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "mode": "parallel",
        "parallel_workers": rollout_config.parallel_workers,
        "elapsed_seconds": elapsed_seconds,
        "output_dir": deck_records.display_path(output_dir),
        "manifest_path": deck_records.display_path(output_dir / "manifest.json"),
        "games_path": deck_records.display_path(output_dir / "games.parquet"),
        "worker_output_dirs": [
            str(summary["output_dir"]) for summary in worker_summaries
        ],
        "worker_summaries": list(worker_summaries),
        "writer": manifest.get("summary", {}),
        "rates": _parallel_rates(manifest, elapsed_seconds),
    }
    return summary


def merge_parallel_rollout_manifests(
    *,
    output_dir: Path,
    worker_summaries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    compression: str,
) -> dict[str, Any]:
    """Merge worker rollout manifests into one training manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = tuple(_worker_manifest(summary) for summary in worker_summaries)
    if not manifests:
        raise ValueError("parallel rollout produced no worker manifests")

    games_path = output_dir / "games.parquet"
    game_rows, game_bytes = _merge_games_parquet(
        manifests,
        games_path=games_path,
        compression=compression,
    )
    shards = _merged_shards(manifests)
    row_bytes = sum(int(shard["bytes"]) for shard in shards)
    row_count = sum(int(shard["rows"]) for shard in shards)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": dict(config),
        "metadata": {
            "parallel_workers": len(worker_summaries),
            "worker_manifests": [
                str(summary["manifest_path"]) for summary in worker_summaries
            ],
        },
        "schema": str(manifests[0].get("schema", "")),
        "schema_version": manifests[0].get("schema_version"),
        "game_schema": str(manifests[0].get("game_schema", "")),
        "summary": {
            "games": game_rows,
            "rows": row_count,
            "shards": len(shards),
            "bytes": row_bytes + game_bytes,
            "row_bytes": row_bytes,
            "game_bytes": game_bytes,
        },
        "shards": shards,
        "games": {
            "path": deck_records.display_path(games_path),
            "rows": game_rows,
            "bytes": game_bytes,
        },
        "output_dir": deck_records.display_path(output_dir),
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def _worker_specs(config: Any, *, output_dir: Path) -> tuple[_WorkerSpec, ...]:
    games = _split_budget(config.total_games, config.parallel_workers)
    steps = _split_budget(config.total_steps, config.parallel_workers)
    specs: list[_WorkerSpec] = []
    for worker_index in range(config.parallel_workers):
        worker_output_dir = (
            output_dir
            / config.parallel_worker_output_subdir
            / f"worker-{worker_index:03d}"
        )
        worker_config = config.model_copy(
            update={
                "output_dir": worker_output_dir,
                "run": config.run.model_copy(
                    update={"version": f"{config.run.version}_w{worker_index:03d}"}
                ),
                "total_games": games[worker_index],
                "total_steps": steps[worker_index],
                "seed": config.seed + worker_index * 1009,
                "parallel_workers": 1,
            }
        )
        specs.append(
            _WorkerSpec(
                worker_index=worker_index,
                config_data=worker_config.model_dump(mode="python"),
                output_dir=worker_output_dir,
            )
        )
    return tuple(specs)


def _run_rollout_worker(spec: _WorkerSpec) -> dict[str, Any]:
    from ptcg_rl.rl.collection import RolloutConfig, _run_single_rollout

    config = RolloutConfig.model_validate(spec.config_data)
    summary = _run_single_rollout(config)
    summary["worker_index"] = spec.worker_index
    summary["worker_output_dir"] = deck_records.display_path(spec.output_dir)
    return summary


def _worker_manifest(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_path = summary.get("manifest_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("worker summary is missing manifest_path")
    path = deck_records.repo_path(Path(raw_path))
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"worker manifest must be a JSON object: {path}")
    return cast(Mapping[str, Any], raw)


def _merged_shards(manifests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for worker_index, manifest in enumerate(manifests):
        raw_shards = manifest.get("shards")
        if not isinstance(raw_shards, Sequence) or isinstance(raw_shards, str | bytes):
            raise ValueError("worker manifest shards must be a sequence")
        for shard_index, shard in enumerate(raw_shards):
            if not isinstance(shard, Mapping):
                raise ValueError("worker shard entries must be objects")
            path = shard.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError("worker shard entry is missing path")
            output.append(
                {
                    "path": path,
                    "rows": int(shard.get("rows", 0)),
                    "bytes": int(shard.get("bytes", 0)),
                    "worker_index": worker_index,
                    "worker_shard_index": shard_index,
                }
            )
    return output


def _merge_games_parquet(
    manifests: Sequence[Mapping[str, Any]],
    *,
    games_path: Path,
    compression: str,
) -> tuple[int, int]:
    writer: pq.ParquetWriter | None = None
    rows = 0
    tmp_path = games_path.with_name(f".{games_path.name}.tmp")
    try:
        for manifest in manifests:
            games = manifest.get("games")
            if not isinstance(games, Mapping):
                continue
            raw_path = games.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            source = deck_records.repo_path(Path(raw_path))
            parquet_file = pq.ParquetFile(source)
            for batch in parquet_file.iter_batches(batch_size=65_536):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(
                        tmp_path,
                        table.schema,
                        compression=compression,
                    )
                writer.write_table(table)
                rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pq.write_table(pa.table({}), tmp_path, compression=compression)
    tmp_path.replace(games_path)
    return (rows, games_path.stat().st_size)


def _split_budget(total: int | None, workers: int) -> tuple[int | None, ...]:
    if total is None:
        return tuple(None for _ in range(workers))
    base, remainder = divmod(int(total), workers)
    return tuple(base + (1 if index < remainder else 0) for index in range(workers))


def _parallel_rates(
    manifest: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, float]:
    summary = manifest.get("summary", {})
    if not isinstance(summary, Mapping) or elapsed_seconds <= 0.0:
        return {
            "recorded_rows_per_second": 0.0,
            "finished_games_per_second": 0.0,
        }
    return {
        "recorded_rows_per_second": float(summary.get("rows", 0)) / elapsed_seconds,
        "finished_games_per_second": float(summary.get("games", 0)) / elapsed_seconds,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
