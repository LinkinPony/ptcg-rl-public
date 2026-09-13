"""Run prepared bundle evaluation on this host and an explicitly configured remote host.

The coordinator shards candidates across both hosts while keeping the full
opponent/seat/block support on each shard, synchronizes required sources and
inputs, collects both Parquet results, then scores them together. ``worker``
mode is an implementation detail used by the local and SSH child processes.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, TextIO

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.bundle_gauntlet import (
    BundleGauntletConfig,
    EvaluationBundleConfig,
    run_bundle_gauntlet,
)
from ptcg_rl.evaluation.deck_strength import DeckStrengthConfig, score_deck_strength
from ptcg_rl.submission.release_assets import (
    load_release_bundle_for_native_execution,
)
from ptcg_rl.training.run_config import resolve_training_output_dir

_DEFAULT_REMOTE_HOST = None
_DEFAULT_REMOTE_REPO = None
_DEFAULT_REMOTE_PYTHON = "python3"
_DEFAULT_WORKERS = 8


def main() -> None:
    """Run one worker shard or coordinate the default two-host evaluation."""
    args = _parse_args()
    if (
        not args.worker
        and not args.local_only
        and (not args.remote_host or args.remote_repo is None)
    ):
        raise ValueError("Use --local-only or supply --remote-host and --remote-repo")
    bundle_config = BundleGauntletConfig.model_validate_json(
        args.bundle_config.read_text(encoding="utf-8")
    )
    if args.worker:
        _run_worker(bundle_config, status_path=args.status_path)
        return
    if args.score_config is None:
        raise ValueError("--score-config is required in coordinator mode")
    score_config = DeckStrengthConfig.model_validate_json(
        args.score_config.read_text(encoding="utf-8")
    )
    if args.local_only:
        score_summary = _run_local(
            bundle_config.model_copy(update={"num_workers": args.local_workers}),
            score_config,
            status_path=args.status_path,
            candidate_shard_limit=args.local_candidate_shards,
        )
        print(json.dumps(score_summary, indent=2, sort_keys=True))
        return
    if args.remote_only:
        try:
            score_summary = _run_remote_only(
                bundle_config,
                score_config,
                bundle_config_path=args.bundle_config,
                status_path=args.status_path,
                remote_host=args.remote_host,
                remote_repo=args.remote_repo,
                remote_python=args.remote_python,
                remote_workers=args.remote_workers,
            )
        except Exception as exc:
            _write_status(
                args.status_path,
                {
                    "status": "failed",
                    "mode": "single_remote_host_all_candidates",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise
        print(json.dumps(score_summary, indent=2, sort_keys=True))
        return
    try:
        score_summary = _run_distributed(
            bundle_config,
            score_config,
            bundle_config_path=args.bundle_config,
            score_config_path=args.score_config,
            status_path=args.status_path,
            remote_host=args.remote_host,
            remote_repo=args.remote_repo,
            remote_python=args.remote_python,
            local_workers=args.local_workers,
            remote_workers=args.remote_workers,
        )
    except Exception as exc:
        _write_status(
            args.status_path,
            {
                "status": "failed",
                "mode": "distributed_local_and_remote",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise
    print(json.dumps(score_summary, indent=2, sort_keys=True))


def _run_local(
    bundle_config: BundleGauntletConfig,
    score_config: DeckStrengthConfig,
    *,
    status_path: Path,
    candidate_shard_limit: int = 1,
) -> Mapping[str, Any]:
    """Run every candidate on one host, with optional sequential VRAM shards."""
    candidate_shards = _local_candidate_shards(
        bundle_config.candidates,
        shard_limit=candidate_shard_limit,
    )
    base_status = {
        "mode": "single_host_all_candidates",
        "candidate_bundles": len(bundle_config.candidates),
        "opponent_bundles": len(bundle_config.opponents),
        "num_workers": bundle_config.num_workers,
        "candidate_shards": len(candidate_shards),
    }
    _write_status(status_path, {"status": "running_games", **base_status})
    games_paths: list[Path] = []
    games_completed = 0
    games_total = _planned_games(bundle_config)
    try:
        for shard_index, shard_candidates in enumerate(candidate_shards):
            shard_config = _local_shard_config(
                bundle_config,
                candidates=shard_candidates,
                shard_index=shard_index,
                shard_count=len(candidate_shards),
            )
            shard_config_path = (
                status_path.parent
                / "candidate_shards"
                / f"shard-{shard_index:02d}-bundle-config.json"
            )
            _write_model_json(shard_config_path, shard_config)

            def report_progress(
                progress: Mapping[str, Any],
                *,
                completed_before: int = games_completed,
                current_shard: int = shard_index,
            ) -> None:
                combined = _local_shard_progress(
                    progress,
                    completed_before=completed_before,
                    games_total=games_total,
                )
                _write_status(
                    status_path,
                    {
                        "status": "running_games",
                        **base_status,
                        "candidate_shard_index": current_shard,
                        **combined,
                    },
                )

            games_summary = run_bundle_gauntlet(
                shard_config,
                progress_callback=report_progress,
            )
            games_path = records.repo_path(Path(str(games_summary["games_path"])))
            games_paths.append(games_path)
            games_completed += int(games_summary["games"])
        canonical_games_path = _canonical_games_path(bundle_config)
        _publish_canonical_games(
            games_paths,
            output_path=canonical_games_path,
            expected_rows=games_completed,
            compression=bundle_config.compression,
        )
        local_score = score_config.model_copy(
            update={"games_paths": (canonical_games_path,)}
        )
        _write_status(
            status_path,
            {
                "status": "scoring",
                **base_status,
                "games": games_completed,
                "games_paths": [str(canonical_games_path)],
                "source_games_paths": [str(path) for path in games_paths],
            },
        )
        score_summary = score_deck_strength(local_score)
    except Exception as exc:
        _write_status(
            status_path,
            {
                "status": "failed",
                **base_status,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise
    _write_status(
        status_path,
        {
            "status": "completed",
            **base_status,
            "games": int(score_summary["games"]["total"]),
            "games_paths": [str(canonical_games_path)],
            "source_games_paths": [str(path) for path in games_paths],
            "quality_warnings": list(score_summary["quality_warnings"]),
            "score_output_dir": str(score_summary["output_dir"]),
        },
    )
    return score_summary


def _local_candidate_shards(
    candidates: Sequence[EvaluationBundleConfig],
    *,
    shard_limit: int,
) -> tuple[tuple[EvaluationBundleConfig, ...], ...]:
    """Group candidates by checkpoint, then cap sequential shard count."""
    if shard_limit <= 0:
        raise ValueError("local candidate shard limit must be positive")
    if shard_limit == 1:
        return (tuple(candidates),)
    checkpoint_groups: dict[str, list[EvaluationBundleConfig]] = {}
    for candidate in candidates:
        checkpoint = candidate.agent.checkpoint_path
        key = str(checkpoint) if checkpoint is not None else candidate.bundle_id
        checkpoint_groups.setdefault(key, []).append(candidate)
    buckets: list[list[EvaluationBundleConfig]] = [
        [] for _ in range(min(shard_limit, len(checkpoint_groups)))
    ]
    for group_index, group in enumerate(checkpoint_groups.values()):
        buckets[group_index % len(buckets)].extend(group)
    return tuple(tuple(bucket) for bucket in buckets if bucket)


def _local_shard_config(
    config: BundleGauntletConfig,
    *,
    candidates: Sequence[EvaluationBundleConfig],
    shard_index: int,
    shard_count: int,
) -> BundleGauntletConfig:
    if shard_count == 1:
        return config
    root_output = records.repo_path(
        resolve_training_output_dir(
            task_name="bundle_gauntlet",
            run=config.run,
            output_dir=config.output_dir,
        )
    )
    return config.model_copy(
        update={
            "candidates": tuple(candidates),
            "output_dir": root_output / "candidate_shards" / f"shard-{shard_index:02d}",
        }
    )


def _planned_games(config: BundleGauntletConfig) -> int:
    games_per_candidate = sum(
        config.schedule.games_per_opponent.get(
            opponent.bundle_id,
            config.schedule.games_per_matchup,
        )
        for opponent in config.opponents
    )
    return len(config.candidates) * games_per_candidate


def _local_shard_progress(
    progress: Mapping[str, Any],
    *,
    completed_before: int,
    games_total: int,
) -> dict[str, Any]:
    committed = completed_before + int(progress.get("games_committed", 0))
    finished = completed_before + int(progress.get("games_finished", 0))
    fraction = committed / games_total if games_total else 1.0
    rate = float(progress.get("session_games_per_second", 0.0) or 0.0)
    remaining = max(0, games_total - committed)
    return {
        **progress,
        "games_total": games_total,
        "games_committed": committed,
        "games_finished": finished,
        "games_remaining": remaining,
        "progress_fraction": fraction,
        "progress_percent": fraction * 100.0,
        "eta_seconds": remaining / rate if rate > 0.0 else None,
    }


def _canonical_games_path(config: BundleGauntletConfig) -> Path:
    """Return the conventional root artifact consumed by later decisions."""
    root_output = records.repo_path(
        resolve_training_output_dir(
            task_name="bundle_gauntlet",
            run=config.run,
            output_dir=config.output_dir,
        )
    )
    return root_output / "games.parquet"


def _publish_canonical_games(
    source_paths: Sequence[Path],
    *,
    output_path: Path,
    expected_rows: int,
    compression: str,
) -> Path:
    """Stream shard Parquets into one atomically published canonical artifact."""
    sources = tuple(records.repo_path(path) for path in source_paths)
    if not sources:
        raise ValueError("cannot publish canonical games without source shards")
    missing = [path for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cannot publish missing game shards: {missing}")
    resolved_output = records.repo_path(output_path)
    if len(sources) == 1 and sources[0] == resolved_output:
        rows = int(pq.ParquetFile(resolved_output).metadata.num_rows)
        if rows != expected_rows:
            raise ValueError(
                "canonical games row count does not match completed games: "
                f"{rows}/{expected_rows}"
            )
        return resolved_output

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_output.with_name(f".{resolved_output.name}.tmp")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    written_rows = 0
    try:
        try:
            for source in sources:
                parquet_file = pq.ParquetFile(source)
                source_schema = parquet_file.schema_arrow.remove_metadata()
                if schema is None:
                    schema = source_schema
                    writer = pq.ParquetWriter(
                        temporary,
                        schema,
                        compression=compression,
                    )
                elif not source_schema.equals(schema, check_metadata=False):
                    raise ValueError(
                        "game shard schemas do not match: "
                        f"{source} differs from {sources[0]}"
                    )
                for batch in parquet_file.iter_batches(batch_size=65_536):
                    table = pa.Table.from_batches([batch]).replace_schema_metadata(None)
                    if not table.schema.equals(schema, check_metadata=False):
                        raise ValueError(f"game shard batch schema changed: {source}")
                    assert writer is not None
                    writer.write_table(table)
                    written_rows += table.num_rows
        finally:
            if writer is not None:
                writer.close()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if writer is None or written_rows != expected_rows:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            "game shards did not compact to the expected row count: "
            f"{written_rows}/{expected_rows}"
        )
    temporary.replace(resolved_output)
    return resolved_output


def _run_distributed(
    bundle_config: BundleGauntletConfig,
    score_config: DeckStrengthConfig,
    *,
    bundle_config_path: Path,
    score_config_path: Path,
    status_path: Path,
    remote_host: str,
    remote_repo: Path,
    remote_python: str,
    local_workers: int,
    remote_workers: int,
) -> Mapping[str, Any]:
    candidates_local, candidates_remote = _split_candidates(bundle_config.candidates)
    root_output = records.repo_path(
        resolve_training_output_dir(
            task_name="bundle_gauntlet",
            run=bundle_config.run,
            output_dir=bundle_config.output_dir,
        )
    )
    local_output = root_output / "shards" / "local"
    remote_output = root_output / "shards" / remote_host
    local_config = bundle_config.model_copy(
        update={
            "candidates": candidates_local,
            "output_dir": local_output,
            "num_workers": local_workers,
        }
    )
    remote_sync_config = bundle_config.model_copy(
        update={
            "candidates": candidates_remote,
            "output_dir": remote_output,
            "num_workers": remote_workers,
        }
    )
    remote_config = _config_for_remote_repo(
        remote_sync_config,
        remote_repo=remote_repo,
    )
    remote_execution_output = _path_for_remote_repo(
        remote_output,
        remote_repo=remote_repo,
    )
    launch_dir = status_path.parent
    shard_dir = launch_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    local_config_path = shard_dir / "local_bundle_config.json"
    remote_config_path = shard_dir / f"{remote_host}_bundle_config.json"
    local_status_path = shard_dir / "local_status.json"
    remote_status_path = shard_dir / f"{remote_host}_status.json"
    local_log_path = shard_dir / "local.log"
    remote_log_path = shard_dir / f"{remote_host}.log"
    remote_pid_path = shard_dir / f"{remote_host}.pid"
    remote_status_absolute = _remote_absolute_path(remote_repo, remote_status_path)
    remote_pid_absolute = _remote_absolute_path(remote_repo, remote_pid_path)
    _write_model_json(local_config_path, local_config)
    _write_model_json(remote_config_path, remote_config)
    _sync_remote_inputs(
        remote_sync_config,
        config_path=remote_config_path,
        remote_host=remote_host,
        remote_repo=remote_repo,
    )

    _write_status(
        status_path,
        {
            "status": "running_games",
            "mode": "distributed_local_and_remote",
            "coordinator_pid": os.getpid(),
            "bundle_config": str(bundle_config_path),
            "score_config": str(score_config_path),
            "local_candidates": len(candidates_local),
            "remote_candidates": len(candidates_remote),
            "remote_host": remote_host,
            "local_workers": local_workers,
            "remote_workers": remote_workers,
        },
    )
    local_command = _worker_command(
        python_executable=sys.executable,
        bundle_config_path=local_config_path,
        status_path=local_status_path,
    )
    remote_command = _remote_worker_command(
        remote_repo=remote_repo,
        remote_python=remote_python,
        bundle_config_path=remote_config_path,
        status_path=remote_status_path,
        pid_path=remote_pid_path,
    )
    local_log = local_log_path.open("w", encoding="utf-8")
    remote_log = remote_log_path.open("w", encoding="utf-8")
    local_process = _start_process(local_command, log=local_log)
    remote_process = _start_process(
        ("ssh", remote_host, f"bash -lc {shlex.quote(remote_command)}"),
        log=remote_log,
    )
    try:
        _wait_for_shards(
            local_process,
            remote_process,
            status_path=status_path,
            local_status_path=local_status_path,
            remote_status_path=remote_status_absolute,
            remote_host=remote_host,
        )
    except BaseException:
        _terminate_process(local_process)
        _terminate_process(remote_process)
        _kill_remote_worker(remote_host, remote_pid_absolute)
        raise
    finally:
        local_log.close()
        remote_log.close()

    _sync_remote_results(
        remote_host=remote_host,
        remote_output=remote_execution_output,
        local_destination=remote_output,
        remote_status_path=remote_status_absolute,
        local_status_path=remote_status_path,
    )
    games_paths = (
        local_output / "games.parquet",
        remote_output / "games.parquet",
    )
    for games_path in games_paths:
        if not games_path.exists():
            raise FileNotFoundError(
                f"distributed shard produced no games: {games_path}"
            )
    distributed_score_config = score_config.model_copy(
        update={"games_paths": games_paths}
    )
    distributed_score_path = shard_dir / "distributed_score_config.json"
    _write_model_json(distributed_score_path, distributed_score_config)
    _write_status(
        status_path,
        {
            "status": "scoring",
            "mode": "distributed_local_and_remote",
            "coordinator_pid": os.getpid(),
            "games_paths": [str(path) for path in games_paths],
        },
    )
    score_summary = score_deck_strength(distributed_score_config)
    _write_status(
        status_path,
        {
            "status": "completed",
            "mode": "distributed_local_and_remote",
            "coordinator_pid": os.getpid(),
            "games": int(score_summary["games"]["total"]),
            "games_paths": [str(path) for path in games_paths],
            "quality_warnings": list(score_summary["quality_warnings"]),
            "score_output_dir": str(score_summary["output_dir"]),
        },
    )
    return score_summary


def _run_remote_only(
    bundle_config: BundleGauntletConfig,
    score_config: DeckStrengthConfig,
    *,
    bundle_config_path: Path,
    status_path: Path,
    remote_host: str,
    remote_repo: Path,
    remote_python: str,
    remote_workers: int,
) -> Mapping[str, Any]:
    """Run every candidate on one idle remote host and score locally."""
    root_output = records.repo_path(
        resolve_training_output_dir(
            task_name="bundle_gauntlet",
            run=bundle_config.run,
            output_dir=bundle_config.output_dir,
        )
    )
    remote_sync_config = bundle_config.model_copy(
        update={"output_dir": root_output, "num_workers": remote_workers}
    )
    remote_config = _config_for_remote_repo(
        remote_sync_config,
        remote_repo=remote_repo,
    )
    remote_execution_output = _path_for_remote_repo(
        root_output,
        remote_repo=remote_repo,
    )
    shard_dir = status_path.parent / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    remote_config_path = shard_dir / f"{remote_host}_all_bundle_config.json"
    remote_status_path = shard_dir / f"{remote_host}_all_status.json"
    remote_log_path = shard_dir / f"{remote_host}_all.log"
    remote_pid_path = shard_dir / f"{remote_host}_all.pid"
    remote_status_absolute = _remote_absolute_path(remote_repo, remote_status_path)
    remote_pid_absolute = _remote_absolute_path(remote_repo, remote_pid_path)
    _write_model_json(remote_config_path, remote_config)
    _sync_remote_inputs(
        remote_sync_config,
        config_path=remote_config_path,
        remote_host=remote_host,
        remote_repo=remote_repo,
    )
    _write_status(
        status_path,
        {
            "status": "running_games",
            "mode": "single_remote_host_all_candidates",
            "coordinator_pid": os.getpid(),
            "bundle_config": str(bundle_config_path),
            "remote_candidates": len(bundle_config.candidates),
            "remote_host": remote_host,
            "remote_workers": remote_workers,
        },
    )
    remote_command = _remote_worker_command(
        remote_repo=remote_repo,
        remote_python=remote_python,
        bundle_config_path=remote_config_path,
        status_path=remote_status_path,
        pid_path=remote_pid_path,
    )
    remote_log = remote_log_path.open("w", encoding="utf-8")
    remote_process = _start_process(
        ("ssh", remote_host, f"bash -lc {shlex.quote(remote_command)}"),
        log=remote_log,
    )
    try:
        _wait_for_remote(
            remote_process,
            status_path=status_path,
            remote_status_path=remote_status_absolute,
            remote_host=remote_host,
        )
    except BaseException:
        _terminate_process(remote_process)
        _kill_remote_worker(remote_host, remote_pid_absolute)
        raise
    finally:
        remote_log.close()

    _sync_remote_results(
        remote_host=remote_host,
        remote_output=remote_execution_output,
        local_destination=root_output,
        remote_status_path=remote_status_absolute,
        local_status_path=remote_status_path,
    )
    games_path = root_output / "games.parquet"
    if not games_path.exists():
        raise FileNotFoundError(f"remote evaluation produced no games: {games_path}")
    remote_score_config = score_config.model_copy(update={"games_paths": (games_path,)})
    _write_status(
        status_path,
        {
            "status": "scoring",
            "mode": "single_remote_host_all_candidates",
            "coordinator_pid": os.getpid(),
            "games_paths": [str(games_path)],
        },
    )
    score_summary = score_deck_strength(remote_score_config)
    _write_status(
        status_path,
        {
            "status": "completed",
            "mode": "single_remote_host_all_candidates",
            "coordinator_pid": os.getpid(),
            "games": int(score_summary["games"]["total"]),
            "games_paths": [str(games_path)],
            "quality_warnings": list(score_summary["quality_warnings"]),
            "score_output_dir": str(score_summary["output_dir"]),
        },
    )
    return score_summary


def _run_worker(config: BundleGauntletConfig, *, status_path: Path) -> None:
    base_status = {
        "candidate_bundles": len(config.candidates),
        "opponent_bundles": len(config.opponents),
        "num_workers": config.num_workers,
    }

    def report_progress(progress: Mapping[str, Any]) -> None:
        _write_status(
            status_path,
            {
                "status": "running_games",
                **base_status,
                **progress,
            },
        )

    _write_status(
        status_path,
        {
            "status": "running_games",
            **base_status,
        },
    )
    try:
        summary = run_bundle_gauntlet(config, progress_callback=report_progress)
    except Exception as exc:
        _write_status(
            status_path,
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise
    _write_status(
        status_path,
        {
            "status": "completed_games",
            "games": int(summary["games"]),
            "games_total": int(summary["planned_games"]),
            "games_committed": int(summary["games"]),
            "games_finished": int(summary["games"]),
            "games_remaining": 0,
            "progress_fraction": 1.0,
            "progress_percent": 100.0,
            "resumed_games": int(summary["resumed_games"]),
            "result_parts": int(summary["result_parts"]),
            "games_path": str(summary["games_path"]),
            "progress_path": str(summary["progress_path"]),
        },
    )


def _split_candidates(
    candidates: Sequence[EvaluationBundleConfig],
) -> tuple[tuple[EvaluationBundleConfig, ...], tuple[EvaluationBundleConfig, ...]]:
    """Alternate candidates across local and remote while preserving order."""
    if len(candidates) < 2:
        raise ValueError("distributed evaluation requires at least two candidates")
    return (tuple(candidates[::2]), tuple(candidates[1::2]))


def _sync_remote_inputs(
    config: BundleGauntletConfig,
    *,
    config_path: Path,
    remote_host: str,
    remote_repo: Path,
) -> None:
    repo_root = records.repo_path(Path("."))
    _run_checked(("ssh", remote_host, f"mkdir -p {shlex.quote(str(remote_repo))}"))
    for source, destination in (
        (
            repo_root / "src" / "ptcg_rl" / "evaluation",
            remote_repo / "src" / "ptcg_rl" / "evaluation",
        ),
        (
            repo_root / "src" / "tools" / "run_bundle_evaluation.py",
            remote_repo / "src" / "tools" / "run_bundle_evaluation.py",
        ),
        (
            repo_root / "src" / "ptcg_rl" / "training" / "arena.py",
            remote_repo / "src" / "ptcg_rl" / "training" / "arena.py",
        ),
        (
            repo_root / "src" / "ptcg_rl" / "agent" / "simple_stateless_runtime.py",
            remote_repo / "src" / "ptcg_rl" / "agent" / "simple_stateless_runtime.py",
        ),
    ):
        source_text = f"{source}/" if source.is_dir() else str(source)
        destination_text = f"{destination}/" if source.is_dir() else str(destination)
        _run_checked(
            (
                "rsync",
                "-az",
                source_text,
                f"{remote_host}:{destination_text}",
            )
        )
    input_paths = _remote_input_paths(config, config_path=config_path)
    relative_paths = [
        str(path.relative_to(repo_root))
        for path in input_paths
        if path.is_relative_to(repo_root)
        and "third_party" not in path.relative_to(repo_root).parts
    ]
    if relative_paths:
        _run_checked(
            (
                "rsync",
                "-azR",
                *relative_paths,
                f"{remote_host}:{remote_repo}/",
            ),
            cwd=repo_root,
        )


def _remote_input_paths(
    config: BundleGauntletConfig,
    *,
    config_path: Path,
) -> tuple[Path, ...]:
    paths = {records.repo_path(config_path)}
    if any(
        bundle.agent.checkpoint_path is not None
        for bundle in (*config.candidates, *config.opponents)
    ):
        paths.add(
            records.repo_path(
                Path("outputs/cards/static_features/card_static_features.npy")
            )
        )
    for bundle in (*config.candidates, *config.opponents):
        paths.add(records.repo_path(bundle.deck_path))
        for optional_path in (
            bundle.agent.checkpoint_path,
            bundle.agent.belief_summary_path,
            bundle.agent.public_catalog_manifest_path,
            bundle.agent.release_manifest_path,
        ):
            if optional_path is not None:
                paths.add(records.repo_path(optional_path))
        if bundle.agent.public_catalog_manifest_path is not None:
            catalog_manifest = records.repo_path(
                bundle.agent.public_catalog_manifest_path
            )
            catalog_payload = _read_json(catalog_manifest)
            if catalog_payload is None:
                raise ValueError(f"invalid public catalog manifest: {catalog_manifest}")
            artifact_filename = catalog_payload.get("artifact_filename")
            if not isinstance(artifact_filename, str) or not artifact_filename:
                raise ValueError(
                    f"public catalog manifest omits artifact filename: "
                    f"{catalog_manifest}"
                )
            paths.add((catalog_manifest.parent / artifact_filename).resolve())
        if bundle.agent.release_manifest_path is not None:
            release = load_release_bundle_for_native_execution(
                records.repo_path(bundle.agent.release_manifest_path)
            )
            paths.update(
                records.repo_path(path)
                for path in (
                    release.checkpoint_path,
                    release.deck_path,
                    release.runtime_archive_path,
                )
            )
            if release.belief_path is not None:
                paths.add(records.repo_path(release.belief_path))
    missing = sorted(path for path in paths if not path.exists())
    if missing:
        raise FileNotFoundError(f"cannot sync missing evaluation inputs: {missing}")
    return tuple(sorted(paths))


def _config_for_remote_repo(
    config: BundleGauntletConfig,
    *,
    remote_repo: Path,
) -> BundleGauntletConfig:
    """Rewrite repository-absolute paths for an isolated remote snapshot."""
    bundles: list[EvaluationBundleConfig] = []
    for bundle in (*config.candidates, *config.opponents):
        agent = bundle.agent
        agent = agent.model_copy(
            update={
                "checkpoint_path": _optional_path_for_remote_repo(
                    agent.checkpoint_path,
                    remote_repo=remote_repo,
                ),
                "belief_summary_path": _optional_path_for_remote_repo(
                    agent.belief_summary_path,
                    remote_repo=remote_repo,
                ),
                "public_catalog_manifest_path": _optional_path_for_remote_repo(
                    agent.public_catalog_manifest_path,
                    remote_repo=remote_repo,
                ),
                "release_manifest_path": _optional_path_for_remote_repo(
                    agent.release_manifest_path,
                    remote_repo=remote_repo,
                ),
            }
        )
        bundles.append(
            bundle.model_copy(
                update={
                    "deck_path": _path_for_remote_repo(
                        bundle.deck_path,
                        remote_repo=remote_repo,
                    ),
                    "agent": agent,
                }
            )
        )
    candidate_count = len(config.candidates)
    return config.model_copy(
        update={
            "candidates": tuple(bundles[:candidate_count]),
            "opponents": tuple(bundles[candidate_count:]),
            "output_dir": _optional_path_for_remote_repo(
                config.output_dir,
                remote_repo=remote_repo,
            ),
        }
    )


def _optional_path_for_remote_repo(
    path: Path | None,
    *,
    remote_repo: Path,
) -> Path | None:
    if path is None:
        return None
    return _path_for_remote_repo(path, remote_repo=remote_repo)


def _path_for_remote_repo(path: Path, *, remote_repo: Path) -> Path:
    """Map an absolute local-repository path under the remote repository."""
    if not path.is_absolute():
        return path
    repo_root = records.repo_path(Path(".")).resolve()
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(repo_root):
        return path
    return remote_repo / resolved.relative_to(repo_root)


def _remote_absolute_path(remote_repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else remote_repo / path


def _worker_command(
    *,
    python_executable: str,
    bundle_config_path: Path,
    status_path: Path,
) -> tuple[str, ...]:
    return (
        "env",
        "PYTHONPATH=data/sample_submission:src",
        "CUDA_VISIBLE_DEVICES=0",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        python_executable,
        "-u",
        "src/tools/run_bundle_evaluation.py",
        "--worker",
        "--bundle-config",
        str(bundle_config_path),
        "--status-path",
        str(status_path),
    )


def _remote_worker_command(
    *,
    remote_repo: Path,
    remote_python: str,
    bundle_config_path: Path,
    status_path: Path,
    pid_path: Path,
) -> str:
    command = _worker_command(
        python_executable=remote_python,
        bundle_config_path=bundle_config_path,
        status_path=status_path,
    )
    quoted_command = " ".join(shlex.quote(part) for part in command)
    return (
        f"cd {shlex.quote(str(remote_repo))} || exit 1; "
        f"echo $$ > {shlex.quote(str(pid_path))}; "
        f"exec {quoted_command}"
    )


def _start_process(command: Sequence[str], *, log: TextIO) -> subprocess.Popen[str]:
    return subprocess.Popen(
        tuple(command),
        cwd=records.repo_path(Path(".")),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _wait_for_shards(
    local_process: subprocess.Popen[str],
    remote_process: subprocess.Popen[str],
    *,
    status_path: Path,
    local_status_path: Path,
    remote_status_path: Path,
    remote_host: str,
) -> None:
    while True:
        local_code = local_process.poll()
        remote_code = remote_process.poll()
        local_status = _read_json(local_status_path)
        remote_status = _read_remote_status(remote_host, remote_status_path)
        _write_status(
            status_path,
            {
                "status": "running_games",
                "mode": "distributed_local_and_remote",
                "coordinator_pid": os.getpid(),
                "remote_host": remote_host,
                "local_pid": local_process.pid,
                "remote_ssh_pid": remote_process.pid,
                "local_exit_code": local_code,
                "remote_exit_code": remote_code,
                **_aggregate_progress(local_status, remote_status),
                "local": local_status,
                "remote": remote_status,
            },
        )
        if local_code not in {None, 0}:
            raise RuntimeError(f"local evaluation shard exited with code {local_code}")
        if remote_code not in {None, 0}:
            raise RuntimeError(
                f"remote evaluation shard exited with code {remote_code}"
            )
        if local_code == 0 and remote_code == 0:
            return
        time.sleep(10.0)


def _wait_for_remote(
    remote_process: subprocess.Popen[str],
    *,
    status_path: Path,
    remote_status_path: Path,
    remote_host: str,
) -> None:
    """Mirror one remote worker's durable progress into coordinator status."""
    while True:
        remote_code = remote_process.poll()
        remote_status = _read_remote_status(remote_host, remote_status_path)
        progress = {
            key: remote_status[key]
            for key in (
                "games_total",
                "games_committed",
                "games_finished",
                "games_remaining",
                "progress_fraction",
                "progress_percent",
                "eta_seconds",
                "session_games_per_second",
            )
            if remote_status is not None and key in remote_status
        }
        _write_status(
            status_path,
            {
                "status": "running_games",
                "mode": "single_remote_host_all_candidates",
                "coordinator_pid": os.getpid(),
                "remote_host": remote_host,
                "remote_ssh_pid": remote_process.pid,
                "remote_exit_code": remote_code,
                **progress,
                "remote": remote_status,
            },
        )
        if remote_code not in {None, 0}:
            raise RuntimeError(
                f"remote evaluation worker exited with code {remote_code}"
            )
        if remote_code == 0:
            return
        time.sleep(10.0)


def _aggregate_progress(
    local_status: Mapping[str, Any] | None,
    remote_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Combine exact durable progress from both worker shards."""
    if local_status is None or remote_status is None:
        return {}
    required = ("games_total", "games_committed", "games_finished")
    if any(
        key not in status
        for status in (local_status, remote_status)
        for key in required
    ):
        return {}
    total = sum(int(status["games_total"]) for status in (local_status, remote_status))
    committed = sum(
        int(status["games_committed"]) for status in (local_status, remote_status)
    )
    finished = sum(
        int(status["games_finished"]) for status in (local_status, remote_status)
    )
    if total <= 0:
        return {}
    return {
        "games_total": total,
        "games_committed": committed,
        "games_finished": finished,
        "games_remaining": total - committed,
        "progress_fraction": committed / total,
        "progress_percent": 100.0 * committed / total,
    }


def _sync_remote_results(
    *,
    remote_host: str,
    remote_output: Path,
    local_destination: Path,
    remote_status_path: Path,
    local_status_path: Path,
) -> None:
    local_destination.mkdir(parents=True, exist_ok=True)
    _run_checked(
        (
            "rsync",
            "-az",
            f"{remote_host}:{remote_output}/",
            f"{local_destination}/",
        )
    )
    local_status_path.parent.mkdir(parents=True, exist_ok=True)
    _run_checked(
        (
            "rsync",
            "-az",
            f"{remote_host}:{remote_status_path}",
            str(local_status_path),
        )
    )


def _kill_remote_worker(remote_host: str, pid_path: Path) -> None:
    command = (
        f"test ! -f {shlex.quote(str(pid_path))} || "
        f"kill -TERM -- -$(cat {shlex.quote(str(pid_path))}) 2>/dev/null || "
        f"kill -TERM $(cat {shlex.quote(str(pid_path))}) 2>/dev/null || true"
    )
    subprocess.run(("ssh", remote_host, command), check=False, timeout=30.0)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, 15)
        process.wait(timeout=10.0)
    except (OSError, subprocess.TimeoutExpired):
        with suppress(OSError):
            os.killpg(process.pid, 9)


def _read_remote_status(remote_host: str, path: Path) -> Mapping[str, Any] | None:
    completed = subprocess.run(
        ("ssh", remote_host, f"cat {shlex.quote(str(path))} 2>/dev/null"),
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    value = json.loads(completed.stdout)
    return value if isinstance(value, Mapping) else None


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, Mapping) else None


def _run_checked(command: Sequence[str], *, cwd: Path | None = None) -> None:
    subprocess.run(
        tuple(command),
        cwd=cwd,
        check=True,
        timeout=600.0,
    )


def _write_model_json(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _write_status(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        **payload,
        "updated_at_unix": time.time(),
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-config", type=Path, required=True)
    parser.add_argument("--score-config", type=Path)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    host_group = parser.add_mutually_exclusive_group()
    host_group.add_argument("--local-only", action="store_true")
    host_group.add_argument("--remote-only", action="store_true")
    parser.add_argument("--remote-host", default=_DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-repo", type=Path, default=_DEFAULT_REMOTE_REPO)
    parser.add_argument("--remote-python", default=_DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--local-workers", type=int, default=_DEFAULT_WORKERS)
    parser.add_argument("--local-candidate-shards", type=int, default=1)
    parser.add_argument("--remote-workers", type=int, default=_DEFAULT_WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    main()
