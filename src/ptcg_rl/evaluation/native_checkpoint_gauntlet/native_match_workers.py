"""Spawn-safe native-match replicas with parent-owned durable artifacts."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

from ptcg_rl.evaluation.continuous_league.models import NativeMatchConfig
from ptcg_rl.evaluation.native_checkpoint_gauntlet.models import (
    NativeCheckpointGauntletConfig,
    ScheduledCrossCheckpointGame,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.native_match_replica import (
    NativeMatchReplica,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.replica_partition import (
    partition_replica_items,
)
from ptcg_rl.evaluation.native_deck_elo.storage import (
    next_part_index,
    write_part,
    write_progress,
)


@dataclass(frozen=True, slots=True)
class _WorkerRow:
    worker_index: int
    preflight: bool
    row: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _WorkerDone:
    worker_index: int
    emitted_rows: int


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    worker_index: int
    detail: str


_WorkerMessage = _WorkerRow | _WorkerDone | _WorkerFailure


def run_native_match_worker_group(
    config: NativeCheckpointGauntletConfig,
    *,
    root: Path,
    candidate_native: NativeMatchConfig,
    baseline_native: NativeMatchConfig,
    checkpoint_paths: Mapping[str, Path],
    games: Sequence[ScheduledCrossCheckpointGame],
    rows: list[dict[str, Any]],
    parts_dir: Path,
    progress_path: Path,
    campaign_fingerprint: str,
    runtime_fingerprint: str,
    belief_fingerprint: str,
    started_clock: float,
    resumed_games: int,
) -> None:
    """Run persistent CUDA replicas while the parent alone commits results."""
    worker_count = min(config.native_worker_replicas, len(games))
    partitions = partition_replica_items(games, workers=worker_count)
    context = mp.get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue(
        maxsize=max(config.flush_shard_games, worker_count * 2)
    )
    require_preflight = resumed_games == 0
    preflight_counts = tuple(
        _preflight_game_count(config, partition_size=len(partition))
        if require_preflight
        else 0
        for partition in partitions
    )
    if not require_preflight:
        start_event.set()
    processes = tuple(
        context.Process(
            target=_native_match_replica_main,
            args=(
                config,
                worker_index,
                root,
                candidate_native,
                baseline_native,
                dict(checkpoint_paths),
                partition,
                campaign_fingerprint,
                runtime_fingerprint,
                belief_fingerprint,
                worker_count,
                preflight_counts[worker_index],
                start_event,
                result_queue,
            ),
            name=f"native-match-gauntlet-worker-{worker_index}",
        )
        for worker_index, partition in enumerate(partitions)
    )
    started_processes: list[Any] = []
    pending_rows: list[dict[str, Any]] = []
    seen_indices = {int(row["game_index"]) for row in rows}
    initial_seen_count = len(seen_indices)
    next_part = next_part_index(parts_dir)
    successful = False
    try:
        for process in processes:
            process.start()
            started_processes.append(process)
        if require_preflight:
            preflight_rows_by_worker = [0] * worker_count
            expected_preflight_rows = sum(preflight_counts)
            received_preflight_rows = 0
            while received_preflight_rows < expected_preflight_rows:
                message = _next_worker_message(result_queue, started_processes)
                if isinstance(message, _WorkerFailure):
                    raise RuntimeError(
                        f"native-match worker {message.worker_index} failed:\n"
                        f"{message.detail}"
                    )
                if not isinstance(message, _WorkerRow) or not message.preflight:
                    raise RuntimeError(
                        "native-match worker violated the preflight barrier"
                    )
                if not 0 <= message.worker_index < worker_count:
                    raise RuntimeError("native-match preflight has an unknown worker")
                if (
                    preflight_rows_by_worker[message.worker_index]
                    >= preflight_counts[message.worker_index]
                ):
                    raise RuntimeError(
                        "native-match worker exceeded its preflight wave"
                    )
                if message.row["terminal_reason"] != "normal":
                    raise RuntimeError(
                        "native-match replica preflight did not finish normally: "
                        f"worker={message.worker_index}, "
                        f"reason={message.row['terminal_reason']}"
                    )
                preflight_rows_by_worker[message.worker_index] += 1
                received_preflight_rows += 1
                _accept_row(
                    message.row,
                    seen_indices=seen_indices,
                    pending_rows=pending_rows,
                )
                next_part = _commit_ready_rows(
                    pending_rows,
                    force=False,
                    next_part=next_part,
                    rows=rows,
                    parts_dir=parts_dir,
                    progress_path=progress_path,
                    config=config,
                    started_clock=started_clock,
                    resumed_games=resumed_games,
                )
            start_event.set()

        done_workers: dict[int, int] = {}
        while len(done_workers) < worker_count:
            message = _next_worker_message(result_queue, started_processes)
            if isinstance(message, _WorkerFailure):
                raise RuntimeError(
                    f"native-match worker {message.worker_index} failed:\n"
                    f"{message.detail}"
                )
            if isinstance(message, _WorkerDone):
                if message.worker_index in done_workers:
                    raise RuntimeError(
                        "native-match worker emitted duplicate completion"
                    )
                done_workers[message.worker_index] = message.emitted_rows
                continue
            if message.preflight:
                raise RuntimeError("native-match worker repeated a preflight result")
            _accept_row(
                message.row,
                seen_indices=seen_indices,
                pending_rows=pending_rows,
            )
            next_part = _commit_ready_rows(
                pending_rows,
                force=False,
                next_part=next_part,
                rows=rows,
                parts_dir=parts_dir,
                progress_path=progress_path,
                config=config,
                started_clock=started_clock,
                resumed_games=resumed_games,
            )

        for process in started_processes:
            process.join(timeout=30.0)
        alive = tuple(
            process.name for process in started_processes if process.is_alive()
        )
        if alive:
            raise TimeoutError(
                f"native-match workers did not exit after completion: {alive}"
            )
        _raise_failed_process(started_processes)
        expected_rows = len(games)
        emitted_rows = sum(done_workers.values())
        if (
            emitted_rows != expected_rows
            or len(seen_indices) - initial_seen_count != expected_rows
        ):
            raise RuntimeError(
                "native-match replicas emitted an incomplete result set: "
                "workers="
                f"{emitted_rows}, accepted={len(seen_indices) - initial_seen_count}, "
                f"expected={expected_rows}"
            )
        _commit_ready_rows(
            pending_rows,
            force=True,
            next_part=next_part,
            rows=rows,
            parts_dir=parts_dir,
            progress_path=progress_path,
            config=config,
            started_clock=started_clock,
            resumed_games=resumed_games,
        )
        successful = True
    finally:
        start_event.set()
        if not successful:
            for process in started_processes:
                if process.is_alive():
                    process.terminate()
        for process in started_processes:
            process.join(timeout=10.0)
        for process in started_processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=10.0)
        result_queue.close()
        result_queue.join_thread()


def _next_worker_message(
    result_queue: Any,
    processes: Sequence[Any],
) -> _WorkerMessage:
    """Receive one message while promptly surfacing replica exits."""
    while True:
        try:
            message = result_queue.get(timeout=0.5)
        except Empty:
            _raise_failed_process(processes)
            if all(not process.is_alive() for process in processes):
                raise RuntimeError(
                    "native-match workers exited without completion"
                ) from None
            continue
        if not isinstance(message, (_WorkerRow, _WorkerDone, _WorkerFailure)):
            raise TypeError("native-match worker emitted an invalid message")
        return message


def _raise_failed_process(processes: Sequence[Any]) -> None:
    failed = tuple(
        process for process in processes if process.exitcode not in (None, 0)
    )
    if failed:
        details = ", ".join(f"{process.name}={process.exitcode}" for process in failed)
        raise RuntimeError(f"native-match worker group failed: {details}")


def _accept_row(
    row: dict[str, Any],
    *,
    seen_indices: set[int],
    pending_rows: list[dict[str, Any]],
) -> None:
    game_index = int(row["game_index"])
    if game_index in seen_indices:
        raise ValueError(f"native-match replicas repeated game {game_index}")
    seen_indices.add(game_index)
    pending_rows.append(row)


def _commit_ready_rows(
    pending_rows: list[dict[str, Any]],
    *,
    force: bool,
    next_part: int,
    rows: list[dict[str, Any]],
    parts_dir: Path,
    progress_path: Path,
    config: NativeCheckpointGauntletConfig,
    started_clock: float,
    resumed_games: int,
) -> int:
    """Atomically publish parent-owned chunks and advance durable progress."""
    while len(pending_rows) >= config.flush_shard_games or (force and pending_rows):
        count = min(len(pending_rows), config.flush_shard_games)
        committed = pending_rows[:count]
        del pending_rows[:count]
        write_part(
            parts_dir / f"part-{next_part:06d}.parquet",
            committed,
            compression=config.compression,
        )
        rows.extend(committed)
        next_part += 1
        write_progress(
            progress_path,
            total_games=config.total_games,
            completed_games=len(rows),
            resumed_games=resumed_games,
            started_clock=started_clock,
            complete=False,
        )
    return next_part


def _native_match_replica_main(
    config: NativeCheckpointGauntletConfig,
    worker_index: int,
    root: Path,
    candidate_native: NativeMatchConfig,
    baseline_native: NativeMatchConfig,
    checkpoint_paths: Mapping[str, Path],
    games: Sequence[ScheduledCrossCheckpointGame],
    campaign_fingerprint: str,
    runtime_fingerprint: str,
    belief_fingerprint: str,
    replica_process_count: int,
    preflight_game_count: int,
    start_event: Any,
    result_queue: Any,
) -> None:
    """Own one persistent model/batcher/lane group in an isolated process."""
    runtime: NativeMatchReplica | None = None
    emitted_rows = 0
    try:
        intraop_threads = _configure_replica_intraop_threads(replica_process_count)
        print(
            f"native-match worker {worker_index}: intraop_threads={intraop_threads}",
            flush=True,
        )
        runtime = NativeMatchReplica(
            config,
            root=root,
            candidate_native=candidate_native,
            baseline_native=baseline_native,
            checkpoint_paths=checkpoint_paths,
            campaign_fingerprint=campaign_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            belief_fingerprint=belief_fingerprint,
            game_capacity=len(games),
        )
        remaining = tuple(games)
        if preflight_game_count:
            for row in runtime.execute_many(remaining[:preflight_game_count]):
                result_queue.put(
                    _WorkerRow(worker_index=worker_index, preflight=True, row=row)
                )
                emitted_rows += 1
            remaining = remaining[preflight_game_count:]
            start_event.wait()
        for row in runtime.execute_many(remaining):
            result_queue.put(
                _WorkerRow(worker_index=worker_index, preflight=False, row=row)
            )
            emitted_rows += 1
        runtime.close()
        runtime = None
        result_queue.put(
            _WorkerDone(worker_index=worker_index, emitted_rows=emitted_rows)
        )
    except BaseException:
        detail = traceback.format_exc()
        with suppress(BaseException):
            result_queue.put(
                _WorkerFailure(worker_index=worker_index, detail=detail),
                timeout=1.0,
            )
        traceback.print_exc()
        raise
    finally:
        if runtime is not None:
            runtime.close()


def _preflight_game_count(
    config: NativeCheckpointGauntletConfig,
    *,
    partition_size: int,
) -> int:
    """Fill one policy batch while bounding uncommitted preflight evidence."""
    return min(config.concurrency, config.policy_batch_max_rows, partition_size)


def _configure_replica_intraop_threads(replica_process_count: int) -> int:
    """Prevent persistent replicas from multiplying one full CPU thread pool."""
    import os

    import torch

    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    target = _replica_intraop_thread_count(
        current_threads=torch.get_num_threads(),
        available_cpus=available_cpus,
        replica_process_count=replica_process_count,
    )
    torch.set_num_threads(target)
    return target


def _replica_intraop_thread_count(
    *,
    current_threads: int,
    available_cpus: int,
    replica_process_count: int,
) -> int:
    """Preserve tighter operator limits while sharing host CPUs evenly."""
    if current_threads <= 0 or available_cpus <= 0 or replica_process_count <= 0:
        raise ValueError("replica CPU thread inputs must be positive")
    return max(1, min(current_threads, available_cpus // replica_process_count))


__all__ = ["run_native_match_worker_group"]
