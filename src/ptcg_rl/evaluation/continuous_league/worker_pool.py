"""Supervise concurrent league slots over one shared native CUDA process."""

from __future__ import annotations

import signal
import sys
import threading
import time
from pathlib import Path
from types import FrameType

from ptcg_rl.evaluation.continuous_league.models import ContinuousLeagueConfig
from ptcg_rl.evaluation.continuous_league.shared_executor import (
    SharedNativeMatchProcess,
)
from ptcg_rl.evaluation.continuous_league.worker import LeagueWorker

_RESTART_SECONDS = 5.0


def run_worker_pool(
    config: ContinuousLeagueConfig,
    *,
    worker_id_prefix: str,
    concurrency: int,
    repo_root: Path,
    coordinator_url: str | None,
    gpu_index: int | None,
    executor_processes: int,
    ignore_resource_load: bool,
    once: bool,
) -> None:
    """Keep independent lease slots over one model/cache-owning child process."""
    if concurrency <= 1:
        raise ValueError("worker pool concurrency must be greater than one")
    if executor_processes <= 0 or executor_processes > concurrency:
        raise ValueError("executor process count must be within worker concurrency")
    if concurrency % executor_processes:
        raise ValueError("worker concurrency must divide across executor processes")
    if concurrency // executor_processes <= 1:
        raise ValueError("each shared executor process must own multiple lanes")
    stopped = threading.Event()
    threads: dict[int, threading.Thread] = {}
    errors: dict[int, BaseException | None] = {}
    errors_lock = threading.Lock()
    restart_after: dict[int, float] = {}
    completed: set[int] = set()
    failures: dict[int, str] = {}
    lanes_per_process = concurrency // executor_processes
    executors = tuple(
        SharedNativeMatchProcess(
            command=config.native_match_command,
            repo_root=repo_root,
            concurrency=lanes_per_process,
            gpu_index=gpu_index,
        )
        for _ in range(executor_processes)
    )

    def stop(_signum: int, _frame: FrameType | None) -> None:
        stopped.set()

    def run_slot(slot: int) -> None:
        error: BaseException | None = None
        try:
            LeagueWorker(
                config,
                worker_id=f"{worker_id_prefix}-{slot:02d}",
                repo_root=repo_root,
                coordinator_url=coordinator_url,
                gpu_index=gpu_index,
                ignore_resource_load=ignore_resource_load,
                executor=executors[slot % executor_processes].slot(),
            ).run(once=once, stop_event=stopped)
        except BaseException as caught:
            error = caught
        finally:
            with errors_lock:
                errors[slot] = error

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    hangup = getattr(signal, "SIGHUP", None)
    if hangup is not None:
        signal.signal(hangup, stop)
    try:
        while not stopped.is_set():
            now = time.monotonic()
            for slot in range(concurrency):
                thread = threads.get(slot)
                if thread is not None and thread.is_alive():
                    continue
                if thread is not None:
                    thread.join(timeout=0.0)
                    threads.pop(slot)
                    with errors_lock:
                        error = errors.pop(slot, None)
                    if once:
                        completed.add(slot)
                        if error is not None:
                            failures[slot] = f"{type(error).__name__}: {error}"
                    else:
                        print(
                            f"worker slot {slot} exited"
                            f"{'' if error is None else f': {type(error).__name__}: {error}'}; "
                            f"restarting in {_RESTART_SECONDS:.0f}s",
                            file=sys.stderr,
                            flush=True,
                        )
                        restart_after[slot] = now + _RESTART_SECONDS
                if once and slot in completed:
                    continue
                if now < restart_after.get(slot, 0.0):
                    continue
                thread = threading.Thread(
                    target=run_slot,
                    args=(slot,),
                    name=f"continuous-league-{worker_id_prefix}-{slot:02d}",
                    daemon=True,
                )
                thread.start()
                threads[slot] = thread
            if once and len(completed) == concurrency:
                if failures:
                    raise RuntimeError(
                        f"worker pool slots failed: {sorted(failures.items())}"
                    )
                return
            stopped.wait(1.0)
    finally:
        stopped.set()
        _join_threads(tuple(threads.values()), timeout=10.0)
        for executor in executors:
            executor.close()
        _join_threads(tuple(threads.values()), timeout=5.0)


def _join_threads(threads: tuple[threading.Thread, ...], *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))


__all__ = ["run_worker_pool"]
