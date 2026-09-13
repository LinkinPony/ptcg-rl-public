"""Remote shard lifecycle for distributed release evaluation."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from ptcg_rl.evaluation.distributed_hosts import HostAllocation
from ptcg_rl.evaluation.distributed_release_h2h import (
    ReleaseH2HShardConfig,
    ReleaseH2HShardResult,
    ShardAssignment,
    build_shard_config,
)
from ptcg_rl.evaluation.release_h2h import ReleaseH2HConfig
from ptcg_rl.evaluation.search_identity import write_identity_atomic


@dataclass(frozen=True)
class PreparedRemoteShard:
    """All local and remote paths required to launch one shard."""

    allocation: HostAllocation
    assignment: ShardAssignment
    config: ReleaseH2HShardConfig
    snapshot_root: Path
    remote_run_root: Path
    remote_config_path: Path
    remote_status_path: Path
    remote_result_path: Path
    remote_pid_path: Path
    local_shard_root: Path
    local_config_path: Path
    local_log_path: Path


@dataclass
class RunningRemoteShard:
    """One SSH process and its owned log handle."""

    prepared: PreparedRemoteShard
    process: subprocess.Popen[str]
    log_handle: TextIO


def prepare_remote_shard(
    release: ReleaseH2HConfig,
    *,
    campaign_fingerprint: str,
    snapshot_fingerprint: str,
    snapshot_root: Path,
    remote_root: Path,
    local_root: Path,
    allocation: HostAllocation,
    assignment: ShardAssignment,
    connect_timeout_seconds: int,
) -> PreparedRemoteShard:
    """Create and upload a content-bound worker envelope."""
    remote_run_root = (
        allocation.home
        / remote_root
        / "runs"
        / campaign_fingerprint
        / assignment.shard_id
    )
    remote_config_path = remote_run_root / "shard_config.json"
    remote_output_dir = remote_run_root / "output"
    remote_status_path = remote_output_dir / "status.json"
    remote_result_path = remote_run_root / "result.json"
    remote_pid_path = remote_run_root / "worker.pid"
    remote_temp_root = remote_run_root / "tmp"
    config = build_shard_config(
        release,
        campaign_fingerprint=campaign_fingerprint,
        snapshot_fingerprint=snapshot_fingerprint,
        allocation=allocation,
        assignment=assignment,
        remote_output_dir=remote_output_dir,
        remote_temp_root=remote_temp_root,
    )
    local_shard_root = local_root / "shards" / assignment.shard_id
    local_config_path = local_shard_root / "shard_config.json"
    local_log_path = local_shard_root / "worker.log"
    local_shard_root.mkdir(parents=True, exist_ok=True)
    write_identity_atomic(local_config_path, config.model_dump(mode="json"))
    _run_checked(
        _ssh_command(
            allocation.target,
            f"mkdir -p {shlex.quote(str(remote_run_root))}",
            timeout=connect_timeout_seconds,
        )
    )
    _run_checked(
        (
            "rsync",
            "-az",
            str(local_config_path),
            f"{allocation.target}:{remote_config_path}",
        )
    )
    return PreparedRemoteShard(
        allocation=allocation,
        assignment=assignment,
        config=config,
        snapshot_root=snapshot_root,
        remote_run_root=remote_run_root,
        remote_config_path=remote_config_path,
        remote_status_path=remote_status_path,
        remote_result_path=remote_result_path,
        remote_pid_path=remote_pid_path,
        local_shard_root=local_shard_root,
        local_config_path=local_config_path,
        local_log_path=local_log_path,
    )


def start_remote_shard(
    prepared: PreparedRemoteShard,
    *,
    connect_timeout_seconds: int,
) -> RunningRemoteShard:
    """Start one SSH-attached worker in its own remote process session."""
    allocation = prepared.allocation
    assignment = prepared.assignment
    affinity = ",".join(str(cpu) for cpu in assignment.cpu_affinity)
    if not affinity:
        raise ValueError(f"remote shard has no CPU affinity: {assignment.shard_id}")
    host_gpu_index = (
        "" if assignment.gpu_index is None else str(assignment.gpu_index)
    )
    cuda_visible = (
        "0"
        if allocation.runtime_backend == "docker"
        and assignment.gpu_index is not None
        else host_gpu_index
    )
    threads = str(allocation.cpu_threads_per_worker)
    environment = [
        "PYTHONPATH=data/sample_submission:src",
        f"CUDA_VISIBLE_DEVICES={cuda_visible}",
        f"OMP_NUM_THREADS={threads}",
        f"MKL_NUM_THREADS={threads}",
        f"OPENBLAS_NUM_THREADS={threads}",
        f"NUMEXPR_NUM_THREADS={threads}",
    ]
    if allocation.runtime_backend == "docker":
        if allocation.container_image is None:
            raise ValueError("docker allocation omitted its immutable image ID")
        environment.extend(
            (
                f"PTCG_EVAL_IMAGE={allocation.container_image}",
                f"PTCG_EVAL_REMOTE_ROOT={prepared.snapshot_root.parents[1]}",
                f"PTCG_EVAL_CPUSET={affinity}",
                f"PTCG_EVAL_GPU_INDEX={host_gpu_index}",
            )
        )
    worker_command = (
        "taskset",
        "-c",
        affinity,
        "env",
        *environment,
        allocation.python_executable,
        "-u",
        "src/tools/eval_release_h2h_shard.py",
        "--shard-config",
        str(prepared.remote_config_path),
        "--status-path",
        str(prepared.remote_status_path),
        "--result-path",
        str(prepared.remote_result_path),
    )
    quoted_worker = " ".join(shlex.quote(part) for part in worker_command)
    remote_command = (
        f"cd {shlex.quote(str(prepared.snapshot_root))} || exit 1; "
        f"echo $$ > {shlex.quote(str(prepared.remote_pid_path))}; "
        f"exec setsid --wait {quoted_worker}"
    )
    log_handle = prepared.local_log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        _ssh_command(
            allocation.target,
            remote_command,
            timeout=connect_timeout_seconds,
        ),
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return RunningRemoteShard(
        prepared=prepared,
        process=process,
        log_handle=log_handle,
    )


def wait_remote_shards(
    running: Sequence[RunningRemoteShard],
    *,
    coordinator_status_path: Path,
    campaign_fingerprint: str,
    poll_interval_seconds: float,
    connect_timeout_seconds: int,
) -> None:
    """Mirror durable shard progress until every SSH worker exits cleanly."""
    while True:
        statuses: dict[str, Mapping[str, Any] | None] = {}
        exit_codes: dict[str, int | None] = {}
        for handle in running:
            shard_id = handle.prepared.assignment.shard_id
            exit_codes[shard_id] = handle.process.poll()
            statuses[shard_id] = _read_remote_json(
                handle.prepared.allocation.target,
                handle.prepared.remote_status_path,
                timeout=connect_timeout_seconds,
            )
        _write_coordinator_status(
            coordinator_status_path,
            campaign_fingerprint=campaign_fingerprint,
            statuses=statuses,
            exit_codes=exit_codes,
        )
        failures = {
            shard_id: code
            for shard_id, code in exit_codes.items()
            if code not in {None, 0}
        }
        if failures:
            raise RuntimeError(f"distributed release shards failed: {failures}")
        if all(code == 0 for code in exit_codes.values()):
            incomplete = {
                shard_id: (
                    None if status is None else str(status.get("status", "missing"))
                )
                for shard_id, status in statuses.items()
                if status is None or status.get("status") != "completed"
            }
            if incomplete:
                raise RuntimeError(
                    "distributed release shards exited without completed durable "
                    f"status: {incomplete}"
                )
            return
        time.sleep(poll_interval_seconds)


def download_remote_result(
    prepared: PreparedRemoteShard,
) -> tuple[ReleaseH2HShardResult, Path]:
    """Fetch one completed run tree and validate its result manifest."""
    prepared.local_shard_root.mkdir(parents=True, exist_ok=True)
    _run_checked(
        (
            "rsync",
            "-az",
            f"{prepared.allocation.target}:{prepared.remote_run_root}/",
            f"{prepared.local_shard_root}/",
        )
    )
    local_result_path = prepared.local_shard_root / "result.json"
    result = ReleaseH2HShardResult.model_validate_json(
        local_result_path.read_text(encoding="utf-8")
    )
    if result.shard_id != prepared.assignment.shard_id:
        raise ValueError("downloaded result manifest belongs to another shard")
    games_path = prepared.local_shard_root / "output" / "games.parquet"
    if not games_path.is_file():
        raise FileNotFoundError(f"remote shard produced no games file: {games_path}")
    return result, games_path


def close_remote_shards(
    running: Sequence[RunningRemoteShard],
    *,
    terminate: bool,
    connect_timeout_seconds: int,
) -> None:
    """Close logs and, on failure, terminate only recorded worker trees."""
    for handle in running:
        if terminate and handle.process.poll() is None:
            _terminate_local_process(handle.process)
            _terminate_remote_worker(
                handle.prepared,
                connect_timeout_seconds=connect_timeout_seconds,
            )
        handle.log_handle.close()


def _write_coordinator_status(
    path: Path,
    *,
    campaign_fingerprint: str,
    statuses: Mapping[str, Mapping[str, Any] | None],
    exit_codes: Mapping[str, int | None],
) -> None:
    totals = {"games_total": 0, "games_committed": 0, "games_finished": 0}
    complete_progress = True
    for status in statuses.values():
        if status is None or "games_total" not in status:
            complete_progress = False
            continue
        totals["games_total"] += int(status["games_total"])
        totals["games_committed"] += int(status.get("games_committed", 0))
        totals["games_finished"] += int(status.get("games_finished", 0))
    payload: dict[str, Any] = {
        "status": "running",
        "protocol": "DISTRIBUTED-RELEASE-H2H-v1",
        "campaign_fingerprint": campaign_fingerprint,
        "shards": statuses,
        "exit_codes": exit_codes,
    }
    if complete_progress and totals["games_total"] > 0:
        payload.update(totals)
        payload["games_remaining"] = (
            totals["games_total"] - totals["games_committed"]
        )
        payload["progress_percent"] = (
            100.0 * totals["games_committed"] / totals["games_total"]
        )
    write_identity_atomic(path, payload)


def _read_remote_json(
    target: str,
    path: Path,
    *,
    timeout: int,
) -> Mapping[str, Any] | None:
    completed = subprocess.run(
        _ssh_command(
            target,
            f"cat {shlex.quote(str(path))} 2>/dev/null",
            timeout=timeout,
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=max(30, timeout * 3),
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    value = json.loads(completed.stdout)
    return value if isinstance(value, Mapping) else None


def _terminate_remote_worker(
    prepared: PreparedRemoteShard,
    *,
    connect_timeout_seconds: int,
) -> None:
    script = _remote_terminate_script(prepared.remote_pid_path)
    subprocess.run(
        _ssh_command(
            prepared.allocation.target,
            f"python3 -c {shlex.quote(script)}",
            timeout=connect_timeout_seconds,
        ),
        check=False,
        timeout=max(30, connect_timeout_seconds * 3),
    )


def _remote_terminate_script(pid_path: Path) -> str:
    return f'''import os
import pathlib
import signal
import time

path = pathlib.Path({str(pid_path)!r})
if not path.is_file():
    raise SystemExit(0)
raw = path.read_text().strip()
if not raw.isdigit():
    raise RuntimeError("invalid evaluation worker pid file")
root = int(raw)

def descendants(parent):
    result = []
    frontier = [parent]
    while frontier:
        current = frontier.pop()
        for proc in pathlib.Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                fields = (proc / "stat").read_text().split()
                ppid = int(fields[3])
            except (OSError, ValueError, IndexError):
                continue
            pid = int(proc.name)
            if ppid == current and pid not in result:
                result.append(pid)
                frontier.append(pid)
    return result

pids = descendants(root) + [root]
for sig in (signal.SIGINT, signal.SIGTERM):
    for pid in reversed(pids):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    time.sleep(2.0)
for pid in reversed(pids):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass'''


def _terminate_local_process(process: subprocess.Popen[str]) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


def _ssh_command(target: str, command: str, *, timeout: int) -> tuple[str, ...]:
    return (
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout}",
        target,
        command,
    )


def _run_checked(command: Sequence[str]) -> None:
    subprocess.run(tuple(command), check=True, timeout=1200.0)


__all__ = [
    "PreparedRemoteShard",
    "RunningRemoteShard",
    "close_remote_shards",
    "download_remote_result",
    "prepare_remote_shard",
    "start_remote_shard",
    "wait_remote_shards",
]
