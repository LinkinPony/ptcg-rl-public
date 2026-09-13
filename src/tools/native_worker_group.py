"""Launch a stable per-GPU native collection worker group in tmux."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_STABLE_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class WorkerLaunch:
    """One deterministic worker process and its assigned CPU set."""

    worker_id: str
    session_name: str
    cpu_list: tuple[int, ...]
    log_path: Path
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MpsLaunch:
    """One group-owned MPS control plane shared by independent workers."""

    session_name: str
    pipe_dir: Path
    log_dir: Path
    ready_path: Path
    command: tuple[str, ...]


def main() -> None:
    """Start, inspect, or gracefully stop one logical GPU worker group."""
    arguments = _arguments()
    repo_root = Path(__file__).resolve().parents[2]
    mps = _mps_launch(arguments, repo_root=repo_root)
    launches = _worker_launches(arguments, repo_root=repo_root, mps=mps)
    if arguments.dry_run:
        for launch in launches:
            print(_display_launch(launch))
        return
    if arguments.status:
        _print_status(launches, mps=mps)
        return
    if arguments.stop:
        _stop_workers(launches, timeout_seconds=arguments.stop_timeout_seconds)
        if mps is not None:
            _stop_mps(mps, timeout_seconds=arguments.stop_timeout_seconds)
        return
    _validate_allowed_cpus(launches)
    _start_workers(launches, repo_root=repo_root, mps=mps)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run several stable native collectors on one GPU so Python-side "
            "arena coordination cannot leave a large accelerator idle."
        )
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-version", required=True)
    parser.add_argument("--coordinator-host", required=True)
    parser.add_argument("--worker-profile", required=True)
    parser.add_argument("--worker-id-prefix", required=True)
    parser.add_argument("--session-prefix", required=True)
    parser.add_argument("--visible-device", required=True)
    parser.add_argument(
        "--source-commit",
        required=True,
        help="full immutable learner source commit shared by every worker",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="formal run output directory; defaults to outputs/training/rl/<run>",
    )
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override forwarded to every stable worker",
    )
    parser.add_argument(
        "--mps",
        action="store_true",
        help="share one group-owned CUDA MPS daemon across all replicas",
    )
    parser.add_argument(
        "--cpu-list",
        required=True,
        help="taskset syntax, for example 104-119 or 104-111,120-127",
    )
    parser.add_argument(
        "--shared-cpu-pool",
        action="store_true",
        help=(
            "give every replica the full CPU list instead of static partitions; "
            "use for blocking engine/fact pipelines that benefit from scheduler "
            "work conservation"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--stop-timeout-seconds", type=float, default=30.0)
    arguments = parser.parse_args()
    selected_operations = sum(
        bool(value) for value in (arguments.dry_run, arguments.status, arguments.stop)
    )
    if selected_operations > 1:
        parser.error("choose at most one of --dry-run, --status, or --stop")
    if arguments.replicas <= 0:
        parser.error("--replicas must be positive")
    if arguments.stop_timeout_seconds <= 0.0:
        parser.error("--stop-timeout-seconds must be positive")
    for name, value in (
        ("run version", arguments.run_version),
        ("worker ID prefix", arguments.worker_id_prefix),
        ("session prefix", arguments.session_prefix),
    ):
        if _STABLE_NAME_PATTERN.fullmatch(value) is None:
            parser.error(f"{name} contains unsupported characters")
    return arguments


def _worker_launches(
    arguments: argparse.Namespace,
    *,
    repo_root: Path,
    mps: MpsLaunch | None,
) -> tuple[WorkerLaunch, ...]:
    parsed_cpus = _parse_cpu_list(arguments.cpu_list)
    cpu_groups = (
        tuple(parsed_cpus for _ in range(arguments.replicas))
        if arguments.shared_cpu_pool
        else _partition_cpus(parsed_cpus, replicas=arguments.replicas)
    )
    launches: list[WorkerLaunch] = []
    for replica, cpu_group in enumerate(cpu_groups):
        worker_id = f"{arguments.worker_id_prefix}_{replica}"
        session_name = f"{arguments.session_prefix}_{replica}"
        hydra_dir = repo_root / "tmp" / "hydra_native_workers" / worker_id
        log_path = (
            repo_root
            / "tmp"
            / "native_worker_logs"
            / arguments.run_version
            / f"{worker_id}.log"
        )
        # The MPS control daemon is pinned with the host-visible physical GPU.
        # A client attached to that single-GPU daemon sees it as logical CUDA
        # device zero, even when the daemon was launched on physical GPU 1.
        client_visible_device = "0" if mps is not None else arguments.visible_device
        environment = [
            "env",
            f"CUDA_VISIBLE_DEVICES={client_visible_device}",
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
        ]
        if mps is not None:
            environment.extend(
                (
                    f"CUDA_MPS_PIPE_DIRECTORY={mps.pipe_dir}",
                    f"CUDA_MPS_LOG_DIRECTORY={mps.log_dir}",
                )
            )
        command = (
            *environment,
            "taskset",
            "-c",
            _format_cpu_list(cpu_group),
            sys.executable,
            str(repo_root / "train.py"),
            "--role",
            "collection-worker",
            "--profile",
            arguments.profile,
            "--run-version",
            arguments.run_version,
            "--output-dir",
            str(
                arguments.output_dir
                or repo_root / "outputs" / "training" / "rl" / arguments.run_version
            ),
            "--worker-id",
            worker_id,
            "--coordinator-host",
            arguments.coordinator_host,
            "--worker-profile",
            arguments.worker_profile,
            "--immutable-source",
            "--source-commit",
            arguments.source_commit,
            f"hydra.run.dir={hydra_dir}",
            *arguments.override,
        )
        launches.append(
            WorkerLaunch(
                worker_id=worker_id,
                session_name=session_name,
                cpu_list=cpu_group,
                log_path=log_path,
                command=command,
            )
        )
    return tuple(launches)


def _start_workers(
    launches: tuple[WorkerLaunch, ...],
    *,
    repo_root: Path,
    mps: MpsLaunch | None,
) -> None:
    existing = tuple(
        launch.session_name for launch in launches if _session_exists(launch.session_name)
    )
    if existing:
        raise RuntimeError(f"native worker tmux sessions already exist: {existing}")
    owned_ids = tuple(
        launch.worker_id for launch in launches if _worker_process_exists(launch.worker_id)
    )
    if owned_ids:
        raise RuntimeError(f"native worker IDs already have live processes: {owned_ids}")
    started: list[str] = []
    try:
        if mps is not None:
            _start_mps(mps, repo_root=repo_root)
            started.append(mps.session_name)
        for launch in launches:
            launch.log_path.parent.mkdir(parents=True, exist_ok=True)
            logged_command = (
                sys.executable,
                str(repo_root / "src" / "tools" / "native_worker_supervisor.py"),
                "--log-path",
                str(launch.log_path),
                "--stop-timeout-seconds",
                "20",
                "--",
                *launch.command,
            )
            subprocess.run(
                (
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    launch.session_name,
                    "-c",
                    str(repo_root),
                    shlex.join(logged_command),
                ),
                check=True,
            )
            started.append(launch.session_name)
            print(_display_launch(launch))
    except BaseException as error:
        if started:
            error.add_note(
                "started sessions were left running for inspection: "
                + ", ".join(started)
            )
        raise


def _print_status(
    launches: tuple[WorkerLaunch, ...],
    *,
    mps: MpsLaunch | None,
) -> None:
    if mps is not None:
        session = _session_exists(mps.session_name)
        ready = mps.ready_path.is_file()
        state = "running" if session and ready else "partial" if session or ready else "stopped"
        print(f"mps: {state} session={mps.session_name} pipe={mps.pipe_dir}")
    for launch in launches:
        session = _session_exists(launch.session_name)
        process = _worker_process_exists(launch.worker_id)
        state = "running" if session and process else "partial" if session or process else "stopped"
        print(
            f"{launch.worker_id}: {state} session={launch.session_name} "
            f"cpus={_format_cpu_list(launch.cpu_list)} log={launch.log_path}"
        )


def _stop_workers(
    launches: tuple[WorkerLaunch, ...],
    *,
    timeout_seconds: float,
) -> None:
    active = tuple(
        launch for launch in launches if _session_exists(launch.session_name)
    )
    for launch in active:
        subprocess.run(
            ("tmux", "send-keys", "-t", launch.session_name, "C-c"),
            check=True,
        )
    deadline = time.monotonic() + timeout_seconds
    remaining = active
    while remaining and time.monotonic() < deadline:
        time.sleep(0.25)
        remaining = tuple(
            launch
            for launch in remaining
            if _session_exists(launch.session_name)
        )
    if remaining:
        raise RuntimeError(
            "native workers did not stop gracefully; sessions were preserved: "
            + ", ".join(launch.session_name for launch in remaining)
        )
    for launch in active:
        print(f"stopped {launch.worker_id} session={launch.session_name}")


def _mps_launch(
    arguments: argparse.Namespace,
    *,
    repo_root: Path,
) -> MpsLaunch | None:
    if not arguments.mps:
        return None
    root = repo_root / "tmp" / "native_worker_mps" / arguments.session_prefix
    pipe_dir = root / "pipe"
    log_dir = root / "log"
    ready_path = root / "ready"
    return MpsLaunch(
        session_name=f"{arguments.session_prefix}_mps",
        pipe_dir=pipe_dir,
        log_dir=log_dir,
        ready_path=ready_path,
        command=(
            "env",
            f"CUDA_VISIBLE_DEVICES={arguments.visible_device}",
            str(repo_root / "src" / "tools" / "native_worker_group_mps.sh"),
            str(pipe_dir),
            str(log_dir),
            str(ready_path),
        ),
    )


def _start_mps(mps: MpsLaunch, *, repo_root: Path) -> None:
    if _session_exists(mps.session_name):
        if mps.ready_path.is_file():
            print(f"reusing mps session={mps.session_name}")
            return
        raise RuntimeError(
            "native worker MPS session exists without a readiness marker: "
            f"{mps.session_name}"
        )
    mps.ready_path.unlink(missing_ok=True)
    subprocess.run(
        (
            "tmux",
            "new-session",
            "-d",
            "-s",
            mps.session_name,
            "-c",
            str(repo_root),
            shlex.join(mps.command),
        ),
        check=True,
    )
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if mps.ready_path.is_file():
            return
        if not _session_exists(mps.session_name):
            raise RuntimeError("native worker MPS supervisor exited during startup")
        time.sleep(0.1)
    raise RuntimeError("native worker MPS supervisor did not become ready")


def _stop_mps(mps: MpsLaunch, *, timeout_seconds: float) -> None:
    if not _session_exists(mps.session_name):
        if mps.ready_path.exists():
            raise RuntimeError("native worker MPS readiness marker is stale")
        return
    subprocess.run(
        ("tmux", "send-keys", "-t", mps.session_name, "C-c"),
        check=True,
    )
    deadline = time.monotonic() + timeout_seconds
    while _session_exists(mps.session_name) and time.monotonic() < deadline:
        time.sleep(0.25)
    if _session_exists(mps.session_name):
        raise RuntimeError(
            f"native worker MPS supervisor did not stop: {mps.session_name}"
        )
    if mps.ready_path.exists():
        raise RuntimeError("native worker MPS readiness marker survived shutdown")
    print(f"stopped mps session={mps.session_name}")


def _validate_allowed_cpus(launches: tuple[WorkerLaunch, ...]) -> None:
    allowed = os.sched_getaffinity(0)
    requested = {cpu for launch in launches for cpu in launch.cpu_list}
    unavailable = sorted(requested.difference(allowed))
    if unavailable:
        raise RuntimeError(
            "native worker CPU list exceeds this process affinity: "
            + _format_cpu_list(tuple(unavailable))
        )


def _session_exists(session_name: str) -> bool:
    return (
        subprocess.run(
            ("tmux", "has-session", "-t", session_name),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _worker_process_exists(worker_id: str) -> bool:
    marker = f"--worker-id {worker_id} "
    result = subprocess.run(
        ("ps", "-eo", "args="),
        check=True,
        capture_output=True,
        text=True,
    )
    return any(marker in f"{line.strip()} " for line in result.stdout.splitlines())


def _parse_cpu_list(value: str) -> tuple[int, ...]:
    cpus: list[int] = []
    for item in value.split(","):
        normalized = item.strip()
        if not normalized:
            raise ValueError("CPU list contains an empty range")
        if "-" not in normalized:
            cpus.append(int(normalized))
            continue
        start_text, stop_text = normalized.split("-", maxsplit=1)
        start = int(start_text)
        stop = int(stop_text)
        if stop < start:
            raise ValueError("CPU list range is reversed")
        cpus.extend(range(start, stop + 1))
    if not cpus or min(cpus) < 0 or len(cpus) != len(set(cpus)):
        raise ValueError("CPU list must contain unique non-negative CPUs")
    return tuple(cpus)


def _partition_cpus(
    cpus: tuple[int, ...],
    *,
    replicas: int,
) -> tuple[tuple[int, ...], ...]:
    if replicas <= 0 or len(cpus) < replicas:
        raise ValueError("each native worker requires at least one CPU")
    base, remainder = divmod(len(cpus), replicas)
    groups: list[tuple[int, ...]] = []
    cursor = 0
    for replica in range(replicas):
        width = base + int(replica < remainder)
        groups.append(cpus[cursor : cursor + width])
        cursor += width
    return tuple(groups)


def _format_cpu_list(cpus: tuple[int, ...]) -> str:
    if not cpus:
        raise ValueError("cannot format an empty CPU list")
    ranges: list[str] = []
    start = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _display_launch(launch: WorkerLaunch) -> str:
    return (
        f"{launch.worker_id}: session={launch.session_name} "
        f"cpus={_format_cpu_list(launch.cpu_list)} log={launch.log_path} "
        f"command={shlex.join(launch.command)}"
    )


if __name__ == "__main__":
    main()
