"""Own one native worker process group and leave no feeder descendants."""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path


def main() -> None:
    """Run a worker command and propagate bounded shutdown to its whole group."""
    arguments = _arguments()
    arguments.log_path.parent.mkdir(parents=True, exist_ok=True)
    logged_command = (
        "bash",
        "-o",
        "pipefail",
        "-c",
        f"{shlex.join(arguments.command)} 2>&1 | "
        f"tee -a {shlex.quote(str(arguments.log_path))}",
    )
    child = subprocess.Popen(logged_command, start_new_session=True)
    requested_signal: int | None = None

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal requested_signal
        if requested_signal is None:
            requested_signal = signum
            _signal_process_group(child.pid, signum)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    while child.poll() is None and requested_signal is None:
        time.sleep(0.1)
    if requested_signal is not None:
        _wait_or_escalate(child, timeout_seconds=arguments.stop_timeout_seconds)
        raise SystemExit(128 + requested_signal)
    return_code = int(child.wait())
    _retire_surviving_group(
        child.pid,
        timeout_seconds=arguments.stop_timeout_seconds,
    )
    raise SystemExit(return_code)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--stop-timeout-seconds", type=float, default=20.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.stop_timeout_seconds <= 0.0:
        parser.error("--stop-timeout-seconds must be positive")
    if arguments.command and arguments.command[0] == "--":
        arguments.command = arguments.command[1:]
    if not arguments.command:
        parser.error("a worker command is required after --")
    return arguments


def _wait_or_escalate(
    child: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if child.poll() is None:
        _signal_process_group(child.pid, signal.SIGTERM)
        deadline = time.monotonic() + min(timeout_seconds, 5.0)
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
    if child.poll() is None:
        _signal_process_group(child.pid, signal.SIGKILL)
        child.wait(timeout=5.0)
    _retire_surviving_group(child.pid, timeout_seconds=min(timeout_seconds, 5.0))


def _retire_surviving_group(group_id: int, *, timeout_seconds: float) -> None:
    if not _process_group_exists(group_id):
        return
    _signal_process_group(group_id, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while _process_group_exists(group_id) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _process_group_exists(group_id):
        _signal_process_group(group_id, signal.SIGKILL)


def _process_group_exists(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_group(group_id: int, signum: int) -> None:
    try:
        os.killpg(group_id, signum)
    except ProcessLookupError:
        return


if __name__ == "__main__":
    main()
