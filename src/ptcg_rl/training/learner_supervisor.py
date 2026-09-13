"""Single-attempt process ownership for formal learner execution."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import IO

from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupervisedCommand:
    """One immutable learner subprocess invocation."""

    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]


def supervise_learner(
    initial_command: SupervisedCommand,
    *,
    output_dir: Path,
) -> int:
    """Own exactly one learner attempt and surface every unexpected exit.

    The wrapper exists only to publish lifecycle state and to retire the whole
    child process group when its tmux owner receives an operator signal. A
    non-zero learner exit is a terminal stop condition: checkpoint selection,
    diagnosis, and recovery require an explicit operator launch.
    """
    resolved_output = output_dir.resolve()
    status_path = resolved_output / "control" / "learner_supervisor.json"
    stderr_path = resolved_output / "control" / "learner_stderr.log"
    requested_signal: int | None = None
    child: subprocess.Popen[bytes] | None = None
    stderr_log: IO[bytes] | None = None

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        nonlocal requested_signal
        if requested_signal is None:
            requested_signal = signum
            if child is not None and child.poll() is None:
                _signal_process_group(child.pid, signum)

    supervised_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in supervised_signals
    }
    try:
        started_at = _utc_now()
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_log = stderr_path.open("ab", buffering=0)
        stderr_log.write(
            f"\n[{started_at}] learner attempt started\n".encode()
        )
        child = subprocess.Popen(
            initial_command.argv,
            cwd=initial_command.cwd,
            env=dict(initial_command.env),
            start_new_session=True,
            stderr=stderr_log,
        )
        _write_status_best_effort(
            status_path,
            state="running",
            child_pid=child.pid,
            started_at=started_at,
            stderr_path=stderr_path,
        )
        while child.poll() is None and requested_signal is None:
            time.sleep(0.2)
        if requested_signal is not None:
            _stop_child(child, requested_signal=requested_signal)
            _write_status_best_effort(
                status_path,
                state="stopped",
                child_pid=None,
                started_at=started_at,
                learner_exitcode=child.returncode,
                stderr_path=stderr_path,
            )
            return 128 + requested_signal

        return_code = int(child.wait())
        child = None
        state = "completed" if return_code == 0 else "failed"
        if return_code != 0:
            _LOGGER.error(
                "learner exited with code %d; automatic restart is disabled",
                return_code,
            )
        _write_status_best_effort(
            status_path,
            state=state,
            child_pid=None,
            started_at=started_at,
            learner_exitcode=return_code,
            stderr_path=stderr_path,
        )
        return return_code
    finally:
        if child is not None and child.poll() is None:
            _stop_child(child, requested_signal=signal.SIGTERM)
        if stderr_log is not None:
            stderr_log.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

def _write_status_best_effort(
    path: Path,
    *,
    state: str,
    child_pid: int | None,
    started_at: str,
    stderr_path: Path,
    learner_exitcode: int | None = None,
) -> None:
    try:
        atomic_write_bytes(
            path,
            json_payload(
                {
                    "format": "single-attempt-learner-owner-v2",
                    "state": state,
                    "recorded_at_utc": _utc_now(),
                    "learner_started_at_utc": started_at,
                    "child_pid": child_pid,
                    "automatic_restart": False,
                    "restart_count": 0,
                    "learner_exitcode": learner_exitcode,
                    "learner_stderr_path": str(stderr_path),
                    "backoff_seconds": None,
                    "resume_checkpoint": None,
                }
            ),
            overwrite=True,
        )
    except Exception:
        _LOGGER.exception("learner supervisor status publication failed")


def _stop_child(
    child: subprocess.Popen[bytes],
    *,
    requested_signal: int,
    timeout_seconds: float = 20.0,
) -> None:
    if child.poll() is not None:
        return
    _signal_process_group(child.pid, requested_signal)
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


def _signal_process_group(group_id: int, signum: int) -> None:
    try:
        os.killpg(group_id, signum)
    except ProcessLookupError:
        return


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "SupervisedCommand",
    "supervise_learner",
]
