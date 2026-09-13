"""Detached worker that owns one schema-v2 task terminal receipt."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ptcg_rl.dashboard.task_models import TaskReceipt
from ptcg_rl.dashboard.task_resources import process_start_ticks
from ptcg_rl.rl.performance_state import atomic_write_json

_cancelled = False
_child: subprocess.Popen[bytes] | None = None


def main() -> int:
    """Execute receipt-owned argv and atomically publish terminal state."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _cancel)
    receipt = _read(args.receipt)
    if receipt.state == "cancelling" or _cancelled:
        _publish_cancelled(args.receipt, receipt)
        return 0
    if receipt.state != "starting":
        return 0
    now = _utc_now()
    running = receipt.model_copy(
        update={
            "state": "running",
            "worker_pid": os.getpid(),
            "worker_start_ticks": process_start_ticks(os.getpid()),
            "started_at_utc": receipt.started_at_utc or now,
            "updated_at_utc": now,
            "queue_reason": None,
        }
    )
    _write(args.receipt, running)
    log_path = Path(running.cwd) / running.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if _cancelled:
            _publish_cancelled(args.receipt, running)
            return 0
        with log_path.open("ab", buffering=0) as log:
            global _child
            _child = subprocess.Popen(  # noqa: S603 - server-generated argv.
                running.argv,
                cwd=running.cwd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if _cancelled:
                _child.terminate()
            exit_code = _child.wait()
        state = (
            "cancelled" if _cancelled else "succeeded" if exit_code == 0 else "failed"
        )
        terminal = running.model_copy(
            update={
                "state": state,
                "exit_code": exit_code,
                "updated_at_utc": _utc_now(),
                "finished_at_utc": _utc_now(),
            }
        )
        _write(args.receipt, terminal)
        return exit_code
    except BaseException as error:
        terminal = running.model_copy(
            update={
                "state": "cancelled" if _cancelled else "failed",
                "updated_at_utc": _utc_now(),
                "finished_at_utc": _utc_now(),
                "detail": f"{type(error).__name__}: {error}",
            }
        )
        _write(args.receipt, terminal)
        raise


def _cancel(_signum: int, _frame: object) -> None:
    global _cancelled
    _cancelled = True
    if _child is not None and _child.poll() is None:
        _child.terminate()


def _publish_cancelled(path: Path, receipt: TaskReceipt) -> None:
    now = _utc_now()
    _write(
        path,
        receipt.model_copy(
            update={
                "state": "cancelled",
                "queue_reason": None,
                "updated_at_utc": now,
                "finished_at_utc": now,
            }
        ),
    )


def _read(path: Path) -> TaskReceipt:
    return TaskReceipt.model_validate_json(path.read_text(encoding="utf-8"))


def _write(path: Path, receipt: TaskReceipt) -> None:
    atomic_write_json(path, receipt.model_dump(mode="json"))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
