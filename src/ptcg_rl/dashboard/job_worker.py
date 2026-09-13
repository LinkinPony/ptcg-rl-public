"""Detached worker that owns one dashboard job terminal receipt."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ptcg_rl.dashboard.job_models import JobReceipt
from ptcg_rl.rl.performance_state import atomic_write_json

_cancelled = False


def main() -> int:
    """Execute the adapter-generated argv and publish a terminal state."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = JobReceipt.model_validate_json(args.receipt.read_text(encoding="utf-8"))
    signal.signal(signal.SIGTERM, _cancel)
    running = receipt.model_copy(
        update={
            "state": "running",
            "worker_pid": os.getpid(),
            "updated_at_utc": _utc_now(),
        }
    )
    _write(args.receipt, running)
    log_path = Path(running.cwd) / running.log_path
    try:
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(  # noqa: S603 - receipt was adapter-generated.
                running.argv,
                cwd=running.cwd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            exit_code = process.wait()
        state = (
            "cancelled" if _cancelled else "succeeded" if exit_code == 0 else "failed"
        )
        terminal = running.model_copy(
            update={
                "state": state,
                "exit_code": exit_code,
                "updated_at_utc": _utc_now(),
            }
        )
        _write(args.receipt, terminal)
        return exit_code
    except BaseException as error:
        terminal = running.model_copy(
            update={
                "state": "cancelled" if _cancelled else "failed",
                "updated_at_utc": _utc_now(),
                "detail": f"{type(error).__name__}: {error}",
            }
        )
        _write(args.receipt, terminal)
        raise


def _cancel(_signum: int, _frame: object) -> None:
    global _cancelled
    _cancelled = True


def _write(path: Path, receipt: JobReceipt) -> None:
    atomic_write_json(path, receipt.model_dump(mode="json"))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
