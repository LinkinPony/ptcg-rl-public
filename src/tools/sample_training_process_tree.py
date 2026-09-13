"""Sample CPU and GPU utilization for one local training process tree."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _process_stat(pid: int) -> tuple[int, int] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().partition(") ")[2].split()
    except OSError:
        return None
    return int(fields[1]), int(fields[11]) + int(fields[12])


def _process_tree(root_pid: int) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    frontier = [root_pid]
    while frontier:
        pid = frontier.pop()
        stat = _process_stat(pid)
        if stat is None:
            continue
        parent_pid, ticks = stat
        rows[str(pid)] = {"parent_pid": parent_pid, "ticks": ticks}
        try:
            children = Path(
                f"/proc/{pid}/task/{pid}/children"
            ).read_text()
        except OSError:
            children = ""
        frontier.extend(int(value) for value in children.split())
    return rows


def _gpu_snapshot() -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()


def sample_process_tree(
    root_pid: int,
    *,
    duration_seconds: float,
    interval_seconds: float,
) -> list[dict[str, Any]]:
    """Return periodic cumulative CPU ticks and GPU snapshots."""
    if root_pid <= 0 or duration_seconds <= 0.0 or interval_seconds <= 0.0:
        raise ValueError("process sampling arguments must be positive")
    samples: list[dict[str, Any]] = []
    started_at = time.monotonic()
    while time.monotonic() - started_at < duration_seconds:
        samples.append(
            {
                "time": time.time(),
                "processes": _process_tree(root_pid),
                "gpu": _gpu_snapshot(),
                "clock_ticks": os.sysconf("SC_CLK_TCK"),
            }
        )
        time.sleep(interval_seconds)
    return samples


def main() -> None:
    """Run the process-tree sampler and persist one small JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root_pid", type=int)
    parser.add_argument("duration_seconds", type=float)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args = parser.parse_args()
    samples = sample_process_tree(
        args.root_pid,
        duration_seconds=args.duration_seconds,
        interval_seconds=args.interval_seconds,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(samples))


if __name__ == "__main__":
    main()
