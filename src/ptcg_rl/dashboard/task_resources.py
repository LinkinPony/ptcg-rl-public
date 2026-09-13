"""Read-only resource gates for dashboard-owned task execution."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

from ptcg_rl.dashboard.task_models import (
    TaskReceipt,
    TaskResourceClass,
    TaskResourceSnapshot,
)

_ACTIVE_STATES = frozenset({"starting", "running", "cancelling"})
_HEAVY_CLASSES = frozenset({"local_heavy", "local_cuda", "remote_heavy"})


class TaskResourceGate:
    """Inspect local processes without changing training or GPU state."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def snapshot(self, receipts: Iterable[TaskReceipt]) -> TaskResourceSnapshot:
        """Return a small current-state view for the wizard and dispatcher."""
        active = tuple(
            receipt for receipt in receipts if receipt.state in _ACTIVE_STATES
        )
        training_pids = self._training_pids()
        gpu_pids = self._gpu_compute_pids()
        task_pids = {
            receipt.worker_pid for receipt in active if receipt.worker_pid is not None
        }
        external_gpu_pids = gpu_pids - task_pids
        heavy = sum(receipt.resource_class in _HEAVY_CLASSES for receipt in active)
        light = sum(receipt.resource_class == "light" for receipt in active)
        details: list[str] = []
        if training_pids:
            details.append("训练正在运行，本机重任务与 CUDA 任务将排队")
        if external_gpu_pids:
            details.append("检测到 dashboard 任务之外的 GPU 计算进程")
        if heavy:
            details.append(f"已有 {heavy} 个重任务执行中")
        if light:
            details.append(f"已有 {light}/2 个轻任务执行中")
        if not details:
            details.append("本机资源门禁当前允许启动任务")
        return TaskResourceSnapshot(
            training_active=bool(training_pids),
            local_gpu_busy=bool(external_gpu_pids),
            running_heavy_tasks=heavy,
            running_light_tasks=light,
            detail="；".join(details),
        )

    def allow(
        self,
        resource_class: TaskResourceClass,
        receipts: Iterable[TaskReceipt],
    ) -> tuple[bool, str | None]:
        """Decide whether one queued task may start now."""
        snapshot = self.snapshot(receipts)
        if resource_class == "light":
            if snapshot.running_light_tasks >= 2:
                return False, "轻任务并发已达到 2 个"
            return True, None
        if snapshot.running_heavy_tasks:
            return False, "已有重任务执行中"
        if resource_class in {"local_heavy", "local_cuda"} and snapshot.training_active:
            return False, "训练运行期间本机重任务保持排队"
        if resource_class == "local_cuda" and snapshot.local_gpu_busy:
            return False, "本机 GPU 正被其他计算进程占用"
        return True, None

    def _training_pids(self) -> set[int]:
        output: set[int] = set()
        repo_text = str(self.repo_root)
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
            if repo_text not in command:
                continue
            if (
                "src/tools/rl_train.py" in command or f"{repo_text}/train.py" in command
            ) and "--dry-run" not in command:
                output.add(int(entry.name))
        return output

    @staticmethod
    def _gpu_compute_pids() -> set[int]:
        try:
            result = subprocess.run(
                (
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ),
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            return set()
        if result.returncode != 0:
            return set()
        return {
            int(line.strip())
            for line in result.stdout.splitlines()
            if line.strip().isdigit()
        }


def process_start_ticks(pid: int) -> int | None:
    """Read Linux proc start ticks for exact PID identity reconciliation."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    close_paren = raw.rfind(")")
    if close_paren < 0:
        return None
    fields = raw[close_paren + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def process_identity_matches(pid: int, start_ticks: int | None) -> bool:
    """Return whether the live PID is the exact receipt-owned process."""
    if start_ticks is None:
        return False
    return process_start_ticks(pid) == start_ticks


def pid_exists(pid: int) -> bool:
    """Best-effort process existence check."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
