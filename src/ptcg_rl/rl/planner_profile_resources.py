"""Physical resource sampling and immutable host identity for planner profiles."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import threading
from dataclasses import dataclass

import torch


def profile_machine_fingerprint() -> str:
    """Bind profile points to one software, accelerator, and MPS host state."""
    cuda_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else "none"
    )
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": cuda_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY", ""),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        b"ptcg-rl/planner-profile-machine/v1\x00" + encoded
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ProfileResourcePeak:
    """Conservative process-tree host RSS and device-wide VRAM peaks."""

    host_bytes: int
    vram_bytes: int


class ProfileResourceSampler:
    """Sample physical resources only during deployment-aligned work windows."""

    def __init__(self, *, include_vram: bool, interval_seconds: float = 0.05) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("profile resource interval must be positive")
        self._include_vram = bool(include_vram)
        self._interval = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_host = 0
        self._peak_vram = 0

    def start(self) -> None:
        """Begin one resource window; repeated windows retain the point peak."""
        if self._thread is not None:
            raise RuntimeError("profile resource sampler is already active")
        self._stop.clear()
        self._sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="planner-profile-resources",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Finish the current resource window and capture its endpoint."""
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout=max(1.0, self._interval * 4.0))
        if thread.is_alive():
            raise RuntimeError("profile resource sampler did not stop")
        self._thread = None
        self._sample_once()

    @property
    def peak(self) -> ProfileResourcePeak:
        return ProfileResourcePeak(
            host_bytes=self._peak_host,
            vram_bytes=self._peak_vram,
        )

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample_once()

    def _sample_once(self) -> None:
        self._peak_host = max(self._peak_host, _process_tree_rss_bytes(os.getpid()))
        if self._include_vram:
            self._peak_vram = max(self._peak_vram, _device_vram_bytes())


def _process_tree_rss_bytes(root_pid: int) -> int:
    page_size = os.sysconf("SC_PAGE_SIZE")
    parents: dict[int, int] = {}
    rss_pages: dict[int, int] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as stat_file:
                raw_stat = stat_file.read()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        close = raw_stat.rfind(")")
        fields = raw_stat[close + 2 :].split()
        if len(fields) < 22:
            continue
        parents[pid] = int(fields[1])
        rss_pages[pid] = max(0, int(fields[21]))
    descendants = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rss_pages.get(pid, 0) for pid in descendants) * page_size


def _device_vram_bytes() -> int:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 0
    values = [
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip().isdigit()
    ]
    return sum(values) * 1024 * 1024


__all__ = [
    "ProfileResourcePeak",
    "ProfileResourceSampler",
    "profile_machine_fingerprint",
]
