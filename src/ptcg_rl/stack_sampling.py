"""In-process wall-clock stack sampler for opt-in runtime diagnosis.

The container blocks ptrace, so external samplers such as py-spy cannot
attach. This module offers an env-gated fallback: a daemon thread samples
``sys._current_frames()`` at a fixed interval and aggregates collapsed
stacks per thread, similar to a flamegraph collapse file.

Enable by setting ``PTCG_RL_STACK_SAMPLER_DIR`` to a writable directory.
Optional ``PTCG_RL_STACK_SAMPLER_INTERVAL_MS`` (default ``10``) controls the
sampling period. Output is one JSON file per process, rewritten atomically
every flush so a live run can be inspected while it executes.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from types import FrameType

_FLUSH_EVERY_SECONDS = 20.0
_MAX_STACK_DEPTH = 60

_sampler_singleton: _StackSampler | None = None
_sampler_lock = threading.Lock()


class _StackSampler:
    def __init__(self, output_path: Path, interval_seconds: float) -> None:
        self._output_path = output_path
        self._interval_seconds = interval_seconds
        self._samples: Counter[str] = Counter()
        self._samples_lock = threading.Lock()
        self._started_at = time.time()
        self._sample_count = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="ptcg-stack-sampler",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        atexit.register(self.flush)

    def _run(self) -> None:
        own_ident = threading.get_ident()
        next_flush = time.monotonic() + _FLUSH_EVERY_SECONDS
        while not self._stop_event.wait(self._interval_seconds):
            names = {
                thread.ident: thread.name
                for thread in threading.enumerate()
                if thread.ident is not None
            }
            frames = sys._current_frames()
            now = time.monotonic()
            with self._samples_lock:
                self._sample_count += 1
                for ident, frame in frames.items():
                    if ident == own_ident:
                        continue
                    stack: list[str] = []
                    depth = 0
                    current: FrameType | None = frame
                    while current is not None and depth < _MAX_STACK_DEPTH:
                        code = current.f_code
                        stack.append(
                            f"{code.co_name} "
                            f"({code.co_filename}:{current.f_lineno})"
                        )
                        current = current.f_back
                        depth += 1
                    collapsed = ";".join(reversed(stack))
                    thread_name = names.get(ident, f"thread-{ident}")
                    self._samples[f"{thread_name};{collapsed}"] += 1
            if now >= next_flush:
                next_flush = now + _FLUSH_EVERY_SECONDS
                self.flush()

    def flush(self) -> None:
        with self._samples_lock:
            payload = {
                "format": "ptcg-stack-samples-v1",
                "pid": os.getpid(),
                "argv": sys.argv,
                "worker_id": os.environ.get("PTCG_RL_WORKER_ID"),
                "role": os.environ.get("PTCG_RL_TRAINING_ROLE"),
                "started_at_unix": self._started_at,
                "written_at_unix": time.time(),
                "interval_seconds": self._interval_seconds,
                "sample_count": self._sample_count,
                "stacks": dict(self._samples),
            }
        tmp_path = self._output_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload))
        tmp_path.replace(self._output_path)


def maybe_start_stack_sampler() -> None:
    """Start the env-gated sampler once per process when configured."""
    global _sampler_singleton
    directory = os.environ.get("PTCG_RL_STACK_SAMPLER_DIR")
    if not directory:
        return
    with _sampler_lock:
        if _sampler_singleton is not None:
            return
        interval_ms = float(
            os.environ.get("PTCG_RL_STACK_SAMPLER_INTERVAL_MS", "10")
        )
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        role = os.environ.get("PTCG_RL_TRAINING_ROLE", "process")
        worker_id = os.environ.get("PTCG_RL_WORKER_ID", "")
        suffix = f"_{worker_id}" if worker_id else ""
        output_path = output_dir / f"{role}{suffix}_{os.getpid()}.json"
        sampler = _StackSampler(
            output_path,
            interval_seconds=max(interval_ms, 1.0) / 1000.0,
        )
        sampler.start()
        _sampler_singleton = sampler


__all__ = ["maybe_start_stack_sampler"]
