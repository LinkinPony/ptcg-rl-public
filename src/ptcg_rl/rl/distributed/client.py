"""Worker-side distributed rollout clients."""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack
import zmq

from ptcg_rl.rl.distributed.compatibility import DistributedModelCompatibility
from ptcg_rl.rl.distributed.transport import serialize_trajectory_batch
from ptcg_rl.rl.experience import GameTrajectory, compact_game_trajectory


@dataclass(frozen=True)
class DistributedTrajectorySenderConfig:
    """Settings for remote trajectory batch uploads."""

    endpoint: str
    worker_id: str
    batch_decisions: int = 2048
    flush_interval_seconds: float = 0.25
    queue_get_timeout_seconds: float = 0.1
    flow_control_sleep_seconds: float = 0.25
    summary_path: Path | None = None
    compatibility: DistributedModelCompatibility | None = None


@dataclass
class DistributedTrajectorySender:
    """Drain a local trajectory queue and upload compact batches."""

    trajectory_queue: Any
    config: DistributedTrajectorySenderConfig
    flow_control_provider: Callable[[], dict[str, Any]] | None = None
    _context: zmq.Context[Any] = field(init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _sent_batches: int = 0
    _sent_trajectories: int = 0
    _sent_decisions: int = 0
    _sent_bytes: int = 0
    _send_seconds: float = 0.0
    _flow_control_pauses: int = 0
    _flow_control_sleep_seconds: float = 0.0
    _last_flow_control: dict[str, Any] = field(default_factory=dict, init=False)
    _errors: Counter[str] = field(default_factory=Counter, init=False)

    def __post_init__(self) -> None:
        """Initialize the ZMQ context after dataclass construction."""
        if self.config.batch_decisions <= 0:
            raise ValueError("batch_decisions must be positive")
        if self.config.flow_control_sleep_seconds <= 0.0:
            raise ValueError("flow_control_sleep_seconds must be positive")
        self._context = zmq.Context.instance()

    def start(self) -> None:
        """Start the sender thread."""
        if self.config.summary_path is not None:
            self.config.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="distributed-trajectory-sender",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stop the sender thread and write a final summary."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.write_summary()

    def summary(self) -> dict[str, Any]:
        """Return current sender counters."""
        with self._lock:
            return {
                "worker_id": self.config.worker_id,
                "endpoint": self.config.endpoint,
                "sent_batches": self._sent_batches,
                "sent_trajectories": self._sent_trajectories,
                "sent_decisions": self._sent_decisions,
                "sent_bytes": self._sent_bytes,
                "send_seconds": self._send_seconds,
                "flow_control_pauses": self._flow_control_pauses,
                "flow_control_sleep_seconds": self._flow_control_sleep_seconds,
                "last_flow_control": dict(self._last_flow_control),
                "errors": dict(self._errors),
            }

    def write_summary(self) -> None:
        """Persist sender counters when a summary path is configured."""
        if self.config.summary_path is None:
            return
        payload = {
            "updated_at": time.time(),
            "summary": self.summary(),
        }
        tmp_path = self.config.summary_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.config.summary_path)

    def _run(self) -> None:
        socket = self._context.socket(zmq.PUSH)
        socket.connect(self.config.endpoint)
        batch: list[GameTrajectory] = []
        decisions = 0
        sequence_id = 0
        last_flush = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                if self._pause_for_flow_control():
                    continue
                timeout = self.config.queue_get_timeout_seconds
                try:
                    trajectory = self.trajectory_queue.get(timeout=timeout)
                except queue.Empty:
                    if batch and _should_flush(
                        last_flush,
                        self.config.flush_interval_seconds,
                    ):
                        sequence_id = self._send_batch(socket, batch, sequence_id)
                        batch = []
                        decisions = 0
                        last_flush = time.perf_counter()
                    continue
                if not isinstance(trajectory, GameTrajectory):
                    self._record_error("non_trajectory")
                    continue
                compact = compact_game_trajectory(trajectory)
                batch.append(compact)
                decisions += compact.decision_count
                if decisions >= self.config.batch_decisions:
                    sequence_id = self._send_batch(socket, batch, sequence_id)
                    batch = []
                    decisions = 0
                    last_flush = time.perf_counter()
            if batch:
                self._send_batch(socket, batch, sequence_id)
        finally:
            socket.close(linger=0)

    def _send_batch(
        self,
        socket: zmq.Socket[Any],
        batch: list[GameTrajectory],
        sequence_id: int,
    ) -> int:
        while not self._stop_event.is_set() and self._pause_for_flow_control():
            pass
        started = time.perf_counter()
        try:
            header, frames = serialize_trajectory_batch(
                batch,
                worker_id=self.config.worker_id,
                sequence_id=sequence_id,
                compatibility=self.config.compatibility,
            )
            socket.send_multipart([header, *frames])
        except (ValueError, TypeError, zmq.ZMQError) as exc:
            self._record_error(f"send:{type(exc).__name__}")
            return sequence_id + 1
        send_seconds = time.perf_counter() - started
        payload_bytes = len(header) + sum(_frame_nbytes(frame) for frame in frames)
        with self._lock:
            self._sent_batches += 1
            self._sent_trajectories += len(batch)
            self._sent_decisions += sum(trajectory.decision_count for trajectory in batch)
            self._sent_bytes += payload_bytes
            self._send_seconds += send_seconds
        if self._sent_batches % 16 == 0:
            self.write_summary()
        return sequence_id + 1

    def _record_error(self, key: str) -> None:
        with self._lock:
            self._errors[key] += 1

    def _pause_for_flow_control(self) -> bool:
        if self.flow_control_provider is None:
            return False
        flow_control = self.flow_control_provider()
        with self._lock:
            self._last_flow_control = dict(flow_control)
        if not bool(flow_control.get("pause")):
            return False
        sleep_seconds = _positive_float(
            flow_control.get("sleep_seconds"),
            default=self.config.flow_control_sleep_seconds,
        )
        self._stop_event.wait(sleep_seconds)
        with self._lock:
            self._flow_control_pauses += 1
            self._flow_control_sleep_seconds += sleep_seconds
        return True


@dataclass(frozen=True)
class DistributedWeightClientConfig:
    """Settings for remote checkpoint polling."""

    endpoint: str
    worker_id: str
    weights_dir: Path
    poll_interval_seconds: float = 5.0
    request_timeout_ms: int = 30_000
    keep_last: int = 2
    summary_path: Path | None = None
    compatibility: DistributedModelCompatibility | None = None


@dataclass
class DistributedWeightClient:
    """Poll coordinator checkpoints into a local weights directory."""

    config: DistributedWeightClientConfig
    _context: zmq.Context[Any] = field(init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _loaded_version: int = -1
    _requests: int = 0
    _updates: int = 0
    _bytes: int = 0
    _pruned_checkpoints: int = 0
    _pruned_temporary_files: int = 0
    _flow_control: dict[str, Any] = field(default_factory=dict, init=False)
    _errors: Counter[str] = field(default_factory=Counter, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        """Initialize the ZMQ context after dataclass construction."""
        if self.config.keep_last <= 0:
            raise ValueError("distributed weight keep_last must be positive")
        self._prepare_weights_dir()
        self._context = zmq.Context.instance()

    def start(self) -> None:
        """Start the background polling thread."""
        self.config.weights_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="distributed-weight-client",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stop the polling thread and write a final summary."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.write_summary()

    def poll_once(self) -> None:
        """Run one synchronous checkpoint poll."""
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.config.request_timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.config.request_timeout_ms)
        socket.connect(self.config.endpoint)
        try:
            self._poll_with_socket(socket)
        finally:
            socket.close(linger=0)

    def summary(self) -> dict[str, Any]:
        """Return current weight client counters."""
        with self._lock:
            return {
                "worker_id": self.config.worker_id,
                "endpoint": self.config.endpoint,
                "loaded_version": self._loaded_version,
                "requests": self._requests,
                "updates": self._updates,
                "bytes": self._bytes,
                "pruned_checkpoints": self._pruned_checkpoints,
                "pruned_temporary_files": self._pruned_temporary_files,
                "flow_control": dict(self._flow_control),
                "errors": dict(self._errors),
            }

    def flow_control(self) -> dict[str, Any]:
        """Return the latest coordinator-side flow-control snapshot."""
        with self._lock:
            return dict(self._flow_control)

    def write_summary(self) -> None:
        """Persist weight client counters when configured."""
        if self.config.summary_path is None:
            return
        self.config.summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at": time.time(), "summary": self.summary()}
        tmp_path = self.config.summary_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.config.summary_path)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.poll_once()
            self._stop_event.wait(self.config.poll_interval_seconds)

    def _poll_with_socket(self, socket: zmq.Socket[Any]) -> None:
        request = {
            "type": "latest_weight",
            "worker_id": self.config.worker_id,
            "loaded_version": self._loaded_version,
        }
        if self.config.compatibility is not None:
            request["compatibility"] = self.config.compatibility.model_dump(mode="json")
        with self._lock:
            self._requests += 1
        try:
            socket.send_multipart([msgpack.packb(request, use_bin_type=True)])
            parts = socket.recv_multipart()
        except zmq.ZMQError as exc:
            self._record_error(f"request:{type(exc).__name__}")
            return
        if not parts:
            self._record_error("empty_response")
            return
        header = msgpack.unpackb(parts[0], raw=False)
        if not isinstance(header, dict):
            self._record_error("invalid_header")
            return
        if header.get("compatible") is False:
            message = str(header.get("error", "distributed compatibility mismatch"))
            raise RuntimeError(message)
        self._update_flow_control(header.get("flow_control"))
        if not header.get("available"):
            return
        if len(parts) < 2:
            self._record_error("missing_payload")
            return
        version = int(header["version"])
        filename = str(header["filename"])
        payload = parts[1]
        self._write_checkpoint(version=version, filename=filename, header=header, payload=payload)

    def _write_checkpoint(
        self,
        *,
        version: int,
        filename: str,
        header: dict[str, Any],
        payload: bytes,
    ) -> None:
        self.config.weights_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.config.weights_dir / filename
        tmp_checkpoint = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        tmp_checkpoint.write_bytes(payload)
        tmp_checkpoint.replace(checkpoint_path)
        latest_record = {
            "version": version,
            "path": str(checkpoint_path),
            "published_at": str(header.get("published_at", "")),
            "metadata": dict(header.get("metadata", {})),
        }
        latest_tmp = self.config.weights_dir / ".latest.distributed.tmp"
        latest_tmp.write_text(json.dumps(latest_record, sort_keys=True), encoding="utf-8")
        latest_tmp.replace(self.config.weights_dir / "latest.json")
        with self._lock:
            self._loaded_version = version
            self._updates += 1
            self._bytes += len(payload)
        self._prune_rolling_weights(active_version=version)
        self.write_summary()

    def _record_error(self, key: str) -> None:
        with self._lock:
            self._errors[key] += 1

    def _update_flow_control(self, value: Any) -> None:
        if not isinstance(value, dict):
            return
        with self._lock:
            self._flow_control = dict(value)

    def _prepare_weights_dir(self) -> None:
        """Recover the active pointer and prune only worker rolling artifacts."""
        directory = self.config.weights_dir
        resolved_directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        removed_temporary = 0
        for path in directory.glob("*.tmp"):
            if path.name.startswith(("policy_v", ".latest")):
                path.unlink(missing_ok=True)
                removed_temporary += 1
        latest_path = directory / "latest.json"
        active_version: int | None = None
        if latest_path.exists():
            try:
                payload = json.loads(latest_path.read_text(encoding="utf-8"))
                active_version = int(payload["version"])
                active_path = Path(str(payload["path"]))
                if not active_path.is_absolute():
                    active_path = directory / active_path.name
                if (
                    active_path.resolve().parent != resolved_directory
                    or not active_path.exists()
                ):
                    active_version = None
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                active_version = None
        with self._lock:
            self._pruned_temporary_files += removed_temporary
            if active_version is not None:
                self._loaded_version = active_version
        self._prune_rolling_weights(active_version=active_version)

    def _prune_rolling_weights(self, *, active_version: int | None) -> None:
        """Keep the active worker checkpoint and its nearest predecessor."""
        versioned = sorted(
            (
                (version, path)
                for path in self.config.weights_dir.glob("policy_v*.pt")
                if (version := _rolling_weight_version(path)) is not None
            ),
            reverse=True,
        )
        if active_version is None:
            keep = {version for version, _path in versioned[: self.config.keep_last]}
        else:
            active_and_previous = [
                version for version, _path in versioned if version <= active_version
            ][: self.config.keep_last]
            keep = {active_version, *active_and_previous}
        removed = 0
        for version, path in versioned:
            if version in keep:
                continue
            path.unlink(missing_ok=True)
            removed += 1
        if removed:
            with self._lock:
                self._pruned_checkpoints += removed


def _should_flush(last_flush: float, interval_seconds: float) -> bool:
    return time.perf_counter() - last_flush >= interval_seconds


def _frame_nbytes(frame: bytes | memoryview) -> int:
    """Return payload bytes for a bytes-like frame."""
    if isinstance(frame, memoryview):
        return int(frame.nbytes)
    return len(frame)


def _positive_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0.0 else default


def _rolling_weight_version(path: Path) -> int | None:
    match = re.fullmatch(r"policy_v(\d+)\.pt", path.name)
    return None if match is None else int(match.group(1))
