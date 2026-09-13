"""Coordinator-side distributed rollout services."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack
import zmq

from ptcg_rl.rl.distributed.compatibility import (
    DistributedModelCompatibility,
    validate_distributed_compatibility,
)
from ptcg_rl.rl.distributed.transport import deserialize_trajectory_batch
from ptcg_rl.rl.experience import GameTrajectory
from ptcg_rl.rl.learner import read_latest_published_weights
from ptcg_rl.rl.performance import TrainingPerformanceReporter
from ptcg_rl.rl.performance_state import PerformanceReporterConfig


@dataclass(frozen=True)
class DistributedCoordinatorConfig:
    """Runtime settings for the coordinator transport services."""

    bind_host: str
    trajectory_port: int
    weight_port: int
    summary_path: Path
    weights_dir: Path
    queue_put_timeout_seconds: float = 5.0
    socket_poll_timeout_ms: int = 100
    weight_poll_interval_versions: int = 5
    max_staleness: int | None = None
    queue_maxsize: int | None = None
    queue_high_watermark_ratio: float = 0.85
    queue_resume_watermark_ratio: float = 0.50
    flow_control_sleep_seconds: float = 0.25
    performance: PerformanceReporterConfig | None = None
    compatibility: DistributedModelCompatibility | None = None


@dataclass
class DistributedCoordinator:
    """Receive remote trajectories and serve low-frequency checkpoints."""

    trajectory_queue: Any
    config: DistributedCoordinatorConfig
    _context: zmq.Context[Any] = field(init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _trajectory_thread: threading.Thread | None = field(default=None, init=False)
    _weight_thread: threading.Thread | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _received_batches: int = 0
    _received_trajectories: int = 0
    _received_decisions: int = 0
    _received_bytes: int = 0
    _dropped_stale_trajectories: int = 0
    _dropped_stale_decisions: int = 0
    _dropped_overflow_trajectories: int = 0
    _dropped_overflow_decisions: int = 0
    _queue_full_events: int = 0
    _queue_put_seconds: float = 0.0
    _weight_requests: int = 0
    _weight_responses: int = 0
    _last_overflow_at: float | None = None
    _flow_paused: bool = False
    _last_batch_by_worker: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
    )
    _errors: Counter[str] = field(default_factory=Counter, init=False)
    _performance: TrainingPerformanceReporter | None = field(
        default=None,
        init=False,
    )

    def __post_init__(self) -> None:
        """Initialize the ZMQ context after dataclass construction."""
        if self.config.queue_maxsize is not None and self.config.queue_maxsize <= 0:
            raise ValueError("queue_maxsize must be positive when set")
        if self.config.queue_put_timeout_seconds <= 0.0:
            raise ValueError("queue_put_timeout_seconds must be positive")
        if not 0.0 <= self.config.queue_resume_watermark_ratio <= 1.0:
            raise ValueError("queue_resume_watermark_ratio must be in [0, 1]")
        if not 0.0 <= self.config.queue_high_watermark_ratio <= 1.0:
            raise ValueError("queue_high_watermark_ratio must be in [0, 1]")
        if (
            self.config.queue_resume_watermark_ratio
            > self.config.queue_high_watermark_ratio
        ):
            raise ValueError(
                "queue_resume_watermark_ratio must be <= queue_high_watermark_ratio"
            )
        if self.config.flow_control_sleep_seconds <= 0.0:
            raise ValueError("flow_control_sleep_seconds must be positive")
        self._context = zmq.Context.instance()
        if self.config.performance is not None:
            self._performance = TrainingPerformanceReporter(self.config.performance)

    def start(self) -> None:
        """Start trajectory receiver and checkpoint server threads."""
        self.config.summary_path.parent.mkdir(parents=True, exist_ok=True)
        if self._performance is not None:
            self._performance.start()
        self._trajectory_thread = threading.Thread(
            target=self._run_trajectory_receiver,
            name="distributed-trajectory-receiver",
            daemon=True,
        )
        self._weight_thread = threading.Thread(
            target=self._run_weight_server,
            name="distributed-weight-server",
            daemon=True,
        )
        self._trajectory_thread.start()
        self._weight_thread.start()

    def close(self) -> None:
        """Stop background services and write a final summary."""
        self._stop_event.set()
        for thread in (self._trajectory_thread, self._weight_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        if self._performance is not None:
            self._performance.close()
        self.write_summary()

    def summary(self) -> dict[str, Any]:
        """Return current coordinator counters."""
        performance = None if self._performance is None else self._performance.summary()
        with self._lock:
            return {
                "received_batches": self._received_batches,
                "received_trajectories": self._received_trajectories,
                "received_decisions": self._received_decisions,
                "received_bytes": self._received_bytes,
                "dropped_stale_trajectories": self._dropped_stale_trajectories,
                "dropped_stale_decisions": self._dropped_stale_decisions,
                "dropped_overflow_trajectories": self._dropped_overflow_trajectories,
                "dropped_overflow_decisions": self._dropped_overflow_decisions,
                "queue_full_events": self._queue_full_events,
                "queue_put_seconds": self._queue_put_seconds,
                "weight_requests": self._weight_requests,
                "weight_responses": self._weight_responses,
                "last_batch_by_worker": dict(self._last_batch_by_worker),
                "errors": dict(self._errors),
                "performance": performance,
                "flow_control": self._flow_control_snapshot_unlocked(),
                "trajectory_endpoint": _endpoint(
                    self.config.bind_host,
                    self.config.trajectory_port,
                ),
                "weight_endpoint": _endpoint(
                    self.config.bind_host,
                    self.config.weight_port,
                ),
            }

    def write_summary(self) -> None:
        """Persist coordinator counters as JSON."""
        payload = {
            "updated_at": time.time(),
            "summary": self.summary(),
        }
        tmp_path = self.config.summary_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.config.summary_path)

    def _run_trajectory_receiver(self) -> None:
        socket = self._context.socket(zmq.PULL)
        socket.setsockopt(zmq.RCVTIMEO, self.config.socket_poll_timeout_ms)
        socket.bind(_endpoint(self.config.bind_host, self.config.trajectory_port))
        try:
            while not self._stop_event.is_set():
                try:
                    parts = socket.recv_multipart()
                except zmq.Again:
                    continue
                except zmq.ZMQError as exc:
                    self._record_error(f"trajectory_zmq:{type(exc).__name__}")
                    continue
                if not parts:
                    continue
                try:
                    batch = deserialize_trajectory_batch(parts[0], parts[1:])
                    if self.config.compatibility is not None:
                        validate_distributed_compatibility(
                            self.config.compatibility,
                            batch.compatibility,
                        )
                        if batch.schema != self.config.compatibility.trajectory_schema_version:
                            raise ValueError(
                                "trajectory payload schema disagrees with compatibility manifest"
                            )
                    if self._performance is not None:
                        self._performance.observe_decoded(
                            worker_id=batch.worker_id,
                            trajectories=batch.trajectories,
                        )
                    trajectories, dropped_stale, dropped_stale_decisions = (
                        self._filter_stale_trajectories(batch.trajectories)
                    )
                    started = time.perf_counter()
                    (
                        queued_trajectories,
                        queued_decisions,
                        dropped_overflow,
                        dropped_overflow_decisions,
                    ) = self._enqueue_with_backpressure(trajectories)
                    if self._performance is not None:
                        self._performance.record_delivery(
                            stale_excluded_games=dropped_stale,
                            queued_games=queued_trajectories,
                        )
                    put_seconds = time.perf_counter() - started
                except (ValueError, TypeError) as exc:
                    self._record_error(f"trajectory_decode:{type(exc).__name__}")
                    continue
                with self._lock:
                    self._received_batches += 1
                    self._received_trajectories += queued_trajectories
                    self._received_decisions += queued_decisions
                    self._received_bytes += batch.payload_bytes
                    self._dropped_stale_trajectories += dropped_stale
                    self._dropped_stale_decisions += dropped_stale_decisions
                    self._dropped_overflow_trajectories += dropped_overflow
                    self._dropped_overflow_decisions += dropped_overflow_decisions
                    if dropped_overflow:
                        self._queue_full_events += 1
                        self._last_overflow_at = time.time()
                        self._flow_paused = True
                    self._queue_put_seconds += put_seconds
                    self._last_batch_by_worker[batch.worker_id] = {
                        "sequence_id": batch.sequence_id,
                        "decisions": batch.decisions,
                        "trajectories": len(batch.trajectories),
                        "queued_decisions": queued_decisions,
                        "queued_trajectories": queued_trajectories,
                        "dropped_stale_trajectories": dropped_stale,
                        "dropped_stale_decisions": dropped_stale_decisions,
                        "dropped_overflow_trajectories": dropped_overflow,
                        "dropped_overflow_decisions": dropped_overflow_decisions,
                        "payload_bytes": batch.payload_bytes,
                        "lag_seconds": max(0.0, batch.received_at - batch.sent_at),
                        "received_at": time.time(),
                    }
                if self._received_batches % 16 == 0:
                    self.write_summary()
        finally:
            socket.close(linger=0)

    def _run_weight_server(self) -> None:
        socket = self._context.socket(zmq.REP)
        socket.setsockopt(zmq.RCVTIMEO, self.config.socket_poll_timeout_ms)
        socket.bind(_endpoint(self.config.bind_host, self.config.weight_port))
        try:
            while not self._stop_event.is_set():
                try:
                    parts = socket.recv_multipart()
                except zmq.Again:
                    continue
                except zmq.ZMQError as exc:
                    self._record_error(f"weight_zmq:{type(exc).__name__}")
                    continue
                request = _unpack_request(parts)
                response, payload = self._weight_response(request)
                frames = [msgpack.packb(response, use_bin_type=True)]
                if payload is not None:
                    frames.append(payload)
                socket.send_multipart(frames)
        finally:
            socket.close(linger=0)

    def _weight_response(
        self,
        request: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bytes | None]:
        with self._lock:
            self._weight_requests += 1
            flow_control = self._flow_control_snapshot_unlocked()
        if self.config.compatibility is not None:
            try:
                validate_distributed_compatibility(
                    self.config.compatibility,
                    request.get("compatibility"),
                )
            except (TypeError, ValueError) as exc:
                return (
                    {
                        "type": "weight",
                        "available": False,
                        "compatible": False,
                        "error": str(exc),
                        "flow_control": flow_control,
                    },
                    None,
                )
        loaded_version = int(request.get("loaded_version", -1))
        latest = read_latest_published_weights(self.config.weights_dir)
        if latest is None or latest.version <= loaded_version:
            return (
                {
                    "type": "weight",
                    "available": False,
                    "compatible": True,
                    "flow_control": flow_control,
                },
                None,
            )
        # A new worker has no policy with which to generate the first learner
        # window. Always serve that bootstrap request; apply the configured
        # low-frequency cadence only after the worker has loaded a version.
        if (
            loaded_version >= 0
            and latest.version % self.config.weight_poll_interval_versions != 0
        ):
            return (
                {
                    "type": "weight",
                    "available": False,
                    "compatible": True,
                    "flow_control": flow_control,
                },
                None,
            )
        payload = latest.path.read_bytes()
        with self._lock:
            self._weight_responses += 1
        return (
            {
                "type": "weight",
                "available": True,
                "compatible": True,
                "version": latest.version,
                "filename": latest.path.name,
                "published_at": latest.published_at,
                "metadata": dict(latest.metadata),
                "size_bytes": len(payload),
                "flow_control": flow_control,
            },
            payload,
        )

    def _enqueue_with_backpressure(
        self,
        trajectories: tuple[GameTrajectory, ...],
    ) -> tuple[int, int, int, int]:
        """Enqueue without loss, letting the bounded queue propagate backpressure."""
        queued_trajectories = 0
        queued_decisions = 0
        for trajectory in trajectories:
            while not self._stop_event.is_set():
                try:
                    self.trajectory_queue.put(
                        trajectory,
                        timeout=self.config.queue_put_timeout_seconds,
                    )
                    break
                except queue.Full:
                    with self._lock:
                        self._queue_full_events += 1
                        self._last_overflow_at = time.time()
                        self._flow_paused = True
            else:
                break
            queued_trajectories += 1
            queued_decisions += trajectory.decision_count
        return (
            queued_trajectories,
            queued_decisions,
            0,
            0,
        )

    def _filter_stale_trajectories(
        self,
        trajectories: tuple[GameTrajectory, ...],
    ) -> tuple[tuple[GameTrajectory, ...], int, int]:
        max_staleness = self.config.max_staleness
        if max_staleness is None:
            return trajectories, 0, 0
        latest = read_latest_published_weights(self.config.weights_dir)
        if latest is None:
            return trajectories, 0, 0
        kept: list[GameTrajectory] = []
        dropped = 0
        dropped_decisions = 0
        for trajectory in trajectories:
            newest_version = _trajectory_newest_policy_version(trajectory)
            if latest.version - newest_version > max_staleness:
                dropped += 1
                dropped_decisions += trajectory.decision_count
                continue
            kept.append(trajectory)
        return tuple(kept), dropped, dropped_decisions

    def _record_error(self, key: str) -> None:
        with self._lock:
            self._errors[key] += 1

    def _flow_control_snapshot_unlocked(self) -> dict[str, Any]:
        queue_size = _queue_size(self.trajectory_queue)
        maxsize = self.config.queue_maxsize
        fill_ratio = (
            queue_size / float(maxsize)
            if queue_size is not None and maxsize is not None
            else None
        )
        if fill_ratio is not None:
            if self._flow_paused:
                self._flow_paused = (
                    fill_ratio > self.config.queue_resume_watermark_ratio
                )
            else:
                self._flow_paused = fill_ratio >= self.config.queue_high_watermark_ratio
        elif self._last_overflow_at is not None:
            self._flow_paused = time.time() - self._last_overflow_at < 5.0
        return {
            "pause": self._flow_paused,
            "sleep_seconds": self.config.flow_control_sleep_seconds,
            "queue_size": queue_size,
            "queue_maxsize": maxsize,
            "queue_fill_ratio": fill_ratio,
            "queue_high_watermark_ratio": self.config.queue_high_watermark_ratio,
            "queue_resume_watermark_ratio": self.config.queue_resume_watermark_ratio,
            "queue_full_events": self._queue_full_events,
            "dropped_overflow_trajectories": self._dropped_overflow_trajectories,
            "dropped_overflow_decisions": self._dropped_overflow_decisions,
        }


def _endpoint(host: str, port: int) -> str:
    return f"tcp://{host}:{int(port)}"


def _unpack_request(parts: list[bytes]) -> Mapping[str, Any]:
    if not parts:
        return {}
    request = msgpack.unpackb(parts[0], raw=False)
    if isinstance(request, Mapping):
        return request
    return {}


def _queue_size(queue_obj: Any) -> int | None:
    qsize = getattr(queue_obj, "qsize", None)
    if not callable(qsize):
        return None
    with suppress(NotImplementedError, OSError, AttributeError):
        return int(qsize())
    return None


def _trajectory_newest_policy_version(trajectory: GameTrajectory) -> int:
    block = trajectory.array_block
    if block is not None and block.policy_versions.size > 0:
        return int(block.policy_versions.max())
    return int(trajectory.metadata.policy_version)
