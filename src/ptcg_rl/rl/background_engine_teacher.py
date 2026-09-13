"""Dedicated non-blocking execution lane for online engine teaching."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from ptcg_rl.rl.engine_teacher import (
    EngineTeacherBatchCompletion,
    EngineTeacherRequest,
    EngineTeacherTarget,
)
from ptcg_rl.rl.online_engine_teacher import OnlineEngineTeacherProducer


@dataclass(frozen=True)
class _QueuedBatch:
    batch_id: int
    requests: tuple[EngineTeacherRequest, ...]
    selected_indices: tuple[int, ...]
    request_count: int
    deadline: float


_STOP = object()


class BackgroundEngineTeacherProducer:
    """Run engine search and its GPU RPCs away from the rollout hot path.

    One thread owns one persistent, killable engine subprocess and one remote
    inference client. The actor only performs bounded queue operations and
    attaches results later, so a slow auxiliary search cannot stall behavior
    collection. A separate inference response queue is required by the caller;
    this class never competes with the behavior client for responses.
    """

    def __init__(
        self,
        producer: OnlineEngineTeacherProducer,
        *,
        queue_batches: int,
        batch_deadline_seconds: float,
    ) -> None:
        if queue_batches <= 0:
            raise ValueError("background teacher queue size must be positive")
        if batch_deadline_seconds <= 0.0:
            raise ValueError("background teacher deadline must be positive")
        self._producer = producer
        self._batch_deadline_seconds = float(batch_deadline_seconds)
        self._tasks: queue.Queue[_QueuedBatch | object] = queue.Queue(
            maxsize=int(queue_batches)
        )
        self._results: queue.Queue[EngineTeacherBatchCompletion] = queue.Queue()
        self._next_batch_id = 0
        self._pending: set[int] = set()
        self._pending_lock = threading.Lock()
        self._summary_lock = threading.Lock()
        self._producer_summary: Mapping[str, Any] = producer.summary()
        self._submitted_batches = 0
        self._submitted_requests = 0
        self._prescreened_batches = 0
        self._prescreened_requests = 0
        self._prescreen_selected_batches = 0
        self._prescreen_selected_requests = 0
        self._prescreen_skipped_batches = 0
        self._prescreen_skipped_requests = 0
        self._completed_batches = 0
        self._completed_requests = 0
        self._saturated_batches = 0
        self._saturated_requests = 0
        self._queue_expired_batches = 0
        self._queue_expired_requests = 0
        self._worker_failures = 0
        self._last_worker_error = ""
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="engine-teacher-background-lane",
            daemon=True,
        )
        self._thread.start()

    def produce(self, request: EngineTeacherRequest) -> EngineTeacherTarget | None:
        """Reject accidental synchronous use of the dedicated async lane."""
        del request
        raise RuntimeError("background engine teacher requires submit_async")

    def produce_batch(
        self,
        requests: Sequence[EngineTeacherRequest],
    ) -> tuple[EngineTeacherTarget | None, ...]:
        """Reject accidental synchronous use of the dedicated async lane."""
        del requests
        raise RuntimeError("background engine teacher requires submit_async")

    def submit_async(
        self,
        requests: Sequence[EngineTeacherRequest],
    ) -> int | None:
        """Accept one actor-step batch without waiting for engine or GPU work."""
        if self._closed:
            raise RuntimeError("background engine teacher is closed")
        frozen = tuple(requests)
        if not frozen:
            raise ValueError("background engine teacher batches cannot be empty")
        selected_indices = self._producer.screen_batch(frozen)
        selected_set = set(selected_indices)
        if len(selected_set) != len(selected_indices) or any(
            index < 0 or index >= len(frozen) for index in selected_indices
        ):
            raise RuntimeError("engine teacher prescreen returned invalid indices")
        selected_requests = tuple(frozen[index] for index in selected_indices)
        skipped_requests = len(frozen) - len(selected_requests)
        with self._summary_lock:
            self._prescreened_batches += 1
            self._prescreened_requests += len(frozen)
            self._prescreen_skipped_requests += skipped_requests
            if selected_requests:
                self._prescreen_selected_batches += 1
                self._prescreen_selected_requests += len(selected_requests)
            else:
                self._prescreen_skipped_batches += 1
        if not selected_requests:
            return None
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        task = _QueuedBatch(
            batch_id=batch_id,
            requests=selected_requests,
            selected_indices=selected_indices,
            request_count=len(frozen),
            deadline=time.perf_counter() + self._batch_deadline_seconds,
        )
        try:
            self._tasks.put_nowait(task)
        except queue.Full:
            with self._summary_lock:
                self._saturated_batches += 1
                self._saturated_requests += len(selected_requests)
            return None
        with self._pending_lock:
            self._pending.add(batch_id)
        with self._summary_lock:
            self._submitted_batches += 1
            self._submitted_requests += len(selected_requests)
        return batch_id

    def poll_completed(self) -> tuple[EngineTeacherBatchCompletion, ...]:
        """Drain ready results without blocking the actor."""
        completed: list[EngineTeacherBatchCompletion] = []
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            self._consume_pending(result.batch_id)
            completed.append(result)
        return tuple(completed)

    def drain(self) -> tuple[EngineTeacherBatchCompletion, ...]:
        """Wait only during orderly actor shutdown for all accepted work."""
        completed = list(self.poll_completed())
        while self._pending_count() > 0:
            result = self._results.get()
            self._consume_pending(result.batch_id)
            completed.append(result)
        return tuple(completed)

    def summary(self) -> Mapping[str, Any]:
        """Return a cached producer snapshot plus background-lane pressure."""
        with self._summary_lock:
            producer_summary = dict(self._producer_summary)
            eligible = self._prescreened_requests
            probability_skips = self._prescreen_skipped_requests
            attempted = int(producer_summary.get("engine_teacher_attempted", 0))
            reason_counts = dict(
                producer_summary.get("engine_teacher_reason_counts", {})
            )
            if probability_skips:
                reason_counts["probability_skip"] = probability_skips
            producer_summary.update(
                {
                    "engine_teacher_eligible": eligible,
                    "engine_teacher_attempt_rate": (
                        attempted / eligible if eligible else 0.0
                    ),
                    "engine_teacher_step_calls": self._prescreened_batches,
                    "engine_teacher_probability_skips": probability_skips,
                    "engine_teacher_reason_counts": dict(sorted(reason_counts.items())),
                }
            )
            lane_summary = {
                "engine_teacher_async": True,
                "engine_teacher_async_prescreened_batches": (self._prescreened_batches),
                "engine_teacher_async_prescreened_requests": (
                    self._prescreened_requests
                ),
                "engine_teacher_async_prescreen_selected_batches": (
                    self._prescreen_selected_batches
                ),
                "engine_teacher_async_prescreen_selected_requests": (
                    self._prescreen_selected_requests
                ),
                "engine_teacher_async_prescreen_skipped_batches": (
                    self._prescreen_skipped_batches
                ),
                "engine_teacher_async_prescreen_skipped_requests": (
                    self._prescreen_skipped_requests
                ),
                "engine_teacher_async_submitted_batches": self._submitted_batches,
                "engine_teacher_async_submitted_requests": self._submitted_requests,
                "engine_teacher_async_completed_batches": self._completed_batches,
                "engine_teacher_async_completed_requests": self._completed_requests,
                "engine_teacher_async_saturated_batches": self._saturated_batches,
                "engine_teacher_async_saturated_requests": self._saturated_requests,
                "engine_teacher_async_queue_expired_batches": (
                    self._queue_expired_batches
                ),
                "engine_teacher_async_queue_expired_requests": (
                    self._queue_expired_requests
                ),
                "engine_teacher_async_worker_failures": self._worker_failures,
                "engine_teacher_async_last_worker_error": self._last_worker_error,
            }
        lane_summary.update(
            {
                "engine_teacher_async_pending_batches": self._pending_count(),
                "engine_teacher_async_queue_depth": self._tasks.qsize(),
            }
        )
        producer_summary.update(lane_summary)
        return producer_summary

    def close(self) -> None:
        """Drain accepted work, then release the underlying native worker."""
        if self._closed:
            return
        self._closed = True
        self._tasks.put(_STOP)
        self._thread.join()
        if self._thread.is_alive():  # pragma: no cover - unbounded host failure
            raise RuntimeError("background engine teacher did not stop")

    def _run(self) -> None:
        try:
            while True:
                task = self._tasks.get()
                if task is _STOP:
                    return
                queued = cast(_QueuedBatch, task)
                selected_targets: tuple[EngineTeacherTarget | None, ...]
                if time.perf_counter() >= queued.deadline:
                    selected_targets = (None,) * len(queued.requests)
                    with self._summary_lock:
                        self._queue_expired_batches += 1
                        self._queue_expired_requests += len(queued.requests)
                else:
                    try:
                        selected_targets = self._producer.produce_batch_until(
                            queued.requests,
                            deadline=queued.deadline,
                        )
                    except Exception as exc:  # Keep auxiliary work fail-open.
                        selected_targets = (None,) * len(queued.requests)
                        with self._summary_lock:
                            self._worker_failures += 1
                            self._last_worker_error = _error_detail(exc)
                if len(selected_targets) != len(queued.selected_indices):
                    selected_targets = (None,) * len(queued.selected_indices)
                    with self._summary_lock:
                        self._worker_failures += 1
                        self._last_worker_error = (
                            "RuntimeError: engine teacher returned the wrong "
                            "prescreened batch size"
                        )
                targets: list[EngineTeacherTarget | None] = [None] * (
                    queued.request_count
                )
                for index, target in zip(
                    queued.selected_indices,
                    selected_targets,
                    strict=True,
                ):
                    targets[index] = target
                completion = EngineTeacherBatchCompletion(
                    batch_id=queued.batch_id,
                    targets=tuple(targets),
                )
                with self._summary_lock:
                    self._completed_batches += 1
                    self._completed_requests += len(queued.requests)
                    self._producer_summary = self._producer.summary()
                self._results.put(completion)
        finally:
            self._producer.close()

    def _pending_count(self) -> int:
        with self._pending_lock:
            return len(self._pending)

    def _consume_pending(self, batch_id: int) -> None:
        with self._pending_lock:
            if batch_id not in self._pending:
                raise RuntimeError("unknown background teacher completion")
            self._pending.remove(batch_id)


def _error_detail(exc: Exception) -> str:
    message = str(exc).replace("\n", " ")[:512]
    return f"{type(exc).__name__}: {message}"


__all__ = ["BackgroundEngineTeacherProducer"]
