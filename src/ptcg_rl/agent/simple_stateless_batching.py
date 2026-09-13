"""Cross-game batching for routed greedy policy evaluation."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import Counter, defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from ptcg_rl.agent.simple_stateless_runtime import (
    PreparedSimpleStatelessDecision,
    SimpleStatelessBatchResult,
)


@dataclass(slots=True)
class _PendingDecision:
    request: PreparedSimpleStatelessDecision
    arrived_at: float
    event: threading.Event
    result: SimpleStatelessBatchResult | None = None
    error: BaseException | None = None


class RoutedPolicyInferenceBatcher:
    """Merge ready game lanes while one thread owns each CUDA model call."""

    def __init__(
        self,
        *,
        maximum_rows: int,
        maximum_wait_seconds: float,
        coalesce_temperatures: bool = True,
    ) -> None:
        if maximum_rows <= 1:
            raise ValueError("routed policy batching requires more than one row")
        if not math.isfinite(maximum_wait_seconds) or maximum_wait_seconds < 0.0:
            raise ValueError("routed policy batch wait must be finite and non-negative")
        self.maximum_rows = maximum_rows
        self.maximum_wait_seconds = maximum_wait_seconds
        self.coalesce_temperatures = coalesce_temperatures
        self._ingress: queue.Queue[_PendingDecision | None] = queue.Queue()
        self._state_lock = threading.Lock()
        self._closed = False
        self._fatal_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._serve,
            name="evaluation-policy-batcher",
            daemon=True,
        )
        self.batch_sizes: Counter[int] = Counter()
        self.batches_by_model: defaultdict[object, int] = defaultdict(int)
        self.rows = 0
        self.service_seconds = 0.0
        self._thread.start()

    def submit(
        self,
        request: PreparedSimpleStatelessDecision,
    ) -> SimpleStatelessBatchResult:
        """Wait for one model-coherent batch result without owning CUDA."""
        pending = _PendingDecision(
            request=request,
            arrived_at=time.perf_counter(),
            event=threading.Event(),
        )
        with self._state_lock:
            if self._closed:
                request.policy.abort_batched_decision(request)
                raise RuntimeError("routed policy inference batcher is closed")
            fatal_error = self._fatal_error
            if fatal_error is not None:
                request.policy.abort_batched_decision(request)
                raise RuntimeError(
                    "routed policy inference batcher failed"
                ) from fatal_error
            self._ingress.put_nowait(pending)
        pending.event.wait()
        if pending.error is not None:
            raise pending.error
        if pending.result is None:
            raise RuntimeError("routed policy batch result is absent")
        return pending.result

    def snapshot(self) -> dict[str, Any]:
        """Return aggregate, non-semantic scheduling telemetry."""
        with self._state_lock:
            return {
                "batch_sizes": dict(self.batch_sizes),
                "batches": sum(self.batch_sizes.values()),
                "rows": self.rows,
                "service_seconds": self.service_seconds,
                "coalesce_temperatures": self.coalesce_temperatures,
            }

    def close(self) -> None:
        """Drain accepted decisions and stop the owner thread exactly once."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._ingress.put_nowait(None)
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            raise TimeoutError("routed policy inference batcher did not stop")

    def _serve(self) -> None:
        pending_by_model: dict[object, deque[_PendingDecision]] = {}
        stop_requested = False
        try:
            while not stop_requested or pending_by_model:
                selected = self._select_ready(pending_by_model, stop_requested)
                if selected is not None:
                    key, batch = selected
                    self._serve_batch(key, batch)
                    continue
                timeout = self._next_wait_seconds(pending_by_model)
                try:
                    item = self._ingress.get(timeout=timeout)
                except queue.Empty:
                    continue
                if item is None:
                    stop_requested = True
                else:
                    key = self._request_batch_key(item)
                    pending_by_model.setdefault(key, deque()).append(item)
                while not stop_requested:
                    try:
                        item = self._ingress.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        stop_requested = True
                        break
                    key = self._request_batch_key(item)
                    pending_by_model.setdefault(key, deque()).append(item)
        except BaseException as error:
            with self._state_lock:
                self._fatal_error = error
            for group in pending_by_model.values():
                self._fail(tuple(group), error)
            while True:
                try:
                    item = self._ingress.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    self._fail((item,), error)

    def _select_ready(
        self,
        pending_by_model: dict[object, deque[_PendingDecision]],
        stop_requested: bool,
    ) -> tuple[object, tuple[_PendingDecision, ...]] | None:
        if not pending_by_model:
            return None
        now = time.perf_counter()
        threshold_keys = tuple(
            key
            for key, group in pending_by_model.items()
            if len(group) >= self.maximum_rows
        )
        if threshold_keys:
            selected_key = min(
                threshold_keys,
                key=lambda key: pending_by_model[key][0].arrived_at,
            )
        else:
            selected_key = min(
                pending_by_model,
                key=lambda key: pending_by_model[key][0].arrived_at,
            )
            oldest = pending_by_model[selected_key][0]
            if not stop_requested and (
                now - oldest.arrived_at < self.maximum_wait_seconds
            ):
                return None
        group = pending_by_model[selected_key]
        batch = tuple(
            group.popleft() for _ in range(min(len(group), self.maximum_rows))
        )
        if not group:
            del pending_by_model[selected_key]
        return selected_key, batch

    def _next_wait_seconds(
        self,
        pending_by_model: dict[object, deque[_PendingDecision]],
    ) -> float | None:
        if not pending_by_model:
            return None
        oldest = min(group[0].arrived_at for group in pending_by_model.values())
        deadline = oldest + self.maximum_wait_seconds
        return max(0.0, deadline - time.perf_counter())

    def _serve_batch(
        self,
        key: object,
        batch: tuple[_PendingDecision, ...],
    ) -> None:
        started = time.perf_counter()
        requests = tuple(item.request for item in batch)
        try:
            actions = requests[0].policy.execute_batched_decisions(requests)
            if len(actions) != len(batch):
                raise RuntimeError("routed policy batch returned the wrong row count")
        except BaseException as error:
            self._fail(batch, error)
            return
        elapsed = time.perf_counter() - started
        batch_size = len(batch)
        for item, action in zip(batch, actions, strict=True):
            item.result = SimpleStatelessBatchResult(
                action=action,
                batch_size=batch_size,
                queue_seconds=max(0.0, started - item.arrived_at),
                service_seconds=elapsed,
            )
            item.event.set()
        with self._state_lock:
            self.batch_sizes[batch_size] += 1
            self.batches_by_model[key] += 1
            self.rows += batch_size
            self.service_seconds += elapsed

    def _request_batch_key(self, item: _PendingDecision) -> object:
        """Choose semantic or model-only grouping for one queued request."""
        policy = item.request.policy
        if self.coalesce_temperatures:
            return getattr(
                policy, "inference_model_batch_key", policy.inference_batch_key
            )
        return policy.inference_batch_key

    @staticmethod
    def _fail(
        batch: tuple[_PendingDecision, ...],
        error: BaseException,
    ) -> None:
        for item in batch:
            with suppress(BaseException):
                item.request.policy.abort_batched_decision(item.request)
            item.error = error
            item.event.set()


__all__ = ["RoutedPolicyInferenceBatcher"]
