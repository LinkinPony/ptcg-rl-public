"""Fixed-shape stage telemetry for the integrated planner hot path."""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Self


class PlannerStage(StrEnum):
    """Stable stages shared by training and packaged-serving profiling."""

    QUEUE_WAIT = "queue_wait"
    BASE_PROPOSAL = "base_proposal"
    CANDIDATE_CONSTRUCTION = "candidate_construction"
    NATIVE_QUEUE_WAIT = "native_queue_wait"
    NATIVE_PACK = "native_pack"
    NATIVE_ENGINE = "native_engine"
    NATIVE_PARSE = "native_parse"
    TENSORIZATION = "tensorization"
    GPU_LEAF_VALUE = "gpu_leaf_value"
    AGGREGATION = "aggregation"
    CANDIDATE_MODEL = "candidate_model"
    BEHAVIOR_SAMPLE = "behavior_sample"
    IPC = "ipc"


@dataclass(frozen=True)
class PlannerStageEvent:
    """One privacy-safe stage duration and bounded work shape."""

    stage: PlannerStage
    seconds: float
    rows: int = 0
    bytes_count: int = 0
    batch_capacity: int = 0

    @property
    def batch_fill(self) -> float | None:
        """Return fill ratio only when a capacity was supplied."""
        if self.batch_capacity <= 0:
            return None
        return self.rows / self.batch_capacity


@dataclass(frozen=True, slots=True)
class PlannerDecisionRuntimeStats:
    """Exact request-local reuse and bounded-work counters."""

    root_context_hits: int = 0
    root_context_misses: int = 0
    engine_reuse_hits: int = 0
    engine_reuse_misses: int = 0
    leaf_reuse_hits: int = 0
    leaf_reuse_misses: int = 0
    prefix_reuse_count: int = 0
    unique_leaf_count: int = 0
    consequence_cell_count: int = 0
    batch_row_position: int = 0

    def __post_init__(self) -> None:
        if min(
            self.root_context_hits,
            self.root_context_misses,
            self.engine_reuse_hits,
            self.engine_reuse_misses,
            self.leaf_reuse_hits,
            self.leaf_reuse_misses,
            self.prefix_reuse_count,
            self.unique_leaf_count,
            self.consequence_cell_count,
            self.batch_row_position,
        ) < 0:
            raise ValueError("planner runtime counters must be non-negative")


@dataclass(frozen=True)
class PlannerTelemetrySummary:
    """Bounded-window latency and lifetime diagnostic counters."""

    decisions: int
    planner_decisions: int
    fallback_decisions: int
    fallback_reasons: dict[str, int]
    stage_counts: dict[str, int]
    stage_seconds: dict[str, float]
    stage_p50_ms: dict[str, float]
    stage_p95_ms: dict[str, float]
    stage_p99_ms: dict[str, float]
    cache_hits: dict[str, int]
    cache_misses: dict[str, int]
    maximum_native_occupancy: int
    maximum_gpu_batch_fill: float


class PlannerStageSpan:
    """Context-managed timer that appends exactly one stage event."""

    def __init__(self, telemetry: PlannerRequestTelemetry, stage: PlannerStage) -> None:
        self._telemetry = telemetry
        self._stage = stage
        self._started = 0.0
        self._closed = False
        self.rows = 0
        self.bytes_count = 0
        self.batch_capacity = 0

    def __enter__(self) -> Self:
        self._started = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self._closed:
            raise RuntimeError("planner telemetry span closed more than once")
        self._closed = True
        self._telemetry.record(
            PlannerStageEvent(
                stage=self._stage,
                seconds=time.perf_counter() - self._started,
                rows=self.rows,
                bytes_count=self.bytes_count,
                batch_capacity=self.batch_capacity,
            )
        )


class PlannerRequestTelemetry:
    """Request-local telemetry collector with no state or private identities."""

    def __init__(self) -> None:
        self._events: list[PlannerStageEvent] = []

    def span(self, stage: PlannerStage) -> PlannerStageSpan:
        """Create one timer for a named hot-path stage."""
        return PlannerStageSpan(self, stage)

    def record(self, event: PlannerStageEvent) -> None:
        """Append one already-measured event after strict shape validation."""
        if not math.isfinite(event.seconds) or event.seconds < 0.0:
            raise ValueError("planner stage seconds must be finite and non-negative")
        if event.rows < 0 or event.bytes_count < 0 or event.batch_capacity < 0:
            raise ValueError("planner stage work shape must be non-negative")
        if event.batch_capacity and event.rows > event.batch_capacity:
            raise ValueError("planner stage rows cannot exceed batch capacity")
        self._events.append(event)

    @property
    def events(self) -> tuple[PlannerStageEvent, ...]:
        """Return immutable events in execution order."""
        return tuple(self._events)


class PlannerTelemetryAccumulator:
    """Thread-safe lifetime counters with bounded percentile windows."""

    def __init__(self, *, latency_window: int) -> None:
        if latency_window <= 0:
            raise ValueError("planner telemetry latency_window must be positive")
        self._window = int(latency_window)
        self._lock = threading.Lock()
        self._decisions = 0
        self._planner_decisions = 0
        self._fallback_reasons: Counter[str] = Counter()
        self._stage_counts: Counter[str] = Counter()
        self._stage_seconds: dict[str, float] = {}
        self._stage_windows: dict[str, deque[float]] = {}
        self._cache_hits: Counter[str] = Counter()
        self._cache_misses: Counter[str] = Counter()
        self._maximum_native_occupancy = 0
        self._maximum_gpu_batch_fill = 0.0

    def record_decision(
        self,
        events: tuple[PlannerStageEvent, ...],
        *,
        planner_used: bool,
        fallback_reason: str | None,
        cache_hits: dict[str, int] | None = None,
        cache_misses: dict[str, int] | None = None,
        native_occupancy: int = 0,
    ) -> None:
        """Record one decision without retaining observation/action payloads."""
        if native_occupancy < 0:
            raise ValueError("native occupancy must be non-negative")
        with self._lock:
            self._decisions += 1
            self._planner_decisions += int(planner_used)
            if fallback_reason is not None:
                self._fallback_reasons[str(fallback_reason)] += 1
            for event in events:
                name = str(event.stage)
                self._stage_counts[name] += 1
                self._stage_seconds[name] = (
                    self._stage_seconds.get(name, 0.0) + event.seconds
                )
                self._stage_windows.setdefault(name, deque(maxlen=self._window)).append(
                    event.seconds * 1000.0
                )
                if event.batch_fill is not None:
                    self._maximum_gpu_batch_fill = max(
                        self._maximum_gpu_batch_fill,
                        event.batch_fill,
                    )
            self._cache_hits.update(cache_hits or {})
            self._cache_misses.update(cache_misses or {})
            self._maximum_native_occupancy = max(
                self._maximum_native_occupancy,
                native_occupancy,
            )

    def summary(self) -> PlannerTelemetrySummary:
        """Return lifetime totals and bounded-window percentiles."""
        with self._lock:
            windows = {
                name: tuple(values) for name, values in self._stage_windows.items()
            }
            return PlannerTelemetrySummary(
                decisions=self._decisions,
                planner_decisions=self._planner_decisions,
                fallback_decisions=sum(self._fallback_reasons.values()),
                fallback_reasons=dict(sorted(self._fallback_reasons.items())),
                stage_counts=dict(sorted(self._stage_counts.items())),
                stage_seconds=dict(sorted(self._stage_seconds.items())),
                stage_p50_ms={
                    name: _percentile(values, 0.50)
                    for name, values in sorted(windows.items())
                },
                stage_p95_ms={
                    name: _percentile(values, 0.95)
                    for name, values in sorted(windows.items())
                },
                stage_p99_ms={
                    name: _percentile(values, 0.99)
                    for name, values in sorted(windows.items())
                },
                cache_hits=dict(sorted(self._cache_hits.items())),
                cache_misses=dict(sorted(self._cache_misses.items())),
                maximum_native_occupancy=self._maximum_native_occupancy,
                maximum_gpu_batch_fill=self._maximum_gpu_batch_fill,
            )


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


__all__ = [
    "PlannerDecisionRuntimeStats",
    "PlannerRequestTelemetry",
    "PlannerStage",
    "PlannerStageEvent",
    "PlannerStageSpan",
    "PlannerTelemetryAccumulator",
    "PlannerTelemetrySummary",
]
