"""Small timing helpers for opt-in runtime profiling."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass


@dataclass(frozen=True)
class StageTiming:
    """Aggregate wall-time measurements for one named stage."""

    seconds: float
    count: int

    @property
    def mean_ms(self) -> float:
        """Return average stage duration in milliseconds."""
        if self.count <= 0:
            return 0.0
        return 1000.0 * self.seconds / float(self.count)


class StageTimer:
    """Accumulate stage wall times with an optional synchronization callback."""

    def __init__(self, *, synchronize: Callable[[], None] | None = None) -> None:
        """Initialize an empty timer."""
        self._synchronize = synchronize
        self._seconds: defaultdict[str, float] = defaultdict(float)
        self._counts: defaultdict[str, int] = defaultdict(int)
        self.synchronizations = 0

    @contextmanager
    def time(self, stage: str) -> Iterator[None]:
        """Measure one stage block."""
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self._seconds[stage] += time.perf_counter() - start
            self._counts[stage] += 1

    def summary(self) -> dict[str, dict[str, float | int]]:
        """Return JSON-friendly timing aggregates."""
        return {
            stage: {
                "seconds": self._seconds[stage],
                "count": self._counts[stage],
                "mean_ms": StageTiming(
                    seconds=self._seconds[stage],
                    count=self._counts[stage],
                ).mean_ms,
            }
            for stage in sorted(self._seconds)
        }

    def _sync(self) -> None:
        if self._synchronize is None:
            return
        self._synchronize()
        self.synchronizations += 1


def time_stage(
    timer: StageTimer | None,
    stage: str,
) -> AbstractContextManager[None]:
    """Return a timing context when profiling is enabled."""
    if timer is None:
        return nullcontext()
    return timer.time(stage)
