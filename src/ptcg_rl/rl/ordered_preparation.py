"""Bounded ordered execution for CPU preparation ahead of device work."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from typing import Generic, TypeVar

_InputT = TypeVar("_InputT")
_OutputT = TypeVar("_OutputT")


@dataclass(frozen=True)
class TimedPreparedItem(Generic[_OutputT]):
    """One ordered preparation result and its observable timing."""

    value: _OutputT
    task_seconds: float
    wait_seconds: float


@dataclass(frozen=True)
class _CompletedPreparation(Generic[_OutputT]):
    """Internal result measured entirely on its executor worker."""

    value: _OutputT
    task_seconds: float


class OrderedPreparationPipeline(Generic[_InputT, _OutputT]):
    """Run bounded preparation concurrently and consume results in input order."""

    def __init__(
        self,
        *,
        executor: Executor,
        prepare: Callable[[_InputT], _OutputT],
        items: Iterable[_InputT],
        capacity: int,
    ) -> None:
        """Submit at most ``capacity`` not-yet-consumed preparation tasks."""
        if capacity <= 0:
            raise ValueError("ordered preparation capacity must be positive")
        self._executor = executor
        self._prepare = prepare
        self._items = iter(items)
        self._capacity = capacity
        self._pending: deque[Future[_CompletedPreparation[_OutputT]]] = deque()
        self._source_exhausted = False
        self._closed = False
        self._fill()

    def __iter__(self) -> Iterator[TimedPreparedItem[_OutputT]]:
        """Yield prepared values in the exact order of their source items."""
        while self._pending:
            future = self._pending.popleft()
            wait_started_at = time.perf_counter()
            try:
                completed = future.result()
            except BaseException:
                self.close()
                raise
            wait_seconds = time.perf_counter() - wait_started_at
            self._fill()
            yield TimedPreparedItem(
                value=completed.value,
                task_seconds=completed.task_seconds,
                wait_seconds=wait_seconds,
            )
        self._closed = True

    def close(self) -> None:
        """Cancel preparation that has not started after a consumer failure."""
        if self._closed:
            return
        self._closed = True
        while self._pending:
            self._pending.popleft().cancel()

    def _fill(self) -> None:
        """Maintain the configured bound without materializing the input."""
        if self._closed or self._source_exhausted:
            return
        while len(self._pending) < self._capacity:
            try:
                item = next(self._items)
            except StopIteration:
                self._source_exhausted = True
                return
            self._pending.append(
                self._executor.submit(_timed_prepare, self._prepare, item)
            )


def _timed_prepare(
    prepare: Callable[[_InputT], _OutputT],
    item: _InputT,
) -> _CompletedPreparation[_OutputT]:
    """Measure one preparation task without synchronizing sibling workers."""
    started_at = time.perf_counter()
    value = prepare(item)
    return _CompletedPreparation(
        value=value,
        task_seconds=time.perf_counter() - started_at,
    )


__all__ = ["OrderedPreparationPipeline", "TimedPreparedItem"]
