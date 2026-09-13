"""Route-aware batching for synchronous native inference feeders."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from ptcg_rl.rl.native_process_client import (
    NativeProcessHistoricalRequest,
    NativeProcessInferenceRequest,
    NativeProcessSequenceRequest,
)
from ptcg_rl.rl.stateless_inference import StatelessInferenceRequest

NativeProcessBrokerRequest = (
    NativeProcessInferenceRequest
    | NativeProcessHistoricalRequest
    | NativeProcessSequenceRequest
    | StatelessInferenceRequest
)
NativeProcessBatchReason = Literal["threshold", "deadline", "unblock"]
_CohortKey = tuple[str, str, float | None]


@dataclass(frozen=True, slots=True)
class NativeProcessBatchSelection:
    """One selected route set and the scheduler reason for releasing it."""

    requests: tuple[NativeProcessBrokerRequest, ...]
    reason: NativeProcessBatchReason


@dataclass(slots=True)
class _PendingCohort:
    """Requests that can share one exact artifact/route forward pass."""

    key: _CohortKey
    requests: list[NativeProcessBrokerRequest]
    rows: int
    workers: set[int]
    first_arrived_at: float
    last_arrived_at: float

    @classmethod
    def create(
        cls,
        request: NativeProcessBrokerRequest,
        *,
        arrived_at: float,
    ) -> _PendingCohort:
        """Create one cohort around its first request."""
        return cls(
            key=_cohort_key(request),
            requests=[request],
            rows=_request_rows(request),
            workers=_native_workers((request,)),
            first_arrived_at=arrived_at,
            last_arrived_at=arrived_at,
        )

    def append(
        self,
        request: NativeProcessBrokerRequest,
        *,
        arrived_at: float,
    ) -> None:
        """Append one compatible request while retaining arrival evidence."""
        if _cohort_key(request) != self.key:
            raise ValueError("native broker request crossed cohort keys")
        self.requests.append(request)
        self.rows += _request_rows(request)
        self.workers.update(_native_workers((request,)))
        self.last_arrived_at = arrived_at


class NativeProcessBatchScheduler:
    """Grow current batches while releasing routes that block their feeders."""

    def __init__(
        self,
        *,
        worker_ids: Iterable[int],
        max_batch_rows: int,
        batch_wait_seconds: float,
        current_wait_multiplier: float,
        current_gather_max_seconds: float,
    ) -> None:
        normalized_workers = frozenset(int(value) for value in worker_ids)
        if (
            max_batch_rows <= 0
            or batch_wait_seconds < 0.0
            or current_wait_multiplier < 1.0
            or current_gather_max_seconds < 0.0
        ):
            raise ValueError("native process scheduler controls are invalid")
        self.worker_ids = normalized_workers
        self.max_batch_rows = int(max_batch_rows)
        self.batch_wait_seconds = float(batch_wait_seconds)
        self.current_wait_multiplier = float(current_wait_multiplier)
        self.current_gather_max_seconds = float(current_gather_max_seconds)
        self._pending: dict[_CohortKey, _PendingCohort] = {}

    @property
    def has_pending(self) -> bool:
        """Return whether at least one feeder is waiting for a response."""
        return bool(self._pending)

    def add(
        self,
        request: NativeProcessBrokerRequest,
        *,
        arrived_at: float,
    ) -> None:
        """Add one validated request to its exact batching cohort."""
        key = _cohort_key(request)
        cohort = self._pending.get(key)
        if cohort is None:
            self._pending[key] = _PendingCohort.create(
                request,
                arrived_at=arrived_at,
            )
        else:
            cohort.append(request, arrived_at=arrived_at)

    def delay(self, seconds: float) -> None:
        """Exclude synchronous control bookkeeping from batching deadlines."""
        if seconds <= 0.0:
            return
        for cohort in self._pending.values():
            cohort.first_arrived_at += seconds
            cohort.last_arrived_at += seconds

    def select(
        self,
        *,
        now: float,
        stop_requested: bool,
    ) -> NativeProcessBatchSelection | None:
        """Release the next actionable route set, if one is ready."""
        ordered = self._ordered()
        keys: tuple[_CohortKey, ...]
        reason: NativeProcessBatchReason
        if stop_requested:
            keys = tuple(cohort.key for cohort in ordered)
            reason = "deadline"
        elif sum(cohort.rows for cohort in ordered) >= self.max_batch_rows:
            keys = tuple(cohort.key for cohort in ordered)
            reason = "threshold"
        else:
            selected = self._select_below_threshold(ordered, now=now)
            if selected is None:
                return None
            keys, reason = selected
        return NativeProcessBatchSelection(
            requests=self._pop(keys),
            reason=reason,
        )

    def wait_seconds(self, *, now: float) -> float:
        """Bound the next queue wait by the earliest actionable deadline."""
        if not self._pending:
            raise RuntimeError("native process scheduler has no pending cohort")
        deadlines = tuple(
            self._current_deadline(cohort)
            if cohort.key[0] == "current"
            else cohort.last_arrived_at + self.batch_wait_seconds
            for cohort in self._pending.values()
        )
        return max(0.0, min(deadlines) - now)

    def _select_below_threshold(
        self,
        ordered: Sequence[_PendingCohort],
        *,
        now: float,
    ) -> tuple[tuple[_CohortKey, ...], NativeProcessBatchReason] | None:
        represented_workers = set().union(
            *(cohort.workers for cohort in ordered),
        )
        current = tuple(cohort for cohort in ordered if cohort.key[0] == "current")
        if current:
            target = current[0]
            if self.worker_ids.issubset(target.workers):
                return self._with_compatible_scripted(ordered, target), "deadline"
            missing_workers = self.worker_ids - target.workers
            blockers = tuple(
                cohort.key
                for cohort in ordered
                if cohort is not target and bool(cohort.workers & missing_workers)
            )
            if blockers:
                return blockers, "unblock"
            if now >= self._current_deadline(target):
                return self._with_compatible_scripted(ordered, target), "deadline"
        elif self.worker_ids and self.worker_ids.issubset(represented_workers):
            # Every synchronous feeder is waiting, so no new merge candidate
            # can arrive until at least one route receives a response.
            return tuple(cohort.key for cohort in ordered), "unblock"

        expired = tuple(
            cohort.key
            for cohort in ordered
            if cohort.key[0] != "current"
            and now >= cohort.last_arrived_at + self.batch_wait_seconds
        )
        if expired:
            return expired, "deadline"
        return None

    def _current_deadline(self, cohort: _PendingCohort) -> float:
        if len(self.worker_ids) <= 1 or not cohort.workers:
            wait_seconds = self.batch_wait_seconds
        else:
            wait_seconds = max(
                self.batch_wait_seconds * self.current_wait_multiplier,
                self.current_gather_max_seconds,
            )
        return cohort.first_arrived_at + wait_seconds

    @staticmethod
    def _with_compatible_scripted(
        cohorts: Sequence[_PendingCohort],
        current: _PendingCohort,
    ) -> tuple[_CohortKey, ...]:
        temperature = current.key[2]
        return (
            current.key,
            *(
                cohort.key
                for cohort in cohorts
                if cohort.key[0] == "scripted" and cohort.key[2] == temperature
            ),
        )

    def _ordered(self) -> tuple[_PendingCohort, ...]:
        return tuple(
            sorted(
                self._pending.values(),
                key=lambda cohort: (cohort.first_arrived_at, cohort.key),
            )
        )

    def _pop(
        self,
        keys: Sequence[_CohortKey],
    ) -> tuple[NativeProcessBrokerRequest, ...]:
        selected = sorted(
            (self._pending.pop(key) for key in keys),
            key=lambda cohort: (cohort.first_arrived_at, cohort.key),
        )
        return tuple(request for cohort in selected for request in cohort.requests)


def _request_rows(request: NativeProcessBrokerRequest) -> int:
    if isinstance(request, StatelessInferenceRequest):
        return len(request.rows)
    return request.shared_batch.batch_size


def _native_workers(
    requests: Sequence[NativeProcessBrokerRequest],
) -> set[int]:
    return {
        request.worker_index
        for request in requests
        if not isinstance(request, StatelessInferenceRequest)
    }


def _cohort_key(request: NativeProcessBrokerRequest) -> _CohortKey:
    if isinstance(request, StatelessInferenceRequest):
        return ("scripted", str(request.policy_route), request.temperature)
    if isinstance(request, NativeProcessHistoricalRequest):
        return ("historical", request.artifact_sha256, None)
    return (request.route_kind, request.artifact_sha256, request.temperature)


__all__ = [
    "NativeProcessBatchScheduler",
    "NativeProcessBatchSelection",
    "NativeProcessBrokerRequest",
]
