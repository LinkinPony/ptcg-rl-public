"""Bounded fair scheduling for heterogeneous inference requests."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

_PURPOSE_ORDER = ("planner_behavior", "behavior", "teacher")


class SchedulableInferenceRequest(Protocol):
    """Read-only request fields needed by the scheduling policy."""

    @property
    def purpose(self) -> str:
        """Return the request purpose."""

    @property
    def request_type(self) -> str:
        """Return the inference stage."""

    @property
    def batch_size(self) -> int:
        """Return the number of decision rows."""

    @property
    def deadline_monotonic(self) -> float:
        """Return the absolute deadline, or zero for ordinary behavior."""

    @property
    def server_received_at(self) -> float:
        """Return the server-side arrival timestamp."""


class InferenceSchedulingConfig(Protocol):
    """Configuration surface shared with the inference server model."""

    @property
    def max_batch(self) -> int:
        """Return the general stage row cap."""

    @property
    def max_planner_proposal_rows(self) -> int:
        """Return the proposal-stage row cap."""

    @property
    def max_root_information_rows(self) -> int:
        """Return the semantic-value row cap."""

    @property
    def purpose_schedule_weights(self) -> tuple[int, int, int]:
        """Return planner, behavior, and teacher shares."""

    @property
    def purpose_aging_seconds(self) -> float:
        """Return the wait threshold for aging rotation."""


RequestT = TypeVar("RequestT", bound=SchedulableInferenceRequest)


def fair_schedule_requests(
    requests: Sequence[RequestT],
    *,
    config: InferenceSchedulingConfig,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[RequestT, ...]:
    """Interleave purposes by bounded shares and age-sensitive rotation.

    Planner and teacher work remains earliest-deadline-first within its
    purpose. Weighted rounds prevent a sustained action-critical stream from
    hiding behavior or auxiliary work already admitted to the bounded drain.
    """
    unknown = {request.purpose for request in requests} - set(_PURPOSE_ORDER)
    if unknown:
        raise ValueError(f"unknown inference request purposes: {sorted(unknown)}")
    weights = dict(zip(_PURPOSE_ORDER, config.purpose_schedule_weights, strict=True))
    pending: dict[str, deque[RequestT]] = {}
    for purpose in _PURPOSE_ORDER:
        purpose_requests = [
            request for request in requests if request.purpose == purpose
        ]
        purpose_requests.sort(key=_purpose_request_order_key)
        pending[purpose] = deque(purpose_requests)

    scheduled: list[RequestT] = []
    while any(pending.values()):
        active = tuple(purpose for purpose in _PURPOSE_ORDER if pending[purpose])
        now = clock()
        aged = tuple(
            purpose
            for purpose in active
            if now - pending[purpose][0].server_received_at
            >= config.purpose_aging_seconds
        )
        first = (
            min(
                aged,
                key=lambda purpose: pending[purpose][0].server_received_at,
            )
            if aged
            else active[0]
        )
        offset = _PURPOSE_ORDER.index(first)
        round_order = _PURPOSE_ORDER[offset:] + _PURPOSE_ORDER[:offset]
        for purpose in round_order:
            for _index in range(weights[purpose]):
                if not pending[purpose]:
                    break
                scheduled.append(pending[purpose].popleft())
    return tuple(scheduled)


def scheduled_stage_batches(
    requests: Sequence[RequestT],
    *,
    config: InferenceSchedulingConfig,
) -> tuple[tuple[RequestT, ...], ...]:
    """Preserve fair order while batching adjacent equal-purpose stages."""
    batches: list[tuple[RequestT, ...]] = []
    index = 0
    while index < len(requests):
        first = requests[index]
        cap = inference_stage_row_cap(first.request_type, config=config)
        if first.batch_size > cap:
            raise RuntimeError(
                f"one {first.request_type} request exceeds its stage row cap"
            )
        group = [first]
        rows = first.batch_size
        index += 1
        while (
            index < len(requests)
            and requests[index].request_type == first.request_type
            and requests[index].purpose == first.purpose
            and rows + requests[index].batch_size <= cap
        ):
            item = requests[index]
            group.append(item)
            rows += item.batch_size
            index += 1
        batches.append(tuple(group))
    return tuple(batches)


def inference_stage_row_cap(
    request_type: str,
    *,
    config: InferenceSchedulingConfig,
) -> int:
    """Return the immutable row cap for one inference stage."""
    if request_type == "root_information_value":
        return config.max_root_information_rows
    if request_type == "planner_proposals":
        return config.max_planner_proposal_rows
    return config.max_batch


def _purpose_request_order_key(
    request: SchedulableInferenceRequest,
) -> tuple[float, int, float]:
    if request.purpose in ("planner_behavior", "teacher"):
        return (
            request.deadline_monotonic,
            _stage_priority(request.request_type),
            request.server_received_at,
        )
    return (
        request.server_received_at,
        _stage_priority(request.request_type),
        request.server_received_at,
    )


def _stage_priority(request_type: str) -> int:
    order = {
        "planner_context_release": 0,
        "recurrent_release": 1,
        "decode": 2,
        "planner_proposals": 3,
        "planner_candidates": 4,
        "root_information_value": 5,
        "value": 6,
    }
    try:
        return order[request_type]
    except KeyError as exc:
        raise ValueError(f"unknown inference request type: {request_type}") from exc


__all__ = [
    "InferenceSchedulingConfig",
    "SchedulableInferenceRequest",
    "fair_schedule_requests",
    "inference_stage_row_cap",
    "scheduled_stage_batches",
]
