"""Fixed-bank coordinator for high-throughput native collection.

This module deliberately contains only the scheduling core.  It does not know
about Hydra, curriculum construction, policy artifacts, or trajectory storage.
Those concerns are supplied through callbacks so the coordinator can be used
as a drop-in alternative to the general ready-arena graph.

Physical arenas are permanently grouped into a fixed ring of banks. Ready banks
may share a policy cohort, optionally after one bounded near-ready wait, then
each bank independently owns its concurrent engine advances. An unfinished
engine barrier is never crossed only to enlarge a policy cohort.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ThreadPoolExecutor,
    wait,
)
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Protocol, TypeVar

ArenaT = TypeVar("ArenaT")
CallbackArenaT = TypeVar("CallbackArenaT", contravariant=True)
DispatchT = TypeVar("DispatchT")
PolicyT = TypeVar("PolicyT")
SubmittedPolicyT = TypeVar("SubmittedPolicyT")
PreparedT = TypeVar("PreparedT")
EngineResultT = TypeVar("EngineResultT")


@dataclass(frozen=True, slots=True)
class NativeBankLayout:
    """Permanent fixed-ring physical arena assignment."""

    banks: tuple[tuple[int, ...], ...] = ((0, 1), (2, 3))

    def __post_init__(self) -> None:
        if len(self.banks) < 2 or any(not bank for bank in self.banks):
            raise ValueError("native bank layout requires at least two non-empty banks")
        flattened = tuple(index for bank in self.banks for index in bank)
        if len(set(flattened)) != len(flattened) or set(flattened) != set(
            range(len(flattened))
        ):
            raise ValueError(
                "native bank layout must cover physical arenas contiguously"
            )

    @property
    def arena_count(self) -> int:
        """Return the exact physical arena count owned by this layout."""
        return sum(len(bank) for bank in self.banks)


class BankedNativeCollectionCallbacks(
    Protocol[
        CallbackArenaT,
        PolicyT,
        SubmittedPolicyT,
        DispatchT,
        PreparedT,
        EngineResultT,
    ]
):
    """Native lifecycle operations owned by the concrete collector adapter."""

    def is_live(self, arena: CallbackArenaT) -> bool:
        """Return whether an arena still has games to collect."""

    def prepare_policy(
        self,
        arenas: tuple[CallbackArenaT, ...],
    ) -> PolicyT:
        """Freeze one policy group and launch only host-side preparation."""

    def submit_prepared_policy(
        self,
        prepared: PolicyT,
        *,
        overlap: Callable[[], None],
    ) -> SubmittedPolicyT:
        """Submit one policy group without waiting for host-visible actions."""

    def finish_policy(
        self,
        submitted: SubmittedPolicyT,
    ) -> Sequence[DispatchT]:
        """Wait one submitted policy group and return its host-visible actions."""

    def abort_policy(self, prepared: PolicyT) -> None:
        """Drain an unconsumed policy preparation before arena teardown."""

    def abort_submitted_policy(self, submitted: SubmittedPolicyT) -> None:
        """Drain one submitted group wave before arena teardown."""

    def prepare_engine(
        self,
        arena: CallbackArenaT,
        dispatch: DispatchT,
    ) -> PreparedT | None:
        """Freeze one engine input, or return None when its rows stay parked."""

    def run_engine(self, prepared: PreparedT) -> EngineResultT:
        """Run only the thread-safe native engine portion."""

    def finish_engine(
        self,
        arena: CallbackArenaT,
        prepared: PreparedT,
        result: EngineResultT,
    ) -> None:
        """Commit one completed engine result on the coordinator thread."""

    def abort_engine(self, prepared: PreparedT) -> None:
        """Release an engine input that could not be committed."""


@dataclass(frozen=True, slots=True)
class BankedNativeCollectionReport:
    """Scheduling-only telemetry returned by the fixed-bank coordinator."""

    policy_waves: int
    policy_group_member_banks: int
    policy_group_max_size: int
    policy_coalescing_misses: int
    policy_cohort_wait_events: int
    policy_cohort_wait_harvests: int
    policy_cohort_wait_seconds: float
    engine_barriers: int
    engine_steps: int
    parked_policy_waves: int
    bank_policy_waves: tuple[int, ...]
    bank_engine_barriers: tuple[int, ...]
    bank_wait_seconds: tuple[float, ...]
    policy_prefetches: int
    policy_overlap_seconds: float
    gpu_feed_gap_seconds: float
    gpu_feed_gap_events: int
    elapsed_seconds: float


class _BankPhase(Enum):
    READY = "ready"
    POLICY = "policy"
    GPU = "gpu"
    ENGINE = "engine"
    FINISHED = "finished"


@dataclass(slots=True)
class _EngineTicket(Generic[ArenaT, PreparedT, EngineResultT]):
    arena_index: int
    arena: ArenaT
    prepared: PreparedT
    future: Future[EngineResultT]
    committed: bool = False


@dataclass(slots=True)
class _BankState(Generic[ArenaT, PreparedT, EngineResultT]):
    bank_index: int
    arena_indices: tuple[int, ...]
    phase: _BankPhase = _BankPhase.READY
    tickets: list[_EngineTicket[ArenaT, PreparedT, EngineResultT]] = field(
        default_factory=list
    )


@dataclass(slots=True)
class _PolicyGroup(
    Generic[ArenaT, PolicyT, SubmittedPolicyT, PreparedT, EngineResultT]
):
    """One prepared or submitted policy wave shared by one or two banks."""

    members: tuple[_BankState[ArenaT, PreparedT, EngineResultT], ...]
    active_by_bank: tuple[tuple[tuple[int, ArenaT], ...], ...]
    policy: PolicyT
    submitted: SubmittedPolicyT | None = None
    completed: bool = False
    aborted: bool = False


def run_banked_native_collection_core(
    arenas: Sequence[ArenaT],
    callbacks: BankedNativeCollectionCallbacks[
        ArenaT,
        PolicyT,
        SubmittedPolicyT,
        DispatchT,
        PreparedT,
        EngineResultT,
    ],
    *,
    layout: NativeBankLayout | None = None,
    engine_executor: Executor | None = None,
    coalesce_two_bank_layout: bool = False,
    policy_group_bank_limit: int = 2,
    policy_cohort_wait_seconds: float = 0.0,
) -> BankedNativeCollectionReport:
    """Collect physical arenas through a continuously fed fixed-bank ring.

    With at least three banks, immediately ready banks share one policy wave
    up to ``policy_group_bank_limit``. A two-bank layout does the same only when
    ``coalesce_two_bank_layout`` explicitly opts into its larger activation
    footprint. A positive ``policy_cohort_wait_seconds`` briefly harvests
    near-ready peer banks before submitting a partial group. Engine barriers
    remain bank-local after a shared policy wave so one faster bank can still
    independently feed a later wave.

    """
    if layout is None:
        layout = NativeBankLayout()
    if policy_group_bank_limit <= 0:
        raise ValueError("policy group bank limit must be positive")
    if policy_cohort_wait_seconds < 0.0:
        raise ValueError("policy cohort wait must be non-negative")
    if len(arenas) != layout.arena_count:
        raise ValueError("banked native collection arenas differ from the fixed layout")

    started_at = time.perf_counter()
    banks: list[_BankState[ArenaT, PreparedT, EngineResultT]] = [
        _BankState[ArenaT, PreparedT, EngineResultT](
            bank_index=bank_index,
            arena_indices=arena_indices,
        )
        for bank_index, arena_indices in enumerate(layout.banks)
    ]
    policy_waves = [0 for _bank in banks]
    engine_barriers = [0 for _bank in banks]
    wait_seconds = [0.0 for _bank in banks]
    prepared_groups: deque[
        _PolicyGroup[
            ArenaT,
            PolicyT,
            SubmittedPolicyT,
            PreparedT,
            EngineResultT,
        ]
    ] = deque()
    submitted_groups: deque[
        _PolicyGroup[
            ArenaT,
            PolicyT,
            SubmittedPolicyT,
            PreparedT,
            EngineResultT,
        ]
    ] = deque()
    owned_groups: dict[
        int,
        _PolicyGroup[
            ArenaT,
            PolicyT,
            SubmittedPolicyT,
            PreparedT,
            EngineResultT,
        ],
    ] = {}
    policy_wave_count = 0
    policy_group_member_banks = 0
    policy_group_max_size = 0
    policy_coalescing_misses = 0
    policy_cohort_wait_events = 0
    policy_cohort_wait_harvests = 0
    policy_cohort_wait_total_seconds = 0.0
    engine_steps = 0
    parked_policy_waves = 0
    policy_prefetches = 0
    policy_overlap_seconds = 0.0
    gpu_feed_gap_seconds = 0.0
    gpu_feed_gap_events = 0
    gpu_feed_gap_started_at: float | None = None
    owned_executor = (
        ThreadPoolExecutor(
            max_workers=len(arenas),
            thread_name_prefix="native-bank-engine",
        )
        if engine_executor is None
        else None
    )
    executor = owned_executor if owned_executor is not None else engine_executor
    if executor is None:
        raise RuntimeError("banked native collection has no engine executor")

    selection_cursor = 0

    def immediate_ready_banks() -> tuple[
        tuple[
            _BankState[ArenaT, PreparedT, EngineResultT],
            tuple[tuple[int, ArenaT], ...],
        ],
        ...,
    ]:
        """Harvest only completed barriers and return live banks in ring order."""
        ready = []
        for offset in range(len(banks)):
            bank = banks[(selection_cursor + offset) % len(banks)]
            if bank.phase is _BankPhase.ENGINE:
                if not _bank_barrier_ready(bank):
                    continue
                wait_seconds[bank.bank_index] += _finish_bank_barrier(
                    bank,
                    callbacks,
                )
                engine_barriers[bank.bank_index] += 1
                bank.phase = _BankPhase.READY
            if bank.phase is not _BankPhase.READY:
                continue
            active = tuple(
                (arena_index, arenas[arena_index])
                for arena_index in bank.arena_indices
                if callbacks.is_live(arenas[arena_index])
            )
            if not active:
                bank.phase = _BankPhase.FINISHED
                continue
            ready.append((bank, active))
        return tuple(ready)

    def prepare_one_policy_group(
        *,
        prefetched: bool,
    ) -> (
        _PolicyGroup[
            ArenaT,
            PolicyT,
            SubmittedPolicyT,
            PreparedT,
            EngineResultT,
        ]
        | None
    ):
        """Prepare one bounded bank group without crossing a live barrier."""
        nonlocal policy_coalescing_misses, policy_prefetches, selection_cursor
        nonlocal policy_cohort_wait_events, policy_cohort_wait_harvests
        nonlocal policy_cohort_wait_total_seconds
        ready = immediate_ready_banks()
        if not ready:
            return None
        maximum_group_size = 1
        if len(banks) >= 3 or coalesce_two_bank_layout:
            maximum_group_size = min(len(banks), policy_group_bank_limit)
        if (
            policy_cohort_wait_seconds > 0.0
            and len(ready) < maximum_group_size
            and any(bank.phase is _BankPhase.ENGINE for bank in banks)
        ):
            wait_started_at = time.perf_counter()
            deadline = wait_started_at + policy_cohort_wait_seconds
            initial_ready = len(ready)
            while len(ready) < maximum_group_size:
                futures = tuple(
                    ticket.future
                    for bank in banks
                    if bank.phase is _BankPhase.ENGINE
                    for ticket in bank.tickets
                    if not ticket.future.done()
                )
                remaining = deadline - time.perf_counter()
                if not futures or remaining <= 0.0:
                    break
                completed, _pending = wait(
                    futures,
                    timeout=remaining,
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    break
                ready = immediate_ready_banks()
            policy_cohort_wait_total_seconds += (
                time.perf_counter() - wait_started_at
            )
            policy_cohort_wait_events += 1
            if len(ready) > initial_ready:
                policy_cohort_wait_harvests += 1
        group_size = 1
        if len(banks) >= 3 or (len(ready) >= 2 and coalesce_two_bank_layout):
            group_size = min(len(ready), policy_group_bank_limit)
        if len(banks) >= 3 and group_size == 1:
            policy_coalescing_misses += 1
        selected_in_ring = ready[:group_size]
        selection_cursor = (selected_in_ring[-1][0].bank_index + 1) % len(banks)
        # Canonical member order makes dispatch slicing independent of the
        # current fairness cursor.
        selected = tuple(sorted(selected_in_ring, key=lambda item: item[0].bank_index))
        active_by_bank = tuple(active for _bank, active in selected)
        policy = callbacks.prepare_policy(
            tuple(arena for active in active_by_bank for _arena_index, arena in active)
        )
        group = _PolicyGroup[
            ArenaT,
            PolicyT,
            SubmittedPolicyT,
            PreparedT,
            EngineResultT,
        ](
            members=tuple(bank for bank, _active in selected),
            active_by_bank=active_by_bank,
            policy=policy,
        )
        for bank in group.members:
            if bank.phase is not _BankPhase.READY:
                raise RuntimeError("native policy group lost a ready member")
            bank.phase = _BankPhase.POLICY
        owned_groups[id(group)] = group
        prepared_groups.append(group)
        if prefetched:
            policy_prefetches += 1
        return group

    def prepare_all_immediate_groups(*, prefetched: bool) -> None:
        """Freeze every currently ready bank without waiting for another."""
        while prepare_one_policy_group(prefetched=prefetched) is not None:
            pass

    def submit_policy_group(
        group: _PolicyGroup[
            ArenaT,
            PolicyT,
            SubmittedPolicyT,
            PreparedT,
            EngineResultT,
        ],
    ) -> None:
        """Submit one shared wave and use its tail for ready peer preparation."""
        nonlocal gpu_feed_gap_events, gpu_feed_gap_seconds
        nonlocal gpu_feed_gap_started_at, policy_group_max_size
        nonlocal policy_group_member_banks, policy_overlap_seconds
        nonlocal policy_wave_count
        if not group.members or any(
            bank.phase is not _BankPhase.POLICY for bank in group.members
        ):
            raise RuntimeError("native policy group is not prepared")
        overlap_called = False
        overlap_seconds = 0.0

        def prepare_peer_groups() -> None:
            nonlocal overlap_called, overlap_seconds, policy_overlap_seconds
            if overlap_called:
                raise RuntimeError("native policy overlap ran more than once")
            overlap_called = True
            overlap_started_at = time.perf_counter()
            try:
                prepare_all_immediate_groups(prefetched=True)
            finally:
                overlap_seconds = time.perf_counter() - overlap_started_at
                policy_overlap_seconds += overlap_seconds

        submitted = callbacks.submit_prepared_policy(
            group.policy,
            overlap=prepare_peer_groups,
        )
        group.submitted = submitted
        if not overlap_called:
            raise RuntimeError("native policy submission omitted its overlap callback")
        for bank in group.members:
            bank.phase = _BankPhase.GPU
            policy_waves[bank.bank_index] += 1
        submitted_groups.append(group)
        group_size = len(group.members)
        policy_wave_count += 1
        policy_group_member_banks += group_size
        policy_group_max_size = max(policy_group_max_size, group_size)
        if gpu_feed_gap_started_at is not None:
            gpu_feed_gap_seconds += max(
                time.perf_counter() - gpu_feed_gap_started_at - overlap_seconds,
                0.0,
            )
            gpu_feed_gap_events += 1
            gpu_feed_gap_started_at = None

    def submit_all_prepared_groups() -> None:
        while prepared_groups:
            submit_policy_group(prepared_groups.popleft())

    def finish_next_policy_group() -> None:
        """Publish one FIFO policy group and return engines to bank ownership."""
        nonlocal engine_steps, gpu_feed_gap_started_at, parked_policy_waves
        if not submitted_groups:
            raise RuntimeError("native policy queue is empty")
        group = submitted_groups[0]
        submitted = group.submitted
        if submitted is None:
            raise RuntimeError("native GPU group lost its submission")
        dispatches = tuple(callbacks.finish_policy(submitted))
        expected_dispatches = sum(len(active) for active in group.active_by_bank)
        if len(dispatches) != expected_dispatches:
            raise RuntimeError(
                "banked policy dispatch count differs from live arena count"
            )
        submitted_groups.popleft()
        if not submitted_groups and not prepared_groups:
            gpu_feed_gap_started_at = time.perf_counter()

        dispatch_offset = 0
        for bank, active in zip(
            group.members,
            group.active_by_bank,
            strict=True,
        ):
            if bank.phase is not _BankPhase.GPU:
                raise RuntimeError("native GPU group lost a member")
            bank.phase = _BankPhase.READY
            bank.tickets = []
            bank_dispatches = dispatches[
                dispatch_offset : dispatch_offset + len(active)
            ]
            dispatch_offset += len(active)
            for (arena_index, arena), dispatch in zip(
                active,
                bank_dispatches,
                strict=True,
            ):
                prepared = callbacks.prepare_engine(arena, dispatch)
                if prepared is None:
                    continue
                try:
                    future = executor.submit(callbacks.run_engine, prepared)
                except BaseException:
                    with suppress(BaseException):
                        callbacks.abort_engine(prepared)
                    raise
                bank.tickets.append(
                    _EngineTicket(
                        arena_index=arena_index,
                        arena=arena,
                        prepared=prepared,
                        future=future,
                    )
                )
            engine_steps += len(bank.tickets)
            if bank.tickets:
                bank.phase = _BankPhase.ENGINE
            else:
                parked_policy_waves += 1
        if dispatch_offset != len(dispatches):
            raise RuntimeError("native policy group dispatch slicing diverged")
        # Keep the submitted policy owner live until every dispatch has either
        # transferred to an engine ticket or parked. If preparation/submission
        # fails midway, group rollback still owns all provisional sequence rows;
        # engine-ticket rollback is deliberately idempotent with it.
        group.completed = True
        group.submitted = None
        del owned_groups[id(group)]

    def wait_for_one_engine_bank() -> None:
        """Cross one deterministic barrier only when CUDA has no queued work."""
        nonlocal selection_cursor
        for offset in range(len(banks)):
            bank = banks[(selection_cursor + offset) % len(banks)]
            if bank.phase is not _BankPhase.ENGINE:
                continue
            wait_seconds[bank.bank_index] += _finish_bank_barrier(
                bank,
                callbacks,
            )
            engine_barriers[bank.bank_index] += 1
            bank.phase = _BankPhase.READY
            selection_cursor = bank.bank_index
            return
        if any(bank.phase is not _BankPhase.FINISHED for bank in banks):
            raise RuntimeError("native scheduler has work but no runnable owner")

    try:
        while True:
            if submitted_groups:
                overlap_started_at = time.perf_counter()
                prefetch_count_before = policy_prefetches
                try:
                    prepare_all_immediate_groups(prefetched=True)
                finally:
                    if policy_prefetches > prefetch_count_before:
                        policy_overlap_seconds += (
                            time.perf_counter() - overlap_started_at
                        )
                submit_all_prepared_groups()
                finish_next_policy_group()
                continue

            if prepared_groups:
                submit_all_prepared_groups()
                continue

            if prepare_one_policy_group(prefetched=False) is not None:
                # CUDA has no queued work. Submit this first immediately; its
                # overlap callback may prepare additional ready banks.
                submit_all_prepared_groups()
                continue

            if all(bank.phase is _BankPhase.FINISHED for bank in banks):
                break
            wait_for_one_engine_bank()
    except BaseException:
        for group in reversed(tuple(owned_groups.values())):
            _abort_policy_group(group, callbacks)
        owned_groups.clear()
        for bank in banks:
            _abort_bank(bank, callbacks)
        raise
    finally:
        if owned_executor is not None:
            owned_executor.shutdown(wait=True, cancel_futures=True)

    return BankedNativeCollectionReport(
        policy_waves=policy_wave_count,
        policy_group_member_banks=policy_group_member_banks,
        policy_group_max_size=policy_group_max_size,
        policy_coalescing_misses=policy_coalescing_misses,
        policy_cohort_wait_events=policy_cohort_wait_events,
        policy_cohort_wait_harvests=policy_cohort_wait_harvests,
        policy_cohort_wait_seconds=policy_cohort_wait_total_seconds,
        engine_barriers=sum(engine_barriers),
        engine_steps=engine_steps,
        parked_policy_waves=parked_policy_waves,
        bank_policy_waves=tuple(policy_waves),
        bank_engine_barriers=tuple(engine_barriers),
        bank_wait_seconds=tuple(wait_seconds),
        policy_prefetches=policy_prefetches,
        policy_overlap_seconds=policy_overlap_seconds,
        gpu_feed_gap_seconds=gpu_feed_gap_seconds,
        gpu_feed_gap_events=gpu_feed_gap_events,
        elapsed_seconds=max(time.perf_counter() - started_at, 0.0),
    )


def _bank_barrier_ready(
    bank: _BankState[ArenaT, PreparedT, EngineResultT],
) -> bool:
    """Return whether a bank barrier can be crossed without waiting."""
    return (
        bank.phase is _BankPhase.ENGINE
        and bool(bank.tickets)
        and all(ticket.future.done() for ticket in bank.tickets)
    )


def _finish_bank_barrier(
    bank: _BankState[ArenaT, PreparedT, EngineResultT],
    callbacks: BankedNativeCollectionCallbacks[
        ArenaT,
        PolicyT,
        SubmittedPolicyT,
        DispatchT,
        PreparedT,
        EngineResultT,
    ],
) -> float:
    """Resolve a whole bank before committing either member."""
    if bank.phase is not _BankPhase.ENGINE or not bank.tickets:
        raise RuntimeError("native bank crossed an empty engine barrier")
    started_at = time.perf_counter()
    try:
        results = tuple(ticket.future.result() for ticket in bank.tickets)
        waited = time.perf_counter() - started_at
        for ticket, result in zip(bank.tickets, results, strict=True):
            callbacks.finish_engine(ticket.arena, ticket.prepared, result)
            ticket.committed = True
    except BaseException:
        _abort_bank(bank, callbacks)
        raise
    bank.tickets.clear()
    return waited


def _abort_policy_group(
    group: _PolicyGroup[
        ArenaT,
        PolicyT,
        SubmittedPolicyT,
        PreparedT,
        EngineResultT,
    ],
    callbacks: BankedNativeCollectionCallbacks[
        ArenaT,
        PolicyT,
        SubmittedPolicyT,
        DispatchT,
        PreparedT,
        EngineResultT,
    ],
) -> None:
    """Abort a shared policy owner exactly once."""
    if group.completed or group.aborted:
        return
    group.aborted = True
    submitted = group.submitted
    group.submitted = None
    if submitted is not None:
        with suppress(BaseException):
            callbacks.abort_submitted_policy(submitted)
        return
    with suppress(BaseException):
        callbacks.abort_policy(group.policy)


def _abort_bank(
    bank: _BankState[ArenaT, PreparedT, EngineResultT],
    callbacks: BankedNativeCollectionCallbacks[
        ArenaT,
        PolicyT,
        SubmittedPolicyT,
        DispatchT,
        PreparedT,
        EngineResultT,
    ],
) -> None:
    """Wait out native writers, then release every uncommitted input."""
    tickets = tuple(bank.tickets)
    bank.tickets.clear()
    for ticket in tickets:
        if ticket.committed:
            continue
        if not ticket.future.cancel():
            with suppress(BaseException):
                ticket.future.result()
        with suppress(BaseException):
            callbacks.abort_engine(ticket.prepared)


__all__ = [
    "BankedNativeCollectionCallbacks",
    "BankedNativeCollectionReport",
    "NativeBankLayout",
    "run_banked_native_collection_core",
]
