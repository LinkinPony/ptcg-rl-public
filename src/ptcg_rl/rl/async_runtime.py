"""Async actor/learner process supervision helpers."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, field_validator


class ManagedProcess(Protocol):
    """Subset of ``multiprocessing.Process`` used by the async supervisor."""

    @property
    def exitcode(self) -> int | None:
        """Return process exit code, or ``None`` while it is still running."""

    def start(self) -> None:
        """Start the process."""

    def join(self, timeout: float | None = None) -> None:
        """Join the process, optionally bounded by timeout."""

    def is_alive(self) -> bool:
        """Return whether the process is still alive."""

    def terminate(self) -> None:
        """Terminate the process."""

    def kill(self) -> None:
        """Unconditionally kill a process that did not terminate."""


ProcessFactory = Callable[[], ManagedProcess]
Sleeper = Callable[[float], None]


@dataclass(frozen=True)
class ActorRecycleRequest:
    """Request replacement of specific live actor process slots."""

    actor_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject ambiguous or invalid actor slot requests."""
        if any(index < 0 for index in self.actor_indices):
            raise ValueError("actor recycle indices must be non-negative")
        if len(set(self.actor_indices)) != len(self.actor_indices):
            raise ValueError("actor recycle indices must be unique")


PollCallback = Callable[
    [int, Sequence[ManagedProcess], ManagedProcess],
    ActorRecycleRequest | None,
]


class AsyncActorLearnerSupervisorConfig(BaseModel):
    """Config for supervising async actor and learner worker processes."""

    model_config = ConfigDict(extra="forbid")

    poll_interval_seconds: float = 1.0
    actor_restart_limit: int = 1
    process_join_timeout_seconds: float = 5.0
    stop_actor_on_learner_exit: bool = True
    max_polls: int | None = None
    actor_heartbeat_timeout_seconds: float | None = 300.0

    @field_validator("poll_interval_seconds")
    @classmethod
    def valid_poll_interval(cls, value: float) -> float:
        """Reject invalid poll intervals."""
        if value <= 0.0:
            raise ValueError("poll_interval_seconds must be positive")
        return value

    @field_validator("actor_restart_limit")
    @classmethod
    def valid_actor_restart_limit(cls, value: int) -> int:
        """Reject invalid actor restart limits."""
        if value < 0:
            raise ValueError("actor_restart_limit must be non-negative")
        return value

    @field_validator("process_join_timeout_seconds")
    @classmethod
    def valid_join_timeout(cls, value: float) -> float:
        """Reject invalid join timeouts."""
        if value <= 0.0:
            raise ValueError("process_join_timeout_seconds must be positive")
        return value

    @field_validator("max_polls")
    @classmethod
    def valid_optional_max_polls(cls, value: int | None) -> int | None:
        """Reject invalid optional poll limits."""
        if value is not None and value <= 0:
            raise ValueError("max_polls must be positive when set")
        return value

    @field_validator("actor_heartbeat_timeout_seconds")
    @classmethod
    def valid_optional_actor_heartbeat_timeout(
        cls,
        value: float | None,
    ) -> float | None:
        """Reject a non-positive actor heartbeat watchdog timeout."""
        if value is not None and value <= 0.0:
            raise ValueError("actor heartbeat timeout must be positive when set")
        return value


@dataclass(frozen=True)
class AsyncActorLearnerSupervisorResult:
    """Summary from a completed actor/learner supervision run."""

    # Restarts consume the failure budget; clean actor work-cycle completions do
    # not.  Keeping the counters separate prevents learner startup latency from
    # looking like actor instability.
    actor_restarts: int
    polls: int
    actor_exitcode: int | None
    learner_exitcode: int | None
    actor_restarts_by_actor: tuple[int, ...] = ()
    actor_exitcodes: tuple[int | None, ...] = ()
    actor_recycles: int = 0
    actor_recycles_by_actor: tuple[int, ...] = ()


class AsyncActorLearnerSupervisorError(RuntimeError):
    """Base class for async actor/learner supervision failures."""


class ActorProcessFailedError(AsyncActorLearnerSupervisorError):
    """Raised when the actor process fails too many times."""


class LearnerProcessFailedError(AsyncActorLearnerSupervisorError):
    """Raised when the learner process exits with a non-zero code."""


class AsyncSupervisorTimeoutError(AsyncActorLearnerSupervisorError):
    """Raised when supervision reaches a configured poll limit."""


def supervise_actor_learner(
    *,
    actor_factory: ProcessFactory,
    learner_factory: ProcessFactory,
    required_processes: Sequence[tuple[str, ManagedProcess]] = (),
    config: AsyncActorLearnerSupervisorConfig | None = None,
    sleeper: Sleeper = time.sleep,
    poll_callback: PollCallback | None = None,
) -> AsyncActorLearnerSupervisorResult:
    """Start and supervise actor and learner processes.

    A clean actor exit completes one bounded collection cycle and is recycled
    while the learner is still running. Non-zero actor exits consume the
    ``actor_restart_limit`` failure budget. A learner non-zero exit fails the
    run.
    """
    return supervise_actor_group_learner(
        actor_factories=(actor_factory,),
        learner_factory=learner_factory,
        required_processes=required_processes,
        config=config,
        sleeper=sleeper,
        poll_callback=poll_callback,
    )


def supervise_actor_group_learner(
    *,
    actor_factories: Sequence[ProcessFactory],
    learner_factory: ProcessFactory,
    required_processes: Sequence[tuple[str, ManagedProcess]] = (),
    config: AsyncActorLearnerSupervisorConfig | None = None,
    sleeper: Sleeper = time.sleep,
    poll_callback: PollCallback | None = None,
) -> AsyncActorLearnerSupervisorResult:
    """Start and supervise a learner with independently restarted actors.

    Required processes are already-started services owned by the caller. An
    unexpected exit fails supervision so actors and the learner cannot remain
    blocked on a service that is no longer available. The caller remains
    responsible for stopping those services during its outer cleanup.
    """
    cfg = config or AsyncActorLearnerSupervisorConfig()
    actors: list[ManagedProcess] = []
    learner: ManagedProcess | None = None
    actor_restarts = [0] * len(actor_factories)
    actor_recycles = [0] * len(actor_factories)
    polls = 0

    try:
        # Start the consumer process before producers; training-specific code
        # performs the stronger service-readiness handshake. Track every
        # process before ``start`` so partial start failure is cleaned up.
        learner = learner_factory()
        learner.start()
        for actor_factory in actor_factories:
            actor = actor_factory()
            actors.append(actor)
            actor.start()
        while cfg.max_polls is None or polls < cfg.max_polls:
            polls += 1
            for actor in actors:
                _refresh_process(actor)
            _refresh_process(learner)
            for _name, process in required_processes:
                _refresh_process(process)
            recycle_request: ActorRecycleRequest | None = None
            if poll_callback is not None:
                recycle_request = poll_callback(polls, tuple(actors), learner)

            for name, process in required_processes:
                if process.exitcode is not None:
                    raise AsyncActorLearnerSupervisorError(
                        f"required process {name!r} exited unexpectedly with code "
                        f"{process.exitcode}"
                    )

            if learner.exitcode is not None:
                if learner.exitcode != 0:
                    raise LearnerProcessFailedError(
                        f"learner exited with code {learner.exitcode}"
                    )
                if cfg.stop_actor_on_learner_exit:
                    stop_managed_processes(
                        actors,
                        join_timeout_seconds=cfg.process_join_timeout_seconds,
                    )
                return AsyncActorLearnerSupervisorResult(
                    actor_restarts=sum(actor_restarts),
                    polls=polls,
                    actor_exitcode=(actors[0].exitcode if len(actors) == 1 else None),
                    learner_exitcode=learner.exitcode,
                    actor_restarts_by_actor=tuple(actor_restarts),
                    actor_exitcodes=tuple(actor.exitcode for actor in actors),
                    actor_recycles=sum(actor_recycles),
                    actor_recycles_by_actor=tuple(actor_recycles),
                )

            if recycle_request is not None:
                _apply_actor_recycle_request(
                    recycle_request,
                    actors=actors,
                    actor_factories=actor_factories,
                    actor_recycles=actor_recycles,
                    join_timeout_seconds=cfg.process_join_timeout_seconds,
                )

            for actor_index, actor in enumerate(actors):
                if actor.exitcode is None:
                    continue
                if actor.exitcode == 0:
                    actor_recycles[actor_index] += 1
                    replacement = actor_factories[actor_index]()
                    actors[actor_index] = replacement
                    replacement.start()
                    continue
                if actor_restarts[actor_index] >= cfg.actor_restart_limit:
                    raise ActorProcessFailedError(
                        f"actor {actor_index} exited with code {actor.exitcode} "
                        "after exhausting failure restart limit "
                        f"({cfg.actor_restart_limit})"
                    )
                actor_restarts[actor_index] += 1
                replacement = actor_factories[actor_index]()
                actors[actor_index] = replacement
                replacement.start()

            sleeper(cfg.poll_interval_seconds)
        raise AsyncSupervisorTimeoutError(
            f"learner did not exit within {cfg.max_polls} polls"
        )
    except BaseException:
        owned_processes = list(actors)
        if learner is not None:
            owned_processes.append(learner)
        stop_managed_processes(
            owned_processes,
            join_timeout_seconds=cfg.process_join_timeout_seconds,
        )
        raise


def _refresh_process(process: ManagedProcess) -> None:
    process.join(timeout=0.0)


def _apply_actor_recycle_request(
    request: ActorRecycleRequest,
    *,
    actors: list[ManagedProcess],
    actor_factories: Sequence[ProcessFactory],
    actor_recycles: list[int],
    join_timeout_seconds: float,
) -> None:
    """Replace only requested live actor slots without spending restart budget."""
    invalid_indices = tuple(
        index for index in request.actor_indices if index >= len(actors)
    )
    if invalid_indices:
        raise AsyncActorLearnerSupervisorError(
            f"actor recycle index is out of range: {invalid_indices[0]}"
        )

    live_targets: list[tuple[int, ManagedProcess]] = []
    for actor_index in request.actor_indices:
        actor = actors[actor_index]
        _refresh_process(actor)
        if actor.exitcode is not None:
            # Let the normal clean-exit/failure path classify this actor.
            continue
        live_targets.append((actor_index, actor))
    if not live_targets:
        return

    target_processes = tuple(actor for _index, actor in live_targets)
    stop_managed_processes(
        target_processes,
        join_timeout_seconds=join_timeout_seconds,
    )
    survivor_ids = {id(actor) for actor in _alive_processes(target_processes)}
    if survivor_ids:
        survivor_indices = tuple(
            index for index, actor in live_targets if id(actor) in survivor_ids
        )
        raise AsyncActorLearnerSupervisorError(
            "actors survived bounded heartbeat recycle: "
            + ", ".join(str(index) for index in survivor_indices)
        )

    for actor_index, _actor in live_targets:
        replacement = actor_factories[actor_index]()
        actors[actor_index] = replacement
        replacement.start()
        actor_recycles[actor_index] += 1


def stop_managed_processes(
    processes: Sequence[ManagedProcess],
    *,
    join_timeout_seconds: float,
) -> None:
    """Terminate, then kill and reap one owned process group."""
    concrete = tuple(processes)

    # Signal the whole group before joining any one process. A sequential
    # terminate-and-join loop can spend the full timeout on every wedged actor
    # while all later actors continue running and holding shared resources.
    for process in _alive_processes(concrete):
        try:
            process.terminate()
        except (AssertionError, OSError, ValueError):
            # A failed/partial ``start`` has no child process to signal.
            continue
    _join_process_group(
        concrete,
        timeout_seconds=join_timeout_seconds,
    )

    survivors = _alive_processes(concrete)
    for process in survivors:
        try:
            process.kill()
        except (AssertionError, OSError, ValueError):
            continue
    _join_process_group(
        survivors,
        timeout_seconds=join_timeout_seconds,
    )

    remaining = _alive_processes(survivors)
    if remaining:
        warnings.warn(
            f"{len(remaining)} managed process(es) survived terminate and kill",
            RuntimeWarning,
            stacklevel=2,
        )


def _stop_processes(
    processes: Sequence[ManagedProcess],
    *,
    join_timeout_seconds: float,
) -> None:
    """Backward-compatible private alias for local callers and tests."""
    stop_managed_processes(
        processes,
        join_timeout_seconds=join_timeout_seconds,
    )


def _alive_processes(
    processes: Sequence[ManagedProcess],
) -> tuple[ManagedProcess, ...]:
    """Return live, successfully started children without masking cleanup."""
    alive: list[ManagedProcess] = []
    for process in processes:
        try:
            if process.is_alive():
                alive.append(process)
        except (AssertionError, ValueError):
            # ``multiprocessing.Process.is_alive`` may reject a child whose
            # ``start`` failed before its process handle was created.
            continue
    return tuple(alive)


def _join_process_group(
    processes: Sequence[ManagedProcess],
    *,
    timeout_seconds: float,
) -> None:
    """Reap a process group within one shared wall-clock timeout."""
    deadline = time.monotonic() + timeout_seconds
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.join(timeout=remaining)
        except (AssertionError, ValueError):
            # There is nothing to join after a failed/partial ``start``.
            continue
