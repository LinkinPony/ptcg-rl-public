"""Deadline-bounded subprocess execution with parent-owned service calls."""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ptcg_rl.rl.bounded_worker_runtime import (
    BrokerCall,
    BrokerReply,
    Ready,
    Shutdown,
    StartupFailure,
    Work,
    WorkerBroker,
    WorkerHandler,
    WorkerInitializer,
    WorkResult,
    bounded_message,
    receive_message,
    send_message,
    worker_main,
)


@dataclass(frozen=True)
class BoundedWorkerStats:
    """Lifecycle diagnostics for one isolated worker."""

    starts: int
    restarts: int
    startup_failures: int
    hard_timeouts: int
    crashes: int
    orderly_closes: int
    forced_terminations: int


class BoundedWorkerError(RuntimeError):
    """Base error for isolated execution failures."""


class BoundedWorkerTimeoutError(TimeoutError, BoundedWorkerError):
    """The hard work deadline expired and the worker was terminated."""


class BoundedWorkerUnavailableError(BoundedWorkerError):
    """The worker could not be started or exited unexpectedly."""


class BoundedWorkerWarmingError(BoundedWorkerUnavailableError):
    """A replacement worker is warming without blocking the caller."""


class BoundedWorkerStartupError(BoundedWorkerUnavailableError):
    """A deterministic child initializer failure prevents safe retries."""


class RemoteWorkerError(BoundedWorkerError):
    """The worker raised an exception while handling one request."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type


class BoundedProcessWorker:
    """Run native/process-global work in one killable serial subprocess.

    The child owns all process-global native state. Parent-owned resources such
    as inference response queues are reached only through ``broker_handler`` in
    :meth:`execute`, so a child can never consume those queues itself. The hard
    deadline covers child/native work and IPC. The parent broker must provide
    its own bounded call (the online teacher uses RemoteInferencePolicy's queue
    timeout). A timeout reaps the whole child before another one is started.
    """

    def __init__(
        self,
        *,
        initializer: WorkerInitializer,
        handler: WorkerHandler,
        initializer_payload: Any,
        startup_timeout_seconds: float,
        terminate_grace_seconds: float = 0.05,
        kill_reap_timeout_seconds: float = 0.5,
        max_startup_attempts: int = 3,
    ) -> None:
        if startup_timeout_seconds <= 0.0:
            raise ValueError("worker startup timeout must be positive")
        if terminate_grace_seconds <= 0.0:
            raise ValueError("worker terminate grace must be positive")
        if kill_reap_timeout_seconds <= 0.0:
            raise ValueError("worker kill/reap timeout must be positive")
        if max_startup_attempts <= 0:
            raise ValueError("worker startup attempt cap must be positive")
        self._initializer = initializer
        self._handler = handler
        self._initializer_payload = initializer_payload
        self._startup_timeout_seconds = float(startup_timeout_seconds)
        self._terminate_grace_seconds = float(terminate_grace_seconds)
        self._kill_reap_timeout_seconds = float(kill_reap_timeout_seconds)
        self._max_startup_attempts = int(max_startup_attempts)
        self._process: Any | None = None
        self._channel: socket.socket | None = None
        self._next_work_id = 0
        self._ever_started = False
        self._closed = False
        self._starts = 0
        self._restarts = 0
        self._startup_failures = 0
        self._hard_timeouts = 0
        self._crashes = 0
        self._orderly_closes = 0
        self._forced_terminations = 0
        self._ready = threading.Event()
        self._prepare_lock = threading.RLock()
        self._warming_thread: threading.Thread | None = None
        self._warming_release: threading.Event | None = None
        self._permanent_failure: BoundedWorkerUnavailableError | None = None

    @property
    def ready(self) -> bool:
        """Whether a fully initialized worker is ready without waiting."""
        with self._prepare_lock:
            if not self._ready.is_set():
                return False
            process = self._process
            if process is not None and process.is_alive():
                return True
            if process is not None:
                self._crashes += 1
                self._reap_stopped_process()
            else:
                self._ready.clear()
            return False

    @property
    def process_id(self) -> int | None:
        """Return the current child PID for lifecycle diagnostics."""
        process = self._process
        return None if process is None else process.pid

    @property
    def process_exit_code(self) -> int | None:
        """Return the current child exit code when it has stopped."""
        process = self._process
        return None if process is None else process.exitcode

    @property
    def stats(self) -> BoundedWorkerStats:
        """Return immutable lifecycle counters."""
        return BoundedWorkerStats(
            starts=self._starts,
            restarts=self._restarts,
            startup_failures=self._startup_failures,
            hard_timeouts=self._hard_timeouts,
            crashes=self._crashes,
            orderly_closes=self._orderly_closes,
            forced_terminations=self._forced_terminations,
        )

    def prepare(self, *, deadline: float | None = None) -> None:
        """Start or deterministically restart a ready, idle subprocess.

        The first eager startup may use the configured startup timeout. Callers
        restarting during a timed operation pass its absolute deadline so
        process warmup cannot escape that operation's cap.
        """
        with self._prepare_lock:
            if self._closed:
                raise BoundedWorkerUnavailableError("bounded worker is closed")
            if self._permanent_failure is not None:
                raise self._permanent_failure
            if self._process is not None:
                if self._process.is_alive() and self._channel is not None:
                    self._ready.set()
                    return
                self._reap_stopped_process()
                self._crashes += 1

            if deadline is not None and time.perf_counter() >= deadline:
                raise BoundedWorkerTimeoutError(
                    "bounded worker restart deadline expired"
                )
            restarting = self._ever_started
            parent_channel, child_channel = socket.socketpair()
            context = mp.get_context("spawn")
            process = context.Process(
                target=worker_main,
                args=(
                    child_channel,
                    self._initializer,
                    self._handler,
                    self._initializer_payload,
                    os.getpid(),
                ),
                daemon=True,
            )
            try:
                process.start()
            except BaseException:
                parent_channel.close()
                child_channel.close()
                self._startup_failures += 1
                raise
            child_channel.close()
            if self._closed:
                parent_channel.close()
                process.terminate()
                process.join(timeout=self._terminate_grace_seconds)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=self._kill_reap_timeout_seconds)
                if not process.is_alive():
                    process.close()
                raise BoundedWorkerUnavailableError("bounded worker is closed")
            self._process = process
            self._channel = parent_channel
            self._starts += 1
            self._restarts += int(restarting)
            self._ever_started = True
            startup_deadline = time.perf_counter() + self._startup_timeout_seconds
            if deadline is not None:
                startup_deadline = min(startup_deadline, deadline)
            try:
                message = receive_message(parent_channel, deadline=startup_deadline)
            except TimeoutError as exc:
                self._startup_failures += 1
                self._hard_timeouts += int(restarting)
                self._terminate_and_reap()
                raise BoundedWorkerTimeoutError(
                    "bounded worker startup deadline expired"
                ) from exc
            except BaseException:
                self._startup_failures += 1
                self._terminate_and_reap()
                raise
            if isinstance(message, Ready):
                if self._closed:
                    self._terminate_and_reap()
                    raise BoundedWorkerUnavailableError("bounded worker is closed")
                self._ready.set()
                return
            self._startup_failures += 1
            self._terminate_and_reap()
            if isinstance(message, StartupFailure):
                failure = BoundedWorkerStartupError(
                    f"worker startup failed: {message.error_type}: {message.message}"
                )
                self._permanent_failure = failure
                raise failure
            raise BoundedWorkerUnavailableError(
                "bounded worker sent an invalid startup reply"
            )

    def restart_in_background(self) -> None:
        """Warm a replacement without blocking an actor step."""
        if self._closed or self.ready:
            return
        if self._permanent_failure is not None:
            raise self._permanent_failure
        thread = self._warming_thread
        if thread is not None and thread.is_alive():
            return
        release = threading.Event()
        self._warming_release = release
        self._warming_thread = threading.Thread(
            target=self._background_prepare,
            args=(release,),
            name="engine-teacher-worker-warmup",
            daemon=True,
        )
        self._warming_thread.start()

    def execute(
        self,
        payload: Any,
        *,
        deadline: float,
        broker_handler: Callable[[Any], Any],
        minimum_broker_seconds: float = 0.0,
    ) -> Any:
        """Execute one item, killing child work at the absolute deadline."""
        if minimum_broker_seconds < 0.0:
            raise ValueError("minimum broker time must be non-negative")
        if not self.ready:
            if self._permanent_failure is not None:
                raise self._permanent_failure
            self.restart_in_background()
            raise BoundedWorkerWarmingError("bounded worker is warming")
        process: Any = self._require_process()
        channel = self._require_channel()
        work_id = self._next_work_id
        self._next_work_id += 1
        try:
            send_message(
                channel,
                Work(work_id=work_id, payload=payload, deadline=float(deadline)),
                deadline=deadline,
            )
            while True:
                if time.perf_counter() >= deadline:
                    raise BoundedWorkerTimeoutError("bounded worker deadline expired")
                message = receive_message(channel, deadline=deadline)
                if isinstance(message, BrokerCall):
                    if message.work_id != work_id:
                        raise BoundedWorkerUnavailableError(
                            "bounded worker broker work identity mismatch"
                        )
                    if time.perf_counter() + minimum_broker_seconds >= deadline:
                        reply = BrokerReply(
                            work_id=work_id,
                            call_id=message.call_id,
                            error_type="TimeoutError",
                            message="insufficient broker deadline",
                        )
                    else:
                        reply = self._handle_broker_call(
                            message,
                            broker_handler=broker_handler,
                        )
                    send_message(channel, reply, deadline=deadline)
                    continue
                if not isinstance(message, WorkResult) or message.work_id != work_id:
                    raise BoundedWorkerUnavailableError(
                        "bounded worker returned an invalid result"
                    )
                if message.error_type is not None:
                    raise RemoteWorkerError(message.error_type, message.message)
                if time.perf_counter() >= deadline:
                    raise BoundedWorkerTimeoutError(
                        "bounded worker result arrived late"
                    )
                return message.result
        except TimeoutError as exc:
            self._hard_timeouts += 1
            self._terminate_and_reap()
            self.restart_in_background()
            raise BoundedWorkerTimeoutError(
                "bounded worker deadline expired"
            ) from exc
        except (EOFError, BrokenPipeError, ConnectionError, OSError) as exc:
            self._crashes += 1
            self._terminate_and_reap()
            self.restart_in_background()
            raise BoundedWorkerUnavailableError(
                "bounded worker exited unexpectedly"
            ) from exc
        except BoundedWorkerUnavailableError:
            self._crashes += 1
            self._terminate_and_reap()
            self.restart_in_background()
            raise
        except RemoteWorkerError:
            raise
        except Exception:
            # Any unclassified transport failure can leave the child blocked on
            # a stale broker call. Never reuse a possibly desynchronized channel.
            self._crashes += 1
            self._terminate_and_reap()
            self.restart_in_background()
            raise
        finally:
            current_process = self._process
            if (
                current_process is process
                and current_process is not None
                and not current_process.is_alive()
            ):
                self._reap_stopped_process()

    def close(self) -> None:
        """Orderly-stop an idle child, then force termination if necessary."""
        if self._closed:
            return
        self._closed = True
        # Wait for an in-flight initializer to observe ``_closed`` and reap its
        # child. A ready supervisor thread remains alive until shutdown below,
        # preserving PR_SET_PDEATHSIG parentage during the orderly close.
        acquired = self._prepare_lock.acquire(
            timeout=(self._startup_timeout_seconds + self._kill_reap_timeout_seconds)
        )
        if not acquired:
            self._stop_and_join_supervisor()
            return
        self._prepare_lock.release()
        process = self._process
        channel = self._channel
        if process is None:
            self._stop_and_join_supervisor()
            return
        if process.is_alive() and channel is not None:
            deadline = time.perf_counter() + self._terminate_grace_seconds
            try:
                send_message(channel, Shutdown(), deadline=deadline)
                process.join(timeout=max(1.0, self._terminate_grace_seconds))
            except (OSError, TimeoutError):
                pass
        if process.is_alive():
            self._terminate_and_reap()
            self._stop_and_join_supervisor()
            return
        self._orderly_closes += 1
        self._reap_stopped_process()
        self._stop_and_join_supervisor()

    def _handle_broker_call(
        self,
        message: BrokerCall,
        *,
        broker_handler: Callable[[Any], Any],
    ) -> BrokerReply:
        try:
            result = broker_handler(message.payload)
        except Exception as exc:
            return BrokerReply(
                work_id=message.work_id,
                call_id=message.call_id,
                error_type=type(exc).__name__,
                message=bounded_message(exc),
            )
        return BrokerReply(
            work_id=message.work_id,
            call_id=message.call_id,
            result=result,
        )

    def _terminate_and_reap(self) -> None:
        with self._prepare_lock:
            self._ready.clear()
            process = self._process
            if process is None:
                self._close_channel()
                return
            if process.is_alive():
                self._forced_terminations += 1
                process.terminate()
                process.join(timeout=self._terminate_grace_seconds)
            if process.is_alive():
                process.kill()
                process.join(timeout=self._kill_reap_timeout_seconds)
            if process.is_alive():
                # Never start another native worker while the old process is alive.
                raise BoundedWorkerUnavailableError(
                    "bounded worker could not be killed"
                )
            self._reap_stopped_process()

    def _reap_stopped_process(self) -> None:
        with self._prepare_lock:
            self._ready.clear()
            process = self._process
            if process is not None:
                process.join(timeout=0.0)
                process.close()
            self._process = None
            self._close_channel()
            release = self._warming_release
            if release is not None:
                release.set()

    def _close_channel(self) -> None:
        if self._channel is not None:
            self._channel.close()
        self._channel = None

    def _require_process(self) -> Any:
        if self._process is None:
            raise BoundedWorkerUnavailableError("bounded worker is not started")
        return self._process

    def _require_channel(self) -> socket.socket:
        if self._channel is None:
            raise BoundedWorkerUnavailableError(
                "bounded worker channel is unavailable"
            )
        return self._channel

    def _background_prepare(self, release: threading.Event) -> None:
        current_release = release
        while not self._closed:
            try:
                self.prepare()
            except BoundedWorkerStartupError:
                return
            except Exception:
                if self._startup_failures >= self._max_startup_attempts:
                    with self._prepare_lock:
                        self._permanent_failure = BoundedWorkerUnavailableError(
                            "bounded worker exhausted startup attempts"
                        )
                return
            # Linux PDEATHSIG follows the thread that spawned the child. Keep
            # this event-driven owner alive; a reap wakes exactly one restart.
            current_release.wait()
            if self._closed:
                return
            current_release = threading.Event()
            self._warming_release = current_release

    def _stop_and_join_supervisor(self) -> None:
        release = self._warming_release
        if release is not None:
            release.set()
        thread = self._warming_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._warming_thread = None
        self._warming_release = None


__all__ = [
    "BoundedProcessWorker",
    "BoundedWorkerError",
    "BoundedWorkerStats",
    "BoundedWorkerTimeoutError",
    "BoundedWorkerStartupError",
    "BoundedWorkerUnavailableError",
    "BoundedWorkerWarmingError",
    "RemoteWorkerError",
    "WorkerBroker",
]
