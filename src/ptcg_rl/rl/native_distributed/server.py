"""Dedicated coordinator I/O thread for continuous worker heartbeats."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import TypeVar, cast

from ptcg_rl.rl.native_distributed.artifact import (
    EncodedBfloat16RolloutArtifact,
)
from ptcg_rl.rl.native_distributed.contracts import (
    NativeCollectionWindow,
    NativeCollectionWindowReceipt,
)
from ptcg_rl.rl.native_distributed.coordinator import (
    CommitCallback,
    NativeCollectionCoordinator,
    NativeCoordinatorError,
)
from ptcg_rl.rl.native_distributed.service import NativeCoordinatorService
from ptcg_rl.rl.native_distributed.transport import (
    NativeCoordinatorSockets,
    NativeDistributedEndpoints,
)
from ptcg_rl.rl.stateless_collection import (
    StatelessAssignedGame,
    StatelessCollectionResult,
)
from ptcg_rl.rl.stateless_quota_assignments import StatelessQuotaAssignmentPlan

ResultT = TypeVar("ResultT")


class NativeCoordinatorServer:
    """Keep all ZMQ socket access on one thread while the H200 learns."""

    def __init__(
        self,
        coordinator: NativeCollectionCoordinator,
        endpoints: NativeDistributedEndpoints,
        *,
        io_threads: int,
        high_watermark: int,
        control_poll_interval_seconds: float,
        status_path: Path | None,
        status_interval_seconds: float,
    ) -> None:
        """Start the I/O owner and fail synchronously on any bind error."""
        self.coordinator = coordinator
        self._endpoints = endpoints
        self._io_threads = io_threads
        self._high_watermark = high_watermark
        self._control_poll_interval_seconds = control_poll_interval_seconds
        self._status_path = status_path
        self._status_interval_seconds = status_interval_seconds
        self._commands: queue.Queue[
            tuple[
                Callable[[NativeCoordinatorService], object],
                Future[object],
            ]
        ] = queue.Queue()
        self._stop = threading.Event()
        self._started = threading.Event()
        self._service: NativeCoordinatorService | None = None
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="native-distributed-coordinator",
            daemon=False,
        )
        self._thread.start()
        if not self._started.wait(timeout=30.0):
            raise TimeoutError("native coordinator I/O thread did not start")
        if self._startup_error is not None:
            raise RuntimeError("native coordinator failed to bind") from (
                self._startup_error
            )

    def begin_window(
        self,
        window: NativeCollectionWindow,
        *,
        artifacts: Sequence[EncodedBfloat16RolloutArtifact],
        assignment_pool: Sequence[StatelessAssignedGame] | StatelessQuotaAssignmentPlan,
        learner_clocked_primary_only: bool = False,
    ) -> NativeCollectionWindow:
        """Open one immutable window on the socket-owning thread."""
        return self._execute(
            lambda service: service.begin_window(
                window,
                artifacts=artifacts,
                assignment_pool=assignment_pool,
                learner_clocked_primary_only=learner_clocked_primary_only,
            )
        )

    def wait_for_workers(
        self,
        worker_ids: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> None:
        """Wait while the I/O thread continues registration and heartbeats."""
        required = set(worker_ids)
        if not required or timeout_seconds <= 0.0:
            raise ValueError("native worker wait requires IDs and a timeout")
        deadline = time.monotonic() + timeout_seconds
        while True:
            connected = set(
                self.coordinator.status(now_unix_ns=time.time_ns()).connected_workers
            )
            if required <= connected:
                return
            if time.monotonic() >= deadline:
                missing = ", ".join(sorted(required - connected))
                raise TimeoutError(
                    f"native distributed worker READY timeout: {missing}"
                )
            self._raise_thread_error()
            time.sleep(self._control_poll_interval_seconds)

    def wait_for_worker_quorum(
        self,
        worker_ids: Sequence[str],
        *,
        minimum_workers: int,
        timeout_seconds: float,
    ) -> None:
        """Wait for any configured minimum without requiring optional workers."""
        eligible = set(worker_ids)
        if (
            not eligible
            or minimum_workers <= 0
            or minimum_workers > len(eligible)
            or timeout_seconds <= 0.0
        ):
            raise ValueError("native worker quorum wait is invalid")
        deadline = time.monotonic() + timeout_seconds
        while True:
            connected = set(
                self.coordinator.status(now_unix_ns=time.time_ns()).connected_workers
            )
            if len(eligible & connected) >= minimum_workers:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "native distributed minimum worker READY quorum timed out: "
                    f"ready={len(eligible & connected)} required={minimum_workers}"
                )
            self._raise_thread_error()
            time.sleep(self._control_poll_interval_seconds)

    def wait_until_complete(self, *, timeout_seconds: float) -> None:
        """Wait for the global target and speculative-tail cutoff."""
        if timeout_seconds <= 0.0:
            raise ValueError("native collection timeout must be positive")
        deadline = time.monotonic() + timeout_seconds
        while not self.coordinator.ready_to_commit():
            exhausted = self.coordinator.exhausted_lease_ids()
            if exhausted:
                raise NativeCoordinatorError(
                    "native collection shard exhausted retries: "
                    + ", ".join(self.coordinator.exhausted_lease_diagnostics())
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("native distributed collection window timed out")
            self._raise_thread_error()
            time.sleep(self._control_poll_interval_seconds)

    def collection_result(self) -> StatelessCollectionResult:
        """Build the complete array window without occupying the socket thread.

        Collection is already immutable once ``wait_until_complete`` returns.
        The service method snapshots coordinator-owned parts under its lock, so
        report merging can safely run on the learner thread while heartbeats
        continue to receive prompt replies.
        """
        return self._require_service().collection_result()

    def unused_assignments(self) -> tuple[StatelessAssignedGame, ...]:
        """Return controller leases that no worker consumed."""
        return self._execute(lambda service: service.unused_assignments())

    def accepted_assignment_ids(self) -> frozenset[str]:
        """Return accepted assignment identities that reached engine start."""
        return self._execute(lambda service: service.accepted_assignment_ids())

    def assignment_planning_seconds(self) -> float:
        """Return measured assignment validation and lazy-plan setup time."""
        return self._execute(lambda service: service.last_assignment_planning_seconds)

    def commit(
        self,
        callback: CommitCallback,
    ) -> NativeCollectionWindowReceipt:
        """Commit on the learner thread so controller writer affinity is kept."""
        self._raise_thread_error()
        return self.coordinator.commit(
            callback,
            now_unix_ns=time.time_ns(),
        )

    def wait_for_receipt_delivery(self, *, timeout_seconds: float) -> None:
        """Wait until every topology worker has observed final settlement."""
        if timeout_seconds <= 0.0:
            raise ValueError("native receipt timeout must be positive")
        deadline = time.monotonic() + timeout_seconds
        while not self._execute(lambda service: service.receipt_delivery_complete()):
            if time.monotonic() >= deadline:
                raise TimeoutError("native distributed receipt delivery timed out")
            self._raise_thread_error()
            time.sleep(self._control_poll_interval_seconds)

    def abort(self, *, reason: str) -> NativeCollectionWindowReceipt:
        """Abort a half-window and release all received frames."""
        self._raise_thread_error()
        return self.coordinator.abort(
            reason=reason,
            now_unix_ns=time.time_ns(),
        )

    def release_parts(self) -> None:
        """Release borrowed ndarray frames after optimizer consumption."""
        self.coordinator.release_parts()

    def close(self) -> None:
        """Stop polling and close all sockets on their owning thread."""
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            raise TimeoutError("native coordinator I/O thread did not stop")
        self._raise_thread_error()

    def _execute(
        self,
        callback: Callable[[NativeCoordinatorService], ResultT],
    ) -> ResultT:
        self._raise_thread_error()
        future: Future[object] = Future()
        self._commands.put((callback, future))
        return cast(ResultT, future.result(timeout=30.0))

    def _run(self) -> None:
        try:
            with NativeCoordinatorSockets(
                self._endpoints,
                io_threads=self._io_threads,
                high_watermark=self._high_watermark,
            ) as sockets:
                service = NativeCoordinatorService(
                    self.coordinator,
                    sockets,
                    control_poll_interval_seconds=(self._control_poll_interval_seconds),
                    status_path=self._status_path,
                    status_interval_seconds=self._status_interval_seconds,
                )
                self._service = service
                self._started.set()
                while not self._stop.is_set():
                    self._drain_commands(service)
                    service.serve_once()
                self._drain_commands(service)
                service.close()
        except BaseException as exc:
            self._startup_error = exc
            self._started.set()
            self._fail_commands(exc)

    def _drain_commands(self, service: NativeCoordinatorService) -> None:
        while True:
            try:
                callback, future = self._commands.get_nowait()
            except queue.Empty:
                return
            if future.cancelled():
                continue
            try:
                future.set_result(callback(service))
            except BaseException as exc:
                future.set_exception(exc)

    def _fail_commands(self, error: BaseException) -> None:
        while True:
            try:
                _callback, future = self._commands.get_nowait()
            except queue.Empty:
                return
            if not future.done():
                future.set_exception(error)

    def _require_service(self) -> NativeCoordinatorService:
        self._raise_thread_error()
        if self._service is None:
            raise RuntimeError("native coordinator service is unavailable")
        return self._service

    def _raise_thread_error(self) -> None:
        if self._startup_error is not None:
            raise RuntimeError("native coordinator I/O thread failed") from (
                self._startup_error
            )


__all__ = ["NativeCoordinatorServer"]
