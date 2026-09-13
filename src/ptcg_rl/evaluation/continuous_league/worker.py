"""Resource-aware worker loop for the native continuous-league match ABI."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import msgpack

from ptcg_rl.evaluation.continuous_league.models import (
    ContinuousLeagueConfig,
    LeaseRequest,
    MatchLease,
    MatchResult,
    WorkerHeartbeat,
    validate_safe_id,
)
from ptcg_rl.evaluation.continuous_league.native_protocol import (
    NativeMatchProtocolError,
    encode_frame,
    read_frame_fd,
)
from ptcg_rl.evaluation.continuous_league.resources import probe_resources
from ptcg_rl.training.source_identity import resolve_training_source_identity


class NativeMatchClient(Protocol):
    """Persistent executor surface shared by serial and multiplexed workers."""

    def request(
        self,
        lease: MatchLease,
        *,
        gpu_index: int | None,
        timeout_seconds: float,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


class LeagueWorker:
    """Finish small native batches, then yield when shared resources become busy."""

    def __init__(
        self,
        config: ContinuousLeagueConfig,
        *,
        worker_id: str,
        repo_root: Path,
        coordinator_url: str | None = None,
        gpu_index: int | None = None,
        ignore_resource_load: bool = False,
        executor: NativeMatchClient | None = None,
    ) -> None:
        self.config = config
        self.worker_id = validate_safe_id(worker_id)
        self.repo_root = repo_root.resolve()
        self.coordinator_url = (
            coordinator_url
            or f"http://{config.coordinator_host}:{config.coordinator_port}"
        ).rstrip("/")
        if gpu_index is not None and gpu_index < 0:
            raise ValueError("gpu_index must be non-negative")
        self.gpu_index = gpu_index
        self.ignore_resource_load = ignore_resource_load
        self.source_commit = _source_commit(self.repo_root)
        self.games_completed = 0
        self.errors = 0
        self.quiet_probes = 0
        self._executor = executor or _NativeMatchProcess(
            command=config.native_match_command,
            repo_root=self.repo_root,
        )

    def run(
        self,
        *,
        once: bool = False,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Poll forever; an absent executor or coordinator is an idle condition."""
        try:
            while stop_event is None or not stop_event.is_set():
                snapshot = probe_resources(
                    self.config.resources,
                    gpu_index=self.gpu_index,
                )
                self.quiet_probes = self.quiet_probes + 1 if snapshot.quiet else 0
                admitted = self.ignore_resource_load or (
                    self.quiet_probes >= self.config.resources.quiet_probes
                )
                snapshot = snapshot.model_copy(update={"quiet": admitted})
                if not admitted:
                    self._executor.close()
                heartbeat = self._heartbeat(snapshot)
                if not self.config.native_match_command:
                    self._post_best_effort(
                        "/worker/v1/heartbeat", heartbeat.model_dump_json()
                    )
                    if once:
                        return
                    _wait(stop_event, self.config.resources.probe_seconds)
                    continue
                lease = self._lease(heartbeat)
                if lease is None:
                    if once:
                        return
                    _wait(stop_event, self.config.resources.probe_seconds)
                    continue
                self._post_best_effort(
                    "/worker/v1/heartbeat",
                    self._heartbeat(
                        snapshot, current_match_id=lease.match_id
                    ).model_dump_json(),
                )
                result = self._execute(lease)
                self._submit_result(result)
                self.games_completed += 1
                if once:
                    return
        finally:
            self._executor.close()

    def _lease(self, heartbeat: WorkerHeartbeat) -> MatchLease | None:
        request = LeaseRequest(heartbeat=heartbeat)
        try:
            payload = self._post("/worker/v1/lease", request.model_dump_json())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"coordinator rejected worker lease: {detail}"
            ) from error
        except (TimeoutError, urllib.error.URLError):
            return None
        if payload is None:
            return None
        lease = MatchLease.model_validate(payload)
        if (
            lease.runtime_fingerprint != self.config.runtime_fingerprint
            or lease.belief_fingerprint != self.config.belief_fingerprint
        ):
            raise RuntimeError("coordinator lease protocol differs from worker profile")
        return lease

    def _execute(self, lease: MatchLease) -> MatchResult:
        started = datetime.now(UTC)
        gpu_index: int | None = None
        if lease.requires_cuda:
            snapshot = probe_resources(
                self.config.resources,
                gpu_index=self.gpu_index,
            )
            if not snapshot.cuda_available or snapshot.gpu_index is None:
                return _infrastructure_result(
                    lease, self.worker_id, started, "CUDA became unavailable"
                )
            gpu_index = snapshot.gpu_index
        try:
            unpacked = self._executor.request(
                lease,
                gpu_index=gpu_index,
                timeout_seconds=self.config.scheduling.lease_seconds,
            )
            payload = dict(unpacked)
            telemetry = payload.pop("telemetry", {})
            payload.update(
                {
                    "match_id": lease.match_id,
                    "worker_id": self.worker_id,
                    "telemetry_msgpack": msgpack.packb(telemetry, use_bin_type=True),
                }
            )
            return MatchResult.model_validate(payload)
        except (
            NativeMatchProtocolError,
            OSError,
            subprocess.SubprocessError,
            TimeoutError,
            TypeError,
            ValueError,
        ) as error:
            self._executor.close()
            self.errors += 1
            return _infrastructure_result(
                lease,
                self.worker_id,
                started,
                f"{type(error).__name__}: {error}",
            )

    def _submit_result(self, result: MatchResult) -> None:
        """Retry an immutable result until the idempotent coordinator records it."""
        body = result.model_dump_json()
        while True:
            try:
                self._post("/worker/v1/results", body)
                return
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"coordinator rejected match result: {detail}"
                ) from error
            except (TimeoutError, urllib.error.URLError):
                time.sleep(self.config.resources.probe_seconds)

    def _heartbeat(
        self, resources: Any, *, current_match_id: str | None = None
    ) -> WorkerHeartbeat:
        return WorkerHeartbeat(
            worker_id=self.worker_id,
            hostname=validate_safe_id(socket.gethostname()),
            source_commit=self.source_commit,
            runtime_fingerprint=self.config.runtime_fingerprint,
            belief_fingerprint=self.config.belief_fingerprint,
            resources=resources,
            current_match_id=current_match_id,
            games_completed=self.games_completed,
            errors=self.errors,
        )

    def _post(self, path: str, body: str) -> Any:
        request = urllib.request.Request(
            f"{self.coordinator_url}{path}",
            data=body.encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=max(5.0, self.config.resources.probe_seconds),
        ) as response:
            raw = response.read()
        return None if not raw or raw == b"null" else json.loads(raw)

    def _post_best_effort(self, path: str, body: str) -> None:
        with suppress(TimeoutError, urllib.error.URLError):
            self._post(path, body)


def _infrastructure_result(
    lease: MatchLease,
    worker_id: str,
    started: datetime,
    detail: str,
) -> MatchResult:
    finished = datetime.now(UTC)
    return MatchResult(
        match_id=lease.match_id,
        worker_id=worker_id,
        outcome="unresolved",
        terminal_reason="infrastructure_error",
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        steps=0,
        duration_seconds=(finished - started).total_seconds(),
        telemetry_msgpack=msgpack.packb({"error": detail}, use_bin_type=True),
    )


def _source_commit(repo_root: Path) -> str:
    return resolve_training_source_identity(repo_root).source_git_commit


def _wait(stop_event: threading.Event | None, seconds: float) -> None:
    if stop_event is None:
        time.sleep(seconds)
    else:
        stop_event.wait(seconds)


class _NativeMatchProcess:
    """Persistent framed subprocess, rebound when CUDA ownership changes."""

    def __init__(self, *, command: tuple[str, ...], repo_root: Path) -> None:
        self.command = command
        self.repo_root = repo_root
        self.process: subprocess.Popen[bytes] | None = None
        self.gpu_index: int | None = None

    def request(
        self,
        lease: MatchLease,
        *,
        gpu_index: int | None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Send one lease and wait for exactly one framed response."""
        self._ensure_process(gpu_index=gpu_index)
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("native match process has no usable pipes")
        if process.poll() is not None:
            raise subprocess.SubprocessError(
                f"native match process exited with code {process.returncode}"
            )
        process.stdin.write(encode_frame(lease.model_dump(mode="json")))
        process.stdin.flush()
        return read_frame_fd(
            process.stdout.fileno(),
            timeout_seconds=timeout_seconds,
        )

    def close(self) -> None:
        """Close stdin for a clean server shutdown, then bound escalation."""
        process = self.process
        self.process = None
        self.gpu_index = None
        if process is None:
            return
        if process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        try:
            process.wait(timeout=5.0)
            return
        except subprocess.TimeoutExpired:
            process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)

    def _ensure_process(self, *, gpu_index: int | None) -> None:
        process = self.process
        if (
            process is not None
            and process.poll() is None
            and self.gpu_index == gpu_index
        ):
            return
        self.close()
        if not self.command:
            raise RuntimeError("native match command is empty")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = (
            "" if gpu_index is None else str(gpu_index)
        )
        self.process = subprocess.Popen(
            list(self.command),
            cwd=self.repo_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=environment,
            bufsize=0,
        )
        self.gpu_index = gpu_index


__all__ = ["LeagueWorker", "NativeMatchClient"]
