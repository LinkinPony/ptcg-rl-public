"""Multiplex league leases over one CUDA-owning native-match subprocess."""

from __future__ import annotations

import os
import subprocess
import threading
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

from ptcg_rl.evaluation.continuous_league.models import MatchLease
from ptcg_rl.evaluation.continuous_league.native_protocol import (
    NativeMatchProtocolError,
    encode_frame,
    read_frame,
)


@dataclass
class _PendingResponse:
    event: threading.Event
    payload: dict[str, Any] | None = None
    error: BaseException | None = None


class SharedNativeMatchProcess:
    """One framed child with concurrent lanes and a shared checkpoint cache."""

    def __init__(
        self,
        *,
        command: tuple[str, ...],
        repo_root: Path,
        concurrency: int,
        gpu_index: int | None,
    ) -> None:
        if concurrency <= 1:
            raise ValueError("shared native match concurrency must exceed one")
        if gpu_index is not None and gpu_index < 0:
            raise ValueError("gpu_index must be non-negative")
        self.command = command
        self.repo_root = repo_root.resolve()
        self.concurrency = concurrency
        self.gpu_index = gpu_index
        self._state_lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._pending: dict[str, _PendingResponse] = {}
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._closed = False

    def slot(self) -> SharedNativeMatchSlot:
        """Return one no-ownership client view for a worker thread."""
        return SharedNativeMatchSlot(self)

    def request(
        self,
        lease: MatchLease,
        *,
        gpu_index: int | None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Submit one lease and await its out-of-order correlated response."""
        if gpu_index != self.gpu_index:
            raise ValueError("shared executor CUDA ownership changed")
        if timeout_seconds <= 0.0:
            raise ValueError("native match response timeout must be positive")
        process = self._ensure_process()
        request_id = uuid.uuid4().hex
        pending = _PendingResponse(event=threading.Event())
        with self._state_lock:
            if self._closed or self._process is not process:
                raise RuntimeError("shared native match process is unavailable")
            self._pending[request_id] = pending
        try:
            if process.stdin is None:
                raise RuntimeError("shared native match process has no stdin")
            with self._write_lock:
                process.stdin.write(
                    encode_frame(
                        {
                            "request_id": request_id,
                            "lease": lease.model_dump(mode="json"),
                        }
                    )
                )
                process.stdin.flush()
        except BaseException:
            with self._state_lock:
                self._pending.pop(request_id, None)
            raise
        if not pending.event.wait(timeout_seconds):
            with self._state_lock:
                self._pending.pop(request_id, None)
            raise TimeoutError("shared native match executor response timed out")
        if pending.error is not None:
            raise pending.error
        if pending.payload is None:
            raise NativeMatchProtocolError("shared native match response is absent")
        return pending.payload

    def close(self) -> None:
        """Stop accepting requests and bound shutdown of the shared child."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            reader = self._reader
            self._process = None
            self._reader = None
            self._fail_pending_locked(RuntimeError("shared executor closed"))
        if process is None:
            return
        if process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=5.0)

    def _ensure_process(self) -> subprocess.Popen[bytes]:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("shared native match executor is closed")
            process = self._process
            if process is not None and process.poll() is None:
                return process
            if not self.command:
                raise RuntimeError("native match command is empty")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = (
                "" if self.gpu_index is None else str(self.gpu_index)
            )
            process = subprocess.Popen(
                [*self.command, "--concurrency", str(self.concurrency)],
                cwd=self.repo_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                env=environment,
                bufsize=0,
            )
            self._process = process
            reader = threading.Thread(
                target=self._read_responses,
                args=(process,),
                name="continuous-league-native-reader",
                daemon=True,
            )
            self._reader = reader
            reader.start()
            return process

    def _read_responses(self, process: subprocess.Popen[bytes]) -> None:
        error: BaseException = NativeMatchProtocolError(
            "shared native match executor closed its response pipe"
        )
        try:
            if process.stdout is None:
                raise RuntimeError("shared native match process has no stdout")
            while True:
                envelope = read_frame(cast(BinaryIO, process.stdout))
                if envelope is None:
                    break
                request_id = str(envelope.get("request_id", ""))
                response = envelope.get("response")
                if not request_id or not isinstance(response, Mapping):
                    raise NativeMatchProtocolError(
                        "shared native match response envelope is invalid"
                    )
                with self._state_lock:
                    pending = self._pending.pop(request_id, None)
                if pending is None:
                    continue
                pending.payload = {str(key): value for key, value in response.items()}
                pending.event.set()
        except BaseException as caught:
            error = caught
        finally:
            with self._state_lock:
                if self._process is process:
                    self._process = None
                    self._reader = None
                self._fail_pending_locked(error)

    def _fail_pending_locked(self, error: BaseException) -> None:
        pending = tuple(self._pending.values())
        self._pending.clear()
        for item in pending:
            item.error = error
            item.event.set()


class SharedNativeMatchSlot:
    """Non-owning client surface used by one independently leased worker slot."""

    def __init__(self, owner: SharedNativeMatchProcess) -> None:
        self._owner = owner

    def request(
        self,
        lease: MatchLease,
        *,
        gpu_index: int | None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return self._owner.request(
            lease,
            gpu_index=gpu_index,
            timeout_seconds=timeout_seconds,
        )

    def close(self) -> None:
        """Leave process ownership with the supervising worker pool."""


__all__ = ["SharedNativeMatchProcess", "SharedNativeMatchSlot"]
