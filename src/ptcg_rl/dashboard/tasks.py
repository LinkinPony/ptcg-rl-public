"""Durable resource-aware queue for typed dashboard tasks."""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ptcg_rl.dashboard.job_models import DashboardSession
from ptcg_rl.dashboard.task_adapters import (
    PreparedTask,
    TaskCommandAdapters,
    task_spec_fingerprint,
)
from ptcg_rl.dashboard.task_catalog import TaskCatalogService
from ptcg_rl.dashboard.task_models import (
    TaskCatalog,
    TaskCreateRequest,
    TaskListPayload,
    TaskReceipt,
)
from ptcg_rl.dashboard.task_resources import (
    process_identity_matches,
    process_start_ticks,
)
from ptcg_rl.dashboard.task_results import TaskResultService
from ptcg_rl.rl.performance_state import atomic_write_json

_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "unknown"})


class DashboardTaskManager:
    """Create, queue, reconcile, cancel, and retry typed task attempts."""

    def __init__(
        self,
        *,
        repo_root: Path,
        enabled: bool,
        refresh_seconds: int = 15,
        request_token: str | None = None,
    ) -> None:
        if refresh_seconds <= 0:
            raise ValueError("dashboard refresh_seconds must be positive")
        self.repo_root = repo_root.resolve()
        self.enabled = enabled
        self.refresh_seconds = refresh_seconds
        self.request_token = (
            request_token
            if enabled and request_token is not None
            else secrets.token_urlsafe(32)
            if enabled
            else None
        )
        self.tasks_root = self.repo_root / "outputs" / "dashboard" / "tasks"
        self.catalog_service = TaskCatalogService(self.repo_root)
        self.adapters = TaskCommandAdapters(self.repo_root, self.catalog_service)
        self.results = TaskResultService(self.repo_root)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._dispatcher: threading.Thread | None = None
        if enabled:
            self._dispatcher = threading.Thread(
                target=self._dispatch_loop,
                name="dashboard-task-dispatcher",
                daemon=True,
            )
            self._dispatcher.start()

    @property
    def session(self) -> DashboardSession:
        """Return action capabilities shared with the workbench."""
        return DashboardSession(
            actions_enabled=self.enabled,
            action_scope="loopback" if self.enabled else "disabled",
            request_token=self.request_token,
            refresh_seconds=self.refresh_seconds,
        )

    def close(self) -> None:
        """Stop only the in-process dispatcher; child tasks keep their receipts."""
        self._closed.set()
        self._wake.set()
        if self._dispatcher is not None:
            self._dispatcher.join(timeout=5.0)

    def catalog(self) -> TaskCatalog:
        """Return task selections and current resource gates."""
        return self.catalog_service.catalog(self._all_receipts())

    def list_tasks(
        self,
        *,
        offset: int,
        limit: int,
        kind: str | None = None,
        state: str | None = None,
    ) -> TaskListPayload:
        """Return filtered paginated task history with live progress."""
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError("task list requires offset >= 0 and 1 <= limit <= 200")
        receipts = list(self._all_receipts())
        if kind is not None:
            receipts = [receipt for receipt in receipts if receipt.kind == kind]
        if state is not None:
            receipts = [receipt for receipt in receipts if receipt.state == state]
        total = len(receipts)
        page = tuple(
            self._with_progress(receipt)
            for receipt in receipts[offset : offset + limit]
        )
        return TaskListPayload(
            tasks=page,
            total=total,
            offset=offset,
            limit=limit,
        )

    def get(self, task_id: str) -> TaskReceipt:
        """Return one reconciled task receipt with projected progress."""
        with self._lock:
            path = self._receipt_path(task_id)
            receipt = self._reconcile(path, _read_receipt(path))
        return self._with_progress(receipt)

    def start(
        self,
        request: TaskCreateRequest,
        *,
        retry_of: str | None = None,
        attempt: int = 1,
    ) -> TaskReceipt:
        """Validate, publish an immutable spec, and enqueue one task."""
        if not self.enabled:
            raise PermissionError("dashboard actions are disabled")
        task_id = uuid.uuid4().hex
        prepared = self.adapters.prepare(request, task_id=task_id)
        task_dir = self.tasks_root / task_id
        task_dir.mkdir(parents=True, exist_ok=False)
        for generated in prepared.files:
            atomic_write_json(task_dir / generated.name, generated.payload)
        request_payload = request.model_dump(mode="json")
        spec_payload = self._spec_payload(
            task_id=task_id,
            request_payload=request_payload,
            prepared=prepared,
        )
        fingerprint = task_spec_fingerprint(spec_payload)
        spec_payload["spec_fingerprint"] = fingerprint
        spec_path = task_dir / "spec.json"
        atomic_write_json(spec_path, spec_payload)
        now = _utc_now()
        existing = self._all_receipts()
        allowed, reason = self.catalog_service.resource_gate.allow(
            prepared.resource_class,
            existing,
        )
        receipt = TaskReceipt(
            task_id=task_id,
            kind=request.kind,
            mode=request.mode,
            label=request.label,
            state="queued",
            resource_class=prepared.resource_class,
            queue_reason=None if allowed else reason,
            created_at_utc=now,
            updated_at_utc=now,
            spec_fingerprint=fingerprint,
            spec_path=_relative(self.repo_root, spec_path),
            output_dir=(
                None
                if prepared.output_dir is None
                else _relative(self.repo_root, prepared.output_dir)
            ),
            status_path=(
                None
                if prepared.status_path is None
                else _relative(self.repo_root, prepared.status_path)
            ),
            log_path=_relative(self.repo_root, task_dir / "task.log"),
            argv=prepared.argv,
            cwd=str(self.repo_root),
            attempt=attempt,
            retry_of=retry_of,
        )
        _write_receipt(task_dir / "receipt.json", receipt)
        self._wake.set()
        return self._with_progress(receipt)

    def retry(self, task_id: str) -> TaskReceipt:
        """Queue a new attempt against the exact same spec and output."""
        if not self.enabled:
            raise PermissionError("dashboard actions are disabled")
        original = self.get(task_id)
        if original.state not in _TERMINAL_STATES:
            raise ValueError("only terminal tasks can be retried")
        new_id = uuid.uuid4().hex
        task_dir = self.tasks_root / new_id
        task_dir.mkdir(parents=True, exist_ok=False)
        original_spec = _read_object(self.repo_root / original.spec_path)
        retry_spec = {
            **original_spec,
            "task_id": new_id,
            "retry_of": original.task_id,
            "original_spec_fingerprint": original.spec_fingerprint,
        }
        spec_path = task_dir / "spec.json"
        atomic_write_json(spec_path, retry_spec)
        now = _utc_now()
        allowed, reason = self.catalog_service.resource_gate.allow(
            original.resource_class,
            self._all_receipts(),
        )
        receipt = original.model_copy(
            update={
                "task_id": new_id,
                "state": "queued",
                "queue_reason": None if allowed else reason,
                "created_at_utc": now,
                "updated_at_utc": now,
                "started_at_utc": None,
                "finished_at_utc": None,
                "spec_path": _relative(self.repo_root, spec_path),
                "log_path": _relative(self.repo_root, task_dir / "task.log"),
                "attempt": original.attempt + 1,
                "retry_of": original.task_id,
                "worker_pid": None,
                "worker_start_ticks": None,
                "exit_code": None,
                "detail": None,
                "progress": None,
            }
        )
        _write_receipt(task_dir / "receipt.json", receipt)
        self._wake.set()
        return self._with_progress(receipt)

    def cancel(self, task_id: str) -> TaskReceipt:
        """Cancel only the exact receipt-owned process group."""
        if not self.enabled:
            raise PermissionError("dashboard actions are disabled")
        with self._lock:
            path = self._receipt_path(task_id)
            receipt = _read_receipt(path)
            if receipt.state == "queued":
                now = _utc_now()
                updated = receipt.model_copy(
                    update={
                        "state": "cancelled",
                        "queue_reason": None,
                        "updated_at_utc": now,
                        "finished_at_utc": now,
                    }
                )
                _write_receipt(path, updated)
                return self._with_progress(updated)
            if receipt.state not in {"starting", "running", "cancelling"}:
                return self._with_progress(receipt)
            pid = receipt.worker_pid
            if pid is None or not process_identity_matches(
                pid, receipt.worker_start_ticks
            ):
                updated = receipt.model_copy(
                    update={
                        "state": "unknown",
                        "updated_at_utc": _utc_now(),
                        "finished_at_utc": _utc_now(),
                        "detail": "worker identity vanished before cancellation",
                    }
                )
                _write_receipt(path, updated)
                return self._with_progress(updated)
            updated = receipt.model_copy(
                update={
                    "state": "cancelling",
                    "updated_at_utc": _utc_now(),
                }
            )
            _write_receipt(path, updated)
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                current = _read_receipt(path)
                if current.state in _TERMINAL_STATES:
                    return self._with_progress(current)
                current = current.model_copy(
                    update={
                        "state": "unknown",
                        "updated_at_utc": _utc_now(),
                        "finished_at_utc": _utc_now(),
                        "detail": "worker vanished during cancellation",
                    }
                )
                _write_receipt(path, current)
                return self._with_progress(current)
            return self._with_progress(updated)

    def _dispatch_loop(self) -> None:
        while not self._closed.is_set():
            # Individual launch failures are recorded by _launch; a transient
            # catalog/proc read must not kill the persistent dispatcher.
            with suppress(Exception):
                self._dispatch_once()
            self._wake.wait(timeout=2.0)
            self._wake.clear()

    def _dispatch_once(self) -> None:
        with self._lock:
            receipts = list(self._all_receipts())
            active = [
                receipt
                for receipt in receipts
                if receipt.state in {"starting", "running", "cancelling"}
            ]
            queued = sorted(
                (receipt for receipt in receipts if receipt.state == "queued"),
                key=lambda receipt: receipt.created_at_utc,
            )
            for receipt in queued:
                allowed, reason = self.catalog_service.resource_gate.allow(
                    receipt.resource_class,
                    tuple(active),
                )
                path = self._receipt_path(receipt.task_id)
                if not allowed:
                    if receipt.queue_reason != reason:
                        receipt = receipt.model_copy(
                            update={
                                "queue_reason": reason,
                                "updated_at_utc": _utc_now(),
                            }
                        )
                        _write_receipt(path, receipt)
                    continue
                launched = self._launch(path, receipt)
                if launched.state in {"starting", "running"}:
                    active.append(launched)

    def _launch(self, path: Path, receipt: TaskReceipt) -> TaskReceipt:
        starting = receipt.model_copy(
            update={
                "state": "starting",
                "queue_reason": None,
                "updated_at_utc": _utc_now(),
            }
        )
        _write_receipt(path, starting)
        worker_argv = (
            sys.executable,
            "-m",
            "ptcg_rl.dashboard.task_worker",
            "--receipt",
            str(path),
        )
        environment = os.environ.copy()
        python_path = os.pathsep.join(
            (
                str(self.repo_root / "data" / "sample_submission"),
                str(self.repo_root / "src"),
            )
        )
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            python_path if not existing else f"{python_path}{os.pathsep}{existing}"
        )
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed worker argv.
                worker_argv,
                cwd=self.repo_root,
                env=environment,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            failed = starting.model_copy(
                update={
                    "state": "failed",
                    "updated_at_utc": _utc_now(),
                    "finished_at_utc": _utc_now(),
                    "detail": f"worker launch failed: {error}",
                }
            )
            _write_receipt(path, failed)
            return failed
        current = _read_receipt(path)
        if current.state == "starting" and current.worker_pid is None:
            current = current.model_copy(
                update={
                    "worker_pid": process.pid,
                    "worker_start_ticks": process_start_ticks(process.pid),
                    "updated_at_utc": _utc_now(),
                }
            )
            _write_receipt(path, current)
        return current

    def _all_receipts(self) -> tuple[TaskReceipt, ...]:
        if not self.tasks_root.is_dir():
            return ()
        output: list[TaskReceipt] = []
        for path in self.tasks_root.glob("*/receipt.json"):
            try:
                with self._lock:
                    output.append(self._reconcile(path, _read_receipt(path)))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        output.sort(key=lambda receipt: receipt.created_at_utc, reverse=True)
        return tuple(output)

    def _reconcile(self, path: Path, receipt: TaskReceipt) -> TaskReceipt:
        if receipt.state not in {"starting", "running", "cancelling"}:
            return receipt
        if receipt.worker_pid is None:
            return receipt
        if process_identity_matches(receipt.worker_pid, receipt.worker_start_ticks):
            return receipt
        updated = receipt.model_copy(
            update={
                "state": "unknown",
                "updated_at_utc": _utc_now(),
                "finished_at_utc": _utc_now(),
                "detail": "worker exited without publishing a terminal receipt",
            }
        )
        _write_receipt(path, updated)
        return updated

    def _with_progress(self, receipt: TaskReceipt) -> TaskReceipt:
        return receipt.model_copy(update={"progress": self.results.progress(receipt)})

    def _receipt_path(self, task_id: str) -> Path:
        if re.fullmatch(r"[0-9a-f]{32}", task_id) is None:
            raise KeyError("unknown dashboard task")
        path = self.tasks_root / task_id / "receipt.json"
        if not path.is_file():
            raise KeyError("unknown dashboard task")
        return path

    def _spec_payload(
        self,
        *,
        task_id: str,
        request_payload: dict[str, Any],
        prepared: PreparedTask,
    ) -> dict[str, Any]:
        generated = {
            item.name: task_spec_fingerprint(item.payload) for item in prepared.files
        }
        return {
            "schema_version": 2,
            "task_id": task_id,
            "request": request_payload,
            "resolved_inputs": prepared.resolved_inputs,
            "resource_class": prepared.resource_class,
            "argv": list(prepared.argv),
            "output_dir": (
                None
                if prepared.output_dir is None
                else _relative(self.repo_root, prepared.output_dir)
            ),
            "status_path": (
                None
                if prepared.status_path is None
                else _relative(self.repo_root, prepared.status_path)
            ),
            "generated_config_fingerprints": generated,
        }


def _read_receipt(path: Path) -> TaskReceipt:
    return TaskReceipt.model_validate_json(path.read_text(encoding="utf-8"))


def _write_receipt(path: Path, receipt: TaskReceipt) -> None:
    atomic_write_json(path, receipt.model_dump(mode="json"))


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("task spec must contain a JSON object")
    return payload


def _relative(repo_root: Path, path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(repo_root):
        raise ValueError("dashboard task path escapes repository")
    return resolved.relative_to(repo_root).as_posix()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
