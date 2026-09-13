"""Trusted-client, receipt-backed dashboard job orchestration."""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ptcg_rl.dashboard.job_adapters import JOB_TEMPLATES, JobCommandAdapters
from ptcg_rl.dashboard.job_models import (
    DashboardSession,
    JobLogPayload,
    JobProgressPayload,
    JobReceipt,
    JobTemplate,
    StartJobRequest,
)
from ptcg_rl.rl.performance_state import atomic_write_json


class DashboardJobManager:
    """Create local jobs through fixed adapters and immutable receipts."""

    def __init__(
        self,
        *,
        repo_root: Path,
        enabled: bool,
        refresh_seconds: int = 15,
    ) -> None:
        if refresh_seconds <= 0:
            raise ValueError("dashboard refresh_seconds must be positive")
        self.repo_root = repo_root.resolve()
        self.enabled = enabled
        self.refresh_seconds = refresh_seconds
        self.request_token = secrets.token_urlsafe(32) if enabled else None
        self.jobs_root = self.repo_root / "outputs" / "dashboard" / "jobs"
        self.adapters = JobCommandAdapters(self.repo_root)

    @property
    def session(self) -> DashboardSession:
        """Return current-process action capabilities."""
        return DashboardSession(
            actions_enabled=self.enabled,
            action_scope="loopback" if self.enabled else "disabled",
            request_token=self.request_token,
            refresh_seconds=self.refresh_seconds,
        )

    def templates(self) -> tuple[JobTemplate, ...]:
        """Return the complete allowlist."""
        return JOB_TEMPLATES

    def list_jobs(self) -> tuple[JobReceipt, ...]:
        """Read durable receipts and reconcile vanished worker processes."""
        if not self.jobs_root.is_dir():
            return ()
        receipts: list[JobReceipt] = []
        for path in self.jobs_root.glob("*/receipt.json"):
            try:
                receipt = _read_receipt(path)
                receipt = self._reconcile(path, receipt)
                receipts.append(receipt)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        receipts.sort(key=lambda receipt: receipt.created_at_utc, reverse=True)
        return tuple(receipts)

    def get_job(self, job_id: str) -> JobReceipt:
        """Return one safe job identifier."""
        return self._reconcile(
            self._receipt_path(job_id),
            _read_receipt(self._receipt_path(job_id)),
        )

    def job_log(self, job_id: str, *, tail_bytes: int) -> JobLogPayload:
        """Read a bounded UTF-8-safe log tail."""
        if not 1 <= tail_bytes <= 1_000_000:
            raise ValueError("job log tail_bytes must be between 1 and 1000000")
        receipt = self.get_job(job_id)
        path = self.repo_root / receipt.log_path
        if not path.is_file():
            return JobLogPayload(job_id=job_id, text="", truncated=False)
        size = path.stat().st_size
        with path.open("rb") as stream:
            if size > tail_bytes:
                stream.seek(-tail_bytes, os.SEEK_END)
            payload = stream.read()
        return JobLogPayload(
            job_id=job_id,
            text=payload.decode("utf-8", errors="replace"),
            truncated=size > tail_bytes,
        )

    def job_progress(self, job_id: str) -> JobProgressPayload:
        """Read structured progress independently from browser or server lifetime."""
        receipt = self.get_job(job_id)
        path = self.jobs_root / job_id / "progress.json"
        raw: dict[str, object] = {}
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("dashboard job progress must be a JSON object")
            raw = payload
        succeeded = receipt.state == "succeeded"
        phase = str(raw.get("phase") or ("completed" if succeeded else receipt.state))
        message = str(
            raw.get("message")
            or receipt.detail
            or ("任务已完成" if succeeded else "等待任务发布结构化进度")
        )
        return JobProgressPayload.model_validate(
            {
                **raw,
                "job_id": job_id,
                "template_id": receipt.template_id,
                "state": receipt.state,
                "phase": phase,
                "message": message,
                "percent": 100.0 if succeeded else raw.get("percent", 0.0),
                "updated_at_utc": (raw.get("updated_at_utc") or receipt.updated_at_utc),
                "detail": receipt.detail,
            }
        )

    def start(self, request: StartJobRequest) -> JobReceipt:
        """Validate adapter parameters, reserve a receipt, and detach a worker."""
        if not self.enabled:
            raise PermissionError("dashboard actions are disabled")
        job_id = uuid.uuid4().hex
        argv, parameters = self.adapters.build(request, job_id=job_id)
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        receipt_path = job_dir / "receipt.json"
        log_path = job_dir / "job.log"
        now = _utc_now()
        receipt = JobReceipt(
            job_id=job_id,
            template_id=request.template_id,
            state="starting",
            created_at_utc=now,
            updated_at_utc=now,
            parameters=parameters,
            argv=argv,
            cwd=str(self.repo_root),
            log_path=str(log_path.relative_to(self.repo_root)),
        )
        _write_receipt(receipt_path, receipt)
        worker_argv = (
            sys.executable,
            "-m",
            "ptcg_rl.dashboard.job_worker",
            "--receipt",
            str(receipt_path),
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
        process = subprocess.Popen(  # noqa: S603 - argv is adapter-generated.
            worker_argv,
            cwd=self.repo_root,
            env=environment,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        current = _read_receipt(receipt_path)
        receipt = current.model_copy(
            update={
                "worker_pid": process.pid,
                "updated_at_utc": _utc_now(),
            }
        )
        _write_receipt(receipt_path, receipt)
        return receipt

    def cancel(self, job_id: str) -> JobReceipt:
        """Terminate only a dashboard-owned process group."""
        if not self.enabled:
            raise PermissionError("dashboard actions are disabled")
        path = self._receipt_path(job_id)
        receipt = _read_receipt(path)
        if receipt.state not in {"starting", "running", "cancelling"}:
            return receipt
        pid = receipt.worker_pid
        if pid is None or not _pid_exists(pid):
            updated = receipt.model_copy(
                update={
                    "state": "unknown",
                    "updated_at_utc": _utc_now(),
                    "detail": "worker vanished before cancellation",
                }
            )
            _write_receipt(path, updated)
            return updated
        cancelling = receipt.model_copy(
            update={"state": "cancelling", "updated_at_utc": _utc_now()}
        )
        _write_receipt(path, cancelling)
        os.killpg(pid, signal.SIGTERM)
        return cancelling

    def _receipt_path(self, job_id: str) -> Path:
        if re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
            raise KeyError("unknown dashboard job")
        path = self.jobs_root / job_id / "receipt.json"
        if not path.is_file():
            raise KeyError("unknown dashboard job")
        return path

    def _reconcile(self, path: Path, receipt: JobReceipt) -> JobReceipt:
        if (
            receipt.state in {"starting", "running", "cancelling"}
            and receipt.worker_pid is not None
            and not _pid_exists(receipt.worker_pid)
        ):
            updated = receipt.model_copy(
                update={
                    "state": "unknown",
                    "updated_at_utc": _utc_now(),
                    "detail": "worker exited without publishing a terminal receipt",
                }
            )
            _write_receipt(path, updated)
            return updated
        return receipt


def _read_receipt(path: Path) -> JobReceipt:
    return JobReceipt.model_validate_json(path.read_text(encoding="utf-8"))


def _write_receipt(path: Path, receipt: JobReceipt) -> None:
    atomic_write_json(path, receipt.model_dump(mode="json"))


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
