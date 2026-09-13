"""Durable, idempotent dispatch for Kaggle submissions.

The local receipt is published before the Kaggle upload starts.  Once a
fingerprint has a receipt, later invocations may only reconcile remote state;
they never call the upload endpoint again.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from ptcg_rl.submission.dispatch_state import (
    DispatchState,
    KaggleSubmission,
    SubmissionDispatchOutcome,
    SubmissionDispatchReceipt,
    SubmissionDispatchRequest,
    append_submission_history,
    exclusive_file_lock,
    file_sha256,
    load_dispatch_receipt,
    replace_dispatch_receipt,
    submission_history_refs,
    write_new_dispatch_receipt,
)

Submitter = Callable[[Path, str, str], None]
SubmissionLister = Callable[[str], Sequence["KaggleSubmission"]]
Clock = Callable[[], str]


def tagged_submission_message(message: str, fingerprint: str) -> str:
    """Add a stable remote lookup token without duplicating it."""
    normalized = message.strip()
    if not normalized:
        raise ValueError("submission message must not be empty")
    token = f"[fp:{fingerprint}]"
    return normalized if token in normalized else f"{normalized} {token}"


def submission_archive_fingerprint(path: Path) -> str:
    """Hash archive member paths and bytes while ignoring tar metadata."""
    digest = hashlib.sha256()
    with tarfile.open(path, "r:*") as archive:
        members = sorted(archive.getmembers(), key=lambda member: member.name)
        for member in members:
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(
                    f"submission archive contains a non-file: {member.name}"
                )
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(
                    f"submission archive contains an unsafe path: {member.name}"
                )
            encoded_path = member_path.as_posix().encode("utf-8")
            digest.update(len(encoded_path).to_bytes(8, "big"))
            digest.update(encoded_path)
            file_obj = archive.extractfile(member)
            if file_obj is None:
                raise ValueError(f"cannot read submission member: {member.name}")
            while chunk := file_obj.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def list_kaggle_submissions(competition: str) -> tuple[KaggleSubmission, ...]:
    """Read recent submissions through the Kaggle JSON interface."""
    completed = subprocess.run(
        [
            "kaggle",
            "competitions",
            "submissions",
            competition,
            "--format",
            "json",
            "--page-size",
            "200",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    raw = json.loads(completed.stdout)
    if not isinstance(raw, list):
        raise ValueError("Kaggle submissions response must be a list")
    return tuple(_normalize_remote_submission(item) for item in raw)


def dispatch_submission_once(
    request: SubmissionDispatchRequest,
    *,
    submitter: Submitter,
    lister: SubmissionLister = list_kaggle_submissions,
    wait_seconds: float = 180.0,
    poll_interval_seconds: float = 5.0,
    clock: Clock | None = None,
) -> SubmissionDispatchOutcome:
    """Upload at most once, then reconcile the immutable fingerprint.

    Existing receipts, including ``reserved`` and uncertain receipts, are
    reconcile-only.  This deliberately favors a missed upload over a duplicate.
    """
    if wait_seconds < 0.0:
        raise ValueError("wait_seconds must be non-negative")
    if poll_interval_seconds <= 0.0:
        raise ValueError("poll_interval_seconds must be positive")
    now = clock or _utc_now
    archive_path = request.archive_path.resolve()
    _verify_archive(request, archive_path)
    dispatch_dir = request.dispatch_dir.resolve()
    receipt_path = dispatch_dir / f"{request.submission_fingerprint}.json"
    lock_path = dispatch_dir / f"{request.submission_fingerprint}.lock"
    history_lock_path = dispatch_dir / "history.lock"

    with exclusive_file_lock(lock_path):
        existing = load_dispatch_receipt(receipt_path)
        if existing is not None:
            return _reconcile_existing(
                existing,
                receipt_path=receipt_path,
                history_path=request.history_path.resolve(),
                history_lock_path=history_lock_path,
                lister=lister,
                wait_seconds=wait_seconds,
                poll_interval_seconds=poll_interval_seconds,
                clock=now,
            )

        history_refs = submission_history_refs(
            request.history_path.resolve(),
            request.submission_fingerprint,
        )
        if history_refs:
            receipt = _new_receipt(
                request,
                state=("duplicate_detected" if len(history_refs) > 1 else "recorded"),
                created_at_utc=now(),
                submission_refs=history_refs,
            )
            write_new_dispatch_receipt(receipt_path, receipt)
            return SubmissionDispatchOutcome(
                receipt=receipt,
                upload_invoked=False,
            )

        try:
            remote = tuple(lister(request.competition))
        except Exception as error:
            raise RuntimeError(
                "cannot confirm remote submission state; refusing to upload"
            ) from error
        matches = _matching_submissions(
            remote,
            message=request.message,
        )
        if matches:
            receipt = _new_receipt(
                request,
                state=("duplicate_detected" if len(matches) > 1 else "recorded"),
                created_at_utc=now(),
                last_checked_at_utc=now(),
                submission_refs=tuple(item.submission_ref for item in matches),
            )
            write_new_dispatch_receipt(receipt_path, receipt)
            append_submission_history(
                request.history_path.resolve(),
                lock_path=history_lock_path,
                bundle_id=request.bundle_id,
                fingerprint=request.submission_fingerprint,
                submissions=matches,
            )
            return SubmissionDispatchOutcome(
                receipt=receipt,
                upload_invoked=False,
                remote_matches=matches,
            )

        reserved = _new_receipt(
            request,
            state="reserved",
            created_at_utc=now(),
        )
        write_new_dispatch_receipt(receipt_path, reserved)
        uploading = reserved.model_copy(
            update={
                "state": "uploading",
                "upload_started_at_utc": now(),
            }
        )
        replace_dispatch_receipt(receipt_path, uploading)
        try:
            submitter(archive_path, request.competition, request.message)
        except BaseException as error:
            uncertain = uploading.model_copy(
                update={
                    "state": "upload_uncertain",
                    "last_error": _error_text(error),
                }
            )
            replace_dispatch_receipt(receipt_path, uncertain)
            raise

        accepted = uploading.model_copy(
            update={
                "state": "accepted_pending_reference",
                "upload_completed_at_utc": now(),
                "last_error": None,
            }
        )
        replace_dispatch_receipt(receipt_path, accepted)
        outcome = _poll_for_reference(
            accepted,
            receipt_path=receipt_path,
            history_path=request.history_path.resolve(),
            history_lock_path=history_lock_path,
            lister=lister,
            wait_seconds=wait_seconds,
            poll_interval_seconds=poll_interval_seconds,
            clock=now,
        )
        return outcome.model_copy(update={"upload_invoked": True})


def _reconcile_existing(
    receipt: SubmissionDispatchReceipt,
    *,
    receipt_path: Path,
    history_path: Path,
    history_lock_path: Path,
    lister: SubmissionLister,
    wait_seconds: float,
    poll_interval_seconds: float,
    clock: Clock,
) -> SubmissionDispatchOutcome:
    """Reconcile a prior reservation without ever uploading again."""
    history_refs = submission_history_refs(
        history_path,
        receipt.submission_fingerprint,
    )
    if history_refs:
        updated = receipt.model_copy(
            update={
                "state": (
                    "duplicate_detected" if len(history_refs) > 1 else "recorded"
                ),
                "submission_refs": history_refs,
                "last_checked_at_utc": clock(),
                "last_error": None,
            }
        )
        replace_dispatch_receipt(receipt_path, updated)
        return SubmissionDispatchOutcome(receipt=updated, upload_invoked=False)
    return _poll_for_reference(
        receipt,
        receipt_path=receipt_path,
        history_path=history_path,
        history_lock_path=history_lock_path,
        lister=lister,
        wait_seconds=wait_seconds,
        poll_interval_seconds=poll_interval_seconds,
        clock=clock,
    )


def _poll_for_reference(
    receipt: SubmissionDispatchReceipt,
    *,
    receipt_path: Path,
    history_path: Path,
    history_lock_path: Path,
    lister: SubmissionLister,
    wait_seconds: float,
    poll_interval_seconds: float,
    clock: Clock,
) -> SubmissionDispatchOutcome:
    """Poll only the existing upload state; never initiate an upload."""
    deadline = time.monotonic() + wait_seconds
    last_error = receipt.last_error
    while True:
        try:
            matches = _matching_submissions(
                lister(receipt.competition),
                message=receipt.message,
            )
        except Exception as error:
            matches = ()
            last_error = _error_text(error)
        checked_at = clock()
        if matches:
            updated = receipt.model_copy(
                update={
                    "state": ("duplicate_detected" if len(matches) > 1 else "recorded"),
                    "submission_refs": tuple(item.submission_ref for item in matches),
                    "last_checked_at_utc": checked_at,
                    "last_error": None,
                }
            )
            replace_dispatch_receipt(receipt_path, updated)
            append_submission_history(
                history_path,
                lock_path=history_lock_path,
                bundle_id=receipt.bundle_id,
                fingerprint=receipt.submission_fingerprint,
                submissions=matches,
            )
            return SubmissionDispatchOutcome(
                receipt=updated,
                upload_invoked=False,
                remote_matches=matches,
            )
        updated = receipt.model_copy(
            update={
                "last_checked_at_utc": checked_at,
                "last_error": last_error,
            }
        )
        replace_dispatch_receipt(receipt_path, updated)
        if time.monotonic() >= deadline:
            return SubmissionDispatchOutcome(
                receipt=updated,
                upload_invoked=False,
            )
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))
        receipt = updated


def _new_receipt(
    request: SubmissionDispatchRequest,
    *,
    state: DispatchState,
    created_at_utc: str,
    last_checked_at_utc: str | None = None,
    submission_refs: tuple[str, ...] = (),
) -> SubmissionDispatchReceipt:
    return SubmissionDispatchReceipt(
        submission_fingerprint=request.submission_fingerprint,
        bundle_id=request.bundle_id,
        profile=request.profile,
        competition=request.competition,
        archive_path=request.archive_path.resolve(),
        archive_sha256=request.archive_sha256,
        archive_size_bytes=request.archive_size_bytes,
        message=request.message,
        state=state,
        created_at_utc=created_at_utc,
        last_checked_at_utc=last_checked_at_utc,
        submission_refs=submission_refs,
    )


def _verify_archive(request: SubmissionDispatchRequest, archive_path: Path) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(f"submission archive does not exist: {archive_path}")
    if archive_path.stat().st_size != request.archive_size_bytes:
        raise ValueError("submission archive size changed before dispatch")
    if file_sha256(archive_path) != request.archive_sha256:
        raise ValueError("submission archive SHA-256 changed before dispatch")


def _matching_submissions(
    submissions: Sequence[KaggleSubmission],
    *,
    message: str,
) -> tuple[KaggleSubmission, ...]:
    return tuple(item for item in submissions if item.description == message)


def _normalize_remote_submission(value: object) -> KaggleSubmission:
    if not isinstance(value, Mapping):
        raise ValueError("Kaggle submission row must be an object")
    submitted_at = str(value.get("date", "")).strip()
    if submitted_at and not submitted_at.endswith("Z"):
        submitted_at += "Z"
    return KaggleSubmission(
        submission_ref=str(value.get("ref", "")),
        file_name=str(value.get("fileName", "")),
        submitted_at_utc=submitted_at,
        description=str(value.get("description", "")),
        status=str(value.get("status", "")),
        public_score=_optional_text(value.get("publicScore")),
        private_score=_optional_text(value.get("privateScore")),
    )


def _optional_text(value: object) -> str | None:
    normalized = str(value).strip() if value is not None else ""
    return normalized or None


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:2000]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "KaggleSubmission",
    "SubmissionDispatchOutcome",
    "SubmissionDispatchReceipt",
    "SubmissionDispatchRequest",
    "dispatch_submission_once",
    "list_kaggle_submissions",
    "submission_archive_fingerprint",
    "tagged_submission_message",
]
