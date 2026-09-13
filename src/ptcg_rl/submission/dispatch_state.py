"""Typed durable state and atomic storage for Kaggle dispatch."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

_HISTORY_PROTOCOL = "SUBMITTED-BUNDLES-v1"
_HEX_DIGITS = frozenset("0123456789abcdef")

DispatchState = Literal[
    "reserved",
    "uploading",
    "accepted_pending_reference",
    "upload_uncertain",
    "recorded",
    "duplicate_detected",
]


class KaggleSubmission(BaseModel):
    """One normalized row returned by the Kaggle submissions API."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    submission_ref: str
    file_name: str
    submitted_at_utc: str
    description: str
    status: str
    public_score: str | None = None
    private_score: str | None = None

    @field_validator(
        "submission_ref",
        "file_name",
        "submitted_at_utc",
        "status",
    )
    @classmethod
    def nonempty_fields(cls, value: str) -> str:
        """Reject malformed remote rows."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Kaggle submission fields must be non-empty")
        return normalized

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        """Allow Kaggle's valid blank descriptions while normalizing text."""
        return value.strip()


class SubmissionDispatchRequest(BaseModel):
    """Immutable inputs for one protected upload attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    submission_fingerprint: str
    bundle_id: str
    profile: str
    competition: str
    archive_path: Path
    archive_sha256: str
    archive_size_bytes: int = Field(gt=0)
    message: str
    dispatch_dir: Path
    history_path: Path

    @field_validator("submission_fingerprint", "archive_sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Normalize content identities."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in _HEX_DIGITS for character in normalized
        ):
            raise ValueError("dispatch hashes must be lowercase SHA-256")
        return normalized

    @field_validator("bundle_id", "profile", "competition", "message")
    @classmethod
    def nonempty_identity(cls, value: str) -> str:
        """Reject ambiguous dispatch metadata."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("dispatch identity fields must be non-empty")
        return normalized


class SubmissionDispatchReceipt(BaseModel):
    """Durable state proving that a fingerprint was reserved for upload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["KAGGLE-SUBMISSION-DISPATCH-v1"] = "KAGGLE-SUBMISSION-DISPATCH-v1"
    submission_fingerprint: str
    bundle_id: str
    profile: str
    competition: str
    archive_path: Path
    archive_sha256: str
    archive_size_bytes: int = Field(gt=0)
    message: str
    state: DispatchState
    created_at_utc: str
    upload_started_at_utc: str | None = None
    upload_completed_at_utc: str | None = None
    last_checked_at_utc: str | None = None
    submission_refs: tuple[str, ...] = ()
    last_error: str | None = None

    @field_validator("submission_fingerprint", "archive_sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Validate persisted content identities."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in _HEX_DIGITS for character in normalized
        ):
            raise ValueError("receipt hashes must be lowercase SHA-256")
        return normalized


class SubmissionDispatchOutcome(BaseModel):
    """Result of either the sole upload or a reconcile-only invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt: SubmissionDispatchReceipt
    upload_invoked: bool
    remote_matches: tuple[KaggleSubmission, ...] = ()


def submission_history_refs(path: Path, fingerprint: str) -> tuple[str, ...]:
    """Return all recorded refs for one effective bundle fingerprint."""
    entries = _read_history(path)
    return tuple(
        str(entry.get("submission_ref", "unknown"))
        for entry in entries
        if str(entry.get("submission_fingerprint", "")) == fingerprint
    )


def append_submission_history(
    path: Path,
    *,
    lock_path: Path,
    bundle_id: str,
    fingerprint: str,
    submissions: Sequence[KaggleSubmission],
) -> None:
    """Atomically append remote references while preserving duplicate evidence."""
    with exclusive_file_lock(lock_path):
        raw = _read_history_object(path)
        entries = cast(list[dict[str, Any]], raw["submissions"])
        known_refs = {str(entry.get("submission_ref", "")) for entry in entries}
        changed = False
        for submission in submissions:
            if submission.submission_ref in known_refs:
                continue
            entries.append(
                {
                    "submission_ref": submission.submission_ref,
                    "submitted_at_utc": submission.submitted_at_utc,
                    "bundle_id": bundle_id,
                    "submission_fingerprint": fingerprint,
                }
            )
            known_refs.add(submission.submission_ref)
            changed = True
        if changed:
            _atomic_write_json(path, raw)


def load_dispatch_receipt(path: Path) -> SubmissionDispatchReceipt | None:
    """Load a receipt, treating malformed existing state as uncertain."""
    if not path.exists():
        return None
    try:
        return SubmissionDispatchReceipt.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as error:
        raise RuntimeError(
            "existing dispatch receipt is unreadable; upload state is uncertain "
            f"and must not be retried: {path}"
        ) from error


def write_new_dispatch_receipt(
    path: Path,
    receipt: SubmissionDispatchReceipt,
) -> None:
    """Publish the initial reservation with no-replace semantics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(receipt.model_dump(mode="json"))
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise RuntimeError(
            "dispatch receipt appeared concurrently; refusing duplicate upload"
        ) from error
    with os.fdopen(descriptor, "wb") as file_obj:
        file_obj.write(payload)
        file_obj.flush()
        os.fsync(file_obj.fileno())
    _fsync_directory(path.parent)


def replace_dispatch_receipt(
    path: Path,
    receipt: SubmissionDispatchReceipt,
) -> None:
    """Atomically advance a receipt after its reservation exists."""
    _atomic_write_bytes(path, _json_bytes(receipt.model_dump(mode="json")))


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Serialize dispatch or history mutations across local processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_history(path: Path) -> tuple[Mapping[str, Any], ...]:
    raw = _read_history_object(path)
    return tuple(cast(Mapping[str, Any], item) for item in raw["submissions"])


def _read_history_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"submission history does not exist: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("protocol") != _HISTORY_PROTOCOL:
        raise ValueError("unsupported submission history protocol")
    entries = raw.get("submissions")
    if not isinstance(entries, list) or not all(
        isinstance(item, dict) for item in entries
    ):
        raise ValueError("submission history submissions must be objects")
    return cast(dict[str, Any], raw)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, _json_bytes(payload))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with temporary.open("xb") as file_obj:
            file_obj.write(payload)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "DispatchState",
    "KaggleSubmission",
    "SubmissionDispatchOutcome",
    "SubmissionDispatchReceipt",
    "SubmissionDispatchRequest",
    "append_submission_history",
    "exclusive_file_lock",
    "file_sha256",
    "load_dispatch_receipt",
    "replace_dispatch_receipt",
    "submission_history_refs",
    "write_new_dispatch_receipt",
]
