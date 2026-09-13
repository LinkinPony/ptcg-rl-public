"""Small durable pointer coordinating remote recurrent publication barriers."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

SERVED_POLICY_POINTER_NAME = "inference_served_latest.json"


class InferenceServedPolicyPointer(BaseModel):
    """Newest candidate snapshot admitting unbound recurrent sequences."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    version: int
    model_fingerprint: str

    @field_validator("schema_version")
    @classmethod
    def valid_schema_version(cls, value: int) -> int:
        """Reject unknown pointer schemas."""
        if value != 1:
            raise ValueError("unsupported served-policy pointer schema")
        return value

    @field_validator("version")
    @classmethod
    def valid_version(cls, value: int) -> int:
        """Reject negative policy versions."""
        if value < 0:
            raise ValueError("served policy version must be non-negative")
        return value

    @field_validator("model_fingerprint")
    @classmethod
    def valid_model_fingerprint(cls, value: str) -> str:
        """Require the full immutable model identity."""
        cleaned = value.strip().lower()
        if len(cleaned) != 64 or any(
            character not in "0123456789abcdef" for character in cleaned
        ):
            raise ValueError("served policy fingerprint must be SHA-256")
        return cleaned


def served_policy_pointer_path(weights_dir: Path) -> Path:
    """Return the shared remote-actor coordination pointer path."""
    return Path(weights_dir) / SERVED_POLICY_POINTER_NAME


def read_inference_served_policy(
    weights_dir: Path,
) -> InferenceServedPolicyPointer | None:
    """Read the current served pointer, tolerating startup absence."""
    path = served_policy_pointer_path(weights_dir)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return InferenceServedPolicyPointer.model_validate_json(raw)


def write_inference_served_policy(
    weights_dir: Path,
    *,
    version: int,
    model_fingerprint: str,
) -> Path:
    """Atomically publish the snapshot accepting new recurrent sequences."""
    pointer = InferenceServedPolicyPointer(
        version=version,
        model_fingerprint=model_fingerprint,
    )
    path = served_policy_pointer_path(weights_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with pending.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(pointer.model_dump(mode="json"), sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        with suppress(OSError):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        pending.unlink(missing_ok=True)
    return path
