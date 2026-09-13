"""Small atomic artifact helpers shared by campaign preparation and execution."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def file_sha256(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: BaseModel | dict[str, Any]) -> None:
    """Publish canonical JSON through a same-directory atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(values, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject scalar/list roots."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return payload


__all__ = ["file_sha256", "read_json", "write_json_atomic"]
