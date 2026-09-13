"""Canonical content identities for belief-runtime semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_belief_fingerprint(domain: bytes, payload: Any) -> str:
    """Hash one JSON-safe semantic payload under a versioned domain."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Stream an immutable source file into a lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["canonical_belief_fingerprint", "file_sha256"]
