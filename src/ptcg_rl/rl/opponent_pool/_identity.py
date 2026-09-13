"""Canonical identities for the standalone opponent-pool V2 core."""

from __future__ import annotations

import hashlib
import json
import threading
import weakref
from collections.abc import Callable
from typing import Annotated, Any

from pydantic import BeforeValidator

_FINGERPRINT_CACHE_LOCK = threading.Lock()
_FINGERPRINT_CACHE: dict[
    tuple[str, int],
    tuple[weakref.ReferenceType[object], str],
] = {}


def normalize_sha256(value: Any) -> str:
    """Normalize and validate one SHA-256 identity."""
    if not isinstance(value, str):
        raise ValueError("SHA-256 identity must be a string")
    cleaned = value.strip().lower()
    if len(cleaned) != 64 or any(
        character not in "0123456789abcdef" for character in cleaned
    ):
        raise ValueError("identity must be a lowercase SHA-256 digest")
    return cleaned


Sha256 = Annotated[str, BeforeValidator(normalize_sha256)]


def canonical_fingerprint(domain: str, payload: object) -> str:
    """Return a domain-separated fingerprint for JSON-compatible payload data."""
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"ptcg-rl/opponent-pool-v2/")
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    digest.update(encoded)
    return digest.hexdigest()


def cached_fingerprint(
    domain: str,
    value: object,
    compute: Callable[[], str],
) -> str:
    """Reuse a canonical fingerprint while it names the same immutable object."""
    key = (domain, id(value))
    with _FINGERPRINT_CACHE_LOCK:
        cached = _FINGERPRINT_CACHE.get(key)
        if cached is not None and cached[0]() is value:
            return cached[1]
    fingerprint = compute()

    def discard(reference: weakref.ReferenceType[object]) -> None:
        with _FINGERPRINT_CACHE_LOCK:
            cached = _FINGERPRINT_CACHE.get(key)
            if cached is not None and cached[0] is reference:
                _FINGERPRINT_CACHE.pop(key, None)

    reference = weakref.ref(value, discard)
    with _FINGERPRINT_CACHE_LOCK:
        _FINGERPRINT_CACHE[key] = (reference, fingerprint)
    return fingerprint
