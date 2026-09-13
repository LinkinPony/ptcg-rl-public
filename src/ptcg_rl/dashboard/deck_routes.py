"""Resolve dashboard deck labels to immutable exact and family routes."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ptcg_rl.decks.identity import canonicalize_deck

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_LABEL_HASH_PATTERN = re.compile(r"(?:^|_)([0-9a-f]{12})(?=_|$)")
_SOURCE_ROOTS = ("docs", "configs", "data/external")


@dataclass(frozen=True, slots=True)
class DeckRouteIdentity:
    """Immutable route identity associated with one performance label."""

    deck_digest: str
    family_id: str | None


@dataclass(frozen=True, slots=True)
class DeckRouteMetadata:
    """Resolved label routes plus diagnostics for a run artifact."""

    identities: Mapping[str, DeckRouteIdentity]
    family_routes_declared: bool
    unresolved_labels: tuple[str, ...]


def resolve_deck_route_metadata(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
    deck_labels: Sequence[str],
) -> DeckRouteMetadata:
    """Bind human-facing labels to path-free routes in ``resolved_config``."""
    model = _mapping(payload.get("resolved_model_config"))
    raw_exact_routes = _sequence(model.get("exact_routes"))
    raw_family_routes = _sequence(model.get("family_routes"))
    family_routes_declared = bool(raw_family_routes)

    family_by_digest: dict[str, str] = {}
    for raw_route in raw_family_routes:
        route = _mapping(raw_route)
        digest = _digest(route.get("deck_digest"))
        family_id = _digest(route.get("family_id"))
        if digest is not None and family_id is not None:
            family_by_digest[digest] = family_id

    token_routes: dict[str, set[str]] = defaultdict(set)
    known_digests: set[str] = set(family_by_digest)
    for raw_route in (*raw_exact_routes, *raw_family_routes):
        route = _mapping(raw_route)
        digest = _digest(route.get("deck_digest"))
        if digest is None:
            continue
        known_digests.add(digest)
        token_routes[digest[:12]].add(digest)
        signature = _text(route.get("signature"))
        if signature is not None:
            token_routes[hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]].add(
                digest
            )

    identities = _explicit_identities(
        payload,
        known_digests=known_digests,
        family_by_digest=family_by_digest,
    )
    unresolved: list[str] = []
    for label in deck_labels:
        if label in identities:
            continue
        candidates = {
            digest
            for token in _label_hashes(label)
            for digest in token_routes.get(token, ())
        }
        if len(candidates) == 1:
            digest = candidates.pop()
            identities[label] = DeckRouteIdentity(
                deck_digest=digest,
                family_id=family_by_digest.get(digest),
            )
        else:
            unresolved.append(label)

    if unresolved and known_digests:
        for label in tuple(unresolved):
            digest = _resolve_source_digest(
                repo_root,
                label=label,
                known_digests=known_digests,
            )
            if digest is None:
                continue
            identities[label] = DeckRouteIdentity(
                deck_digest=digest,
                family_id=family_by_digest.get(digest),
            )
            unresolved.remove(label)

    return DeckRouteMetadata(
        identities=identities,
        family_routes_declared=family_routes_declared,
        unresolved_labels=tuple(unresolved),
    )


def label_hash(label: str) -> str | None:
    """Return the first legacy compact identity embedded in a deck label."""
    hashes = _label_hashes(label)
    return hashes[0] if hashes else None


def _explicit_identities(
    payload: Mapping[str, Any],
    *,
    known_digests: set[str],
    family_by_digest: Mapping[str, str],
) -> dict[str, DeckRouteIdentity]:
    identities: dict[str, DeckRouteIdentity] = {}
    for raw_route in _sequence(payload.get("active_deck_routes")):
        route = _mapping(raw_route)
        label = _text(route.get("label"))
        digest = _digest(route.get("deck_digest"))
        if label is None or digest is None or digest not in known_digests:
            continue
        declared_family = _digest(route.get("family_id"))
        resolved_family = family_by_digest.get(digest)
        if (
            declared_family is not None
            and resolved_family is not None
            and declared_family != resolved_family
        ):
            continue
        identities[label] = DeckRouteIdentity(
            deck_digest=digest,
            family_id=resolved_family or declared_family,
        )
    return identities


def _resolve_source_digest(
    repo_root: Path,
    *,
    label: str,
    known_digests: set[str],
) -> str | None:
    candidates: set[str] = set()
    for token in _label_hashes(label):
        for relative_root in _SOURCE_ROOTS:
            source_root = (repo_root / relative_root).resolve()
            if not source_root.is_dir():
                continue
            for path in source_root.rglob(f"*{token}*.csv"):
                digest = _deck_file_digest(path, source_root=source_root)
                if digest in known_digests:
                    candidates.add(digest)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _deck_file_digest(path: Path, *, source_root: Path) -> str | None:
    try:
        resolved = path.resolve()
        if not resolved.is_relative_to(source_root) or not resolved.is_file():
            return None
        card_ids = [
            int(line.strip().strip(","))
            for line in resolved.read_text(encoding="utf-8").splitlines()
            if line.strip().strip(",")
        ]
        return canonicalize_deck(card_ids).deck_digest
    except (OSError, UnicodeError, ValueError):
        return None


def _label_hashes(label: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_LABEL_HASH_PATTERN.findall(label)))


def _digest(value: Any) -> str | None:
    text = _text(value)
    return text if text is not None and _DIGEST_PATTERN.fullmatch(text) else None


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


__all__ = [
    "DeckRouteIdentity",
    "DeckRouteMetadata",
    "label_hash",
    "resolve_deck_route_metadata",
]
