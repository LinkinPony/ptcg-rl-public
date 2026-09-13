"""Resolve authoritative deck and public-catalog provenance for inventory."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.decks.traceability import authoritative_deck_hash
from ptcg_rl.evaluation.native_collection_campaign.artifact_io import (
    file_sha256,
    read_json,
)
from ptcg_rl.evaluation.native_collection_campaign.models import HistoricalDeckRecord


@dataclass(frozen=True)
class ResolvedRouteEvidence:
    """Compact deck identity observed under one exact registry fingerprint."""

    deck_hash: str
    label: str
    provenance_paths: tuple[Path, ...]


def extract_authoritative_deck_hash(
    label: str,
    *,
    explicit: str | None = None,
) -> str:
    """Resolve only a compact ID explicitly stored in deck metadata or label."""
    return authoritative_deck_hash(label, explicit=explicit)


def discover_decks(
    registry_paths: Sequence[Path],
    *,
    root: Path,
) -> dict[str, tuple[HistoricalDeckRecord, ...]]:
    """Discover canonical deck bytes and their historical compact identifiers."""
    candidates: defaultdict[str, list[HistoricalDeckRecord]] = defaultdict(list)
    for registry_path in registry_paths:
        try:
            payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        for item in _walk_mappings(payload):
            label = item.get("label")
            raw_path = item.get("path")
            if not isinstance(label, str) or not isinstance(raw_path, str):
                continue
            if "${" in raw_path or not raw_path.lower().endswith(".csv"):
                continue
            path = _resolve_path(Path(raw_path), root=root)
            if not path.is_file():
                continue
            try:
                deck_hash = extract_authoritative_deck_hash(
                    label,
                    explicit=(
                        str(item["deck_hash"])
                        if isinstance(item.get("deck_hash"), str)
                        else None
                    ),
                )
                canonical = canonicalize_deck(records.read_deck(path))
            except (KeyError, TypeError, ValueError):
                continue
            candidates[canonical.deck_digest].append(
                HistoricalDeckRecord(
                    deck_digest=canonical.deck_digest,
                    deck_hash=deck_hash,
                    label=label,
                    path=_display_path(path, root=root),
                    deck_signature=canonical.signature,
                    provenance_paths=(_display_path(registry_path, root=root),),
                )
            )
    output: dict[str, tuple[HistoricalDeckRecord, ...]] = {}
    for digest, values in sorted(candidates.items()):
        signatures = {item.deck_signature for item in values}
        if len(signatures) != 1:
            raise ValueError(
                f"conflicting canonical signatures for exact deck {digest}"
            )
        output[digest] = tuple(
            sorted(
                values,
                key=lambda item: (item.deck_hash, str(item.path), item.label),
            )
        )
    return output


def discover_catalogs(
    manifest_paths: Sequence[Path],
    *,
    root: Path,
) -> dict[str, tuple[Path, str]]:
    """Index immutable public-catalog manifests by their canonical fingerprint."""
    output: dict[str, tuple[Path, str]] = {}
    for path in manifest_paths:
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            continue
        fingerprint = payload.get("catalog_fingerprint")
        if not isinstance(fingerprint, str):
            continue
        fingerprint = fingerprint.strip().lower()
        if len(fingerprint) != 64:
            continue
        value = (_display_path(path, root=root), file_sha256(path))
        previous = output.setdefault(fingerprint, value)
        if previous[1] != value[1]:
            raise ValueError(
                f"public catalog fingerprint {fingerprint} has conflicting manifests"
            )
    return output


def discover_resolved_route_evidence(
    config_paths: Sequence[Path],
    *,
    root: Path,
) -> dict[tuple[str, str], ResolvedRouteEvidence]:
    """Index compact IDs by the exact registry identity that carried them."""
    grouped: defaultdict[tuple[str, str], list[tuple[str, str, Path]]] = defaultdict(
        list
    )
    for path in config_paths:
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            continue
        registry = payload.get("exact_registry_fingerprint")
        routes = payload.get("active_deck_routes")
        if not isinstance(registry, str) or not isinstance(routes, list):
            continue
        for route in routes:
            if not isinstance(route, Mapping):
                continue
            digest = route.get("deck_digest")
            label = route.get("label")
            explicit = route.get("deck_hash")
            if not isinstance(digest, str) or not isinstance(label, str):
                continue
            if explicit is not None and not isinstance(explicit, str):
                continue
            try:
                deck_hash = extract_authoritative_deck_hash(
                    label,
                    explicit=explicit,
                )
            except ValueError:
                continue
            grouped[(registry, digest)].append(
                (deck_hash, label, _display_path(path, root=root))
            )
    output: dict[tuple[str, str], ResolvedRouteEvidence] = {}
    for key, values in sorted(grouped.items()):
        hashes = {item[0] for item in values}
        if len(hashes) != 1:
            raise ValueError(
                "one exact registry maps a deck to conflicting compact IDs: "
                f"registry={key[0]} deck={key[1]}"
            )
        selected = min(values, key=lambda item: (str(item[2]), item[1]))
        output[key] = ResolvedRouteEvidence(
            deck_hash=selected[0],
            label=selected[1],
            provenance_paths=tuple(sorted({item[2] for item in values}, key=str)),
        )
    return output


def checkpoint_deck_records(
    active: Sequence[str],
    *,
    run_root: Path,
    identity: Mapping[str, Any],
    candidates: Mapping[str, Sequence[HistoricalDeckRecord]],
    route_evidence: Mapping[tuple[str, str], ResolvedRouteEvidence],
    root: Path,
) -> tuple[HistoricalDeckRecord, ...]:
    """Bind exact deck bytes to checkpoint-local authoritative route labels."""
    resolved_path = run_root / "resolved_config.json"
    resolved_routes = _resolved_route_identities(
        resolved_path,
        active=active,
        identity=identity,
    )
    registry_fingerprint = _required_text(identity, "exact_registry_fingerprint")
    output: list[HistoricalDeckRecord] = []
    for digest in active:
        values = tuple(candidates[digest])
        evidence = route_evidence.get((registry_fingerprint, digest))
        local_identity = None if resolved_routes is None else resolved_routes[digest]
        local_hash: str | None = None
        if local_identity is not None:
            try:
                local_hash = extract_authoritative_deck_hash(
                    local_identity[0],
                    explicit=local_identity[1],
                )
            except ValueError:
                local_identity = None
        if (
            local_hash is not None
            and evidence is not None
            and evidence.deck_hash != local_hash
        ):
            raise ValueError(
                "resolved route identity conflicts with registry-matched evidence: "
                f"{digest}"
            )
        if local_identity is not None and local_hash is not None:
            label = local_identity[0]
            deck_hash = local_hash
            evidence_paths: tuple[Path, ...] = ()
        elif evidence is not None:
            label = evidence.label
            deck_hash = evidence.deck_hash
            evidence_paths = evidence.provenance_paths
        else:
            hashes = {item.deck_hash for item in values}
            if len(hashes) != 1:
                raise ValueError(
                    "exact deck has conflicting compact IDs and the checkpoint "
                    f"has no resolved route identity: {digest}"
                )
            selected = min(values, key=lambda item: (str(item.path), item.label))
            output.append(selected)
            continue
        matching = tuple(item for item in values if item.deck_hash == deck_hash)
        selected = min(
            matching or values,
            key=lambda item: (str(item.path), item.label),
        )
        output.append(
            HistoricalDeckRecord(
                deck_digest=digest,
                deck_hash=deck_hash,
                label=label,
                path=selected.path,
                deck_signature=selected.deck_signature,
                provenance_paths=tuple(
                    sorted(
                        {
                            *selected.provenance_paths,
                            _display_path(resolved_path, root=root),
                            *evidence_paths,
                        },
                        key=str,
                    )
                ),
            )
        )
    return tuple(output)


def _resolved_route_identities(
    path: Path,
    *,
    active: Sequence[str],
    identity: Mapping[str, Any],
) -> dict[str, tuple[str, str | None]] | None:
    """Read checkpoint-local deck labels from the immutable resolved config."""
    if not path.is_file():
        return None
    payload = read_json(path)
    expected_config = identity.get("resolved_config_fingerprint")
    if (
        isinstance(expected_config, str)
        and payload.get("resolved_config_fingerprint") != expected_config
    ):
        raise ValueError("resolved config fingerprint differs from checkpoint pair")
    expected_registry = identity.get("exact_registry_fingerprint")
    if (
        isinstance(expected_registry, str)
        and payload.get("exact_registry_fingerprint") != expected_registry
    ):
        raise ValueError("resolved exact registry differs from checkpoint pair")
    raw_routes = payload.get("active_deck_routes")
    if not isinstance(raw_routes, list):
        return None
    routes: dict[str, tuple[str, str | None]] = {}
    for raw in raw_routes:
        if not isinstance(raw, Mapping):
            raise ValueError("resolved active deck route must be an object")
        digest = raw.get("deck_digest")
        label = raw.get("label")
        explicit = raw.get("deck_hash")
        if not isinstance(digest, str) or not isinstance(label, str):
            raise ValueError("resolved active deck route identity is incomplete")
        if explicit is not None and not isinstance(explicit, str):
            raise ValueError("resolved active deck_hash must be text")
        if digest in routes:
            raise ValueError("resolved active deck routes contain duplicates")
        routes[digest] = (label, explicit)
    if set(routes) != set(active):
        raise ValueError("resolved active deck routes differ from checkpoint pair")
    return routes


def _walk_mappings(value: object) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _walk_mappings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_mappings(item)


def _required_text(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"artifact is missing {key}")
    return value.strip()


def _resolve_path(path: Path, *, root: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def _display_path(path: Path, *, root: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root)
    except ValueError:
        return resolved


__all__ = [
    "ResolvedRouteEvidence",
    "checkpoint_deck_records",
    "discover_catalogs",
    "discover_decks",
    "discover_resolved_route_evidence",
    "extract_authoritative_deck_hash",
]
