"""Typed source and resolved registries for exact private deck modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.cards.static_features import DEFAULT_NUM_CARD_IDS
from ptcg_rl.decks.identity import (
    CanonicalDeck,
    canonicalize_deck,
    deck_module_key,
)

_SHA256_HEX_LENGTH = 64
_REGISTRY_V2_SCHEMA = "deck-expert-registry-v2"
_FAMILY_REGISTRY_SCHEMA = "deck-strategy-family-registry-v1"


class PrivateDeckSourceConfig(BaseModel):
    """Human-facing source entry used only while resolving a training run."""

    model_config = ConfigDict(extra="forbid")

    label: str
    path: Path
    expert_id: str | None = None
    family_id: str | None = None

    @field_validator("label")
    @classmethod
    def valid_label(cls, value: str) -> str:
        """Require a non-empty diagnostic label."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("private deck source label must be non-empty")
        return normalized

    @field_validator("expert_id")
    @classmethod
    def valid_optional_expert_id(cls, value: str | None) -> str | None:
        """Normalize an optional stable expert lineage identifier."""
        if value is None:
            return None
        return _validate_expert_id(value)

    @field_validator("family_id")
    @classmethod
    def valid_optional_family_id(cls, value: str | None) -> str | None:
        """Normalize an optional stable strategy-family lineage identifier."""
        if value is None:
            return None
        return _validate_family_id(value)


class PrivateDeckRegistrySourceConfig(BaseModel):
    """Hydra-facing private deck declarations containing source paths."""

    model_config = ConfigDict(extra="forbid")

    decks: tuple[PrivateDeckSourceConfig, ...] = ()

    @model_validator(mode="after")
    def unique_labels(self) -> Self:
        """Reject ambiguous operational labels within one source registry."""
        labels = [entry.label for entry in self.decks]
        if len(labels) != len(set(labels)):
            raise ValueError("private deck source labels must be unique")
        return self


class PrivateDeckProfile(BaseModel):
    """Path-free exact deck identity persisted in a model configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_card_ids: tuple[int, ...]
    signature: str
    deck_digest: str
    module_key: str

    @model_validator(mode="after")
    def consistent_identity(self) -> Self:
        """Require all persisted identity fields to describe the same deck."""
        deck = canonicalize_deck(self.canonical_card_ids)
        if self.canonical_card_ids != deck.card_ids:
            raise ValueError("private profile canonical_card_ids must be sorted")
        if self.signature != deck.signature:
            raise ValueError("private profile signature does not match card IDs")
        if self.deck_digest != deck.deck_digest:
            raise ValueError("private profile deck_digest does not match signature")
        if self.module_key != deck.module_key:
            raise ValueError("private profile module_key does not match digest")
        return self


class ResolvedPrivateDeckRegistry(BaseModel):
    """Deterministically ordered private profiles plus their fingerprint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profiles: tuple[PrivateDeckProfile, ...]
    resolved_registry_sha256: str

    @model_validator(mode="after")
    def consistent_registry(self) -> Self:
        """Validate ordering, uniqueness, and the canonical fingerprint."""
        _validate_profiles(self.profiles)
        if self.resolved_registry_sha256 != private_registry_fingerprint(self.profiles):
            raise ValueError("resolved registry fingerprint does not match profiles")
        return self


class DeckExpertRoute(BaseModel):
    """One exact deck routed to a stable private expert lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_card_ids: tuple[int, ...]
    signature: str
    deck_digest: str
    expert_id: str

    @field_validator("expert_id")
    @classmethod
    def valid_expert_id(cls, value: str) -> str:
        """Require a state-dict-safe full SHA256 expert identifier."""
        return _validate_expert_id(value)

    @model_validator(mode="after")
    def consistent_identity(self) -> Self:
        """Require the exact identity fields to match the canonical cards."""
        deck = canonicalize_deck(self.canonical_card_ids)
        if self.canonical_card_ids != deck.card_ids:
            raise ValueError("expert route canonical_card_ids must be sorted")
        if self.signature != deck.signature:
            raise ValueError("expert route signature does not match card IDs")
        if self.deck_digest != deck.deck_digest:
            raise ValueError("expert route deck_digest does not match signature")
        return self

    @property
    def module_key(self) -> str:
        """Return the stable physical module key for this expert lineage."""
        return deck_module_key(self.expert_id)


class ResolvedDeckExpertRegistry(BaseModel):
    """Deterministic exact-deck routes plus their generation fingerprint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    routes: tuple[DeckExpertRoute, ...]
    resolved_registry_sha256: str

    @model_validator(mode="after")
    def consistent_registry(self) -> Self:
        """Validate route ordering, exact identities, and generation hash."""
        _validate_expert_routes(self.routes)
        if self.resolved_registry_sha256 != deck_expert_registry_fingerprint(
            self.routes
        ):
            raise ValueError(
                "resolved expert registry fingerprint does not match routes"
            )
        return self


class DeckFamilyRoute(BaseModel):
    """Route one exact deck to a shareable private strategy family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_card_ids: tuple[int, ...]
    signature: str
    deck_digest: str
    family_id: str

    @field_validator("family_id")
    @classmethod
    def valid_family_id(cls, value: str) -> str:
        """Require a state-dict-safe full SHA256 family lineage identifier."""
        return _validate_family_id(value)

    @model_validator(mode="after")
    def consistent_identity(self) -> Self:
        """Require the exact identity fields to match the canonical cards."""
        deck = canonicalize_deck(self.canonical_card_ids)
        if self.canonical_card_ids != deck.card_ids:
            raise ValueError("family route canonical_card_ids must be sorted")
        if self.signature != deck.signature:
            raise ValueError("family route signature does not match card IDs")
        if self.deck_digest != deck.deck_digest:
            raise ValueError("family route deck_digest does not match signature")
        return self

    @property
    def module_key(self) -> str:
        """Return the stable physical module key for this family lineage."""
        return deck_module_key(self.family_id)


class ResolvedDeckFamilyRegistry(BaseModel):
    """Deterministic exact-deck-to-family routes and their generation hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    routes: tuple[DeckFamilyRoute, ...]
    resolved_registry_sha256: str

    @model_validator(mode="after")
    def consistent_registry(self) -> Self:
        """Validate route ordering, exact identities, and generation hash."""
        _validate_family_routes(self.routes)
        if self.resolved_registry_sha256 != deck_family_registry_fingerprint(
            self.routes
        ):
            raise ValueError(
                "resolved family registry fingerprint does not match routes"
            )
        return self


def private_deck_profile(deck: CanonicalDeck) -> PrivateDeckProfile:
    """Build a path-free resolved profile from a canonical deck."""
    return PrivateDeckProfile(
        canonical_card_ids=deck.card_ids,
        signature=deck.signature,
        deck_digest=deck.deck_digest,
        module_key=deck.module_key,
    )


def deck_expert_route(
    deck: CanonicalDeck,
    *,
    expert_id: str | None = None,
) -> DeckExpertRoute:
    """Build a path-free v2 route, starting a new lineage by default."""
    return DeckExpertRoute(
        canonical_card_ids=deck.card_ids,
        signature=deck.signature,
        deck_digest=deck.deck_digest,
        expert_id=deck.deck_digest if expert_id is None else expert_id,
    )


def deck_family_route(
    deck: CanonicalDeck,
    *,
    family_id: str,
) -> DeckFamilyRoute:
    """Build one path-free exact-deck-to-family route."""
    return DeckFamilyRoute(
        canonical_card_ids=deck.card_ids,
        signature=deck.signature,
        deck_digest=deck.deck_digest,
        family_id=family_id,
    )


def resolve_private_registry(
    source: PrivateDeckRegistrySourceConfig,
    *,
    base_dir: Path | None = None,
    max_card_id: int = DEFAULT_NUM_CARD_IDS,
) -> ResolvedPrivateDeckRegistry:
    """Resolve source files into a deterministic, path-free registry."""
    profiles: list[PrivateDeckProfile] = []
    for entry in source.decks:
        if entry.expert_id is not None or entry.family_id is not None:
            raise ValueError(
                "v1 private registry entries cannot declare expert_id or family_id"
            )
        path = entry.path if base_dir is None else base_dir / entry.path
        cards = _read_deck_file(path)
        profiles.append(
            private_deck_profile(canonicalize_deck(cards, max_card_id=max_card_id))
        )
    ordered = tuple(sorted(profiles, key=lambda profile: profile.module_key))
    _validate_profiles(ordered)
    return ResolvedPrivateDeckRegistry(
        profiles=ordered,
        resolved_registry_sha256=private_registry_fingerprint(ordered),
    )


def resolve_deck_expert_registry(
    source: PrivateDeckRegistrySourceConfig,
    *,
    base_dir: Path | None = None,
    max_card_id: int = DEFAULT_NUM_CARD_IDS,
) -> ResolvedDeckExpertRegistry:
    """Resolve deck files into deterministic v2 exact-to-expert routes."""
    routes: list[DeckExpertRoute] = []
    for entry in source.decks:
        path = entry.path if base_dir is None else base_dir / entry.path
        deck = canonicalize_deck(
            _read_deck_file(path),
            max_card_id=max_card_id,
        )
        routes.append(deck_expert_route(deck, expert_id=entry.expert_id))
    ordered = tuple(sorted(routes, key=lambda route: route.deck_digest))
    _validate_expert_routes(ordered)
    return ResolvedDeckExpertRegistry(
        routes=ordered,
        resolved_registry_sha256=deck_expert_registry_fingerprint(ordered),
    )


def resolve_deck_family_registry(
    source: PrivateDeckRegistrySourceConfig,
    *,
    base_dir: Path | None = None,
    max_card_id: int = DEFAULT_NUM_CARD_IDS,
) -> ResolvedDeckFamilyRegistry:
    """Resolve deck files into deterministic exact-to-family routes."""
    routes: list[DeckFamilyRoute] = []
    for entry in source.decks:
        if entry.family_id is None:
            raise ValueError("family-private registry entries require family_id")
        path = entry.path if base_dir is None else base_dir / entry.path
        deck = canonicalize_deck(
            _read_deck_file(path),
            max_card_id=max_card_id,
        )
        routes.append(deck_family_route(deck, family_id=entry.family_id))
    ordered = tuple(sorted(routes, key=lambda route: route.deck_digest))
    _validate_family_routes(ordered)
    return ResolvedDeckFamilyRegistry(
        routes=ordered,
        resolved_registry_sha256=deck_family_registry_fingerprint(ordered),
    )


def validate_active_exact_strategy_routes(
    routes: tuple[DeckExpertRoute, ...],
) -> None:
    """Require one physical expert lineage for every active exact deck.

    Historical artifact schemas may contain shared expert lineages and must
    remain loadable. Trainable active rosters use this stricter lifecycle gate
    at launch instead of changing the meaning of those immutable registries.
    """
    for field_name in ("expert_id", "module_key"):
        values = tuple(getattr(route, field_name) for route in routes)
        if len(values) != len(set(values)):
            raise ValueError(
                f"active exact strategy {field_name} values must be unique"
            )


def private_registry_fingerprint(
    profiles: tuple[PrivateDeckProfile, ...],
) -> str:
    """Hash the canonical structured payload for sorted resolved profiles."""
    import hashlib

    ordered = tuple(sorted(profiles, key=lambda profile: profile.module_key))
    payload = {
        "profiles": [
            {
                "canonical_card_ids": list(profile.canonical_card_ids),
                "deck_digest": profile.deck_digest,
                "module_key": profile.module_key,
                "signature": profile.signature,
            }
            for profile in ordered
        ]
    }
    serialized = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def deck_expert_registry_fingerprint(
    routes: tuple[DeckExpertRoute, ...],
) -> str:
    """Hash the canonical v2 exact-deck-to-expert routing generation."""
    import hashlib

    ordered = tuple(sorted(routes, key=lambda route: route.deck_digest))
    payload = {
        "schema": _REGISTRY_V2_SCHEMA,
        "expert_ids": sorted({route.expert_id for route in ordered}),
        "routes": [
            {
                "canonical_card_ids": list(route.canonical_card_ids),
                "deck_digest": route.deck_digest,
                "expert_id": route.expert_id,
                "module_key": route.module_key,
                "signature": route.signature,
            }
            for route in ordered
        ],
    }
    serialized = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def deck_family_registry_fingerprint(
    routes: tuple[DeckFamilyRoute, ...],
) -> str:
    """Hash the canonical exact-deck-to-family routing generation."""
    import hashlib

    ordered = tuple(sorted(routes, key=lambda route: route.deck_digest))
    payload = {
        "schema": _FAMILY_REGISTRY_SCHEMA,
        "family_ids": sorted({route.family_id for route in ordered}),
        "routes": [
            {
                "canonical_card_ids": list(route.canonical_card_ids),
                "deck_digest": route.deck_digest,
                "family_id": route.family_id,
                "module_key": route.module_key,
                "signature": route.signature,
            }
            for route in ordered
        ],
    }
    serialized = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_profiles(profiles: tuple[PrivateDeckProfile, ...]) -> None:
    module_keys = tuple(profile.module_key for profile in profiles)
    if module_keys != tuple(sorted(module_keys)):
        raise ValueError("private profiles must be sorted by module_key")
    for field_name in ("signature", "deck_digest", "module_key"):
        values = [getattr(profile, field_name) for profile in profiles]
        if len(values) != len(set(values)):
            raise ValueError(f"private profile {field_name} values must be unique")


def _validate_expert_routes(routes: tuple[DeckExpertRoute, ...]) -> None:
    deck_digests = tuple(route.deck_digest for route in routes)
    if deck_digests != tuple(sorted(deck_digests)):
        raise ValueError("expert routes must be sorted by deck_digest")
    for field_name in ("signature", "deck_digest"):
        values = [getattr(route, field_name) for route in routes]
        if len(values) != len(set(values)):
            raise ValueError(f"expert route {field_name} values must be unique")


def _validate_family_routes(routes: tuple[DeckFamilyRoute, ...]) -> None:
    deck_digests = tuple(route.deck_digest for route in routes)
    if deck_digests != tuple(sorted(deck_digests)):
        raise ValueError("family routes must be sorted by deck_digest")
    for field_name in ("signature", "deck_digest"):
        values = [getattr(route, field_name) for route in routes]
        if len(values) != len(set(values)):
            raise ValueError(f"family route {field_name} values must be unique")


def _validate_expert_id(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("expert_id must be a 64-character lowercase SHA256")
    return normalized


def _validate_family_id(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("family_id must be a 64-character lowercase SHA256")
    return normalized


def _read_deck_file(path: Path) -> tuple[int, ...]:
    values: list[int] = []
    with path.open(encoding="utf-8", newline="") as file_obj:
        for raw_line in file_obj:
            value = raw_line.strip().strip(",")
            if value:
                values.append(int(value))
    return tuple(values)
