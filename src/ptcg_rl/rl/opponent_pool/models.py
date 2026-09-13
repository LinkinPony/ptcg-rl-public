"""Immutable identities and bounded active revisions for opponent-pool RL."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptcg_rl.rl.opponent_pool._identity import (
    Sha256,
    cached_fingerprint,
    canonical_fingerprint,
    normalize_sha256,
)

SemanticRole = Literal[
    "sentinel",
    "champion",
    "retired_champion",
    "counter_train",
    "frontier",
    "recent",
    "age_landmark",
    "probe",
]
RuntimeKind = Literal["legacy_resident", "wire_bf16"]
QuotaStratum = Literal[
    "protected",
    "counter_frontier",
    "recent",
    "age_diverse",
    "probe_reentry",
]
CandidateSeat = Literal[0, 1]

STRATUM_ORDER: tuple[QuotaStratum, ...] = (
    "protected",
    "counter_frontier",
    "recent",
    "age_diverse",
    "probe_reentry",
)


class OpponentArtifact(BaseModel):
    """One immutable executable policy artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: Sha256
    runtime_kind: RuntimeKind
    source_fingerprint: Sha256
    execution_fingerprint: Sha256
    compatibility_fingerprint: Sha256
    source_policy_version: int = Field(ge=0)

    @classmethod
    def from_content(
        cls,
        *,
        runtime_kind: RuntimeKind,
        source_fingerprint: str,
        execution_fingerprint: str,
        compatibility_fingerprint: str,
        source_policy_version: int,
    ) -> Self:
        """Create an artifact whose ID commits to all executable content."""
        normalized_source = normalize_sha256(source_fingerprint)
        normalized_execution = normalize_sha256(execution_fingerprint)
        normalized_compatibility = normalize_sha256(compatibility_fingerprint)
        content = {
            "runtime_kind": runtime_kind,
            "source_fingerprint": normalized_source,
            "execution_fingerprint": normalized_execution,
            "compatibility_fingerprint": normalized_compatibility,
            "source_policy_version": source_policy_version,
        }
        return cls(
            artifact_id=canonical_fingerprint("opponent-artifact", content),
            runtime_kind=runtime_kind,
            source_fingerprint=normalized_source,
            execution_fingerprint=normalized_execution,
            compatibility_fingerprint=normalized_compatibility,
            source_policy_version=source_policy_version,
        )

    @model_validator(mode="after")
    def identity_matches_content(self) -> Self:
        """Reject aliases that reuse an ID for different executable content."""
        if self.artifact_id != opponent_artifact_fingerprint(self):
            raise ValueError("artifact_id does not match artifact content")
        return self


def opponent_artifact_fingerprint(artifact: OpponentArtifact) -> str:
    """Return the immutable artifact content identity."""
    return canonical_fingerprint(
        "opponent-artifact",
        artifact.model_dump(mode="json", exclude={"artifact_id"}),
    )


class OpponentRoute(BaseModel):
    """An immutable exact-deck route into one executable artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route_id: Sha256
    artifact_id: Sha256
    exact_deck_digest: Sha256
    route_contract_fingerprint: Sha256

    @classmethod
    def from_content(
        cls,
        *,
        artifact_id: str,
        exact_deck_digest: str,
        route_contract_fingerprint: str,
    ) -> Self:
        """Create a route whose ID commits to artifact, deck, and contract."""
        content = {
            "artifact_id": normalize_sha256(artifact_id),
            "exact_deck_digest": normalize_sha256(exact_deck_digest),
            "route_contract_fingerprint": normalize_sha256(route_contract_fingerprint),
        }
        return cls(
            route_id=canonical_fingerprint("opponent-route", content),
            **content,
        )

    @model_validator(mode="after")
    def identity_matches_content(self) -> Self:
        """Reject route aliases."""
        if self.route_id != opponent_route_fingerprint(self):
            raise ValueError("route_id does not match route content")
        return self


def opponent_route_fingerprint(route: OpponentRoute) -> str:
    """Return the immutable route content identity."""
    return canonical_fingerprint(
        "opponent-route",
        route.model_dump(mode="json", exclude={"route_id"}),
    )


class ActiveMatchup(BaseModel):
    """One schedulable candidate-deck, opponent-route, and seat cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_deck_digest: Sha256
    route_id: Sha256
    candidate_seat: CandidateSeat

    @property
    def key(self) -> tuple[str, str, int]:
        """Return the canonical cell key."""
        return (
            self.candidate_deck_digest,
            self.route_id,
            self.candidate_seat,
        )


class PoolEntry(BaseModel):
    """Active scheduling metadata for exactly one artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: Sha256
    semantic_roles: tuple[SemanticRole, ...]
    stratum: QuotaStratum
    admission_generation: int = Field(ge=0)

    @model_validator(mode="after")
    def roles_are_canonical(self) -> Self:
        """Keep role membership non-empty, unique, and deterministic."""
        if not self.semantic_roles:
            raise ValueError("an active artifact needs at least one semantic role")
        if self.semantic_roles != tuple(sorted(set(self.semantic_roles))):
            raise ValueError("semantic_roles must be sorted and unique")
        return self


class PoolRevision(BaseModel):
    """A complete, bounded snapshot of the currently schedulable pool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    revision_sequence: int = Field(ge=0)
    maximum_active_artifacts: int = Field(gt=0)
    compatibility_target_fingerprint: Sha256
    artifacts: tuple[OpponentArtifact, ...]
    routes: tuple[OpponentRoute, ...]
    entries: tuple[PoolEntry, ...]
    active_matchups: tuple[ActiveMatchup, ...]

    @model_validator(mode="after")
    def active_graph_is_closed_and_bounded(self) -> Self:
        """Validate the closed active graph and the compatibility gate."""
        artifact_ids = tuple(item.artifact_id for item in self.artifacts)
        if not artifact_ids:
            raise ValueError("an active revision must contain artifacts")
        if artifact_ids != tuple(sorted(artifact_ids)):
            raise ValueError("artifacts must be sorted by artifact_id")
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("artifact IDs must be unique")
        if len(artifact_ids) > self.maximum_active_artifacts:
            raise ValueError("active revision exceeds its artifact bound")
        if any(
            artifact.compatibility_fingerprint != self.compatibility_target_fingerprint
            for artifact in self.artifacts
        ):
            raise ValueError("artifact fails the revision compatibility gate")

        route_ids = tuple(item.route_id for item in self.routes)
        if route_ids != tuple(sorted(route_ids)):
            raise ValueError("routes must be sorted by route_id")
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("route IDs must be unique")
        artifact_id_set = set(artifact_ids)
        if any(route.artifact_id not in artifact_id_set for route in self.routes):
            raise ValueError("route references an artifact outside the revision")
        routed_artifacts = {route.artifact_id for route in self.routes}
        if routed_artifacts != artifact_id_set:
            raise ValueError("every active artifact must have an active route")

        entry_ids = tuple(item.artifact_id for item in self.entries)
        if entry_ids != tuple(sorted(entry_ids)):
            raise ValueError("entries must be sorted by artifact_id")
        if len(set(entry_ids)) != len(entry_ids):
            raise ValueError("pool entries must be unique")
        if set(entry_ids) != artifact_id_set:
            raise ValueError("pool entries must exactly cover active artifacts")

        matchup_keys = tuple(item.key for item in self.active_matchups)
        if matchup_keys != tuple(sorted(matchup_keys)):
            raise ValueError("active matchups must be sorted by cell key")
        if len(set(matchup_keys)) != len(matchup_keys):
            raise ValueError("active matchup cells must be unique")
        route_id_set = set(route_ids)
        if any(
            matchup.route_id not in route_id_set for matchup in self.active_matchups
        ):
            raise ValueError("active matchup references an inactive route")
        matched_artifacts = {
            route.artifact_id
            for route in self.routes
            if route.route_id in {matchup.route_id for matchup in self.active_matchups}
        }
        if matched_artifacts != artifact_id_set:
            raise ValueError("every active artifact needs an active matchup")
        return self

    @property
    def fingerprint(self) -> str:
        """Return the revision identity used by planning stale gates."""
        return cached_fingerprint(
            "pool-revision",
            self,
            lambda: pool_revision_fingerprint(self),
        )


def pool_revision_fingerprint(revision: PoolRevision) -> str:
    """Fingerprint one complete active revision."""
    return canonical_fingerprint(
        "pool-revision",
        revision.model_dump(mode="json"),
    )
