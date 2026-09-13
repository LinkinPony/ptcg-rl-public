"""Versioned identities and external records for exact engine consequences."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from enum import IntEnum, StrEnum
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

COMPACT_CONSEQUENCE_SCHEMA_VERSION: Literal[1] = 1
ROOT_OBSERVATION_ENCODING: Literal["root_observable_numeric_json_v1"] = (
    "root_observable_numeric_json_v1"
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_UINT64 = int(np.iinfo(np.uint64).max)
_WEIGHT_SUM_TOLERANCE = 1.0e-6
_SCENARIO_DOMAIN = b"ptcg-rl/compact-consequence/scenario/v1\x00"
_SUPPORT_DOMAIN = b"ptcg-rl/compact-consequence/scenario-support/v1\x00"
_CELL_IDENTITY_DOMAIN = b"ptcg-rl/compact-consequence/cell-identity/v1\x00"


class SemanticEndpoint(IntEnum):
    """Semantic boundary reached by an exact decision transition.

    ``ROOT_STRATEGIC_PROMPT`` and ``CHANCE_PROMPT`` are continuation
    boundaries, not directly comparable completed option outcomes.
    """

    INVALID = 0
    TERMINAL = 1
    SAME_SEAT_MAIN = 2
    TURN_HANDOFF = 3
    ROOT_STRATEGIC_PROMPT = 4
    CHANCE_PROMPT = 5


class ScenarioSupportMode(StrEnum):
    """Supported relationship between belief worlds and engine chance."""

    SAMPLED_BELIEF_NO_CHANCE = "sampled_belief_no_chance"
    SAMPLED_BELIEF_MANUAL_COIN_ENUMERATED = (
        "sampled_belief_manual_coin_enumerated"
    )
    SAMPLED_BELIEF_CHANCE_UNSUPPORTED = "sampled_belief_chance_unsupported"


class ScenarioHandle(BaseModel):
    """One opaque engine scenario handle and its immutable content identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    belief_world_handle: int = Field(ge=0, le=_MAX_UINT64)
    chance_support_handle: int = Field(ge=0, le=_MAX_UINT64)
    belief_world_fingerprint: str
    chance_support_fingerprint: str
    scenario_fingerprint: str
    weight: float = Field(gt=0.0)

    @field_validator(
        "belief_world_fingerprint",
        "chance_support_fingerprint",
        "scenario_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require canonical lowercase SHA-256 identities."""
        return _validate_sha256(value)

    @field_validator("weight")
    @classmethod
    def finite_weight(cls, value: float) -> float:
        """Reject NaN and infinite scenario mass."""
        if not math.isfinite(value):
            raise ValueError("scenario weight must be finite")
        return value

    @model_validator(mode="after")
    def identity_matches_components(self) -> Self:
        """Bind the scenario identity to belief and chance identities."""
        expected = scenario_fingerprint(
            belief_world_fingerprint=self.belief_world_fingerprint,
            chance_support_fingerprint=self.chance_support_fingerprint,
        )
        if self.scenario_fingerprint != expected:
            raise ValueError("scenario fingerprint does not match its components")
        return self


class ScenarioSupport(BaseModel):
    """Ordered, normalized paired support shared by every retained candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ScenarioSupportMode
    scenarios: tuple[ScenarioHandle, ...]
    support_fingerprint: str

    @field_validator("support_fingerprint")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require a canonical support SHA-256 identity."""
        return _validate_sha256(value)

    @model_validator(mode="after")
    def valid_paired_support(self) -> Self:
        """Reject empty, duplicate, unnormalized, or misidentified support."""
        if not self.scenarios:
            raise ValueError("scenario support must not be empty")
        handle_pairs = tuple(
            (item.belief_world_handle, item.chance_support_handle)
            for item in self.scenarios
        )
        if len(set(handle_pairs)) != len(handle_pairs):
            raise ValueError("scenario handle pairs must be unique")
        fingerprints = tuple(item.scenario_fingerprint for item in self.scenarios)
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("scenario fingerprints must be unique")
        if not math.isclose(
            math.fsum(item.weight for item in self.scenarios),
            1.0,
            rel_tol=0.0,
            abs_tol=_WEIGHT_SUM_TOLERANCE,
        ):
            raise ValueError("scenario weights must be normalized to one")
        expected = scenario_support_fingerprint(self.mode, self.scenarios)
        if self.support_fingerprint != expected:
            raise ValueError("scenario support fingerprint does not match its rows")
        return self


class ConsequenceExactnessFlags(BaseModel):
    """Independent facts that must never be collapsed into one ``exact`` flag."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rules_exact: bool
    support_exhaustive: bool
    scenario_grid_complete: bool
    leaf_bootstrapped: bool


class RootCandidateScenarioIdentity(BaseModel):
    """Immutable root/candidate/scenario identity for one consequence cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = COMPACT_CONSEQUENCE_SCHEMA_VERSION
    candidate_index: int = Field(ge=0)
    scenario_index: int = Field(ge=0)
    root_state_fingerprint: str
    candidate_fingerprint: str
    scenario_fingerprint: str
    scenario_support_fingerprint: str
    identity_fingerprint: str

    @field_validator(
        "root_state_fingerprint",
        "candidate_fingerprint",
        "scenario_fingerprint",
        "scenario_support_fingerprint",
        "identity_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require canonical lowercase SHA-256 identities."""
        return _validate_sha256(value)

    @model_validator(mode="after")
    def identity_matches_components(self) -> Self:
        """Reject a cell identity assembled from mismatched artifacts."""
        expected = root_candidate_scenario_fingerprint(
            root_state_fingerprint=self.root_state_fingerprint,
            candidate_fingerprint=self.candidate_fingerprint,
            scenario_fingerprint=self.scenario_fingerprint,
            scenario_support_fingerprint=self.scenario_support_fingerprint,
        )
        if self.identity_fingerprint != expected:
            raise ValueError("cell identity fingerprint does not match its components")
        return self


class CompactConsequenceMetadata(BaseModel):
    """Validated external metadata for one compact consequence buffer set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = COMPACT_CONSEQUENCE_SCHEMA_VERSION
    observation_encoding: Literal["root_observable_numeric_json_v1"] = (
        ROOT_OBSERVATION_ENCODING
    )
    observation_schema_fingerprint: str
    root_state_fingerprint: str
    candidate_fingerprints: tuple[str, ...]
    legal_action_count: int = Field(ge=1)
    scenario_support: ScenarioSupport
    support_exhaustive: bool
    scenario_grid_complete: bool

    @field_validator(
        "observation_schema_fingerprint",
        "root_state_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require canonical lowercase SHA-256 identities."""
        return _validate_sha256(value)

    @field_validator("candidate_fingerprints")
    @classmethod
    def valid_candidate_fingerprints(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty, duplicate, or malformed candidate identities."""
        if not values:
            raise ValueError("candidate fingerprints must not be empty")
        canonical = tuple(_validate_sha256(value) for value in values)
        if len(set(canonical)) != len(canonical):
            raise ValueError("candidate fingerprints must be unique")
        return canonical

    @model_validator(mode="after")
    def valid_support_coverage(self) -> Self:
        """Make root support exhaustiveness explicit and internally consistent."""
        candidate_count = len(self.candidate_fingerprints)
        if self.legal_action_count < candidate_count:
            raise ValueError("legal action count cannot be below candidate count")
        expected_exhaustive = self.legal_action_count == candidate_count
        if self.support_exhaustive != expected_exhaustive:
            raise ValueError(
                "support_exhaustive must match retained versus legal action count"
            )
        return self


def scenario_fingerprint(
    *,
    belief_world_fingerprint: str,
    chance_support_fingerprint: str,
) -> str:
    """Fingerprint one belief-world/chance-support content pair."""
    belief = bytes.fromhex(_validate_sha256(belief_world_fingerprint))
    chance = bytes.fromhex(_validate_sha256(chance_support_fingerprint))
    return hashlib.sha256(_SCENARIO_DOMAIN + belief + chance).hexdigest()


def scenario_support_fingerprint(
    mode: ScenarioSupportMode,
    scenarios: tuple[ScenarioHandle, ...],
) -> str:
    """Fingerprint ordered scenario identities, mode, and normalized weights."""
    digest = hashlib.sha256()
    digest.update(_SUPPORT_DOMAIN)
    digest.update(mode.value.encode("ascii"))
    digest.update(struct.pack(">I", len(scenarios)))
    for item in scenarios:
        digest.update(bytes.fromhex(item.scenario_fingerprint))
        digest.update(struct.pack(">d", item.weight))
    return digest.hexdigest()


def root_candidate_scenario_fingerprint(
    *,
    root_state_fingerprint: str,
    candidate_fingerprint: str,
    scenario_fingerprint: str,
    scenario_support_fingerprint: str,
) -> str:
    """Fingerprint all immutable identities required to consume one cell."""
    components = (
        root_state_fingerprint,
        candidate_fingerprint,
        scenario_fingerprint,
        scenario_support_fingerprint,
    )
    payload = b"".join(bytes.fromhex(_validate_sha256(item)) for item in components)
    return hashlib.sha256(_CELL_IDENTITY_DOMAIN + payload).hexdigest()


def _validate_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("fingerprints must be lowercase SHA-256 hex strings")
    return value


__all__ = [
    "COMPACT_CONSEQUENCE_SCHEMA_VERSION",
    "ROOT_OBSERVATION_ENCODING",
    "CompactConsequenceMetadata",
    "ConsequenceExactnessFlags",
    "RootCandidateScenarioIdentity",
    "ScenarioHandle",
    "ScenarioSupport",
    "ScenarioSupportMode",
    "SemanticEndpoint",
    "root_candidate_scenario_fingerprint",
    "scenario_fingerprint",
    "scenario_support_fingerprint",
]
