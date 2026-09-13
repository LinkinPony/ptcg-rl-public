"""Contracts for local engine transitions and hierarchical option outcomes."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from dataclasses import dataclass
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.agent.search.planner_fallback import (
    PlannerEvidenceError,
    PlannerFallbackReason,
)
from ptcg_rl.engine.compact_consequence import ScenarioSupport, SemanticEndpoint

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTROLLER_DOMAIN = b"ptcg-rl/deployment-continuation-controller/v1\x00"
_STABLE_CONTROLLER_DOMAIN = b"ptcg-rl/deployment-continuation-controller/v2\x00"
STABLE_CONTINUATION_CONTROLLER_IDENTITY_SCHEMA_VERSION: Final[Literal[2]] = 2
_COMPARABLE_ENDPOINTS = frozenset(
    {
        SemanticEndpoint.TERMINAL,
        SemanticEndpoint.SAME_SEAT_MAIN,
        SemanticEndpoint.TURN_HANDOFF,
    }
)


class DeploymentContinuationControllerIdentity(BaseModel):
    """Immutable identity of the controller used after the root action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    controller_version: str
    model_fingerprint: str
    constructor_fingerprint: str
    scorer_fingerprint: str
    resolved_config_fingerprint: str
    controller_fingerprint: str

    @field_validator("controller_version")
    @classmethod
    def nonempty_version(cls, value: str) -> str:
        """Reject ambiguous empty controller versions."""
        canonical = value.strip()
        if not canonical:
            raise ValueError("controller_version must not be empty")
        return canonical

    @field_validator(
        "model_fingerprint",
        "constructor_fingerprint",
        "scorer_fingerprint",
        "resolved_config_fingerprint",
        "controller_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require canonical SHA-256 identities."""
        if _SHA256.fullmatch(value) is None:
            raise ValueError("fingerprints must be lowercase SHA-256 hex")
        return value

    @model_validator(mode="after")
    def identity_matches_fields(self) -> Self:
        """Bind the public controller identity to every semantic dependency."""
        expected = continuation_controller_fingerprint(
            controller_version=self.controller_version,
            model_fingerprint=self.model_fingerprint,
            constructor_fingerprint=self.constructor_fingerprint,
            scorer_fingerprint=self.scorer_fingerprint,
            resolved_config_fingerprint=self.resolved_config_fingerprint,
        )
        if self.controller_fingerprint != expected:
            raise ValueError("controller fingerprint does not match its fields")
        return self

    @classmethod
    def create(
        cls,
        *,
        controller_version: str,
        model_fingerprint: str,
        constructor_fingerprint: str,
        scorer_fingerprint: str,
        resolved_config_fingerprint: str,
    ) -> DeploymentContinuationControllerIdentity:
        """Construct a content-bound deployment controller identity."""
        fingerprint = continuation_controller_fingerprint(
            controller_version=controller_version.strip(),
            model_fingerprint=model_fingerprint,
            constructor_fingerprint=constructor_fingerprint,
            scorer_fingerprint=scorer_fingerprint,
            resolved_config_fingerprint=resolved_config_fingerprint,
        )
        return cls(
            controller_version=controller_version,
            model_fingerprint=model_fingerprint,
            constructor_fingerprint=constructor_fingerprint,
            scorer_fingerprint=scorer_fingerprint,
            resolved_config_fingerprint=resolved_config_fingerprint,
            controller_fingerprint=fingerprint,
        )


class StableContinuationControllerIdentity(BaseModel):
    """Stable continuation semantics independent of a model publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity_schema_version: Literal[2] = (
        STABLE_CONTINUATION_CONTROLLER_IDENTITY_SCHEMA_VERSION
    )
    controller_version: str
    constructor_fingerprint: str
    scorer_fingerprint: str
    continuation_semantics_fingerprint: str
    controller_fingerprint: str

    @field_validator("controller_version")
    @classmethod
    def nonempty_stable_version(cls, value: str) -> str:
        """Canonicalize the stable controller implementation version."""
        canonical = value.strip()
        if not canonical:
            raise ValueError("controller_version must not be empty")
        return canonical

    @field_validator(
        "constructor_fingerprint",
        "scorer_fingerprint",
        "continuation_semantics_fingerprint",
        "controller_fingerprint",
    )
    @classmethod
    def valid_stable_fingerprint(cls, value: str) -> str:
        """Require canonical SHA-256 identities."""
        return _require_fingerprint(value)

    @model_validator(mode="after")
    def stable_identity_matches_fields(self) -> Self:
        """Bind the public v2 identity to stable semantic dependencies."""
        expected = stable_continuation_controller_fingerprint(
            controller_version=self.controller_version,
            constructor_fingerprint=self.constructor_fingerprint,
            scorer_fingerprint=self.scorer_fingerprint,
            continuation_semantics_fingerprint=(
                self.continuation_semantics_fingerprint
            ),
        )
        if self.controller_fingerprint != expected:
            raise ValueError("stable controller fingerprint does not match its fields")
        return self

    @classmethod
    def create(
        cls,
        *,
        controller_version: str,
        constructor_fingerprint: str,
        scorer_fingerprint: str,
        continuation_semantics_fingerprint: str,
    ) -> StableContinuationControllerIdentity:
        """Construct a content-bound stable deployment controller identity."""
        fingerprint = stable_continuation_controller_fingerprint(
            controller_version=controller_version,
            constructor_fingerprint=constructor_fingerprint,
            scorer_fingerprint=scorer_fingerprint,
            continuation_semantics_fingerprint=continuation_semantics_fingerprint,
        )
        return cls(
            controller_version=controller_version,
            constructor_fingerprint=constructor_fingerprint,
            scorer_fingerprint=scorer_fingerprint,
            continuation_semantics_fingerprint=continuation_semantics_fingerprint,
            controller_fingerprint=fingerprint,
        )


@dataclass(frozen=True, slots=True)
class DecisionTransition:
    """One complete selection advanced to the next non-forced boundary."""

    candidate_index: int
    scenario_index: int
    candidate_fingerprint: str
    scenario_fingerprint: str
    scenario_support_fingerprint: str
    source_information_history_fingerprint: str
    reached_information_history_fingerprint: str
    endpoint: SemanticEndpoint
    transition_steps: int
    rules_exact: bool
    error_code: int = 0

    def __post_init__(self) -> None:
        """Reject malformed local transition evidence."""
        if self.candidate_index < 0 or self.scenario_index < 0:
            raise ValueError("transition indices must be non-negative")
        for value in (
            self.candidate_fingerprint,
            self.scenario_fingerprint,
            self.scenario_support_fingerprint,
            self.source_information_history_fingerprint,
            self.reached_information_history_fingerprint,
        ):
            _require_fingerprint(value)
        if not isinstance(self.endpoint, SemanticEndpoint):
            raise TypeError("endpoint must be SemanticEndpoint")
        if self.endpoint is SemanticEndpoint.INVALID:
            raise ValueError("decision transitions cannot use INVALID endpoint")
        if self.transition_steps <= 0:
            raise ValueError("decision transitions must execute at least one step")
        if not isinstance(self.rules_exact, bool):
            raise TypeError("rules_exact must be bool")
        if self.error_code < 0:
            raise ValueError("error_code must be non-negative")

    @property
    def comparable(self) -> bool:
        """Whether this local transition already reached an option boundary."""
        return (
            self.error_code == 0
            and self.rules_exact
            and self.endpoint in _COMPARABLE_ENDPOINTS
        )


@dataclass(frozen=True, slots=True)
class ContinuationDecision:
    """One controller choice made from a root-observable information history."""

    scenario_index: int
    information_history_fingerprint: str
    action_fingerprint: str

    def __post_init__(self) -> None:
        """Validate public information-set and action identities."""
        if self.scenario_index < 0:
            raise ValueError("scenario_index must be non-negative")
        _require_fingerprint(self.information_history_fingerprint)
        _require_fingerprint(self.action_fingerprint)


@dataclass(frozen=True, slots=True)
class OptionOutcomeCell:
    """Final, comparable result of one option under one paired scenario."""

    scenario_index: int
    scenario_fingerprint: str
    weight: float
    endpoint: SemanticEndpoint
    information_history_fingerprint: str
    leaf_index: int | None
    terminal_result: float | None
    rules_exact: bool = True

    def __post_init__(self) -> None:
        """Require exact terminal or root-information value inputs."""
        if self.scenario_index < 0:
            raise ValueError("scenario_index must be non-negative")
        _require_fingerprint(self.scenario_fingerprint)
        _require_fingerprint(self.information_history_fingerprint)
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("scenario weight must be positive and finite")
        if not isinstance(self.endpoint, SemanticEndpoint):
            raise TypeError("endpoint must be SemanticEndpoint")
        if self.endpoint not in _COMPARABLE_ENDPOINTS:
            raise ValueError("option outcomes require a comparable endpoint")
        if not isinstance(self.rules_exact, bool):
            raise TypeError("rules_exact must be bool")
        if self.endpoint is SemanticEndpoint.TERMINAL:
            if self.leaf_index is not None:
                raise ValueError("terminal outcomes must not reference a value leaf")
            if isinstance(self.terminal_result, bool) or self.terminal_result not in (
                -1.0,
                0.0,
                1.0,
            ):
                raise ValueError("terminal_result must be root-perspective W/D/L")
        else:
            if self.leaf_index is None or self.leaf_index < 0:
                raise ValueError("nonterminal outcomes require a value leaf index")
            if self.terminal_result is not None:
                raise ValueError("nonterminal outcomes cannot carry terminal_result")


@dataclass(frozen=True, slots=True)
class OptionOutcome:
    """Paired-scenario outcome distribution for one hierarchical root option."""

    root_information_history_fingerprint: str
    root_candidate_fingerprint: str
    scenario_support: ScenarioSupport
    controller: StableContinuationControllerIdentity
    cells: tuple[OptionOutcomeCell, ...]
    continuation_trace_complete: bool
    continuation_decisions: tuple[ContinuationDecision, ...] = ()

    def __post_init__(self) -> None:
        """Validate complete support and information-set nonanticipativity."""
        _require_fingerprint(self.root_information_history_fingerprint)
        _require_fingerprint(self.root_candidate_fingerprint)
        if not isinstance(self.scenario_support, ScenarioSupport):
            raise TypeError("scenario_support must be ScenarioSupport")
        if not isinstance(self.continuation_trace_complete, bool):
            raise TypeError("continuation_trace_complete must be bool")
        if not self.continuation_trace_complete:
            raise PlannerEvidenceError(
                PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                "option outcome has an incomplete continuation trace",
            )
        if not self.cells:
            raise PlannerEvidenceError(
                PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                "option outcome has no scenario cells",
            )
        indices = tuple(cell.scenario_index for cell in self.cells)
        if indices != tuple(range(len(self.cells))):
            raise PlannerEvidenceError(
                PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                "option outcome cells must cover ordered scenario indices",
            )
        if len(self.cells) != len(self.scenario_support.scenarios):
            raise PlannerEvidenceError(
                PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                "option outcome cells do not cover their declared support",
            )
        for cell, scenario in zip(
            self.cells,
            self.scenario_support.scenarios,
            strict=True,
        ):
            if (
                cell.scenario_fingerprint != scenario.scenario_fingerprint
                or cell.weight != scenario.weight
            ):
                raise PlannerEvidenceError(
                    PlannerFallbackReason.FINGERPRINT_MISMATCH,
                    "option outcome cell differs from its declared support row",
                )
        if not math.isclose(
            math.fsum(cell.weight for cell in self.cells),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise PlannerEvidenceError(
                PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                "option outcome scenario weights must sum to one",
            )
        scenario_fingerprints = tuple(
            cell.scenario_fingerprint for cell in self.cells
        )
        if len(set(scenario_fingerprints)) != len(scenario_fingerprints):
            raise PlannerEvidenceError(
                PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                "option outcome contains duplicate scenario identities",
            )
        if any(not cell.rules_exact for cell in self.cells):
            raise PlannerEvidenceError(
                PlannerFallbackReason.RULES_INEXACT,
                "option outcome contains inexact engine evidence",
            )
        leaf_by_history: dict[str, int] = {}
        for cell in self.cells:
            if cell.leaf_index is None:
                continue
            previous = leaf_by_history.setdefault(
                cell.information_history_fingerprint,
                cell.leaf_index,
            )
            if previous != cell.leaf_index:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.FINGERPRINT_MISMATCH,
                    "one information history maps to multiple value leaves",
                )
        _validate_nonanticipativity(self.continuation_decisions, len(self.cells))

    @property
    def scenario_support_fingerprint(self) -> str:
        """Return the content-bound identity of the originating support."""
        return self.scenario_support.support_fingerprint


def continuation_controller_fingerprint(
    *,
    controller_version: str,
    model_fingerprint: str,
    constructor_fingerprint: str,
    scorer_fingerprint: str,
    resolved_config_fingerprint: str,
) -> str:
    """Fingerprint all inputs that define deployment continuation semantics."""
    version = controller_version.strip().encode("utf-8")
    if not version:
        raise ValueError("controller_version must not be empty")
    identities = tuple(
        bytes.fromhex(_require_fingerprint(value))
        for value in (
            model_fingerprint,
            constructor_fingerprint,
            scorer_fingerprint,
            resolved_config_fingerprint,
        )
    )
    digest = hashlib.sha256()
    digest.update(_CONTROLLER_DOMAIN)
    digest.update(struct.pack(">I", len(version)))
    digest.update(version)
    for identity in identities:
        digest.update(identity)
    return digest.hexdigest()


def stable_continuation_controller_fingerprint(
    *,
    controller_version: str,
    constructor_fingerprint: str,
    scorer_fingerprint: str,
    continuation_semantics_fingerprint: str,
) -> str:
    """Fingerprint only stable continuation-controller semantics."""
    version = controller_version.strip().encode("utf-8")
    if not version:
        raise ValueError("controller_version must not be empty")
    identities = tuple(
        bytes.fromhex(_require_fingerprint(value))
        for value in (
            constructor_fingerprint,
            scorer_fingerprint,
            continuation_semantics_fingerprint,
        )
    )
    digest = hashlib.sha256()
    digest.update(_STABLE_CONTROLLER_DOMAIN)
    digest.update(
        struct.pack(
            ">II",
            STABLE_CONTINUATION_CONTROLLER_IDENTITY_SCHEMA_VERSION,
            len(version),
        )
    )
    digest.update(version)
    for identity in identities:
        digest.update(identity)
    return digest.hexdigest()


def _validate_nonanticipativity(
    decisions: tuple[ContinuationDecision, ...],
    scenario_count: int,
) -> None:
    action_by_history: dict[str, str] = {}
    seen: set[tuple[int, str]] = set()
    for decision in decisions:
        if decision.scenario_index >= scenario_count:
            raise PlannerEvidenceError(
                PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                "continuation decision references an absent scenario",
            )
        occurrence = (
            decision.scenario_index,
            decision.information_history_fingerprint,
        )
        if occurrence in seen:
            raise PlannerEvidenceError(
                PlannerFallbackReason.NONANTICIPATIVITY_VIOLATION,
                "duplicate continuation decision for one information history",
            )
        seen.add(occurrence)
        previous = action_by_history.setdefault(
            decision.information_history_fingerprint,
            decision.action_fingerprint,
        )
        if previous != decision.action_fingerprint:
            raise PlannerEvidenceError(
                PlannerFallbackReason.NONANTICIPATIVITY_VIOLATION,
                "controller chose different actions in one information set",
            )


def _require_fingerprint(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise ValueError("expected a lowercase SHA-256 fingerprint")
    return value


__all__ = [
    "STABLE_CONTINUATION_CONTROLLER_IDENTITY_SCHEMA_VERSION",
    "ContinuationDecision",
    "DecisionTransition",
    "DeploymentContinuationControllerIdentity",
    "OptionOutcome",
    "OptionOutcomeCell",
    "StableContinuationControllerIdentity",
    "continuation_controller_fingerprint",
    "stable_continuation_controller_fingerprint",
]
