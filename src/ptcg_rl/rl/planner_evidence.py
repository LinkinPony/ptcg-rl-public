"""Immutable schema-9 planner behavior evidence.

The records in this module are deliberately independent from schema-8 search
evidence.  They describe the distribution that actually sampled a complete
action before the engine advanced, together with the immutable retained
support needed to replay that distribution in PPO.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from ptcg_rl.agent.search.planner_fallback import PlannerFallbackReason
from ptcg_rl.engine.search_evidence import (
    SEARCH_EVIDENCE_DOWNSIDE_CVAR_INDEX,
    SEARCH_EVIDENCE_EXACT_INDEX,
    SEARCH_EVIDENCE_FEATURE_SIZE,
    SEARCH_EVIDENCE_MAX_SCORE_INDEX,
    SEARCH_EVIDENCE_MEAN_SCORE_INDEX,
    SEARCH_EVIDENCE_MIN_SCORE_INDEX,
    SEARCH_EVIDENCE_ROBUST_SCORE_INDEX,
    SEARCH_EVIDENCE_SCORE_STD_INDEX,
    SearchCandidateEvidence,
)

PLANNER_TRAJECTORY_SCHEMA_VERSION = 9
PLANNER_SOURCE_NAMES = (
    "base",
    "proposal",
    "cardinality",
    "stochastic",
    "structural",
    "random",
    "mutation",
    "novelty",
    "exhaustive",
)
PLANNER_SOURCE_COUNT = len(PLANNER_SOURCE_NAMES)
_EVIDENCE_DOMAIN = b"ptcg-rl/schema-9-planner-evidence/v1\x00"
_INT32_MAX = np.iinfo(np.int32).max
_INT64_MAX = np.iinfo(np.int64).max
_UINT16_MAX = np.iinfo(np.uint16).max


class PlannerBehaviorBranch(IntEnum):
    """Wire-stable distribution used for one behavior action."""

    BASE_FALLBACK = 0
    PLANNER_CONDITIONED = 1


class ScenarioSupportMode(IntEnum):
    """Wire-stable belief/chance support semantics."""

    UNSPECIFIED = 0
    BELIEF_SAMPLED_CHANCE_ENUMERATED = 1
    BELIEF_SAMPLED_CHANCE_SAMPLED = 2


@dataclass(frozen=True, slots=True)
class PlannerCandidateEvidence:
    """One retained complete action and its detached collection evidence."""

    action: tuple[int, ...]
    aggregate_features: tuple[float, ...]
    base_logprob: float
    proposal_logprob: float
    score_prior: float
    target_probability: float
    behavior_probability: float
    robust_score: float
    source_bits: int
    rules_exact: bool

    def __post_init__(self) -> None:
        """Validate a compact, replayable candidate row."""
        if len(set(self.action)) != len(self.action) or any(
            index < 0 or index > _INT32_MAX for index in self.action
        ):
            raise ValueError("planner candidate action indices exceed int32 bounds")
        if len(self.aggregate_features) != SEARCH_EVIDENCE_FEATURE_SIZE:
            raise ValueError("planner candidate aggregate feature width is invalid")
        SearchCandidateEvidence(
            action=self.action,
            features=self.aggregate_features,
        )
        numeric = (
            *self.aggregate_features,
            self.base_logprob,
            self.proposal_logprob,
            self.score_prior,
            self.target_probability,
            self.behavior_probability,
            self.robust_score,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("planner candidate values must be finite")
        if self.target_probability < 0.0 or self.behavior_probability <= 0.0:
            raise ValueError("planner candidate probabilities are invalid")
        if self.base_logprob > 1.0e-6 or self.proposal_logprob > 1.0e-6:
            raise ValueError("planner complete-action log-probabilities must be <= 0")
        if not math.isclose(
            self.robust_score,
            self.aggregate_features[SEARCH_EVIDENCE_ROBUST_SCORE_INDEX],
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise ValueError("planner robust score differs from aggregate features")
        expected_exact = 1.0 if self.rules_exact else 0.0
        if self.aggregate_features[SEARCH_EVIDENCE_EXACT_INDEX] != expected_exact:
            raise ValueError("planner rules-exact flag differs from aggregate features")
        _validate_aggregate_score_order(self.aggregate_features)
        if self.source_bits <= 0 or self.source_bits >= (1 << PLANNER_SOURCE_COUNT):
            raise ValueError("planner candidate source bitset is invalid")


@dataclass(frozen=True, slots=True)
class PlannerBehaviorEvidence:
    """One behavior branch plus immutable evidence/version identities."""

    branch: PlannerBehaviorBranch
    fallback_reason: PlannerFallbackReason
    candidates: tuple[PlannerCandidateEvidence, ...]
    selected_candidate_index: int
    root_information_fingerprint: str
    scenario_support_fingerprint: str
    model_fingerprint: str
    constructor_fingerprint: str
    scorer_fingerprint: str
    controller_fingerprint: str
    planner_fingerprint: str
    policy_version: int
    proposal_version: int
    constructor_version: int
    planner_version: int
    scenario_support_mode: ScenarioSupportMode
    legal_action_count: int
    scenario_count: int
    support_exhaustive: bool
    scenario_grid_complete: bool
    leaf_bootstrapped: bool
    support_censored: bool
    configured_source_quotas: tuple[int, ...]
    used_source_quotas: tuple[int, ...]
    engine_transition_limit: int
    engine_transitions_used: int
    prefix_node_limit: int
    prefix_nodes_used: int
    wall_clock_limit_ms: int
    wall_clock_used_ms: int
    planner_temperature: float

    def __post_init__(self) -> None:
        """Reject evidence that cannot define the recorded behavior policy."""
        for fingerprint in (
            self.root_information_fingerprint,
            self.scenario_support_fingerprint,
            self.model_fingerprint,
            self.constructor_fingerprint,
            self.scorer_fingerprint,
            self.controller_fingerprint,
            self.planner_fingerprint,
        ):
            _require_fingerprint(fingerprint)
        for version in (
            self.policy_version,
            self.proposal_version,
            self.constructor_version,
            self.planner_version,
        ):
            if version < 0 or version > _INT32_MAX:
                raise ValueError("planner evidence version exceeds int32 wire bounds")
        if not 0 < self.legal_action_count <= _INT64_MAX:
            raise ValueError("planner evidence requires a positive legal action count")
        if not 0 <= self.scenario_count <= _INT32_MAX:
            raise ValueError("planner scenario count exceeds int32 wire bounds")
        if (
            len(self.configured_source_quotas) != PLANNER_SOURCE_COUNT
            or len(self.used_source_quotas) != PLANNER_SOURCE_COUNT
        ):
            raise ValueError("planner source quota vectors have the wrong width")
        if any(
            value < 0 or value > _UINT16_MAX for value in self.configured_source_quotas
        ) or any(value < 0 or value > _UINT16_MAX for value in self.used_source_quotas):
            raise ValueError("planner source quotas exceed uint16 wire bounds")
        _validate_budget_pair(
            self.engine_transitions_used,
            self.engine_transition_limit,
            "engine transitions",
        )
        _validate_budget_pair(
            self.prefix_nodes_used,
            self.prefix_node_limit,
            "prefix nodes",
        )
        _validate_budget_pair(
            self.wall_clock_used_ms,
            self.wall_clock_limit_ms,
            "wall clock",
            allow_overrun=self.branch is PlannerBehaviorBranch.BASE_FALLBACK,
        )
        if not math.isfinite(self.planner_temperature) or self.planner_temperature <= 0:
            raise ValueError("planner temperature must be finite and positive")
        if self.support_censored == self.support_exhaustive:
            raise ValueError(
                "support_censored must be the inverse of support_exhaustive"
            )
        if len({candidate.action for candidate in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("planner retained actions must be unique")
        if len(self.candidates) > self.legal_action_count:
            raise ValueError("planner support exceeds the legal action count")
        if self.selected_candidate_index < -1 or (
            self.selected_candidate_index > _INT32_MAX
        ):
            raise ValueError("selected candidate index exceeds int32 wire bounds")
        if self.branch is PlannerBehaviorBranch.PLANNER_CONDITIONED:
            self._validate_planner_branch()
        else:
            self._validate_fallback_branch()

    @property
    def selected_action(self) -> tuple[int, ...] | None:
        """Return the categorical action or ``None`` for base fallback."""
        if self.branch is PlannerBehaviorBranch.BASE_FALLBACK:
            return None
        return self.candidates[self.selected_candidate_index].action

    @property
    def refill_slots_used(self) -> int:
        """Return actual source admissions beyond reserved source slots."""
        return sum(
            max(0, used - configured)
            for used, configured in zip(
                self.used_source_quotas,
                self.configured_source_quotas,
                strict=True,
            )
        )

    @property
    def selected_old_logprob(self) -> float | None:
        """Return the old categorical log-probability for a planner row."""
        if self.branch is PlannerBehaviorBranch.BASE_FALLBACK:
            return None
        probability = self.candidates[
            self.selected_candidate_index
        ].behavior_probability
        return math.log(probability)

    @property
    def evidence_fingerprint(self) -> str:
        """Bind support, detached evidence, flags, budgets, and versions."""
        digest = hashlib.sha256()
        digest.update(_EVIDENCE_DOMAIN)
        digest.update(
            struct.pack(
                ">iiiiiiiiqq????iiiiiiif",
                PLANNER_TRAJECTORY_SCHEMA_VERSION,
                int(self.branch),
                int(self.fallback_reason),
                int(self.scenario_support_mode),
                self.policy_version,
                self.proposal_version,
                self.constructor_version,
                self.planner_version,
                self.legal_action_count,
                self.scenario_count,
                self.support_exhaustive,
                self.scenario_grid_complete,
                self.leaf_bootstrapped,
                self.support_censored,
                self.engine_transition_limit,
                self.engine_transitions_used,
                self.prefix_node_limit,
                self.prefix_nodes_used,
                self.wall_clock_limit_ms,
                self.wall_clock_used_ms,
                self.selected_candidate_index,
                np.float32(self.planner_temperature),
            )
        )
        for fingerprint in (
            self.root_information_fingerprint,
            self.scenario_support_fingerprint,
            self.model_fingerprint,
            self.constructor_fingerprint,
            self.scorer_fingerprint,
            self.controller_fingerprint,
            self.planner_fingerprint,
        ):
            digest.update(bytes.fromhex(fingerprint))
        digest.update(np.asarray(self.configured_source_quotas, dtype=">u2").tobytes())
        digest.update(np.asarray(self.used_source_quotas, dtype=">u2").tobytes())
        for candidate in self.candidates:
            digest.update(struct.pack(">I", len(candidate.action)))
            digest.update(np.asarray(candidate.action, dtype=">i4").tobytes())
            digest.update(
                np.asarray(
                    (
                        *candidate.aggregate_features,
                        candidate.base_logprob,
                        candidate.proposal_logprob,
                        candidate.score_prior,
                        candidate.target_probability,
                        candidate.behavior_probability,
                        candidate.robust_score,
                    ),
                    dtype=">f4",
                ).tobytes()
            )
            digest.update(
                struct.pack(">HB", candidate.source_bits, candidate.rules_exact)
            )
        return digest.hexdigest()

    def _validate_planner_branch(self) -> None:
        if self.fallback_reason is not PlannerFallbackReason.NONE:
            raise ValueError("planner-conditioned rows cannot carry fallback reasons")
        if not self.candidates:
            raise ValueError("planner-conditioned rows require retained candidates")
        if not 0 <= self.selected_candidate_index < len(self.candidates):
            raise ValueError("selected planner candidate index is invalid")
        if self.scenario_count <= 0 or not self.scenario_grid_complete:
            raise ValueError("planner behavior requires a complete scenario grid")
        if self.scenario_support_mode is ScenarioSupportMode.UNSPECIFIED:
            raise ValueError("planner behavior requires explicit scenario semantics")
        if any(not candidate.rules_exact for candidate in self.candidates):
            raise ValueError("planner behavior requires exact engine transitions")
        if self.support_exhaustive and len(self.candidates) != self.legal_action_count:
            raise ValueError(
                "exhaustive planner support must contain every legal action"
            )
        if sum(self.used_source_quotas) > len(self.candidates):
            raise ValueError("planner source usage exceeds the retained support")
        for source_index, used in enumerate(self.used_source_quotas):
            provenance_count = sum(
                bool(candidate.source_bits & (1 << source_index))
                for candidate in self.candidates
            )
            if used > provenance_count:
                raise ValueError("planner source usage exceeds candidate provenance")
        target_sum = math.fsum(
            candidate.target_probability for candidate in self.candidates
        )
        behavior_sum = math.fsum(
            candidate.behavior_probability for candidate in self.candidates
        )
        if not math.isclose(target_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError("planner target probabilities must sum to one")
        if not math.isclose(behavior_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError("planner behavior probabilities must sum to one")
        target_logits = tuple(
            (candidate.base_logprob + candidate.score_prior) / self.planner_temperature
            for candidate in self.candidates
        )
        target_max = max(target_logits)
        target_weights = tuple(math.exp(value - target_max) for value in target_logits)
        target_total = math.fsum(target_weights)
        expected_target = tuple(value / target_total for value in target_weights)
        if any(
            not math.isclose(
                candidate.target_probability,
                expected,
                rel_tol=2.0e-5,
                abs_tol=2.0e-6,
            )
            for candidate, expected in zip(
                self.candidates,
                expected_target,
                strict=True,
            )
        ):
            raise ValueError("planner target probabilities differ from saved logits")

    def _validate_fallback_branch(self) -> None:
        if self.fallback_reason is PlannerFallbackReason.NONE:
            raise ValueError("base fallback rows require an explicit reason")
        if self.candidates or self.selected_candidate_index != -1:
            raise ValueError("base fallback rows cannot retain planner candidates")
        if self.support_exhaustive:
            raise ValueError(
                "base fallback cannot claim materialized exhaustive support"
            )
        if self.scenario_grid_complete or self.leaf_bootstrapped:
            raise ValueError(
                "base fallback cannot claim a complete scored planner grid"
            )


def planner_source_bits(sources: tuple[str, ...]) -> int:
    """Encode merged candidate provenance with a stable compact bitset."""
    bits = 0
    for source in sources:
        try:
            index = PLANNER_SOURCE_NAMES.index(source)
        except ValueError as exc:
            raise ValueError(f"unknown planner candidate source: {source}") from exc
        bits |= 1 << index
    return bits


def _validate_budget_pair(
    used: int,
    limit: int,
    label: str,
    *,
    allow_overrun: bool = False,
) -> None:
    if (
        limit < 0
        or limit > _INT32_MAX
        or used < 0
        or used > _INT32_MAX
        or (used > limit and not allow_overrun)
    ):
        raise ValueError(f"planner {label} usage exceeds its limit")


def _validate_aggregate_score_order(features: tuple[float, ...]) -> None:
    """Require the public aggregate row to obey its score invariants."""
    score_std = features[SEARCH_EVIDENCE_SCORE_STD_INDEX]
    minimum = features[SEARCH_EVIDENCE_MIN_SCORE_INDEX]
    downside = features[SEARCH_EVIDENCE_DOWNSIDE_CVAR_INDEX]
    mean = features[SEARCH_EVIDENCE_MEAN_SCORE_INDEX]
    maximum = features[SEARCH_EVIDENCE_MAX_SCORE_INDEX]
    tolerance = 1.0e-6
    if score_std < 0.0:
        raise ValueError("planner aggregate score standard deviation is negative")
    if not (
        minimum - tolerance <= downside <= mean + tolerance
        and minimum - tolerance <= mean <= maximum + tolerance
    ):
        raise ValueError("planner aggregate score ordering is invalid")


def _require_fingerprint(value: str) -> None:
    if len(value) != 64:
        raise ValueError("planner identities must be lowercase SHA-256")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("planner identities must be lowercase SHA-256") from exc
    if len(raw) != 32 or value != value.lower():
        raise ValueError("planner identities must be lowercase SHA-256")


__all__ = [
    "PLANNER_SOURCE_COUNT",
    "PLANNER_SOURCE_NAMES",
    "PLANNER_TRAJECTORY_SCHEMA_VERSION",
    "PlannerBehaviorBranch",
    "PlannerBehaviorEvidence",
    "PlannerCandidateEvidence",
    "ScenarioSupportMode",
    "planner_source_bits",
]
